"""Pure JAX implementation of MNFIX — Fortran-faithful partial-transfer algorithm.

Port of mnfix.f (Peter Adams, September 2000).

When average mass per particle drifts outside a bin's boundaries, this routine
performs a **partial transfer**: it splits the bin's population so that some
particles stay (at the geometric mean mass) and the excess moves to the
appropriate neighbor bin. This conserves both total number and total mass.

Key difference from a naive "move-all" approach:
  - Fortran/this code: partial split → smooth gradual redistribution
  - Naive move-all: entire bin dumps to neighbor → oscillation/pulses

Phases 1-2 touch only bin k, so they run on all bins at once. Phase 3 stays a
``jax.lax.fori_loop`` in the Fortran in-place order (an upward shift feeds bins
processed later) but each iteration updates only rows k and kk. Results are
bit-identical to the original one-bin-at-a-time port
(tests/reference/mnfix_sequential.py, tests/test_mnfix.py).

References:
    - mnfix.f (Peter Adams, September 2000)
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp
from typing import Tuple

from .config import ICOMP_NODIAG


# Number MNFIX leaves in an empty bin [#/cell] (Fortran mnfix.f: parameter(Neps=1E-5))
MNFIX_N_EMPTY = 1.0e-5


def mnfix_jax(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    icomp_nodiag: int = ICOMP_NODIAG
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Correct mass/number drift using Fortran-faithful partial transfer.

    Port of mnfix.f: when avg mass exceeds bin boundaries, split the bin
    population so the remainder sits at the geometric mean and the excess
    moves to the appropriate neighbor bin.

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        xk: Bin boundaries [kg], shape (ibins+1,)
        icomp_nodiag: Number of non-diagnostic species.

    Returns:
        Nk_new: Corrected number concentration
        Mk_new: Corrected mass concentration
    """
    nbins = Nk.shape[0]
    icomp = Mk.shape[1]

    # Constants matching Fortran mnfix.f
    EPS = 1.0e-40
    NEPS = MNFIX_N_EMPTY

    # Enforce physical positivity
    Nk = jnp.maximum(Nk, 0.0)
    Mk = jnp.maximum(Mk, 0.0)

    # =========================================================================
    # Phase 1: Fix empty/near-empty bins (mnfix.f lines 56-72)
    # If Nk < Neps: zero out dry mass, set Nk=Neps, Mk_so4 = Neps*sqrt(xk*xk+1)
    # =========================================================================
    xk_geo = jnp.sqrt(xk[:-1] * xk[1:])
    is_empty = Nk < NEPS
    Mk_empty = jnp.zeros_like(Mk).at[:, 0].set(NEPS * xk_geo)
    Nk = jnp.where(is_empty, NEPS, Nk)
    Mk = jnp.where(is_empty[:, None], Mk_empty, Mk)

    # =========================================================================
    # Phase 2: Fix extreme out-of-range (mnfix.f lines 75-94)
    # avg > xk[ibins+1]: trim mass
    # avg < xk[0]: trim number
    # =========================================================================
    drymass = jnp.sum(Mk[:, :icomp_nodiag], axis=1)
    too_high = drymass / (Nk + EPS) > xk[nbins]
    scale = (Nk * xk[nbins] / 1.2) / (drymass + EPS)
    Mk_trimmed = Mk * scale[:, None]
    Mk = jnp.where(too_high[:, None], Mk_trimmed, Mk)
    # Recompute drymass after potential trim
    drymass2 = jnp.where(too_high, jnp.sum(Mk_trimmed[:, :icomp_nodiag], axis=1), drymass)
    too_low = drymass2 / (Nk + EPS) < xk[0]
    Nk = jnp.where(too_low, drymass2 / (xk[0] * 1.2), Nk)

    # =========================================================================
    # Phase 3: Partial transfer for out-of-range bins (mnfix.f lines 96-156)
    # This is the CRITICAL part that differs from the old "move-all" approach.
    #
    # When avg > xk[k+1]:
    #   xold = sqrt(xk[k]*xk[k+1])  (geometric mean of current bin)
    #   xnew = xk[kk+1] / 1.1       (target mass in destination bin kk)
    #   nshift = (drymass - xold*number) / (xnew - xold)
    #   Particles staying in bin k get avg mass = xold
    #   Particles moving to bin kk get avg mass = xnew
    #
    # Similarly for avg < xk[k] (downward shift).
    #
    # Only bins that are out of range when the loop reaches them change, so
    # the loop runs from the first out-of-range bin to `hi`: the last one, or
    # the highest bin an upward shift has fed so far (targets can be several
    # bins up). Downward shifts land in bins already passed, which the Fortran
    # order never revisits either. Skipped iterations would change nothing,
    # so the result is bit-identical to looping over every bin.
    # =========================================================================
    log_p = jnp.log(xk[1] / xk[0])  # log of the bin mass ratio
    idx = jnp.arange(nbins)

    # Out-of-range bins after phases 1-2, with the loop body's own formulas
    drymass = jnp.sum(Mk[:, :icomp_nodiag], axis=1)
    avg = drymass / (Nk + EPS)
    avg = jnp.where((Nk == 0.0) & (drymass <= 0.0), xk_geo, avg)
    out = (((avg > xk[1:]) & (idx < nbins - 1))
           | ((avg < xk[:-1]) & (idx > 0)))
    k_first = jnp.where(jnp.any(out), jnp.argmax(out), nbins)
    hi_first = jnp.max(jnp.where(out, idx, -1))

    def fix_drift(k, state):
        Nk_s, Mk_s = state
        row = Mk_s[k]

        drymass = jnp.sum(row[:icomp_nodiag])
        number = Nk_s[k]
        avg = drymass / (number + EPS)

        # --- Handle N=0 edge case (Fortran lines 103-110) ---
        avg = jnp.where(
            (number == 0.0) & (drymass <= 0.0),
            jnp.sqrt(xk[k] * xk[k + 1]),  # use geometric mean
            avg
        )

        xk_lo = xk[k]
        xk_hi = xk[k + 1]
        xold = jnp.sqrt(xk_lo * xk_hi)  # geometric mean of current bin

        # Composition fractions
        fj = row / (drymass + EPS)

        # ===== UPWARD SHIFT (avg > xk[k+1]) =====
        # Target bin kk such that xk[kk+1]/1.1 > avg (Fortran goto loop); for
        # xk[j] = xk[0] * p^j: kk = ceil(log_p(avg*1.1/xk[0])) - 1
        needs_up = (avg > xk_hi) & (k < nbins - 1)
        ratio_up = avg * 1.1 / (xk[0] + EPS)
        kk_up_raw = jnp.ceil(jnp.log(jnp.maximum(ratio_up, 1.0)) / log_p).astype(jnp.int32) - 1
        kk_up = jnp.clip(kk_up_raw, k + 1, nbins - 1)
        xnew_up = xk[jnp.minimum(kk_up + 1, nbins)] / 1.1

        # ===== DOWNWARD SHIFT (avg < xk[k]) =====
        # Target bin kk such that xk[kk]*1.1 < avg: kk = floor(log_p(avg/(1.1*xk[0])))
        needs_down = (avg < xk_lo) & (k > 0)
        ratio_dn = avg / (1.1 * xk[0] + EPS)
        kk_dn_raw = jnp.floor(jnp.log(jnp.maximum(ratio_dn, 1.0)) / log_p).astype(jnp.int32)
        kk_dn = jnp.clip(kk_dn_raw, 0, jnp.maximum(k - 1, 0))
        xnew_dn = xk[kk_dn] * 1.1

        nshift_up = (drymass - xold * number) / (xnew_up - xold + EPS)
        nshift_up = jnp.maximum(nshift_up, 0.0)  # Safety: no negative shifts
        nshift_up = jnp.minimum(nshift_up, number)  # Can't shift more than we have
        mshift_up = xnew_up * nshift_up
        nshift_dn = (drymass - xold * number) / (xnew_dn - xold + EPS)
        nshift_dn = jnp.maximum(nshift_dn, 0.0)
        nshift_dn = jnp.minimum(nshift_dn, number)
        mshift_dn = xnew_dn * nshift_dn

        # Up and down are exclusive (avg cannot exceed xk_hi and be below
        # xk_lo). Only rows k and kk change: the particles left in bin k get
        # avg mass xold with composition fj, the shifted ones add mshift*fj
        # to bin kk.
        shift = needs_up | needs_down
        kk = jnp.where(needs_up, kk_up, kk_dn)
        nshift = jnp.where(needs_up, nshift_up, nshift_dn)
        mshift = jnp.where(needs_up, mshift_up, mshift_dn)
        n_remain = number - nshift
        nk_k = jnp.where(shift, number + (-nshift), number)
        mk_k = jnp.where(shift, xold * n_remain * fj, row)
        nk_kk = Nk_s[kk] + jnp.where(shift, nshift, 0.0)
        Nk_s = Nk_s.at[k].set(nk_k)
        Mk_s = Mk_s.at[k].set(mk_k)
        mk_kk = jnp.where(shift, Mk_s[kk] + mshift * fj, Mk_s[kk])
        Nk_s = Nk_s.at[kk].set(jnp.where(shift, nk_kk, Nk_s[kk]))
        Mk_s = Mk_s.at[kk].set(mk_kk)
        return Nk_s, Mk_s, jnp.where(needs_up, kk_up, -1)

    def body(state):
        k, hi, Nk_s, Mk_s = state
        Nk_s, Mk_s, kk_up = fix_drift(k, (Nk_s, Mk_s))
        return k + 1, jnp.maximum(hi, kk_up), Nk_s, Mk_s

    _, _, Nk, Mk = jax.lax.while_loop(
        lambda s: (s[0] <= s[1]) & (s[0] < nbins), body,
        (k_first, hi_first, Nk, Mk))

    return Nk, Mk
