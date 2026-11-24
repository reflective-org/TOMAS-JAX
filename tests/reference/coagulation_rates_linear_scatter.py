"""Original linear sub-bin coagulation rates (scatter-add formulation).

Verbatim copy of ``calc_coagulation_rates_linear`` (renamed) and of the
helpers it calls from ``tomas_jax/physics/coagulation_rates_linear.py``
(``linear_subbin_reconstruction``, ``_partial_moments``, ``_NARROW``,
``_TINY``) as they were before the 2026-11-24 rewrite, which evaluates only
the pairs i <= j, finds the destination bin in closed form on geometric grids
and accumulates with one windowed scatter per array. The helpers are copied
rather than imported so that later changes to the library (for example a
double-where in the reconstruction) do not change this reference. The only
library import is ``_preprocess_concentrations``, the empty-bin cleanup shared
with TFL.

Kept as the reference that ``tests/test_coagulation_rates_linear_fast.py``
compares the library function against (values and tangents to round-off) and
that ``benchmarks/python/coag_rates_profile.py`` times. Not used by the
library.
"""
import jax.numpy as jnp
# float64 enforced by core/config.py
from typing import Tuple

from tomas_jax.core.config import ICOMP_NODIAG
from tomas_jax.physics.coagulation_rates import _preprocess_concentrations

_NARROW = 1.0e-6
_TINY = 1.0e-300


def linear_subbin_reconstruction(
    xbar: jnp.ndarray, xk: jnp.ndarray
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Per-particle linear number density within each bin.

    Returns the support [lo, hi] and the endpoint densities (n_lo, n_hi) of a
    non-negative linear density with unit integral and mean xbar:

      - xbar within w/6 of the bin centre: full-bin linear profile;
      - xbar further toward an edge: a ramp to zero on a sub-interval
        (support [3*xbar - 2*b, b] or [a, 3*xbar - 2*a]);
      - xbar at or outside an edge: a narrow uniform centred on xbar.

    Args:
        xbar: Mean particle mass per bin [kg], shape (nbins,)
        xk: Bin boundaries [kg], shape (nbins+1,)

    Returns:
        lo, hi, n_lo, n_hi: each shape (nbins,), densities in [1/kg].
    """
    a, b = xk[:-1], xk[1:]
    w = b - a
    c = 0.5 * (a + b)

    # Full-bin profile: n(x) = 1/w + 12 (xbar - c)(x - c) / w^3
    g = 6.0 * (xbar - c) / (w * w)
    lo, hi, n_lo, n_hi = a, b, 1.0 / w - g, 1.0 / w + g

    # Ramp rising to the top edge (zero at x0)
    x0 = 3.0 * xbar - 2.0 * b
    up = xbar > c + w / 6.0
    lo = jnp.where(up, x0, lo)
    n_lo = jnp.where(up, 0.0, n_lo)
    n_hi = jnp.where(up, 2.0 / jnp.maximum(b - x0, _TINY), n_hi)

    # Ramp falling to zero at x1 from the bottom edge
    x1 = 3.0 * xbar - 2.0 * a
    dn = xbar < c - w / 6.0
    hi = jnp.where(dn, x1, hi)
    n_lo = jnp.where(dn, 2.0 / jnp.maximum(x1 - a, _TINY), n_lo)
    n_hi = jnp.where(dn, 0.0, n_hi)

    # Near-delta when the ramp degenerates (xbar at or beyond an edge)
    d = _NARROW * w
    narrow = (xbar >= b - d) | (xbar <= a + d)
    lo = jnp.where(narrow, xbar - d, lo)
    hi = jnp.where(narrow, xbar + d, hi)
    n_lo = jnp.where(narrow, 0.5 / d, n_lo)
    n_hi = jnp.where(narrow, 0.5 / d, n_hi)

    return lo, hi, n_lo, n_hi


def _partial_moments(lo, hi, n_lo, n_hi, t):
    """Zeroth and first moments of the linear density over [lo, t].

    t must already lie in [lo, hi].
    """
    slope = (n_hi - n_lo) / jnp.maximum(hi - lo, _TINY)
    n_t = n_lo + slope * (t - lo)
    span = t - lo
    num = 0.5 * (n_lo + n_t) * span
    # x * n(x) is quadratic, so Simpson's rule is exact
    mom = span / 6.0 * (lo * (2.0 * n_lo + n_t) + t * (n_lo + 2.0 * n_t))
    return num, mom


def calc_coagulation_rates_linear_scatter(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    kij: jnp.ndarray,
    xk: jnp.ndarray,
    icomp_nodiag: int = ICOMP_NODIAG
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Calculate coagulation rates dNdt and dMdt for any bin mass ratio.

    Same interface and return values as calc_coagulation_rates. Requires
    bin widths that do not decrease with size (true for any constant mass
    ratio grid), which bounds each pair's products to two destination bins.

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        kij: Coagulation kernel [s⁻¹], shape (ibins, ibins)
        xk: Bin boundaries [kg], shape (ibins+1,)
        icomp_nodiag: Number of non-diagnostic species.

    Returns:
        dNdt: Rate of change of number, shape (ibins,)
        dMdt: Rate of change of mass, shape (ibins, icomp)
        dM_overflow: Mass rate of products above the top bin boundary,
            shape (icomp,).
    """
    nbins = Nk.shape[0]
    Nk_s, Mk_s = _preprocess_concentrations(Nk, Mk, xk)
    M = Mk_s[:, :icomp_nodiag]
    Mtot = jnp.sum(M, axis=1)
    xbar = Mtot / Nk_s
    frac = M / jnp.maximum(Mtot, _TINY)[:, None]  # composition of each bin
    mbar = M / Nk_s[:, None]                      # species mass per particle

    # Collision rates R[j, i] for the larger bin j and partner i <= j
    # (self-coagulation counted once per pair of particles).
    pair_w = jnp.tril(jnp.ones((nbins, nbins)), k=-1) + 0.5 * jnp.eye(nbins)
    R = kij * Nk_s[:, None] * Nk_s[None, :] * pair_w

    # Products of (j, i): bin j's support shifted up by xbar_i
    lo, hi, n_lo, n_hi = linear_subbin_reconstruction(xbar, xk)
    y = xbar[None, :]
    start = lo[:, None] + y
    # d0 = bin holding the lowest product; nbins means "above the grid"
    d0 = jnp.clip(jnp.searchsorted(xk, start, side='right') - 1, 0, nbins)
    xk_ext = jnp.concatenate([xk, jnp.array([jnp.inf])])
    t = jnp.clip(xk_ext[d0 + 1] - y, lo[:, None], hi[:, None])

    # Fraction of each collision's number and bin-j mass going to d0;
    # the remainder goes to d0 + 1 (complements keep conservation exact).
    nu_lo, mu_lo = _partial_moments(
        lo[:, None], hi[:, None], n_lo[:, None], n_hi[:, None], t)
    nu_hi = 1.0 - nu_lo
    mu_hi = xbar[:, None] - mu_lo

    # Scatter into nbins + 2 destinations (last two collect overflow).
    jj = jnp.broadcast_to(jnp.arange(nbins)[:, None], (nbins, nbins))
    ii = jnp.broadcast_to(jnp.arange(nbins)[None, :], (nbins, nbins))
    A = (jnp.zeros((nbins + 2, nbins))
         .at[d0, jj].add(R * mu_lo)
         .at[d0 + 1, jj].add(R * mu_hi))  # bin-j mass reaching each dest
    B = (jnp.zeros((nbins + 2, nbins))
         .at[d0, ii].add(R * nu_lo)
         .at[d0 + 1, ii].add(R * nu_hi))  # partner count reaching each dest
    gain_N = jnp.sum(B, axis=1)
    gain_M = A @ frac + B @ mbar

    loss_N = jnp.sum(R, axis=1) + jnp.sum(R, axis=0)

    dNdt = gain_N[:nbins] - loss_N
    dMdt_nodiag = gain_M[:nbins] - loss_N[:, None] * mbar

    dMdt = jnp.zeros_like(Mk).at[:, :icomp_nodiag].set(dMdt_nodiag)
    dM_overflow = jnp.zeros(Mk.shape[1]).at[:icomp_nodiag].set(
        jnp.sum(gain_M[nbins:], axis=0))

    return dNdt, dMdt, dM_overflow
