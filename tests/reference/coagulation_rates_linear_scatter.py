"""Original linear sub-bin coagulation rates (scatter-add formulation).

Verbatim copy of ``calc_coagulation_rates_linear`` as it was before the
dense masked-reduction rewrite (2026-11-24). Kept as the reference that
``tests/test_coagulation_rates_linear_fast.py`` compares the library function
against (values and tangents to round-off) and that
``benchmarks/python/coag_rates_profile.py`` times. Not used by the library.
"""
import jax.numpy as jnp
# float64 enforced by core/config.py
from typing import Tuple

from tomas_jax.core.config import ICOMP_NODIAG
from tomas_jax.physics.coagulation_rates import _preprocess_concentrations
from tomas_jax.physics.coagulation_rates_linear import (
    _TINY, _partial_moments, linear_subbin_reconstruction)


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
