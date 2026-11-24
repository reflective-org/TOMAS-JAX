"""Two-moment coagulation rates for arbitrary bin mass ratio (linear sub-bin scheme).

The TFL algorithm in coagulation_rates.py assumes mass-doubling bins
(xk[k+1] = 2*xk[k]): self-coagulation products of bin k land in bin k+1, and
a collision with a smaller particle moves a particle at most one bin. Neither
holds for other ratios -- at p = sqrt(2) (80 bins) self-coagulation products
belong in bin k+2, at p = 2^(1/4) (160 bins) in bin k+4 -- and the TFL
transfer terms produce spurious oscillations that grow with resolution.

This scheme makes no assumption about the bin ratio. For every colliding pair
of bins (j, i) with i <= j:

  - particles in the larger bin j follow a positive-definite linear number
    density on [xk[j], xk[j+1]] that reproduces N_j and the mean mass xbar_j
    (Simmel et al. 2002 "linear discrete method" reconstruction);
  - particles in the partner bin i are taken at their mean mass xbar_i;
  - the products x + xbar_i fill a shifted copy of bin j's support. Since bin
    widths grow with mass, that interval overlaps at most two destination
    bins, and the number and per-species mass landing in each are found by
    integrating the linear density exactly.

Losses are exact (K_ij N_i N_j collisions remove one particle from each bin),
so number and mass are conserved to round-off independent of the grid.

References:
    Simmel, Trautmann, and Tetzlaff (2002) "Numerical solution of the
    stochastic collection equation -- comparison of the Linear Discrete Method
    with other methods", Atmos. Res., 61, 135-148.
    Tzivion, Feingold, and Levin (1987), J. Atmos. Sci., 44, 3139-3149.
"""
import jax
import jax.numpy as jnp
import numpy as np
# float64 enforced by core/config.py
from typing import Tuple

from ..core.config import ICOMP_NODIAG
from .coagulation_rates import _preprocess_concentrations

# Half-width of the near-delta reconstruction used when xbar sits at (or
# outside) a bin edge, as a fraction of the bin width.
_NARROW = 1.0e-6
_TINY = 1.0e-300
# Largest |log_p(xk[k]/xk[0]) - k| for which the closed-form destination bin
# is used; one +-1 correction then makes it exact (see _destination_bin).
_GEOMETRIC_TOL = 0.25


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


def _destination_bin_geometric(xk, x):
    """searchsorted(xk, x, side='right') - 1 clipped to [0, nbins], closed form.

    Exact when every edge satisfies |log_p(xk[k]/xk[0]) - k| < _GEOMETRIC_TOL
    with p = xk[1]/xk[0]: the computed log_p(x/xk[0]) then lies within 1.25
    bins of the true index, floor() is off by at most one, and comparing x
    with the two neighbouring edges corrects it. Non-positive x goes below
    the grid and NaN above it, as searchsorted does.
    """
    nbins = xk.shape[0] - 1
    r = jnp.log(jnp.maximum(x, 0.0) / xk[0]) / jnp.log(xk[1] / xk[0])
    c = jnp.where(jnp.isnan(r), float(nbins), jnp.clip(jnp.floor(r), -1.0, float(nbins)))
    c = c.astype(jnp.int32)
    xk_pad = jnp.concatenate([jnp.array([-jnp.inf]), xk, jnp.array([jnp.inf])])
    c = (c + (xk_pad[c + 2] <= x).astype(jnp.int32)
         - (xk_pad[c + 1] > x).astype(jnp.int32))
    return jnp.clip(c, 0, nbins)


def _destination_bin_search(xk, x):
    nbins = xk.shape[0] - 1
    return jnp.clip(jnp.searchsorted(xk, x, side='right') - 1, 0, nbins).astype(jnp.int32)


def _destination_bin(xk, x):
    """Bin holding mass x (nbins = above the grid), exact for any grid.

    The closed form is about 5x cheaper than jnp.searchsorted at 80 bins; a
    grid that is not geometric within _GEOMETRIC_TOL takes searchsorted. The
    choice depends on xk only, so it stays a real branch under vmap unless xk
    itself is batched.
    """
    x = jax.lax.stop_gradient(x)
    k = jnp.arange(xk.shape[0], dtype=xk.dtype)
    log_ratio = jnp.log(xk / xk[0]) / jnp.log(xk[1] / xk[0])
    geometric = jnp.all(jnp.abs(log_ratio - k) < _GEOMETRIC_TOL)
    return jax.lax.cond(geometric, _destination_bin_geometric,
                        _destination_bin_search, xk, x)


def _materialize(x):
    """Return x unchanged, computed once into its own buffer.

    XLA CPU fuses an elementwise producer into a gather and re-evaluates it
    for every gathered element, so gathering the per-bin reconstruction for
    the n(n+1)/2 pairs would redo its where-chain per pair (~20% of the call
    at 80 bins). A cumulative sum is not fused into the gather; its first
    element is x exactly (no arithmetic), and its tangent and transpose are
    the identity. lax.optimization_barrier does not work here: XLA CPU
    removes it before fusion.
    """
    return jax.lax.cumsum(jnp.stack([x, jnp.zeros_like(x)]), axis=0)[0]


def calc_coagulation_rates_linear(
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
    On a geometric grid the destination bin is found in closed form, on any
    other grid by jnp.searchsorted (same result, see _destination_bin).

    Only the pairs i <= j are evaluated and each array is accumulated with
    one windowed scatter; the original all-pairs scatter-add version is kept
    in tests/reference/coagulation_rates_linear_scatter.py, and the two agree
    to round-off (docs/coagulation_bin_ratio.md, Cost).

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

    loss_N = jnp.sum(R, axis=1) + jnp.sum(R, axis=0)

    # Products of (j, i): bin j's support shifted up by xbar_i. R is zero
    # above the diagonal, so only the n(n+1)/2 pairs i <= j are evaluated
    # (index arrays fixed by the array shape).
    lo, hi, n_lo, n_hi = linear_subbin_reconstruction(xbar, xk)
    jp, ip = np.tril_indices(nbins)
    per_j = _materialize(jnp.stack([lo, hi, n_lo, n_hi, xbar], axis=-1))[jp]
    lo_p, hi_p, n_lo_p, n_hi_p, xbar_j = (per_j[:, q] for q in range(5))
    y = xbar[ip]
    R_p = R.reshape(-1)[jp * nbins + ip]
    # d0 = bin holding the lowest product; nbins means "above the grid"
    d0 = _destination_bin(xk, lo_p + y)
    xk_ext = jnp.concatenate([xk, jnp.array([jnp.inf])])
    t = jnp.clip(xk_ext[d0 + 1] - y, lo_p, hi_p)

    # Fraction of each collision's number and bin-j mass going to d0;
    # the remainder goes to d0 + 1 (complements keep conservation exact).
    nu_lo, mu_lo = _partial_moments(lo_p, hi_p, n_lo_p, n_hi_p, t)
    nu_hi = 1.0 - nu_lo
    mu_hi = xbar_j - mu_lo

    # Accumulate into nbins + 2 destinations (last two collect overflow):
    # A[d, j] = bin-j mass reaching d, B[d, i] = partner count reaching d.
    # One scatter per array adds each pair's (d0, d0 + 1) shares as a
    # length-2 window at row d0; the two planes are then offset by a row.
    A2 = jnp.zeros((nbins + 1, nbins, 2)).at[d0, jp].add(
        jnp.stack([R_p * mu_lo, R_p * mu_hi], axis=-1), mode='promise_in_bounds')
    B2 = jnp.zeros((nbins + 1, nbins, 2)).at[d0, ip].add(
        jnp.stack([R_p * nu_lo, R_p * nu_hi], axis=-1), mode='promise_in_bounds')
    zero_row = jnp.zeros((1, nbins))
    A = (jnp.concatenate([A2[..., 0], zero_row])
         + jnp.concatenate([zero_row, A2[..., 1]]))
    B = (jnp.concatenate([B2[..., 0], zero_row])
         + jnp.concatenate([zero_row, B2[..., 1]]))
    gain_N = jnp.sum(B, axis=1)
    gain_M = A @ frac + B @ mbar

    dNdt = gain_N[:nbins] - loss_N
    dMdt_nodiag = gain_M[:nbins] - loss_N[:, None] * mbar

    dMdt = jnp.zeros_like(Mk).at[:, :icomp_nodiag].set(dMdt_nodiag)
    dM_overflow = jnp.zeros(Mk.shape[1]).at[:icomp_nodiag].set(
        jnp.sum(gain_M[nbins:], axis=0))

    return dNdt, dMdt, dM_overflow
