"""Tests for coagulation on grids that are not mass-doubling.

Tests cover:
    - Linear sub-bin reconstruction: non-negative, unit integral, exact mean
    - Linear scheme conservation: per-species mass (incl. overflow), number
    - Constant-kernel Smoluchowski analytic solution: error falls with
      resolution (TFL's grows at 80/160 bins)
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import ICOMP, ICOMP_NODIAG, SRTSO4, make_grid
from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.physics.coagulation_rates import _preprocess_concentrations
from tomas_jax.physics.coagulation_rates_linear import (
    calc_coagulation_rates_linear,
    linear_subbin_reconstruction,
)

XK0_1_7NM = 3.4728e-24
BOXVOL = 1.0e6


def grid(nbins):
    """API-style grid: 1.7 nm start, fixed diameter range, ratio 2^(40/nbins)."""
    return make_grid(nbins, XK0_1_7NM, 2.0 ** (40.0 / nbins))


# =========================================================================
# Linear sub-bin reconstruction
# =========================================================================

class TestLinearReconstruction:

    @pytest.mark.parametrize("nbins", [20, 40, 80, 160])
    @pytest.mark.parametrize("pos", [-0.1, 0.0, 0.05, 0.2, 0.5, 0.7, 0.95,
                                     0.999999, 1.0, 1.1])
    def test_density_valid(self, nbins, pos):
        """Density is non-negative, integrates to 1, and has mean xbar."""
        xk = grid(nbins)
        a, b = np.asarray(xk[:-1]), np.asarray(xk[1:])
        xbar = a + pos * (b - a)
        lo, hi, n_lo, n_hi = (np.asarray(v) for v in
                              linear_subbin_reconstruction(jnp.asarray(xbar), xk))
        assert np.all(hi > lo)
        assert np.all(n_lo >= 0.0) and np.all(n_hi >= 0.0)
        integral = 0.5 * (n_lo + n_hi) * (hi - lo)
        mean = (hi - lo) / 6.0 * (lo * (2 * n_lo + n_hi) + hi * (n_lo + 2 * n_hi))
        np.testing.assert_allclose(integral, 1.0, rtol=1e-9)
        np.testing.assert_allclose(mean, xbar, rtol=1e-9)

    def test_support_inside_bin_when_xbar_inside(self):
        xk = grid(80)
        a, b = np.asarray(xk[:-1]), np.asarray(xk[1:])
        for pos in (0.01, 0.3, 0.5, 0.8, 0.99):
            lo, hi, _, _ = linear_subbin_reconstruction(
                jnp.asarray(a + pos * (b - a)), xk)
            assert np.all(np.asarray(lo) >= a * (1 - 1e-12))
            assert np.all(np.asarray(hi) <= b * (1 + 1e-12))


# =========================================================================
# Conservation
# =========================================================================

def _random_state(nbins, seed, empty_top=0):
    rng = np.random.default_rng(seed)
    xk = grid(nbins)
    N = rng.lognormal(10.0, 1.0, nbins)
    N[:3] = 0.0                       # exercise the empty-bin preprocessing
    if empty_top:
        N[-empty_top:] = 0.0
    comp = rng.uniform(0.2, 1.0, (nbins, 3))
    x_geo = np.sqrt(np.asarray(xk[:-1]) * np.asarray(xk[1:]))
    M = np.zeros((nbins, ICOMP))
    M[:, :3] = comp * (N * x_geo / comp.sum(1))[:, None]
    kij = rng.uniform(0.5, 1.5, (nbins, nbins)) * 1e-12
    kij = 0.5 * (kij + kij.T)
    return xk, jnp.asarray(N), jnp.asarray(M), jnp.asarray(kij)


class TestLinearConservation:

    @pytest.mark.parametrize("nbins", [20, 40, 80, 160])
    def test_species_mass_conserved_with_overflow(self, nbins):
        xk, N, M, kij = _random_state(nbins, seed=nbins)
        dN, dM, ov = calc_coagulation_rates_linear(N, M, kij, xk, ICOMP_NODIAG)
        scale = jnp.sum(jnp.abs(dM), axis=0)
        resid = jnp.abs(jnp.sum(dM, axis=0) + ov)
        assert float(jnp.max(resid[:3] / scale[:3])) < 1e-12
        assert float(jnp.max(jnp.abs(dM[:, ICOMP_NODIAG:]))) == 0.0

    @pytest.mark.parametrize("nbins", [40, 80, 160])
    def test_one_particle_lost_per_collision(self, nbins):
        """With the top of the grid empty, dN_total = -(collision rate).

        Only the NEPS placeholder particles in the empty top bins can overflow,
        which is far below the tolerance.
        """
        xk, N, M, kij = _random_state(nbins, seed=1, empty_top=nbins // 4)
        dN, _, _ov = calc_coagulation_rates_linear(N, M, kij, xk, ICOMP_NODIAG)
        Ns, _ = _preprocess_concentrations(N, M, xk)
        Ns, K = np.asarray(Ns), np.asarray(kij)
        collisions = (np.tril(K * np.outer(Ns, Ns), -1).sum()
                      + 0.5 * (np.diag(K) * Ns ** 2).sum())
        np.testing.assert_allclose(float(jnp.sum(dN)), -collisions, rtol=1e-12)

    def test_jit(self):
        xk, N, M, kij = _random_state(80, seed=3)
        eager = calc_coagulation_rates_linear(N, M, kij, xk, ICOMP_NODIAG)
        jitted = jax.jit(calc_coagulation_rates_linear,
                         static_argnames="icomp_nodiag")(N, M, kij, xk,
                                                         icomp_nodiag=ICOMP_NODIAG)
        for e, j in zip(eager, jitted):
            np.testing.assert_allclose(np.asarray(j), np.asarray(e), rtol=1e-12)


# =========================================================================
# Constant-kernel analytic solution
# =========================================================================
# Exponential initial distribution n(x,0) = N0/x0 exp(-x/x0) under constant
# kernel K stays exponential:  n(x,t) = N0/(x0 s^2) exp(-x/(x0 s)),
# s = 1 + K N0 t / 2.

N0 = 1.0e4                                    # cm-3
X0 = np.pi / 6.0 * 1770.0 * (100e-9) ** 3     # kg, ~100 nm sulfate particle
TAU_END = 10.0                                # N falls 6x
DT, NSTEPS = 60.0, 1440                       # 24 h
KCONST = TAU_END / (N0 * DT * NSTEPS)         # cm3/s


def _exact_bins(xk, tau):
    s = 1.0 + tau / 2.0
    xs = X0 * s
    a, b = np.asarray(xk[:-1]), np.asarray(xk[1:])
    Nk = N0 / s * (np.exp(-a / xs) - np.exp(-b / xs))
    Mk = N0 / s * xs * ((a / xs + 1) * np.exp(-a / xs) - (b / xs + 1) * np.exp(-b / xs))
    return Nk, Mk


def _run_constant_kernel(rates_fn, nbins):
    xk = grid(nbins)
    Nk0, Mk0 = _exact_bins(xk, 0.0)
    N = jnp.asarray(Nk0) * BOXVOL
    M = jnp.zeros((nbins, ICOMP)).at[:, SRTSO4].set(jnp.asarray(Mk0) * BOXVOL)
    kij = jnp.full((nbins, nbins), KCONST / BOXVOL)

    def step(carry, _):
        N, M = carry
        dN, dM, _ov = rates_fn(N, M, kij, xk, ICOMP_NODIAG)
        N = jnp.maximum(N + DT * dN, 0.0)
        M = jnp.maximum(M + DT * dM, 0.0)
        return mnfix_jax(N, M, xk, ICOMP_NODIAG), None

    (N, _M), _ = jax.jit(
        lambda N, M: jax.lax.scan(step, (N, M), None, length=NSTEPS))(N, M)
    N = np.asarray(N) / BOXVOL
    Ne, _ = _exact_bins(xk, TAU_END)
    populated = Ne > 1e-3 * Ne.max()
    return np.sum(np.abs(N - Ne)[populated]) / np.sum(Ne[populated])


class TestConstantKernelAnalytic:

    @pytest.mark.coag_only
    def test_linear_error_falls_with_resolution(self):
        err = {nb: _run_constant_kernel(calc_coagulation_rates_linear, nb)
               for nb in (40, 80, 160)}
        assert err[160] < err[80] < err[40]
        assert err[80] < 0.01, err
        assert err[160] < 0.005, err
