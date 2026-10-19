"""Tests for the selectable water uptake schemes and water_every_process.

Tests cover:
    - tabazadeh_h2so4_wt: matches the numpy radiative-forcing reference,
      monotonic in RH, bounded, sane extrapolation at 298 K, JIT
    - calc_equilibrium_water_h2so4: H2SO4 water, organics on the bisulfate
      ratio, no water on NH4
    - equilibrate_water: static and traced scheme selection agree
    - Solver options: water_every_process gives coag-only runs an RH
      dependence; water_scheme reaches condensation; defaults unchanged
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import (
    ICOMP, N_GAS_SPECIES, SRTSO4, SRTORG1, SRTNH4, SRTH2O,
    MW_H2SO4, AVOGADRO, make_grid,
)
from tomas_jax.physics.radiative_forcing import h2so4_equilibrium_wt
from tomas_jax.physics.water_equilibrium import (
    WATER_SCHEME_BISULFATE, WATER_SCHEME_H2SO4,
    calc_equilibrium_water, calc_equilibrium_water_h2so4,
    equilibrate_water, tabazadeh_h2so4_wt, water_uptake_sulfate,
)
from tomas_jax.solvers.condensation import (
    make_step, run_combined_scan_ppm, run_condensation_scan,
)


# =========================================================================
# Tabazadeh weight percent
# =========================================================================

class TestTabazadehWt:

    @pytest.mark.parametrize("temp", [190.0, 215.0, 240.0, 260.0])
    @pytest.mark.parametrize("rh", [1.0, 10.0, 50.0, 90.0, 98.0])
    def test_matches_numpy_reference(self, temp, rh):
        assert float(tabazadeh_h2so4_wt(temp, rh)) == pytest.approx(
            h2so4_equilibrium_wt(temp, rh), abs=1e-9)

    @pytest.mark.parametrize("temp", [200.0, 250.0, 298.0])
    def test_decreases_with_rh(self, temp):
        wt = np.array([float(tabazadeh_h2so4_wt(temp, rh))
                       for rh in (5, 20, 40, 60, 80, 95)])
        assert np.all(np.diff(wt) < 0)

    @pytest.mark.parametrize("temp", [180.0, 298.0, 350.0])
    @pytest.mark.parametrize("rh", [0.0, 100.0])
    def test_bounded(self, temp, rh):
        wt = float(tabazadeh_h2so4_wt(temp, rh))
        assert 10.0 <= wt <= 80.0

    def test_room_temperature_extrapolation(self):
        """25 C water-activity data give ~43 wt% at a_w = 0.5."""
        assert 40.0 < float(tabazadeh_h2so4_wt(298.15, 50.0)) < 47.0

    def test_jit(self):
        f = jax.jit(tabazadeh_h2so4_wt)
        assert float(f(220.0, 30.0)) == pytest.approx(
            float(tabazadeh_h2so4_wt(220.0, 30.0)), rel=1e-14)


# =========================================================================
# H2SO4 water and scheme selection
# =========================================================================

def _mixed_mk(nbins=6):
    rng = np.random.default_rng(0)
    Mk = np.zeros((nbins, ICOMP))
    Mk[:, SRTSO4] = rng.uniform(1e-18, 1e-16, nbins)
    Mk[:, SRTORG1] = rng.uniform(1e-19, 1e-17, nbins)
    Mk[:, SRTNH4] = rng.uniform(1e-19, 1e-17, nbins)
    return jnp.asarray(Mk)


class TestH2SO4Water:

    def test_sulfate_water_from_wt(self):
        Mk = jnp.zeros((3, ICOMP)).at[:, SRTSO4].set(jnp.array([1.0, 2.0, 3.0]))
        rh, temp = 0.4, 220.0
        wt = float(tabazadeh_h2so4_wt(temp, rh * 100))
        expected = np.array([1.0, 2.0, 3.0]) * 98.0 / 96.0 * (100.0 / wt - 1.0)
        np.testing.assert_allclose(
            np.asarray(calc_equilibrium_water_h2so4(Mk, rh, temp)[:, SRTH2O]),
            expected, rtol=1e-14)

    def test_organics_use_bisulfate_ratio_and_nh4_takes_none(self):
        rh, temp = 0.7, 250.0
        Mk = jnp.zeros((2, ICOMP)).at[0, SRTORG1].set(1.0).at[1, SRTNH4].set(1.0)
        water = np.asarray(calc_equilibrium_water_h2so4(Mk, rh, temp)[:, SRTH2O])
        wr = float(water_uptake_sulfate(rh * 100))
        assert water[0] == pytest.approx(wr - 1.0, rel=1e-14)
        assert water[1] == 0.0

    def test_sulfuric_acid_wetter_than_bisulfate(self):
        Mk = jnp.zeros((1, ICOMP)).at[0, SRTSO4].set(1.0)
        for rh in (0.1, 0.5, 0.9):
            h2so4 = float(calc_equilibrium_water_h2so4(Mk, rh, 260.0)[0, SRTH2O])
            bisulf = float(calc_equilibrium_water(Mk, rh)[0, SRTH2O])
            assert h2so4 > bisulf

    def test_only_water_column_changes(self):
        Mk = _mixed_mk()
        out = calc_equilibrium_water_h2so4(Mk, 0.5, 230.0)
        np.testing.assert_array_equal(np.asarray(out[:, :SRTH2O]),
                                      np.asarray(Mk[:, :SRTH2O]))


class TestEquilibrateWater:

    @pytest.mark.parametrize("scheme,ref", [
        (WATER_SCHEME_BISULFATE, lambda Mk, rh, T: calc_equilibrium_water(Mk, rh)),
        (WATER_SCHEME_H2SO4, calc_equilibrium_water_h2so4),
    ])
    def test_static_selection(self, scheme, ref):
        Mk = _mixed_mk()
        np.testing.assert_array_equal(
            np.asarray(equilibrate_water(Mk, 0.6, 240.0, scheme)),
            np.asarray(ref(Mk, 0.6, 240.0)))

    @pytest.mark.parametrize("scheme", [WATER_SCHEME_BISULFATE, WATER_SCHEME_H2SO4])
    def test_traced_selection_matches_static(self, scheme):
        Mk = _mixed_mk()
        traced = jax.jit(equilibrate_water)(Mk, 0.6, 240.0, jnp.float64(scheme))
        static = equilibrate_water(Mk, 0.6, 240.0, scheme)
        np.testing.assert_allclose(np.asarray(traced), np.asarray(static), rtol=1e-14)


# =========================================================================
# Solver options
# =========================================================================

BOXVOL = 1.0e6


def _state(n_cm3=1e5, gmd=100e-9):
    xk = make_grid(40, 3.4728e-24, 2.0)
    m = jnp.sqrt(xk[:-1] * xk[1:])
    dp = jnp.cbrt(m / 1770.0 * 6.0 / np.pi)
    Nk = (n_cm3 / (np.sqrt(2 * np.pi) * np.log(1.6))
          * jnp.exp(-(jnp.log(dp) - np.log(gmd)) ** 2 / (2 * np.log(1.6) ** 2))
          * jnp.log(xk[1:] / xk[:-1]) / 3.0 * BOXVOL)
    Mk = jnp.zeros((40, ICOMP)).at[:, SRTSO4].set(Nk * m)
    Gc = jnp.zeros(N_GAS_SPECIES).at[SRTSO4].set(
        1e7 * BOXVOL * MW_H2SO4 / 1000.0 / AVOGADRO)
    return Nk, Mk, Gc, xk


def _coag_only(rh, nsteps=60, **opts):
    Nk, Mk, Gc, xk = _state()
    step = make_step(['coagulation'], **opts)
    for _ in range(nsteps):
        Nk, Mk, Gc = step(Nk, Mk, Gc, xk, 298.0, 101325.0, BOXVOL, rh, 1.0, 60.0)
    return float(jnp.sum(Nk)), Mk


class TestWaterEveryProcess:

    def test_default_coag_only_ignores_rh(self):
        """Legacy behaviour (kept as the library default): coag-only is dry."""
        n_dry, Mk = _coag_only(0.1)
        n_wet, _ = _coag_only(0.9)
        assert n_dry == n_wet
        assert float(jnp.sum(Mk[:, SRTH2O])) == 0.0

    @pytest.mark.parametrize("scheme", [WATER_SCHEME_BISULFATE, WATER_SCHEME_H2SO4])
    def test_coag_only_sees_rh_when_enabled(self, scheme):
        n10, _ = _coag_only(0.1, water_every_process=True, water_scheme=scheme)
        n90, Mk90 = _coag_only(0.9, water_every_process=True, water_scheme=scheme)
        assert float(jnp.sum(Mk90[:, SRTH2O])) > 0.0
        # A single ~100 nm mode coagulates more slowly once wet (larger,
        # out of the slip regime), so more particles remain at high RH.
        assert n90 > n10

    def test_combined_scan_option_changes_result_default_does_not(self):
        Nk, Mk, Gc, xk = _state()
        prod = 1e7 * BOXVOL * MW_H2SO4 / 1000.0 / AVOGADRO
        args = (Nk, Mk, Gc, xk, 298.0, 101325.0, BOXVOL, 0.9, 1.0, 60.0, 30, prod)
        base = run_combined_scan_ppm(*args)
        explicit = run_combined_scan_ppm(*args, water_scheme=WATER_SCHEME_BISULFATE,
                                         water_every_process=False)
        every = run_combined_scan_ppm(*args, water_every_process=True)
        for a, b in zip(base[:3], explicit[:3]):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
        assert not np.array_equal(np.asarray(base[0]), np.asarray(every[0]))

    def test_condensation_uses_selected_scheme(self):
        Nk, Mk, Gc, xk = _state()
        args = (Nk, Mk, Gc, xk, 220.0, 5000.0, BOXVOL, 0.5, 1.0, 60.0, 5, 0.0)
        water = {}
        for scheme in (WATER_SCHEME_BISULFATE, WATER_SCHEME_H2SO4):
            _, Mk_f, _, _ = run_condensation_scan(*args, water_scheme=scheme)
            water[scheme] = float(jnp.sum(Mk_f[:, SRTH2O]))
        assert water[WATER_SCHEME_H2SO4] > 2.0 * water[WATER_SCHEME_BISULFATE] > 0.0
