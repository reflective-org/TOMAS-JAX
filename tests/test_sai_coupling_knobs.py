"""Tests for the step kwargs the coupled SAI plume model drives.

Tests cover:
    - coag_kernel_scale: scales the coagulation kernel (2x kernel -> 2x
      number loss over one forward-Euler substep); 1.0 equals omitting it
      bitwise, through make_step and coag_euler_step
    - nuc_org_mass_frac: 0.0 puts all nucleated mass in SRTSO4 and leaves
      the organic slot alone; sulfur is conserved over a nucleation-only
      step; omitted equals 0.1 bitwise; a traced value works under jit
    - make_step with water_every_process + WATER_SCHEME_H2SO4 + both knobs,
      nucleation/coagulation/condensation on the 80-bin grid: no warnings,
      finite after one step

Sulfur bookkeeping convention (as in the Fortran and in condensation): gas
H2SO4 (Gc[SRTSO4]) and particulate sulfate (Mk[:, SRTSO4]) are exchanged
kg-for-kg in the normal nucleation path, so total sulfur is
Gc[SRTSO4] + sum(Mk[:, SRTSO4]) in kg. Only the gas-limited (clamped) path
converts H2SO4 -> SO4 with 96/98, which conserves moles of S instead.
"""
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import (
    NBINS, ICOMP, N_GAS_SPECIES, SRTSO4, SRTORGLAST, SRTH2O,
    MW_H2SO4, MW_SO4, AVOGADRO, make_grid_80bin,
)
from tomas_jax.physics.nucleation import (
    nucleation_step, zhao2024_nucleation_step, NUC_ORG_MASS_FRAC_DEFAULT,
)
from tomas_jax.physics.water_equilibrium import WATER_SCHEME_H2SO4
from tomas_jax.solvers.condensation import make_step
from tomas_jax.solvers.diffrax import coag_euler_step

BOXVOL = 1e6  # cm3

# Stratospheric box (B1-like): cold, low pressure, dry
T_STRAT, P_STRAT, RH_STRAT = 210.0, 5000.0, 0.017


def _gc(h2so4_molec_cm3):
    return jnp.zeros(N_GAS_SPECIES).at[SRTSO4].set(
        h2so4_molec_cm3 * BOXVOL * MW_H2SO4 / 1000.0 / AVOGADRO)


@pytest.fixture
def state(xk):
    """1e6 per bin, sulfate at bin centre, a little organic everywhere."""
    Nk = jnp.ones(NBINS) * 1e6
    m = jnp.sqrt(xk[:-1] * xk[1:])
    Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(0.95 * Nk * m)
    Mk = Mk.at[:, SRTORGLAST].set(0.05 * Nk * m)
    return Nk, Mk, _gc(1e7), xk


def _assert_trees_equal(a, b):
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


# =========================================================================
# coag_kernel_scale
# =========================================================================

class TestCoagKernelScale:

    def test_double_kernel_doubles_number_loss(self, state):
        """One forward-Euler substep is linear in the kernel: 2x -> 2x dN."""
        Nk, Mk, Gc, xk = state
        step = make_step(['coagulation'], n_coag_substeps=1)
        args = (xk, 298.0, 101325.0, BOXVOL, 0.5, 1.0, 1.0)
        Nk1, _, _ = step(Nk, Mk, Gc, *args, coag_kernel_scale=1.0)
        Nk2, _, _ = step(Nk, Mk, Gc, *args, coag_kernel_scale=2.0)
        loss1 = float(jnp.sum(Nk) - jnp.sum(Nk1))
        loss2 = float(jnp.sum(Nk) - jnp.sum(Nk2))
        assert loss1 > 0.0
        np.testing.assert_allclose(loss2 / loss1, 2.0, rtol=1e-6)

    def test_zero_kernel_is_noop_on_number(self, state):
        Nk, Mk, Gc, xk = state
        step = make_step(['coagulation'], n_coag_substeps=1)
        Nk0, _, _ = step(Nk, Mk, Gc, xk, 298.0, 101325.0, BOXVOL, 0.5, 1.0,
                         60.0, coag_kernel_scale=0.0)
        np.testing.assert_allclose(float(jnp.sum(Nk0)), float(jnp.sum(Nk)),
                                   rtol=1e-12)

    def test_unit_scale_equals_omitted_make_step(self, state):
        Nk, Mk, Gc, xk = state
        step = make_step(['coagulation'])
        args = (Nk, Mk, Gc, xk, 298.0, 101325.0, BOXVOL, 0.5, 1.0, 600.0)
        _assert_trees_equal(step(*args), step(*args, coag_kernel_scale=1.0))

    def test_unit_scale_equals_omitted_coag_euler_step(self, state):
        Nk, Mk, _, xk = state
        a = coag_euler_step(Nk, Mk, xk, 298.0, 101325.0, BOXVOL, dt=600.0)
        b = coag_euler_step(Nk, Mk, xk, 298.0, 101325.0, BOXVOL, dt=600.0,
                            coag_kernel_scale=1.0)
        _assert_trees_equal(a, b)

    def test_traced_scale_under_jit(self, state):
        """A traced scale gives the same result as the Python float."""
        Nk, Mk, Gc, xk = state
        step = make_step(['coagulation'])
        args = (Nk, Mk, Gc, xk, 298.0, 101325.0, BOXVOL, 0.5, 1.0, 600.0)
        _assert_trees_equal(step(*args, coag_kernel_scale=2.0),
                            step(*args, coag_kernel_scale=jnp.float64(2.0)))


# =========================================================================
# nuc_org_mass_frac
# =========================================================================

_NUC_ARGS = (278.0, 101325.0, BOXVOL, 60.0,   # temp, pres, boxvol, dt
             1e7, 1e9, 3.0)                   # org_conc, nh3_conc, fion


def _sulfur_kg(Mk, Gc):
    return float(Gc[SRTSO4] + jnp.sum(Mk[:, SRTSO4]))


def _sulfur_mol(Mk, Gc):
    return float(Gc[SRTSO4] / MW_H2SO4 + jnp.sum(Mk[:, SRTSO4]) / MW_SO4)


class TestNucOrgMassFrac:

    def test_default_constant_is_historical_split(self):
        assert NUC_ORG_MASS_FRAC_DEFAULT == 0.1

    @pytest.mark.parametrize("step_fn", [nucleation_step, zhao2024_nucleation_step])
    def test_zero_puts_all_mass_in_so4(self, state, step_fn):
        Nk, Mk, Gc, xk = state
        Nk1, Mk1, Gc1 = step_fn(Nk, Mk, Gc, xk, *_NUC_ARGS, nuc_org_mass_frac=0.0)
        dN = float(Nk1[0] - Nk[0])
        assert dN > 0.0
        # Organic slot untouched, bitwise
        np.testing.assert_array_equal(np.asarray(Mk1[:, SRTORGLAST]),
                                      np.asarray(Mk[:, SRTORGLAST]))
        # All of the cluster mass is sulfate: dM_SO4 = dN * m_cluster
        from tomas_jax.physics.nucleation import _MNUC
        np.testing.assert_allclose(float(Mk1[0, SRTSO4] - Mk[0, SRTSO4]),
                                   dN * _MNUC, rtol=1e-12)

    @pytest.mark.parametrize("f", [0.0, 0.1, 0.5])
    def test_sulfur_conserved_unclamped(self, state, f):
        """Normal path: gas lost == particulate SO4 gained, kg-for-kg."""
        Nk, Mk, Gc, xk = state
        Nk1, Mk1, Gc1 = nucleation_step(Nk, Mk, Gc, xk, *_NUC_ARGS,
                                        nuc_org_mass_frac=f)
        assert float(Gc1[SRTSO4]) > 0.0                 # not gas-limited
        assert float(Gc1[SRTSO4]) < float(Gc[SRTSO4])   # nucleation happened
        np.testing.assert_allclose(_sulfur_kg(Mk1, Gc1), _sulfur_kg(Mk, Gc),
                                   rtol=1e-14)

    def test_zero_conserves_total_mass(self, state):
        """With f=0 no organic mass is created: gas lost == particle mass gained."""
        Nk, Mk, Gc, xk = state
        _, Mk1, Gc1 = nucleation_step(Nk, Mk, Gc, xk, *_NUC_ARGS,
                                      nuc_org_mass_frac=0.0)
        # Difference per element (summing Mk first loses the tiny delta in
        # the ~1e-5 kg total)
        d_particle = float(jnp.sum(Mk1 - Mk))
        d_gas = float(Gc[SRTSO4] - Gc1[SRTSO4])
        assert d_gas > 0.0
        np.testing.assert_allclose(d_particle, d_gas, rtol=1e-12)

    @pytest.mark.parametrize("f", [0.0, 0.1])
    def test_sulfur_moles_conserved_clamped(self, xk, f):
        """Gas-limited path: all H2SO4 used, added as SO4 (96/98)."""
        Nk = jnp.ones(NBINS) * 1e6
        Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(
            Nk * jnp.sqrt(xk[:-1] * xk[1:]))
        Gc = _gc(1e3)
        _, Mk1, Gc1 = nucleation_step(
            Nk, Mk, Gc, xk, 278.0, 101325.0, BOXVOL, 3600.0, 1e15, 1e9, 3.0,
            nuc_org_mass_frac=f)
        assert float(Gc1[SRTSO4]) == 0.0
        np.testing.assert_allclose(_sulfur_mol(Mk1, Gc1), _sulfur_mol(Mk, Gc),
                                   rtol=1e-14)

    def test_make_step_nucleation_only_zero(self, state):
        """Through make_step (substeps + MNFIX): organic total and S conserved."""
        Nk, Mk, Gc, xk = state
        step = make_step(['nucleation'])
        Nk1, Mk1, Gc1 = step(Nk, Mk, Gc, xk, *_NUC_ARGS[:3], 0.5, 1.0, 60.0,
                             org_conc=1e7, nh3_conc=1e9, fion=3.0,
                             nuc_org_mass_frac=0.0)
        assert float(jnp.sum(Nk1)) > float(jnp.sum(Nk))
        np.testing.assert_allclose(float(jnp.sum(Mk1[:, SRTORGLAST])),
                                   float(jnp.sum(Mk[:, SRTORGLAST])), rtol=1e-12)
        np.testing.assert_allclose(_sulfur_kg(Mk1, Gc1), _sulfur_kg(Mk, Gc),
                                   rtol=1e-12)

    @pytest.mark.parametrize("nucl_scheme", ['ricco_dunne', 'zhao2024'])
    def test_omitted_equals_default_bitwise(self, state, nucl_scheme):
        Nk, Mk, Gc, xk = state
        step = make_step(['nucleation', 'coagulation', 'condensation'],
                         nucl_scheme=nucl_scheme)
        args = (Nk, Mk, Gc, xk, 278.0, 101325.0, BOXVOL, 0.5, 1.0, 600.0)
        kw = dict(org_conc=1e7, nh3_conc=1e9, fion=3.0)
        _assert_trees_equal(step(*args, **kw),
                            step(*args, **kw, nuc_org_mass_frac=0.1))

    def test_omitted_equals_default_bitwise_step_fn(self, state):
        Nk, Mk, Gc, xk = state
        _assert_trees_equal(
            nucleation_step(Nk, Mk, Gc, xk, *_NUC_ARGS),
            nucleation_step(Nk, Mk, Gc, xk, *_NUC_ARGS, nuc_org_mass_frac=0.1))

    def test_traced_fraction_under_jit(self, state):
        Nk, Mk, Gc, xk = state
        step = make_step(['nucleation'])
        args = (Nk, Mk, Gc, xk, 278.0, 101325.0, BOXVOL, 0.5, 1.0, 60.0)
        kw = dict(org_conc=1e7, nh3_conc=1e9, fion=3.0)
        _assert_trees_equal(step(*args, **kw, nuc_org_mass_frac=0.0),
                            step(*args, **kw, nuc_org_mass_frac=jnp.float64(0.0)))
        # vmap over the fraction: organic mass added grows with f
        fs = jnp.array([0.0, 0.1, 0.3])
        _, Mk_v, _ = jax.vmap(
            lambda f: step(*args, **kw, nuc_org_mass_frac=f))(fs)
        org = np.asarray(jnp.sum(Mk_v[:, :, SRTORGLAST], axis=1))
        assert org[0] == pytest.approx(float(jnp.sum(Mk[:, SRTORGLAST])), rel=1e-12)
        assert org[0] < org[1] < org[2]


# =========================================================================
# Combined: the coupled-model configuration at 80 bins
# =========================================================================

def _state_80bin():
    xk = make_grid_80bin()
    nb = xk.shape[0] - 1
    m = jnp.sqrt(xk[:-1] * xk[1:])
    dp_um = 1e6 * jnp.cbrt(6.0 * m / (1770.0 * np.pi))
    # Background stratospheric mode, ~10 /cm3 around 0.15 um
    dlog = jnp.log(xk[1:] / xk[:-1]) / 3.0
    Nk = (10.0 / (np.sqrt(2 * np.pi) * np.log(1.6))
          * jnp.exp(-(jnp.log(dp_um) - np.log(0.15)) ** 2 / (2 * np.log(1.6) ** 2))
          * dlog * BOXVOL)
    Nk = jnp.maximum(Nk, 1e-3)
    Mk = jnp.zeros((nb, ICOMP)).at[:, SRTSO4].set(Nk * m)
    return Nk, Mk, _gc(1e8), xk


class TestCoupledConfiguration:

    def test_80bin_all_options_one_step(self):
        Nk, Mk, Gc, xk = _state_80bin()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            step = make_step(
                ['nucleation', 'coagulation', 'condensation'],
                cond_method='ppm_jit', nucl_scheme='ricco_dunne',
                water_scheme=WATER_SCHEME_H2SO4, water_every_process=True,
            )
            Nk1, Mk1, Gc1 = step(
                Nk, Mk, Gc, xk, T_STRAT, P_STRAT, BOXVOL, RH_STRAT, 1.0, 600.0,
                org_conc=0.0, nh3_conc=0.0, fion=3.0, fn_scale=1.0,
                coag_kernel_scale=1.5, nuc_org_mass_frac=0.0,
            )
        assert Nk1.shape == (80,) and Mk1.shape == (80, ICOMP)
        for a in (Nk1, Mk1, Gc1):
            assert bool(jnp.all(jnp.isfinite(a)))
        assert float(jnp.min(Nk1)) >= 0.0
        # H2SO4 scheme set water on the particles
        assert float(jnp.sum(Mk1[:, SRTH2O])) > 0.0
        # Nothing was put in the organic slot
        assert float(jnp.sum(Mk1[:, SRTORGLAST])) == 0.0
