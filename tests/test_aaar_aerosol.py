"""Tests for the AAAR tutorial aerosol step (docs/aaar/).

Covers physics/soa_partitioning.py (vbs-jax adapter), physics/
heterogeneous.py (N2O5 uptake) and box/aerosol_step.py (outer step).
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import (
    ICOMP, N_GAS_SPECIES, SRTSO4, SRTSOA1, SRTNIT, SRTORG_NUC, SRTH2O,
    N_SOA_CLASSES, PI, R_GAS, AVOGADRO, MW_H2SO4, MW_N2O5, make_grid,
)
from tomas_jax.physics.density import calc_density
from tomas_jax.physics.heterogeneous import n2o5_uptake
from tomas_jax.physics.ppm_remap import BIN_TRANSFERS
from tomas_jax.physics.soa_partitioning import soa_partition_step
from tomas_jax.box.aerosol_step import (
    AerosolForcing, AerosolState, SOAParams, aerosol_outer_step,
)
from tomas_jax.box.config import AerosolStepConfig
from vbs_jax import vbs_equilibrium

BOXVOL = 1.0e10            # cm^3 (tutorial default)
BOXVOL_M3 = BOXVOL * 1e-6
NBINS = 40
XK = make_grid(NBINS)
XMID = jnp.sqrt(XK[:-1] * XK[1:])
ALL = ('nucleation', 'coagulation', 'h2so4_condensation', 'soa_partitioning')

soa_step = jax.jit(soa_partition_step, static_argnames=('bin_transfer',))


def kg(ug_m3):
    """ug/m^3 -> kg/cell."""
    return ug_m3 * 1e-9 * BOXVOL_M3


def ug(kg_cell):
    """kg/cell -> ug/m^3."""
    return float(kg_cell) / BOXVOL_M3 * 1e9


def lognormal(n_cm3, gmd, gsd, rho=1770.0):
    """Lognormal number per bin [#/cell] on the 40-bin grid."""
    dp = jnp.cbrt(XMID / rho * 6.0 / PI)
    pdf = jnp.exp(-0.5 * (jnp.log(dp / gmd) / jnp.log(gsd)) ** 2)
    return n_cm3 * BOXVOL * pdf / jnp.sum(pdf)


def sulfate_seed(n_cm3, gmd, gsd):
    Nk = lognormal(n_cm3, gmd, gsd)
    return Nk, jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(Nk * XMID)


def soa_params(cstar):
    """Test classes: dHvap = 131 - 11 log10 c* [kJ/mol] (Epstein et al.
    2010, SOM-TOMAS form), MW 200 g/mol, alpha 1."""
    cstar = jnp.atleast_1d(jnp.asarray(cstar, dtype=jnp.float64))
    return SOAParams(cstar_ref=cstar, dhvap=131.0 - 11.0 * jnp.log10(cstar),
                     mw=jnp.full(cstar.shape, 200.0),
                     alpha=jnp.float64(1.0))


def make_state(Nk, Mk, Gc=None, temp=298.0, rh=0.5):
    Gc = jnp.zeros(N_GAS_SPECIES) if Gc is None else Gc
    f = jnp.float64
    return AerosolState(Nk, Mk, Gc, XK, f(temp), f(101325.0), f(rh),
                        f(BOXVOL))


def make_forcing(K, h2so4_rate=0.0, org_rate=None, nitrate_rate=0.0,
                 nitrate_weights=None, **nucl):
    return AerosolForcing(
        h2so4_rate=jnp.float64(h2so4_rate),
        org_rate=jnp.zeros(K) if org_rate is None else jnp.asarray(org_rate),
        nitrate_rate=jnp.float64(nitrate_rate),
        nitrate_weights=(jnp.zeros(NBINS) if nitrate_weights is None
                         else jnp.asarray(nitrate_weights)),
        **{k: jnp.float64(v) for k, v in nucl.items()})


def soa_particle(Mk, K):
    return np.array([ug(jnp.sum(Mk[:, SRTSOA1 + j])) for j in range(K)])


def soa_gas(Gc, K):
    return np.array([ug(Gc[SRTSOA1 + j]) for j in range(K)])


def run_soa(Nk, Mk, Gc, cstar, dt, nsteps, bin_transfer='ppm'):
    p = soa_params(cstar)
    for _ in range(nsteps):
        Nk, Mk, Gc, diag = soa_step(Nk, Mk, Gc, XK, 298.0, 101325.0, BOXVOL,
                                    dt, p.cstar_ref, p.dhvap, p.mw, p.alpha,
                                    bin_transfer=bin_transfer)
    return Nk, Mk, Gc, diag


def large_seed_with_oa(nonvol_ug):
    """1.5 um seeds (Kelvin < 0.5%) carrying non-volatile OA in SRTORG_NUC."""
    Nk, Mk = sulfate_seed(20.0, 1.5e-6, 1.2)
    share = Mk[:, SRTSO4] / jnp.sum(Mk[:, SRTSO4])
    return Nk, Mk.at[:, SRTORG_NUC].set(share * kg(nonvol_ug))


# =========================================================================
# (a) zero-vapour stability
# =========================================================================

def test_zero_vapour_stability_12h():
    """12 h, no sources: no NaN, dry mass conserved, N changes only by coag."""
    Nk = lognormal(5000.0, 50e-9, 1.6)
    Mk = (jnp.zeros((NBINS, ICOMP))
          .at[:, SRTSO4].set(0.6 * Nk * XMID)
          .at[:, SRTORG_NUC].set(0.3 * Nk * XMID)
          .at[:, SRTNIT].set(0.1 * Nk * XMID))
    st0 = make_state(Nk, Mk)
    p = soa_params([1.0, 100.0])
    f = make_forcing(2)

    def dry_total(st):
        return float(jnp.sum(st.Mk) - jnp.sum(st.Mk[:, SRTH2O])
                     + jnp.sum(st.Gc))

    def run(processes):
        st, n_tot = st0, [float(jnp.sum(Nk))]
        for _ in range(144):
            st, diag = aerosol_outer_step(st, f, p, 300.0,
                                          processes=processes)
            n_tot.append(float(jnp.sum(st.Nk)))
        return st, np.array(n_tot), diag

    st, n_all, diag = run(ALL)
    _, n_coag, _ = run(('coagulation',))

    assert not bool(jnp.any(jnp.isnan(st.Mk)) | jnp.any(jnp.isnan(st.Nk)))
    assert bool(diag.soa_time_integrated)
    assert float(diag.j_nuc) == 0.0
    assert dry_total(st) == pytest.approx(dry_total(st0), rel=1e-10)
    assert np.all(np.diff(n_all) <= 1e-12 * n_all[0])
    assert n_all[-1] < 0.9 * n_all[0]          # coagulation did act
    np.testing.assert_allclose(n_all, n_coag, rtol=1e-10)


# =========================================================================
# (b), (c) equilibrium limits
# =========================================================================

@pytest.mark.parametrize("bin_transfer", BIN_TRANSFERS)
@pytest.mark.parametrize("start_in_gas", [True, False])
def test_single_class_pankow_equilibrium(start_in_gas, bin_transfer):
    """C_OA = C_nv + C_tot/(1 + c*/C_OA), approached from either side."""
    nonvol, c_tot, cstar = 2.0, 10.0, 10.0
    Nk, Mk = large_seed_with_oa(nonvol)
    Gc = jnp.zeros(N_GAS_SPECIES)
    if start_in_gas:
        Gc = Gc.at[SRTSOA1].set(kg(c_tot))
    else:
        Mk = Mk.at[:, SRTSOA1].set(Mk[:, SRTORG_NUC] * c_tot / nonvol)

    Nk, Mk, Gc, diag = run_soa(Nk, Mk, Gc, [cstar], 600.0, 24, bin_transfer)

    c_oa = nonvol + c_tot
    for _ in range(500):
        c_oa = nonvol + c_tot / (1.0 + cstar / c_oa)
    assert float(diag.time_integrated) == pytest.approx(600.0, rel=1e-12)
    assert soa_particle(Mk, 1)[0] == pytest.approx(c_oa - nonvol, rel=0.01)
    assert soa_particle(Mk, 1)[0] + soa_gas(Gc, 1)[0] == pytest.approx(
        c_tot, rel=1e-10)


@pytest.mark.parametrize("bin_transfer", BIN_TRANSFERS)
def test_multi_class_matches_vbs_equilibrium(bin_transfer):
    """6 Donahue classes relax to vbs_jax.vbs_equilibrium."""
    nonvol = 1.0
    cstar = [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]
    c_tot = np.array([0.5, 1.0, 2.0, 4.0, 8.0, 16.0])
    Nk, Mk = large_seed_with_oa(nonvol)
    Gc = jnp.zeros(N_GAS_SPECIES).at[SRTSOA1:SRTSOA1 + 6].set(kg(c_tot))

    Nk, Mk, Gc, _ = run_soa(Nk, Mk, Gc, cstar, 600.0, 24, bin_transfer)

    c_part, _ = vbs_equilibrium(jnp.asarray(c_tot) * 1e-9,
                                jnp.asarray(cstar), nonvol * 1e-9)
    want = np.asarray(c_part) * 1e9
    got = soa_particle(Mk, 6)
    np.testing.assert_allclose(got, want, rtol=0.01)
    assert got.sum() == pytest.approx(want.sum(), rel=0.01)


# =========================================================================
# (d) zero-OA bootstrap
# =========================================================================

def test_zero_oa_bootstrap():
    """Pure sulfate seed: low c* condenses, high c* stays in the gas."""
    Nk, Mk = sulfate_seed(1000.0, 100e-9, 1.6)
    Gc = jnp.zeros(N_GAS_SPECIES).at[SRTSOA1:SRTSOA1 + 2].set(kg(2.0))
    st = make_state(Nk, Mk, Gc)
    p = soa_params([0.01, 1000.0])
    f = make_forcing(2)
    for _ in range(6):
        st, diag = aerosol_outer_step(st, f, p, 300.0,
                                      processes=('soa_partitioning',))
    assert not bool(jnp.any(jnp.isnan(st.Mk)))
    assert bool(diag.soa_time_integrated)
    part = soa_particle(st.Mk, 2)
    assert part[0] > 0.5 * 2.0                 # c* = 0.01 mostly condensed
    assert part[1] < 0.01 * 2.0                # c* = 1000 stays gas


def test_zero_oa_high_cstar_never_spikes():
    """Alone on a pure seed, c* = 1000 vapour never transiently condenses."""
    Nk, Mk = sulfate_seed(1000.0, 100e-9, 1.6)
    st = make_state(Nk, Mk, jnp.zeros(N_GAS_SPECIES).at[SRTSOA1].set(kg(2.0)))
    p = soa_params([1000.0])
    f = make_forcing(1)
    for _ in range(30):
        st, _ = aerosol_outer_step(st, f, p, 60.0,
                                   processes=('soa_partitioning',))
        assert soa_particle(st.Mk, 1)[0] < 1e-3 * 2.0


# =========================================================================
# (e) per-class mass conservation over an outer step
# =========================================================================

@pytest.mark.parametrize("bin_transfer", BIN_TRANSFERS)
def test_class_mass_conservation_all_processes(bin_transfer):
    K = 3
    Nk, Mk = sulfate_seed(2000.0, 60e-9, 1.6)
    Mk = Mk.at[:, SRTSOA1:SRTSOA1 + K].set(0.1 * Mk[:, [SRTSO4]])
    Gc = (jnp.zeros(N_GAS_SPECIES)
          .at[SRTSOA1:SRTSOA1 + K].set(kg(jnp.array([0.5, 1.0, 2.0])))
          .at[SRTSO4].set(kg(0.01)))
    st0 = make_state(Nk, Mk, Gc)
    org_rate = kg(jnp.array([0.3, 0.5, 1.0])) / 3600.0
    f = make_forcing(K, h2so4_rate=kg(0.2) / 3600.0, org_rate=org_rate,
                     nh3_conc=2.5e9, fion=3.0)
    p = soa_params([0.1, 10.0, 1000.0])
    dt = 600.0

    st, diag = aerosol_outer_step(
        st0, f, p, dt, cfg=AerosolStepConfig(growth_bin_transfer=bin_transfer))

    def class_total(s):
        return (np.asarray(jnp.sum(s.Mk[:, SRTSOA1:SRTSOA1 + K], axis=0))
                + np.asarray(s.Gc[SRTSOA1:SRTSOA1 + K]))

    want = class_total(st0) + np.asarray(org_rate) * dt
    np.testing.assert_allclose(class_total(st), want, rtol=1e-10)
    assert bool(diag.soa_time_integrated)


# =========================================================================
# (f) N2O5 uptake
# =========================================================================

@pytest.mark.parametrize("gamma", [1e-3, 0.02, 1.0])
def test_n2o5_free_molecular_limit(gamma):
    """Few-nm particles (Kn >> 1): k_het = gamma * c_bar * A / 4."""
    Nk = jnp.zeros(NBINS).at[2:6].set(1e4 * BOXVOL)
    Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(Nk * XMID)
    k_het, w = n2o5_uptake(Nk, Mk, XK, 298.0, 101325.0, BOXVOL, gamma)

    dp = jnp.cbrt(jnp.sum(Mk, axis=1) / jnp.maximum(Nk, 1e-30)
                  / calc_density(Mk) * 6.0 / PI)
    area = float(jnp.sum(jnp.where(Nk > 0, PI * dp**2 * Nk, 0.0))) / BOXVOL_M3
    c_bar = np.sqrt(8.0 * R_GAS * 298.0 / (PI * MW_N2O5 * 1e-3))
    assert float(k_het) == pytest.approx(gamma * c_bar * area / 4.0, rel=0.01)
    assert float(jnp.sum(w)) == pytest.approx(1.0, rel=1e-12)


def test_n2o5_gamma_zero():
    Nk, Mk = sulfate_seed(1000.0, 100e-9, 1.6)
    k_het, w = n2o5_uptake(Nk, Mk, XK, 298.0, 101325.0, BOXVOL, 0.0)
    assert float(k_het) == 0.0
    assert not bool(jnp.any(jnp.isnan(w)))
    assert float(jnp.sum(w)) == pytest.approx(1.0, rel=1e-12)


# =========================================================================
# (g) nitrate injection
# =========================================================================

def test_nitrate_injection_mass_and_slot():
    Nk, Mk = sulfate_seed(1000.0, 100e-9, 1.6)
    st0 = make_state(Nk, Mk)
    _, w = n2o5_uptake(Nk, Mk, XK, 298.0, 101325.0, BOXVOL, 0.02)
    assert float(Nk[-1]) < 1e10               # top bin is empty
    w_empty = w.at[-1].set(0.5)               # weight on the empty bin
    rate, dt = kg(1.0) / 3600.0, 300.0
    p = soa_params([1.0])

    out = {}
    for name, weights in (('w', w), ('w_empty', w_empty)):
        f = make_forcing(1, nitrate_rate=rate, nitrate_weights=weights)
        out[name] = aerosol_outer_step(st0, f, p, dt, processes=())
    st, diag = out['w']

    added = rate * dt
    assert float(diag.nitrate_added) == pytest.approx(added, rel=1e-12)
    assert float(jnp.sum(st.Mk[:, SRTNIT])) == pytest.approx(added, rel=1e-12)
    # Other dry columns: only MNFIX's empty-bin SO4 fill (~1e-12) changes them
    dry_cols = [c for c in range(ICOMP) if c not in (SRTNIT, SRTH2O)]
    np.testing.assert_allclose(
        np.asarray(jnp.sum(st.Mk[:, dry_cols], axis=0)),
        np.asarray(jnp.sum(st0.Mk[:, dry_cols], axis=0)), rtol=1e-10)
    # Weight on an empty bin is dropped and the rest renormalised
    np.testing.assert_allclose(np.asarray(out['w_empty'][0].Mk),
                               np.asarray(st.Mk), rtol=1e-12)


# =========================================================================
# (h) Level-1-like run, (i) no recompilation on c* change
# =========================================================================

def _level1_run(cstar, hours=12):
    Nk, Mk = sulfate_seed(1000.0, 80e-9, 1.6)
    st = make_state(Nk, Mk)
    h2so4_rate = 1e5 * BOXVOL * MW_H2SO4 * 1e-3 / AVOGADRO  # 1e5 cm^-3 s^-1
    f = make_forcing(1, h2so4_rate=h2so4_rate,
                     org_rate=jnp.array([kg(1.0) / 3600.0]),
                     nh3_conc=2.5e9, fion=3.0)
    p = soa_params([cstar])
    for _ in range(hours * 12):
        st, diag = aerosol_outer_step(st, f, p, 300.0)
    return st, diag


def test_level1_low_cstar_grows_more():
    st_lo, d_lo = _level1_run(1.0)
    st_hi, d_hi = _level1_run(1000.0)
    for st, d in ((st_lo, d_lo), (st_hi, d_hi)):
        assert not bool(jnp.any(jnp.isnan(st.Mk)))
        assert bool(d.soa_time_integrated)

    def dry_mass(st):
        return float(jnp.sum(st.Mk) - jnp.sum(st.Mk[:, SRTH2O]))

    assert soa_particle(st_lo.Mk, 1)[0] > 10.0 * soa_particle(st_hi.Mk, 1)[0]
    assert dry_mass(st_lo) > 1.5 * dry_mass(st_hi)
    assert float(d_lo.cs_h2so4) > float(d_hi.cs_h2so4)


def test_cstar_change_does_not_recompile():
    Nk, Mk = sulfate_seed(1000.0, 100e-9, 1.6)
    st = make_state(Nk, Mk)
    f = make_forcing(1, org_rate=jnp.array([kg(1.0) / 3600.0]))
    aerosol_outer_step(st, f, soa_params([1.0]), 120.0)
    n_compiled = aerosol_outer_step._cache_size()
    for cstar in (0.1, 1000.0):
        aerosol_outer_step(st, f, soa_params([cstar]), 120.0)
    assert aerosol_outer_step._cache_size() == n_compiled


def test_too_many_classes_rejected():
    Nk, Mk = sulfate_seed(1000.0, 100e-9, 1.6)
    K = N_SOA_CLASSES + 1
    p = soa_params(np.logspace(-2, 3, K))
    with pytest.raises(ValueError, match="N_SOA_CLASSES"):
        soa_partition_step(Nk, Mk, jnp.zeros(N_GAS_SPECIES), XK, 298.0,
                           101325.0, BOXVOL, 60.0, p.cstar_ref, p.dhvap,
                           p.mw, p.alpha)
