"""PPM bin transfer of in-place particle mass changes (AAAR tutorial).

Covers physics/ppm_remap.py and the AerosolStepConfig.growth_bin_transfer
setting ('ppm' default, 'mnfix' = the previous scheme) for SOA growth,
SOA evaporation, the zero-OA bootstrap and N2O5-uptake nitrate. See
docs/aaar/aerosol_step.md and docs/aaar/soa_partitioning.md.
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.box.aerosol_step import (
    AerosolForcing, AerosolState, SOAParams, aerosol_outer_step,
)
from tomas_jax.box.config import AerosolStepConfig
from tomas_jax.box.coupling import mass_grid
from tomas_jax.box.driver import run
from tomas_jax.box.scenario import load_preset
from tomas_jax.core.config import (
    ICOMP, ICOMP_NODIAG, N_GAS_SPECIES, PI, SRTH2O, SRTNIT, SRTORG_NUC,
    SRTSO4, SRTSOA1,
)
from tomas_jax.physics.ppm_remap import (
    BIN_TRANSFERS, keep_means_in_bins, remap_mass_change,
)
from tomas_jax.physics.soa_partitioning import soa_partition_step

BOXVOL = 1.0e10                    # cm^3 (tutorial default)
BOXVOL_M3 = BOXVOL * 1e-6

remap = jax.jit(remap_mass_change)


def kg(ug_m3):
    """ug/m^3 -> kg/cell."""
    return ug_m3 * 1e-9 * BOXVOL_M3


def grid(nbins):
    xk = jnp.asarray(mass_grid(nbins))
    return xk, jnp.sqrt(xk[:-1] * xk[1:])


def seed(nbins, n_cm3, gmd, gsd, comp, rho=1770.0):
    """Lognormal number [#/cell] with each particle's mass split by
    comp = {slot: fraction} (bin-centre mass)."""
    xk, xm = grid(nbins)
    dp = jnp.cbrt(xm / rho * 6.0 / PI)
    pdf = jnp.exp(-0.5 * (jnp.log(dp / gmd) / jnp.log(gsd)) ** 2)
    Nk = n_cm3 * BOXVOL * pdf / jnp.sum(pdf)
    Mk = jnp.zeros((nbins, ICOMP))
    for slot, frac in comp.items():
        Mk = Mk.at[:, slot].set(frac * Nk * xm)
    return xk, Nk, Mk


def extrema(y, frac=0.05):
    """Interior local extrema of y above frac * max(y)."""
    y = np.asarray(y)
    thr = frac * y.max()
    return sum(1 for i in range(1, y.size - 1) if y[i] > thr and (
        (y[i] > y[i - 1] and y[i] > y[i + 1])
        or (y[i] < y[i - 1] and y[i] < y[i + 1])))


def means_inside(Nk, Mk, xk, rel_n=1e-6):
    """Every bin with Nk > rel_n * max(Nk) has its mean dry mass in-bin."""
    Nk, xk = np.asarray(Nk), np.asarray(xk)
    m = np.asarray(jnp.sum(Mk[:, :ICOMP_NODIAG], axis=1)) / np.maximum(Nk, 1e-300)
    pop = Nk > rel_n * Nk.max()
    return bool(np.all((m[pop] >= xk[:-1][pop]) & (m[pop] <= xk[1:][pop])))


def number_mean_diameter(Nk, Mk, rho=1770.0):
    m = jnp.sum(Mk[:, :ICOMP_NODIAG], axis=1) / jnp.maximum(Nk, 1e-30)
    return float(jnp.sum(Nk * jnp.cbrt(m / rho * 6.0 / PI)) / jnp.sum(Nk))


def state(xk, Nk, Mk, Gc=None, temp=298.15, rh=0.5):
    Gc = jnp.zeros(N_GAS_SPECIES) if Gc is None else Gc
    f = jnp.float64
    return AerosolState(Nk, Mk, Gc, xk, f(temp), f(101325.0), f(rh), f(BOXVOL))


def forcing(nbins, K=1, org_ugm3_h=0.0, nitrate_rate=0.0, nitrate_weights=None):
    return AerosolForcing(
        h2so4_rate=jnp.float64(0.0),
        org_rate=jnp.full(K, kg(org_ugm3_h) / 3600.0),
        nitrate_rate=jnp.float64(nitrate_rate),
        nitrate_weights=(jnp.zeros(nbins) if nitrate_weights is None
                         else jnp.asarray(nitrate_weights)))


def soa_params(cstar):
    cstar = jnp.atleast_1d(jnp.asarray(cstar, dtype=jnp.float64))
    return SOAParams(cstar_ref=cstar, dhvap=131.0 - 11.0 * jnp.log10(cstar),
                     mw=jnp.full(cstar.shape, 200.0), alpha=jnp.float64(1.0))


def cfg(transfer):
    return AerosolStepConfig(growth_bin_transfer=transfer)


# =========================================================================
# remap_mass_change on its own
# =========================================================================

COMP = {SRTSO4: 0.5, SRTSOA1: 0.3, SRTORG_NUC: 0.1, SRTNIT: 0.1, SRTH2O: 0.2}


@pytest.mark.parametrize("nbins", [40, 80, 160])
@pytest.mark.parametrize("change", ["grow", "evaporate"])
def test_remap_conserves_every_species(nbins, change):
    xk, Nk, Mk = seed(nbins, 2000.0, 60e-9, 1.6, COMP)
    if change == "grow":       # +150 % of the dry mass as SOA
        d = 1.5 * jnp.sum(Mk[:, :ICOMP_NODIAG], axis=1)
    else:                      # 90 % of the SOA evaporates
        d = -0.9 * Mk[:, SRTSOA1]
    Mk_new = Mk.at[:, SRTSOA1].add(d)

    N2, M2 = remap(Nk, Mk, Mk_new, xk)

    assert float(jnp.sum(N2)) == pytest.approx(float(jnp.sum(Nk)), rel=1e-12)
    np.testing.assert_allclose(np.asarray(jnp.sum(M2, axis=0)),
                               np.asarray(jnp.sum(Mk_new, axis=0)),
                               rtol=1e-12, atol=0.0)
    assert bool(jnp.all(N2 >= 0.0)) and bool(jnp.all(M2 >= 0.0))
    assert bool(jnp.all(jnp.isfinite(N2))) and bool(jnp.all(jnp.isfinite(M2)))
    assert means_inside(N2, M2, xk)
    assert extrema(N2) == 1
    # every particle keeps its composition (the same change in every bin)
    pop = N2 > 1e-3 * jnp.max(N2)
    so4_share = M2[:, SRTSO4] / jnp.sum(M2[:, :ICOMP_NODIAG], axis=1)
    want = float(jnp.sum(Mk_new[:, SRTSO4])
                 / jnp.sum(Mk_new[:, :ICOMP_NODIAG]))
    np.testing.assert_allclose(np.asarray(so4_share[pop]), want, rtol=1e-6)


@pytest.mark.parametrize("nbins", [40, 80])
def test_remap_moves_the_mode_by_the_mass_change(nbins):
    """All particles x G: the number distribution moves by ln G in ln m."""
    xk, Nk, Mk = seed(nbins, 2000.0, 60e-9, 1.6, COMP)
    g = 1.7
    xi_c = jnp.log(jnp.sqrt(xk[:-1] * xk[1:]))

    N2, M2 = remap(Nk, Mk, g * Mk, xk)

    shift = float(jnp.sum(N2 * xi_c) / jnp.sum(N2)
                  - jnp.sum(Nk * xi_c) / jnp.sum(Nk))
    assert shift == pytest.approx(np.log(g), rel=0.02)
    assert means_inside(N2, M2, xk)


def test_remap_without_change_is_identity():
    xk, Nk, Mk = seed(80, 2000.0, 60e-9, 1.6, COMP)
    N2, M2 = remap(Nk, Mk, Mk, xk)
    np.testing.assert_allclose(np.asarray(N2), np.asarray(Nk), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(M2), np.asarray(Mk), rtol=1e-14)


def test_keep_means_in_bins_moves_few_particles():
    """A bin whose mean crossed its upper edge sends a small share up."""
    xk, Nk, Mk = seed(80, 2000.0, 60e-9, 1.6, COMP)
    k = int(jnp.argmax(Nk))
    m_k = float(jnp.sum(Mk[k, :ICOMP_NODIAG]) / Nk[k])
    Mk_bad = Mk.at[k].multiply(float(xk[k + 1]) * 1.001 / m_k)

    N2, M2 = keep_means_in_bins(Nk, Mk_bad, xk)

    assert means_inside(N2, M2, xk)
    assert 0.0 < float(Nk[k] - N2[k]) < 0.1 * float(Nk[k])
    assert float(jnp.sum(N2)) == pytest.approx(float(jnp.sum(Nk)), rel=1e-14)
    np.testing.assert_allclose(np.asarray(jnp.sum(M2, axis=0)),
                               np.asarray(jnp.sum(Mk_bad, axis=0)), rtol=1e-13)


# =========================================================================
# SOA growth, evaporation, bootstrap through the aerosol step
# =========================================================================

def _class_total(st, K=1):
    return (np.asarray(jnp.sum(st.Mk[:, SRTSOA1:SRTSOA1 + K], axis=0))
            + np.asarray(st.Gc[SRTSOA1:SRTSOA1 + K]))


@pytest.mark.parametrize("transfer", BIN_TRANSFERS)
def test_soa_growth_smooth_and_conserved(transfer):
    """Level-1-like growth (c* = 1, 0.5 ug/m3/h, 12 h, 80 bins, SOA only)."""
    xk, Nk, Mk = seed(80, 2000.0, 50e-9, 1.6,
                      {SRTSO4: 0.7, SRTORG_NUC: 0.3})
    st = state(xk, Nk, Mk)
    f = forcing(80, org_ugm3_h=0.5)
    p = soa_params([1.0])
    n_ext = []
    for _ in range(144):
        st, diag = aerosol_outer_step(st, f, p, 300.0,
                                      processes=('soa_partitioning',),
                                      cfg=cfg(transfer))
        n_ext.append(extrema(st.Nk))

    assert bool(diag.soa_time_integrated)
    np.testing.assert_allclose(_class_total(st), kg(0.5) * 12.0, rtol=1e-10)
    assert float(jnp.sum(st.Nk)) == pytest.approx(float(jnp.sum(Nk)), rel=1e-10)
    assert bool(jnp.all(st.Mk >= 0.0)) and bool(jnp.all(st.Nk >= 0.0))
    assert float(jnp.sum(st.Mk[:, SRTSOA1])) > kg(4.0)
    if transfer == 'ppm':
        assert max(n_ext) == 1
        assert means_inside(st.Nk, st.Mk, xk)
    else:                       # the previous scheme combs the mode
        assert n_ext[-1] > 2


@pytest.mark.parametrize("transfer", BIN_TRANSFERS)
def test_soa_evaporation_moves_particles_down(transfer):
    """Particles carrying 75 % c* = 10 SOA, no vapour: the SOA evaporates
    and the mode shrinks back to the sulfate seed."""
    nbins = 80
    xk, Nk, Mk = seed(nbins, 1000.0, 100e-9, 1.5, {SRTSO4: 1.0})
    d_seed = number_mean_diameter(Nk, Mk)
    shift = int(round(np.log(4.0) / float(jnp.log(xk[1] / xk[0]))))
    Nk = jnp.roll(Nk, shift)                 # 4x the seed mass per particle
    _, xm = grid(nbins)
    Mk = (jnp.zeros((nbins, ICOMP)).at[:, SRTSO4].set(0.25 * Nk * xm)
          .at[:, SRTSOA1].set(0.75 * Nk * xm))
    st = state(xk, Nk, Mk)
    total0 = _class_total(st)
    so4_0 = float(jnp.sum(Mk[:, SRTSO4]))
    d_prev = number_mean_diameter(Nk, Mk)
    f, p = forcing(nbins), soa_params([10.0])
    for _ in range(60):
        st, diag = aerosol_outer_step(st, f, p, 60.0,
                                      processes=('soa_partitioning',),
                                      cfg=cfg(transfer))
        np.testing.assert_allclose(_class_total(st), total0, rtol=1e-10)
        assert bool(jnp.all(st.Mk >= 0.0)) and bool(jnp.all(st.Nk >= 0.0))
        d_now = number_mean_diameter(st.Nk, st.Mk)
        assert d_now <= d_prev * (1.0 + 1e-9)
        d_prev = d_now
        if transfer == 'ppm':
            assert extrema(st.Nk) == 1
            assert means_inside(st.Nk, st.Mk, xk)

    assert bool(diag.soa_time_integrated)
    assert float(jnp.sum(st.Mk[:, SRTSOA1])) < 1e-3 * float(total0[0])
    assert float(jnp.sum(st.Nk)) == pytest.approx(float(jnp.sum(Nk)), rel=1e-12)
    # MNFIX's empty-bin fill adds ~1e-12 relative SO4
    assert float(jnp.sum(st.Mk[:, SRTSO4])) == pytest.approx(so4_0, rel=1e-10)
    assert d_prev == pytest.approx(d_seed, rel=0.05)


@pytest.mark.parametrize("transfer", BIN_TRANSFERS)
def test_zero_oa_bootstrap_both_transfers(transfer):
    """Pure sulfate seed (the absorbing floor AC1 acts the same with either
    transfer): c* = 0.01 condenses and c* = 1000 stays in the gas; alone,
    c* = 1000 never transiently condenses."""
    xk, Nk, Mk = seed(40, 1000.0, 100e-9, 1.6, {SRTSO4: 1.0})
    Gc = jnp.zeros(N_GAS_SPECIES).at[SRTSOA1:SRTSOA1 + 2].set(kg(2.0))
    st = state(xk, Nk, Mk, Gc)
    f, p = forcing(40, K=2), soa_params([0.01, 1000.0])
    for _ in range(6):
        st, diag = aerosol_outer_step(st, f, p, 300.0,
                                      processes=('soa_partitioning',),
                                      cfg=cfg(transfer))
    part = np.asarray(jnp.sum(st.Mk[:, SRTSOA1:SRTSOA1 + 2], axis=0))
    assert bool(diag.soa_time_integrated)
    assert part[0] > 0.5 * kg(2.0)
    assert part[1] < 0.01 * kg(2.0)
    np.testing.assert_allclose(_class_total(st, 2), kg(2.0), rtol=1e-10)
    assert not bool(jnp.any(jnp.isnan(st.Mk)))

    st = state(xk, Nk, Mk, jnp.zeros(N_GAS_SPECIES).at[SRTSOA1].set(kg(2.0)))
    for _ in range(30):
        st, _ = aerosol_outer_step(st, forcing(40), soa_params([1000.0]),
                                   60.0, processes=('soa_partitioning',),
                                   cfg=cfg(transfer))
        assert float(jnp.sum(st.Mk[:, SRTSOA1])) < 1e-3 * kg(2.0)


def test_transfers_agree_on_soa_mass():
    """Same partitioning physics: class totals within 1 % after 2 h."""
    out = {}
    for transfer in BIN_TRANSFERS:
        xk, Nk, Mk = seed(40, 2000.0, 50e-9, 1.6,
                          {SRTSO4: 0.7, SRTORG_NUC: 0.3})
        st = state(xk, Nk, Mk)
        for _ in range(24):
            st, _ = aerosol_outer_step(st, forcing(40, org_ugm3_h=2.0),
                                       soa_params([1.0]), 300.0,
                                       processes=('soa_partitioning',),
                                       cfg=cfg(transfer))
        out[transfer] = float(jnp.sum(st.Mk[:, SRTSOA1]))
    assert out['ppm'] == pytest.approx(out['mnfix'], rel=0.01)


# =========================================================================
# nitrate (N2O5 uptake) injection
# =========================================================================

@pytest.mark.parametrize("transfer", BIN_TRANSFERS)
def test_nitrate_injection_conserved(transfer):
    xk, Nk, Mk = seed(80, 2000.0, 80e-9, 1.6, {SRTSO4: 1.0})
    st0 = state(xk, Nk, Mk)
    w = Nk * jnp.cbrt(jnp.sqrt(xk[:-1] * xk[1:])) ** 2     # ~ surface
    rate, dt = kg(2.0) / 3600.0, 3600.0
    f = forcing(80, nitrate_rate=rate, nitrate_weights=w / jnp.sum(w))

    st, diag = aerosol_outer_step(st0, f, soa_params([1.0]), dt,
                                  processes=(), cfg=cfg(transfer))

    assert float(diag.nitrate_added) == pytest.approx(rate * dt, rel=1e-12)
    assert float(jnp.sum(st.Mk[:, SRTNIT])) == pytest.approx(rate * dt,
                                                             rel=1e-12)
    assert float(jnp.sum(st.Nk)) == pytest.approx(float(jnp.sum(Nk)), rel=1e-12)
    assert float(jnp.sum(st.Mk[:, SRTSO4])) == pytest.approx(
        float(jnp.sum(Mk[:, SRTSO4])), rel=1e-10)
    if transfer == 'ppm':
        assert extrema(st.Nk) == 1
        assert means_inside(st.Nk, st.Mk, xk)


# =========================================================================
# settings
# =========================================================================

def test_unknown_transfer_rejected():
    with pytest.raises(ValueError, match="growth_bin_transfer"):
        AerosolStepConfig(growth_bin_transfer="tfl")
    xk, Nk, Mk = seed(40, 1000.0, 100e-9, 1.6, {SRTSO4: 1.0})
    p = soa_params([1.0])
    with pytest.raises(ValueError, match="bin_transfer"):
        soa_partition_step(Nk, Mk, jnp.zeros(N_GAS_SPECIES), xk, 298.0,
                           101325.0, BOXVOL, 60.0, p.cstar_ref, p.dhvap,
                           p.mw, p.alpha, bin_transfer="tfl")


def test_default_is_ppm_and_driver_reports_changes():
    assert AerosolStepConfig().growth_bin_transfer == 'ppm'
    s = load_preset("level1_volatility").replace(duration_h=0.5)
    r_ppm = run(s)
    r_mnfix = run(s, aerosol_cfg=cfg('mnfix'))
    assert r_ppm.diagnostics["aerosol_cfg"] == {}
    assert r_mnfix.diagnostics["aerosol_cfg"] == {"growth_bin_transfer": "mnfix"}
    assert r_ppm.totals["SOA_ugm3"][-1] == pytest.approx(
        r_mnfix.totals["SOA_ugm3"][-1], rel=0.01)
