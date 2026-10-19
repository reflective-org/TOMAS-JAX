"""PPM bin transfer of in-place particle mass changes (AAAR tutorial).

Covers physics/ppm_remap.py and the AerosolStepConfig.growth_bin_transfer
setting ('ppm' default, 'mnfix' = the previous scheme) for SOA growth,
SOA evaporation, the zero-OA bootstrap and N2O5-uptake nitrate. See
docs/aaar/aerosol_step.md and docs/aaar/soa_partitioning.md.
"""
import math
from fractions import Fraction

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
from tomas_jax.physics import soa_partitioning
from tomas_jax.physics.ppm_remap import (
    BIN_TRANSFERS, _phi, cap_top_bin, keep_means_in_bins, remap_mass_change,
)
from tomas_jax.physics.soa_partitioning import soa_partition_step
from tomas_jax.physics.water_equilibrium import equilibrate_water

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


def _two_populations(lo_change):
    """80 bins: SOA-only particles in bins 30-39 (lo_change) or 50-59, and
    unchanged sulfate particles in bins 40-49, 1e12 #/cell each."""
    nbins = 80
    xk, xm = grid(nbins)
    k = np.arange(nbins)
    soa = (k >= 30) & (k < 40) if lo_change else (k >= 50) & (k < 60)
    so4 = (k >= 40) & (k < 50)
    Nk = jnp.where(soa | so4, 1e12, 0.0)
    Mk = (jnp.zeros((nbins, ICOMP))
          .at[:, SRTSOA1].set(jnp.where(soa, Nk * xm, 0.0))
          .at[:, SRTSO4].set(jnp.where(so4, Nk * xm, 0.0)))
    return xk, Nk, Mk, soa, so4


@pytest.mark.parametrize("lo_change, factor", [
    (True, 3.0), (True, 10.0),        # grow into / through the sulfate bins
    (False, 0.25), (False, 0.01),     # shrink into / through them
    (True, 0.01),                     # shrink away from them
])
def test_remap_moves_particles_past_unchanged_ones(lo_change, factor):
    """Two externally mixed populations side by side; only the SOA
    particles change. The sulfate particles stay exactly where they are
    (number and mass per bin), the SOA particles move by their own
    displacement however many bins that is (here 3 to 13 bins), without
    piling up where they cross the sulfate bins (at most the two
    populations' 2e12 #/cell plus a few percent from the PPM profile at
    the edge of the top-hat; the earlier flux-form remap stacked up to
    8e12)."""
    xk, Nk, Mk, soa, so4 = _two_populations(lo_change)
    k = np.arange(Nk.shape[0])
    dxi = float(jnp.log(xk[1] / xk[0]))
    Mk_new = Mk.at[:, SRTSOA1].multiply(factor)

    N2, M2 = remap(Nk, Mk, Mk_new, xk)

    np.testing.assert_allclose(np.asarray(M2[:, SRTSO4]),
                               np.asarray(Mk[:, SRTSO4]), rtol=1e-14,
                               atol=1e-14 * float(jnp.max(Mk[:, SRTSO4])))
    n_soa = np.asarray(N2) - np.where(so4, 1e12, 0.0)      # SOA particles
    assert n_soa.min() > -1e-3
    exact = float(np.mean(k[soa])) + np.log(factor) / dxi
    assert np.sum(n_soa * k) / np.sum(n_soa) == pytest.approx(exact, abs=0.02)
    assert float(jnp.max(N2)) <= 2.05e12
    assert float(jnp.sum(N2)) == pytest.approx(float(jnp.sum(Nk)), rel=1e-13)
    np.testing.assert_allclose(np.asarray(jnp.sum(M2, axis=0)),
                               np.asarray(jnp.sum(Mk_new, axis=0)),
                               rtol=1e-13, atol=0.0)
    assert means_inside(N2, M2, xk)


@pytest.mark.parametrize("factor", [10.0, 0.1])
def test_remap_multi_bin_shift_keeps_the_mode(factor):
    """160 bins, every particle x10 or /10 at once (13.3 bins): the mode
    moves by ln(10) in ln(m) with its width and single maximum kept."""
    xk, Nk, Mk = seed(160, 2000.0, 60e-9, 1.6, COMP)
    ln_c = np.log(np.asarray(jnp.sqrt(xk[:-1] * xk[1:])))

    N2, M2 = remap(Nk, Mk, factor * Mk, xk)

    def moments(N):
        N = np.asarray(N)
        mu = np.sum(N * ln_c) / N.sum()
        return mu, np.sqrt(np.sum(N * (ln_c - mu) ** 2) / N.sum())

    (mu0, sd0), (mu1, sd1) = moments(Nk), moments(N2)
    assert mu1 - mu0 == pytest.approx(np.log(factor), rel=1e-6)
    assert sd1 == pytest.approx(sd0, rel=1e-6)
    assert extrema(N2) == 1
    assert means_inside(N2, M2, xk)


def test_phi_matches_the_exact_series():
    """The closed-form/Taylor mass integrals phi_j(z) = int_0^1 v^j e^(zv)
    agree with an exact rational series to 1e-14 (both branches and the
    switch at z = 1)."""
    zs = [0.0, 1e-9, 0.3, 0.999999, 1.0, 1.000001, 2.5, 7.0, 20.0]
    got = _phi(jnp.asarray(zs))
    for j in range(4):
        for i, z in enumerate(zs):
            zf = Fraction(z)
            ref = float(sum(zf ** n / (math.factorial(n) * (n + j + 1))
                            for n in range(120)))
            assert float(got[j][i]) == pytest.approx(ref, rel=1e-14)


def test_cap_top_bin_takes_back_only_the_added_species():
    xk, Nk, Mk = seed(40, 1000.0, 100e-9, 1.6, {SRTSO4: 1.0})
    Nk = Nk.at[-1].set(1e6)
    m_top = float(jnp.sqrt(xk[-2] * xk[-1]))
    Mk = Mk.at[-1].set(0.0).at[-1, SRTSO4].set(1e6 * m_top)
    Mk = Mk.at[-1, SRTSOA1].set(1e6 * 0.8 * float(xk[-1]))  # mean 1.5x top

    M2, removed = cap_top_bin(Nk, Mk, xk, (SRTSOA1, SRTSOA1 + 1))

    mean = float(jnp.sum(M2[-1, :ICOMP_NODIAG])) / 1e6
    assert float(xk[-2]) < mean <= float(xk[-1])
    assert float(M2[-1, SRTSO4]) == float(Mk[-1, SRTSO4])
    assert float(removed[0]) == pytest.approx(
        float(Mk[-1, SRTSOA1] - M2[-1, SRTSOA1]), rel=1e-14)
    np.testing.assert_array_equal(np.asarray(M2[:-1]), np.asarray(Mk[:-1]))
    M3, removed3 = cap_top_bin(Nk, M2, xk, (SRTSOA1, SRTSOA1 + 1))
    assert float(removed3[0]) == 0.0           # inside the grid: no change


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


@pytest.mark.parametrize("transfer, nbins", [
    ("ppm", 40), ("ppm", 160), ("mnfix", 40)])
def test_temperature_rise_evaporates_smoothly(transfer, nbins):
    """Grow SOA on a seed at 288 K (c* = 1, 2 ug/m3/h, 6 h), then raise T
    to 318 K at once with no source: most of the SOA evaporates, the first
    60 s step moves the smallest particles several bins. The class budget
    closes every step, nothing goes negative and the mode moves down
    monotonically; with PPM number is conserved and one extremum kept
    (the earlier flux-form remap had 5 at 160 bins)."""
    xk, Nk, Mk = seed(nbins, 2000.0, 50e-9, 1.6,
                      {SRTSO4: 0.7, SRTORG_NUC: 0.3})
    st = state(xk, Nk, Mk, temp=288.0)
    p = SOAParams(cstar_ref=jnp.array([1.0]), dhvap=jnp.array([131.0]),
                  mw=jnp.array([200.0]), alpha=jnp.float64(1.0))
    for _ in range(72):
        st, _ = aerosol_outer_step(st, forcing(nbins, org_ugm3_h=2.0), p, 300.0,
                                   processes=('soa_partitioning',),
                                   cfg=cfg(transfer))
    soa_grown = float(jnp.sum(st.Mk[:, SRTSOA1]))
    st = st._replace(temp=jnp.float64(318.0))
    n0 = float(jnp.sum(st.Nk))
    d_prev = number_mean_diameter(st.Nk, st.Mk)
    for _ in range(36):
        tot0 = _class_total(st)
        st, diag = aerosol_outer_step(st, forcing(nbins), p, 300.0,
                                      processes=('soa_partitioning',),
                                      cfg=cfg(transfer))
        np.testing.assert_allclose(_class_total(st), tot0, rtol=1e-12)
        assert bool(jnp.all(st.Mk >= 0.0)) and bool(jnp.all(st.Nk >= 0.0))
        d_now = number_mean_diameter(st.Nk, st.Mk)
        assert d_now <= d_prev * (1.0 + 1e-9)
        d_prev = d_now
        if transfer == 'ppm':   # MNFIX drops particles below the grid (0.13 %)
            assert float(jnp.sum(st.Nk)) == pytest.approx(n0, rel=1e-10)
            assert extrema(st.Nk) == 1
            assert means_inside(st.Nk, st.Mk, xk)
    assert bool(diag.soa_time_integrated)
    assert float(jnp.sum(st.Mk[:, SRTSOA1])) < 0.05 * soa_grown


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


@pytest.mark.parametrize("transfer", BIN_TRANSFERS)
def test_top_bin_growth_keeps_class_budget(transfer):
    """Particles in the top three bins, 2000 ug/m3 of c* = 0.01 vapour,
    12 x 600 s: the top bin's mean reaches the grid's upper edge. Gas +
    particle closes per class every step with both settings; with PPM the
    SOA that does not fit stays in the gas and no sulfate is trimmed."""
    nbins = 40
    xk, xm = grid(nbins)
    k = np.arange(nbins)
    Nk = jnp.where(k >= nbins - 3, 1e6, 0.0) + 1e-5
    Mk = jnp.zeros((nbins, ICOMP)).at[:, SRTSO4].set(Nk * xm)
    Gc = jnp.zeros(N_GAS_SPECIES).at[SRTSOA1].set(kg(2000.0))
    p = soa_params([0.01])
    step = jax.jit(soa_partition_step, static_argnames=('bin_transfer',))
    so4_0 = float(jnp.sum(Mk[:, SRTSO4]))
    top_mean = []
    for _ in range(12):
        tot0 = float(jnp.sum(Mk[:, SRTSOA1]) + Gc[SRTSOA1])
        Nk, Mk, Gc, diag = step(Nk, Mk, Gc, xk, 298.0, 101325.0, BOXVOL, 600.0,
                                p.cstar_ref, p.dhvap, p.mw, p.alpha,
                                bin_transfer=transfer)
        tot1 = float(jnp.sum(Mk[:, SRTSOA1]) + Gc[SRTSOA1])
        assert tot1 == pytest.approx(tot0, rel=1e-12)
        assert bool(jnp.all(Mk >= 0.0)) and float(Gc[SRTSOA1]) >= 0.0
        top_mean.append(float(jnp.sum(Mk[-1, :ICOMP_NODIAG]) / Nk[-1]))
    assert float(jnp.sum(Mk[:, SRTSOA1])) > kg(0.5)
    if transfer == 'ppm':
        assert max(top_mean) <= float(xk[-1])
        assert max(top_mean) > 0.95 * float(xk[-1])   # the cap did act
        assert float(jnp.sum(Mk[:, SRTSO4])) == pytest.approx(so4_0, rel=1e-12)


def _soa_hours(Nk, Mk, Gc, xk, phases, rho=1500.0):
    """SOA partitioning (c* = 1, 'ppm') with 60 s micro-steps and water,
    as aerosol_outer_step does; phases = [(T [K], hours, ug/m3/h), ...].
    Returns the final (Nk, Mk), the extrema count after every hour and the
    number-mean dry diameter [nm] and GSD from each bin's mean mass."""
    p = soa_params([1.0])

    @jax.jit
    def hour(Nk, Mk, Gc, temp, prod):
        def micro(_, c):
            Nk, Mk, Gc = c
            Gc = Gc.at[SRTSOA1].add(prod * 60.0)
            Nk, Mk, Gc, _ = soa_partitioning.soa_partition_step(
                Nk, Mk, Gc, xk, temp, 101325.0, BOXVOL, 60.0, p.cstar_ref,
                p.dhvap, p.mw, p.alpha)
            return Nk, equilibrate_water(Mk, 0.5, temp, 0.0), Gc
        return jax.lax.fori_loop(0, 60, micro, (Nk, Mk, Gc))

    ext = []
    for temp, hours, rate in phases:
        for _ in range(hours):
            Nk, Mk, Gc = hour(Nk, Mk, Gc, temp, kg(rate) / 3600.0)
            ext.append(extrema(Nk))
    N = np.asarray(Nk)
    pop = N > 1e-6 * N.max()
    m = np.asarray(jnp.sum(Mk[:, :ICOMP_NODIAG], axis=1))[pop] / N[pop]
    d = np.cbrt(6.0 * m / (PI * rho)) * 1e9
    w = N[pop] / N[pop].sum()
    mu = np.sum(w * np.log(d))
    gsd = np.exp(np.sqrt(np.sum(w * (np.log(d) - mu) ** 2)))
    return ext, float(np.sum(w * d)), float(gsd)


def _moving_sections(monkeypatch):
    """No bin transfer at all: every section keeps its particles and only
    its mass changes (same vbs-jax physics) -- the reference."""
    monkeypatch.setattr(soa_partitioning, "remap_mass_change",
                        lambda Nk, Mk_old, Mk_new, xk, *a, **k: (Nk, Mk_new))
    monkeypatch.setattr(soa_partitioning, "cap_top_bin",
                        lambda Nk, Mk, xk, cols, *a, **k:
                        (Mk, jnp.zeros(cols[1] - cols[0])))
    monkeypatch.setattr(soa_partitioning, "mnfix_jax",
                        lambda Nk, Mk, xk, *a, **k: (Nk, Mk))


@pytest.mark.parametrize("case", ["round_trip", "bootstrap_ripening"])
def test_soa_matches_moving_sections(case, monkeypatch):
    """80 bins against moving sections (no transfer between bins).
    round_trip: grow 3 h at 283 K (1 ug/m3/h), then 1 h at 313 K without
    source, which evaporates most of the SOA; particles must return to the
    sizes that go with their composition (per-bin mean composition alone
    gave +1.0 % in mean diameter and -2.3 % in GSD, the earlier flux-form
    remap +1.3 % / -2.8 %). bootstrap_ripening: pure sulfate seed with
    5 ug/m3 of c* = 1 vapour, 4 h at 298 K (zero-OA bootstrap, then Kelvin
    ripening that sends the smallest particles back towards their cores)."""
    nbins = 80
    if case == "round_trip":
        xk, Nk, Mk = seed(nbins, 2000.0, 80e-9, 1.6,
                          {SRTSO4: 0.7, SRTORG_NUC: 0.3})
        phases, gas0 = [(283.15, 3, 1.0), (313.15, 1, 0.0)], 0.0
    else:
        xk, Nk, Mk = seed(nbins, 2000.0, 80e-9, 1.5, {SRTSO4: 1.0})
        phases, gas0 = [(298.15, 4, 0.0)], 5.0
    Mk = equilibrate_water(Mk, 0.5, phases[0][0], 0.0)
    Gc = jnp.zeros(N_GAS_SPECIES).at[SRTSOA1].set(kg(gas0))

    ext, d_ppm, g_ppm = _soa_hours(Nk, Mk, Gc, xk, phases)
    _moving_sections(monkeypatch)
    _, d_ref, g_ref = _soa_hours(Nk, Mk, Gc, xk, phases)

    assert d_ppm == pytest.approx(d_ref, rel=0.003)
    assert g_ppm == pytest.approx(g_ref, rel=0.005)
    if case == "round_trip":
        assert max(ext) == 1


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


def test_nitrate_that_does_not_fit_the_grid_is_reported():
    """Nitrate pushed onto top-bin particles: with PPM, what would push
    the top bin past the grid is not added, nitrate_added says so (the
    driver then warns) and no sulfate is trimmed."""
    nbins = 40
    xk, xm = grid(nbins)
    n_top = 2e10
    Nk = jnp.zeros(nbins).at[-1].set(n_top)
    Mk = jnp.zeros((nbins, ICOMP)).at[-1, SRTSO4].set(
        n_top * 0.9 * float(xk[-1]))
    w = jnp.zeros(nbins).at[-1].set(1.0)
    dt = 600.0
    rate = n_top * 0.5 * float(xk[-1]) / dt          # +0.5 x top edge
    st, diag = aerosol_outer_step(state(xk, Nk, Mk),
                                  forcing(nbins, nitrate_rate=rate,
                                          nitrate_weights=w),
                                  soa_params([1.0]), dt, processes=(),
                                  cfg=cfg('ppm'))
    added = float(diag.nitrate_added)
    assert 0.0 < added < 0.5 * rate * dt
    assert float(jnp.sum(st.Mk[:, SRTNIT])) == pytest.approx(added, rel=1e-12)
    assert float(jnp.sum(st.Mk[:, SRTSO4])) == pytest.approx(
        float(jnp.sum(Mk[:, SRTSO4])), rel=1e-10)
    assert float(jnp.sum(st.Mk[-1, :ICOMP_NODIAG]) / st.Nk[-1]) <= float(xk[-1])


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


@pytest.mark.parametrize("lo, hi", [(3.0, 1.5), (0.5, 0.25), (3.0, 0.5)])
def test_unchanged_bin_between_changing_neighbours_stays(lo, hi):
    """One bin of sulfate-only particles (bin 40) between SOA-only bins 30-39
    and 41-50 whose particles change by different factors: the sulfate
    particles keep their bin exactly (minmod edge slope), whatever passes
    through it. Before the minmod slope, (3.0, 1.5) moved 43 % of the
    sulfate into bin 41."""
    nbins = 80
    xk, xm = grid(nbins)
    k = np.arange(nbins)
    lo_bins, hi_bins = (k >= 30) & (k < 40), (k >= 41) & (k <= 50)
    so4 = k == 40
    Nk = jnp.where(lo_bins | hi_bins | so4, 1e12, 0.0)
    Mk = (jnp.zeros((nbins, ICOMP))
          .at[:, SRTSOA1].set(jnp.where(lo_bins | hi_bins, Nk * xm, 0.0))
          .at[:, SRTSO4].set(jnp.where(so4, Nk * xm, 0.0)))
    factor = jnp.where(lo_bins, lo, jnp.where(hi_bins, hi, 1.0))
    Mk_new = Mk.at[:, SRTSOA1].multiply(factor)

    N2, M2 = remap(Nk, Mk, Mk_new, xk)

    np.testing.assert_allclose(np.asarray(M2[:, SRTSO4]), np.asarray(Mk[:, SRTSO4]),
                               rtol=1e-14, atol=1e-14 * float(jnp.max(Mk[:, SRTSO4])))
    assert float(jnp.sum(N2)) == pytest.approx(float(jnp.sum(Nk)), rel=1e-13)
    np.testing.assert_allclose(np.asarray(jnp.sum(M2, axis=0)),
                               np.asarray(jnp.sum(Mk_new, axis=0)), rtol=1e-13, atol=0.0)


def test_driver_passes_micro_dt():
    """AerosolStepConfig.micro_dt reaches the aerosol step: 60 s (the default)
    reproduces a default run bit for bit, 20 s changes the result slightly."""
    import dataclasses
    s = load_preset("level1_volatility").replace(duration_h=1.0)
    base = run(s)
    same = run(s, aerosol_cfg=dataclasses.replace(AerosolStepConfig(), micro_dt=60.0))
    finer = run(s, aerosol_cfg=dataclasses.replace(AerosolStepConfig(), micro_dt=20.0))
    np.testing.assert_array_equal(np.asarray(same.dndlog10dp_cm3), np.asarray(base.dndlog10dp_cm3))
    assert not np.array_equal(np.asarray(finer.dndlog10dp_cm3), np.asarray(base.dndlog10dp_cm3))
    assert finer.totals["SOA_ugm3"][-1] == pytest.approx(base.totals["SOA_ugm3"][-1], rel=0.02)

