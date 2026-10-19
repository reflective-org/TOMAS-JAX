"""Where N2O5-uptake nitrate goes in the AAAR aerosol micro-step.

box/aerosol_step.py places nitrate on every bin with Nk > NITRATE_N_MIN
(2e-5 #/cell, just above MNFIX's empty-bin floor), each bin's share
following its number within the outer step. It used NEPS_CONDSINK
(1e10 #/cell = 1 cm^-3 in the tutorial box), which left thin tail bins
without nitrate: in Level 3 they stopped growing at night and stayed
behind as a band of a few cm^-3 at 40-90 nm. See docs/aaar/aerosol_step.md.
"""
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.box.aerosol_step import (
    NITRATE_N_MIN, AerosolForcing, AerosolState, SOAParams, aerosol_outer_step,
)
from tomas_jax.box.config import AerosolStepConfig
from tomas_jax.box.coupling import mass_grid
from tomas_jax.core.config import ICOMP, N_GAS_SPECIES, SRTNIT, SRTSO4
from tomas_jax.core.mnfix_jax import MNFIX_N_EMPTY
from tomas_jax.physics.ppm_remap import BIN_TRANSFERS

BOXVOL = 1.0e10                    # cm^3 (tutorial default)
NBINS = 40
MODE = slice(24, 31)               # main mode, 200 cm^-3 per bin
THIN = 12                          # 0.5 cm^-3: below the old 1 cm^-3 gate
TINY = 8                           # 1e-3 #/cell (1e-13 cm^-3): above the gate
EMPTY = 4                          # N = 0, but given a nitrate weight
FLOOR = 36                         # MNFIX's empty-bin floor, given a weight
BELOW = 16                         # 1e-11 #/cell: below the gate, given a weight
MNFIX_FLOOR = MNFIX_N_EMPTY        # [#/cell] mnfix.f Neps


def _case():
    xk = jnp.asarray(mass_grid(NBINS))
    xm = jnp.sqrt(xk[:-1] * xk[1:])
    Nk = (jnp.zeros(NBINS).at[MODE].set(2.0e12).at[THIN].set(5.0e9)
          .at[TINY].set(1.0e-3).at[BELOW].set(1.0e-11)
          .at[FLOOR].set(MNFIX_FLOOR))
    Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(Nk * xm)
    # surface-like weights for the bins with particles, plus weights on
    # the empty, floor and below-gate bins, which must not be used (the
    # driver's weights are frozen over a coupling interval, so a bin that
    # emptied meanwhile can still carry one)
    w = Nk * xm ** (2.0 / 3.0)
    tot = jnp.sum(w)
    w = (w.at[EMPTY].set(0.2 * tot).at[BELOW].set(0.1 * tot)
         .at[FLOOR].set(0.1 * tot))
    return xk, Nk, Mk, w / jnp.sum(w)


def test_gate_is_just_above_the_mnfix_floor():
    assert MNFIX_FLOOR < NITRATE_N_MIN < 1.0e-3


@pytest.mark.parametrize("transfer", BIN_TRANSFERS)
def test_thin_bin_gets_its_nitrate_share_and_empty_bin_none(transfer):
    xk, Nk, Mk, w = _case()
    f64 = jnp.float64
    st0 = AerosolState(Nk, Mk, jnp.zeros(N_GAS_SPECIES), xk, f64(293.15),
                       f64(101325.0), f64(0.6), f64(BOXVOL))
    dt = 60.0                                   # one micro-step
    total = 0.01 * float(jnp.sum(Mk[:, SRTSO4]))  # +1 % of the dry mass
    rate = total / dt
    forcing = AerosolForcing(
        h2so4_rate=f64(0.0), org_rate=jnp.zeros(1), nitrate_rate=f64(rate),
        nitrate_weights=w)
    soa = SOAParams(cstar_ref=jnp.ones(1), dhvap=jnp.full(1, 131.0),
                    mw=jnp.full(1, 200.0), alpha=f64(1.0))

    st, diag = aerosol_outer_step(
        st0, forcing, soa, dt, processes=(),
        cfg=AerosolStepConfig(growth_bin_transfer=transfer))

    nk, nit = np.asarray(st.Nk), np.asarray(st.Mk[:, SRTNIT])
    assert np.all(np.isfinite(nk)) and np.all(np.isfinite(np.asarray(st.Mk)))
    assert np.all(nit >= 0.0)

    # shares renormalised over the bins with particles (Nk > NITRATE_N_MIN)
    used = np.asarray(Nk) > NITRATE_N_MIN
    share = np.asarray(w) / np.sum(np.asarray(w)[used])
    # the 0.5 cm^-3 bin gets its share (the remap may move some of its
    # particles up a bin or two; nothing else is nearby)
    assert nit[THIN - 2:THIN + 3].sum() == pytest.approx(
        share[THIN] * total, rel=1e-10)
    # empty, floor and below-gate bins get none, whatever their weight
    assert nit[EMPTY] == 0.0
    assert nit[BELOW] == 0.0
    assert nit[FLOOR] == 0.0 and nit[-1] == 0.0
    # a bin far below one particle per cm^3 still gets its (tiny) share
    assert nit[TINY:TINY + 3].sum() == pytest.approx(share[TINY] * total,
                                                      rel=1e-10)
    assert nit[MODE.start - 2:MODE.stop + 2].sum() == pytest.approx(
        share[MODE].sum() * total, rel=1e-10)

    # all of it is added and every bit of it is in the particles
    added = float(diag.nitrate_added)
    assert added == pytest.approx(total, rel=1e-12)
    assert float(nit.sum()) == pytest.approx(added, rel=1e-12)
    assert float(jnp.sum(st.Nk)) == pytest.approx(float(jnp.sum(Nk)),
                                                  rel=1e-12)


@pytest.mark.slow
def test_level3_leaves_no_band_below_the_mode():
    """Level 3 at 80 bins, 24 h: the nitrate-grown mode leaves no band of
    stranded tail particles at 40-70 nm (it held 3.2 cm^-3 with the old
    1 cm^-3 gate, and more at finer grids; now 1.22 with the Jathar c*
    table at the 300 s preset interval, 1.35 at 900 s, 0.93 with the Li et
    al. 2016 c* at 900 s), and the nitrate total is unchanged."""
    from tomas_jax.box.driver import run
    from tomas_jax.box.scenario import load_preset

    r = run(load_preset("level3_day_night").replace(numerics={"nbins": 80}))
    e = np.asarray(r.diameter_edges_nm)
    dm = np.sqrt(e[:-1] * e[1:])
    n_bin = np.asarray(r.dndlog10dp_cm3[-1]) * np.diff(np.log10(e))
    band = (dm >= 40.0) & (dm < 70.0)
    assert n_bin[band].sum() < 1.5                    # cm^-3 (old gate: 3.16)
    q = np.cumsum(n_bin) / n_bin.sum()
    assert dm[np.searchsorted(q, 1e-3)] > 75.0        # 0.1 % quantile [nm]
    # 3.281 with the Jathar c* table at the 300 s preset interval (2026-10-17; 3.134
    # at 900 s; 5.151 with the Li et al. 2016 c* at 900 s,
    # whose ~8 ug/m3 of nitrate SOA added surface area for N2O5 uptake)
    assert r.totals["NIT_ugm3"][-1] == pytest.approx(3.281, rel=2e-3)
    assert r.diagnostics["warnings"] == []
