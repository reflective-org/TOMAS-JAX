"""SABRE-binned output operator (sensitivities/dunne_nuc/sabre_bins.py)."""
import jax
import jax.numpy as jnp
import numpy as np

from tomas_jax.core.config import make_grid, make_grid_80bin, XK0, PI, SRTSO4, SRTH2O
from sensitivities.dunne_nuc.common import lognormal_state, BOXVOL
from sensitivities.dunne_nuc.sabre_bins import (
    SABRE_EDGES_M, SABRE_MID_M, NMASS_CUTOFFS_M,
    dry_density, model_bin_diameters, rebin_to_sabre, nmass_cumulative,
)

GRIDS = [make_grid(40, XK0, 2.0), make_grid_80bin()]


def _total_from_dndlog(dndlog):
    return float(jnp.sum(dndlog / 20.0))


def test_edges_match_paper():
    assert SABRE_EDGES_M.size == 64
    np.testing.assert_allclose(SABRE_EDGES_M[0], 3e-9)
    np.testing.assert_allclose(np.diff(np.log10(SABRE_EDGES_M)), 0.05)
    assert 4e-6 < SABRE_EDGES_M[-1] < 4.5e-6


def test_dry_density_ignores_water_and_box_size():
    xk = GRIDS[0]
    _, Mk, _ = lognormal_state(xk)
    rho = dry_density(Mk)
    # pure H2SO4, CRC at 100 wt%, in every bin incl. sparse ones
    np.testing.assert_allclose(np.array(rho), 1836.0, rtol=1e-12)
    wet = Mk.at[:, SRTH2O].set(Mk[:, SRTSO4])
    np.testing.assert_allclose(np.array(dry_density(wet)), np.array(rho), rtol=1e-12)
    np.testing.assert_allclose(np.array(dry_density(Mk * 1e-12)), np.array(rho),
                               rtol=1e-12)


def test_total_number_conserved():
    for xk in GRIDS:
        Nk, Mk, _ = lognormal_state(xk, n_total=10.0, gmd=100e-9, gsd=1.6)
        got = _total_from_dndlog(rebin_to_sabre(Nk, Mk, xk, BOXVOL))
        # model bins entirely inside 3 nm - 4.24 um
        d_lo, d_hi = model_bin_diameters(Mk, xk)
        inside = (d_lo >= SABRE_EDGES_M[0]) & (d_hi <= SABRE_EDGES_M[-1])
        expect = float(jnp.sum(jnp.where(inside, Nk, 0.0)) / BOXVOL)
        assert abs(got - expect) / expect < 1e-6


def test_lognormal_gmd_recovered():
    """With the init density, the SABRE-binned GMD is the input GMD."""
    for xk in GRIDS:
        Nk, Mk, _ = lognormal_state(xk, n_total=10.0, gmd=100e-9, gsd=1.6)
        dndlog = np.array(rebin_to_sabre(Nk, Mk, xk, BOXVOL))
        gmd = np.exp(np.sum(dndlog * np.log(SABRE_MID_M)) / np.sum(dndlog))
        # model density (~1890) vs init density (1770): Dp scaled by (1770/rho)^(1/3)
        rho = float(dry_density(Mk)[0])
        np.testing.assert_allclose(gmd, 100e-9 * (1770.0 / rho) ** (1 / 3), rtol=0.02)


def test_nmass_consistent_with_bins():
    xk = GRIDS[1]
    Nk, Mk, _ = lognormal_state(xk, n_total=10.0, gmd=100e-9, gsd=1.6)
    Nk = Nk.at[5].add(5e6)   # add a nucleation-mode spike (~4 nm)
    cum = np.array(nmass_cumulative(Nk, Mk, xk, BOXVOL))
    assert np.all(np.diff(cum) <= 0.0)
    # 3 nm is the first SABRE edge, so N(>3 nm) = sum of all SABRE bins
    np.testing.assert_allclose(
        cum[0], _total_from_dndlog(rebin_to_sabre(Nk, Mk, xk, BOXVOL)), rtol=1e-6)


def test_ammonium_bins_capped_at_100_wt():
    """With NH4, calc_density path: dry (NH4)HSO4 sits at 100 wt%, not above."""
    from tomas_jax.core.config import ICOMP, SRTNH4
    Mk = jnp.zeros((1, ICOMP)).at[0, SRTSO4].set(96.0).at[0, SRTNH4].set(18.0)
    x = 100.0   # Tang (1997) ammonium bisulfate fit at 100 wt%
    expect = 1000.0 * (0.9971 + x * (5.87e-3 + x * (-1.89e-6 + x * 1.763e-7)))
    np.testing.assert_allclose(float(dry_density(Mk)[0]), expect, rtol=1e-12)


def test_water_frac_shifts_up():
    xk = GRIDS[1]
    Nk, Mk, _ = lognormal_state(xk)
    d0, _ = model_bin_diameters(Mk, xk)
    d1, _ = model_bin_diameters(Mk, xk, water_frac=0.1)
    ratio = np.array(d1 / d0)
    assert np.all(ratio > 1.0) and np.all(ratio < 1.1)


def test_differentiable_and_linear_in_Nk():
    xk = GRIDS[0]
    Nk, Mk, _ = lognormal_state(xk)
    J = jax.jacfwd(lambda n: rebin_to_sabre(n, Mk, xk, BOXVOL))(Nk)
    assert np.all(np.isfinite(np.array(J)))
    np.testing.assert_allclose(np.array(J @ Nk),
                               np.array(rebin_to_sabre(Nk, Mk, xk, BOXVOL)), rtol=1e-12)
    Jm = jax.jacfwd(lambda m: nmass_cumulative(Nk, m, xk, BOXVOL))(Mk)
    assert np.all(np.isfinite(np.array(Jm)))
