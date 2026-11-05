"""Phase 3/4 scenarios: stratospheric background (mid-latitude, TTL) and an
SAI plume.

All start from the Phase 0-2 background aerosol (common.lognormal_state:
10 cm-3, GMD 100 nm, GSD 1.6), run 48 h with fion = 10, NH3 = 0, the H2SO4
water scheme and sparse density 1700, and constant OH = 1e6 (common.OH_CONC).
"""
import numpy as np

from tomas_jax.physics.so2_chemistry import _BUCK_A, _BUCK_B, _BUCK_C, _BUCK_D
from .common import lognormal_state, ppbv_to_molec

HOURS = 48


def rh_from_h2o(h2o_vmr, temp, pres):
    """RH over liquid water from an H2O mixing ratio (Buck 1981, as calc_k1_so2_oh)."""
    tc = temp - 273.15
    e_sat = _BUCK_A * np.exp((_BUCK_B - tc / _BUCK_D) * tc / (tc + _BUCK_C))
    return h2o_vmr * pres / e_sat


def _scenario(temp, pres, h2o_vmr, so2_ppbv, kdil=None, background=None,
              below_cloud=False, dt=None, grids=(40, 80), dt_note=''):
    return dict(cond=dict(temp=temp, pres=pres, rh=rh_from_h2o(h2o_vmr, temp, pres),
                          alpha=1.0),
                h2o_vmr=h2o_vmr, so2_ppbv=so2_ppbv, kdil=kdil,
                background=background, below_cloud=below_cloud, dt=dt,
                grids=grids, dt_note=dt_note)


SCENARIOS = {
    # ~20 km mid-latitude
    'bg_midlat': _scenario(217.0, 5500.0, 5.0e-6, 0.02),
    # tropical tropopause layer: below CLOUD's 208 K (Dunne fit extrapolated)
    'bg_ttl': _scenario(195.0, 10000.0, 4.0e-6, 0.05, below_cloud=True),
    # 10 ppmv SO2 plume diluting toward bg_midlat at volume doubling per day
    # (benchmark_dilution stratosphere profile)
    # dt fixed at 0.5 s (user decision 2026-11-03): the 48 h dt ladder gave
    # N(>3 nm) 4.75e4 / 4.30e4 at 2 / 0.5 s (10% apart) while N(>12 nm)
    # (0.5%) and the H2SO4-calibration sensitivity (1.75 / 1.76) had
    # converged; 0.125 s would cost ~4x more again. First run on 40 bins only;
    # the 80-bin Jacobian and Phase 4 were added afterwards (the FD check
    # stays 40-bin, as for the other scenarios).
    'sai_plume': _scenario(217.0, 5500.0, 5.0e-6, 1.0e4,
                           kdil=np.log(2.0) / 86400.0, background='bg_midlat',
                           dt=0.5, grids=(40, 80),
                           dt_note='N(>3 nm) ~10% from converged at 0.5 s'),
}


def initial_state(name, xk):
    """(Nk, Mk, Gc) at t = 0 for a scenario."""
    sc = SCENARIOS[name]
    c = sc['cond']
    return lognormal_state(xk, so2=ppbv_to_molec(sc['so2_ppbv'], c['temp'], c['pres']))


def runner_kwargs(name, xk):
    """make_runner kwargs for a scenario (conditions, dilution)."""
    sc = SCENARIOS[name]
    kw = dict(cond=sc['cond'])
    if sc['kdil'] is not None:
        Nk_bg, Mk_bg, Gc_bg = initial_state(sc['background'], xk)
        kw.update(kdil=sc['kdil'], Nk_bg=Nk_bg, Mk_bg=Mk_bg, Gc_bg=Gc_bg)
    return kw
