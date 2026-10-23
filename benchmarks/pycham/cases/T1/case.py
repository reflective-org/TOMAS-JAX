"""Case T1: gas phase only, toluene + NOx + O3 in sunlight (definition shared by run.py and
analyse.py).

TOMAS-JAX: preset ``level1_urban`` gases (O3 30, NO 0.5, NO2 15, CO 200, CH4 1900, H2 500 ppb)
with 5 ppb toluene (the Urban toggle's "on" value), 6 h from 10:00 EDT on 1 July at 40 N,
75 W, TUV-x photolysis, no NO emission, every aerosol process off. The run integrates the
limonene + toluene mechanism (box.coupling.uses_toluene). PyCHAM: the same gases, T, P, RH,
the same merged mechanism written as one KPP file (pycham_inputs.write_merged_eqn), J
prescribed from TOMAS-JAX per coupling interval, no particles, no walls, <34> rate 0.
"""
from __future__ import annotations

from pathlib import Path

from tomas_jax.box.scenario import Scenario, load_preset

from ...config import T1

CASE = T1["name"]
CASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CASE_DIR / "results"
FIG_DIR = CASE_DIR / "figures"

OURS_NPZ = {300: RESULTS_DIR / "ours_dt300.npz", 60: RESULTS_DIR / "ours_dt60.npz"}
PYCHAM_NPZ = RESULTS_DIR / "pycham.npz"
KEY_NPZ = RESULTS_DIR / "key_species.npz"
RUN_INFO_JSON = RESULTS_DIR / "run_info.json"
SUMMARY_JSON = RESULTS_DIR / "summary.json"


def scenario(coupling_dt_s: float | None = None) -> Scenario:
    """The T1 TOMAS-JAX scenario (preset coupling unless ``coupling_dt_s`` is given)."""
    s = load_preset(T1["preset"])
    dt = coupling_dt_s or T1["coupling_dt_s"]
    return s.replace(name=f"benchmark {CASE}", start_utc=T1["start_utc"], duration_h=T1["duration_h"],
                     gases_ppb={**s.gases_ppb, "TOLUENE": T1["toluene_ppb"]},
                     emissions_ppb_per_h={}, emission_profile=None,
                     processes=dict(T1["processes"], chemistry=True), gamma_n2o5=0.0,
                     numerics={"coupling_dt_s": dt, "output_dt_s": dt})
