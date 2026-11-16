"""Cases T1-T3: gas phase only, an aromatic VOC + NOx + O3 in sunlight (definition shared by
run.py and analyse.py; T2 and T3 are thin wrappers that select their case).

TOMAS-JAX: preset ``level1_urban`` gases (O3 30, NO 0.5, NO2 15, CO 200, CH4 1900, H2 500 ppb)
with 5 ppb of the VOC (the Urban selector's level), 6 h from 10:00 EDT on 1 July at 40 N,
75 W, TUV-x photolysis, no NO emission, every aerosol process off. The run integrates the
limonene mechanism with that VOC's MCM subset merged in (box.coupling.aromatics_used).
PyCHAM: the same gases, T, P, RH, the same merged mechanism written as one KPP file
(pycham_inputs.write_merged_eqn), J prescribed from TOMAS-JAX per coupling interval, no
particles, no walls, <34> rate 0.
"""
from __future__ import annotations

from pathlib import Path

from tomas_jax.box.scenario import Scenario, load_preset

from ...config import AROMATIC_CASES, CASES_DIR


class AromaticCase:
    """Paths and the scenario of one of the cases T1 (toluene), T2 (benzene), T3 (m-xylene)."""

    def __init__(self, name: str):
        if name not in AROMATIC_CASES:
            raise ValueError(f"unknown aromatic case {name!r}; known: {sorted(AROMATIC_CASES)}")
        self.cfg = AROMATIC_CASES[name]
        self.CASE = name
        self.VOC = self.cfg["voc"]
        self.CASE_DIR = CASES_DIR / name
        self.RESULTS_DIR = self.CASE_DIR / "results"
        self.FIG_DIR = self.CASE_DIR / "figures"
        self.OURS_NPZ = {300: self.RESULTS_DIR / "ours_dt300.npz", 60: self.RESULTS_DIR / "ours_dt60.npz"}
        self.PYCHAM_NPZ = self.RESULTS_DIR / "pycham.npz"
        self.KEY_NPZ = self.RESULTS_DIR / "key_species.npz"
        self.RUN_INFO_JSON = self.RESULTS_DIR / "run_info.json"
        self.SUMMARY_JSON = self.RESULTS_DIR / "summary.json"

    def scenario(self, coupling_dt_s: float | None = None) -> Scenario:
        """The TOMAS-JAX scenario (preset coupling unless ``coupling_dt_s`` is given)."""
        cfg = self.cfg
        s = load_preset(cfg["preset"])
        dt = coupling_dt_s or cfg["coupling_dt_s"]
        return s.replace(name=f"benchmark {self.CASE}", start_utc=cfg["start_utc"], duration_h=cfg["duration_h"],
                         gases_ppb={**s.gases_ppb, self.VOC: cfg["voc_ppb"]},
                         emissions_ppb_per_h={}, emission_profile=None,
                         processes=dict(cfg["processes"], chemistry=True), gamma_n2o5=0.0,
                         numerics={"coupling_dt_s": dt, "output_dt_s": dt})


DEFAULT_CASE = "T1"
