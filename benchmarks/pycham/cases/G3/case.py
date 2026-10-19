"""Case G3: gas phase only, Level 3 day/night conditions (shared by run.py and analyse.py).

TOMAS-JAX: preset ``level3_day_night`` (24 h from 2026-07-01T16:00Z = 12:00 EDT, Buffalo
42.89 N 78.88 W, T 293.15 K, RH 0.6, TUV-x photolysis, NO2 held at 1 ppb) with every
aerosol process off and gamma(N2O5) = 0, so the seed is a spectator.
PyCHAM: same gases/T/P/RH, no particles, walls or dilution; J prescribed from TOMAS-JAX
through J.xlsx; NO2 held with ``const_comp = NO2``; <34> N2O5 -> NA rate set to 0;
update_step 60 s.

The case settings (``G3``) live in ``benchmarks/pycham/config.py`` with the other cases.
"""
from __future__ import annotations

from pathlib import Path

from tomas_jax.box.scenario import Scenario, load_preset

from ...config import G3


# every coarser grid must contain the finer ones (comparisons pair times exactly)
for _fine, _coarse in (("key_dt_s", "compare_dt_s"), ("compare_dt_s", "attribution_dt_s"),
                       ("coupling_dt_s", "compare_dt_s")):
    if G3[_coarse] % G3[_fine] != 0:
        raise ValueError(f"G3[{_coarse!r}] must be a multiple of G3[{_fine!r}]")
for _, _t_h in G3["snapshots"]:
    if (_t_h * 3600.0) % G3["compare_dt_s"] != 0:
        raise ValueError(f"snapshot {_t_h} h is not on the compare grid")

CASE = G3["name"]
CASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CASE_DIR / "results"
FIG_DIR = CASE_DIR / "figures"
FIG_PREFIX = "G3_"

# PyCHAM variants: J on the 60 s grid (= TOMAS-JAX at 60 s coupling, the reference), J on
# the 900 s grid (= TOMAS-JAX at the preset coupling), and the 60 s J at rtol 1e-6.
PYCHAM_VARIANTS = {
    "j60": {"j_dt_s": 60.0, "tight": False},
    "j900": {"j_dt_s": 900.0, "tight": False},
    "j60_tight": {"j_dt_s": 60.0, "tight": True},
}
PYCHAM_NPZ = {v: RESULTS_DIR / f"pycham_{v}.npz" for v in PYCHAM_VARIANTS}

# TOMAS-JAX runs: tag -> (coupling [s], tight tolerance, PyCHAM water, NO2 held in-solver)
OURS_RUNS = {
    "dt900": (900.0, False, False, False),
    "dt60": (60.0, False, False, False),
    "dt900_heldNO2": (900.0, False, False, True),
    "dt60_tight": (60.0, True, False, False),
    "dt60_tight_pychamH2O": (60.0, True, True, False),
    "dt60_tight_pychamH2O_heldNO2": (60.0, True, True, True),
}
OURS_NPZ = {tag: RESULTS_DIR / f"ours_{tag}.npz" for tag in OURS_RUNS}
# all-species files on the compare grid (900 s); every other run on the attribution grid
MAIN_RUNS = ("dt900", "dt60")
MAIN_PYCHAM = ("j60",)
KEY_NPZ = RESULTS_DIR / "key_species.npz"
TOP_CONTRIBUTORS_JSON = RESULTS_DIR / "level3_top_contributors.json"
RUN_INFO_JSON = RESULTS_DIR / "run_info.json"
RATES_CHECK_JSON = RESULTS_DIR / "rate_constant_check.json"
SUN_JSON = RESULTS_DIR / "sun.json"


def scenario(coupling_dt_s: float | None = None, seed_number_cm3: float | None = None) -> Scenario:
    """The G3 TOMAS-JAX scenario (G3["coupling_dt_s"], 900 s, unless ``coupling_dt_s`` is given)."""
    s = load_preset(G3["preset"])
    if tuple(s.hold_constant) != G3["held_species"]:
        raise ValueError(f"preset holds {s.hold_constant}, G3 expects {G3['held_species']}")
    dt = coupling_dt_s or G3["coupling_dt_s"]
    s = s.replace(name=f"benchmark {CASE}", processes=dict(G3["processes"], chemistry=True),
                  gamma_n2o5=G3["gamma_n2o5"],
                  numerics={"coupling_dt_s": dt, "output_dt_s": dt})
    if seed_number_cm3 is not None:
        modes = [{"number_cm3": seed_number_cm3, "gmd_nm": m.gmd_nm, "gsd": m.gsd}
                 for m in s.seed.modes]
        s = s.replace(seed={"modes": modes})
    return s


def full_level3_scenario() -> Scenario:
    """Unmodified Level 3 preset (aerosol on): source of the particle-relevant species."""
    return load_preset(G3["preset"])
