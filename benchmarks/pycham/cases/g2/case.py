"""Case G2: gas phase only, Level 2 conditions (definition shared by run.py and analyse.py).

TOMAS-JAX: preset ``level2_limonene_soa`` (initial gases, T, P, RH, 6 h, constant light
J_NO2 = 5e-3 s-1 at SZA 30) with every aerosol process off, so the seed is a spectator.
PyCHAM: same gases/T/P/RH, no particles (number_size_bins = 0), no walls, no dilution,
J prescribed from TOMAS-JAX (J.xlsx), <34> N2O5 -> NA rate set to 0 (TOMAS-JAX replaces
it by gamma uptake, gamma = 0 here), update_step 60 s.
"""
from __future__ import annotations

from pathlib import Path

from tomas_jax.box.scenario import Scenario, load_preset

from ...config import G2

CASE = G2["name"]
CASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CASE_DIR / "results"
FIG_DIR = CASE_DIR / "figures"

# result files (small, kept in the repo)
OURS_NPZ = {300: RESULTS_DIR / "ours_dt300.npz", 60: RESULTS_DIR / "ours_dt60.npz"}
# attribution runs (coupling 60 s): tight solver tolerance, + PyCHAM's water vapour
OURS_ATTR_NPZ = {"dt60_tight": RESULTS_DIR / "ours_dt60_tight.npz",
                 "dt60_tight_pychamH2O": RESULTS_DIR / "ours_dt60_tight_pychamH2O.npz"}
PYCHAM_NPZ = RESULTS_DIR / "pycham.npz"
PYCHAM_TIGHT_NPZ = RESULTS_DIR / "pycham_tight.npz"
KEY_NPZ = RESULTS_DIR / "key_species_60s.npz"
TOP_CONTRIBUTORS_JSON = RESULTS_DIR / "level2_top_contributors.json"
RUN_INFO_JSON = RESULTS_DIR / "run_info.json"
RATES_CHECK_JSON = RESULTS_DIR / "rate_constant_check.json"


def scenario(coupling_dt_s: float | None = None, seed_number_cm3: float | None = None) -> Scenario:
    """The G2 TOMAS-JAX scenario (preset coupling unless ``coupling_dt_s`` is given)."""
    s = load_preset(G2["preset"])
    numerics = {"coupling_dt_s": coupling_dt_s or G2["coupling_dt_s"]}
    numerics["output_dt_s"] = numerics["coupling_dt_s"]
    s = s.replace(name=f"benchmark {CASE}", processes=dict(G2["processes"], chemistry=True),
                  gamma_n2o5=0.0, numerics=numerics)
    if seed_number_cm3 is not None:
        modes = [{"number_cm3": seed_number_cm3, "gmd_nm": m.gmd_nm, "gsd": m.gsd}
                 for m in s.seed.modes]
        s = s.replace(seed={"modes": modes})
    return s


def full_level2_scenario() -> Scenario:
    """Unmodified Level 2 preset (aerosol on): source of the particle-relevant species."""
    return load_preset(G2["preset"])
