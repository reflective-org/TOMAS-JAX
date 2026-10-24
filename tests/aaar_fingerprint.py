"""Bit-level fingerprint of a ``RunResult`` (used by tests/test_aaar_presets_unchanged.py).

``fingerprint(result)`` hashes every numeric output of a run (size distributions, totals,
gases, radicals, J, c* classes, top contributors, condensation sink) as float64 bytes,
so two runs give the same fingerprint only when every number is identical to the last
digit. ``diagnostics["runtime_s"]`` and ``diagnostics["code_version"]`` are left out.

Regenerate the stored file after an intended change of a preset's result:

    python -m tests.aaar_fingerprint          (writes tests/reference/aaar_preset_fingerprints.json)
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np

FINGERPRINT_FILE = Path(__file__).parent / "reference" / "aaar_preset_fingerprints.json"
# (preset, duration_h, slow): the short runs guard every commit, the full presets are
# the tutorial's own numbers (pytest -m slow).
CASES = (
    ("level1_volatility", 1.0, False),
    ("level2_limonene_soa", 1.0, False),
    ("level3_day_night", 1.0, False),
    ("level1_volatility", 12.0, True),
    ("level2_limonene_soa", 6.0, True),
    ("level3_day_night", 24.0, True),
    # Level 1 urban air as a whole day with traffic emissions (2026-10-22, A26); the earlier
    # 6 h midday urban preset was not fingerprinted
    ("level1_urban", 1.0, False),
    ("level1_urban", 24.0, True),
)


# The outputs that existed before 2026-10-19 (sorted names). New outputs added later (surface
# and volume distributions, O:C, kappa, loss budgets) are not part of the fingerprint, so
# adding an output never changes it; changing an old number always does.
TOTALS_KEYS = ("H2O_ugm3", "NH4_ugm3", "NIT_ugm3", "N_cm3", "POA_ugm3", "SO4_ugm3", "SOA_ugm3",
               "dry_ugm3")
SERIES_KEYS = {
    "gases_ppb": ("CO", "H2O2", "HCHO", "HNO3", "LIMONENE", "N2O5", "NO", "NO2", "O3", "SO2"),
    "radicals_cm3": ("HO2", "NO3", "OH"),
    "photolysis": ("J_HCHO_H", "J_NO2", "J_NO3_NO2", "J_O3_O1D", "sza_deg"),
}


def _hash(h: "hashlib._Hash", name: str, value) -> None:
    arr = np.asarray(value, dtype=np.float64)
    h.update(name.encode())
    h.update(np.ascontiguousarray(arr).tobytes())


def fingerprint(result) -> dict:
    """{"sha256": ..., "sample": {...}} for a RunResult."""
    h = hashlib.sha256()
    _hash(h, "time_h", result.time_h)
    _hash(h, "diameter_nm", result.diameter_nm)
    _hash(h, "dndlog10dp_cm3", result.dndlog10dp_cm3)
    _hash(h, "dmdlog10dp_ugm3", result.dmdlog10dp_ugm3)
    for key in TOTALS_KEYS:
        _hash(h, f"totals.{key}", result.totals[key])
    for group, keys in SERIES_KEYS.items():
        d = getattr(result, group)
        for key in keys:
            if key in d:
                _hash(h, f"{group}.{key}", [np.nan if v is None else v for v in d[key]])
    _hash(h, "h2so4_cm3", result.h2so4_cm3)
    for key in ("gas_ugm3", "log10_cstar", "particle_ugm3"):
        _hash(h, f"volatility.{key}", result.volatility[key])
    for c in result.top_contributors:
        h.update(c["species"].encode())
        _hash(h, "top.particle_ugm3", c["particle_ugm3"])
    _hash(h, "cs_per_s", result.diagnostics["cs_per_s"])
    _hash(h, "nucleation_rate_cm3_s", result.diagnostics["nucleation_rate_cm3_s"])
    sample = {
        "N_cm3_end": float(result.totals["N_cm3"][-1]),
        "dry_ugm3_end": float(result.totals["dry_ugm3"][-1]),
        "SOA_ugm3_end": float(result.totals["SOA_ugm3"][-1]),
        "NIT_ugm3_end": float(result.totals["NIT_ugm3"][-1]),
    }
    if result.gases_ppb:
        sample["O3_ppb_end"] = float(result.gases_ppb["O3"][-1])
    return {"sha256": h.hexdigest(), "sample": sample}


def case_key(name: str, duration_h: float) -> str:
    return f"{name}@{duration_h:g}h"


def main() -> None:
    from tomas_jax.box.driver import run
    from tomas_jax.box.scenario import load_preset

    out = {}
    for name, hours, _ in CASES:
        out[case_key(name, hours)] = fingerprint(run(load_preset(name).replace(duration_h=hours)))
        print(case_key(name, hours), out[case_key(name, hours)])
    FINGERPRINT_FILE.parent.mkdir(exist_ok=True)
    FINGERPRINT_FILE.write_text(json.dumps(out, indent=2) + "\n")
    print("wrote", FINGERPRINT_FILE)


if __name__ == "__main__":
    main()
