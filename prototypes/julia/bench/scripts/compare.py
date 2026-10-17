"""Score a run file against a reference with the benchmark metric.

    python compare.py <run.json> <reference.json>

Both files have the reference format (key_species at every interval end, and a sibling
*_hourly.npz with every species at every full hour). Prints the key-species metric and
the all-species hourly metric as JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import METRIC_FLOOR, all_species_error, key_errors  # noqa: E402


def score(run_path: Path, ref_path: Path) -> dict:
    run = json.loads(Path(run_path).read_text())
    ref = json.loads(Path(ref_path).read_text())
    assert run["problem"] == ref["problem"] and run["coupling_s"] == ref["coupling_s"]
    res = {"run": Path(run_path).name, "reference": Path(ref_path).name,
           "floor_molec_cm3": METRIC_FLOOR,
           "key": key_errors(run["key_species"], ref["key_species"])}
    hr = Path(str(run_path).replace(".json", "_hourly.npz"))
    hf = Path(str(ref_path).replace(".json", "_hourly.npz"))
    if hr.exists() and hf.exists():
        res["all_species_hourly_max"] = all_species_error(np.load(hr)["y"], np.load(hf)["y"])
    return res


if __name__ == "__main__":
    print(json.dumps(score(Path(sys.argv[1]), Path(sys.argv[2])), indent=1))
