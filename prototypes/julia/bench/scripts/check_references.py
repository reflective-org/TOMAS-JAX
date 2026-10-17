"""Cross-checks of the reference solutions.

1. Method check: scipy Radau (rtol 1e-9, atol 1e-3) against the BDF reference
   (rtol 1e-10, atol 1e-4) for the limonene problems.
2. Equivalence check: a synthetic problem with the precursor split equally over N copies
   is mathematically the limonene Level 3 problem (no reaction has two C>=4 reactants or
   makes a C>=4 species from smaller ones), so shared species must equal the lim726_L3
   reference and each copy of a C>=4 species must equal 1/N of it.

Writes ../reference/CHECKS.json and prints it.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import METRIC_FLOOR, REFERENCE, load_problem  # noqa: E402
from compare import score  # noqa: E402


def main():
    out = {"floor_molec_cm3": METRIC_FLOOR, "radau_vs_bdf": {}, "synthetic_vs_limonene": {}}
    for p in ("lim726_L2", "lim726_L3"):
        for c in (900, 60):
            a, b = REFERENCE / f"chkradau_{p}_dt{c}.json", REFERENCE / f"ref_{p}_dt{c}.json"
            if a.exists() and b.exists():
                s = score(a, b)
                out["radau_vs_bdf"][f"{p}_dt{c}"] = {"key_max": s["key"]["max"],
                                                     "key_worst": s["key"]["worst_species"],
                                                     "all_species_hourly_max":
                                                         s["all_species_hourly_max"]}
    lim = load_problem("lim726_L3")
    big = [s for s in lim.mech.species if s not in ("SA", "NA_HET")]
    for name, ncopy in (("syn1353_L3", 2), ("syn4488_L3", 7), ("syn5742_L3", 9)):
        P = load_problem(name)
        for c in (900, 60):
            fs, fl = REFERENCE / f"ref_{name}_dt{c}_hourly.npz", REFERENCE / f"ref_lim726_L3_dt{c}_hourly.npz"
            if not (fs.exists() and fl.exists()):
                continue
            ys, yl = np.load(fs)["y"], np.load(fl)["y"]
            # map every synthetic species onto its limonene species and copy factor
            worst, worst_sp = 0.0, None
            for i, s in enumerate(P.mech.species):
                base, _, suffix = s.rpartition("_")
                if suffix.isdigit() and base in lim.mech.species:
                    j, fac = lim.mech.index(base), 1.0 / ncopy
                else:
                    j = lim.mech.index(s)
                    is_copied = any(f"{s}_{k}" in P.mech.species for k in range(1, ncopy))
                    fac = 1.0 / ncopy if is_copied else 1.0
                e = np.max(np.abs(ys[:, i] - fac * yl[:, j])
                           / np.maximum(np.abs(fac * yl[:, j]), METRIC_FLOOR / ncopy))
                if e > worst:
                    worst, worst_sp = float(e), s
            out["synthetic_vs_limonene"][f"{name}_dt{c}"] = {"all_species_hourly_max": worst,
                                                              "worst_species": worst_sp}
    (REFERENCE / "CHECKS.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
