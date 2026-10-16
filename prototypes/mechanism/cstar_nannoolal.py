"""c* at 298.15 K from SMILES with the Nannoolal et al. (2008) vapour pressure and
Nannoolal boiling point (UManSysProp, as PyCHAM's default 'nannoolal' option calls it),
then C* = 1e6 MW Psat / (8.314 * 298.15) as in Shantanu Jathar's table.

Purpose: (1) check that this reproduces Shantanu's limonene table, which would make the
same pipeline usable for products of other VOCs; (2) count how many species of the MCM
SMILES file it covers.

Run with the pycham-bench venv (has openbabel), isolated:
  <pycham-bench>/.venv/bin/python -I cstar_nannoolal.py <umansysprop_root> <mcm_xml> \
      <shantanu_csv> <out.json> [max_species]
"""
import csv
import json
import math
import re
import sys
import time

sys.path.insert(0, sys.argv[1])
from openbabel import pybel  # noqa: E402
from umansysprop import boiling_points, vapour_pressures  # noqa: E402

xml_path, table_path, out_path = sys.argv[2], sys.argv[3], sys.argv[4]
limit = int(sys.argv[5]) if len(sys.argv) > 5 else None
T = 298.15
R = 8.314

smiles = dict(re.findall(r'species_name="([^"]+)">\s*<smiles>([^<]+)</smiles>',
                         open(xml_path).read()))
with open(table_path, newline="") as f:
    rows = [r for r in csv.DictReader(line for line in f if not line.startswith("#"))]
ref = {r["species"].strip(): float(r["cstar_ug_m3_298.15K"]) for r in rows}


def cstar(smi):
    mol = pybel.readstring("smi", smi)
    tb = boiling_points.nannoolal(mol)
    log10_atm = vapour_pressures.nannoolal(mol, T, tb)
    psat_pa = 10.0 ** log10_atm * 101325.0
    return 1e6 * mol.molwt * psat_pa / (R * T)


t0 = time.perf_counter()
cmp_rows, failed = [], []
for name, c_ref in ref.items():
    if name not in smiles:
        failed.append((name, "no SMILES"))
        continue
    try:
        c = cstar(smiles[name])
        cmp_rows.append((name, c_ref, c, math.log10(c) - math.log10(c_ref)))
    except Exception as e:  # noqa: BLE001
        failed.append((name, repr(e)[:80]))
t_ref = time.perf_counter() - t0

# coverage of the whole SMILES file (closed-shell rule applied later, in tomas-jax)
names = sorted(smiles)[:limit] if limit else sorted(smiles)
t0 = time.perf_counter()
ok = bad = 0
for name in names:
    try:
        cstar(smiles[name])
        ok += 1
    except Exception:  # noqa: BLE001
        bad += 1
t_all = time.perf_counter() - t0

d = sorted(abs(r[3]) for r in cmp_rows)
out = {"xml_species": len(smiles), "table_rows": len(ref), "compared": len(cmp_rows),
       "failed": failed[:20], "n_failed": len(failed),
       "abs_dlog10_median": d[len(d) // 2] if d else None,
       "abs_dlog10_p99": d[int(0.99 * (len(d) - 1))] if d else None,
       "abs_dlog10_max": d[-1] if d else None,
       "within_0.01_dex": sum(x < 0.01 for x in d),
       "worst": sorted(cmp_rows, key=lambda r: -abs(r[3]))[:8],
       "seconds_for_table": t_ref,
       "whole_file": {"species_tried": len(names), "ok": ok, "failed": bad, "seconds": t_all}}
json.dump(out, open(out_path, "w"), indent=1)
print(json.dumps(out, indent=1))
