"""MCM v3.3.1 (inorganic module) vs SANDBOX JPL 19-5 rate coefficients for shared reactions.

Matches by reactant set after mapping names and folding bath species (M, O2, N2, H2O)
into pseudo-order coefficients; sums every channel with the same reactants on each side
(so lumped vs branched descriptions compare as totals). Reads results/sandbox_rates.json
(from sandbox_dump.py), writes results/rate_crosscheck.json.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tomas_jax.gaschem.kpp_parser import parse_eqn  # noqa: E402
from tomas_jax.gaschem.mechanism import RateModel  # noqa: E402

KB = 1.380649e-23
BATH = {"M", "O2", "N2", "H2O"}
TO_MCM = {"HNO4": "HO2NO2", "H2SO4": "SA", "Cl": "CL", "CH3": "CH3O2"}

sb = json.loads((HERE / "results" / "sandbox_rates.json").read_text())
mech = parse_eqn(HERE / "modules" / "mcm_inorganic_c1_sulfur.eqn")
rm = RateModel(mech)

out = {}
for cname, (t, p, x) in sb["conditions"].items():
    m = p / (KB * t) * 1e-6
    env = {"TEMP": t, "M": m, "O2": 0.2095 * m, "N2": 0.7809 * m, "H2O": x * m}
    k_th, ro2_dep, _ = rm.at_env(env)
    mcm = collections.defaultdict(list)
    for r, k, dep in zip(mech.reactions, k_th, ro2_dep):
        if r.photolysis or dep:
            continue
        mcm[tuple(sorted(r.reactants))].append((r.label, float(k)))
    jpl = collections.defaultdict(list)
    for row in sb["reactions"]:
        k = row["k"][cname]
        if isinstance(k, str):
            continue
        reac = []
        for s, n in row["reactants"].items():
            if s in BATH:
                k *= env[s] ** n if s != "M" else 1.0   # SANDBOX folds M into k already
            else:
                reac += [TO_MCM.get(s, s)] * int(n)
        jpl[tuple(sorted(reac))].append((row["R"], row["equation"], k))
    rows = []
    for key in sorted(set(mcm) & set(jpl)):
        km = sum(k for _, k in mcm[key])
        kj = sum(k for *_, k in jpl[key])
        rows.append({"reactants": " + ".join(key),
                     "mcm": [lab for lab, _ in mcm[key]], "jpl": [rr for rr, *_ in jpl[key]],
                     "k_mcm": km, "k_jpl": kj, "ratio_mcm_over_jpl": km / kj if kj else None})
    out[cname] = {"matched": len(rows),
                  "jpl_only": sorted(" + ".join(k) for k in set(jpl) - set(mcm)),
                  "rows": rows}
(HERE / "results" / "rate_crosscheck.json").write_text(json.dumps(out, indent=1))
for cname, d in out.items():
    print(f"== {cname}: {d['matched']} shared reactant sets")
    for r in d["rows"]:
        print(f"  {r['reactants']:16s} MCM {','.join(r['mcm']):10s} JPL {','.join(r['jpl']):9s} "
              f"k_mcm {r['k_mcm']:.3e}  k_jpl {r['k_jpl']:.3e}  ratio {r['ratio_mcm_over_jpl'] or float('nan'):.3f}")
print("JPL-only reactant sets (surface):", out["surface_298K"]["jpl_only"])
