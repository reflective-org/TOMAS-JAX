"""Composition rules on real files: inorganic module + limonene, a rate conflict, a
JPL-vs-MCM overlap, the RO2 check, atom-balance classes, SANDBOX species coverage and
photolysis-name coverage. Writes results/compose_summary.json.

Run (after build_inorganic.py):
    PYTHONPATH=<mech-wt> TUVX_JAX_ROOT=<tuvx-jax> python compose_demo.py
"""
from __future__ import annotations

import collections
import dataclasses
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from mechlib import (ComposeError, Module, atom_imbalance, compose,  # noqa: E402
                     explain_residual, reaction_key, ro2_consistency)
from tomas_jax.gaschem.config import LIMONENE_EQN  # noqa: E402
from tomas_jax.gaschem.kpp_parser import ParsedMechanism, Reaction, parse_eqn  # noqa: E402
from tomas_jax.gaschem.mcm_rates import MCM_J_PARAMS  # noqa: E402
from tomas_jax.gaschem.mechanism import RateModel  # noqa: E402
from tomas_jax.gaschem.photolysis import SOURCES, TUVX_DIRECT, TUVX_SPLIT  # noqa: E402

ENV = {"TEMP": 298.15, "M": 2.46e19, "O2": 0.2095 * 2.46e19, "N2": 0.7809 * 2.46e19,
       "H2O": 3.9e17}
out: dict = {}

inorg = parse_eqn(HERE / "modules" / "mcm_inorganic_c1_sulfur.eqn")
lim = parse_eqn(LIMONENE_EQN)

# 1. inorganic + limonene: every inorganic reaction is a duplicate, result == limonene
comp, rep = compose([Module("mcm_inorganic", inorg, "mcm"), Module("limonene", lim, "mcm")])
same = (sorted(map(reaction_key, comp.reactions)) == sorted(map(reaction_key, lim.reactions)))
out["inorganic_plus_limonene"] = {
    "species": len(comp.species), "reactions": len(comp.reactions),
    "duplicates_removed": len(rep.duplicates), "equals_limonene_file": same,
    "ro2_members": len(comp.ro2), "ro2_equal": sorted(comp.ro2) == sorted(lim.ro2)}

# 2. same reaction, different rate (e.g. someone swaps KMT12 for a Sun et al. 2022 value)
patched = dataclasses.replace(inorg, reactions=[
    dataclasses.replace(r, rate_expr="2.0E-12") if r.label == "31" else r for r in inorg.reactions])
try:
    compose([Module("mcm_inorganic", inorg, "mcm"), Module("patched", patched, "mcm")])
    out["rate_conflict"] = "not detected"
except ComposeError as e:
    out["rate_conflict"] = str(e)

# 3. JPL (SANDBOX R79, lumped OH + SO2 -> SO3 + HO2) next to MCM (<31>, <32>, via HSO3)
jpl = ParsedMechanism(species=["SO2", "OH", "SO3", "HO2"],
                      atoms={s: inorg.atoms[s] for s in ("SO2", "OH", "SO3", "HO2")},
                      reactions=[Reaction("R79", ("SO2", "OH"), (("SO3", 1.0), ("HO2", 1.0)),
                                          "KMT12", False)], ro2=[])
try:
    compose([Module("mcm_inorganic", inorg, "mcm"), Module("jpl_sulfur", jpl, "jpl")])
    out["overlap_default"] = "not detected"
except ComposeError as e:
    out["overlap_default"] = str(e)
c2, _ = compose([Module("mcm_inorganic", inorg, "mcm"), Module("jpl_sulfur", jpl, "jpl")],
                allow_overlap={("OH", "SO2"): "jpl_sulfur"})
out["overlap_resolved_jpl_wins"] = {
    "reactions": len(c2.reactions),
    "so2_oh_channels": [r.label for r in c2.reactions if sorted(r.reactants) == ["OH", "SO2"]],
    "HSO3_still_present": "HSO3" in c2.species,
}

# 4. RO2: every RO2-dependent reaction's reactant must be in the RO2 sum
rm = RateModel(lim, external=("34",))
_, ro2_dep, j_coef = rm.at_env(ENV)
dep_labels = {lab for lab, d in zip(rm.labels, ro2_dep) if d}
out["ro2_check_limonene"] = {"ro2_dependent": len(dep_labels),
                             "reactant_not_in_ro2": ro2_consistency(lim, dep_labels)}

# 5. atom balance classes for the limonene file
cls = collections.Counter()
other = []
for r in lim.reactions:
    imb = atom_imbalance(lim, r)
    if r.label in dep_labels and imb:
        cls["RO2 permutation (partner RO2 not written)"] += 1
        continue
    if not imb:
        cls["balanced"] += 1
        continue
    ex = explain_residual(imb)
    if ex is None:
        cls["not CO2/H2O/O2"] += 1
        other.append((r.label, imb))
    else:
        cls["omitted " + "+".join(k for k, v in ex.items() if v)] += 1
out["atom_balance_limonene"] = {"classes": dict(cls), "unexplained": other[:20],
                                "n_unexplained": len(other)}

# 6. SANDBOX species (stratchem-jax, 36) -> canonical MCM names
SANDBOX = ["Cl", "ClO", "ClOOCl", "ClONO2", "HCl", "Cl2", "NO", "NO2", "O", "O2", "O3", "CH3",
           "CH4", "HNO3", "H2O", "HOCl", "N2O5", "NO3", "OH", "HO2", "O1D", "HONO", "HNO4",
           "OClO", "Br", "BrO", "BrONO2", "BrCl", "HBr", "HOBr", "HNO3aq", "C2H6", "SO2",
           "H2O2", "SO3", "H2SO4"]
TO_MCM = {"HNO4": "HO2NO2", "H2SO4": "SA", "Cl": "CL", "HCl": "HCL", "Br": "BR", "CH3": "CH3O2"}
cover = {}
for s in SANDBOX:
    name = TO_MCM.get(s, s.upper() if any(h in s for h in ("Cl", "Br")) else s)
    if s in ("O2", "H2O"):
        cover[s] = "environment (fixed; enters rates as O2, H2O)"
    elif s == "HNO3aq":
        cover[s] = "aerosol counter (like NA_HET), not a gas species"
    elif s == "CH3":
        cover[s] = "folded: CH3 + O2 -> CH3O2 is instantaneous in MCM"
    elif name in inorg.species:
        cover[s] = f"mcm_inorganic as {name}"
    elif name in lim.atoms:
        cover[s] = f"declared in limonene_mcm.eqn as {name} but no chemistry"
    elif s == "C2H6":
        cover[s] = "MCM ethane export (not in the file)"
    else:
        cover[s] = "not in MCM: JPL halogen module"
out["sandbox_coverage"] = cover
out["sandbox_coverage_counts"] = dict(collections.Counter(v.split(" as ")[0].split(":")[0]
                                                          for v in cover.values()))

# 7. photolysis names: what MCM v3.3.1 uses (34 J) vs what photolysis.SOURCES maps today
out["j_names"] = {
    "mcm_parameterised": len(MCM_J_PARAMS),
    "mapped_in_photolysis_SOURCES": len(SOURCES),
    "mcm_without_source": sorted(set(MCM_J_PARAMS) - set(SOURCES)),
    "tuvx_direct": len(TUVX_DIRECT), "tuvx_split": len(TUVX_SPLIT),
}
try:
    from tomas_jax.gaschem.photolysis import tuvx_root
    from tuvx_photolysis import PhotolysisCalculator
    from tomas_jax.gaschem.config import TUVX_CONFIG
    root = tuvx_root()
    calc = PhotolysisCalculator.from_tuvx_json(root / TUVX_CONFIG, data_root=root)
    names = set(calc.reaction_names())
    want = ["CH3CHO+hv->CH3+HCO", "C2H5CHO+hv->C2H5+HCO", "C2H5ONO2+hv->C2H5O+NO2",
            "C(CH3)3(ONO2)+hv->C(CH3)(O.)+NO2", "N2O5+hv->NO2+NO3", "HNO4+hv->HO2+NO2",
            "PAN+hv->CH3CO(OO)+NO2", "Cl2+hv->Cl+Cl", "ClONO2+hv->Cl+NO3",
            "ClONO2+hv->ClO+NO2", "ClOOCl+hv->Cl+ClOO", "HOCl+hv->HO+Cl", "OClO+hv->Products",
            "ClNO2+hv->Cl+NO2", "BrO+hv->Br+O", "BrONO2+hv->Br+NO3", "BrONO2+hv->BrO+NO2",
            "BrCl+hv->Br+Cl", "HOBr+hv->OH+Br", "HCl+hv->H+Cl", "Br2+hv->Br+Br"]
    out["tuvx_port"] = {"n_reactions": len(names),
                        "needed_for_new_modules": {w: (w in names) for w in want}}
except Exception as e:  # tuvx not importable: record and continue
    out["tuvx_port"] = f"not checked: {type(e).__name__}: {e}"

(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "compose_summary.json").write_text(json.dumps(out, indent=1, default=str))
print(json.dumps(out, indent=1, default=str))
