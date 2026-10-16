"""Extract the MCM inorganic + CH4 + sulfur subset from limonene_mcm.eqn and check it.

Seeds: every species without carbon, plus CH4 and CO. The closure adds every product of
a reaction whose reactants are all already in the set, so the result is what an MCM
export of "methane + inorganic" contains, as far as limonene_mcm.eqn holds it.
MCM's three gas-particle sinks are kept (<33> HNO3 = NA, <34> N2O5 = NA + NA,
<35> SO3 = SA); the registry lets a mechanism keep, drop or replace them (the
tutorial replaces <34> by gamma-based uptake, ASSUMPTIONS A1).

Run: PYTHONPATH=<mech-wt> python build_inorganic.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from mechlib import atom_imbalance, extract_closure, write_eqn  # noqa: E402
from tomas_jax.gaschem.config import LIMONENE_EQN  # noqa: E402
from tomas_jax.gaschem.kpp_parser import parse_eqn  # noqa: E402
from tomas_jax.gaschem.mechanism import RateModel  # noqa: E402

full = parse_eqn(LIMONENE_EQN)
seeds = {s for s, a in full.atoms.items() if not a.get("C")} | {"CH4", "CO"}
sub = extract_closure(full, seeds)
out = HERE / "modules" / "mcm_inorganic_c1_sulfur.eqn"
out.parent.mkdir(exist_ok=True)
write_eqn(sub, out, [
    "MCM v3.3.1 inorganic + methane + SO2 subset, extracted (prototype) from",
    "tomas_jax/gaschem/mechanisms/limonene_mcm.eqn by prototypes/mechanism/build_inorganic.py.",
    "Labels and rate expressions verbatim. <33>-<35> are MCM's gas-particle sinks (NA, SA).",
])
back = parse_eqn(out)
assert [r.label for r in back.reactions] == [r.label for r in sub.reactions]
assert back.species == sub.species and back.ro2 == sub.ro2

rm = RateModel(back)
env = {"TEMP": 298.15, "M": 2.46e19, "O2": 0.2095 * 2.46e19, "N2": 0.7809 * 2.46e19,
       "H2O": 3.9e17}
k_th, ro2_dep, j_coef = rm.at_env(env)

imb = {r.label: atom_imbalance(back, r) for r in back.reactions}
by_el = collections.Counter(el for d in imb.values() for el in d)
sulfur = [f"<{r.label}> {' + '.join(r.reactants)} = {' + '.join(p for p, _ in r.products)}"
          for r in back.reactions if any("S" in back.atoms.get(s, {}) for s in r.reactants)]
summary = {
    "species": len(back.species),
    "reactions": len(back.reactions),
    "photolysis": sum(r.photolysis for r in back.reactions),
    "ro2_members": back.ro2,
    "ro2_dependent": int(ro2_dep.sum()),
    "j_names": rm.j_names,
    "species_list": back.species,
    "sulfur_reactions": sulfur,
    "unbalanced_by_element": dict(by_el),
    "unbalanced_examples": {lab: d for lab, d in imb.items() if d and set(d) - {"O", "H"}},
    "excluded_criegee_so2": [f"<{r.label}>" for r in full.reactions
                             if "SO2" in r.reactants and r not in sub.reactions],
}
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "inorganic_summary.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k != "species_list"}, indent=1))
