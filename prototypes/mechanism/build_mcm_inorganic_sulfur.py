"""Extract the MCM inorganic + CH4 + sulfur subset that the SANDBOX comparison uses.

Seeds (the species a user would set initial values for):
    {O3, NO, NO2, CO, CH4, SO2, H2O2, HONO, HNO3, N2O5, H2, H2O}
H2O is not a state species in MCM (it appears inside rate expressions) and is ignored
as a seed. The closure adds every product of a reaction whose reactants are all already
in the set (fixed point), so no limonene-derived species can enter: limonene itself and
every species made from it are never reachable from these seeds. Rate expressions and
J(...) references are copied verbatim.

Atom balance: for every reaction, products minus reactants per element. MCM KPP files
omit O2, H2O and CO2 as products, declare the aerosol sinks NA and SA as IGNORE and
write RO2 self/cross reactions as pseudo-first-order in one RO2, so the script explains each imbalance by one of those causes and fails if anything is
left unexplained.

Run: PYTHONPATH=<mech-wt> <mech-wt>/.venv/bin/python build_mcm_inorganic_sulfur.py
Writes mcm_inorganic_sulfur.eqn and results/mcm_inorganic_sulfur_build.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))

from mechlib import atom_imbalance, extract_closure, write_eqn  # noqa: E402
from tomas_jax.gaschem.config import LIMONENE_EQN  # noqa: E402
from tomas_jax.gaschem.kpp_parser import parse_eqn  # noqa: E402
from tomas_jax.gaschem.mechanism import RateModel  # noqa: E402

SEEDS = {"O3", "NO", "NO2", "CO", "CH4", "SO2", "H2O2", "HONO", "HNO3", "N2O5", "H2", "H2O"}
OUT = HERE / "mcm_inorganic_sulfur.eqn"
HEADER = [
    "MCM v3.3.1 inorganic + methane + sulfur subset. PROTOTYPING ONLY.",
    "Derived from MCM v3.3.1 (http://mcm.york.ac.uk) via the limonene subset Shantanu Jathar",
    "sent (tomas_jax/gaschem/mechanisms/limonene_mcm.eqn, SCENTS 04272022 export), not from",
    "a fresh MCM export. Extracted by prototypes/mechanism/build_mcm_inorganic_sulfur.py as",
    "the closure of {O3, NO, NO2, CO, CH4, SO2, H2O2, HONO, HNO3, N2O5, H2, H2O}: every reaction",
    "whose reactants are all reachable from these seeds. Labels, rate expressions and J()",
    "references are verbatim. <33> HNO3 = NA and <34> N2O5 = NA + NA are MCM's fixed",
    "first-order aerosol sinks; the box comparison sets them to zero (no aerosol).",
]


def explain(parsed, r, imb: dict[str, float]) -> str:
    """Name the reason for an atom imbalance (or '' when it cannot be explained)."""
    if not imb:
        return "balanced"
    if any(not parsed.atoms.get(p, {}) for p, _ in r.products):
        return "product declared IGNORE (MCM aerosol sink NA or SA)"
    if "RO2" in r.rate_expr:
        return "RO2 pseudo-first-order (the partner peroxy radical is not written)"
    rest = dict(imb)
    reasons = []
    if rest.get("C", 0) < 0:                       # CO2 omitted
        n = -rest.pop("C")
        rest["O"] = rest.get("O", 0) + 2 * n
        reasons.append(f"{n:g} CO2 omitted")
    dh = -rest.pop("H", 0.0)                        # H2O omitted (or consumed if < 0)
    n_h2o = dh / 2
    rest["O"] = rest.get("O", 0) + n_h2o
    if n_h2o:
        reasons.append(f"{n_h2o:+g} H2O {'omitted' if n_h2o > 0 else 'consumed (in k)'}")
    do = -rest.pop("O", 0.0)
    n_o2 = do / 2
    if abs(n_o2) > 1e-9:
        reasons.append(f"{n_o2:+g} O2 {'omitted' if n_o2 > 0 else 'consumed (in k)'}")
    if any(abs(v) > 1e-9 for v in rest.values()):
        return ""
    return ", ".join(reasons)


full = parse_eqn(LIMONENE_EQN)
seeds = SEEDS - {"H2O"}                             # not a state species in MCM
missing = sorted(s for s in seeds if s not in full.atoms)
if missing:
    raise SystemExit(f"seed species not in limonene_mcm.eqn: {missing}")
sub = extract_closure(full, seeds)
write_eqn(sub, OUT, HEADER)
back = parse_eqn(OUT)
assert [r.label for r in back.reactions] == [r.label for r in sub.reactions]
assert [r.rate_expr for r in back.reactions] == [r.rate_expr for r in sub.reactions]
assert back.species == sub.species and back.ro2 == sub.ro2

rm = RateModel(back)                                # translates every rate expression
balance = []
for r in back.reactions:
    imb = atom_imbalance(back, r)
    why = explain(back, r, imb)
    eq = (f"<{r.label}> {' + '.join(r.reactants)}{' + hv' if r.photolysis else ''} = "
          f"{' + '.join(p if c == 1 else f'{c:g}{p}' for p, c in r.products) or 'PROD'}")
    balance.append({"reaction": eq, "imbalance": imb, "explained_by": why})
unexplained = [b for b in balance if not b["explained_by"]]
if unexplained:
    raise SystemExit(f"unexplained atom imbalance: {unexplained}")

lim_species = {s for s in full.species if s not in back.species}
summary = {
    "source": str(LIMONENE_EQN.name),
    "seeds": sorted(SEEDS),
    "species": len(back.species),
    "reactions": len(back.reactions),
    "photolysis_reactions": sum(r.photolysis for r in back.reactions),
    "species_list": back.species,
    "ro2_members": back.ro2,
    "j_names": rm.j_names,
    "sulfur_reactions": [b["reaction"] for b in balance
                         if any("S" in back.atoms.get(s, {}) for s in
                                b["reaction"].split("=")[0].replace("<", " ").split())],
    "criegee_so2_reactions_not_reachable": [
        f"<{r.label}> {' + '.join(r.reactants)}" for r in full.reactions
        if "SO2" in r.reactants and r.label not in {x.label for x in back.reactions}],
    "atom_balance": {
        "balanced": sum(b["explained_by"] == "balanced" for b in balance),
        "explained_by_omitted_O2_H2O_CO2_or_IGNORE": sum(
            b["explained_by"] not in ("balanced", "") for b in balance),
        "unexplained": len(unexplained),
        "per_reaction": balance,
    },
    "limonene_file_species_excluded": len(lim_species),
}
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "mcm_inorganic_sulfur_build.json").write_text(json.dumps(summary, indent=1))
print(json.dumps({k: v for k, v in summary.items() if k not in ("atom_balance", "species_list")},
                 indent=1))
print("atom balance:", {k: v for k, v in summary["atom_balance"].items() if k != "per_reaction"})
for b in balance:
    if b["explained_by"] != "balanced":
        print(f"  {b['reaction']:45s} {b['explained_by']}")
