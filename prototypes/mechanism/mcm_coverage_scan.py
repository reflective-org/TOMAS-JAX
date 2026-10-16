"""Map the 36 SANDBOX species onto the MCM v3.3.1 limonene subset (limonene_mcm.eqn).

Planning prototype for docs/pro/mechanism/mcm_coverage.md. Read-only: it parses the
mechanism file with the package's own pure-Python parser and prints counts and the
carbon-free ("inorganic") reactions of every SANDBOX species. No JAX needed.

Run (from the worktree root):
    python prototypes/mechanism/mcm_coverage_scan.py [--json out.json]
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
EQN = ROOT / "tomas_jax" / "gaschem" / "mechanisms" / "limonene_mcm.eqn"


def _load_parser():
    """Import kpp_parser without importing tomas_jax/__init__ (which pulls in JAX)."""
    gas = ROOT / "tomas_jax" / "gaschem"
    pkg = "gaschem_ro"
    spec = importlib.util.spec_from_file_location(pkg, gas / "__init__.py",
                                                  submodule_search_locations=[str(gas)])
    mod = importlib.util.module_from_spec(spec)
    sys.modules[pkg] = mod
    for sub in ("config", "kpp_parser"):
        s = importlib.util.spec_from_file_location(f"{pkg}.{sub}", gas / f"{sub}.py")
        m = importlib.util.module_from_spec(s)
        sys.modules[f"{pkg}.{sub}"] = m
        s.loader.exec_module(m)
    return sys.modules[f"{pkg}.kpp_parser"]


# SANDBOX name -> MCM v3.3.1 name (None: MCM has no such species).
# "aux": MCM treats it as an auxiliary variable inside rate expressions, not a species.
SANDBOX_TO_MCM = {
    "Cl": "CL", "ClO": None, "ClOOCl": None, "ClONO2": None, "HCl": "HCL", "Cl2": None,
    "NO": "NO", "NO2": "NO2", "O": "O", "O2": "aux:O2", "O3": "O3", "CH3": None,
    "CH4": "CH4", "HNO3": "HNO3", "H2O": "aux:H2O", "HOCl": None, "N2O5": "N2O5",
    "NO3": "NO3", "OH": "OH", "HO2": "HO2", "O1D": "O1D", "HONO": "HONO",
    "HNO4": "HO2NO2", "OClO": None, "Br": None, "BrO": None, "BrONO2": None, "BrCl": None,
    "HBr": None, "HOBr": None, "HNO3aq": "NA", "C2H6": "C2H6", "SO2": "SO2",
    "H2O2": "H2O2", "SO3": "SO3", "H2SO4": "SA",
}


def fmt(r) -> str:
    lhs = " + ".join(r.reactants) + (" + hv" if r.photolysis else "")
    rhs = " + ".join((f"{c:g} {p}" if c != 1 else p) for p, c in r.products) or "(nothing)"
    return f"<{r.label}> {lhs} = {rhs} : {r.rate_expr}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=Path, default=None)
    args = ap.parse_args()

    kp = _load_parser()
    mech = kp.parse_eqn(EQN)
    used = set(mech.species)

    def carbon(name: str) -> bool:
        return mech.atoms.get(name, {}).get("C", 0) > 0

    def is_inorganic(r) -> bool:
        names = [*r.reactants, *(p for p, _ in r.products)]
        return not any(carbon(n) for n in names)

    inorganic = [r for r in mech.reactions if is_inorganic(r)]
    report = {"n_species_used": len(mech.species), "n_declared": len(mech.atoms),
              "n_reactions": len(mech.reactions), "n_ro2": len(mech.ro2),
              "n_inorganic": len(inorganic),
              "inorganic": [fmt(r) for r in inorganic], "species": {}}

    print(f"{EQN.name}: {len(mech.atoms)} declared, {len(mech.species)} used, "
          f"{len(mech.reactions)} reactions, RO2 sum {len(mech.ro2)} members, "
          f"{len(inorganic)} carbon-free reactions")
    for sb, mcm in SANDBOX_TO_MCM.items():
        entry = {"mcm": mcm, "in_file": False, "as_reactant": 0, "as_product": 0,
                 "in_rate_expr": 0, "inorganic": [], "organic_examples": []}
        if mcm is not None and mcm.startswith("aux:"):
            tok = mcm[4:]
            pat = re.compile(rf"(?<![A-Za-z0-9_]){tok}(?![A-Za-z0-9_])")
            entry["in_rate_expr"] = sum(bool(pat.search(r.rate_expr)) for r in mech.reactions)
        elif mcm is not None and mcm in used:
            entry["in_file"] = True
            for r in mech.reactions:
                prods = [p for p, _ in r.products]
                hit_r, hit_p = mcm in r.reactants, mcm in prods
                if not (hit_r or hit_p):
                    continue
                entry["as_reactant"] += hit_r
                entry["as_product"] += hit_p
                if is_inorganic(r):
                    entry["inorganic"].append(fmt(r))
                elif len(entry["organic_examples"]) < 3:
                    entry["organic_examples"].append(fmt(r))
        report["species"][sb] = entry
        print(f"\n{sb:7s} -> {mcm}: in file {entry['in_file']}, reactant in "
              f"{entry['as_reactant']}, product in {entry['as_product']}, "
              f"rate-expr uses {entry['in_rate_expr']}, carbon-free {len(entry['inorganic'])}")
        for line in entry["inorganic"]:
            print("    " + line)

    # MCM primary VOCs (browse page names) already present in the subset.
    primaries = ["CH4", "C2H6", "C3H8", "CO", "HCHO", "CH3CHO", "C2H5CHO", "CH3OH", "C2H5OH",
                 "CH3COCH3", "MEK", "HCOOH", "CH3CO2H", "MACR", "MVK", "C5H8", "APINENE",
                 "BPINENE", "LIMONENE", "TOLUENE", "BENZENE", "C2H4", "C3H6", "DMS", "CL",
                 "HCL", "C2H2", "GLYOX", "MGLYOX", "BIACET", "IPRCHO", "C3H7CHO", "ACR"]
    present = {p: (p in used) for p in primaries}
    report["primaries_present"] = present
    print("\nMCM primary / small species present in the subset:")
    print("  present: " + ", ".join(p for p, v in present.items() if v))
    print("  absent:  " + ", ".join(p for p, v in present.items() if not v))

    # Reactions that need each generic / complex rate token (for merge planning).
    tokens = {}
    for r in mech.reactions:
        for tok in ("KMT01", "KMT02", "KMT03", "KMT04", "KMT05", "KMT06", "KMT07", "KMT08",
                    "KMT09", "KMT10", "KMT11", "KMT12", "KMT13", "KMT14", "KMT15", "KMT16",
                    "KMT17", "KMT18", "KFPAN", "KBPAN", "KBPPN", "KRO2NO", "KRO2HO2",
                    "KRO2NO3", "KDEC", "KCH3O2", "KAPNO", "KAPHO2", "KNO3AL", "KROPRIM",
                    "KROSEC", "K14ISOM1", "K298CH3O2", "RO2"):
            if tok in r.rate_expr:
                tokens[tok] = tokens.get(tok, 0) + 1
    report["rate_tokens"] = tokens
    print("\nrate-token use counts:", json.dumps(tokens))

    # C1 (methane-chain) reactions: every reactant and product has at most one carbon.
    def c1(r) -> bool:
        names = [*r.reactants, *(p for p, _ in r.products)]
        return (any(carbon(n) for n in names)
                and all(mech.atoms.get(n, {}).get("C", 0) <= 1 for n in names))
    c1r = [r for r in mech.reactions if c1(r)]
    report["c1_reactions"] = [fmt(r) for r in c1r]
    print(f"\nC1-only reactions (methane chain + CO-free C1 steps): {len(c1r)}")
    for r in c1r:
        print("    " + fmt(r))

    if args.json:
        args.json.write_text(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
