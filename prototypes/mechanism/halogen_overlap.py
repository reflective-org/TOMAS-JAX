"""Overlap of SANDBOX (stratchem-jax, JPL 19-5) with the MCM file in tomas-jax (planning only).

For every SANDBOX gas-phase thermal reaction, find the MCM reactions with the same reactants
(after renaming to MCM names and moving bath species into the rate) and compare the total rate
coefficient at three conditions. Photolysis and heterogeneous reactions are listed, not compared.

The two codes live in different environments, so the script runs in three steps:

  1. SANDBOX side (SANDBOX venv):
     .../SANDBOX/.venv/bin/python -I halogen_overlap.py sandbox OUT_sandbox.json
  2. MCM side (any venv that imports tomas_jax; run from the tomas-jax worktree root):
     python halogen_overlap.py mcm OUT_mcm.json
  3. Join (any python):
     python halogen_overlap.py join OUT_sandbox.json OUT_mcm.json OUT.md

Nothing in either package is changed. Rate coefficients are pseudo-first-order in the bath
species: k_eff = k * [O2]^a [H2O]^b [N2]^c, so the two sides are comparable whatever each
code writes into the stoichiometry.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SANDBOX_DIR = "/Users/ali/Documents/GitHub/gas-phase-chemistry/SANDBOX/stratchem-jax"

# SANDBOX name -> MCM-style name (upper case, as KPP/Fortran would fold case anyway).
TO_MCM = {
    "Cl": "CL", "ClO": "CLO", "ClOOCl": "CL2O2", "ClONO2": "CLONO2", "HCl": "HCL",
    "Cl2": "CL2", "HOCl": "HOCL", "OClO": "OCLO", "Br": "BR", "BrO": "BRO",
    "BrONO2": "BRONO2", "BrCl": "BRCL", "HBr": "HBR", "HOBr": "HOBR",
    "HNO4": "HO2NO2", "H2SO4": "SA", "HNO3aq": "NA", "CH3": "CH3O2",
}
BATH = ("O2", "H2O", "N2", "M")

# Conditions: name, T [K], M [molec cm-3], H2O [molec cm-3]
CONDITIONS = [
    ("surface 298 K", 298.0, 2.46e19, 2.46e17),          # 1013 hPa, H2O 1 %
    ("20 km 217 K", 217.0, 1.84e18, 5.0e-6 * 1.84e18),   # 55 hPa, H2O 5 ppm
    ("25 km 222 K", 222.0, 8.4e17, 5.0e-6 * 8.4e17),     # 25.5 hPa, H2O 5 ppm
]


def _bath(M: float, H2O: float) -> dict[str, float]:
    return {"O2": 0.2095 * M, "N2": 0.7808 * M, "H2O": H2O, "M": 1.0}


def _key(reactants: dict[str, float]) -> str:
    names = []
    for s, n in reactants.items():
        if s in BATH:
            continue
        names += [TO_MCM.get(s, s)] * int(round(n))
    return " + ".join(sorted(names))


# ------------------------------------------------------------------ step 1: SANDBOX side
def sandbox_side(out: str) -> None:
    sys.path.insert(0, SANDBOX_DIR)
    import reactions  # noqa: PLC0415 (SANDBOX module, needs its own venv)
    from mechanism import Env  # noqa: PLC0415

    rows = []
    for i, r in enumerate(reactions.REACTIONS, 1):
        row = {"R": f"R{i}", "klabel": r.klabel, "kind": r.kind, "equation": r.equation,
               "active": r.active, "note": r.note, "key": _key(r.reactants),
               "products": {TO_MCM.get(s, s): n for s, n in r.products.items()}, "k": {}}
        if r.kind == "gas":
            for name, T, M, H2O in CONDITIONS:
                env = Env(T=T, M=M, P=0.0, opt=0, H2O=H2O, sulfur_chain=True)
                k = float(r.coeff_fn(env))
                bath = _bath(M, H2O)
                for s, n in r.reactants.items():
                    if s in BATH:
                        k *= bath[s] ** n
                row["k"][name] = k
        rows.append(row)
    Path(out).write_text(json.dumps(rows, indent=1))


# ------------------------------------------------------------------ step 2: MCM side
def mcm_side(out: str) -> None:
    sys.path.insert(0, str(Path.cwd()))
    from tomas_jax.gaschem.kpp_parser import parse_eqn  # noqa: PLC0415
    from tomas_jax.gaschem.mechanism import RateModel  # noqa: PLC0415

    parsed = parse_eqn("tomas_jax/gaschem/mechanisms/limonene_mcm.eqn")
    model = RateModel(parsed)
    ks = {}
    for name, T, M, H2O in CONDITIONS:
        env = {"TEMP": T, "M": M, "O2": 0.2095 * M, "N2": 0.7808 * M, "H2O": H2O}
        k_thermal, ro2_dep, _ = model.at_env(env)
        ks[name] = k_thermal.tolist()
    rows = []
    for i, r in enumerate(parsed.reactions):
        if r.photolysis or ro2_dep[i]:
            continue
        reac: dict[str, float] = {}
        for s in r.reactants:
            reac[s] = reac.get(s, 0) + 1
        rows.append({"label": r.label, "key": _key(reac), "rate": r.rate_expr,
                     "equation": " + ".join(r.reactants) + " = "
                     + " + ".join(p if c == 1 else f"{c:g}{p}" for p, c in r.products),
                     "k": {name: ks[name][i] for name, *_ in CONDITIONS}})
    Path(out).write_text(json.dumps(rows, indent=1))


# ------------------------------------------------------------------ step 3: join
def join(sandbox_json: str, mcm_json: str, out_md: str) -> None:
    sb = json.loads(Path(sandbox_json).read_text())
    mcm = json.loads(Path(mcm_json).read_text())
    by_key: dict[str, list[dict]] = {}
    for r in mcm:
        by_key.setdefault(r["key"], []).append(r)
    cond = [c[0] for c in CONDITIONS]

    lines = ["# SANDBOX vs MCM overlap (generated by halogen_overlap.py)", "",
             "k_eff in cm3 molec-1 s-1 (bimolecular) or s-1 (pseudo-first-order in bath "
             "species). Ratio = k_SANDBOX / k_MCM, summed over channels with the same reactants.",
             "", "| SANDBOX | reactants (MCM names) | MCM labels | "
             + " | ".join(f"ratio {c}" for c in cond) + " |",
             "|---|---|---|" + "---|" * len(cond)]
    gas_keys_done = set()
    no_match = []
    for r in sb:
        if r["kind"] != "gas" or not r["active"]:
            continue
        key = r["key"]
        if key in gas_keys_done:
            continue
        gas_keys_done.add(key)
        same = [x for x in sb if x["kind"] == "gas" and x["active"] and x["key"] == key]
        hits = by_key.get(key, [])
        if not hits:
            no_match.append(", ".join(f"{x['R']} {x['equation']}" for x in same))
            continue
        ratios = []
        for c in cond:
            ksb = sum(x["k"][c] for x in same)
            km = sum(h["k"][c] for h in hits)
            ratios.append(f"{ksb / km:.3g}" if km else "MCM k = 0")
        lines.append(f"| {', '.join(x['R'] for x in same)} | {key} | "
                     f"{', '.join('<' + h['label'] + '>' for h in hits)} | " + " | ".join(ratios)
                     + " |")
    lines += ["", "## SANDBOX gas reactions with no MCM reaction of the same reactants", ""]
    lines += [f"- {x}" for x in no_match]
    lines += ["", "## SANDBOX photolysis and heterogeneous reactions (not compared)", ""]
    lines += [f"- {x['R']} ({x['kind']}{'' if x['active'] else ', disabled'}): {x['equation']}"
              for x in sb if x["kind"] != "gas"]
    Path(out_md).write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    mode = sys.argv[1]
    if mode == "sandbox":
        sandbox_side(sys.argv[2])
    elif mode == "mcm":
        mcm_side(sys.argv[2])
    elif mode == "join":
        join(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        raise SystemExit(f"unknown mode {mode!r}")
