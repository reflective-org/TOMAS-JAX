"""Export the shared gas-chemistry benchmark problems to neutral JSON files.

Run with the benchmark venv (TOMAS-JAX worktree _work/julia/tj installed) and
TUVX_JAX_ROOT pointing at a tuvx-jax checkout (for the Level 3 TUV-x J table):

    PYTHONPATH=<tuvx-jax> TUVX_JAX_ROOT=<tuvx-jax> python export_problems.py

Writes to ../problems/:
  mech_<m>.json       structure of mechanism m (species, reactions, RO2 list)
  problem_<p>.json    conditions, rate-constant inputs, J tables at 900 s and 60 s
                      coupling, initial state, post-interval rules, key species
  checks_<p>.json     f(y) and J(y) v at a test state (unit tests for a port)
  MANIFEST.json       sizes and sha256 of every file

Mechanisms: lim726 is the tutorial limonene mechanism (MCM v3.3.1 subset, 725 species
plus the NA_HET counter, 2135 reactions; reaction <34> N2O5 -> NA_HET is external).
synN are N copies of its chemistry of species with 4 or more C atoms, renamed per copy,
sharing the inorganic and C1-C3 chemistry (the construction of
prototypes/mechanism/scaling.py, applied to the tutorial mechanism).
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tomas_jax.box.config import AerosolStepConfig
from tomas_jax.box.coupling import (N2O5_HET_LABEL, NA_HET, mass_grid, seed_distribution,
                                    tutorial_mechanism)
from tomas_jax.box.scenario import load_preset
from tomas_jax.gaschem.environment import air_environment
from tomas_jax.gaschem.kpp_parser import ParsedMechanism
from tomas_jax.gaschem.mechanism import Mechanism, RateModel
from tomas_jax.gaschem.photolysis import JProvider, PhotolysisSettings
from tomas_jax.physics.heterogeneous import n2o5_uptake
from tomas_jax.physics.water_equilibrium import equilibrate_water

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import NumpyChem, load_problem  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "problems"
DURATION_S = 86400.0
COUPLINGS_S = (900, 60)
RTOL, ATOL = 1e-3, 1e4                     # tutorial solver tolerances (gaschem/config.py)
COPIES = {"syn1353": 2, "syn4488": 7, "syn5742": 9}
KEY_SHARED = ["O3", "OH", "HO2", "NO", "NO2", "NO3", "N2O5", "HNO3", "H2O2", "HCHO", "PAN",
              "SA"]
KEY_ORGANIC = ["LIMONENE", "LIMAL", "LIMKET", "LIMONONIC", "LIMAOOH", "LIMANO3"]


def copies(parsed: ParsedMechanism, n: int) -> tuple[ParsedMechanism, np.ndarray]:
    """n copies of the C>=4 chemistry (same construction as scaling.py::copies)."""
    big = {s for s, a in parsed.atoms.items() if a.get("C", 0) >= 4}
    reactions, origin, atoms, ro2 = [], [], dict(parsed.atoms), []
    seen = set()
    for c in range(n):
        ren = (lambda s, c=c: f"{s}_{c}" if (s in big and c > 0) else s)
        for i, r in enumerate(parsed.reactions):
            involves_big = any(s in big for s in (*r.reactants, *(p for p, _ in r.products)))
            if c > 0 and not involves_big:
                continue
            rr = dataclasses.replace(r, label=f"{r.label}_{c}" if c > 0 else r.label,
                                     reactants=tuple(map(ren, r.reactants)),
                                     products=tuple((ren(p), k) for p, k in r.products))
            reactions.append(rr)
            origin.append(i)
        for s in parsed.ro2:
            t = ren(s)
            if t not in seen:
                seen.add(t)
                ro2.append(t)
        for s in big:
            atoms[ren(s)] = parsed.atoms[s]
    species, sp_seen = [], set()
    for r in reactions:
        for s in (*r.reactants, *(p for p, _ in r.products)):
            if s not in sp_seen:
                sp_seen.add(s)
                species.append(s)
    return (ParsedMechanism(species=species, atoms={s: atoms[s] for s in species},
                            reactions=reactions, ro2=ro2), np.asarray(origin))


def mech_json(name: str, parsed: ParsedMechanism, origin: np.ndarray, base: str,
              n_copies: int) -> dict:
    index = {s: i for i, s in enumerate(parsed.species)}
    return {
        "format": "tomas-gas-bench/mechanism/1",
        "name": name,
        "description": ("MCM v3.3.1 limonene subset as used by the AAAR tutorial "
                        "(TOMAS-JAX box.coupling.tutorial_mechanism)" if n_copies == 1 else
                        f"synthetic: {n_copies} copies of the C>=4 chemistry of {base}, "
                        "sharing the inorganic and C1-C3 chemistry (scaling.py construction)"),
        "index_base": 0,
        "n_species": len(parsed.species),
        "n_reactions": len(parsed.reactions),
        "species": list(parsed.species),
        "labels": [r.label for r in parsed.reactions],
        "origin_reaction": origin.tolist(),
        "reactants": [[index[s] for s in r.reactants] for r in parsed.reactions],
        "products": [[[index[p], float(c)] for p, c in r.products] for r in parsed.reactions],
        "ro2": [index[s] for s in parsed.ro2],
        "notes": [
            "reactants: species indices, one entry per occurrence (A + A appears twice); "
            "rate r = k * product of y over the entries; no reaction has more than 2",
            "products: [species index, stoichiometric coefficient] (duplicates merged)",
            "ro2: species whose sum is the RO2 that multiplies RO2-dependent rate constants",
            "origin_reaction: index of the lim726 reaction this one copies",
        ],
    }


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, separators=(",", ":")))


def initial_k_het(preset) -> float:
    """k_het(N2O5) [1/s] of the preset's initial seed (held constant in the benchmark)."""
    cfg = AerosolStepConfig()
    env = preset.environment
    xk = mass_grid(preset.numerics.nbins)
    nk, mk = seed_distribution(preset.seed.modes, preset.seed.organic_mass_fraction, xk,
                               cfg.boxvol_cm3)
    mk = equilibrate_water(jnp.asarray(mk), env.rh, env.temp_k, cfg.water_scheme)
    k_het, _ = n2o5_uptake(jnp.asarray(nk), mk, jnp.asarray(xk), jnp.float64(env.temp_k),
                           jnp.float64(env.pres_pa), jnp.float64(cfg.boxvol_cm3),
                           jnp.float64(preset.gamma_n2o5))
    return float(k_het)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    lim = tutorial_mechanism()
    rate_model = RateModel(lim, external=(N2O5_HET_LABEL,))
    i_het = rate_model.labels.index(N2O5_HET_LABEL)

    # --- mechanisms ---------------------------------------------------------------------
    mechs = {"lim726": (lim, np.arange(len(lim.reactions)), 1)}
    for name, n in COPIES.items():
        parsed, origin = copies(lim, n)
        assert len(parsed.species) == int(name[3:]), (name, len(parsed.species))
        mechs[name] = (parsed, origin, n)
    for name, (parsed, origin, n) in mechs.items():
        write_json(OUT / f"mech_{name}.json", mech_json(name, parsed, origin, "lim726", n))
        # the export must reproduce TOMAS-JAX's own index arrays
        built = Mechanism.build(parsed)
        exported = json.loads((OUT / f"mech_{name}.json").read_text())
        assert exported["n_species"] == built.n_species
        flat = np.full((len(parsed.reactions), 2), built.n_species)
        for j, rs in enumerate(exported["reactants"]):
            flat[j, :len(rs)] = rs
        assert np.array_equal(flat, np.asarray(built.reactant_idx))
        assert exported["ro2"] == np.asarray(built.ro2_idx).tolist()
        print(f"{name}: {len(parsed.species)} species, {len(parsed.reactions)} reactions, "
              f"{len(parsed.ro2)} RO2")

    # --- conditions ---------------------------------------------------------------------
    levels = {}
    for lev, preset_name in (("L2", "level2_limonene_soa"), ("L3", "level3_day_night")):
        pr = load_preset(preset_name)
        env = pr.environment
        air = air_environment(env.temp_k, env.pres_pa, env.rh)
        k_th, ro2_dep, j_coef = rate_model.at_env(air)
        jp = JProvider(rate_model.j_names, PhotolysisSettings(
            pr.photolysis.mode, pr.photolysis.latitude_deg, pr.photolysis.longitude_deg,
            pr.start_utc, pr.photolysis.jno2_per_s, pr.photolysis.constant_sza_deg))
        jtab = {}
        for dt in COUPLINGS_S:
            n_int = int(round(DURATION_S / dt))
            jtab[str(dt)] = [jp.at(i * dt + 0.5 * dt).tolist() for i in range(n_int)]
        k_het = initial_k_het(pr) if (pr.processes.n2o5_uptake and pr.gamma_n2o5 > 0) else 0.0
        levels[lev] = dict(preset=pr, air=air, k_th=k_th, ro2_dep=ro2_dep, j_coef=j_coef,
                           jtab=jtab, k_het=k_het)
        print(f"{lev}: k_het(N2O5) = {k_het:.4g} 1/s, max J_NO2 = "
              f"{max(r[rate_model.j_names.index('J_NO2')] for r in jtab['900']):.4g}")

    # --- problems -------------------------------------------------------------------------
    problems = [("lim726", "L2"), ("lim726", "L3"), ("syn1353", "L3"), ("syn4488", "L3"),
                ("syn5742", "L3")]
    for mname, lev in problems:
        parsed, origin, ncopy = mechs[mname]
        L = levels[lev]
        pr, air = L["preset"], L["air"]
        index = {s: i for i, s in enumerate(parsed.species)}
        big = {s for s, a in lim.atoms.items() if a.get("C", 0) >= 4}
        y0 = {}
        for sp_name, ppb in pr.gases_ppb.items():
            val = ppb * 1e-9 * air["M"]
            if sp_name in big:                     # precursor split equally over the copies
                for c in range(ncopy):
                    y0[index[sp_name if c == 0 else f"{sp_name}_{c}"]] = val / ncopy
            else:
                y0[index[sp_name]] = val
        # map lim726 reaction-level arrays onto this mechanism through origin
        k_th = L["k_th"][origin]
        ro2_dep = L["ro2_dep"][origin]
        jc_full = L["j_coef"][origin]
        jr, jcol = np.nonzero(jc_full)
        j_entries = [[int(r), int(c), float(jc_full[r, c])] for r, c in zip(jr, jcol)]
        ext = [{"reaction": int(j), "label": parsed.reactions[j].label,
                "k": float(L["k_het"])} for j in np.flatnonzero(origin == i_het)]
        held = [index[s] for s in pr.hold_constant]
        key = list(KEY_SHARED)
        for s in KEY_ORGANIC:
            key.append(s)
        if ncopy > 1:
            key += [f"LIMONENE_{ncopy - 1}", f"LIMONONIC_{ncopy - 1}"]
        pname = f"{mname}_{lev}"
        start = pr.start_utc.astimezone(timezone.utc).isoformat()
        prob = {
            "format": "tomas-gas-bench/problem/1",
            "name": pname,
            "mechanism_file": f"mech_{mname}.json",
            "index_base": 0,
            "description": (
                f"Gas chemistry only, {lev[-1]} = conditions of the AAAR tutorial Level "
                f"{lev[-1]} preset ({pr.name}), run for 24 h. Rate constants frozen per "
                "coupling interval (KPP semantics, as TOMAS-JAX)."),
            "conditions": {
                "temp_K": pr.environment.temp_k, "pres_Pa": pr.environment.pres_pa,
                "rh": pr.environment.rh, "M": air["M"], "O2": air["O2"], "N2": air["N2"],
                "H2O": air["H2O"],
                "photolysis": {"mode": pr.photolysis.mode,
                               "latitude_deg": pr.photolysis.latitude_deg,
                               "longitude_deg": pr.photolysis.longitude_deg,
                               "start_utc": start,
                               "jno2_per_s": pr.photolysis.jno2_per_s,
                               "constant_sza_deg": pr.photolysis.constant_sza_deg},
                "n2o5_uptake": {"gamma": pr.gamma_n2o5 if pr.processes.n2o5_uptake else 0.0,
                                "k_het_per_s": L["k_het"],
                                "note": "k_het of the preset's initial seed aerosol, held "
                                        "constant (no aerosol in this benchmark)"},
            },
            "duration_s": DURATION_S,
            "couplings_s": list(COUPLINGS_S),
            "tolerances": {"rtol": RTOL, "atol": ATOL, "atol_units": "molec/cm3",
                           "note": "the tutorial's Kvaerno3 settings; the reference uses "
                                   "much tighter ones"},
            "rates": {
                "rule": "k_i = k_thermal * (RO2(y(t_i)) if ro2_dependent else 1) "
                        "+ J_coef @ J_i; then k_i[external.reaction] = external.k",
                "k_thermal": k_th.tolist(),
                "ro2_dependent": np.flatnonzero(ro2_dep).tolist(),
                "j_names": list(rate_model.j_names),
                "j_coef": j_entries,
                "external": ext,
            },
            "photolysis": L["jtab"],
            "photolysis_note": ("photolysis[str(coupling)][i] = J [1/s] in j_names order at "
                                "the midpoint t_i + dt/2 of interval i"),
            "initial": {
                "units": "molec/cm3",
                "y0_nonzero": [[int(i), float(v)] for i, v in sorted(y0.items())],
                "mixing_ratios_ppb": dict(pr.gases_ppb),
                "note": ("C>=4 initial species (LIMONENE) split equally over the copies, "
                         "so the total precursor matches the tutorial" if ncopy > 1 else ""),
            },
            "post_interval": {
                "rule": "after each interval: clip y < 0 to 0 (this is the scored output), "
                        "then set reset_to_zero to 0 and reset_to_initial to y0",
                "reset_to_zero": [index["SA"], index[NA_HET]],
                "reset_to_initial": held,
                "reset_to_initial_names": list(pr.hold_constant),
            },
            "key_species": key,
            "key_species_index": [index[s] for s in key],
        }
        write_json(OUT / f"problem_{pname}.json", prob)

        # unit checks for a port: f(y) and J(y) v at the start state of interval 0 (900 s)
        P = load_problem(pname, OUT)
        chem = NumpyChem(P.mech)
        rng = np.random.default_rng(7)
        y_test = P.y0 + 10.0 ** rng.uniform(4, 9, P.mech.n)
        k_test = P.k(y_test, P.j_table(900)[0])
        v = rng.standard_normal(P.mech.n) * y_test
        checks = {
            "format": "tomas-gas-bench/checks/1",
            "problem": pname,
            "how": "y_test = y0 + 10**U(4, 9) (numpy default_rng(7)); k_test = rate rule at "
                   "y_test with J of interval 0 at 900 s coupling; v = N(0,1) * y_test",
            "y_test": y_test.tolist(), "v": v.tolist(), "k_test": k_test.tolist(),
            "ro2_test": float(y_test[P.mech.ro2].sum()),
            "f_test": chem.rhs(y_test, k_test).tolist(),
            "jac_v_test": (chem.jac(y_test, k_test) @ v).tolist(),
            "jac_nnz": int(chem.nnz),
        }
        write_json(OUT / f"checks_{pname}.json", checks)
        print(f"{pname}: n={P.mech.n} nr={P.mech.nr} jac_nnz={chem.nnz} "
              f"external={[(e['label'], e['k']) for e in ext]} held={pr.hold_constant}")

    manifest = {"generated_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "files": {p.name: {"bytes": p.stat().st_size, "sha256": sha(p)}
                          for p in sorted(OUT.glob("*.json")) if p.name != "MANIFEST.json"}}
    write_json(OUT / "MANIFEST.json", manifest)
    for f, d in manifest["files"].items():
        print(f"{f:32s} {d['bytes'] / 1e6:8.2f} MB")


if __name__ == "__main__":
    main()
