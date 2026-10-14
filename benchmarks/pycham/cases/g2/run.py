"""Case G2 runner: write PyCHAM inputs, run PyCHAM and TOMAS-JAX, save small results.

    cd tomas-jax && .venv/bin/python -m benchmarks.pycham.cases.g2.run [--skip-pycham] [--skip-ours]

Outputs (``cases/g2/results``): ours_dt300.npz, ours_dt60.npz, pycham.npz (all species
on the 300 s grid, float32), the attribution runs ours_dt60_tight*.npz and
pycham_tight.npz (900 s grid), key_species_60s.npz (selected species at full
resolution), level2_top_contributors.json, rate_constant_check.json, run_info.json,
pycham_model_var*.txt, j_prescribed.csv. Raw PyCHAM output stays in
``$PYCHAM_BENCH/outputs/g2*``. Every model run is bounded by a timeout (config).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tomas_jax.box.driver import _chemistry, code_version, run
from tomas_jax.box.coupling import N2O5_HET_LABEL
from tomas_jax.gaschem.environment import air_environment
from tomas_jax.gaschem.mechanism import EnvRates

from ... import jvalues, ours, pycham_inputs
from ...config import (ABUNDANCE_THRESHOLD_PPB, COMPARE_DT_S, G2, MCM_J_NUMBER, OURS_TIMEOUT_S,
                       PYCHAM_PYTHON, PYCHAM_SHA, PYCHAM_TIMEOUT_S, PYCHAM_UPDATE_STEP_S,
                       UMANSYSPROP_SHA, BENCH_DIR)
from ...load_pycham import load_pycham
from . import case

RATE_CHECK_TIMEOUT_S = 300.0
PAN_TAG = "PAN"


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist()
                               if isinstance(o, np.ndarray) else float(o)) + "\n")


def make_inputs() -> dict:
    """PyCHAM inputs for G2 in $PYCHAM_BENCH/inputs/g2; returns their paths."""
    s = case.scenario()
    in_dir = pycham_inputs.case_input_dir(case.CASE)
    t0, j_names, j = jvalues.j_per_interval(s)
    rows = jvalues.to_pycham_rows(j_names, j)
    _, xlsx = jvalues.write_j_files(in_dir, t0, rows)
    with open(case.RESULTS_DIR / "j_prescribed.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["j_name", "mcm_number", "j_first_interval_per_s", "j_min_per_s", "j_max_per_s"])
        for col, name in enumerate(j_names):
            w.writerow([name, MCM_J_NUMBER[name], f"{j[0, col]:.6e}", f"{j[:, col].min():.6e}",
                        f"{j[:, col].max():.6e}"])
    eqn = pycham_inputs.write_eqn(in_dir / "limonene_mcm_g2.eqn", G2["eqn_rate_overrides"])
    xml = pycham_inputs.write_xml(in_dir / "limonene_mcm.xml")
    constants = pycham_inputs.write_constants(in_dir / "constants_mcm.f90")
    mv = pycham_inputs.scenario_model_var(
        s, res_file=pycham_inputs.case_output_dir(case.CASE), eqn=eqn, xml=xml,
        constants=constants, j_xlsx=xlsx)
    mv_path = pycham_inputs.write_model_var(in_dir / "g2_model_var.txt", mv)
    shutil.copyfile(mv_path, case.RESULTS_DIR / "pycham_model_var.txt")
    atol, rtol = G2["pycham_tight_int_tol"]
    mv_tight = dict(mv, res_file_name=str(pycham_inputs.case_output_dir(case.CASE, "tight")),
                    int_tol=f"{atol:g}, {rtol:g}")
    mv_tight_path = pycham_inputs.write_model_var(in_dir / "g2_tight_model_var.txt", mv_tight)
    shutil.copyfile(mv_tight_path, case.RESULTS_DIR / "pycham_model_var_tight.txt")
    return {"model_var": mv_path, "model_var_tight": mv_tight_path, "xlsx": xlsx, "eqn": eqn,
            "in_dir": in_dir}


def run_pycham_case(model_var: Path, tag: str) -> dict:
    """One PyCHAM run in the case's own checkout copy (runs are sequential)."""
    root = pycham_inputs.ensure_pycham_root(case.CASE)
    log = model_var.parent / f"pycham_{tag}.log"
    print(f"PyCHAM: {model_var} (root {root}, timeout {PYCHAM_TIMEOUT_S:.0f} s)", flush=True)
    wall = pycham_inputs.run_pycham(model_var, root, log)
    print(f"PyCHAM done in {wall:.1f} s", flush=True)
    return {"root": root, "wall_s": wall, "log": log}


def save_pycham(out_dir: Path, npz: Path, every_s: float = COMPARE_DT_S) -> dict:
    r = load_pycham(str(out_dir))
    keep = ours.on_grid(r["t_s"], every_s)
    np.savez_compressed(npz, t_s=r["t_s"][keep], species=np.array(r["names"]),
                        gas_ppb=r["gas_ppb"][keep].astype(np.float32),
                        ppb_to_molec_cm3=r["ppb_to_molec_cm3"][keep],
                        env=r["env"][keep])
    return r


def check_ro2_pool(out_dir: Path) -> dict:
    """PyCHAM's RO2 pool (from the .eqn inline RO2 sum) vs the TOMAS-JAX parser's."""
    names = [str(n) for n in np.load(out_dir / "comp_namelist.npy", allow_pickle=True)]
    idx = np.load(out_dir / "organic_peroxy_radical_pool_index.npy", allow_pickle=True)
    pool = {names[i] for i in np.asarray(idx, dtype=int).ravel()}
    ref = set(_chemistry()[0].ro2)
    return {"n_pycham": len(pool), "n_ours": len(ref), "identical": pool == ref,
            "only_pycham": sorted(pool - ref), "only_ours": sorted(ref - pool)}


def particle_relevant_species() -> list[dict]:
    """Top particle-phase contributors of the unmodified Level 2 preset (aerosol on)."""
    deadline = time.monotonic() + OURS_TIMEOUT_S
    res = run(case.full_level2_scenario(), cancel=lambda: time.monotonic() > deadline)
    top = res.top_contributors
    _json(case.TOP_CONTRIBUTORS_JSON, {
        "scenario": "level2_limonene_soa (unmodified preset, aerosol on, 6 h)",
        "soa_ugm3_end": res.totals["SOA_ugm3"][-1],
        "top_contributors": top})
    return top


def check_rates(pycham_root: Path, xlsx: Path, ref_run: dict) -> dict:
    """All rate coefficients, PyCHAM (generated rate_coeffs.py) vs TOMAS-JAX, at t = 1 h."""
    s = case.scenario()
    env = s.environment
    air = air_environment(env.temp_k, env.pres_pa, env.rh)
    parsed, rate_model, mech = _chemistry()
    k_th, ro2_dep, j_coef = rate_model.at_env(air)
    rates = EnvRates(jnp.asarray(k_th), jnp.asarray(ro2_dep), jnp.asarray(j_coef))
    i = int(np.argmin(np.abs(ref_run["t_s"] - 3600.0)))
    y = ref_run["gas_ppb"][i] * 1e-9 * air["M"]
    ro2 = float(mech.ro2(jnp.asarray(y)))
    j = ref_run["j"][i]
    k_ours = np.array(rates.k(jnp.asarray(ro2), jnp.asarray(j)))
    k_ours[rate_model.labels.index(N2O5_HET_LABEL)] = 0.0          # k_het = 0 (gamma = 0)
    idx = {n: k for k, n in enumerate(parsed.species)}
    h2o_pycham = ours.h2o_hyland_wexler_cm3(env.temp_k, env.rh)   # PyCHAM's own water
    out = {"t_h": float(ref_run["t_s"][i] / 3600.0), "RO2_cm3": ro2,
           "H2O_ours_buck_cm3": air["H2O"], "H2O_pycham_hyland_wexler_cm3": float(h2o_pycham)}
    for tag, h2o in (("same_h2o", air["H2O"]), ("pycham_h2o", float(h2o_pycham))):
        envd = {"TEMP": air["TEMP"], "M": air["M"], "N2": air["N2"], "O2": air["O2"], "H2O": h2o,
                "RO2": ro2, "RH": env.rh, "NO": float(y[idx["NO"]]), "HO2": float(y[idx["HO2"]]),
                "NO3": float(y[idx["NO3"]]), "sumt": float(ref_run["t_s"][i] - 1.0)}
        with tempfile.TemporaryDirectory() as tmp:
            env_path, out_path = Path(tmp) / "env.json", Path(tmp) / "k.npy"
            env_path.write_text(json.dumps(envd))
            proc = subprocess.run([str(PYCHAM_PYTHON), "-I", str(BENCH_DIR / "check_rates_pycham.py"),
                                   str(pycham_root), str(xlsx), str(env_path), str(out_path)],
                                  capture_output=True, text=True, timeout=RATE_CHECK_TIMEOUT_S)
            if proc.returncode != 0:
                raise RuntimeError(f"rate check failed: {proc.stderr[-2000:]}")
            k_py = np.load(out_path)
        if k_py.shape != k_ours.shape:
            raise RuntimeError(f"reaction count differs: PyCHAM {k_py.shape} vs ours {k_ours.shape}")
        rel = np.abs(k_py - k_ours) / np.maximum(np.abs(k_ours), 1e-300)
        rel[(k_py == 0) & (k_ours == 0)] = 0.0
        worst = np.argsort(-rel)[:10]
        out[tag] = {"n_reactions": int(k_ours.size), "max_rel_diff": float(rel.max()),
                    "n_rel_diff_gt_1e-10": int(np.sum(rel > 1e-10)),
                    "n_rel_diff_gt_1e-6": int(np.sum(rel > 1e-6)),
                    "worst": [{"label": rate_model.labels[w], "rel_diff": float(rel[w]),
                               "k_ours": float(k_ours[w]), "k_pycham": float(k_py[w]),
                               "rate_expr": parsed.reactions[w].rate_expr} for w in worst]}
    _json(case.RATES_CHECK_JSON, out)
    return out


def key_species(top: list[dict], pycham: dict | None) -> list[str]:
    names = list(G2["key_species"]) + [c["species"] for c in top]
    all_species = _chemistry()[0].species
    names += [n for n in all_species if PAN_TAG in n]
    names.append("SA")
    if pycham is not None:
        names = [n for n in names if n in pycham["names"]]
    return list(dict.fromkeys(names))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-pycham", action="store_true", help="reuse the existing PyCHAM output")
    ap.add_argument("--skip-ours", action="store_true", help="reuse existing TOMAS-JAX results")
    args = ap.parse_args(argv)
    case.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    info_path = case.RUN_INFO_JSON
    info = json.loads(info_path.read_text()) if info_path.exists() else {}

    inputs = make_inputs()
    out_dir = pycham_inputs.case_output_dir(case.CASE)
    out_tight = pycham_inputs.case_output_dir(case.CASE, "tight")
    if not args.skip_pycham:
        pr = run_pycham_case(inputs["model_var"], "g2")
        info["pycham"] = {"wall_s": pr["wall_s"], "sha": PYCHAM_SHA, "umansysprop_sha": UMANSYSPROP_SHA,
                          "update_step_s": PYCHAM_UPDATE_STEP_S, "root": str(pr["root"]),
                          "output": str(out_dir), "log": str(pr["log"])}
        pt = run_pycham_case(inputs["model_var_tight"], "g2_tight")
        info["pycham_tight"] = {"wall_s": pt["wall_s"], "int_tol": G2["pycham_tight_int_tol"],
                                "output": str(out_tight)}
    pycham = save_pycham(out_dir, case.PYCHAM_NPZ)
    info["ro2_pool"] = check_ro2_pool(out_dir)
    if not info["ro2_pool"]["identical"]:
        raise RuntimeError(f"RO2 pools differ: {info['ro2_pool']}")
    save_pycham(out_tight, case.PYCHAM_TIGHT_NPZ, G2["attribution_dt_s"])

    top = particle_relevant_species()
    keys = key_species(top, pycham)
    key_arrays = {"species": np.array(keys)}
    ip = [pycham["names"].index(n) for n in keys]
    key_arrays["pycham_t_s"] = pycham["t_s"]
    key_arrays["pycham_ppb"] = pycham["gas_ppb"][:, ip].astype(np.float32)
    pyt = load_pycham(str(out_tight))
    key_arrays["pycham_tight_t_s"] = pyt["t_s"]
    key_arrays["pycham_tight_ppb"] = pyt["gas_ppb"][:, [pyt["names"].index(n) for n in keys]
                                                    ].astype(np.float32)

    runs = {}
    if not args.skip_ours:
        for dt in (G2["coupling_dt_s"], G2["coupling_dt_sensitivity_s"]):
            print(f"TOMAS-JAX: coupling {dt:.0f} s", flush=True)
            runs[int(dt)] = ours.run_ours(case.scenario(coupling_dt_s=dt))
            ours.save_npz(case.OURS_NPZ[int(dt)], runs[int(dt)], every_s=COMPARE_DT_S)
            info[f"ours_dt{int(dt)}"] = {
                "wall_s": float(runs[int(dt)]["runtime_s"]),
                "gas_solver_steps": int(runs[int(dt)]["gas_solver_steps"]),
                "code_version": code_version()}
        dt_s = G2["coupling_dt_sensitivity_s"]
        for tag, water in (("dt60_tight", False), ("dt60_tight_pychamH2O", True)):
            print(f"TOMAS-JAX attribution run {tag}", flush=True)
            with ours.numerics_override(**G2["ours_tight_tol"], pycham_water=water):
                rd = ours.run_ours(case.scenario(coupling_dt_s=dt_s))
            ours.save_npz(case.OURS_ATTR_NPZ[tag], rd, every_s=G2["attribution_dt_s"])
            info[f"ours_{tag}"] = {"wall_s": float(rd["runtime_s"]),
                                   "gas_solver_steps": int(rd["gas_solver_steps"]),
                                   "solver_tol": G2["ours_tight_tol"], "pycham_water": water}
        print("TOMAS-JAX: coupling 300 s, seed number ~0 (aerosol-independence check)", flush=True)
        noseed = ours.run_ours(case.scenario(seed_number_cm3=1e-6))
        a, b = runs[300]["gas_ppb"], noseed["gas_ppb"]
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-30)
        rel[(a < ABUNDANCE_THRESHOLD_PPB) & (b < ABUNDANCE_THRESHOLD_PPB)] = 0.0
        info["aerosol_independence"] = {
            "description": "gas phase with the 2000 cm-3 Level 2 seed vs a 1e-6 cm-3 seed, "
                           "all aerosol processes off, coupling 300 s",
            "max_rel_diff_above_threshold": float(rel.max()),
            "max_abs_diff_ppb": float(np.abs(a - b).max())}
        for dt, rd in runs.items():
            io = [ours.species_index(rd)[n] for n in keys]
            key_arrays[f"ours_dt{dt}_t_s"] = rd["t_s"]
            key_arrays[f"ours_dt{dt}_ppb"] = rd["gas_ppb"][:, io].astype(np.float32)
            key_arrays[f"ours_dt{dt}_sa_cum_ppb"] = (rd["sa_cum_cm3"] / rd["m_air_cm3"] * 1e9
                                                     ).astype(np.float32)
        np.savez_compressed(case.KEY_NPZ, **key_arrays)
        rate_check = check_rates(Path(info["pycham"]["root"]), inputs["xlsx"], runs[300])
        info["rate_constant_check"] = {k: v["max_rel_diff"] for k, v in rate_check.items()
                                       if isinstance(v, dict)}
        info["m_air_cm3"] = {"ours": float(runs[300]["m_air_cm3"]),
                             "pycham": float(pycham["ppb_to_molec_cm3"][0] * 1e9)}
    info["threshold_ppb"] = ABUNDANCE_THRESHOLD_PPB
    _json(info_path, info)
    print(json.dumps({k: v for k, v in info.items() if k != "pycham"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
