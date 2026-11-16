"""Cases T1-T3 runner: write PyCHAM inputs (merged mechanism, J), run PyCHAM and TOMAS-JAX,
save small results.

    cd tomas-jax && .venv/bin/python -m benchmarks.pycham.cases.T1.run [--case T1|T2|T3] [--skip-pycham] [--skip-ours]

(``cases/T2/run.py`` and ``cases/T3/run.py`` select their case.) Outputs (``cases/<case>/results``):
ours_dt300.npz, ours_dt60.npz, pycham.npz (all species on the 300 s grid, float32),
key_species.npz (key species at full resolution), run_info.json (wall times, RO2 pool check,
air density). Raw PyCHAM output stays in ``$PYCHAM_BENCH/outputs/<case>``. Every model run is
bounded by a timeout (config).
"""
from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import numpy as np

from tomas_jax.box.coupling import merged_mechanism
from tomas_jax.box.driver import code_version

from ... import jvalues, ours, pycham_inputs
from ...config import (COMPARE_DT_S, MCM_J_NUMBER, PYCHAM_SHA, PYCHAM_UPDATE_STEP_S,
                       UMANSYSPROP_SHA)
from ...load_pycham import load_pycham
from .case import DEFAULT_CASE, AromaticCase


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist()
                               if isinstance(o, np.ndarray) else float(o)) + "\n")


def make_inputs(case: AromaticCase) -> dict:
    """PyCHAM inputs for the case in $PYCHAM_BENCH/inputs/<case>; returns their paths."""
    T1 = case.cfg
    s = case.scenario()
    in_dir = pycham_inputs.case_input_dir(case.CASE)
    t0, j_names, j = jvalues.j_per_interval(s)
    if case.VOC != "BENZENE" and ("J_CH3CHO" not in j_names or "J_MVK_CO" not in j_names):
        raise RuntimeError("the aromatic J names are missing: the run did not select the merged mechanism")
    rows = jvalues.to_pycham_rows(j_names, j)
    _, xlsx = jvalues.write_j_files(in_dir, t0, rows)
    with open(case.RESULTS_DIR / "j_prescribed.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["j_name", "mcm_number", "j_first_interval_per_s", "j_min_per_s", "j_max_per_s"])
        for col, name in enumerate(j_names):
            w.writerow([name, MCM_J_NUMBER[name], f"{j[0, col]:.6e}", f"{j[:, col].min():.6e}",
                        f"{j[:, col].max():.6e}"])
    eqn = pycham_inputs.write_merged_eqn(in_dir / f"limonene_{case.VOC.lower()}_mcm_{case.CASE}.eqn",
                                         T1["eqn_rate_overrides"], aromatics=(case.VOC,))
    xml = pycham_inputs.write_xml(in_dir / f"limonene_{case.VOC.lower()}_mcm.xml")
    constants = pycham_inputs.write_constants(in_dir / "constants_mcm.f90")
    mv = pycham_inputs.scenario_model_var(
        s, res_file=pycham_inputs.case_output_dir(case.CASE), eqn=eqn, xml=xml,
        constants=constants, j_xlsx=xlsx)
    mv_path = pycham_inputs.write_model_var(in_dir / f"{case.CASE}_model_var.txt", mv)
    shutil.copyfile(mv_path, case.RESULTS_DIR / "pycham_model_var.txt")
    return {"model_var": mv_path, "xlsx": xlsx, "eqn": eqn, "in_dir": in_dir}


def run_pycham_case(case: AromaticCase, model_var: Path) -> dict:
    root = pycham_inputs.ensure_pycham_root(case.CASE)
    log = model_var.parent / f"pycham_{case.CASE}.log"
    print(f"PyCHAM: {model_var} (root {root})", flush=True)
    wall = pycham_inputs.run_pycham(model_var, root, log)
    print(f"PyCHAM done in {wall:.1f} s", flush=True)
    return {"root": root, "wall_s": wall, "log": log}


def save_pycham(out_dir: Path, npz: Path, every_s: float = COMPARE_DT_S) -> dict:
    r = load_pycham(str(out_dir))
    keep = ours.on_grid(r["t_s"], every_s)
    np.savez_compressed(npz, t_s=r["t_s"][keep], species=np.array(r["names"]),
                        gas_ppb=r["gas_ppb"][keep].astype(np.float32),
                        ppb_to_molec_cm3=r["ppb_to_molec_cm3"][keep], env=r["env"][keep])
    return r


def check_ro2_pool(case: AromaticCase, out_dir: Path) -> dict:
    """PyCHAM's RO2 pool (from the merged file's inline RO2 sum) vs the in-code merge."""
    names = [str(n) for n in np.load(out_dir / "comp_namelist.npy", allow_pickle=True)]
    idx = np.load(out_dir / "organic_peroxy_radical_pool_index.npy", allow_pickle=True)
    pool = {names[i] for i in np.asarray(idx, dtype=int).ravel()}
    ref = set(merged_mechanism((case.VOC,)).ro2)
    return {"n_pycham": len(pool), "n_ours": len(ref), "identical": pool == ref,
            "only_pycham": sorted(pool - ref), "only_ours": sorted(ref - pool)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--case", default=DEFAULT_CASE, help="T1 (toluene), T2 (benzene) or T3 (m-xylene)")
    ap.add_argument("--skip-pycham", action="store_true", help="reuse the existing PyCHAM output")
    ap.add_argument("--skip-ours", action="store_true", help="reuse existing TOMAS-JAX results")
    args = ap.parse_args(argv)
    case = AromaticCase(args.case)
    T1 = case.cfg
    case.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    info = json.loads(case.RUN_INFO_JSON.read_text()) if case.RUN_INFO_JSON.exists() else {}

    inputs = make_inputs(case)
    out_dir = pycham_inputs.case_output_dir(case.CASE)
    if not args.skip_pycham:
        pr = run_pycham_case(case, inputs["model_var"])
        info["pycham"] = {"wall_s": pr["wall_s"], "sha": PYCHAM_SHA, "umansysprop_sha": UMANSYSPROP_SHA,
                          "update_step_s": PYCHAM_UPDATE_STEP_S, "root": str(pr["root"]),
                          "output": str(out_dir), "log": str(pr["log"])}
    pycham = save_pycham(out_dir, case.PYCHAM_NPZ)
    info["ro2_pool"] = check_ro2_pool(case, out_dir)
    if not info["ro2_pool"]["identical"]:
        raise RuntimeError(f"RO2 pools differ: {info['ro2_pool']}")
    info["n_species_pycham"] = len(pycham["names"])

    keys = [n for n in T1["key_species"] if n in pycham["names"]]
    key_arrays = {"species": np.array(keys), "pycham_t_s": pycham["t_s"],
                  "pycham_ppb": pycham["gas_ppb"][:, [pycham["names"].index(n) for n in keys]].astype(np.float32)}
    if not args.skip_ours:
        for dt in (T1["coupling_dt_s"], T1["coupling_dt_sensitivity_s"]):
            print(f"TOMAS-JAX: coupling {dt:.0f} s", flush=True)
            rd = ours.run_ours(case.scenario(coupling_dt_s=dt))
            n_expected = len(merged_mechanism((case.VOC,)).species) + 1          # + NA_HET
            if len(rd["species"]) != n_expected:
                raise RuntimeError(f"expected the merged mechanism ({n_expected} species), got {len(rd['species'])}")
            ours.save_npz(case.OURS_NPZ[int(dt)], rd, every_s=COMPARE_DT_S)
            info[f"ours_dt{int(dt)}"] = {"wall_s": float(rd["runtime_s"]),
                                         "gas_solver_steps": int(rd["gas_solver_steps"]),
                                         "code_version": code_version()}
            io = [ours.species_index(rd)[n] for n in keys]
            key_arrays[f"ours_dt{int(dt)}_t_s"] = rd["t_s"]
            key_arrays[f"ours_dt{int(dt)}_ppb"] = rd["gas_ppb"][:, io].astype(np.float32)
            info["m_air_cm3"] = {"ours": float(rd["m_air_cm3"]),
                                 "pycham": float(pycham["ppb_to_molec_cm3"][0] * 1e9)}
        np.savez_compressed(case.KEY_NPZ, **key_arrays)
    _json(case.RUN_INFO_JSON, info)
    print(json.dumps({k: v for k, v in info.items() if k != "pycham"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
