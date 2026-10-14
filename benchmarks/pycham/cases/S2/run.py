"""Case S2 runner: write PyCHAM inputs, run PyCHAM and TOMAS-JAX, save small results.

    cd tomas-jax && .venv/bin/python -m benchmarks.pycham.cases.S2.run [--skip-pycham] [--skip-ours]
    cd tomas-jax && .venv/bin/python -m benchmarks.pycham.cases.S2.run --smoke   # 30 min PyCHAM test

PyCHAM runs (sequential, in pycham-bench/PyCHAM_S2, raw output in pycham-bench/outputs/S2_*):
  class    - every condensable species at the c* of its TOMAS-JAX class centre (main run)
  exact    - every condensable species at its own Li et al. (2016) c* (lumping sensitivity)
  nopart   - no particles (gas-phase counterpart, to size the partitioning feedback)
  massfrac - attribution: class-centre c* converted to Pa with one molar mass, which turns
             PyCHAM's mole-fraction Raoult term into a mass-fraction one (TOMAS-JAX form)
  massfrac_nokelvin - attribution: as massfrac in the surfT = 0 copy PyCHAM_S2_nokelvin
  native   - volatility sensitivity: the condensables keep PyCHAM's own vapour pressures
TOMAS-JAX runs: coupling 300 s (preset), 60 s (= PyCHAM update step), 300 s with SOA
partitioning and H2SO4 condensation off, and the attribution runs (60 s, organic
condensation sink scaled by the PyCHAM/TOMAS-JAX transfer-rate ratio, with and without
the Kelvin term; attribution.py, each in a subprocess). Results (``cases/S2/results``): npz on the 300 s grid (float32),
pycham_volatility.csv, setup.json, transfer_rates.json, run_info.json and copies of the
model-variables files. Every model run is bounded by a timeout (config).
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from tomas_jax.box.driver import _chemistry, _class_map, code_version

from ... import jvalues, ours, pycham_inputs
from ...config import (COMPARE_DT_S, EXTRA_XML_SPECIES, OURS_TIMEOUT_S, PYCHAM_SHA,
                       PYCHAM_TIMEOUT_S, PYCHAM_UPDATE_STEP_S, TOMAS_JAX_ROOT,
                       UMANSYSPROP_SHA)
from ...load_pycham import load_pycham
from . import case, inputs, transfer
from .case import S2

SMOKE_HOURS = 0.5
# particle mass of components PyCHAM must not partition (above ppartit_cutoff) [ug/m3]
NON_PARTITIONING_MAX_UGM3 = 1e-6
# PyCHAM runs whose npz keep the gas phase of the condensable species (+ the feedback
# species) only, and no per-bin arrays (file size; the analysis uses nothing else)
GAS_SUBSET_VARIANTS = ("exact", "massfrac", "massfrac_nokelvin", "native")
NO_BIN_VARIANTS = ("massfrac", "massfrac_nokelvin", "native")
# model-variables copies kept in results/ (exact / massfrac* differ from class only in volP,
# which pycham_volatility.csv records per variant; all files stay in pycham-bench/inputs/S2)
MODEL_VAR_COPIES = ("class", "nopart", "native")


def _json(path: Path, obj) -> None:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.integer,)):
            return int(o)
        return float(o)
    path.write_text(json.dumps(obj, indent=2, default=default) + "\n")


def make_inputs(smoke: bool = False) -> dict:
    """All PyCHAM inputs for S2 in $PYCHAM_BENCH/inputs/S2; returns their paths + setup."""
    s = case.scenario()
    in_dir = pycham_inputs.case_input_dir(case.CASE)
    case.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    # photolysis: our J per coupling interval (constant light -> one value per J)
    t0, j_names, j = jvalues.j_per_interval(s)
    if np.ptp(j, axis=0).max() > 0.0:
        raise RuntimeError("S2 expects constant J (Level 2 constant light)")
    _, xlsx = jvalues.write_j_files(in_dir, t0, jvalues.to_pycham_rows(j_names, j))
    # mechanism: <34> rate 0 (gamma = 0 in TOMAS-JAX) + a dead reaction declaring SEEDOA
    eqn = pycham_inputs.write_eqn(in_dir / "limonene_mcm_s2.eqn", S2["eqn_rate_overrides"])
    text = eqn.read_text().rstrip("\n") + "\n" + S2["seed_dead_reaction"] + "\n"
    eqn.write_text(text)
    xml = pycham_inputs.write_xml(in_dir / "limonene_mcm_s2.xml",
                                  {**EXTRA_XML_SPECIES, S2["seed_oa_name"]: S2["seed_oa_smiles"]})
    constants = pycham_inputs.write_constants(in_dir / "constants_mcm.f90")
    mw = inputs.pycham_molar_masses(xml, in_dir / "pycham_mw.json")
    temp = s.environment.temp_k
    rows = {v: inputs.volatility_table(v, mw, temp)
            for v in (S2["pycham_variants"] + S2["pycham_attribution_variants"]
                      + S2["pycham_volatility_variants"])}
    inputs.write_volatility_csv(rows)
    part_keys, seed_record = inputs.particle_keys(s, mw)
    extra = {"recording_time_step": f"{S2['pycham_recording_step_s']:.1f}"}
    if smoke:
        extra["total_model_time"] = f"{SMOKE_HOURS * 3600.0:.1f}"
    paths = {}
    for variant in case.PYCHAM_ALL_VARIANTS:
        tag = f"smoke_{variant}" if smoke else variant
        res = pycham_inputs.case_output_dir(case.CASE, tag)
        if variant == S2["pycham_gas_only_variant"]:
            particles, vol = None, inputs.volatility_keys(rows["class"])
        else:
            particles, vol = part_keys, inputs.volatility_keys(rows[variant])
        mv = pycham_inputs.scenario_model_var(s, res_file=res, eqn=eqn, xml=xml,
                                              constants=constants, j_xlsx=xlsx,
                                              particles=particles, extra={**extra, **vol})
        p = pycham_inputs.write_model_var(in_dir / f"S2_{tag}_model_var.txt", mv)
        if not smoke and variant in MODEL_VAR_COPIES:
            shutil.copyfile(p, case.RESULTS_DIR / f"pycham_model_var_{variant}.txt")
        paths[variant] = {"model_var": p, "output": res}
    cond = [r for r in rows["class"] if r["category"] == "condensable"]
    setup = {
        "seed": seed_record,
        "seed_oa_mw_pycham": mw[S2["seed_oa_name"]],
        "n_pycham_components_overridden": len(rows["class"]),
        "categories": {c: sum(r["category"] == c for r in rows["class"])
                       for c in sorted({r["category"] for r in rows["class"]})},
        "condensable_vp_pa_range": {
            v: [min(r["vp_pa"] for r in rows[v] if r["category"] == "condensable"),
                max(r["vp_pa"] for r in rows[v] if r["category"] == "condensable")]
            for v in rows if v not in S2["pycham_volatility_variants"]},
        "condensable_mw_max_rel_diff_pycham_vs_ours": float(max(
            abs(r["mw_pycham"] / r["mw_ours"] - 1.0) for r in cond)),
        "j_constant_per_s": {n: float(j[0, k]) for k, n in enumerate(j_names)},
        "temperature_k": temp,
    }
    _json(case.SETUP_JSON, setup)
    return {"paths": paths, "xlsx": xlsx, "eqn": eqn, "xml": xml, "mw": mw, "rows": rows,
            "setup": setup}


def run_pycham_variant(model_var: Path, tag: str, kelvin_off: bool = False) -> float:
    root = pycham_inputs.ensure_pycham_root(case.CASE, kelvin_off=kelvin_off)
    log = model_var.parent / f"pycham_{tag}.log"
    print(f"PyCHAM {tag}: {model_var} (root {root}, timeout {PYCHAM_TIMEOUT_S:.0f} s)", flush=True)
    wall = pycham_inputs.run_pycham(model_var, root, log)
    print(f"PyCHAM {tag} done in {wall:.1f} s", flush=True)
    return wall


def save_pycham(out_dir: Path, npz: Path, mw: dict[str, float],
                condensables_only: bool = False, per_bin: bool = True) -> dict:
    """PyCHAM output -> small npz (300 s grid, float32); returns the loaded arrays.

    ``condensables_only`` keeps the gas phase of the 245 condensable species and the
    feedback species only (sensitivity/attribution runs, whose other gases the analysis
    does not use); ``per_bin = False`` stores bin totals instead of per-bin arrays.
    """
    r = load_pycham(str(out_dir))
    names = r["names"]
    keep = ours.on_grid(r["t_s"], COMPARE_DT_S)
    if keep.sum() < 2:
        raise RuntimeError(f"no {COMPARE_DT_S:.0f} s output grid in {out_dir}")
    cols = ([names.index(n) for n in list(_class_map(None).species) + list(S2["feedback_species"])]
            if condensables_only else list(range(len(names))))
    arrays = {"t_s": r["t_s"][keep], "species": np.array(names)[cols],
              "gas_ppb": r["gas_ppb"][keep][:, cols].astype(np.float32),
              "ppb_to_molec_cm3": r["ppb_to_molec_cm3"][keep], "mw_g_mol": r["mw_g_mol"][cols],
              "psat_298_pa": r["psat_298_Pa"][cols]}
    # PyCHAM must use exactly the molar masses the vapour pressures were converted with
    for n, m in mw.items():
        if n in names and abs(r["mw_g_mol"][names.index(n)] / m - 1.0) > 1e-12:
            raise RuntimeError(f"PyCHAM MW of {n} = {r['mw_g_mol'][names.index(n)]} != {m}")
    nb = r["part_ug_m3"].shape[1]
    if nb:
        p = r["part_ug_m3"][keep]                                   # [t, bin, comp]
        iw = names.index("H2O")
        not_dry = [iw, names.index("core")]
        dry = np.delete(p, not_dry, axis=2).sum(axis=2)
        cmap = _class_map(None)
        icond = [names.index(sn) for sn in cmap.species]
        n_bin = r["N_bin"]                       # number of each (moving-centre) bin
        # particle phase per species only for the components that can be there; the
        # rest must be ~0 (they are above ppartit_cutoff) and is checked here
        pnames = list(cmap.species) + ["SA", S2["seed_oa_name"], "H2O"]
        ip = [names.index(n) for n in pnames]
        rest = np.delete(p, ip + [names.index("core")], axis=2).sum(axis=(1, 2))
        if rest.max() > NON_PARTITIONING_MAX_UGM3:
            raise RuntimeError(f"non-partitioning species in PyCHAM particles: {rest.max():g} ug/m3")
        bins = {"n_bin_cm3": n_bin[keep], "rad_wet_um": r["rad_um"][keep],
                "bin_dry_ugm3": dry, "bin_water_ugm3": p[:, :, iw],
                "bin_soa_ugm3": p[:, :, icond].sum(axis=2),
                "bin_sa_ugm3": p[:, :, names.index("SA")],
                "bin_seedoa_ugm3": p[:, :, names.index(S2["seed_oa_name"])]}
        if not per_bin:          # bin totals, kept as one-bin arrays (same analysis code)
            bins = {k: v.sum(axis=1, keepdims=True) for k, v in bins.items()
                    if k not in ("rad_wet_um",)}
        arrays.update({
            "particle_names": np.array(pnames),
            "particle_species_ugm3": p.sum(axis=1)[:, ip].astype(np.float32),  # [t, pnames]
            "other_particle_max_ugm3": np.float64(rest.max()),
            **{k: np.asarray(v, dtype=np.float32) for k, v in bins.items()},
        })
    np.savez_compressed(npz, **arrays)
    return r


SPECIES_KEYED = ("gas_ppb", "mw_g_mol", "psat_298_pa")   # arrays indexed by "species"


def prune_gas_species(paths: list[Path]) -> dict:
    """Drop gas species below S2["store_floor_ppb"] at every time in EVERY run (both models)
    from the stored npz files; condensable and feedback species are always kept."""
    keep = (set(_class_map(None).species) | set(S2["feedback_species"]) | set(S2["key_species"]))
    data = {p: dict(np.load(p)) for p in paths}
    for d in data.values():
        g = d["gas_ppb"].astype(float)
        keep |= {str(s) for s, m in zip(d["species"], g.max(axis=0)) if m >= S2["store_floor_ppb"]}
    dropped = 0
    for p, d in data.items():
        sel = np.array([str(s) in keep for s in d["species"]])
        dropped = max(dropped, int((~sel).sum()))
        d["species"] = d["species"][sel]
        for k in SPECIES_KEYED:
            if k in d:
                d[k] = d[k][..., sel] if d[k].ndim == 2 else d[k][sel]
        np.savez_compressed(p, **d)
    return {"store_floor_ppb": S2["store_floor_ppb"], "n_species_kept": len(keep),
            "max_dropped_per_file": dropped}


def check_pycham_output(r: dict, rows: list[dict]) -> dict:
    """Vapour pressures PyCHAM recorded (Pa at 298.15 K) vs those it was given."""
    names = r["names"]
    rel = []
    for row in rows:
        if not np.isfinite(row["vp_pa"]):         # not overridden (PyCHAM's own value)
            continue
        got = r["psat_298_Pa"][names.index(row["species"])]
        rel.append(abs(got / row["vp_pa"] - 1.0))
    out = {"max_rel_diff_psat_vs_volP": float(max(rel))}
    if out["max_rel_diff_psat_vs_volP"] > 1e-10:
        raise RuntimeError(f"PyCHAM did not take the vapour pressures: {out}")
    return out


def check_ro2_pool(out_dir: Path) -> dict:
    names = [str(n) for n in np.load(out_dir / "comp_namelist.npy", allow_pickle=True)]
    idx = np.load(out_dir / "organic_peroxy_radical_pool_index.npy", allow_pickle=True)
    pool = {names[i] for i in np.asarray(idx, dtype=int).ravel()}
    ref = set(_chemistry()[0].ro2)
    out = {"n_pycham": len(pool), "n_ours": len(ref), "identical": pool == ref}
    if not out["identical"]:
        raise RuntimeError(f"RO2 pools differ: {sorted(pool ^ ref)[:20]}")
    return out


def initial_particle_check(r: dict, o: dict) -> dict:
    """t = 0 particle state of PyCHAM vs TOMAS-JAX (number, dry mass, water)."""
    names = r["names"]
    p0 = r["part_ug_m3"][0]
    iw = names.index("H2O")
    dry0 = np.delete(p0, [iw, names.index("core")], axis=1).sum()
    comp = list(map(str, o["component_names"]))
    pc = o["particle_component_ugm3"][0]
    ours_dry = float(pc[[comp.index("SO4"), comp.index("POA")]].sum())
    return {"N_cm3": {"pycham": float(r["N_bin"][0].sum()),
                      "ours": float(o["result"].totals["N_cm3"][0])},
            "dry_ugm3": {"pycham": float(dry0), "ours": ours_dry},
            "SA_SO4_ugm3": {"pycham": float(p0[:, names.index("SA")].sum()),
                            "ours": float(pc[comp.index("SO4")])},
            "seed_OA_ugm3": {"pycham": float(p0[:, names.index(S2["seed_oa_name"])].sum()),
                             "ours": float(pc[comp.index("POA")])},
            "water_ugm3": {"pycham": float(p0[:, iw].sum()), "ours": float(pc[comp.index("H2O")])}}


def transfer_rates(made: dict, py_class: dict) -> dict:
    """PyCHAM / TOMAS-JAX condensation-sink ratio of every condensable species on the same
    size distribution (t = 0: TOMAS-JAX seed; 6 h: PyCHAM's wet radii and numbers),
    weighted by PyCHAM's 6 h particle-phase composition -> factor for the attribution run."""
    cmap = _class_map(None)
    s = case.scenario()
    env = s.environment
    in_dir = pycham_inputs.case_input_dir(case.CASE)
    dv = transfer.pycham_diffusion_volumes(in_dir / "pycham_mw.json",
                                           in_dir / "pycham_diffvol.json", list(cmap.species))
    seed = inputs.seed_state(s)
    r0 = (3.0 * seed["m_mid_kg"] * (1.0 + seed["water_to_dry"]) / seed["rho_wet_tomas"]
          / (4.0 * np.pi)) ** (1.0 / 3.0)
    names = py_class["names"]
    w = py_class["part_ug_m3"][-1][:, [names.index(n) for n in cmap.species]].sum(axis=0)
    w = w / w.sum()
    n_bin = py_class["N_bin"]
    out = {"weights": "PyCHAM class-centre run, particle-phase mass at 6 h"}
    for tag, r_m, n_cm3 in (("t0_tomas_seed", r0, seed["n_cm3"]),
                            ("6h_pycham_sizes", py_class["rad_um"][-1] * 1e-6, n_bin[-1])):
        res = transfer.transfer_ratios(r_m, n_cm3, made["mw"], dv, env.temp_k, env.pres_pa)
        out[tag] = {"cs_ratio_weighted": float(np.sum(w * res["cs_ratio"])),
                    "d_ratio_weighted": float(np.sum(w * res["d_ratio"])),
                    "cs_ours_weighted_per_s": float(np.sum(w * res["cs_ours"])),
                    "cs_pycham_weighted_per_s": float(np.sum(w * res["cs_pycham"]))}
    top = np.argsort(-w)[:8]
    res0 = transfer.transfer_ratios(r0, seed["n_cm3"], made["mw"], dv, env.temp_k, env.pres_pa)
    out["top_species_t0"] = [{"species": cmap.species[k], "diff_vol_pycham": float(dv[cmap.species[k]]),
                              "diff_vol_ours": float(transfer.SV_ORG),
                              "d_ratio": float(res0["d_ratio"][k]),
                              "cs_ratio": float(res0["cs_ratio"][k])} for k in top]
    out["attribution_factor"] = out["t0_tomas_seed"]["cs_ratio_weighted"]
    _json(case.TRANSFER_JSON, out)
    return out


def run_ours_attribution(factor: float, info: dict) -> None:
    """TOMAS-JAX with its organic condensation sink scaled, with and without the Kelvin
    term (each in a fresh process, bounded)."""
    dt = S2["ours_attribution_dt_s"]
    for tag, opts in S2["ours_attribution_runs"].items():
        print(f"TOMAS-JAX attribution {tag}: coupling {dt:.0f} s, organic CS x {factor:.3f}, "
              f"Kelvin off {opts['kelvin_off']}", flush=True)
        t0 = time.perf_counter()
        cmd = [sys.executable, "-m", "benchmarks.pycham.cases.S2.attribution",
               f"{factor:.10g}", f"{dt:g}", str(case.OURS_ATTR_NPZ[tag])]
        if opts["kelvin_off"]:
            cmd.append("--kelvin-off")
        proc = subprocess.run(cmd, cwd=TOMAS_JAX_ROOT, capture_output=True, text=True,
                              timeout=OURS_TIMEOUT_S)
        if proc.returncode != 0:
            raise RuntimeError(f"attribution run {tag} failed ({proc.returncode}): "
                               f"{proc.stderr[-3000:]}")
        info[f"ours_{tag}"] = {"wall_s": time.perf_counter() - t0, "cs_factor": factor,
                               "coupling_dt_s": dt, "kelvin_off": opts["kelvin_off"],
                               "stdout": proc.stdout.strip()[-500:]}


def run_ours_all(info: dict) -> dict:
    runs = {}
    for dt in (S2["coupling_dt_s"], S2["coupling_dt_sensitivity_s"]):
        print(f"TOMAS-JAX: coupling {dt:.0f} s", flush=True)
        runs[int(dt)] = ours.run_ours(case.scenario(coupling_dt_s=dt))
        ours.save_npz(case.OURS_NPZ[int(dt)], runs[int(dt)], every_s=COMPARE_DT_S)
        info[f"ours_dt{int(dt)}"] = {"wall_s": float(runs[int(dt)]["runtime_s"]),
                                     "gas_solver_steps": int(runs[int(dt)]["gas_solver_steps"]),
                                     "warnings": runs[int(dt)]["result"].diagnostics["warnings"],
                                     "code_version": code_version()}
    print("TOMAS-JAX: coupling 300 s, SOA partitioning and H2SO4 condensation off", flush=True)
    nop = ours.run_ours(case.scenario(partitioning=False))
    ours.save_npz(case.OURS_NOPART_NPZ, nop, every_s=COMPARE_DT_S)
    info["ours_dt300_nopart"] = {"wall_s": float(nop["runtime_s"])}
    # seed OA molar mass check: number-mean MW of the TOMAS-JAX particle-phase SOA
    cmap = _class_map(None)
    o = runs[int(S2["coupling_dt_s"])]
    chk = {}
    for th in (3.0, 6.0):
        i = int(np.argmin(np.abs(o["t_s"] - th * 3600.0)))
        p = o["condensable_particle_ugm3"][i]
        chk[f"{th:g}h_number_mean_mw"] = float(p.sum() / np.sum(p / cmap.mw))
    info["seed_oa_mw_check"] = chk
    return runs


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-pycham", action="store_true", help="reuse the existing PyCHAM output")
    ap.add_argument("--skip-ours", action="store_true", help="reuse existing TOMAS-JAX results")
    ap.add_argument("--smoke", action="store_true",
                    help=f"only run PyCHAM 'class' for {SMOKE_HOURS} h (no results saved)")
    ap.add_argument("--variants", default=None,
                    help="comma list of PyCHAM variants to run (default: all)")
    args = ap.parse_args(argv)
    case.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    made = make_inputs(smoke=args.smoke)
    if args.smoke:
        wall = run_pycham_variant(made["paths"]["class"]["model_var"], "smoke_class")
        r = load_pycham(str(made["paths"]["class"]["output"]))
        print(json.dumps({"wall_s": wall, "t_s": r["t_s"].tolist(),
                          "psat": check_pycham_output(r, made["rows"]["class"])}, indent=1))
        return 0
    info_path = case.RUN_INFO_JSON
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    variants = case.PYCHAM_ALL_VARIANTS
    selected = variants if args.variants is None else tuple(args.variants.split(","))
    if not args.skip_pycham:
        for v in selected:
            wall = run_pycham_variant(made["paths"][v]["model_var"], v,
                                      kelvin_off=v in S2["pycham_kelvin_off_variants"])
            info[f"pycham_{v}"] = {"wall_s": wall, "output": str(made["paths"][v]["output"]),
                                   "sha": PYCHAM_SHA, "umansysprop_sha": UMANSYSPROP_SHA,
                                   "update_step_s": PYCHAM_UPDATE_STEP_S,
                                   "recording_step_s": S2["pycham_recording_step_s"]}
    loaded = {}
    for v in variants:
        out = made["paths"][v]["output"]
        loaded[v] = save_pycham(out, case.PYCHAM_NPZ[v], made["mw"],
                                condensables_only=v in GAS_SUBSET_VARIANTS,
                                per_bin=v not in NO_BIN_VARIANTS)
        rows = made["rows"]["class" if v == S2["pycham_gas_only_variant"] else v]
        info.setdefault(f"pycham_{v}", {})["psat_check"] = check_pycham_output(loaded[v], rows)
        with np.load(case.PYCHAM_NPZ[v]) as f:
            if "other_particle_max_ugm3" in f.files:
                info[f"pycham_{v}"]["other_particle_max_ugm3"] = float(f["other_particle_max_ugm3"])
    info["ro2_pool"] = check_ro2_pool(made["paths"]["class"]["output"])
    info["transfer_rates"] = transfer_rates(made, loaded["class"])
    if not args.skip_ours:
        runs = run_ours_all(info)
        info["initial_particles"] = initial_particle_check(loaded["class"],
                                                           runs[int(S2["coupling_dt_s"])])
        run_ours_attribution(info["transfer_rates"]["attribution_factor"], info)
    gas_files = ([case.PYCHAM_NPZ[v] for v in variants] + list(case.OURS_NPZ.values())
                 + [case.OURS_NOPART_NPZ] + list(case.OURS_ATTR_NPZ.values()))
    info["stored_species"] = prune_gas_species([p for p in gas_files if p.exists()])
    info["setup"] = made["setup"]
    _json(info_path, info)
    print(json.dumps({k: v for k, v in info.items() if k != "setup"}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    t0 = time.perf_counter()
    rc = main()
    print(f"total {time.perf_counter() - t0:.0f} s")
    sys.exit(rc)
