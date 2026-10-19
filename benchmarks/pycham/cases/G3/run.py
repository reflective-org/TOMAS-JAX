"""Case G3 runner: write PyCHAM inputs, run PyCHAM and TOMAS-JAX, save small results.

    cd tomas-jax && .venv/bin/python -m benchmarks.pycham.cases.G3.run [--skip-pycham] [--skip-ours]

PyCHAM (own checkout ``pycham-bench/PyCHAM_G3``, runs sequential, each bounded):
  j60        J.xlsx on the 60 s grid (our J at each 60 s interval midpoint) - reference
  j900       J.xlsx on the 900 s grid (= the J staircase of TOMAS-JAX at 900 s coupling)
  j60_tight  as j60 with int_tol rtol 1e-6 (PyCHAM's own solver error)
TOMAS-JAX (``case.OURS_RUNS``): preset coupling 900 s, 60 s, and attribution runs (NO2
held inside the solver like PyCHAM, tight tolerances, PyCHAM water vapour), plus a
near-zero-seed run (aerosol independence) and the unmodified Level 3 preset (aerosol on)
for the particle-relevant species.

Checks: our recorded J equals the exported J exactly; all 2135 rate coefficients from
PyCHAM's generated rate_coeffs.py (with our J.xlsx) vs ours at five times incl. the
sunset/sunrise intervals; RO2 pool; NO2 constant in both; photostationary state.

Outputs in ``cases/G3/results`` (see case.py); raw PyCHAM output in
``$PYCHAM_BENCH/outputs/G3_*``.
"""
from __future__ import annotations

import argparse
import contextlib
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
from scipy.optimize import brentq

from tomas_jax.box.coupling import N2O5_HET_LABEL
from tomas_jax.box.driver import _chemistry, code_version, run
from tomas_jax.gaschem.environment import air_environment
from tomas_jax.gaschem.mechanism import EnvRates
from tomas_jax.gaschem.photolysis import JProvider, PhotolysisSettings
from tomas_jax.gaschem.solver import integrate_interval

from ... import jvalues, ours, pycham_inputs
from ...config import (ABUNDANCE_THRESHOLD_PPB, BENCH_DIR, MCM_J_NUMBER, OURS_TIMEOUT_S,
                       PYCHAM_PYTHON, PYCHAM_SHA, PYCHAM_TIMEOUT_S, PYCHAM_UPDATE_STEP_S,
                       UMANSYSPROP_SHA)
from ...load_pycham import load_pycham
from . import case, hold
from .case import G3

RATE_CHECK_TIMEOUT_S = 300.0
KEY_J = ("J_NO2", "J_O3_O1D", "J_NO3_NO2", "J_HCHO_H")


def _json(path: Path, obj) -> None:
    path.write_text(json.dumps(obj, indent=2, default=lambda o: o.tolist()
                               if isinstance(o, np.ndarray) else float(o)) + "\n")


# --- sun -------------------------------------------------------------------------------
def sun_info() -> dict:
    """Solar zenith angle on a 60 s grid and the sunset/sunrise times (SZA = 90 deg)."""
    s = case.scenario()
    p = s.photolysis
    jp = JProvider(_chemistry()[1].j_names, PhotolysisSettings(
        p.mode, p.latitude_deg, p.longitude_deg, s.start_utc, p.jno2_per_s,
        p.constant_sza_deg))
    t = np.arange(0.0, s.duration_h * 3600.0 + 1.0, 60.0)
    sza = np.array([jp.sza_deg(x) for x in t])
    night = sza >= 90.0
    edges = np.flatnonzero(np.diff(night.astype(int)))
    if edges.size != 2:
        raise RuntimeError(f"expected one sunset and one sunrise, got {edges.size} transitions")
    cross = [brentq(lambda x: jp.sza_deg(x) - 90.0, t[e], t[e + 1], xtol=1e-3) for e in edges]
    out = {"t_sunset_h": cross[0] / 3600.0, "t_sunrise_h": cross[1] / 3600.0,
           "start_utc": s.start_utc.isoformat(), "utc_offset_h": G3["utc_offset_h"],
           "sza_min_deg": float(sza.min()), "t_sza_min_h": float(t[np.argmin(sza)] / 3600.0)}
    _json(case.SUN_JSON, out)
    return {**out, "t_s": t, "sza_deg": sza}


# --- inputs ----------------------------------------------------------------------------
def make_inputs() -> dict:
    """PyCHAM inputs for G3 in $PYCHAM_BENCH/inputs/G3; returns paths and exported J."""
    in_dir = pycham_inputs.case_input_dir(case.CASE)
    eqn = pycham_inputs.write_eqn(in_dir / "limonene_mcm_G3.eqn", G3["eqn_rate_overrides"])
    xml = pycham_inputs.write_xml(in_dir / "limonene_mcm.xml")
    constants = pycham_inputs.write_constants(in_dir / "constants_mcm.f90")
    xlsx, j_export = {}, {}
    for j_dt in sorted({v["j_dt_s"] for v in case.PYCHAM_VARIANTS.values()}):
        t0, j_names, j = jvalues.j_per_interval(case.scenario(coupling_dt_s=j_dt))
        _, xlsx[j_dt] = jvalues.write_j_files(in_dir, t0, jvalues.to_pycham_rows(j_names, j),
                                              stem=f"J_dt{int(j_dt)}")
        j_export[j_dt] = {"t0_s": t0, "j_names": j_names, "j": j}
    j60 = j_export[60.0]
    with open(case.RESULTS_DIR / "j_prescribed.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["j_name", "mcm_number", "j_max_per_s", "t_of_max_h", "j_at_noon_end_per_s"])
        for col, name in enumerate(j60["j_names"]):
            k = int(np.argmax(j60["j"][:, col]))
            w.writerow([name, MCM_J_NUMBER[name], f"{j60['j'][k, col]:.6e}",
                        f"{(j60['t0_s'][k] + 30.0) / 3600.0:.3f}", f"{j60['j'][-1, col]:.6e}"])

    model_vars = {}
    for variant, spec in case.PYCHAM_VARIANTS.items():
        s = case.scenario(coupling_dt_s=spec["j_dt_s"])
        # hold_constant -> PyCHAM const_comp (written by the shared harness)
        mv = pycham_inputs.scenario_model_var(
            s, res_file=pycham_inputs.case_output_dir(case.CASE, variant),
            eqn=eqn, xml=xml, constants=constants, j_xlsx=xlsx[spec["j_dt_s"]])
        if mv.get("const_comp") != ", ".join(G3["held_species"]):
            raise RuntimeError(f"const_comp {mv.get('const_comp')!r} != {G3['held_species']}")
        if spec["tight"]:
            atol, rtol = G3["pycham_tight_int_tol"]
            mv["int_tol"] = f"{atol:g}, {rtol:g}"
        path = pycham_inputs.write_model_var(in_dir / f"G3_{variant}_model_var.txt", mv)
        shutil.copyfile(path, case.RESULTS_DIR / f"pycham_model_var_{variant}.txt")
        model_vars[variant] = path
    return {"model_vars": model_vars, "xlsx": xlsx, "j_export": j_export, "eqn": eqn,
            "in_dir": in_dir}


# --- PyCHAM ----------------------------------------------------------------------------
def run_pycham_variant(model_var: Path, variant: str) -> dict:
    root = pycham_inputs.ensure_pycham_root(case.CASE)
    log = model_var.parent / f"pycham_{variant}.log"
    print(f"PyCHAM {variant}: {model_var} (root {root}, timeout {PYCHAM_TIMEOUT_S:.0f} s)",
          flush=True)
    wall = pycham_inputs.run_pycham(model_var, root, log)
    print(f"PyCHAM {variant} done in {wall:.1f} s", flush=True)
    return {"root": str(root), "wall_s": wall, "log": str(log)}


def check_ro2_pool(out_dir: Path) -> dict:
    names = [str(n) for n in np.load(out_dir / "comp_namelist.npy", allow_pickle=True)]
    idx = np.load(out_dir / "organic_peroxy_radical_pool_index.npy", allow_pickle=True)
    pool = {names[i] for i in np.asarray(idx, dtype=int).ravel()}
    ref = set(_chemistry()[0].ro2)
    return {"n_pycham": len(pool), "n_ours": len(ref), "identical": pool == ref,
            "only_pycham": sorted(pool - ref), "only_ours": sorted(ref - pool)}


# --- TOMAS-JAX -------------------------------------------------------------------------
def run_ours_tag(tag: str) -> dict:
    dt, tight, water, held = case.OURS_RUNS[tag]
    print(f"TOMAS-JAX {tag}", flush=True)
    with contextlib.ExitStack() as stack:
        if tight or water:
            tol = G3["ours_tight_tol"] if tight else {}
            stack.enter_context(ours.numerics_override(**tol, pycham_water=water))
        if held:
            stack.enter_context(hold.held_in_solver(G3["held_species"]))
        rd = ours.run_ours(case.scenario(coupling_dt_s=dt))
    print(f"  {rd['runtime_s']:.1f} s, {int(rd['gas_solver_steps'])} solver steps", flush=True)
    return rd


def check_j_identical(rd: dict, exported: dict) -> float:
    """Our recorded J per interval vs the J written to J.xlsx (must be identical)."""
    if list(map(str, rd["j_names"])) != list(exported["j_names"]):
        raise RuntimeError("J name order differs between the run and the export")
    diff = float(np.max(np.abs(rd["j"][1:] - exported["j"])))
    if diff != 0.0:
        raise RuntimeError(f"J used by TOMAS-JAX differs from the exported J by {diff:g}")
    return diff


def particle_relevant_species() -> list[dict]:
    """Top particle-phase contributors of the unmodified Level 3 preset (aerosol on)."""
    deadline = time.monotonic() + OURS_TIMEOUT_S
    t0 = time.perf_counter()
    res = run(case.full_level3_scenario(), cancel=lambda: time.monotonic() > deadline)
    _json(case.TOP_CONTRIBUTORS_JSON, {
        "scenario": "level3_day_night (unmodified preset, aerosol on, 24 h)",
        "soa_ugm3_end": res.totals["SOA_ugm3"][-1], "nit_ugm3_end": res.totals["NIT_ugm3"][-1],
        "wall_s": time.perf_counter() - t0, "top_contributors": res.top_contributors})
    return res.top_contributors


# --- rate constants --------------------------------------------------------------------
def _pycham_k(root: Path, xlsx: Path, envd: dict) -> np.ndarray:
    with tempfile.TemporaryDirectory() as tmp:
        env_path, out_path = Path(tmp) / "env.json", Path(tmp) / "k.npy"
        env_path.write_text(json.dumps(envd))
        proc = subprocess.run([str(PYCHAM_PYTHON), "-I", str(BENCH_DIR / "check_rates_pycham.py"),
                               str(root), str(xlsx), str(env_path), str(out_path)],
                              capture_output=True, text=True, timeout=RATE_CHECK_TIMEOUT_S)
        if proc.returncode != 0:
            raise RuntimeError(f"rate check failed: {proc.stderr[-2000:]}")
        return np.load(out_path)


def check_rates(root: Path, xlsx: dict, runs: dict) -> dict:
    """All k: PyCHAM's rate_coeffs.py + J.xlsx vs TOMAS-JAX, at G3 times (60 s and 900 s J)."""
    s = case.scenario()
    env = s.environment
    air = air_environment(env.temp_k, env.pres_pa, env.rh)
    parsed, rate_model, mech = _chemistry()
    k_th, ro2_dep, j_coef = rate_model.at_env(air)
    rates = EnvRates(jnp.asarray(k_th), jnp.asarray(ro2_dep), jnp.asarray(j_coef))
    i_het = rate_model.labels.index(N2O5_HET_LABEL)
    idx = {n: k for k, n in enumerate(parsed.species)}
    h2o_pycham = ours.h2o_hyland_wexler_cm3(env.temp_k, env.rh)
    out = {"H2O_ours_buck_cm3": air["H2O"], "H2O_pycham_hyland_wexler_cm3": h2o_pycham,
           "checks": []}
    for t_h in G3["rate_check_times_h"]:
        for j_dt, tag in ((60.0, "dt60"), (900.0, "dt900")):
            rd = runs[tag]
            i = int(np.floor(t_h * 3600.0 / j_dt + 1e-9))      # interval holding t_h
            y = rd["gas_ppb"][i] * 1e-9 * rd["m_air_cm3"]       # state at interval start
            ro2 = float(mech.ro2(jnp.asarray(y)))
            j = rd["j"][i + 1]                                  # J used over that interval
            k_ours = np.array(rates.k(jnp.asarray(ro2), jnp.asarray(j)))
            k_ours[i_het] = 0.0                                 # gamma = 0
            row = {"t_h": t_h, "j_grid_s": j_dt, "interval_start_h": i * j_dt / 3600.0,
                   "RO2_cm3": ro2, "J_NO2_per_s": float(j[list(rd["j_names"]).index("J_NO2")])}
            for wtag, h2o in (("same_h2o", air["H2O"]), ("pycham_h2o", h2o_pycham)):
                envd = {"TEMP": air["TEMP"], "M": air["M"], "N2": air["N2"], "O2": air["O2"],
                        "H2O": h2o, "RO2": ro2, "RH": env.rh, "NO": float(y[idx["NO"]]),
                        "HO2": float(y[idx["HO2"]]), "NO3": float(y[idx["NO3"]]),
                        "sumt": i * j_dt + 0.5 * j_dt}
                k_py = _pycham_k(root, xlsx[j_dt], envd)
                if k_py.shape != k_ours.shape:
                    raise RuntimeError(f"reaction count differs: {k_py.shape} vs {k_ours.shape}")
                rel = np.abs(k_py - k_ours) / np.maximum(np.abs(k_ours), 1e-300)
                rel[(k_py == 0) & (k_ours == 0)] = 0.0
                jdep = np.any(j_coef != 0.0, axis=1)
                worst = np.argsort(-rel)[:5]
                row[wtag] = {"n_reactions": int(k_ours.size), "max_rel_diff": float(rel.max()),
                             "max_rel_diff_photolysis": float(rel[jdep].max()),
                             "n_photolysis_nonzero": int(np.sum(jdep & (k_ours > 0))),
                             "worst": [{"label": rate_model.labels[w], "rel_diff": float(rel[w]),
                                        "k_ours": float(k_ours[w]), "k_pycham": float(k_py[w]),
                                        "rate_expr": parsed.reactions[w].rate_expr}
                                       for w in worst]}
            out["checks"].append(row)
            print(f"  rate check t={t_h} h J grid {j_dt:.0f} s: max rel diff "
                  f"{row['same_h2o']['max_rel_diff']:.2e} (same H2O), "
                  f"{row['pycham_h2o']['max_rel_diff']:.2e} (PyCHAM H2O)", flush=True)
    out["max_rel_diff_same_h2o"] = max(c["same_h2o"]["max_rel_diff"] for c in out["checks"])
    out["max_rel_diff_pycham_h2o"] = max(c["pycham_h2o"]["max_rel_diff"] for c in out["checks"])
    _json(case.RATES_CHECK_JSON, out)
    return out


def held_within_intervals(rd: dict, sub_dt_s: float) -> dict:
    """What the held species does *inside* each coupling interval of a TOMAS-JAX run.

    The driver integrates every interval from its recorded start state with frozen k and
    J and resets the held species afterwards, so outputs only ever show the reset value.
    This re-integrates each interval from the recorded start state with the same k (RO2
    at the start, the J of the interval, k_het = 0) to every ``sub_dt_s`` and returns
    the held species there. Self-check: at the full interval every other species must
    reproduce the recorded state at the interval end.
    """
    s = case.scenario(coupling_dt_s=float(rd["coupling_dt_s"]))
    env = s.environment
    air = air_environment(env.temp_k, env.pres_pa, env.rh)
    parsed, rate_model, mech = _chemistry()
    rates = EnvRates(*(jnp.asarray(a) for a in rate_model.at_env(air)))
    i_het = rate_model.labels.index(N2O5_HET_LABEL)
    names = [str(n) for n in rd["species"]]
    held = [names.index(sp) for sp in G3["held_species"]]
    skip = set(held) | {names.index(sp) for sp in ("SA", "NA_HET")}
    others = np.array([k for k in range(len(names)) if k not in skip])
    dt = float(rd["coupling_dt_s"])
    subs = np.arange(sub_dt_s, dt + 0.5 * sub_dt_s, sub_dt_s)
    m_air = float(rd["m_air_cm3"])
    n_int = len(rd["t_s"]) - 1
    vals = np.zeros((n_int, subs.size, len(held)))
    worst = 0.0
    for i in range(n_int):
        y0 = jnp.asarray(rd["gas_ppb"][i] * 1e-9 * m_air)
        k = rates.k(mech.ro2(y0), jnp.asarray(rd["j"][i + 1])).at[i_het].set(0.0)
        for m, sub in enumerate(subs):
            y, _ = integrate_interval(mech, y0, k, jnp.float64(sub))
            y = np.asarray(y)
            vals[i, m] = y[held] / m_air * 1e9
        end = y[others] / m_air * 1e9
        rec = rd["gas_ppb"][i + 1][others]
        ok = rec >= ABUNDANCE_THRESHOLD_PPB
        worst = max(worst, float(np.max(np.abs(end[ok] - rec[ok]) / rec[ok])))
    if worst > 1e-6:
        raise RuntimeError(f"re-integrated intervals differ from the run by {worst:g}")
    t = (rd["t_s"][:-1, None] + subs[None, :]).ravel()
    return {"t_s": t, "ppb": vals.reshape(-1, len(held)), "t_start_s": rd["t_s"][:-1],
            "self_check_max_rel_diff": worst}


def photostationary(t_s, ppb, names, j_t0, j_no2, m_air) -> dict:
    """Leighton ratio J_NO2[NO2] / (k_NO+O3 [NO][O3]) in daylight (SZA < 80 deg via J)."""
    parsed, rate_model, _ = _chemistry()
    lab = [r.label for r in parsed.reactions
           if sorted(r.reactants) == ["NO", "O3"]]
    if len(lab) != 1:
        raise RuntimeError(f"expected one NO + O3 reaction, got {lab}")
    s = case.scenario()
    air = air_environment(s.environment.temp_k, s.environment.pres_pa, s.environment.rh)
    k3 = float(rate_model.at_env(air)[0][rate_model.labels.index(lab[0])])
    # J over (t - dt, t]: the value of the interval that ends at t
    col = np.clip(np.searchsorted(j_t0, t_s, side="left") - 1, 0, len(j_t0) - 1)
    jn = j_no2[col]
    no, no2, o3 = (ppb[:, names.index(n)] * 1e-9 * m_air for n in ("NO", "NO2", "O3"))
    day = (jn > 0.3 * j_no2.max()) & (t_s > 0)
    phi = jn[day] * no2[day] / (k3 * no[day] * o3[day])
    return {"n_times": int(day.sum()), "median": float(np.median(phi)),
            "min": float(phi.min()), "max": float(phi.max()), "k_NO_O3": k3, "label": lab[0]}


# --- storage ---------------------------------------------------------------------------
def stored_species(arrays: list[tuple[list[str], np.ndarray]]) -> list[str]:
    """Species ever >= store_floor_ppb in any run (union, mechanism order)."""
    keep = set()
    for names, ppb in arrays:
        mx = np.nanmax(ppb, axis=0)
        keep |= {n for n, v in zip(names, mx) if v >= G3["store_floor_ppb"]}
    order = list(_chemistry()[0].species) + sorted(keep - set(_chemistry()[0].species))
    return [n for n in order if n in keep]


def save_pycham_npz(r: dict, npz: Path, keep_species: list[str], every_s: float) -> None:
    keep_t = ours.on_grid(r["t_s"], every_s)
    sp = [n for n in keep_species if n in r["names"]]
    cols = [r["names"].index(n) for n in sp]
    np.savez_compressed(npz, t_s=r["t_s"][keep_t], species=np.array(sp),
                        gas_ppb=r["gas_ppb"][keep_t][:, cols].astype(np.float32),
                        ppb_to_molec_cm3=r["ppb_to_molec_cm3"][keep_t], env=r["env"][keep_t])


def save_ours_npz(rd: dict, npz: Path, keep_species: list[str], every_s: float) -> None:
    names = [str(n) for n in rd["species"]]
    sp = [n for n in keep_species if n in names]
    sub = dict(rd, species=np.array(sp), gas_ppb=rd["gas_ppb"][:, [names.index(n) for n in sp]])
    # the per-species particle tracker is all zero in a gas-only case
    sub.pop("condensable_particle_ugm3", None)
    sub.pop("condensable_species", None)
    ours.save_npz(npz, sub, every_s=every_s)


def key_species_list(top: list[dict], pycham_names: list[str]) -> list[str]:
    names = list(G3["key_species"]) + list(G3["product_species"]) + [c["species"] for c in top]
    names += list(G3["transition_species"]) + ["SA"]
    missing = [n for n in names if n not in pycham_names]
    if missing:
        raise RuntimeError(f"key species missing in PyCHAM output: {missing}")
    return list(dict.fromkeys(names))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-pycham", action="store_true", help="reuse the existing PyCHAM output")
    ap.add_argument("--skip-ours", action="store_true",
                    help="reuse saved TOMAS-JAX results (no rate check / key-species refresh)")
    args = ap.parse_args(argv)
    case.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    info = json.loads(case.RUN_INFO_JSON.read_text()) if case.RUN_INFO_JSON.exists() else {}

    sun = sun_info()
    inputs = make_inputs()
    out_dirs = {v: pycham_inputs.case_output_dir(case.CASE, v) for v in case.PYCHAM_VARIANTS}
    if not args.skip_pycham:
        for variant in case.PYCHAM_VARIANTS:
            pr = run_pycham_variant(inputs["model_vars"][variant], variant)
            spec = case.PYCHAM_VARIANTS[variant]
            info[f"pycham_{variant}"] = {
                **pr, "sha": PYCHAM_SHA, "umansysprop_sha": UMANSYSPROP_SHA,
                "update_step_s": PYCHAM_UPDATE_STEP_S, "j_grid_s": spec["j_dt_s"],
                "int_tol": G3["pycham_tight_int_tol"] if spec["tight"] else "default",
                "output": str(out_dirs[variant])}
    pycham = {v: load_pycham(str(d)) for v, d in out_dirs.items()}
    info["ro2_pool"] = check_ro2_pool(out_dirs["j60"])
    if not info["ro2_pool"]["identical"]:
        raise RuntimeError(f"RO2 pools differ: {info['ro2_pool']}")
    for v, r in pycham.items():
        no2 = r["gas_ppb"][:, r["names"].index("NO2")]
        info.setdefault("no2_held", {})[f"pycham_{v}"] = {
            "min_ppb": float(no2.min()), "max_ppb": float(no2.max())}

    if args.skip_ours:
        runs = {tag: ours.load_npz(case.OURS_NPZ[tag]) for tag in case.OURS_RUNS}
    else:
        runs = {tag: run_ours_tag(tag) for tag in case.OURS_RUNS}
        for tag, rd in runs.items():
            dt, tight, water, held = case.OURS_RUNS[tag]
            info[f"ours_{tag}"] = {"coupling_dt_s": dt, "wall_s": float(rd["runtime_s"]),
                                   "gas_solver_steps": int(rd["gas_solver_steps"]),
                                   "solver_tol": G3["ours_tight_tol"] if tight else "default",
                                   "pycham_water": water, "no2_held_in_solver": held,
                                   "code_version": code_version()}
        info["j_identical"] = {tag: check_j_identical(runs[tag], inputs["j_export"][dt])
                               for tag, dt in (("dt900", 900.0), ("dt60", 60.0))}
        print("TOMAS-JAX dt900, seed number ~0 (aerosol-independence check)", flush=True)
        noseed = ours.run_ours(case.scenario(seed_number_cm3=1e-6))
        a, b = runs["dt900"]["gas_ppb"], noseed["gas_ppb"]
        rel = np.abs(a - b) / np.maximum(np.abs(a), 1e-30)
        rel[(a < ABUNDANCE_THRESHOLD_PPB) & (b < ABUNDANCE_THRESHOLD_PPB)] = 0.0
        info["aerosol_independence"] = {
            "description": "gas phase with the 3000 cm-3 Level 3 seed vs a 1e-6 cm-3 seed, "
                           "all aerosol processes off, coupling 900 s",
            "max_rel_diff_above_threshold": float(rel.max()),
            "max_abs_diff_ppb": float(np.abs(a - b).max())}
        sp = [str(n) for n in runs["dt60"]["species"]]
        info["no2_held"]["ours_dt900_at_outputs"] = {
            "min_ppb": float(runs["dt900"]["gas_ppb"][:, sp.index("NO2")].min()),
            "max_ppb": float(runs["dt900"]["gas_ppb"][:, sp.index("NO2")].max())}

    top = particle_relevant_species() if not args.skip_ours else json.loads(
        case.TOP_CONTRIBUTORS_JSON.read_text())["top_contributors"]

    if not args.skip_ours:
        info["rate_constant_check"] = {
            k: v for k, v in check_rates(Path(info["pycham_j60"]["root"]), inputs["xlsx"],
                                         runs).items() if k.startswith("max_rel")}
        j60 = inputs["j_export"][60.0]
        jn = j60["j"][:, j60["j_names"].index("J_NO2")]
        py = pycham["j60"]
        o60 = runs["dt60"]
        info["photostationary_state"] = {
            "pycham_j60": photostationary(py["t_s"], py["gas_ppb"], py["names"], j60["t0_s"], jn,
                                          float(o60["m_air_cm3"])),
            "ours_dt60": photostationary(o60["t_s"], o60["gas_ppb"], [str(n) for n in o60["species"]],
                                         j60["t0_s"], jn, float(o60["m_air_cm3"]))}

        print("TOMAS-JAX: held NO2 inside the coupling intervals (re-integration)", flush=True)
        held_in = {tag: held_within_intervals(runs[tag], min(G3["no2_substep_s"],
                                                                float(runs[tag]["coupling_dt_s"])))
                   for tag in case.MAIN_RUNS}
        info["held_within_intervals_self_check"] = {
            tag: h["self_check_max_rel_diff"] for tag, h in held_in.items()}

        # ---- small result files ------------------------------------------------------
        arrays = [(r["names"], r["gas_ppb"]) for r in pycham.values()]
        arrays += [([str(n) for n in rd["species"]], rd["gas_ppb"]) for rd in runs.values()]
        keep = stored_species(arrays)
        info["stored_species"] = {"n": len(keep), "floor_ppb": G3["store_floor_ppb"]}
        for v, r in pycham.items():
            save_pycham_npz(r, case.PYCHAM_NPZ[v], keep, G3["compare_dt_s"]
                            if v in case.MAIN_PYCHAM else G3["attribution_dt_s"])
        for tag, rd in runs.items():
            save_ours_npz(rd, case.OURS_NPZ[tag], keep, G3["compare_dt_s"]
                          if tag in case.MAIN_RUNS else G3["attribution_dt_s"])
        info["storage_grid_s"] = {"main": G3["compare_dt_s"], "attribution": G3["attribution_dt_s"],
                                  "key_species_60s_runs": G3["key_dt_s"]}

        keys = key_species_list(top, pycham["j60"]["names"])
        key = {"species": np.array(keys), "sza_t_s": sun["t_s"],
               "sza_deg": sun["sza_deg"].astype(np.float32)}
        for j_dt, ex in inputs["j_export"].items():
            key[f"j_dt{int(j_dt)}_t0_s"] = ex["t0_s"]
            key[f"j_dt{int(j_dt)}_names"] = np.array(KEY_J)
            key[f"j_dt{int(j_dt)}"] = ex["j"][:, [ex["j_names"].index(n) for n in KEY_J]]
        # time series for figures: the 60 s runs on the key grid (120 s), the 900 s runs at
        # every output, the attribution-only runs on the compare grid (900 s)
        for v, r in pycham.items():
            kt = ours.on_grid(r["t_s"], G3["compare_dt_s"] if v == "j60_tight" else G3["key_dt_s"])
            key[f"pycham_{v}_t_s"] = r["t_s"][kt]
            key[f"pycham_{v}_ppb"] = r["gas_ppb"][kt][:, [r["names"].index(n) for n in keys]
                                                      ].astype(np.float32)
            key[f"pycham_{v}_cfac"] = r["ppb_to_molec_cm3"][kt]
        for tag, rd in runs.items():
            sp = [str(n) for n in rd["species"]]
            kt = ours.on_grid(rd["t_s"], G3["key_dt_s"] if tag == "dt60" else G3["compare_dt_s"])
            key[f"ours_{tag}_t_s"] = rd["t_s"][kt]
            key[f"ours_{tag}_ppb"] = rd["gas_ppb"][kt][:, [sp.index(n) for n in keys]
                                                       ].astype(np.float32)
            key[f"ours_{tag}_sa_cum_ppb"] = (rd["sa_cum_cm3"][kt] / rd["m_air_cm3"] * 1e9
                                             ).astype(np.float32)
        for tag, h in held_in.items():
            key[f"held_{tag}_t_s"] = h["t_s"]
            key[f"held_{tag}_ppb"] = h["ppb"].astype(np.float32)
        key["held_species"] = np.array(G3["held_species"])
        key["m_air_cm3"] = np.float64(runs["dt900"]["m_air_cm3"])
        np.savez_compressed(case.KEY_NPZ, **key)
        info["m_air_cm3"] = {"ours": float(runs["dt900"]["m_air_cm3"]),
                             "pycham": float(pycham["j60"]["ppb_to_molec_cm3"][0] * 1e9)}
        sizes = {p.name: p.stat().st_size for p in case.RESULTS_DIR.iterdir() if p.is_file()}
        info["result_bytes_total"] = int(sum(sizes.values()))
    info["sun"] = {k: v for k, v in sun.items() if k not in ("t_s", "sza_deg")}
    info["threshold_ppb"] = ABUNDANCE_THRESHOLD_PPB
    _json(case.RUN_INFO_JSON, info)
    print(json.dumps({k: v for k, v in info.items() if not k.startswith("pycham_")}, indent=1,
                     default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
