"""Case S1 runner: write PyCHAM inputs, run PyCHAM and TOMAS-JAX, save small results.

    cd tomas-jax && .venv/bin/python -m benchmarks.pycham.cases.S1.run [--skip-pycham] [--skip-ours]

For c* = 1 and 100 ug/m3:

* PyCHAM ``base`` (update step 60 s, int_tol default), ``tight`` (update step 10 s, rtol
  1e-6; numerical sensitivity), ``nokelvin`` (surface tension 0 in the patched copy
  PyCHAM_S1_nokelvin from ``pycham_inputs.ensure_pycham_root(kelvin_off=True)``;
  attribution only) and ``bins160`` (TOMAS-JAX's 160-bin initial state).
* TOMAS-JAX ``base``, ``micro10`` (aerosol micro-step 10 s), ``nokelvin`` (vbs-jax sigma
  0), ``nokelvin_pytransfer`` (sigma 0 and PyCHAM's gas-to-particle transfer
  coefficient; attribution only), ``bins80`` and ``bins160`` (bin resolution). Each in its
  own subprocess (ours_s1.py).

Results (``cases/S1/results``): ours_<variant>_cstar<c>.npz, pycham_<variant>_cstar<c>.npz
(time series and size statistics at 60 s / 300 s, per-bin arrays every 30 min),
pycham_setup.json, run_info.json and copies of the PyCHAM model-variables files. Raw PyCHAM output stays in
``$PYCHAM_BENCH/outputs/S1_*``. Every run is bounded by a timeout; any failure raises.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

from tomas_jax.box.driver import code_version

from ... import pycham_inputs
from ...config import PYCHAM_SHA, PYCHAM_TIMEOUT_S, UMANSYSPROP_SHA
from . import case, ours_s1, pycham_s1, sizestats
from .case import S1

PYCHAM_VARIANTS = ("base", "tight", "nokelvin", "bins160")
OURS_VARIANTS = ("base", "micro10", "nokelvin", "nokelvin_pytransfer", "bins80", "bins160")
MASS_CLOSURE_TOL = 1e-6                    # relative, gas + particle surrogate vs P*t


def _json(path: Path, obj) -> None:
    def default(o):
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, (np.floating, np.integer)):
            return o.item()
        if isinstance(o, Path):
            return str(o)
        raise TypeError(type(o))
    path.write_text(json.dumps(obj, indent=2, default=default) + "\n")


def make_inputs() -> dict:
    """Scheme, xml, constants and one model-variables file per (c*, variant)."""
    in_dir = pycham_inputs.case_input_dir(case.CASE)
    scheme = pycham_s1.write_scheme(in_dir / "s1_scheme.eqn")
    xml = pycham_s1.write_xml(in_dir / "s1_species.xml")
    constants = pycham_inputs.write_constants(in_dir / "constants_mcm.f90")
    shutil.copyfile(scheme, case.RESULTS_DIR / "pycham_s1_scheme.eqn")
    shutil.copyfile(xml, case.RESULTS_DIR / "pycham_s1_species.xml")
    paths, setup = {}, {}
    for cstar in S1["cstar_ugm3"]:
        setup[f"cstar{cstar:g}"] = case.pycham_setup(case.scenario(cstar))
        for variant in PYCHAM_VARIANTS:
            s = case.scenario(cstar, nbins=ours_s1.variant_nbins(variant))
            kw = {}
            if variant == "tight":
                kw = {"update_step_s": S1["pycham_tight"]["update_step_s"],
                      "int_tol": S1["pycham_tight"]["int_tol"]}
            tag = f"{variant}_cstar{cstar:g}"
            mv = pycham_s1.model_var(s, res_file=pycham_inputs.case_output_dir(case.CASE, tag),
                                     scheme=scheme, xml=xml, constants=constants, **kw)
            p = pycham_inputs.write_model_var(in_dir / f"S1_{tag}_model_var.txt", mv)
            shutil.copyfile(p, case.RESULTS_DIR / f"pycham_model_var_{tag}.txt")
            paths[(cstar, variant)] = p
    _json(case.SETUP_JSON, setup)
    return {"model_var": paths, "setup": setup, "in_dir": in_dir}


def run_pycham_variant(model_var: Path, variant: str, tag: str) -> dict:
    root = pycham_inputs.ensure_pycham_root(case.CASE, kelvin_off=variant == "nokelvin")
    log = model_var.parent / f"pycham_{tag}.log"
    print(f"PyCHAM {tag}: root {root.name}, timeout {PYCHAM_TIMEOUT_S:.0f} s", flush=True)
    wall = pycham_inputs.run_pycham(model_var, root, log)
    print(f"  done in {wall:.1f} s", flush=True)
    return {"wall_s": wall, "root": str(root), "log": str(log),
            "output": str(pycham_inputs.case_output_dir(case.CASE, tag))}


def check_pycham(red: dict, setup: dict, s) -> dict:
    """PyCHAM read back the inputs as intended (raises otherwise)."""
    names = [str(n) for n in red["names"]]
    sur, poa = names.index(S1["surrogate_name"]), names.index(S1["poa_name"])
    h2o, core = names.index("H2O"), names.index("core")
    mw = red["mw_g_mol"]
    rho = mw / red["molar_volume_cm3_mol"]
    checks = {
        "mw_surr": float(mw[sur]), "mw_poa": float(mw[poa]), "mw_h2o": float(mw[h2o]),
        "mw_core": float(mw[core]),
        "rho_surr_poa_core_g_cm3": [float(rho[sur]), float(rho[poa]), float(rho[core])],
        "rho_h2o_g_cm3": float(rho[h2o]),
        "psat_surr_pa": float(red["psat_298_pa"][sur]),
    }
    want = {"mw_surr": S1["pycham_organic_mw"], "mw_poa": S1["pycham_organic_mw"],
            "mw_h2o": S1["pycham_water_mw"], "mw_core": S1["sulfate_mw"]}
    for k, v in want.items():
        if abs(checks[k] / v - 1.0) > 1e-9:
            raise RuntimeError(f"PyCHAM {k} = {checks[k]} != {v}")
    if max(abs(r / setup["rho_dry_g_cm3"] - 1.0) for r in checks["rho_surr_poa_core_g_cm3"]) > 1e-9:
        raise RuntimeError(f"PyCHAM densities {checks['rho_surr_poa_core_g_cm3']}")
    if abs(checks["psat_surr_pa"] / setup["volP_surr_pa"] - 1.0) > 1e-9:
        raise RuntimeError(f"PyCHAM volP {checks['psat_surr_pa']} != {setup['volP_surr_pa']}")
    # initial state: number and dry mass per bin as TOMAS-JAX (bins >= threshold)
    n0 = red["n_bin_cm3"][0]
    ref = setup["n_bin_cm3"]
    ok = ref > 0
    checks["initial_number_max_rel_diff"] = float(np.max(np.abs(n0[ok] / ref[ok] - 1.0)))
    xk, nk, mk, bv = case.initial_state(s)
    m_mid = np.sqrt(xk[:-1] * xk[1:])
    dry_ref = ref * m_mid * 1e15                      # ug/m3 per bin
    checks["initial_dry_mass_max_rel_diff"] = float(
        np.max(np.abs(red["dry_ugm3_bin"][0][ok] / dry_ref[ok] - 1.0)))
    if checks["initial_number_max_rel_diff"] > 1e-6 or checks["initial_dry_mass_max_rel_diff"] > 1e-6:
        raise RuntimeError(f"PyCHAM initial state differs from TOMAS-JAX: {checks}")
    checks["mass_closure_max_rel"] = mass_closure(red, s)
    return checks


def mass_closure(d: dict, s) -> float:
    """max |gas + particle surrogate - P t| / (P t_end) (both models must conserve)."""
    total = d["gas_surr_ugm3"] + d["soa_ugm3"]
    expect = s.surrogate.production_ugm3_per_h * d["t_s"] / 3600.0
    rel = float(np.max(np.abs(total - expect)) / expect[-1])
    if rel > MASS_CLOSURE_TOL:
        raise RuntimeError(f"surrogate mass not conserved: {rel:.3g}")
    return rel


def save_pycham(red: dict, path: Path) -> None:
    out = {k: red[k] for k in ("t_s", "gas_surr_ugm3", "soa_ugm3", "poa_ugm3", "sulfate_ugm3",
                               "water_ugm3", "names", "mw_g_mol", "molar_volume_cm3_mol")}
    out.update({k: red[k] for k in sizestats.PER_BIN_KEYS})
    np.savez_compressed(path, **sizestats.decimate_per_bin(out))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--skip-pycham", action="store_true", help="reuse the existing PyCHAM output")
    ap.add_argument("--skip-ours", action="store_true", help="reuse existing TOMAS-JAX results")
    args = ap.parse_args(argv)
    case.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    info = json.loads(case.RUN_INFO_JSON.read_text()) if case.RUN_INFO_JSON.exists() else {}
    inputs = make_inputs()
    info["pycham_version"] = {"sha": PYCHAM_SHA, "umansysprop_sha": UMANSYSPROP_SHA}

    for cstar in S1["cstar_ugm3"]:
        for variant in PYCHAM_VARIANTS:
            s = case.scenario(cstar, nbins=ours_s1.variant_nbins(variant))
            setup = case.pycham_setup(s)
            tag = f"{variant}_cstar{cstar:g}"
            if not args.skip_pycham:
                info[f"pycham_{tag}"] = run_pycham_variant(inputs["model_var"][(cstar, variant)],
                                                           variant, tag)
            red = pycham_s1.reduce_output(pycham_inputs.case_output_dir(case.CASE, tag), s)
            info.setdefault(f"pycham_{tag}", {})["checks"] = check_pycham(red, setup, s)
            save_pycham(red, case.pycham_npz(cstar, variant))
        if not args.skip_ours:
            for variant in OURS_VARIANTS:
                tag = f"{variant}_cstar{cstar:g}"
                print(f"TOMAS-JAX {tag}", flush=True)
                path = case.ours_npz(cstar, variant)
                res = ours_s1.run_in_subprocess(cstar, variant, path)
                if res["warnings"]:
                    raise RuntimeError(f"TOMAS-JAX {tag} warnings: {res['warnings']}")
                d = dict(np.load(path))
                res["mass_closure_max_rel"] = mass_closure(d, case.scenario(cstar))
                res["code_version"] = code_version()
                info[f"ours_{tag}"] = res
    _json(case.RUN_INFO_JSON, info)
    print(json.dumps({k: (v.get("wall_s", v) if isinstance(v, dict) else v)
                      for k, v in info.items()}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
