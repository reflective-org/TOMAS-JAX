"""TOMAS-JAX runs for case S1 (Level 1 surrogate, no chemistry).

The harness ``ours.run_ours`` requires chemistry; this records a Level 1 run through the
same opt-in ``driver.run(observer=...)`` hook: at t = 0 and after every coupling
interval the surrogate gas, particle mass per component and per bin, number per bin and
each bin's wet diameter (calc_wet_diameters, the size the SOA step uses).

Variants (benchmark-side monkeypatches for numerical / attribution runs only; the model
code is untouched). Each run executes in its own subprocess (``python -m ...ours_s1``)
so a patch can never leak into a cached jit trace of another run:

* ``base``      - the model as is
* ``micro10``   - aerosol micro-step 10 s instead of 60 s (numerical sensitivity)
* ``nokelvin``  - vbs-jax surface tension 0 -> Kelvin factor 1 (attribution)
* ``nokelvin_pytransfer`` - as nokelvin, plus PyCHAM's gas-to-particle transfer
  coefficient (transfer.pycham_sink_patch) in place of vbs-jax's (attribution)
* ``bins80`` / ``bins160`` - the model as is on 80 / 160 bins (scenario nbins; resolution)

    .venv/bin/python -m benchmarks.pycham.cases.S1.ours_s1 --cstar 1 --variant base --out f.npz
"""
from __future__ import annotations

import argparse
import functools
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ...config import OURS_TIMEOUT_S, TOMAS_JAX_ROOT
from ...particles import dry_diameter_nm
from . import case, sizestats

VARIANTS = {"base": (), "micro10": ("micro10",), "nokelvin": ("nokelvin",),
            "nokelvin_pytransfer": ("nokelvin", "pytransfer"), "bins80": (), "bins160": ()}


def variant_nbins(variant: str) -> int | None:
    """Bin count of a resolution variant (``bins<N>``), None for the preset."""
    return int(variant[4:]) if variant.startswith("bins") else None


def _patch(variant: str) -> None:
    from tomas_jax.box import driver
    from tomas_jax.physics import soa_partitioning
    import vbs_jax.partitioning
    from .transfer import pycham_sink_patch
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant!r}; valid: {list(VARIANTS)}")
    for patch in VARIANTS[variant]:
        if patch == "micro10":
            driver.aerosol_outer_step = functools.partial(
                driver.aerosol_outer_step, micro_dt=case.S1["ours_micro_dt_sensitivity_s"])
        elif patch == "nokelvin":
            soa_partitioning.DEFAULT_VBS_CONFIG = (
                soa_partitioning.DEFAULT_VBS_CONFIG._replace(sigma=0.0))
        elif patch == "pytransfer":
            vbs_jax.partitioning.organic_condensation_sink = pycham_sink_patch


def run_s1(cstar: float, variant: str = "base", timeout_s: float = OURS_TIMEOUT_S) -> dict:
    """One TOMAS-JAX S1 run (in this process) -> dict of numpy arrays."""
    import jax.numpy as jnp
    from tomas_jax.box import driver
    from tomas_jax.box.coupling import kg_cell_to_ugm3, mass_grid
    from tomas_jax.box.config import AerosolStepConfig
    from tomas_jax.core.config import SRTH2O, SRTORG_NUC, SRTSO4, SRTSOA1
    from tomas_jax.physics.condensation_sink import calc_wet_diameters

    _patch(variant)
    s = case.scenario(cstar, nbins=variant_nbins(variant))
    snaps: list[tuple[float, dict]] = []
    deadline = time.monotonic() + timeout_s
    t_wall = time.perf_counter()
    result = driver.run(s, cancel=lambda: time.monotonic() > deadline,
                        observer=lambda t, snap: snaps.append((t, snap)))
    wall = time.perf_counter() - t_wall
    boxvol = AerosolStepConfig().boxvol_cm3
    xk = mass_grid(s.numerics.nbins)
    nk = np.array([sn["Nk"] for _, sn in snaps])                     # [t, bin] #/cell
    mk = np.array([sn["Mk"] for _, sn in snaps])                     # [t, bin, comp] kg/cell
    gc = np.array([sn["Gc"] for _, sn in snaps])
    dp_wet = np.array([np.asarray(calc_wet_diameters(jnp.asarray(n), jnp.asarray(m),
                                                     jnp.asarray(xk))[0])
                       for n, m in zip(nk, mk)])                     # [t, bin] m
    ug = functools.partial(kg_cell_to_ugm3, boxvol_cm3=boxvol)
    dry_bin = ug(mk[:, :, :SRTH2O].sum(axis=2))
    n_cm3 = nk / boxvol
    with np.errstate(divide="ignore", invalid="ignore"):
        m_dry_pp = np.where(n_cm3 > 0, dry_bin / n_cm3, np.nan) * 1e-12   # g per particle
    d_dry = dry_diameter_nm(m_dry_pp * 1e-3)
    return {
        "t_s": np.array([t for t, _ in snaps]),
        "gas_surr_ugm3": ug(gc[:, SRTSOA1]),
        "soa_ugm3": ug(mk[:, :, SRTSOA1].sum(axis=1)),
        "poa_ugm3": ug(mk[:, :, SRTORG_NUC].sum(axis=1)),
        "sulfate_ugm3": ug(mk[:, :, SRTSO4].sum(axis=1)),
        "water_ugm3": ug(mk[:, :, SRTH2O].sum(axis=1)),
        "n_bin_cm3": n_cm3,
        "dry_ugm3_bin": dry_bin,
        "soa_ugm3_bin": ug(mk[:, :, SRTSOA1]),
        "poa_ugm3_bin": ug(mk[:, :, SRTORG_NUC]),
        "water_ugm3_bin": ug(mk[:, :, SRTH2O]),
        "d_dry_nm_bin": d_dry,
        "d_wet_nm_bin": dp_wet * 1e9,
        "populated_bin": nk > _neps(),
        "xk_kg": xk,
        "runtime_s": np.float64(wall),
        "driver_runtime_s": np.float64(result.diagnostics["runtime_s"]),
        "warnings": np.array(result.diagnostics["warnings"], dtype=str),
        "cstar_source": np.array(result.diagnostics["cstar_source"]),
    }


def _neps() -> float:
    from tomas_jax.physics.condensation_sink import NEPS_CONDSINK
    return NEPS_CONDSINK


def save(path: Path, data: dict) -> None:
    """Size statistics at full resolution, per-bin arrays on sizestats.PER_BIN_EVERY_S."""
    arrays = {}
    for k, v in sizestats.decimate_per_bin(data).items():
        v = np.asarray(v)
        if v.dtype == np.float64 and v.ndim >= 2:
            v = v.astype(np.float32)
        arrays[k] = v
    arrays.pop("populated_bin")
    np.savez_compressed(path, **arrays)


def run_in_subprocess(cstar: float, variant: str, out: Path,
                      timeout_s: float = OURS_TIMEOUT_S) -> dict:
    """Run one variant in a fresh interpreter (bounded) and return its run info."""
    cmd = [sys.executable, "-m", "benchmarks.pycham.cases.S1.ours_s1", "--cstar", f"{cstar:g}",
           "--variant", variant, "--out", str(out), "--timeout", f"{timeout_s:g}"]
    proc = subprocess.run(cmd, cwd=TOMAS_JAX_ROOT, capture_output=True, text=True,
                          timeout=timeout_s + 120.0)
    if proc.returncode != 0:
        raise RuntimeError(f"TOMAS-JAX S1 run failed (exit {proc.returncode}):\n"
                           f"{proc.stdout[-2000:]}\n{proc.stderr[-4000:]}")
    return json.loads(proc.stdout.strip().splitlines()[-1])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="one TOMAS-JAX S1 run")
    ap.add_argument("--cstar", type=float, required=True)
    ap.add_argument("--variant", choices=list(VARIANTS), default="base")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--timeout", type=float, default=OURS_TIMEOUT_S)
    args = ap.parse_args(argv)
    data = run_s1(args.cstar, args.variant, args.timeout)
    save(args.out, data)
    print(json.dumps({"cstar": args.cstar, "variant": args.variant,
                      "wall_s": float(data["runtime_s"]),
                      "warnings": [str(w) for w in data["warnings"]],
                      "soa_end_ugm3": float(data["soa_ugm3"][-1]),
                      "gas_end_ugm3": float(data["gas_surr_ugm3"][-1])}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
