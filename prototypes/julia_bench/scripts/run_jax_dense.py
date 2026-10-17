"""Time the current TOMAS-JAX gas solver (dense Kvaerno3, tomas_jax.gaschem.solver.
integrate_interval) on a benchmark problem with the benchmark protocol.

    python run_jax_dense.py <problem> <coupling_s> [--warm 3] [--max-wall-s 1200]

One fresh process per call. Measures
  t_setup     script start -> imports done, problem loaded, mechanism arrays on device
  t_first     first 24 h run, including JIT compilation (first-call latency)
  t_first_interval  compilation + the first interval alone
  t_warm[]    each further 24 h run (same process, compiled)
  peak RSS of the process, load average before and after every run, solver steps.
The interval-end outputs of the first run are written in the reference format to
../results/jaxdense_<problem>_dt<coupling>.json (+ _hourly.npz) and scored.
If a run passes --max-wall-s it is stopped; the partial result is reported with the
extrapolated 24 h time.
"""
from __future__ import annotations

import time

T0 = time.perf_counter()

import argparse  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import resource  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402

import tomas_jax  # noqa: E402,F401  (float64 config)
from tomas_jax.gaschem.config import SOLVER, SOLVER_MAX_STEPS  # noqa: E402
from tomas_jax.gaschem.mechanism import Mechanism  # noqa: E402
from tomas_jax.gaschem.solver import integrate_interval  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import REFERENCE, RESULTS, load_problem  # noqa: E402
from compare import score  # noqa: E402


class WallCap(Exception):
    pass


def uptime() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True, timeout=10).stdout.strip()


def jax_mechanism(P) -> Mechanism:
    m = P.mech
    return Mechanism(
        n_species=m.n,
        reactant_idx=jnp.asarray(m.reactants, dtype=jnp.int32),
        prod_rxn=jnp.asarray(m.prod_rxn, dtype=jnp.int32),
        prod_sp=jnp.asarray(m.prod_sp, dtype=jnp.int32),
        prod_coef=jnp.asarray(m.prod_coef, dtype=jnp.float64),
        loss_rxn=jnp.asarray(m.loss_rxn, dtype=jnp.int32),
        loss_sp=jnp.asarray(m.loss_sp, dtype=jnp.int32),
        ro2_idx=jnp.asarray(m.ro2, dtype=jnp.int32))


def one_run(P, mech, coupling, max_wall, record=False):
    jtab = P.j_table(coupling)
    n_int = P.n_intervals(coupling)
    y = P.y0.copy()
    out = np.zeros((n_int, P.mech.n)) if record else None
    ro2 = np.zeros(n_int)
    steps = 0
    t_start = time.perf_counter()
    t_first_interval = None
    for i in range(n_int):
        ro2[i] = y[P.mech.ro2].sum()
        k = P.k(y, jtab[i])
        y_new, nsteps = integrate_interval(mech, jnp.asarray(y), jnp.asarray(k), float(coupling),
                                           rtol=P.rtol, atol=P.atol)
        steps += int(nsteps)
        y_end = np.maximum(np.asarray(y_new), 0.0)
        if i == 0:
            t_first_interval = time.perf_counter() - t_start
        if record:
            out[i] = y_end
        y = P.post_interval(y_end.copy())
        el = time.perf_counter() - t_start
        if el > max_wall:
            raise WallCap({"intervals_done": i + 1, "n_intervals": n_int, "wall_s": el,
                           "steps": steps, "t_first_interval_s": t_first_interval,
                           "extrapolated_24h_s": el / (i + 1) * n_int})
    return time.perf_counter() - t_start, t_first_interval, steps, out, ro2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("coupling", type=int)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--max-wall-s", type=float, default=1200.0)
    ap.add_argument("--tag", default="jaxdense")
    a = ap.parse_args()

    P = load_problem(a.problem)
    mech = jax_mechanism(P)
    jax.block_until_ready(mech.reactant_idx)
    t_setup = time.perf_counter() - T0
    res = {"problem": a.problem, "coupling_s": a.coupling, "n_species": P.mech.n,
           "n_reactions": P.mech.nr,
           "solver": f"TOMAS-JAX integrate_interval: diffrax {SOLVER}, dense autodiff Jacobian, "
                     f"rtol {P.rtol}, atol {P.atol}, max_steps {SOLVER_MAX_STEPS}",
           "versions": {"jax": jax.__version__,
                        "diffrax": __import__("diffrax").__version__,
                        "python": platform.python_version()},
           "machine": platform.platform(), "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
           "t_setup_s": t_setup, "runs": []}
    up = uptime()
    try:
        wall, tfi, steps, out, ro2 = one_run(P, mech, a.coupling, a.max_wall_s, record=True)
    except WallCap as cap:
        res.update({"status": "stopped at wall cap", "cap_s": a.max_wall_s, "partial": cap.args[0],
                    "uptime_before": up, "uptime_after": uptime(),
                    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20})
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{a.tag}_{a.problem}_dt{a.coupling}_timing.json").write_text(
            json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    res.update({"status": "complete", "t_first_s": wall, "t_first_interval_s": tfi,
                "accepted_steps": steps})
    res["runs"].append({"kind": "first (includes compile)", "wall_s": wall,
                        "uptime_before": up, "uptime_after": uptime()})

    # outputs in the reference format, then scored
    n_int = out.shape[0]
    per_hour = int(round(3600 / a.coupling))
    hourly_idx = np.arange(per_hour - 1, n_int, per_hour)
    run_file = {"format": "tomas-gas-bench/reference/1", "problem": a.problem,
                "coupling_s": a.coupling, "solver": res["solver"],
                "t_end_s": [(i + 1) * a.coupling for i in range(n_int)],
                "key_species": {s: out[:, P.mech.index(s)].tolist() for s in P.key_species},
                "ro2_start": ro2.tolist(),
                "hourly_t_s": [(int(i) + 1) * a.coupling for i in hourly_idx]}
    RESULTS.mkdir(exist_ok=True)
    base = RESULTS / f"{a.tag}_{a.problem}_dt{a.coupling}"
    Path(f"{base}.json").write_text(json.dumps(run_file, separators=(",", ":")))
    np.savez_compressed(f"{base}_hourly.npz", t_s=np.asarray(run_file["hourly_t_s"], float),
                        y=out[hourly_idx], species=np.asarray(P.mech.species))
    ref = REFERENCE / f"ref_{a.problem}_dt{a.coupling}.json"
    if ref.exists():
        res["accuracy"] = score(Path(f"{base}.json"), ref)

    for r in range(a.warm):
        up = uptime()
        try:
            wall, _, steps_w, _, _ = one_run(P, mech, a.coupling, a.max_wall_s)
        except WallCap as cap:
            res["runs"].append({"kind": "warm", "status": "stopped at wall cap",
                                "partial": cap.args[0], "uptime_before": up})
            break
        assert steps_w == steps, (steps_w, steps)
        res["runs"].append({"kind": "warm", "wall_s": wall, "uptime_before": up,
                            "uptime_after": uptime()})
    warm = [r["wall_s"] for r in res["runs"] if r["kind"] == "warm" and "wall_s" in r]
    if warm:
        res["t_warm_median_s"] = float(np.median(warm))
        res["t_warm_min_s"] = float(np.min(warm))
        res["ms_per_step_warm_median"] = 1e3 * res["t_warm_median_s"] / steps
    res["compile_estimate_s"] = res["t_first_s"] - (res.get("t_warm_median_s") or 0.0)
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    (RESULTS / f"{a.tag}_{a.problem}_dt{a.coupling}_timing.json").write_text(
        json.dumps(res, indent=1))
    short = {k: res.get(k) for k in ("problem", "coupling_s", "t_setup_s", "t_first_s",
                                     "t_first_interval_s", "t_warm_median_s", "accepted_steps",
                                     "ms_per_step_warm_median", "peak_rss_mb")}
    if "accuracy" in res:
        short["key_max_err"] = res["accuracy"]["key"]["max"]
        short["key_worst"] = res["accuracy"]["key"]["worst_species"]
        short["all_species_hourly_max"] = res["accuracy"].get("all_species_hourly_max")
    print(json.dumps(short))


if __name__ == "__main__":
    main()
