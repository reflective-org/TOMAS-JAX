"""Time the JAX sparse Rosenbrock spike on a benchmark problem with the benchmark protocol
(bench/BENCHMARK.md section 7).

    python run_jax_sparse.py <problem> <coupling_s> [--method rodas3] [--rtol 1e-3]
        [--atol 1e4] [--h0 hairer|carry] [--mode loop|scan] [--tail 0] [--warm 3]
        [--max-wall-s 1200] [--tag jaxsparse]

One fresh process per call.
  mode loop: Python loop over the intervals, one jitted call per interval (as the dense run)
  mode scan: the whole 24 h (rate constants, solves, clipping, resets) is one jitted
             lax.scan; t_first_interval is then the compile time of that program
  h0 hairer: each interval starts with Hairer's initial-step rule (as diffrax dt0=None)
  h0 carry:  each interval starts with the step size the controller proposed at the end
             of the previous interval (allowed by the benchmark; reported)
Outputs (reference format) and timings go to ../results/<tag>_<problem>_dt<coupling>*.
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

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from jax import lax  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from spike import chem as C  # noqa: E402
from spike.rosenbrock import TABLEAUS, make_integrator  # noqa: E402
from spike.sparse_lu import make_factor, make_factor_v2, make_solve, make_solve_v2  # noqa: E402

from bench_common import REFERENCE, load_problem  # noqa: E402  (path set by spike.chem)
from compare import score  # noqa: E402

RESULTS = HERE.parent / "results"


class WallCap(Exception):
    pass


def uptime() -> str:
    return subprocess.run(["uptime"], capture_output=True, text=True, timeout=10).stdout.strip()


def build(P, a):
    t0 = time.perf_counter()
    ch = C.build(P, dense_tail=a.tail, order=a.order)
    t_sym = time.perf_counter() - t0
    pl = ch.plan
    if a.solve == "v1":
        factor, solve = make_factor(pl), make_solve(pl)
    else:
        prepare, solve = make_solve_v2(pl, pl.lrow, pl.urow_l, pl.flev, pl.blev)
        factor1 = make_factor_v2(pl) if (a.factor == "v2" and a.tail == 0) else make_factor(pl)

        def factor(vals):
            return prepare(factor1(vals))
    integ = make_integrator(ch.rhs, ch.neg_jac_lu, ch.diag_mask, factor, solve,
                            TABLEAUS[a.method], a.rtol, a.atol, max_steps=a.max_steps,
                            err_norm=a.norm)
    dt = float(a.coupling)
    carry = a.h0 == "carry"

    def interval(y_start, j_row, h_in):
        ro2 = jnp.sum(y_start[jnp.asarray(P.mech.ro2)])
        k = ch.rate_constants(y_start, j_row)
        y_end, h_carry, stats, ok = integ(y_start, k, dt,
                                          h_in * a.carry_scale if carry else -1.0)
        y_end = jnp.maximum(y_end, 0.0)
        return y_end, ch.post_interval(y_end), ro2, h_carry, stats, ok

    def day(y0, jtab):
        def body(c, j_row):
            y, h = c
            y_end, y_next, ro2, h_c, stats, ok = interval(y, j_row, h)
            return (y_next, h_c), (y_end, ro2, stats, ok)
        _, out = lax.scan(body, (y0, jnp.asarray(-1.0)), jtab)
        return out

    return ch, t_sym, jax.jit(interval), jax.jit(day)


def run_loop(P, interval_j, jtab, y0, max_wall):
    n_int = jtab.shape[0]
    t_start = time.perf_counter()
    y, h = y0, jnp.asarray(-1.0)
    outs, ro2s, stats, oks = [], [], [], []
    t_first_interval = None
    for i in range(n_int):
        y_end, y, ro2, h, st, ok = interval_j(y, jtab[i], h)
        jax.block_until_ready(y)
        if i == 0:
            t_first_interval = time.perf_counter() - t_start
        outs.append(y_end)
        ro2s.append(ro2)
        stats.append(st)
        oks.append(ok)
        el = time.perf_counter() - t_start
        if el > max_wall:
            raise WallCap({"intervals_done": i + 1, "n_intervals": n_int, "wall_s": el,
                           "t_first_interval_s": t_first_interval,
                           "extrapolated_24h_s": el / (i + 1) * n_int})
    out = np.asarray(jnp.stack(outs))
    res = (out, np.asarray(jnp.stack(ro2s)), np.asarray(jnp.stack(stats)).sum(0),
           bool(np.all(np.asarray(jnp.stack(oks)))))
    return time.perf_counter() - t_start, t_first_interval, res


def run_scan(day_j, jtab, y0):
    t_start = time.perf_counter()
    out, ro2, stats, ok = jax.block_until_ready(day_j(y0, jtab))
    res = (np.asarray(out), np.asarray(ro2), np.asarray(stats).sum(0), bool(np.all(ok)))
    return time.perf_counter() - t_start, None, res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("coupling", type=int)
    ap.add_argument("--method", default="ros3", choices=sorted(TABLEAUS))
    ap.add_argument("--rtol", type=float, default=None)
    ap.add_argument("--atol", type=float, default=None)
    ap.add_argument("--h0", default="hairer", choices=["hairer", "carry"])
    ap.add_argument("--mode", default="loop", choices=["loop", "scan"])
    ap.add_argument("--tail", type=int, default=0)
    ap.add_argument("--order", default="markowitz", choices=["markowitz", "mindeg"])
    ap.add_argument("--solve", default="v2", choices=["v1", "v2"])
    ap.add_argument("--factor", default="v2", choices=["v1", "v2"])
    ap.add_argument("--norm", default="rms", choices=["rms", "max"])
    ap.add_argument("--carry-scale", type=float, default=1.0,
                    help="with --h0 carry: start each interval at this fraction of the carried step")
    ap.add_argument("--max-steps", type=int, default=2000)
    ap.add_argument("--warm", type=int, default=3)
    ap.add_argument("--max-wall-s", type=float, default=1200.0)
    ap.add_argument("--tag", default="jaxsparse")
    a = ap.parse_args()

    P = load_problem(a.problem)
    a.rtol = P.rtol if a.rtol is None else a.rtol
    a.atol = P.atol if a.atol is None else a.atol
    ch, t_sym, interval_j, day_j = build(P, a)
    jtab = jnp.asarray(P.j_table(a.coupling))
    y0 = jnp.asarray(P.y0)
    jax.block_until_ready((jtab, y0, ch.diag_mask))
    t_setup = time.perf_counter() - T0
    pl = ch.plan
    res = {"problem": a.problem, "coupling_s": a.coupling, "n_species": P.mech.n,
           "n_reactions": P.mech.nr,
           "solver": f"JAX sparse spike: {a.method} (KPP coefficients and step control), "
                     f"analytic sparse Jacobian, levelised sparse LU ({a.order} order, "
                     f"dense tail {a.tail}, factor {a.factor}, solve {a.solve}), {a.norm} error norm, rtol {a.rtol}, atol {a.atol}, "
                     f"max_steps {a.max_steps}, h0 {a.h0}, mode {a.mode}",
           "settings": {"method": a.method, "rtol": a.rtol, "atol": a.atol, "h0": a.h0,
                        "mode": a.mode, "tail": a.tail, "order": a.order, "factor": a.factor, "solve": a.solve, "norm": a.norm, "carry_scale": a.carry_scale,
                        "max_steps": a.max_steps},
           "lu": {"jac_nnz": int(ch.numpy_chem.nnz), "lu_nnz": int(pl.nnz),
                  "factor_levels": len(pl.levels), "fwd_groups": len(pl.fwd),
                  "bwd_groups": len(pl.bwd), "flops": int(pl.flops)},
           "versions": {"jax": jax.__version__, "python": platform.python_version()},
           "machine": platform.platform(), "XLA_FLAGS": os.environ.get("XLA_FLAGS", ""),
           "t_symbolic_s": t_sym, "t_setup_s": t_setup, "runs": []}

    def one(record_cap=True):
        if a.mode == "loop":
            return run_loop(P, interval_j, jtab, y0, a.max_wall_s)
        return run_scan(day_j, jtab, y0)

    up = uptime()
    if a.mode == "scan":
        t0 = time.perf_counter()
        day_j.lower(y0, jtab).compile()
        res["t_compile_s"] = time.perf_counter() - t0
    try:
        wall, tfi, (out, ro2, stats, ok) = one()
    except WallCap as cap:
        res.update({"status": "stopped at wall cap", "cap_s": a.max_wall_s, "partial": cap.args[0],
                    "uptime_before": up, "uptime_after": uptime(),
                    "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20})
        RESULTS.mkdir(exist_ok=True)
        (RESULTS / f"{a.tag}_{a.problem}_dt{a.coupling}_timing.json").write_text(
            json.dumps(res, indent=1))
        print(json.dumps(res))
        return
    if a.mode == "scan":
        wall += res["t_compile_s"]
        tfi = res["t_compile_s"]
    names = ["accepted", "rejected", "rhs_evals", "jac_evals", "lu_factorisations"]
    work = {k: int(v) for k, v in zip(names, stats)}
    res.update({"status": "complete" if ok else "FAILED (max_steps or no progress)",
                "t_first_s": wall, "t_first_interval_s": tfi,
                "accepted_steps": work["accepted"], "work": work})
    res["runs"].append({"kind": "first (includes compile)", "wall_s": wall,
                        "uptime_before": up, "uptime_after": uptime()})

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

    for _ in range(a.warm):
        up = uptime()
        try:
            wall, _, (_, _, st_w, _) = one()
        except WallCap as cap:
            res["runs"].append({"kind": "warm", "status": "stopped at wall cap",
                                "partial": cap.args[0], "uptime_before": up})
            break
        assert int(st_w[0]) == work["accepted"], (st_w, work)
        res["runs"].append({"kind": "warm", "wall_s": wall, "uptime_before": up,
                            "uptime_after": uptime()})
    warm = [r["wall_s"] for r in res["runs"] if r["kind"] == "warm" and "wall_s" in r]
    if warm:
        res["t_warm_median_s"] = float(np.median(warm))
        res["t_warm_min_s"] = float(np.min(warm))
        res["ms_per_step_warm_median"] = 1e3 * res["t_warm_median_s"] / work["accepted"]
    res["compile_estimate_s"] = res["t_first_s"] - (res.get("t_warm_median_s") or 0.0)
    res["peak_rss_mb"] = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
    (RESULTS / f"{a.tag}_{a.problem}_dt{a.coupling}_timing.json").write_text(
        json.dumps(res, indent=1))
    short = {k: res.get(k) for k in ("problem", "coupling_s", "status", "t_setup_s",
                                     "t_symbolic_s", "t_first_s", "t_first_interval_s",
                                     "t_warm_median_s", "accepted_steps",
                                     "ms_per_step_warm_median", "peak_rss_mb")}
    short["work"] = work
    short["warm"] = [round(w, 3) for w in warm]
    if "accuracy" in res:
        short["key_max_err"] = res["accuracy"]["key"]["max"]
        short["key_worst"] = res["accuracy"]["key"]["worst_species"]
        short["all_species_hourly_max"] = res["accuracy"].get("all_species_hourly_max")
    short["settings"] = res["settings"]
    print(json.dumps(short))


if __name__ == "__main__":
    main()
