"""Correctness and cost of the pieces of one Rosenbrock step, per problem and elimination order.

    python test_pieces.py <problem> [--order markowitz|mindeg]

Checks against the benchmark's unit-test vectors (checks_<problem>.json): f(y_test) and
J(y_test) v; checks the sparse LU solve of (c I - J) x = b by its residual and against
scipy's SuperLU. Times each piece jitted on its own (best of 9) and prints one JSON line.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax  # noqa: E402

jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import scipy.sparse as sp  # noqa: E402
import scipy.sparse.linalg as spla  # noqa: E402

from spike import chem as C  # noqa: E402
from spike.sparse_lu import make_factor_v2, make_solve_v2  # noqa: E402

from bench_common import PROBLEMS, load_problem  # noqa: E402


def best(fn, *a, reps=9):
    jax.block_until_ready(fn(*a))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*a))
        ts.append(time.perf_counter() - t0)
    return min(ts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("--order", default="markowitz")
    a = ap.parse_args()
    P = load_problem(a.problem)
    t0 = time.perf_counter()
    ch = C.build(P, order=a.order)
    t_sym = time.perf_counter() - t0
    pl = ch.plan
    chk = json.loads((PROBLEMS / f"checks_{a.problem}.json").read_text())
    y, v, k = (np.asarray(chk[q]) for q in ("y_test", "v", "k_test"))
    yj, kj = jnp.asarray(y), jnp.asarray(k)
    rhs, jac = jax.jit(ch.rhs), jax.jit(ch.neg_jac_lu)
    f = np.asarray(rhs(yj, kj))
    fr = np.asarray(chk["f_test"])
    nj = np.asarray(jac(yj, kj))
    inv = {q: e for e, q in pl.pos.items()}
    ij = np.array([inv[q] for q in range(pl.nnz)])
    Jm = sp.csr_matrix((-nj, (pl.perm[ij[:, 0]], pl.perm[ij[:, 1]])), shape=(ch.n, ch.n))
    jvr = np.asarray(chk["jac_v_test"])
    c = 1.0 / (10.0 * 0.4358665215)             # 1/(h gamma) for h = 10 s, Ros3 gamma
    vals = jnp.asarray(nj) + c * ch.diag_mask
    prepare, solve = make_solve_v2(pl, pl.lrow, pl.urow_l, pl.flev, pl.blev)
    fac = jax.jit(make_factor_v2(pl))
    t0 = time.perf_counter()
    F = jax.block_until_ready(fac(vals))
    t_cf = time.perf_counter() - t0
    prep, sol = jax.jit(prepare), jax.jit(solve)
    F2 = prep(F)
    t0 = time.perf_counter()
    x = np.asarray(jax.block_until_ready(sol(F2, yj)))
    t_cs = time.perf_counter() - t0
    A = (c * sp.identity(ch.n) - Jm).tocsc()
    xr = spla.spsolve(A, y)
    t0 = time.perf_counter()
    for _ in range(5):
        lu = spla.splu(A)
    t_splu = (time.perf_counter() - t0) / 5
    t0 = time.perf_counter()
    for _ in range(20):
        lu.solve(y)
    t_spsv = (time.perf_counter() - t0) / 20
    out = {
        "problem": a.problem, "order": a.order, "n": ch.n, "t_symbolic_s": t_sym,
        "jac_nnz": int(ch.numpy_chem.nnz), "lu_nnz": int(pl.nnz),
        "lu_updates": int(sum(L.u_l.size for L in pl.levels)),
        "factor_levels": len(pl.levels), "fwd_levels": len(pl.fwd), "bwd_levels": len(pl.bwd),
        "rhs_max_err_rel_to_max": float(np.abs(f - fr).max() / np.abs(fr).max()),
        "jv_max_err_rel_to_max": float(np.abs(Jm @ v - jvr).max() / np.abs(jvr).max()),
        "solve_residual_rel": float(np.abs(A @ x - y).max() / np.abs(y).max()),
        "solve_vs_superlu_max_rel": float(np.max(np.abs(x - xr) / np.maximum(np.abs(xr), 1e-30))),
        "compile_factor_s": t_cf, "compile_solve_s": t_cs,
        "t_rhs_ms": 1e3 * best(rhs, yj, kj), "t_jac_ms": 1e3 * best(jac, yj, kj),
        "t_factor_ms": 1e3 * best(fac, vals), "t_prepare_ms": 1e3 * best(prep, F),
        "t_solve_ms": 1e3 * best(sol, F2, yj),
        "t_superlu_factor_ms": 1e3 * t_splu, "t_superlu_solve_ms": 1e3 * t_spsv,
        "uptime": subprocess.run(["uptime"], capture_output=True, text=True,
                                 timeout=10).stdout.strip()}
    out["ros3_step_model_ms"] = (2 * out["t_rhs_ms"] + out["t_jac_ms"] + out["t_factor_ms"]
                                 + out["t_prepare_ms"] + 3 * out["t_solve_ms"])
    print(json.dumps(out))


if __name__ == "__main__":
    main()
