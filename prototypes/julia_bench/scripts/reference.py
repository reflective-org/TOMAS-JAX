"""Tight-tolerance reference solutions of the benchmark problems (scipy, analytic sparse
Jacobian).

    python reference.py <problem> <coupling_s> [--method BDF|Radau] [--rtol 1e-10]
                        [--atol 1e-4] [--tag ref] [--out-dir ../reference]

Writes <out-dir>/<tag>_<problem>_dt<coupling>.json (key species at every interval end,
RO2 at every interval start, solver statistics) and ..._hourly.npz (every species at
every full hour, the interval-end outputs at t = 1 h ... 24 h). With --rtol 1e-3
--atol 1e4 it is the scipy sparse baseline at the tutorial tolerances.
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import BDF, Radau

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import REFERENCE, NumpyChem, load_problem, run_intervals  # noqa: E402

METHODS = {"BDF": BDF, "Radau": Radau}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("problem")
    ap.add_argument("coupling", type=int)
    ap.add_argument("--method", default="BDF")
    ap.add_argument("--rtol", type=float, default=1e-10)
    ap.add_argument("--atol", type=float, default=1e-4)
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--max-wall-s", type=float, default=3600.0)
    ap.add_argument("--out-dir", default=str(REFERENCE))
    a = ap.parse_args()

    P = load_problem(a.problem)
    chem = NumpyChem(P.mech)
    stats = {"steps": 0, "nfev": 0, "njev": 0, "nlu": 0}
    t_start = time.perf_counter()
    load0 = os.getloadavg()

    def solve(y0, k, dt):
        if time.perf_counter() - t_start > a.max_wall_s:
            raise RuntimeError(f"reference run exceeded {a.max_wall_s} s")
        s = METHODS[a.method](lambda t, y: chem.rhs(y, k), 0.0, y0, dt, rtol=a.rtol,
                              atol=a.atol, jac=lambda t, y: chem.jac(y, k))
        while s.status == "running":
            msg = s.step()
            if s.status == "failed":
                raise RuntimeError(f"{a.method} failed: {msg}")
            stats["steps"] += 1
        stats["nfev"] += s.nfev
        stats["njev"] += s.njev
        stats["nlu"] += s.nlu
        return s.y

    out, ro2 = run_intervals(P, a.coupling, solve)
    wall = time.perf_counter() - t_start
    n_int = out.shape[0]
    per_hour = int(round(3600 / a.coupling))
    hourly_idx = np.arange(per_hour - 1, n_int, per_hour)
    idx = {s: P.mech.index(s) for s in P.key_species}
    meta = {
        "format": "tomas-gas-bench/reference/1",
        "problem": a.problem, "coupling_s": a.coupling,
        "solver": f"scipy.integrate.{a.method} (scipy {__import__('scipy').__version__}), "
                  "analytic sparse Jacobian (CSC, SuperLU)",
        "rtol": a.rtol, "atol": a.atol,
        "t_end_s": [(i + 1) * a.coupling for i in range(n_int)],
        "key_species": {s: out[:, i].tolist() for s, i in idx.items()},
        "ro2_start": ro2.tolist(),
        "hourly_t_s": [(int(i) + 1) * a.coupling for i in hourly_idx],
        "stats": {**stats, "wall_s": wall, "load_before": load0, "load_after": os.getloadavg(),
                  "peak_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20},
    }
    out_dir = Path(a.out_dir)
    out_dir.mkdir(exist_ok=True)
    base = out_dir / f"{a.tag}_{a.problem}_dt{a.coupling}"
    Path(f"{base}.json").write_text(json.dumps(meta, separators=(",", ":")))
    np.savez_compressed(f"{base}_hourly.npz", t_s=np.asarray(meta["hourly_t_s"], dtype=float),
                        y=out[hourly_idx], species=np.asarray(P.mech.species))
    print(json.dumps({"problem": a.problem, "coupling": a.coupling, "method": a.method,
                      "rtol": a.rtol, "atol": a.atol, **meta["stats"]}))


if __name__ == "__main__":
    main()
