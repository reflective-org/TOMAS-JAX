"""Per-step cost models and 24 h gas-solve estimates from results/scaling_*.json.

Dense Kvaerno3 (today):  c_d = a_d * (t_jacfwd + t_LU + 8 (t_solve + t_rhs))
Sparse Ros3 (Pro plan):  c_s = a_s * (t_jacvals + t_factor + 3 t_solve + 3 t_rhs)
a_d, a_s calibrate the sums to measured per-step costs at 725 species: 11.3 ms
(run_inorganic.py, Kvaerno3 in the model) and 0.39 ms (Pro plan 3.2: 3.13 s / 8029 steps).
Steps per 24 h are the Level 3 counts from the Pro plan (assumed independent of size).
"""
import json
from pathlib import Path

R = Path(__file__).parent / "results"
names = ["inorganic", "c3", "limonene", "copies2", "copies4", "copies7", "copies9"]
rs = {n: json.loads((R / f"scaling_{n}.json").read_text()) for n in names}
L = rs["limonene"]
sum_d = lambda r: r["t_jacfwd_ms"] + r["t_dense_lu_ms"] + 8 * (r["t_dense_solve_ms"] + r["t_rhs_ms"])  # noqa: E731
sum_s = lambda r: r["t_jacvals_ms"] + r["t_sparse_factor_ms"] + 3 * (r["t_sparse_solve_ms"] + r["t_rhs_ms"])  # noqa: E731
a_d, a_s = 11.3 / sum_d(L), 0.39 / sum_s(L)
STEPS = {"kvaerno3_dense": {"900s": 1061, "120s": 4855}, "ros3_sparse": {"900s": 1576, "120s": 8029}}
rows = []
for n in names:
    r = rs[n]
    cd, cs = a_d * sum_d(r), a_s * sum_s(r)
    rows.append({"name": n, "species": r["species"], "reactions": r["reactions"],
                 "ms_per_step_dense": cd, "ms_per_step_sparse": cs,
                 "gas_24h_s_dense_900": cd * STEPS["kvaerno3_dense"]["900s"] / 1e3,
                 "gas_24h_s_dense_120": cd * STEPS["kvaerno3_dense"]["120s"] / 1e3,
                 "gas_24h_s_sparse_900": cs * STEPS["ros3_sparse"]["900s"] / 1e3,
                 "gas_24h_s_sparse_120": cs * STEPS["ros3_sparse"]["120s"] / 1e3,
                 "speedup": cd / cs})
out = {"a_dense": a_d, "a_sparse": a_s, "steps": STEPS, "rows": rows}
(R / "runtime_estimates.json").write_text(json.dumps(out, indent=1))
print(f"a_d={a_d:.3f} a_s={a_s:.3f}")
for x in rows:
    print(f"{x['name']:9s} n={x['species']:5d} nr={x['reactions']:6d} dense {x['ms_per_step_dense']:8.2f} ms "
          f"sparse {x['ms_per_step_sparse']:6.3f} ms  x{x['speedup']:6.1f} | 24h dense 900/120 s: "
          f"{x['gas_24h_s_dense_900']:8.1f} {x['gas_24h_s_dense_120']:8.1f} | sparse: "
          f"{x['gas_24h_s_sparse_900']:6.2f} {x['gas_24h_s_sparse_120']:6.2f}")
