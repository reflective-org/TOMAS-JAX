"""Compare the two triangular-solve variants: result and time (best of 7)."""
import sys, json, time, subprocess, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
from spike import chem as C
from spike.sparse_lu import make_factor, make_solve, make_solve_v2
from bench_common import load_problem, PROBLEMS
def best(fn, *a, reps=7):
    jax.block_until_ready(fn(*a)); ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); jax.block_until_ready(fn(*a)); ts.append(time.perf_counter()-t0)
    return min(ts)
ap=argparse.ArgumentParser(); ap.add_argument("problem"); ap.add_argument("--tail", type=int, default=0); ap.add_argument("--order", default="markowitz"); a=ap.parse_args()
P=load_problem(a.problem); ch=C.build(P, dense_tail=a.tail, order=a.order); pl=ch.plan
chk=json.loads((PROBLEMS/f"checks_{a.problem}.json").read_text())
y=jnp.asarray(chk["y_test"]); k=jnp.asarray(chk["k_test"])
vals=jax.jit(ch.neg_jac_lu)(y,k)+0.2*ch.diag_mask
fac=jax.jit(make_factor(pl)); F=fac(vals)
s1=jax.jit(make_solve(pl))
prep, s2 = make_solve_v2(pl, pl.lrow, pl.urow_l, pl.flev, pl.blev)
prep=jax.jit(prep); s2=jax.jit(s2)
t0=time.perf_counter(); F2=jax.block_until_ready(prep(F)); tc_prep=time.perf_counter()-t0
t0=time.perf_counter(); x2=jax.block_until_ready(s2(F2,y)); tc2=time.perf_counter()-t0
x1=s1(F,y)
out={"problem":a.problem,"tail":a.tail,"order":a.order,"lu_nnz":pl.nnz,"levels":[len(pl.levels),len(pl.fwd),len(pl.bwd)],"t_rhs_ms":1e3*best(jax.jit(ch.rhs),y,k),"t_jac_ms":1e3*best(jax.jit(ch.neg_jac_lu),y,k),"maxrel_v2_vs_v1":float(jnp.max(jnp.abs(x2-x1)/jnp.maximum(jnp.abs(x1),1e-300))),
 "t_solve_v1_ms":1e3*best(s1,F,y),"t_solve_v2_ms":1e3*best(s2,F2,y),"t_prepare_ms":1e3*best(prep,F),
 "compile_v2_s":tc2,"t_factor_ms":1e3*best(fac,vals),
 "uptime":subprocess.run(["uptime"],capture_output=True,text=True,timeout=10).stdout.strip()}
print(json.dumps(out))
