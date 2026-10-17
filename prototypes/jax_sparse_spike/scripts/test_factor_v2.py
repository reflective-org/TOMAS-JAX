"""Factor v2 (unscaled L, one scatter per level) against v1: values and time."""
import sys, json, time, subprocess, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from spike import chem as C
from spike.sparse_lu import make_factor, make_factor_v2
from bench_common import load_problem, PROBLEMS
def best(fn, *a, reps=9):
    jax.block_until_ready(fn(*a)); ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); jax.block_until_ready(fn(*a)); ts.append(time.perf_counter()-t0)
    return min(ts)
ap=argparse.ArgumentParser(); ap.add_argument("problem"); a=ap.parse_args()
P=load_problem(a.problem); ch=C.build(P); pl=ch.plan
chk=json.loads((PROBLEMS/f"checks_{a.problem}.json").read_text())
y=jnp.asarray(chk["y_test"]); k=jnp.asarray(chk["k_test"])
vals=jax.jit(ch.neg_jac_lu)(y,k)+0.2*ch.diag_mask
f1=jax.jit(make_factor(pl)); f2=jax.jit(make_factor_v2(pl, True)); f3=jax.jit(make_factor_v2(pl, False))
v1=f1(vals)[0]; v2=f2(vals)[0]; v3=f3(vals)[0]
rel=lambda a,b: float(jnp.max(jnp.abs(a-b)/jnp.maximum(jnp.abs(b),1e-300)))
print(json.dumps({"problem":a.problem,"v2dup_vs_v1":rel(v2,v1),"v2seg_vs_v1":rel(v3,v1),
  "t_v1_ms":1e3*best(f1,vals),"t_v2dup_ms":1e3*best(f2,vals),"t_v2seg_ms":1e3*best(f3,vals),
  "uptime":subprocess.run(["uptime"],capture_output=True,text=True,timeout=10).stdout.strip()}))
