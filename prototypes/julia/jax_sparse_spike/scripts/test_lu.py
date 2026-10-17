"""Check the JAX pieces against the benchmark's unit-test vectors and the sparse LU against
scipy; time each piece (best of 5, jitted separately).

    python test_lu.py <problem> [--tail 0] [--perm mindeg|superlu]
"""
from __future__ import annotations
import argparse, json, sys, time, subprocess
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from spike import chem as C
from bench_common import load_problem, PROBLEMS

def best(fn, *a, reps=5):
    jax.block_until_ready(fn(*a)); ts=[]
    for _ in range(reps):
        t0=time.perf_counter(); jax.block_until_ready(fn(*a)); ts.append(time.perf_counter()-t0)
    return min(ts)

ap=argparse.ArgumentParser(); ap.add_argument("problem"); ap.add_argument("--tail", type=int, default=0)
a=ap.parse_args()
P=load_problem(a.problem)
t0=time.perf_counter(); ch=C.build(P, dense_tail=a.tail); t_sym=time.perf_counter()-t0
pl=ch.plan
res={"problem":a.problem,"n":ch.n,"tail":a.tail,"t_symbolic_s":t_sym,"jac_nnz":ch.numpy_chem.nnz,
     "lu_nnz":pl.nnz,"fill":pl.nnz/(ch.numpy_chem.nnz + 0),"flops":pl.flops,"factor_levels":len(pl.levels),
     "fwd_groups":len(pl.fwd),"bwd_groups":len(pl.bwd)}
chk=json.loads((PROBLEMS/f"checks_{a.problem}.json").read_text())
y=np.asarray(chk["y_test"]); v=np.asarray(chk["v"]); k=np.asarray(chk["k_test"])
f=np.asarray(jax.jit(ch.rhs)(jnp.asarray(y), jnp.asarray(k)))
fr=np.asarray(chk["f_test"])
res["rhs_maxrel_vs_checks"]=float(np.max(np.abs(f-fr)/np.maximum(np.abs(fr),1e-300*0+np.abs(fr).max()*1e-16)))
# Jv from LU-layout values: rebuild J (original numbering) from neg_jac_lu
nj=np.asarray(jax.jit(ch.neg_jac_lu)(jnp.asarray(y), jnp.asarray(k)))
inv={q:e for e,q in pl.pos.items()}
rows=[];cols=[];vals=[]
for q,val in enumerate(nj):
    i,j=inv[q]; rows.append(pl.perm[i]); cols.append(pl.perm[j]); vals.append(-val)
Jm=sp.csr_matrix((vals,(rows,cols)),shape=(ch.n,ch.n))
jv=Jm@v; jvr=np.asarray(chk["jac_v_test"])
res["jv_maxrel_vs_checks"]=float(np.max(np.abs(jv-jvr))/np.abs(jvr).max())
# LU solve of (c I - J) x = b against scipy
c=1.0/(10.0*0.5)
from spike.sparse_lu import make_factor, make_solve
fac=jax.jit(make_factor(pl)); sol=jax.jit(make_solve(pl))
t0=time.perf_counter(); F=jax.block_until_ready(fac(jnp.asarray(nj)+c*ch.diag_mask)); res["compile_factor_s"]=time.perf_counter()-t0
b=jnp.asarray(y)
t0=time.perf_counter(); x=np.asarray(jax.block_until_ready(sol(F,b))); res["compile_solve_s"]=time.perf_counter()-t0
A=(c*sp.identity(ch.n)-Jm).tocsc()
xr=spla.spsolve(A,y)
res["solve_maxrel_vs_scipy"]=float(np.max(np.abs(x-xr)/np.maximum(np.abs(xr),1e-30)))
res["residual_rel"]=float(np.abs(A@x-y).max()/np.abs(y).max())
yj=jnp.asarray(y); kj=jnp.asarray(k)
res["t_rhs_ms"]=1e3*best(jax.jit(ch.rhs),yj,kj)
res["t_jac_ms"]=1e3*best(jax.jit(ch.neg_jac_lu),yj,kj)
vals0=jnp.asarray(nj)+c*ch.diag_mask
res["t_factor_ms"]=1e3*best(fac,vals0)
res["t_solve_ms"]=1e3*best(sol,F,b)
# scipy splu for reference
t0=time.perf_counter()
for _ in range(5): lu=spla.splu(A)
res["t_scipy_splu_ms"]=1e3*(time.perf_counter()-t0)/5
t0=time.perf_counter()
for _ in range(20): lu.solve(y)
res["t_scipy_solve_ms"]=1e3*(time.perf_counter()-t0)/20
res["uptime"]=subprocess.run(["uptime"],capture_output=True,text=True,timeout=10).stdout.strip()
print(json.dumps(res,indent=1))
