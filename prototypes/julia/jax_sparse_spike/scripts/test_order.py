"""Convergence order of the Rosenbrock tableaux on the Brusselator (dense linear algebra):
fixed steps h = 1/N on [0, 1]; error of the main and of the embedded solution."""
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from scipy.integrate import solve_ivp
from spike.rosenbrock import TABLEAUS, make_integrator

def rhs(y, k):
    return jnp.array([1 + y[0]**2*y[1] - 4*y[0], 3*y[0] - y[0]**2*y[1]])
def neg_jac(y, k):
    return -jax.jacfwd(rhs)(y, k).ravel()
diag = jnp.eye(2).ravel()
factor = lambda v: jsl.lu_factor(v.reshape(2, 2))
solve = lambda f, b: jsl.lu_solve(f, b)
y0 = np.array([1.5, 3.0])
ref = solve_ivp(lambda t, y: np.array(rhs(y, None)), (0, 1), y0, method="DOP853", rtol=1e-13, atol=1e-13).y[:, -1]
out = {}
for name, tab in TABLEAUS.items():
    integ = make_integrator(rhs, neg_jac, diag, factor, solve, tab, 1e-3, 1e-3)
    st = jax.jit(integ.fixed_step)
    errs, errs_emb = [], []
    for N in (20, 40, 80, 160):
        y = jnp.asarray(y0); h = 1.0 / N
        ye = None
        for _ in range(N):
            y, ev = st(y, None, h)
        errs.append(float(np.abs(np.asarray(y) - ref).max()))
        # embedded order: local error of one step vs h
    for N in (20, 40, 80, 160):
        h = 1.0 / N
        y1, ev = st(jnp.asarray(y0), None, h)
        loc = solve_ivp(lambda t, y: np.array(rhs(y, None)), (0, h), y0, method="DOP853", rtol=1e-14, atol=1e-14).y[:, -1]
        errs_emb.append(float(np.abs(np.asarray(y1) - np.asarray(ev) - loc).max()))
    o = [np.log2(errs[i]/errs[i+1]) for i in range(3)]
    oe = [np.log2(errs_emb[i]/errs_emb[i+1]) - 1 for i in range(3)]
    out[name] = {"global_err": errs, "global_order": o, "embedded_local_err": errs_emb, "embedded_order": oe}
print(json.dumps(out, indent=1))
