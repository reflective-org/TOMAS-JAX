"""Diagnose which coagulation intervals bind coag_max_substeps (15 km, 100000 ppbv, euler_adaptive).
Wraps solvers.euler.euler_step as called from make_step and prints, for every interval with
>= 1000 sub-steps, its length, n_sub, dt_left and the limiting bin/species at start and end.
usage: python -m benchmarks.python.diag_coag_cap FRAC HOURS CAP   (docs/gas_substep.md)
"""
import sys
import numpy as np, jax, jax.numpy as jnp
from tomas_jax.solvers import condensation as cd
from tomas_jax.solvers import euler as eu
from tomas_jax.physics.properties import calc_particle_properties
from tomas_jax.physics.coagulation_kernel import calc_coagulation_kernel
from tomas_jax.physics.coagulation_rates import calc_coagulation_rates
from tomas_jax.core.config import ICOMP_NODIAG, TINY_N, TINY_M
import benchmarks.python.gas_substep_coag_interval as m

frac, hours, cap = float(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3])
orig = eu.euler_step

def limiter(Nk, Mk, xk, temp, pres, boxvol, sd):
    Dp, D, c = calc_particle_properties(Nk, Mk, temp, pres, sd)
    k = calc_coagulation_kernel(Dp, D, c, boxvol)
    dN, dM, _ = calc_coagulation_rates(Nk, Mk, k, xk, ICOMP_NODIAG)
    tN = jnp.where(Nk > TINY_N, 0.25 * jnp.abs(Nk) / jnp.maximum(jnp.abs(dN), 1e-30), jnp.inf)
    Mp, dMp = Mk[:, :ICOMP_NODIAG], dM[:, :ICOMP_NODIAG]
    tM = jnp.where(Mp > TINY_M, 10 * jnp.abs(Mp) / jnp.maximum(jnp.abs(dMp), 1e-30), jnp.inf)
    kN = jnp.argmin(tN); flat = jnp.argmin(tM)
    return (jnp.min(tN), kN, Nk[kN], dN[kN], jnp.min(tM), flat // ICOMP_NODIAG,
            flat % ICOMP_NODIAG, Nk)

def report(dt, n, left, s0, s1):
    if int(n) < 1000:
        return
    f = lambda s: (f"dtN={float(s[0]):.3g}s bin{int(s[1])} N={float(s[2]):.3g} dN/dt={float(s[3]):.3g} | "
                   f"dtM={float(s[4]):.3g}s bin{int(s[5])} sp{int(s[6])}")
    print(f"interval {float(dt):.4g}s n_sub={int(n)} dt_left={float(left):.3g}\n   start: {f(s0)}\n   end:   {f(s1)}",
          flush=True)

def wrapped(Nk, Mk, xk, temp, pres, boxvol, dt, **kw):
    s0 = limiter(Nk, Mk, xk, temp, pres, boxvol, kw['sparse_density'])
    Nk2, Mk2, n, left = orig(Nk, Mk, xk, temp, pres, boxvol, dt, return_info=True, **kw)
    s1 = limiter(Nk2, Mk2, xk, temp, pres, boxvol, kw['sparse_density'])
    jax.debug.callback(report, dt, n, left, s0[:7], s1[:7])
    return Nk2, Mk2, left

cd.euler_step = wrapped
m.KINDS[f'd{frac:g}'] = frac
m.DOUT = '/tmp/diag_coag_cap_out'
m.COAG_MAX_SUBSTEPS = cap
m.REPORT_HOURS = (hours,)
m.build.__globals__['REPORT_HOURS'] = (hours,)
import benchmarks.python.gas_substep_convergence as g
m.run_one.__globals__['REPORT_HOURS'] = (hours,)
m.run_one(15.0, 100000.0, f'd{frac:g}')
