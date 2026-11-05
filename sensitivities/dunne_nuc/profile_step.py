"""Per-process cost of one model step for a Phase 3 scenario.

Runs the scenario forward to a few representative times, then times each
process of a step on its own (SO2 chemistry, nucleation, coagulation,
condensation, dilution) and the full step, as a jitted lax.scan of REPS
repeated steps (dispatch overhead amortised). Also times the full step under
jax.jvp (1 tangent) and jax.jacfwd (25 tangents), the costs behind the Phase
3 dt ladder and Jacobian.

Usage::

    python -m sensitivities.dunne_nuc.profile_step                     # sai_plume, dt 2 s
    python -m sensitivities.dunne_nuc.profile_step --dt 0.5 --grids 40
"""
import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.physics import nucleation as nuc
from tomas_jax.solvers.condensation import make_step
from .common import (BOXVOL, FION, NUC_PARAMS, OH_CONC, SPARSE_DENSITY, WATER_SCHEME,
                     make_runner)
from .run_map import GRIDS
from .scenarios import SCENARIOS, initial_state, runner_kwargs

REPS = 200
P0 = jax.tree.map(jnp.float64, NUC_PARAMS)


def states_at(name, xk, dt, hours):
    """Model state (Nk, Mk, Gc) at each of `hours` (forward run)."""
    rk = runner_kwargs(name, xk)
    out, state, t_prev = {}, initial_state(name, xk), 0.0
    for h in hours:
        n = round((h - t_prev) * 3600.0 / dt)
        run = jax.jit(make_runner(xk, n, dt=dt, **rk))
        Nk, Mk, Gc, d = run(P0, *state)
        state, t_prev = (Nk, Mk, Gc), h
        out[h] = dict(state=state, n_sub=int(np.max(d['n_sub'])),
                      N=float(jnp.sum(Nk)) / BOXVOL)
    return out


def timed(fn, *args):
    """Seconds per call of jit(fn) after compilation (median of 3)."""
    f = jax.jit(fn)
    jax.block_until_ready(f(*args))
    ts = []
    for _ in range(3):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def profile(name, nbins, dt, hours):
    xk = GRIDS[nbins]()
    sc = SCENARIOS[name]
    c = sc['cond']
    env = (xk, c['temp'], c['pres'], BOXVOL, c['rh'], c['alpha'], dt)
    common = dict(sparse_density=SPARSE_DENSITY, water_scheme=WATER_SCHEME, jit=False)
    rk = runner_kwargs(name, xk)
    nucl_kw = dict(org_conc=0.0, nh3_conc=0.0, fion=FION)
    dil_kw = {k: rk[k] for k in ('kdil', 'Nk_bg', 'Mk_bg', 'Gc_bg') if k in rk}
    procs = {
        'so2_chemistry': (make_step(['so2_chemistry'], jit=False), dict(oh_conc=OH_CONC)),
        'nucleation': (make_step(['nucleation'], **common), nucl_kw),
        'coagulation': (make_step(['coagulation'], **common), {}),
        'condensation': (make_step(['condensation'], **common), {}),
    }
    if dil_kw:
        procs['dilution'] = (make_step(['dilution'], **common), dil_kw)
    full = make_step(list(procs), **common)
    full_kw = dict(oh_conc=OH_CONC, **nucl_kw, **dil_kw)

    def repeat(step, kw):
        def f(p, Nk, Mk, Gc):
            def body(s, _):
                return step(*s, *env, **kw, nucl_params=p) if 'fion' in kw else \
                    step(*s, *env, **kw), None
            return jax.lax.scan(body, (Nk, Mk, Gc), None, length=REPS)[0]
        return f

    print(f"\n{name}, {nbins} bins, dt = {dt:g} s ({REPS} repeated steps per timing; "
          f"per-step cost in µs)")
    print(f"  {'state':>12s} " + ' '.join(f'{p:>13s}' for p in procs)
          + f" {'sum':>8s} {'full step':>9s} {'jvp(1)':>8s} {'jacfwd(25)':>10s}  n_sub  N[cm-3]")
    for h, st in states_at(name, xk, dt, hours).items():
        args = (P0, *st['state'])
        row = {p: timed(repeat(s, kw), *args) / REPS * 1e6 for p, (s, kw) in procs.items()}
        f_full = repeat(full, full_kw)
        t_full = timed(f_full, *args) / REPS * 1e6
        out = lambda p, *s: f_full(p, *s)[0]
        t_jvp = timed(lambda p, *s: jax.jvp(lambda q: out(q, *s), (p,), (p,)), *args) / REPS * 1e6
        t_jac = timed(lambda p, *s: jax.jacfwd(lambda q: out(q, *s))(p), *args) / REPS * 1e6
        print(f"  {f'{h:g} h':>12s} " + ' '.join(f'{row[p]:13.1f}' for p in procs)
              + f" {sum(row.values()):8.1f} {t_full:9.1f} {t_jvp:8.1f} {t_jac:10.1f}"
              + f"  {st['n_sub']:5d}  {st['N']:.2e}", flush=True)
    steps_48h = 48 * 3600 / dt
    print(f"  48 h = {steps_48h:.0f} steps: full step x steps = "
          f"{t_full * steps_48h / 1e6 / 60:.1f} min forward, "
          f"{t_jac * steps_48h / 1e6 / 3600:.1f} h jacfwd (at the last state's cost)")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--scenario', default='sai_plume', choices=list(SCENARIOS))
    ap.add_argument('--dt', type=float, default=2.0)
    ap.add_argument('--grids', type=int, nargs='+', default=[40, 80])
    ap.add_argument('--hours', type=float, nargs='+', default=[0.5, 6.0, 24.0])
    args = ap.parse_args()
    for g in args.grids:
        profile(args.scenario, g, args.dt, args.hours)


if __name__ == '__main__':
    main()
