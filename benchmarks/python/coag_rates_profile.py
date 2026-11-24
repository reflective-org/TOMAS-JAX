"""Cost of the linear sub-bin coagulation rates: library vs the scatter reference.

Times, per call, the linear scheme as it is now
(physics/coagulation_rates_linear.calc_coagulation_rates_linear) and the
original scatter-add formulation it replaced
(tests/reference/coagulation_rates_linear_scatter.py), next to TFL on the same
grid, the kernel, MNFIX (its cost depends on how many bins it shifts), one
coag_euler_step and one make_step(['coagulation']) step with each of the two
linear implementations (40 bins: the dispatcher picks TFL, so the step rows do
not depend on the linear scheme). Also prints the largest
difference between the two (relative to each output's largest value).

States: sai_plume at 6 h (sensitivities/dunne_nuc, dt = 2 s, the same state
as sensitivities/dunne_nuc/profile_step.py) on 40 and 80 bins, and a bimodal
lognormal on 160 bins (running the plume 6 h on 160 bins would take most of
the benchmark's time).

Every timing is a jitted lax.scan of REPS calls whose input depends on the
previous call's output (a 1e-300-scaled feedback), so XLA cannot hoist the
work out of the loop; the median of 5 runs is reported.

Run from the repository root with -m (it imports tests.reference and
sensitivities.dunne_nuc):

    python -m benchmarks.python.coag_rates_profile                 # 40, 80, 160 bins
    python -m benchmarks.python.coag_rates_profile --grids 80 --trace

--trace also prints the most expensive XLA operations of each linear
implementation (jax.profiler; per call).
"""
import argparse
import collections
import contextlib
import glob
import gzip
import json
import os
import tempfile
import time

import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.core.config import ICOMP, XK0, make_grid
from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.physics import coagulation_rates_linear as linear_mod
from tomas_jax.physics.coagulation_kernel import calc_coagulation_kernel
from tomas_jax.physics.coagulation_rates import calc_coagulation_rates_tfl
from tomas_jax.physics.coagulation_rates_linear import calc_coagulation_rates_linear
from tomas_jax.physics.properties import calc_particle_properties
from tomas_jax.solvers.condensation import make_step
from tomas_jax.solvers.diffrax import coag_euler_step
from tests.reference.coagulation_rates_linear_scatter import (
    calc_coagulation_rates_linear_scatter)

REPS = 500
EPS = 1e-300
PLUME_DT = 2.0
PLUME_HOURS = 6.0


def grid(nbins):
    """1.7 nm start, fixed range, mass ratio 2^(40/nbins) (= make_grid_80bin at 80)."""
    return make_grid(nbins, XK0, 2.0 ** (40.0 / nbins))


def bimodal_state(xk, rho=1770.0):
    xk = np.asarray(xk)
    xc = np.sqrt(xk[:-1] * xk[1:])
    d = (6.0 * xc / (np.pi * rho)) ** (1.0 / 3.0)
    N = (1e10 * np.exp(-0.5 * (np.log(d / 5e-9) / 0.4) ** 2)
         + 1e8 * np.exp(-0.5 * (np.log(d / 1e-7) / 0.5) ** 2))
    M = np.zeros((xc.size, ICOMP))
    M[:, 0] = 0.8 * N * xc
    M[:, 1] = 0.2 * N * xc
    return jnp.asarray(N), jnp.asarray(M)


def state_for(nbins, xk):
    if nbins in (40, 80):
        from sensitivities.dunne_nuc.profile_step import states_at
        Nk, Mk, _ = states_at('sai_plume', xk, PLUME_DT, [PLUME_HOURS])[PLUME_HOURS]['state']
        return Nk, Mk, f'sai_plume at {PLUME_HOURS:g} h'
    Nk, Mk = bimodal_state(xk)
    return Nk, Mk, 'bimodal lognormal (5 nm / 100 nm)'


def timed(f, Nk, Mk):
    """µs per call of f(Nk, Mk) inside a dependent scan."""
    def run(Nk, Mk):
        def body(c, _):
            N, M = c
            s = sum(jnp.sum(jnp.abs(x)) for x in jax.tree.leaves(f(N, M))) * EPS
            return (N + s, M + s), None
        return jax.lax.scan(body, (Nk, Mk), None, length=REPS)[0]
    g = jax.jit(run)
    jax.block_until_ready(g(Nk, Mk))
    ts = []
    for _ in range(5):
        t0 = time.perf_counter()
        jax.block_until_ready(g(Nk, Mk))
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts)) / REPS * 1e6


def top_ops(f, Nk, Mk, n=12, reps=200):
    """Most expensive XLA operations of f inside the dependent scan [µs/call]."""
    def run(Nk, Mk):
        def body(c, _):
            N, M = c
            s = sum(jnp.sum(jnp.abs(x)) for x in jax.tree.leaves(f(N, M))) * EPS
            return (N + s, M + s), None
        return jax.lax.scan(body, (Nk, Mk), None, length=reps)[0]
    g = jax.jit(run)
    jax.block_until_ready(g(Nk, Mk))
    with tempfile.TemporaryDirectory() as d:
        with jax.profiler.trace(d, create_perfetto_trace=True):
            jax.block_until_ready(g(Nk, Mk))
        files = glob.glob(os.path.join(d, '**', '*.trace.json.gz'), recursive=True)
        if not files:
            raise FileNotFoundError(f"jax.profiler wrote no trace under {d}")
        events = json.load(gzip.open(files[-1]))
    events = events['traceEvents'] if isinstance(events, dict) else events
    tot, cnt = collections.Counter(), collections.Counter()
    for e in events:
        if e.get('ph') == 'X':
            tot[e['name']] += e.get('dur', 0)
            cnt[e['name']] += 1
    # XLA kernels run once per scan iteration; drop the enclosing spans
    ops = [(k, t) for k, t in tot.items() if cnt[k] == reps and k != 'ThunkExecutor::Execute']
    return sorted(ops, key=lambda kt: -kt[1])[:n], reps


@contextlib.contextmanager
def linear_impl(fn):
    """Route the dispatcher (and so every solver) to `fn` while tracing."""
    saved = linear_mod.calc_coagulation_rates_linear
    linear_mod.calc_coagulation_rates_linear = fn
    try:
        yield
    finally:
        linear_mod.calc_coagulation_rates_linear = saved


def max_rel_diff(a, b):
    return tuple(float(jnp.max(jnp.abs(x - y)) / jnp.maximum(jnp.max(jnp.abs(y)), 1e-300))
                 for x, y in zip(a, b))


def profile(nbins, trace):
    from sensitivities.dunne_nuc.common import BOXVOL, SPARSE_DENSITY
    from sensitivities.dunne_nuc.scenarios import SCENARIOS
    c = SCENARIOS['sai_plume']['cond']
    T, P = c['temp'], c['pres']
    xk = grid(nbins)
    Nk, Mk, label = state_for(nbins, xk)
    Dpk, Dk, ck = calc_particle_properties(Nk, Mk, T, P, SPARSE_DENSITY)
    kij = calc_coagulation_kernel(Dpk, Dk, ck, BOXVOL)
    new, old = calc_coagulation_rates_linear, calc_coagulation_rates_linear_scatter

    print(f"\n{nbins} bins, {label} (µs per call, {REPS} dependent calls, median of 5)")
    diff = max_rel_diff(new(Nk, Mk, kij, xk), old(Nk, Mk, kij, xk))
    print("  max |new - old| / max|old|: dNdt {:.1e}, dMdt {:.1e}, dM_overflow {:.1e}".format(*diff))

    rows = [
        ('linear rates, scatter reference', lambda N, M: old(N, M, kij, xk)),
        ('linear rates, library', lambda N, M: new(N, M, kij, xk)),
        ('TFL rates (same grid)', lambda N, M: calc_coagulation_rates_tfl(N, M, kij, xk)),
        ('kernel (from properties)', lambda N, M: calc_coagulation_kernel(
            *calc_particle_properties(N, M, T, P, SPARSE_DENSITY), BOXVOL)),
        ('MNFIX on this state', lambda N, M: mnfix_jax(N, M, xk)),
    ]
    for name, f in rows:
        print(f"  {name:44s} {timed(f, Nk, Mk):9.1f}", flush=True)

    step = make_step(['coagulation'], sparse_density=SPARSE_DENSITY, jit=False)
    Gc = jnp.zeros(44)
    for impl, fn in (('scatter reference', old), ('library', new)):
        with linear_impl(fn):
            t_euler = timed(lambda N, M: coag_euler_step(
                N, M, xk, T, P, BOXVOL, dt=PLUME_DT, n_substeps=10,
                sparse_density=SPARSE_DENSITY), Nk, Mk)
            t_step = timed(lambda N, M: step(N, M, Gc, xk, T, P, BOXVOL, c['rh'], c['alpha'],
                                             PLUME_DT)[:2], Nk, Mk)
        print(f"  {'coag_euler_step (10 sub-steps), ' + impl:44s} {t_euler:9.1f}")
        print(f"  {'make_step([coagulation]) step, ' + impl:44s} {t_step:9.1f}", flush=True)

    if trace:
        for impl, fn in (('scatter reference', old), ('library', new)):
            ops, reps = top_ops(lambda N, M: fn(N, M, kij, xk), Nk, Mk)
            print(f"  most expensive XLA ops, {impl} (µs per call):")
            for name, t in ops:
                print(f"    {t / reps:8.2f}  {name}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--grids', type=int, nargs='+', default=[40, 80, 160])
    ap.add_argument('--trace', action='store_true',
                    help='also print the most expensive XLA ops (jax.profiler)')
    args = ap.parse_args()
    print(f"jax {jax.__version__}, backend {jax.default_backend()}")
    for nb in args.grids:
        profile(nb, args.trace)


if __name__ == '__main__':
    main()
