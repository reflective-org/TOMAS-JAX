"""Truth reference for the coag_n_floor coagulation-only check (docs/gas_substep.md).

Same 40-bin TFL discretisation and MNFIX as euler_step (kernel recomputed every sub-step),
integrated with the Fortran sub-step limit divided by `s` (floor off): the time-integration
error goes to zero as s grows, what stays is the discretisation itself. Then euler_step with
the floor off / 1e-3 / 1e-2 is compared with that truth.

State: lognormal background (N = 500 cm^-3, Dg = 60 nm, sigma_g = 1.8, pure SO4) plus a burst
of 1.7 nm clusters (bin 0) of 1e7 or 3e9 cm^-3; 217 K, 5500 Pa (rh/gas irrelevant: coagulation
only), boxvol 1e6 cm^3, 10 x 360 s.  Dry mass is SO4 only.

    python benchmarks/python/coag_n_floor_truth.py [--burst 1e7 3e9] [--s 4 16] [--steps 10]

Prints per case: the truth's own convergence (s_lo vs s_hi), and for each floor the sub-step
count, total-N and L1 distance of the 40-bin number distribution to the truth, and the largest
relative error over bins with N > 1e-3 #/cell, next to the no-floor values. Saves NPZ in
benchmarks/results/coag_n_floor_truth/.
"""
import os
import sys
import time
import argparse

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.environ.setdefault('JAX_PLATFORMS', 'cpu')

import numpy as np
import jax
import jax.numpy as jnp
from functools import partial

from tomas_jax.core.config import NBINS, ICOMP, ICOMP_NODIAG, SRTSO4, make_grid
from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.physics.properties import calc_particle_properties
from tomas_jax.physics.coagulation_kernel import calc_coagulation_kernel
from tomas_jax.physics.coagulation_rates import calc_coagulation_rates
from tomas_jax.solvers.euler import euler_step, _compute_adaptive_dt

TEMP, PRES, BOXVOL, DT = 217.0, 5500.0, 1e6, 360.0
DENS = 1770.0
OUT = os.path.join(ROOT, 'benchmarks', 'results', 'coag_n_floor_truth')


def grid():
    return make_grid()


def initial(xk, burst_cm3):
    xm = np.sqrt(np.array(xk[:-1]) * np.array(xk[1:]))
    dp = (6.0 * xm / (np.pi * DENS)) ** (1.0 / 3.0)
    dlogd = np.log10(np.array(xk[1:]) / np.array(xk[:-1])) / 3.0
    dg, sg, n_tot = 60e-9, 1.8, 500.0
    dn = n_tot * dlogd / (np.sqrt(2 * np.pi) * np.log10(sg)) * np.exp(
        -0.5 * (np.log10(dp / dg) / np.log10(sg)) ** 2)
    n = np.maximum(dn, 1e-3) * BOXVOL
    n[0] += burst_cm3 * BOXVOL
    Nk = jnp.array(n)
    Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(Nk * jnp.array(xm))
    return Nk, Mk


@partial(jax.jit, static_argnames=('s', 'max_sub'))
def truth_step(Nk, Mk, xk, s, max_sub=20_000_000):
    """euler_step(recompute_kernel=True, floor off) with every sub-step divided by s."""
    def cond(c):
        return (c[2] < DT - 1e-12) & (c[3] < max_sub)

    def body(c):
        N, M, t, n = c
        Dp, D, cb = calc_particle_properties(N, M, TEMP, PRES, None)
        kij = calc_coagulation_kernel(Dp, D, cb, BOXVOL)
        dN, dM, _ = calc_coagulation_rates(N, M, kij, xk, ICOMP_NODIAG)
        dt = _compute_adaptive_dt(N, M, dN, dM, DT - t, ICOMP_NODIAG)
        dt = jnp.minimum(dt, (DT - t)) if s == 1 else jnp.minimum(dt / s, DT - t)
        N2 = jnp.maximum(N + dt * dN, 0.0)
        M2 = jnp.maximum(M + dt * dM, 0.0)
        N2, M2 = mnfix_jax(N2, M2, xk, ICOMP_NODIAG)
        return N2, M2, t + dt, n + 1

    N, M, t, n = jax.lax.while_loop(cond, body, (Nk, Mk, jnp.float64(0.0), jnp.int32(0)))
    return N, M, n, DT - t


@partial(jax.jit, static_argnames=('floor',))
def floor_step(Nk, Mk, xk, floor):
    return euler_step(Nk, Mk, xk, TEMP, PRES, BOXVOL, DT, recompute_kernel=True,
                      max_substeps=200000, return_info=True, return_dt_left=True,
                      n_floor=floor)


def run(step, Nk, Mk, n_steps):
    subs, left = [], 0.0
    for _ in range(n_steps):
        out = step(Nk, Mk)
        Nk, Mk = out[0], out[1]
        subs.append(int(out[2]))
        left += float(out[3])
    return Nk, Mk, subs, left


def dist(Nk, Nt):
    Nk, Nt = np.array(Nk), np.array(Nt)
    l1 = np.abs(Nk - Nt).sum() / Nt.sum()
    # L1 of the mass-weighted (per-log-bin) distribution is the same sum for equal bins
    big = Nt > 1e-3
    rel = np.abs(Nk[big] / Nt[big] - 1.0).max()
    return Nk.sum() / Nt.sum() - 1.0, l1, rel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--burst', nargs='*', type=float, default=[1e7, 3e9])
    ap.add_argument('--s', nargs=2, type=int, default=[4, 16], metavar=('S_LO', 'S_HI'))
    ap.add_argument('--steps', type=int, default=10)
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    xk = grid()
    for burst in a.burst:
        print(f'\n=== burst {burst:.0e} cm^-3, {a.steps} x {DT:.0f} s ===', flush=True)
        N0, M0 = initial(xk, burst)
        res = {}
        for s in (1, *a.s):
            t0 = time.time()
            N, M, subs, left = run(lambda n, m, s=s: truth_step(n, m, xk, s=s), N0, M0, a.steps)
            res[s] = (N, M, subs)
            print(f'truth s={s:3d}: {sum(subs):9d} sub-steps  {time.time()-t0:7.1f} s'
                  f'  dt_left {left:.1e}', flush=True)
        Nt = res[a.s[1]][0]
        tn, l1, rel = dist(res[a.s[0]][0], Nt)
        print(f'truth convergence s={a.s[0]} vs s={a.s[1]}: dN {tn:+.2e}  L1 {l1:.2e}  '
              f'max rel (N>1e-3) {rel:.2e}')
        np.savez(os.path.join(OUT, f'burst{burst:.0e}.npz'), truth=np.array(Nt),
                 **{f's{s}': np.array(res[s][0]) for s in res})
        print(f'{"floor":>8} {"sub-steps":>10} {"dN/N":>10} {"L1":>9} {"max rel":>9}'
              f' {"dry M err":>10}')
        m0 = float(M0[:, :ICOMP_NODIAG].sum())
        for f in (None, 1e-3, 1e-2):
            N, M, subs, left = run(lambda n, m, f=f: floor_step(n, m, xk, floor=f),
                                   N0, M0, a.steps)
            tn, l1, rel = dist(N, Nt)
            print(f'{str(f):>8} {sum(subs):10d} {tn:+10.2e} {l1:9.2e} {rel:9.2e}'
                  f' {abs(float(M[:, :ICOMP_NODIAG].sum())-m0)/m0:10.1e}'
                  + (f'  dt_left {left:.1e}' if left else ''), flush=True)


if __name__ == '__main__':
    main()
