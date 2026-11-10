"""Exact vs frozen-shape PPM tangent (make_step ppm_tangent) against central FD.

    python -m benchmarks.python.ppm_tangent_stability [--hours 18] [--cases ...]

Prescribed-J runs (sensitivities.dunne_nuc.prescribed_j's set-up: 217 K,
Phase 2 conditions, dt_for(SO2), 40 bins), d N(>3 nm) / d J [s] hourly from
- a jvp with ppm_tangent='exact' (the default derivative),
- a jvp with ppm_tangent='frozen_shape',
- a central FD at J(1 +- FD_REL) (forward values are the same in both modes).
Cases (CASES): 100 ppbv at J = 0.01 cm-3 s-1, where the exact tangent grows
x22 per hour (docs/prescribed_j.md), and J = J_ref (the Phase 2 map's mean
Dunne J) at 1, 100 and 1000 ppbv, where it was stable. The forward runs of the
two modes must be bit-identical (checked; raises otherwise).
Writes benchmarks/results/ppm_tangent/stability.npz and prints the table.
"""
import os
import argparse

import numpy as np
import jax
import jax.numpy as jnp

from sensitivities.dunne_nuc.common import (STRAT, BOXVOL, lognormal_state,
                                            make_prescribed_j_runner, ppbv_to_molec)
from sensitivities.dunne_nuc.run_map import GRIDS, dt_for
from sensitivities.dunne_nuc.prescribed_j import j_reference, FD_REL
from sensitivities.dunne_nuc.sabre_bins import nmass_cumulative

OUTDIR = os.path.join(os.path.dirname(__file__), '..', 'results', 'ppm_tangent')
TEMP = 217.0
NBINS = 40
# (name, SO2 [ppbv], J [cm-3 s-1] or None for J_ref)
CASES = {'low_J': (100.0, 0.01), 'ref_1ppb': (1.0, None),
         'ref_100ppb': (100.0, None), 'ref_1ppm': (1000.0, None)}


def build(xk, dt, hours, ppm_tangent):
    run_hour = make_prescribed_j_runner(xk, nsteps=round(3600.0 / dt), dt=dt,
                                        cond=dict(STRAT, temp=TEMP),
                                        ppm_tangent=ppm_tangent)
    Nk0, Mk0, _ = lognormal_state(xk)

    def sim(J, Gc0):
        def hour(carry, _):
            Nk, Mk, Gc, _ = run_hour(J, *carry)
            return (Nk, Mk, Gc), (nmass_cumulative(Nk, Mk, xk, BOXVOL)[0], Nk)
        _, ys = jax.lax.scan(hour, (Nk0, Mk0, Gc0), None, length=hours)
        return ys
    return sim


def run_case(name, hours):
    so2, J = CASES[name]
    xk = GRIDS[NBINS]()
    if J is None:
        J = float(j_reference(NBINS, TEMP, [so2])[0])
    dt = dt_for(so2)
    gc = lognormal_state(xk, so2=ppbv_to_molec(so2, TEMP, STRAT['pres']))[2]
    out = {}
    for mode in ('exact', 'frozen_shape'):
        sim = build(xk, dt, hours, mode)
        (n3, Nk), (dn3, _) = jax.jit(lambda j: jax.jvp(lambda x: sim(x, gc), (j,),
                                                       (jnp.ones_like(j),)))(jnp.float64(J))
        out[mode] = (np.asarray(n3), np.asarray(Nk), np.asarray(dn3))
    if not (np.array_equal(out['exact'][1], out['frozen_shape'][1])
            and np.array_equal(out['exact'][0], out['frozen_shape'][0])):
        raise RuntimeError(f"{name}: forward values differ between ppm_tangent modes")
    sim = jax.jit(lambda j: build(xk, dt, hours, 'exact')(j, gc)[0])
    fd = (np.asarray(sim(J * (1 + FD_REL))) - np.asarray(sim(J * (1 - FD_REL)))) / (2 * FD_REL * J)
    return dict(so2=so2, J=J, dt=dt, n3=out['exact'][0], ad_exact=out['exact'][2],
                ad_frozen=out['frozen_shape'][2], fd=fd)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--hours', type=int, default=18)
    parser.add_argument('--cases', nargs='+', choices=list(CASES), default=list(CASES))
    args = parser.parse_args()
    os.makedirs(OUTDIR, exist_ok=True)
    res = {}
    for name in args.cases:
        r = res[name] = run_case(name, args.hours)
        print(f"\n{name}: SO2 {r['so2']:g} ppbv, J {r['J']:.3g} cm-3 s-1, dt {r['dt']:g} s; "
              "dN(>3 nm)/dJ [s]")
        print(f"{'hour':>4} {'N(>3) cm-3':>11} {'AD exact':>11} {'AD frozen':>11} {'FD':>11}"
              f" {'frozen/FD-1':>11} {'exact/FD-1':>11}")
        for h in range(args.hours):
            fd, ex, fr = r['fd'][h], r['ad_exact'][h], r['ad_frozen'][h]
            print(f"{h + 1:4d} {r['n3'][h]:11.4g} {ex:11.4g} {fr:11.4g} {fd:11.4g}"
                  f" {fr / fd - 1:11.3g} {ex / fd - 1:11.3g}")
    np.savez(os.path.join(OUTDIR, 'stability.npz'),
             **{f'{n}_{k}': v for n, r in res.items() for k, v in r.items()})


if __name__ == '__main__':
    main()
