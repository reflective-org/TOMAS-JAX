"""coag_interval_frac ladder: euler_adaptive coagulation inside adaptive gas sub-steps
(f = 0.1) at outer dt = 360 s, tropical altitude grid (13-25 km x SO2, 40 bins, 24 h).

coag_interval_frac closes a coagulation interval once nucleation has added that fraction
of N; larger values give fewer, longer intervals (cheaper, coarser). New runs (forward
only, same state and settings as gas_substep_euler_adaptive, hour means every 360 s):
    ci0.003, ci0.03, ci0.1, ci0.3, ci0.5
The 0.01 rung (make_step's default) is the existing ea_f0.1_360 run in
euler_adaptive_tropical/, read from there and not recomputed (its wall time was measured
under a different machine load).

Reference: the smallest-dt split-chemistry run per cell in psd40_tropical/: ref_split_half
(dt 0.125 s) where it exists (SO2 >= 100 ppbv), otherwise ref_split. The one used is
printed per cell. Error is the hour-mean L1 number and dry mass at 6 h and 24 h. Pass:
L1 N <= 0.05 at 6 h and 24 h, finite, no gas-sub-step cap, no CoagSubstepCapWarning
(read from the run's .log; the cap drops the rest of dt).

    python -m benchmarks.python.gas_substep_coag_interval --run 13 100 ci0.1   # one run
    python -m benchmarks.python.gas_substep_coag_interval --all                # missing runs, then --compare
    python -m benchmarks.python.gas_substep_coag_interval --compare            # table + coag_interval_compare.npy

--all runs NJOBS (default 1, for timing) at a time, highest SO2 first; COAG_MAX_SUBSTEPS
(default 10000). Unattended: `run_awake -m benchmarks.python.gas_substep_coag_interval --all`.

Output: benchmarks/results/gas_substep/coag_interval_tropical/z<km>_<so2>_<kind>.{npz,log}.
"""
import os
import sys
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import jax.numpy as jnp

from tomas_jax.core.config import make_grid, XK0
from sensitivities.dunne_nuc import common as c
from sensitivities.dunne_nuc.run_map import SO2_PPBV, directions
from benchmarks.python.gas_substep_convergence import build, l1, REPORT_HOURS, OUTDIR
from benchmarks.python.gas_substep_dt360 import (CELLS, D40, SAMPLE_S, GAS_MAX_SUBSTEPS,
                                                 fname, need)

DOUT = os.path.join(OUTDIR, 'coag_interval_tropical')
DEA = os.path.join(OUTDIR, 'euler_adaptive_tropical')
COAG_MAX_SUBSTEPS = int(os.environ.get('COAG_MAX_SUBSTEPS', 10000))
TOL = 0.05
LADDER = (0.003, 0.03, 0.1, 0.3, 0.5)
DT = 360.0
FRAC = 0.1
KINDS = {f'ci{x:g}': x for x in LADDER}     # kind -> coag_interval_frac
# coag_n_floor rungs: coag_interval_frac 0.01 (the default) with the euler_adaptive number
# limit ignoring bins at or below the floor [#/cell] (kind -> floor)
NFLOOR = {'nf0.01': 1e-2, 'nf0.001': 1e-3}
KINDS.update({k: 0.01 for k in NFLOOR})
BASE_KIND, BASE_FILE = 'ci0.01', 'ea_f0.1_360'   # existing run (default 0.01)
ALL_KINDS = sorted(KINDS, key=lambda k: KINDS[k])
ALL_KINDS.insert(sum(KINDS[k] < 0.01 for k in ALL_KINDS), BASE_KIND)


def pout(z, so2, kind, ext='npz'):
    return os.path.join(DOUT, fname(z, so2, kind).replace('.npz', f'.{ext}'))


def pkind(z, so2, kind):
    if kind == BASE_KIND:
        return os.path.join(DEA, fname(z, so2, BASE_FILE))
    return pout(z, so2, kind)


def plog(z, so2, kind):
    return (os.path.join(DEA, fname(z, so2, BASE_FILE).replace('.npz', '.log'))
            if kind == BASE_KIND else pout(z, so2, kind, 'log'))


def done(z, so2, kind):
    p = pout(z, so2, kind)
    return os.path.exists(p) and 'Nk_avg' in np.load(p).files


def reference(z, so2):
    """(name, data) of the smallest-dt split-chemistry run of the cell."""
    for name in ('ref_split_half', 'ref_split'):
        p = os.path.join(D40, fname(z, so2, name))
        if os.path.exists(p):
            return name, np.load(p)
    raise FileNotFoundError(f"No ref_split_half or ref_split for z={z:g} SO2={so2:g} in {D40}.")


def run_one(z, so2, kind):
    _, T, pres = next(x for x in CELLS if x[0] == z)
    xk = jnp.asarray(make_grid(40, XK0, 2.0))
    direction = jnp.asarray(directions(T)['H2SO4 cal'])
    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(so2, T, pres))
    out, _, info = build(xk, DT, REPORT_HOURS, FRAC, fd_alpha=0.0, pres=pres,
                         sample_s=SAMPLE_S, gas_max_substeps=GAS_MAX_SUBSTEPS,
                         coag_solver='euler_adaptive', coag_max_substeps=COAG_MAX_SUBSTEPS,
                         coag_interval_frac=KINDS[kind],
                         coag_n_floor=NFLOOR.get(kind))(direction, T, *state)
    os.makedirs(DOUT, exist_ok=True)
    np.savez(pout(z, so2, kind), T=T, pres=pres, so2=so2, kind=kind, dt=DT,
             coag_interval_frac=KINDS[kind], coag_n_floor=NFLOOR.get(kind, 0.0), coag_solver='euler_adaptive',
             hours=np.array(sorted(REPORT_HOURS)), Nk=out['Nk'], Mk=out['Mk'],
             nmass=out['nmass'], Nk_avg=out['Nk_avg'], Mk_avg=out['Mk_avg'],
             Nk_series=out['Nk_series'], Mk_series=out['Mk_series'], t=info['t'],
             t_compile=info['t_compile'], n_sub=info['n_sub'], knuc=info['knuc'],
             cap=info['cap'])
    print(f"z={z:g} km SO2={so2:g} {kind} dt={DT:g} n_sub={info['n_sub']} cap={info['cap']} "
          f"t={info['t'][-1]:.0f}s (+{info['t_compile']:.0f}s compile)", flush=True)


def coag_cap_hit(z, so2, kind):
    p = plog(z, so2, kind)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}: the coagulation cap flag is read from the log. "
                                f"Run through --all (or redirect --run output to it).")
    return 'CoagSubstepCapWarning' in open(p).read()


def row(z, T, so2, kind, ref_name, ref):
    d = np.load(pkind(z, so2, kind))
    ok = bool(np.all(np.isfinite(d['Nk_avg'])) and np.all(np.isfinite(d['Mk_avg'])))
    r = dict(z=z, T=T, so2=so2, kind=kind, ref=ref_name, finite=ok, gas_cap=bool(d['cap']),
             coag_cap=coag_cap_hit(z, so2, kind), n_sub=int(d['n_sub']))
    for i, h in enumerate(d['hours']):
        j = list(ref['hours']).index(h)
        r[f't_{int(h)}h'] = float(d['t'][i])
        for q, key in (('N', 'Nk_avg'), ('M', 'Mk_avg')):
            r[f'e{q}_{int(h)}h'] = (l1(d[key][i], ref[key][j], ref[key][j]) if ok else np.nan)
    r['valid'] = bool(ok and not r['gas_cap'] and not r['coag_cap']
                      and r['eN_6h'] <= TOL and r['eN_24h'] <= TOL)
    return r


def compare():
    missing = [pout(z, s, k) for z, _, _ in CELLS for s in SO2_PPBV for k in KINDS
               if not done(z, s, k)]
    missing += [pkind(z, s, BASE_KIND) for z, _, _ in CELLS for s in SO2_PPBV
                if not os.path.exists(pkind(z, s, BASE_KIND))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} runs missing (first {missing[:3]}); "
                                f"run --all first.")
    rows = []
    for z, T, _ in CELLS:
        for s in SO2_PPBV:
            ref_name, ref = reference(z, s)
            cell = [row(z, T, s, k, ref_name, ref) for k in ALL_KINDS]
            t_base = next(r['t_24h'] for r in cell if r['kind'] == BASE_KIND)
            for r in cell:
                r['x_base_24h'] = t_base / max(r['t_24h'], 1e-3)
            rows += cell
    np.save(os.path.join(OUTDIR, 'coag_interval_compare.npy'),
            np.array(rows, dtype=object), allow_pickle=True)

    print(f"{'z':>3s} {'SO2':>8s} {'kind':>8s} {'ref':>14s} {'t24 [s]':>8s} {'x ci0.01':>8s} "
          f"{'n_sub':>8s} {'L1N 6h':>7s} {'L1N 24h':>8s} {'L1M 24h':>8s}  flags")
    for r in rows:
        flags = ('' if r['finite'] else 'NaN ') + ('GASCAP ' if r['gas_cap'] else '') + \
                ('COAGCAP ' if r['coag_cap'] else '') + ('ok' if r['valid'] else '')
        print(f"{r['z']:3g} {r['so2']:8g} {r['kind']:>8s} {r['ref']:>14s} {r['t_24h']:8.1f} "
              f"{r['x_base_24h']:8.2f} {r['n_sub']:8d} {r['eN_6h']:7.3f} {r['eN_24h']:8.3f} "
              f"{r['eM_24h']:8.3f}  {flags}")
    print(f"\nper SO2 (sum of wall time to 24 h over altitudes; valid = finite, no cap, L1 N <= "
          f"{TOL} at 6 h and 24 h; {BASE_KIND} wall time is from another machine load):")
    for s in SO2_PPBV:
        parts = []
        for k in ALL_KINDS:
            sel = [r for r in rows if r['so2'] == s and r['kind'] == k]
            parts.append(f"{k} {sum(r['t_24h'] for r in sel):.0f}s {sum(r['valid'] for r in sel)}/6")
        print(f"  SO2 {s:g}: " + ' | '.join(parts))
    return rows


def all_jobs():
    return [(z, s, k) for s in sorted(SO2_PPBV, reverse=True) for k in KINDS
            for z, _, _ in CELLS if not done(z, s, k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--run', nargs=3, metavar=('ALT_KM', 'SO2_PPBV', 'KIND'))
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    if args.run:
        z, so2, kind = float(args.run[0]), float(args.run[1]), args.run[2]
        if z not in [x[0] for x in CELLS] or so2 not in SO2_PPBV or kind not in KINDS:
            raise ValueError(f"--run {args.run}: need ALT_KM in {[x[0] for x in CELLS]}, "
                             f"SO2 in {SO2_PPBV}, kind in {tuple(KINDS)} "
                             f"({BASE_KIND} is read from {DEA}, not run here)")
        run_one(z, so2, kind)
    elif args.compare:
        compare()
    elif args.all:
        jobs = all_jobs()
        os.makedirs(DOUT, exist_ok=True)
        print(f"{len(jobs)} runs to do", flush=True)

        def go(job):
            z, s, k = job
            with open(pout(z, s, k, 'log'), 'w') as f:
                rc = subprocess.run(
                    [sys.executable, '-m', 'benchmarks.python.gas_substep_coag_interval',
                     '--run', f'{z:g}', f'{s:g}', k], stdout=f,
                    stderr=subprocess.STDOUT).returncode
            print(f"job {z:g} km SO2={s:g} {k} {'done' if rc == 0 else 'FAILED'}", flush=True)
            return rc

        with ThreadPoolExecutor(int(os.environ.get('NJOBS', 1))) as ex:
            rcs = list(ex.map(go, jobs))
        failed = [j for j, rc in zip(jobs, rcs) if rc]
        if failed:
            raise RuntimeError(f"{len(failed)} runs failed (see {DOUT}/*.log): {failed}")
        compare()
    else:
        parser.error('give --all, --run or --compare')


if __name__ == '__main__':
    main()
