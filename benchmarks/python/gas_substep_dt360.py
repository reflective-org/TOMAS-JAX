"""Adaptive gas sub-steps at a long outer dt (default 360 s; --outer-dt 600) against
constant dt = 60 s and the same constant dt, on the tropical altitude grid
(13-25 km x SO2, 40 bins, 24 h).

Three kinds per cell (N = the outer dt in s), all forward-only (no tangent), same
initial state and settings as psd40_tropical:
    f0.1_N    make_step(gas_substep='adaptive', gas_substep_frac=0.1), outer dt = N s
    dt60      constant dt = 60 s
    dtN       constant dt = N s
Hour means are sampled every N s for all three (3600/N samples per hour; the
psd40_tropical references sample every 60 s, so a sampling difference is in the
L1 values, larger for larger N). Error is the hour-mean L1 number and dry mass against the
split-chemistry reference psd40_tropical/ref_split, at 6 h and 24 h; the table
adds wall time (compile excluded), the speed-up against dt60, the adaptive
sub-step count and the cap flag. A run that goes NaN or hits `gas_max_substeps`
is reported as such, not dropped. The existing f0.1 at outer dt = 60 s
(psd40_tropical/f0.1) is listed for time and error alongside.

    python -m benchmarks.python.gas_substep_dt360 --all              # missing runs
    python -m benchmarks.python.gas_substep_dt360 --run 13 100 dt60   # one run (ALT_KM, SO2 ppbv, kind)
    python -m benchmarks.python.gas_substep_dt360 --compare          # table + dt360_compare.npy

Options: --outer-dt N (360 default; the kinds, sample interval and output directory
follow, e.g. dt600_tropical; the gas sub-step cap is 131072 above 360 s: 17 km /
100 ppmv needs ~73000 per 600 s), --hours H (one report hour only, directory suffix
_<H>h), --so2-max X (only SO2 < X ppbv in --all and --compare), --repeats R (wall time
= the minimum of up to R executions, stopping at 5 s timed),
--split-chemistry (constant-dt kinds split chemistry around nucleation as the adaptive
kind does; directory suffix _split, copy the adaptive runs in from the unsplit directory).

Wall times are only comparable when the runs share a machine load: --all runs
NJOBS (default 1) at a time, so leave it at 1 for a timing comparison and raise
it only for the error check. For an unattended run use
`run_awake -m benchmarks.python.gas_substep_dt360 --all`.

Output: benchmarks/results/gas_substep/dt<N>_tropical[_<H>h]/z<km>_<so2>_<kind>.npz.
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

D40 = os.path.join(OUTDIR, 'psd40_tropical')
HOURS = tuple(REPORT_HOURS)
REPEATS = 1
SO2S = list(SO2_PPBV)
CELLS = [(z, T, P) for z, (T, P) in c.TROPICAL_AFGL.items()]
KINDS = {}


def configure(outer_dt=360.0, hours=None, so2_max=None, split=False, with_f60=False):
    """Set the module's run settings (other modules import the 360 s defaults)."""
    global OUTER_DT, SAMPLE_S, GAS_MAX_SUBSTEPS, DOUT, HOURS, SO2S, SPLIT
    SPLIT = bool(split)
    OUTER_DT = SAMPLE_S = float(outer_dt)
    n = f'{OUTER_DT:g}'
    # sub-steps are set by the H2SO4 relaxation time, so the cap scales with the outer dt:
    # 17 km / 100 ppmv averages ~43900 per 360 s (~73000 per 600 s)
    GAS_MAX_SUBSTEPS = int(os.environ.get('GAS_MAX_SUBSTEPS', 65536 if OUTER_DT <= 360 else 131072))
    DOUT = os.path.join(OUTDIR, f'dt{n}_tropical')
    if hours:
        HOURS, DOUT = (hours,), DOUT + f'_{hours}h'
    if SPLIT:
        DOUT += '_split'
    SO2S = [x for x in SO2_PPBV if so2_max is None or x < so2_max]
    # kind -> (dt [s], gas_substep_frac or None)
    KINDS.clear()
    KINDS.update({f'f0.1_{n}': (OUTER_DT, 0.1), 'dt60': (60.0, None), f'dt{n}': (OUTER_DT, None)})
    if with_f60:    # forward-only adaptive at the 60 s outer step (the psd40 f0.1 runs carry a tangent)
        KINDS['f0.1_60'] = (60.0, 0.1)
    if 3600.0 % OUTER_DT or OUTER_DT % 60.0:
        raise ValueError(f"outer dt {OUTER_DT} must divide 3600 s and be a multiple of 60 s")


configure()


def fname(z, so2, kind):
    return f'z{z:g}_{so2:g}_{kind}.npz'


def pout(z, so2, kind):
    return os.path.join(DOUT, fname(z, so2, kind))


def done(z, so2, kind):
    p = pout(z, so2, kind)
    return os.path.exists(p) and 'Nk_avg' in np.load(p).files


def run_one(z, so2, kind):
    _, T, pres = next(x for x in CELLS if x[0] == z)
    dt, frac = KINDS[kind]
    xk = jnp.asarray(make_grid(40, XK0, 2.0))
    direction = jnp.asarray(directions(T)['H2SO4 cal'])
    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(so2, T, pres))
    # fd_alpha = 0 runs forward only at the unmoved coefficients
    run = build(xk, dt, HOURS, frac, fd_alpha=0.0, pres=pres, sample_s=SAMPLE_S,
                gas_max_substeps=GAS_MAX_SUBSTEPS,
                split_chemistry=SPLIT and frac is None)   # adaptive always splits
    # Repeated executions of the one compiled program (same state, same result);
    # the first also carries t_compile. Wall time = the minimum, the least disturbed.
    # Up to REPEATS executions, stopping once 5 s of execution is timed: ms-long runs are
    # averaged over noise, runs of 5 s or more are measured once. (Runs made before
    # 2026-11-17 17:55 took at least 3 executions.)
    runs = [run(direction, T, *state)]
    while len(runs) < REPEATS and sum(r[2]['t'][-1] for r in runs) < 5.0:
        runs.append(run(direction, T, *state))
    out, _, info = runs[0]
    t_all = np.array([r[2]['t'] for r in runs])
    os.makedirs(DOUT, exist_ok=True)
    np.savez(pout(z, so2, kind), T=T, pres=pres, so2=so2, kind=kind, dt=dt,
             split_chemistry=bool(frac is not None or SPLIT),
             hours=np.array(sorted(HOURS)), Nk=out['Nk'], Mk=out['Mk'],
             nmass=out['nmass'], Nk_avg=out['Nk_avg'], Mk_avg=out['Mk_avg'],
             Nk_series=out['Nk_series'], Mk_series=out['Mk_series'],
             t=t_all.min(0), t_all=t_all, t_compile=info['t_compile'], n_sub=info['n_sub'],
             knuc=info['knuc'], cap=info['cap'])
    print(f"z={z:g} km T={T:g} P={pres:g} SO2={so2:g} {kind} dt={dt:g} n_sub={info['n_sub']} "
          f"cap={info['cap']} t={t_all.min(0)[-1]:.4f}s (min of {len(runs)}, "
          f"max/min {t_all[:, -1].max() / t_all[:, -1].min():.2f}, "
          f"+{info['t_compile']:.1f}s compile)", flush=True)


def need(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing {path}. Run the simulations first.")
    return np.load(path)


def compare():
    """Per cell and kind: hour-mean L1 error (number, dry mass) at 6 and 24 h
    against psd40_tropical ref_split, wall time, speed-up vs dt60 and vs the
    adaptive run. Raises if any run or reference is missing."""
    missing = [pout(z, s, k) for z, _, _ in CELLS for s in SO2S for k in KINDS
               if not done(z, s, k)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} runs missing (first {missing[:3]}); "
                                f"run --all first.")
    rows = []
    hh = '_'.join(f'{h}h' for h in HOURS)
    h0, h1 = HOURS[0], HOURS[-1]
    for z, T, _ in CELLS:
        for s in SO2S:
            ref = need(os.path.join(D40, fname(z, s, 'ref_split')))
            runs = {k: np.load(pout(z, s, k)) for k in KINDS}
            t60 = float(runs['dt60']['t'][-1])
            extra = []
            if HOURS == tuple(REPORT_HOURS) and 'f0.1_60' not in KINDS:   # adaptive, outer 60 s (jvp timing: tangent included)
                extra = [('f0.1_60', need(os.path.join(D40, fname(z, s, 'f0.1'))))]
            for k, d in extra + list(runs.items()):
                row = dict(z=z, T=T, so2=s, kind=k, t=float(d['t'][-1]),
                           n_sub=int(d['n_sub']), cap=bool(d['cap']),
                           speedup_vs_dt60=t60 / float(d['t'][-1]))
                ok = bool(np.all(np.isfinite(d['Nk_avg'])) and np.all(np.isfinite(d['Mk_avg'])))
                row['finite'] = ok
                for i, h in enumerate(d['hours']):
                    j = list(ref['hours']).index(h)
                    for q, key in (('N', 'Nk_avg'), ('M', 'Mk_avg')):
                        row[f'e{q}_{int(h)}h'] = (l1(d[key][i], ref[key][j], ref[key][j])
                                                  if ok else np.nan)
                rows.append(row)
    n = f'{OUTER_DT:g}'
    tag = os.path.join(OUTDIR, f'dt{n}_compare.npy' if HOURS == tuple(REPORT_HOURS)
                       else f'dt{n}_compare_{hh}' + ('_split' if SPLIT else '') + '.npy')
    np.save(tag, np.array(rows, dtype=object), allow_pickle=True)

    print(f"{'z km':>5s} {'SO2 ppbv':>9s} {'kind':>9s} {'t [s]':>8s} {'x dt60':>7s} "
          f"{'n_sub':>9s} {f'L1 N {h0}h':>8s} {f'L1 N {h1}h':>9s} {f'L1 M {h1}h':>9s}  flags")
    for r in rows:
        flags = ('' if r['finite'] else 'NaN ') + ('CAP' if r['cap'] else '')
        print(f"{r['z']:5g} {r['so2']:9g} {r['kind']:>9s} {r['t']:8.4f} "
              f"{r['speedup_vs_dt60']:7.2f} {r['n_sub']:9d} {r[f'eN_{h0}h']:8.3f} "
              f"{r[f'eN_{h1}h']:9.3f} {r[f'eM_{h1}h']:9.3f}  {flags}")
    print(f"\nsummary (SO2 < {max(SO2S) * 10:g} ppbv grid: {len(SO2S)} SO2 values, kinds "
          f"{', '.join(KINDS)}; finite and uncapped cells only for time and error):")
    for k in (['f0.1_60'] if HOURS == tuple(REPORT_HOURS) and 'f0.1_60' not in KINDS else []) + list(KINDS):
        sel = [r for r in rows if r['kind'] == k]
        good = [r for r in sel if r['finite'] and not r['cap']]
        sp = np.array([r['speedup_vs_dt60'] for r in good])
        print(f"{k:>9s}: {len(sel) - len(good)} NaN/capped of {len(sel)}; "
              f"time vs dt60 median {np.median(sp):.2f}x (min {sp.min():.2f}, max {sp.max():.2f}); "
              f"L1 N {h1}h median {np.median([r[f'eN_{h1}h'] for r in good]):.3f} "
              f"max {max(r[f'eN_{h1}h'] for r in good):.3f}")
    return rows


def all_jobs():
    # heaviest first: highest SO2
    return [(z, s, k) for s in sorted(SO2S, reverse=True) for k in KINDS
            for z, _, _ in CELLS if not done(z, s, k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--run', nargs=3, metavar=('ALT_KM', 'SO2_PPBV', 'KIND'))
    parser.add_argument('--compare', action='store_true')
    parser.add_argument('--hours', type=int, help='simulate and report this one hour only '
                        '(default: the report hours 6 and 24); results go to dt360_tropical_<h>h')
    parser.add_argument('--repeats', type=int, default=1, help='executions per run; the wall '
                        'time is their minimum (stopping at 5 s timed; use ~10)')
    parser.add_argument('--outer-dt', type=float, default=360.0, help='outer dt [s] of the '
                        'adaptive and long constant-dt kinds (360 or 600)')
    parser.add_argument('--split-chemistry', action='store_true', help='constant-dt kinds run '
                        'make_step(split_chemistry=True) like the adaptive kind (always split); '
                        'directory suffix _split. Adaptive runs are unchanged: copy them in')
    parser.add_argument('--with-f60', action='store_true', help='also the adaptive f = 0.1 kind at '
                        'outer dt = 60 s (f0.1_60, forward only)')
    parser.add_argument('--so2-max', type=float, help='only SO2 below this [ppbv] in --all/--compare')
    args = parser.parse_args()
    global REPEATS
    configure(args.outer_dt, args.hours, args.so2_max, args.split_chemistry, args.with_f60)
    REPEATS = args.repeats
    if args.run:
        z, so2, kind = float(args.run[0]), float(args.run[1]), args.run[2]
        if z not in [x[0] for x in CELLS] or so2 not in SO2_PPBV or kind not in KINDS:
            raise ValueError(f"--run {args.run}: need ALT_KM in {[x[0] for x in CELLS]}, "
                             f"SO2 in {SO2_PPBV}, kind in {tuple(KINDS)}")
        run_one(z, so2, kind)
    elif args.compare:
        compare()
    elif args.all:
        jobs = all_jobs()
        os.makedirs(DOUT, exist_ok=True)
        print(f"{len(jobs)} runs to do", flush=True)

        def go(job):
            z, s, k = job
            with open(pout(z, s, k).replace('.npz', '.log'), 'w') as f:
                rc = subprocess.run(
                    [sys.executable, '-m',
                     'benchmarks.python.' + os.path.splitext(os.path.basename(__file__))[0],
                     '--run', f'{z:g}', f'{s:g}', k, '--repeats', str(REPEATS),
                     '--outer-dt', f'{OUTER_DT:g}']
                    + (['--split-chemistry'] if SPLIT else [])
                    + (['--with-f60'] if 'f0.1_60' in KINDS else [])
                    + (['--hours', str(HOURS[0])] if len(HOURS) == 1 else []), stdout=f,
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
