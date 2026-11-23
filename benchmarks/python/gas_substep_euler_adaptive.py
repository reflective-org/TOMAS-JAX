"""euler_adaptive coagulation inside adaptive gas sub-steps and at constant dt, on the
tropical altitude grid (13-25 km x SO2, 40 bins, 24 h): accuracy and wall time.

Question: does make_step(coag_solver='euler_adaptive') (Fortran multicoag.f sub-step
limits, kernel recomputed) let constant dt = 60 s / 360 s survive and stay accurate
at high SO2, and what does it cost inside the adaptive gas sub-steps against the
default euler_fixed (10 equal sub-steps, kernel frozen)?

New runs (forward-only, same state and settings as gas_substep_dt360, hour means
sampled every 360 s):
    f0.1_60      euler_fixed, adaptive f = 0.1, outer 60 s   (baseline; no forward-only run exists)
    ea_f0.1_60   euler_adaptive, adaptive f = 0.1, outer 60 s
    ea_f0.1_360  euler_adaptive, adaptive f = 0.1, outer 360 s
    ea_dt60      euler_adaptive, constant dt = 60 s
    ea_dt360     euler_adaptive, constant dt = 360 s
Baselines read from dt360_tropical (euler_fixed): f0.1_360, dt60, dt360. Reference:
psd40_tropical/ref_split (euler_fixed, small constant dt).

Reported per cell and kind: hour-mean L1 number and dry mass at 6 h and 24 h against the
reference, wall time to 6 h and 24 h (compile excluded), time relative to the euler_fixed
run of the same step (ea_f0.1_360 vs f0.1_360, ea_dt60 vs dt60, ea_dt360 vs dt360,
ea_f0.1_60 vs f0.1_60), and flags: NaN, gas sub-step cap, and CoagSubstepCapWarning
(found in the run's .log: the cap drops the rest of dt, so N comes out too high and the
run does not count as valid). Pass: L1 <= 0.05 at 6 h and 24 h, finite, no cap.

    python -m benchmarks.python.gas_substep_euler_adaptive --run 13 100 ea_dt60   # one run
    python -m benchmarks.python.gas_substep_euler_adaptive --all                  # missing runs, then --compare
    python -m benchmarks.python.gas_substep_euler_adaptive --compare              # table + euler_adaptive_compare.npy

SPLIT=1 reruns the constant-dt kinds (dt60, dt360, ea_dt60, ea_dt360) with
make_step(split_chemistry=True), like the reference (the default runs are unsplit; the
adaptive kinds always split): output in euler_adaptive_tropical_split/, compare file
euler_adaptive_compare_split.npy (combines with BINS=80: ..._80_split).
    SPLIT=1 python -m benchmarks.python.gas_substep_euler_adaptive --all

BINS=80 (environment variable, default 40) runs the same study on the 80-bin grid (mass
ratio sqrt 2, linear sub-bin coagulation): output in euler_adaptive_tropical_80/, reference
psd80_tropical/ref_split, L1 on hour means with bin pairs merged to the 40-bin grid (as
gas_substep_psd80), and the euler_fixed baselines f0.1_360, dt60, dt360 are run here too
(no 80-bin dt360 runs exist), so 8 kinds and 384 runs. An 80-bin run costs ~5x a 40-bin one.
    BINS=80 python -m benchmarks.python.gas_substep_euler_adaptive --all

--all runs NJOBS (default 1, for timing) at a time, highest SO2 first. The constant-dt
euler_adaptive runs at 100 ppmv are the unknown cost: set COAG_MAX_SUBSTEPS (default
10000, make_step's forward default) and read the cap flag before raising it. Unattended:
`run_awake -m benchmarks.python.gas_substep_euler_adaptive --all`.

Output: benchmarks/results/gas_substep/euler_adaptive_tropical/z<km>_<so2>_<kind>.{npz,log}.
"""
import os
import sys
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import jax.numpy as jnp

from tomas_jax.core.config import make_grid, make_grid_80bin, XK0
from sensitivities.dunne_nuc import common as c
from sensitivities.dunne_nuc.run_map import SO2_PPBV, directions
from benchmarks.python.gas_substep_convergence import build, l1, REPORT_HOURS, OUTDIR
from benchmarks.python.gas_substep_dt360 import (CELLS, D40, DOUT as DBASE, SAMPLE_S,
                                                 GAS_MAX_SUBSTEPS, fname, need)

BINS = int(os.environ.get('BINS', 40))
if BINS not in (40, 80):
    raise ValueError(f"BINS={BINS}: 40 or 80")
MERGE = BINS // 40      # grid bins per 40-bin bin in the L1
SPLIT = os.environ.get('SPLIT', '0') == '1'
SFX = ('' if BINS == 40 else f'_{BINS}') + ('_split' if SPLIT else '')
DOUT = os.path.join(OUTDIR, 'euler_adaptive_tropical' + SFX)
DREF = D40 if BINS == 40 else os.path.join(OUTDIR, 'psd80_tropical')
COAG_MAX_SUBSTEPS = int(os.environ.get('COAG_MAX_SUBSTEPS', 10000))
TOL = 0.05
# kind -> (dt [s], gas_substep_frac or None, coag_solver)
KINDS = {'f0.1_60': (60.0, 0.1, 'euler_fixed'),
         'ea_f0.1_60': (60.0, 0.1, 'euler_adaptive'),
         'ea_f0.1_360': (360.0, 0.1, 'euler_adaptive'),
         'ea_dt60': (60.0, None, 'euler_adaptive'),
         'ea_dt360': (360.0, None, 'euler_adaptive'),
         # error-controlled Heun coagulation (make_step coag_solver='heun_adaptive')
         'ha_f0.1_360': (360.0, 0.1, 'heun_adaptive')}   # h0 = dt (hs_*: h0 = 1 s, older runs)
HEUN_RTOL = {}       # kind -> coag_heun_rtol (default 3e-3)
# kind -> (directory, file kind) of the euler_fixed run it is compared with
PAIR = {'ea_f0.1_60': 'f0.1_60', 'ea_f0.1_360': 'f0.1_360', 'ea_dt60': 'dt60',
        'ea_dt360': 'dt360', 'ha_f0.1_360': 'f0.1_360'}
BASE = {'f0.1_60': DOUT, 'f0.1_360': DBASE, 'dt60': DBASE, 'dt360': DBASE}
FIXED = ['f0.1_360', 'dt60', 'dt360']       # euler_fixed baselines
if BINS == 80:      # no 80-bin baselines exist: run them here
    KINDS.update({'f0.1_360': (360.0, 0.1, 'euler_fixed'), 'dt60': (60.0, None, 'euler_fixed'),
                  'dt360': (360.0, None, 'euler_fixed')})
    BASE = {k: DOUT for k in BASE}
ALL_KINDS = FIXED + [k for k in KINDS if k not in FIXED]
if SPLIT:   # constant-dt kinds with split_chemistry=True; the adaptive kinds always split
    KINDS = {'dt60': (60.0, None, 'euler_fixed'), 'dt360': (360.0, None, 'euler_fixed'),
             'ea_dt60': (60.0, None, 'euler_adaptive'), 'ea_dt360': (360.0, None, 'euler_adaptive')}
    PAIR = {'ea_dt60': 'dt60', 'ea_dt360': 'dt360'}
    BASE = {'dt60': DOUT, 'dt360': DOUT}
    ALL_KINDS = list(KINDS)


def merged(x):
    return np.asarray(x).reshape(-1, MERGE).sum(1)


def pout(z, so2, kind, ext='npz'):
    return os.path.join(DOUT, fname(z, so2, kind).replace('.npz', f'.{ext}'))


def pkind(z, so2, kind):
    return os.path.join(BASE.get(kind, DOUT), fname(z, so2, kind))


def done(z, so2, kind):
    p = pout(z, so2, kind)
    return os.path.exists(p) and 'Nk_avg' in np.load(p).files


def run_one(z, so2, kind):
    _, T, pres = next(x for x in CELLS if x[0] == z)
    dt, frac, solver = KINDS[kind]
    xk = jnp.asarray(make_grid(40, XK0, 2.0) if BINS == 40 else make_grid_80bin())
    direction = jnp.asarray(directions(T)['H2SO4 cal'])
    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(so2, T, pres))
    out, _, info = build(xk, dt, REPORT_HOURS, frac, fd_alpha=0.0, pres=pres,
                         sample_s=SAMPLE_S, gas_max_substeps=GAS_MAX_SUBSTEPS,
                         coag_solver=solver, coag_max_substeps=COAG_MAX_SUBSTEPS,
                         coag_heun_rtol=HEUN_RTOL.get(kind),
                         split_chemistry=SPLIT and frac is None)(
        direction, T, *state)
    os.makedirs(DOUT, exist_ok=True)
    np.savez(pout(z, so2, kind), T=T, pres=pres, so2=so2, kind=kind, dt=dt,
             coag_solver=solver, split_chemistry=bool(frac is not None or SPLIT),
             hours=np.array(sorted(REPORT_HOURS)), Nk=out['Nk'],
             Mk=out['Mk'], nmass=out['nmass'], Nk_avg=out['Nk_avg'], Mk_avg=out['Mk_avg'],
             Nk_series=out['Nk_series'], Mk_series=out['Mk_series'], t=info['t'],
             t_compile=info['t_compile'], n_sub=info['n_sub'], knuc=info['knuc'],
             cap=info['cap'])
    print(f"z={z:g} km SO2={so2:g} {kind} dt={dt:g} {solver} n_sub={info['n_sub']} "
          f"cap={info['cap']} t={info['t'][-1]:.0f}s (+{info['t_compile']:.0f}s compile)",
          flush=True)


def coag_cap_hit(z, so2, kind):
    """CoagSubstepCapWarning in the run's log (the callback fires after the step)."""
    p = pout(z, so2, kind, 'log')
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}: the coagulation cap flag is read from the log. "
                                f"Run through --all (or redirect --run output to it).")
    return 'CoagSubstepCapWarning' in open(p).read()


def row(z, T, so2, kind, ref):
    d = np.load(pkind(z, so2, kind))
    ok = bool(np.all(np.isfinite(d['Nk_avg'])) and np.all(np.isfinite(d['Mk_avg'])))
    r = dict(z=z, T=T, so2=so2, kind=kind, finite=ok, gas_cap=bool(d['cap']),
             coag_cap=coag_cap_hit(z, so2, kind) if kind in KINDS and kind != 'f0.1_60'
             and KINDS[kind][2] in ('euler_adaptive', 'heun_adaptive') else False)
    for i, h in enumerate(d['hours']):
        j = list(ref['hours']).index(h)
        r[f't_{int(h)}h'] = float(d['t'][i])
        for q, key in (('N', 'Nk_avg'), ('M', 'Mk_avg')):
            r[f'e{q}_{int(h)}h'] = l1(merged(d[key][i]), merged(ref[key][j]), merged(ref[key][j])) if ok else np.nan
    r['valid'] = bool(ok and not r['gas_cap'] and not r['coag_cap']
                      and r['eN_6h'] <= TOL and r['eN_24h'] <= TOL)
    return r


def compare():
    missing = [pout(z, s, k) for z, _, _ in CELLS for s in SO2_PPBV for k in KINDS
               if not done(z, s, k)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} runs missing (first {missing[:3]}); "
                                f"run --all first.")
    rows = []
    for z, T, _ in CELLS:
        for s in SO2_PPBV:
            ref = need(os.path.join(DREF, fname(z, s, 'ref_split')))
            for k in ALL_KINDS:
                r = row(z, T, s, k, ref)
                if k in PAIR:
                    base = np.load(pkind(z, s, PAIR[k]))
                    for h, i in ((6, 0), (24, 1)):
                        r[f'x_fixed_{h}h'] = float(base['t'][i]) / max(r[f't_{h}h'], 1e-3)
                if k.startswith('ha'):      # against the Fortran-limit euler_adaptive run
                    ea = np.load(pkind(z, s, 'ea_f0.1_360'))
                    r['x_ea_24h'] = float(ea['t'][1]) / max(r['t_24h'], 1e-3)
                    r['eN_ea_24h'] = l1(merged(ea['Nk_avg'][1]), merged(ref['Nk_avg'][1]),
                                        merged(ref['Nk_avg'][1]))
                rows.append(r)
    np.save(os.path.join(OUTDIR, 'euler_adaptive_compare' + SFX + '.npy'),
            np.array(rows, dtype=object), allow_pickle=True)

    print(f"{'z':>3s} {'SO2':>8s} {'kind':>12s} {'t24 [s]':>8s} {'fixed/ea':>8s} "
          f"{'L1N 6h':>7s} {'L1N 24h':>8s} {'L1M 24h':>8s}  flags")
    for r in rows:
        flags = ('' if r['finite'] else 'NaN ') + ('GASCAP ' if r['gas_cap'] else '') + \
                ('COAGCAP ' if r['coag_cap'] else '') + ('ok' if r['valid'] else '')
        print(f"{r['z']:3g} {r['so2']:8g} {r['kind']:>12s} {r['t_24h']:8.1f} "
              f"{r.get('x_fixed_24h', float('nan')):8.2f} {r['eN_6h']:7.3f} "
              f"{r['eN_24h']:8.3f} {r['eM_24h']:8.3f}  {flags}")
    for hk in [k for k in ALL_KINDS if k.startswith('ha')]:
        ha = [r for r in rows if r['kind'] == hk]
        print(f"\n{hk} vs ea_f0.1_360 (euler_adaptive): ea/ha time and L1 N 24 h of each")
        for r in ha:
            print(f"  {r['z']:3g} km SO2 {r['so2']:8g}: x{r['x_ea_24h']:5.2f}  "
                  f"ha {r['eN_24h']:.3f}  ea {r['eN_ea_24h']:.3f}")
    print("\nper SO2 (sum of wall time to 24 h over altitudes; valid = finite, no cap, L1 N <= "
          f"{TOL} at 6 h and 24 h):")
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
            with open(pout(z, s, k, 'log'), 'w') as f:
                rc = subprocess.run(
                    [sys.executable, '-m', 'benchmarks.python.gas_substep_euler_adaptive',
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
