"""The 40-bin rerun of the 80-bin temperature grid (results/gas_substep/psd80/T*.npz),
for a like-for-like 40- vs 80-bin comparison.

Matched to the 80-bin files on disk: the same T x SO2 cells at STRAT's 5500 Pa,
the same initial lognormal state and 'H2SO4 cal' direction, the same run kinds
(ref, f0.3, f0.1, f0.03, dt_for, and dt60/dt0.25 where an 80-bin file exists),
and the same step size per kind (ref: 0.25 s up to 100 ppbv, else
gas_substep_convergence.ref_dt, as the 80-bin ref80_dt; dt_for: run_map.dt_for;
adaptive: outer dt 60 s). The 40-bin grid is make_grid(40, XK0, 2.0), whose bins
are the 80-bin grid's merged in pairs. Saved like the 80-bin files, with the
hour means (Nk_avg, Mk_avg, sampled once a minute) and the hourly series, so
the comparison is hour mean against hour mean, not snapshot against snapshot
(80-bin snapshots are noisy in bins 0-2).

    python -m benchmarks.python.gas_substep_psd40 --all             # every missing run
    python -m benchmarks.python.gas_substep_psd40 --run 217 1 ref   # one run (T K, SO2 ppbv, kind)
    python -m benchmarks.python.gas_substep_psd40 --compare         # 40 vs 80 bins, no runs

Add `--grid tropical` for the altitude grid (psd80_tropical/z<km>_<so2>_<kind>.npz, T and
P from common.TROPICAL_AFGL, 13-25 km x SO2, including the split-chemistry references
ref_split / ref_split_half): files go to psd40_tropical/, `--run` takes ALT_KM instead of
T_K, and `--kinds a,b` restricts the kinds (default: every kind that has an 80-bin file).
In both grids a run is made for every 80-bin file that exists, with the 80-bin file's dt.

Output: benchmarks/results/gas_substep/psd40/T<T>_<so2>_<kind>.npz. Cost: about
a fifth of the 80-bin run per cell; the reference at 100 ppmv is the longest.
`--all` runs NJOBS (default 6, the performance cores) at a time, heaviest
first; for an unattended run use `run_awake -m benchmarks.python.gas_substep_psd40 --all`.
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
from sensitivities.dunne_nuc.run_map import TEMPS, SO2_PPBV, directions
from benchmarks.python.gas_substep_convergence import build, l1, REPORT_HOURS, OUTDIR

GRIDS = {
    # name: (80-bin dir, 40-bin dir, cell file prefix, cells (key, T, pres))
    'T': (os.path.join(OUTDIR, 'psd80'), os.path.join(OUTDIR, 'psd40'), 'T',
          [(T, T, c.STRAT['pres']) for T in TEMPS]),
    'tropical': (os.path.join(OUTDIR, 'psd80_tropical'), os.path.join(OUTDIR, 'psd40_tropical'),
                 'z', [(z, T, P) for z, (T, P) in c.TROPICAL_AFGL.items()]),
}
# kind -> (gas_substep_frac or None for fixed dt, chemistry split around nucleation)
KINDS = {'ref_split_half': (None, True), 'ref_split': (None, True), 'ref': (None, False),
         'f0.3': (0.3, False), 'f0.1': (0.1, False), 'f0.03': (0.03, False),
         'dt_for': (None, False), 'dt60': (None, False), 'dt0.25': (None, False)}


class Grid:
    def __init__(self, name, kinds=None):
        self.name = name
        self.d80, self.d40, self.prefix, self.cells = GRIDS[name]
        self.kinds = tuple(KINDS) if kinds is None else tuple(kinds)
        unknown = set(self.kinds) - set(KINDS)
        if unknown:
            raise ValueError(f"unknown kinds {sorted(unknown)}; choose from {tuple(KINDS)}")

    def fname(self, key, so2, kind):
        return f'{self.prefix}{key:g}_{so2:g}_{kind}.npz'

    def p40(self, key, so2, kind):
        return os.path.join(self.d40, self.fname(key, so2, kind))

    def p80(self, key, so2, kind):
        return os.path.join(self.d80, self.fname(key, so2, kind))

    def kinds_for(self, key, so2):
        """The kinds with an 80-bin file in this cell."""
        return [k for k in self.kinds if os.path.exists(self.p80(key, so2, k))]

    def cell(self, key):
        return next(x for x in self.cells if x[0] == key)

    def done(self, key, so2, kind):
        p = self.p40(key, so2, kind)
        return os.path.exists(p) and {'Nk_avg', 'Mk_avg', 'Nk_series'} <= set(np.load(p).files)


def run_one(grid, key, so2, kind):
    _, T, pres = grid.cell(key)
    xk = jnp.asarray(make_grid(40, XK0, 2.0))
    frac, split = KINDS[kind]
    f80 = grid.p80(key, so2, kind)
    if not os.path.exists(f80):
        raise FileNotFoundError(f"No 80-bin run {f80} to match")
    dt = float(np.load(f80)['dt'])
    direction = jnp.asarray(directions(T)['H2SO4 cal'])
    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(so2, T, pres))
    out, tan, info = build(xk, dt, REPORT_HOURS, frac, pres=pres, split_chemistry=split)(
        direction, T, *state)
    os.makedirs(grid.d40, exist_ok=True)
    np.savez(grid.p40(key, so2, kind), T=T, pres=pres, so2=so2, kind=kind, dt=dt,
             hours=np.array(sorted(REPORT_HOURS)), Nk=out['Nk'], Mk=out['Mk'],
             nmass=out['nmass'], dNk=tan['Nk'], dnmass=tan['nmass'],
             Nk_avg=out['Nk_avg'], Mk_avg=out['Mk_avg'], dNk_avg=tan['Nk_avg'],
             Nk_series=out['Nk_series'], Mk_series=out['Mk_series'],
             dNk_series=tan['Nk_series'],
             t=info['t'], t_compile=info['t_compile'], n_sub=info['n_sub'],
             knuc=info['knuc'], cap=info['cap'])
    print(f"{grid.name} {key:g} T={T:g} P={pres:g} SO2={so2:g} {kind} dt={dt:g} "
          f"n_sub={info['n_sub']} cap={info['cap']} t={info['t'][-1]:.0f}s "
          f"(+{info['t_compile']:.0f}s compile)", flush=True)


def merge(x):
    return x.reshape(-1, 2, *x.shape[1:]).sum(1)


def compare(grid):
    """40 vs 80 bins, per kind and report hour: L1 number and dry mass of the
    40-bin distribution against the 80-bin one merged in pairs, from the hour
    means (and from the snapshots, for reference); N(>3 nm) ratio from the
    snapshots (NMASS). Raises if any 40-bin run is missing."""
    jobs = [(k_, s, kind) for k_, _, _ in grid.cells for s in SO2_PPBV
            for kind in grid.kinds_for(k_, s)]
    if not jobs:
        raise FileNotFoundError(f"No 80-bin files in {grid.d80}")
    missing = [grid.p40(*j) for j in jobs if not grid.done(*j)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} runs missing (first {missing[:3]}); "
                                f"run --all first.")
    rows = []
    for key, s, k in jobs:
        a, b = np.load(grid.p40(key, s, k)), np.load(grid.p80(key, s, k))
        if abs(float(a['dt']) - float(b['dt'])) > 1e-12:
            raise ValueError(f"{key} SO2={s} {k}: dt {a['dt']} vs {b['dt']}")
        for i, h in enumerate(a['hours']):
            N8, M8 = merge(b['Nk_avg'][i]), merge(b['Mk_avg'][i])
            S8, SM8 = merge(b['Nk'][i]), merge(b['Mk'][i])
            rows.append(dict(
                key=key, T=float(a['T']), so2=s, kind=k, hour=int(h),
                eN=l1(a['Nk_avg'][i], N8, N8), eM=l1(a['Mk_avg'][i], M8, M8),
                eN_snap=l1(a['Nk'][i], S8, S8), eM_snap=l1(a['Mk'][i], SM8, SM8),
                rN=a['Nk_avg'][i].sum() / N8.sum(),
                r3=float(a['nmass'][i][0] / b['nmass'][i][0])))
    tag = '' if grid.name == 'T' else f'_{grid.name}'
    np.save(os.path.join(OUTDIR, f'compare_40_80_hourmean{tag}.npy'),
            np.array(rows, dtype=object), allow_pickle=True)
    for k in grid.kinds:
        for h in REPORT_HOURS:
            r = [x for x in rows if x['kind'] == k and x['hour'] == h]
            if not r:
                continue
            g = lambda key: np.array([x[key] for x in r])
            print(f"{k:14s} {h:2d} h ({len(r)} cells) hour-mean L1 N med {np.median(g('eN')):.3f} "
                  f"max {g('eN').max():.3f} | M med {np.median(g('eM')):.3f} max "
                  f"{g('eM').max():.3f} | N40/N80 {g('rN').min():.2f}-{g('rN').max():.2f} | "
                  f"snapshot L1 N med {np.median(g('eN_snap')):.3f} | "
                  f"N3 {g('r3').min():.2f}-{g('r3').max():.2f}")
    return rows


def all_jobs(grid):
    # heaviest first: highest SO2, then in KINDS order
    return [(key, s, k) for s in sorted(SO2_PPBV, reverse=True) for k in KINDS
            for key, _, _ in grid.cells if k in grid.kinds_for(key, s)
            and not grid.done(key, s, k)]


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--grid', choices=tuple(GRIDS), default='T')
    parser.add_argument('--kinds', help='comma-separated subset of ' + ','.join(KINDS))
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--run', nargs=3, metavar=('T_K_or_ALT_KM', 'SO2_PPBV', 'KIND'))
    parser.add_argument('--compare', action='store_true')
    args = parser.parse_args()
    grid = Grid(args.grid, args.kinds.split(',') if args.kinds else None)
    if args.run:
        key, so2, kind = float(args.run[0]), float(args.run[1]), args.run[2]
        if key not in [x[0] for x in grid.cells] or so2 not in SO2_PPBV or kind not in KINDS:
            raise ValueError(f"--run {args.run}: need a cell in {[x[0] for x in grid.cells]}, "
                             f"SO2 in {SO2_PPBV}, kind in {tuple(KINDS)}")
        run_one(grid, key, so2, kind)
    elif args.compare:
        compare(grid)
    elif args.all:
        jobs = all_jobs(grid)
        os.makedirs(grid.d40, exist_ok=True)
        print(f"{len(jobs)} runs to do", flush=True)

        def go(job):
            key, s, k = job
            with open(grid.p40(key, s, k).replace('.npz', '.log'), 'w') as f:
                rc = subprocess.run(
                    [sys.executable, '-m', 'benchmarks.python.gas_substep_psd40',
                     '--grid', grid.name, '--run', f'{key:g}', f'{s:g}', k], stdout=f,
                    stderr=subprocess.STDOUT).returncode
            print(f"job {key:g} SO2={s:g} {k} {'done' if rc == 0 else 'FAILED'}", flush=True)
            return rc

        with ThreadPoolExecutor(int(os.environ.get('NJOBS', 6))) as ex:
            rcs = list(ex.map(go, jobs))
        failed = [j for j, rc in zip(jobs, rcs) if rc]
        if failed:
            raise RuntimeError(f"{len(failed)} runs failed (see {grid.d40}/*.log): {failed}")
        compare(grid)
    else:
        parser.error('give --all, --run or --compare')


if __name__ == '__main__':
    main()
