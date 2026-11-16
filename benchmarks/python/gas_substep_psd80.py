"""The gas_substep_convergence grid on 80 bins (mass ratio sqrt 2), 6 h and
24 h, over altitude x SO2: each altitude takes T and P together from the AFGL
tropical atmosphere (common.TROPICAL_AFGL, 13-25 km). The reference is a
fixed-dt run with the step's chemistry split around nucleation (SPLIT_KINDS,
reference()); the unsplit 'ref' is kept for comparison. For every cell, a fixed-dt reference, adaptive f = 0.3, 0.1, 0.03,
run_map.dt_for and constant dt = 60 s and 0.25 s in every cell, each a forward
pass plus a jvp along 'H2SO4 cal' (KINDS). A constant-dt run that is the same
run as dt_for or the reference in a cell (60 s up to 10 ppbv, 0.25 s up to
100 ppbv) is read from that file (alias), not run again.
Used by plot_gas_substep_psd (reference and f = 0.1) and plot_gas_substep_grid.

    python -m benchmarks.python.gas_substep_psd80 --all            # every missing run
    python -m benchmarks.python.gas_substep_psd80 --run 20 1e4 ref   # one run (altitude km)
    python -m benchmarks.python.gas_substep_psd80 --check          # pass/fail only

Pass (check, also run at the end of --all), at every report hour, f = 0.1
against the reference: the gas_substep_convergence L1 distances (number and
dry mass) <= TOL, computed from the hour means with pairs of 80-bin
bins merged (MERGE; the merged bins are the 40-bin grid's), plus no gas-clamp
firing and no sub-step cap hit. The per-80-bin distances are printed, not
tested: at low SO2 and 200 K, fixed 0.125 s and 0.25 s put the same particles
one 80-bin bin apart in bins 0-2 for hours (L1 sensitivity 0.160 per 80-bin
bin, 0.010 merged; 2026-11-07). The snapshots are not tested either: at 80 bins
the two smallest bins trade particles from minute to minute, and a snapshot
catches that exchange at a different moment in each run (217 K / 10 ppmv,
f = 0.1: snapshot L1 number up to 0.082, hour mean <= 0.014, 2026-11-06).
The L1 sensitivity is printed, not tested (2026-11-12): at 15 km / 100 ppbv,
6 h, a central FD (+-FD_H) put every run's tangent, the references and
f = 0.1 alike, 0.16-0.28 off in the merged bins 1-2 (~2-4 nm), where the
one-sided differences have opposite signs (MNFIX bin selection, which the
tangent does not see), while the FDs of split and unsplit chemistry agreed
to 0.004 and the total agreed (FD 0.548, jvp 0.553); and the tangent blows
up in some cells (docs/prescribed_j.md). Any failure exits non-zero.

Finite differences (FD_CELLS, (altitude, SO2) cells): on the earlier
temperature grid at 5500 Pa, the 80-bin linear sub-bin coagulation tangent blew
up after ~12 h at 223 K and 240 K / 100 ppmv in every run, the reference
included (sum|dN|/sum N 1e6-1e45 at 24 h; docs/prescribed_j.md). Where it does
on this grid (the printed dN reaches ~1e3 or more), add the cell. There every
kind also runs forward only at alpha = +-FD_H along the direction (runs
<kind>_fd+ and <kind>_fd-), and load_run replaces the jvp tangents by the
central difference. FD_H = 0.05 agreed with the jvp at 6 h in these cells
(L1 within 0.01, docs/prescribed_j.md); --check prints FD against the jvp at
every hour where the jvp is still valid.

Reference step (ref80_dt): 0.25 s up to 100 ppbv, else gas_substep_convergence.ref_dt
(0.5 s at 1 ppmv, 0.1 s at 10 ppmv, 0.02 s at 100 ppmv). At 208 K / 0.1 ppbv, 6 h,
80 bins, fixed 1 s was 0.17 off a 0.125 s run in L1 sensitivity, 0.5 s 0.056 and
0.25 s 0.014 (2026-11-06); the 40-bin reference's 15 s is coarser still.
An 80-bin run costs about 5x the
40-bin one: the reference ~65 min per cell at 10 ppmv and ~5 h at 100 ppmv,
f = 0.03 at 190 K / 100 ppmv ~3 h (single process).
`--all` skips runs whose file exists and runs NJOBS (default 11) at a time,
heaviest first; for an unattended run use `run_awake -m ...`.
Output: benchmarks/results/gas_substep/psd80_tropical/z<km>_<so2>_<kind>.npz
(z, T, pres saved) with, per
report hour, Nk [#/cm3], dry Mk per bin, NMASS N(>3), N(>12) and their tangents,
the hour means Nk_avg, Mk_avg (and dNk_avg; sampled once a minute), the hour
means of every hour Nk_series, Mk_series, dNk_series (SERIES_KINDS need them;
--all reruns those without),
t (execution time to that hour, without compilation) and t_compile.
"""
import os
import sys
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.core.config import make_grid_80bin
from sensitivities.dunne_nuc import common as c
from sensitivities.dunne_nuc.run_map import SO2_PPBV, directions, dt_for
from benchmarks.python.gas_substep_convergence import (
    build, ref_dt, l1, REPORT_HOURS, OUTDIR, TOL, PASS_FRAC)

PSD80 = os.path.join(OUTDIR, 'psd80_tropical')
ALTITUDES = c.TROPICAL_AFGL     # km -> (T [K], P [Pa])
# run kind: (gas_substep_frac or None for fixed dt, outer dt [s] or None for its rule)
KINDS = {'ref_split_half': (None, None), 'ref_split': (None, None),
         'ref': (None, None), 'f0.03': (0.03, 60.0), 'f0.1': (0.1, 60.0),
         'f0.3': (0.3, 60.0), 'dt_for': (None, None),
         'dt60': (None, 60.0), 'dt0.25': (None, 0.25)}
# Fixed-dt kinds with the step's SO2 chemistry split around nucleation
# (make_step(split_chemistry=True)): ref_split at ref80_dt, ref_split_half at
# half of it, only at HALF_SO2 (100 ppbv, 1, 10 and 100 ppmv: 0.125, 0.25,
# 0.05 and 0.01 s; ~15 h a run at 100 ppmv). At 100 ppbv because the split
# 0.25 s run's 6 h sensitivity at 15 km was 0.08 from both f = 0.1 and the
# unsplit 0.25 s run (2026-11-12). The unsplit 'ref' is first order in dt through the H2SO4 that
# nucleation sees and was ~2% high in N at 13 km / 1 ppmv; split 0.5 s is
# within 0.0014 (L1 number, 6 h) of split 0.125 s there (2026-11-11).
SPLIT_KINDS = ('ref_split', 'ref_split_half')
HALF_SO2 = (100.0, 1e3, 1e4, 1e5)
# kinds the figures plot: their runs (or the run they alias) must have the hourly series
SERIES_KINDS = ('ref', 'f0.1', 'f0.3', 'dt60', 'dt0.25') + SPLIT_KINDS
MERGE = 2               # 80-bin bins per tested bin: the 40-bin grid's bins
INVALID_TANGENT = 1e3   # reference sum|dN|/sum N above this: tangent blew up (valid cells <= 2.4)
FD_CELLS = ()           # (altitude km, SO2 ppbv): sensitivity by finite differences
FD_H = 0.05             # central-difference step in alpha along the direction
FD_SIDES = ('_fd+', '_fd-')


def split(kind):
    """'f0.1_fd-' -> ('f0.1', -1); 'f0.1' -> ('f0.1', 0)."""
    for side, sign in zip(FD_SIDES, (1, -1)):
        if kind.endswith(side):
            return kind[:-len(side)], sign
    return kind, 0


def runs_for(z, so2, kind):
    """The run names that load_run(z, so2, kind) reads."""
    return [kind] + ([kind + side for side in FD_SIDES] if (z, so2) in FD_CELLS else [])


def ref80_dt(so2_ppbv):
    return 0.25 if so2_ppbv <= 100 else ref_dt(so2_ppbv)


def kinds_for(so2):
    """The kinds run in a cell at this SO2."""
    return [k for k in KINDS if k != 'ref_split_half' or so2 in HALF_SO2]


def reference(so2):
    """The kind --check and the figures use as the reference: the finest split run."""
    return 'ref_split_half' if so2 in HALF_SO2 else 'ref_split'


def alias(so2, kind):
    """The kind whose run a constant-dt kind is in this cell, else kind
    (an FD side keeps its suffix)."""
    base, sign = split(kind)
    suffix = kind[len(base):]
    frac, dt = KINDS[base]
    if frac is None and dt is not None:
        for other, rule in (('ref', ref80_dt), ('dt_for', dt_for)):
            if rule(so2) == dt:
                return other + suffix
    return kind


def path(z, so2, kind):
    return os.path.join(PSD80, f'z{z:g}_{so2:g}_{alias(so2, kind)}.npz')


def run_one(z, so2, kind):
    T, pres = ALTITUDES[z]
    kind = alias(so2, kind)
    base, sign = split(kind)
    xk = jnp.asarray(make_grid_80bin())
    frac, dt = KINDS[base]
    if dt is None:
        dt = {'ref': ref80_dt(so2), 'ref_split': ref80_dt(so2),
              'ref_split_half': ref80_dt(so2) / 2, 'dt_for': dt_for(so2)}[base]
    direction = jnp.asarray(directions(T)['H2SO4 cal'])
    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(so2, T, pres))
    out, tan, info = build(xk, dt, REPORT_HOURS, frac,
                           fd_alpha=sign * FD_H if sign else None,
                           pres=pres, split_chemistry=base in SPLIT_KINDS)(
                               direction, T, *state)
    os.makedirs(PSD80, exist_ok=True)
    np.savez(path(z, so2, kind), z=z, T=T, pres=pres, so2=so2, kind=kind, dt=dt,
             hours=np.array(sorted(REPORT_HOURS)), Nk=out['Nk'], Mk=out['Mk'],
             nmass=out['nmass'], dNk=tan['Nk'], dnmass=tan['nmass'],
             Nk_avg=out['Nk_avg'], Mk_avg=out['Mk_avg'], dNk_avg=tan['Nk_avg'],
             Nk_series=out['Nk_series'], Mk_series=out['Mk_series'],
             dNk_series=tan['Nk_series'],
             t=info['t'], t_compile=info['t_compile'], n_sub=info['n_sub'],
             knuc=info['knuc'], cap=info['cap'])
    print(f"z={z:g} km T={T:g} P={pres:g} SO2={so2:g} {kind} dt={dt:g} n_sub={info['n_sub']} "
          f"cap={info['cap']} t={info['t'][-1]:.0f}s "
          f"(+{info['t_compile']:.0f}s compile)", flush=True)


def done(z, so2, kind):
    # files from before Mk and the hour means were saved (imported 2026-11-06) are
    # redone, and for SERIES_KINDS files from before the hourly series (2026-11-09)
    p = path(z, so2, kind)
    need = {'Mk', 'Nk_avg', 'Mk_avg', 'dNk_avg'}
    if split(kind)[0] in SERIES_KINDS or split(kind)[1]:
        need |= {'Nk_series', 'Mk_series'}
    return os.path.exists(p) and need <= set(np.load(p).files)


def load_run(z, so2, kind):
    """A run's file as a dict; in FD_CELLS its tangents (dNk, dnmass, dNk_avg,
    dNk_series) are the central difference of the two FD sides, the jvp's kept
    as <key>_jvp. 'fd' says which."""
    A = dict(np.load(path(z, so2, kind)))
    A['fd'] = (z, so2) in FD_CELLS
    if A['fd']:
        missing = [path(z, so2, k) for k in runs_for(z, so2, kind)[1:] if not done(z, so2, k)]
        if missing:
            raise FileNotFoundError(f"Missing FD runs {missing}; run --all first.")
        P, M = (np.load(path(z, so2, kind + side)) for side in FD_SIDES)
        for key, dkey in (('Nk', 'dNk'), ('nmass', 'dnmass'), ('Nk_avg', 'dNk_avg'),
                          ('Nk_series', 'dNk_series')):
            if dkey in A:
                A[dkey + '_jvp'] = A[dkey]
            A[dkey] = (P[key] - M[key]) / (2 * FD_H)
    return A


def fd_vs_jvp():
    """Print, per FD cell and kind, the L1 distance (MERGE bins merged) of the FD
    sensitivity to the jvp's at every hour where the jvp is still valid."""
    for z, s in FD_CELLS:
        for k in SERIES_KINDS:
            A = load_run(z, s, k)
            N, fd, jvp = (merge(A[key].T).T for key in
                          ('Nk_series', 'dNk_series', 'dNk_series_jvp'))
            size = np.abs(jvp).sum(1) / N.sum(1)
            ok = size < INVALID_TANGENT
            dist = np.abs(fd - jvp).sum(1) / N.sum(1)
            print(f"FD vs jvp z={z:g} km SO2={s:g} {k}: valid jvp to hour {ok.argmin() if not ok.all() else len(ok)}, "
                  f"L1 merged x{MERGE} by hour " + " ".join(
                      f"{h + 1}:{dist[h]:.2g}" for h in range(len(ok)) if ok[h]), flush=True)


def merge(x):
    return x.reshape(-1, MERGE, *x.shape[1:]).sum(1)


def hour_mean_errors(A, R, i):
    """Distances of run A to reference R at report-hour index i, from the hour
    means: per 80-bin bin (eN, eM, edN), with MERGE bins merged (eN40, eM40,
    edN40), and the reference's sum|dN|/sum N (ref_dN)."""
    N, M, dN = A['Nk_avg'][i], A['Mk_avg'][i], A['dNk_avg'][i]
    rN, rM, rdN = R['Nk_avg'][i], R['Mk_avg'][i], R['dNk_avg'][i]
    mN, mM, mdN, mrN, mrM, mrdN = map(merge, (N, M, dN, rN, rM, rdN))
    return dict(eN=l1(N, rN, rN), eM=l1(M, rM, rM), edN=l1(dN, rdN, rN),
                eN40=l1(mN, mrN, mrN), eM40=l1(mM, mrM, mrM), edN40=l1(mdN, mrdN, mrN),
                ref_dN=float(np.abs(rdN).sum() / rN.sum()))


def check():
    """Pass/fail of f = PASS_FRAC against the reference (reference(so2)), from
    the hour means with MERGE bins merged. Also prints, per cell, the distance
    of the split reference at ref80_dt and of the unsplit 'ref' to it."""
    kind = f'f{PASS_FRAC:g}'
    missing = [path(z, s, r) for z in ALTITUDES for s in SO2_PPBV
               for k in {'ref', 'ref_split', reference(s), kind}
               for r in runs_for(z, s, k) if not done(z, s, r)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} 80-bin runs missing or without hour means "
                                f"(first {missing[:3]}); run --all first.")
    fd_vs_jvp()
    fails = []
    for s in SO2_PPBV:
        for z, (T, _) in ALTITUDES.items():
            A, R = load_run(z, s, kind), load_run(z, s, reference(s))
            others = [k for k in ('ref_split', 'ref') if k != reference(s)]
            for i, h in enumerate(A['hours']):
                for k in others:
                    eo = hour_mean_errors(load_run(z, s, k), R, i)
                    print(f"z={z:4.0f} km T={T:5.1f} SO2={s:8.0e} {int(h):2d}h {k} vs "
                          f"{reference(s)}: N={eo['eN40']:.4f} M={eo['eM40']:.4f} "
                          f"dN={eo['edN40']:.3g}", flush=True)
                e = hour_mean_errors(A, R, i)
                print(f"z={z:4.0f} km T={T:5.1f} SO2={s:8.0e} {int(h):2d}h {kind} hour mean, L1 "
                      f"merged x{MERGE} (per 80-bin bin): N={e['eN40']:.3f} ({e['eN']:.3f}) "
                      f"M={e['eM40']:.3f} ({e['eM']:.3f}) dN={e['edN40']:.3g} ({e['edN']:.3g}) | "
                      f"kdt={float(A['knuc']):.2g} cap={bool(A['cap'])}"
                      + (" [dN by FD]" if A['fd'] else ""), flush=True)
                where = f"z={z:g} km ({T:g} K) SO2={s:g} ppbv {int(h)} h: "
                bad = [f"{name} {e[k]:.3f}" for k, name in
                       (('eN40', 'L1 number'), ('eM40', 'L1 dry mass')) if e[k] > TOL]
                if float(A['knuc']) > 1.0 or bool(A['cap']):
                    bad.append(f"clamp/cap (k_nuc dt {float(A['knuc']):.2g}, "
                               f"cap {bool(A['cap'])})")
                if bad:
                    fails.append(where + ", ".join(bad))
    label = (f"80 bins merged x{MERGE}, f={PASS_FRAC} vs split-chemistry reference, "
             f"hour-mean L1 number and dry mass tolerance {TOL}; sensitivity not tested")
    if fails:
        print(f"FAIL ({label}):\n  " + "\n  ".join(fails))
    if fails:
        sys.exit(1)
    print(f"PASS ({label})")


def all_jobs():
    # heaviest first: highest SO2, then in KINDS order (ref, f = 0.03, ...)
    # a constant-dt kind that is missing its series reruns the run it aliases
    jobs = [(z, s, alias(s, r)) for s in sorted(SO2_PPBV, reverse=True)
            for k in kinds_for(s) for z in ALTITUDES for r in runs_for(z, s, k)
            if not done(z, s, r)]
    return list(dict.fromkeys(jobs))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--all', action='store_true')
    parser.add_argument('--run', nargs=3, metavar=('ALT_KM', 'SO2_PPBV', 'KIND'))
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args()
    if args.run:
        z, so2, kind = float(args.run[0]), float(args.run[1]), args.run[2]
        if z not in ALTITUDES or so2 not in SO2_PPBV or split(kind)[0] not in KINDS:
            raise ValueError(f"--run {args.run}: need altitude in {tuple(ALTITUDES)} km, SO2 in "
                             f"{SO2_PPBV}, kind in {tuple(KINDS)} (+ _fd+ / _fd-)")
        run_one(z, so2, kind)
    elif args.check:
        check()
    elif args.all:
        jobs = all_jobs()
        os.makedirs(PSD80, exist_ok=True)
        print(f"{len(jobs)} runs to do", flush=True)

        def go(job):
            z, s, k = job
            log = path(z, s, k).replace('.npz', '.log')
            with open(log, 'w') as f:
                rc = subprocess.run(
                    [sys.executable, '-m', 'benchmarks.python.gas_substep_psd80',
                     '--run', f'{z:g}', f'{s:g}', k], stdout=f,
                    stderr=subprocess.STDOUT).returncode
            print(f"job z={z:g} km SO2={s:g} {k} {'done' if rc == 0 else 'FAILED'}",
                  flush=True)
            return rc

        with ThreadPoolExecutor(int(os.environ.get('NJOBS', 11))) as ex:
            rcs = list(ex.map(go, jobs))
        failed = [j for j, rc in zip(jobs, rcs) if rc]
        if failed:
            raise RuntimeError(f"{len(failed)} runs failed (see {PSD80}/*.log): {failed}")
        check()
    else:
        parser.error('give --all, --run or --check')


if __name__ == '__main__':
    main()
