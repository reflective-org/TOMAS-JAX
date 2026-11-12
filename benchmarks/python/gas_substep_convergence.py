"""Adaptive H2SO4 sub-steps against small fixed dt on the Phase 2 map grid.

For every T x SO2 cell of sensitivities/dunne_nuc/run_map (40 bins, the
stratospheric study settings, constant OH), runs max(REPORT_HOURS) h with
    - make_step(gas_substep='adaptive') at dt = 60 s, f in FRACS;
    - run_map.dt_for (the hand-tuned workaround);
    - a fixed-dt reference: dt_for / 4, 0.1 s at 10 ppmv and 0.02 s at
      100 ppmv (at 10 ppmv, dt_for / 4 = 0.5 s was itself ~0.04 off).
Each run is a forward pass plus one jvp along the 'H2SO4 cal' direction
(run_map.directions). At each of REPORT_HOURS it records the size
distribution in the model bins (number and dry mass per bin) and its tangent.

Pass, at every report hour, f = 0.1 against the reference, three L1
distances over the whole distribution (the 40 mass-doubling bins all have the
same width in log Dp, so these are L1 distances of dN/dlogDp):
    number       sum|N_k - N_k,ref| / sum N_k,ref          <= TOL
    dry mass     sum|M_k - M_k,ref| / sum M_k,ref          <= TOL
    sensitivity  sum|dN_k - dN_k,ref| / sum N_k,ref        <= TOL
(dN_k = d N_k / d alpha along the direction; the number distance is dominated
by the smallest particles, the mass distance by the largest), with no
gas-clamp firing and no sub-step cap hit. Any failure exits non-zero.
NMASS N(>3 nm), N(>12 nm) and delta ln N(>3 nm) are logged, not tested.

    python -m benchmarks.python.gas_substep_convergence            # full grid
    python -m benchmarks.python.gas_substep_convergence --smoke    # 2 T x 3 SO2, 1 h
    # the full grid in parallel parts, then merge (pass/fail on the merge):
    python -m benchmarks.python.gas_substep_convergence --temps 190 --so2 1e5 \
        --runs f=0.1 --part T190_1e5_f0.1
    python -m benchmarks.python.gas_substep_convergence --merge
benchmarks/results/gas_substep/run_grid_parts.sh runs the whole grid this way.
"""
import os
import sys
import glob
import time
import argparse

import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.core.config import make_grid, XK0, ICOMP, SRTH2O
from tomas_jax.solvers.condensation import make_step
from sensitivities.dunne_nuc import common as c
from sensitivities.dunne_nuc.run_map import (
    TEMPS, SO2_PPBV, FIELDS, directions, dt_for)
from sensitivities.dunne_nuc.sabre_bins import nmass_cumulative

OUTDIR = os.path.join("benchmarks", "results", "gas_substep")
PROCS = ['so2_chemistry', 'nucleation', 'coagulation', 'condensation']
FRACS = (0.3, 0.1, 0.03)
PASS_FRAC = 0.1
REPORT_HOURS = (6, 24)
TOL = 0.05
NMASS_N3, NMASS_N12 = 0, 2      # NMASS_CUTOFFS_M = 3, 8, 12, 27, 53 nm


def ref_dt(so2_ppbv):
    if so2_ppbv > 1e4:
        return 0.02
    return 0.1 if so2_ppbv > 1e3 else dt_for(so2_ppbv) / 4.0


def build(xk, dt, report_hours, frac=None, ppm_tangent='exact', fd_alpha=None,
          pres=None, split_chemistry=False):
    """run(direction, temp, Nk0, Mk0, Gc0) -> (out, tangent, info). out and
    tangent hold, at each report hour, Nk [#/cm3], dry Mk per bin [kg/cell]
    and NMASS N(>3 nm), N(>12 nm) [#/cm3], and Nk_avg, Mk_avg: the means over
    the hour before it of the state sampled once a minute (the same 60
    samples for every dt, which must divide 60 s). Nk_series, Mk_series are
    those hour means for every hour of the run, (max(report_hours), nbins). info: n_sub, knuc, cap (whole
    run), t_compile [s] and t [s, per report hour]: the execution time from
    the start to that hour, without compilation.

    The run is one compiled segment per report hour (a jvp that carries the
    state's tangent), so each hour's time is measured on its own. Each
    segment length compiles once per build (temp and the state are traced).
    ppm_tangent: make_step's (the tangent's PPM derivative; values unchanged).
    fd_alpha: run forward only, at the coefficients moved by fd_alpha along the
    direction (one side of a finite difference); the tangent is returned as 0.
    pres: pressure [Pa]; None is STRAT's 5500 Pa.
    split_chemistry: make_step's (fixed dt only; adaptive always splits).
    """
    kw = dict(sparse_density=c.SPARSE_DENSITY, water_scheme=c.WATER_SCHEME,
              ppm_tangent=ppm_tangent, jit=False)
    if frac is not None:
        kw.update(gas_substep='adaptive', gas_substep_frac=frac,
                  return_substep_info=True)
    elif split_chemistry:
        kw.update(split_chemistry=True)
    step = make_step(PROCS, **kw)
    per_minute = int(round(60.0 / dt))
    if dt > 60.0 or abs(per_minute * dt - 60.0) > 1e-9:
        raise ValueError(f"dt = {dt} s must divide 60 s (hour means sample once a minute)")
    P = jax.tree.map(jnp.float64, c.NUC_PARAMS)
    pres = c.STRAT['pres'] if pres is None else pres
    rh, alpha_acc = c.STRAT['rh'], c.STRAT['alpha']
    dry = jnp.arange(ICOMP) != SRTH2O

    def segment(alpha, state, direction, temp, n_hours):
        p = c.nuc.NucleationParams(*(getattr(P, n) + alpha * direction[k]
                                     for k, n in enumerate(FIELDS)))

        def body(carry, _):
            out = step(*carry, xk, temp, pres, c.BOXVOL, rh, alpha_acc, dt,
                       oh_conc=c.OH_CONC, org_conc=0.0, nh3_conc=0.0,
                       fion=c.FION, nucl_params=p)
            if frac is None:
                return out, (0, 0.0, False)
            info = out[3]
            return out[:3], (info['n_sub'], info['knuc_dt_max'],
                             info['hit_cap'])

        def dry_mass(Mk):
            return jnp.sum(jnp.where(dry, Mk, 0.0), 1)

        def minute(carry, _):
            state, sum_N, sum_M = carry
            state, diag = jax.lax.scan(body, state, None, length=per_minute)
            return (state, sum_N + state[0], sum_M + dry_mass(state[1])), diag

        def hour(carry, _):
            zero = (jnp.zeros_like(carry[0]), jnp.zeros(carry[1].shape[0]))
            (carry, sum_N, sum_M), diag = jax.lax.scan(
                minute, (carry, *zero), None, length=60)
            return carry, (diag, sum_N / 60.0, sum_M / 60.0)

        carry, ((n_sub, knuc, cap), mean_N, mean_M) = jax.lax.scan(
            hour, state, None, length=n_hours)
        Nk, Mk = carry[0], carry[1]
        n = nmass_cumulative(Nk, Mk, xk, c.BOXVOL)
        out = dict(Nk=Nk / c.BOXVOL, Mk=dry_mass(Mk),
                   nmass=n[jnp.array([NMASS_N3, NMASS_N12])],
                   Nk_avg=mean_N[-1] / c.BOXVOL, Mk_avg=mean_M[-1],
                   Nk_series=mean_N / c.BOXVOL, Mk_series=mean_M)
        return (carry, out), (jnp.sum(n_sub), jnp.max(knuc), jnp.any(cap))

    def seg_jvp(state, dstate, direction, temp, n_hours):
        return jax.jvp(
            lambda a, s: segment(a, s, direction, temp, n_hours),
            (jnp.float64(0.0), state), (jnp.float64(1.0), dstate), has_aux=True)

    def seg_fwd(state, dstate, direction, temp, n_hours):
        primal, aux = segment(jnp.float64(fd_alpha), state, direction, temp, n_hours)
        return primal, jax.tree.map(jnp.zeros_like, primal), aux

    seg_jit = jax.jit(seg_jvp if fd_alpha is None else seg_fwd, static_argnums=4)
    compiled = {}
    hours = sorted(report_hours)
    lengths = [hours[0]] + [b - a for a, b in zip(hours, hours[1:])]

    def run(direction, temp, Nk0, Mk0, Gc0):
        temp = jnp.float64(temp)
        state = (Nk0, Mk0, Gc0)
        dstate = jax.tree.map(jnp.zeros_like, state)
        t_compile, t, ts, outs, tans, diags = 0.0, 0.0, [], [], [], []
        for n in lengths:
            if n not in compiled:
                t0 = time.time()
                compiled[n] = seg_jit.lower(state, dstate, direction, temp,
                                            n).compile()
                t_compile += time.time() - t0
            t0 = time.time()
            (state, out), (dstate, dout), diag = jax.block_until_ready(
                compiled[n](state, dstate, direction, temp))
            t += time.time() - t0
            ts.append(t)
            outs.append(out)
            tans.append(dout)
            diags.append(diag)
        # report-hour values stack; the hourly series of the segments join
        stack = lambda xs: {k: (np.concatenate if k.endswith('_series') else np.stack)(
            [np.asarray(x[k]) for x in xs]) for k in xs[0]}
        info = dict(n_sub=int(sum(int(d[0]) for d in diags)),
                    knuc=float(max(float(d[1]) for d in diags)),
                    cap=bool(any(bool(d[2]) for d in diags)),
                    t_compile=t_compile, t=np.array(ts))
        return stack(outs), stack(tans), info

    return run


def l1(a, b, norm):
    return float(np.sum(np.abs(a - b)) / np.sum(norm))


# Smoothed distances (figures; not part of the pass condition)
RHO_GRID = 1770.0       # kg/m3, sulfate (the grid density): dry diameter of a bin edge
KDE_SIGMA = 0.075       # decades of Dp, 0.75 of a 40-bin width
KDE_X = np.linspace(0.0, np.log10(5000.0), 2000)     # log10 Dp [nm]


def edges_nm(xk):
    """Dry diameter [nm] of the bin edges."""
    return (6 * np.asarray(xk) / (np.pi * RHO_GRID)) ** (1 / 3) * 1e9


def kde(N, xk, sigma=KDE_SIGMA):
    """dN/dlog10 Dp on KDE_X: a Gaussian of width sigma decades at each bin centre."""
    e = np.log10(edges_nm(xk))
    z = (KDE_X[:, None] - 0.5 * (e[:-1] + e[1:])[None, :]) / sigma
    return (np.exp(-0.5 * z ** 2) / (sigma * np.sqrt(2 * np.pi))) @ np.asarray(N)


def kde_l1(a, xa, b, xb, norm, sigma=KDE_SIGMA):
    """L1 distance of the two KDEs over log10 Dp, divided by sum(norm).
    a and b may be on different grids (xa, xb)."""
    return float(np.abs(kde(a, xa, sigma) - kde(b, xb, sigma)).sum()
                 * (KDE_X[1] - KDE_X[0]) / np.sum(norm))


def compare(rows):
    """Add the distances to the cell's reference to every row (in place)."""
    refs = {(r['T'], r['so2'], r['hour']): r for r in rows if r['run'] == 'ref'}
    for r in rows:
        ref = refs[(r['T'], r['so2'], r['hour'])]
        r.update(eN=l1(r['Nk'], ref['Nk'], ref['Nk']),
                 eM=l1(r['Mk'], ref['Mk'], ref['Mk']),
                 edN=l1(r['dNk'], ref['dNk'], ref['Nk']),
                 rN3=r['N3'] / ref['N3'], rN12=r['N12'] / ref['N12'],
                 dd3=r['d3'] - ref['d3'])


def run_keys():
    return [f'f={f:g}' for f in FRACS] + ['dt_for', 'ref']


def report(r):
    print(f"T={r['T']:5.0f} SO2={r['so2']:8.0e} {r['hour']:2d}h {r['run']:8s} "
          f"dt={r['dt']:6.3g} L1 N={r['eN']:.3f} M={r['eM']:.3f} "
          f"dN={r['edN']:.3f} | N3 x{r['rN3']:.3f} N12 x{r['rN12']:.3f} "
          f"dlnN3 {r['dd3']:+.3f} | n_sub={r['n_sub']} kdt={r['knuc']:.2g} "
          f"cap={r['cap']} t={r['t']:.1f}s", flush=True)


def failures(rows):
    out = []
    for r in rows:
        if r['run'] != f'f={PASS_FRAC:g}':
            continue
        bad = [f"{name} {r[k]:.3f}" for k, name in
               (('eN', 'L1 number'), ('eM', 'L1 dry mass'),
                ('edN', 'L1 sensitivity')) if r[k] > TOL]
        if r['knuc'] > 1.0 or r['cap']:
            bad.append(f"clamp/cap (k_nuc dt {r['knuc']:.2g}, cap {r['cap']})")
        if bad:
            out.append(f"T={r['T']:g} SO2={r['so2']:g} ppbv {r['hour']} h: "
                       + ", ".join(bad))
    return out


def save(rows, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, **{k: np.array([r[k] for r in rows]) for k in rows[0]})
    print(f"wrote {path}")


def load(path):
    d = np.load(path)
    return [{k: (d[k][i] if d[k].ndim > 1 else d[k][i].item()) for k in d.files}
            for i in range(len(d['T']))]


def finish(rows):
    fails = failures(rows)
    if fails:
        print(f"FAIL (f={PASS_FRAC}, L1 tolerance {TOL}):\n  "
              + "\n  ".join(fails))
        sys.exit(1)
    print(f"PASS (f={PASS_FRAC}, L1 tolerance {TOL})")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--smoke', action='store_true')
    parser.add_argument('--temps', help='comma-separated subset of TEMPS')
    parser.add_argument('--so2', help='comma-separated subset of SO2_PPBV')
    parser.add_argument('--runs', help='comma-separated subset of '
                        + ','.join(run_keys()))
    parser.add_argument('--part', help='write grid_part_<part>.npz')
    parser.add_argument('--merge', action='store_true',
                        help='merge grid_part_*.npz into grid.npz and judge')
    parser.add_argument('--outdir', default=OUTDIR)
    args = parser.parse_args()

    if args.runs is not None and args.part is None:
        parser.error('--runs needs --part (the reference may be in another part)')

    if args.merge:
        parts = sorted(glob.glob(os.path.join(args.outdir, 'grid_part_*.npz')))
        if not parts:
            raise FileNotFoundError(f"No grid_part_*.npz in {args.outdir}")
        rows = [r for p in parts for r in load(p)]
        keys = [(r['T'], r['so2'], r['run'], r['hour']) for r in rows]
        want = [(T, s, k, h) for T in TEMPS for s in SO2_PPBV
                for k in run_keys() for h in REPORT_HOURS]
        missing = sorted(set(want) - set(keys))
        dup = sorted({k for k in keys if keys.count(k) > 1})
        if missing or dup:
            raise ValueError(f"{len(parts)} parts: {len(missing)} runs missing "
                             f"(first {missing[:5]}), duplicated {dup[:5]}")
        compare(rows)
        for r in sorted(rows, key=lambda r: (r['T'], r['so2'], r['hour'])):
            report(r)
        save(rows, os.path.join(args.outdir, 'grid.npz'))
        finish(rows)
        return

    if args.smoke:
        temps, so2s, hours = (208.0, 217.0), (1.0, 1e3, 1e5), (1,)
    else:
        temps = TEMPS if args.temps is None else tuple(
            float(t) for t in args.temps.split(','))
        unknown = set(temps) - set(TEMPS)
        if unknown:
            raise ValueError(f"--temps {sorted(unknown)} not in TEMPS {TEMPS}")
        so2s = SO2_PPBV if args.so2 is None else tuple(
            float(x) for x in args.so2.split(','))
        unknown = set(so2s) - set(SO2_PPBV)
        if unknown:
            raise ValueError(f"--so2 {sorted(unknown)} not in SO2_PPBV {SO2_PPBV}")
        hours = REPORT_HOURS

    xk = jnp.asarray(make_grid(40, XK0, 2.0))
    configs = [('adaptive', f, 60.0) for f in FRACS] + [('dt_for', None, None),
                                                        ('ref', None, None)]
    if args.runs is not None:
        runs = args.runs.split(',')
        unknown = set(runs) - set(run_keys())
        if unknown:
            raise ValueError(f"--runs {sorted(unknown)} not in {run_keys()}")
        configs = [cf for cf, k in zip(configs, run_keys()) if k in runs]
    compiled = {}

    def runner(dt, frac):
        if (dt, frac) not in compiled:
            compiled[(dt, frac)] = build(xk, dt, hours, frac)
        return compiled[(dt, frac)]

    rows = []
    for T in temps:
        direction = jnp.asarray(directions(T)['H2SO4 cal'])
        for so2 in so2s:
            Nk0, Mk0, Gc0 = c.lognormal_state(
                xk, so2=c.ppbv_to_molec(so2, T, c.STRAT['pres']))
            cell = []
            for name, frac, dt in configs:
                dt = {'dt_for': dt_for(so2), 'ref': ref_dt(so2)}.get(name, dt)
                key = name if frac is None else f'f={frac:g}'
                out, tan, info = runner(dt, frac)(direction, T, Nk0, Mk0, Gc0)
                for i, h in enumerate(hours):
                    cell.append(dict(
                        T=T, so2=so2, run=key, hour=h, dt=dt,
                        t=float(info['t'][i]), t_compile=info['t_compile'],
                        n_sub=info['n_sub'], knuc=info['knuc'], cap=info['cap'],
                        Nk=out['Nk'][i], Mk=out['Mk'][i], dNk=tan['Nk'][i],
                        Nk_avg=out['Nk_avg'][i], Mk_avg=out['Mk_avg'][i],
                        dNk_avg=tan['Nk_avg'][i],
                        N3=float(out['nmass'][i, 0]),
                        N12=float(out['nmass'][i, 1]),
                        d3=float(tan['nmass'][i, 0] / out['nmass'][i, 0])))
                print(f"T={T:5.0f} SO2={so2:8.0e} {key:8s} dt={dt:6.3g} "
                      f"n_sub={info['n_sub']} cap={info['cap']} "
                      f"t={info['t'][-1]:.1f}s (+{info['t_compile']:.1f}s compile)",
                      flush=True)
            if args.part is None:
                compare(cell)
                for r in sorted(cell, key=lambda r: r['hour']):
                    report(r)
            rows += cell

    name = ('smoke' if args.smoke else
            f'grid_part_{args.part}' if args.part else 'grid')
    save(rows, os.path.join(args.outdir, f'{name}.npz'))
    if args.part:
        print(f"part {args.part} done; judge with --merge")
        return
    finish(rows)


if __name__ == '__main__':
    main()
