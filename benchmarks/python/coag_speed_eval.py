"""Quick evaluation of coagulation / gas sub-step speed candidates.

Runs the gas_substep_coag_interval.run_one simulation (adaptive gas sub-steps f = 0.1,
euler_adaptive coagulation, outer dt = 360 s, 40 bins, tropical AFGL cell, 24 h, forward
only) for a few cells and named make_step configs, times the post-compile execution,
and prints per run: wall time to 24 h, speed-up vs the first config, gas sub-step count,
hour-mean L1 number and dry mass at 6 h and 24 h vs the cell's reference (ref_split_half
where it exists, else ref_split, as gas_substep_coag_interval.reference), and flags
(NaN, GASCAP, COAGCAP from the warnings, ok = finite, no cap, L1 N <= 0.05 at 6 h and 24 h).

A config is NAME=JSON, the JSON holding make_step kwargs that override the run's own
(coag_interval_frac, coag_max_substeps, gas_substep_frac, gas_max_substeps, coag_solver,
or any new opt-in kwarg). Keys frac-like ones go to the harness's build(); every other key
is passed to make_step through a wrapper.

    python benchmarks/python/coag_speed_eval.py \
        --config base01='{"coag_interval_frac": 0.01}' --config c03='{"coag_interval_frac": 0.03}'
    python benchmarks/python/coag_speed_eval.py --root /path/to/worktree \
        --refs /Users/dmcguffin/Documents/Modeling/Microphysics/TOMAS-JAX/benchmarks/results/gas_substep \
        --cells 13:100 20:1000 --config mine='{"new_kwarg": 1}'

--root: directory whose tomas_jax / sensitivities / benchmarks are imported (default: this
repo). --refs: gas_substep results dir holding psd40_tropical/ (default: <root>/benchmarks/
results/gas_substep, falling back to the main repo). --cells: Z:SO2 pairs (km:ppbv).
--out FILE: also save the rows as JSON. Run one job at a time.
"""
import os
import sys
import json
import argparse
import warnings

MAIN = '/Users/dmcguffin/Documents/Modeling/Microphysics/TOMAS-JAX'
DEFAULT_CELLS = [(13, 10), (13, 100), (13, 1000), (13, 10000),
                 (20, 10), (20, 100), (20, 1000),
                 (25, 10), (25, 100), (25, 1000)]
TOL = 0.05
DT = 360.0
FRAC = 0.1
GAS_MAX = 65536
COAG_MAX = 10000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--root', default=os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__)))))
    p.add_argument('--refs', default=None)
    p.add_argument('--cells', nargs='*', default=None, help='Z:SO2 pairs, km:ppbv')
    p.add_argument('--config', action='append', default=None, metavar='NAME=JSON')
    p.add_argument('--out', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    root = os.path.abspath(args.root)
    sys.path.insert(0, root)
    os.chdir(root)
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')

    import numpy as np
    import jax.numpy as jnp
    import tomas_jax
    from tomas_jax.core.config import make_grid, XK0
    from sensitivities.dunne_nuc import common as c
    from sensitivities.dunne_nuc.run_map import directions
    from benchmarks.python import gas_substep_convergence as gsc
    from benchmarks.python.gas_substep_convergence import l1, REPORT_HOURS

    print(f"tomas_jax: {tomas_jax.__file__}", flush=True)
    if not os.path.abspath(tomas_jax.__file__).startswith(root):
        raise RuntimeError(f"tomas_jax imported from {tomas_jax.__file__}, not under {root}")

    refs = args.refs or os.path.join(root, 'benchmarks', 'results', 'gas_substep')
    if not os.path.isdir(os.path.join(refs, 'psd40_tropical')):
        refs = os.path.join(MAIN, 'benchmarks', 'results', 'gas_substep')
    d40 = os.path.join(refs, 'psd40_tropical')
    print(f"refs: {d40}", flush=True)

    cells = DEFAULT_CELLS if not args.cells else [
        (float(a.split(':')[0]), float(a.split(':')[1])) for a in args.cells]
    cfgs = []
    for s in (args.config or ['frac0.01={"coag_interval_frac": 0.01}',
                              'frac0.03={"coag_interval_frac": 0.03}']):
        name, js = s.split('=', 1)
        cfgs.append((name, json.loads(js)))

    afgl = c.TROPICAL_AFGL

    def fname(z, so2, kind):
        return f'z{z:g}_{so2:g}_{kind}.npz'

    def reference(z, so2):
        for name in ('ref_split_half', 'ref_split'):
            pth = os.path.join(d40, fname(z, so2, name))
            if os.path.exists(pth):
                return name, np.load(pth)
        raise FileNotFoundError(f"No ref_split_half/ref_split for z={z:g} SO2={so2:g} in {d40}")

    # extra make_step kwargs reach make_step through a wrapper around the one build() uses
    orig_make_step = gsc.make_step
    extra = {}
    gsc.make_step = lambda procs, **kw: orig_make_step(procs, **{**kw, **extra})

    rows = []
    hdr = (f"{'z':>3s} {'SO2':>7s} {'config':>14s} {'ref':>14s} {'t6 [s]':>7s} {'t24 [s]':>8s} "
           f"{'x first':>7s} {'n_sub':>8s} {'N 6h':>6s} {'N 24h':>6s} {'M 6h':>6s} "
           f"{'M 24h':>6s} {'comp [s]':>8s}  flags")
    print(hdr, flush=True)
    for z, so2 in cells:
        if z not in afgl:
            raise ValueError(f"z={z} not in TROPICAL_AFGL {list(afgl)}")
        T, pres = afgl[z]
        ref_name, ref = reference(z, so2)
        xk = jnp.asarray(make_grid(40, XK0, 2.0))
        direction = jnp.asarray(directions(T)['H2SO4 cal'])
        t_first = None
        for name, kw in cfgs:
            kw = dict(kw)
            frac = kw.pop('gas_substep_frac', FRAC)
            gmax = kw.pop('gas_max_substeps', GAS_MAX)
            cmax = kw.pop('coag_max_substeps', COAG_MAX)
            solver = kw.pop('coag_solver', 'euler_adaptive')
            civ = kw.pop('coag_interval_frac', None)
            extra.clear()
            extra.update(kw)
            state = c.lognormal_state(xk, so2=c.ppbv_to_molec(so2, T, pres))
            with warnings.catch_warnings(record=True) as wl:
                warnings.simplefilter('always')
                out, _, info = gsc.build(
                    xk, DT, REPORT_HOURS, frac, fd_alpha=0.0, pres=pres, sample_s=DT,
                    gas_max_substeps=gmax, coag_solver=solver, coag_max_substeps=cmax,
                    coag_interval_frac=civ)(direction, T, *state)
            names = {type(w.message).__name__ for w in wl}
            gas_cap = bool(info['cap']) or 'GasSubstepCapWarning' in names
            coag_cap = 'CoagSubstepCapWarning' in names
            finite = bool(np.all(np.isfinite(out['Nk_avg'])) and np.all(np.isfinite(out['Mk_avg'])))
            r = dict(z=z, so2=so2, config=name, ref=ref_name, finite=finite, gas_cap=gas_cap,
                     coag_cap=coag_cap, n_sub=int(info['n_sub']), t_compile=info['t_compile'],
                     t_6h=float(info['t'][0]), t_24h=float(info['t'][-1]))
            hrs = [int(h) for h in sorted(REPORT_HOURS)]
            for i, h in enumerate(hrs):
                j = [int(x) for x in ref['hours']].index(h)
                for q, key in (('N', 'Nk_avg'), ('M', 'Mk_avg')):
                    r[f'e{q}_{h}h'] = (l1(np.asarray(out[key][i]), ref[key][j], ref[key][j])
                                       if finite else float('nan'))
            r['valid'] = bool(finite and not gas_cap and not coag_cap
                              and r['eN_6h'] <= TOL and r['eN_24h'] <= TOL)
            t_first = r['t_24h'] if t_first is None else t_first
            r['x_first'] = t_first / max(r['t_24h'], 1e-3)
            rows.append(r)
            flags = ('' if finite else 'NaN ') + ('GASCAP ' if gas_cap else '') + \
                    ('COAGCAP ' if coag_cap else '') + ('ok' if r['valid'] else 'FAIL')
            print(f"{z:3g} {so2:7g} {name:>14s} {ref_name:>14s} {r['t_6h']:7.1f} "
                  f"{r['t_24h']:8.1f} {r['x_first']:7.2f} {r['n_sub']:8d} {r['eN_6h']:6.3f} "
                  f"{r['eN_24h']:6.3f} {r['eM_6h']:6.3f} {r['eM_24h']:6.3f} "
                  f"{r['t_compile']:8.1f}  {flags}", flush=True)
    gsc.make_step = orig_make_step

    print("\nper config (sum over cells):")
    for name, _ in cfgs:
        sel = [r for r in rows if r['config'] == name]
        print(f"  {name:>14s}: t24 sum {sum(r['t_24h'] for r in sel):8.1f} s, "
              f"valid {sum(r['valid'] for r in sel)}/{len(sel)}, "
              f"max L1N {max(max(r['eN_6h'], r['eN_24h']) for r in sel):.3f}")
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(rows, f, indent=1)


if __name__ == '__main__':
    main()
