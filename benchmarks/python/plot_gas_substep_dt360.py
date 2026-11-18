"""Figure for gas_substep_dt360: wall time and L1 number error vs SO2 for adaptive
f = 0.1 at outer dt = 360 s, constant dt = 60 s and constant dt = 360 s (tropical
altitude grid, 40 bins). Reads results/gas_substep/dt360_compare.npy (run
`python -m benchmarks.python.gas_substep_dt360 --compare` first).

    python -m benchmarks.python.plot_gas_substep_dt360 [--hours 6|24] [--outer-dt 360|600] [--from-6h-runs]

--hours picks the report hour (default 24): the wall time to that hour and the
hour-mean L1 number error there. Output dt360_compare.png (24) or
dt360_compare_6h.png (6).
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmarks.python.gas_substep_convergence import OUTDIR

TOL = 5.0      # percent (L1 fraction 0.05)


def style(n):
    """kind -> (label, colour, marker); Okabe-Ito hues, marker shape as second encoding."""
    return {f'f0.1_{n}': (f'adaptive f = 0.1, outer {n} s', '#0072B2', 'o'),
            'f0.1_60': ('adaptive f = 0.1, outer 60 s', '#56B4E9', 'D'),
            'dt60': ('constant dt = 60 s', '#E69F00', 's'),
            f'dt{n}': (f'constant dt = {n} s', '#CC79A7', '^')}


def wall_time(druns, z, s, kind, hour):
    """Execution time [s] from the start to the report hour (compile excluded)."""
    p = os.path.join(druns, f'z{z:g}_{s:g}_{kind}.npz')
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing {p}. Run simulations first.")
    d = np.load(p)
    return float(d['t'][list(d['hours']).index(hour)])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--hours', type=int, choices=(6, 24), default=24)
    parser.add_argument('--outer-dt', type=int, default=360)
    parser.add_argument('--split', action='store_true', help='read the _split runs (constant dt with split_chemistry)')
    parser.add_argument('--from-6h-runs', action='store_true',
                        help='with --hours 6: read the dedicated 6 h runs (dt<N>_tropical_6h, '
                             'repeated timings) instead of hour 6 of the 24 h runs')
    args = parser.parse_args()
    hour, n = args.hours, args.outer_dt
    dedicated = args.from_6h_runs
    if dedicated and hour != 6:
        parser.error('--from-6h-runs needs --hours 6')
    src = os.path.join(OUTDIR, f'dt{n}_compare_6h.npy' if dedicated else f'dt{n}_compare.npy')
    druns = os.path.join(OUTDIR, f'dt{n}_tropical_6h' if dedicated else f'dt{n}_tropical')
    sfx = '_split' if args.split else ''
    src, druns = src.replace('.npy', sfx + '.npy'), druns + sfx
    out = os.path.join(OUTDIR, f'dt{n}_compare' + ('' if hour == 24 else f'_{hour}h')
                       + ('_dedicated' if dedicated else '') + sfx + '.png')
    STYLE = style(n)
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing {src}. Run gas_substep_dt360 --compare first.")
    rows = list(np.load(src, allow_pickle=True))
    so2s = sorted({r['so2'] for r in rows})
    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    present = {r['kind'] for r in rows}
    for kind, (label, col, mk) in STYLE.items():
        if kind not in present:
            continue
        med, lo, hi, emax, nbad, x_bad = [], [], [], [], [], []
        for s in so2s:
            sel = [r for r in rows if r['kind'] == kind and r['so2'] == s]
            if len(sel) != 6:
                raise ValueError(f"{kind} SO2={s}: {len(sel)} altitudes, expected 6")
            t = np.array([max(wall_time(druns, r['z'], s, kind, hour), 1e-3) for r in sel])   # floor: log axis only
            med.append(np.median(t)); lo.append(t.min()); hi.append(t.max())
            good = [100.0 * r[f'eN_{hour}h'] for r in sel if np.isfinite(r[f'eN_{hour}h'])]
            emax.append(max(good) if good else np.nan)
            nbad.append(6 - len(good))
        x = np.array(so2s)
        ax_t.plot(x, med, color=col, marker=mk, lw=2, ms=7, mec='white', mew=1, label=label)
        ax_t.fill_between(x, lo, hi, color=col, alpha=0.15, lw=0)
        ax_e.plot(x, emax, color=col, marker=mk, lw=2, ms=7, mec='white', mew=1, label=label)
        for xi, e, n in zip(x, emax, nbad):
            if n:
                ax_e.annotate(f'{n}/6 NaN', (xi, e if np.isfinite(e) else 3), color=col,
                              textcoords='offset points', xytext=(0, 9 if kind != 'dt360' else -15),
                              ha='center', fontsize=7)
    ax_e.axhline(TOL, color='0.4', ls='--', lw=1)
    ax_e.text(so2s[0], TOL * 1.15, 'tolerance 5 %', color='0.35', fontsize=8)
    for ax in (ax_t, ax_e):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('SO$_2$ [ppbv]')
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    ax_t.set_ylabel(f'wall time to {hour} h [s]')
    ax_t.set_title('Cost: median over 6 altitudes (band: min-max)', fontsize=10, loc='left')
    ax_e.set_ylabel(f'hour-mean L1 number error at {hour} h [%]')
    ax_e.set_title('Error vs split reference: worst of the finite altitudes', fontsize=10, loc='left')
    ax_e.set_ylim(top=600)
    ax_t.legend(frameon=False, fontsize=8, loc='best')
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
