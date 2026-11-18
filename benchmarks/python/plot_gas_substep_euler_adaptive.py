"""Figure for gas_substep_euler_adaptive: wall time and L1 number error vs SO2 for
euler_fixed (solid) and euler_adaptive (dashed, open markers) coagulation, with adaptive
gas sub-steps (f = 0.1, outer 60 s / 360 s) and at constant dt = 60 s / 360 s (tropical
altitude grid). Reads results/gas_substep/euler_adaptive_compare.npy (run
`python -m benchmarks.python.gas_substep_euler_adaptive --compare` first).

    python -m benchmarks.python.plot_gas_substep_euler_adaptive [--hours 6|24]

Output euler_adaptive_compare[_6h].png. An x marks a point where an altitude hit a
coagulation sub-step cap (the run does not count as valid); NaN runs are annotated.
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from benchmarks.python.gas_substep_convergence import OUTDIR

TOL = 5.0      # percent
# kind -> (label, colour, marker, euler_adaptive?)
STYLE = {'f0.1_360': ('fixed, adaptive gas f = 0.1, outer 360 s', '#0072B2', 'o', False),
         'ea_f0.1_360': ('adaptive coag, f = 0.1, outer 360 s', '#0072B2', 'o', True),
         'f0.1_60': ('fixed, adaptive gas f = 0.1, outer 60 s', '#009E73', 'D', False),
         'ea_f0.1_60': ('adaptive coag, f = 0.1, outer 60 s', '#009E73', 'D', True),
         'dt60': ('fixed, constant dt = 60 s', '#E69F00', 's', False),
         'ea_dt60': ('adaptive coag, constant dt = 60 s', '#E69F00', 's', True),
         'dt360': ('fixed, constant dt = 360 s', '#CC79A7', '^', False),
         'ea_dt360': ('adaptive coag, constant dt = 360 s', '#CC79A7', '^', True)}


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--hours', type=int, choices=(6, 24), default=24)
    parser.add_argument('--bins', type=int, choices=(40, 80), default=40,
                        help='read euler_adaptive_compare_80.npy (BINS=80 runs)')
    parser.add_argument('--no-fixed-constant', action='store_true',
                        help='leave out the euler_fixed constant-dt kinds (dt60, dt360)')
    parser.add_argument('--split', action='store_true',
                        help='constant-dt kinds from the SPLIT=1 runs (split chemistry, like the '
                             'reference); adaptive kinds always split and come from the default file')
    args = parser.parse_args()
    hour = args.hours
    sfx = '' if args.bins == 40 else f'_{args.bins}'
    src = os.path.join(OUTDIR, f'euler_adaptive_compare{sfx}.npy')
    if not os.path.exists(src):
        raise FileNotFoundError(f"Missing {src}. Run gas_substep_euler_adaptive --compare first.")
    rows = list(np.load(src, allow_pickle=True))
    if args.split:
        src_s = os.path.join(OUTDIR, f'euler_adaptive_compare{sfx}_split.npy')
        if not os.path.exists(src_s):
            raise FileNotFoundError(f"Missing {src_s}. Run SPLIT=1 gas_substep_euler_adaptive --all first.")
        const = ('dt60', 'dt360', 'ea_dt60', 'ea_dt360')
        rows = [r for r in rows if r['kind'] not in const] + \
               [r for r in np.load(src_s, allow_pickle=True) if r['kind'] in const]
    so2s = sorted({r['so2'] for r in rows})
    fig, (ax_t, ax_e) = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    for kind, (label, col, mk, ea) in STYLE.items():
        if args.no_fixed_constant and kind in ('dt60', 'dt360'):
            continue
        sel_k = [r for r in rows if r['kind'] == kind]
        if not sel_k:
            raise ValueError(f"No rows for kind {kind} in {src}")
        med, lo, hi, emax, ncap, nnan = [], [], [], [], [], []
        for s in so2s:
            sel = [r for r in sel_k if r['so2'] == s]
            if len(sel) != 6:
                raise ValueError(f"{kind} SO2={s}: {len(sel)} altitudes, expected 6")
            t = np.array([max(r[f't_{hour}h'], 1e-3) for r in sel])      # floor: log axis only
            med.append(np.median(t)); lo.append(t.min()); hi.append(t.max())
            good = [100.0 * r[f'eN_{hour}h'] for r in sel if np.isfinite(r[f'eN_{hour}h'])]
            emax.append(max(good) if good else np.nan)
            ncap.append(sum(r['coag_cap'] or r['gas_cap'] for r in sel))
            nnan.append(6 - len(good))
        kw = dict(color=col, marker=mk, lw=2, ms=7, mew=1.5,
                  ls=(0, (3, 2)) if ea else '-', mfc='white' if ea else col, mec=col if ea else 'white')
        ax_t.plot(so2s, med, **kw)
        ax_t.fill_between(so2s, lo, hi, color=col, alpha=0.10 if ea else 0.15, lw=0)
        ax_e.plot(so2s, emax, **kw)
        for ax, y in ((ax_t, med), (ax_e, emax)):
            for xi, yi, nc in zip(so2s, y, ncap):
                if nc and np.isfinite(yi):
                    ax.plot(xi, yi, 'x', color='k', ms=11, mew=1.8, zorder=5)
        for xi, e, nn in zip(so2s, emax, nnan):
            if nn:
                ax_e.annotate(f'{nn}/6 NaN', (xi, e if np.isfinite(e) else 3), color=col,
                              textcoords='offset points', xytext=(0, 9), ha='center', fontsize=7)
    ax_e.axhline(TOL, color='0.4', ls=':', lw=1)
    ax_e.text(1e3, TOL * 1.15, 'tolerance 5 %', color='0.35', fontsize=8, ha='center')
    for ax in (ax_t, ax_e):
        ax.set_xscale('log'); ax.set_yscale('log')
        ax.set_xlabel('SO$_2$ [ppbv]')
        ax.grid(alpha=0.25, lw=0.6)
        for sp in ('top', 'right'):
            ax.spines[sp].set_visible(False)
    ax_t.set_ylabel(f'wall time to {hour} h [s]')
    ax_t.set_title(f'Cost ({args.bins} bins): median over 6 altitudes (band: min-max)',
                   fontsize=10, loc='left')
    ax_e.set_ylabel(f'hour-mean L1 number error at {hour} h [%]')
    ax_e.set_title('Error vs split-chemistry reference: worst of the finite altitudes', fontsize=10, loc='left')
    ax_e.set_ylim(top=600)
    from matplotlib.lines import Line2D
    steps = [('adaptive gas sub-steps, outer 60 s', '#009E73', 'D'),
             ('adaptive gas sub-steps, outer 360 s', '#0072B2', 'o'),
             ('constant dt = 60 s', '#E69F00', 's'),
             ('constant dt = 360 s', '#CC79A7', '^')]
    h_step = [Line2D([], [], color=c_, marker=m_, lw=2.5, ms=7, mec=c_, label=l_)
              for l_, c_, m_ in steps]
    h_coag = [Line2D([], [], color='0.25', ls='-', lw=2.5, label='euler_fixed (filled markers)'),
              Line2D([], [], color='0.25', ls=(0, (3, 2)), lw=2.5,
                     label='euler_adaptive (open markers)'),
              Line2D([], [], color='k', ls='', marker='x', ms=9, mew=1.8,
                     label='coagulation cap hit')]
    leg1 = ax_t.legend(handles=h_step, title='Time stepping (colour)', frameon=False,
                       fontsize=8, title_fontsize=8, loc='upper left', alignment='left', handlelength=3)
    ax_t.add_artist(leg1)
    fig.canvas.draw()       # place the second legend directly under the first
    box = leg1.get_window_extent().transformed(ax_t.transAxes.inverted())
    ax_t.legend(handles=h_coag, title='Coagulation solver', frameon=False, fontsize=8,
                title_fontsize=8, loc='upper left', bbox_to_anchor=(0.0, box.y0 - 0.02),
                alignment='left', handlelength=3)
    out = os.path.join(OUTDIR, f'euler_adaptive_compare{sfx}' + ('' if hour == 24 else f'_{hour}h')
                       + ('_no_fixed_constant' if args.no_fixed_constant else '')
                       + ('_split' if args.split else '') + '.png')
    fig.savefig(out, dpi=200)
    print(f"wrote {out}")


if __name__ == '__main__':
    main()
