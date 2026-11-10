"""Size-distribution error and wall time per T x SO2 cell from
gas_substep_convergence (40 bins) or gas_substep_psd80 (80 bins).

    python -m benchmarks.python.plot_gas_substep_grid [--bins 40|80]

Writes one figure per report hour: grid_error_time_<h>h.png (40 bins, from
grid.npz) or grid_error_time_80bins_<h>h.png (80 bins, from psd80/; every run
must exist). Series: adaptive f = 0.3 and 0.1, and in the 80-bin figure
constant dt = 60 s and 0.25 s in every cell (run_map.dt_for is not shown).
Columns: temperature. Rows: the L1 distances to the fixed-dt
reference on the same grid that the benchmark tests (number, dry mass,
sensitivity of the number distribution); then wall time. 40-bin distances are from the
snapshots (grid.npz has no hour means); 80-bin ones from the hour means, as in
gas_substep_psd80's pass condition: solid with pairs of 80-bin bins merged
(tested), dashed per 80-bin bin. Distances above CEIL (the PPM tangent
instability) are drawn at CEIL with a cross. Runs that store per-hour
timing (t_compile present) show the execution time to the figure's hour
without compilation; older ones only have the whole 24 h run, compile
included, and the row says so. In the 40-bin figure, a run whose 80-bin
version and 80-bin reference both exist in psd80/ with hour means is shown
at 80 bins merged to the 40-bin grid, hour mean (ringed); wall time stays the
40-bin run's.
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sensitivities.dunne_nuc.run_map import TEMPS, SO2_PPBV
from benchmarks.python.gas_substep_convergence import (
    OUTDIR, TOL)
from benchmarks.python.gas_substep_psd80 import (
    KINDS, MERGE, path, done, hour_mean_errors, load_run, runs_for)

INK, INK2, GRID, SURFACE = '#1f1f1e', '#5f5e58', '#e4e3df', '#fcfcfb'
SERIES = [  # (run key, label, colour, marker)
    ('f=0.3', 'adaptive f = 0.3', '#2a78d6', 'o'),
    ('f=0.1', 'adaptive f = 0.1 (default)', '#eb6834', 's'),
    ('dt=60 s', 'constant dt = 60 s', '#eda100', 'D'),
    ('dt=0.25 s', 'constant dt = 0.25 s (the reference to 100 ppbv)', '#7b52ae', '^'),
    ('ref', 'reference', '#8a8980', 'v'),
]
CONSTANT_DT = ('dt=60 s', 'dt=0.25 s')    # 80-bin runs only
REF_LABEL = {40: 'reference (dt_for/4; 0.1 s at 10, 0.02 s at 100 ppmv)',
             80: 'reference (0.25 s to 100 ppbv; 0.5, 0.1, 0.02 s at 1, 10, 100 ppmv)'}
RUN_KEY = {'ref': 'ref', 'f0.03': 'f=0.03', 'f0.1': 'f=0.1', 'f0.3': 'f=0.3',
           'dt_for': 'dt_for', 'dt60': 'dt=60 s', 'dt0.25': 'dt=0.25 s'}
FLOOR = 1e-4    # distances below this are drawn at it (log axis)
CEIL = 10.0     # and above this (tangent blew up, reference included)


def plot_hour(d, hour, outdir, nbins):
    sel = d['hour'] == hour
    temps = np.unique(d['T'][sel])
    timed = 't_compile' in d
    merged = nbins == 80
    rows = [('L1 number', 'eN40' if merged else 'eN'),
            ('L1 dry mass', 'eM40' if merged else 'eM'),
            ('L1 sensitivity', 'edN40' if merged else 'edN'),
            (f'wall time to {hour} h [s]\n(excl. compile)' if timed else
             'wall time [s], whole\n24 h run incl. compile', 't')]
    height = 2.1 * len(rows) + 0.6
    fig, axes = plt.subplots(len(rows), len(temps),
                             figsize=(2.6 * len(temps), height), sharex=True,
                             sharey='row', facecolor=SURFACE)
    for j, T in enumerate(temps):
        for i, (ylabel, key_y) in enumerate(rows):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for s in ('top', 'right'):
                ax.spines[s].set_visible(False)
            is_err = key_y != 't'
            if is_err:
                ax.axhspan(FLOOR, TOL, color=GRID, alpha=0.6, lw=0)
                ax.axhline(TOL, color=INK2, lw=0.8)
            for key, label, colour, marker in SERIES:
                if (key == 'ref' and is_err) or (key in CONSTANT_DT and nbins == 40):
                    continue
                if key == 'ref':
                    label = REF_LABEL[nbins]
                m = sel & (d['T'] == T) & (d['run'] == key)
                if not m.any():
                    raise ValueError(f"No '{key}' rows at T={T} hour={hour}")
                order = np.argsort(d['so2'][m])     # merged parts are unsorted
                x, y = d['so2'][m][order], d[key_y][m][order]
                if is_err:
                    if not np.isfinite(y).all():
                        raise ValueError(f"Non-finite {key_y} for '{key}' at T={T} "
                                         f"hour={hour}, SO2 {x[~np.isfinite(y)]}")
                    over = y > CEIL
                    y = np.clip(y, FLOOR, CEIL)
                ax.plot(x, y, color=colour, lw=2, marker=marker, ms=5,
                        mec=SURFACE, mew=1, label=label)
                if is_err and over.any():
                    ax.plot(x[over], y[over], ls='none', marker='X', ms=9, color=INK,
                            label=f'> {CEIL:g}, drawn at {CEIL:g} (tangent unstable)')
                if merged and key_y.endswith('40'):
                    fine = np.clip(d[key_y[:-2]][m][order], FLOOR, CEIL)
                    ax.plot(x, fine, color=colour, lw=1, ls='--', alpha=0.8)
                b80 = d['bins80'][m][order] if (is_err and 'bins80' in d) else None
                if b80 is not None and b80.any():
                    ax.plot(x[b80], y[b80], ls='none', marker='o', ms=9,
                            mfc='none', mec=colour, mew=1.6,
                            label='ringed: 80 bins merged to 40, hour mean, vs 80-bin reference')
                cap = d['cap'][m][order]
                if cap.any():
                    ax.plot(x[cap], y[cap], ls='none', marker='o', ms=11,
                            mfc='none', mec=INK, mew=1.2,
                            label='sub-step cap hit')
                fd = d['fd'][m][order] if (key_y.startswith('edN') and 'fd' in d) else None
                if fd is not None and fd.any():
                    ax.plot(x[fd], y[fd], ls='none', marker='s', ms=10,
                            mfc='none', mec=INK2, mew=1.0,
                            label='boxed: sensitivity by central FD (jvp tangent unstable)')
            ax.set_xscale('log')
            ax.set_yscale('log')
            if i == 0:
                ax.set_title(f'{T:g} K', color=INK)
            if j == 0:
                ax.set_ylabel(ylabel)
            if i == len(rows) - 1:
                ax.set_xlabel('SO₂ [ppbv]')

    handles, labels = [], []
    for ax in axes.flat:
        for h, l in zip(*ax.get_legend_handles_labels()):
            if l not in labels:
                handles.append(h)
                labels.append(l)
    if merged:
        handles.append(plt.Line2D([], [], color=INK2, lw=1, ls='--'))
        labels.append('dashed: per 80-bin bin (not tested)')
    for last in ('sub-step cap hit', f'> {CEIL:g}, drawn at {CEIL:g} (tangent unstable)'):
        if last in labels:
            k = labels.index(last)
            handles.append(handles.pop(k))
            labels.append(labels.pop(k))
    fig.legend(handles, labels, loc='upper center',
               ncol=min(len(labels), 4) if len(temps) >= 5 else 2,
               frameon=False, bbox_to_anchor=(0.5, 1.0))
    fig.suptitle(f'Size distribution vs fixed-dt reference at {hour} h, '
                 + ('40 bins, snapshot' if nbins == 40 else
                    f'80 bins, hour mean (L1 rows: solid {MERGE} bins merged, tested; '
                    'dashed per 80-bin bin)')
                 + (' (ringed: 80 bins merged to 40, hour mean, where run and reference exist)'
                    if d.get('bins80', np.zeros(1, bool)).any() else '')
                 + f'\n(grey band: pass, ≤ {TOL}; values below '
                 f'{FLOOR:g} drawn at {FLOOR:g})',
                 y=1 - 0.6 / height, color=INK2, fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 1 - 1.0 / height))
    name = (f'grid_error_time_{hour}h.png' if nbins == 40
            else f'grid_error_time_80bins_{hour}h.png')
    out = os.path.join(outdir, name)
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def load40(outdir):
    path40 = os.path.join(outdir, 'grid.npz')
    if not os.path.exists(path40):
        raise FileNotFoundError(f"Missing NPZ: {path40}. Run "
                                "benchmarks.python.gas_substep_convergence first.")
    z = np.load(path40)
    for key in ('eN', 'eM', 'edN', 'Nk', 'dNk'):
        if key not in z.files:
            raise KeyError(f"{path40} has no '{key}': it predates the "
                           "size-distribution metrics; rerun the benchmark.")
    return dict(z)


def overlay80(d):
    """Replace the error metrics of 40-bin rows by the 80-bin hour-mean ones (L1
    with MERGE bins merged, i.e. on the 40-bin grid) where psd80/ has that run
    and its reference, both with hour means, at that hour."""
    kind_of = {v: k for k, v in RUN_KEY.items()}
    d = {**d, 'bins80': np.zeros(len(d['T']), bool)}
    for k in ('eN', 'eM', 'edN'):
        d[k] = d[k].astype(float).copy()
    for i, (T, s, h, r) in enumerate(zip(d['T'], d['so2'], d['hour'], d['run'])):
        if r == 'ref':
            continue
        T, s, kind = float(T), float(s), kind_of[str(r)]
        if not all(done(T, s, r) for k in ('ref', kind) for r in runs_for(T, s, k)):
            continue
        A, R = load_run(T, s, kind), load_run(T, s, 'ref')
        if int(h) not in A['hours'] or int(h) not in R['hours']:
            continue
        e = hour_mean_errors(A, R, list(A['hours']).index(int(h)))
        for k in ('eN', 'eM', 'edN'):
            d[k][i] = e[k + '40']
        d['bins80'][i] = True
    return d


def load80():
    files = [(T, s, k) for T in TEMPS for s in SO2_PPBV for k in KINDS]
    missing = [path(T, s, r) for T, s, k in files for r in runs_for(T, s, k)
               if not done(T, s, r)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} 80-bin runs missing or without hour means "
                                f"(first {missing[:3]}); run "
                                "benchmarks.python.gas_substep_psd80 --all first.")
    rows = {k: [] for k in ('T', 'so2', 'run', 'hour', 'cap', 't', 't_compile',
                            'eN', 'eM', 'edN', 'eN40', 'eM40', 'edN40', 'ref_dN', 'fd')}
    for T, s, k in files:
        A, R = load_run(T, s, k), load_run(T, s, 'ref')
        for i, h in enumerate(A['hours']):
            if list(A['hours']) != list(R['hours']):
                raise ValueError(f"{path(T, s, k)}: hours {A['hours']} differ from the reference's")
            e = hour_mean_errors(A, R, i)
            for key, v in (('T', T), ('so2', s), ('run', RUN_KEY[k]), ('hour', int(h)),
                           ('cap', bool(A['cap'])), ('t', float(A['t'][i])),
                           ('t_compile', float(A['t_compile'])), ('fd', bool(A['fd'])),
                           *e.items()):
                rows[key].append(v)
    return {k: np.array(v) for k, v in rows.items()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=OUTDIR)
    parser.add_argument('--bins', type=int, choices=(40, 80), default=40)
    args = parser.parse_args()
    d = overlay80(load40(args.outdir)) if args.bins == 40 else load80()
    plt.rcParams.update({'font.size': 9, 'axes.edgecolor': INK2,
                         'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK})
    for hour in np.unique(d['hour']):
        plot_hour(d, int(hour), args.outdir, args.bins)


if __name__ == '__main__':
    main()
