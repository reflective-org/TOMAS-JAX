"""Size distributions per cell: adaptive f = 0.1 against the fixed-dt
reference on the same grid, at 6 h and 24 h.

    python -m benchmarks.python.plot_gas_substep_psd [--sigma-bins 40|80] [--hours 6 24]

Reads psd80_tropical/ (80-bin runs from gas_substep_psd80, altitude x SO2 with
T and P from the AFGL tropical atmosphere, hour means: the pass condition
there) and, for --sigma-bins 40, grid.npz (40-bin runs, T x SO2 at 5500 Pa,
snapshots). Writes:
  - tropical/dndlogD_<h>h.png: the model's bins (steps, dN/dlog10 Dp; the
    reference solid, constant dt = 60 s thinner, adaptive dashed on top), 80
    bins; the panel gives the L1 number, dry
    mass and sensitivity distances with pairs of bins merged, as tested by
    gas_substep_psd80 --check. Every cell needs both 80-bin runs.
  - tropical/residual_<h>h.png: the number concentration error per merged
    bin (pairs of 80-bin bins, as tested), N_k - N_k,ref [cm-3], of constant
    dt = 60 s, the old unsplit-chemistry reference and adaptive f = 0.1, on a
    symlog axis (linear up to the largest f = 0.1 / old-reference error);
    sum |error| / sum N_ref is the L1@40 number distance; shaded: +-5% of
    the reference's concentration in each bin.
  - tropical/dndlogD_sigma_80bins_<h>h.png or dndlogD_sigma_40bins_<h>h.png:
    the KDE and its L1 number distance at several widths (SIGMAS).
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

from tomas_jax.core.config import make_grid, make_grid_80bin, XK0
from sensitivities.dunne_nuc.run_map import TEMPS, SO2_PPBV
from benchmarks.python.gas_substep_convergence import (
    OUTDIR, TOL, KDE_X, edges_nm, kde, kde_l1)
from benchmarks.python.gas_substep_psd80 import (
    ALTITUDES, MERGE, path, done, hour_mean_errors, load_run, runs_for, reference, merge)

INK, INK2, GRID, SURF = '#1f1f1e', '#5f5e58', '#e4e3df', '#fcfcfb'
C_REF40, C_REF80, C_40, C_80 = '#76756e', INK, '#eb6834', '#2a78d6'
C_60 = '#eda100'                                       # constant dt = 60 s
C_OLD = '#1f9e89'                                      # old unsplit reference
C_BAND, BAND = '#dcdad3', 0.05                         # +-5% of the reference per bin
C_PASS, C_FAIL = '#1a7f37', '#c4261d'
# decades above the highest curve for n lines of text, in a 7-decade panel
HEADROOM = {3: 3.2, 4: 6.5, 5: 9.8}
SIGMAS = (0.05, 0.075, 0.1)                           # decades of Dp
C_SIGMA = ('#8fbbe3', '#3f7fc4', '#0b3d75')


class Grid:
    def __init__(self, xk, nbins):
        self.xk, self.nbins = np.asarray(xk), nbins
        self.edge = edges_nm(xk)
        self.w = np.log10(2) / 3 * 40 / nbins        # bin width in log10 Dp


G40 = Grid(make_grid(40, XK0, 2.0), 40)
G80 = Grid(make_grid_80bin(), 80)


def load_cells(outdir):
    d = np.load(os.path.join(outdir, 'grid.npz'))
    cells = {}
    for i in range(len(d['T'])):
        k = (float(d['T'][i]), float(d['so2'][i]), int(d['hour'][i]))
        run = str(d['run'][i])
        if run in ('ref', 'f=0.1'):
            cells.setdefault(k, {})[run] = dict(Nk=d['Nk'][i], dNk=d['dNk'][i],
                                                dt=float(d['dt'][i]), eM=float(d['eM'][i]))
    return cells


def require80(kinds):
    # 'ref' stands for the cell's reference (gas_substep_psd80.reference)
    missing = [path(z, s, r) for z in ALTITUDES for s in SO2_PPBV for k in kinds
               for r in runs_for(z, s, {'ref': reference(s), 'ref_unsplit': 'ref'}.get(k, k))
               if not done(z, s, r)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} 80-bin runs missing or without hour means "
                                f"(first {missing[:3]}); "
                                "run benchmarks.python.gas_substep_psd80 --all first.")


def load80(z, so2, kind, hour):
    """The hour mean at `hour`; kind 'ref' is the cell's reference."""
    if kind == 'ref':
        kind = reference(so2)
    elif kind == 'ref_unsplit':
        kind = 'ref'
    A = load_run(z, so2, kind)
    if hour not in A['hours']:
        raise ValueError(f"{path(z, so2, kind)} has no hour {hour} (hours {A['hours']})")
    i = list(A['hours']).index(hour)
    return dict(Nk=A['Nk_avg'][i], dNk=A['dNk_avg'][i], dt=float(A['dt']))


def errors80(z, so2, hour):
    """Distances of f = 0.1 to the reference, pairs of bins merged (as tested)."""
    A, R = load_run(z, so2, 'f0.1'), load_run(z, so2, reference(so2))
    e = hour_mean_errors(A, R, list(A['hours']).index(hour))
    return dict(eN=e['eN40'], eM=e['eM40'], edN=e['edN40'])


def column_title(z):
    T, P = ALTITUDES[z]
    return f'{z:g} km ({T:g} K, {P / 100:.0f} hPa)'


def outdir_for(outdir, nbins):
    if nbins == 40:
        return outdir
    out = os.path.join(outdir, 'tropical')
    os.makedirs(out, exist_ok=True)
    return out


def style(ax):
    ax.set_facecolor(SURF)
    ax.grid(color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(1.5, 3000)
    ax.tick_params(labelsize=8)


def dist(label, v):
    return (f'{label} {v:.3f}' if v < 10 else f'{label} {v:.1e}') + (' ✗' if v > TOL else ' ✓'), \
        C_FAIL if v > TOL else C_PASS


def write_lines(ax, lines):
    for k, (t, col) in enumerate(lines):
        ax.text(0.98, 0.97 - 0.11 * k, t, transform=ax.transAxes, fontsize=8.5,
                ha='right', va='top', family='monospace', color=col, fontweight='bold',
                bbox=dict(fc=SURF, ec='none', alpha=0.8, pad=0.8))


def label_axes(ax, i, j, title):
    ax.set_title(title, fontsize=9.5, loc='left', color=INK)
    if j == 0:
        ax.set_ylabel('dN/dlog₁₀Dp [cm⁻³]', fontsize=8.5)
    if i == len(SO2_PPBV) - 1:
        ax.set_xlabel('dry diameter [nm]', fontsize=8.5)


def new_figure(ncols):
    return plt.subplots(len(SO2_PPBV), ncols,
                        figsize=(3.3 * ncols, 2.8 * len(SO2_PPBV)), facecolor=SURF)


def save(fig, outdir, name):
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(outdir, name)
    fig.savefig(out, dpi=100, facecolor=SURF)
    plt.close(fig)
    print(f"wrote {out}")


def colours(g):
    return (C_REF80, C_80) if g is G80 else (C_REF40, C_40)


def plot_hour(hour, outdir):
    require80(('ref', 'f0.1', 'dt60'))
    g = G80
    c_ref, c_mod = colours(g)
    fig, axes = new_figure(len(ALTITUDES))
    for i, so2 in enumerate(SO2_PPBV[::-1]):
        for j, z in enumerate(ALTITUDES):
            ax = axes[i, j]
            style(ax)
            ref, mod = load80(z, so2, 'ref', hour), load80(z, so2, 'f0.1', hour)
            dt60 = load80(z, so2, 'dt60', hour)
            e = errors80(z, so2, hour)
            # reference wide and solid, dt = 60 s thinner, adaptive dashed on top so
            # the reference shows through
            for N, col, lw, ls in ((ref['Nk'], c_ref, 3.0, '-'),
                                   (dt60['Nk'], C_60, 1.4, '-'),
                                   (mod['Nk'], c_mod, 1.6, (0, (3, 2)))):
                ax.stairs(N / g.w, g.edge, color=col, lw=lw, ls=ls, baseline=None)
            lines = [(f'ref {g.nbins} bins, dt {ref["dt"]:g} s, hour mean', c_ref),
                     dist('L1@40 number  ', e['eN']),
                     dist('L1@40 dry mass', e['eM']),
                     dist('L1@40 sens.   ', e['edN'])]
            if not np.isfinite(dt60['Nk']).all():
                lines.append(('dt 60 s run went NaN', C_60))
            top = (ref['Nk'] / g.w).max()
            ax.set_ylim(top * 1e-7, top * 10 ** HEADROOM[len(lines)])
            write_lines(ax, lines)
            label_axes(ax, i, j, f'{column_title(z)}, SO₂ {so2:g} ppbv')
    fig.legend(handles=[
        Line2D([], [], color=C_REF80, lw=3, label='reference, 80 bins, hour mean'),
        Line2D([], [], color=C_60, lw=1.4, label='constant dt = 60 s, 80 bins, hour mean'),
        Line2D([], [], color=C_80, lw=1.6, ls=(0, (3, 2)), label='adaptive f = 0.1, 80 bins, hour mean')],
        loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=10.5)
    fig.suptitle(f'Size distribution at {hour} h, AFGL tropical altitudes, 80 bins (hour mean): '
                 f'adaptive f = 0.1 and constant dt = 60 s vs split-chemistry fixed-dt reference; L1@40 (f = 0.1): distances with pairs of bins '
                 f'merged, as tested; tolerance {TOL}', y=0.985, fontsize=10, color=INK2)
    save(fig, outdir_for(outdir, 80), f'dndlogD_{hour}h.png')


def plot_residual(hour, outdir):
    require80(('ref', 'f0.1', 'ref_unsplit', 'dt60'))
    edge = G80.edge[::MERGE]
    fig, axes = new_figure(len(ALTITUDES))
    for i, so2 in enumerate(SO2_PPBV[::-1]):
        for j, z in enumerate(ALTITUDES):
            ax = axes[i, j]
            style(ax)
            N_ref = merge(load80(z, so2, 'ref', hour)['Nk'])
            ax.stairs(BAND * N_ref, edge, baseline=-BAND * N_ref, fill=True,
                      color=C_BAND, lw=0, zorder=0)
            lines, small = [], 0.0
            for kind, col, lw, ls, name in (
                    ('dt60', C_60, 1.4, '-', 'dt 60 s'),
                    ('ref_unsplit', C_OLD, 1.4, '-', 'old ref'),
                    ('f0.1', C_80, 1.6, (0, (3, 2)), 'f = 0.1')):
                dN = merge(load80(z, so2, kind, hour)['Nk']) - N_ref
                if not np.isfinite(dN).all():
                    lines.append((f'{name:8s} went NaN', col))
                    continue
                ax.stairs(dN, edge, color=col, lw=lw, ls=ls, baseline=None)
                if kind != 'dt60':
                    small = max(small, np.abs(dN).max())
                lines.append((f'{name:8s} L1@40 {np.abs(dN).sum() / N_ref.sum():.3f}', col))
            # linear where f = 0.1 and the old reference are, log beyond (dt = 60 s)
            ax.set_yscale('symlog', linthresh=max(small, 1e-12), linscale=2.0)
            ax.axhline(0.0, color=INK2, lw=0.8)
            lim = max(np.abs(ax.get_ylim()))
            ax.set_ylim(-lim, lim * 10 ** (0.6 * len(lines)) if lim > small else lim * 3)
            write_lines(ax, lines)
            ax.set_title(f'{column_title(z)}, SO₂ {so2:g} ppbv', fontsize=9.5,
                         loc='left', color=INK)
            if j == 0:
                ax.set_ylabel('N − N_ref per merged bin [cm⁻³]', fontsize=8.5)
            if i == len(SO2_PPBV) - 1:
                ax.set_xlabel('dry diameter [nm]', fontsize=8.5)
    fig.legend(handles=[
        Line2D([], [], color=C_60, lw=1.4, label='constant dt = 60 s'),
        Line2D([], [], color=C_OLD, lw=1.4,
               label='old reference, chemistry not split (2x the dt from 1 ppmv)'),
        Line2D([], [], color=C_80, lw=1.6, ls=(0, (3, 2)), label='adaptive f = 0.1'),
        Patch(color=C_BAND, label=f'±{BAND:.0%} of the reference in that bin')],
        loc='upper center', ncol=4, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=10.5)
    fig.suptitle(f'Number concentration error at {hour} h vs the split-chemistry reference, 80-bin '
                 f'hour means with pairs of bins merged (as tested); symlog y: linear up to the '
                 f'largest f = 0.1 / old-reference error, log beyond; L1@40 = sum|error| / N_ref, '
                 f'tolerance {TOL}', y=0.985, fontsize=10, color=INK2)
    save(fig, outdir_for(outdir, 80), f'residual_{hour}h.png')


def plot_sigma(cells, hour, outdir, nbins):
    """KDE at each of SIGMAS: adaptive f = 0.1 against the reference, same grid."""
    if nbins == 80:
        require80(('ref', 'f0.1'))
    g = G40 if nbins == 40 else G80
    c_ref, c_mod = colours(g)
    cols = TEMPS if nbins == 40 else list(ALTITUDES)
    fig, axes = new_figure(len(cols))
    for i, so2 in enumerate(SO2_PPBV[::-1]):
        for j, cell in enumerate(cols):
            ax = axes[i, j]
            style(ax)
            if nbins == 40:
                ref, mod = cells[(cell, so2, hour)]['ref'], cells[(cell, so2, hour)]['f=0.1']
            else:
                ref, mod = load80(cell, so2, 'ref', hour), load80(cell, so2, 'f0.1', hour)
            ax.stairs(ref['Nk'] / g.w, g.edge, color=c_ref, lw=1.0, baseline=None)
            ax.stairs(mod['Nk'] / g.w, g.edge, color=c_mod, lw=1.0, baseline=None)
            lines = []
            for s, col in zip(SIGMAS, C_SIGMA):
                ax.plot(10 ** KDE_X, kde(ref['Nk'], g.xk, s), color=col, lw=2.6)
                lines.append(dist(f'KDE σ {s:<5g}', kde_l1(mod['Nk'], g.xk, ref['Nk'], g.xk,
                                                           ref['Nk'], s)))
            for s in SIGMAS:     # adaptive on top, in its own colour
                ax.plot(10 ** KDE_X, kde(mod['Nk'], g.xk, s), color=c_mod, lw=1.2,
                        ls=(0, (3, 1.5)))
            top = (ref['Nk'] / g.w).max()
            ax.set_ylim(top * 1e-7, top * 10 ** HEADROOM[len(lines)])
            write_lines(ax, lines)
            title = f'{cell:g} K' if nbins == 40 else column_title(cell)
            label_axes(ax, i, j, f'{title}, SO₂ {so2:g} ppbv, ref dt {ref["dt"]:g} s')
    fig.legend(handles=[
        Line2D([], [], color=c_ref, lw=1.0, drawstyle='steps-mid', label='reference, per bin'),
        Line2D([], [], color=c_mod, lw=1.0, drawstyle='steps-mid', label='adaptive f = 0.1, per bin')]
        + [Line2D([], [], color=col, lw=2.6, label=f'reference KDE σ = {s}')
           for s, col in zip(SIGMAS, C_SIGMA)]
        + [Line2D([], [], color=c_mod, lw=1.2, ls=(0, (3, 1.5)), label='adaptive KDE (each σ)')],
        loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=10.5)
    fig.suptitle(f'Size distribution at {hour} h, {nbins} bins '
                 f'({"snapshot, 5500 Pa" if nbins == 40 else "hour mean, AFGL tropical altitudes"}): adaptive dt (f = 0.1) vs fixed-dt '
                 f'reference; KDE L1 number distance at each σ [decades of Dp; a 40-bin width is 0.1] '
                 f'(tolerance {TOL})', y=0.985, fontsize=10, color=INK2)
    save(fig, outdir_for(outdir, nbins), f'dndlogD_sigma_{nbins}bins_{hour}h.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=OUTDIR)
    parser.add_argument('--sigma-bins', type=int, choices=(40, 80), default=80)
    parser.add_argument('--hours', type=int, nargs='+', choices=(6, 24), default=[6, 24])
    args = parser.parse_args()
    cells = None
    if args.sigma_bins == 40:
        if not os.path.exists(os.path.join(args.outdir, 'grid.npz')):
            raise FileNotFoundError(f"Missing NPZ: {args.outdir}/grid.npz. Run "
                                    "benchmarks.python.gas_substep_convergence first.")
        cells = load_cells(args.outdir)
    for hour in args.hours:
        plot_hour(hour, args.outdir)
        plot_residual(hour, args.outdir)
        plot_sigma(cells, hour, args.outdir, args.sigma_bins)


if __name__ == '__main__':
    main()
