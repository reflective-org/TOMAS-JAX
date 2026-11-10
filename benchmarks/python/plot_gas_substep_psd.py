"""Size distributions per T x SO2 cell: adaptive f = 0.1 against the fixed-dt
reference on the same grid, at 6 h and 24 h.

    python -m benchmarks.python.plot_gas_substep_psd [--sigma-bins 40|80] [--hours 6 24]

Reads grid.npz (40-bin runs, snapshots) and psd80/ (80-bin runs from
gas_substep_psd80, hour means: the pass condition there); a cell with both
80-bin runs (with hour means) is drawn at 80 bins as the hour mean, the others
at 40 as the snapshot, and every panel says which. Writes, in benchmarks/results/gas_substep/:
  - dndlogD_<h>h.png: the model's bins (thin steps, dN/dlog10 Dp) and their KDE
    (thick; gas_substep_convergence.kde, a Gaussian of KDE_SIGMA decades at
    each bin centre); the panel gives the L1 number, dry mass and sensitivity
    distances on the 40-bin grid (80-bin cells: pairs merged, as tested by
    gas_substep_psd80 --check).
  - dndlogD_sigma_<bins>bins_<h>h.png: the KDE and its L1 number distance at
    several widths (SIGMAS). With --sigma-bins 80 every
    cell needs both 80-bin runs.
"""
import os
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from tomas_jax.core.config import make_grid, make_grid_80bin, XK0
from sensitivities.dunne_nuc.run_map import TEMPS, SO2_PPBV
from benchmarks.python.gas_substep_convergence import (
    OUTDIR, TOL, KDE_SIGMA, KDE_X, edges_nm, kde, kde_l1, l1)
from benchmarks.python.gas_substep_psd80 import path, done, hour_mean_errors, load_run, runs_for

INK, INK2, GRID, SURF = '#1f1f1e', '#5f5e58', '#e4e3df', '#fcfcfb'
C_REF40, C_REF80, C_40, C_80 = '#76756e', INK, '#eb6834', '#2a78d6'
C_PASS, C_FAIL = '#1a7f37', '#c4261d'
# decades above the highest curve for n lines of text, in a 7-decade panel
HEADROOM = {3: 3.2, 4: 6.5}
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


def load80(T, so2, kind, hour):
    """The hour mean at `hour`, or None if the run (with hour means) is absent."""
    if not all(done(T, so2, r) for r in runs_for(T, so2, kind)):
        return None
    z = load_run(T, so2, kind)
    if hour not in z['hours']:
        return None
    i = list(z['hours']).index(hour)
    return dict(Nk=z['Nk_avg'][i], dNk=z['dNk_avg'][i], dt=float(z['dt']))


def pair(cells, T, so2, hour):
    """(grid, reference, model, distances on the 40-bin grid): the 80-bin runs
    if both exist (pairs of bins merged, as gas_substep_psd80 --check), else 40-bin."""
    ref80, mod80 = load80(T, so2, 'ref', hour), load80(T, so2, 'f0.1', hour)
    if ref80 is not None and mod80 is not None:
        A, R = load_run(T, so2, 'f0.1'), load_run(T, so2, 'ref')
        e = hour_mean_errors(A, R, list(A['hours']).index(hour))
        return G80, ref80, mod80, dict(eN=e['eN40'], eM=e['eM40'], edN=e['edN40'])
    c = cells[(T, so2, hour)]
    ref, mod = c['ref'], c['f=0.1']
    return G40, ref, mod, dict(eN=l1(mod['Nk'], ref['Nk'], ref['Nk']), eM=mod['eM'],
                               edN=l1(mod['dNk'], ref['dNk'], ref['Nk']))


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


def new_figure():
    return plt.subplots(len(SO2_PPBV), len(TEMPS),
                        figsize=(3.3 * len(TEMPS), 2.8 * len(SO2_PPBV)), facecolor=SURF)


def save(fig, outdir, name):
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = os.path.join(outdir, name)
    fig.savefig(out, dpi=100, facecolor=SURF)
    plt.close(fig)
    print(f"wrote {out}")


def colours(g):
    return (C_REF80, C_80) if g is G80 else (C_REF40, C_40)


def plot_hour(cells, hour, outdir):
    fig, axes = new_figure()
    n80 = 0
    for i, so2 in enumerate(SO2_PPBV[::-1]):
        for j, T in enumerate(TEMPS):
            ax = axes[i, j]
            style(ax)
            g, ref, mod, e = pair(cells, T, so2, hour)
            n80 += g is G80
            c_ref, c_mod = colours(g)
            for N, col, lw, ls in ((ref['Nk'], c_ref, 3.0, '-'),
                                   (mod['Nk'], c_mod, 1.8, (0, (4, 1.5)))):
                ax.stairs(N / g.w, g.edge, color=col, lw=0.8, baseline=None, alpha=0.9)
                ax.plot(10 ** KDE_X, kde(N, g.xk), color=col, lw=lw, ls=ls)
            lines = [(f'ref {g.nbins} bins, dt {ref["dt"]:g} s, '
                      + ('hour mean' if g is G80 else 'snapshot'), c_ref),
                     dist('L1@40 number  ', e['eN']),
                     dist('L1@40 dry mass', e['eM']),
                     dist('L1@40 sens.   ', e['edN'])]
            top = (ref['Nk'] / g.w).max()
            ax.set_ylim(top * 1e-7, top * 10 ** HEADROOM[len(lines)])
            write_lines(ax, lines)
            label_axes(ax, i, j, f'{T:g} K, SO₂ {so2:g} ppbv')
    fig.legend(handles=[
        Line2D([], [], color=C_REF80, lw=3, label='reference, 80 bins, hour mean'),
        Line2D([], [], color=C_80, lw=1.8, ls=(0, (4, 1.5)), label='adaptive f = 0.1, 80 bins, hour mean'),
        Line2D([], [], color=C_REF40, lw=3, label='reference, 40 bins, snapshot'),
        Line2D([], [], color=C_40, lw=1.8, ls=(0, (4, 1.5)), label='adaptive f = 0.1, 40 bins, snapshot'),
        Line2D([], [], color=INK2, lw=0.8, drawstyle='steps-mid', label='thin: per bin'),
        Line2D([], [], color=INK2, lw=2, label=f'thick: KDE, σ = {KDE_SIGMA} decade')],
        loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=10.5)
    fig.suptitle(f'Size distribution at {hour} h, adaptive f = 0.1 vs fixed-dt reference on the '
                 f'same grid: 80 bins (hour mean) in {n80} of {len(SO2_PPBV) * len(TEMPS)} cells, '
                 f'40 (snapshot) in the rest; L1@40: distances on the 40-bin grid (80 bins: pairs '
                 f'merged, as tested); tolerance {TOL}', y=0.985, fontsize=10, color=INK2)
    save(fig, outdir, f'dndlogD_{hour}h.png')


def plot_sigma(cells, hour, outdir, nbins):
    """KDE at each of SIGMAS: adaptive f = 0.1 against the reference, same grid."""
    if nbins == 80:
        missing = [path(T, s, k) for T in TEMPS for s in SO2_PPBV for k in ('ref', 'f0.1')
                   if not done(T, s, k)]
        if missing:
            raise FileNotFoundError(f"{len(missing)} 80-bin runs missing or without hour means "
                                    f"(first {missing[:3]}); "
                                    "run benchmarks.python.gas_substep_psd80 --all first.")
    g = G40 if nbins == 40 else G80
    c_ref, c_mod = colours(g)
    fig, axes = new_figure()
    for i, so2 in enumerate(SO2_PPBV[::-1]):
        for j, T in enumerate(TEMPS):
            ax = axes[i, j]
            style(ax)
            if nbins == 40:
                ref, mod = cells[(T, so2, hour)]['ref'], cells[(T, so2, hour)]['f=0.1']
            else:
                ref, mod = load80(T, so2, 'ref', hour), load80(T, so2, 'f0.1', hour)
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
            label_axes(ax, i, j, f'{T:g} K, SO₂ {so2:g} ppbv, ref dt {ref["dt"]:g} s')
    fig.legend(handles=[
        Line2D([], [], color=c_ref, lw=1.0, drawstyle='steps-mid', label='reference, per bin'),
        Line2D([], [], color=c_mod, lw=1.0, drawstyle='steps-mid', label='adaptive f = 0.1, per bin')]
        + [Line2D([], [], color=col, lw=2.6, label=f'reference KDE σ = {s}')
           for s, col in zip(SIGMAS, C_SIGMA)]
        + [Line2D([], [], color=c_mod, lw=1.2, ls=(0, (3, 1.5)), label='adaptive KDE (each σ)')],
        loc='upper center', ncol=6, frameon=False, bbox_to_anchor=(0.5, 1.0), fontsize=10.5)
    fig.suptitle(f'Size distribution at {hour} h, {nbins} bins '
                 f'({"snapshot" if nbins == 40 else "hour mean"}): adaptive dt (f = 0.1) vs fixed-dt '
                 f'reference; KDE L1 number distance at each σ [decades of Dp; a 40-bin width is 0.1] '
                 f'(tolerance {TOL})', y=0.985, fontsize=10, color=INK2)
    save(fig, outdir, f'dndlogD_sigma_{nbins}bins_{hour}h.png')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', default=OUTDIR)
    parser.add_argument('--sigma-bins', type=int, choices=(40, 80), default=40)
    parser.add_argument('--hours', type=int, nargs='+', choices=(6, 24), default=[6, 24])
    args = parser.parse_args()
    if not os.path.exists(os.path.join(args.outdir, 'grid.npz')):
        raise FileNotFoundError(f"Missing NPZ: {args.outdir}/grid.npz. Run "
                                "benchmarks.python.gas_substep_convergence first.")
    cells = load_cells(args.outdir)
    for hour in args.hours:
        plot_hour(cells, hour, args.outdir)
        plot_sigma(cells, hour, args.outdir, args.sigma_bins)


if __name__ == '__main__':
    main()
