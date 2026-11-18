"""40- vs 80-bin difference (hour means, 80-bin bins merged in pairs) from
results/gas_substep/psd40 and psd80.

  grid_40v80_<kind>.png   heatmaps over T x SO2 at 6 h and 24 h: hour-mean L1 number and dry mass
                          (sum_k |X40_k - X80_k| / X80_tot on the 40-bin grid) and N40/N80; outlined: |N40/N80 - 1| > 8%, dry-mass L1 > 19% (SABRE inversion error)
  psd_40v80_<kind>[_6h][_linear].png  dN/dlog10 Dp (log and linear y), 40 vs 80 bins (native 80), 24 h (6 h with _6h) hour mean,
                          every SO2 level (columns) x every T or altitude (rows, increasing upward as in the heatmaps), x from 1 nm
  --grid tropical: the same over altitude (13-25 km) x SO2, files
  grid_40v80_tropical_<kind>.png, psd_40v80_tropical_<kind>.png

    python -m benchmarks.python.plot_gas_substep_40v80 [--grid tropical] [--kind f0.1]
"""
import os
import argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.patches import Rectangle

from tomas_jax.core.config import make_grid, make_grid_80bin, XK0
from sensitivities.dunne_nuc.run_map import SO2_PPBV
from benchmarks.python.gas_substep_convergence import OUTDIR, l1, edges_nm


def l2(a, ref):
    """Relative L2 distance |a - ref|_2 / |ref|_2 of per-bin values."""
    return float(np.sqrt(np.sum((a - ref) ** 2) / np.sum(ref ** 2)))

from benchmarks.python.gas_substep_psd40 import Grid, merge


# SABRE inversion standard relative error (Lyu et al. 2026 SuppMat, Table S2 and text: number
# < 8%, volume < 19%); cells beyond it are outlined. Dry mass is volume.
SABRE_ERR = {'eM': 0.19, 'rN': 0.08}      # no outline on the number L1 (its bound is for total number)


def load(grid, key, s, kind):
    out = []
    for f in (grid.p40(key, s, kind), grid.p80(key, s, kind)):
        if not os.path.exists(f):
            raise FileNotFoundError(f"Missing NPZ: {f}. Run gas_substep_psd40 --all / gas_substep_psd80 first.")
        out.append(np.load(f))
    return out


def grids(grid, kind):
    keys_ = [x[0] for x in grid.cells]
    shape = (len(keys_), len(SO2_PPBV))
    keys = ('eN', 'eM', 'rN')
    G = {h: {k: np.full(shape, np.nan) for k in keys} for h in (6, 24)}
    for i, key in enumerate(keys_):
        for j, s in enumerate(SO2_PPBV):
            a, b = load(grid, key, s, kind)
            for hi, h in enumerate(a['hours']):
                N8, M8 = merge(b['Nk_avg'][hi]), merge(b['Mk_avg'][hi])
                g = G[int(h)]
                g['eN'][i, j] = l1(a['Nk_avg'][hi], N8, N8)
                g['eM'][i, j] = l1(a['Mk_avg'][hi], M8, M8)
                g['rN'][i, j] = a['Nk_avg'][hi].sum() / N8.sum()
    if any(np.isnan(v).any() for g in G.values() for v in g.values()):
        raise ValueError('NaN in the 40-vs-80 grids')
    return G


def heatmaps(grid, kind, out):
    G = grids(grid, kind)
    rows = [f'{x[1]:g}' if grid.name == 'T' else f'{x[0]:g} km\n{x[1]:.0f} K' for x in grid.cells]
    panels = [('eN', r'$\sum_k |N^{40}_k - N^{80}_k| \, / \, N^{80}_{tot}$', 'viridis',
               LogNorm(0.005, 0.5), '{:.2f}'),
              ('eM', r'$\sum_k |M^{40}_k - M^{80}_k| \, / \, M^{80}_{tot}$', 'viridis',
               LogNorm(0.005, 0.5), '{:.2f}'),
              ('rN', r'$N^{40}_{tot} \, / \, N^{80}_{tot}$', 'RdBu_r',
               TwoSlopeNorm(1.0, 0.5, 1.5), '{:.2f}')]
    fig, ax = plt.subplots(2, 3, figsize=(15, 7.5), constrained_layout=True)
    for r, h in enumerate((6, 24)):
        for c_, (key, title, cmap, norm, fmt) in enumerate(panels):
            a, v = ax[r, c_], G[h][key]
            im = a.imshow(v, origin='lower', cmap=cmap, norm=norm, aspect='auto')
            for i in range(v.shape[0]):
                for j in range(v.shape[1]):
                    a.text(j, i, fmt.format(v[i, j]), ha='center', va='center', fontsize=7)
            err = np.abs(v - 1.0) if key == 'rN' else v
            for i, j in zip(*np.nonzero(err > SABRE_ERR.get(key, np.inf))):
                a.add_patch(Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec='k', lw=2.0))
            a.set_xticks(range(len(SO2_PPBV)), [f'{s:g}' for s in SO2_PPBV], rotation=45, fontsize=8)
            a.set_yticks(range(len(rows)), rows, fontsize=8)
            a.set_title(f'{title}, {h} h', fontsize=11)
            if r == 1:
                a.set_xlabel('SO$_2$ [ppbv]')
            if c_ == 0:
                a.set_ylabel('T [K]' if grid.name == 'T' else 'altitude')
            fig.colorbar(im, ax=a, shrink=0.8)
    fig.suptitle(f'40 vs 80 bins ({grid.name} grid), run {kind}: hour means; k = 40-bin grid bins, X$^{{80}}_k$ = 80-bin pairs merged\n'
                 f'black outline: above SABRE inversion error (Lyu et al. 2026, Table S2): total number 8% (|N$^{{40}}$/N$^{{80}}$ - 1|), volume 19%')
    fig.savefig(out, dpi=130)
    plt.close(fig)


def psd(grid, kind, out, so2s=SO2_PPBV, yscale='log', hour=24):
    cells = list(grid.cells)[::-1]     # first cell at the bottom, as in the heatmaps (origin lower)
    e40 = edges_nm(make_grid(40, XK0, 2.0))
    e80 = edges_nm(make_grid_80bin())
    fig, ax = plt.subplots(len(cells), len(so2s), figsize=(3.1 * len(so2s), 2.6 * len(cells)),
                           sharex=True, constrained_layout=True)
    for i, (key, T, _) in enumerate(cells):
        for j, s in enumerate(so2s):
            a40, a80 = load(grid, key, s, kind)
            hi = list(a40['hours']).index(hour)
            for N, e, lab, col, ls, lw in (
                    (a80['Nk_avg'][hi], e80, '80 bins (raw)', 'k', '-', 1.0),
                    (merge(a80['Nk_avg'][hi]), e40, '80 bins merged to 40-bin grid', 'C0', '--', 1.6),
                    (a40['Nk_avg'][hi], e40, '40 bins', 'C3', '-', 1.3)):
                x = np.log10(e)
                y = N / np.diff(x)
                ax[i, j].stairs(np.maximum(y, 1e-6) if yscale == 'log' else y, 10 ** x,
                                color=col, ls=ls, label=lab, lw=lw)
            ax[i, j].set_xscale('log')
            ax[i, j].set_xlim(1.0, None)
            ax[i, j].set_yscale(yscale)
            ax[i, j].set_ylim(1e-3 if yscale == 'log' else 0, None)
            ax[i, j].set_title((f'{key:g} km, ' if grid.name != 'T' else '') + f'{T:g} K, {s:g} ppbv', fontsize=8)
            if j == 0:
                ax[i, j].set_ylabel('dN/dlog$_{10}$D$_p$ [cm$^{-3}$]')
            if i == len(cells) - 1:
                ax[i, j].set_xlabel('dry D$_p$ [nm]')
    h, l = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc='outside lower center', ncol=3, fontsize=10, frameon=False)
    fig.suptitle(f'{hour} h hour-mean size distributions ({grid.name} grid), run {kind}')
    fig.savefig(out, dpi=130)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--kind', default='f0.1')
    p.add_argument('--grid', default='T', choices=('T', 'tropical'))
    args = p.parse_args()
    grid = Grid(args.grid)
    tag = '' if args.grid == 'T' else f'{args.grid}_'
    heatmaps(grid, args.kind, os.path.join(OUTDIR, f'grid_40v80_{tag}{args.kind}.png'))
    for hour in (24, 6):
        sfx = '' if hour == 24 else f'_{hour}h'
        base = os.path.join(OUTDIR, f'psd_40v80_{tag}{args.kind}{sfx}')
        psd(grid, args.kind, base + '.png', hour=hour)
        psd(grid, args.kind, base + '_linear.png', yscale='linear', hour=hour)

if __name__ == '__main__':
    main()
