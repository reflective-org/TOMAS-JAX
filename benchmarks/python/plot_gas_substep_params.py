"""Hourly total number and effective radius of the 80-bin gas_substep_psd80
runs against the fixed-dt reference: one figure per quantity, rows SO2,
columns temperature, x the hour (1-24; each point the mean distribution over
the hour before it), y |x / x_ref - 1|.

    python -m benchmarks.python.plot_gas_substep_params

    params_N.png     total number concentration, sum over the bins;
    params_reff.png  effective radius sum N r^3 / sum N r^2 from the bins
                     (benchmarks/python/lognormal_fit.moments; no fit).
Runs: adaptive f = 0.3, 0.1 and constant dt = 60 s, 0.25 s (all need the hourly
series; gas_substep_psd80 --all).
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tomas_jax.core.config import make_grid_80bin
from sensitivities.dunne_nuc import common as c
from sensitivities.dunne_nuc.run_map import TEMPS, SO2_PPBV
from benchmarks.python.gas_substep_convergence import OUTDIR, TOL
from benchmarks.python.gas_substep_psd80 import path, done
from benchmarks.python.plot_gas_substep_grid import (
    INK, INK2, GRID, SURFACE, SERIES, RUN_KEY, FLOOR, CEIL)
from benchmarks.python import lognormal_fit as lf

RUNS = ('f0.3', 'f0.1', 'dt60', 'dt0.25')
QUANTITIES = (('N', 'total number concentration'),
              ('reff', 'effective radius (bins)'))


def load():
    """{'N', 'reff'}: (T, SO2, 1 + len(RUNS), hours), the reference first."""
    cells = [(T, s) for T in TEMPS for s in SO2_PPBV]
    missing = [path(T, s, k) for T, s in cells for k in ('ref',) + RUNS
               if not done(T, s, k)]
    if missing:
        raise FileNotFoundError(f"{len(missing)} 80-bin runs missing or without the hourly "
                                f"series (first {missing[:3]}); run "
                                "benchmarks.python.gas_substep_psd80 --all first.")
    xk = np.asarray(make_grid_80bin())
    N, reff = [], []
    for T, s in cells:
        files = [np.load(path(T, s, k)) for k in ('ref',) + RUNS]
        N.append([A['Nk_series'].sum(-1) for A in files])
        reff.append([[lf.moments(n, m, xk, c.BOXVOL)['reff']
                      for n, m in zip(A['Nk_series'], A['Mk_series'])] for A in files])
    shape = (len(TEMPS), len(SO2_PPBV), 1 + len(RUNS), -1)
    return dict(N=np.reshape(N, shape), reff=np.reshape(reff, shape))


def rel(a, b):
    e = np.abs(a / b - 1.0)
    if not np.isfinite(e).all():
        raise ValueError("Non-finite relative error (a value is 0, NaN or inf)")
    return np.clip(e, FLOOR, CEIL)


def style(kind):
    key = RUN_KEY[kind]
    return next((label, colour) for k, label, colour, _ in SERIES if k == key)


def plot(d, name, label):
    fig, axes = plt.subplots(len(SO2_PPBV), len(TEMPS),
                             figsize=(2.3 * len(TEMPS), 1.45 * len(SO2_PPBV) + 1.0),
                             sharex=True, sharey=True, facecolor=SURFACE)
    hours = np.arange(1, d[name].shape[-1] + 1)
    for i, s in enumerate(SO2_PPBV):
        for j, T in enumerate(TEMPS):
            ax = axes[i, j]
            ax.set_facecolor(SURFACE)
            ax.grid(color=GRID, lw=0.6)
            ax.set_axisbelow(True)
            for side in ('top', 'right'):
                ax.spines[side].set_visible(False)
            ax.set_yscale('log')
            ax.axhspan(FLOOR, TOL, color=GRID, alpha=0.6, lw=0)
            ref = d[name][j, i, 0]
            for r, kind in enumerate(RUNS):
                run_label, colour = style(kind)
                ax.plot(hours, rel(d[name][j, i, 1 + r], ref), color=colour, lw=1.3,
                        label=run_label)
            if i == 0:
                ax.set_title(f'{T:g} K', color=INK)
            if j == 0:
                ax.set_ylabel(f'{s:g} ppbv', color=INK)
            if i == len(SO2_PPBV) - 1:
                ax.set_xlabel('hour')
    fig.suptitle(f'{label} vs fixed-dt reference, 80 bins\n'
                 f'|x / x_ref - 1|, hour means; grey band ≤ {TOL}; '
                 f'values outside [{FLOOR:g}, {CEIL:g}] drawn at the limit',
                 color=INK2, fontsize=9, y=0.998)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc='upper center', ncol=4,
               frameon=False, bbox_to_anchor=(0.5, 0.962))
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    out = os.path.join(OUTDIR, f'params_{name}.png')
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    print(f"wrote {out}")


def main():
    plt.rcParams.update({'font.size': 8, 'axes.edgecolor': INK2,
                         'axes.labelcolor': INK, 'xtick.color': INK2,
                         'ytick.color': INK2, 'text.color': INK})
    d = load()
    for name, label in QUANTITIES:
        plot(d, name, label)


if __name__ == '__main__':
    main()
