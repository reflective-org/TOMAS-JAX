"""Phase 4: does the linear (AD) prediction hold for published-size changes?

For each Phase 3 scenario and grid, the model is rerun (forward, vmapped)
with the coefficients moved alpha in {-2, -1, +1, +2} along the published
uncertainty directions J_bn x2.5, J_bi x2.5 and the H2SO4 calibration, and a
joint direction (J_all x2.5 + H2SO4 calibration): 16 runs, at the Phase 3 dt
and coagulation solver. The nonlinear change ln(Y_alpha / Y_0) is compared with
the linear prediction alpha (J.d) / Y_0 from the Phase 3 Jacobian, for the
NMASS channels and the SABRE bins at 6, 24 and 48 h.

A cell is 'nonlinear' where |nonlinear - linear| > 20% of |nonlinear| at
alpha = +-1, counted only where the change exceeds SABRE's 8% (smaller
changes are not detectable anyway).

Usage::

    python -m sensitivities.dunne_nuc.phase4_linearity            # needs Phase 3 NPZs
    python -m sensitivities.dunne_nuc.phase4_linearity --smoke    # on phase3_smoke
    python -m sensitivities.dunne_nuc.phase4_linearity --plot-only
"""
import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from .phase3_scenarios import (RESULTS, eval_hours, load, make_sim, scenario_grids,
                               shifted)  # first: sets the host device count
from .common import vmap_over_devices
from .run_map import GRIDS
from .sabre_bins import (NMASS_CUTOFFS_M, NUC_MODE_CH, NUC_MODE_LABEL, SABRE_MID_M,
                         SABRE_NUMBER_REL_ERR, add_nuc_mode_channel)
from .scenarios import SCENARIOS

ALPHAS = (-2.0, -1.0, 1.0, 2.0)
PERTURB = ('J_bn x2.5', 'J_bi x2.5', 'H2SO4 cal', 'joint')
NONLIN_TOL = 0.20
FLOOR_NMASS, FLOOR_DNDLOG = 1e-3, 1e-2
CHANS = (0, NUC_MODE_CH, 2, 4)


def direction_vectors(d3):
    names = list(d3['dir_names'])
    D = {n: d3['directions'][names.index(n)] for n in names}
    D['joint'] = D['J_all x2.5'] + D['H2SO4 cal']
    return {p: D[p] for p in PERTURB}


def run_perturbations(name, nbins, p3dir, outdir):
    d3 = load(p3dir, name, nbins)
    hours, dt, coag = int(d3['hours']), float(d3['dt']), str(d3['coag_solver'])
    dirs = direction_vectors(d3)
    deltas = [a * dirs[p] for p in PERTURB for a in ALPHAS]
    f = make_sim(name, GRIDS[nbins](), hours, dt, coag)
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *[shifted(jnp.array(d)) for d in deltas])
    t0 = time.time()
    ys, aux = jax.block_until_ready(vmap_over_devices(f, batch))
    out = dict(nmass=np.array(ys['nmass']).reshape(len(PERTURB), len(ALPHAS), hours, -1),
               dndlog=np.array(ys['dndlog']).reshape(len(PERTURB), len(ALPHAS), hours, -1),
               clamp=np.array(aux['frac_clamp']).reshape(len(PERTURB), len(ALPHAS), hours),
               directions=np.stack([dirs[p] for p in PERTURB]),
               perturb=np.array(PERTURB), alphas=np.array(ALPHAS))
    for k in ('nmass', 'dndlog'):
        if not np.all(np.isfinite(out[k])):
            raise RuntimeError(f"{name} {nbins}-bin: non-finite perturbed {k}")
    np.savez(os.path.join(outdir, f'{name}_{nbins}bin.npz'), **out)
    print(f"    {name} {nbins}-bin: {len(deltas)} runs {time.time() - t0:.0f} s", flush=True)
    return out


def load4(outdir, name, nbins):
    fname = os.path.join(outdir, f'{name}_{nbins}bin.npz')
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing NPZ: {fname}. Run simulations first.")
    return add_nuc_mode_channel(np.load(fname))


def chan_label(ch):
    return NUC_MODE_LABEL if ch == NUC_MODE_CH else f'>{NMASS_CUTOFFS_M[ch] * 1e9:.0f} nm'


def compare(d3, d4, key, floor):
    """(nonlinear, linear) Delta ln Y: (nperturb, nalpha, hours, nout); NaN below floor."""
    y0 = d3[key]
    J = d3[f'jac_{key}']
    lin = np.stack([np.stack([a * (J @ d) for a in ALPHAS]) for d in d4['directions']])
    lin = lin / np.where(y0 > 0, y0, 1.0)
    ya = d4[key]
    ok = (y0 > floor) & (ya > floor)
    nl = np.where(ok, np.log(np.where(ok, ya, 1.0) / np.where(y0 > floor, y0, 1.0)), np.nan)
    return nl, np.where(ok, lin, np.nan)


def nonlinear_flags(nl, lin):
    """At alpha = +-1: detectable (> SABRE 8%) and |nl - lin| > 20% |nl|."""
    det = np.log(1.0 + SABRE_NUMBER_REL_ERR)
    idx = [ALPHAS.index(-1.0), ALPHAS.index(1.0)]
    a, b = nl[:, idx], lin[:, idx]
    with np.errstate(invalid='ignore'):
        return (np.abs(a) > det) & (np.abs(a - b) > NONLIN_TOL * np.abs(a))


def summary_table(names, grids, p3dir, outdir):
    lines = ["| scenario | grid | perturbation | channel | hour | Δln N(−1) nl / lin | "
             "Δln N(+1) nl / lin | Δln N(+2) nl / lin | nonlinear at ±1 |",
             "|---|---|---|---|---|---|---|---|---|"]
    n_flag = {}
    for name in names:
        for g in scenario_grids(name, grids):
            d3 = load(p3dir, name, g)
            d4 = load4(outdir, name, g)
            nl, lin = compare(d3, d4, 'nmass', FLOOR_NMASS)
            flags = nonlinear_flags(nl, lin)
            nl_s, lin_s = compare(d3, d4, 'dndlog', FLOOR_DNDLOG)
            n_flag[(name, g)] = (int(np.sum(nonlinear_flags(nl_s, lin_s))),
                                 int(np.sum(np.abs(nl_s[:, [1, 2]]) >
                                            np.log(1 + SABRE_NUMBER_REL_ERR))))
            for p, pn in enumerate(PERTURB):
                for hr in eval_hours(int(d3['hours'])):
                    for ch in CHANS:
                        v = lambda a: (f"{nl[p, ALPHAS.index(a), hr - 1, ch]:+.3f} / "
                                       f"{lin[p, ALPHAS.index(a), hr - 1, ch]:+.3f}")
                        fl = flags[p, :, hr - 1, ch]
                        lines.append(f"| {name} | {g} | {pn} | {chan_label(ch)} "
                                     f"| {hr} | {v(-1.0)} | {v(1.0)} | {v(2.0)} | "
                                     f"{'**yes**' if fl.any() else 'no'} |")
    lines += ["", "SABRE bins (all hours, alpha = ±1): nonlinear / detectable cells", ""]
    lines += [f"- {n} {g} bins: {a} / {b}" for (n, g), (a, b) in n_flag.items()]
    with open(os.path.join(outdir, 'linearity_summary.md'), 'w') as fh:
        fh.write('\n'.join(lines) + '\n')
    return n_flag


def plot(names, grids, p3dir, outdir):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    det = np.log(1.0 + SABRE_NUMBER_REL_ERR)
    gmax = lambda name: max(scenario_grids(name, grids))
    chans = CHANS

    # --- A: Delta ln N vs alpha at the last hour, per channel ---------------
    fig, axes = plt.subplots(len(names), len(chans), figsize=(4.2 * len(chans), 3.4 * len(names)),
                             squeeze=False)
    for r, name in enumerate(names):
        d3, d4 = load(p3dir, name, gmax(name)), load4(outdir, name, gmax(name))
        nl, lin = compare(d3, d4, 'nmass', FLOOR_NMASS)
        a = np.array(ALPHAS)
        for c, ch in enumerate(chans):
            ax = axes[r, c]
            for p, pn in enumerate(PERTURB):
                col = f'C{p}'
                ax.plot(a, nl[p, :, -1, ch], 'o', color=col, label=pn)
                ax.plot([a[0], 0, a[-1]], [lin[p, 0, -1, ch], 0, lin[p, -1, -1, ch]],
                        '--', color=col, lw=0.9)
            ax.axhspan(-det, det, color='0.9')
            ax.axhline(0, color='k', lw=0.5)
            ax.set_xlabel(r'$\alpha$ (published-size steps)')
            ax.set_title(f'{name}: $\\Delta$ln N({chan_label(ch)}), '
                         f'{int(d3["hours"])} h, {gmax(name)} bins', fontsize=9)
        axes[r, 0].legend(fontsize=7)
    fig.suptitle('Phase 4: nonlinear runs (points) vs linear AD prediction (dashed); '
                 'band: SABRE ±8%', fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'phase4_alpha.png'), dpi=130)
    plt.close(fig)

    # --- B: spectra at alpha = +-1, last hour -------------------------------
    fig, axes = plt.subplots(len(names), len(PERTURB), figsize=(4 * len(PERTURB),
                                                                3.2 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        d3 = load(p3dir, name, gmax(name))
        d4 = load4(outdir, name, gmax(name))
        nl, lin = compare(d3, d4, 'dndlog', FLOOR_DNDLOG)
        for p, pn in enumerate(PERTURB):
            ax = axes[r, p]
            for a, col in ((-1.0, 'C0'), (1.0, 'C3')):
                k = ALPHAS.index(a)
                ax.semilogx(SABRE_MID_M * 1e9, nl[p, k, -1], '-', color=col,
                            label=f'nonlinear α={a:+g}')
                ax.semilogx(SABRE_MID_M * 1e9, lin[p, k, -1], '--', color=col,
                            label=f'linear α={a:+g}')
            ax.axhspan(-det, det, color='0.9')
            ax.set_title(f'{name}: {pn}', fontsize=9)
            ax.set_xlabel('D$_p$ [nm]')
        axes[r, 0].set_ylabel(r'$\Delta$ln dN/dlogD$_p$')
        axes[r, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'phase4_spectra.png'), dpi=130)
    plt.close(fig)
    print(f"  wrote figures to {outdir}")


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--plot-only', action='store_true')
    ap.add_argument('--grids', type=int, nargs='+', default=[40, 80])
    ap.add_argument('--scenarios', nargs='+', default=list(SCENARIOS))
    args = ap.parse_args()
    if args.smoke:
        names, grids = ['sai_plume'], [40]
        p3dir, outdir = (os.path.join(RESULTS, d) for d in ('phase3_smoke', 'phase4_smoke'))
    else:
        names, grids = args.scenarios, args.grids
        p3dir, outdir = (os.path.join(RESULTS, d) for d in ('phase3', 'phase4'))
    os.makedirs(outdir, exist_ok=True)
    if not args.plot_only:
        for name in names:
            for g in scenario_grids(name, grids):
                run_perturbations(name, g, p3dir, outdir)
    n_flag = summary_table(names, grids, p3dir, outdir)
    for (n, g), (a, b) in n_flag.items():
        print(f"    {n} {g} bins: {a} nonlinear of {b} detectable SABRE-bin changes at α=±1")
    plot(names, grids, p3dir, outdir)


if __name__ == '__main__':
    main()
