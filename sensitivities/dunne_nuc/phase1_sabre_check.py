"""Phase 1 check: SABRE-binned output and its coefficient Jacobian.

One stratospheric point (217 K, 55 hPa, SO2 1 ppbv, background aerosol),
6 h, on the 40- and 80-bin grids. Saves the SABRE-binned dry dN/dlogDp, the
NMASS cumulative channels, and jacfwd of both w.r.t. NucleationParams.

Usage::

    python -m sensitivities.dunne_nuc.phase1_sabre_check            # run + plot
    python -m sensitivities.dunne_nuc.phase1_sabre_check --plot-only
"""
import os
import time
import argparse

import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.core.config import make_grid, make_grid_80bin, XK0
from tomas_jax.physics import nucleation as nuc
from .common import (STRAT, BOXVOL, NUC_PARAMS, lognormal_state, make_runner,
                     ppbv_to_molec)
from .sabre_bins import (SABRE_EDGES_M, SABRE_MID_M, NMASS_CUTOFFS_M,
                         SABRE_NUMBER_REL_ERR, rebin_to_sabre, nmass_cumulative,
                         model_bin_diameters)

OUTDIR = os.path.join(os.path.dirname(__file__), 'results', 'phase1')
GRIDS = {40: lambda: make_grid(40, XK0, 2.0), 80: make_grid_80bin}
SO2_PPBV = 1.0
HOURS = 6
PLOT_FIELDS = ('u_bn', 'u_bi', 'p_bi', 'r_nuc_nm')


def npz_path(nbins):
    return os.path.join(OUTDIR, f'sabre_jac_{nbins}bin.npz')


def run_grid(nbins):
    xk = GRIDS[nbins]()
    so2 = ppbv_to_molec(SO2_PPBV, STRAT['temp'], STRAT['pres'])
    Nk0, Mk0, Gc0 = lognormal_state(xk, so2=so2)
    run = make_runner(xk, nsteps=HOURS * 60)
    P = jax.tree.map(jnp.float64, NUC_PARAMS)

    def outputs(p):
        Nk, Mk, _, diag = run(p, Nk0, Mk0, Gc0)
        return (rebin_to_sabre(Nk, Mk, xk, BOXVOL),
                nmass_cumulative(Nk, Mk, xk, BOXVOL), (diag, Nk, Mk))

    t0 = time.time()
    dndlog, nmass, (diag, Nk, Mk) = jax.block_until_ready(jax.jit(outputs)(P))
    t_fwd = time.time() - t0
    t0 = time.time()
    J_bins, J_nmass = jax.block_until_ready(
        jax.jit(jax.jacfwd(lambda p: outputs(p)[:2]))(P))
    t_jac = time.time() - t0

    fields = nuc.NucleationParams._fields
    jac_bins = np.stack([np.array(getattr(J_bins, f)) for f in fields])
    jac_nmass = np.stack([np.array(getattr(J_nmass, f)) for f in fields])
    for name, arr in (('bins', jac_bins), ('nmass', jac_nmass)):
        if not np.all(np.isfinite(arr)):
            raise RuntimeError(f"{nbins}-bin: non-finite Jacobian ({name})")

    # Native model bins on the same dry-diameter edges the SABRE rebinning uses
    d_lo, d_hi = (np.array(d) for d in model_bin_diameters(Mk, xk))
    native_dndlog = np.array(Nk) / BOXVOL / np.log10(d_hi / d_lo)

    n_sub = np.array(diag['n_sub'])
    clamp = np.array(diag['clamp'])
    print(f"  {nbins}-bin: forward {t_fwd:.1f} s, jacfwd {t_jac:.1f} s "
          f"(incl. compile); N(>3 nm)={float(nmass[0]):.3e} cm-3; "
          f"steps with n_sub>1: {np.sum(n_sub > 1)}, clamp: {np.sum(clamp)}")
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez(npz_path(nbins), fields=np.array(fields),
             params=np.array(NUC_PARAMS, dtype=float),
             dndlog=np.array(dndlog), nmass=np.array(nmass),
             jac_bins=jac_bins, jac_nmass=jac_nmass, n_sub=n_sub, clamp=clamp,
             native_dndlog=native_dndlog, native_d_lo=d_lo, native_d_hi=d_hi,
             h2so4=np.array(diag['h2so4']), J=np.array(diag['J']))


def plot():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = {}
    for nbins in GRIDS:
        if not os.path.exists(npz_path(nbins)):
            raise FileNotFoundError(f"Missing NPZ: {npz_path(nbins)}. "
                                    f"Run without --plot-only first.")
        data[nbins] = np.load(npz_path(nbins))

    fig, axes = plt.subplots(1 + len(PLOT_FIELDS), 1, figsize=(7, 3 + 2.2 * len(PLOT_FIELDS)),
                             sharex=True)
    dmid_nm = SABRE_MID_M * 1e9
    ax = axes[0]
    for nbins, ls, mk, c in ((40, '--', 's', 'C0'), (80, '-', 'o', 'C1')):
        d = data[nbins]
        if 'native_dndlog' not in d:
            raise KeyError(f"{npz_path(nbins)} has no native_dndlog. "
                           "Rerun without --plot-only.")
        ax.stairs(d['native_dndlog'], np.append(d['native_d_lo'], d['native_d_hi'][-1]) * 1e9,
                  color=c, lw=4, alpha=0.3, label=f'{nbins}-bin model, native bins')
        ax.step(dmid_nm, d['dndlog'], where='mid', ls=ls, color=c,
                label=f'{nbins}-bin model, SABRE bins')
        ax.plot(NMASS_CUTOFFS_M * 1e9, d['nmass'], mk, ms=5, mfc='none', color=c,
                label=f'N(>Dc) NMASS, {nbins}-bin')
    ax.set_xscale('log'); ax.set_yscale('log')
    ax.set_ylim(1e-3, None)
    ax.set_ylabel('dN/dlogDp [cm$^{-3}$]\n(o: cumulative N)')
    ax.axvline(SABRE_EDGES_M[0] * 1e9, color='k', lw=0.6, ls=':', label='SABRE lower edge')
    ax.set_title(f'{HOURS} h at {STRAT["temp"]:.0f} K, SO$_2$ {SO2_PPBV} ppbv, '
                 'native and SABRE bins (dry)')
    ax.legend(fontsize=7)

    fields = list(data[80]['fields'])
    for ax, f in zip(axes[1:], PLOT_FIELDS):
        i = fields.index(f)
        for nbins, ls in ((40, '--'), (80, '-')):
            d = data[nbins]
            y = d['dndlog']
            rel = np.where(y > 1e-3, d['jac_bins'][i] / np.where(y > 0, y, 1.0), np.nan)
            ax.step(dmid_nm, rel, where='mid', ls=ls, label=f'{nbins}-bin')
        ax.axhspan(-SABRE_NUMBER_REL_ERR, SABRE_NUMBER_REL_ERR, color='0.85',
                   label='SABRE ±8% (per unit change)')
        ax.axhline(0, color='k', lw=0.5)
        ax.set_ylabel(f'd ln(dN/dlogDp)\n/ d {f}')
        ax.legend(fontsize=7)
    axes[-1].set_xlabel('Dry diameter [nm]')
    # Start below SABRE's 3 nm edge so the native grid's 1.7-3 nm bins show
    axes[-1].set_xlim(1., 2e3)
    fig.tight_layout()
    out = os.path.join(OUTDIR, 'sabre_jacobian.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--plot-only', action='store_true')
    args = parser.parse_args()
    if not args.plot_only:
        for nbins in GRIDS:
            run_grid(nbins)
    plot()


if __name__ == '__main__':
    main()
