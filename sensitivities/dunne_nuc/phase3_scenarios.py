"""Phase 3: full-coefficient sensitivity for focused scenarios, with FD checks.

For each scenario in scenarios.SCENARIOS (bg_midlat, bg_ttl, sai_plume):

  1. dt ladder (40 bins): 60 -> 10 -> 2 -> 0.5 s; the coarsest dt whose
     N(>3 nm), N(>12 nm) and H2SO4-calibration sensitivity after 48 h agree
     with the next finer dt (5% / 0.05) is used. At that dt, fixed vs adaptive
     coagulation are compared; adaptive is used if they differ by > 5%.
  2. jax.jacfwd of hourly SABRE dN/dlogDp and NMASS N(>Dp) w.r.t. all 25
     NucleationParams fields, 40 and 80 bins, plus a Dunne Table S2 run.
  3. FD check (40 bins): the 4 published directions (alpha = +-0.1, +-0.01)
     and the 4 coefficients with the largest elasticity (h = 1e-3, 1e-4 |theta|),
     least-squares slope over a 9-point stencil vs AD at 6, 24 and 48 h.

Usage::

    python -m sensitivities.dunne_nuc.phase3_scenarios            # all + plots
    python -m sensitivities.dunne_nuc.phase3_scenarios --smoke    # 40 bins, 2 h, sai_plume
    python -m sensitivities.dunne_nuc.phase3_scenarios --plot-only
"""
import argparse
import json
import os
import time

# Six virtual CPU devices (the 6 performance cores), for the forward batches
# (FD here, Phase 4) split by common.vmap_over_devices: about 3x the
# throughput of one vmapped batch. jacfwd stays on one device, which already
# uses several threads. Must be set before JAX starts its backend; an
# explicit device count in XLA_FLAGS wins.
if 'xla_force_host_platform_device_count' not in os.environ.get('XLA_FLAGS', ''):
    os.environ['XLA_FLAGS'] = (os.environ.get('XLA_FLAGS', '')
                               + ' --xla_force_host_platform_device_count=6').strip()

import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.physics import nucleation as nuc
from .common import NUC_PARAMS, NUC_PARAMS_S2, vmap_over_devices
from .run_map import build_sim, directions, dlnY_dlnJall, FIELDS, GRIDS
from .sabre_bins import (NMASS_CUTOFFS_M, SABRE_MID_M, SABRE_NUMBER_REL_ERR,
                         NUC_MODE_CH, NUC_MODE_LABEL, add_nuc_mode_channel)
from .scenarios import SCENARIOS, HOURS, initial_state, runner_kwargs

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
DT_LADDER = (60.0, 10.0, 2.0, 0.5)
TOL_N, TOL_S = 0.05, 0.05
FD_HOURS = (6, 24, 48)
FD_ALPHA = 0.1              # direction step (alpha = 1: the published size)
FD_REL_STEP = 1e-2          # coefficient step, relative to |theta| (wide enough
                            # to average the step-to-step jitter, see fd_check)
FD_STENCIL = (-1.0, -0.75, -0.5, -0.25, 0.25, 0.5, 0.75, 1.0)
FD_TOL = 1e-3
FD_TOL_SABRE = 0.05         # agreement that matters against SABRE's 8%
P0 = jax.tree.map(jnp.float64, NUC_PARAMS)
S2 = jax.tree.map(jnp.float64, NUC_PARAMS_S2)


def scenario_grids(name, grids):
    """Requested grids that the scenario is run on (sai_plume: 40 bins only)."""
    return sorted(g for g in grids if g in SCENARIOS[name]['grids'])


def eval_hours(hours):
    """FD_HOURS within the run (the last hour for short smoke runs)."""
    return [h for h in FD_HOURS if h <= hours] or [hours]


def shifted(delta):
    """P0 + delta (delta: (nfields,) array)."""
    return nuc.NucleationParams(*(getattr(P0, f) + delta[i] for i, f in enumerate(FIELDS)))


def make_sim(name, xk, hours, dt, coag_solver='euler_fixed', dunne_form='double_exp',
             **overrides):
    rk = runner_kwargs(name, xk)
    rk.update(overrides)
    sim = build_sim(xk, rk['cond']['temp'], hours, dt=dt, coag_solver=coag_solver,
                    dunne_form=dunne_form, runner_kw=rk)
    Nk0, Mk0, Gc0 = initial_state(name, xk)
    return lambda p: sim(p, Nk0, Mk0, Gc0)


def check_finite(tag, tree):
    bad = [k for k, v in jax.tree_util.tree_leaves_with_path(tree)
           if not np.all(np.isfinite(np.asarray(v)))]
    if bad:
        raise RuntimeError(f"{tag}: non-finite values in {bad}")


# =========================================================================
# 1. dt ladder
# =========================================================================

def dt_ladder(name, hours, ladder=DT_LADDER, allow_unconverged=False):
    """Coarsest converged dt (and coag solver) for a scenario, 40 bins."""
    xk = GRIDS[40]()
    d_cal = jnp.array(directions(SCENARIOS[name]['cond']['temp'])['H2SO4 cal'])
    rows = []
    for dt in ladder:
        f = make_sim(name, xk, hours, dt)
        t0 = time.time()
        (ys, aux), (dys, _) = jax.jit(lambda a: jax.jvp(
            lambda s: f(shifted(s * d_cal)), (a,), (jnp.float64(1.0),)))(jnp.float64(0.0))
        y, dy = np.array(ys['nmass'][-1]), np.array(dys['nmass'][-1])
        rows.append(dict(dt=dt, n3=float(y[0]), n12=float(y[2]), s3=float(dy[0] / y[0]),
                         clamp=float(np.mean(aux['frac_clamp'])), sec=time.time() - t0))
        print(f"    {name} dt={dt:5.1f}s: N>3 {y[0]:.3e} N>12 {y[2]:.3e} "
              f"dlnN>3 {dy[0] / y[0]:+.3f} clamp {rows[-1]['clamp']:.2f} "
              f"({rows[-1]['sec']:.0f} s)", flush=True)
        if len(rows) >= 2:
            a, b = rows[-2], rows[-1]
            if (abs(a['n3'] / b['n3'] - 1) < TOL_N and abs(a['n12'] / b['n12'] - 1) < TOL_N
                    and abs(a['s3'] - b['s3']) < TOL_S):
                chosen = a['dt']
                break
    else:
        msg = f"{name}: dt ladder not converged down to {ladder[-1]} s: {rows}"
        if not allow_unconverged:
            raise RuntimeError(msg)
        print(f"    WARNING (smoke only): {msg}; using {ladder[-1]} s", flush=True)
        chosen = ladder[-1]

    return dict(dt=chosen, ladder=rows, **coag_check(name, hours, chosen))


def coag_check(name, hours, dt):
    """Fixed vs adaptive coagulation at dt (40 bins, forward)."""
    xk = GRIDS[40]()
    chosen = dt
    f_fix = make_sim(name, xk, hours, chosen)
    f_ad = make_sim(name, xk, hours, chosen, coag_solver='euler_adaptive')
    yf = np.array(jax.jit(f_fix)(P0)[0]['nmass'][-1])
    ya = np.array(jax.jit(f_ad)(P0)[0]['nmass'][-1])
    coag_diff = float(np.max(np.abs(ya[[0, 2]] / yf[[0, 2]] - 1)))
    coag = 'euler_adaptive' if coag_diff > TOL_N else 'euler_fixed'
    print(f"    {name}: dt = {chosen:g} s; adaptive vs fixed coag {100 * coag_diff:.1f}% "
          f"-> {coag}", flush=True)
    return dict(coag_solver=coag, coag_diff=coag_diff)


# =========================================================================
# 2. Jacobian + Table S2
# =========================================================================

def run_jacobian(name, nbins, hours, dt, coag, outdir):
    xk = GRIDS[nbins]()
    f = make_sim(name, xk, hours, dt, coag)
    f_s2 = make_sim(name, xk, hours, dt, coag, dunne_form='single_exp')
    def g(p):
        ys, aux = f(p)
        return ys, (ys, aux)

    t0 = time.time()
    jac, (ys, aux) = jax.block_until_ready(jax.jit(jax.jacfwd(g, has_aux=True))(P0))
    t_jac = time.time() - t0
    ys_s2, aux_s2 = jax.jit(f_s2)(S2)
    T = SCENARIOS[name]['cond']['temp']
    dirs = directions(T)
    out = dict(
        scenario=name, nbins=nbins, hours=hours, dt=dt, coag_solver=coag,
        fields=np.array(FIELDS), dir_names=np.array(list(dirs)),
        directions=np.stack(list(dirs.values())),
        theta=np.array([float(getattr(P0, k)) for k in FIELDS]),
        dndlog=np.array(ys['dndlog']), nmass=np.array(ys['nmass']),
        jac_dndlog=np.stack([np.array(getattr(jac['dndlog'], k)) for k in FIELDS], -1),
        jac_nmass=np.stack([np.array(getattr(jac['nmass'], k)) for k in FIELDS], -1),
        dndlog_s2=np.array(ys_s2['dndlog']), nmass_s2=np.array(ys_s2['nmass']),
        **{f'aux_{k}': np.array(v) for k, v in aux.items()},
        **{f'aux_s2_{k}': np.array(v) for k, v in aux_s2.items()})
    check_finite(f"{name} {nbins}-bin", {k: v for k, v in out.items()
                                          if isinstance(v, np.ndarray) and v.dtype.kind == 'f'})
    np.savez(os.path.join(outdir, f'{name}_{nbins}bin.npz'), **out)
    print(f"    {name} {nbins}-bin: jacfwd {t_jac:.0f} s; N>3/12/53 at {hours} h = "
          + ', '.join(f'{v:.2e}' for v in out['nmass'][-1, [0, 2, 4]]), flush=True)
    return out


# =========================================================================
# 3. FD check
# =========================================================================

def fd_check(name, hours, dt, coag, jac40, outdir):
    """Regression FD vs AD for directions and top-elasticity coefficients.

    The model output carries step-to-step numerical jitter (MNFIX bin
    selection, PPM sub-steps): ~1% in N(>12 nm) at the leading edge of a
    growing mode, larger than the change a small FD step produces. A
    two-point FD then measures the jitter, not the slope. Instead the slope
    is a least-squares fit through FD_STENCIL x step (plus the base run), with
    a standard error from the residuals. AD passes if
    |slope - AD| <= max(FD_TOL |AD|, 3 sigma); entries with 3 sigma > |AD| are
    'noise-limited' (sensitivity below the model's numerical noise).
    """
    xk = GRIDS[40]()
    f = make_sim(name, xk, hours, dt, coag)
    y = jac40['nmass']                                  # (hours, 5)
    J = jac40['jac_nmass']                              # (hours, 5, nfields)
    theta = jac40['theta']
    elast = np.abs(theta * J[-1, 0] / y[-1, 0])
    top = [int(i) for i in np.argsort(elast)[::-1][:4]]

    cases = [('direction', str(nd), FD_ALPHA, d * FD_ALPHA)
             for nd, d in zip(jac40['dir_names'], jac40['directions'])]
    for i in top:
        h = FD_REL_STEP * abs(theta[i]) if theta[i] != 0 else FD_REL_STEP
        cases.append(('coefficient', FIELDS[i], h, np.eye(len(FIELDS))[i] * h))
    deltas = [d * m for _, _, _, d in cases for m in FD_STENCIL]
    batch = jax.tree.map(lambda *xs: jnp.stack(xs), *[shifted(jnp.array(d)) for d in deltas])
    t0 = time.time()
    ys, aux = jax.block_until_ready(vmap_over_devices(f, batch))
    print(f"    {name}: FD batch of {len(deltas)} runs {time.time() - t0:.0f} s", flush=True)
    ns = len(FD_STENCIL)
    nm = np.array(ys['nmass']).reshape(len(cases), ns, hours, -1)
    clamp = np.array(aux['frac_clamp']).reshape(len(cases), ns, hours)
    x = np.concatenate([[0.0], FD_STENCIL])           # in units of one step

    rows = []
    for c, (kind, label, step, d) in enumerate(cases):
        ad = J @ d                                      # change per one step
        for hr in eval_hours(hours):
            for ch, cut in enumerate(NMASS_CUTOFFS_M):
                yy = np.concatenate([[y[hr - 1, ch]], nm[c, :, hr - 1, ch]])
                A = np.vstack([np.ones_like(x), x]).T
                coef, *_ = np.linalg.lstsq(A, yy, rcond=None)
                resid = yy - A @ coef
                sig = float(np.sqrt(resid @ resid / (len(x) - 2)
                                    / np.sum((x - x.mean()) ** 2)))
                a_, f_ = float(ad[hr - 1, ch]), float(coef[1])
                tol = max(FD_TOL * abs(a_), 3.0 * sig)
                rows.append(dict(kind=kind, label=label, step=float(step), hour=hr,
                                 channel=f'>{cut * 1e9:.0f} nm', ad=a_, fd=f_, sigma=sig,
                                 rel=abs(f_ - a_) / max(abs(a_), 1e-300),
                                 ok=bool(abs(f_ - a_) <= tol),
                                 within_5pct=bool(abs(f_ - a_) <= max(FD_TOL_SABRE * abs(a_),
                                                                      3.0 * sig)),
                                 noise_limited=bool(3.0 * sig > abs(a_)),
                                 rel_change=float(abs(a_) / y[hr - 1, ch] / step
                                                  if kind == 'direction' else 0.0),
                                 clamp=float(clamp[c, :, hr - 1].max())))
    with open(os.path.join(outdir, f'{name}_fd_check.json'), 'w') as fh:
        json.dump(dict(top_coefficients=[FIELDS[i] for i in top], stencil=list(FD_STENCIL),
                       rows=rows), fh, indent=1)
    return rows, [FIELDS[i] for i in top]


def fd_summary(rows):
    """Per (kind, label): number checked, failed, and noise-limited entries."""
    out = {}
    for r in rows:
        s = out.setdefault((r['kind'], r['label']), dict(n=0, fail=[], noise=0, w5=0))
        s['n'] += 1
        s['w5'] += r['within_5pct']
        s['noise'] += r['noise_limited']
        if not r['ok']:
            s['fail'].append(r)
    return out


# =========================================================================
# Plots
# =========================================================================

def load(outdir, name, nbins):
    fname = os.path.join(outdir, f'{name}_{nbins}bin.npz')
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing NPZ: {fname}. Run simulations first.")
    return add_nuc_mode_channel(np.load(fname))


def plot(outdir, names, grids):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    det = np.log(1.0 + SABRE_NUMBER_REL_ERR)
    gmax = lambda name: max(scenario_grids(name, grids))
    nm_lab = [f'>{c * 1e9:.0f} nm' for c in NMASS_CUTOFFS_M] + [NUC_MODE_LABEL]
    chans = ((0, 'C0'), (NUC_MODE_CH, 'C3'), (2, 'C1'), (4, 'C2'))

    # --- A: evolution, NMASS with linear envelopes, OH/H2SO4 -------------
    fig, axes = plt.subplots(len(names), 4, figsize=(20, 3.6 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        d = load(outdir, name, gmax(name))
        hrs = np.arange(1, int(d['hours']) + 1)
        ax = axes[r, 0]
        im = ax.pcolormesh(hrs, SABRE_MID_M * 1e9, np.maximum(d['dndlog'].T, 1e-2),
                           norm=LogNorm(1e-1, None), shading='nearest', cmap='viridis')
        ax.set_yscale('log')
        ax.set_ylabel('D$_p$ [nm] (dry)')
        ax.set_title(f'{name}: dN/dlogD$_p$ [cm$^{{-3}}$], {gmax(name)} bins', fontsize=9)
        fig.colorbar(im, ax=ax, shrink=0.8)
        ax = axes[r, 1]
        for ch, col in chans:
            y = d['nmass'][:, ch]
            ax.semilogy(hrs, y, color=col, label=f'N({nm_lab[ch]})')
            ax.fill_between(hrs, y * (1 - SABRE_NUMBER_REL_ERR),
                            y * (1 + SABRE_NUMBER_REL_ERR), color=col, alpha=0.15)
            for k, nd in enumerate(d['dir_names']):
                if nd not in ('H2SO4 cal', 'J_all x2.5'):
                    continue
                dy = d['jac_nmass'][:, ch] @ d['directions'][k]
                ls = '--' if nd == 'H2SO4 cal' else ':'
                ax.semilogy(hrs, y * np.exp(dy / y), ls, color=col, lw=0.8)
                ax.semilogy(hrs, y * np.exp(-dy / y), ls, color=col, lw=0.8)
        ax.set_title('NMASS (band: SABRE ±8%; -- H$_2$SO$_4$ cal ±1, : J ×2.5 ±1)',
                     fontsize=9)
        ax.legend(fontsize=7)
        # d ln N / d ln J_all as in the last row of the Phase 2 map (1 = N ∝ J).
        # The shaded band is what SABRE's 8% could not see for a x2.5 change in J.
        ax = axes[r, 2]
        ej = dlnY_dlnJall(d)
        for ch, col in chans:
            ax.plot(hrs, ej[:, ch], color=col, label=f'N({nm_lab[ch]})')
        ax.axhspan(-det / np.log(2.5), det / np.log(2.5), color='0.9')
        ax.axhline(1.0, color='k', ls=':', lw=0.8)
        ax.axhline(0.0, color='k', lw=0.5)
        ax.set_title('d ln N / d ln J$_{all}$ (band: below SABRE ±8% for J ×2.5)',
                     fontsize=9)
        ax.legend(fontsize=7)
        ax = axes[r, 3]
        ax.semilogy(hrs, d['aux_h2so4_mean'], label='H$_2$SO$_4$')
        ax.semilogy(hrs, d['aux_so2_mean'], label='SO$_2$')
        ax.set_title(f'hourly mean gas [cm$^{{-3}}$] (dt {float(d["dt"]):g} s, '
                     f'{d["coag_solver"]})', fontsize=9)
        ax.legend(fontsize=7)
        for a in axes[r]:
            a.set_xlabel('hour')
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'phase3_evolution.png'), dpi=130)
    plt.close(fig)

    # --- B: sensitivity spectra along the published directions ------------
    dnames = list(load(outdir, names[0], gmax(names[0]))['dir_names'])
    fig, axes = plt.subplots(len(names), len(dnames), figsize=(4 * len(dnames), 3.2 * len(names)),
                             squeeze=False)
    for r, name in enumerate(names):
        for g, ls in zip(scenario_grids(name, grids)[::-1], ('-', '--')):
            d = load(outdir, name, g)
            for c, nd in enumerate(dnames):
                ax = axes[r, c]
                k = list(d['dir_names']).index(nd)
                for hr, col in zip(eval_hours(int(d['hours'])), ('C0', 'C1', 'C2')):
                    y = d['dndlog'][hr - 1]
                    dy = d['jac_dndlog'][hr - 1] @ d['directions'][k]
                    v = np.where(y > 1e-2, dy / np.where(y > 0, y, 1), np.nan)
                    ax.semilogx(SABRE_MID_M * 1e9, v, ls, color=col,
                                label=f'{hr} h, {g} bins' if c == 0 else None)
                ax.axhspan(-det, det, color='0.9')
                ax.set_title(f'{name}: {nd}', fontsize=9)
                ax.set_xlabel('D$_p$ [nm]')
        axes[r, 0].set_ylabel(r'$\delta$ ln dN/dlogD$_p$')
        axes[r, 0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'phase3_spectra.png'), dpi=130)
    plt.close(fig)

    # --- C: response to each published direction at the final hour -------
    # Same units for every bar: the change for the published uncertainty.
    # Dunne gives no per-coefficient sigma, so coefficient elasticities
    # (1% changes) are not comparable with each other or with J.
    fig, axes = plt.subplots(len(names), 1, figsize=(8, 3 * len(names)), squeeze=False)
    for r, name in enumerate(names):
        d = load(outdir, name, gmax(name))
        x = np.arange(len(dnames))
        for j, (ch, col) in enumerate(chans):
            dy = np.array([d['jac_nmass'][-1, ch] @ d['directions'][list(d['dir_names']).index(nd)]
                           for nd in dnames]) / d['nmass'][-1, ch]
            axes[r, 0].bar(x + (j - 1.5) * 0.2, dy, 0.2, color=col,
                           label=f'N({nm_lab[ch]})')
        axes[r, 0].axhspan(-det, det, color='0.9', zorder=0)
        axes[r, 0].axhline(0.0, color='k', lw=0.5)
        axes[r, 0].set_xticks(x, dnames, fontsize=8)
        axes[r, 0].set_ylabel(r'$\delta$ ln N (published size)')
        axes[r, 0].set_title(f'{name}: response after {int(d["hours"])} h, {gmax(name)} bins '
                             f'(band: SABRE ±{100 * SABRE_NUMBER_REL_ERR:.0f}%)', fontsize=9)
        axes[r, 0].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'phase3_directions.png'), dpi=130)
    plt.close(fig)

    # --- D: FD vs AD -------------------------------------------------------
    fig, axes = plt.subplots(1, len(names), figsize=(4.5 * len(names), 4), squeeze=False)
    for c, name in enumerate(names):
        fname = os.path.join(outdir, f'{name}_fd_check.json')
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing FD check: {fname}. Run simulations first.")
        rows = json.load(open(fname))['rows']
        ax = axes[0, c]
        for kind, m in (('direction', 'o'), ('coefficient', 's')):
            for ok, col in ((True, 'C0'), (False, 'C3')):
                rr = [r for r in rows if r['kind'] == kind and r['ok'] == ok]
                ax.errorbar([r['ad'] / max(abs(r['ad']), 1e-300) for r in rr],
                            [(r['fd'] - r['ad']) / max(abs(r['ad']), 1e-300) for r in rr],
                            yerr=[3 * r['sigma'] / max(abs(r['ad']), 1e-300) for r in rr],
                            fmt=m, ms=3, color=col, alpha=0.6, lw=0.6,
                            label=f"{kind} {'agree' if ok else 'disagree'}")
        ax.axhline(0.0, color='k', lw=0.8)
        ax.set_yscale('symlog', linthresh=FD_TOL)
        ax.set_xlabel('sign of AD change')
        ax.set_ylabel('(FD slope - AD) / |AD|  (bars: 3$\\sigma$)')
        ax.set_title(f'{name}: regression FD vs AD (all hours, channels)', fontsize=9)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(outdir, 'phase3_fd_check.png'), dpi=130)
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
        names, grids, hours = ['sai_plume'], [40], 2
        outdir = os.path.join(RESULTS, 'phase3_smoke')
    else:
        names, grids, hours = args.scenarios, args.grids, HOURS
        outdir = os.path.join(RESULTS, 'phase3')
    ladder = DT_LADDER
    os.makedirs(outdir, exist_ok=True)

    if not args.plot_only:
        for name in names:
            print(f"  {name}:", flush=True)
            sc = SCENARIOS[name]
            if sc['dt'] is not None and not args.smoke:
                sel = dict(dt=sc['dt'], note=sc['dt_note'], **coag_check(name, hours, sc['dt']))
                print(f"    {name}: dt fixed at {sc['dt']:g} s ({sc['dt_note']})", flush=True)
            else:
                sel = dt_ladder(name, hours, ladder, allow_unconverged=args.smoke)
            with open(os.path.join(outdir, f'{name}_dt.json'), 'w') as fh:
                json.dump(sel, fh, indent=1)
            jac = {g: run_jacobian(name, g, hours, sel['dt'], sel['coag_solver'], outdir)
                   for g in scenario_grids(name, grids)}
            rows, top = fd_check(name, hours, sel['dt'], sel['coag_solver'], jac[40], outdir)
            for (kind, label), sm in fd_summary(rows).items():
                worst = max(sm['fail'], key=lambda r: r['rel'], default=None)
                print(f"    FD {kind:11s} {label:10s}: {sm['n'] - len(sm['fail'])}/{sm['n']} "
                      f"agree ({sm['w5']} within 5%), {sm['noise']} noise-limited"
                      + (f"; worst fail {worst['channel']} {worst['hour']} h AD {worst['ad']:.3e} "
                         f"FD {worst['fd']:.3e}±{worst['sigma']:.1e}" if worst else ''),
                      flush=True)
    plot(outdir, names, grids)


if __name__ == '__main__':
    main()
