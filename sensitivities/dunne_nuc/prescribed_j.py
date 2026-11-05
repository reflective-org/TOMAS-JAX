"""Response of the size distribution to a prescribed nucleation rate J.

The nucleation scheme is bypassed: clusters are inserted at a constant J
(common.make_prescribed_j_runner), and SO2 chemistry, coagulation and
condensation run as in Phase 2 (same conditions, dt_for(SO2), 24 h).

J is centred on the scheme: at each T and SO2, J = J_ref x 10^(-1..+1), where
J_ref is the 24 h mean Dunne J from the Phase 2 map (run_map --resolved-dt)
on the same grid. A constant J_ref nucleates the same number over 24 h as the
scheme. Far below J_ref (high H2SO4, few new particles) is not a state the
scheme produces, and there the PPM condensation tangent is unstable.

It records, hourly, SABRE dN/dlogDp and NMASS (plus N(3-10.65 nm)), their
derivative w.r.t. J from one forward-mode tangent, and a central finite
difference at J(1 +- FD_REL) as a check on it (AD is blind to the sub-step
count and gas clamp).

Usage::

    python -m sensitivities.dunne_nuc.prescribed_j                 # 217 K, 40 bins
    python -m sensitivities.dunne_nuc.prescribed_j --temps 208 217 223
    python -m sensitivities.dunne_nuc.prescribed_j --plot-only
"""
import os
import time
import argparse

import numpy as np
import jax
import jax.numpy as jnp

from .common import (STRAT, BOXVOL, OH_CONC, lognormal_state,
                     make_prescribed_j_runner, ppbv_to_molec)
from .run_map import GRIDS, SO2_PPBV, HOURS, dt_for, conditions_note, npz_path as map_npz
from .sabre_bins import (rebin_to_sabre, nmass_cumulative, NMASS_CUTOFFS_M,
                         NUC_MODE_CH, NUC_MODE_LABEL, add_nuc_mode_channel)

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
OUTDIR = os.path.join(RESULTS, 'prescribed_j')
J_FACTORS = 10.0 ** np.linspace(-1.0, 1.0, 9)    # J / J_ref
FD_REL = 0.02                                     # central FD step in J (relative)
TEMPS = [217.0]


def npz_path(nbins, temp):
    return os.path.join(OUTDIR, f'pj_{nbins}bin_T{temp:.0f}.npz')


def j_reference(nbins, temp, so2_ppbv):
    """24 h mean Dunne J [cm-3 s-1] per SO2 level, from the Phase 2 map."""
    fname = map_npz(os.path.join(RESULTS, 'phase2_dt'), nbins, temp)
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing Phase 2 NPZ: {fname} (J reference). "
                                f"Run run_map --resolved-dt --grids {nbins}.")
    m = np.load(fname)
    jm = dict(zip(m['so2_ppbv'].tolist(), m['aux_J_mean'].mean(1)))
    missing = [c for c in so2_ppbv if c not in jm]
    if missing:
        raise ValueError(f"{fname}: no SO2 level {missing} ppbv")
    return np.array([jm[c] for c in so2_ppbv])


def build_sim(xk, temp, hours, dt):
    """sim(J, Gc0) -> (hourly outputs, hourly diagnostics), Phase 2 initial aerosol."""
    run_hour = make_prescribed_j_runner(xk, nsteps=round(3600.0 / dt), dt=dt,
                                        cond=dict(STRAT, temp=temp))
    Nk0, Mk0, _ = lognormal_state(xk)

    def sim(J, Gc0):
        def hour(carry, _):
            Nk, Mk, Gc, d = run_hour(J, *carry)
            y = dict(dndlog=rebin_to_sabre(Nk, Mk, xk, BOXVOL),
                     nmass=nmass_cumulative(Nk, Mk, xk, BOXVOL))
            aux = dict(h2so4_mean=jnp.mean(d['h2so4']),
                       J_scheme_mean=jnp.mean(d['J']),
                       frac_sub=jnp.mean(d['n_sub'] > 1),
                       max_sub=jnp.max(d['n_sub']),
                       frac_clamp=jnp.mean(d['clamp']))
            return (Nk, Mk, Gc), (y, aux)

        _, (ys, aux) = jax.lax.scan(hour, (Nk0, Mk0, Gc0), None, length=hours)
        return ys, aux

    return sim


def run_temp(nbins, temp, hours=HOURS, so2_ppbv=SO2_PPBV, j_factors=J_FACTORS):
    xk = GRIDS[nbins]()
    j_ref = j_reference(nbins, temp, so2_ppbv)
    J = j_ref[:, None] * j_factors[None, :]          # (nso2, nf)
    gcs = np.stack([np.array(lognormal_state(xk, so2=ppbv_to_molec(c, temp, STRAT['pres']))[2])
                    for c in so2_ppbv])
    dts = np.array([dt_for(c) for c in so2_ppbv])
    nf = len(j_factors)
    out, timing = {}, []
    # One compiled program per dt group, vmapped over (SO2 in the group) x J
    for dt in sorted(set(dts.tolist()), reverse=True):
        idx = np.where(dts == dt)[0]
        sim = build_sim(xk, temp, hours, dt)
        gc_b = jnp.array(np.repeat(gcs[idx], nf, axis=0))
        J_b = jnp.array(J[idx].ravel())

        def cell(J, gc):
            # d/dJ: one tangent, plus a central FD at J(1 +- FD_REL) to check it
            (ys, aux), (dys, _) = jax.jvp(lambda j: sim(j, gc), (J,), (jnp.ones_like(J),))
            up, _ = sim(J * (1.0 + FD_REL), gc)
            dn, _ = sim(J * (1.0 - FD_REL), gc)
            fd = jax.tree.map(lambda a, b: (a - b) / (2.0 * FD_REL * J), up, dn)
            return ys, aux, dys, fd

        t0 = time.time()
        ys, aux, dys, fd = jax.block_until_ready(jax.jit(jax.vmap(cell))(J_b, gc_b))
        timing.append(f"dt={dt:g}s x{len(idx) * nf}: {time.time() - t0:.0f} s")
        shape = lambda a: np.array(a).reshape(len(idx), nf, *np.shape(a)[1:])
        part = dict(dndlog=shape(ys['dndlog']), nmass=shape(ys['nmass']),
                    dJ_dndlog=shape(dys['dndlog']), dJ_nmass=shape(dys['nmass']),
                    fdJ_dndlog=shape(fd['dndlog']), fdJ_nmass=shape(fd['nmass']),
                    **{f'aux_{k}': shape(v) for k, v in aux.items()})
        for k, v in part.items():
            out.setdefault(k, np.zeros((len(so2_ppbv),) + v.shape[1:], v.dtype))[idx] = v

    bad = [k for k, v in out.items() if v.dtype.kind == 'f' and not np.all(np.isfinite(v))]
    if bad:
        raise RuntimeError(f"{nbins}-bin T={temp} K: non-finite {bad}")
    out.update(nbins=nbins, temp=temp, hours=hours, so2_ppbv=np.array(so2_ppbv),
               j_ref=j_ref, j_factors=np.array(j_factors), J=J, fd_rel=FD_REL,
               dt_s=dts, oh_conc=OH_CONC, coag_solver='euler_fixed')
    os.makedirs(OUTDIR, exist_ok=True)
    np.savez(npz_path(nbins, temp), **out)
    print(f"  {nbins}-bin T={temp:5.1f} K: {'; '.join(timing)}; "
          f"cells clamped >1% of steps: {int(np.sum(out['aux_frac_clamp'].mean(-1) > 0.01))}"
          f"/{out['aux_frac_clamp'][..., 0].size}", flush=True)


def load(nbins, temp):
    fname = npz_path(nbins, temp)
    if not os.path.exists(fname):
        raise FileNotFoundError(f"Missing NPZ: {fname}. Run simulations first.")
    d = dict(np.load(fname))
    # Append N(3-10.65 nm) to the values and to both derivatives. The helper
    # expects Jacobians with a parameter axis last, so add (and drop) one.
    fd = add_nuc_mode_channel(dict(dndlog=d['dndlog'], nmass=d['nmass'],
                                   jac_dndlog=d['fdJ_dndlog'][..., None],
                                   jac_nmass=d['fdJ_nmass'][..., None]))
    d['fdJ_nmass'] = fd['jac_nmass'][..., 0]
    for k in ('dndlog', 'nmass'):
        d[f'jac_{k}'] = d.pop(f'dJ_{k}')[..., None]
    d = add_nuc_mode_channel(d)
    d['dJ_nmass'] = d.pop('jac_nmass')[..., 0]
    d.pop('jac_dndlog')
    return d


def elasticity(d, key, ch, hour=-1, floor=1e-3):
    """d ln N / d ln J at one hour, (nso2, nf); NaN where N < floor [cm-3]."""
    N = d['nmass'][:, :, hour, ch]
    return np.where(N > floor, d[key][:, :, hour, ch] * d['J'] / np.where(N > 0, N, 1), np.nan)


def plot(nbins, temps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm

    nm = [f'>{c * 1e9:.0f} nm' for c in NMASS_CUTOFFS_M] + [NUC_MODE_LABEL]
    chans = [NUC_MODE_CH, 0, 2]
    for T in temps:
        d = load(nbins, T)
        so2, f = d['so2_ppbv'], d['j_factors']
        xt, yt = np.arange(len(so2)), np.arange(len(f))
        clamped = d['aux_frac_clamp'].mean(-1) > 0.01

        fig, axes = plt.subplots(len(chans), 3, figsize=(15, 3.6 * len(chans)))
        for r, ch in enumerate(chans):
            ad, fd = elasticity(d, 'dJ_nmass', ch), elasticity(d, 'fdJ_nmass', ch)
            # AD and FD disagree: more than 0.05 + 10% apart
            off = np.abs(np.nan_to_num(ad - fd)) > 0.05 + 0.1 * np.abs(np.nan_to_num(fd))
            panels = [(f'N({nm[ch]}) [cm$^{{-3}}$]',
                       np.maximum(d['nmass'][:, :, -1, ch], 1e-3), dict(norm=LogNorm()), 'viridis'),
                      (f'd ln N({nm[ch]}) / d ln J, AD', ad, dict(vmin=-1.2, vmax=1.2), 'RdBu_r'),
                      (f'd ln N({nm[ch]}) / d ln J, FD ±{100 * FD_REL:.0f}%', fd,
                       dict(vmin=-1.2, vmax=1.2), 'RdBu_r')]
            for c, (title, z, kw, cmap) in enumerate(panels):
                ax = axes[r, c]
                im = ax.pcolormesh(xt, yt, z.T, shading='nearest', cmap=cmap, **kw)
                fig.colorbar(im, ax=ax, shrink=0.8)
                ii, jj = np.where(clamped)
                ax.plot(xt[ii], yt[jj], 'x', color='w', ms=4, mew=0.8)
                if c > 0:
                    ii, jj = np.where(off)
                    ax.plot(xt[ii], yt[jj], 'k+', ms=6, mew=0.8)
                ax.axhline(len(f) // 2, color='k', lw=0.6, ls='--')
                ax.set_xticks(xt, [f'{c:g}' for c in so2], rotation=60, fontsize=7)
                ax.set_yticks(yt, [f'{v:.2g}' for v in f], fontsize=7)
                ax.set_title(title, fontsize=8)
                if c == 0:
                    ax.set_ylabel('J / J$_{ref}$')
                if r == len(chans) - 1:
                    ax.set_xlabel('SO$_2$ [ppbv]')
        fig.suptitle(f'Prescribed J, {int(d["hours"])} h, {nbins} bins, {T:.0f} K. '
                     'J$_{ref}$ (dashed): Dunne 24 h mean J at this T and SO$_2$ (Phase 2). '
                     'x: gas clamp on >1% of steps; +: AD and FD differ', fontsize=10)
        fig.text(0.01, 0.003, conditions_note(dict(dt_s=d['dt_s'], oh_conc=d['oh_conc'],
                                                   coag_solver=d['coag_solver'])),
                 fontsize=7, va='bottom', wrap=True)
        fig.tight_layout(rect=(0, 0.04, 1, 1))
        out = os.path.join(OUTDIR, f'pj_{nbins}bin_T{T:.0f}.png')
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--grids', type=int, nargs='+', default=[40])
    parser.add_argument('--temps', type=float, nargs='+', default=TEMPS)
    parser.add_argument('--smoke', action='store_true',
                        help='2 h, SO2 0.1/10 ppbv, J/J_ref 0.1/10; no plots')
    args = parser.parse_args()
    if args.smoke:
        global OUTDIR
        OUTDIR = os.path.join(RESULTS, 'prescribed_j_smoke')
        for nbins in args.grids:
            for T in args.temps:
                run_temp(nbins, T, hours=2, so2_ppbv=[0.1, 10.0],
                         j_factors=np.array([0.1, 10.0]))
        return
    for nbins in args.grids:
        if not args.plot_only:
            for T in args.temps:
                run_temp(nbins, T)
        plot(nbins, args.temps)


if __name__ == '__main__':
    main()
