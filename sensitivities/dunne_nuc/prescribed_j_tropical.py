"""Prescribed-J sensitivity on the AFGL tropical altitudes with adaptive gas sub-steps.

The prescribed_j study (constant J inserted in place of the scheme, d N / d J
by one forward-mode tangent and a central FD check) redone on the
gas_substep_psd80 grid: altitude 13-25 km (T and P together,
common.TROPICAL_AFGL) x SO2, 24 h, make_step(gas_substep='adaptive',
f = 0.1) with fn_prescribed = J, so the sub-step criterion sees the prescribed
J's own H2SO4 loss. docs/prescribed_j.md.

J_ref is the 24 h mean of the scheme's Dunne J from an adaptive-sub-step scheme
run in the same cell (60 s end-of-step values, as in the Phase 2 map); J =
J_ref x 10^(-1..1), 9 factors.

Unit of work = one file, so a run can be stopped and resumed, and the cells
run in separate processes (adaptive while-loops under vmap run at the pace of
the slowest lane, so cells and J factors are not batched):

    jref_z<km>_<so2>.npz            scheme run -> J_ref
    pj_z<km>_<so2>_j<i>.npz         one J factor: values, AD tangent, FD +-FD_REL

Usage (python -m sensitivities.dunne_nuc.prescribed_j_tropical ...)::

    --smoke                          2 h, 15 km, SO2 10 ppbv, J factors 0 and 8
    --all  [--bins 40] [--so2 ...] [--alts ...]   every missing file, NJOBS at a time
    --run Z SO2 JIDX | --jref Z SO2  one file
    --plot [--bins 40]               figures for every SO2 level with all files (errors if < 2)

Default SO2 is every level up to 10 ppmv; 100 ppmv is run with --so2 1e5
(hours; see run_prescribed_j_overnight.sh).
"""
import os
import sys
import time
import argparse
import subprocess
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.core.config import make_grid_80bin
from . import common as c
from .common import (STRAT, BOXVOL, OH_CONC, lognormal_state,
                     make_prescribed_j_runner, make_runner, ppbv_to_molec)
from .run_map import GRIDS, SO2_PPBV, HOURS
from .sabre_bins import rebin_to_sabre, nmass_cumulative

RESULTS = os.path.join(os.path.dirname(__file__), 'results', 'prescribed_j_tropical')
J_FACTORS = 10.0 ** np.linspace(-1.0, 1.0, 9)    # J / J_ref
FD_REL = 0.02                                     # central FD step in J (relative)
FRAC = 0.1                                        # gas_substep_frac
DT = 60.0                                         # outer step [s]
# Floor of the gas sub-step [s]: a J above the H2SO4 supply clamps on every
# sub-step whatever its length (common.make_prescribed_j_runner)
MIN_DT = 0.1
SO2_DEFAULT = [s for s in SO2_PPBV if s <= 1e4]
ALTITUDES = c.TROPICAL_AFGL                       # km -> (T [K], P [Pa])


def outdir(nbins, smoke=False):
    return os.path.join(RESULTS + ('_smoke' if smoke else ''), f'{nbins}bin')


def jref_path(nbins, z, so2, smoke=False):
    return os.path.join(outdir(nbins, smoke), f'jref_z{z:g}_{so2:g}.npz')


def pj_path(nbins, z, so2, j, smoke=False):
    return os.path.join(outdir(nbins, smoke), f'pj_z{z:g}_{so2:g}_j{j}.npz')


def grid(nbins):
    return GRIDS[nbins]() if nbins in GRIDS else make_grid_80bin()


def setup(nbins, z, so2):
    xk = grid(nbins)
    T, P = ALTITUDES[z]
    cond = dict(STRAT, temp=T, pres=P)
    state = lognormal_state(xk, so2=ppbv_to_molec(so2, T, P))
    return xk, cond, state


def run_jref(nbins, z, so2, hours=HOURS, smoke=False):
    """24 h mean scheme J [cm-3 s-1] from an adaptive-sub-step scheme run."""
    xk, cond, (Nk0, Mk0, Gc0) = setup(nbins, z, so2)
    run = make_runner(xk, nsteps=round(hours * 3600.0 / DT), cond=cond, dt=DT,
                      gas_substep='adaptive', gas_substep_frac=FRAC)
    t0 = time.time()
    Nk, Mk, Gc, d = jax.block_until_ready(
        jax.jit(run)(c.NUC_PARAMS, Nk0, Mk0, Gc0))
    J = np.array(d['J'])
    if not np.all(np.isfinite(J)) or not np.all(np.isfinite(np.array(Nk))):
        raise RuntimeError(f"jref z={z} SO2={so2}: non-finite")
    os.makedirs(outdir(nbins, smoke), exist_ok=True)
    np.savez(jref_path(nbins, z, so2, smoke), z=z, so2=so2, nbins=nbins,
             hours=hours, j_ref=J.mean(), j_hourly=J.reshape(hours, -1).mean(1),
             N_final=float(jnp.sum(Nk) / BOXVOL), t=time.time() - t0)
    print(f"jref z={z:g} km SO2={so2:g}: J_ref={J.mean():.4g} cm-3 s-1 "
          f"({time.time() - t0:.0f} s)", flush=True)


def build_sim(nbins, z, hours):
    xk, cond, (Nk0, Mk0, _) = setup(nbins, z, 1.0)
    run_hour = make_prescribed_j_runner(
        xk, nsteps=round(3600.0 / DT), dt=DT, cond=cond,
        gas_substep='adaptive', gas_substep_frac=FRAC, gas_min_dt=MIN_DT)

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


def run_j(nbins, z, so2, jidx, hours=HOURS, smoke=False, j_factors=J_FACTORS):
    jp = jref_path(nbins, z, so2, smoke)
    if not os.path.exists(jp):
        raise FileNotFoundError(f"Missing {jp}: run --jref {z:g} {so2:g} first.")
    j_ref = float(np.load(jp)['j_ref'])
    J = j_ref * float(j_factors[jidx])
    _, _, (_, _, Gc0) = setup(nbins, z, so2)
    sim = build_sim(nbins, z, hours)

    @jax.jit
    def cell(J):
        (ys, aux), (dys, _) = jax.jvp(lambda j: sim(j, Gc0), (J,), (jnp.ones_like(J),))
        up, _ = sim(J * (1.0 + FD_REL), Gc0)
        dn, _ = sim(J * (1.0 - FD_REL), Gc0)
        fd = jax.tree.map(lambda a, b: (a - b) / (2.0 * FD_REL * J), up, dn)
        return ys, aux, dys, fd

    t0 = time.time()
    ys, aux, dys, fd = jax.block_until_ready(cell(jnp.float64(J)))
    out = dict(dndlog=ys['dndlog'], nmass=ys['nmass'], dJ_dndlog=dys['dndlog'],
               dJ_nmass=dys['nmass'], fdJ_dndlog=fd['dndlog'],
               fdJ_nmass=fd['nmass'], **{f'aux_{k}': v for k, v in aux.items()})
    out = {k: np.array(v) for k, v in out.items()}
    bad = [k for k, v in out.items() if v.dtype.kind == 'f' and not np.all(np.isfinite(v))]
    if bad:
        raise RuntimeError(f"z={z} SO2={so2} j{jidx}: non-finite {bad}")
    np.savez(pj_path(nbins, z, so2, jidx, smoke), nbins=nbins, z=z,
             temp=ALTITUDES[z][0], pres=ALTITUDES[z][1], so2=so2, jidx=jidx,
             j_factor=j_factors[jidx], J=J, j_ref=j_ref, hours=hours,
             fd_rel=FD_REL, frac=FRAC, min_dt=MIN_DT, t=time.time() - t0, **out)
    print(f"z={z:g} km SO2={so2:g} j{jidx} (J/J_ref={j_factors[jidx]:.3g}): "
          f"{time.time() - t0:.0f} s, max sub-steps {int(out['aux_max_sub'].max())}, "
          f"clamped {float(out['aux_frac_clamp'].mean()):.3g}", flush=True)


def jobs_for(nbins, alts, so2s, smoke, nf):
    """Missing (kind, z, so2, j) files, heaviest SO2 first, jref before its J runs."""
    todo = []
    for so2 in sorted(so2s, reverse=True):
        for z in alts:
            if not os.path.exists(jref_path(nbins, z, so2, smoke)):
                todo.append(('jref', z, so2, 0))
    for so2 in sorted(so2s, reverse=True):
        for z in alts:
            for j in range(nf):
                if not os.path.exists(pj_path(nbins, z, so2, j, smoke)):
                    todo.append(('pj', z, so2, j))
    return todo


def run_all(nbins, alts, so2s, smoke=False, nf=len(J_FACTORS)):
    os.makedirs(outdir(nbins, smoke), exist_ok=True)
    njobs = int(os.environ.get('NJOBS', 6))

    def launch(kind, z, so2, j):
        base = [sys.executable, '-m', 'sensitivities.dunne_nuc.prescribed_j_tropical',
                '--bins', str(nbins)] + (['--smoke-dir'] if smoke else [])
        cmd = base + (['--jref', f'{z:g}', f'{so2:g}'] if kind == 'jref'
                      else ['--run', f'{z:g}', f'{so2:g}', str(j)])
        log = (jref_path(nbins, z, so2, smoke) if kind == 'jref'
               else pj_path(nbins, z, so2, j, smoke)).replace('.npz', '.log')
        with open(log, 'w') as f:
            rc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT).returncode
        print(f"{kind} z={z:g} SO2={so2:g} j{j}: {'done' if rc == 0 else 'FAILED'}",
              flush=True)
        return rc

    # Stage 1: J_ref (the J runs read it), stage 2: the J runs. Each stage
    # waits for its last job; stage 2's heaviest jobs start first.
    for stage in ('jref', 'pj'):
        jobs = [j for j in jobs_for(nbins, alts, so2s, smoke, nf) if j[0] == stage]
        print(f"{stage}: {len(jobs)} files to do, {njobs} at a time", flush=True)
        with ThreadPoolExecutor(njobs) as ex:
            rcs = list(ex.map(lambda j: launch(*j), jobs))
        failed = [j for j, rc in zip(jobs, rcs) if rc]
        if failed:
            raise RuntimeError(f"{len(failed)} {stage} runs failed (see the .log "
                               f"files in {outdir(nbins, smoke)}): {failed}")


def assemble(nbins, z, so2s, nf=len(J_FACTORS), smoke=False):
    """Arrays (nso2, nf, ...) from the per-J files; errors on any missing file."""
    out = {}
    for i, so2 in enumerate(so2s):
        for j in range(nf):
            p = pj_path(nbins, z, so2, j, smoke)
            if not os.path.exists(p):
                raise FileNotFoundError(f"Missing {p}. Run --all first.")
            f = np.load(p)
            for k in f.files:
                v = f[k]
                if v.dtype.kind == 'f' or k in ('max_sub',):
                    out.setdefault(k, np.zeros((len(so2s), nf) + v.shape, v.dtype))[i, j] = v
    out['so2_ppbv'] = np.array(so2s)
    out['j_factors'] = J_FACTORS[:nf]
    return out


def complete_levels(nbins, alts, nf=len(J_FACTORS)):
    """SO2 levels whose 9 J files exist at every altitude (what --plot draws)."""
    return [so2 for so2 in SO2_PPBV
            if all(os.path.exists(pj_path(nbins, z, so2, j))
                   for z in alts for j in range(nf))]


def plot(nbins, alts, so2s, smoke=False):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm
    from .prescribed_j import elasticity
    from .sabre_bins import (NMASS_CUTOFFS_M, NUC_MODE_CH, NUC_MODE_LABEL,
                             add_nuc_mode_channel)

    nm = [f'>{x * 1e9:.0f} nm' for x in NMASS_CUTOFFS_M] + [NUC_MODE_LABEL]
    chans = [NUC_MODE_CH, 0, 2]
    summary = {}
    for z in alts:
        d = assemble(nbins, z, so2s, smoke=smoke)
        fd = add_nuc_mode_channel(dict(
            dndlog=d['dndlog'], nmass=d['nmass'],
            jac_dndlog=d['fdJ_dndlog'][..., None], jac_nmass=d['fdJ_nmass'][..., None]))
        ad = add_nuc_mode_channel(dict(
            dndlog=d['dndlog'], nmass=d['nmass'],
            jac_dndlog=d['dJ_dndlog'][..., None], jac_nmass=d['dJ_nmass'][..., None]))
        d['nmass'] = ad['nmass']
        d['dJ_nmass'] = ad['jac_nmass'][..., 0]
        d['fdJ_nmass'] = fd['jac_nmass'][..., 0]
        d['J'] = d['J'].reshape(len(so2s), -1)
        xt, yt = np.arange(len(so2s)), np.arange(d['J'].shape[1])
        clamped = d['aux_frac_clamp'].mean(-1) > 0.01
        fig, axes = plt.subplots(len(chans), 3, figsize=(15, 3.6 * len(chans)))
        for r, ch in enumerate(chans):
            a, f = elasticity(d, 'dJ_nmass', ch), elasticity(d, 'fdJ_nmass', ch)
            summary[(z, ch)] = (a, f)
            off = np.abs(np.nan_to_num(a - f)) > 0.05 + 0.1 * np.abs(np.nan_to_num(f))
            panels = [(f'N({nm[ch]}) [cm$^{{-3}}$]',
                       np.maximum(d['nmass'][:, :, -1, ch], 1e-3), dict(norm=LogNorm()), 'viridis'),
                      (f'd ln N({nm[ch]}) / d ln J, AD', a, dict(vmin=-1.2, vmax=1.2), 'RdBu_r'),
                      (f'd ln N({nm[ch]}) / d ln J, FD ±{100 * FD_REL:.0f}%', f,
                       dict(vmin=-1.2, vmax=1.2), 'RdBu_r')]
            for cc, (title, zz, kw, cmap) in enumerate(panels):
                ax = axes[r, cc]
                im = ax.pcolormesh(xt, yt, zz.T, shading='nearest', cmap=cmap, **kw)
                fig.colorbar(im, ax=ax, shrink=0.8)
                ii, jj = np.where(clamped)
                ax.plot(xt[ii], yt[jj], 'x', color='w', ms=4, mew=0.8)
                if cc > 0:
                    ii, jj = np.where(off)
                    ax.plot(xt[ii], yt[jj], 'k+', ms=6, mew=0.8)
                ax.axhline(len(yt) // 2, color='k', lw=0.6, ls='--')
                ax.set_xticks(xt, [f'{s:g}' for s in so2s], rotation=60, fontsize=7)
                ax.set_yticks(yt, [f'{v:.2g}' for v in d['j_factors']], fontsize=7)
                ax.set_title(title, fontsize=8)
                if cc == 0:
                    ax.set_ylabel('J / J$_{ref}$')
                if r == len(chans) - 1:
                    ax.set_xlabel('SO$_2$ [ppbv]')
        T, P = ALTITUDES[z]
        fig.suptitle(f'Prescribed J, adaptive gas sub-steps (f = {FRAC:g}), 24 h, {nbins} bins, '
                     f'{z:g} km ({T:g} K, {P / 100:.0f} hPa). J$_{{ref}}$ (dashed): scheme 24 h '
                     'mean J. x: gas clamp on >1% of steps; +: AD and FD differ', fontsize=10)
        fig.tight_layout()
        out = os.path.join(outdir(nbins, smoke), f'pj_{nbins}bin_z{z:g}.png')
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print(f"  wrote {out}")

    # d ln N(>3 nm) / d ln J at J = J_ref vs SO2, one line per altitude
    mid = len(J_FACTORS) // 2
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.8), sharey=True)
    for z in alts:
        a, f = summary[(z, 0)]
        ax[0].semilogx(so2s, a[:, mid], 'o-', label=f'{z:g} km')
        ax[1].semilogx(so2s, f[:, mid], 'o-')
    for x, t in zip(ax, ('AD', f'FD ±{100 * FD_REL:.0f}%')):
        x.set_title(f'd ln N(>3 nm) / d ln J at J$_{{ref}}$, {t}', fontsize=9)
        x.set_xlabel('SO$_2$ [ppbv]')
    ax[0].legend(fontsize=7)
    fig.tight_layout()
    out = os.path.join(outdir(nbins, smoke), f'pj_{nbins}bin_elasticity_vs_so2.png')
    fig.savefig(out, dpi=140)
    print(f"  wrote {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('--bins', type=int, default=40, choices=[40, 80])
    p.add_argument('--all', action='store_true')
    p.add_argument('--run', nargs=3, metavar=('ALT_KM', 'SO2_PPBV', 'JIDX'))
    p.add_argument('--jref', nargs=2, metavar=('ALT_KM', 'SO2_PPBV'))
    p.add_argument('--plot', action='store_true')
    p.add_argument('--smoke', action='store_true')
    p.add_argument('--smoke-dir', action='store_true', help=argparse.SUPPRESS)
    p.add_argument('--so2', type=float, nargs='+', default=SO2_DEFAULT)
    p.add_argument('--alts', type=float, nargs='+', default=list(ALTITUDES))
    a = p.parse_args()
    for z in a.alts:
        if z not in ALTITUDES:
            p.error(f'altitude {z} not in {list(ALTITUDES)}')
    if a.smoke:
        # 2 h, one cell, J factors 0.1 and 10 only
        global J_FACTORS
        J_FACTORS = J_FACTORS[[0, -1]]
        run_jref(a.bins, 15.0, 10.0, hours=2, smoke=True)
        for j in range(2):
            run_j(a.bins, 15.0, 10.0, j, hours=2, smoke=True, j_factors=J_FACTORS)
        return
    sm = a.smoke_dir
    if a.jref:
        run_jref(a.bins, float(a.jref[0]), float(a.jref[1]), smoke=sm)
    elif a.run:
        run_j(a.bins, float(a.run[0]), float(a.run[1]), int(a.run[2]), smoke=sm)
    elif a.all:
        run_all(a.bins, a.alts, a.so2)
        levels = complete_levels(a.bins, a.alts)
        if len(levels) > 1:
            plot(a.bins, a.alts, levels)
    elif a.plot:
        levels = complete_levels(a.bins, a.alts)
        if len(levels) < 2:
            raise FileNotFoundError(f"--plot needs two or more complete SO2 levels in "
                                    f"{outdir(a.bins)}; complete: {levels}. Run --all first.")
        plot(a.bins, a.alts, levels)
    else:
        p.error('give --all, --run, --jref, --plot or --smoke')


if __name__ == '__main__':
    main()
