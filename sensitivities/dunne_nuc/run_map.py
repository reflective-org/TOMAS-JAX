"""Phase 2: nucleation-coefficient sensitivity map over T x SO2 (chemistry on).

For each grid (40, 80 bins), temperature and SO2 level, it runs 24 h of SO2
chemistry + nucleation + coagulation + condensation at 55 hPa, starting from
background stratospheric aerosol. It records, hourly:

  - SABRE-binned dry dN/dlogDp and the NMASS cumulative channels
  - jax.jacfwd of both w.r.t. all 25 NucleationParams fields (raw Jacobian)
  - a forward run with the Dunne 2016 Table S2 single-exponential fit
  - regime diagnostics: H2SO4, J, J_bn, J_bi, sub-stepping and clamp fractions

Dunne 2016 publishes no per-coefficient sigma (see
docs/nucleation_sensitivity.md), so the Jacobian is contracted with the
uncertainty directions it does publish (`directions()` below). These are linear
estimates; Phase 4 checks linearity.

Usage::

    python -m sensitivities.dunne_nuc.run_map              # full run + plots
    python -m sensitivities.dunne_nuc.run_map --plot-only
    python -m sensitivities.dunne_nuc.run_map --smoke      # 40 bins, 2 T x 3 SO2 x 2 h
    python -m sensitivities.dunne_nuc.run_map --resolved-dt --grids 40
        # dt per SO2 level (dt_for), Jacobian along directions() only
    python -m sensitivities.dunne_nuc.run_map --gas-substep adaptive --grids 40
        # dt = 60 s with adaptive H2SO4 sub-steps (docs/gas_substep.md)
"""
import os
import time
import argparse

import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.core.config import make_grid, make_grid_80bin, XK0
from tomas_jax.physics import nucleation as nuc
from .common import (STRAT, BOXVOL, OH_CONC, DT, NUC_PARAMS, NUC_PARAMS_S2,
                     FION, BG_N, BG_GMD, BG_GSD, SPARSE_DENSITY,
                     lognormal_state, make_runner, ppbv_to_molec)
from .sabre_bins import (NMASS_CUTOFFS_M, SABRE_NUMBER_REL_ERR,
                         rebin_to_sabre, nmass_cumulative,
                         NUC_MODE_CH, NUC_MODE_LABEL, add_nuc_mode_channel)

RESULTS = os.path.join(os.path.dirname(__file__), 'results')
GRIDS = {40: lambda: make_grid(40, XK0, 2.0), 80: make_grid_80bin}
TEMPS = [190.0, 200.0, 208.0, 217.0, 223.0, 240.0]
# 10 pptv - 100 ppmv. Above 100 ppmv, constant OH makes ~1e11 H2SO4 cm-3 per
# minute, nucleation puts ~3e9 cm-3 in bin 0 and the fixed 10-sub-step Euler
# coagulation goes unstable (NaN). Near-source plumes need OH titration and
# dilution: Phase 3.
SO2_PPBV = [1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5]
HOURS = 24
CLOUD_T_MIN = 208.0   # lowest CLOUD temperature in Dunne 2016

FIELDS = nuc.NucleationParams._fields
MECHS = ('bn', 'tn', 'bi', 'ti')


def h2so4_cal_factor(temp):
    """Dunne SM Sect. 19 H2SO4 calibration factor: 1.5 (+50%/-33%), x1.5 at
    223 K and x2 at 208 K; log-linear in T between, constant outside."""
    w = np.clip((223.0 - temp) / (223.0 - 208.0), 0.0, 1.0)
    return float(np.exp((1.0 - w) * np.log(1.5) + w * np.log(2.0)))


def directions(temp, params=NUC_PARAMS):
    """Published uncertainty directions as {name: delta-theta vector}.

    J x 2.5 (measured nucleation rate): d u_x = ln 2.5, since dlnJ_x/du_x = 1.
    H2SO4 calibration factor f: fitting data with [H2SO4] off by f shifts
    each rate constant by d u_x = p_x ln f (systematic, all mechanisms).
    """
    def vec(**du):
        v = np.zeros(len(FIELDS))
        for k, val in du.items():
            v[FIELDS.index(k)] = val
        return v

    ln25 = np.log(2.5)
    lnf = np.log(h2so4_cal_factor(temp))
    return {
        'J_bn x2.5': vec(u_bn=ln25),
        'J_bi x2.5': vec(u_bi=ln25),
        'J_all x2.5': vec(**{f'u_{m}': ln25 for m in MECHS}),
        'H2SO4 cal': vec(**{f'u_{m}': getattr(params, f'p_{m}') * lnf
                            for m in MECHS}),
    }


def dt_for(so2_ppbv):
    """Time step [s] that resolves H2SO4 at this SO2 level.

    From the 2026-10-29 convergence test (docs/nucleation_sensitivity.md):
    at dt = 60 s, >= 1 ppmv SO2 makes more H2SO4 per step than nucleation
    uses, the gas clamp fires and the coefficient sensitivity drops to 0.
    The 100 ppmv level keeps an ~8% residual in N at 0.5 s (vs 0.25 s).
    """
    if so2_ppbv <= 10.0:
        return 60.0
    if so2_ppbv <= 100.0:
        return 10.0
    if so2_ppbv <= 1e4:
        return 2.0
    return 0.5


def npz_path(outdir, nbins, temp):
    return os.path.join(outdir, f'map_{nbins}bin_T{temp:.0f}.npz')


def build_sim(xk, temp, hours, dunne_form='double_exp', oh_conc=OH_CONC,
              coag_solver='euler_fixed', dt=DT, runner_kw=None,
              gas_substep=None, ppm_tangent='exact'):
    """sim(params, Nk0, Mk0, Gc0) -> (hourly outputs, hourly diagnostics).

    params enters only through nucleation, so sim is a function of the fit
    coefficients that jax.jacfwd can differentiate. runner_kw (e.g. from
    scenarios.runner_kwargs: cond, dilution) overrides the map defaults.
    """
    kw = dict(cond=dict(STRAT, temp=temp), oh_conc=oh_conc,
              dunne_form=dunne_form, coag_solver=coag_solver,
              gas_substep=gas_substep, ppm_tangent=ppm_tangent)
    kw.update(runner_kw or {})
    # One hour = 3600/dt steps (SO2 chem -> nucl -> coag -> cond), as a scan.
    run_hour = make_runner(xk, nsteps=round(3600.0 / dt), dt=dt, **kw)

    def sim(params, Nk0, Mk0, Gc0):
        # Outer scan over hours: carry the aerosol and gas state forward and
        # emit the SABRE-comparable outputs once per hour. Only `y` is
        # differentiated; `aux` holds regime diagnostics (no tangents needed).
        def hour(carry, _):
            Nk, Mk, Gc, d = run_hour(params, *carry)
            # Model bins -> SABRE dry dN/dlogDp (20/decade) and NMASS N(>Dp)
            y = dict(dndlog=rebin_to_sabre(Nk, Mk, xk, BOXVOL),
                     nmass=nmass_cumulative(Nk, Mk, xk, BOXVOL))
            aux = dict(N_total=jnp.sum(Nk) / BOXVOL,
                       h2so4_mean=jnp.mean(d['h2so4']),
                       h2so4_max=jnp.max(d['h2so4']),
                       so2_mean=jnp.mean(d['so2']),
                       J_mean=jnp.mean(d['J']),
                       J_bn_mean=jnp.mean(d['J_bn']),
                       J_bi_mean=jnp.mean(d['J_bi']),
                       # Where these are non-zero the gradient is blind to the
                       # sub-step count / gas clamp (piecewise, not smooth)
                       frac_sub=jnp.mean(d['n_sub'] > 1),
                       max_sub=jnp.max(d['n_sub']),
                       frac_clamp=jnp.mean(d['clamp']))
            return (Nk, Mk, Gc), (y, aux)

        _, (ys, aux) = jax.lax.scan(hour, (Nk0, Mk0, Gc0), None, length=hours)
        return ys, aux

    return sim


def run_cell_set(nbins, temp, so2_ppbv, hours, outdir, oh_conc,
                 coag_solver='euler_fixed', resolved_dt=False,
                 gas_substep=None, ppm_tangent='exact'):
    """All SO2 levels at one (grid, T), vmapped over SO2.

    resolved_dt=False: dt = 60 s everywhere, full Jacobian (25 fields).
    resolved_dt=True: dt = dt_for(SO2), SO2 levels grouped by dt, and the
    Jacobian only along directions(temp) (4 tangents instead of 25).
    gas_substep='adaptive': dt = 60 s with make_step's H2SO4-budget
    sub-steps (docs/gas_substep.md), Jacobian along the directions as for
    resolved_dt. The SO2 levels are still batched by dt_for, so that a
    vmapped batch does not wait for the stiffest level.
    ppm_tangent: make_step's (frozen_shape: stable PPM tangent, same values).
    """
    dir_jac = resolved_dt or gas_substep is not None
    # Initial state: same background aerosol for every cell; only the SO2 in
    # the gas array differs, so Gc0 is the one batched input (nso2, ngas).
    xk = GRIDS[nbins]()
    so2 = np.array([ppbv_to_molec(c, temp, STRAT['pres']) for c in so2_ppbv])
    states = [lognormal_state(xk, so2=c) for c in so2]
    Nk0, Mk0 = states[0][0], states[0][1]
    Gc0 = jnp.stack([s[2] for s in states])

    # Coefficients as float64 JAX arrays so jacfwd treats every field as an
    # input (Python floats would be closed over as constants).
    P = jax.tree.map(jnp.float64, NUC_PARAMS)
    S2 = jax.tree.map(jnp.float64, NUC_PARAMS_S2)
    dirs = directions(temp)
    D = jnp.array(np.stack(list(dirs.values())))     # (ndir, nfields)

    dts = np.array([dt_for(c) if resolved_dt or gas_substep else DT
                    for c in so2_ppbv])
    parts, timing = [], []
    # One compiled program per dt group: batched while_loops run until the
    # slowest member finishes, so mixing dt (or SO2 regimes) in one vmap
    # would make every level pay for the most expensive one.
    for group in sorted(set(dts.tolist()), reverse=True):
        idx = np.where(dts == group)[0]
        dt = DT if gas_substep else group
        sim = build_sim(xk, temp, hours, oh_conc=oh_conc,
                        coag_solver=coag_solver, dt=dt,
                        gas_substep=gas_substep, ppm_tangent=ppm_tangent)
        sim_s2 = build_sim(xk, temp, hours, dunne_form='single_exp',
                           oh_conc=oh_conc, coag_solver=coag_solver, dt=dt,
                           gas_substep=gas_substep, ppm_tangent=ppm_tangent)

        # Forward-mode Jacobian. has_aux returns the primal outputs and
        # diagnostics from the same pass, so no separate baseline run is
        # needed. Forward mode is the cheaper one here (hundreds of outputs,
        # 4-25 inputs); make_runner(ad_mode='reverse') would allow jax.grad.
        if dir_jac:
            # theta = P + alpha @ D: d/d alpha_i is the derivative along
            # direction i, one tangent per direction.
            def with_jac(gc):
                def f(alpha):
                    shift = alpha @ D
                    p = nuc.NucleationParams(*(getattr(P, n) + shift[k]
                                               for k, n in enumerate(FIELDS)))
                    ys, aux = sim(p, Nk0, Mk0, gc)
                    return ys, (ys, aux)
                return jax.jacfwd(f, has_aux=True)(jnp.zeros(D.shape[0]))
        else:
            def with_jac(gc):
                def f(p):
                    ys, aux = sim(p, Nk0, Mk0, gc)
                    return ys, (ys, aux)
                return jax.jacfwd(f, has_aux=True)(P)

        t0 = time.time()
        jac, (ys, aux) = jax.block_until_ready(
            jax.jit(jax.vmap(with_jac))(Gc0[idx]))
        t_jac = time.time() - t0
        # Structural alternative: Dunne Table S2 single-exponential fit,
        # forward run only (a different functional form, not a perturbation).
        t0 = time.time()
        ys_s2, aux_s2 = jax.block_until_ready(
            jax.jit(jax.vmap(lambda gc: sim_s2(S2, Nk0, Mk0, gc)))(Gc0[idx]))
        t_s2 = time.time() - t0
        label = f"dt_for={group:g}s group" if gas_substep else f"dt={dt:g}s"
        timing.append(f"{label} x{len(idx)}: {t_jac:.0f}+{t_s2:.0f} s")

        if dir_jac:
            # jac[key]: (n, hours, nout, ndir) already
            jac_d = {k: np.array(jac[k]) for k in ('dndlog', 'nmass')}
        else:
            # jac[key] is a NucleationParams of arrays (n, hours, nout); stack
            # the fields last -> (n, hours, nout, nfields)
            jac_d = {k: np.stack([np.array(getattr(jac[k], f)) for f in FIELDS],
                                 axis=-1) for k in ('dndlog', 'nmass')}
        parts.append((idx, dict(
            dndlog=np.array(ys['dndlog']), nmass=np.array(ys['nmass']),
            dndlog_s2=np.array(ys_s2['dndlog']), nmass_s2=np.array(ys_s2['nmass']),
            **{f'jac{"_dir" if dir_jac else ""}_{k}': v
               for k, v in jac_d.items()},
            **{f'aux_{k}': np.array(v) for k, v in aux.items()},
            **{f'aux_s2_{k}': np.array(v) for k, v in aux_s2.items()})))

    # Reassemble the dt groups in the original SO2 order
    order = np.argsort(np.concatenate([idx for idx, _ in parts]))
    out = {k: np.concatenate([d[k] for _, d in parts])[order]
           for k in parts[0][1]}
    out.update(nbins=nbins, temp=temp, hours=hours,
               so2_ppbv=np.array(so2_ppbv), dt_s=dts, oh_conc=oh_conc,
               coag_solver=coag_solver, fields=np.array(FIELDS),
               dir_names=np.array(list(dirs)), directions=np.array(D))

    # Fail loudly on any non-finite output or tangent, naming the SO2 cells.
    bad = [k for k, v in out.items()
           if isinstance(v, np.ndarray) and v.dtype.kind == 'f'
           and v.ndim > 0 and v.shape[0] == len(so2_ppbv)
           and not np.all(np.isfinite(v))]
    if bad:
        cells = sorted({float(so2_ppbv[i]) for k in bad
                        for i in np.where(~np.all(np.isfinite(
                            out[k].reshape(len(so2_ppbv), -1)), axis=1))[0]})
        raise RuntimeError(f"{nbins}-bin T={temp} K: non-finite {bad} at "
                           f"SO2 = {cells} ppbv")

    os.makedirs(outdir, exist_ok=True)
    np.savez(npz_path(outdir, nbins, temp), **out)
    print(f"  {nbins}-bin T={temp:5.1f} K: {'; '.join(timing)}; N(>3nm, end) = "
          + ', '.join(f'{v:.2e}' for v in out['nmass'][:, -1, 0])
          + f"; max sub-steps {int(out['aux_max_sub'].max())}, "
          f"clamp {100 * out['aux_frac_clamp'].mean():.1f}% of steps", flush=True)


# =========================================================================
# Post-processing and plots
# =========================================================================

def load(outdir, nbins, temps):
    data = []
    for T in temps:
        fname = npz_path(outdir, nbins, T)
        if not os.path.exists(fname):
            raise FileNotFoundError(f"Missing NPZ: {fname}. Run simulations first.")
        data.append(add_nuc_mode_channel(np.load(fname)))
    return data


def delta_ln(d, key, name, temp, floor):
    """Linear d ln(Y) along directions(temp)[name]; NaN where Y < floor."""
    # (J @ dtheta) / Y: the first-order change in ln Y when the coefficients
    # move by dtheta. Masked below `floor` where ln Y is meaningless.
    y = d[key]
    if f'jac_dir_{key}' in d:      # --resolved-dt: Jacobian per direction
        dy = d[f'jac_dir_{key}'][..., list(d['dir_names']).index(name)]
    else:
        dy = d[f'jac_{key}'] @ directions(temp)[name]
    return np.where(y > floor, dy / np.where(y > 0, y, 1.0), np.nan)


def conditions_note(d):
    """Figure footnote: the conditions every cell of the map assumes."""
    dts = ', '.join(f'{dt:g}' for dt in sorted(set(d['dt_s'].tolist()), reverse=True))
    return (f"Conditions: {STRAT['pres'] / 100:.0f} hPa (~20 km), RH {100 * STRAT['rh']:.0f}%, "
            f"constant OH {float(d['oh_conc']):.0e} cm$^{{-3}}$ (not titrated by SO$_2$), "
            f"{FION:g} ion pairs cm$^{{-3}}$ s$^{{-1}}$, no NH$_3$ or nucleating organics, no dilution; "
            f"initial sulfate {BG_N:g} cm$^{{-3}}$, GMD {BG_GMD * 1e9:.0f} nm, GSD {BG_GSD:g}; "
            f"cluster and sparse-bin density {NUC_PARAMS.rho_nuc:.0f}/{SPARSE_DENSITY:.0f} kg m$^{{-3}}$; "
            f"dt {dts} s by SO$_2$; {str(d['coag_solver'])} coagulation.\n"
            f"SO$_2$ is a mixing ratio at {STRAT['pres'] / 100:.0f} hPa. At higher pressure the same "
            f"ppbv makes more H$_2$SO$_4$ (about 4.7x at 13 km, 160 hPa), and at $\\geq$1 ppmv "
            f"real OH would be titrated, so H$_2$SO$_4$ production here is an upper bound.")


def dlnY_dlnJall(d, key='nmass', floor=1e-3):
    """d ln Y / d ln J_all: the four Dunne rates scaled by one common factor.

    d ln J_x = d u_x, so this is the J_all direction per unit ln J (Phase 2)
    or the sum of the u_x columns of the full Jacobian (Phase 3). NaN where
    Y < floor.
    """
    y = d[key]
    if f'jac_dir_{key}' in d:
        k = list(d['dir_names']).index('J_all x2.5')
        dy = d[f'jac_dir_{key}'][..., k] / np.log(2.5)
    else:
        dy = d[f'jac_{key}'][..., [FIELDS.index(f'u_{m}') for m in MECHS]].sum(-1)
    return np.where(y > floor, dy / np.where(y > 0, y, 1.0), np.nan)


def plot(outdir, nbins, temps):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import LogNorm, TwoSlopeNorm

    data = load(outdir, nbins, temps)
    so2 = data[0]['so2_ppbv']
    hours = int(data[0]['hours'])
    xt = np.arange(len(so2))
    yt = np.arange(len(temps))
    xlab = [f'{c:g}' for c in so2]
    nm = [f'>{c * 1e9:.0f} nm' for c in NMASS_CUTOFFS_M] + [NUC_MODE_LABEL]
    det = np.log(1.0 + SABRE_NUMBER_REL_ERR)

    def frame(ax, title):
        ax.set_xticks(xt, xlab, rotation=60, fontsize=7)
        ax.set_yticks(yt, [f'{T:.0f}' for T in temps], fontsize=7)
        below = [i for i, T in enumerate(temps) if T < CLOUD_T_MIN]
        if below:
            ax.axhline(max(below) + 0.5, color='k', ls='--', lw=1)
        ax.set_title(title, fontsize=8)

    # --- Fig A: baseline state and regime -------------------------------
    stack = lambda f: np.array([f(d) for d in data])
    panels = [
        ('N(>3 nm) [cm$^{-3}$]', stack(lambda d: d['nmass'][:, -1, 0]), LogNorm()),
        (f'N({NUC_MODE_LABEL}) [cm$^{{-3}}$]',
         stack(lambda d: np.maximum(d['nmass'][:, -1, NUC_MODE_CH], 1e-3)), LogNorm()),
        ('N(>12 nm) [cm$^{-3}$]', stack(lambda d: d['nmass'][:, -1, 2]), LogNorm()),
        ('mean H$_2$SO$_4$ [cm$^{-3}$]', stack(lambda d: d['aux_h2so4_mean'].mean(1)), LogNorm()),
        ('J$_{bi}$ / (J$_{bn}$+J$_{bi}$)', stack(lambda d: d['aux_J_bi_mean'].mean(1)
                                    / np.maximum(d['aux_J_bn_mean'].mean(1)
                                                 + d['aux_J_bi_mean'].mean(1), 1e-300)), None),
        ('steps sub-stepped / clamped', stack(lambda d: np.maximum(
            d['aux_frac_sub'].mean(1), d['aux_frac_clamp'].mean(1))), None),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for ax, (title, z, norm) in zip(axes.flat, panels):
        im = ax.pcolormesh(xt, yt, z, norm=norm, shading='nearest',
                           cmap='viridis')
        fig.colorbar(im, ax=ax, shrink=0.8)
        frame(ax, title)
    for ax in axes[-1]:
        ax.set_xlabel('SO$_2$ [ppbv]')
    for ax in axes[:, 0]:
        ax.set_ylabel('T [K]')
    fig.suptitle(f'Phase 2 baseline after {hours} h, {nbins} bins '
                 f'(dashed: below CLOUD 208 K)', fontsize=10)
    fig.text(0.01, 0.005, conditions_note(data[0]), fontsize=7, va='bottom', wrap=True)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    out = os.path.join(outdir, f'map_{nbins}bin_baseline.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")

    # --- Fig B: sensitivity along published directions + Table S2 --------
    # Rows: uncertainty directions (linear, from the Jacobian) plus the Table S2
    # forward run (exact ln ratio). Columns: N(>3, 3-10.65, >12 nm) at the
    # final hour. Dots mark changes SABRE's 8% number uncertainty could detect.
    # Last row: elasticity to a common scaling of all Dunne rates (1 = N ∝ J)
    # Colour limits come from the 95th percentile of each row, not its maximum:
    # a few 80-bin cells have tangents of ~1e42 (blown-up AD, not a response),
    # which a max-based scale turns into one flat colour. Crosses mark cells
    # with |value| > BLOWUP, left grey like the masked (NaN) cells.
    BLOWUP = 10.0
    rows = list(directions(temps[0]).keys()) + ['Table S2 fit', 'J_all elasticity']
    chans = [0, NUC_MODE_CH, 2]
    fig, axes = plt.subplots(len(rows), len(chans), figsize=(11.5, 2.3 * len(rows)))
    for r, name in enumerate(rows):
        z = np.full((len(chans), len(temps), len(so2)), np.nan)
        for i, (d, T) in enumerate(zip(data, temps)):
            if name == 'J_all elasticity':
                v = dlnY_dlnJall(d)[:, -1]
            elif name == 'Table S2 fit':
                y, y2 = d['nmass'][:, -1], d['nmass_s2'][:, -1]
                v = np.where((y > 1e-3) & (y2 > 1e-3),
                             np.log(np.maximum(y2, 1e-300) / np.maximum(y, 1e-300)),
                             np.nan)
            else:
                v = delta_ln(d, 'nmass', name, T, 1e-3)[:, -1]
            z[:, i, :] = v[:, chans].T
        absz = np.abs(z[np.isfinite(z)])
        vmax = max(np.percentile(absz, 95), 2 * det) if absz.size else 1.0
        for c, ch in enumerate(chans):
            ax = axes[r, c]
            blown = np.abs(np.nan_to_num(z[c])) > BLOWUP
            zc = np.where(blown, np.nan, z[c])
            ax.set_facecolor('0.8')
            if name == 'J_all elasticity':
                im = ax.pcolormesh(xt, yt, zc, shading='nearest', cmap='viridis',
                                   vmin=0.0, vmax=1.2)
                fig.colorbar(im, ax=ax, shrink=0.8, extend='both')
                frame(ax, f'd ln N({nm[ch]}) / d ln J$_{{all}}$')
            else:
                im = ax.pcolormesh(xt, yt, zc, shading='nearest', cmap='RdBu_r',
                                   norm=TwoSlopeNorm(0.0, -vmax, vmax))
                jj, ii = np.where((np.abs(np.nan_to_num(z[c])) > det) & ~blown)
                ax.plot(xt[ii], yt[jj], 'k.', ms=3)
                fig.colorbar(im, ax=ax, shrink=0.8, extend='both')
                frame(ax, f'{name}: $\\delta$ ln N({nm[ch]})')
            jj, ii = np.where(blown)
            ax.plot(xt[ii], yt[jj], 'kx', ms=6, mew=1.2)
    for ax in axes[-1]:
        ax.set_xlabel('SO$_2$ [ppbv]')
    fig.suptitle(f'Phase 2: response after {hours} h, {nbins} bins. Linear '
                 f'(AD) except Table S2 (forward run). Dots: |change| > SABRE '
                 f'±{100 * SABRE_NUMBER_REL_ERR:.0f}%. Crosses: |value| > '
                 f'{BLOWUP:g}. Grey: masked', fontsize=10)
    fig.text(0.01, 0.003, conditions_note(data[0]), fontsize=7, va='bottom', wrap=True)
    fig.tight_layout(rect=(0, 0.035, 1, 1))
    out = os.path.join(outdir, f'map_{nbins}bin_sensitivity.png')
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  wrote {out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--plot-only', action='store_true')
    parser.add_argument('--smoke', action='store_true',
                        help='40 bins, T 208/217 K, SO2 0.1/1/100 ppbv, 2 h '
                             '(results/phase2_smoke)')
    parser.add_argument('--grids', type=int, nargs='+', default=list(GRIDS))
    parser.add_argument('--oh', type=float, default=OH_CONC,
                        help='constant OH [molec/cm3]; no OH titration by SO2')
    parser.add_argument('--coag-solver', default='euler_fixed',
                        choices=['euler_fixed', 'euler_adaptive'],
                        help='euler_adaptive writes to results/phase2_adaptive')
    parser.add_argument('--resolved-dt', action='store_true',
                        help='dt per SO2 level (dt_for) and Jacobian along the '
                             'published directions only; writes results/phase2_dt')
    parser.add_argument('--gas-substep', choices=['adaptive'], default=None,
                        help="dt = 60 s with make_step(gas_substep='adaptive') "
                             'and the directional Jacobian; writes '
                             'results/phase2_gas')
    parser.add_argument('--ppm-tangent', choices=['exact', 'frozen_shape'],
                        default='exact',
                        help="make_step's PPM derivative; frozen_shape (stable, "
                             "not exact) appends _frozen to the results directory")
    args = parser.parse_args()
    if args.resolved_dt and args.gas_substep:
        parser.error('--resolved-dt and --gas-substep are alternatives')

    if args.smoke:
        grids, temps, so2, hours = [40], [208.0, 217.0], [1e-1, 1.0, 1e2], 2
        outdir = os.path.join(RESULTS, 'phase2_smoke')
    else:
        grids, temps, so2, hours = args.grids, TEMPS, SO2_PPBV, HOURS
        outdir = os.path.join(RESULTS, 'phase2')
    if args.resolved_dt:
        outdir += '_dt'
    if args.gas_substep:
        outdir += '_gas'
    if args.coag_solver != 'euler_fixed':
        outdir += '_adaptive'
    if args.ppm_tangent == 'frozen_shape':
        outdir += '_frozen'

    if not args.plot_only:
        for nbins in grids:
            for T in temps:
                run_cell_set(nbins, T, so2, hours, outdir, args.oh,
                             args.coag_solver, args.resolved_dt,
                             args.gas_substep, args.ppm_tangent)
    for nbins in grids:
        plot(outdir, nbins, temps)


if __name__ == '__main__':
    main()
