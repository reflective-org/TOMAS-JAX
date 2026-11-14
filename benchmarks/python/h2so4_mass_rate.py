"""H2SO4 condensation / evaporation rate of single sulfate-water droplets vs
gas-phase H2SO4, with the droplet's equilibrium vapour pressure (Kelvin term
included) and without it (C_eq = 0, as TOMAS-JAX and TOMAS Fortran condense).

    python -m benchmarks.python.h2so4_mass_rate

Rate per particle (continuum flux with the Fuchs-Sutugin correction, as
condensation_sink.py / getCondSink.f, alpha = 1):
    dn/dt = 2 pi Dp_wet Di beta(Kn) (C - C_eq,flat K(Dp_wet)),  Kn = 2 mfp / Dp_wet
plotted as the fractional rate of the droplet's H2SO4 content,
    100 * 3600 * (dn/dt) / n_H2SO4   [% per hour].
The water follows the H2SO4 at fixed composition (wt % from RH), so this is
also the fractional rate of the wet mass, and of the dry (SO4) mass.
Sizes are dry radii (SO4 at DENS_INIT = 1770 kg/m3, the grid's density):
1, 10, 101, 1024 nm. The wet diameter used for the flux and the Kelvin term
is the dry diameter / dry_over_wet(wt).

C_eq,flat (Ayers / Kulmala & Laaksonen pure acid x Giauque activity),
Kelvin factor (sigma = 0.08 N/m) and the Tabazadeh composition are those of
h2so4_kelvin_driving_force.py. Signed rate on a symlog axis
(linear between -1e-4 and +1e-4 %/h, as Kleinschmitt's Fig. 3.4): solid =
net rate (positive = condensation, negative = evaporation), dotted = C_eq = 0
(the model). The dot on each curve is C = C_eq (zero net rate); the C grid is
refined around it so the zero crossing is resolved. Grey bands: hourly-mean H2SO4 of
the Phase 3 runs (hours 2-48), bg_midlat and sai_plume, both run at 217 K.

(a) 217 K, 5500 Pa (sai_plume / bg_midlat), 5 ppmv H2O -> 72 wt%.
(b) 250 K, 5000 Pa, 5 ppmv H2O -> RH below the Tabazadeh table, clamped at
    80 wt%: C_eq is a lower bound (90 wt% would raise it ~30x; the C = C_eq
    point at 90 wt% is marked with an open circle).

--kleinschmitt [--kl-wt 75]: Kleinschmitt (2017, PhD thesis, LMDZ-S3A,
Sect. 3.3.4, Eqs. 3.9-3.14, 3.26) Fig. 3.4 at 250 K / 50 hPa, changed one
setting at a time towards ours (KL_STEPS, cumulative; each panel shows the
previous step faintly), written to h2so4_mass_rate_kleinschmitt.png:
    (a) her formulation (reproduces her Fig. 3.4);
    (b) + alpha = 1 (ours, TOMAS);
    (c) + Kelvin with the H2SO4 molar volume in the solution, M_H2SO4 / rho_p,
        in place of m_H2O / rho_p (sigma, rho_p kept);
    (d) + the H2SO4 mean free path, 2 D / vbar (gas_properties, as
        getCondSink.f), in place of the air mean free path in Kn (her flux
        formula kept: with it the continuum limit becomes 4 pi r D C);
    (e) ours (panel (b) of the main figure). What still differs between (d)
        and (e): the flat-surface p_sat (Ayers / 0.086 vs the CARMA Ayers /
        Kulmala form), composition (75 vs 80 wt%), sigma (0.072 vs 0.08),
        density, Fuchs-Sutugin vs her transition formula, wet vs dry radius.
    (f) legend and a table: C = C_eq and the rate at 1e5 / 1e9 cm-3 per step.
Her formulation:
    p_sat [Pa] = 101325 / 0.086 exp(-10156/T + 16.259 + (mu - mu0)/(R T))
        (Ayers et al. 1980 / 0.086, Hamill et al. 1982; no Kulmala low-T term)
    Kelvin exp(2 sigma m_H2O / (rho_p k_B T r)), sigma = 0.072 N/m, m_H2O the
        molecular mass of WATER (the H2SO4 molar volume belongs here)
    J = pi r^2 vbar alpha (C - C_sat) (1 + Kn) / (1 + Kn + alpha / (2 Kn)),
        alpha = 0.1, Kn = lambda_air / r,
        lambda_air = 6.6e-8 m (1013.25 hPa / p) (T / 293.15 K)
    rho_p = (A w^2 + B w + C) (1 - 0.02/30 (T - 293)) g/cm3 (Eq. 3.26)
    rate = J / N_H2SO4 (Eq. 3.14), r = the particle's (wet) radius.
Two inputs are not in the thesis and are stand-ins: (mu - mu0) is taken as
-E(wt), the Giauque enthalpy fit used for our curves (she uses the Giauque
Table I chemical potentials), and the composition, which her Fig. 3.4 does not
state (her model takes it from Steele & Hamill / Tabazadeh at the local H2O);
--kl-wt sets it, default 75 wt% (the composition of her Fig. 2.2).
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tomas_jax.core.config import AVOGADRO, MW_H2SO4, MW_SO4, PI, SV_H2SO4
from tomas_jax.physics.gas_properties import (calc_fuchs_sutugin_correction,
                                              calc_gas_diffusivity,
                                              calc_mean_free_path)
from sensitivities.dunne_nuc.common import DENS_INIT
from sensitivities.dunne_nuc.scenarios import SCENARIOS, rh_from_h2o

from .h2so4_kelvin_driving_force import (K_B, OUTDIR, R_GAS, SKIP_HOURS, PHASE3,
                                         WT_SENS, WT_TABLE_MAX, _check, c_eq_flat,
                                         dry_over_wet, giauque_enthalpy, kelvin,
                                         sanity, wt_percent)

R_DRY_NM = (1.0, 10.0, 101.0, 1024.0)
C_GRID = np.geomspace(1e5, 1e9, 400)          # molec/cm3
H2O_VMR = 5.0e-6
CASES = ((217.0, 5500.0), (250.0, 5000.0))
# sequential (ordered sizes): one blue ramp, light -> dark
SIZE_COLOR = ('#7fb0ec', '#3d86dc', '#1f5fae', '#0d2f5e')
Y_LIN = 1e-4      # symlog: linear for |rate| < Y_LIN [%/h]
Y_LIM = (-1e4, 1e5)
BAND_SCEN = ('bg_midlat', 'sai_plume')


def rates(T, p, wt, r_dry_nm, C):
    """(net, nonvolatile) fractional rate [%/h] and C_eq [cm-3] for one size."""
    dp_dry = 2.0 * r_dry_nm * 1e-9
    dp_wet = dp_dry / dry_over_wet(wt)
    Di = float(calc_gas_diffusivity(T, p, MW_H2SO4, SV_H2SO4))       # m2/s
    mfp = float(calc_mean_free_path(T, p, MW_H2SO4, SV_H2SO4))
    beta = float(calc_fuchs_sutugin_correction(2.0 * mfp / dp_wet, 1.0))
    c_eq = float(c_eq_flat(T, wt) * kelvin(T, wt, dp_wet * 1e9))
    k = 2.0 * PI * dp_wet * Di * beta * 1e6                         # cm3/s
    n_h2so4 = PI / 6.0 * dp_dry**3 * DENS_INIT / (MW_SO4 / 1000.0) * AVOGADRO
    to_pct_h = 100.0 * 3600.0 / n_h2so4
    return k * (C - c_eq) * to_pct_h, k * C * to_pct_h, c_eq


KL_ALPHA = 0.1
KL_SIGMA = 0.072
KL_CASE = (250.0, 5000.0)
MW_H2O = 18.015
KL_BASE = dict(alpha=KL_ALPHA, kelvin_mw=MW_H2O, mfp='air')
KL_STEPS = (
    ('Kleinschmitt (2017)', 'her formulation',
     KL_BASE),
    ('+ $\\alpha$ = 1', '$\\alpha$ 0.1 $\\rightarrow$ 1',
     dict(KL_BASE, alpha=1.0)),
    ('+ Kelvin with H$_2$SO$_4$ in solution',
     'Kelvin m$_{H_2O}$ $\\rightarrow$ M$_{H_2SO_4}$ / $\\rho_p$',
     dict(KL_BASE, alpha=1.0, kelvin_mw=MW_H2SO4)),
    ('+ H$_2$SO$_4$ mean free path',
     'Kn: $\\lambda_{air}$ $\\rightarrow$ $\\lambda_{H_2SO_4}$ = 2D/$\\bar v$',
     dict(KL_BASE, alpha=1.0, kelvin_mw=MW_H2SO4, mfp='h2so4')),
)


def kl_density(T, wt):
    """Kleinschmitt Eq. 3.26 [kg/m3]."""
    rho = 7.8681252e-6 * wt**2 + 8.2185978e-3 * wt + 0.97968381
    return 1000.0 * rho * (1.0 - 0.02 / 30.0 * (T - 293.0))


def rates_kleinschmitt(T, p, wt, r_nm, C, alpha=KL_ALPHA, kelvin_mw=MW_H2O, mfp='air'):
    """As rates(), with Kleinschmitt's formulation; r_nm is the (wet) radius.
    alpha, kelvin_mw (molar mass [g/mol] in the Kelvin term; hers is water's)
    and mfp ('air' = hers, 'h2so4' = 2 D / vbar) switch her settings to ours."""
    r = r_nm * 1e-9
    rho = kl_density(T, wt)
    p_sat = 101325.0 / 0.086 * np.exp(-10156.0 / T + 16.259
                                      - giauque_enthalpy(wt) / (R_GAS * T))
    m_kelvin = kelvin_mw * 1e-3 / AVOGADRO
    c_eq = float(p_sat * np.exp(2.0 * KL_SIGMA * m_kelvin / (rho * K_B * T * r))
                 / (K_B * T) * 1e-6)
    vbar = np.sqrt(8.0 * K_B * T / (PI * MW_H2SO4 * 1e-3 / AVOGADRO))   # m/s
    if mfp == 'air':
        lam = 6.6e-8 * (101325.0 / p) * (T / 293.15)
    elif mfp == 'h2so4':
        lam = float(calc_mean_free_path(T, p, MW_H2SO4, SV_H2SO4))
    else:
        raise ValueError(f'mfp = {mfp!r}')
    kn = lam / r
    corr = (1.0 + kn) / (1.0 + kn + alpha / (2.0 * kn))
    k = PI * r**2 * vbar * alpha * corr * 1e6                          # cm3/s
    n_h2so4 = 4.0 / 3.0 * PI * r**3 * rho * wt / 100.0 / (MW_H2SO4 * 1e-3) * AVOGADRO
    to_pct_h = 100.0 * 3600.0 / n_h2so4
    return k * (C - c_eq) * to_pct_h, k * C * to_pct_h, c_eq


def h2so4_bands():
    out = []
    for name in BAND_SCEN:
        fname = os.path.join(PHASE3, f'{name}_40bin.npz')
        if not os.path.exists(fname):
            raise FileNotFoundError(f'Missing NPZ: {fname}. Run phase3_scenarios first.')
        h = _check(name, np.load(fname)['aux_h2so4_mean'])[SKIP_HOURS:]
        out.append((name, h.min(), h.max()))
    return out


def panel(ax, T, p, bands, letter, kl=None, prev=None, heading=None):
    """kl not None: Kleinschmitt's formulation, dict(wt, alpha, kelvin_mw, mfp).
    prev: the previous step's kl, drawn faintly. Returns per-size
    (r, C_eq, rate at 1e5, rate at 1e9)."""
    rh = 100.0 * rh_from_h2o(H2O_VMR, T, p)
    wt = float(wt_percent(T, rh)) if kl is None else kl['wt']
    capped = kl is None and wt >= WT_TABLE_MAX
    if kl is None:
        rate_fn = rates
    else:
        opts = {k: v for k, v in kl.items() if k != 'wt'}
        rate_fn = lambda T_, p_, w_, r_, C_: rates_kleinschmitt(T_, p_, w_, r_, C_, **opts)
    rsym = 'r$_{dry}$' if kl is None else 'r'
    for name, lo, hi in bands:
        ax.axvspan(lo, hi, color='0.88', lw=0, zorder=0)
        ax.text(np.sqrt(lo * hi), 0.985, f'{name}\n(217 K run)', transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=8.5, color='0.35')
    print(f'\n{T:g} K, {p:g} Pa: RH {rh:.3f} %, {wt:.1f} wt%'
          f'{" (clamped)" if capped else ""}, dry/wet {dry_over_wet(wt):.3f}')
    if kl is not None:
        print(f'  Kleinschmitt formulation, r = wet radius: {kl}')
    print(f'  {"r nm":>8} {"C_eq cm-3":>10} {"%/h @1e5":>10} {"%/h @1e7":>10} {"%/h @1e9":>10}')
    below, summary = [], []
    if prev is not None:
        popts = {k: v for k, v in prev.items() if k != 'wt'}
        for r, col in zip(R_DRY_NM, SIZE_COLOR):
            c_prev = rates_kleinschmitt(T, p, prev['wt'], r, C_GRID, **popts)[2]
            eps = np.geomspace(1e-9, 3e-2, 40)
            cf = np.unique(np.concatenate([C_GRID, c_prev * (1 - eps), [c_prev], c_prev * (1 + eps)]))
            cf = cf[(cf >= C_GRID[0]) & (cf <= C_GRID[-1])]
            ax.plot(cf, rates_kleinschmitt(T, p, prev['wt'], r, cf, **popts)[0], '-',
                    color=col, lw=1.2, alpha=0.35, zorder=2)
    for r, col in zip(R_DRY_NM, SIZE_COLOR):
        net, nv, c_eq = rate_fn(T, p, wt, r, C_GRID)
        if c_eq < C_GRID[0]:
            below.append((r, c_eq))
        _check(f'nonvolatile rate r={r}', nv)
        ax.plot(C_GRID, nv, ':', color=col, lw=1.4)
        # refine around C = C_eq: the rate crosses the linear band within a tiny dC
        eps = np.geomspace(1e-9, 3e-2, 40)
        c_fine = np.unique(np.concatenate([C_GRID, c_eq * (1.0 - eps), [c_eq], c_eq * (1.0 + eps)]))
        c_fine = c_fine[(c_fine >= C_GRID[0]) & (c_fine <= C_GRID[-1])]
        net_fine, _, _ = rate_fn(T, p, wt, r, c_fine)
        ax.plot(c_fine, net_fine, '-', color=col, lw=2.2, label=f'{rsym} = {r:g} nm')
        if C_GRID[0] < c_eq < C_GRID[-1]:
            ax.plot(c_eq, 0.0, 'o', color=col, ms=8, mec='white', mew=2, zorder=5)
        if capped:
            _, _, c_eq90 = rates(T, p, WT_SENS, r, C_GRID)
            if C_GRID[0] < c_eq90 < C_GRID[-1]:
                ax.plot(c_eq90, 0.0, 'o', mfc='white', mec=col, mew=1.8, ms=8, zorder=5)
        pick = [np.interp(np.log(x), np.log(C_GRID), net) for x in (1e5, 1e7, 1e9)]
        print(f'  {r:8g} {c_eq:10.2e} ' + ' '.join(f'{v:10.3g}' for v in pick))
        summary.append((r, c_eq, pick[0], pick[2]))
    if below:
        txt = ('C = C$_{eq}$ is left of the axis for\n'
               + '\n'.join(f'{rsym} = {r:g} nm: C$_{{eq}}$ = {c:.2g} cm$^{{-3}}$' for r, c in below)
               + f'\nso the curves overlay C$_{{eq}}$ = 0 (dotted)\n'
               f'to within {100 * max(c for _, c in below) / C_GRID[0]:.0f} % at 10$^5$ cm$^{{-3}}$')
        ax.text(0.97, 0.03, txt, transform=ax.transAxes, ha='right', va='bottom', fontsize=9, multialignment='left',
                bbox=dict(boxstyle='round,pad=0.4', fc='white', ec='0.8'))
    title = f'({letter}) {T:g} K, {p / 100:g} hPa, {rh:.2f} % RH, {wt:.0f} wt% H$_2$SO$_4$'
    if kl is not None:
        title = f'({letter}) {heading}'
    elif heading is not None:
        title = f'({letter}) {heading}\n{T:g} K, {p / 100:g} hPa, {wt:.0f} wt% (Tabazadeh table limit; open circles: 90 wt%)'
    if capped and heading is None:
        title += ' (table limit:\nC$_{eq}$ is a lower bound; open circles = C = C$_{eq}$ at 90 wt%)'
    ax.set_title(title, fontsize=11, loc='left')
    ax.set_xscale('log')
    ax.set_yscale('symlog', linthresh=Y_LIN, linscale=1.0)
    ax.axhline(0.0, color='0.4', lw=0.8, zorder=1)
    ax.set_xlim(C_GRID[0], C_GRID[-1])
    ax.set_ylim(*Y_LIM)
    ax.set_xlabel('Gas-phase H$_2$SO$_4$ [molec cm$^{-3}$]')
    ax.grid(True, which='major', color='0.9', lw=0.6)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    return summary


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--kleinschmitt', action='store_true',
                    help='Kleinschmitt (2017) Fig. 3.4 formulation next to ours at 250 K, 50 hPa')
    ap.add_argument('--kl-wt', type=float, default=75.0,
                    help='H2SO4 wt%% for the Kleinschmitt formulation (not given in her Fig. 3.4)')
    args = ap.parse_args()
    sanity()
    os.makedirs(OUTDIR, exist_ok=True)
    bands = h2so4_bands()
    from matplotlib.lines import Line2D
    style_h = [Line2D([], [], color='0.3', lw=2.2),
               Line2D([], [], color='0.3', lw=1.4, ls=':'),
               Line2D([], [], color='0.3', marker='o', ls='', ms=8, mec='white', mew=2)]
    style_l = ['net rate (with C$_{eq}$, Kelvin)',
               'C$_{eq}$ = 0' + ('' if args.kleinschmitt else ' (the model)'),
               'C = C$_{eq}$ (zero net rate)']
    ylabel = ('(d m / d t) / m  [% per hour]\n'
              f'< 0 evaporation, > 0 condensation; linear for |rate| < {Y_LIN:g}')
    if args.kleinschmitt:
        fig, axes = plt.subplots(2, 3, figsize=(20, 12), sharey=True)
        axes = axes.ravel()
        rows, prev = [], None
        T, p = KL_CASE
        for i, (name, change, opts) in enumerate(KL_STEPS):
            kl = dict(opts, wt=args.kl_wt)
            ttl = (f'{name}\n{change}' if i else
                   f'{name}: {T:g} K, {p / 100:g} hPa, {args.kl_wt:g} wt% (assumed)\n'
                   f'$\\alpha$ = {KL_ALPHA:g}, Kelvin with m$_{{H_2O}}$, $\\lambda_{{air}}$, '
                   f'r = wet radius')
            rows.append((name, panel(axes[i], T, p, bands, 'abcd'[i], kl=kl, prev=prev,
                                     heading=ttl)))
            prev = kl
        rows.append(('ours', panel(axes[4], T, p, bands, 'e',
                                   heading='ours (TOMAS-JAX + C$_{eq}$), r = dry radius')))
        for ax in axes[:5]:
            ax.tick_params(labelleft=True)
        axes[0].set_ylabel(ylabel)
        axes[3].set_ylabel(ylabel)
        ax = axes[5]
        ax.axis('off')
        h, l = axes[0].get_legend_handles_labels()
        h = h + style_h + [Line2D([], [], color='0.3', lw=1.2, alpha=0.35)]
        l = l + style_l + ['previous step (faint)']
        ax.legend(h, l, loc='upper left', fontsize=10, frameon=False)
        lines = ['C = C_eq [cm-3] and rate [%/h] at C = 1e5 / 1e9 cm-3', '',
                 f'{"":33s}{"C_eq":>9s}{"1e5":>9s}{"1e9":>9s}{"C_eq":>10s}{"1e5":>9s}{"1e9":>8s}',
                 f'{"":33s}{"---- r = 1 nm ----":>27s}{"--- r = 1024 nm ---":>27s}']
        for name, summ in rows:
            nm = (name.replace('$\\alpha$', 'alpha').replace('H$_2$SO$_4$', 'H2SO4')
                  .replace('$', ''))
            (r1, c1, a1, b1), (r4, c4, a4, b4) = summ[0], summ[-1]
            lines.append(f'{nm:33s}{c1:9.1e}{a1:9.2g}{b1:9.2g}{c4:10.1e}{a4:9.2g}{b4:8.2g}')
        ax.text(-0.05, 0.45, '\n'.join(lines), transform=ax.transAxes, va='top', ha='left',
                family='monospace', fontsize=8.6)
        fig.suptitle('H$_2$SO$_4$ condensation / evaporation rate at 250 K, 50 hPa: from '
                     'Kleinschmitt (2017) Fig. 3.4 to ours, one setting at a time',
                     fontsize=14, x=0.01, ha='left')
    else:
        fig, axes = plt.subplots(1, 2, figsize=(14, 6.4), sharey=True)
        for ax, (T, p), letter in zip(axes, CASES, 'ab'):
            panel(ax, T, p, bands, letter)
        axes[0].set_ylabel('(d m / d t) / m  [% per hour]  (H$_2$SO$_4$ content = dry and wet mass)\n'
                           f'< 0 evaporation, > 0 condensation; linear for |rate| < {Y_LIN:g}')
        h, l = axes[0].get_legend_handles_labels()
        axes[0].legend(h + style_h, l + style_l, loc='upper left', bbox_to_anchor=(0.0, 0.91),
                       fontsize=8.5, frameon=True, framealpha=0.95)
        fig.suptitle('H$_2$SO$_4$ condensation / evaporation rate of sulfate–water droplets '
                     '(5 ppmv H$_2$O, $\\alpha$ = 1, Fuchs–Sutugin)', fontsize=13, x=0.01, ha='left')
    fig.tight_layout()
    fname = os.path.join(OUTDIR, 'h2so4_mass_rate_kleinschmitt.png' if args.kleinschmitt
                         else 'h2so4_mass_rate.png')
    fig.savefig(fname, dpi=150)
    print(f'\nwrote {fname}')


if __name__ == '__main__':
    main()
