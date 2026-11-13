"""Growth lost by treating H2SO4 as nonvolatile in the SAI plume: equilibrium
H2SO4 over sulfate-water droplets, Kelvin (curvature) term included, against
the plume's H2SO4. TOMAS-JAX condenses H2SO4 with C_eq = 0 (no vapour
pressure, so no Kelvin term), as TOMAS Fortran does.

    python -m benchmarks.python.h2so4_kelvin_driving_force

Growth is proportional to C - C_eq, so with S = C_H2SO4 / C_eq(Dp)
    growth / nonvolatile growth = 1 - 1/S,  growth lost = C_eq / C = 1/S.
The figure plots 1/S on a log axis (1 - 1/S is within 1e-3 of 1 for the
plume and would be a flat line). Reference lines: 1/S = 1 (S = 1, growth
stops; above it the droplet evaporates), 0.1 (10 % slower), 0.01 (1 %).

Equilibrium H2SO4 over a binary solution (as in CARMA/WACCM,
vaporp_H2SO4_Ayers1980.F90):
    ln p [atm] = -10156/T0 + 16.259
                 + 10156 (-1/T + 1/T0 + 0.38/(Tc - T0) (1 + ln(T0/T) - T0/T))
                 - E(wt) / (R T)
    T0 = 340 K, Tc = 905 K: Ayers et al. (1980) pure-acid fit with the
    Kulmala & Laaksonen (1990) low-temperature extrapolation;
    E(wt) = 4.184 (23624.8 - 1.14208e8 / ((wt - 105.318)^2 + 4798.69)) J/mol,
    a fit to Giauque et al. (1960) partial molar enthalpies (the activity term).
Solution composition: Tabazadeh et al. (1997), tabazadeh_h2so4_wt, at the RH
of the water vapour (scenarios.rh_from_h2o, Buck 1981 over liquid water).
Kelvin factor exp(4 sigma v / (R T Dp)), v = M_H2SO4 / rho_solution,
sigma = 0.08 N/m (upper end of 0.07-0.08 for 50-75 wt%: conservative).

(a) 1/S vs wet diameter (1.5-100 nm) for the Phase 3 sai_plume scenario
    (sensitivities/dunne_nuc/scenarios.py: fixed 217 K, 5500 Pa, 5 ppmv H2O
    -> RH 0.89 %, 72 wt%). Line = median, band = range of the hourly-mean
    H2SO4 over the 48 h 40-bin run (results/phase3/sai_plume_40bin.npz,
    aux_h2so4_mean: 6.0e7-1.1e8 cm-3); dotted = flat surface (no Kelvin)
    at the median H2SO4, which the droplet curve approaches at large Dp.
(b) 1/S at 1.7 nm (solid), 3 nm (dashed) and for a flat surface (dotted) vs
    altitude, AFGL tropical 13-22 km (common.TROPICAL_AFGL, linear T / log p
    between levels; temperature on the top axis), H2O 4 ppmv (drier than the
    scenario's 5 ppmv: more concentrated droplets, higher C_eq), at the
    sai_plume run's lowest hourly-mean H2SO4 (conservative).

Result (2026-11-12): at most 6e-4 at 1.7 nm (lowest plume hour), ~2e-5 at
3 nm, ~5e-7 for a flat surface; the altitude worst case is 22 km (213.6 K,
0.8 % RH), 1.6e-4 at 1.7 nm. The bulk Kelvin formula is rough at 1.7 nm,
where the Dunne nucleation rate already sets cluster survival.

Output: benchmarks/results/h2so4_kelvin/h2so4_kelvin_driving_force.png and a
printed table (S, i.e. the inverse of the plotted quantity).
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tomas_jax.core.config import MW_H2SO4
from tomas_jax.physics.water_equilibrium import tabazadeh_h2so4_wt
from tomas_jax.physics.radiative_forcing import h2so4_solution_density
from sensitivities.dunne_nuc.common import TROPICAL_AFGL
from sensitivities.dunne_nuc.scenarios import SCENARIOS, rh_from_h2o

OUTDIR = os.path.join('benchmarks', 'results', 'h2so4_kelvin')
PHASE3 = os.path.join('sensitivities', 'dunne_nuc', 'results', 'phase3')
R_GAS = 8.3143          # J/mol/K (as in CARMA)
K_B = 1.380649e-23      # J/K
ATM = 101325.0          # Pa
SIGMA = 0.08            # N/m
DP_NM = np.geomspace(1.5, 100.0, 240)
H2O_VMR_ALT = 4.0e-6
ALT_KM = np.linspace(13.0, 22.0, 91)
SCEN_ORDER = ('sai_plume',)

INK, INK2, MUTED, GRID, SURFACE = '#1f1f1e', '#5f5e58', '#898781', '#e4e3df', '#fcfcfb'
SCEN_COLOR = {'sai_plume': '#eb6834'}
SCEN_LABEL = {'sai_plume': 'sai_plume (217 K, 5.5 kPa, 10 ppmv SO$_2$)'}
LEVEL_COLOR = {'h_lo': '#eb6834'}
LEVEL_LABEL = {'h_lo': 'lowest hour'}


def afgl(z_km):
    """T [K], p [Pa] at z: AFGL tropical, linear T and log p between levels."""
    zs = np.array(sorted(TROPICAL_AFGL))
    T = np.array([TROPICAL_AFGL[z][0] for z in zs])
    p = np.array([TROPICAL_AFGL[z][1] for z in zs])
    z_km = np.asarray(z_km, dtype=np.float64)
    return np.interp(z_km, zs, T), np.exp(np.interp(z_km, zs, np.log(p)))


def wt_percent(T, rh_percent):
    T, rh = np.broadcast_arrays(np.asarray(T, dtype=np.float64),
                                np.asarray(rh_percent, dtype=np.float64))
    out = [float(tabazadeh_h2so4_wt(t, h)) for t, h in zip(T.ravel(), rh.ravel())]
    return np.array(out).reshape(T.shape)


def ln_p_pure_atm(T):
    T0, Tc = 340.0, 905.0
    return (-10156.0 / T0 + 16.259
            + 10156.0 * (-1.0 / T + 1.0 / T0
                         + 0.38 / (Tc - T0) * (1.0 + np.log(T0 / T) - T0 / T)))


def giauque_enthalpy(wt):
    return np.maximum(4.184 * (23624.8 - 1.14208e8 / ((wt - 105.318)**2 + 4798.69)), 0.0)


def c_eq_flat(T, wt):
    """Equilibrium H2SO4 over a flat solution surface [molec/cm3]."""
    p = np.exp(ln_p_pure_atm(T) - giauque_enthalpy(wt) / (R_GAS * T)) * ATM
    return p / (K_B * T) * 1e-6


def kelvin(T, wt, dp_nm, sigma=SIGMA):
    v = (MW_H2SO4 / 1000.0) / np.asarray(h2so4_solution_density(wt), dtype=np.float64)
    return np.exp(4.0 * sigma * v / (R_GAS * T * dp_nm * 1e-9))


def _check(name, a):
    a = np.asarray(a)
    if a.size == 0 or not np.all(np.isfinite(a)) or np.any(a <= 0):
        raise ValueError(f'{name}: empty, non-finite or non-positive values')
    return a


def sanity():
    p298 = c_eq_flat(298.0, 100.0) * 1e6 * K_B * 298.0
    if not 1e-4 < p298 < 1e-1:
        raise ValueError(f'pure H2SO4 p_sat(298 K) = {p298:.2e} Pa, expected ~1e-3 Pa')
    k = kelvin(220.0, 70.0, 1.7, 0.075)
    if not 30.0 < k < 1000.0:
        raise ValueError(f'Kelvin factor at 1.7 nm / 220 K = {k:.1f}, expected ~1e2')
    print(f'check: pure H2SO4 p_sat(298 K) = {p298:.2e} Pa; '
          f'Kelvin(1.7 nm, 220 K, 70 wt%, 0.075 N/m) = {k:.0f}')


def scenario_rows():
    rows = []
    for name in SCEN_ORDER:
        fname = os.path.join(PHASE3, f'{name}_40bin.npz')
        if not os.path.exists(fname):
            raise FileNotFoundError(f'Missing NPZ: {fname}. Run phase3_scenarios first.')
        h = _check(f'{name} aux_h2so4_mean', np.load(fname)['aux_h2so4_mean'])
        sc = SCENARIOS[name]
        T, p = sc['cond']['temp'], sc['cond']['pres']
        rh = 100.0 * rh_from_h2o(sc['h2o_vmr'], T, p)
        wt = float(wt_percent(T, rh))
        flat = float(_check(f'{name} C_eq flat', c_eq_flat(T, wt)))
        c_eq = _check(f'{name} C_eq', flat * kelvin(T, wt, DP_NM))
        rows.append(dict(name=name, T=T, rh=rh, wt=wt, h_lo=h.min(), h_hi=h.max(),
                         h_med=np.median(h), c_eq=c_eq, flat=flat))
    return rows


def altitude_profile():
    T, p = afgl(ALT_KM)
    rh = 100.0 * rh_from_h2o(H2O_VMR_ALT, T, p)
    wt = wt_percent(T, rh)
    flat = c_eq_flat(T, wt)
    return dict(T=T, rh=rh, wt=wt, flat=_check('C_eq flat', flat),
                c_eq={d: _check(f'C_eq {d} nm', flat * kelvin(T, wt, d)) for d in (1.7, 3.0)})


def print_tables(rows, prof):
    def s_at(r, d):
        return np.interp(np.log(d), np.log(DP_NM), np.log(r['c_eq']))
    print('\n(a) sai_plume, S at the lowest / median hourly-mean H2SO4')
    print(f'{"scenario":>10} {"T K":>6} {"RH %":>6} {"wt %":>5} {"H2SO4 min":>10} '
          f'{"median":>10} {"S flat med":>10} {"S1.7 min":>9} {"S1.7 med":>9} {"S3 min":>9}')
    for r in rows:
        ce17, ce3 = np.exp(s_at(r, 1.7)), np.exp(s_at(r, 3.0))
        print(f'{r["name"]:>10} {r["T"]:6.1f} {r["rh"]:6.2f} {r["wt"]:5.1f} {r["h_lo"]:10.2e} '
              f'{r["h_med"]:10.2e} {r["h_med"] / r["flat"]:10.2e} {r["h_lo"] / ce17:9.2e} {r["h_med"] / ce17:9.2e} '
              f'{r["h_lo"] / ce3:9.2e}')
    print(f'\n(b) AFGL tropical, H2O {H2O_VMR_ALT * 1e6:g} ppmv: minimum S over 13-22 km')
    for d in (1.7, 3.0):
        for key in ('h_lo', 'h_med'):
            n = rows[0][key]
            S = n / prof['c_eq'][d]
            i = int(np.argmin(S))
            print(f'  {d:g} nm, H2SO4 {n:.0e}: min S = {S[i]:.2e} at {ALT_KM[i]:.1f} km '
                  f'({prof["T"][i]:.1f} K, RH {prof["rh"][i]:.2f} %, {prof["wt"][i]:.1f} wt%)')


def _reference_lines(ax, label):
    # growth with C_eq / growth with C_eq = 0 (the model) = (C - C_eq) / C = 1 - 1/S,
    # so the growth lost by treating H2SO4 as nonvolatile is 1/S = C_eq / C
    for y, text in ((1.0, 'growth stops (S = 1)'), (0.1, 'growth 10 % slower (S = 10)'),
                    (0.01, 'growth 1 % slower (S = 100)')):
        ax.axhline(y, color=INK2 if y == 1.0 else MUTED, lw=1.2 if y == 1.0 else 0.8,
                   ls='-' if y == 1.0 else '--', zorder=1)
        if label:
            ax.annotate(text, (1.0, y), xycoords=('axes fraction', 'data'),
                        xytext=(-4, -3), textcoords='offset points', ha='right',
                        va='top', fontsize=8.5, color=INK2)
    ax.axhspan(1.0, 1e3, color=GRID, alpha=0.7, lw=0, zorder=0)


def plot(rows, prof, fname):
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(13, 5.8), facecolor=SURFACE,
                                     layout='constrained')
    for ax in (ax_a, ax_b):
        ax.set_facecolor(SURFACE)
        ax.grid(color=GRID, lw=0.6)
        ax.set_yscale('log')
        ax.set_ylim(1e-18, 3.0)
        for s in ax.spines.values():
            s.set_color('#c3c2b7')
        ax.tick_params(colors=INK2)
        ax.set_ylabel('Growth lost to the vapour pressure: 1/S = C$_{eq}$(D$_p$) / C$_{H_2SO_4}$',
                      color=INK)

    _reference_lines(ax_a, label=True)
    for r in rows:
        col = SCEN_COLOR[r['name']]
        ax_a.fill_between(DP_NM, r['c_eq'] / r['h_hi'], r['c_eq'] / r['h_lo'],
                          color=col, alpha=0.3, lw=0)
        ax_a.plot(DP_NM, r['c_eq'] / r['h_med'], color=col, lw=2,
                  label=f'{SCEN_LABEL[r["name"]]}, {r["wt"]:.0f} wt%')
        ax_a.axhline(r['flat'] / r['h_med'], color=col, lw=1.4, ls=':')
    ax_a.plot([], [], color=INK2, lw=2, label='droplet with Kelvin term')
    ax_a.plot([], [], color=INK2, lw=1.4, ls=':', label='flat surface (no Kelvin), median H$_2$SO$_4$')
    ax_a.axvline(1.7, color=MUTED, lw=0.8, ls=':')
    ax_a.annotate('1.7 nm\nnucleated\ncluster', (1.7, 1e-7), xytext=(4, 0),
                  textcoords='offset points', fontsize=8.5, color=INK2, va='top')
    ax_a.set_xscale('log')
    ax_a.set_xlim(DP_NM[0], DP_NM[-1])
    ax_a.set_xticks([1.5, 2, 3, 5, 10, 20, 50, 100])
    ax_a.set_xticklabels(['1.5', '2', '3', '5', '10', '20', '50', '100'])
    ax_a.set_xlabel('Wet particle diameter D$_p$ [nm]', color=INK)
    ax_a.set_title('(a) sai_plume: line = median, band = range of hourly-mean '
                   'H$_2$SO$_4$ over 48 h', color=INK, loc='left', fontsize=10)
    ax_a.legend(loc='lower right', frameon=False, fontsize=8.5)
    ax_a.set_ylim(1e-9, 3.0)

    _reference_lines(ax_b, label=True)
    for key in ('h_lo',):
        n = rows[0][key]
        name = f'sai_plume {LEVEL_LABEL[key]} {n:.1e} cm$^{{-3}}$'
        for d, ls in ((1.7, '-'), (3.0, '--')):
            ax_b.plot(ALT_KM, prof['c_eq'][d] / n, color=LEVEL_COLOR[key], lw=2, ls=ls,
                      label=f'{name}, {d:g} nm')
        ax_b.plot(ALT_KM, prof['flat'] / n, color=LEVEL_COLOR[key], lw=1.4, ls=':',
                  label=f'{name}, flat surface')
    top = ax_b.secondary_xaxis('top')
    zt = [z for z in sorted(TROPICAL_AFGL) if ALT_KM[0] <= z <= ALT_KM[-1]] + [ALT_KM[-1]]
    top.set_xticks(zt)
    top.set_xticklabels([f'{afgl(z)[0]:.0f} K' for z in zt], fontsize=8, color=INK2)
    top.tick_params(colors=MUTED)
    ax_b.set_xlim(ALT_KM[0], ALT_KM[-1])
    ax_b.set_xlabel('Altitude [km] (AFGL tropical; temperature on the top axis)', color=INK)
    ax_b.set_title(f'(b) 1.7 nm (solid), 3 nm (dashed), flat (dotted) vs altitude, H$_2$O '
                   f'{H2O_VMR_ALT * 1e6:g} ppmv', color=INK, loc='left', fontsize=10)
    handles, labels = ax_b.get_legend_handles_labels()
    ax_b.legend(handles, labels, loc='lower left', frameon=False, fontsize=8.5)
    ax_b.set_ylim(1e-19, 3.0)   # room for the legend beside the 17 km minimum

    fig.suptitle('Growth lost by treating H$_2$SO$_4$ as nonvolatile, sulfate–water droplets with the '
                 f'Kelvin term (σ = {SIGMA} N/m): growth ∝ C − C$_{{eq}}$, so 1 − growth / nonvolatile growth = '
                 'C$_{{eq}}$ / C = 1/S\n'
                 'Ayers (1980) + Kulmala & Laaksonen (1990) pure acid, Giauque (1960) activity, '
                 'Tabazadeh (1997) composition', fontsize=10, color=INK)
    fig.savefig(fname, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    sanity()
    rows = scenario_rows()
    prof = altitude_profile()
    print_tables(rows, prof)
    os.makedirs(OUTDIR, exist_ok=True)
    fname = os.path.join(OUTDIR, 'h2so4_kelvin_driving_force.png')
    plot(rows, prof, fname)
    print(f'\nwrote {fname}')


if __name__ == '__main__':
    main()
