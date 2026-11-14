"""Growth lost by treating H2SO4 as nonvolatile, in the SAI plume and in
background air (low H2SO4): equilibrium H2SO4 over sulfate-water droplets,
Kelvin (curvature) term included, against the modelled H2SO4. TOMAS-JAX condenses H2SO4 with C_eq = 0 (no vapour
pressure, so no Kelvin term), as TOMAS Fortran does.

    python -m benchmarks.python.h2so4_kelvin_driving_force
    python -m benchmarks.python.h2so4_kelvin_driving_force --temp 250 --pres 5000

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

(a) 1/S vs wet diameter (1.5-100 nm) for two Phase 3 scenarios
    (sensitivities/dunne_nuc/scenarios.py), both at fixed 217 K, 5500 Pa,
    5 ppmv H2O -> RH 0.89 %, 72 wt%, so they differ only in H2SO4:
      sai_plume  10 ppmv SO2, hourly-mean H2SO4 6.0e7-1.0e8 cm-3;
      bg_midlat  0.02 ppbv SO2 (low H2SO4), 1.1e5-7.5e5 cm-3.
    H2SO4 = aux_h2so4_mean of the 48 h 40-bin runs
    (results/phase3/<name>_40bin.npz), hour 1 dropped (SKIP_HOURS: H2SO4 is
    still rising from zero; bg_midlat's hour 1 is 3.9e4). Line = median, band = lowest to highest
    hourly mean; dotted = flat surface (no Kelvin) at the median H2SO4, which
    the droplet curve approaches at large Dp. The min / median / max are
    written on the figure. Top axis: dry diameter (SO4 at DENS_INIT = 1770
    kg/m3, the grid's density, so it is the model's bin size; dry_over_wet).
    The nucleated cluster is 1.7 nm dry = bin 0 edge (1.96 nm wet at 72 wt%).
    RH matters beyond the wet size: it sets wt %, hence the H2SO4 activity
    (the dominant term in C_eq) and the density in the Kelvin factor.
    Panel (b) sizes are wet.
(b) 1/S at 1.7 nm (solid), 3 nm (dashed) and for a flat surface (dotted) vs
    altitude 13-22 km, AFGL tropical (Anderson et al. 1986, AFGL-TR-86-0110;
    AFGL_TROPICAL, 1 km levels: linear T and H2O, log p), with its H2O
    profile, so RH and the droplet composition follow altitude: 9.9 ppmv at
    13 km, 2.9 at 17 km, 2.6-2.8 at 19-22 km (the scenarios use 5 ppmv).
    T and wt% on the top axis. Each run's lowest hourly-mean H2SO4, reused at
    every altitude (not re-run).
(c) 1/S vs gas-phase H2SO4 (1e4-1e10 cm-3) at 13, 15, 17, 19, 22 km (Z_C,
    same profile as (b)): 1.7 nm wet (solid) and flat (dotted); 1/S =
    C_eq / C, so each is a line of slope -1 and growth stops where it crosses
    1/S = 1 (C = C_eq). Shaded columns: the runs' hourly-mean H2SO4 ranges.

--temp/--pres: panel (a) alone at prescribed T and p, same H2SO4 series (from
the 217 K runs, so only indicative: at another T the run's H2SO4 would
differ), the scenarios' 5 ppmv H2O (RH follows). Where the RH is below the
Tabazadeh table, tabazadeh_h2so4_wt clamps at 80 wt% and C_eq is a lower
bound; a dash-dot line then shows 90 wt%. 1/S > 1 means net evaporation.
Output h2so4_kelvin_driving_force_<T>K_<p>Pa.png.

Result (2026-11-13), 217 K: sai_plume at most 6e-4 at 1.7 nm (lowest hour),
~2e-5 at 3 nm, ~5e-7 flat. Altitude (b), AFGL tropical with its H2O, each
run's lowest hourly H2SO4: the largest 1/S over 13-22 km is at 22 km (214.6
K, 2.8 ppmv, RH 0.5 %, 74 wt%): bg_midlat 0.28 at 1.7 nm, 0.019 at 3 nm;
sai_plume 5e-4 at 1.7 nm. C_eq at 1.7 nm is 250 cm-3 at 13 km (wetter,
63 wt%), 1e-3 at the 17 km cold point and 3e4 at 22 km, so the margin
shrinks fast above ~20 km. bg_midlat 0.03 at 1.7 nm (median hour), 0.2 at its lowest
hour, < 0.013 at 3 nm. 250 K / 50 hPa (RH 0.026 %, >= 80 wt%): background
droplets evaporate at every size (flat S = 0.27); in the plume particles
below ~2.6 nm evaporate and 3 nm growth is 60-80 % slower (80 wt%); at 90 wt%
every plume size is at or below S = 1. The bulk Kelvin formula is rough at 1.7 nm,
where the Dunne nucleation rate already sets cluster survival.

Output: benchmarks/results/h2so4_kelvin/h2so4_kelvin_driving_force.png and a
printed table (S, i.e. the inverse of the plotted quantity).
"""
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from tomas_jax.core.config import MW_H2SO4, MW_SO4
from tomas_jax.physics.water_equilibrium import tabazadeh_h2so4_wt
from tomas_jax.physics.radiative_forcing import h2so4_solution_density
from sensitivities.dunne_nuc.common import TROPICAL_AFGL, DENS_INIT
from sensitivities.dunne_nuc.scenarios import SCENARIOS, rh_from_h2o

OUTDIR = os.path.join('benchmarks', 'results', 'h2so4_kelvin')
PHASE3 = os.path.join('sensitivities', 'dunne_nuc', 'results', 'phase3')
R_GAS = 8.3143          # J/mol/K (as in CARMA)
K_B = 1.380649e-23      # J/K
ATM = 101325.0          # Pa
SIGMA = 0.08            # N/m
DP_NM = np.geomspace(1.5, 100.0, 240)
ALT_KM = np.linspace(13.0, 22.0, 181)
# AFGL tropical (Anderson et al. 1986, AFGL-TR-86-0110), 13-25 km:
# z [km], p [hPa], T [K], H2O [ppmv]. Same T and p as common.TROPICAL_AFGL at
# its levels (checked in afgl()); this adds the 1 km levels and the H2O.
AFGL_TROPICAL = np.array([
    (13.0, 182.0, 217.0, 9.90), (14.0, 156.0, 210.3, 6.22), (15.0, 132.0, 203.7, 4.00),
    (16.0, 111.0, 197.0, 3.00), (17.0, 93.7, 194.8, 2.90), (18.0, 78.9, 198.8, 2.75),
    (19.0, 66.6, 202.7, 2.60), (20.0, 56.5, 206.7, 2.60), (21.0, 48.0, 210.7, 2.65),
    (22.0, 40.9, 214.6, 2.80), (23.0, 35.0, 217.0, 2.90), (24.0, 30.0, 219.2, 3.20),
    (25.0, 25.7, 221.4, 3.25)])
SCEN_ORDER = ('bg_midlat', 'sai_plume')
SKIP_HOURS = 1          # hour 1: H2SO4 still rising from zero (spin-up)
WT_TABLE_MAX = 80.0     # top of the Tabazadeh Table 1; tabazadeh_h2so4_wt clamps here
WT_SENS = 90.0          # composition for the sensitivity line when the table clamps

INK, INK2, MUTED, GRID, SURFACE = '#1f1f1e', '#5f5e58', '#898781', '#e4e3df', '#fcfcfb'
SCEN_COLOR = {'bg_midlat': '#2a78d6', 'sai_plume': '#eb6834'}
SCEN_LABEL = {'bg_midlat': 'bg_midlat (0.02 ppbv SO$_2$)',
              'sai_plume': 'sai_plume (10 ppmv SO$_2$)'}


def afgl(z_km):
    """T [K], p [Pa], H2O mixing ratio at z: AFGL tropical, linear T and H2O,
    log p between the 1 km levels."""
    zs, p, T, h2o = AFGL_TROPICAL.T
    for z, (Tc, pc) in TROPICAL_AFGL.items():
        k = int(np.argmin(np.abs(zs - z)))
        if zs[k] == z and (abs(T[k] - Tc) > 0.05 or abs(p[k] * 100 / pc - 1) > 0.01):
            raise ValueError(f'AFGL_TROPICAL disagrees with common.TROPICAL_AFGL at {z} km')
    z_km = np.asarray(z_km, dtype=np.float64)
    if np.any(z_km < zs[0]) or np.any(z_km > zs[-1]):
        raise ValueError(f'altitude outside AFGL_TROPICAL ({zs[0]:g}-{zs[-1]:g} km)')
    return (np.interp(z_km, zs, T), np.exp(np.interp(z_km, zs, np.log(p * 100.0))),
            np.interp(z_km, zs, h2o) * 1e-6)


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


def dry_over_wet(wt):
    """Dry / wet diameter of a sulfate-water droplet. Dry = the SO4 mass alone at
    DENS_INIT (1770 kg/m3), the density that defines the grid (XK0 = 1.7 nm), so a
    dry diameter is the model's bin size. Wet = the H2SO4 (98/96 x SO4) solution at
    wt % and the solution density."""
    rho_sol = float(h2so4_solution_density(wt))
    return (rho_sol * wt / 100.0 / (DENS_INIT * MW_H2SO4 / MW_SO4)) ** (1.0 / 3.0)


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


def scenario_rows(temp=None, pres=None, wt=None):
    """Per scenario: modelled H2SO4 stats and C_eq(Dp). temp/pres replace the
    scenario's own (its H2O mixing ratio is kept, so RH follows); wt replaces
    the Tabazadeh composition. Hour 1 is dropped (H2SO4 still rising from 0)."""
    rows = []
    for name in SCEN_ORDER:
        fname = os.path.join(PHASE3, f'{name}_40bin.npz')
        if not os.path.exists(fname):
            raise FileNotFoundError(f'Missing NPZ: {fname}. Run phase3_scenarios first.')
        h = _check(f'{name} aux_h2so4_mean', np.load(fname)['aux_h2so4_mean'])[SKIP_HOURS:]
        sc = SCENARIOS[name]
        T = sc['cond']['temp'] if temp is None else temp
        p = sc['cond']['pres'] if pres is None else pres
        rh = 100.0 * rh_from_h2o(sc['h2o_vmr'], T, p)
        wt_tab = float(wt_percent(T, rh))
        w = wt_tab if wt is None else wt
        flat = float(_check(f'{name} C_eq flat', c_eq_flat(T, w)))
        c_eq = _check(f'{name} C_eq', flat * kelvin(T, w, DP_NM))
        rows.append(dict(name=name, T=T, p=p, rh=rh, wt=w, wt_capped=wt_tab >= WT_TABLE_MAX,
                         h_lo=h.min(), h_hi=h.max(), h_med=np.median(h),
                         hour_lo=int(np.argmin(h)) + 1 + SKIP_HOURS,
                         hour_hi=int(np.argmax(h)) + 1 + SKIP_HOURS,
                         c_eq=c_eq, flat=flat))
    return rows


def altitude_profile():
    T, p, h2o = afgl(ALT_KM)
    rh = 100.0 * rh_from_h2o(h2o, T, p)
    wt = wt_percent(T, rh)
    capped = wt >= WT_TABLE_MAX
    wt_hi = np.where(capped, WT_SENS, wt)
    flat = c_eq_flat(T, wt)
    flat_hi = c_eq_flat(T, wt_hi)
    return dict(T=T, p=p, h2o=h2o, rh=rh, wt=wt, capped=capped, flat=_check('C_eq flat', flat),
                flat_hi=_check('C_eq flat 90', flat_hi),
                c_eq={d: _check(f'C_eq {d} nm', flat * kelvin(T, wt, d)) for d in (1.7, 3.0)},
                c_eq_hi={d: _check(f'C_eq {d} nm 90', flat_hi * kelvin(T, wt_hi, d))
                         for d in (1.7, 3.0)})


def print_tables(rows, prof):
    def s_at(r, d):
        return np.interp(np.log(d), np.log(DP_NM), np.log(r['c_eq']))
    print(f'\n(a) Phase 3 scenarios at {rows[0]["T"]:g} K, {rows[0]["p"]:g} Pa, S at the lowest / '
          f'median hourly-mean H2SO4 (hours {SKIP_HOURS + 1}-48)')
    print(f'{"scenario":>10} {"T K":>6} {"RH %":>6} {"wt %":>5} {"H2SO4 min":>10} '
          f'{"median":>10} {"S flat med":>10} {"S1.7 min":>9} {"S1.7 med":>9} {"S3 min":>9}')
    for r in rows:
        ce17, ce3 = np.exp(s_at(r, 1.7)), np.exp(s_at(r, 3.0))
        print(f'{r["name"]:>10} {r["T"]:6.1f} {r["rh"]:6.2f} {r["wt"]:5.1f} {r["h_lo"]:10.2e} '
              f'{r["h_med"]:10.2e} {r["h_med"] / r["flat"]:10.2e} {r["h_lo"] / ce17:9.2e} {r["h_med"] / ce17:9.2e} '
              f'{r["h_lo"] / ce3:9.2e}')
    if prof is None:
        return
    print(f'\n(b) AFGL tropical with its H2O, each run\'s lowest '
          f'hourly-mean H2SO4: lowest altitude where 1/S reaches 0.1 / 1 (80 wt%; 90 wt% in [])')

    def first(y, lev):
        i = np.nonzero(y >= lev)[0]
        return f'{ALT_KM[i[0]]:5.1f}' if i.size else f'  >{ALT_KM[-1]:g}'
    for r in rows:
        n = r['h_lo']
        for key, lab in ((1.7, '1.7 nm'), (3.0, '3 nm'), (None, 'flat')):
            lo = (prof['c_eq'][key] if key else prof['flat']) / n
            hi = (prof['c_eq_hi'][key] if key else prof['flat_hi']) / n
            print(f'  {r["name"]:>10} {n:.1e} {lab:>6}: 0.1 at {first(lo, 0.1)} [{first(hi, 0.1)}] km, '
                  f'1 at {first(lo, 1.0)} [{first(hi, 1.0)}] km; max 1/S {lo.max():.2e} '
                  f'at {ALT_KM[int(np.argmax(lo))]:.1f} km')
    print('   z km     T K  H2O ppmv    RH %  wt %   C_eq 1.7 nm   C_eq flat')
    for z in (13.0, 15.0, 17.0, 19.0, 20.0, 22.0):
        k = int(np.argmin(np.abs(ALT_KM - z)))
        print(f'  {z:5.1f} {prof["T"][k]:7.1f} {prof["h2o"][k] * 1e6:9.2f} {prof["rh"][k]:7.3f} '
              f'{prof["wt"][k]:5.1f} {prof["c_eq"][1.7][k]:13.2e} {prof["flat"][k]:11.2e}')
    i = np.nonzero(prof['capped'])[0]
    if i.size:
        print(f'  composition clamped at {WT_TABLE_MAX:g} wt% above {ALT_KM[i[0]]:.1f} km '
              f'({prof["T"][i[0]]:.1f} K, RH {prof["rh"][i[0]]:.3f} %)')


def _reference_lines(ax, label, label_x=1.0, above=False):
    # growth with C_eq / growth with C_eq = 0 (the model) = (C - C_eq) / C = 1 - 1/S,
    # so the growth lost by treating H2SO4 as nonvolatile is 1/S = C_eq / C
    for y, text in ((1.0, 'growth stops (S = 1)'), (0.1, 'growth 10 % slower (S = 10)'),
                    (0.01, 'growth 1 % slower (S = 100)')):
        ax.axhline(y, color=INK2 if y == 1.0 else MUTED, lw=1.2 if y == 1.0 else 0.8,
                   ls='-' if y == 1.0 else '--', zorder=1)
        if label:
            ax.annotate(text, (label_x, y), xycoords=('axes fraction', 'data'),
                        xytext=(-4, 3 if above else -3), textcoords='offset points', ha='right',
                        va='bottom' if above else 'top', fontsize=8.5, color=INK2)
    ax.axhspan(1.0, 1e8, color=GRID, alpha=0.7, lw=0, zorder=0)


def _style(ax, ylim):
    ax.set_facecolor(SURFACE)
    ax.grid(color=GRID, lw=0.6)
    ax.set_yscale('log')
    ax.set_ylim(*ylim)
    for sp in ax.spines.values():
        sp.set_color('#c3c2b7')
    ax.tick_params(colors=INK2)
    ax.set_ylabel('Growth lost to the vapour pressure: 1/S = C$_{eq}$(D$_p$) / C$_{H_2SO_4}$',
                  color=INK)


def _panel_a(ax, rows, title, ylim, stats_xy, legend_loc, sens_rows=None):
    _style(ax, ylim)
    _reference_lines(ax, label=True)
    for r in rows:
        col = SCEN_COLOR[r['name']]
        ax.fill_between(DP_NM, r['c_eq'] / r['h_hi'], r['c_eq'] / r['h_lo'],
                        color=col, alpha=0.3, lw=0)
        ax.plot(DP_NM, r['c_eq'] / r['h_med'], color=col, lw=2, label=SCEN_LABEL[r['name']])
        ax.axhline(r['flat'] / r['h_med'], color=col, lw=1.4, ls=':')
    for r in sens_rows or ():
        ax.plot(DP_NM, r['c_eq'] / r['h_med'], color=SCEN_COLOR[r['name']], lw=1.4, ls='-.')
    ax.plot([], [], color=INK2, lw=2, label=f'droplet with Kelvin term, {rows[0]["wt"]:.0f} wt%')
    ax.plot([], [], color=INK2, lw=1.4, ls=':', label='flat surface (no Kelvin), median H$_2$SO$_4$')
    if sens_rows:
        ax.plot([], [], color=INK2, lw=1.4, ls='-.',
                label=f'droplet with Kelvin term, {sens_rows[0]["wt"]:.0f} wt%, median')
    fdw = dry_over_wet(rows[0]['wt'])
    d_nuc = 1.7 / fdw   # nucleated cluster = bin 0 edge, 1.7 nm dry
    ax.axvline(d_nuc, color=MUTED, lw=0.8, ls=':')
    ax.annotate(f'1.7 nm dry\n({d_nuc:.2f} nm wet)\nnucleated\ncluster', (d_nuc, 0.04),
                xycoords=('data', 'axes fraction'),
                xytext=(4, 0), textcoords='offset points', fontsize=8.5, color=INK2, va='bottom')
    ax.set_xscale('log')
    ax.set_xlim(DP_NM[0], DP_NM[-1])
    ax.set_xticks([1.5, 2, 3, 5, 10, 20, 50, 100])
    ax.set_xticklabels(['1.5', '2', '3', '5', '10', '20', '50', '100'])
    ax.set_xlabel(f'Wet particle diameter D$_p$ [nm] ({rows[0]["wt"]:.0f} wt% H$_2$SO$_4$ droplet; '
                  'the Kelvin term uses this)', color=INK)
    top = ax.secondary_xaxis('top', functions=(lambda d: d * fdw, lambda d: d / fdw))
    top.set_xticks([1.7, 2, 3, 5, 10, 20, 50])
    top.set_xticklabels(['1.7', '2', '3', '5', '10', '20', '50'], fontsize=8, color=INK2)
    top.tick_params(colors=MUTED)
    top.set_xlabel(f'Dry diameter [nm] (SO$_4$ at {DENS_INIT:.0f} kg m$^{{-3}}$, the model grid; '
                   f'dry/wet = {fdw:.3f})', color=INK2, fontsize=9)
    ax.set_title(title, color=INK, loc='left', fontsize=10)
    ax.legend(loc=legend_loc, frameon=False, fontsize=8.5)
    stats = [f'Hourly-mean H$_2$SO$_4$ [cm$^{{-3}}$], 48 h run, hours {SKIP_HOURS + 1}–48:']
    for r in rows:
        stats.append(f'{r["name"]}: min {r["h_lo"]:.1e} (hour {r["hour_lo"]}), '
                     f'median {r["h_med"]:.1e}, max {r["h_hi"]:.1e} (hour {r["hour_hi"]})')
    ax.text(*stats_xy, '\n'.join(stats), transform=ax.transAxes, ha='left', va='bottom',
            fontsize=8, color=INK2, linespacing=1.5)


SUPTITLE = ('Growth lost by treating H$_2$SO$_4$ as nonvolatile, sulfate–water droplets with the '
            f'Kelvin term (σ = {SIGMA} N/m): growth ∝ C − C$_{{eq}}$, so 1 − growth / nonvolatile '
            'growth = C$_{{eq}}$ / C = 1/S\n'
            'Ayers (1980) + Kulmala & Laaksonen (1990) pure acid, Giauque (1960) activity, '
            'Tabazadeh (1997) composition')


Z_C = (13.0, 15.0, 17.0, 19.0, 22.0)
# categorical: five hues kept apart from the scenario blue / orange; checked with the
# dataviz palette validator (CVD and normal-vision separation, contrast on white)
Z_COLOR = ('#13864a', '#e05a9e', '#b08900', '#5a3fc0', '#8a2f12')
# (worst colour-blind pair 13/17 km is in the 6-8 dE band: lines are also labelled)
YLIM_BC = (1e-24, 10.0)   # shared by panels (b) and (c)
C_AXIS = np.geomspace(1e4, 1e10, 200)


def _panel_c(ax, rows):
    """1/S vs gas-phase H2SO4 at several altitudes (1.7 nm wet and flat)."""
    _style(ax, YLIM_BC)
    _reference_lines(ax, label=False)
    for r in rows:
        ax.axvspan(r['h_lo'], r['h_hi'], color=SCEN_COLOR[r['name']], alpha=0.18, lw=0, zorder=0)
        ax.text(np.sqrt(r['h_lo'] * r['h_hi']), 0.985, r['name'], transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=8.5, color=INK2, rotation=90)
    T, p, h2o = afgl(np.array(Z_C))
    rh = 100.0 * rh_from_h2o(h2o, T, p)
    wt = wt_percent(T, rh)
    print('\n(c) C_eq [cm-3] at the panel (c) altitudes (80 wt% where clamped; 90 wt% in [])')
    for z, t, w, col in zip(Z_C, T, wt, Z_COLOR):
        flat = float(c_eq_flat(t, w))
        ce = flat * float(kelvin(t, w, 1.7))
        ax.plot(C_AXIS, ce / C_AXIS, color=col, lw=2.2,
                label=f'{z:g} km ({t:.0f} K, {w:.0f} wt%{"+" if w >= WT_TABLE_MAX else ""})')
        ax.plot(C_AXIS, flat / C_AXIS, color=col, lw=1.3, ls=':')
        x_lab = 2e9   # direct label near the right end of the solid line
        ax.annotate(f'{z:g} km', (x_lab, ce / x_lab), xytext=(0, 4), textcoords='offset points',
                    ha='center', va='bottom', fontsize=8.5, color=INK, rotation=-11,
                    rotation_mode='anchor')
        extra = ''
        if w >= WT_TABLE_MAX:
            ce_hi = float(c_eq_flat(t, WT_SENS) * kelvin(t, WT_SENS, 1.7))
            ax.plot(C_AXIS, ce_hi / C_AXIS, color=col, lw=1.2, ls='-.')
            extra = f' [{ce_hi:.2e}]'
        print(f'  {z:4.0f} km {t:6.1f} K {w:5.1f} wt%: 1.7 nm {ce:.2e}{extra}, flat {flat:.2e}')
    ax.set_xscale('log')
    ax.set_xlim(C_AXIS[0], C_AXIS[-1])
    ax.set_xlabel('Gas-phase H$_2$SO$_4$ [molec cm$^{-3}$]', color=INK)
    ax.set_title('(c) vs H$_2$SO$_4$ at fixed altitudes: 1.7 nm wet (solid), flat (dotted),\n'
                 'AFGL tropical T, p and H$_2$O', color=INK, loc='left', fontsize=10)
    h, l = ax.get_legend_handles_labels()
    from matplotlib.lines import Line2D
    if np.any(wt >= WT_TABLE_MAX):
        h.append(Line2D([], [], color=INK2, lw=1.2, ls='-.'))
        l.append(f'1.7 nm at {WT_SENS:g} wt% (wt%+: table limit, solid = {WT_TABLE_MAX:g} wt%)')
    h += [Line2D([], [], color=INK2, lw=1.2), Line2D([], [], color=MUTED, lw=0.8, ls='--')]
    l += ['S = 1 (shaded above: evaporation)', 'S = 10, 100']
    ax.legend(h, l, loc='lower left', frameon=False, fontsize=8.5)


def plot(rows, prof, fname):
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(19.5, 6.2), facecolor=SURFACE,
                                           layout='constrained')
    _panel_a(ax_a, rows, f'(a) 217 K, 5.5 kPa, {rows[0]["wt"]:.0f} wt%: line = median, '
             'band = hourly-mean range', (1e-13, 3.0), (0.12, 0.30), 'lower right')

    _style(ax_b, YLIM_BC)   # room for the legend below the 17 km minimum
    _reference_lines(ax_b, label=False)   # 25 decades: labels would collide; in the legend
    cap = prof['capped']
    z_cap = ALT_KM[np.nonzero(cap)[0][0]] if cap.any() else None
    for r in rows:
        n, col = r['h_lo'], SCEN_COLOR[r['name']]
        for d, ls in ((1.7, '-'), (3.0, '--')):
            ax_b.plot(ALT_KM, prof['c_eq'][d] / n, color=col, lw=2, ls=ls)
            ax_b.fill_between(ALT_KM, prof['c_eq'][d] / n, prof['c_eq_hi'][d] / n, where=cap,
                              color=col, alpha=0.15, lw=0)
        ax_b.plot(ALT_KM, prof['flat'] / n, color=col, lw=1.4, ls=':')
        ax_b.fill_between(ALT_KM, prof['flat'] / n, prof['flat_hi'] / n, where=cap,
                          color=col, alpha=0.15, lw=0)
        ax_b.plot([], [], color=col, lw=2,
                  label=f'{r["name"]}, lowest hourly mean {n:.1e} cm$^{{-3}}$')
    if z_cap is not None:
        ax_b.axvline(z_cap, color=MUTED, lw=0.8, ls=':')
        ax_b.annotate(f'RH below the Tabazadeh table above {z_cap:.0f} km:\n'
                      f'line = {WT_TABLE_MAX:g} wt% (lower bound), band to {WT_SENS:g} wt%',
                      (z_cap, 1e-23), xytext=(4, 0), textcoords='offset points',
                      fontsize=8.5, color=INK2, va='bottom')
    for d, ls in ((1.7, '-'), (3.0, '--')):
        ax_b.plot([], [], color=INK2, lw=2, ls=ls, label=f'{d:g} nm wet')
    ax_b.plot([], [], color=INK2, lw=1.4, ls=':', label='flat surface')
    ax_b.plot([], [], color=INK2, lw=1.2, label='S = 1 (growth stops; shaded: evaporation)')
    ax_b.plot([], [], color=MUTED, lw=0.8, ls='--', label='S = 10, 100 (10 %, 1 % slower)')
    top = ax_b.secondary_xaxis('top')
    zt = np.array([13.0, 15.0, 17.0, 19.0, 21.0, 22.0])
    T_t, p_t, h_t = afgl(zt)
    wt_t = wt_percent(T_t, 100.0 * rh_from_h2o(h_t, T_t, p_t))
    top.set_xticks(zt)
    top.set_xticklabels([f'{t:.0f} K\n{w:.0f} wt%' for t, w in zip(T_t, wt_t)], fontsize=8,
                        color=INK2)
    top.tick_params(colors=MUTED)
    ax_b.set_xlim(ALT_KM[0], ALT_KM[-1])
    ax_b.set_xlabel('Altitude [km] (AFGL tropical T, p and H$_2$O; T and droplet wt% '
                    'on the top axis)', color=INK)
    ax_b.set_title('(b) wet 1.7 nm (solid), 3 nm (dashed), flat (dotted) vs altitude,\n'
                   'AFGL tropical H$_2$O (9.9 ppmv at 13 km, 2.6–2.9 ppmv at 17–22 km)',
                   color=INK, loc='left', fontsize=10)
    ax_b.legend(loc='lower left', frameon=False, fontsize=8.5)
    _panel_c(ax_c, rows)

    fig.suptitle(SUPTITLE, fontsize=10, color=INK)
    fig.savefig(fname, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_conditions(rows, sens_rows, fname):
    """Panel (a) alone at prescribed T and p (same modelled H2SO4 series)."""
    r0 = rows[0]
    fig, ax = plt.subplots(figsize=(11.5, 6.6), facecolor=SURFACE, layout='constrained')
    cap = (f'RH {r0["rh"]:.3f} %, beyond the Tabazadeh table: {r0["wt"]:.0f} wt% (its top, '
           'C$_{eq}$ a lower bound)' if r0['wt_capped'] else f'{r0["wt"]:.0f} wt%')
    _panel_a(ax, rows, f'{r0["T"]:g} K, {r0["p"] / 100:g} hPa, H$_2$O 5 ppmv; {cap}\n'
             'line = median, band = hourly-mean range; H$_2$SO$_4$ from the 217 K runs',
             (1e-3, 1e6), (0.22, 0.02), 'upper right', sens_rows=sens_rows)
    fig.suptitle(SUPTITLE, fontsize=9, color=INK)
    fig.savefig(fname, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def main():
    import argparse
    ap = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    ap.add_argument('--temp', type=float, help='temperature [K] for a single-panel version')
    ap.add_argument('--pres', type=float, help='pressure [Pa] for a single-panel version')
    args = ap.parse_args()
    sanity()
    os.makedirs(OUTDIR, exist_ok=True)
    if args.temp is None and args.pres is None:
        rows = scenario_rows()
        prof = altitude_profile()
        print_tables(rows, prof)
        fname = os.path.join(OUTDIR, 'h2so4_kelvin_driving_force.png')
        plot(rows, prof, fname)
    else:
        if args.temp is None or args.pres is None:
            raise SystemExit('--temp and --pres go together')
        rows = scenario_rows(args.temp, args.pres)
        print_tables(rows, None)
        sens_rows = None
        if rows[0]['wt_capped']:
            print(f'warning: RH {rows[0]["rh"]:.4f} % is below the Tabazadeh table '
                  f'(clamped at {WT_TABLE_MAX:g} wt%); C_eq is a lower bound. '
                  f'Sensitivity line at {WT_SENS:g} wt%.')
            sens_rows = scenario_rows(args.temp, args.pres, wt=WT_SENS)
            print_tables(sens_rows, None)
        fname = os.path.join(OUTDIR, f'h2so4_kelvin_driving_force_{args.temp:g}K_'
                                     f'{args.pres:g}Pa.png')
        plot_conditions(rows, sens_rows, fname)
    print(f'\nwrote {fname}')


if __name__ == '__main__':
    main()
