"""How much the vapour molecule's size (and the particle's own motion) changes
the per-particle H2SO4 condensation rate, i.e. the growth rate dDp/dt.

    python -m benchmarks.python.vapour_cross_section

TOMAS (getCondSink.f, condensation_sink.py) uses
    k = 2 pi Dp D_v beta(Kn),  Kn = 2 lambda / Dp,  lambda = 2 D_v / c_v,
    beta = (1 + Kn) / (1 + 2 Kn (1 + Kn) / alpha),
whose kinetic limit is  pi/4 Dp^2 c_v alpha  (Eq. 4 of the figure below with
D_l = 0). Three forms are compared, all on the same Fuchs-Sutugin beta and
TOMAS's lambda = 2 D / c so that only the size/motion terms differ:

  tomas      : as above.
  cross      : collision diameter Dp -> Dp + d_v in the prefactor and in Kn
               (Eq. 4 as written: phi = N pi/4 (Dp + D_l)^2 alpha v beta dC).
  full       : Lehtinen & Kulmala (2003) / Nieminen et al. (2010):
               cross, plus D_v -> D_v + D_p (particle Brownian diffusivity,
               Stokes-Einstein with Cunningham slip) and c_v -> sqrt(c_v^2 +
               c_p^2) (particle thermal speed); kinetic limit
               pi/4 (Dp + d_v)^2 sqrt(c_v^2 + c_p^2) alpha.

The ratio k_form / k_tomas is the ratio of the growth rates dDp/dt of one
particle at the same vapour concentration (and of each size's contribution to
the condensation sink). d_v from the molecular volume, (6 M / (pi rho N_A))^(1/3):
bare H2SO4 (1830 kg/m3) = 0.55 nm; H2SO4(H2O)2 at 1600 kg/m3 = 0.64 nm as a
hydrated upper case. Dp is the wet diameter, particle density 1770 kg/m3.
"""
import os

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from tomas_jax.core.config import AVOGADRO, MW_H2SO4, PI, R_GAS, SV_H2SO4
from tomas_jax.physics.gas_properties import (calc_air_viscosity,
                                              calc_fuchs_sutugin_correction,
                                              calc_gas_diffusivity,
                                              calc_mean_molecular_speed)

OUTDIR = os.path.join('benchmarks', 'results', 'vapour_cross_section')
K_B = R_GAS / AVOGADRO
MW_AIR_KG = 28.97e-3
RHO_P = 1770.0
ALPHA = 1.0
DP = np.geomspace(1.0e-9, 1.0e-6, 500)                  # wet diameter [m]
DP_TABLE_NM = (1.7, 2.0, 3.0, 5.0, 10.0, 20.0, 50.0, 100.0, 500.0)
CASES = (('stratosphere, 217 K, 55 hPa', 217.0, 5500.0),
         ('surface, 298 K, 1013 hPa', 298.0, 101325.0))


def molecular_diameter(mw_g, rho):
    return (6.0 * mw_g * 1e-3 / (PI * rho * AVOGADRO)) ** (1.0 / 3.0)


D_V_BARE = molecular_diameter(MW_H2SO4, 1830.0)                       # 0.55 nm
D_V_HYD = molecular_diameter(MW_H2SO4 + 2 * 18.015, 1600.0)           # 0.70 nm


def particle_diffusivity(dp, T, p):
    """Stokes-Einstein with Cunningham slip [m2/s]."""
    mu = calc_air_viscosity(T)
    rho_air = p * MW_AIR_KG / (R_GAS * T)
    c_air = np.sqrt(8.0 * R_GAS * T / (PI * MW_AIR_KG))
    lam_air = 2.0 * mu / (rho_air * c_air)
    kn = 2.0 * lam_air / dp
    cc = 1.0 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))
    return K_B * T * cc / (3.0 * PI * mu * dp)


def particle_speed(dp, T):
    m_p = RHO_P * PI / 6.0 * dp**3
    return np.sqrt(8.0 * K_B * T / (PI * m_p))


def rate(dp, T, p, form, d_v=D_V_BARE):
    """Per-particle collision rate k [m3/s] (flux = k (C - C_eq))."""
    D = float(calc_gas_diffusivity(T, p, MW_H2SO4, SV_H2SO4))
    c = float(calc_mean_molecular_speed(T, MW_H2SO4))
    if form == 'tomas':
        d = dp
    elif form == 'cross':
        d = dp + d_v
    elif form == 'full':
        d = dp + d_v
        D = D + particle_diffusivity(dp, T, p)
        c = np.sqrt(c**2 + particle_speed(dp, T)**2)
    else:
        raise ValueError(form)
    lam = 2.0 * D / c
    beta = np.asarray(calc_fuchs_sutugin_correction(2.0 * lam / d, ALPHA))
    return 2.0 * PI * d * D * beta


def sanity():
    # kinetic limit of 'full' must equal pi/4 (Dp+dv)^2 c_rel alpha
    T, p, dp = 217.0, 10.0, 1e-9        # 0.1 hPa: Kn >> 1
    c = float(calc_mean_molecular_speed(T, MW_H2SO4))
    kin = PI / 4 * (dp + D_V_BARE)**2 * np.sqrt(c**2 + particle_speed(dp, T)**2) * ALPHA
    r = rate(dp, T, p, 'full') / kin
    assert abs(r - 1) < 1e-3, r
    # 'tomas' must equal condensation_sink.py's per-particle term
    from tomas_jax.physics.gas_properties import calc_mean_free_path
    T, p, dp = 217.0, 5500.0, 1e-8
    mfp = float(calc_mean_free_path(T, p, MW_H2SO4, SV_H2SO4))
    D = float(calc_gas_diffusivity(T, p, MW_H2SO4, SV_H2SO4))
    ref = 2 * PI * dp * D * float(calc_fuchs_sutugin_correction(2 * mfp / dp, ALPHA))
    assert abs(rate(dp, T, p, 'tomas') / ref - 1) < 1e-12
    print(f'd_v bare {D_V_BARE * 1e9:.3f} nm, hydrated {D_V_HYD * 1e9:.3f} nm; sanity ok')


def main():
    sanity()
    os.makedirs(OUTDIR, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), sharey=True)
    for ax, (name, T, p), letter in zip(axes, CASES, 'ab'):
        k0 = rate(DP, T, p, 'tomas')
        curves = (('cross', D_V_BARE, '#3d86dc', '--', 'Dp + d$_v$ only (Eq. 4)'),
                  ('full', D_V_BARE, '#0d2f5e', '-', '+ particle diffusivity and speed'),
                  ('full', D_V_HYD, '#c0504d', ':', 'full, hydrated d$_v$ = '
                   f'{D_V_HYD * 1e9:.2f} nm'))
        print(f'\n{name}: k / k_tomas  (d_v = {D_V_BARE * 1e9:.2f} nm unless "hyd")')
        print(f'  {"Dp nm":>7} {"Kn_tomas":>9} {"cross":>7} {"full":>7} {"full hyd":>9}')
        D = float(calc_gas_diffusivity(T, p, MW_H2SO4, SV_H2SO4))
        lam = 2 * D / float(calc_mean_molecular_speed(T, MW_H2SO4))
        for dnm in DP_TABLE_NM:
            dp = dnm * 1e-9
            b = rate(dp, T, p, 'tomas')
            vals = [float(rate(dp, T, p, f, dv) / b) for f, dv in
                    (('cross', D_V_BARE), ('full', D_V_BARE), ('full', D_V_HYD))]
            print(f'  {dnm:7g} {2 * lam / dp:9.3g} {vals[0]:7.3f} {vals[1]:7.3f} {vals[2]:9.3f}')
        for form, dv, col, ls, lab in curves:
            ax.plot(DP * 1e9, rate(DP, T, p, form, dv) / k0, ls, color=col, lw=2, label=lab)
        ax.axhline(1.0, color='0.4', lw=0.8)
        ax.axvline(1.7, color='0.6', lw=0.8, ls='-.')
        ax.text(1.75, 0.98, '1.7 nm (bin 0)', transform=ax.get_xaxis_transform(),
                va='top', fontsize=8.5, color='0.4')
        ax.set_xscale('log')
        ax.set_yscale('log')
        ax.set_xlim(1, 1000)
        ax.set_xlabel('Particle (wet) diameter Dp [nm]')
        ax.set_title(f'({letter}) {name}', loc='left', fontsize=11)
        ax.grid(True, which='both', color='0.92', lw=0.6)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
    axes[0].set_ylabel('growth rate / TOMAS growth rate\n(per-particle H$_2$SO$_4$ collision rate)')
    axes[0].legend(fontsize=9, frameon=False)
    fig.suptitle('Effect of the H$_2$SO$_4$ molecule size and particle motion on condensation '
                 f'($\\alpha$ = {ALPHA:g}, Fuchs–Sutugin, d$_v$ = {D_V_BARE * 1e9:.2f} nm)',
                 fontsize=12, x=0.01, ha='left')
    fig.tight_layout()
    fname = os.path.join(OUTDIR, 'vapour_cross_section.png')
    fig.savefig(fname, dpi=150)
    print(f'\nwrote {fname}')


if __name__ == '__main__':
    main()
