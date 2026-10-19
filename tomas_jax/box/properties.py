"""Reported particle properties (numpy, on recorded states): surface and volume
distributions, the O:C of the particle organics and the hygroscopicity kappa.

Surface and volume distributions from each bin's dry mass and number (rho = RHO_REPORT,
1.4 g/cm3). The volume is the dry mass over the density, so coagulation (which adds
volumes) keeps it exactly; the surface uses the bin's mean-mass diameter Dp:

    dV/dlog10Dp = (dM/dlog10Dp) / rho = (pi/6) Dp^3 dN/dlog10Dp   [um3/cm3]
    dS/dlog10Dp = pi Dp^2 dN/dlog10Dp                             [um2/cm3]

with Dp = (6 m / (pi rho))^(1/3) and m = M / N the mean dry mass per particle in the bin
(the bin-centre diameter where a bin is empty). 1 ug/m3 at 1 g/cm3 is 1 um3/cm3.

O:C of the particle organics (ASSUMPTIONS A18): the SOA species are known from the
per-species tracker and the mechanism's atom counts, n_O = sum_i P_i O_i / MW_i and
n_C = sum_i P_i C_i / MW_i; the seed organic (slot SRTORG_NUC) has the user's O:C with
H:C = 2 - O:C (the Van Krevelen line), so its mass per mole of carbon is
12.011 + 1.008 (2 - O:C) + 15.999 O:C g/mol. O:C = (n_O,SOA + O:C_seed n_C,seed) /
(n_C,SOA + n_C,seed).

kappa (A18): kappa_org = 0.18 O:C + 0.03 (Lambe et al. 2011,
https://doi.org/10.5194/acp-11-8913-2011), kappa of ammonium sulfate 0.61 and ammonium
nitrate 0.67 (Petters and Kreidenweis 2007, https://doi.org/10.5194/acp-7-1961-2007),
mixed by dry volume: kappa = sum_j eps_j kappa_j with eps_j = (m_j / rho_j) /
sum_k (m_k / rho_k). Sulfate counts as ammonium sulfate (m_SO4 x 132/96) and nitrate
as ammonium nitrate (m_NO3 x 80/62) whether or not the particles carry ammonium, so the
ammonium slot is not counted again. Densities: 1.77, 1.72 and 1.4 g/cm3.
"""
from __future__ import annotations

import numpy as np

from ..core.config import MW_NH4, MW_NO3, MW_SO4, SRTNIT, SRTORG_NUC, SRTSO4, SRTSOA1

KAPPA_OC_SLOPE, KAPPA_OC_INTERCEPT = 0.18, 0.03   # Lambe et al. (2011)
KAPPA_AMMONIUM_SULFATE = 0.61                       # Petters and Kreidenweis (2007)
KAPPA_AMMONIUM_NITRATE = 0.67
RHO_AMMONIUM_SULFATE_G_CM3 = 1.77
RHO_AMMONIUM_NITRATE_G_CM3 = 1.72
RHO_ORGANIC_G_CM3 = 1.4
M_C, M_H, M_O = 12.011, 1.008, 15.999               # [g/mol]


N_EMPTY_CM3 = 1e-12          # bins with less number than this use the bin-centre diameter


def surface_volume_distributions(dndlog10dp_cm3: np.ndarray, dmdlog10dp_ugm3: np.ndarray,
                                 diameter_nm: np.ndarray, rho_g_cm3: float):
    """(dS/dlog10Dp [um2/cm3], dV/dlog10Dp [um3/cm3]) for [t, bin] number and dry mass
    distributions; diameter_nm is the bin centre (used only for empty bins)."""
    dn = np.asarray(dndlog10dp_cm3)
    dv = np.asarray(dmdlog10dp_ugm3) / rho_g_cm3
    full = dn > N_EMPTY_CM3
    v_mean_um3 = np.where(full, dv / np.where(full, dn, 1.0), 0.0)
    dp_um = np.where(full, (6.0 * v_mean_um3 / np.pi) ** (1.0 / 3.0),
                     np.broadcast_to(np.asarray(diameter_nm) * 1e-3, dn.shape))
    return np.pi * dp_um**2 * dn, dv


def kappa_organic(o_to_c: np.ndarray) -> np.ndarray:
    return KAPPA_OC_SLOPE * np.asarray(o_to_c) + KAPPA_OC_INTERCEPT


def organic_o_to_c(soa_mass_kg: np.ndarray, soa_o_per_mass, soa_c_per_mass,
                   seed_organic_kg: np.ndarray, seed_o_to_c: float) -> np.ndarray:
    """O:C of all particle organics at each time.

    soa_mass_kg: [t, species] per-species particle mass (tracker); soa_o_per_mass and
    soa_c_per_mass: [species] moles of O and C per kg of each species (atoms / MW);
    seed_organic_kg: [t] mass in SRTORG_NUC. Times with no organic mass give NaN.
    """
    m_per_c = (M_C + M_H * (2.0 - seed_o_to_c) + M_O * seed_o_to_c) * 1e-3     # kg per mol C
    n_c_seed = np.asarray(seed_organic_kg) / m_per_c
    n_o = np.asarray(soa_mass_kg) @ np.asarray(soa_o_per_mass) + seed_o_to_c * n_c_seed
    n_c = np.asarray(soa_mass_kg) @ np.asarray(soa_c_per_mass) + n_c_seed
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_c > 0, n_o / np.where(n_c > 0, n_c, 1.0), np.nan)


def kappa_mixture(mk_dry: np.ndarray, k_cls: int, o_to_c: np.ndarray) -> dict[str, np.ndarray]:
    """kappa of the dry particle population per time from [t, bin, comp] masses.

    Returns kappa (volume-mixed), kappa_organic, kappa_inorganic and the organic volume
    fraction. Times without dry mass give NaN.
    """
    mk = np.asarray(mk_dry)
    m_as = mk[:, :, SRTSO4].sum(axis=1) * (MW_SO4 + 2.0 * MW_NH4) / MW_SO4
    m_an = mk[:, :, SRTNIT].sum(axis=1) * (MW_NO3 + MW_NH4) / MW_NO3
    m_org = mk[:, :, SRTSOA1:SRTSOA1 + k_cls].sum(axis=(1, 2)) + mk[:, :, SRTORG_NUC].sum(axis=1)
    v_as, v_an, v_org = (m_as / RHO_AMMONIUM_SULFATE_G_CM3, m_an / RHO_AMMONIUM_NITRATE_G_CM3,
                         m_org / RHO_ORGANIC_G_CM3)
    v_inorg, v_tot = v_as + v_an, v_as + v_an + v_org
    k_org = np.where(np.isnan(o_to_c), KAPPA_OC_INTERCEPT, kappa_organic(np.nan_to_num(o_to_c)))
    with np.errstate(invalid="ignore", divide="ignore"):
        k_inorg = np.where(v_inorg > 0, (v_as * KAPPA_AMMONIUM_SULFATE + v_an * KAPPA_AMMONIUM_NITRATE)
                           / np.where(v_inorg > 0, v_inorg, 1.0), np.nan)
        kappa = np.where(v_tot > 0, (np.nan_to_num(v_inorg * k_inorg) + v_org * k_org)
                         / np.where(v_tot > 0, v_tot, 1.0), np.nan)
        eps_org = np.where(v_tot > 0, v_org / np.where(v_tot > 0, v_tot, 1.0), np.nan)
    return {"kappa": kappa, "kappa_organic": np.where(m_org > 0, k_org, np.nan),
            "kappa_inorganic": k_inorg, "organic_volume_fraction": eps_org}
