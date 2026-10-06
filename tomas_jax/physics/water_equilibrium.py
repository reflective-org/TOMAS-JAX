"""Water equilibrium calculations for TOMAS-JAX.

Exact port of ezwatereqm.f, waterso4.f, and waternacl.f.

Calculates equilibrium water uptake by aerosol particles based on
relative humidity. Two schemes, chosen with ``equilibrate_water``:

  - ``WATER_SCHEME_BISULFATE`` (default, ``calc_equilibrium_water``):
    TOMAS ezwatereqm -- piecewise polynomial fits to ISORROPIA results for
    ammonium bisulfate at 273 K.
  - ``WATER_SCHEME_H2SO4`` (``calc_equilibrium_water_h2so4``): binary
    H2SO4/H2O equilibrium of Tabazadeh et al. (1997), for pure sulfuric
    acid aerosol (no ammonia).

Assumptions (bisulfate scheme):
    - Sulfate particles treated as ammonium bisulfate (NH4HSO4),
      converted from SO4 mass with a factor of 1.2.
    - Organic aerosol assumed to have the same water uptake as sulfate.
    - Sea salt water uptake is computed but currently disabled in TOMAS.
    - Instantaneous equilibrium (no kinetic limitations).

References:
    - ezwatereqm.f (Peter Adams, March 2000)
    - waterso4.f (Peter Adams, November 2001) - ammonium bisulfate fits
    - waternacl.f (Peter Adams, November 2001) - sea salt fits
    - ISORROPIA thermodynamic model (Nenes et al., 1998)
    - Tabazadeh, Toon, Clegg, and Hamill (1997), GRL, 24(15), 1931-1934
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp
import numpy as np
from typing import Union

from ..core.config import SRTSO4, SRTORG1, SRTH2O, IORG, MW_H2SO4, MW_SO4

WATER_SCHEME_BISULFATE = 0.0   # TOMAS ammonium-bisulfate fit (ezwatereqm)
WATER_SCHEME_H2SO4 = 1.0       # Tabazadeh (1997) binary H2SO4/H2O


def water_uptake_sulfate(rh_percent: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """Wet/dry mass ratio for ammonium bisulfate aerosol (waterso4.f).

    Piecewise polynomial fit based on ISORROPIA results at 273 K.

    Args:
        rh_percent: Relative humidity [%], range 0-100

    Returns:
        wr: Wet/dry mass ratio (1.0 = dry)
    """
    rh = jnp.clip(rh_percent, 1.0, 99.0)

    wr_96 = (0.7540688 * rh**3 - 218.5647 * rh**2
             + 21118.19 * rh - 6.801999e5)
    wr_91 = 8.517e-2 * rh**2 - 15.388 * rh + 698.25
    wr_81 = 8.2696e-3 * rh**2 - 1.3076 * rh + 53.697
    wr_61 = 9.3562e-4 * rh**2 - 0.10427 * rh + 4.3155
    wr_41 = 1.9149e-4 * rh**2 - 8.8619e-3 * rh + 1.2535
    wr_low = 5.1337e-5 * rh**2 + 2.6266e-3 * rh + 1.0149

    wr = jnp.where(
        rh > 96.0, wr_96,
        jnp.where(rh > 91.0, wr_91,
        jnp.where(rh > 81.0, wr_81,
        jnp.where(rh > 61.0, wr_61,
        jnp.where(rh > 41.0, wr_41, wr_low)))))

    return jnp.clip(wr, 1.0, 30.0)


def water_uptake_seasalt(rh_percent: Union[float, jnp.ndarray]) -> Union[float, jnp.ndarray]:
    """Wet/dry mass ratio for sea salt aerosol (waternacl.f).

    Piecewise polynomial fit based on ISORROPIA results at 273 K.

    Args:
        rh_percent: Relative humidity [%], range 0-100

    Returns:
        wr: Wet/dry mass ratio (1.0 = dry)
    """
    rh = jnp.clip(rh_percent, 1.0, 99.0)

    wr_90 = (5.1667642e-2 * rh**3 - 14.153121 * rh**2
             + 1292.8377 * rh - 3.9373536e4)
    wr_80 = (1.0629e-3 * rh**3 - 0.25281 * rh**2
             + 20.171 * rh - 5.3558e2)
    wr_50 = (4.2967e-5 * rh**3 - 7.3654e-3 * rh**2
             + 0.46312 * rh - 7.5731)
    wr_20 = (2.9443e-5 * rh**3 - 2.4739e-3 * rh**2
             + 7.3430e-2 * rh + 1.3727)
    wr_low = 1.17

    wr = jnp.where(
        rh > 90.0, wr_90,
        jnp.where(rh > 80.0, wr_80,
        jnp.where(rh > 50.0, wr_50,
        jnp.where(rh > 20.0, wr_20, wr_low))))

    return jnp.clip(wr, 1.0, 45.0)


def calc_equilibrium_water(
    Mk: jnp.ndarray,
    rh: float,
) -> jnp.ndarray:
    """Calculate equilibrium water content for each aerosol bin.

    Exact port of ezwatereqm.f (Peter Adams, March 2000).

    Args:
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        rh: Relative humidity [fraction 0-1]

    Returns:
        Mk_new: Updated mass with equilibrium water, shape (ibins, icomp)
    """
    rh_percent = jnp.clip(rh * 100.0, 1.0, 99.0)
    wr_so4 = water_uptake_sulfate(rh_percent)
    wr_nacl = water_uptake_seasalt(rh_percent)

    # Sulfate mass converted to NH4HSO4 (factor 1.2)
    so4_mass = Mk[:, SRTSO4] * 1.2

    # Sea salt (disabled in TOMAS)
    nacl_mass = jnp.zeros(Mk.shape[0])

    # Organic mass (same water uptake as sulfate)
    org_mass = jnp.sum(Mk[:, SRTORG1:SRTORG1 + IORG], axis=1)

    # Water = dry_mass * (wr - 1)
    water_mass = ((so4_mass + org_mass) * (wr_so4 - 1.0)
                  + nacl_mass * (wr_nacl - 1.0))

    Mk_new = Mk.at[:, SRTH2O].set(water_mass)
    return Mk_new


# =========================================================================
# Tabazadeh et al. (1997) binary H2SO4/H2O equilibrium
# =========================================================================
# Table 1: water vapour pressure over aqueous H2SO4,
#   P_H2O(mb) = exp[a + b/T + c/T^2].  Columns: (weight_percent, a, b, c).
# Also used by radiative_forcing.py.
TABAZADEH_TABLE1 = np.array([
    [10, 19.726, -4364.8, -147620],
    [15, 19.747, -4390.9, -144690],
    [20, 19.761, -4414.7, -142940],
    [25, 19.794, -4451.1, -140870],
    [30, 19.883, -4519.2, -136500],
    [35, 20.078, -4644.0, -127240],
    [40, 20.379, -4828.5, -112550],
    [45, 20.637, -5011.5, -98811],
    [50, 20.682, -5121.3, -94033],
    [55, 20.555, -5177.6, -96984],
    [60, 20.405, -5252.1, -100840],
    [65, 20.383, -5422.4, -97966],
    [70, 20.585, -5743.8, -83701],
    [75, 21.169, -6310.6, -48396],
    [80, 21.808, -6985.9, -12170],
], dtype=np.float64)

# Equation (1): saturation vapour pressure of pure water [mbar]
#   ln P_H2O = c0 + c1/T + c2/T^2 + c3/T^3
TABAZADEH_EQ1 = (18.452406985, -3505.1578801, -330918.55082, 12725068.262)

# Fitted temperature range of Table 1 [K]
TABAZADEH_T_MIN = 185.0
TABAZADEH_T_MAX = 260.0


def tabazadeh_h2so4_wt(temp, rh_percent):
    """Equilibrium H2SO4 weight percent of a binary H2SO4/H2O droplet.

    JIT-safe counterpart of radiative_forcing.h2so4_equilibrium_wt. The
    weight percent is where the solution water vapour pressure (Table 1)
    equals the ambient partial pressure RH/100 * P_sat(T).

    Table 1 is fitted for T = 185-260 K and 10-80 wt%. Outside that T range
    each row's ln P = a + b/T + c/T^2 is extrapolated smoothly rather than
    clamped (at 298 K and RH 50% this gives ~43 wt%, close to 25 C
    water-activity data). RH is clipped to [1e-3, 99] %, so the result stays
    in [10, 80] wt% and water is at most 9x the acid mass.

    Args:
        temp: Temperature [K]
        rh_percent: Relative humidity [%]

    Returns:
        H2SO4 weight percent, in [10, 80].
    """
    temp = jnp.asarray(temp, dtype=jnp.float64)
    rh = jnp.clip(jnp.asarray(rh_percent, dtype=jnp.float64), 1e-3, 99.0)

    c0, c1, c2, c3 = TABAZADEH_EQ1
    ln_p_sat = c0 + c1 / temp + c2 / temp**2 + c3 / temp**3
    # -ln P increases with concentration, so interpolate on -ln P
    x = -(jnp.log(rh / 100.0) + ln_p_sat)

    table = jnp.asarray(TABAZADEH_TABLE1)
    xp = -(table[:, 1] + table[:, 2] / temp + table[:, 3] / temp**2)
    return jnp.interp(x, xp, table[:, 0])


def calc_equilibrium_water_h2so4(
    Mk: jnp.ndarray,
    rh: float,
    temp: float,
) -> jnp.ndarray:
    """Equilibrium water treating sulfate as pure H2SO4/H2O (Tabazadeh 1997).

    SO4 mass is converted to H2SO4 (x 98/96) and takes the water of a binary
    solution at the Tabazadeh equilibrium weight percent. Organic mass keeps
    the TOMAS ammonium-bisulfate wet/dry ratio (as in calc_equilibrium_water);
    NH4 takes no water.

    Args:
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        rh: Relative humidity [fraction 0-1]
        temp: Temperature [K]

    Returns:
        Mk_new: Updated mass with equilibrium water, shape (ibins, icomp)
    """
    wt = tabazadeh_h2so4_wt(temp, rh * 100.0)
    acid_mass = Mk[:, SRTSO4] * (MW_H2SO4 / MW_SO4)

    wr_org = water_uptake_sulfate(jnp.clip(rh * 100.0, 1.0, 99.0))
    org_mass = jnp.sum(Mk[:, SRTORG1:SRTORG1 + IORG], axis=1)

    water_mass = acid_mass * (100.0 / wt - 1.0) + org_mass * (wr_org - 1.0)
    return Mk.at[:, SRTH2O].set(water_mass)


def equilibrate_water(Mk, rh, temp, water_scheme=WATER_SCHEME_BISULFATE):
    """Set the aerosol water column with the selected scheme.

    Args:
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        rh: Relative humidity [fraction 0-1]
        temp: Temperature [K] (used by the H2SO4 scheme only)
        water_scheme: WATER_SCHEME_BISULFATE (0.0, default) or
            WATER_SCHEME_H2SO4 (1.0). A Python number picks the scheme at
            trace time, so the default compiles to exactly
            calc_equilibrium_water; a traced value selects with jnp.where.

    Returns:
        Mk_new: Updated mass with equilibrium water, shape (ibins, icomp)
    """
    if isinstance(water_scheme, (int, float)):
        if water_scheme > 0.5:
            return calc_equilibrium_water_h2so4(Mk, rh, temp)
        return calc_equilibrium_water(Mk, rh)
    return jnp.where(
        water_scheme > 0.5,
        calc_equilibrium_water_h2so4(Mk, rh, temp),
        calc_equilibrium_water(Mk, rh),
    )
