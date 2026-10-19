"""Particle set-up shared by the particle cases (S1, S2): how PyCHAM is told to hold the
same water and to have the same wet volume as TOMAS-JAX, and how dry diameters are drawn.

* ``water_mass_ratios``: TOMAS-JAX particle water per unit mass of sulfate and of
  organics (ezwatereqm: water = (wr - 1) (1.2 SO4 + organics)), measured by perturbing
  the model's own ``equilibrate_water`` - never from results.
* ``pycham_diss_wrtw``: PyCHAM "dissociation constant with respect to water" that makes
  PyCHAM's ideal-solution water equal to a given water/dry mass ratio when the Kelvin
  factor is 1. PyCHAM's water Raoult term at RH gives n_w = r0 sum_j d_j n_j with
  r0 = RH / (1 - RH), so d_j = w_j MW_j / (MW_w r0) for w_j water mass per mass of j.
* ``effective_dry_density``: PyCHAM adds volumes and fixes water at 1000 kg/m3; the
  dry density that gives TOMAS-JAX's wet volume for the same dry and water mass.
* ``dry_diameter_nm``: display diameter of a dry mass per particle at the one display
  density of every particle figure (config.DISPLAY_DRY_DENSITY_KG_M3).
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from tomas_jax.box.config import AerosolStepConfig
from tomas_jax.box.scenario import Scenario
from tomas_jax.core.config import ICOMP, SRTH2O, SRTSO4, SRTSOA1
from tomas_jax.physics.water_equilibrium import equilibrate_water

from .config import DISPLAY_DRY_DENSITY_KG_M3

PYCHAM_WATER_DENSITY = 1000.0   # [kg/m3] PyCHAM prop_calc.py (a dens_Comp entry for H2O is ignored)
PYCHAM_WATER_MW = 18.0          # [g/mol] PyCHAM init_conc.py


def water_mass_ratios(s: Scenario) -> dict[str, float]:
    """Particle water [kg] per kg of sulfate (SO4 slot) and per kg of organics in
    TOMAS-JAX at the scenario RH and T (the model's own ``equilibrate_water``)."""
    out = {}
    for name, col in (("sulfate", SRTSO4), ("organic", SRTSOA1)):
        mk = np.zeros((1, ICOMP))
        mk[0, col] = 1.0
        w = np.asarray(equilibrate_water(jnp.asarray(mk), s.environment.rh, s.environment.temp_k,
                                         AerosolStepConfig().water_scheme))[0, SRTH2O]
        out[name] = float(w)
    return out


def pycham_diss_wrtw(water_per_mass: float, mw_g_mol: float, rh: float) -> float:
    """PyCHAM dissociation constant w.r.t. water giving ``water_per_mass`` kg water per
    kg of a component of molar mass ``mw_g_mol`` at ``rh`` (Kelvin factor 1)."""
    if not 0.0 < rh < 1.0:
        raise ValueError(f"rh must be in (0, 1), got {rh}")
    r0 = rh / (1.0 - rh)
    return water_per_mass * mw_g_mol / (PYCHAM_WATER_MW * r0)


def effective_dry_density(rho_wet: float, water_to_dry: float,
                          rho_water: float = PYCHAM_WATER_DENSITY) -> float:
    """Dry density [kg/m3] with m_dry/rho_d + m_w/rho_water = (m_dry + m_w)/rho_wet."""
    return 1.0 / ((1.0 + water_to_dry) / rho_wet - water_to_dry / rho_water)


def dry_diameter_nm(mass_per_particle_kg) -> np.ndarray:
    """Dry diameter [nm] of a dry mass per particle at the display density."""
    m = np.asarray(mass_per_particle_kg, dtype=float)
    return (6.0 * m / (np.pi * DISPLAY_DRY_DENSITY_KG_M3)) ** (1.0 / 3.0) * 1e9
