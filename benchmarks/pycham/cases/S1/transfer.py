"""Gas-to-particle transfer and Kelvin factors of the two models (S1 diagnostics).

* ``pycham_sink``: PyCHAM v5.8.0's partitioning coefficient (partit_var_prep.py +
  partit_var.py): Fuller diffusivity with the SMILES diffusion volume (air 19.7, constant
  1.013e-2), mean free path 3 D / c, Fuchs-Sutugin in the Jacobson (2005) form,
  k_i = 4 pi r N D F(Kn). Works on numpy or jax arrays.
* ``pycham_sink_patch``: drop-in for ``vbs_jax.partitioning.organic_condensation_sink``
  (attribution run ``nokelvin_pytransfer`` only).
* ``tomas_sink``: TOMAS-JAX's own (vbs-jax) organic condensation sink per bin.
* ``kelvin_tomas`` / ``kelvin_pycham``: the two Kelvin factors.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from .case import S1

KB = 1.380649e-23
NA = 6.02214076e23
R_GAS = 8.314462618
# PyCHAM diff_vol_est atom increments (C 15.9, H 2.31, O 6.11); the surrogate is C10H16O4
PYCHAM_DIFF_VOL = 10 * 15.9 + 16 * 2.31 + 4 * 6.11          # 220.4
PYCHAM_SURF_TENSION = 72.0e-3                                 # N/m (partit_var_prep.surfT)


def pycham_diffusivity(temp, pres, mw=S1["pycham_organic_mw"], diff_vol=PYCHAM_DIFF_VOL):
    """Gas diffusivity [m2/s] as partit_var_prep.py."""
    ma = 28.966
    return (1.013e-2 * temp ** 1.75 * ((mw + ma) / (mw * ma)) ** 0.5
            / (pres * (diff_vol ** (1 / 3) + 19.7 ** (1 / 3)) ** 2))


def pycham_sink(n_m3, dp_m, temp, pres, alpha=1.0, mw=S1["pycham_organic_mw"]):
    """Per-bin partitioning coefficient [1/s] (n [#/m3], dp wet diameter [m])."""
    d = pycham_diffusivity(temp, pres, mw)
    c = (8.0 * KB * temp / (np.pi * mw / NA * 1e-3)) ** 0.5
    r = jnp.maximum(dp_m / 2.0, 1e-30)
    kn = 3.0 * d / c / r
    corr = 1.0 / (1.0 + ((1.33 + 0.71 / kn) / (1.0 + 1.0 / kn)
                         + 4.0 * (1.0 - alpha) / (3.0 * alpha)) * kn)
    return 4.0 * np.pi * r * n_m3 * d * corr


def pycham_sink_patch(n, dp, rho_p, temp, pres, mw, sv, alpha, cstar_ug, dbk=1e-10, kc=0.0,
                      r_gas=R_GAS):
    """vbs-jax ``organic_condensation_sink`` signature with PyCHAM's coefficient."""
    k = pycham_sink(n[:, None], dp[:, None], temp, pres, alpha,
                    mw=S1["pycham_organic_mw"] + 0.0 * mw[None, :])
    k = jnp.where(dp[:, None] > 0.0, k, 0.0)
    cs = jnp.sum(k, axis=0)
    safe = jnp.where(cs > 1e-30, cs, 1.0)
    sinkfrac = jnp.where(cs[None, :] > 1e-30, k / safe[None, :], 0.0)
    return cs, sinkfrac, jnp.ones(n.shape[0])


def tomas_sink(n_m3, dp_m, rho_p, temp, pres, cstar_ug):
    """TOMAS-JAX (vbs-jax defaults) per-bin organic sink [1/s] for one class, MW 200."""
    from vbs_jax import DEFAULT_VBS_CONFIG as cfg
    from vbs_jax.condensation_sink import organic_condensation_sink
    cs, frac, _ = organic_condensation_sink(
        jnp.asarray(n_m3), jnp.asarray(dp_m), jnp.asarray(rho_p), temp, pres,
        jnp.asarray([200.0]), cfg.sv, 1.0, jnp.asarray([cstar_ug]), dbk=cfg.dbk, kc=cfg.kc,
        r_gas=cfg.r_gas)
    return np.asarray(frac[:, 0] * cs[0])


def kelvin_tomas(dp_m, temp, mw=200.0):
    """vbs-jax Kelvin factor (sigma 0.025 N/m, rho_org 1200 kg/m3, species MW)."""
    from vbs_jax import DEFAULT_VBS_CONFIG as cfg
    return np.exp(4.0 * cfg.sigma * mw * 1e-3 / (R_GAS * temp * cfg.rho * dp_m))


def kelvin_pycham(dp_m, temp, av_mw, rho_p_kg_m3):
    """PyCHAM partit_var.py: exp(2 avMW sigma / (R T r rho_p)), avMW mass-weighted."""
    return np.exp(2.0 * av_mw * 1e-3 * PYCHAM_SURF_TENSION
                  / (R_GAS * temp * dp_m / 2.0 * rho_p_kg_m3))
