"""Ventilation, surface and wall losses of the indoor level (numpy, applied by the driver
once per coupling interval; docs/aaar/coupling.md "Emissions and losses").

Every loss is first order. For a quantity C with total loss rate r and a constant
source S (ventilation brings outdoor air in: S = lambda C_out) the interval solution is

    C(t) = C* + (C0 - C*) exp(-r t),    C* = S / r,

and the time integral over the interval, which the budgets need, is

    int_0^dt C dt = C* dt + (C0 - C*) (1 - exp(-r dt)) / r.

Each sink then removed (its rate) x (that integral), so in + removals = change exactly.

Particle deposition follows Lai and Nazaroff (2000), J. Aerosol Sci. 31, 463-476
(https://doi.org/10.1016/S0021-8502(99)00536-4): deposition velocities to vertical,
upward-facing (floor) and downward-facing (ceiling) smooth surfaces from the
near-surface friction velocity u*, particle Brownian diffusion and settling.
"""
from __future__ import annotations

import numpy as np

from ..core.config import KB

G_M_S2 = 9.80665            # gravity [m/s2]
R_AIR_J_KG_K = 287.05       # specific gas constant of dry air [J/(kg K)]
MU_REF_PA_S = 1.81e-5       # air viscosity at T_REF [Pa s] (Sutherland)
T_REF_K = 293.15
SUTHERLAND_K = 110.4
MFP_REF_M = 0.0665e-6       # air mean free path at T_REF, 101325 Pa [m]
LN_INTEGRAL_CONSTANT = 39.0  # Lai-Nazaroff I: the 4.3 < y+ <= 30 part of the integral
LN_R_PLUS_MAX = 4.3         # the closed form below holds for r+ <= 4.3 (see the paper)


def relax(c0, c_star, rate, dt):
    """First-order step toward c_star: returns (c(dt), int_0^dt c dt).

    rate = 0 (no loss) gives c0 and c0 dt. Works element-wise on arrays.
    """
    c0 = np.asarray(c0, dtype=float)
    rate = np.asarray(rate, dtype=float)
    c_star = np.asarray(c_star, dtype=float)
    on = rate > 0.0
    safe = np.where(on, rate, 1.0)
    decay = np.exp(-safe * dt)
    c1 = np.where(on, c_star + (c0 - c_star) * decay, c0)
    integral = np.where(on, c_star * dt + (c0 - c_star) * (1.0 - decay) / safe, c0 * dt)
    return c1, integral


def air_properties(temp_k: float, pres_pa: float) -> tuple[float, float, float]:
    """(dynamic viscosity [Pa s], kinematic viscosity [m2/s], mean free path [m])."""
    mu = MU_REF_PA_S * (temp_k / T_REF_K) ** 1.5 * (T_REF_K + SUTHERLAND_K) / (temp_k + SUTHERLAND_K)
    rho = pres_pa / (R_AIR_J_KG_K * temp_k)
    mfp = MFP_REF_M * (temp_k / T_REF_K) * (101325.0 / pres_pa)
    return mu, mu / rho, mfp


def lai_nazaroff_beta(dp_m, temp_k: float, pres_pa: float, friction_velocity_cm_s: float,
                      surface_to_volume_per_m: float, height_m: float,
                      particle_density_g_cm3: float) -> np.ndarray:
    """Deposition loss rate beta [1/s] per particle diameter (Lai and Nazaroff 2000).

    beta = (S_v v_dv + S_u v_du + S_d v_dd) / V with S_u/V = S_d/V = 1/height (floor and
    ceiling) and S_v/V the rest of the surface-to-volume ratio. Velocities (their
    eqs. 12-14): v_dv = u*/I, v_du = v_s / (1 - exp(-v_s I/u*)), v_dd = v_s /
    (exp(v_s I/u*) - 1), with I = 3.64 Sc^(2/3) (a - b) + 39 for r+ <= 4.3, Sc = nu/D
    the particle Schmidt number, r+ = (d/2) u*/nu, and a, b the closed-form integral over
    their near-wall eddy diffusivity (coefficient 7.669e-4 (y+)^3, 10.92 = 7.669e-4^(-1/3)).
    """
    d = np.asarray(dp_m, dtype=float)
    mu, nu, mfp = air_properties(temp_k, pres_pa)
    u_star = friction_velocity_cm_s * 1e-2
    rho_p = particle_density_g_cm3 * 1e3
    kn = 2.0 * mfp / d
    cc = 1.0 + kn * (1.257 + 0.4 * np.exp(-1.1 / kn))           # Cunningham slip correction
    diff = KB * temp_k * cc / (3.0 * np.pi * mu * d)              # Brownian diffusivity [m2/s]
    v_s = rho_p * d**2 * G_M_S2 * cc / (18.0 * mu)                # settling velocity [m/s]
    sc = nu / diff
    r_plus = 0.5 * d * u_star / nu
    if np.any(r_plus > LN_R_PLUS_MAX):
        raise ValueError("Lai-Nazaroff closed form needs r+ = (d/2) u*/nu <= 4.3; largest is "
                         f"{float(np.max(r_plus)):.3g} (particles too large or u* too high)")
    m = 10.92 * sc ** (-1.0 / 3.0)
    sc_inv = 1.0 / sc
    a = (0.5 * np.log((m + 4.3) ** 3 / (sc_inv + 0.0609))
         + np.sqrt(3.0) * np.arctan((8.6 - m) / (np.sqrt(3.0) * m)))
    b = (0.5 * np.log((m + r_plus) ** 3 / (sc_inv + 7.669e-4 * r_plus**3))
         + np.sqrt(3.0) * np.arctan((2.0 * r_plus - m) / (np.sqrt(3.0) * m)))
    integral = 3.64 * sc ** (2.0 / 3.0) * (a - b) + LN_INTEGRAL_CONSTANT
    x = np.minimum(v_s * integral / u_star, 700.0)                # settling / diffusion ratio
    v_dv = u_star / integral
    v_du = v_s / (-np.expm1(-x))
    v_dd = v_s / np.expm1(x)
    sv_floor = 1.0 / height_m
    sv_wall = surface_to_volume_per_m - 2.0 * sv_floor
    return sv_wall * v_dv + sv_floor * v_du + sv_floor * v_dd


def particle_loss_step(nk, mk, nk_out, mk_out, lam_s, beta_s, dt):
    """Ventilation toward outdoor particles plus deposition, per bin and species.

    dN_i/dt = lambda (N_out,i - N_i) - beta_i N_i (same for every species mass).
    Returns (nk, mk, budget) with budget = dict of per-bin arrays: ``in`` (ventilated in),
    ``out`` (ventilated out) and ``deposited`` for number ``n_*`` and mass ``m_*`` [per cell].
    """
    lam_s, beta_s = np.asarray(lam_s, dtype=float), np.asarray(beta_s, dtype=float)
    rate = lam_s + beta_s                                          # (nbins,)
    on = rate > 0
    n_star = np.where(on, lam_s * nk_out / np.where(on, rate, 1.0), 0.0)
    nk1, n_int = relax(nk, n_star, rate, dt)
    lam_m, rate_m, beta_m = lam_s[:, None], rate[:, None], beta_s[:, None]   # (nbins, 1)
    m_star = np.where(rate_m > 0, lam_m * mk_out / np.where(rate_m > 0, rate_m, 1.0), 0.0)
    mk1, m_int = relax(mk, m_star, rate_m, dt)
    budget = {
        "n_in": lam_s * nk_out * dt, "n_out": lam_s * n_int, "n_deposited": beta_s * n_int,
        "m_in": lam_m * mk_out * dt, "m_out": lam_m * m_int, "m_deposited": beta_m * m_int,
    }
    return nk1, mk1, budget


def gas_loss_step(y, y_out, lam_s, extra_rate_s, dt):
    """Ventilation toward outdoor air plus per-species first-order sinks (O3 surface loss,
    condensable vapour wall loss).

    dC/dt = lambda (C_out - C) - k_extra C. Returns (y, budget) with per-species arrays
    ``in``, ``out`` (ventilation) and ``sink`` (= k_extra x integral of C).
    """
    rate = lam_s + extra_rate_s
    c_star = np.where(rate > 0, lam_s * y_out / np.where(rate > 0, rate, 1.0), 0.0)
    y1, integral = relax(y, c_star, rate, dt)
    return y1, {"in": lam_s * y_out * dt, "out": lam_s * integral, "sink": extra_rate_s * integral}
