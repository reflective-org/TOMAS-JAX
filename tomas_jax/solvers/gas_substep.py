"""Adaptive sub-step for the processes that compete for H2SO4.

make_step(..., gas_substep='adaptive') sub-steps SO2 chemistry -> nucleation
-> condensation inside the outer dt. This module picks each sub-step from the
H2SO4 budget at its start (docs/gas_substep.md):

    lambda(dt) = CS + n_eff * k_nuc(C_seen(dt)),   accept dt if lambda*dt <= frac

- C_seen(dt) = C + dH2SO4_chem(dt): the gas nucleation sees after the
  chemistry sub-step and before condensation (exact SO2 decay).
- CS: condensation sink [s^-1] of the current distribution.
- k_nuc = (1 - org_frac) J m_nuc boxvol / C_seen [s^-1]: the gas loss
  frequency to nucleation, the same mass nucleation_step removes.
- n_eff = d ln J / d ln C (finite ratio over 10%, floored at 1): J ~ C^n makes
  the nucleation sink n times stiffer than k_nuc.

Since n_eff >= 1, k_nuc*dt <= frac < 1 on every accepted sub-step, so the
nucleation gas clamp cannot fire. CS alone is not enough: the clamp fires
where CS is small (clean background) and k_nuc is orders of magnitude larger.

Candidates are dt_rem / 2^k, k = 0..n_levels-1, floored at min(min_dt,
dt_rem); the largest one that passes is taken. Everything here is under
stop_gradient: gradients do not see the sub-step choice.
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp

from ..core.config import SRTSO4, SRTSO2, MW_H2SO4, MW_SO2
from ..physics.condensation_sink import calc_condensation_sink
from ..physics.so2_chemistry import calc_k1_so2_oh
from ..physics.nucleation import (
    ricco_dunne_nucleation_rate, _cluster_mass, DEFAULT_NUCLEATION_PARAMS,
)

# Relative perturbation for the effective nucleation order n_eff
_N_EFF_RATIO = 1.1
_N_EFF_MAX = 20.0


def h2so4_loss_frequency(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dts,
                         oh_conc, nuc_kwargs, has_chem=True, has_nuc=True,
                         sparse_density=None, dunne_form='double_exp'):
    """Linearised H2SO4 loss frequency lambda [s^-1] for each candidate dt.

    Args:
        dts: candidate sub-steps [s], shape (K,)
        oh_conc: OH [molec/cm3] (ignored if has_chem is False)
        nuc_kwargs: dict with org_conc, nh3_conc, fion, enable_organic,
            enable_inorganic, fn_scale, nucl_params (make_step kwargs)
        has_chem / has_nuc: whether the gas loop contains SO2 chemistry /
            nucleation (static)

    Returns:
        lam: shape (K,), CS + n_eff * k_nuc at C_seen(dt)
        CS: scalar condensation sink [s^-1]
        k_nuc: shape (K,), nucleation loss frequency [s^-1]
    """
    CS, _ = calc_condensation_sink(
        Nk, Mk, temp, pres, boxvol, alpha=alpha, xk=xk,
        sparse_density=sparse_density)
    if not has_nuc:
        return jnp.full_like(dts, CS), CS, jnp.zeros_like(dts)

    C0 = Gc[SRTSO4]
    if has_chem:
        k1 = calc_k1_so2_oh(temp, pres, rh)
        dso2 = Gc[SRTSO2] * -jnp.expm1(-k1 * oh_conc * dts)
        C_seen = C0 + dso2 * (MW_H2SO4 / MW_SO2)
    else:
        C_seen = jnp.full_like(dts, C0)

    params = nuc_kwargs.get('nucl_params')
    p = DEFAULT_NUCLEATION_PARAMS if params is None else params
    loss_per_J = (1.0 - p.org_frac) * _cluster_mass(p) * boxvol

    def rate(c):
        return ricco_dunne_nucleation_rate(
            Gc.at[SRTSO4].set(c), temp, pres, boxvol,
            nuc_kwargs.get('org_conc', 0.0), nuc_kwargs.get('nh3_conc', 0.0),
            nuc_kwargs.get('fion', 0.0),
            nuc_kwargs.get('enable_organic', 1.0),
            nuc_kwargs.get('enable_inorganic', 1.0),
            nuc_kwargs.get('fn_scale', 1.0),
            params, dunne_form)

    J = jax.vmap(rate)(C_seen)
    J_up = jax.vmap(rate)(C_seen * _N_EFF_RATIO)

    pos = (J > 0.0) & (C_seen > 0.0)
    J_safe = jnp.where(pos, J, 1.0)
    C_safe = jnp.where(pos, C_seen, 1.0)
    k_nuc = jnp.where(pos, loss_per_J * J_safe / C_safe, 0.0)
    n_eff = jnp.where(pos & (J_up > 0.0),
                      jnp.log(jnp.where(pos & (J_up > 0.0), J_up / J_safe, 1.0))
                      / jnp.log(_N_EFF_RATIO),
                      1.0)
    n_eff = jnp.clip(n_eff, 1.0, _N_EFF_MAX)
    return CS + n_eff * k_nuc, CS, k_nuc


def choose_gas_substep(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt_rem,
                       oh_conc, nuc_kwargs, *, frac=0.1, min_dt=1e-3,
                       n_levels=24, has_chem=True, has_nuc=True,
                       sparse_density=None, dunne_form='double_exp'):
    """Largest dt_rem / 2^k with lambda*dt <= frac.

    Returns:
        dt_sub: chosen sub-step [s]
        floored: True if no candidate passed (dt_sub is the floor)
        knuc_dt: k_nuc * dt_sub (> 1 means the nucleation clamp fires)
    """
    args = jax.lax.stop_gradient(
        (Nk, Mk, Gc, temp, pres, boxvol, rh, alpha, dt_rem, oh_conc))
    Nk, Mk, Gc, temp, pres, boxvol, rh, alpha, dt_rem, oh_conc = args
    nuc_kwargs = jax.lax.stop_gradient(nuc_kwargs)

    floor = jnp.minimum(min_dt, dt_rem)
    dts = jnp.maximum(dt_rem * 0.5 ** jnp.arange(n_levels), floor)
    lam, _, k_nuc = h2so4_loss_frequency(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dts, oh_conc,
        nuc_kwargs, has_chem, has_nuc, sparse_density, dunne_form)
    ok = lam * dts <= frac
    floored = ~jnp.any(ok)
    idx = jnp.where(floored, n_levels - 1, jnp.argmax(ok))
    return dts[idx], floored, k_nuc[idx] * dts[idx]
