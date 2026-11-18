"""Adaptive sub-step for the processes that compete for H2SO4.

make_step(..., gas_substep='adaptive') sub-steps SO2 chemistry -> nucleation
-> condensation inside the outer dt. This module picks each sub-step from the
H2SO4 budget at its start (docs/gas_substep.md):

    lambda(dt) = CS + n_eff * k_nuc(C_seen(dt)),   accept dt if lambda*dt <= frac

- C_seen(dt) = C + dH2SO4_chem(dt): the gas nucleation sees after the
  chemistry sub-step and before condensation (exact SO2 decay).
- CS: condensation sink [s^-1] of the current distribution.
- k_nuc = (1 - org_frac) J m_nuc boxvol / C_seen [s^-1]: the gas loss
  frequency to nucleation, the same mass nucleation_step removes. J is the
  rate of the scheme the step applies (nucl_scheme): ricco_dunne, or the
  zhao2024 total over the enabled mechanisms. zhao2024_nucleation_step
  removes 0.9 J m_nuc boxvol of H2SO4 for every mechanism, including those
  that do not depend on H2SO4 (iodine, pure organic); their n_eff is 1.
- n_eff = d ln J / d ln C (finite ratio over 10%, floored at 1): J ~ C^n makes
  the nucleation sink n times stiffer than k_nuc.

Since n_eff >= 1, k_nuc*dt <= frac < 1 on every accepted sub-step, so the
nucleation gas clamp cannot fire. CS alone is not enough: the clamp fires
where CS is small (clean background) and k_nuc is orders of magnitude larger.

Candidates are dt_rem / 2^k, k = 0..n_levels-1, floored at min(min_dt,
dt_rem), then the floor itself as a last rung; the largest one that
passes is taken. Everything here is under stop_gradient: gradients do
not see the sub-step choice.
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp

from ..core.config import SRTSO4, SRTSO2, MW_H2SO4, MW_SO2
from ..physics.condensation_sink import calc_condensation_sink
from ..physics.so2_chemistry import calc_k1_so2_oh
from ..physics.nucleation import (
    ricco_dunne_nucleation_rate, zhao2024_nucleation_rate, _cluster_mass,
    _MNUC, default_nucleation_params, ZHAO2024_ALL_ENABLED,
)

NUCL_SCHEMES = ('ricco_dunne', 'zhao2024')

# Relative perturbation for the effective nucleation order n_eff
_N_EFF_RATIO = 1.1
_N_EFF_MAX = 20.0


def h2so4_loss_frequency(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dts,
                         oh_conc, nuc_kwargs, has_chem=True, has_nuc=True,
                         sparse_density=None, dunne_form='double_exp',
                         nucl_scheme='ricco_dunne'):
    """Linearised H2SO4 loss frequency lambda [s^-1] for each candidate dt.

    Args:
        dts: candidate sub-steps [s], shape (K,)
        oh_conc: OH [molec/cm3] (ignored if has_chem is False)
        nuc_kwargs: dict of the make_step nucleation kwargs: org_conc,
            nh3_conc, fion, fn_scale, and enable_organic, enable_inorganic,
            nucl_params, fn_prescribed (ricco_dunne; fn_prescribed = a
            constant J [cm-3 s-1] replacing the scheme) or hno3, ulvoc, dma, hio3,
            enable_masks (zhao2024)
        has_chem / has_nuc: whether the gas loop contains SO2 chemistry /
            nucleation (static)
        nucl_scheme: the scheme the nucleation step applies (static)

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

    if nucl_scheme == 'ricco_dunne':
        params = nuc_kwargs.get('nucl_params')
        p = default_nucleation_params(dunne_form) if params is None else params
        loss_per_J = (1.0 - p.org_frac) * _cluster_mass(p) * boxvol

        fn_prescribed = nuc_kwargs.get('fn_prescribed')

        def rate(c):
            # Prescribed J does not depend on C: n_eff = 1, k_nuc = J m / C
            if fn_prescribed is not None:
                return jnp.asarray(fn_prescribed, dtype=jnp.float64) + 0.0 * c
            return ricco_dunne_nucleation_rate(
                Gc.at[SRTSO4].set(c), temp, pres, boxvol,
                nuc_kwargs.get('org_conc', 0.0), nuc_kwargs.get('nh3_conc', 0.0),
                nuc_kwargs.get('fion', 0.0),
                nuc_kwargs.get('enable_organic', 1.0),
                nuc_kwargs.get('enable_inorganic', 1.0),
                nuc_kwargs.get('fn_scale', 1.0),
                params, dunne_form)
    elif nucl_scheme == 'zhao2024':
        # zhao2024_nucleation_step: fixed cluster (_MNUC), 90% SO4
        loss_per_J = 0.9 * _MNUC * boxvol

        def rate(c):
            return zhao2024_nucleation_rate(
                Gc.at[SRTSO4].set(c), temp, pres, boxvol,
                nuc_kwargs.get('org_conc', 0.0), nuc_kwargs.get('nh3_conc', 0.0),
                nuc_kwargs.get('fion', 0.0),
                hno3=nuc_kwargs.get('hno3', 0.0),
                ulvoc=nuc_kwargs.get('ulvoc', 0.0),
                dma=nuc_kwargs.get('dma', 0.0),
                hio3=nuc_kwargs.get('hio3', 0.0),
                enable_masks=nuc_kwargs.get('enable_masks',
                                            ZHAO2024_ALL_ENABLED),
                fn_scale=nuc_kwargs.get('fn_scale', 1.0))[0]
    else:
        raise ValueError(f"Unknown nucl_scheme '{nucl_scheme}'. "
                         f"Valid: {NUCL_SCHEMES}")

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
                       sparse_density=None, dunne_form='double_exp',
                       nucl_scheme='ricco_dunne', force_dt=None, force=False):
    """Largest dt_rem / 2^k with lambda*dt <= frac.

    force_dt / force: the sub-step the caller will take instead when
    force is True (make_step's last allowed sub-step, which covers the
    rest of the outer step). It is evaluated as one more candidate, so
    knuc_dt describes the step actually taken.

    Returns:
        dt_sub: chosen sub-step [s] (the criterion's choice, never force_dt)
        floored: True if no candidate passed (dt_sub is the floor) and the
            step is not forced
        knuc_dt: k_nuc * (the step taken); > 1 means the nucleation clamp
            fires
    """
    args = jax.lax.stop_gradient(
        (Nk, Mk, Gc, temp, pres, boxvol, rh, alpha, dt_rem, oh_conc))
    Nk, Mk, Gc, temp, pres, boxvol, rh, alpha, dt_rem, oh_conc = args
    nuc_kwargs = jax.lax.stop_gradient(nuc_kwargs)

    floor = jnp.minimum(min_dt, dt_rem)
    # The floor is its own last rung: dt_rem / 2^(n_levels-1) is above
    # min_dt when dt_rem > 2^(n_levels-1) * min_dt (long coagulation
    # intervals), and the ladder alone could not reach it.
    dts = jnp.append(
        jnp.maximum(dt_rem * 0.5 ** jnp.arange(n_levels), floor), floor)
    n_cand = n_levels + 1
    if force_dt is not None:
        dts = jnp.append(dts, jax.lax.stop_gradient(force_dt))
    lam, _, k_nuc = h2so4_loss_frequency(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dts, oh_conc,
        nuc_kwargs, has_chem, has_nuc, sparse_density, dunne_form,
        nucl_scheme)
    ok = (lam * dts <= frac)[:n_cand]
    floored = ~jnp.any(ok)
    idx = jnp.where(floored, n_cand - 1, jnp.argmax(ok))
    knuc_dt = k_nuc[idx] * dts[idx]
    if force_dt is not None:
        knuc_dt = jnp.where(force, k_nuc[n_cand] * dts[n_cand], knuc_dt)
        floored = floored & ~force
    return dts[idx], floored, knuc_dt
