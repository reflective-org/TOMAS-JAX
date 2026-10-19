"""Aerosol side of the AAAR tutorial box model (JIT-compiled).

``aerosol_outer_step`` advances the TOMAS aerosol over one coupling
(outer) step with fixed micro-steps. Each micro-step:

    1. sources: H2SO4 vapour -> Gc[SRTSO4], SOA class vapour ->
       Gc[SRTSOA1:SRTSOA1+K], particle nitrate -> Mk[:, SRTNIT]
       (by forcing.nitrate_weights over bins with Nk > NITRATE_N_MIN),
       then MNFIX
    2. nucleation -> coagulation -> H2SO4 condensation
       (solvers.condensation.make_step components)
    3. SOA partitioning (physics.soa_partitioning, vbs-jax)
    4. water equilibrium (equilibrate_water, cfg.water_scheme)

The nitrate (1) and SOA (3) mass changes are made in place and then move
particles between bins with the PPM remap (physics/ppm_remap.py) when
cfg.growth_bin_transfer = 'ppm' (default), or with MNFIX ('mnfix').

All rates are per grid cell: [kg/cell/s]. K = len(soa_params.cstar_ref)
(1 for the Level 1 surrogate, up to N_SOA_CLASSES); c*, dHvap, MW and
alpha are traced, so changing their values never recompiles. See
docs/aaar/soa_partitioning.md.
"""
import math
from typing import NamedTuple

import jax
# float64 enforced by core/config.py
import jax.numpy as jnp

from ..core.config import (
    ICOMP_NODIAG, SRTSO4, SRTSOA1, SRTNIT, MW_H2SO4, SV_H2SO4,
)
from ..core.mnfix_jax import MNFIX_N_EMPTY, mnfix_jax
from ..physics.condensation_sink import calc_condensation_sink
from ..physics.nucleation import ricco_dunne_nucleation_rate
from ..physics.ppm_remap import (
    BIN_TRANSFER_PPM, cap_top_bin, remap_mass_change,
)
from ..physics.soa_partitioning import soa_partition_step
from ..physics.water_equilibrium import equilibrate_water
from ..solvers.condensation import make_step
from .config import AerosolStepConfig

AEROSOL_PROCESSES = (
    'nucleation', 'coagulation', 'h2so4_condensation', 'soa_partitioning',
)

# AerosolStepConfig process names -> make_step process names
_MAKE_STEP_PROCESS = {
    'nucleation': 'nucleation',
    'coagulation': 'coagulation',
    'h2so4_condensation': 'condensation',
}

_DEFAULT_CFG = AerosolStepConfig()

# Nitrate placement gate [#/cell]: N2O5-uptake nitrate goes to every bin
# with particles, as SOA and the PPM remap do (they use 1e-10 #/cell).
# It sits just above MNFIX's empty-bin floor (1e-5 #/cell, mnfix.f Neps)
# so the floor's placeholder particles get none. Inside the outer step a
# bin's share follows its number (see micro), so a bin that empties does
# not pile its start-of-step share onto the particles left. The gate was
# NEPS_CONDSINK (1e10 #/cell = 1 cm^-3 in the tutorial box): thin tail
# bins then got no nitrate, stopped growing at night and left a band of a
# few cm^-3 at 40-90 nm in Level 3 that grew with resolution.
NITRATE_N_MIN = 2.0 * MNFIX_N_EMPTY   # 2e-5 #/cell


class AerosolState(NamedTuple):
    """Box aerosol state (TOMAS conventions)."""
    Nk: jnp.ndarray      # (nbins,) number [#/cell]
    Mk: jnp.ndarray      # (nbins, ICOMP) mass [kg/cell]
    Gc: jnp.ndarray      # (N_GAS_SPECIES,) gas mass [kg/cell]
    xk: jnp.ndarray      # (nbins+1,) bin boundaries [kg]
    temp: jnp.ndarray    # [K]
    pres: jnp.ndarray    # [Pa]
    rh: jnp.ndarray      # [0-1]
    boxvol: jnp.ndarray  # [cm^3]


class AerosolForcing(NamedTuple):
    """Sources and nucleation inputs, constant over one outer step."""
    h2so4_rate: jnp.ndarray       # H2SO4 vapour source [kg/cell/s]
    org_rate: jnp.ndarray         # (K,) SOA class vapour sources [kg/cell/s]
    nitrate_rate: jnp.ndarray     # particle NO3 source (as NO3) [kg/cell/s]
    nitrate_weights: jnp.ndarray  # (nbins,) per-bin share at state.Nk, e.g.
                                  # n2o5_uptake; scaled by Nk/state.Nk as
                                  # bins fill or empty within the step
    nh3_conc: jnp.ndarray = 0.0   # nucleation NH3 [molec/cm^3]
    fion: jnp.ndarray = 0.0       # ion-pair production [pairs/cm^3/s]
    org_conc: jnp.ndarray = 0.0   # Riccobono organic vapour [molec/cm^3]


class SOAParams(NamedTuple):
    """Volatility classes (vbs-jax VBSConfig fields that vary by level).

    The class arrays are defined by the caller (gaschem/volatility.py for
    the tutorial); this module applies no c*/dHvap rule of its own.
    """
    cstar_ref: jnp.ndarray  # (K,) c* at 298 K [ug/m^3]
    dhvap: jnp.ndarray      # (K,) enthalpy of vaporisation [kJ/mol]
    mw: jnp.ndarray         # (K,) molecular weight [g/mol]
    alpha: jnp.ndarray      # mass accommodation coefficient [-]


class AerosolDiag(NamedTuple):
    """Diagnostics of one outer step."""
    cs_h2so4: jnp.ndarray       # H2SO4 condensation sink at step end [s^-1]
    j_nuc: jnp.ndarray          # mean Ricco+Dunne nucleation rate [cm^-3 s^-1]
    soa_cs: jnp.ndarray         # (K,) SOA sink, last vbs sub-step [s^-1]
    soa_time_integrated: jnp.ndarray  # bool: every micro-step fully advanced
    nitrate_added: jnp.ndarray  # particle NO3 actually added [kg/cell]


def _aerosol_outer_step(state, forcing, soa_params, dt_outer, *,
                        micro_dt=_DEFAULT_CFG.micro_dt,
                        processes=AEROSOL_PROCESSES, cfg=_DEFAULT_CFG):
    """Advance the aerosol by dt_outer with fixed micro-steps.

    Args:
        state: AerosolState
        forcing: AerosolForcing (rates held constant over dt_outer)
        soa_params: SOAParams with K classes in slots SRTSOA1..SRTSOA1+K-1
        dt_outer: Outer step [s] (static)
        micro_dt: Target micro-step [s] (static); the outer step is split
            into ceil(dt_outer/micro_dt) equal micro-steps
        processes: Static tuple, subset of AEROSOL_PROCESSES (order fixed)
        cfg: AerosolStepConfig (static)

    Returns:
        (AerosolState, AerosolDiag)
    """
    unknown = set(processes) - set(AEROSOL_PROCESSES)
    if unknown:
        raise ValueError(
            f"Unknown processes {sorted(unknown)}. Valid: {AEROSOL_PROCESSES}")
    n_classes = soa_params.cstar_ref.shape[0]
    if jnp.shape(forcing.org_rate) != (n_classes,):
        raise ValueError(
            f"forcing.org_rate shape {jnp.shape(forcing.org_rate)} != "
            f"({n_classes},) from soa_params.cstar_ref")
    s0, s1 = SRTSOA1, SRTSOA1 + n_classes

    n_micro = max(1, math.ceil(dt_outer / micro_dt - 1e-9))
    dt = dt_outer / n_micro

    do_nucl = 'nucleation' in processes
    do_soa = 'soa_partitioning' in processes
    do_nitrate_remap = cfg.growth_bin_transfer == BIN_TRANSFER_PPM
    tomas_step = make_step(
        [_MAKE_STEP_PROCESS[p] for p in AEROSOL_PROCESSES[:3]
         if p in processes],
        water_scheme=cfg.water_scheme,
        water_every_process=cfg.water_every_process, jit=False)

    xk, temp, pres, rh, boxvol = (
        state.xk, state.temp, state.pres, state.rh, state.boxvol)
    nucl_kw = dict(nh3_conc=forcing.nh3_conc, fion=forcing.fion,
                   org_conc=forcing.org_conc)
    # nitrate weight per particle of each bin at the start of the step
    # (the weights are per-bin shares of state.Nk; N2O5 uptake per
    # particle depends on size, not on how many particles a bin holds)
    n0_ok = state.Nk > NITRATE_N_MIN
    nit_w_pp = jnp.where(
        n0_ok, forcing.nitrate_weights / jnp.where(n0_ok, state.Nk, 1.0), 0.0)

    def micro(_, carry):
        Nk, Mk, Gc, j_sum, soa_ok, soa_cs, nit_sum = carry

        # 1. Sources. Nitrate goes to every bin with particles
        # (Nk > NITRATE_N_MIN), by weight per particle x Nk, renormalised;
        # an empty bin (N = 0 or MNFIX's floor) gets none.
        Gc = Gc.at[SRTSO4].add(forcing.h2so4_rate * dt)
        Gc = Gc.at[s0:s1].add(forcing.org_rate * dt)
        w = jnp.where(Nk > NITRATE_N_MIN, nit_w_pp * Nk, 0.0)
        w_sum = jnp.sum(w)
        w = jnp.where(w_sum > 0.0, w / jnp.where(w_sum > 0.0, w_sum, 1.0), 0.0)
        d_nit = forcing.nitrate_rate * dt * w
        Mk_old = Mk
        Mk = Mk.at[:, SRTNIT].add(d_nit)
        nit_sum = nit_sum + jnp.sum(d_nit)
        if do_nitrate_remap:
            Nk, Mk = remap_mass_change(Nk, Mk_old, Mk, xk, ICOMP_NODIAG)
            # nitrate that would push the top bin past the grid is not
            # added (counted out of nitrate_added; the driver warns)
            Mk, nit_back = cap_top_bin(Nk, Mk, xk, (SRTNIT, SRTNIT + 1),
                                       ICOMP_NODIAG)
            nit_sum = nit_sum - nit_back[0]
        Nk, Mk = mnfix_jax(Nk, Mk, xk, ICOMP_NODIAG)

        # 2. Nucleation -> coagulation -> H2SO4 condensation
        if do_nucl:
            j_sum = j_sum + ricco_dunne_nucleation_rate(
                Gc, temp, pres, boxvol, forcing.org_conc, forcing.nh3_conc,
                forcing.fion)
        Nk, Mk, Gc = tomas_step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh,
                                cfg.alpha_h2so4, dt, **nucl_kw)

        # 3. SOA partitioning
        if do_soa:
            Nk, Mk, Gc, sdiag = soa_partition_step(
                Nk, Mk, Gc, xk, temp, pres, boxvol, dt,
                soa_params.cstar_ref, soa_params.dhvap, soa_params.mw,
                soa_params.alpha, n_substeps=cfg.soa_n_substeps,
                bin_transfer=cfg.growth_bin_transfer)
            soa_ok = soa_ok & (sdiag.time_integrated >= dt * (1.0 - 1e-9))
            soa_cs = sdiag.cs

        # 4. Water
        Mk = equilibrate_water(Mk, rh, temp, cfg.water_scheme)
        return Nk, Mk, Gc, j_sum, soa_ok, soa_cs, nit_sum

    zero = jnp.zeros((), dtype=jnp.float64)
    carry0 = (state.Nk, state.Mk, state.Gc, zero, jnp.array(True),
              jnp.zeros(n_classes), zero)
    Nk, Mk, Gc, j_sum, soa_ok, soa_cs, nit_sum = jax.lax.fori_loop(
        0, n_micro, micro, carry0)

    cs_h2so4, _ = calc_condensation_sink(
        Nk, Mk, temp, pres, boxvol, MW_H2SO4, SV_H2SO4, cfg.alpha_h2so4,
        xk=xk)
    diag = AerosolDiag(
        cs_h2so4=cs_h2so4, j_nuc=j_sum / n_micro, soa_cs=soa_cs,
        soa_time_integrated=soa_ok, nitrate_added=nit_sum)
    return state._replace(Nk=Nk, Mk=Mk, Gc=Gc), diag


aerosol_outer_step = jax.jit(
    _aerosol_outer_step,
    static_argnames=('dt_outer', 'micro_dt', 'processes', 'cfg'))
