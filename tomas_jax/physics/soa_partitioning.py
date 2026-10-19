"""SOA gas-particle partitioning for TOMAS-JAX via vbs-jax (JIT-compilable).

Adapter between the TOMAS per-cell state (Nk [#/cell], Mk/Gc [kg/cell],
boxvol [cm^3]) and the per-m^3 kinetic VBS solver
``vbs_jax.vbs_partition_step`` (Zaveri et al. 2014 two-film kinetics,
soacond.f Raoult/Kelvin driving force).

Slot map (core/config.py, docs/aaar/soa_partitioning.md):
    - K = len(cstar_ref) volatility classes live in slots
      SRTSOA1 .. SRTSOA1+K-1 of both Gc (vapour) and Mk (particle).
      K <= N_SOA_CLASSES; the class count comes from the array shape, so
      the step compiles once per (nbins, K) and c*/dHvap/MW are traced.
    - m_org_nonvol (absorbing, never evaporates): every other organic slot
      SRTORG1..SRTORGLAST except SRTNIT -- in practice the nucleated
      organic SRTORG_NUC, plus any unused class slots >= K -- plus the
      absorbing floor below.
    - m_inorg: everything else (SO4, NIT, NH4, H2O). With the TOMAS
      geometry hook it only completes the wet-mass bookkeeping.

Absorbing floor: with zero organic mass in a bin vbs-jax sets psat = 0,
so any vapour (even c* = 1000) condenses, evaporates in the next step and
re-condenses whenever the bin returns to exactly zero (a sawtooth). The
Raoult denominator therefore gets SOA_ABSORBING_FLOOR x (bin dry mass
outside the SOA classes). The floor is never stored in Mk, so no mass is
created; it biases equilibrium C_OA by at most that amount (ASSUMPTIONS.md
AC1).

Bin transfer (``bin_transfer``, static): how the per-bin class mass change
moves particles between the fixed bins.
    - 'ppm' (default): vbs-jax changes the class masses in place in every
      sub-step (``redistribute`` only copies them into the per-cell Mk);
      after the step, ppm_remap.remap_mass_change moves number and every
      species to the bins they now belong to: each bin's PPM sub-bin
      profile (as in H2SO4 condensation) is carried by that bin's own
      log-mass displacement, however far, and deposited onto the fixed
      bins, with each species' share following its composition slope
      inside the bin. SOA that would push the top bin past the grid stays
      in the gas (cap_top_bin); then MNFIX restores its floor in emptied
      bins, and any class mass it deletes returns to the vapour, so gas +
      particle closes per class as with 'mnfix'. Growth and evaporation
      both stay smooth.
    - 'mnfix': the previous scheme; ``redistribute`` calls mnfix_jax after
      every vbs-jax sub-step. Its two-packet split spreads a growing mode
      into a comb (kept for comparison).
``geometry_fn`` = calc_wet_diameters (the same wet Dp/density as the H2SO4
condensation sink, NEPS_CONDSINK = 1e10 #/cell), from the actual mean
masses. The full per-cell (Nk, Mk) travel through vbs-jax as ``aux``, so
non-organic columns are never round-tripped through per-m^3 units.

References:
    - Donahue et al. (2006), Environ. Sci. Technol. 40, 2635-2643
    - Zaveri et al. (2014), Atmos. Chem. Phys. 14, 5153-5181
    - Pankow (1994), Atmos. Environ. 28, 185-188
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp
import numpy as np

from vbs_jax import DEFAULT_VBS_CONFIG, ParticleState, vbs_partition_step

from ..core.config import (
    ICOMP_NODIAG, SRTORG1, SRTORGLAST, SRTSOA1, N_SOA_CLASSES, SRTNIT,
    SRTH2O,
)
from ..core.mnfix_jax import mnfix_jax
from .condensation_sink import calc_wet_diameters
from .ppm_remap import (
    BIN_TRANSFER_PPM, BIN_TRANSFERS, cap_top_bin, remap_mass_change,
)

# vbs-jax defaults: max adaptive sub-steps and Raoult fixed-point iterations
SOA_N_SUBSTEPS = 10
SOA_N_ITER = 3

# Number gate [#/cell] for the per-bin equilibrium (fodc) correction; the
# value of the tomas-jax coupled solver vbs-jax was extracted from.
SOA_N_TINY_CELL = 1.0e-10

# Absorbing floor [-]: fraction of each bin's dry mass outside the SOA
# classes added to the Raoult denominator (never to Mk). See module doc.
SOA_ABSORBING_FLOOR = 1.0e-4


def _slot_masks(icomp, n_classes):
    """Static 0/1 column masks: (non-volatile organic, inorganic + water,
    dry mass outside the classes)."""
    is_class = np.zeros(icomp)
    is_class[SRTSOA1:SRTSOA1 + n_classes] = 1.0
    nonvol = np.zeros(icomp)
    nonvol[SRTORG1:SRTORGLAST + 1] = 1.0
    nonvol[SRTNIT] = 0.0
    nonvol = nonvol - is_class
    seed = 1.0 - is_class
    seed[SRTH2O] = 0.0
    return (jnp.asarray(nonvol), jnp.asarray(1.0 - is_class - nonvol),
            jnp.asarray(seed))


def soa_partition_step(Nk, Mk, Gc, xk, temp, pres, boxvol, dt,
                       cstar_ref, dhvap, mw_org, alpha_org,
                       n_substeps=SOA_N_SUBSTEPS, n_iter=SOA_N_ITER,
                       bin_transfer=BIN_TRANSFER_PPM):
    """Advance SOA gas-particle partitioning of K c*-classes by dt.

    Args:
        Nk: Number per bin [#/cell], shape (ibins,)
        Mk: Mass per bin [kg/cell], shape (ibins, icomp)
        Gc: Gas masses [kg/cell], shape (N_GAS_SPECIES,)
        xk: Bin boundaries [kg], shape (ibins+1,)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm^3]
        dt: Time step [s]
        cstar_ref: Class saturation concentrations at 298 K [ug/m^3],
            shape (K,), K <= N_SOA_CLASSES
        dhvap: Enthalpy of vaporisation per class [kJ/mol], shape (K,)
        mw_org: Molecular weight per class [g/mol], shape (K,)
        alpha_org: Mass accommodation coefficient [-]
        n_substeps: Static max adaptive sub-steps (check
            diag.time_integrated == dt)
        n_iter: Static Raoult fixed-point iterations
        bin_transfer: Static, 'ppm' (default) or 'mnfix' (module doc)

    Returns:
        (Nk, Mk, Gc, diag): updated state (Gc changes only in the class
        slots) and vbs_jax.VBSDiagnostics (per-m^3 / SI units).
    """
    if bin_transfer not in BIN_TRANSFERS:
        raise ValueError(
            f"bin_transfer={bin_transfer!r}; valid: {BIN_TRANSFERS}")
    n_classes = cstar_ref.shape[0]
    if n_classes > N_SOA_CLASSES:
        raise ValueError(
            f"{n_classes} SOA classes > N_SOA_CLASSES={N_SOA_CLASSES}")
    s0, s1 = SRTSOA1, SRTSOA1 + n_classes
    nonvol_mask, inorg_mask, seed_mask = _slot_masks(Mk.shape[1], n_classes)
    boxvol_m3 = boxvol * 1.0e-6

    cfg = DEFAULT_VBS_CONFIG._replace(
        n_species=n_classes, cstar_ref_ug=cstar_ref,
        delta_hvap_kj=dhvap, mw=mw_org)

    def to_state(state, Nk_c, Mk_c):
        Mk_m3 = Mk_c / boxvol_m3
        return state._replace(
            n=Nk_c / boxvol_m3,
            m_org=Mk_m3[:, s0:s1],
            m_org_nonvol=(jnp.sum(Mk_m3 * nonvol_mask, axis=1)
                          + SOA_ABSORBING_FLOOR
                          * jnp.sum(Mk_m3 * seed_mask, axis=1)),
            m_inorg=jnp.sum(Mk_m3 * inorg_mask, axis=1),
        )

    def geometry_fn(state, aux):
        return calc_wet_diameters(aux[0], aux[1], xk)

    def redistribute(state, aux):
        Nk_c, Mk_c = aux
        Mk_c = Mk_c.at[:, s0:s1].set(state.m_org * boxvol_m3)
        if bin_transfer == BIN_TRANSFER_PPM:
            return state, (Nk_c, Mk_c)      # in place; remapped after
        Nk_c, Mk_c = mnfix_jax(Nk_c, Mk_c, xk, ICOMP_NODIAG)
        return to_state(state, Nk_c, Mk_c), (Nk_c, Mk_c)

    dp0, rho0 = calc_wet_diameters(Nk, Mk, xk)
    nbins = Nk.shape[0]
    state0 = to_state(
        ParticleState(n=Nk, m_org=Mk[:, s0:s1], m_org_nonvol=jnp.zeros(nbins),
                      m_inorg=jnp.zeros(nbins), rho_p=rho0, dp_empty=dp0),
        Nk, Mk)

    Mk_in = Mk
    _, gas, (Nk, Mk), diag = vbs_partition_step(
        state0, Gc[s0:s1] / boxvol_m3, temp, pres, dt,
        cfg=cfg, alpha=alpha_org, aux=(Nk, Mk),
        redistribute=redistribute, geometry_fn=geometry_fn,
        n_substeps=n_substeps, n_iter=n_iter,
        n_tiny=SOA_N_TINY_CELL / boxvol_m3)
    gas = gas * boxvol_m3
    if bin_transfer == BIN_TRANSFER_PPM:
        Nk, Mk = remap_mass_change(Nk, Mk_in, Mk, xk, ICOMP_NODIAG,
                                   n_min=SOA_N_TINY_CELL)
        # SOA that would push the top bin past the grid stays in the gas
        Mk, back = cap_top_bin(Nk, Mk, xk, (s0, s1), ICOMP_NODIAG,
                               n_min=SOA_N_TINY_CELL)
        # MNFIX restores its floor in bins the remap emptied. Class mass it
        # deletes there (or trims) returns to the vapour, as vbs-jax does
        # after the 'mnfix' hook, so gas + particle closes per class.
        cls = jnp.sum(Mk[:, s0:s1], axis=0)
        Nk, Mk = mnfix_jax(Nk, Mk, xk, ICOMP_NODIAG)
        back = back + jnp.maximum(cls - jnp.sum(Mk[:, s0:s1], axis=0), 0.0)
        gas = gas + back

    Gc = Gc.at[s0:s1].set(gas)
    return Nk, Mk, Gc, diag
