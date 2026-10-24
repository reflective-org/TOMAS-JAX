"""Shared setup for the Dunne nucleation sensitivity study.

Stratospheric reference conditions, lognormal initial state, and a
differentiable scan driver that also records nucleation regime diagnostics.
"""
import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    ICOMP, N_GAS_SPECIES, SRTSO4, SRTSO2,
    MW_H2SO4, MW_SO2, AVOGADRO, KB, PI,
)
from tomas_jax.physics import nucleation as nuc
from tomas_jax.solvers.condensation import make_step

# ~20 km lower stratosphere
STRAT = dict(temp=217.0, pres=5500.0, rh=0.02, alpha=1.0)
FION = 10.0          # ion pairs/cm3/s
OH_CONC = 1e6        # molec/cm3 (constant)
BOXVOL = 1.0e6       # cm3
DT = 60.0            # s
DENS_INIT = 1770.0   # kg/m3 (sulfate)

# Background stratospheric aerosol
BG_N = 10.0          # #/cm3
BG_GMD = 100e-9      # m
BG_GSD = 1.6


def air_density(temp, pres):
    """Air number density [molec/cm3]."""
    return pres / (KB * temp) * 1e-6


def ppbv_to_molec(ppbv, temp, pres):
    return ppbv * 1e-9 * air_density(temp, pres)


def lognormal_state(xk, n_total=BG_N, gmd=BG_GMD, gsd=BG_GSD,
                    h2so4=0.0, so2=0.0, boxvol=BOXVOL):
    """Pure-sulfate lognormal Nk, Mk and gas Gc (concentrations in molec/cm3).

    Same Fortran-style discretisation as
    benchmarks/python/benchmark_so2_sensitivity.make_initial_state.
    """
    xk = np.asarray(xk)
    nbins = xk.shape[0] - 1
    Dl = ((6.0 * xk[:-1]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
    Dh = ((6.0 * xk[1:]) / (DENS_INIT * PI)) ** (1.0 / 3.0)
    Dk = np.sqrt(Dl * Dh)
    Nk = (n_total * boxvol / (np.sqrt(2.0 * PI) * Dk * np.log(gsd))
          * np.exp(-np.log(Dk / gmd) ** 2 / (2.0 * np.log(gsd) ** 2))
          * (Dh - Dl))
    Nk = np.maximum(Nk, 1e-3)
    Mk = np.zeros((nbins, ICOMP))
    Mk[:, SRTSO4] = Nk * np.sqrt(xk[:-1] * xk[1:])

    Gc = np.zeros(N_GAS_SPECIES)
    Gc[SRTSO4] = h2so4 * boxvol * (MW_H2SO4 / 1000.0) / AVOGADRO
    Gc[SRTSO2] = so2 * boxvol * (MW_SO2 / 1000.0) / AVOGADRO
    return jnp.array(Nk), jnp.array(Mk), jnp.array(Gc)


def make_runner(xk, nsteps, cond=STRAT, fion=FION, oh_conc=OH_CONC,
                boxvol=BOXVOL, dt=DT, max_nuc_substeps=20,
                max_nucleation_frac=0.5):
    """Build run(params, Nk, Mk, Gc) -> (Nk, Mk, Gc, diag).

    One step = SO2 chemistry, then nucleation + coagulation + condensation,
    identical to make_step(['so2_chemistry', 'nucleation', 'coagulation',
    'condensation']). The split lets us record, from the post-chemistry
    state, the nucleation sub-step count and whether the gas clamp fires on
    the first sub-step. Gradients are blind to both (integer/branch).

    Not JIT-wrapped: callers apply jax.jit / jax.jacfwd as needed.
    """
    chem = make_step(['so2_chemistry'], jit=False)
    rest = make_step(['nucleation', 'coagulation', 'condensation'],
                     max_nucleation_frac=max_nucleation_frac,
                     max_nuc_substeps=max_nuc_substeps, jit=False)
    temp, pres, rh, alpha = (cond[k] for k in ('temp', 'pres', 'rh', 'alpha'))

    def run(params, Nk, Mk, Gc):
        kw = dict(org_conc=0.0, nh3_conc=0.0, fion=fion, nucl_params=params)

        def body(carry, _):
            Nk, Mk, Gc = carry
            Nk, Mk, Gc = chem(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha,
                              dt, oh_conc=oh_conc)
            fn = nuc.ricco_dunne_nucleation_rate(
                Gc, temp, pres, boxvol, 0.0, 0.0, fion, params=params)
            n_sub = nuc.compute_nucleation_substeps(
                fn, boxvol, dt, jnp.sum(Nk),
                max_nucleation_frac, max_nuc_substeps)
            mnuc = nuc._cluster_mass(params)
            clamp = ((1.0 - params.org_frac) * fn * mnuc * boxvol * dt / n_sub
                     > Gc[SRTSO4])
            Nk, Mk, Gc = rest(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha,
                              dt, **kw)
            h2so4 = Gc[SRTSO4] / boxvol * 1000.0 / MW_H2SO4 * AVOGADRO
            diag = dict(J=fn, n_sub=n_sub, clamp=clamp, h2so4=h2so4,
                        N_total=jnp.sum(Nk) / boxvol)
            return (Nk, Mk, Gc), diag

        (Nk, Mk, Gc), diag = jax.lax.scan(body, (Nk, Mk, Gc), None,
                                          length=nsteps)
        return Nk, Mk, Gc, diag

    return run
