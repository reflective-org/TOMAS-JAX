"""Shared setup for the Dunne nucleation sensitivity study.

Stratospheric reference conditions, lognormal initial state, and a
differentiable scan driver that also records nucleation regime diagnostics.
"""
import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.core.config import (
    ICOMP, ICOMP_NODIAG, N_GAS_SPECIES, SRTSO4, SRTSO2, SRTORGLAST,
    MW_H2SO4, MW_SO2, AVOGADRO, KB, PI,
)
from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.physics import nucleation as nuc
from tomas_jax.physics.water_equilibrium import WATER_SCHEME_H2SO4
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

# Low-occupancy bin density for coag kernel + condensation sink. The Fortran
# fallbacks (1000 / 1500 kg/m3) trigger on per-cell thresholds that sparse
# stratospheric bins in a 1e6 cm3 box fall below (see make_step). Set to
# DENS_INIT, the density that puts the grid's lower edge (XK0) at 1.7 nm.
SPARSE_DENSITY = DENS_INIT

# Nucleated cluster at the grid density, so its mass equals xk[0]. The library
# default (1350 kg/m3) is 0.76 xk[0]: MNFIX then keeps the mass and resets the
# number, so the new particles are in effect born at 1.2 xk[0], two thirds as many.
# Pure sulfate clusters: the library's 10% organic (from TRACER_SOM-TOMAS) is
# added without any organic vapour, and these runs have none.
NUC_PARAMS = nuc.DEFAULT_NUCLEATION_PARAMS._replace(rho_nuc=DENS_INIT, org_frac=0.0)
NUC_PARAMS_S2 = nuc.DUNNE_TABLE_S2_PARAMS._replace(rho_nuc=DENS_INIT, org_frac=0.0)

# Binary H2SO4/H2O water uptake (Tabazadeh 1997) instead of the TOMAS
# ammonium-bisulfate default: stratospheric aerosol has no NH4.
WATER_SCHEME = WATER_SCHEME_H2SO4


def vmap_over_devices(f, batch):
    """jax.vmap(f) over the leading axis of `batch`, split across the host devices.

    A vmapped forward run uses about one core, so a batch is split into equal
    chunks with pmap (one chunk per device, run in parallel). The batch is
    padded with copies of its last element to a multiple of the device count;
    the padding is dropped from the output. With one device this is
    jit(vmap(f)). The Phase 4 background runs came out bit-identical to one
    vmapped batch, but that is not guaranteed: XLA may fuse multiply-adds
    differently for a different batch size (differences of a few ulps).
    """
    n = jax.tree.leaves(batch)[0].shape[0]
    nd = min(jax.local_device_count(), n)
    per = -(-n // nd)

    def split(x):
        x = jnp.concatenate([x, jnp.repeat(x[-1:], nd * per - n, axis=0)])
        return x.reshape(nd, per, *x.shape[1:])

    out = jax.pmap(jax.vmap(f), devices=jax.local_devices()[:nd])(jax.tree.map(split, batch))
    return jax.tree.map(lambda x: x.reshape(nd * per, *x.shape[2:])[:n], out)


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
                max_nucleation_frac=0.5, sparse_density=SPARSE_DENSITY,
                water_scheme=WATER_SCHEME, dunne_form='double_exp',
                coag_solver='euler_fixed', kdil=None, Nk_bg=None, Mk_bg=None,
                Gc_bg=None):
    """Build run(params, Nk, Mk, Gc) -> (Nk, Mk, Gc, diag).

    One step = SO2 chemistry, then nucleation + coagulation + condensation,
    identical to make_step(['so2_chemistry', 'nucleation', 'coagulation',
    'condensation']). The split lets us record, from the post-chemistry
    state, the nucleation sub-step count and whether the gas clamp fires on
    the first sub-step. Gradients are blind to both (integer/branch).

    sparse_density, water_scheme, dunne_form, coag_solver: see make_step.
    diag also carries the binary neutral and binary ion rates (J_bn, J_bi)
    and SO2 [molec/cm3]. kdil (with Nk_bg, Mk_bg, Gc_bg) appends make_step's
    'dilution' process.

    Not JIT-wrapped: callers apply jax.jit / jax.jacfwd as needed.
    """
    chem = make_step(['so2_chemistry'], jit=False)
    procs = ['nucleation', 'coagulation', 'condensation']
    if kdil is not None:
        procs.append('dilution')
    rest = make_step(procs,
                     max_nucleation_frac=max_nucleation_frac,
                     max_nuc_substeps=max_nuc_substeps,
                     sparse_density=sparse_density,
                     water_scheme=water_scheme, dunne_form=dunne_form,
                     coag_solver=coag_solver, jit=False)
    temp, pres, rh, alpha = (cond[k] for k in ('temp', 'pres', 'rh', 'alpha'))

    def run(params, Nk, Mk, Gc):
        kw = dict(org_conc=0.0, nh3_conc=0.0, fion=fion, nucl_params=params)
        if kdil is not None:
            kw.update(kdil=kdil, Nk_bg=Nk_bg, Mk_bg=Mk_bg, Gc_bg=Gc_bg)

        def body(carry, _):
            Nk, Mk, Gc = carry
            Nk, Mk, Gc = chem(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha,
                              dt, oh_conc=oh_conc)
            fn = nuc.ricco_dunne_nucleation_rate(
                Gc, temp, pres, boxvol, 0.0, 0.0, fion, params=params,
                dunne_form=dunne_form)
            h2so4_pre = Gc[SRTSO4] / boxvol * 1000.0 / MW_H2SO4 * AVOGADRO
            mair = 2.69e19 * 273.15 / temp * pres / 101325.0
            _, J_bn, _, J_bi, _ = nuc.dunne_nucleation_rate(
                temp, fion, h2so4_pre, 0.0, mair, params, dunne_form)
            n_sub = nuc.compute_nucleation_substeps(
                fn, boxvol, dt, jnp.sum(Nk),
                max_nucleation_frac, max_nuc_substeps)
            mnuc = nuc._cluster_mass(params)
            clamp = ((1.0 - params.org_frac) * fn * mnuc * boxvol * dt / n_sub
                     > Gc[SRTSO4])
            Nk, Mk, Gc = rest(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha,
                              dt, **kw)
            h2so4 = Gc[SRTSO4] / boxvol * 1000.0 / MW_H2SO4 * AVOGADRO
            so2 = Gc[SRTSO2] / boxvol * 1000.0 / MW_SO2 * AVOGADRO
            diag = dict(J=fn, J_bn=J_bn, J_bi=J_bi, n_sub=n_sub, clamp=clamp,
                        h2so4=h2so4, so2=so2, N_total=jnp.sum(Nk) / boxvol)
            return (Nk, Mk, Gc), diag

        (Nk, Mk, Gc), diag = jax.lax.scan(body, (Nk, Mk, Gc), None,
                                          length=nsteps)
        return Nk, Mk, Gc, diag

    return run


def make_prescribed_j_runner(xk, nsteps, cond=STRAT, fion=FION, oh_conc=OH_CONC,
                             boxvol=BOXVOL, dt=DT, max_nuc_substeps=20,
                             max_nucleation_frac=0.5, sparse_density=SPARSE_DENSITY,
                             water_scheme=WATER_SCHEME, coag_solver='euler_fixed',
                             params=NUC_PARAMS, cond_method='ppm_jit'):
    """Build run(J, Nk, Mk, Gc) -> (Nk, Mk, Gc, diag) with a prescribed J [cm-3 s-1].

    Same step as make_runner, except that nucleation inserts clusters at the
    constant rate J instead of the Riccobono/Dunne rate. Everything else is
    nucleation.nucleation_step + make_step: cluster mass and organic fraction
    from `params`, SO4 mass taken from the gas, the gas clamp, the sub-step
    count from compute_nucleation_substeps and MNFIX after each sub-step.
    diag['J_scheme'] is the Riccobono/Dunne rate at the same H2SO4, for
    reference. Gradients w.r.t. J are blind to the sub-step count and clamp.
    """
    chem = make_step(['so2_chemistry'], jit=False)
    rest = make_step(['coagulation', 'condensation'], cond_method=cond_method,
                     sparse_density=sparse_density, water_scheme=water_scheme,
                     coag_solver=coag_solver, jit=False)
    temp, pres, rh, alpha = (cond[k] for k in ('temp', 'pres', 'rh', 'alpha'))
    mnuc = nuc._cluster_mass(params)
    org_frac = params.org_frac

    def insert(Nk, Mk, Gc, J, dt_nuc):
        # nucleation_step with fn = J (same clamp: SO4 mass vs gas H2SO4)
        dN = J * boxvol * dt_nuc
        dM_so4 = (1.0 - org_frac) * dN * mnuc
        clamp = dM_so4 > Gc[SRTSO4]
        dN = jnp.where(clamp, Gc[SRTSO4] * (96.0 / 98.0) / mnuc, dN)
        dM_so4 = jnp.where(clamp, Gc[SRTSO4] * (96.0 / 98.0), dM_so4)
        gas = jnp.where(clamp, Gc[SRTSO4], dM_so4)
        Nk = Nk.at[0].add(dN)
        Mk = Mk.at[0, SRTSO4].add(dM_so4).at[0, SRTORGLAST].add(org_frac * dN * mnuc)
        return Nk, Mk, Gc.at[SRTSO4].add(-gas), clamp

    def run(J, Nk, Mk, Gc):
        def body(carry, _):
            Nk, Mk, Gc = carry
            Nk, Mk, Gc = chem(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha,
                              dt, oh_conc=oh_conc)
            J_scheme = nuc.ricco_dunne_nucleation_rate(
                Gc, temp, pres, boxvol, 0.0, 0.0, fion, params=params)
            n_sub = nuc.compute_nucleation_substeps(
                J, boxvol, dt, jnp.sum(Nk), max_nucleation_frac, max_nuc_substeps)
            dt_nuc = dt / n_sub

            def nuc_body(i, c):
                Nk_s, Mk_s, Gc_s, cl = c
                Nk_s, Mk_s, Gc_s, cl_s = insert(Nk_s, Mk_s, Gc_s, J, dt_nuc)
                Nk_s, Mk_s = mnfix_jax(Nk_s, Mk_s, xk, ICOMP_NODIAG)
                return Nk_s, Mk_s, Gc_s, cl | cl_s

            Nk, Mk, Gc, clamp = jax.lax.fori_loop(0, n_sub, nuc_body,
                                                  (Nk, Mk, Gc, False))
            Nk, Mk, Gc = rest(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt)
            h2so4 = Gc[SRTSO4] / boxvol * 1000.0 / MW_H2SO4 * AVOGADRO
            so2 = Gc[SRTSO2] / boxvol * 1000.0 / MW_SO2 * AVOGADRO
            diag = dict(J=J_scheme, n_sub=n_sub, clamp=clamp, h2so4=h2so4, so2=so2,
                        N_total=jnp.sum(Nk) / boxvol)
            return (Nk, Mk, Gc), diag

        (Nk, Mk, Gc), diag = jax.lax.scan(body, (Nk, Mk, Gc), None, length=nsteps)
        return Nk, Mk, Gc, diag

    return run
