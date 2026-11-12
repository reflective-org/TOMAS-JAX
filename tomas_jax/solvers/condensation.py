"""Top-level condensation operator-split step for TOMAS-JAX.

Orchestrates the full condensation pipeline in sequence:

    1. H2SO4 condensation — exponential gas depletion + ezcond
    2. NH3 equilibrium   — stoichiometric NH4:SO4 partitioning
    3. Water equilibrium  — hygroscopic uptake (ISORROPIA fits)
    4. MNFIX             — mass-number consistency cleanup

This is called once per model timestep, after the coagulation step.

Condensation methods:
    - method='ppm_jit': Fully JIT-compiled PPM pipeline (default, fastest)
    - method='tfl_jit': Fully JIT-compiled TFL (Fortran-matching)
    - method='tfl': Sequential (non-JIT), Fortran-faithful TFL algorithm
    - method='ppm': Sequential wrapper, PPM advection (JIT internally)

Scan-fused time loops (single XLA program, zero Python dispatch):

    Function                          | Coag      | Cond | Nucl
    ----------------------------------|-----------|------|-----
    run_condensation_scan             | —         | PPM  | —
    run_condensation_scan_tfl         | —         | TFL  | —
    run_combined_scan_ppm             | Euler(3)  | PPM  | —
    run_combined_scan_tfl             | Euler(3)  | TFL  | —
    run_nucleation_condensation_scan  | —         | both | Yes
    run_full_scan                     | Euler(10) | TFL  | Yes

Composable process API:

    step_fn = make_step(['nucleation', 'coagulation', 'condensation'],
                        cond_method='ppm_jit')
    Nk, Mk, Gc = step_fn(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                          org_conc=..., nh3_conc=..., fion=...)

Usage::

    from tomas_jax.solvers.condensation import condensation_step
    Nk_new, Mk_new, Gc_new = condensation_step(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt
    )

    # JIT-compiled version with scan-fused time loop:
    from tomas_jax.solvers.condensation import run_condensation_scan_tfl
    Nk_f, Mk_f, Gc_f, history = run_condensation_scan_tfl(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha,
        dt=60.0, nsteps=1440, prod_rate=h2so4_prod_kg_s
    )
    # history shape (nsteps, 3): columns [N_tot, M_dry, Gc_SO4]
"""
import warnings
import jax
import numpy as np
import jax.numpy as jnp
from typing import Tuple
from functools import partial

from ..core.config import (
    SRTSO4, SRTSO2, SRTNH4, SRTH2O, ICOMP_NODIAG,
    MW_H2SO4, SV_H2SO4, CS_EPS,
)
from ..physics.so2_chemistry import (
    so2_oxidation_step, calc_k1_so2_oh,
    calc_solar_zenith_angle, calc_oh_concentration,
)
from ..physics.dilution import dilution_step
from ..physics.nucleation import (
    nucleation_step, ricco_dunne_nucleation_rate, compute_nucleation_substeps,
    zhao2024_nucleation_step, ZHAO2024_ALL_ENABLED, DUNNE_FORMS,
)
from ..physics.condensation_sink import calc_condensation_sink
from ..physics.ezcond import ezcond
from ..physics.ezcond_ppm_jax import ezcond_ppm_jax
from ..physics.condensation_tfl_jax import ezcond_tfl_jax
from ..physics.nh3_equilibrium import eznh3eqm
from ..physics.water_equilibrium import (
    calc_equilibrium_water, equilibrate_water, WATER_SCHEME_BISULFATE,
)
from ..core.mnfix_jax import mnfix_jax
from ..solvers.diffrax import coag_euler_step
from ..solvers.euler import euler_step
from ..solvers.gas_substep import choose_gas_substep
from ..core.bounded_loop import loop_fori, bounded_while

# Valid kwargs for make_step() step function — used for typo detection
_VALID_MAKE_STEP_KWARGS = {
    # so2_chemistry
    'oh_conc',
    # nucleation (common)
    'org_conc', 'nh3_conc', 'fion', 'fn_scale',
    # nucleation (ricco_dunne)
    'enable_organic', 'enable_inorganic', 'nucl_params',
    # nucleation (zhao2024)
    'hno3', 'ulvoc', 'dma', 'hio3', 'enable_masks',
    # coagulation
    'icomp_nodiag',
    # dilution
    'kdil', 'Nk_bg', 'Mk_bg', 'Gc_bg',
}

# Processes that compete for H2SO4 (sub-stepped by gas_substep='adaptive')
_GAS_PROCESSES = {'so2_chemistry', 'nucleation', 'condensation'}

# Canonical physical process ordering
_CANONICAL_PROCESS_ORDER = [
    'so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution',
]


# =========================================================================
# Layer 0: Non-JIT dispatcher (sequential numpy paths)
# =========================================================================

# =========================================================================
# Layer 0: Non-JIT dispatcher (sequential numpy paths)
# =========================================================================

def condensation_step(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    Gc: jnp.ndarray,
    xk: jnp.ndarray,
    temp: float,
    pres: float,
    boxvol: float,
    rh: float,
    alpha: float,
    dt: float,
    method: str = 'ppm_jit'
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Execute one condensation operator-split step.

    Steps:
        1. H2SO4 condensation via ezcond (TFL) or ezcond_ppm (PPM)
        2. NH3 equilibrium
        3. Water equilibrium
        4. MNFIX cleanup

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        Gc: Gas-phase concentrations [kg/grid cell], shape (N_GAS_SPECIES,)
        xk: Bin boundaries [kg], shape (ibins+1,)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm^3]
        rh: Relative humidity [fraction 0-1]
        alpha: Accommodation coefficient
        dt: Timestep [s]
        method: Condensation method - 'ppm_jit' (default), 'tfl', 'ppm', or 'tfl_jit'

    Returns:
        Nk_new: Updated number concentration
        Mk_new: Updated mass concentration
        Gc_new: Updated gas concentrations
    """
    # Dispatch to pure-JAX path if requested
    if method == 'ppm_jit':
        return condensation_step_jax(
            Nk, Mk, Gc, xk,
            jnp.asarray(temp), jnp.asarray(pres),
            jnp.asarray(boxvol), jnp.asarray(rh),
            jnp.asarray(alpha), jnp.asarray(dt)
        )

    if method == 'tfl_jit':
        return condensation_step_tfl_jax(
            Nk, Mk, Gc, xk,
            jnp.asarray(temp), jnp.asarray(pres),
            jnp.asarray(boxvol), jnp.asarray(rh),
            jnp.asarray(alpha), jnp.asarray(dt)
        )

    # Legacy numpy paths — not JIT-compilable, not GPU-compatible
    warnings.warn(
        f"method='{method}' uses sequential numpy and is NOT JIT-compilable. "
        "Use 'ppm_jit' or 'tfl_jit' for GPU deployment.",
        DeprecationWarning, stacklevel=2,
    )

    # Convert to numpy for sequential operations
    Nk_np = np.array(Nk)
    Mk_np = np.array(Mk)
    Gc_np = np.array(Gc)
    xk_np = np.array(xk)

    # =========================================================
    # 1. H2SO4 Condensation
    # =========================================================
    # Calculate condensation sink for H2SO4
    CS, sinkfrac = calc_condensation_sink(
        Nk, Mk, temp, pres, boxvol,
        MW_H2SO4, SV_H2SO4, alpha, xk=xk
    )
    CS_val = float(CS)

    if CS_val > CS_EPS and Gc_np[SRTSO4] > 0.0:
        # Exponential gas depletion: mcond = Gc * (1 - exp(-CS*dt))
        mcond_so4 = Gc_np[SRTSO4] * (1.0 - np.exp(-CS_val * dt))

        # Deplete gas phase
        Gc_np[SRTSO4] -= mcond_so4

        # Condense onto particles
        if method == 'ppm':
            from ..physics.ezcond_ppm import ezcond_ppm
            Nk_np, Mk_np = ezcond_ppm(
                Nk_np, Mk_np, mcond_so4, SRTSO4,
                xk_np, temp, pres, boxvol, alpha
            )
        else:
            Nk_np, Mk_np = ezcond(
                Nk_np, Mk_np, mcond_so4, SRTSO4,
                xk_np, temp, pres, boxvol, alpha
            )
    elif Gc_np[SRTSO4] > 0.0:
        # CS too small — dump all gas into first bin
        Mk_np[0, SRTSO4] += Gc_np[SRTSO4]
        Nk_np[0] += Gc_np[SRTSO4] / np.sqrt(xk_np[0] * xk_np[1])
        Gc_np[SRTSO4] = 0.0

    # =========================================================
    # 2. NH3 Equilibrium
    # =========================================================
    Gc_jax = jnp.array(Gc_np)
    Mk_jax = jnp.array(Mk_np)
    Gc_jax, Mk_jax = eznh3eqm(Gc_jax, Mk_jax)
    Gc_np = np.array(Gc_jax)
    Mk_np = np.array(Mk_jax)

    # =========================================================
    # 3. Water Equilibrium
    # =========================================================
    Mk_jax = jnp.array(Mk_np)
    Mk_jax = calc_equilibrium_water(Mk_jax, rh)
    Mk_np = np.array(Mk_jax)

    # =========================================================
    # 4. MNFIX
    # =========================================================
    Nk_jax, Mk_jax = mnfix_jax(
        jnp.array(Nk_np), jnp.array(Mk_np),
        jnp.array(xk_np), ICOMP_NODIAG
    )

    return Nk_jax, Mk_jax, jnp.array(Gc_np)


# =========================================================================
# Layer 1: _condensation_step_core — unified JIT condensation
# =========================================================================

def _condensation_step_core(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                            ezcond_fn, water_scheme=WATER_SCHEME_BISULFATE,
                            sparse_density=None, bounded=False):
    """Core condensation: MNFIX -> CS -> gas depletion -> ezcond_fn -> NH3 -> water -> MNFIX.

    This is the single implementation that both PPM and TFL JIT paths share.
    The only difference is which ezcond function is passed in.
    water_scheme selects the water uptake (see equilibrate_water).
    sparse_density: low-occupancy bin density for the condensation sink
    (see calc_wet_diameters); None = Fortran defaults.
    bounded: static; bounded MNFIX and ezcond loops (reverse-mode AD).
    """
    # 0. MNFIX input
    Nk, Mk = mnfix_jax(Nk, Mk, xk, ICOMP_NODIAG, bounded=bounded)

    # 1. Condensation sink
    CS, sinkfrac = calc_condensation_sink(
        Nk, Mk, temp, pres, boxvol,
        MW_H2SO4, SV_H2SO4, alpha, xk=xk, sparse_density=sparse_density
    )

    # 2. H2SO4 condensation
    gc_so4 = Gc[SRTSO4]
    mcond_so4 = gc_so4 * (1.0 - jnp.exp(-CS * dt))
    mcond_so4 = jnp.where((CS > CS_EPS) & (gc_so4 > 0.0), mcond_so4, 0.0)

    Gc_new = Gc.at[SRTSO4].add(-mcond_so4)

    # Dump path (CS too small but gas available)
    dump_N = gc_so4 / jnp.sqrt(xk[0] * xk[1])
    Nk_dump = Nk.at[0].add(dump_N)
    Mk_dump = Mk.at[0, SRTSO4].add(gc_so4)
    Gc_dump = Gc.at[SRTSO4].set(0.0)

    should_dump = (CS <= CS_EPS) & (gc_so4 > 0.0)

    # Condense via ezcond_fn (PPM or TFL)
    Nk_cond, Mk_cond = ezcond_fn(
        Nk, Mk, mcond_so4, SRTSO4,
        xk, temp, pres, boxvol, alpha, sparse_density=sparse_density,
        bounded=bounded,
    )

    # Select: dump path vs condensation path vs no-op
    has_gas = gc_so4 > 0.0
    Nk = jnp.where(should_dump, Nk_dump,
                    jnp.where(has_gas & (CS > CS_EPS), Nk_cond, Nk))
    Mk = jnp.where(should_dump, Mk_dump,
                    jnp.where(has_gas & (CS > CS_EPS), Mk_cond, Mk))
    Gc = jnp.where(should_dump, Gc_dump, Gc_new)

    # 3. NH3 equilibrium
    Gc, Mk = eznh3eqm(Gc, Mk)

    # 4. Water equilibrium
    Mk = equilibrate_water(Mk, rh, temp, water_scheme)

    # 5. MNFIX cleanup
    Nk, Mk = mnfix_jax(Nk, Mk, xk, ICOMP_NODIAG, bounded=bounded)

    return Nk, Mk, Gc


# =========================================================================
# Layer 1 wrappers: backward-compatible public API
# =========================================================================

def condensation_step_jax(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    Gc: jnp.ndarray,
    xk: jnp.ndarray,
    temp: jnp.ndarray,
    pres: jnp.ndarray,
    boxvol: jnp.ndarray,
    rh: jnp.ndarray,
    alpha: jnp.ndarray,
    dt: jnp.ndarray,
    water_scheme=WATER_SCHEME_BISULFATE,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure-JAX condensation step using PPM — fully JIT-compilable."""
    return _condensation_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_ppm_jax, water_scheme=water_scheme,
    )


condensation_step_jit = jax.jit(condensation_step_jax)


def condensation_step_tfl_jax(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    Gc: jnp.ndarray,
    xk: jnp.ndarray,
    temp: jnp.ndarray,
    pres: jnp.ndarray,
    boxvol: jnp.ndarray,
    rh: jnp.ndarray,
    alpha: jnp.ndarray,
    dt: jnp.ndarray,
    water_scheme=WATER_SCHEME_BISULFATE,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Pure-JAX condensation step using TFL — Fortran-matching, JIT-compilable."""
    return _condensation_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_tfl_jax, water_scheme=water_scheme,
    )


condensation_step_tfl_jit = jax.jit(condensation_step_tfl_jax)


# =========================================================================
# Layer 2: _combined_step_core / _full_step_core — multi-process steps
# =========================================================================

def _combined_step_core(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                        ezcond_fn, icomp_nodiag=ICOMP_NODIAG, n_coag_substeps=3,
                        water_scheme=WATER_SCHEME_BISULFATE,
                        water_every_process=False):
    """Coagulation + condensation in one step, parameterized by ezcond_fn.

    water_every_process re-equilibrates aerosol water after coagulation too
    (TOMAS box.f order); by default water is only set by condensation.
    """
    # 1. Coagulation (forward Euler + MNFIX)
    Nk, Mk = coag_euler_step(
        Nk, Mk, xk, temp, pres, boxvol,
        dt=dt, icomp_nodiag=icomp_nodiag,
        n_substeps=n_coag_substeps,
    )
    if water_every_process:
        Mk = equilibrate_water(Mk, rh, temp, water_scheme)

    # 2. Condensation
    return _condensation_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_fn, water_scheme=water_scheme,
    )


def _full_step_core(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                    ezcond_fn, org_conc, nh3_conc, fion,
                    enable_organic=1.0, enable_inorganic=1.0, fn_scale=1.0,
                    icomp_nodiag=ICOMP_NODIAG, n_coag_substeps=10,
                    max_nucleation_frac=0.5, max_nuc_substeps=20,
                    water_scheme=WATER_SCHEME_BISULFATE,
                    water_every_process=False):
    """Nucleation + coagulation + condensation in one step.

    Nucleation uses adaptive sub-stepping: when dN would exceed
    max_nucleation_frac * N_total, the nucleation timestep is subdivided
    (up to max_nuc_substeps) with MNFIX between substeps.

    water_every_process re-equilibrates aerosol water after nucleation and
    after coagulation too (TOMAS box.f order).
    """
    # 1. Adaptive nucleation sub-stepping
    fn = ricco_dunne_nucleation_rate(
        Gc, temp, pres, boxvol,
        org_conc, nh3_conc, fion,
        enable_organic, enable_inorganic, fn_scale,
    )
    N_total = jnp.sum(Nk)
    n_nuc = compute_nucleation_substeps(
        fn, boxvol, dt, N_total,
        max_nucleation_frac, max_nuc_substeps,
    )
    dt_nuc = dt / n_nuc

    def nuc_body(i, carry):
        Nk_s, Mk_s, Gc_s = carry
        Nk_s, Mk_s, Gc_s = nucleation_step(
            Nk_s, Mk_s, Gc_s, xk, temp, pres, boxvol, dt_nuc,
            org_conc, nh3_conc, fion,
            enable_organic, enable_inorganic, fn_scale,
        )
        Nk_s, Mk_s = mnfix_jax(Nk_s, Mk_s, xk, icomp_nodiag)
        return (Nk_s, Mk_s, Gc_s)

    Nk, Mk, Gc = jax.lax.fori_loop(0, n_nuc, nuc_body, (Nk, Mk, Gc))
    if water_every_process:
        Mk = equilibrate_water(Mk, rh, temp, water_scheme)

    # 2. Coagulation (forward Euler + MNFIX)
    Nk, Mk = coag_euler_step(
        Nk, Mk, xk, temp, pres, boxvol,
        dt=dt, icomp_nodiag=icomp_nodiag,
        n_substeps=n_coag_substeps,
    )
    if water_every_process:
        Mk = equilibrate_water(Mk, rh, temp, water_scheme)

    # 3. Condensation
    return _condensation_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_fn, water_scheme=water_scheme,
    )


# =========================================================================
# Layer 2 wrappers: backward-compatible public API
# =========================================================================

def combined_step_ppm_jax(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    icomp_nodiag=ICOMP_NODIAG, n_coag_substeps=3,
    water_scheme=WATER_SCHEME_BISULFATE, water_every_process=False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Combined coagulation + PPM condensation step (JIT-compilable)."""
    return _combined_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_ppm_jax,
        icomp_nodiag=icomp_nodiag, n_coag_substeps=n_coag_substeps,
        water_scheme=water_scheme, water_every_process=water_every_process,
    )


def combined_step_tfl_jax(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    icomp_nodiag=ICOMP_NODIAG, n_coag_substeps=3,
    water_scheme=WATER_SCHEME_BISULFATE, water_every_process=False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Combined coagulation + TFL condensation step (JIT-compilable)."""
    return _combined_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_tfl_jax,
        icomp_nodiag=icomp_nodiag, n_coag_substeps=n_coag_substeps,
        water_scheme=water_scheme, water_every_process=water_every_process,
    )


def condensation_step_with_nucleation_jax(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    org_conc, nh3_conc, fion,
    enable_organic=1.0, enable_inorganic=1.0, fn_scale=1.0,
    use_tfl=1.0,
    max_nucleation_frac=0.5, max_nuc_substeps=20,
    water_scheme=WATER_SCHEME_BISULFATE, water_every_process=False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Nucleation then condensation in one JIT-compilable step.

    Args:
        use_tfl: 1.0 for TFL condensation, 0.0 for PPM (Python-level dispatch)
        max_nucleation_frac: Max dN/N_total per nucleation substep (0.5 = 50%)
        max_nuc_substeps: Hard cap on nucleation substeps
        water_scheme: Water uptake scheme (see equilibrate_water)
        water_every_process: Also re-equilibrate water after nucleation
    """
    # Python-level dispatch — no double compute
    ezcond_fn = ezcond_tfl_jax if float(use_tfl) > 0.5 else ezcond_ppm_jax

    # 1. Adaptive nucleation sub-stepping
    fn = ricco_dunne_nucleation_rate(
        Gc, temp, pres, boxvol,
        org_conc, nh3_conc, fion,
        enable_organic, enable_inorganic, fn_scale,
    )
    N_total = jnp.sum(Nk)
    n_nuc = compute_nucleation_substeps(
        fn, boxvol, dt, N_total,
        max_nucleation_frac, max_nuc_substeps,
    )
    dt_nuc = dt / n_nuc

    def nuc_body(i, carry):
        Nk_s, Mk_s, Gc_s = carry
        Nk_s, Mk_s, Gc_s = nucleation_step(
            Nk_s, Mk_s, Gc_s, xk, temp, pres, boxvol, dt_nuc,
            org_conc, nh3_conc, fion,
            enable_organic, enable_inorganic, fn_scale,
        )
        Nk_s, Mk_s = mnfix_jax(Nk_s, Mk_s, xk, ICOMP_NODIAG)
        return (Nk_s, Mk_s, Gc_s)

    Nk, Mk, Gc = jax.lax.fori_loop(0, n_nuc, nuc_body, (Nk, Mk, Gc))
    if water_every_process:
        Mk = equilibrate_water(Mk, rh, temp, water_scheme)

    # 2. Condensation
    return _condensation_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_fn, water_scheme=water_scheme,
    )


condensation_step_with_nucleation_jit = jax.jit(
    condensation_step_with_nucleation_jax,
    static_argnums=(16,),  # use_tfl must be static for Python-level dispatch
    # max_nucleation_frac (17) and max_nuc_substeps (18) are captured as tracers
    static_argnames=('use_tfl', 'water_scheme', 'water_every_process'),
)


def full_step_jax(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    org_conc, nh3_conc, fion,
    enable_organic=1.0, enable_inorganic=1.0, fn_scale=1.0,
    use_tfl=1.0, icomp_nodiag=ICOMP_NODIAG,
    max_nucleation_frac=0.5, max_nuc_substeps=20,
    water_scheme=WATER_SCHEME_BISULFATE, water_every_process=False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Full step: nucleation + coagulation + condensation (JIT-compilable).

    Uses fixed-step forward Euler coagulation (no diffrax) for minimal XLA overhead.
    Nucleation uses adaptive sub-stepping when dN would exceed max_nucleation_frac * N_total.
    """
    ezcond_fn = ezcond_tfl_jax if float(use_tfl) > 0.5 else ezcond_ppm_jax
    return _full_step_core(
        Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
        ezcond_fn=ezcond_fn,
        org_conc=org_conc, nh3_conc=nh3_conc, fion=fion,
        enable_organic=enable_organic, enable_inorganic=enable_inorganic,
        fn_scale=fn_scale, icomp_nodiag=icomp_nodiag, n_coag_substeps=10,
        max_nucleation_frac=max_nucleation_frac,
        max_nuc_substeps=max_nuc_substeps,
        water_scheme=water_scheme, water_every_process=water_every_process,
    )


# =========================================================================
# Layer 3: _run_scan — single scan implementation
# =========================================================================

def _run_scan(Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, diag_mode='rich',
              so2_prod_rate=0.0):
    """Internal scan: adds H2SO4/SO2 prod, calls step_fn, collects diagnostics.

    Args:
        step_fn: Callable(Nk, Mk, Gc) -> (Nk, Mk, Gc)
        nsteps: Number of scan steps
        dt: Timestep [s]
        prod_rate: H2SO4 production rate [kg/s]
        diag_mode: 'rich' returns (N_tot, M_dry, Gc_SO4) per step,
                   'light' returns N_tot only
        so2_prod_rate: SO2 emission rate [kg/s] (added to Gc[SRTSO2] each step)

    Returns:
        (Nk_f, Mk_f, Gc_f), diagnostics
    """
    if diag_mode == 'rich':
        def body(carry, _):
            Nk_c, Mk_c, Gc_c = carry
            Gc_c = Gc_c.at[SRTSO4].add(prod_rate * dt)
            Gc_c = Gc_c.at[SRTSO2].add(so2_prod_rate * dt)
            Nk_c, Mk_c, Gc_c = step_fn(Nk_c, Mk_c, Gc_c)
            diag = jnp.array([jnp.sum(Nk_c),
                               jnp.sum(Mk_c[:, :SRTH2O]),
                               Gc_c[SRTSO4]])
            return (Nk_c, Mk_c, Gc_c), diag
    else:
        def body(carry, _):
            Nk_c, Mk_c, Gc_c = carry
            Gc_c = Gc_c.at[SRTSO4].add(prod_rate * dt)
            Gc_c = Gc_c.at[SRTSO2].add(so2_prod_rate * dt)
            Nk_c, Mk_c, Gc_c = step_fn(Nk_c, Mk_c, Gc_c)
            return (Nk_c, Mk_c, Gc_c), jnp.sum(Nk_c)

    (Nk_f, Mk_f, Gc_f), history = jax.lax.scan(
        body, (Nk, Mk, Gc), None, length=nsteps
    )
    return (Nk_f, Mk_f, Gc_f), history


# =========================================================================
# Layer 3 wrappers: backward-compatible scan-fused loops
# =========================================================================

@partial(jax.jit, static_argnums=(10,), static_argnames=('nsteps', 'water_scheme'))
def run_condensation_scan(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    nsteps: int,
    prod_rate,
    water_scheme=WATER_SCHEME_BISULFATE,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run nsteps PPM condensation steps fused into a single XLA program.

    Returns:
        Nk_f, Mk_f, Gc_f: Final state arrays
        history: shape (nsteps, 3) — columns [N_tot, M_dry, Gc_SO4]
    """
    def step_fn(Nk_c, Mk_c, Gc_c):
        return _condensation_step_core(
            Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt,
            ezcond_fn=ezcond_ppm_jax, water_scheme=water_scheme,
        )
    (Nk_f, Mk_f, Gc_f), history = _run_scan(
        Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, 'rich'
    )
    return Nk_f, Mk_f, Gc_f, history


@partial(jax.jit, static_argnums=(10,), static_argnames=('nsteps', 'water_scheme'))
def run_condensation_scan_tfl(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    nsteps: int,
    prod_rate,
    water_scheme=WATER_SCHEME_BISULFATE,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Run nsteps TFL condensation steps fused into a single XLA program.

    Returns:
        Nk_f, Mk_f, Gc_f: Final state arrays
        history: shape (nsteps, 3) — columns [N_tot, M_dry, Gc_SO4]
    """
    def step_fn(Nk_c, Mk_c, Gc_c):
        return _condensation_step_core(
            Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt,
            ezcond_fn=ezcond_tfl_jax, water_scheme=water_scheme,
        )
    (Nk_f, Mk_f, Gc_f), history = _run_scan(
        Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, 'rich'
    )
    return Nk_f, Mk_f, Gc_f, history


@partial(jax.jit, static_argnums=(10,),
         static_argnames=('nsteps', 'water_scheme', 'water_every_process'))
def run_combined_scan_ppm(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    nsteps: int,
    prod_rate,
    water_scheme=WATER_SCHEME_BISULFATE,
    water_every_process=False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan-fused Euler coagulation + PPM condensation loop."""
    def step_fn(Nk_c, Mk_c, Gc_c):
        return _combined_step_core(
            Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt,
            ezcond_fn=ezcond_ppm_jax,
            water_scheme=water_scheme, water_every_process=water_every_process,
        )
    (Nk_f, Mk_f, Gc_f), history = _run_scan(
        Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, 'light'
    )
    return Nk_f, Mk_f, Gc_f, history


@partial(jax.jit, static_argnums=(10,),
         static_argnames=('nsteps', 'water_scheme', 'water_every_process'))
def run_combined_scan_tfl(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    nsteps: int,
    prod_rate,
    water_scheme=WATER_SCHEME_BISULFATE,
    water_every_process=False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan-fused Euler coagulation + TFL condensation loop."""
    def step_fn(Nk_c, Mk_c, Gc_c):
        return _combined_step_core(
            Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt,
            ezcond_fn=ezcond_tfl_jax,
            water_scheme=water_scheme, water_every_process=water_every_process,
        )
    (Nk_f, Mk_f, Gc_f), history = _run_scan(
        Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, 'light'
    )
    return Nk_f, Mk_f, Gc_f, history


@partial(jax.jit, static_argnums=(10, 18),
         static_argnames=('nsteps', 'use_tfl', 'water_scheme', 'water_every_process'))
def run_nucleation_condensation_scan(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    nsteps: int,
    prod_rate,
    org_conc, nh3_conc, fion,
    enable_organic, enable_inorganic, fn_scale,
    use_tfl,
    max_nucleation_frac=0.5, max_nuc_substeps=20,
    water_scheme=WATER_SCHEME_BISULFATE,
    water_every_process=False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan-fused nucleation + condensation loop with adaptive sub-stepping."""
    ezcond_fn = ezcond_tfl_jax if float(use_tfl) > 0.5 else ezcond_ppm_jax

    def step_fn(Nk_c, Mk_c, Gc_c):
        # Adaptive nucleation sub-stepping
        fn = ricco_dunne_nucleation_rate(
            Gc_c, temp, pres, boxvol,
            org_conc, nh3_conc, fion,
            enable_organic, enable_inorganic, fn_scale,
        )
        N_total = jnp.sum(Nk_c)
        n_nuc = compute_nucleation_substeps(
            fn, boxvol, dt, N_total,
            max_nucleation_frac, max_nuc_substeps,
        )
        dt_nuc = dt / n_nuc

        def nuc_body(i, carry):
            Nk_s, Mk_s, Gc_s = carry
            Nk_s, Mk_s, Gc_s = nucleation_step(
                Nk_s, Mk_s, Gc_s, xk, temp, pres, boxvol, dt_nuc,
                org_conc, nh3_conc, fion,
                enable_organic, enable_inorganic, fn_scale,
            )
            Nk_s, Mk_s = mnfix_jax(Nk_s, Mk_s, xk, ICOMP_NODIAG)
            return (Nk_s, Mk_s, Gc_s)

        Nk_c, Mk_c, Gc_c = jax.lax.fori_loop(
            0, n_nuc, nuc_body, (Nk_c, Mk_c, Gc_c))
        if water_every_process:
            Mk_c = equilibrate_water(Mk_c, rh, temp, water_scheme)

        return _condensation_step_core(
            Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt,
            ezcond_fn=ezcond_fn, water_scheme=water_scheme,
        )

    (Nk_f, Mk_f, Gc_f), history = _run_scan(
        Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, 'light'
    )
    return Nk_f, Mk_f, Gc_f, history


@partial(jax.jit, static_argnums=(10, 18),
         static_argnames=('nsteps', 'use_tfl', 'water_scheme', 'water_every_process'))
def run_full_scan(
    Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
    nsteps: int,
    prod_rate,
    org_conc, nh3_conc, fion,
    enable_organic, enable_inorganic, fn_scale,
    use_tfl,
    max_nucleation_frac=0.5, max_nuc_substeps=20,
    water_scheme=WATER_SCHEME_BISULFATE,
    water_every_process=False,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Scan-fused nucleation + coagulation + condensation loop with adaptive sub-stepping."""
    ezcond_fn = ezcond_tfl_jax if float(use_tfl) > 0.5 else ezcond_ppm_jax

    def step_fn(Nk_c, Mk_c, Gc_c):
        return _full_step_core(
            Nk_c, Mk_c, Gc_c, xk, temp, pres, boxvol, rh, alpha, dt,
            ezcond_fn=ezcond_fn,
            org_conc=org_conc, nh3_conc=nh3_conc, fion=fion,
            enable_organic=enable_organic, enable_inorganic=enable_inorganic,
            fn_scale=fn_scale,
            max_nucleation_frac=max_nucleation_frac,
            max_nuc_substeps=max_nuc_substeps,
            water_scheme=water_scheme, water_every_process=water_every_process,
        )

    (Nk_f, Mk_f, Gc_f), history = _run_scan(
        Nk, Mk, Gc, step_fn, nsteps, dt, prod_rate, 'light'
    )
    return Nk_f, Mk_f, Gc_f, history


# =========================================================================
# Layer 4: make_step() — public composable API
# =========================================================================

def make_step(processes, cond_method='ppm_jit', nucl_scheme='ricco_dunne',
              dunne_form='double_exp',
              n_coag_substeps=10,
              max_nucleation_frac=0.5, max_nuc_substeps=20,
              water_scheme=WATER_SCHEME_BISULFATE, water_every_process=False,
              sparse_density=None, coag_solver='euler_fixed',
              coag_max_substeps=None, gas_substep=None,
              gas_substep_frac=0.1, gas_max_substeps=16384, gas_min_dt=1e-3,
              coag_interval_frac=0.01, return_substep_info=False,
              split_chemistry=False,
              ad_mode='forward', ppm_max_substeps=64, ppm_tangent='exact',
              jit=True):
    """Build a step function from an ordered list of process names.

    The returned function has signature:
        step_fn(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, **kwargs)
    and returns (Nk, Mk, Gc).

    The ``for process in processes`` loop is Python-level — unrolled at JAX
    trace time. This makes it trivial to reorder, skip, or add processes.

    Nucleation uses adaptive sub-stepping: when dN would exceed
    max_nucleation_frac * N_total, the nucleation timestep is subdivided.

    Args:
        processes: Ordered list of process names, e.g.
            ['so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution']
        cond_method: 'ppm_jit' or 'tfl_jit'
        nucl_scheme: 'ricco_dunne' (Riccobono 2014 + Dunne 2016) or
            'zhao2024' (11-mechanism Zhao et al. 2024). For zhao2024,
            pass extra kwargs: hno3, ulvoc, dma, hio3, enable_masks.
        dunne_form: ricco_dunne only. 'double_exp' (Dunne Table S1,
            default) or 'single_exp' (Table S2 cross-check fit; pass
            nucl_params=DUNNE_TABLE_S2_PARAMS with it). Also used for the
            sub-step rate estimate.
        n_coag_substeps: Number of forward-Euler substeps for coagulation
            (coag_solver='euler_fixed' only)
        coag_solver: 'euler_fixed' (default, coag_euler_step: n_coag_substeps
            equal sub-steps, kernel frozen over the step) or 'euler_adaptive'
            (solvers/euler.euler_step, Fortran multicoag.f: sub-step
            dt <= min(0.25 N/|dN/dt|, 10 M/|dM/dt|), kernel recomputed every
            sub-step, MNFIX after each). The fixed scheme goes unstable when
            K*N*dt/n_coag_substeps >~ 1 (e.g. ~3e9 cm-3 of 2 nm particles at
            dt = 60 s); the adaptive one stays stable but its cost scales
            with the number of sub-steps. Adaptive is a while_loop: forward-
            mode differentiable only, and gradients do not see its sub-step
            count.
        coag_max_substeps: Sub-step cap for 'euler_adaptive' (None:
            10000, or 1024 with ad_mode='reverse'). Hitting it truncates
            the coagulation step silently.
        max_nucleation_frac: Max dN/N_total per nucleation substep (0.5 = 50%)
        max_nuc_substeps: Hard cap on nucleation substeps
        water_scheme: Aerosol water uptake, WATER_SCHEME_BISULFATE (default,
            TOMAS ammonium-bisulfate fit) or WATER_SCHEME_H2SO4 (Tabazadeh
            1997 pure H2SO4/H2O). See physics/water_equilibrium.py.
        water_every_process: If True, re-equilibrate aerosol water after
            nucleation, coagulation and dilution as well as condensation
            (TOMAS box.f order). Default False: only condensation sets water.
        sparse_density: Density [kg/m3] for low-occupancy bins, used in the
            coagulation kernel and the condensation sink. None (default)
            keeps the Fortran values: 1000 for bins with < 1e-15 kg/cell
            (aerodens.f) and 1500 for bins with <= 1e10 #/cell
            (getCondSink.f). Both thresholds are per grid cell, so in small
            boxes (stratosphere, SAI plumes) real sulfate bins hit them.
            For those cases pass a sulfate density, e.g.
            ``radiative_forcing.DENSITY_H2SO4_SOLUTION`` (1700, ~75 wt%
            H2SO4 at 220 K). See docs/nucleation_sensitivity.md.
        gas_substep: None (default, every process on dt) or 'adaptive':
            so2_chemistry, nucleation and condensation are sub-stepped in a
            while_loop until dt is covered, and the other processes
            (coagulation, dilution) then run once on dt, after them. Each
            sub-step is the largest dt_rem/2^k with
            (CS + n_eff*k_nuc)*dt_sub <= gas_substep_frac, from the H2SO4
            budget at its start (solvers/gas_substep.py,
            docs/gas_substep.md). Forward-mode differentiable only;
            gradients do not see the sub-step choice.
        gas_substep_frac: The criterion's f (default 0.1). k_nuc*dt_sub <= f
            keeps the nucleation gas clamp from firing.
        gas_max_substeps: Sub-step cap per outer step; the last one covers
            the rest of dt.
        gas_min_dt: Smallest sub-step [s]; taken when no larger one passes.
        return_substep_info: With gas_substep='adaptive', the step function
            returns (Nk, Mk, Gc, info), info = dict(n_sub, dt_min, hit_cap,
            n_floor: sub-steps at gas_min_dt that failed the criterion,
            knuc_dt_max: largest k_nuc*dt_sub, > 1 means the clamp fired).
        split_chemistry: gas_substep=None only (the adaptive path always
            splits). False (default): so2_chemistry runs on the whole dt
            before nucleation, which then sees H2SO4 at the top of the step's
            sawtooth, about C*(1 + CS*dt/2), and J ~ C^n is high by about
            n*CS*dt/2 (first order in dt). True: half of dt's chemistry before
            nucleation and half after, as in the adaptive path; the bias is
            O((CS*dt)^2). Needs so2_chemistry before nucleation in processes.
        ad_mode: 'forward' (default) or 'reverse'. Every loop with a traced
            trip count (MNFIX phase 3, PPM CFL sub-steps, nucleation
            sub-steps, euler_adaptive, the gas sub-steps) is a while_loop or
            fori_loop, which supports forward-mode AD only. 'reverse' runs
            them as bounded loops (core/bounded_loop.py: static length,
            lax.cond skips, jax.checkpoint) so jax.grad/vjp work. Forward
            values match 'forward' to round-off unless a cap binds
            (ppm_max_substeps, coag_max_substeps). Checkpoint the caller's
            time loop too (jax.checkpoint(step_fn) inside the scan) or the
            backward pass keeps every step's intermediates.
        ppm_max_substeps: PPM CFL sub-step cap with ad_mode='reverse'
            (static). A binding cap raises the Courant number above 0.8.
        ppm_tangent: 'exact' (default) or 'frozen_shape' (PPM condensation
            only). 'frozen_shape' leaves every value unchanged but
            differentiates the PPM reconstruction with each bin's sub-bin
            shape held fixed (condensation_ppm._freeze_shape). The exact
            tangent can grow without bound through the reconstruction when few
            particles grow fast in high H2SO4 (docs/prescribed_j.md). It is not
            the exact derivative of the discrete model.
            Callers no longer need to wrap manually. Double-JIT is a no-op.

    Returns:
        A callable step function (JIT-compiled by default).

    For JIT stability, always pass the same set of kwargs across calls.

    Example::

        step = make_step(['nucleation', 'coagulation', 'condensation'],
                         cond_method='ppm_jit')
        Nk, Mk, Gc = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                           org_conc=org_conc, nh3_conc=nh3_conc, fion=fion)

        # With SO2 chemistry:
        step = make_step(['so2_chemistry', 'nucleation', 'coagulation', 'condensation'])
        Nk, Mk, Gc = step(..., oh_conc=1e6)

        # Zhao 2024 11-mechanism scheme:
        step = make_step(['nucleation', 'coagulation', 'condensation'],
                         nucl_scheme='zhao2024')
        Nk, Mk, Gc = step(..., org_conc=..., nh3_conc=..., fion=...,
                           hno3=..., ulvoc=..., dma=..., hio3=...)
    """
    if ad_mode not in ('forward', 'reverse'):
        raise ValueError(f"Unknown ad_mode '{ad_mode}'. "
                         f"Valid: ['forward', 'reverse']")
    bounded = ad_mode == 'reverse'
    if coag_max_substeps is None:
        coag_max_substeps = 1024 if bounded else 10000

    if ppm_tangent not in ('exact', 'frozen_shape'):
        raise ValueError(f"Unknown ppm_tangent '{ppm_tangent}'. "
                         f"Valid: ['exact', 'frozen_shape']")
    ezcond_fn = ezcond_tfl_jax if 'tfl' in cond_method else ezcond_ppm_jax
    if ppm_tangent == 'frozen_shape' and ezcond_fn is not ezcond_ppm_jax:
        raise ValueError("ppm_tangent='frozen_shape' needs PPM condensation "
                         f"(cond_method='ppm_jit'), not '{cond_method}'")
    ppm_opts = {}
    if bounded and ezcond_fn is ezcond_ppm_jax:
        ppm_opts['ppm_max_substeps'] = ppm_max_substeps
    if ppm_tangent == 'frozen_shape':
        ppm_opts['ppm_freeze_shape'] = True
    if ppm_opts:
        ezcond_fn = partial(ezcond_ppm_jax, **ppm_opts)

    valid = {'so2_chemistry', 'nucleation', 'coagulation', 'condensation', 'dilution'}
    for p in processes:
        if p not in valid:
            raise ValueError(f"Unknown process '{p}'. Valid: {sorted(valid)}")

    # Warn if process order deviates from canonical physical ordering
    canonical_indices = [_CANONICAL_PROCESS_ORDER.index(p) for p in processes]
    if canonical_indices != sorted(canonical_indices):
        canonical_subset = [p for p in _CANONICAL_PROCESS_ORDER if p in processes]
        warnings.warn(
            f"Non-standard process order: {list(processes)}. "
            f"Canonical physical order is: {canonical_subset}. "
            f"Non-standard ordering may produce physically incorrect results.",
            UserWarning, stacklevel=2,
        )

    if coag_solver not in ('euler_fixed', 'euler_adaptive'):
        raise ValueError(f"Unknown coag_solver '{coag_solver}'. "
                         f"Valid: ['euler_fixed', 'euler_adaptive']")

    valid_schemes = {'ricco_dunne', 'zhao2024'}
    if nucl_scheme not in valid_schemes:
        raise ValueError(f"Unknown nucl_scheme '{nucl_scheme}'. Valid: {sorted(valid_schemes)}")

    use_zhao = nucl_scheme == 'zhao2024'
    if dunne_form not in DUNNE_FORMS:
        raise ValueError(f"Unknown dunne_form '{dunne_form}'. Valid: {DUNNE_FORMS}")

    if gas_substep not in (None, 'adaptive'):
        raise ValueError(f"Unknown gas_substep '{gas_substep}'. "
                         f"Valid: [None, 'adaptive']")
    if gas_substep is not None:
        if not _GAS_PROCESSES.intersection(processes):
            raise ValueError("gas_substep='adaptive' needs at least one of "
                             f"{sorted(_GAS_PROCESSES)} in processes")
        if gas_substep_frac <= 0.0 or gas_max_substeps < 1 or gas_min_dt <= 0.0:
            raise ValueError("gas_substep_frac and gas_min_dt must be > 0 and "
                             "gas_max_substeps >= 1")
    elif return_substep_info:
        raise ValueError("return_substep_info needs gas_substep='adaptive'")
    if split_chemistry and not ('so2_chemistry' in processes and 'nucleation' in processes
                                and processes.index('so2_chemistry')
                                < processes.index('nucleation')):
        raise ValueError("split_chemistry needs so2_chemistry before nucleation "
                         f"in processes, got {list(processes)}")

    def apply(process, Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
              kwargs):
        if process == 'so2_chemistry':
            oh_conc = kwargs.get('oh_conc', 0.0)
            Gc = so2_oxidation_step(Gc, temp, pres, boxvol, dt, oh_conc, rh)

        elif process == 'nucleation':
            # Adaptive nucleation sub-stepping
            # Rate estimate always uses Ricco+Dunne (fast, conservative)
            fn = ricco_dunne_nucleation_rate(
                Gc, temp, pres, boxvol,
                kwargs.get('org_conc', 0.0), kwargs.get('nh3_conc', 0.0),
                kwargs.get('fion', 0.0),
                kwargs.get('enable_organic', 1.0),
                kwargs.get('enable_inorganic', 1.0),
                kwargs.get('fn_scale', 1.0),
                kwargs.get('nucl_params'),
                dunne_form,
            )
            N_total = jnp.sum(Nk)
            n_nuc = compute_nucleation_substeps(
                fn, boxvol, dt, N_total,
                max_nucleation_frac, max_nuc_substeps,
            )
            dt_nuc = dt / n_nuc

            if use_zhao:
                enable_masks = kwargs.get('enable_masks', ZHAO2024_ALL_ENABLED)

                def nuc_body(i, carry):
                    Nk_s, Mk_s, Gc_s = carry
                    Nk_s, Mk_s, Gc_s = zhao2024_nucleation_step(
                        Nk_s, Mk_s, Gc_s, xk, temp, pres, boxvol, dt_nuc,
                        kwargs.get('org_conc', 0.0),
                        kwargs.get('nh3_conc', 0.0),
                        kwargs.get('fion', 0.0),
                        hno3=kwargs.get('hno3', 0.0),
                        ulvoc=kwargs.get('ulvoc', 0.0),
                        dma=kwargs.get('dma', 0.0),
                        hio3=kwargs.get('hio3', 0.0),
                        enable_masks=enable_masks,
                        fn_scale=kwargs.get('fn_scale', 1.0),
                    )
                    Nk_s, Mk_s = mnfix_jax(Nk_s, Mk_s, xk, ICOMP_NODIAG,
                                           bounded=bounded)
                    return (Nk_s, Mk_s, Gc_s)
            else:
                def nuc_body(i, carry):
                    Nk_s, Mk_s, Gc_s = carry
                    Nk_s, Mk_s, Gc_s = nucleation_step(
                        Nk_s, Mk_s, Gc_s, xk, temp, pres, boxvol, dt_nuc,
                        kwargs.get('org_conc', 0.0),
                        kwargs.get('nh3_conc', 0.0),
                        kwargs.get('fion', 0.0),
                        kwargs.get('enable_organic', 1.0),
                        kwargs.get('enable_inorganic', 1.0),
                        kwargs.get('fn_scale', 1.0),
                        kwargs.get('nucl_params'),
                        dunne_form,
                    )
                    Nk_s, Mk_s = mnfix_jax(Nk_s, Mk_s, xk, ICOMP_NODIAG,
                                           bounded=bounded)
                    return (Nk_s, Mk_s, Gc_s)

            Nk, Mk, Gc = loop_fori(
                0, n_nuc, nuc_body, (Nk, Mk, Gc), bounded,
                max_iter=max_nuc_substeps)
            if water_every_process:
                Mk = equilibrate_water(Mk, rh, temp, water_scheme)

        elif process == 'coagulation':
            if coag_solver == 'euler_adaptive':
                Nk, Mk = euler_step(
                    Nk, Mk, xk, temp, pres, boxvol, dt,
                    icomp_nodiag=kwargs.get('icomp_nodiag', ICOMP_NODIAG),
                    recompute_kernel=True,
                    max_substeps=coag_max_substeps,
                    sparse_density=sparse_density,
                    bounded=bounded,
                )
            else:
                Nk, Mk = coag_euler_step(
                    Nk, Mk, xk, temp, pres, boxvol,
                    dt=dt,
                    icomp_nodiag=kwargs.get('icomp_nodiag', ICOMP_NODIAG),
                    n_substeps=n_coag_substeps,
                    sparse_density=sparse_density,
                    bounded=bounded,
                )
            if water_every_process:
                Mk = equilibrate_water(Mk, rh, temp, water_scheme)
        elif process == 'condensation':
            Nk, Mk, Gc = _condensation_step_core(
                Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                ezcond_fn=ezcond_fn, water_scheme=water_scheme,
                sparse_density=sparse_density, bounded=bounded,
            )
        elif process == 'dilution':
            kdil = kwargs.get('kdil', 0.0)
            Nk, Mk, Gc = dilution_step(
                Nk, Mk, Gc, dt, kdil,
                kwargs.get('Nk_bg', jnp.zeros_like(Nk)),
                kwargs.get('Mk_bg', jnp.zeros_like(Mk)),
                kwargs.get('Gc_bg', jnp.zeros_like(Gc)),
            )
            if water_every_process:
                Mk = equilibrate_water(Mk, rh, temp, water_scheme)
        return Nk, Mk, Gc

    gas_procs = [p for p in processes if p in _GAS_PROCESSES]
    other_procs = [p for p in processes if p not in _GAS_PROCESSES]
    # Half the sub-step's chemistry before nucleation, half after: with all
    # of it before, nucleation sees the top of the H2SO4 sawtooth,
    # C*(1 + CS dt/2), and J ~ C^n is n CS dt/2 high (11% at 240 K / 10 ppmv,
    # f = 0.1). Split, the bias is O((CS dt)^2).
    split_chem = ('so2_chemistry' in gas_procs and 'nucleation' in gas_procs
                  and gas_procs.index('so2_chemistry')
                  < gas_procs.index('nucleation'))

    def _adaptive_gas_step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                           kwargs):
        """Cover dt with coagulation intervals of adaptive gas sub-steps.

        Gas sub-steps (choose_gas_substep) run until the interval target is
        reached or nucleation has added coag_interval_frac * N. Then the
        other processes (coagulation, dilution) run on the interval. The
        next target shrinks if coagulation removed more than
        coag_interval_frac * N, and grows at most x2 otherwise.
        Forward mode: nested while_loops (intervals, gas sub-steps).
        Reverse mode: one bounded loop over gas sub-steps, ending an
        interval with lax.cond; same operations in the same order.
        """
        nuc_kwargs = {k: kwargs.get(k, d) for k, d in (
            ('org_conc', 0.0), ('nh3_conc', 0.0), ('fion', 0.0),
            ('enable_organic', 1.0), ('enable_inorganic', 1.0),
            ('fn_scale', 1.0), ('nucl_params', None))}
        oh_conc = kwargs.get('oh_conc', 0.0)
        dt = jnp.asarray(dt, dtype=jnp.float64)
        eps_t = 1e-12 * dt

        def gas_substep_once(Nk_s, Mk_s, Gc_s, t, t_end, n, dN, diag):
            dt_min, n_floor, knuc_dt_max = diag
            dt_sub, floored, knuc_dt = choose_gas_substep(
                Nk_s, Mk_s, Gc_s, xk, temp, pres, boxvol, rh, alpha, t_end - t,
                oh_conc, nuc_kwargs, frac=gas_substep_frac,
                min_dt=gas_min_dt, has_chem='so2_chemistry' in gas_procs,
                has_nuc='nucleation' in gas_procs,
                sparse_density=sparse_density, dunne_form=dunne_form)
            # The last allowed sub-step covers the rest of the outer step
            dt_sub = jnp.where(n == gas_max_substeps - 1, dt - t, dt_sub)
            N_before = jnp.sum(Nk_s)
            for process in gas_procs:
                dt_p = 0.5 * dt_sub if (split_chem and
                                        process == 'so2_chemistry') else dt_sub
                Nk_s, Mk_s, Gc_s = apply(process, Nk_s, Mk_s, Gc_s, xk, temp,
                                         pres, boxvol, rh, alpha, dt_p,
                                         kwargs)
                if split_chem and process == 'nucleation':
                    Nk_s, Mk_s, Gc_s = apply(
                        'so2_chemistry', Nk_s, Mk_s, Gc_s, xk, temp, pres,
                        boxvol, rh, alpha, 0.5 * dt_sub, kwargs)
            dN = dN + jnp.maximum(jnp.sum(Nk_s) - N_before, 0.0)
            diag = (jnp.minimum(dt_min, dt_sub), n_floor + floored,
                    jnp.maximum(knuc_dt_max, knuc_dt))
            return Nk_s, Mk_s, Gc_s, t + dt_sub, n + 1, dN, diag

        def end_interval(Nk_s, Mk_s, Gc_s, t_int, target):
            r_coag = jnp.float64(0.0)
            for process in other_procs:
                N_before = jnp.sum(Nk_s)
                Nk_s, Mk_s, Gc_s = apply(process, Nk_s, Mk_s, Gc_s, xk, temp,
                                         pres, boxvol, rh, alpha, t_int,
                                         kwargs)
                if process == 'coagulation':
                    r_coag = jax.lax.stop_gradient(
                        (N_before - jnp.sum(Nk_s)) / jnp.maximum(N_before, 1e-30))
            shrunk = (t_int * coag_interval_frac
                      / jnp.where(r_coag > 0.0, r_coag, 1.0))
            # Rounded down to a power of 2 s: a target continuous in the
            # inputs would move the interval (and gas sub-step) lengths
            # smoothly, and gradients cannot see that (stop_gradient)
            shrunk = 2.0 ** jnp.floor(jnp.log2(jnp.maximum(shrunk, 1e-30)))
            target = jnp.minimum(
                2.0 * target,
                jnp.where(r_coag > coag_interval_frac, shrunk, 2.0 * target))
            return Nk_s, Mk_s, Gc_s, target

        def interval_open(t, t_end, n, dN, N0):
            return ((t_end - t > eps_t) & (n < gas_max_substeps)
                    & (dN < coag_interval_frac * N0))

        def n0(Nk_s):
            return jax.lax.stop_gradient(jnp.sum(Nk_s))

        diag0 = (dt, jnp.int32(0), jnp.float64(0.0))
        if not bounded:
            def gas_cond(state):
                return interval_open(state[3], state[4], state[5], state[6],
                                     state[7])

            def gas_body(state):
                Nk_s, Mk_s, Gc_s, t, t_end, n, dN, N0, diag = state
                Nk_s, Mk_s, Gc_s, t, n, dN, diag = gas_substep_once(
                    Nk_s, Mk_s, Gc_s, t, t_end, n, dN, diag)
                return Nk_s, Mk_s, Gc_s, t, t_end, n, dN, N0, diag

            def interval_body(state):
                Nk_s, Mk_s, Gc_s, t, target, n, n_coag, diag = state
                t_start = t
                t_end = t + jnp.minimum(target, dt - t)
                (Nk_s, Mk_s, Gc_s, t, _, n, _, _, diag) = jax.lax.while_loop(
                    gas_cond, gas_body,
                    (Nk_s, Mk_s, Gc_s, t, t_end, n, jnp.float64(0.0),
                     n0(Nk_s), diag))
                Nk_s, Mk_s, Gc_s, target = end_interval(
                    Nk_s, Mk_s, Gc_s, t - t_start, target)
                return Nk_s, Mk_s, Gc_s, t, target, n, n_coag + 1, diag

            init = (Nk, Mk, Gc, jnp.float64(0.0), dt, jnp.int32(0),
                    jnp.int32(0), diag0)
            Nk, Mk, Gc, _, _, n_sub, n_coag, diag = jax.lax.while_loop(
                lambda s: dt - s[3] > eps_t, interval_body, init)
        else:
            def flat_body(state):
                (Nk_s, Mk_s, Gc_s, t, t_start, t_end, target, n, n_coag, dN,
                 N0, diag) = state
                Nk_s, Mk_s, Gc_s, t, n, dN, diag = gas_substep_once(
                    Nk_s, Mk_s, Gc_s, t, t_end, n, dN, diag)

                def close(c):
                    Nk_c, Mk_c, Gc_c, target_c = end_interval(
                        c[0], c[1], c[2], t - t_start, c[3])
                    t_end_c = t + jnp.minimum(target_c, dt - t)
                    return (Nk_c, Mk_c, Gc_c, target_c, t, t_end_c,
                            n_coag + 1, jnp.float64(0.0), n0(Nk_c))

                def keep(c):
                    return (c[0], c[1], c[2], c[3], t_start, t_end, n_coag,
                            dN, N0)

                (Nk_s, Mk_s, Gc_s, target, t_start, t_end, n_coag, dN,
                 N0) = jax.lax.cond(
                    interval_open(t, t_end, n, dN, N0), keep, close,
                    (Nk_s, Mk_s, Gc_s, target))
                return (Nk_s, Mk_s, Gc_s, t, t_start, t_end, target, n,
                        n_coag, dN, N0, diag)

            init = (Nk, Mk, Gc, jnp.float64(0.0), jnp.float64(0.0), dt, dt,
                    jnp.int32(0), jnp.int32(0), jnp.float64(0.0), n0(Nk),
                    diag0)
            (Nk, Mk, Gc, _, _, _, _, n_sub, n_coag, _, _,
             diag) = bounded_while(lambda s: dt - s[3] > eps_t, flat_body,
                                   init, gas_max_substeps)
        dt_min, n_floor, knuc_dt_max = diag
        info = dict(n_sub=n_sub, n_coag=n_coag, dt_min=dt_min,
                    hit_cap=n_sub >= gas_max_substeps, n_floor=n_floor,
                    knuc_dt_max=knuc_dt_max)
        return Nk, Mk, Gc, info

    def step_fn(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, **kwargs):
        unknown = set(kwargs) - _VALID_MAKE_STEP_KWARGS
        if unknown:
            warnings.warn(
                f"Unknown kwargs passed to make_step step function: {unknown}. "
                f"These will be silently ignored. "
                f"Valid kwargs: {sorted(_VALID_MAKE_STEP_KWARGS)}",
                UserWarning, stacklevel=2,
            )
        if gas_substep is None:
            for process in processes:
                dt_p = 0.5 * dt if (split_chemistry and
                                    process == 'so2_chemistry') else dt
                Nk, Mk, Gc = apply(process, Nk, Mk, Gc, xk, temp, pres, boxvol,
                                   rh, alpha, dt_p, kwargs)
                if split_chemistry and process == 'nucleation':
                    Nk, Mk, Gc = apply('so2_chemistry', Nk, Mk, Gc, xk, temp,
                                       pres, boxvol, rh, alpha, 0.5 * dt, kwargs)
            return Nk, Mk, Gc
        Nk, Mk, Gc, info = _adaptive_gas_step(
            Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, kwargs)
        if return_substep_info:
            return Nk, Mk, Gc, info
        return Nk, Mk, Gc
    return jax.jit(step_fn) if jit else step_fn
