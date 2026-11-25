"""Forward Euler integrator — exact port of multicoag.f time-stepping.

This module provides a FORTRAN-equivalent Forward Euler solver with
adaptive sub-stepping for validation against the original FORTRAN code.

The FORTRAN multicoag.f uses:
  - Forward Euler: y(t+dt) = y(t) + dt * f(y(t))
  - Adaptive dt: dtlimit = 0.25 * Nk / |dNdt|, itlimit = 10 * Mk / |dMdt|
  - Sub-stepping: dt = min(dtlimit, itlimit, remaining_time)
  - MNFIX after each sub-step

For production use, prefer diffrax.py (Tsit5 adaptive RK) which is:
  - Higher-order accurate (5th order vs 1st order)
  - Better error control (embedded error estimator)
  - More efficient (larger stable timesteps)
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp
import numpy as np
from functools import partial
from typing import Tuple, Optional

from ..physics.properties import calc_particle_properties
from ..physics.coagulation_kernel import calc_coagulation_kernel
from ..physics.coagulation_rates import calc_coagulation_rates
from ..core.mnfix_jax import mnfix_jax
from ..core.mnfix_fortran import mnfix_fortran
from ..core.bounded_loop import loop_while
from ..core.config import ICOMP_NODIAG, TINY_N, TINY_M


def check_coag_kernel_scale(coag_kernel_scale):
    """Raise ValueError for a concrete coag_kernel_scale that is < 0 or
    not finite. A traced value (inside jit/vmap) cannot be checked and is
    passed through, so the check never breaks tracing."""
    if isinstance(coag_kernel_scale, jax.core.Tracer):
        return
    v = np.asarray(coag_kernel_scale)
    if not np.all(np.isfinite(v) & (v >= 0.0)):
        raise ValueError(
            f"coag_kernel_scale must be finite and >= 0, got {coag_kernel_scale!r}")


def _compute_adaptive_dt(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    dNdt: jnp.ndarray,
    dMdt: jnp.ndarray,
    dt_remaining: float,
    icomp_nodiag: int = ICOMP_NODIAG,
    n_floor=None,
) -> jnp.ndarray:
    """Compute adaptive timestep matching FORTRAN multicoag.f logic.

    FORTRAN:
        dtlimit = 0.25 * Nk(k) / max(abs(dNdt(k)), 1e-30)
        itlimit = 10.0 * Mk(k,j) / max(abs(dMdt(k,j)), 1e-30)
        dt = min(dtlimit, itlimit) over all k,j

    Args:
        Nk: Number concentrations, shape (ibins,)
        Mk: Mass concentrations, shape (ibins, icomp)
        dNdt: Number rates, shape (ibins,)
        dMdt: Mass rates, shape (ibins, icomp)
        dt_remaining: Time remaining in the outer step [s]
        icomp_nodiag: Number of non-diagnostic species

    Returns:
        dt: Adaptive sub-step size [s]
    """
    # Number-based limit: dt < 0.25 * N / |dN/dt|
    # Mask out empty bins (Fortran skips bins with Nk < Neps)
    abs_dNdt = jnp.maximum(jnp.abs(dNdt), 1e-30)
    n_cut = TINY_N if n_floor is None else jnp.maximum(n_floor, TINY_N)
    dtlimit = jnp.where(Nk > n_cut, 0.25 * jnp.abs(Nk) / abs_dNdt, jnp.inf)
    dt_N = jnp.min(dtlimit)

    # Mass-based limit: dt < 10.0 * M / |dM/dt|
    # Mask out near-zero mass entries
    Mk_prog = Mk[:, :icomp_nodiag]
    dMdt_prog = dMdt[:, :icomp_nodiag]
    abs_dMdt = jnp.maximum(jnp.abs(dMdt_prog), 1e-30)
    itlimit = jnp.where(Mk_prog > TINY_M, 10.0 * jnp.abs(Mk_prog) / abs_dMdt, jnp.inf)
    dt_M = jnp.min(itlimit)

    # Take minimum of all constraints
    dt = jnp.minimum(dt_N, dt_M)
    dt = jnp.minimum(dt, dt_remaining)

    # Safety floor to prevent infinitely small steps
    dt = jnp.maximum(dt, 1e-10)

    return dt


def euler_step(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    temp: float,
    pres: float,
    boxvol: float,
    dt: float,
    icomp_nodiag: int = ICOMP_NODIAG,
    use_fortran_mnfix: bool = False,
    recompute_kernel: bool = False,
    max_substeps: int = 10000,
    return_info: bool = False,
    sparse_density=None,
    bounded: bool = False,
    mnfix_trim: bool = False,
    return_dt_left: bool = False,
    n_floor=None,
    coag_kernel_scale=1.0,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Forward Euler integration of coagulation over timestep dt.

    Matches the FORTRAN multicoag.f approach:
    1. Compute particle properties and coagulation kernel
    2. Compute rates (dNdt, dMdt)
    3. Determine adaptive sub-step dt_sub
    4. Apply Euler update: N += dt_sub * dNdt, M += dt_sub * dMdt
    5. Apply MNFIX
    6. Repeat until dt is consumed

    Args:
        Nk: Number concentration [#/grid cell], shape (ibins,)
        Mk: Mass concentration [kg/grid cell], shape (ibins, icomp)
        xk: Bin boundaries [kg], shape (ibins+1,)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm^3]
        dt: Total integration time [s]
        icomp_nodiag: Number of non-diagnostic species
        use_fortran_mnfix: If True, use FORTRAN-equivalent mnfix
        recompute_kernel: If True, recompute kernel each sub-step (slower but more accurate)
        max_substeps: Maximum number of sub-steps
        return_info: If True, returns (Nk, Mk, n_substeps_used) so callers
            can detect when max_substeps was hit (integration truncated).
        sparse_density: Density [kg/m3] for low-mass bins in the kernel
            (see calc_particle_properties). None = 1000 (Fortran).
        bounded: If True (static), the sub-step loop is a bounded loop of
            max_substeps iterations (core/bounded_loop.py), so reverse-mode
            AD works.
        mnfix_trim: Static; mnfix_jax(trim=...). True is faster but
            forward-mode AD only.
        n_floor: Opt-in (None = Fortran, bit-identical). Bins with
            Nk <= n_floor [#/cell] are left out of the number limit
            0.25 N/|dN/dt| (the mask is Nk > TINY_N otherwise), so a nearly
            empty bin that MNFIX keeps resetting to its 1e-5 floor cannot
            collapse the sub-step. Mass limit unchanged.
        return_dt_left: Static; if True, also return the time [s] left
            uncovered when max_substeps bound (0 otherwise). That part of
            dt gets no coagulation.
        coag_kernel_scale: Multiplier on the Brownian coagulation kernel
            (sensitivity knob, default 1.0 = unscaled and bit-identical),
            applied to the initial kernel and to every recomputed one. A
            scalar or 0-d array, may be traced (no recompile on a new value);
            a concrete value < 0 or non-finite raises ValueError.

    Returns:
        Nk_final: Updated number concentration
        Mk_final: Updated mass concentration
        n_substeps: (only if return_info=True) Number of substeps used
        dt_left: (only if return_dt_left=True, after n_substeps) time [s]
            not integrated because max_substeps bound
    """
    # Select MNFIX function
    mnfix_fn = (mnfix_fortran if use_fortran_mnfix
                else partial(mnfix_jax, trim=mnfix_trim))

    check_coag_kernel_scale(coag_kernel_scale)

    # Initial kernel computation
    Dpk, Dk, ck = calc_particle_properties(Nk, Mk, temp, pres, sparse_density)
    kij = calc_coagulation_kernel(Dpk, Dk, ck, boxvol) * coag_kernel_scale

    # State for while_loop: (Nk, Mk, kij, time_elapsed, substep_count)
    def cond_fn(state):
        _, _, _, t_elapsed, n_steps = state
        return (t_elapsed < dt - 1e-15) & (n_steps < max_substeps)

    def body_fn(state):
        Nk_s, Mk_s, kij_s, t_elapsed, n_steps = state

        # Optionally recompute kernel (FORTRAN recomputes each sub-step)
        def recompute():
            Dpk_new, Dk_new, ck_new = calc_particle_properties(
                Nk_s, Mk_s, temp, pres, sparse_density)
            return (calc_coagulation_kernel(Dpk_new, Dk_new, ck_new, boxvol)
                    * coag_kernel_scale)

        kij_use = jax.lax.cond(
            recompute_kernel & (n_steps > 0),
            recompute,
            lambda: kij_s
        )

        # Compute rates
        dNdt, dMdt, _overflow = calc_coagulation_rates(Nk_s, Mk_s, kij_use, xk, icomp_nodiag)

        # Adaptive timestep
        dt_remaining = dt - t_elapsed
        dt_sub = _compute_adaptive_dt(
            Nk_s, Mk_s, dNdt, dMdt, dt_remaining, icomp_nodiag,
            n_floor=n_floor,
        )

        # Forward Euler update
        Nk_new = Nk_s + dt_sub * dNdt
        Mk_new = Mk_s + dt_sub * dMdt

        # Enforce positivity
        Nk_new = jnp.maximum(Nk_new, 0.0)
        Mk_new = jnp.maximum(Mk_new, 0.0)

        # Apply MNFIX
        Nk_new, Mk_new = mnfix_fn(Nk_new, Mk_new, xk, icomp_nodiag)

        return Nk_new, Mk_new, kij_use, t_elapsed + dt_sub, n_steps + 1

    init_state = (Nk, Mk, kij, jnp.float64(0.0), jnp.int32(0))
    Nk_final, Mk_final, _, t_final, n_substeps = loop_while(
        cond_fn, body_fn, init_state, bounded, max_iter=max_substeps)

    out = (Nk_final, Mk_final)
    if return_info:
        out += (n_substeps,)
    if return_dt_left:
        # The loop stops early only at the cap; the rest of dt is dropped
        dt_left = jnp.where(t_final < dt - 1e-15, dt - t_final, 0.0)
        out += (jax.lax.stop_gradient(dt_left),)
    return out


def heun_step(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    temp: float,
    pres: float,
    boxvol: float,
    dt: float,
    icomp_nodiag: int = ICOMP_NODIAG,
    rtol: float = 3e-3,
    atol_frac: float = 1e-6,
    max_substeps: int = 10000,
    sparse_density=None,
    bounded: bool = False,
    mnfix_trim: bool = False,
    return_info: bool = False,
    return_dt_left: bool = False,
    h0=None,
    safety: float = 0.9,
    fac_max: float = 5.0,
    coag_kernel_scale=1.0,
):
    """Error-controlled Heun (explicit trapezoid) coagulation over dt.

    Not in Fortran multicoag.f. Each iteration computes the kernel at the
    current state (frozen over both stages), the Euler predictor N1 and the
    Heun update N + h/2 (f0 + f1), then MNFIX on the Heun result. The error
    estimate is the Euler-Heun difference h/2 |f1 - f0| per bin, scaled by
    atol_frac * sum(N) + rtol * N: it is an absolute error against the total,
    so a nearly empty bin that MNFIX keeps resetting has no weight and cannot
    collapse the step (the failure of the Fortran limit 0.25 N/|dN/dt|, see
    docs/gas_substep.md). h is rejected and retried when the scaled error
    exceeds 1; the next h = h * clip(safety / sqrt(err), 0.2, fac_max).
    Second order in h (the Fortran Euler step is first order). Dry mass
    closes through the top-bin overflow as in euler_step.

    max_substeps bounds the iterations (accepted + rejected); return_info
    gives the accepted count, return_dt_left the time not integrated when
    the cap bound. bounded: static, bounded loop for reverse-mode AD.
    h0: first trial step [s]; None (default) = dt, so a short interval is
    covered by one step. (h0 = 1 s needed 3 iterations on a 12 s interval,
    1 s then 5 s then the rest: docs/gas_substep.md.)
    coag_kernel_scale: multiplier on the kernel computed every iteration
    (default 1.0, bit-identical; may be traced; concrete < 0 raises). The
    controller sees the scaled rates, so it takes ~1/scale times the steps.
    """
    check_coag_kernel_scale(coag_kernel_scale)

    def kernel(N, M):
        Dpk, Dk, ck = calc_particle_properties(N, M, temp, pres, sparse_density)
        return calc_coagulation_kernel(Dpk, Dk, ck, boxvol) * coag_kernel_scale

    def cond_fn(state):
        t, n_it = state[2], state[5]
        return (t < dt - 1e-12) & (n_it < max_substeps)

    def body_fn(state):
        N, M, t, h, n_acc, n_it = state
        h = jnp.minimum(h, dt - t)
        kij = kernel(N, M)
        dN0, dM0, _ = calc_coagulation_rates(N, M, kij, xk, icomp_nodiag)
        N1 = jnp.maximum(N + h * dN0, 0.0)
        M1 = jnp.maximum(M + h * dM0, 0.0)
        dN1, dM1, _ = calc_coagulation_rates(N1, M1, kij, xk, icomp_nodiag)
        Nh = jnp.maximum(N + 0.5 * h * (dN0 + dN1), 0.0)
        Mh = jnp.maximum(M + 0.5 * h * (dM0 + dM1), 0.0)
        atol = jnp.maximum(atol_frac * jnp.sum(N), TINY_N)
        err = jnp.max(0.5 * h * jnp.abs(dN1 - dN0)
                      / (atol + rtol * jnp.maximum(N, 1e-300)))
        ok = (err <= 1.0) | (h < 1e-9)
        Nn, Mn = mnfix_jax(Nh, Mh, xk, icomp_nodiag, trim=mnfix_trim)
        fac = jnp.clip(safety * jnp.maximum(err, 1e-10) ** -0.5, 0.2, fac_max)
        return (jnp.where(ok, Nn, N), jnp.where(ok, Mn, M),
                jnp.where(ok, t + h, t), h * fac, n_acc + ok.astype(jnp.int32),
                n_it + 1)

    init = (Nk, Mk, jnp.float64(0.0), jnp.minimum(jnp.float64(dt if h0 is None else h0), dt),
            jnp.int32(0), jnp.int32(0))
    N, M, t, _, n_acc, _ = loop_while(cond_fn, body_fn, init, bounded,
                                      max_iter=max_substeps)
    out = (N, M)
    if return_info:
        out += (n_acc,)
    if return_dt_left:
        out += (jax.lax.stop_gradient(jnp.where(t < dt - 1e-12, dt - t, 0.0)),)
    return out


def euler_single_step(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    kij: jnp.ndarray,
    xk: jnp.ndarray,
    dt_sub: float,
    icomp_nodiag: int = ICOMP_NODIAG
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """Single Forward Euler step without MNFIX (for benchmarking level 9).

    Args:
        Nk: Number concentration, shape (ibins,)
        Mk: Mass concentration, shape (ibins, icomp)
        kij: Pre-computed kernel, shape (ibins, ibins)
        xk: Bin boundaries, shape (ibins+1,)
        dt_sub: Sub-step size [s]
        icomp_nodiag: Number of non-diagnostic species

    Returns:
        Nk_new, Mk_new after one Euler step
    """
    dNdt, dMdt, _overflow = calc_coagulation_rates(Nk, Mk, kij, xk, icomp_nodiag)

    Nk_new = Nk + dt_sub * dNdt
    Mk_new = Mk + dt_sub * dMdt

    Nk_new = jnp.maximum(Nk_new, 0.0)
    Mk_new = jnp.maximum(Mk_new, 0.0)

    return Nk_new, Mk_new
