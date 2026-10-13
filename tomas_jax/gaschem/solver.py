"""Stiff integration of one coupling interval with frozen rate constants (diffrax)."""
from __future__ import annotations

import diffrax as dfx
import equinox as eqx
import jax
import jax.numpy as jnp

# float64 enforced by tomas_jax.core.config (imported by the tomas_jax package)
from .config import SOLVER, SOLVER_ATOL, SOLVER_MAX_STEPS, SOLVER_RTOL
from .mechanism import Mechanism

_SOLVERS = {"kvaerno5": dfx.Kvaerno5, "kvaerno3": dfx.Kvaerno3}


def integrate_interval(
    mech: Mechanism,
    y0: jax.Array,
    k: jax.Array,
    dt: jax.Array,
    rtol: float = SOLVER_RTOL,
    atol: float = SOLVER_ATOL,
    solver: str = SOLVER,
    max_steps: int = SOLVER_MAX_STEPS,
) -> tuple[jax.Array, jax.Array]:
    """Integrate dy/dt = mech.rhs(y, k) over [0, dt] with k held fixed.

    Args:
        mech: mechanism index arrays.
        y0: concentrations [molec/cm3], shape (n_species,).
        k: rate constants for this interval, shape (n_reactions,).
        dt: interval length [s].
        max_steps: budget of accepted + rejected steps for this interval.

    Returns:
        (y [molec/cm3] clipped at zero, number of accepted steps).

    Raises:
        RuntimeError: the solve did not reach dt, e.g. ``max_steps`` ran out. The
            message leads with diffrax's reason (not a JIT stack trace).
    """
    y, nsteps, result = _integrate(mech, y0, k, dt, rtol, atol, solver, max_steps)
    if result != dfx.RESULTS.successful:
        raise RuntimeError(f"gas-phase chemistry solver failed on a {float(dt):g} s coupling "
                           f"interval (max_steps={max_steps}): {dfx.RESULTS[result]}")
    return y, nsteps


@eqx.filter_jit
def _integrate(mech, y0, k, dt, rtol, atol, solver, max_steps):
    term = dfx.ODETerm(lambda t, y, args: mech.rhs(y, args))
    sol = dfx.diffeqsolve(
        term,
        _SOLVERS[solver](),
        t0=0.0,
        t1=dt,
        dt0=None,
        y0=y0,
        args=k,
        stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol),
        saveat=dfx.SaveAt(t1=True),
        max_steps=max_steps,
        throw=False,                    # checked by integrate_interval
    )
    # KPP-style: tiny negative overshoots from the implicit solver are clipped.
    return jnp.maximum(sol.ys[-1], 0.0), sol.stats["num_accepted_steps"], sol.result
