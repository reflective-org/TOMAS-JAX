"""Attribution only: hold species constant *inside* the gas solver, as PyCHAM does.

TOMAS-JAX's ``hold_constant`` resets the species to its initial value after every
coupling interval; within the interval it evolves freely (driver step 5). PyCHAM's
``const_comp`` sets its rate of change (and Jacobian row) to zero inside the ODE, so it
never moves. ``held_in_solver`` reproduces PyCHAM's semantics in a TOMAS-JAX run by
masking the species' derivative in a subclass of the mechanism, so the difference the
two conventions make can be measured.

Benchmark-side monkeypatch of the name imported into ``tomas_jax.box.driver`` (like
``ours.numerics_override``, and composable with it); model code and defaults are
untouched and restored on exit.
"""
from __future__ import annotations

import contextlib
import dataclasses

import jax
import jax.numpy as jnp
import numpy as np

from tomas_jax.box import driver as _driver
from tomas_jax.gaschem.mechanism import Mechanism


class HeldMechanism(Mechanism):
    """Mechanism whose RHS is zero for the held species (mask 0), unchanged elsewhere."""
    hold_mask: jax.Array

    def rhs(self, y: jax.Array, k: jax.Array) -> jax.Array:
        return Mechanism.rhs(self, y, k) * self.hold_mask


def held_mechanism(species: tuple[str, ...]) -> tuple[Mechanism, HeldMechanism]:
    parsed, _, mech = _driver._chemistry()
    mask = np.ones(mech.n_species)
    mask[[parsed.species.index(s) for s in species]] = 0.0
    fields = {f.name: getattr(mech, f.name) for f in dataclasses.fields(mech)}
    return mech, HeldMechanism(**fields, hold_mask=jnp.asarray(mask))


@contextlib.contextmanager
def held_in_solver(species: tuple[str, ...]):
    """Inside the block, driver.run integrates with d[species]/dt = 0 (PyCHAM const_comp)."""
    mech, held = held_mechanism(species)
    inner = _driver.integrate_interval          # may already be numerics_override's partial

    def integrate(m, y0, k, dt, **kw):
        if m is not mech:
            raise RuntimeError("held_in_solver: unexpected mechanism object")
        return inner(held, y0, k, dt, **kw)

    _driver.integrate_interval = integrate
    try:
        yield
    finally:
        _driver.integrate_interval = inner
