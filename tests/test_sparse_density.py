"""sparse_density option: sulfate density for low-occupancy bins (stratosphere/SAI)."""
import jax
import jax.numpy as jnp
import numpy as np

from tomas_jax.core.config import NBINS, ICOMP, N_GAS_SPECIES, SRTSO4, AVOGADRO
from tomas_jax.physics.density import calc_density
from tomas_jax.physics.condensation_sink import (
    calc_wet_diameters, calc_condensation_sink, NEPS_CONDSINK,
)
from tomas_jax.physics.properties import calc_particle_properties
from tomas_jax.physics.radiative_forcing import DENSITY_H2SO4_SOLUTION
from tomas_jax.solvers.condensation import make_step

RHO_S = DENSITY_H2SO4_SOLUTION
BOXVOL = 1e6


def _state(xk, n_cm3):
    """Pure-sulfate state with n_cm3 particles per bin."""
    Nk = jnp.full(NBINS, n_cm3 * BOXVOL)
    Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(Nk * jnp.sqrt(xk[:-1] * xk[1:]))
    return Nk, Mk


def test_calc_density_empty_value(xk):
    Nk, Mk = _state(xk, 1.0)
    rho_default = calc_density(Mk)
    rho_s = calc_density(Mk, RHO_S)
    empty = np.array(jnp.sum(Mk, axis=1) < 1e-15)
    assert empty.any() and (~empty).any()
    np.testing.assert_array_equal(np.array(rho_default)[empty], 1000.0)
    np.testing.assert_array_equal(np.array(rho_s)[empty], RHO_S)
    np.testing.assert_array_equal(np.array(rho_s)[~empty], np.array(rho_default)[~empty])


def test_none_is_bit_identical(xk):
    Nk, Mk = _state(xk, 1.0)
    for a, b in [
        (calc_particle_properties(Nk, Mk, 217.0, 5500.0),
         calc_particle_properties(Nk, Mk, 217.0, 5500.0, None)),
        (calc_wet_diameters(Nk, Mk, xk), calc_wet_diameters(Nk, Mk, xk, None)),
    ]:
        for x, y in zip(a, b):
            np.testing.assert_array_equal(np.array(x), np.array(y))


def test_wet_diameters_sparse_bins_only(xk):
    Nk, Mk = _state(xk, 1.0)               # 1e6 #/cell < NEPS_CONDSINK
    Nk = Nk.at[20].set(2.0 * NEPS_CONDSINK)  # one populated bin
    Mk = Mk.at[20, SRTSO4].set(Nk[20] * jnp.sqrt(xk[20] * xk[21]))
    _, rho0 = calc_wet_diameters(Nk, Mk, xk)
    _, rho1 = calc_wet_diameters(Nk, Mk, xk, RHO_S)
    sparse = np.array(Nk) <= NEPS_CONDSINK
    np.testing.assert_array_equal(np.array(rho0)[sparse], 1500.0)
    np.testing.assert_array_equal(np.array(rho1)[sparse], RHO_S)
    assert float(rho1[20]) == float(rho0[20])
    cs0, _ = calc_condensation_sink(Nk, Mk, 217.0, 5500.0, BOXVOL, xk=xk)
    cs1, _ = calc_condensation_sink(Nk, Mk, 217.0, 5500.0, BOXVOL, xk=xk,
                                    sparse_density=RHO_S)
    # denser sparse bins -> smaller diameters -> smaller CS
    assert float(cs1) < float(cs0)


def test_make_step_option(xk):
    Nk, Mk = _state(xk, 1.0)
    Gc = jnp.zeros(N_GAS_SPECIES).at[SRTSO4].set(1e7 * BOXVOL * 98e-3 / AVOGADRO)
    args = (Nk, Mk, Gc, xk, 217.0, 5500.0, BOXVOL, 0.02, 1.0, 60.0)
    procs = ['coagulation', 'condensation']
    base = make_step(procs)(*args)
    same = make_step(procs, sparse_density=None)(*args)
    sulf = make_step(procs, sparse_density=RHO_S)(*args)
    for x, y in zip(base, same):
        np.testing.assert_array_equal(np.array(x), np.array(y))
    assert all(np.all(np.isfinite(np.array(x))) for x in sulf)
    assert not np.array_equal(np.array(base[0]), np.array(sulf[0]))
