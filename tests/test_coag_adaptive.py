"""make_step(coag_solver='euler_adaptive'): Fortran multicoag.f adaptive Euler."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import NBINS, ICOMP, N_GAS_SPECIES, SRTSO4, SRTH2O
from tomas_jax.physics.nucleation import DEFAULT_NUCLEATION_PARAMS
from tomas_jax.solvers.condensation import make_step
from tomas_jax.solvers.diffrax import coag_euler_step

BOXVOL = 1e6
ENV = (190.0, 5500.0, BOXVOL, 0.02, 1.0, 60.0)   # temp, pres, boxvol, rh, alpha, dt


def _state(xk, n_cm3_by_bin):
    Nk = jnp.full(NBINS, 1e-3)
    for k, n in n_cm3_by_bin.items():
        Nk = Nk.at[k].set(n * BOXVOL)
    Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(Nk * jnp.sqrt(xk[:-1] * xk[1:]))
    return Nk, Mk, jnp.zeros(N_GAS_SPECIES)


def _dry(Mk):
    return float(jnp.sum(Mk[:, :SRTH2O]))


def test_default_is_fixed_euler(xk):
    """Default = 'euler_fixed' exactly, and equal to coag_euler_step up to
    XLA round-off (traced vs closed-over temp/pres fold differently)."""
    Nk, Mk, Gc = _state(xk, {0: 1e4, 20: 10.0})
    out = make_step(['coagulation'])(Nk, Mk, Gc, xk, *ENV)
    same = make_step(['coagulation'], coag_solver='euler_fixed')(Nk, Mk, Gc, xk, *ENV)
    for a, b in zip(out, same):
        np.testing.assert_array_equal(np.array(a), np.array(b))
    ref = coag_euler_step(Nk, Mk, xk, ENV[0], ENV[1], BOXVOL, dt=60.0, n_substeps=10)
    np.testing.assert_allclose(np.array(out[0]), np.array(ref[0]), rtol=1e-13)
    np.testing.assert_allclose(np.array(out[1]), np.array(ref[1]), rtol=1e-13, atol=1e-30)


def test_unknown_solver_rejected():
    with pytest.raises(ValueError, match='coag_solver'):
        make_step(['coagulation'], coag_solver='rk4')


def test_adaptive_stable_where_fixed_fails(xk):
    """~3e9 cm-3 of 1.7 nm clusters (the Phase 2 crash state at 1000 ppmv SO2)."""
    Nk, Mk, Gc = _state(xk, {0: 3.3e9, 20: 10.0})
    fixed = make_step(['coagulation'])(Nk, Mk, Gc, xk, *ENV)
    assert not np.all(np.isfinite(np.array(fixed[0])))   # documents the failure
    ad = make_step(['coagulation'], coag_solver='euler_adaptive')(Nk, Mk, Gc, xk, *ENV)
    assert np.all(np.isfinite(np.array(ad[0]))) and np.all(np.isfinite(np.array(ad[1])))
    assert float(jnp.sum(ad[0])) < float(jnp.sum(Nk)) * 0.1   # strong self-coagulation
    assert abs(_dry(ad[1]) / _dry(Mk) - 1.0) < 1e-6


def test_adaptive_matches_fixed_when_fixed_is_converged(xk):
    """Moderate N: kernel barely changes within 60 s, so the frozen-kernel
    fixed scheme with many sub-steps and the adaptive scheme agree."""
    Nk, Mk, Gc = _state(xk, {6: 1e4, 20: 10.0})
    fixed = make_step(['coagulation'], n_coag_substeps=200)(Nk, Mk, Gc, xk, *ENV)
    ad = make_step(['coagulation'], coag_solver='euler_adaptive')(Nk, Mk, Gc, xk, *ENV)
    dN_fixed = float(jnp.sum(Nk) - jnp.sum(fixed[0]))
    dN_ad = float(jnp.sum(Nk) - jnp.sum(ad[0]))
    assert dN_fixed > 0.0
    np.testing.assert_allclose(dN_ad, dN_fixed, rtol=0.02)


def test_forward_mode_through_adaptive(xk):
    """jacfwd w.r.t. nucleation coefficients through nucl + adaptive coag."""
    step = make_step(['nucleation', 'coagulation'], coag_solver='euler_adaptive',
                     jit=False)
    Nk0, Mk0, _ = _state(xk, {20: 10.0})
    Gc0 = jnp.zeros(N_GAS_SPECIES).at[SRTSO4].set(1e8 * BOXVOL * 98e-3 / 6.022e23)

    def n_total(p):
        def body(c, _):
            return step(*c, xk, *ENV, fion=10.0, nucl_params=p), None
        (Nk, _, _), _ = jax.lax.scan(body, (Nk0, Mk0, Gc0), None, length=3)
        return jnp.sum(Nk)

    P = DEFAULT_NUCLEATION_PARAMS
    Pj = jax.tree.map(jnp.float64, P)
    g = jax.jit(jax.jacfwd(n_total))(Pj)
    f = jax.jit(n_total)
    h = 1e-5 * abs(P.u_bi)
    fd = (f(jax.tree.map(jnp.float64, P._replace(u_bi=P.u_bi + h)))
          - f(jax.tree.map(jnp.float64, P._replace(u_bi=P.u_bi - h)))) / (2 * h)
    assert np.isfinite(float(g.u_bi))
    np.testing.assert_allclose(float(g.u_bi), float(fd), rtol=1e-4)
