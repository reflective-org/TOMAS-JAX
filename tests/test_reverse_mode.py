"""make_step(ad_mode='reverse'): bounded loops (core/bounded_loop.py) for jax.grad."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.bounded_loop import bounded_while, bounded_fori
from tomas_jax.core.config import make_grid, XK0
from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.solvers.condensation import make_step
from sensitivities.dunne_nuc import common as c
from tests.test_mnfix import _random_states

PROCS = ['so2_chemistry', 'nucleation', 'coagulation', 'condensation']
FIELDS = c.nuc.NucleationParams._fields


def test_bounded_loops_match_and_differentiate():
    def body(s):
        i, x = s
        return i + 1, x * 1.5 + jnp.sin(x)

    def cond(s):
        return s[0] < 7

    x0 = jnp.float64(0.3)
    ref = jax.lax.while_loop(cond, body, (0, x0))
    out = bounded_while(cond, body, (0, x0), 10)
    assert int(out[0]) == 7 and float(out[1]) == pytest.approx(float(ref[1]), rel=1e-15)

    f = lambda x: bounded_while(cond, body, (0, x), 10)[1]
    np.testing.assert_allclose(float(jax.grad(f)(x0)), float(jax.jacfwd(f)(x0)),
                               rtol=1e-13)
    with pytest.raises(ValueError):
        jax.grad(lambda x: jax.lax.while_loop(cond, body, (0, x))[1])(x0)

    g = lambda x, n: bounded_fori(0, n, lambda i, y: y * (1.0 + 0.1 * i), x, 20)
    assert float(g(x0, 5)) == pytest.approx(
        float(jax.lax.fori_loop(0, 5, lambda i, y: y * (1.0 + 0.1 * i), x0)),
        rel=1e-15)
    assert float(jax.grad(g)(x0, 5)) == pytest.approx(
        np.prod([1.0 + 0.1 * i for i in range(5)]), rel=1e-14)


@pytest.mark.parametrize('nbins,ratio', [(40, 2.0), (80, 2.0 ** 0.5), (30, 3.0)])
def test_bounded_mnfix_matches_reference(nbins, ratio):
    """Round-off only: XLA fuses the loop body differently inside lax.cond."""
    from tests.reference.mnfix_sequential import mnfix_reference
    xk = make_grid(nbins, XK0, ratio)
    new = jax.jit(lambda N, M: mnfix_jax(N, M, xk, bounded=True))
    ref = jax.jit(mnfix_reference)
    for Nk, Mk in _random_states(xk, seed=nbins):
        a, b = new(Nk, Mk), ref(Nk, Mk, xk)
        np.testing.assert_allclose(np.array(a[0]), np.array(b[0]), rtol=1e-12)
        np.testing.assert_allclose(np.array(a[1]), np.array(b[1]), rtol=1e-12,
                                   atol=1e-300)


def test_bad_ad_mode_rejected():
    with pytest.raises(ValueError, match='ad_mode'):
        make_step(PROCS, ad_mode='backward')


def _model(xk, so2_ppbv, nsteps, mode, **step_kw):
    temp = 217.0
    step = make_step(PROCS, sparse_density=c.SPARSE_DENSITY,
                     water_scheme=c.WATER_SCHEME, ad_mode=mode, jit=False,
                     **step_kw)
    if mode == 'reverse':
        step = jax.checkpoint(step)
    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(so2_ppbv, temp,
                                                      c.STRAT['pres']))

    def f(p):
        def body(carry, _):
            return step(*carry, xk, temp, c.STRAT['pres'], c.BOXVOL,
                        c.STRAT['rh'], c.STRAT['alpha'], 60.0,
                        oh_conc=c.OH_CONC, org_conc=0.0, nh3_conc=0.0,
                        fion=c.FION, nucl_params=p), None
        (Nk, _, _), _ = jax.lax.scan(body, state, None, length=nsteps)
        return jnp.log(jnp.sum(Nk))
    return f


def _as_vec(p):
    return np.array([float(getattr(p, k)) for k in FIELDS])


@pytest.mark.parametrize('so2_ppbv,nsteps,step_kw', [
    (1e3, 10, dict(gas_substep='adaptive')),
    (1e3, 10, dict()),
    (1e5, 3, dict(gas_substep='adaptive', coag_solver='euler_adaptive')),
    (1e3, 10, dict(gas_substep='adaptive', ppm_tangent='frozen_shape')),
])
def test_grad_matches_jacfwd(xk, so2_ppbv, nsteps, step_kw):
    """jax.grad of the reverse-mode model = jacfwd of the forward-mode model,
    and the two models agree in value to round-off."""
    P = jax.tree.map(jnp.float64, c.NUC_PARAMS)
    fr = _model(xk, so2_ppbv, nsteps, 'reverse', **step_kw)
    ff = _model(xk, so2_ppbv, nsteps, 'forward', **step_kw)
    np.testing.assert_allclose(float(jax.jit(fr)(P)), float(jax.jit(ff)(P)),
                               rtol=1e-12)
    g = _as_vec(jax.jit(jax.grad(fr))(P))
    J = _as_vec(jax.jit(jax.jacfwd(ff))(P))
    assert np.all(np.isfinite(g))
    np.testing.assert_allclose(g, J, rtol=1e-8, atol=1e-10 * np.max(np.abs(J)))


def test_grad_matches_central_fd(xk):
    """Directional derivative along the Dunne binary-neutral rate (u_bn).

    The sub-step ladder (dt_rem / 2^k) is piecewise constant in the
    parameters, so central FD with h >= ~1e-3 can straddle a ladder switch
    and scatter by a few % (0.056-0.061 here); h = 1e-4 stays on one branch.
    """
    P = jax.tree.map(jnp.float64, c.NUC_PARAMS)
    fr = jax.jit(_model(xk, 1e3, 10, 'reverse', gas_substep='adaptive'))
    g = float(jax.grad(fr)(P).u_bn)
    h = 1e-4
    fd = (float(fr(P._replace(u_bn=P.u_bn + h)))
          - float(fr(P._replace(u_bn=P.u_bn - h)))) / (2 * h)
    np.testing.assert_allclose(g, fd, rtol=1e-3)
