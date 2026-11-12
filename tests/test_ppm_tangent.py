"""make_step(ppm_tangent='frozen_shape'): same values, frozen-shape PPM derivative."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.physics.condensation_ppm import ppm_reconstruct
from tomas_jax.solvers.condensation import make_step
from sensitivities.dunne_nuc import common as c

PROCS = ['so2_chemistry', 'nucleation', 'coagulation', 'condensation']


def test_reconstruct_values_and_tangent():
    rng = np.random.default_rng(0)
    q = jnp.asarray(np.concatenate([rng.lognormal(0.0, 2.0, 30), np.zeros(5),
                                    rng.lognormal(0.0, 2.0, 5)]))
    exact = ppm_reconstruct(q)
    frozen = ppm_reconstruct(q, freeze_shape=True)
    for a, b in zip(exact, frozen):
        assert np.array_equal(np.array(a), np.array(b))

    dq = jnp.asarray(rng.normal(size=q.shape))
    _, tan = jax.jvp(lambda x: ppm_reconstruct(x, freeze_shape=True), (q,), (dq,))
    occ = np.array(q) > 0
    for v, t, flat in zip(exact, tan, (1.0, 1.0, 0.0)):
        s = np.where(occ, np.array(v) / np.where(occ, np.array(q), 1.0), flat)
        np.testing.assert_allclose(np.array(t), s * np.array(dq), rtol=1e-14, atol=1e-300)


def _run(xk, mode, nsteps=20, so2_ppbv=1e2):
    temp = 217.0
    step = make_step(PROCS, sparse_density=c.SPARSE_DENSITY,
                     water_scheme=c.WATER_SCHEME, ppm_tangent=mode, jit=False)
    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(so2_ppbv, temp, c.STRAT['pres']))

    def f(scale):
        def body(carry, _):
            return step(*carry, xk, temp, c.STRAT['pres'], c.BOXVOL,
                        c.STRAT['rh'], c.STRAT['alpha'], 10.0,
                        oh_conc=c.OH_CONC * scale, org_conc=0.0, nh3_conc=0.0,
                        fion=c.FION, nucl_params=c.NUC_PARAMS), None
        (Nk, Mk, Gc), _ = jax.lax.scan(body, state, None, length=nsteps)
        return Nk, Mk, Gc
    return jax.jit(lambda s: jax.jvp(f, (s,), (jnp.ones_like(s),)))(jnp.float64(1.0))


def test_make_step_values_bit_identical(xk):
    (v_e, t_e), (v_f, t_f) = _run(xk, 'exact'), _run(xk, 'frozen_shape')
    for a, b in zip(v_e, v_f):
        assert np.array_equal(np.array(a), np.array(b))
    for t in t_f:
        assert np.all(np.isfinite(np.array(t)))
    # same derivative to within the shape term (a few % here), not identical
    rel = abs(float(jnp.sum(t_f[0])) / float(jnp.sum(t_e[0])) - 1.0)
    assert rel < 0.1


def test_bad_ppm_tangent_rejected():
    with pytest.raises(ValueError, match='ppm_tangent'):
        make_step(PROCS, ppm_tangent='frozen')
    with pytest.raises(ValueError, match='PPM condensation'):
        make_step(PROCS, cond_method='tfl_jit', ppm_tangent='frozen_shape')
