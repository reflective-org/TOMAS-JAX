"""fn_prescribed (a constant J in place of the scheme) with adaptive gas sub-steps."""
import jax
import jax.numpy as jnp
import numpy as np

from sensitivities.dunne_nuc import common as c
from sensitivities.dunne_nuc.run_map import GRIDS
from tomas_jax.core.config import SRTSO4
from tomas_jax.physics import nucleation as nuc
from tomas_jax.solvers.gas_substep import h2so4_loss_frequency

T, P = c.TROPICAL_AFGL[15.0]
COND = dict(c.STRAT, temp=T, pres=P)


def _state(so2_ppbv=10.0, h2so4=0.0):
    xk = GRIDS[40]()
    Nk, Mk, Gc = c.lognormal_state(xk, so2=c.ppbv_to_molec(so2_ppbv, T, P),
                                   h2so4=h2so4)
    return xk, Nk, Mk, Gc


def test_nucleation_step_prescribed_default_and_scheme_rate():
    xk, Nk, Mk, Gc = _state(h2so4=3e7)
    args = (Nk, Mk, Gc, xk, T, P, c.BOXVOL, 10.0, 0.0, 0.0, c.FION)
    ref = nuc.nucleation_step(*args, params=c.NUC_PARAMS)
    none = nuc.nucleation_step(*args, params=c.NUC_PARAMS, fn_prescribed=None)
    J = nuc.ricco_dunne_nucleation_rate(Gc, T, P, c.BOXVOL, 0.0, 0.0, c.FION,
                                        params=c.NUC_PARAMS)
    same = nuc.nucleation_step(*args, params=c.NUC_PARAMS, fn_prescribed=J)
    for a, b, d in zip(ref, none, same):
        assert np.array_equal(a, b)
        np.testing.assert_allclose(d, a, rtol=1e-12, atol=0)


def test_controller_sees_prescribed_j():
    # k_nuc = (1 - org_frac) m_nuc boxvol J / C, and J does not depend on C (n_eff = 1)
    xk, Nk, Mk, Gc = _state(h2so4=3e7)
    Gc = Gc.at[SRTSO4].set(1e-12)
    J = 50.0
    lam, CS, k = h2so4_loss_frequency(
        Nk, Mk, Gc, xk, T, P, c.BOXVOL, 0.02, 1.0, jnp.array([1.0]), 0.0,
        dict(fion=c.FION, nucl_params=c.NUC_PARAMS, fn_prescribed=J),
        has_chem=False)
    expect = ((1 - c.NUC_PARAMS.org_frac) * nuc._cluster_mass(c.NUC_PARAMS)
              * c.BOXVOL * J / 1e-12)
    np.testing.assert_allclose(k, expect, rtol=1e-12)
    np.testing.assert_allclose(lam, CS + k, rtol=1e-12)


def test_adaptive_matches_fixed_dt_runner():
    xk, Nk, Mk, Gc = _state(so2_ppbv=10.0)
    J = 2.0
    ad = c.make_prescribed_j_runner(xk, nsteps=60, cond=COND, dt=60.0,
                                    gas_substep='adaptive')
    fx = c.make_prescribed_j_runner(xk, nsteps=3600, cond=COND, dt=1.0)
    a = jax.jit(ad)(J, Nk, Mk, Gc)
    b = jax.jit(fx)(J, Nk, Mk, Gc)
    np.testing.assert_allclose(jnp.sum(a[0]), jnp.sum(b[0]), rtol=2e-3)
    np.testing.assert_allclose(a[2][SRTSO4], b[2][SRTSO4], rtol=5e-2)
    assert not bool(jnp.any(a[3]['clamp']))


def test_adaptive_tangent_in_j_matches_fd():
    xk, Nk, Mk, Gc = _state(so2_ppbv=10.0)
    run = c.make_prescribed_j_runner(xk, nsteps=30, cond=COND, dt=60.0,
                                     gas_substep='adaptive')

    def n(J):
        return jnp.sum(run(J, Nk, Mk, Gc)[0])

    J = 2.0
    ad = jax.jvp(n, (J,), (1.0,))[1]
    h = 0.02
    fd = (n(J * (1 + h)) - n(J * (1 - h))) / (2 * h * J)
    np.testing.assert_allclose(ad, fd, rtol=0.05)
