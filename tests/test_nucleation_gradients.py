"""Differentiability of the ricco_dunne nucleation scheme w.r.t. its fit
coefficients (NucleationParams). See docs/nucleation_sensitivity.md."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import (
    NBINS, ICOMP, N_GAS_SPECIES, SRTSO4, SRTORGLAST, AVOGADRO,
)
from tomas_jax.physics.nucleation import (
    NucleationParams, DEFAULT_NUCLEATION_PARAMS,
    dunne_nucleation_rate, ricco_dunne_nucleation_rate, nucleation_step,
    _MNUC,
)
from tomas_jax.solvers.condensation import make_step

T_STRAT, P_STRAT = 217.0, 5500.0
MAIR = 2.69e19 * 273.15 / T_STRAT * P_STRAT / 101325.0
BOXVOL = 1e6


def _jax_params(p=DEFAULT_NUCLEATION_PARAMS):
    return jax.tree.map(jnp.float64, p)


def _gc(h2so4=5e6):
    return jnp.zeros(N_GAS_SPECIES).at[SRTSO4].set(
        h2so4 * BOXVOL * 98e-3 / AVOGADRO)


@pytest.fixture
def state(xk):
    Nk = jnp.ones(NBINS) * 1e6
    Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(
        Nk * jnp.sqrt(xk[:-1] * xk[1:]))
    return Nk, Mk, _gc(1e7), xk


class TestDefaults:
    def test_default_cluster_mass_and_split_bit_identical(self):
        from tomas_jax.physics.nucleation import _cluster_mass
        assert _cluster_mass(DEFAULT_NUCLEATION_PARAMS) == _MNUC
        assert 1.0 - DEFAULT_NUCLEATION_PARAMS.org_frac == 0.9

    def test_params_none_equals_default_pytree(self):
        args = (_gc(), T_STRAT, P_STRAT, BOXVOL, 1e5, 1e7, 10.0)
        a = ricco_dunne_nucleation_rate(*args)
        b = ricco_dunne_nucleation_rate(*args, params=_jax_params())
        assert float(a) == float(b)

    def test_step_matches_legacy_constants(self, state):
        """Default step reproduces the pre-params arithmetic exactly."""
        Nk, Mk, Gc, xk = state
        dt = 60.0
        Nk1, Mk1, Gc1 = nucleation_step(Nk, Mk, Gc, xk, 278.0, 101325.0,
                                        BOXVOL, dt, 1e7, 1e8, 3.0)
        fn = ricco_dunne_nucleation_rate(Gc, 278.0, 101325.0, BOXVOL,
                                         1e7, 1e8, 3.0)
        dM = fn * _MNUC * BOXVOL * dt
        assert float(Nk1[0] - Nk[0]) == float(Nk[0] + fn * BOXVOL * dt) - float(Nk[0])
        assert float(Mk1[0, SRTORGLAST]) == float(Mk[0, SRTORGLAST] + 0.1 * dM)
        assert float(Mk1[0, SRTSO4]) == float(Mk[0, SRTSO4] + 0.9 * dM)


class TestRateGradients:
    def test_binary_neutral_analytic(self):
        """d lnJbn/d(u, p, v, w) match closed forms."""
        h2so4 = 5e6
        P = DEFAULT_NUCLEATION_PARAMS

        def ln_jbn(p):
            return jnp.log(dunne_nucleation_rate(T_STRAT, 10.0, h2so4, 0.0,
                                                 MAIR, p)[1])

        g = jax.jacfwd(ln_jbn)(_jax_params())
        t = T_STRAT / 1000.0
        e = np.exp(P.v_bn * (t - P.w_bn))
        np.testing.assert_allclose(float(g.u_bn), 1.0, rtol=1e-12)
        np.testing.assert_allclose(float(g.p_bn), np.log(h2so4 * 1e-6), rtol=1e-12)
        np.testing.assert_allclose(float(g.v_bn), -e * (t - P.w_bn), rtol=1e-12)
        np.testing.assert_allclose(float(g.w_bn), P.v_bn * e, rtol=1e-12)
        assert float(g.u_bi) == 0.0 and float(g.k_m) == 0.0

    @pytest.mark.parametrize('nh3,fion,org', [(0.0, 0.0, 0.0), (0.0, 10.0, 0.0),
                                              (1e7, 0.0, 1e5)])
    def test_tangents_finite_at_zero_inputs(self, state, nh3, fion, org):
        Nk, Mk, Gc, xk = state

        def rate(p):
            return ricco_dunne_nucleation_rate(Gc, T_STRAT, P_STRAT, BOXVOL,
                                               org, nh3, fion, params=p)

        def step(p):
            out = nucleation_step(Nk, Mk, Gc, xk, T_STRAT, P_STRAT, BOXVOL,
                                  60.0, org, nh3, fion, params=p)
            return jnp.concatenate([out[0], out[1].ravel(), out[2]])

        for f in (rate, step):
            for leaf in jax.tree.leaves(jax.jacfwd(f)(_jax_params())):
                assert np.all(np.isfinite(np.array(leaf)))

    def test_plain_python_floats_no_zero_division(self):
        """nh3 = 0 with all-Python-float inputs must not raise."""
        out = dunne_nucleation_rate(T_STRAT, 10.0, 5e6, 0.0, MAIR)
        assert float(out[2]) == 0.0 and float(out[4]) == 0.0


class TestMakeStepGradients:
    def test_jacfwd_through_scan_matches_fd(self, xk):
        """Forward-mode Jacobian through 5 full steps matches central FD."""
        step = make_step(['so2_chemistry', 'nucleation', 'coagulation',
                          'condensation'], jit=False)
        Nk0 = jnp.ones(NBINS) * 1e4
        Mk0 = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(
            Nk0 * jnp.sqrt(xk[:-1] * xk[1:]))
        Gc0 = _gc(1e7)

        def n_total(p):
            def body(c, _):
                return step(*c, xk, T_STRAT, P_STRAT, BOXVOL, 0.02, 1.0, 60.0,
                            oh_conc=1e6, org_conc=0.0, nh3_conc=0.0,
                            fion=10.0, nucl_params=p), None
            (Nk, _, _), _ = jax.lax.scan(body, (Nk0, Mk0, Gc0), None, length=5)
            return jnp.sum(Nk)

        f = jax.jit(n_total)
        P = DEFAULT_NUCLEATION_PARAMS
        g = jax.jit(jax.jacfwd(n_total))(_jax_params())
        for name in ('u_bn', 'u_bi', 'r_nuc_nm'):
            assert np.isfinite(float(getattr(g, name)))
            val = getattr(P, name)
            h = 1e-5 * abs(val)
            fd = (f(_jax_params(P._replace(**{name: val + h})))
                  - f(_jax_params(P._replace(**{name: val - h})))) / (2 * h)
            np.testing.assert_allclose(float(getattr(g, name)), float(fd),
                                       rtol=1e-5)

    def test_unknown_kwarg_not_warned(self, xk):
        step = make_step(['nucleation'], jit=False)
        Nk = jnp.ones(NBINS) * 1e4
        Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(
            Nk * jnp.sqrt(xk[:-1] * xk[1:]))
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter('error', UserWarning)
            step(Nk, Mk, _gc(), xk, T_STRAT, P_STRAT, BOXVOL, 0.02, 1.0, 60.0,
                 fion=10.0, nucl_params=NucleationParams())


class TestDunneTableS2:
    """Dunne 2016 Table S2 single-exponential cross-check fit."""

    def test_single_exp_rate_constant(self):
        from tomas_jax.physics.nucleation import DUNNE_TABLE_S2_PARAMS as S2
        h2so4, t = 5e6, T_STRAT / 1000.0
        _, Jbn, _, Jbi, _ = dunne_nucleation_rate(
            T_STRAT, 10.0, h2so4, 0.0, MAIR, S2, dunne_form='single_exp')
        cna = h2so4 * 1e-6
        np.testing.assert_allclose(float(Jbn),
                                   np.exp(S2.u_bn - S2.v_bn * t) * cna**S2.p_bn,
                                   rtol=1e-12)
        assert float(Jbi) > 0.0

    def test_same_order_as_baseline_at_cloud_temperatures(self):
        """Both fits describe the same CLOUD data (208-298 K): binary
        neutral rates agree within 2 orders of magnitude there."""
        from tomas_jax.physics.nucleation import DUNNE_TABLE_S2_PARAMS as S2
        for T in (208.0, 223.0, 248.0):
            Mair = 2.69e19 * 273.15 / T * P_STRAT / 101325.0
            base = dunne_nucleation_rate(T, 10.0, 1e7, 0.0, Mair)[1]
            alt = dunne_nucleation_rate(T, 10.0, 1e7, 0.0, Mair, S2,
                                        dunne_form='single_exp')[1]
            assert abs(np.log10(float(alt) / float(base))) < 2.0

    def test_make_step_rejects_unknown_form(self):
        with pytest.raises(ValueError, match='dunne_form'):
            make_step(['nucleation'], dunne_form='triple_exp')

    def test_make_step_single_exp_runs_and_differs(self, xk):
        from tomas_jax.physics.nucleation import DUNNE_TABLE_S2_PARAMS as S2
        Nk = jnp.ones(NBINS) * 1e4
        Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(
            Nk * jnp.sqrt(xk[:-1] * xk[1:]))
        args = (Nk, Mk, _gc(1e7), xk, T_STRAT, P_STRAT, BOXVOL, 0.02, 1.0, 60.0)
        base = make_step(['nucleation'])(*args, fion=10.0)
        alt = make_step(['nucleation'], dunne_form='single_exp')(
            *args, fion=10.0, nucl_params=S2)
        assert np.all(np.isfinite(np.array(alt[0])))
        assert float(alt[0][0]) != float(base[0][0])
