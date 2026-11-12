"""make_step(gas_substep='adaptive'): H2SO4-budget sub-stepping (docs/gas_substep.md)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import SRTSO4, SRTSO2, MW_H2SO4, MW_SO2
from tomas_jax.solvers.condensation import make_step
from sensitivities.dunne_nuc import common as c

PROCS = ['so2_chemistry', 'nucleation', 'coagulation', 'condensation']
PRES, RH, ALPHA = c.STRAT['pres'], c.STRAT['rh'], c.STRAT['alpha']
STUDY = dict(sparse_density=c.SPARSE_DENSITY, water_scheme=c.WATER_SCHEME)
KW = dict(oh_conc=c.OH_CONC, org_conc=0.0, nh3_conc=0.0, fion=c.FION,
          nucl_params=c.NUC_PARAMS)


def _run(xk, temp, so2_ppbv, dt, nsteps, **step_kw):
    step = make_step(PROCS, jit=False, **STUDY, **step_kw)
    info_on = step_kw.get('return_substep_info', False)

    def body(carry, _):
        out = step(*carry, xk, temp, PRES, c.BOXVOL, RH, ALPHA, dt, **KW)
        return out[:3], (out[3] if info_on else 0)

    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(so2_ppbv, temp, PRES))
    (Nk, Mk, Gc), info = jax.jit(
        lambda s: jax.lax.scan(body, s, None, length=nsteps))(state)
    return state, (Nk, Mk, Gc), info


def _n_above(Nk, Mk, d):
    Nk, Mk = np.asarray(Nk), np.asarray(Mk)
    m = Mk.sum(1) / np.maximum(Nk, 1e-30)
    D = (6 * m / (c.DENS_INIT * np.pi)) ** (1 / 3)
    return Nk[D > d].sum() / c.BOXVOL


def _sulfur(Mk, Gc):
    # Nucleation and condensation move SO4 mass 1:1 from Gc[SRTSO4]
    return (float(Gc[SRTSO2]) * MW_H2SO4 / MW_SO2 + float(Gc[SRTSO4])
            + float(jnp.sum(Mk[:, SRTSO4])))


def test_bad_arguments_rejected():
    with pytest.raises(ValueError, match='gas_substep'):
        make_step(PROCS, gas_substep='rk45')
    with pytest.raises(ValueError, match='needs at least one'):
        make_step(['coagulation'], gas_substep='adaptive')
    with pytest.raises(ValueError, match='return_substep_info'):
        make_step(PROCS, return_substep_info=True)
    with pytest.raises(ValueError, match='split_chemistry'):
        make_step(['so2_chemistry', 'coagulation'], split_chemistry=True)


def test_split_chemistry_fixed_dt(xk):
    """Fixed dt, chemistry split around nucleation: sulfur conserved, and
    nucleation sees less H2SO4 than with all the chemistry before it."""
    state, (Nk, Mk, Gc), _ = _run(xk, 217.0, 1e3, 0.5, 120, split_chemistry=True)
    _, (Nu, Mu, Gu), _ = _run(xk, 217.0, 1e3, 0.5, 120)
    s0, s1 = _sulfur(state[1], state[2]), _sulfur(Mk, Gc)
    assert abs(s1 / s0 - 1.0) < 1e-10
    assert float(jnp.sum(Nk)) < float(jnp.sum(Nu))


def test_low_so2_takes_the_outer_step(xk):
    """10 ppbv: the criterion passes at 60 s, one sub-step and one interval."""
    _, _, info = _run(xk, 217.0, 10.0, 60.0, 10, gas_substep='adaptive',
                      return_substep_info=True)
    assert np.all(np.asarray(info['n_sub']) == 1)
    assert np.all(np.asarray(info['n_coag']) == 1)


@pytest.mark.parametrize('temp', [190.0, 217.0])
def test_clamp_never_fires_at_100_ppmv(xk, temp):
    state, (Nk, Mk, Gc), info = _run(xk, temp, 1e5, 60.0, 5,
                                     gas_substep='adaptive',
                                     return_substep_info=True)
    assert float(np.max(info['knuc_dt_max'])) <= 0.1 + 1e-12
    assert not np.any(np.asarray(info['hit_cap']))
    assert np.all(np.asarray(info['n_floor']) == 0)
    assert np.all(np.asarray(info['n_sub']) > 100)    # the burst is resolved
    s0, s1 = _sulfur(state[1], state[2]), _sulfur(Mk, Gc)
    assert abs(s1 / s0 - 1.0) < 1e-10


def test_matches_small_fixed_dt_at_1_ppmv(xk):
    """1 h at 217 K, 1 ppmv: outer 60 s adaptive vs fixed 0.1 s."""
    _, (Nk, Mk, _), _ = _run(xk, 217.0, 1e3, 60.0, 60, gas_substep='adaptive')
    _, (Nr, Mr, _), _ = _run(xk, 217.0, 1e3, 0.1, 36000)
    for d in (3e-9, 12e-9):
        np.testing.assert_allclose(_n_above(Nk, Mk, d), _n_above(Nr, Mr, d),
                                   rtol=0.05)


def test_forward_mode_and_vmap(xk):
    """jvp along the Dunne binary-neutral H2SO4 exponent matches central FD;
    vmap over two SO2 levels runs."""
    step = make_step(PROCS, jit=False, gas_substep='adaptive', **STUDY)
    temp = 217.0
    states = [c.lognormal_state(xk, so2=c.ppbv_to_molec(s, temp, PRES))
              for s in (1e2, 1e4)]

    def n_total(scale, state):
        p = c.NUC_PARAMS._replace(p_bn=c.NUC_PARAMS.p_bn * scale)
        kw = dict(KW, nucl_params=p)

        def body(carry, _):
            return step(*carry, xk, temp, PRES, c.BOXVOL, RH, ALPHA, 60.0,
                        **kw), None
        (Nk, _, _), _ = jax.lax.scan(body, state, None, length=3)
        return jnp.log(jnp.sum(Nk))

    f = jax.jit(lambda s: n_total(s, states[0]))
    _, tangent = jax.jvp(f, (1.0,), (1.0,))
    h = 1e-4
    fd = (f(1.0 + h) - f(1.0 - h)) / (2 * h)
    assert np.isfinite(float(tangent))
    np.testing.assert_allclose(float(tangent), float(fd), rtol=0.05, atol=1e-3)

    batch = jax.tree_util.tree_map(lambda *a: jnp.stack(a), *states)
    out = jax.jit(jax.vmap(lambda s: n_total(1.0, s)))(batch)
    assert np.all(np.isfinite(np.asarray(out)))
