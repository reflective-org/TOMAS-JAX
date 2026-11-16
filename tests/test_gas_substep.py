"""make_step(gas_substep='adaptive'): H2SO4-budget sub-stepping (docs/gas_substep.md)."""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import SRTSO4, SRTSO2, MW_H2SO4, MW_SO2
from tomas_jax.solvers.condensation import make_step, GasSubstepCapWarning
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
    with pytest.raises(ValueError, match='split_chemistry'):
        make_step(PROCS, gas_substep='adaptive', split_chemistry=True)
    with pytest.raises(ValueError, match='dunne_form'):
        make_step(PROCS, nucl_scheme='zhao2024', dunne_form='single_exp')
    for frac in (0.0, -1.0):
        with pytest.raises(ValueError, match='coag_interval_frac'):
            make_step(PROCS, gas_substep='adaptive', coag_interval_frac=frac)


@pytest.mark.parametrize('ad_mode', ['forward', 'reverse'])
def test_empty_box_terminates(xk, ad_mode):
    """N0 = 0 with no coagulation to add TINY values: the interval starts
    closed (dN < frac*N0 is 0 < 0). Every interval must still take a
    sub-step; the forward loop used to spin forever here."""
    procs = ['so2_chemistry', 'nucleation', 'condensation']
    step = make_step(procs, gas_substep='adaptive', ad_mode=ad_mode,
                     return_substep_info=True, **STUDY)
    Nk, Mk, Gc = c.lognormal_state(
        xk, so2=c.ppbv_to_molec(10.0, 217.0, PRES))
    Nk, Mk = jnp.zeros_like(Nk), jnp.zeros_like(Mk)
    Nk1, Mk1, Gc1, info = step(Nk, Mk, Gc, xk, 217.0, PRES, c.BOXVOL, RH,
                               ALPHA, 60.0, **KW)
    assert int(info['n_sub']) >= 1 and not bool(info['hit_cap'])
    assert np.all(np.isfinite(np.asarray(Nk1)))
    assert float(jnp.sum(Nk1)) > 0.0


@pytest.mark.parametrize('ad_mode', ['forward', 'reverse'])
def test_binding_cap_warns_and_reports_the_forced_step(xk, ad_mode):
    """A binding cap forces the rest of dt into one sub-step: hit_cap,
    dt_capped and a knuc_dt_max that describes that step (the clamp fires),
    plus a GasSubstepCapWarning without return_substep_info."""
    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(1e5, 217.0, PRES))
    args = (xk, 217.0, PRES, c.BOXVOL, RH, ALPHA, 60.0)
    step = make_step(PROCS, gas_substep='adaptive', gas_max_substeps=64,
                     ad_mode=ad_mode, return_substep_info=True, **STUDY)
    with pytest.warns(GasSubstepCapWarning, match='gas_max_substeps=64'):
        *_, info = step(*state, *args, **KW)
        jax.block_until_ready(info)
    assert bool(info['hit_cap'])
    assert 1.0 < float(info['dt_capped']) < 60.0
    assert float(info['knuc_dt_max']) > 1.0

    quiet = make_step(PROCS, gas_substep='adaptive', gas_max_substeps=64,
                      ad_mode=ad_mode, **STUDY)
    with pytest.warns(GasSubstepCapWarning):
        jax.block_until_ready(quiet(*state, *args, **KW))


def test_cap_warning_under_vmap(xk):
    """Only the batch member that hits the cap is reported; a batch that
    does not hit it stays silent."""
    import warnings
    step = make_step(PROCS, gas_substep='adaptive', gas_max_substeps=64,
                     return_substep_info=True, **STUDY)
    states = [c.lognormal_state(xk, so2=c.ppbv_to_molec(s, 217.0, PRES))
              for s in (10.0, 1e5)]
    batch = jax.tree.map(lambda *a: jnp.stack(a), *states)
    f = jax.vmap(lambda s: step(*s, xk, 217.0, PRES, c.BOXVOL, RH, ALPHA,
                                60.0, **KW))
    with pytest.warns(GasSubstepCapWarning, match='in 1 step'):
        *_, info = f(batch)
        jax.block_until_ready(info)
    assert list(np.asarray(info['hit_cap'])) == [False, True]
    low = jax.tree.map(lambda *a: jnp.stack(a), states[0], states[0])
    with warnings.catch_warnings():
        warnings.simplefilter('error', GasSubstepCapWarning)
        jax.block_until_ready(f(low))


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


def test_zhao2024_loss_frequency_is_the_step_removal(xk):
    """With nucl_scheme='zhao2024' k_nuc is the H2SO4 that
    zhao2024_nucleation_step removes (amine and iodine mechanisms included)."""
    from tomas_jax.solvers.gas_substep import h2so4_loss_frequency
    from tomas_jax.physics.nucleation import zhao2024_nucleation_step
    temp = 217.0
    Nk, Mk, Gc = c.lognormal_state(xk, so2=0.0)
    Gc = Gc.at[SRTSO4].set(1e6 * c.BOXVOL * 98e-3 / 6.02214076e23)
    zkw = dict(org_conc=0.0, nh3_conc=0.0, fion=c.FION, fn_scale=1.0,
               hno3=0.0, ulvoc=0.0, dma=1e8, hio3=2e7)
    dt = 1e-3
    _, _, k_nuc = h2so4_loss_frequency(
        Nk, Mk, Gc, xk, temp, PRES, c.BOXVOL, RH, ALPHA, jnp.array([dt]),
        0.0, zkw, has_chem=False, nucl_scheme='zhao2024')
    _, _, Gc1 = zhao2024_nucleation_step(
        Nk, Mk, Gc, xk, temp, PRES, c.BOXVOL, dt, 0.0, 0.0, c.FION,
        dma=1e8, hio3=2e7)
    removed = float(Gc[SRTSO4] - Gc1[SRTSO4])
    assert removed > 0.0
    np.testing.assert_allclose(float(k_nuc[0]) * dt * float(Gc[SRTSO4]),
                               removed, rtol=1e-10)
    # The Ricco+Dunne rate (the old criterion) is far smaller here
    _, _, k_rd = h2so4_loss_frequency(
        Nk, Mk, Gc, xk, temp, PRES, c.BOXVOL, RH, ALPHA, jnp.array([dt]),
        0.0, dict(KW), has_chem=False)
    assert float(k_rd[0]) < 1e-3 * float(k_nuc[0])


def test_zhao2024_rejects_nucl_params(xk):
    step = make_step(['nucleation'], nucl_scheme='zhao2024', jit=False)
    Nk, Mk, Gc = c.lognormal_state(xk, so2=0.0)
    with pytest.raises(ValueError, match='nucl_params'):
        step(Nk, Mk, Gc, xk, 217.0, PRES, c.BOXVOL, RH, ALPHA, 60.0, **KW)


def test_zhao2024_iodine_clamp_never_fires(xk):
    """1 ppmv, iodine-dominated zhao2024 nucleation: the sub-steps follow
    the zhao2024 rate (knuc_dt reaches past what Dunne alone gives, 0.026)
    and stay within the criterion."""
    temp = 217.0
    step = make_step(PROCS, nucl_scheme='zhao2024', gas_substep='adaptive',
                     return_substep_info=True, **STUDY)
    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(1e3, temp, PRES))
    kw = dict(oh_conc=c.OH_CONC, org_conc=0.0, nh3_conc=0.0, fion=c.FION,
              hio3=3e7)
    s0 = _sulfur(state[1], state[2])
    knuc = []
    for _ in range(3):
        *state, info = step(*state, xk, temp, PRES, c.BOXVOL, RH, ALPHA,
                            60.0, **kw)
        assert int(info['n_floor']) == 0 and not bool(info['hit_cap'])
        knuc.append(float(info['knuc_dt_max']))
    assert max(knuc) <= 0.1 + 1e-12
    assert max(knuc) > 0.05
    assert abs(_sulfur(state[1], state[2]) / s0 - 1.0) < 1e-10


@pytest.mark.parametrize('force', [False, True])
def test_floor_reached_on_long_interval(xk, force):
    """dt_rem > 2^23 * min_dt: the ladder's bottom rung is above min_dt,
    so the floor is a rung of its own; a failing criterion lands on min_dt."""
    from tomas_jax.solvers.gas_substep import choose_gas_substep
    temp = 217.0
    Nk, Mk, Gc = c.lognormal_state(xk, so2=0.0)
    # No chemistry; a large H2SO4 concentration with frac tiny so nothing passes
    Gc = Gc.at[SRTSO4].set(1e9 * c.BOXVOL * 98e-3 / 6.02214076e23)
    dt_rem = 1e5
    assert dt_rem / 2 ** 23 > 1e-3
    dt_sub, floored, _ = choose_gas_substep(
        Nk, Mk, Gc, xk, temp, PRES, c.BOXVOL, RH, ALPHA, dt_rem, 0.0, dict(KW),
        frac=1e-12, min_dt=1e-3, has_chem=False, force_dt=dt_rem,
        force=jnp.asarray(force))
    assert float(dt_sub) == 1e-3
    assert bool(floored) is (not force)
