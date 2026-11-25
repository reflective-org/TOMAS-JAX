"""make_step(...)(..., coag_kernel_scale=s): multiplier on the Brownian
coagulation kernel (sensitivity knob), in all three coag_solvers."""
import json
import os
import subprocess
import sys
import textwrap

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import NBINS, ICOMP, N_GAS_SPECIES, SRTSO4, SRTH2O
from tomas_jax.solvers.condensation import make_step
from tomas_jax.solvers.diffrax import coag_euler_step
from tomas_jax.solvers.euler import euler_step, heun_step
from sensitivities.dunne_nuc import common as c

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SOLVERS = ['euler_fixed', 'euler_adaptive', 'heun_adaptive']
BOXVOL = 1e6
ENV = (190.0, 5500.0, BOXVOL, 0.02, 1.0, 60.0)   # temp, pres, boxvol, rh, alpha, dt
PROCS = ['so2_chemistry', 'nucleation', 'coagulation', 'condensation']
STUDY = dict(sparse_density=c.SPARSE_DENSITY, water_scheme=c.WATER_SCHEME)
KW = dict(oh_conc=c.OH_CONC, org_conc=0.0, nh3_conc=0.0, fion=c.FION,
          nucl_params=c.NUC_PARAMS)


def _state(xk, n_cm3_by_bin):
    Nk = jnp.full(NBINS, 1e-3)
    for k, n in n_cm3_by_bin.items():
        Nk = Nk.at[k].set(n * BOXVOL)
    Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(Nk * jnp.sqrt(xk[:-1] * xk[1:]))
    return Nk, Mk, jnp.zeros(N_GAS_SPECIES)


# Nucleation mode + Aitken mode + accumulation tail: coagulation removes
# ~15-40% of N in 60 s, enough to see every solver respond to the scale.
MODES = {0: 3e6, 3: 1e5, 8: 1e4, 20: 10.0}


def _smooth_state(xk):
    """Nucleation, Aitken and accumulation modes with every bin occupied
    (coagulation removes 20-75% of N in 60-190 s). No bin is nearly empty
    next to a full one, so euler_adaptive's absolute 1e-10 s sub-step floor
    does not bind (see test_time_rescaling_identity)."""
    i = np.arange(NBINS)
    n = (3e6 * np.exp(-0.5 * ((i - 1) / 1.5) ** 2)
         + 1e5 * np.exp(-0.5 * ((i - 8) / 3.0) ** 2)
         + 1e2 * np.exp(-0.5 * ((i - 20) / 3.0) ** 2))
    Nk = jnp.asarray(np.maximum(n * BOXVOL, 1e-3))
    Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(Nk * jnp.sqrt(xk[:-1] * xk[1:]))
    return Nk, Mk, jnp.zeros(N_GAS_SPECIES)


def _dry(Mk):
    return float(jnp.sum(Mk[:, :SRTH2O]))


def _coag(solver, **factory):
    return make_step(['coagulation'], coag_solver=solver, **factory)


# ---------------------------------------------------------------- a. default

@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('jit', [True, False])
def test_scale_one_is_bit_identical(xk, solver, jit):
    """coag_kernel_scale=1.0 (Python float, or a jnp array, i.e. traced
    under the step's jit) equals not passing it, bit for bit."""
    Nk, Mk, Gc = _state(xk, MODES)
    step = _coag(solver, jit=jit)
    ref = step(Nk, Mk, Gc, xk, *ENV)
    for s in (1.0, jnp.float64(1.0)):
        out = step(Nk, Mk, Gc, xk, *ENV, coag_kernel_scale=s)
        for a, b in zip(ref, out):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _production_run(xk, nsteps, scale=None, traced=True):
    """The AIDE-SAI cesm_adaptive_substep configuration: heun_adaptive with
    adaptive gas sub-steps, under jit(scan). scale None = kwarg not passed;
    traced=False closes over it as a Python constant."""
    step = make_step(PROCS, coag_solver='heun_adaptive', gas_substep='adaptive',
                     jit=False, **STUDY)
    temp = 217.0
    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(100.0, temp, c.STRAT['pres']))

    def run(state, s):
        kw = dict(KW) if s is None else dict(KW, coag_kernel_scale=s)

        def body(carry, _):
            return step(*carry, xk, temp, c.STRAT['pres'], c.BOXVOL,
                        c.STRAT['rh'], c.STRAT['alpha'], 60.0, **kw), None
        return jax.lax.scan(body, state, None, length=nsteps)[0]

    if scale is None or not traced:
        return state, jax.jit(lambda st: run(st, scale))(state)
    return state, jax.jit(run)(state, scale)


def test_scale_one_bit_identical_production_config(xk):
    _, ref = _production_run(xk, 3)
    for traced in (True, False):
        _, out = _production_run(xk, 3, scale=1.0, traced=traced)
        for a, b in zip(ref, out):
            np.testing.assert_array_equal(np.asarray(a), np.asarray(b))
    # and the knob reaches the coagulation in this path
    state, half = _production_run(xk, 3, scale=0.5)
    _, double = _production_run(xk, 3, scale=2.0)
    assert float(jnp.sum(double[0])) < float(jnp.sum(ref[0])) < float(jnp.sum(half[0]))


SULFATE_WORKER = textwrap.dedent('''
    import sys, json
    import numpy as np
    import jax, jax.numpy as jnp
    from tomas_jax.core.config import (NBINS, ICOMP, N_GAS_SPECIES, SRTSO4,
                                       SRTSO2, HAS_ORG, xk_boundaries)
    from tomas_jax.solvers.condensation import make_step
    from tomas_jax.physics.nucleation import DEFAULT_NUCLEATION_PARAMS
    NUC = DEFAULT_NUCLEATION_PARAMS._replace(org_frac=0.0)
    out_path = sys.argv[1]
    BOXVOL = 1e12
    xk = jnp.asarray(xk_boundaries())
    i = np.arange(NBINS)
    Nk = jnp.asarray(1e4 * BOXVOL * np.exp(-0.5 * ((i - 6) / 3.0) ** 2))
    Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(Nk * xk[:NBINS] * 1.4)
    Gc = jnp.zeros(N_GAS_SPECIES).at[SRTSO4].set(1e-12).at[SRTSO2].set(1e-6)
    args = (Nk, Mk, Gc, xk, 215., 3000., BOXVOL, 0.2, 1.0, 60.0)
    kw = dict(oh_conc=1e6, nucl_params=NUC, fion=10.0)
    res = dict(has_org=HAS_ORG)
    for sub in (None, "adaptive"):
        step = make_step(["so2_chemistry", "nucleation", "coagulation",
                          "condensation"], coag_solver="heun_adaptive",
                         gas_substep=sub)
        ref = step(*args, **kw)
        one = step(*args, coag_kernel_scale=1.0, **kw)
        two = step(*args, coag_kernel_scale=2.0, **kw)
        key = str(sub)
        res[key + "_bitident"] = all(bool(np.array_equal(np.asarray(a), np.asarray(b)))
                                     for a, b in zip(ref, one))
        res[key + "_N"] = [float(jnp.sum(ref[0])), float(jnp.sum(two[0]))]
        try:
            step(*args, coag_kernel_scale=-1.0, **kw)
            res[key + "_neg_raises"] = False
        except ValueError:
            res[key + "_neg_raises"] = True
    json.dump(res, open(out_path, "w"))
''')


@pytest.mark.parametrize('layout', ['default', 'sulfate'])
def test_scale_one_bit_identical_in_both_layouts(tmp_path, layout):
    """TOMAS_SPECIES is fixed at import, so each layout runs in its own
    subprocess (CPU only). The sulfate layout's jitted step goes through the
    concrete-value guard wrapper, so the negative check fires there too."""
    env = dict(os.environ, JAX_PLATFORMS='cpu', CUDA_VISIBLE_DEVICES='',
               PYTHONPATH=ROOT)
    env.pop('TOMAS_SPECIES', None)
    if layout == 'sulfate':
        env['TOMAS_SPECIES'] = 'sulfate'
    out = tmp_path / f'{layout}.json'
    p = subprocess.run([sys.executable, '-c', SULFATE_WORKER, str(out)],
                       env=env, cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    r = json.load(open(out))
    assert r['has_org'] == (layout == 'default')
    for sub in ('None', 'adaptive'):
        assert r[sub + '_bitident'], (layout, sub)
        n1, n2 = r[sub + '_N']
        assert n2 < n1, (layout, sub, r)
    # The sulfate layout checks concrete kwargs before its jit. (The jitted
    # default-layout step sees the value traced and cannot check it.)
    if layout == 'sulfate':
        assert r['None_neg_raises'] and r['adaptive_neg_raises'], r


# ---------------------------------------------------------------- b. zero

@pytest.mark.parametrize('solver', SOLVERS)
def test_scale_zero_switches_coagulation_off(xk, solver):
    Nk, Mk, Gc = _state(xk, MODES)
    out = _coag(solver)(Nk, Mk, Gc, xk, *ENV, coag_kernel_scale=0.0)
    np.testing.assert_array_equal(np.asarray(out[0]), np.asarray(Nk))
    np.testing.assert_array_equal(np.asarray(out[1]), np.asarray(Mk))


# ---------------------------------------------------------- c. time rescaling

# dN/dt = s f(N) is autonomous: scale s over dt is scale 1 over s*dt.
# euler_fixed: the same arithmetic up to the order of one multiplication.
# euler_adaptive / heun_adaptive: the controllers see h*s*f in both runs,
# so they pick the same sub-steps (h/s) and differ only by rounding in the
# step lengths. Observed max rel diff <= 1e-15 on 40 bins (CPU); 1e-10
# leaves room for platform FMA differences and is still ~1e7 below the
# solvers' own tolerance (heun rtol 3e-3), so a kernel that missed the
# scale anywhere (e.g. euler_adaptive's per-sub-step recompute) fails it.
# Exception, not tested: euler_adaptive's Fortran controller has an
# absolute sub-step floor of 1e-10 s, which does not scale with s. It binds
# when a nearly empty bin next to a full one receives coagulation products
# (MODES: bin 1 empty below 3e6 cm-3 in bin 0); there the two runs take
# different sub-steps (316 vs 319 at s = 2) and differ by up to 1e-6 in
# total N. That is the controller, not the scale.
RESCALE_RTOL = {'euler_fixed': 1e-12, 'euler_adaptive': 1e-10,
                'heun_adaptive': 1e-10}


@pytest.mark.parametrize('solver', SOLVERS)
@pytest.mark.parametrize('s', [0.37, 2.0, 3.1])
def test_time_rescaling_identity(xk, solver, s):
    Nk, Mk, Gc = _smooth_state(xk)
    temp, pres, boxvol, rh, alpha, dt = ENV
    step = _coag(solver)
    a = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt, coag_kernel_scale=s)
    b = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, s * dt,
             coag_kernel_scale=1.0)
    rtol = RESCALE_RTOL[solver]
    # the step changed the state, else the identity is empty
    assert float(jnp.sum(a[0])) < 0.99 * float(jnp.sum(Nk))
    np.testing.assert_allclose(np.asarray(a[0]), np.asarray(b[0]), rtol=rtol,
                               atol=rtol * float(jnp.sum(Nk)) * 1e-6)
    np.testing.assert_allclose(np.asarray(a[1]), np.asarray(b[1]), rtol=rtol,
                               atol=rtol * float(jnp.max(Mk)) * 1e-6)


def test_solver_functions_take_the_scale_directly(xk):
    """The three solver functions (used outside make_step) scale too:
    scale s over dt equals scale 1 over s*dt, and the default is unchanged."""
    Nk, Mk, _ = _smooth_state(xk)
    temp, pres = ENV[0], ENV[1]
    for fn, kw in ((coag_euler_step, dict(n_substeps=10)),
                   (euler_step, dict(recompute_kernel=True)),
                   (heun_step, {})):
        ref = fn(Nk, Mk, xk, temp, pres, BOXVOL, 60.0, **kw)
        one = fn(Nk, Mk, xk, temp, pres, BOXVOL, 60.0, coag_kernel_scale=1.0, **kw)
        for x, y in zip(ref, one):
            np.testing.assert_array_equal(np.asarray(x), np.asarray(y))
        a = fn(Nk, Mk, xk, temp, pres, BOXVOL, 60.0, coag_kernel_scale=2.0, **kw)
        b = fn(Nk, Mk, xk, temp, pres, BOXVOL, 120.0, **kw)
        np.testing.assert_allclose(np.asarray(a[0]), np.asarray(b[0]), rtol=1e-10)
        with pytest.raises(ValueError, match='coag_kernel_scale'):
            fn(Nk, Mk, xk, temp, pres, BOXVOL, 60.0, coag_kernel_scale=-0.5, **kw)


# ------------------------------------------------- d. monotonic, conservation

@pytest.mark.parametrize('solver', SOLVERS)
def test_monotonic_response_and_mass_closure(xk, solver):
    Nk, Mk, Gc = _state(xk, MODES)
    step = _coag(solver)
    N0, dry0 = float(jnp.sum(Nk)), _dry(Mk)
    n, closure = {}, {}
    for s in (0.5, 1.0, 2.0):
        out = step(Nk, Mk, Gc, xk, *ENV, coag_kernel_scale=s)
        n[s] = float(jnp.sum(out[0]))
        closure[s] = abs(_dry(out[1]) / dry0 - 1.0)
    assert n[2.0] < n[1.0] < n[0.5] < N0, n
    # loss grows with the scale, sub-linearly (the loss depletes N)
    loss = {s: N0 - n[s] for s in n}
    assert loss[1.0] < 2.0 * loss[0.5] and loss[2.0] < 2.0 * loss[1.0], loss
    # dry mass closes as tightly as it does unscaled (no top-bin overflow here)
    for s in (0.5, 2.0):
        assert closure[s] <= max(1e-13, 10.0 * closure[1.0]), closure


# -------------------------------------------------------------- e. jit, vmap

@pytest.mark.parametrize('solver', SOLVERS)
def test_new_value_does_not_retrace(xk, solver):
    Nk, Mk, Gc = _state(xk, MODES)
    inner = _coag(solver, jit=False)
    traces = []

    @jax.jit
    def f(Nk, Mk, Gc, s):
        traces.append(1)
        return inner(Nk, Mk, Gc, xk, *ENV, coag_kernel_scale=s)

    a = f(Nk, Mk, Gc, 0.5)
    b = f(Nk, Mk, Gc, 2.0)
    assert len(traces) == 1
    assert float(jnp.sum(b[0])) < float(jnp.sum(a[0]))
    # A float64 array is a different (strong) type from a Python float, so
    # JAX traces once more for it, and then never again.
    c_ = f(Nk, Mk, Gc, jnp.float64(1.3))
    f(Nk, Mk, Gc, jnp.float64(0.7))
    assert len(traces) == 2
    assert np.all(np.isfinite(np.asarray(c_[0])))

    # make_step's own jit: one compiled entry for all values
    step = _coag(solver)
    for s in (0.5, 2.0, 3.0):
        step(Nk, Mk, Gc, xk, *ENV, coag_kernel_scale=s)
    if hasattr(step, '_cache_size'):
        assert step._cache_size() == 1


def test_vmap_per_cell_scale(xk):
    """Production config, vmapped over cells with a per-cell scale; each
    cell equals its own unbatched run (batched vs single: round-off)."""
    step = make_step(PROCS, coag_solver='heun_adaptive', gas_substep='adaptive',
                     jit=False, cap_warnings=False, **STUDY)
    temp, pres = 217.0, c.STRAT['pres']
    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(100.0, temp, pres))
    scales = jnp.array([0.5, 1.0, 2.0, 0.0])

    def cell(s):
        return step(*state, xk, temp, pres, c.BOXVOL, c.STRAT['rh'],
                    c.STRAT['alpha'], 60.0, coag_kernel_scale=s, **KW)

    batched = jax.jit(jax.vmap(cell))(scales)
    single = jax.jit(cell)
    for i, s in enumerate(scales):
        ref = single(s)
        for x, y in zip(batched, ref):
            np.testing.assert_allclose(np.asarray(x[i]), np.asarray(y),
                                       rtol=1e-12, atol=1e-300)
    N = np.asarray(jnp.sum(batched[0], axis=1))
    # 0 < 0.5 < 1 < 2 in coagulation strength -> N decreasing
    assert N[3] > N[0] > N[1] > N[2], N


# ------------------------------------------------------------- argument check

def test_negative_or_nonfinite_rejected(xk):
    Nk, Mk, Gc = _state(xk, MODES)
    step = _coag('heun_adaptive', jit=False)
    for bad in (-1.0, float('nan'), float('inf'), np.array([1.0, -0.1])):
        with pytest.raises(ValueError, match='coag_kernel_scale'):
            step(Nk, Mk, Gc, xk, *ENV, coag_kernel_scale=bad)
    # a traced value is not checked (and must not break tracing)
    out = jax.jit(lambda s: step(Nk, Mk, Gc, xk, *ENV,
                                 coag_kernel_scale=s))(1.0)
    assert np.all(np.isfinite(np.asarray(out[0])))


def test_kwarg_is_known(xk):
    """No 'unknown kwargs' warning."""
    import warnings
    Nk, Mk, Gc = _state(xk, MODES)
    with warnings.catch_warnings():
        warnings.simplefilter('error', UserWarning)
        _coag('euler_fixed', jit=False)(Nk, Mk, Gc, xk, *ENV, coag_kernel_scale=1.5)
