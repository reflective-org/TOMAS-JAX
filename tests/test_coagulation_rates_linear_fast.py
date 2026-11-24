"""calc_coagulation_rates_linear against the original scatter-add formulation.

The library function evaluates only the pairs i <= j, finds each pair's
destination bin in closed form on geometric grids, and accumulates with one
windowed scatter per array (2026-11-24). The original is kept verbatim in
tests/reference/coagulation_rates_linear_scatter.py. Tests cover:
    - destination bin == searchsorted(xk, x, 'right') - 1 clipped to
      [0, nbins] for every input (edges, ulp neighbours, outside the grid,
      0, negative, inf, NaN) on geometric and non-geometric grids
    - values equal the reference to round-off (<= 1e-12 of the gross loss
      rates, see _gross_scales; eager <= 1e-15) for lognormal,
      nucleation-mode, near-empty, top-bin, xbar-at-edge and random states at
      20/40/80/160 bins and a non-geometric grid, and on sai_plume states (slow)
    - jvp, jacfwd and grad equal the reference's to round-off (calibrated on
      the reference itself, see _assert_tree_close) with the same non-finite
      entries (no new NaN)
    - jit and vmap (shared and batched grids)
    - species mass conserved with overflow
"""
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import ICOMP, ICOMP_NODIAG, XK0, make_grid
from tomas_jax.physics.coagulation_kernel import calc_coagulation_kernel
from tomas_jax.physics.coagulation_rates import _preprocess_concentrations
from tomas_jax.physics.coagulation_rates_linear import (
    _destination_bin, _destination_bin_geometric, calc_coagulation_rates_linear)
from tomas_jax.physics.properties import calc_particle_properties
from tests.reference.coagulation_rates_linear_scatter import (
    calc_coagulation_rates_linear_scatter as reference)

TOL = 1e-12
BOXVOL = 1.0e6
NEPS = 1.0e-20


def grid(nbins):
    return make_grid(nbins, XK0, 2.0 ** (40.0 / nbins))


def non_geometric_grid(nbins=60):
    """Mass ratio rising from 1.2 to 1.9 across the grid (not geometric)."""
    ratios = np.linspace(1.2, 1.9, nbins)
    return jnp.asarray(XK0 * np.concatenate([[1.0], np.cumprod(ratios)]))


GRIDS = {'20': lambda: grid(20), '40': lambda: grid(40), '80': lambda: grid(80),
         '160': lambda: grid(160), 'nongeo60': non_geometric_grid}


# =========================================================================
# Destination bin
# =========================================================================

def _search(xk, x):
    return np.clip(np.searchsorted(np.asarray(xk), np.asarray(x), side='right') - 1,
                   0, xk.shape[0] - 1)


def _probe_points(xk, seed=0):
    xk = np.asarray(xk)
    rng = np.random.default_rng(seed)
    edges = np.concatenate([xk, np.nextafter(xk, np.inf), np.nextafter(xk, -np.inf),
                            xk * (1 + 1e-15), xk * (1 - 1e-15)])
    spread = xk[0] * np.exp(rng.uniform(-3.0, np.log(xk[-1] / xk[0]) + 3.0, 4000))
    special = np.array([0.0, -0.0, -1.0, -np.inf, np.inf, np.nan, 1e300, -1e300,
                        5e-324, 1e-320, np.finfo(float).max, np.finfo(float).tiny,
                        xk[0] * 0.5, xk[-1] * 2.0])
    return np.concatenate([edges, spread, special])


class TestDestinationBin:

    @pytest.mark.parametrize("name", list(GRIDS))
    def test_equals_searchsorted(self, name):
        xk = GRIDS[name]()
        x = _probe_points(xk)
        got = np.asarray(jax.jit(_destination_bin)(xk, jnp.asarray(x)))
        np.testing.assert_array_equal(got, _search(xk, x))

    @pytest.mark.parametrize("name", ['20', '40', '80', '160'])
    def test_closed_form_exact_on_geometric_grids(self, name):
        """The closed-form branch itself (not the fallback) is exact."""
        xk = GRIDS[name]()
        x = _probe_points(xk, seed=1)
        got = np.asarray(jax.jit(_destination_bin_geometric)(xk, jnp.asarray(x)))
        np.testing.assert_array_equal(got, _search(xk, x))

    def test_non_geometric_grid_takes_fallback(self):
        """The closed form is wrong on this grid, so the dispatcher must not use it."""
        xk = non_geometric_grid()
        x = jnp.asarray(_probe_points(xk, seed=2))
        closed = np.asarray(_destination_bin_geometric(xk, x))
        assert np.any(closed != _search(xk, x))
        np.testing.assert_array_equal(np.asarray(_destination_bin(xk, x)), _search(xk, x))

    def test_vmap_over_grids(self):
        """Batched xk turns the branch into a select; both grids still exact."""
        xks = jnp.stack([grid(60), non_geometric_grid(60)])
        x = jnp.stack([jnp.asarray(_probe_points(xk, seed=3)) for xk in xks])
        got = np.asarray(jax.vmap(_destination_bin)(xks, x))
        for b in range(2):
            np.testing.assert_array_equal(got[b], _search(xks[b], x[b]))


# =========================================================================
# States
# =========================================================================

def _x_geo(xk):
    xk = np.asarray(xk)
    return np.sqrt(xk[:-1] * xk[1:])


def _diam(x, rho=1770.0):
    return (6.0 * x / (np.pi * rho)) ** (1.0 / 3.0)


def _fill_mass(xk, N, xbar, rng):
    """Mk with per-particle mass xbar over SO4, two organics and NH4, plus water."""
    nb = N.size
    comp = rng.uniform(0.1, 1.0, (nb, 4))
    comp /= comp.sum(1, keepdims=True)
    M = np.zeros((nb, ICOMP))
    M[:, [0, 1, 2, 42]] = comp * (N * xbar)[:, None]
    M[:, ICOMP - 1] = 0.3 * N * xbar     # water (diagnostic): must not move
    return M


def _state(family, xk, seed=0):
    rng = np.random.default_rng(seed)
    xk_np = np.asarray(xk)
    nb = xk_np.size - 1
    a, b = xk_np[:-1], xk_np[1:]
    xg = _x_geo(xk_np)
    d = _diam(xg)
    xbar = a + rng.uniform(0.05, 0.95, nb) * (b - a)
    if family == 'lognormal':
        N = (1e10 * np.exp(-0.5 * (np.log(d / 5e-9) / 0.4) ** 2)
             + 1e8 * np.exp(-0.5 * (np.log(d / 1e-7) / 0.5) ** 2))
    elif family == 'nucleation':
        N = 1e3 * np.exp(-0.5 * (np.log(d / 1e-7) / 0.5) ** 2)
        N[:4] = [3e12, 8e11, 1e11, 5e9]
        xbar[:4] = a[:4] * 1.001          # just above the lower edge
    elif family == 'near_empty':
        N = 1e8 * np.exp(-0.5 * (np.log(d / 3e-8) / 0.3) ** 2)
        N[N < 1e-6] = 0.0
        idx = rng.choice(nb, nb // 3, replace=False)
        N[idx] = rng.choice([0.0, 1e-25, 0.5 * NEPS, NEPS, 1.0001 * NEPS, 1e-15], idx.size)
    elif family == 'top_bin':
        N = np.zeros(nb)
        N[-1] = 1e6
    elif family == 'edges':
        N = 1e9 * np.exp(-0.5 * (np.log(d / 2e-8) / 0.8) ** 2) + 1.0
        pos = np.array([-0.1, 0.0, 1e-7, 1e-6, 1e-3, 0.2, 0.5 - 1 / 6, 0.5, 0.5 + 1 / 6,
                        0.8, 1 - 1e-6, 1 - 1e-7, 1.0, 1.1])
        xbar = a + np.resize(pos, nb) * (b - a)
    elif family == 'random':
        N = rng.lognormal(10.0, 1.0, nb)
        N[:3] = 0.0
    else:
        raise ValueError(family)
    M = _fill_mass(xk_np, N, xbar, rng)
    return jnp.asarray(N), jnp.asarray(M)


def _kernel(xk, N, M, kind, seed=0):
    nb = N.shape[0]
    if kind == 'brownian':
        props = calc_particle_properties(N, M, 217.0, 5500.0, 1770.0)
        return calc_coagulation_kernel(*props, BOXVOL)
    rng = np.random.default_rng(seed)
    k = rng.uniform(0.5, 1.5, (nb, nb)) * 1e-12
    return jnp.asarray(0.5 * (k + k.T))


FAMILIES = ['lognormal', 'nucleation', 'near_empty', 'top_bin', 'edges', 'random']


def _gross_scales(N, M, kij, xk):
    """Gross loss rates: number [#/s] and per-species mass [kg/s], largest bin.

    dMdt is gain - loss and dM_overflow a small remainder of the gross
    transfer, so their round-off is set by the gross rates, not by their own
    size: the reference alone differs between eager and jit by up to ~2e-11
    of max|dMdt| and ~1e-5 of max|dM_overflow| on sai_plume states (XLA
    fuses gain - loss * mbar into an FMA under jit), but by < 1e-15 of these
    scales.
    """
    Ns, Ms = _preprocess_concentrations(N, M, xk)
    nb = N.shape[0]
    w = jnp.tril(jnp.ones((nb, nb)), k=-1) + 0.5 * jnp.eye(nb)
    R = kij * Ns[:, None] * Ns[None, :] * w
    loss = jnp.sum(R, axis=1) + jnp.sum(R, axis=0)
    mbar = Ms[:, :ICOMP_NODIAG] / Ns[:, None]
    return float(jnp.max(loss)), float(jnp.max(loss[:, None] * mbar))


def _assert_close(new, old, scales, tol=TOL, what=''):
    s_N, s_M = scales
    for name, n, o, s in zip(('dNdt', 'dMdt', 'dM_overflow'), new, old, (s_N, s_M, s_M)):
        n, o = np.asarray(n), np.asarray(o)
        np.testing.assert_array_equal(np.isfinite(n), np.isfinite(o), err_msg=str((what, name)))
        fin = np.isfinite(o)
        err = np.max(np.abs(n - o)[fin], initial=0.0)
        assert err <= tol * s, (what, name, err, s)


# =========================================================================
# Values
# =========================================================================

class TestEquivalence:

    @pytest.mark.parametrize("grid_name", list(GRIDS))
    @pytest.mark.parametrize("family", FAMILIES)
    @pytest.mark.parametrize("kernel", ['brownian', 'random'])
    def test_matches_reference(self, grid_name, family, kernel):
        xk = GRIDS[grid_name]()
        N, M = _state(family, xk, seed=len(family))
        kij = _kernel(xk, N, M, kernel)
        f = jax.jit(calc_coagulation_rates_linear)
        g = jax.jit(reference)
        _assert_close(f(N, M, kij, xk), g(N, M, kij, xk), _gross_scales(N, M, kij, xk),
                      what=(grid_name, family))

    @pytest.mark.parametrize("family", FAMILIES)
    def test_eager_matches_reference(self, family):
        xk = grid(80)
        N, M = _state(family, xk, seed=11)
        kij = _kernel(xk, N, M, 'brownian')
        # eager: same accumulation order and no fusion, so (nearly) bit-identical
        _assert_close(calc_coagulation_rates_linear(N, M, kij, xk),
                      reference(N, M, kij, xk), _gross_scales(N, M, kij, xk), tol=1e-15,
                      what=family)

    @pytest.mark.parametrize("grid_name", ['80', '160', 'nongeo60'])
    @pytest.mark.parametrize("family", FAMILIES)
    def test_mass_conserved_with_overflow(self, grid_name, family):
        xk = GRIDS[grid_name]()
        N, M = _state(family, xk, seed=5)
        kij = _kernel(xk, N, M, 'brownian')
        dN, dM, ov = jax.jit(calc_coagulation_rates_linear)(N, M, kij, xk)
        scale = np.asarray(jnp.sum(jnp.abs(dM), axis=0))
        resid = np.abs(np.asarray(jnp.sum(dM, axis=0) + ov))
        live = scale > 0
        assert np.all(resid[live] <= 1e-12 * scale[live])
        assert float(jnp.max(jnp.abs(dM[:, ICOMP_NODIAG:]))) == 0.0

    @pytest.mark.slow
    @pytest.mark.parametrize("nbins", [40, 80])
    def test_matches_reference_on_plume_states(self, nbins):
        """sai_plume (sensitivities/dunne_nuc) after 0.25 and 1 h at dt = 2 s."""
        from sensitivities.dunne_nuc.common import BOXVOL as PLUME_BOXVOL, SPARSE_DENSITY
        from sensitivities.dunne_nuc.profile_step import states_at
        from sensitivities.dunne_nuc.scenarios import SCENARIOS
        c = SCENARIOS['sai_plume']['cond']
        xk = grid(nbins)
        for h, st in states_at('sai_plume', xk, 2.0, [0.25, 1.0]).items():
            Nk, Mk, _ = st['state']
            props = calc_particle_properties(Nk, Mk, c['temp'], c['pres'], SPARSE_DENSITY)
            kij = calc_coagulation_kernel(*props, PLUME_BOXVOL)
            scales = _gross_scales(Nk, Mk, kij, xk)
            _assert_close(jax.jit(calc_coagulation_rates_linear)(Nk, Mk, kij, xk),
                          jax.jit(reference)(Nk, Mk, kij, xk), scales, what=(nbins, h))
            _assert_close(calc_coagulation_rates_linear(Nk, Mk, kij, xk),
                          reference(Nk, Mk, kij, xk), scales, tol=1e-15, what=(nbins, h))


# =========================================================================
# Derivatives
# =========================================================================

def _functional(rates_fn, xk, w):
    wN, wM, wO = w

    def F(N, M, kij):
        dN, dM, ov = rates_fn(N, M, kij, xk)
        return jnp.sum(wN * dN) + jnp.sum(wM * dM) + jnp.sum(wO * ov)
    return F


def _weights(nb, seed=0):
    rng = np.random.default_rng(seed)
    return (jnp.asarray(rng.normal(size=nb)), jnp.asarray(rng.normal(size=(nb, ICOMP))),
            jnp.asarray(rng.normal(size=ICOMP)))


def _ulp(x, seed):
    """x with every entry moved by one unit in the last place, random sign."""
    rng = np.random.default_rng(seed)
    return x * (1.0 + 2.0 ** -52 * jnp.asarray(rng.choice([-1.0, 1.0], size=x.shape)))


def _assert_tree_close(new, ref, ref_variants, what='', factor=50.0):
    """Same non-finite entries as the reference (no new NaN), finite ones close.

    "Close" is calibrated on the reference itself. Its tangents change by a
    round-off-sized amount between jit and eager (fusion, FMA contraction)
    and when its inputs move by one ulp, because cancellation in gain - loss
    amplifies round-off; relative to a leaf's largest entry that reaches
    ~1e-11 in the gradients below. The new tangents may differ from the
    jitted reference by `factor` times the largest of those changes (floor
    1e-13 of the largest entry). Observed: up to ~30x the one-ulp change,
    3e-10 of the leaf's largest entry.

    The reference's reverse-mode gradient already has NaN entries on these
    states (0 * inf in the transpose of the reconstruction's unselected
    where-branches); the new function must have exactly the same ones.
    """
    for x, y, *zs in zip(*(jax.tree.leaves(t) for t in (new, ref, *ref_variants))):
        x, y = np.asarray(x), np.asarray(y)
        np.testing.assert_array_equal(np.isfinite(x), np.isfinite(y), err_msg=str(what))
        fin = np.isfinite(y)
        for z in zs:
            fin &= np.isfinite(np.asarray(z))
        scale = np.max(np.abs(y[fin]), initial=0.0)
        floor = max(np.max(np.abs(y - np.asarray(z))[fin], initial=0.0) for z in zs)
        err = np.max(np.abs(x - y)[fin], initial=0.0)
        assert err <= max(factor * floor, 1e-13 * scale), (what, err, floor, scale)


class TestDerivatives:

    @pytest.mark.parametrize("grid_name", ['80', '160', 'nongeo60'])
    @pytest.mark.parametrize("family", ['lognormal', 'nucleation', 'near_empty', 'edges'])
    def test_jvp_matches_reference(self, grid_name, family):
        xk = GRIDS[grid_name]()
        N, M = _state(family, xk, seed=2)
        kij = _kernel(xk, N, M, 'brownian')
        rng = np.random.default_rng(3)
        tN = jnp.asarray(rng.normal(size=N.shape)) * N
        tM = jnp.asarray(rng.normal(size=M.shape)) * M
        tK = jnp.asarray(rng.normal(size=kij.shape)) * kij

        def jvp(f, args=(N, M, kij), jit=True):
            g = lambda *p: jax.jvp(lambda N, M, K: f(N, M, K, xk), p, (tN, tM, tK))
            return (jax.jit(g) if jit else g)(*args)
        (v_new, t_new), (v_old, t_old) = jvp(calc_coagulation_rates_linear), jvp(reference)
        _assert_close(v_new, v_old, _gross_scales(N, M, kij, xk), what=(grid_name, family))
        variants = [jvp(reference, jit=False)[1],
                    jvp(reference, args=(_ulp(N, 1), _ulp(M, 2), _ulp(kij, 3)))[1]]
        _assert_tree_close(t_new, t_old, variants, what=(grid_name, family, 'jvp'))

    @pytest.mark.parametrize("grid_name", ['80', '160', 'nongeo60'])
    @pytest.mark.parametrize("family", ['lognormal', 'nucleation', 'near_empty', 'edges'])
    def test_grad_matches_reference(self, grid_name, family):
        xk = GRIDS[grid_name]()
        N, M = _state(family, xk, seed=4)
        kij = _kernel(xk, N, M, 'brownian')
        w = _weights(N.shape[0])

        def grad(f, args=(N, M, kij), jit=True):
            g = jax.grad(_functional(f, xk, w), argnums=(0, 1, 2))
            return (jax.jit(g) if jit else g)(*args)
        variants = [grad(reference, jit=False),
                    grad(reference, args=(_ulp(N, 1), _ulp(M, 2), _ulp(kij, 3)))]
        _assert_tree_close(grad(calc_coagulation_rates_linear), grad(reference), variants,
                           what=(grid_name, family, 'grad'))

    def test_jacfwd_matches_reference(self):
        xk = grid(80)
        N, M = _state('lognormal', xk, seed=6)
        kij = _kernel(xk, N, M, 'brownian')

        def jac(f, N=N, jit=True):
            g = jax.jacfwd(lambda N: f(N, M, kij, xk)[0])
            return (jax.jit(g) if jit else g)(N)
        variants = [jac(reference, jit=False), jac(reference, N=_ulp(N, 1))]
        _assert_tree_close(jac(calc_coagulation_rates_linear), jac(reference), variants,
                           what='jacfwd')


# =========================================================================
# vmap
# =========================================================================

class TestVmap:

    def test_vmap_shared_grid(self):
        xk = grid(80)
        states = [_state(f, xk, seed=9) for f in ('lognormal', 'nucleation', 'edges')]
        N = jnp.stack([s[0] for s in states])
        M = jnp.stack([s[1] for s in states])
        K = jnp.stack([_kernel(xk, n, m, 'brownian') for n, m in states])
        out = jax.jit(jax.vmap(calc_coagulation_rates_linear, in_axes=(0, 0, 0, None)))(N, M, K, xk)
        for b in range(N.shape[0]):
            # vmap may fuse differently (FMA), so compare to round-off
            _assert_close([o[b] for o in out], reference(N[b], M[b], K[b], xk),
                          _gross_scales(N[b], M[b], K[b], xk), what=b)

    def test_vmap_batched_grids(self):
        """Geometric and non-geometric grid in one batch."""
        xks = jnp.stack([grid(60), non_geometric_grid(60)])
        states = [_state('lognormal', xk, seed=10) for xk in xks]
        N = jnp.stack([s[0] for s in states])
        M = jnp.stack([s[1] for s in states])
        K = jnp.stack([_kernel(xk, n, m, 'random', seed=1) for xk, (n, m) in zip(xks, states)])
        out = jax.jit(jax.vmap(calc_coagulation_rates_linear))(N, M, K, xks)
        for b in range(2):
            _assert_close([o[b] for o in out], reference(N[b], M[b], K[b], xks[b]),
                          _gross_scales(N[b], M[b], K[b], xks[b]), what=b)
