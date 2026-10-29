"""Tests for the ``molar_so4_transfer`` sulfur-bookkeeping option.

Mk[:, SRTSO4] holds SO4 mass (MW 96) and Gc[SRTSO4] holds H2SO4 mass
(MW 98). With ``molar_so4_transfer=True`` every gas->particle sulfate
transfer conserves moles of S,

    S [mol-ish] = Gc[SRTSO4] / 98 + sum(Mk[:, SRTSO4]) / 96,

to ~1e-12 relative (the remaining ~1e-13 comes from coagulation's own mass
bookkeeping and is the same with the option off). With the default False the
transfers stay kg-for-kg (Fortran parity) and every output is bit-identical
to omitting the option.

Tests cover, on the 80-bin grid (make_grid_80bin):
    - nucleation only: nucleation_step and zhao2024_nucleation_step, normal
      and gas-limited (clamped) paths, directly and through make_step
    - condensation only: PPM-JIT and TFL-JIT through make_step, the numpy
      reference paths (condensation_step method='ppm'/'tfl'), and the
      tiny-sink dump to bin 0
    - nucleation + coagulation + condensation through make_step (normal,
      clamped and dump) and every step/scan entry point that takes the
      option; each also checks that False does NOT conserve moles there,
      so the option is known to reach that transfer site
    - False == omitted, bitwise, for make_step, the nucleation steps, the
      condensation steps and the scans
"""
import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.core.config import (
    ICOMP, N_GAS_SPECIES, SRTSO4, SRTORGLAST, MW_H2SO4, MW_SO4, AVOGADRO,
    SV_H2SO4, CS_EPS, make_grid_80bin,
)
from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.physics.condensation_sink import calc_condensation_sink
from tomas_jax.physics.nucleation import (
    nucleation_step, zhao2024_nucleation_step, _MNUC,
)
from tomas_jax.solvers.condensation import (
    make_step, condensation_step, condensation_step_jax,
    condensation_step_tfl_jax, condensation_step_jit,
    combined_step_ppm_jax, combined_step_tfl_jax,
    condensation_step_with_nucleation_jit, full_step_jax,
    run_condensation_scan, run_condensation_scan_tfl,
    run_combined_scan_ppm, run_combined_scan_tfl,
    run_nucleation_condensation_scan, run_full_scan,
)

BOXVOL = 1e6        # cm3
BOXVOL_DUMP = 1e15  # cm3: MNFIX's empty-bin floor then gives CS < CS_EPS
RTOL_S = 1e-12      # relative tolerance on total moles of S

XK80 = make_grid_80bin()
P, RH, ALPHA = 5000.0, 0.017, 1.0
ALL3 = ['nucleation', 'coagulation', 'condensation']

# (temp, h2so4 [molec/cm3], step kwargs, dt): nucleation that leaves gas
# (normal path) and nucleation the gas cannot supply (clamped path; for
# Zhao 2024 via its H2SO4-free pure-organic ULVOC mechanisms)
NORMAL = (240.0, 1e7, dict(org_conc=0.0, nh3_conc=0.0, fion=3.0), 600.0)
CLAMPED = (278.0, 1e3, dict(org_conc=1e15, nh3_conc=1e9, fion=3.0), 600.0)
CLAMPED_ZHAO = (278.0, 1e3, dict(org_conc=1e7, nh3_conc=1e9, fion=3.0,
                                  ulvoc=1e7), 600.0)


def _clamped(scheme):
    """The clamped case for a nucleation scheme (name or step function)."""
    zhao = scheme in ('zhao2024', zhao2024_nucleation_step)
    return CLAMPED_ZHAO if zhao else CLAMPED


def _gc(h2so4_molec_cm3, boxvol=BOXVOL):
    return jnp.zeros(N_GAS_SPECIES).at[SRTSO4].set(
        h2so4_molec_cm3 * boxvol * MW_H2SO4 / 1000.0 / AVOGADRO)


def _strat_state():
    """Background stratospheric mode, ~10 /cm3 around 0.15 um, 80 bins."""
    xk = XK80
    m = jnp.sqrt(xk[:-1] * xk[1:])
    dp_um = 1e6 * jnp.cbrt(6.0 * m / (1770.0 * np.pi))
    dlog = jnp.log(xk[1:] / xk[:-1]) / 3.0
    Nk = (10.0 / (np.sqrt(2 * np.pi) * np.log(1.6))
          * jnp.exp(-(jnp.log(dp_um) - np.log(0.15)) ** 2
                    / (2 * np.log(1.6) ** 2))
          * dlog * BOXVOL)
    Nk = jnp.maximum(Nk, 1e-3)
    Mk = jnp.zeros((xk.shape[0] - 1, ICOMP)).at[:, SRTSO4].set(Nk * m)
    return Nk, Mk


def _uniform_state():
    """1e6 per bin at the bin centre, 80 bins. Condensation-only use: with
    coagulation the top bin's self-coagulation overflows the grid and loses
    mass (documented, the same in both modes)."""
    xk = XK80
    Nk = jnp.ones(xk.shape[0] - 1) * 1e6
    Mk = jnp.zeros((xk.shape[0] - 1, ICOMP)).at[:, SRTSO4].set(
        Nk * jnp.sqrt(xk[:-1] * xk[1:]))
    return Nk, Mk


def _empty_state():
    nb = XK80.shape[0] - 1
    return jnp.zeros(nb), jnp.zeros((nb, ICOMP))


def _s_mol(Mk, Gc):
    """Moles of S (x 1e-3: kg / (g/mol)) in gas H2SO4 + particulate SO4."""
    return float(Gc[SRTSO4] / MW_H2SO4 + jnp.sum(Mk[:, SRTSO4]) / MW_SO4)


def _s_rel_err(Mk0, Gc0, Mk1, Gc1, added_gas_kg=0.0):
    s0 = _s_mol(Mk0, Gc0) + added_gas_kg / MW_H2SO4
    return abs(_s_mol(Mk1, Gc1) - s0) / s0


def _assert_trees_equal(a, b):
    assert len(a) == len(b)
    for x, y in zip(a, b):
        np.testing.assert_array_equal(np.asarray(x), np.asarray(y))


def _env(temp, dt, boxvol=BOXVOL):
    """Positional (xk, temp, pres, boxvol, rh, alpha, dt) for make_step."""
    return (XK80, temp, P, boxvol, RH, ALPHA, dt)


# =========================================================================
# Nucleation only
# =========================================================================

_NUC_STEPS = [nucleation_step, zhao2024_nucleation_step]


def _nuc_args(case):
    temp, _, kw, dt = case
    return (XK80, temp, P, BOXVOL, dt,
            kw['org_conc'], kw['nh3_conc'], kw['fion'])


def _nuc_kw(case):
    """Keyword-only extras of a case (zhao2024_nucleation_step's ulvoc)."""
    return {k: v for k, v in case[2].items()
            if k not in ('org_conc', 'nh3_conc', 'fion')}


class TestNucleation:

    @pytest.mark.parametrize("step_fn", _NUC_STEPS)
    @pytest.mark.parametrize("f", [0.0, 0.1, 0.5])
    def test_normal_path(self, step_fn, f):
        """Same particle gain and dN as kg-for-kg; gas debit x 98/96."""
        Nk, Mk = _strat_state()
        Gc = _gc(NORMAL[1])
        a = step_fn(Nk, Mk, Gc, *_nuc_args(NORMAL), nuc_org_mass_frac=f)
        b = step_fn(Nk, Mk, Gc, *_nuc_args(NORMAL), nuc_org_mass_frac=f,
                    molar_so4_transfer=True)
        assert 0.0 < float(b[2][SRTSO4]) < float(Gc[SRTSO4])  # not gas-limited
        # particle side unchanged by the option, bitwise
        np.testing.assert_array_equal(np.asarray(a[0]), np.asarray(b[0]))
        np.testing.assert_array_equal(np.asarray(a[1]), np.asarray(b[1]))
        d_so4 = float(b[1][0, SRTSO4] - Mk[0, SRTSO4])
        d_gas = float(Gc[SRTSO4] - b[2][SRTSO4])
        assert d_so4 > 0.0
        np.testing.assert_allclose(d_gas, d_so4 * MW_H2SO4 / MW_SO4, rtol=1e-9)
        assert _s_rel_err(Mk, Gc, b[1], b[2]) < RTOL_S
        # and kg-for-kg does not conserve moles at this transfer
        assert _s_rel_err(Mk, Gc, a[1], a[2]) > 1e-10

    @pytest.mark.parametrize("step_fn", _NUC_STEPS)
    @pytest.mark.parametrize("f", [0.0, 0.1, 0.5])
    def test_clamped_path(self, step_fn, f):
        """Gas-limited: all gas used, Gc*96/98 as SO4, dN from cluster SO4."""
        Nk, Mk = _strat_state()
        case = _clamped(step_fn)
        Gc = _gc(case[1])
        Nk1, Mk1, Gc1 = step_fn(Nk, Mk, Gc, *_nuc_args(case), **_nuc_kw(case),
                                nuc_org_mass_frac=f, molar_so4_transfer=True)
        assert float(Gc1[SRTSO4]) == 0.0
        d_so4 = float(Mk1[0, SRTSO4] - Mk[0, SRTSO4])
        np.testing.assert_allclose(d_so4, float(Gc[SRTSO4]) * MW_SO4 / MW_H2SO4,
                                   rtol=1e-9)
        dN = float(Nk1[0] - Nk[0])
        np.testing.assert_allclose(dN * (1.0 - f) * _MNUC, d_so4, rtol=1e-9)
        d_org = float(Mk1[0, SRTORGLAST] - Mk[0, SRTORGLAST])
        np.testing.assert_allclose(d_org, dN * f * _MNUC, rtol=1e-9, atol=0.0)
        assert _s_rel_err(Mk, Gc, Mk1, Gc1) < RTOL_S

    @pytest.mark.parametrize("nucl_scheme", ['ricco_dunne', 'zhao2024'])
    @pytest.mark.parametrize("path", ['normal', 'clamped'])
    def test_make_step_nucleation_only(self, nucl_scheme, path):
        """Through make_step (adaptive substeps + MNFIX)."""
        case = NORMAL if path == 'normal' else _clamped(nucl_scheme)
        temp, c, kw, dt = case
        Nk, Mk = _strat_state()
        Gc = _gc(c)
        step = make_step(['nucleation'], nucl_scheme=nucl_scheme,
                         molar_so4_transfer=True)
        Nk1, Mk1, Gc1 = step(Nk, Mk, Gc, *_env(temp, dt), **kw)
        assert float(jnp.sum(Nk1)) > float(jnp.sum(Nk))
        if path == 'clamped':
            assert float(Gc1[SRTSO4]) == 0.0
        else:
            assert float(Gc1[SRTSO4]) > 0.0
        assert _s_rel_err(Mk, Gc, Mk1, Gc1) < RTOL_S


# =========================================================================
# Condensation only
# =========================================================================

class TestCondensation:

    @pytest.mark.parametrize("cond_method", ['ppm_jit', 'tfl_jit'])
    def test_make_step_condensation_only(self, cond_method):
        Nk, Mk = _uniform_state()
        Gc = _gc(1e7)
        kw = dict(cond_method=cond_method)
        a = make_step(['condensation'], **kw)(Nk, Mk, Gc, *_env(278.0, 600.0))
        b = make_step(['condensation'], molar_so4_transfer=True, **kw)(
            Nk, Mk, Gc, *_env(278.0, 600.0))
        # same gas debit, particles get 96/98 of it
        np.testing.assert_array_equal(np.asarray(a[2]), np.asarray(b[2]))
        d_gas = float(Gc[SRTSO4] - b[2][SRTSO4])
        assert d_gas > 0.0
        d_part_a = float(jnp.sum(a[1][:, SRTSO4] - Mk[:, SRTSO4]))
        d_part_b = float(jnp.sum(b[1][:, SRTSO4] - Mk[:, SRTSO4]))
        np.testing.assert_allclose(d_part_b / d_part_a, MW_SO4 / MW_H2SO4,
                                   rtol=1e-6)
        assert _s_rel_err(Mk, Gc, b[1], b[2]) < RTOL_S
        assert _s_rel_err(Mk, Gc, a[1], a[2]) > 1e-10

    def test_strat_state_ppm(self):
        """The coupled-model background (sparse, 1e-3 floors), PPM-JIT."""
        Nk, Mk = _strat_state()
        Gc = _gc(1e8)
        _, Mk1, Gc1 = make_step(['condensation'], molar_so4_transfer=True)(
            Nk, Mk, Gc, *_env(210.0, 600.0))
        assert float(Gc1[SRTSO4]) < float(Gc[SRTSO4])
        assert _s_rel_err(Mk, Gc, Mk1, Gc1) < RTOL_S

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("method", ['ppm_jit', 'tfl_jit', 'ppm', 'tfl'])
    def test_condensation_step_all_methods(self, method):
        """condensation_step, including the numpy reference paths."""
        Nk, Mk = _uniform_state()
        Nk, Mk = mnfix_jax(Nk, Mk, XK80)
        Gc = _gc(1e7)
        args = (Nk, Mk, Gc, XK80, 278.0, 101325.0, BOXVOL, 0.5, 1.0, 600.0)
        a = condensation_step(*args, method=method)
        b = condensation_step(*args, method=method, molar_so4_transfer=True)
        assert float(b[2][SRTSO4]) < float(Gc[SRTSO4])
        assert _s_rel_err(Mk, Gc, b[1], b[2]) < RTOL_S
        assert _s_rel_err(Mk, Gc, a[1], a[2]) > 1e-10


class TestDumpPath:
    """CS <= CS_EPS: all gas H2SO4 goes to bin 0."""

    @staticmethod
    def _floored_initial():
        """Empty aerosol as MNFIX leaves it (the step's first action)."""
        Nk0, Mk0 = _empty_state()
        Nkf, Mkf = mnfix_jax(Nk0, Mk0, XK80)
        cs, _ = calc_condensation_sink(Nkf, Mkf, 210.0, P, BOXVOL_DUMP,
                                       MW_H2SO4, SV_H2SO4, ALPHA, xk=XK80)
        assert float(cs) <= CS_EPS     # precondition: the dump path is taken
        return Nk0, Mk0, Nkf, Mkf

    def _check(self, Nkf, Mkf, Gc, Nk1, Mk1, Gc1):
        assert float(Gc1[SRTSO4]) == 0.0
        dump_so4 = float(Gc[SRTSO4]) * MW_SO4 / MW_H2SO4
        np.testing.assert_allclose(float(jnp.sum(Mk1[:, SRTSO4] - Mkf[:, SRTSO4])),
                                   dump_so4, rtol=1e-12)
        np.testing.assert_allclose(
            float(jnp.sum(Nk1) - jnp.sum(Nkf)),
            dump_so4 / float(jnp.sqrt(XK80[0] * XK80[1])), rtol=1e-12)
        assert _s_rel_err(Mkf, Gc, Mk1, Gc1) < RTOL_S

    @pytest.mark.parametrize("cond_method", ['ppm_jit', 'tfl_jit'])
    def test_make_step_condensation(self, cond_method):
        Nk0, Mk0, Nkf, Mkf = self._floored_initial()
        Gc = _gc(1e8, BOXVOL_DUMP)
        out = make_step(['condensation'], cond_method=cond_method,
                        molar_so4_transfer=True)(
            Nk0, Mk0, Gc, *_env(210.0, 600.0, BOXVOL_DUMP))
        self._check(Nkf, Mkf, Gc, *out)

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("method", ['ppm', 'tfl'])
    def test_numpy_reference(self, method):
        Nk0, Mk0, Nkf, Mkf = self._floored_initial()
        Gc = _gc(1e8, BOXVOL_DUMP)
        out = condensation_step(Nk0, Mk0, Gc, XK80, 210.0, P, BOXVOL_DUMP,
                                RH, ALPHA, 600.0, method=method,
                                molar_so4_transfer=True)
        self._check(Nkf, Mkf, Gc, *out)

    def test_full_step_no_nucleation(self):
        """Nucleation (scaled to zero) + coagulation + condensation dump."""
        Nk0, Mk0, Nkf, Mkf = self._floored_initial()
        Gc = _gc(1e8, BOXVOL_DUMP)
        Nk1, Mk1, Gc1 = make_step(ALL3, molar_so4_transfer=True)(
            Nk0, Mk0, Gc, *_env(210.0, 600.0, BOXVOL_DUMP),
            org_conc=0.0, nh3_conc=0.0, fion=3.0, fn_scale=0.0)
        assert float(Gc1[SRTSO4]) == 0.0
        assert _s_rel_err(Mkf, Gc, Mk1, Gc1) < RTOL_S


# =========================================================================
# Nucleation + coagulation + condensation
# =========================================================================

class TestFullStep:

    @pytest.mark.parametrize("nucl_scheme", ['ricco_dunne', 'zhao2024'])
    @pytest.mark.parametrize("path", ['normal', 'clamped'])
    def test_make_step_80bin_ppm(self, nucl_scheme, path):
        case = NORMAL if path == 'normal' else _clamped(nucl_scheme)
        temp, c, kw, dt = case
        Nk, Mk = _strat_state()
        Gc = _gc(c)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            step = make_step(ALL3, nucl_scheme=nucl_scheme,
                             molar_so4_transfer=True)
            Nk1, Mk1, Gc1 = step(Nk, Mk, Gc, *_env(temp, dt), **kw)
        assert float(jnp.sum(Nk1)) != float(jnp.sum(Nk))
        if path == 'clamped':
            # the gas-limited nucleation path ran: nucleation alone uses it all
            _, _, g_nuc = make_step(['nucleation'], nucl_scheme=nucl_scheme,
                                    molar_so4_transfer=True)(
                Nk, Mk, Gc, *_env(temp, dt), **kw)
            assert float(g_nuc[SRTSO4]) == 0.0
        assert _s_rel_err(Mk, Gc, Mk1, Gc1) < RTOL_S

    def test_make_step_80bin_tfl(self):
        temp, c, kw, dt = NORMAL
        Nk, Mk = _strat_state()
        Gc = _gc(c)
        _, Mk1, Gc1 = make_step(ALL3, cond_method='tfl_jit',
                                molar_so4_transfer=True)(
            Nk, Mk, Gc, *_env(temp, dt), **kw)
        assert _s_rel_err(Mk, Gc, Mk1, Gc1) < RTOL_S

    def test_with_water_options(self):
        """Coupled-model config: H2SO4 water scheme, water every process."""
        from tomas_jax.physics.water_equilibrium import WATER_SCHEME_H2SO4
        temp, c, kw, dt = NORMAL
        Nk, Mk = _strat_state()
        Gc = _gc(c)
        _, Mk1, Gc1 = make_step(
            ALL3, water_scheme=WATER_SCHEME_H2SO4, water_every_process=True,
            molar_so4_transfer=True)(
            Nk, Mk, Gc, *_env(temp, dt), **kw, nuc_org_mass_frac=0.0)
        assert _s_rel_err(Mk, Gc, Mk1, Gc1) < RTOL_S


# Every other public entry point that takes the option. Each callable maps
# (Nk, Mk, Gc, molar) -> ((Nk, Mk, Gc), gas added [kg]).
_T, _C, _KW, _DT = NORMAL
_ENV = (XK80, _T, P, BOXVOL, RH, ALPHA, _DT)
_NUC = (_KW['org_conc'], _KW['nh3_conc'], _KW['fion'])
_NSTEPS, _PROD = 3, 1e-16   # scans: 3 steps, H2SO4 production [kg/s]
_SCAN_ADD = _NSTEPS * _PROD * _DT


def _scan(fn, *extra, **kw):
    def run(Nk, Mk, Gc, molar):
        out = fn(Nk, Mk, Gc, *_ENV, _NSTEPS, _PROD, *extra,
                 molar_so4_transfer=molar, **kw)
        return out[:3], _SCAN_ADD
    return run


def _single(fn, *extra, **kw):
    def run(Nk, Mk, Gc, molar):
        return fn(Nk, Mk, Gc, *_ENV, *extra, molar_so4_transfer=molar, **kw), 0.0
    return run


_ENTRY_POINTS = {
    'condensation_step_jax': _single(condensation_step_jax),
    'condensation_step_tfl_jax': _single(condensation_step_tfl_jax),
    'condensation_step_jit': _single(condensation_step_jit),
    'combined_step_ppm_jax': _single(combined_step_ppm_jax),
    'combined_step_tfl_jax': _single(combined_step_tfl_jax),
    'condensation_step_with_nucleation_jit': _single(
        condensation_step_with_nucleation_jit, *_NUC, 1.0, 1.0, 1.0, 0.0),
    'full_step_jax': _single(full_step_jax, *_NUC, use_tfl=0.0),
    'run_condensation_scan': _scan(run_condensation_scan),
    'run_condensation_scan_tfl': _scan(run_condensation_scan_tfl),
    'run_combined_scan_ppm': _scan(run_combined_scan_ppm),
    'run_combined_scan_tfl': _scan(run_combined_scan_tfl),
    'run_nucleation_condensation_scan': _scan(
        run_nucleation_condensation_scan, *_NUC, 1.0, 1.0, 1.0, 0.0),
    'run_full_scan': _scan(run_full_scan, *_NUC, 1.0, 1.0, 1.0, 0.0),
}


class TestEntryPoints:

    @pytest.mark.parametrize("name", sorted(_ENTRY_POINTS))
    def test_moles_conserved_only_when_true(self, name):
        Nk, Mk = _strat_state()
        Gc = _gc(_C)
        run = _ENTRY_POINTS[name]
        (_, Mk1, Gc1), added = run(Nk, Mk, Gc, True)
        assert _s_rel_err(Mk, Gc, Mk1, Gc1, added) < 2 * RTOL_S
        (_, Mk0, Gc0), added = run(Nk, Mk, Gc, False)
        assert _s_rel_err(Mk, Gc, Mk0, Gc0, added) > 1e-10

    @pytest.mark.parametrize("name", sorted(_ENTRY_POINTS))
    def test_false_equals_omitted_bitwise(self, name):
        Nk, Mk = _strat_state()
        Gc = _gc(_C)
        run = _ENTRY_POINTS[name]
        (explicit, _) = run(Nk, Mk, Gc, False)
        # same call without the keyword
        omitted = _call_omitted(name, Nk, Mk, Gc)
        _assert_trees_equal(explicit, omitted)


def _call_omitted(name, Nk, Mk, Gc):
    """The _ENTRY_POINTS call with molar_so4_transfer left out."""
    fns = {
        'condensation_step_jax': lambda: condensation_step_jax(Nk, Mk, Gc, *_ENV),
        'condensation_step_tfl_jax': lambda: condensation_step_tfl_jax(Nk, Mk, Gc, *_ENV),
        'condensation_step_jit': lambda: condensation_step_jit(Nk, Mk, Gc, *_ENV),
        'combined_step_ppm_jax': lambda: combined_step_ppm_jax(Nk, Mk, Gc, *_ENV),
        'combined_step_tfl_jax': lambda: combined_step_tfl_jax(Nk, Mk, Gc, *_ENV),
        'condensation_step_with_nucleation_jit': lambda: (
            condensation_step_with_nucleation_jit(
                Nk, Mk, Gc, *_ENV, *_NUC, 1.0, 1.0, 1.0, 0.0)),
        'full_step_jax': lambda: full_step_jax(Nk, Mk, Gc, *_ENV, *_NUC, use_tfl=0.0),
        'run_condensation_scan': lambda: run_condensation_scan(
            Nk, Mk, Gc, *_ENV, _NSTEPS, _PROD),
        'run_condensation_scan_tfl': lambda: run_condensation_scan_tfl(
            Nk, Mk, Gc, *_ENV, _NSTEPS, _PROD),
        'run_combined_scan_ppm': lambda: run_combined_scan_ppm(
            Nk, Mk, Gc, *_ENV, _NSTEPS, _PROD),
        'run_combined_scan_tfl': lambda: run_combined_scan_tfl(
            Nk, Mk, Gc, *_ENV, _NSTEPS, _PROD),
        'run_nucleation_condensation_scan': lambda: run_nucleation_condensation_scan(
            Nk, Mk, Gc, *_ENV, _NSTEPS, _PROD, *_NUC, 1.0, 1.0, 1.0, 0.0),
        'run_full_scan': lambda: run_full_scan(
            Nk, Mk, Gc, *_ENV, _NSTEPS, _PROD, *_NUC, 1.0, 1.0, 1.0, 0.0),
    }
    return tuple(fns[name]())[:3]


# =========================================================================
# Default False == omitted, bitwise
# =========================================================================

class TestDefaultBitwise:

    @pytest.mark.parametrize("processes", [
        ['nucleation'], ['condensation'], ALL3,
        ['so2_chemistry', 'nucleation', 'coagulation', 'condensation',
         'dilution'],
    ])
    @pytest.mark.parametrize("cond_method", ['ppm_jit', 'tfl_jit'])
    @pytest.mark.parametrize("nucl_scheme", ['ricco_dunne', 'zhao2024'])
    def test_make_step(self, processes, cond_method, nucl_scheme):
        Nk, Mk = _strat_state()
        Gc = _gc(1e8)
        kw = dict(org_conc=1e7, nh3_conc=1e9, fion=3.0)
        opts = dict(cond_method=cond_method, nucl_scheme=nucl_scheme)
        a = make_step(processes, **opts)(Nk, Mk, Gc, *_env(240.0, 600.0), **kw)
        b = make_step(processes, molar_so4_transfer=False, **opts)(
            Nk, Mk, Gc, *_env(240.0, 600.0), **kw)
        _assert_trees_equal(a, b)

    @pytest.mark.parametrize("step_fn", _NUC_STEPS)
    @pytest.mark.parametrize("path", ['normal', 'clamped'])
    def test_nucleation_steps(self, step_fn, path):
        case = NORMAL if path == 'normal' else _clamped(step_fn)
        Nk, Mk = _strat_state()
        Gc = _gc(case[1])
        _assert_trees_equal(
            step_fn(Nk, Mk, Gc, *_nuc_args(case), **_nuc_kw(case)),
            step_fn(Nk, Mk, Gc, *_nuc_args(case), **_nuc_kw(case),
                    molar_so4_transfer=False))

    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    @pytest.mark.parametrize("method", ['ppm_jit', 'tfl_jit', 'ppm', 'tfl'])
    def test_condensation_step(self, method):
        Nk, Mk = _uniform_state()
        Gc = _gc(1e7)
        args = (Nk, Mk, Gc, XK80, 278.0, 101325.0, BOXVOL, 0.5, 1.0, 600.0)
        _assert_trees_equal(
            condensation_step(*args, method=method),
            condensation_step(*args, method=method, molar_so4_transfer=False))

    def test_dump_path(self):
        Nk0, Mk0 = _empty_state()
        Gc = _gc(1e8, BOXVOL_DUMP)
        env = _env(210.0, 600.0, BOXVOL_DUMP)
        _assert_trees_equal(
            make_step(['condensation'])(Nk0, Mk0, Gc, *env),
            make_step(['condensation'], molar_so4_transfer=False)(
                Nk0, Mk0, Gc, *env))

    def test_box_config_default(self):
        pytest.importorskip("vbs_jax")   # box/ imports soa_partitioning
        from tomas_jax.box.config import AerosolStepConfig
        assert AerosolStepConfig().molar_so4_transfer is False
