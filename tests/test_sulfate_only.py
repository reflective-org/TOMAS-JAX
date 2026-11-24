"""TOMAS_SPECIES=sulfate must reproduce the default 44-column layout bit-for-bit
on SO4-only inputs. The layout is fixed at import, so each layout runs in its
own subprocess (CPU only) and the saved outputs are compared here."""
import json
import os
import subprocess
import sys
import textwrap

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WORKER = textwrap.dedent('''
    import sys, json
    import numpy as np
    import jax, jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)
    from tomas_jax.core.config import (NBINS, ICOMP, N_GAS_SPECIES, SRTSO4,
                                       SRTH2O, SRTSO2, SRTORGLAST, HAS_ORG)
    from tomas_jax.solvers.condensation import make_step
    from tomas_jax.physics.nucleation import DEFAULT_NUCLEATION_PARAMS
    NUC = DEFAULT_NUCLEATION_PARAMS._replace(org_frac=0.0)   # sulfate layout requires it
    out_path, procs = sys.argv[1], json.loads(sys.argv[2])
    assert NBINS == 40, NBINS
    BOXVOL = 1e12
    xk = jnp.asarray(1.4e-22 * 2.0 ** np.arange(NBINS + 1))
    # (T K, P Pa, RH, SO2 kg/box, H2SO4 kg/box, OH molec/cm3, Nmode cm-3)
    cells = [
        (200., 1000., 0.01, 1e-12, 1e-16, 1e5, 1e2),
        (210., 2000., 0.05, 1e-10, 1e-14, 1e6, 1e3),
        (215., 3000., 0.20, 1e-8,  1e-12, 1e6, 1e4),
        (225., 4000., 0.35, 1e-6,  1e-11, 5e5, 1e5),
        (230., 5000., 0.50, 1e-9,  1e-13, 1e6, 1e4),
        (205., 2500., 0.10, 5.0,   1e-10, 1e6, 1e3),   # high-SO2 plume
    ]
    step = make_step(procs, coag_solver="heun_adaptive")
    res = {k: [] for k in ("Nk", "so4", "so4_org", "h2o", "so2", "h2so4")}
    for (T, P, RH, so2, h2so4, oh, n0) in cells:
        i = np.arange(NBINS)
        Nk = jnp.asarray(n0 * BOXVOL * np.exp(-0.5 * ((i - 12) / 4.0) ** 2))
        Mk = jnp.zeros((NBINS, ICOMP)).at[:, SRTSO4].set(Nk * xk[:NBINS] * 1.0)
        Gc = (jnp.zeros(N_GAS_SPECIES).at[SRTSO4].set(h2so4).at[SRTSO2].set(so2))
        r = step(Nk, Mk, Gc, xk, T, P, BOXVOL, RH, 1.0, 60.0, oh_conc=oh,
                 org_conc=0.0, nh3_conc=0.0, nucl_params=NUC)
        res["Nk"].append(np.asarray(r[0]))
        res["so4"].append(np.asarray(r[1][:, SRTSO4]))
        # nucleation's organic share: default -> SRTORGLAST column, sulfate -> folded into SO4
        res["so4_org"].append(np.asarray(r[1][:, SRTSO4] + (r[1][:, SRTORGLAST] if HAS_ORG else 0.0)))
        res["h2o"].append(np.asarray(r[1][:, SRTH2O]))
        res["so2"].append(np.asarray(r[2][SRTSO2]))
        res["h2so4"].append(np.asarray(r[2][SRTSO4]))
    np.savez(out_path, **{k: np.array(v) for k, v in res.items()})
''')

CASES = {
    "full_chain": ["so2_chemistry", "nucleation", "coagulation", "condensation"],
    "coag_only": ["coagulation"],
    "cond_only": ["condensation"],
}
FIELDS = ("Nk", "so4", "so4_org", "h2o", "so2", "h2so4")


def _run(tmp_path, layout, name):
    env = dict(os.environ, JAX_PLATFORMS="cpu", PYTHONPATH=ROOT)
    env.pop("TOMAS_SPECIES", None)
    if layout == "sulfate":
        env["TOMAS_SPECIES"] = "sulfate"
    out = tmp_path / f"{layout}_{name}.npz"
    p = subprocess.run([sys.executable, "-c", WORKER, str(out), json.dumps(CASES[name])],
                       env=env, cwd=ROOT, capture_output=True, text=True)
    assert p.returncode == 0, p.stderr[-3000:]
    return np.load(out)


def _maxrel(a, b):
    den = np.maximum(np.abs(a), np.abs(b))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den == 0, 0.0, np.abs(a - b) / den)
    return float(np.nanmax(r))


PLUME = 5      # index of the high-SO2 plume cell in the worker
TIGHT = 1e-12  # round-off level (summation order differs with column count)


def _close(a, b, rtol):
    scale = np.abs(a).max(axis=-1, keepdims=True) if a.ndim > 1 else np.abs(a)
    return bool(np.all(np.abs(a - b) <= rtol * np.maximum(scale, 1e-300)))


@pytest.mark.parametrize("name", list(CASES))
def test_sulfate_matches_default(tmp_path, name):
    d = _run(tmp_path, "default", name)
    s = _run(tmp_path, "sulfate", name)
    diffs = {f: _maxrel(d[f], s[f]) for f in FIELDS}
    print(name, "max rel diff:", diffs)
    # Gas phase is bit-identical in every case.
    for f in ("so2", "h2so4"):
        assert np.array_equal(d[f], s[f], equal_nan=True), (name, f, diffs)
    # org_frac = 0 in both layouts: every field agrees to round-off, plume cell included.
    for f in ("Nk", "so4", "so4_org", "h2o"):
        assert _close(d[f], s[f], TIGHT), (name, f, diffs)


def test_reduced_layout_rejects_unsupported_kwargs_under_jit():
    """jit traces every kwarg; the guard must still fire (default jit=True)."""
    import os
    import pytest
    if os.environ.get("TOMAS_SPECIES", "default") != "sulfate":
        pytest.skip("reduced layout only")
    import jax.numpy as jnp
    from tomas_jax.solvers.condensation import make_step
    from tomas_jax.physics.nucleation import DEFAULT_NUCLEATION_PARAMS as D
    NUC10 = D._replace(org_frac=0.1)
    NUC0 = D._replace(org_frac=0.0)
    for jit in (True, False):
        st = make_step(['so2_chemistry', 'nucleation', 'coagulation',
                        'condensation'], coag_solver='heun_adaptive', jit=jit)
        args = (jnp.ones(40), jnp.ones((40, 3)), jnp.ones(3), jnp.ones(41),
                270., 1e5, 1e9, .5, 1., 10.)
        for kw in (dict(org_conc=1e6), dict(nh3_conc=1e9), dict(ulvoc=1.0),
                   dict(enable_organic=0.5)):
            with pytest.raises(ValueError):
                st(*args, nucl_params=NUC0, **kw)
        for kw in (dict(nucl_params=object()), dict(nucl_params=None),
                   dict(nucl_params=NUC10), dict()):    # missing / org_frac != 0
            with pytest.raises(ValueError, match="org_frac"):
                st(*args, **kw)

        # the rule: org_frac == 0 is accepted, with and without jit
        st(*args, nucl_params=NUC0)
