"""Time-step caps with nucleation on (tomas_jax.box.driver.time_steps; ASSUMPTIONS A13).

With nucleation on, the driver uses a coupling interval of at most
NUCLEATION_MAX_COUPLING_DT_S and an aerosol micro-step of at most
NUCLEATION_MAX_MICRO_DT_S, keeps the requested output times, records both values in the
diagnostics and adds one warning. Runs without nucleation are unchanged.
"""

import numpy as np
import pytest

from tomas_jax.box.config import AerosolStepConfig
from tomas_jax.box.driver import (NUCLEATION_MAX_COUPLING_DT_S, NUCLEATION_MAX_MICRO_DT_S,
                                  run, time_steps)
from tomas_jax.box.scenario import list_presets, load_preset


def _nucleating(s):
    return s.replace(processes={"nucleation": True})


def _short(s):
    """One requested coupling interval, output at its end."""
    dt = s.numerics.coupling_dt_s
    return s.replace(duration_h=dt / 3600.0, numerics={"output_dt_s": dt})


def test_caps_are_one_minute_and_five_seconds():
    assert NUCLEATION_MAX_COUPLING_DT_S == 60.0
    assert NUCLEATION_MAX_MICRO_DT_S == 5.0


@pytest.mark.parametrize("name", list_presets())
def test_time_steps_per_preset(name):
    s = load_preset(name)
    req = s.numerics.coupling_dt_s
    off = time_steps(s)
    assert (off.coupling_dt_s, off.micro_dt_s, off.split) == (req, AerosolStepConfig().micro_dt, 1)
    on = time_steps(_nucleating(s))
    assert on.coupling_dt_s == 60.0 and on.micro_dt_s == 5.0
    assert on.split * on.coupling_dt_s == req                 # output times stay on the grid
    assert (on.requested_coupling_dt_s, on.requested_micro_dt_s) == (req, 60.0)


def test_time_steps_divides_the_requested_interval():
    s = _nucleating(load_preset("level1_volatility"))
    for req, used in ((900.0, 60.0), (600.0, 60.0), (120.0, 60.0), (90.0, 45.0),
                      (100.0, 50.0), (60.0, 60.0), (45.0, 45.0)):
        ts = time_steps(s.replace(numerics={"coupling_dt_s": req}))
        assert ts.coupling_dt_s == used and ts.split * used == req
    fine = AerosolStepConfig(micro_dt=2.0)                    # a finer micro-step is kept
    assert time_steps(s, fine).micro_dt_s == 2.0
    assert time_steps(s, nucleation_caps=False) == (300.0, 60.0, 1, 300.0, 60.0)


@pytest.mark.parametrize("name", list_presets())
def test_run_uses_capped_steps_and_warns(name):
    """Each preset with nucleation on: the run takes 60 s intervals (observer calls), the
    diagnostics record requested and used values, one warning, same output times."""
    s = _short(_nucleating(load_preset(name)))
    req = s.numerics.coupling_dt_s
    seen = []
    r = run(s, observer=lambda t, snap: seen.append(t))
    assert seen == pytest.approx(np.arange(0.0, req + 1.0, 60.0))
    d = r.diagnostics
    assert (d["requested_coupling_dt_s"], d["coupling_dt_s"]) == (req, 60.0)
    assert (d["requested_micro_dt_s"], d["micro_dt_s"]) == (60.0, 5.0)
    msgs = [w for w in d["warnings"] if w.startswith("Nucleation is on")]
    assert msgs == [f"Nucleation is on, so the coupling interval is 60 s instead of {req:g} s "
                    "and the aerosol micro-step is 5 s instead of 60 s, to follow the "
                    "nucleation burst. Output times are unchanged."]
    assert r.time_h == pytest.approx([0.0, req / 3600.0])


def test_no_warning_when_nothing_is_shortened():
    s = _nucleating(load_preset("level1_volatility")).replace(
        duration_h=120 / 3600.0, numerics={"coupling_dt_s": 60.0, "output_dt_s": 60.0})
    r = run(s, aerosol_cfg=AerosolStepConfig(micro_dt=5.0))
    assert r.diagnostics["warnings"] == []
    assert r.diagnostics["coupling_dt_s"] == 60.0 and r.diagnostics["micro_dt_s"] == 5.0


@pytest.mark.parametrize("name", list_presets())
def test_nucleation_off_is_unchanged(name):
    """Without nucleation the requested steps are used, no warning is added, and the
    results equal those of a run with the caps disabled."""
    s = _short(load_preset(name))
    r = run(s)
    ref = run(s, nucleation_caps=False)
    d = r.diagnostics
    assert d["coupling_dt_s"] == d["requested_coupling_dt_s"] == s.numerics.coupling_dt_s
    assert d["micro_dt_s"] == d["requested_micro_dt_s"] == 60.0
    assert not any(w.startswith("Nucleation is on") for w in d["warnings"])
    a, b = r.to_dict(), ref.to_dict()
    for key in ("time_h", "dndlog10dp_cm3", "dmdlog10dp_ugm3", "totals", "gases_ppb",
                "radicals_cm3", "h2so4_cm3", "volatility"):
        assert a[key] == b[key], key


def test_level1_nucleation_burst_converges():
    """Level 1 at 265 K with 1e6 cm-3 s-1 H2SO4 production: with the caps the peak
    number is within 5 % of a 30 s coupling / 1 s micro-step reference (the study gave
    +1.6 %); without them (300 s / 60 s) it is about 5 % high."""
    s = _nucleating(load_preset("level1_volatility")).replace(
        h2so4_production_cm3_s=1e6, environment={"temp_k": 265.0}, duration_h=1.0,
        numerics={"output_dt_s": 300.0})
    capped = run(s)
    uncapped = run(s, nucleation_caps=False)
    ref = run(s.replace(numerics={"coupling_dt_s": 30.0}), nucleation_caps=False,
              aerosol_cfg=AerosolStepConfig(micro_dt=1.0))
    assert capped.time_h == ref.time_h == uncapped.time_h
    peak = {k: max(r.totals["N_cm3"]) for k, r in
            (("capped", capped), ("uncapped", uncapped), ("ref", ref))}
    assert peak["ref"] > 2.5e4                                      # a real burst
    err_capped = peak["capped"] / peak["ref"] - 1.0
    err_uncapped = peak["uncapped"] / peak["ref"] - 1.0
    assert abs(err_capped) < 0.05
    assert abs(err_uncapped) > 2.0 * abs(err_capped)


def test_warmup_covers_capped_preset_runs():
    """After warmup(), a run of each preset at its own coupling interval (300 s in all
    three since 2026-10-17), with nucleation off and on, needs no new compile of the
    aerosol step or of the gas solver (the API workers rely on this)."""
    from tomas_jax.box.aerosol_step import aerosol_outer_step
    from tomas_jax.box.driver import warmup
    from tomas_jax.gaschem.solver import _integrate

    warmup()
    n_aerosol, n_gas = aerosol_outer_step._cache_size(), _integrate._cached._cache_size()
    for name in list_presets():
        s = load_preset(name)
        run(_short(s))
        run(_short(_nucleating(s)))
    assert aerosol_outer_step._cache_size() == n_aerosol
    assert _integrate._cached._cache_size() == n_gas
