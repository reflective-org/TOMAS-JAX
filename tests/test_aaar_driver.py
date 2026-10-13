"""End-to-end tests of the AAAR tutorial driver (tomas_jax.box.driver)."""
import json

import numpy as np
import pytest

from tomas_jax.box.coupling import ClassBridge, tutorial_mechanism
from tomas_jax.box.driver import _class_map, code_version, run
from tomas_jax.box.scenario import Scenario, list_presets, load_preset


def test_presets_load():
    assert list_presets() == ["level1_volatility", "level2_limonene_soa", "level3_day_night"]
    assert [load_preset(n).level for n in list_presets()] == [1, 2, 3]


def test_level1_volatility_controls_growth():
    s = load_preset("level1_volatility").replace(duration_h=6.0)
    soa = {}
    for cstar in (0.01, 100.0):
        r = run(s.replace(surrogate={"cstar_ugm3": cstar}))
        soa[cstar] = r.totals["SOA_ugm3"][-1]
        produced = 0.5 * 6.0
        gas = r.volatility["gas_ugm3"][-1][0]
        assert soa[cstar] + gas == pytest.approx(produced, rel=1e-3)     # organic budget closes
        assert r.diagnostics["warnings"] == []
    assert soa[0.01] > 0.95 * 3.0 and soa[100.0] < 0.05
    json.dumps(r.to_dict())                                             # JSON-ready


def test_level1_zero_vapor_only_coagulates():
    s = load_preset("level1_volatility").replace(
        duration_h=6.0, surrogate={"production_ugm3_per_h": 0.0}, h2so4_production_cm3_s=0.0)
    r = run(s)
    dry = np.array(r.totals["dry_ugm3"])
    n = np.array(r.totals["N_cm3"])
    assert np.all(np.isfinite(dry)) and np.all(np.isfinite(n))
    assert dry[-1] == pytest.approx(dry[0], rel=1e-6)                   # no source, no loss
    assert np.all(np.diff(n) <= 0)                                      # coagulation only


def test_class_bridge_conserves_mass():
    parsed = tutorial_mechanism()
    cm = _class_map(None)
    bridge = ClassBridge(cm, boxvol_cm3=1e10)
    rng = np.random.default_rng(0)
    y = np.zeros(len(parsed.species))
    y[cm.species_idx] = rng.uniform(1e6, 1e9, cm.species_idx.size)
    before = bridge.gas_mass(y)
    cls_before = bridge.class_sum(before)
    # condense half of every class
    y1 = bridge.scatter(y, before, 0.5 * cls_before, 0.5 * cls_before)
    gas1 = bridge.gas_mass(y1)
    np.testing.assert_allclose(bridge.class_sum(gas1) + bridge.class_sum(bridge.particle),
                               cls_before, rtol=1e-12)
    # evaporate a quarter of the particle mass back
    p_cls = bridge.class_sum(bridge.particle)
    y2 = bridge.scatter(y1, gas1, bridge.class_sum(gas1) + 0.25 * p_cls, 0.75 * p_cls)
    np.testing.assert_allclose(bridge.class_sum(bridge.gas_mass(y2)) + bridge.class_sum(bridge.particle),
                               cls_before, rtol=1e-12)


@pytest.mark.slow
def test_level2_limonene_forms_soa():
    r = run(load_preset("level2_limonene_soa").replace(duration_h=1.0))
    g = r.gases_ppb
    assert g["LIMONENE"][-1] < 0.5 * g["LIMONENE"][0]
    assert r.totals["SOA_ugm3"][-1] > 0.5
    assert len(r.top_contributors) > 0
    assert r.diagnostics["cstar_source"].startswith("Li et al.")
    # per-class tracker agrees with the size-resolved SOA mass
    total_tracker = sum(c["particle_ugm3"] for c in r.top_contributors)
    assert total_tracker <= r.totals["SOA_ugm3"][-1] * (1 + 1e-9)


@pytest.mark.slow
def test_level3_gamma_controls_nitrate():
    s = load_preset("level3_day_night")
    nit = {g: run(s.replace(gamma_n2o5=g)).totals["NIT_ugm3"][-1] for g in (0.0, 0.02)}
    assert nit[0.0] == 0.0
    assert nit[0.02] > 1.0


def test_code_version_is_stable():
    assert code_version() == code_version() and len(code_version()) == 16


def test_unknown_species_rejected():
    s = load_preset("level2_limonene_soa").replace(gases_ppb={"NOT_A_SPECIES": 1.0})
    with pytest.raises(ValueError, match="unknown mechanism species"):
        run(s)


def test_surrogate_and_chemistry_exclusive():
    s = load_preset("level2_limonene_soa").replace(surrogate={"cstar_ugm3": 1.0})
    assert isinstance(s, Scenario)
    with pytest.raises(ValueError, match="either the surrogate"):
        run(s)
