"""End-to-end tests of the AAAR tutorial driver (tomas_jax.box.driver)."""
import json

import numpy as np
import pytest

from tomas_jax.box.coupling import ClassBridge, tutorial_mechanism
from tomas_jax.box.driver import _class_map, _cstar_fallback_warnings, code_version, run
from tomas_jax.box.scenario import Scenario, list_presets, load_preset


def test_presets_load():
    # the original levels plus the series added 2026-10-19 (urban warm-up, indoor air)
    assert list_presets() == ["level1_urban", "level1_volatility", "level2_indoor",
                              "level2_limonene_soa", "level3_day_night"]
    assert [load_preset(n).level for n in list_presets()] == [1, 1, 2, 2, 3]
    # every preset couples gas and aerosol every 300 s (Level 3 used 900 s until
    # 2026-10-17; ASSUMPTIONS A12)
    assert [load_preset(n).numerics.coupling_dt_s for n in list_presets()] == [300.0] * 5


def test_presets_use_the_jathar_cstar_table():
    from tomas_jax.gaschem.config import DEFAULT_CSTAR_TABLE, MECHANISM_DIR
    assert DEFAULT_CSTAR_TABLE == "mcm_limonene_cstar_298K.csv"
    assert (MECHANISM_DIR / DEFAULT_CSTAR_TABLE).is_file()
    assert load_preset("level1_volatility").cstar_table is None          # no chemistry
    for name in ("level2_limonene_soa", "level3_day_night"):
        assert load_preset(name).cstar_table == DEFAULT_CSTAR_TABLE
    cm = _class_map(DEFAULT_CSTAR_TABLE)
    assert cm.source.startswith("Shantanu Jathar (2026-10-16) table: Nannoolal et al. (2008) "
                                "vapour pressures via PyCHAM 5.8.0")
    assert Scenario().cstar_table is None                                # Li et al. (2016)
    assert _class_map(None).source.startswith("Li et al. (2016)") and not _class_map(None).fallback


def test_partial_cstar_table_warns():
    """A table that lists only a few species leaves the rest on Li et al. (2016): the run
    warns. The shipped table leaves no condensable species on Li, and Li runs never warn."""
    from tomas_jax.gaschem.volatility import assign_classes
    parsed = tutorial_mechanism()
    partial = assign_classes(parsed, {"C822PAN": 5.0, "LIMANO3": 5.0})
    (msg,) = _cstar_fallback_warnings(partial, "cstar_user.csv")
    assert f"does not list {len(partial.species)} of {len(partial.species)}" in msg
    assert "merge them into the full table" in msg
    assert _cstar_fallback_warnings(_class_map("mcm_limonene_cstar_298K.csv"),
                                    "mcm_limonene_cstar_298K.csv") == []
    assert _cstar_fallback_warnings(_class_map(None), None) == []


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


@pytest.mark.parametrize("table", [None, "mcm_limonene_cstar_298K.csv"])
def test_class_bridge_conserves_mass(table):
    parsed = tutorial_mechanism()
    cm = _class_map(table)
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
    assert r.diagnostics["cstar_source"].startswith("Shantanu Jathar (2026-10-16) table")
    assert r.diagnostics["cstar_fallback"] == ["CO", "CH4", "LIMONENE", "D9"]
    # per-class tracker agrees with the size-resolved SOA mass
    total_tracker = sum(c["particle_ugm3"] for c in r.top_contributors)
    assert total_tracker <= r.totals["SOA_ugm3"][-1] * (1 + 1e-9)


@pytest.mark.slow
def test_level2_soa_with_the_jathar_table():
    """Level 2 preset, 6 h, Jathar c* (A8): ~8.7 ug/m3 of mostly hydroperoxide SOA
    (Li et al. 2016 gave 15.9 ug/m3, 98 % nitrogen-containing). PyCHAM with the same
    Nannoolal volatilities: 7.96 ug/m3, 5 % N."""
    r = run(load_preset("level2_limonene_soa"))
    assert 6.0 < r.totals["SOA_ugm3"][-1] < 12.0
    atoms = tutorial_mechanism().atoms
    top = r.top_contributors
    n_share = (sum(c["particle_ugm3"] for c in top if atoms[c["species"]].get("N"))
               / sum(c["particle_ugm3"] for c in top))
    assert n_share < 0.2
    assert top[0]["species"] == "LIMALOOH"
    assert r.diagnostics["warnings"] == []


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


# --- regressions from the code review --------------------------------------------------
from datetime import datetime, timedelta, timezone  # noqa: E402

from tomas_jax.box.result import _jsonable  # noqa: E402
from tomas_jax.box.scenario import from_dict  # noqa: E402


def test_start_utc_normalised():
    naive = from_dict({"start_utc": "2026-07-01T16:00:00"})
    assert naive.start_utc == datetime(2026, 7, 1, 16, tzinfo=timezone.utc)      # naive = UTC
    ny = timezone(timedelta(hours=-4))
    aware = Scenario(start_utc=datetime(2026, 7, 1, 12, tzinfo=ny))
    assert aware.start_utc == datetime(2026, 7, 1, 16, tzinfo=timezone.utc)
    assert aware.start_utc.utcoffset() == timedelta(0)
    replaced = Scenario().replace(start_utc="2026-01-01T00:00:00+01:00")
    assert replaced.start_utc == datetime(2025, 12, 31, 23, tzinfo=timezone.utc)


def test_partial_seed_override_keeps_modes_and_rejects_typos():
    s = from_dict({"seed": {"organic_mass_fraction": 0.5}})
    assert len(s.seed.modes) == 1 and s.seed.organic_mass_fraction == 0.5
    with pytest.raises(TypeError):
        from_dict({"seed": {"organic_fraction": 0.5}})


def test_counter_species_and_bad_duration_rejected():
    s = load_preset("level2_limonene_soa")
    with pytest.raises(ValueError, match="counter"):
        run(s.replace(hold_constant=("SA",)))
    with pytest.raises(ValueError, match="multiple of the coupling"):
        run(s.replace(duration_h=0.1))


def test_cstar_table_must_be_a_plain_file_name():
    with pytest.raises(ValueError, match="file name"):
        run(load_preset("level2_limonene_soa").replace(cstar_table="../../../etc/passwd"))


def test_json_has_no_nan():
    assert _jsonable({"a": np.float64("nan"), "b": [np.float32(1.5)]}) == {"a": None, "b": [1.5]}
    r = run(load_preset("level1_volatility").replace(duration_h=0.5))
    d = r.to_dict()
    json.dumps(d, allow_nan=False)
    assert d["diagnostics"]["cs_per_s"][0] > 0          # initial sink computed, not NaN/null


def test_observer_is_opt_in_and_sees_every_interval():
    """The benchmark hook sees every interval and does not change the run."""
    from tomas_jax.box.driver import _chemistry
    s = load_preset("level2_limonene_soa").replace(duration_h=1.0, numerics={"output_dt_s": 600.0})
    seen = []
    r_obs = run(s, observer=lambda t, snap: seen.append((t, snap)))
    r_ref = run(s)
    assert [t for t, _ in seen] == [300.0 * i for i in range(13)]
    for key in ("LIMONENE", "O3", "NO2"):
        np.testing.assert_array_equal(r_obs.gases_ppb[key], r_ref.gases_ppb[key])
    parsed, rate_model, _ = _chemistry()
    i_o3 = parsed.species.index("O3")
    m_air = seen[0][1]["y"][i_o3] / (r_ref.gases_ppb["O3"][0] * 1e-9)
    assert seen[-1][1]["y"][i_o3] / m_air * 1e9 == pytest.approx(r_ref.gases_ppb["O3"][-1], rel=1e-12)
    assert seen[0][1]["j"] is None
    j_no2 = seen[1][1]["j"][rate_model.j_names.index("J_NO2")]
    assert j_no2 == pytest.approx(5e-3, rel=1e-12)
    assert all(snap["counters"]["SA"] > 0 for _, snap in seen[1:])   # SO2 + OH -> SA each interval
