"""Scenario fields added for the urban and indoor levels (2026-10-19): emissions, cloud
fraction, losses (ventilation, O3 surface loss, organic vapour wall loss, Lai-Nazaroff
deposition with room parameters) and the seed salts. Defaults leave every existing
preset unchanged (tests/test_aaar_presets_unchanged.py).
"""
import dataclasses
from datetime import datetime, timezone

import numpy as np
import pytest

from tomas_jax.box.scenario import (CLOUD_J_LOSS, TRAFFIC_HOURLY_WEIGHTS, VENTILATION_PRESETS_PER_H,
                                    EmissionProfile, Losses, Room,
                                    Scenario, Seed, SeedMode, from_dict, list_presets, load_preset)


def _error(match, **changes):
    with pytest.raises(ValueError, match=match):
        Scenario().replace(**changes)


def test_defaults_are_off():
    s = Scenario()
    assert s.emissions_ppb_per_h == {} and s.photolysis.cloud_fraction == 0.0
    assert s.photolysis.j_factor == 1.0
    assert not s.losses.any and s.losses.outdoor_seed is None
    assert s.seed.ammonium_nitrate_mass_fraction == 0.0 and s.seed.ammonium is False
    assert s.seed.sulfate_mass_fraction == pytest.approx(0.7)
    for name in ("level1_volatility", "level2_limonene_soa", "level3_day_night"):
        p = load_preset(name)
        assert p.emissions_ppb_per_h == {} and not p.losses.any
        assert p.photolysis.cloud_fraction == 0.0 and p.seed.ammonium is False


def test_emissions_and_cloud_fraction():
    s = Scenario().replace(emissions_ppb_per_h={"NO": 2.0}, photolysis={"cloud_fraction": 0.4})
    assert s.emissions_ppb_per_h == {"NO": 2.0}
    assert s.photolysis.j_factor == pytest.approx(1.0 - CLOUD_J_LOSS * 0.4)
    _error(r"emissions_ppb_per_h\['NO'\] must be at least 0", emissions_ppb_per_h={"NO": -1.0})
    _error(r"emissions_ppb_per_h keys must be species names", emissions_ppb_per_h={1: 1.0})
    _error(r"photolysis\.cloud_fraction .*did you mean 0\.5\?", photolysis={"cloud_fraction": 50})


def test_losses_fields_and_checks():
    s = Scenario().replace(losses={
        "ventilation_per_h": 0.5, "outdoor_gases_ppb": {"O3": 40.0}, "o3_surface_per_h": 3.0,
        "organic_vapour_wall_per_s": 1e-4, "particle_deposition": "lai_nazaroff",
        "outdoor_seed": {"modes": [{"number_cm3": 5000.0, "gmd_nm": 100.0, "gsd": 1.6}],
                         "organic_mass_fraction": 0.5, "ammonium": True},
        "room": {"friction_velocity_cm_s": 2.0}})
    assert s.losses.any and isinstance(s.losses.outdoor_seed, Seed)
    assert s.losses.outdoor_seed.modes[0] == SeedMode(5000.0, 100.0, 1.6)
    assert s.losses.room == Room(friction_velocity_cm_s=2.0)     # other room values kept
    # a second partial override keeps the nested values
    s2 = s.replace(losses={"ventilation_per_h": 3.0})
    assert s2.losses.ventilation_per_h == 3.0 and s2.losses.outdoor_seed == s.losses.outdoor_seed
    assert s2.losses.room.friction_velocity_cm_s == 2.0
    _error(r"losses\.ventilation_per_h must be at least 0", losses={"ventilation_per_h": -1})
    _error(r"losses\.particle_deposition must be one of", losses={"particle_deposition": "on"})
    _error(r"losses\.organic_vapour_wall_per_s .*per second", losses={"organic_vapour_wall_per_s": 2})
    _error(r"losses\.outdoor_gases_ppb\['O3'\]", losses={"outdoor_gases_ppb": {"O3": -5}})
    _error(r"losses\.room\.friction_velocity_cm_s must be above 0",
           losses={"room": {"friction_velocity_cm_s": 0}})
    _error(r"floor and ceiling alone", losses={"room": {"height_m": 0.5}})
    with pytest.raises(TypeError, match="Losses got unexpected keys"):
        from_dict({"losses": {"ventilation": 0.5}})
    assert VENTILATION_PRESETS_PER_H == {"closed_room": 0.5, "window_open": 3.0, "hvac": 6.0}


def test_seed_salts():
    s = Scenario().replace(seed={"organic_mass_fraction": 0.5, "ammonium_nitrate_mass_fraction": 0.2,
                                 "ammonium": True})
    assert s.seed.sulfate_mass_fraction == pytest.approx(0.3)
    _error(r"must be at most 1 \(the rest is sulfate\)",
           seed={"organic_mass_fraction": 0.6, "ammonium_nitrate_mass_fraction": 0.5})
    _error(r"seed\.ammonium must be True or False", seed={"ammonium": "yes"})
    _error(r"seed\.organic_o_to_c", seed={"organic_o_to_c": 3.0})


def test_from_dict_round_trip_with_losses():
    s = Scenario().replace(losses={"ventilation_per_h": 1.0, "outdoor_seed": {}})
    assert s.losses.outdoor_seed == Seed()
    d = dataclasses.asdict(s)
    d["start_utc"] = s.start_utc
    assert from_dict(d) == s


def test_preset_list_still_sorted():
    assert list_presets() == sorted(list_presets())


# --- diurnal emission profile (A26) -----------------------------------------------------------
MIDNIGHT_EDT = datetime(2026, 7, 1, 4, 0, tzinfo=timezone.utc)      # 00:00 local at UTC-4


def test_traffic_weights_are_the_published_profile():
    w = np.array(TRAFFIC_HOURLY_WEIGHTS)
    assert w.size == 24 and w.sum() == pytest.approx(24.0)             # mean 1, as published
    assert w.argmax() == 17 and w.argmin() == 3                        # evening peak, 03:00 low
    assert w[8] > 1.8 and w[17] > 2.0 and w[:6].max() < 0.25           # two rush hours, quiet night


def test_profile_mean_over_a_day_is_one_whatever_the_scale():
    for scale in (1.0, 7.3):
        p = EmissionProfile(utc_offset_h=-4.0, hourly_weights=tuple(scale * w for w in TRAFFIC_HOURLY_WEIGHTS))
        for start in (MIDNIGHT_EDT, datetime(2026, 7, 1, 9, 17, tzinfo=timezone.utc)):
            assert p.mean_weight(start, 0.0, 86400.0) == pytest.approx(1.0, rel=1e-12)
            steps = [p.mean_weight(start, t, t + 300.0) for t in np.arange(0.0, 86400.0, 300.0)]
            assert np.mean(steps) == pytest.approx(1.0, rel=1e-12)


def test_profile_uses_local_time_and_averages_across_hour_edges():
    p = EmissionProfile(utc_offset_h=-4.0)
    w = np.array(TRAFFIC_HOURLY_WEIGHTS)
    assert p.mean_weight(MIDNIGHT_EDT, 0.0, 300.0) == pytest.approx(w[0])           # 00:00 local
    assert p.mean_weight(MIDNIGHT_EDT, 8 * 3600.0, 8 * 3600.0 + 300.0) == pytest.approx(w[8])
    # 07:45-08:15 local: half of hour 7 and half of hour 8
    assert p.mean_weight(MIDNIGHT_EDT, 7.75 * 3600.0, 8.25 * 3600.0) == pytest.approx(0.5 * (w[7] + w[8]))
    # the same instant in UTC with offset 0 is hour 4 of the day
    assert EmissionProfile().mean_weight(MIDNIGHT_EDT, 0.0, 300.0) == pytest.approx(w[4])
    # past local midnight the profile wraps
    assert p.mean_weight(MIDNIGHT_EDT, 24 * 3600.0, 24 * 3600.0 + 300.0) == pytest.approx(w[0])


def test_emission_profile_scenario_field_and_checks():
    assert Scenario().emission_profile is None
    s = Scenario().replace(emission_profile={"utc_offset_h": -4.0})
    assert s.emission_profile == EmissionProfile(-4.0)
    assert s.replace(emission_profile={"utc_offset_h": -5.0}).emission_profile.hourly_weights == \
        TRAFFIC_HOURLY_WEIGHTS
    assert load_preset("level1_urban").emission_profile.utc_offset_h == -4.0
    for name in list_presets():
        if name != "level1_urban":
            assert load_preset(name).emission_profile is None
    _error(r"emission_profile\.utc_offset_h must be from -12 to 14", emission_profile={"utc_offset_h": -20})
    _error(r"emission_profile\.hourly_weights must be 24 numbers", emission_profile={"hourly_weights": (1.0,) * 23})
    _error(r"hourly_weights\[3\] must be at least 0", emission_profile={"hourly_weights": (1.0,) * 3 + (-1.0,) + (1.0,) * 20})
    _error(r"must not all be zero", emission_profile={"hourly_weights": (0.0,) * 24})
    with pytest.raises(ValueError, match="emission_profile must be a scenario.EmissionProfile"):
        Scenario(emission_profile="traffic")
