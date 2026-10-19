"""Scenario fields added for the urban and indoor levels (2026-10-19): emissions, cloud
fraction, losses (ventilation, O3 surface loss, organic vapour wall loss, Lai-Nazaroff
deposition with room parameters) and the seed salts. Defaults leave every existing
preset unchanged (tests/test_aaar_presets_unchanged.py).
"""
import dataclasses

import pytest

from tomas_jax.box.scenario import (CLOUD_J_LOSS, VENTILATION_PRESETS_PER_H, Losses, Room,
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
