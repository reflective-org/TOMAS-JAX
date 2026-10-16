"""Input checks on the scenario classes (tomas_jax.box.scenario).

Common slips used to run silently: RH 80 ran as RH = 80, hold_constant="NO2" held the
"species" N, O and 2, and coagulation="off" stayed on (a non-empty string is true). Each
scenario class now checks its values when it is built and raises ValueError naming the
field. The checks only raise: valid scenarios, and so their results, are unchanged.
"""

import dataclasses
import math

import numpy as np
import pytest

from tomas_jax.box.scenario import (Environment, Processes, Scenario, SeedMode, from_dict,
                                    list_presets, load_preset)


def _error(match, **changes):
    with pytest.raises(ValueError, match=match):
        Scenario().replace(**changes)


def test_presets_and_defaults_still_build():
    for name in list_presets():
        load_preset(name)
    Scenario()


def test_valid_values_pass_through_unchanged():
    s = Scenario().replace(environment={"rh": 0.8}, processes={"nucleation": True},
                           hold_constant=("NO2", "O3"), gases_ppb={"O3": 40.0})
    assert s.environment.rh == 0.8 and s.processes.nucleation is True
    assert s.hold_constant == ("NO2", "O3") and s.gases_ppb == {"O3": 40.0}
    assert dataclasses.asdict(s.seed) == dataclasses.asdict(Scenario().seed)
    for rh in (0.0, 1.0):                                    # the ends of the range are allowed
        assert Environment(rh=rh).rh == rh


def test_rh_in_percent_gets_a_hint():
    _error(r"environment\.rh .*got 80; .*did you mean 0\.8\?", environment={"rh": 80})
    _error(r"environment\.rh must be from 0 to 1, got -0\.1$", environment={"rh": -0.1})
    _error(r"environment\.rh must be a number, got '0\.8'", environment={"rh": "0.8"})


def test_temperature_in_celsius_gets_a_hint():
    _error(r"environment\.temp_k .*got 25; temperature is in kelvin; did you mean 298\.15 K\?",
           environment={"temp_k": 25})
    _error(r"environment\.temp_k", environment={"temp_k": 500.0})
    _error(r"environment\.pres_pa", environment={"pres_pa": 0.0})


def test_one_species_name_is_one_species():
    assert Scenario(hold_constant="NO2").hold_constant == ("NO2",)
    assert Scenario().replace(hold_constant="NO2").hold_constant == ("NO2",)
    assert Scenario(hold_constant=["NO2", "O3"]).hold_constant == ("NO2", "O3")
    with pytest.raises(ValueError, match="hold_constant must be species names"):
        Scenario(hold_constant=[1, 2])


@pytest.mark.parametrize("value", ["off", "on", 0, 1, None, 1.0])
def test_process_switches_are_true_or_false(value):
    _error(r"processes\.coagulation must be True or False", processes={"coagulation": value})


def test_numpy_bools_are_switches():
    assert Processes(nucleation=np.bool_(True)).nucleation


def test_seed_checks():
    with pytest.raises(ValueError, match="gsd must be above 1"):
        SeedMode(1000.0, 80.0, 1.0)
    with pytest.raises(ValueError, match="gmd_nm must be above 0"):
        SeedMode(1000.0, 0.0, 1.6)
    with pytest.raises(ValueError, match="number_cm3 must be at least 0"):
        SeedMode(-5.0, 80.0, 1.6)
    _error(r"seed\.organic_mass_fraction .*did you mean 0\.3\?", seed={"organic_mass_fraction": 30})


def test_photolysis_numerics_and_amounts():
    _error(r"photolysis\.mode must be one of", photolysis={"mode": "TUVX"})
    _error(r"photolysis\.latitude_deg", photolysis={"latitude_deg": 100.0})
    _error(r"photolysis\.constant_sza_deg", photolysis={"constant_sza_deg": 95.0})
    _error(r"numerics\.nbins must be a whole number", numerics={"nbins": 40.0})
    _error(r"numerics\.nbins must be a whole number", numerics={"nbins": True})
    _error(r"numerics\.coupling_dt_s must be above 0", numerics={"coupling_dt_s": 0})
    _error(r"gases_ppb\['O3'\] must be at least 0", gases_ppb={"O3": -1.0})
    _error(r"gases_ppb\['O3'\]", gases_ppb={"O3": math.nan})
    _error(r"gamma_n2o5 must be from 0 to 1", gamma_n2o5=2.0)
    _error(r"duration_h must be above 0", duration_h=0.0)
    _error(r"h2so4_production_cm3_s must be at least 0", h2so4_production_cm3_s=-1.0)
    _error(r"surrogate\.cstar_ugm3 must be above 0", surrogate={"cstar_ugm3": 0.0})


def test_dict_inputs_are_checked_too():
    with pytest.raises(ValueError, match=r"environment\.rh .*did you mean 0\.6\?"):
        from_dict({"environment": {"rh": 60}})
    with pytest.raises(ValueError, match=r"environment must be a scenario\.Environment object"):
        Scenario(environment={"rh": 0.5})
