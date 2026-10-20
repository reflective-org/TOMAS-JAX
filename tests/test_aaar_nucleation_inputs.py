"""Nucleation inputs besides H2SO4 (ion-pair production, NH3, the nucleating organic vapour)
and the Level 1 coupling with nucleation on (ASSUMPTIONS A13, A21-A24; docs/aaar/coupling.md
"Nucleation inputs").

Nucleation is off in every preset, so the stored fingerprints
(tests/test_aaar_presets_unchanged.py) and the series preset tests guard the runs
without it; these tests cover the inputs and the runs with it on.
"""
import datetime
import math

import jax.numpy as jnp
import numpy as np
import pytest

from tomas_jax.box.driver import _processes, run
from tomas_jax.box.scenario import (NUCLEATION_ORGANIC_LOG10_CSTAR_MAX, Nucleation, Scenario,
                                    from_dict, list_presets, load_preset)
from tomas_jax.core.config import AVOGADRO, MW_H2SO4, N_GAS_SPECIES, SRTSO4
from tomas_jax.gaschem.environment import air_environment
from tomas_jax.physics.nucleation import ricco_dunne_nucleation_rate

BOXVOL = 1e10   # [cm3]


def _rate(h2so4_cm3, temp_k, *, q=0.0, nh3_ppb=0.0, org_cm3=0.0, pres_pa=101325.0):
    """The code's own Riccobono + Dunne rate [cm-3 s-1] for the scenario-style inputs."""
    gc = np.zeros(N_GAS_SPECIES)
    gc[SRTSO4] = h2so4_cm3 * BOXVOL * MW_H2SO4 * 1e-3 / AVOGADRO
    m = air_environment(temp_k, pres_pa, 0.5)["M"]
    # jnp scalars, as the driver passes them (numpy scalars warn on the 0/0 of the
    # unused ternary branch)
    return float(ricco_dunne_nucleation_rate(jnp.asarray(gc), jnp.float64(temp_k),
                                             jnp.float64(pres_pa), jnp.float64(BOXVOL),
                                             jnp.float64(org_cm3), jnp.float64(nh3_ppb * 1e-9 * m),
                                             jnp.float64(q)))


# --- inputs ---------------------------------------------------------------------------------
def test_defaults_and_presets():
    n = Nucleation()
    assert (n.ion_pairs_cm3_s, n.nh3_ppb, n.organic) == (4.0, 3.0, True)
    assert n.organic_log10_cstar_max == NUCLEATION_ORGANIC_LOG10_CSTAR_MAX == -0.5
    for name in list_presets():
        s = load_preset(name)
        assert s.processes.nucleation is False               # off everywhere (fingerprints)
        assert isinstance(s.nucleation, Nucleation)
    assert load_preset("level2_indoor").nucleation.nh3_ppb == 30.0      # indoor sources
    for name in ("level1_urban", "level3_day_night"):
        assert load_preset(name).nucleation == Nucleation()


def test_replace_and_from_dict_build_the_inputs():
    s = Scenario().replace(nucleation={"nh3_ppb": 10.0, "organic": False})
    assert s.nucleation == Nucleation(nh3_ppb=10.0, organic=False)
    s = from_dict({"nucleation": {"ion_pairs_cm3_s": 0.0}, "processes": {"nucleation": True}})
    assert s.nucleation.ion_pairs_cm3_s == 0.0 and s.nucleation.nh3_ppb == 3.0
    with pytest.raises(TypeError, match="unexpected keys"):
        Scenario().replace(nucleation={"nh3": 3.0})
    with pytest.raises(ValueError, match=r"nucleation must be a scenario\.Nucleation object"):
        Scenario(nucleation={"nh3_ppb": 3.0})


@pytest.mark.parametrize("field,value,match", [
    ("ion_pairs_cm3_s", -1.0, r"nucleation\.ion_pairs_cm3_s must be from 0 to 10000, got -1\.0; "
                              r"near the ground it is a few \(2-10\) ion pairs"),
    ("ion_pairs_cm3_s", "4", r"nucleation\.ion_pairs_cm3_s must be a number, got '4'"),
    ("nh3_ppb", 7.4e10, r"nucleation\.nh3_ppb must be from 0 to 10000, got 74000000000\.0; "
                        r"NH3 is in ppb .* not molec/cm3"),
    ("nh3_ppb", math.nan, r"nucleation\.nh3_ppb"),
    ("organic", "off", r"nucleation\.organic must be True or False, got 'off'"),
    ("organic", 1, r"nucleation\.organic must be True or False"),
    ("organic_log10_cstar_max", 0.3e-3, None),             # a valid log10 value (no error)
    ("organic_log10_cstar_max", 1e3, r"nucleation\.organic_log10_cstar_max must be from -15 to 10, "
                                     r"got 1000\.0; it is log10 of c\* in ug/m3"),
])
def test_inputs_are_checked(field, value, match):
    if match is None:
        assert getattr(Nucleation(**{field: value}), field) == value
        return
    with pytest.raises(ValueError, match=match):
        Nucleation(**{field: value})
    with pytest.raises(ValueError, match=match):
        Scenario().replace(nucleation={field: value})


def test_valid_inputs_pass_through_unchanged():
    n = Nucleation(ion_pairs_cm3_s=0.0, nh3_ppb=0.0, organic=False, organic_log10_cstar_max=-3.5)
    assert (n.ion_pairs_cm3_s, n.nh3_ppb, n.organic, n.organic_log10_cstar_max) == (0.0, 0.0, False, -3.5)
    assert Nucleation(ion_pairs_cm3_s=np.float64(2.5)).ion_pairs_cm3_s == 2.5


# --- the rate with the default inputs --------------------------------------------------------
def test_rate_order_binary_ions_nh3_organic():
    """At 1e8 cm-3 H2SO4 and 298 K: binary alone < with ions < with ions and NH3; the
    organic path adds on top. With everything at 0 the rate is the pure binary one."""
    d = Nucleation()
    binary = _rate(1e8, 298.15)
    ions = _rate(1e8, 298.15, q=d.ion_pairs_cm3_s)
    ions_nh3 = _rate(1e8, 298.15, q=d.ion_pairs_cm3_s, nh3_ppb=d.nh3_ppb)
    everything = _rate(1e8, 298.15, q=d.ion_pairs_cm3_s, nh3_ppb=d.nh3_ppb, org_cm3=1e7)
    assert 0.0 < binary < 1e-7                              # 6e-9 measured 2026-10-20
    assert binary < ions < 1e-2                             # 2e-4
    assert 1.0 < ions_nh3 < 100.0                            # about 6 cm-3 s-1
    assert ions < ions_nh3 < everything
    assert everything - ions_nh3 == pytest.approx(_rate(1e8, 298.15, org_cm3=1e7) - binary, rel=1e-9)
    # the indoor NH3 (30 ppb) gives more ternary nucleation than the outdoor 3 ppb
    indoor = load_preset("level2_indoor").nucleation
    assert _rate(1e8, 295.15, q=indoor.ion_pairs_cm3_s, nh3_ppb=indoor.nh3_ppb) > ions_nh3


# --- the driver -----------------------------------------------------------------------------
def test_nucleation_turns_h2so4_condensation_on_only_with_nucleation():
    urban = load_preset("level1_urban")
    assert _processes(urban, False) == ("coagulation",)
    assert _processes(urban.replace(processes={"nucleation": True}), False) == (
        "nucleation", "coagulation", "h2so4_condensation")
    l3 = load_preset("level3_day_night")
    assert _processes(l3, True) == ("coagulation", "h2so4_condensation", "soa_partitioning")
    assert _processes(l3.replace(processes={"nucleation": True}), True) == (
        "nucleation", "coagulation", "h2so4_condensation", "soa_partitioning")


def test_nucleation_off_passes_no_inputs():
    s = load_preset("level1_volatility").replace(duration_h=300 / 3600.0, numerics={"output_dt_s": 300.0})
    r = run(s)
    assert r.diagnostics["nucleation_inputs"] is None
    assert r.diagnostics["nucleating_organic_cm3"] == [0.0, 0.0]
    assert not any("H2SO4 condensation is on as well" in w for w in r.diagnostics["warnings"])


def _urban_so2(ppb, hours):
    s = load_preset("level1_urban").replace(processes={"nucleation": True}, duration_h=hours)
    return s.replace(gases_ppb={**s.gases_ppb, "SO2": ppb})


def test_level1_urban_high_so2_forms_new_particles():
    """Urban air with nucleation on and 30 ppb SO2 (a polluted daytime value): the
    chemistry's H2SO4 reaches the particles (it was 0 before 2026-10-20), new particles
    appear in the smallest bins and grow, while the total N still falls by coagulation
    (ASSUMPTIONS A24). 1 h fast; the 6 h preset run is in the slow test."""
    s = _urban_so2(30.0, 1.0)
    r = run(s)
    d = r.diagnostics
    assert d["processes"] == ["nucleation", "coagulation", "h2so4_condensation"]
    assert any(w == "H2SO4 condensation is on as well, because nucleation is on: the new particles "
               "need it to grow." for w in d["warnings"])
    # the time-step warning is still the only one starting "Nucleation is on"
    assert sum(w.startswith("Nucleation is on") for w in d["warnings"]) == 1
    inputs = d["nucleation_inputs"]
    m = air_environment(s.environment.temp_k, s.environment.pres_pa, s.environment.rh)["M"]
    assert inputs["ion_pairs_cm3_s"] == 4.0
    assert inputs["nh3_cm3"] == pytest.approx(3.0e-9 * m)
    assert inputs["organic_species"] and inputs["organic_log10_cstar_max"] == -0.5
    assert max(d["nucleating_organic_cm3"]) == 0.0           # no limonene in urban air
    h2so4 = np.array(r.h2so4_cm3)
    j = np.array(d["nucleation_rate_cm3_s"])
    assert h2so4[0] == 0.0 and h2so4[1:].min() > 1e7 and h2so4.max() < 1e9
    assert j[1:].min() > 0.05 and j.max() < 10.0            # about 0.1 -> 0.5 cm-3 s-1
    dn = np.array(r.dndlog10dp_cm3)
    small = dn[:, np.array(r.diameter_nm) < 10.0].sum(axis=1) * math.log10(2.0) / 3.0
    assert small[0] < 20.0 and small[-1] > 5.0 * small[0]   # new particles below 10 nm
    assert np.all(np.diff(small) > 0)
    n = np.array(r.totals["N_cm3"])
    assert n[-1] < n[0]                                     # coagulation still wins overall
    assert r.totals["SO4_ugm3"][-1] > r.totals["SO4_ugm3"][0]   # H2SO4 condenses


def test_level1_urban_without_so2_has_no_h2so4():
    """Nucleation on but no SO2: the H2SO4 source is zero, so the run is the coagulation
    run at the nucleation time steps (no burst, no condensation)."""
    s = load_preset("level1_urban").replace(processes={"nucleation": True}, duration_h=0.5,
                                            numerics={"output_dt_s": 600.0})
    r = run(s)
    assert max(r.h2so4_cm3) == 0.0 and max(r.diagnostics["nucleation_rate_cm3_s"]) == 0.0
    assert r.totals["SO4_ugm3"][-1] == pytest.approx(r.totals["SO4_ugm3"][0], rel=1e-9)


@pytest.mark.slow
def test_level1_urban_high_so2_six_hours():
    """The 6 h urban preset with 30 ppb SO2 (measured 2026-10-20): H2SO4 peaks at
    4.4e7 cm-3 and J at 1.0 cm-3 s-1 near solar noon (3 h in); the sub-10 nm number
    rises from 10 to 220 cm-3 and falls again as the sun goes down, the new particles
    being scavenged by the 2e4 cm-3 of seed before they grow past 10 nm; total N ends
    at 12.6e3 against 11.5e3 without SO2."""
    r = run(_urban_so2(30.0, 6.0))
    t = np.array(r.time_h)
    h2so4, j = np.array(r.h2so4_cm3), np.array(r.diagnostics["nucleation_rate_cm3_s"])
    assert 3e7 < h2so4.max() < 6e7 and 0.5 < j.max() < 2.0
    assert 2.0 <= t[j.argmax()] <= 4.0                       # daytime, near solar noon
    dn = np.array(r.dndlog10dp_cm3)
    small = dn[:, np.array(r.diameter_nm) < 10.0].sum(axis=1) * math.log10(2.0) / 3.0
    assert 100.0 < small.max() < 500.0 and small[-1] < 0.5 * small.max()
    n = np.array(r.totals["N_cm3"])
    assert 1.2e4 < n[-1] < 1.35e4 and n[-1] < n[0]


def _indoor_so2(ppb, hours):
    s = load_preset("level2_indoor").replace(processes={"nucleation": True}, duration_h=hours)
    return s.replace(gases_ppb={**s.gases_ppb, "SO2": ppb})


def test_level2_indoor_high_so2_bursts_in_the_dark():
    """Indoor air with nucleation on and 30 ppb SO2: there is no sunlight, but the
    ozonolysis OH (peak 8e5 cm-3) oxidises SO2 onto only 3000 cm-3 of particles, so
    H2SO4 reaches 9e7 cm-3 within 10 minutes and, with 30 ppb NH3, the ternary rate
    reaches 170 cm-3 s-1: N 3000 -> 7.4e4 cm-3 at 20 min, then back to 4.6e3 at 4 h
    as the limonene (and so OH and H2SO4) runs out and the 2 nm particles coagulate
    (measured 2026-10-20; 1 h here, the 4 h preset in the slow test). BioOxOrg is
    4e6 cm-3 (13 LVOC species) and adds little next to the ternary path."""
    r = run(_indoor_so2(30.0, 1.0))
    d = r.diagnostics
    assert d["processes"] == ["nucleation", "coagulation", "h2so4_condensation", "soa_partitioning"]
    assert not any("H2SO4 condensation is on as well" in w for w in d["warnings"])
    assert d["nucleation_inputs"]["nh3_cm3"] == pytest.approx(
        30e-9 * air_environment(295.15, 101325.0, 0.4)["M"])
    bio = np.array(d["nucleating_organic_cm3"])
    assert bio[0] == 0.0 and 1e6 < bio[1:].max() < 1e7
    h2so4, j = np.array(r.h2so4_cm3), np.array(d["nucleation_rate_cm3_s"])
    assert 5e7 < h2so4.max() < 2e8 and 50.0 < j.max() < 500.0
    assert h2so4.argmax() == 1 and j.argmax() == 1          # first output (10 min)
    n = np.array(r.totals["N_cm3"])
    assert 3e4 < n.max() < 1.5e5 and n.argmax() == 2        # 20 min
    assert n[-1] < 0.5 * n.max()                            # the burst coagulates away
    dn = np.array(r.dndlog10dp_cm3)
    small = dn[:, np.array(r.diameter_nm) < 10.0].sum(axis=1) * math.log10(2.0) / 3.0
    assert small.max() > 0.8 * n.max()                      # the new particles are below 10 nm


@pytest.mark.slow
def test_level2_indoor_high_so2_four_hours():
    r = run(_indoor_so2(30.0, 4.0))
    n = np.array(r.totals["N_cm3"])
    h2so4 = np.array(r.h2so4_cm3)
    assert 5e4 < n.max() < 1e5 and 4e3 < n[-1] < 5e3
    assert h2so4[-1] < 1e-2 * h2so4.max()                  # the limonene is gone
    # some of the new particles grew past the smallest bins (growth by H2SO4 and SOA)
    gain = np.array(r.dndlog10dp_cm3)[-1] - np.array(r.dndlog10dp_cm3)[0]
    d = np.array(r.diameter_nm)
    assert gain[(d > 4.0) & (d < 10.0)].max() > 0.0


_NOON_EDT = datetime.datetime(2026, 7, 1, 16, 0, tzinfo=datetime.timezone.utc)


def _l3(hours, **nucleation):
    # The preset itself starts at 00:00 local (A25); these tests are about the daytime burst
    # and were measured from 12:00 local, so they keep that start.
    s = load_preset("level3_day_night").replace(processes={"nucleation": True}, duration_h=hours,
                                                start_utc=_NOON_EDT, numerics={"output_dt_s": 1800.0})
    return s.replace(nucleation=nucleation) if nucleation else s


def test_level3_defaults_burst_in_daylight():
    """Day & night with nucleation on and the default inputs (start 12:00 local): the
    chemistry's H2SO4 (2 ppb SO2) reaches 5.9e7 cm-3 at 15:30 and the ternary rate
    about 10 cm-3 s-1, so N climbs from 3000 to 2.3e4 cm-3 by 16:30 (measured
    2026-10-20; here the first 3 h: 1.8e4 cm-3, J 9.8). Before the inputs J was 1.4e-8."""
    r = run(_l3(3.0))
    t = np.array(r.time_h)
    n, j = np.array(r.totals["N_cm3"]), np.array(r.diagnostics["nucleation_rate_cm3_s"])
    h2so4 = np.array(r.h2so4_cm3)
    assert np.all(np.diff(n) > 0) and 1.4e4 < n[-1] < 2.2e4
    assert 5.0 < j.max() < 20.0 and j.argmax() == len(j) - 1
    assert 4e7 < h2so4[-1] < 8e7
    assert np.all(np.array(r.photolysis["J_NO2"]) > 1e-3)   # daylight throughout
    dn = np.array(r.dndlog10dp_cm3)
    small = dn[:, np.array(r.diameter_nm) < 10.0].sum(axis=1) * math.log10(2.0) / 3.0
    assert small[-1] > 0.7 * (n[-1] - n[0])                 # the new particles are still small
    bio = np.array(r.diagnostics["nucleating_organic_cm3"])
    assert 1e6 < bio[1:].max() < 2e7
    assert t[-1] == 3.0


def test_level3_rate_order_within_the_run():
    """Same run, inputs switched off one by one (first half hour): binary alone < with
    ions < with ions and NH3 < everything (the organic path adds ~30 %)."""
    runs = {
        "binary": _l3(0.5, ion_pairs_cm3_s=0.0, nh3_ppb=0.0, organic=False),
        "ions": _l3(0.5, nh3_ppb=0.0, organic=False),
        "ions_nh3": _l3(0.5, organic=False),
        "all": _l3(0.5),
    }
    j = {k: run(s).diagnostics["nucleation_rate_cm3_s"][-1] for k, s in runs.items()}
    assert 0.0 < j["binary"] < 1e-7                         # 2e-9
    assert j["binary"] < j["ions"] < 1e-2                   # 7e-5
    assert 1.0 < j["ions_nh3"] < j["all"] < 10.0            # 1.8 and 2.4
    assert 1.1 < j["all"] / j["ions_nh3"] < 2.0


@pytest.mark.slow
def test_level3_defaults_full_day():
    """24 h: burst to 2.3e4 cm-3 at 4.5 h, the new mode grows past 25 nm overnight, N
    falls to 5.0e3 by 20 h and the second morning starts a new burst (9.8e3 at 24 h);
    wall about 2 minutes (measured 2026-10-20)."""
    r = run(_l3(24.0))
    t = np.array(r.time_h)
    n, j = np.array(r.totals["N_cm3"]), np.array(r.diagnostics["nucleation_rate_cm3_s"])
    assert 2.0e4 < n.max() < 2.7e4 and 4.0 <= t[n.argmax()] <= 5.0
    assert 8.0 < j.max() < 14.0 and 3.0 <= t[j.argmax()] <= 4.0
    night = (t > 10.0) & (t < 18.0)
    assert j[night].max() < 1e-3                           # no nucleation at night
    assert 4.5e3 < n[(t > 19.0) & (t < 20.5)].min() < 5.5e3
    assert n[-1] > 1.5 * n[(t > 19.0) & (t < 20.5)].min()  # the second morning's burst
    dn = np.array(r.dndlog10dp_cm3)
    d = np.array(r.diameter_nm)
    gain = dn[-1] - dn[0]
    assert gain[(d > 25.0) & (d < 50.0)].max() > 0.0        # the day's mode grew past 25 nm
