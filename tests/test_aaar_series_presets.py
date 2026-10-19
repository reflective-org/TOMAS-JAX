"""The urban air and indoor air presets (series of 2026-10-19; ASSUMPTIONS A19, A20).

Fast tests run 2 h of each; the full presets (6 h urban, 4 h indoor) are slow tests.
The indoor preset holds O3 at 30 ppb (Shantanu: "O3 = 30 ppbv"); the free-O3 variant,
in which O3 relaxes to the balance of outdoor supply and surface loss, is tested too.
"""
import numpy as np
import pytest

from tomas_jax.box.driver import run
from tomas_jax.box.scenario import load_preset
from tomas_jax.gaschem.environment import air_environment


def _pss_ratio(r, scenario):
    """k[O3][NO] / (J(NO2)[NO2]) at the output times: 1 in photostationary state."""
    env = scenario.environment
    m = air_environment(env.temp_k, env.pres_pa, env.rh)["M"]
    k = 1.4e-12 * np.exp(-1310.0 / env.temp_k)              # MCM <7> NO + O3
    o3, no, no2 = (np.array(r.gases_ppb[x]) * 1e-9 * m for x in ("O3", "NO", "NO2"))
    return k * o3 * no / (np.array(r.photolysis["J_NO2"]) * no2)


def _check_urban(r, s):
    assert r.diagnostics["warnings"] == []
    assert r.diagnostics["processes"] == ["coagulation"]
    # gases: photostationary state holds at every daytime output after the first interval
    ratio = _pss_ratio(r, s)
    np.testing.assert_allclose(ratio[1:], 1.0, atol=0.05)
    assert np.all(np.array(r.photolysis["J_NO2"]) > 5e-3)   # sunlit throughout
    assert r.gases_ppb["HNO3"][-1] > 1.0                      # NO2 + OH takes NOx out
    np.testing.assert_allclose(r.losses["emitted_ppb"]["NO"], 1.0 * np.array(r.time_h), rtol=1e-9)
    # particles: coagulation only. N falls; dry mass and volume stay (coagulation adds volumes)
    n, dry, v = (np.array(r.totals[k]) for k in ("N_cm3", "dry_ugm3", "V_um3cm3"))
    assert n[0] == pytest.approx(20003.0, rel=1e-3) and np.all(np.diff(n) < 0)
    np.testing.assert_allclose(dry, dry[0], rtol=1e-6)
    np.testing.assert_allclose(v, v[0], rtol=1e-6)
    assert np.all(np.diff(r.totals["S_um2cm3"]) < 0)          # surface area falls as N does
    assert r.totals["SOA_ugm3"][-1] == 0.0 and r.totals["NIT_ugm3"][-1] == pytest.approx(r.totals["NIT_ugm3"][0])
    assert r.totals["NH4_ugm3"][0] > 0                         # ammonium salts
    assert r.properties["o_to_c"][-1] == pytest.approx(0.5)    # seed organic only
    assert 0.3 < r.properties["kappa"][-1] < 0.4
    assert all(v == 0.0 for k, v in r.losses.items() if k.endswith("_ugm3") or k.endswith("_N_cm3") for v in [v[-1]])


def test_urban_two_hours():
    s = load_preset("level1_urban").replace(duration_h=2.0)
    _check_urban(run(s), s)


@pytest.mark.slow
def test_urban_full_preset():
    s = load_preset("level1_urban")
    r = run(s)
    _check_urban(r, s)
    n = np.array(r.totals["N_cm3"])
    assert 0.5 < n[-1] / n[0] < 0.7                           # about 2e4 -> 1.2e4 cm-3 in 6 h
    # J(NO2) peaks near 13:00 EDT (solar noon at 75 W), 3 h in
    t_peak = r.time_h[int(np.argmax(r.photolysis["J_NO2"]))]
    assert 2.5 <= t_peak <= 3.5
    assert 35.0 < r.gases_ppb["O3"][-1] < 40.0


def _check_indoor(r, s):
    assert r.diagnostics["warnings"] == []
    assert np.all(np.array(r.photolysis["J_NO2"]) == 0.0)      # dark room
    # O3 is held at 30 ppb: every output shows the held value (the hold is the O3 source)
    assert s.hold_constant == ("O3",)
    np.testing.assert_allclose(r.gases_ppb["O3"], 30.0, rtol=1e-9)
    # limonene budget: initial + ventilated in - ventilated out - remaining = reacted
    lim = np.array(r.gases_ppb["LIMONENE"])
    vent = np.array(r.losses["ventilated_gases_ppb"]["LIMONENE"])
    reacted = lim[0] - lim[-1] - vent[-1]
    assert lim[-1] < lim[0] and vent[-1] > 0 and reacted > 5.0   # 13 ppb react in 4 h
    # ozonolysis makes OH and products; SOA forms but the walls and the window take most vapour
    assert max(r.radicals_cm3["OH"]) > 1e5
    assert r.totals["SOA_ugm3"][-1] > 0.0 and len(r.top_contributors) > 0
    assert r.top_contributors[0]["species"] == "LIMALOOH"
    L = r.losses
    assert L["wall_vapour_ugm3"][-1] > 0 and L["ventilated_vapour_ugm3"][-1] > 0
    assert L["wall_vapour_ugm3"][-1] > 5.0 * max(r.totals["SOA_ugm3"])
    assert L["o3_surface_ppb"][-1] > 10.0                        # the surfaces take O3, the hold replaces it
    assert L["deposited_N_cm3"][-1] > 0 and L["deposited_dry_ugm3"][-1] > 0
    assert L["ventilated_N_cm3"][-1] < 0                        # outdoor air brings particles in
    for key in ("wall_vapour_ugm3", "o3_surface_ppb", "deposited_dry_ugm3", "deposited_N_cm3"):
        assert np.all(np.diff(L[key]) >= 0)                       # cumulative
    # properties: the SOA (O:C about 0.4-0.5) hardly moves the O:C from the seed value 0.5
    p = r.properties
    assert 0.4 <= p["o_to_c"][-1] < 0.8
    assert 0.1 < p["kappa"][-1] < 0.65


def _check_indoor_free_o3(r, s):
    """O3 not held: it relaxes to the balance of outdoor supply and surface loss."""
    loss = s.losses
    lam, ks = loss.ventilation_per_h, loss.o3_surface_per_h
    o3 = np.array(r.gases_ppb["O3"])
    o3_balance = lam * loss.outdoor_gases_ppb["O3"] / (lam + ks)
    assert o3[0] == pytest.approx(30.0) and np.all(np.diff(o3[:3]) < 0)
    assert o3[-1] == pytest.approx(o3_balance, rel=0.15)


def test_indoor_two_hours():
    s = load_preset("level2_indoor").replace(duration_h=2.0)
    _check_indoor(run(s), s)


def test_indoor_two_hours_free_o3():
    s = load_preset("level2_indoor").replace(duration_h=2.0, hold_constant=())
    _check_indoor_free_o3(run(s), s)


@pytest.mark.slow
def test_indoor_full_preset_and_wall_loss_switch():
    s = load_preset("level2_indoor")
    r = run(s)
    _check_indoor(r, s)
    assert 0.5 < max(r.totals["SOA_ugm3"]) < 2.0                # about 1 ug/m3 with the walls on
    # without wall losses and ventilation, the same chemistry gives several ug/m3 of SOA
    r_free = run(s.replace(losses={"ventilation_per_h": 0.0, "o3_surface_per_h": 0.0,
                                   "organic_vapour_wall_per_s": 0.0, "particle_deposition": "off"}))
    assert r_free.totals["SOA_ugm3"][-1] > 5.0 and r_free.losses["wall_vapour_ugm3"][-1] == 0.0
    # opening the window ventilates the limonene before it reacts: less SOA, more out
    r_open = run(s.replace(losses={"ventilation_per_h": 3.0}))
    assert max(r_open.totals["SOA_ugm3"]) < 0.5 * max(r.totals["SOA_ugm3"])
    assert (r_open.losses["ventilated_gases_ppb"]["LIMONENE"][-1]
            > r.losses["ventilated_gases_ppb"]["LIMONENE"][-1])
    # O3 free: it falls to lambda O3_out / (lambda + k_s) = 5.7 ppb and SOA stays below 0.3 (A20)
    r_o3 = run(s.replace(hold_constant=()))
    _check_indoor_free_o3(r_o3, s)
    assert max(r_o3.totals["SOA_ugm3"]) < 0.3
