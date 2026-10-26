"""Driver tests for the series additions (2026-10-19): emissions, cloud fraction, ventilation,
O3 surface loss, organic vapour wall loss, particle deposition, seed salts and the new
outputs (surface and volume distributions, O:C, kappa, loss budgets).

Where the physics allows it the checks are exact: with every other process off, a
first-order loss applied once per coupling interval is the analytic solution at the
interval ends, and each budget closes to rounding.
"""
import numpy as np
import pytest

from tomas_jax.box.coupling import kg_cell_to_ugm3, mass_grid, seed_distribution
from tomas_jax.box.driver import run
from tomas_jax.box.scenario import load_preset
from tomas_jax.core.config import SRTNH4, SRTNIT, SRTORG_NUC, SRTSO4

OFF = {"coagulation": False, "h2so4_condensation": False, "soa_partitioning": False}
DARK_ONLY_O3 = load_preset("level2_limonene_soa").replace(
    duration_h=1.0, gases_ppb={"O3": 30.0}, photolysis={"mode": "off"}, processes=OFF,
    numerics={"output_dt_s": 300.0})


def _particles_only(**losses):
    return load_preset("level1_volatility").replace(
        duration_h=2.0, processes=OFF, numerics={"output_dt_s": 600.0},
        surrogate={"production_ugm3_per_h": 0.0, "initial_gas_ugm3": 0.0}, losses=losses)


def test_particle_ventilation_is_the_exact_exponential_toward_outdoor_air():
    outdoor = {"modes": [{"number_cm3": 500.0, "gmd_nm": 150.0, "gsd": 1.5}],
               "organic_mass_fraction": 0.5}
    r = run(_particles_only(ventilation_per_h=1.0, outdoor_seed=outdoor))
    n = np.array(r.dndlog10dp_cm3)
    t = np.array(r.time_h)
    n_out = np.array(run(_particles_only().replace(seed=outdoor, duration_h=1.0)).dndlog10dp_cm3)[0]
    expect = n_out + (n[0] - n_out) * np.exp(-t)[:, None]
    big = n[0] > 1.0
    np.testing.assert_allclose(n[:, big], expect[:, big], rtol=1e-6)
    total = np.array(r.totals["N_cm3"])
    np.testing.assert_allclose(total[0] - total, r.losses["ventilated_N_cm3"], rtol=1e-9)
    dry = np.array(r.totals["dry_ugm3"])
    np.testing.assert_allclose(dry[0] - dry, r.losses["ventilated_dry_ugm3"], rtol=1e-9)
    assert r.losses["deposited_N_cm3"][-1] == 0.0


def test_particle_deposition_budget_closes():
    r = run(_particles_only(particle_deposition="lai_nazaroff"))
    n, dry = np.array(r.totals["N_cm3"]), np.array(r.totals["dry_ugm3"])
    assert n[-1] < n[0] and dry[-1] < dry[0]
    np.testing.assert_allclose(n + r.losses["deposited_N_cm3"], n[0], rtol=1e-9)
    np.testing.assert_allclose(dry + r.losses["deposited_dry_ugm3"], dry[0], rtol=1e-9)
    assert r.losses["ventilated_N_cm3"][-1] == 0.0
    # about a percent per hour for an 80 nm mode in a typical room
    assert 0.005 < (n[0] - n[-1]) / n[0] / 2.0 < 0.05
    # per size bin: the deposited amounts add up to the totals, and beta(D) dips between
    # the diffusion (small) and settling (large) ends
    dlog10 = np.log10(np.array(r.diameter_edges_nm[1:]) / np.array(r.diameter_edges_nm[:-1]))
    dep_n = np.array(r.losses["deposited_dndlog10dp_cm3"]) * dlog10
    dep_m = np.array(r.losses["deposited_dmdlog10dp_ugm3"]) * dlog10
    np.testing.assert_allclose(dep_n.sum(axis=1), r.losses["deposited_N_cm3"], rtol=1e-9, atol=1e-12)
    np.testing.assert_allclose(dep_m.sum(axis=1), r.losses["deposited_dry_ugm3"], rtol=1e-9, atol=1e-12)
    assert (np.diff(dep_n, axis=0) >= 0).all()
    beta = np.array(r.losses["deposition_rate_per_h"])
    d = np.array(r.diameter_nm)
    i_min = int(np.argmin(beta))
    assert 50 < d[i_min] < 2000 and beta[0] > 10 * beta[i_min] and beta[-1] > 10 * beta[i_min]
    assert r.losses["wall_vapour_by_species"] == [] and r.losses["wall_vapour_other_ugm3"][-1] == 0.0


def test_runs_without_losses_report_no_resolved_losses():
    r = run(_particles_only())
    assert "deposited_dndlog10dp_cm3" not in r.losses and "wall_vapour_by_species" not in r.losses


def test_gas_ventilation_and_o3_surface_loss_are_exact_in_the_dark():
    r = run(DARK_ONLY_O3.replace(losses={"ventilation_per_h": 1.0, "outdoor_gases_ppb": {"O3": 40.0},
                                         "o3_surface_per_h": 3.0}))
    t = np.array(r.time_h)
    o3 = np.array(r.gases_ppb["O3"])
    o3_star = 1.0 * 40.0 / 4.0
    np.testing.assert_allclose(o3, o3_star + (30.0 - o3_star) * np.exp(-4.0 * t), rtol=1e-6)
    # change = ventilated in - ventilated out - lost to surfaces
    np.testing.assert_allclose(o3 - 30.0, -np.array(r.losses["ventilated_gases_ppb"]["O3"])
                               - np.array(r.losses["o3_surface_ppb"]), rtol=1e-6, atol=1e-9)
    assert r.losses["o3_surface_ppb"][-1] > 0 and r.losses["ventilated_gases_ppb"]["O3"][-1] < 0


def test_no_emission_accumulates_linearly_in_the_dark():
    r = run(DARK_ONLY_O3.replace(gases_ppb={}, emissions_ppb_per_h={"NO": 2.0}))
    t = np.array(r.time_h)
    # within the gas solver's tolerance (NO + NO + O2 is the only dark sink, 1e-5 per hour)
    np.testing.assert_allclose(r.gases_ppb["NO"], 2.0 * t, rtol=1e-4, atol=1e-9)
    np.testing.assert_allclose(r.losses["emitted_ppb"]["NO"], 2.0 * t, rtol=1e-12, atol=1e-12)
    with pytest.raises(ValueError, match="need chemistry"):
        run(load_preset("level1_volatility").replace(emissions_ppb_per_h={"NO": 1.0}))
    with pytest.raises(ValueError, match="unknown mechanism species"):
        run(DARK_ONLY_O3.replace(emissions_ppb_per_h={"NOX": 1.0}))


def test_vapour_wall_loss_of_the_surrogate_closes():
    s = _particles_only(organic_vapour_wall_per_s=1e-4).replace(
        surrogate={"cstar_ugm3": 1e6, "initial_gas_ugm3": 10.0, "production_ugm3_per_h": 0.0})
    r = run(s)
    t_s = np.array(r.time_h) * 3600.0
    gas = np.array(r.volatility["gas_ugm3"])[:, 0]
    np.testing.assert_allclose(gas, 10.0 * np.exp(-1e-4 * t_s), rtol=1e-9)
    np.testing.assert_allclose(gas + r.losses["wall_vapour_ugm3"], 10.0, rtol=1e-9)
    by = r.losses["wall_vapour_by_species"]
    assert [e["species"] for e in by] == ["c* class 1"]
    np.testing.assert_allclose(by[0]["ugm3"], r.losses["wall_vapour_ugm3"], rtol=1e-12)


def test_vapour_wall_loss_with_chemistry_removes_condensables_only():
    s = load_preset("level2_limonene_soa").replace(duration_h=0.5, numerics={"output_dt_s": 300.0})
    r0, r1 = run(s), run(s.replace(losses={"organic_vapour_wall_per_s": 1e-3}))
    assert r1.losses["wall_vapour_ugm3"][-1] > 0.1
    # per species: the largest few plus the rest add up to the total at every time
    by = r1.losses["wall_vapour_by_species"]
    assert 0 < len(by) <= 15 and all(e["formula"] and e["species"] for e in by)
    final = [e["ugm3"][-1] for e in by]
    assert final == sorted(final, reverse=True)
    total = np.sum([e["ugm3"] for e in by], axis=0) + np.array(r1.losses["wall_vapour_other_ugm3"])
    np.testing.assert_allclose(total, r1.losses["wall_vapour_ugm3"], rtol=1e-9, atol=1e-12)
    assert r1.totals["SOA_ugm3"][-1] < 0.8 * r0.totals["SOA_ugm3"][-1]
    # limonene itself is not condensable: it is not wall-lost
    assert r1.gases_ppb["LIMONENE"][-1] == pytest.approx(r0.gases_ppb["LIMONENE"][-1], rel=0.02)


def test_cloud_fraction_scales_every_j():
    s = load_preset("level2_limonene_soa").replace(duration_h=0.5, numerics={"output_dt_s": 300.0})
    r = run(s.replace(photolysis={"cloud_fraction": 1.0}))
    for name in ("J_NO2", "J_O3_O1D"):
        ref = np.array(run(s).photolysis[name])
        np.testing.assert_allclose(r.photolysis[name], 0.25 * ref, rtol=1e-12)
    assert r.photolysis["J_NO2"][-1] == pytest.approx(0.25 * 5e-3, rel=1e-12)


def test_seed_salts_stoichiometry():
    xk = mass_grid(40)
    modes = load_preset("level1_volatility").seed.modes
    nk, mk = seed_distribution(modes, 0.5, xk, 1e10, ammonium_nitrate_mass_fraction=0.2, ammonium=True)
    tot = mk.sum(axis=0)
    dry = tot.sum()
    assert tot[SRTORG_NUC] / dry == pytest.approx(0.5)
    assert tot[SRTNIT] / dry == pytest.approx(0.2 * 62.0 / 80.0)
    assert tot[SRTSO4] / dry == pytest.approx(0.3 * 96.0 / 132.0)
    assert tot[SRTNH4] / dry == pytest.approx(0.3 * 36.0 / 132.0 + 0.2 * 18.0 / 80.0)
    # the old call is unchanged: sulfate and organic only
    _, mk_old = seed_distribution(modes, 0.3, xk, 1e10)
    assert mk_old[:, SRTNH4].sum() == 0.0 and mk_old[:, SRTNIT].sum() == 0.0
    assert mk_old[:, SRTSO4].sum() / mk_old.sum() == pytest.approx(0.7)
    # without ammonium the nitrate is NO3 alone
    _, mk_no = seed_distribution(modes, 0.5, xk, 1e10, ammonium_nitrate_mass_fraction=0.2)
    assert mk_no.sum(axis=0)[SRTNIT] / mk_no.sum() == pytest.approx(0.2)
    assert mk_no[:, SRTNH4].sum() == 0.0
    r = run(load_preset("level1_volatility").replace(
        duration_h=0.5, seed={"organic_mass_fraction": 0.5, "ammonium_nitrate_mass_fraction": 0.2,
                              "ammonium": True}))
    assert r.totals["NIT_ugm3"][0] > 0 and r.totals["NH4_ugm3"][0] > 0
    assert kg_cell_to_ugm3(tot[SRTNIT], 1e10) == pytest.approx(r.totals["NIT_ugm3"][0], rel=1e-9)


def test_surface_volume_distributions_and_kappa():
    r = run(load_preset("level1_volatility").replace(duration_h=1.0))
    dn, ds, dv = (np.array(getattr(r, k)) for k in ("dndlog10dp_cm3", "dsdlog10dp_um2cm3",
                                                     "dvdlog10dp_um3cm3"))
    dm = np.array(r.dmdlog10dp_ugm3)
    # V is the dry mass over 1.4 g/cm3 (exact at every time); S uses each bin's mean-mass
    # diameter, which is the bin centre at the start (seed particles sit at the centre)
    np.testing.assert_allclose(dv, dm / 1.4, rtol=1e-12)
    full = dn > 1e-12
    dp_mean_um = (6.0 * dv[full] / (np.pi * dn[full])) ** (1.0 / 3.0)
    np.testing.assert_allclose(ds[full], np.pi * dp_mean_um**2 * dn[full], rtol=1e-12)
    dp_um = np.array(r.diameter_nm) * 1e-3
    np.testing.assert_allclose(ds[0], np.pi * dp_um**2 * dn[0], rtol=1e-9)
    dry, vol = np.array(r.totals["dry_ugm3"]), np.array(r.totals["V_um3cm3"])
    np.testing.assert_allclose(dry, 1.4 * vol, rtol=1e-12)
    p = r.properties
    assert p["o_to_c"][0] == pytest.approx(0.5)                   # seed organic only
    assert p["kappa_inorganic"][0] == pytest.approx(0.61)         # sulfate as ammonium sulfate
    assert p["kappa_organic"][0] == pytest.approx(0.18 * 0.5 + 0.03)
    assert 0.12 < p["kappa"][-1] < 0.61 and p["kappa"][-1] < p["kappa"][0]   # SOA dilutes the salt
    # pure sulfate seed: kappa 0.61 and no organic O:C
    r2 = run(load_preset("level1_volatility").replace(
        duration_h=0.5, seed={"organic_mass_fraction": 0.0}, surrogate={"production_ugm3_per_h": 0.0}))
    assert r2.properties["kappa"][0] == pytest.approx(0.61) and np.isnan(r2.properties["o_to_c"][0])
    assert r2.to_dict()["properties"]["o_to_c"][0] is None           # NaN -> null in JSON
