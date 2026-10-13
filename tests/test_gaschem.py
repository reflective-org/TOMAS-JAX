"""Tests for tomas_jax.gaschem: KPP parsing, MCM rate constants, vectorised RHS, solver."""
import numpy as np
import jax.numpy as jnp
import pytest

from tomas_jax.gaschem.config import LIMONENE_EQN
from tomas_jax.gaschem.environment import air_environment
from tomas_jax.gaschem.kpp_parser import parse_eqn
from tomas_jax.gaschem.mcm_rates import MCM_J_PARAMS, complex_rates, mcm_photolysis
from tomas_jax.gaschem.mechanism import EnvRates, Mechanism, RateModel
from tomas_jax.gaschem.solver import integrate_interval

ENV = air_environment(298.15, 101325.0, 0.5)


@pytest.fixture(scope="module")
def parsed():
    return parse_eqn(LIMONENE_EQN)


@pytest.fixture(scope="module")
def mech(parsed):
    return Mechanism.build(parsed)


def test_parse_counts(parsed):
    assert len(parsed.reactions) == 2135
    assert len(parsed.species) == 725
    assert len(parsed.ro2) == 146
    assert sorted(parsed.unused_declared) == ["BR", "H2O", "HCL"]
    assert sum(r.photolysis for r in parsed.reactions) == 342
    assert parsed.molecular_weight("LIMONENE") == pytest.approx(136.238, abs=1e-3)
    assert all("PROD" not in dict(r.products) for r in parsed.reactions)


def test_atom_balance(parsed):
    """N balances in every reaction except one documented mechanism quirk; C is only
    lost (MCM omits CO2) apart from four documented reactions."""
    def n_atoms(el, terms):
        return sum(parsed.atoms[n].get(el, 0) * c for n, c in terms)

    ignore_involving, n_bad, c_gain = [], [], []
    for r in parsed.reactions:
        names = list(r.reactants) + [n for n, _ in r.products]
        if any(not parsed.atoms[n] for n in names):   # NA / SA are IGNORE sinks
            ignore_involving.append(r.label)
            continue
        lhs = [(n, 1) for n in r.reactants]
        if n_atoms("N", lhs) != n_atoms("N", r.products):
            n_bad.append(r.label)
        if n_atoms("C", lhs) < n_atoms("C", r.products):
            c_gain.append(r.label)
    assert ignore_involving == ["33", "34", "35"]
    assert n_bad == ["13917"]                          # CO2N3PAN + OH = CO2N3CO3 (no NO2)
    assert sorted(c_gain) == sorted(["14551", "13369", "7727", "7729"])


# MCM v3.3.1 values at 298.15 K, 1 atm, RH 50 %; agree with the independent F0AM
# MCMv331_K.m transcription to machine precision (KNO3AL differs by 2 % there because
# F0AM uses the older 1.4e-12*exp(-1860/T); we follow the KPP/MCM export).
REFERENCE_K = {
    "KMT01": 2.258300e-12, "KMT03": 1.241023e-12, "KMT04": 4.541237e-02,
    "KMT05": 2.283940e-13, "KMT06": 1.863030e+00, "KMT08": 9.879639e-12,
    "KMT11": 1.540912e-13, "KMT12": 8.937689e-13, "KFPAN": 8.942947e-12,
    "KBPAN": 4.402784e-04, "KRO2NO": 9.031312e-12, "KRO2HO2": 2.277787e-11,
    "KAPHO2": 1.391559e-11, "KNO3AL": 2.793289e-15,
}


def test_mcm_complex_rates_reference():
    K = complex_rates(ENV["TEMP"], ENV["M"], ENV["O2"], ENV["N2"], ENV["H2O"])
    for name, ref in REFERENCE_K.items():
        assert float(K[name]) == pytest.approx(ref, rel=1e-6), name


def test_mcm_photolysis_parameterization():
    j = mcm_photolysis(0.0)
    l, m, n = MCM_J_PARAMS["J_NO2"]
    assert j["J_NO2"] == pytest.approx(l * np.exp(-n))
    night = mcm_photolysis(np.deg2rad(95.0))
    assert all(v == 0.0 for v in night.values())


def test_rate_model_decomposition(parsed):
    rm = RateModel(parsed, external=("34",))
    k_thermal, ro2_dep, j_coef = rm.at_env(ENV)
    assert len(rm.j_names) == 28
    assert int(ro2_dep.sum()) == 280
    assert int((j_coef != 0).any(axis=1).sum()) == 342
    i34 = rm.labels.index("34")
    assert k_thermal[i34] == 0.0 and not j_coef[i34].any()
    # thermal reactions are positive; photolysis reactions have no thermal part except
    # the 7 PAN-type "KBPAN + J(...)" reactions (thermal decomposition + photolysis)
    photo = (j_coef != 0).any(axis=1)
    assert np.all(k_thermal[~photo & (np.arange(len(rm.labels)) != i34)] > 0)
    mixed = [rm.labels[i] for i in np.flatnonzero(photo & (k_thermal != 0))]
    assert len(mixed) == 7
    assert all(parsed.reactions[rm.labels.index(lab)].rate_expr.startswith("KBPAN+J(")
               for lab in mixed)


def test_rate_model_rejects_unknown_names(tmp_path):
    eqn = tmp_path / "bad.eqn"
    eqn.write_text("A = C ;\nB = C ;\n<1> A = B : 1e-12*__import__ ;\n")
    with pytest.raises(ValueError, match="unknown names"):
        RateModel(parse_eqn(eqn))


def test_rhs_matches_naive_loop(parsed, mech):
    rng = np.random.default_rng(1)
    y = rng.uniform(1e3, 1e11, mech.n_species)
    k = rng.uniform(1e-15, 1e-10, len(parsed.reactions))
    index = {s: i for i, s in enumerate(parsed.species)}
    ref = np.zeros(mech.n_species)
    for j, r in enumerate(parsed.reactions):
        rate = k[j] * np.prod([y[index[s]] for s in r.reactants])
        for s in r.reactants:
            ref[index[s]] -= rate
        for s, c in r.products:
            ref[index[s]] += c * rate
    got = np.asarray(mech.rhs(jnp.asarray(y), jnp.asarray(k)))
    np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max())


def _initial_state(parsed, ppb):
    y = np.zeros(len(parsed.species))
    index = {s: i for i, s in enumerate(parsed.species)}
    for s, v in ppb.items():
        y[index[s]] = v * 1e-9 * ENV["M"]
    return jnp.asarray(y), index


def test_dark_ozonolysis_initial_rate(parsed, mech):
    """With only LIMONENE + O3 present, d[LIM]/dt = -k_O3 [O3][LIM] exactly."""
    rm = RateModel(parsed)
    k_thermal, ro2_dep, j_coef = rm.at_env(ENV)
    rates = EnvRates(jnp.asarray(k_thermal), jnp.asarray(ro2_dep), jnp.asarray(j_coef))
    y, index = _initial_state(parsed, {"LIMONENE": 10.0, "O3": 50.0})
    k = rates.k(mech.ro2(y), jnp.zeros(len(rm.j_names)))
    k_o3 = sum(k_thermal[j] for j, r in enumerate(parsed.reactions)
               if sorted(r.reactants) == ["LIMONENE", "O3"])
    dydt = mech.rhs(y, k)
    expected = -k_o3 * y[index["O3"]] * y[index["LIMONENE"]]
    assert float(dydt[index["LIMONENE"]]) == pytest.approx(float(expected), rel=1e-12)
    assert k_o3 == pytest.approx(2.8e-15 * np.exp(-770 / 298.15), rel=0.05)  # MCM/IUPAC k(LIM+O3)


def test_integrate_daytime_hour(parsed, mech):
    rm = RateModel(parsed, external=("34",))
    k_thermal, ro2_dep, j_coef = rm.at_env(ENV)
    rates = EnvRates(jnp.asarray(k_thermal), jnp.asarray(ro2_dep), jnp.asarray(j_coef))
    y, index = _initial_state(parsed, {"LIMONENE": 10, "O3": 40, "NO": 0.5, "NO2": 2,
                                       "SO2": 1, "CO": 120, "CH4": 1900, "H2": 500})
    jd = mcm_photolysis(np.deg2rad(30.0), rm.j_names)
    jv = jnp.asarray([jd[n] for n in rm.j_names])
    for _ in range(6):                                  # 6 x 600 s, KPP-style frozen k
        y, nsteps = integrate_interval(mech, y, rates.k(mech.ro2(y), jv), 600.0)
        assert int(nsteps) > 0
    y = np.asarray(y)
    assert np.all(np.isfinite(y)) and np.all(y >= 0)
    assert 1e5 < y[index["OH"]] < 1e8                   # daytime OH
    assert y[index["LIMONENE"]] < 0.5 * 10e-9 * ENV["M"]
    assert y[index["SA"]] > 0                           # SO2 -> H2SO4 counter


def test_integrate_exhausted_step_budget_raises_clear_error(parsed, mech):
    """Running out of max_steps raises with the reason first, not a JIT stack trace."""
    rm = RateModel(parsed, external=("34",))
    rates = EnvRates(*(jnp.asarray(a) for a in rm.at_env(ENV)))
    y, _ = _initial_state(parsed, {"LIMONENE": 10, "O3": 40, "NO": 0.5, "NO2": 2})
    jd = mcm_photolysis(np.deg2rad(30.0), rm.j_names)
    k = rates.k(mech.ro2(y), jnp.asarray([jd[n] for n in rm.j_names]))
    with pytest.raises(RuntimeError, match=r"^gas-phase chemistry solver failed on a 300 s "
                       r"coupling interval \(max_steps=2\): The maximum number of solver steps"):
        integrate_interval(mech, y, k, 300.0, max_steps=2)


# --- photolysis ------------------------------------------------------------------
from datetime import datetime, timezone  # noqa: E402

from tomas_jax.gaschem.photolysis import (  # noqa: E402
    MCM_ONLY, SOURCES, TUVX_DIRECT, JProvider, PhotolysisSettings, tuvx_root)

BUFFALO_JULY = PhotolysisSettings("tuvx", 42.9, -78.9, datetime(2026, 7, 1, 5, 0, tzinfo=timezone.utc))


@pytest.fixture(scope="module")
def j_names(parsed):
    return RateModel(parsed).j_names


def test_every_mechanism_j_has_a_source(j_names):
    assert set(j_names) == set(SOURCES)
    assert set(MCM_ONLY) == {"J_C3H7CHO_HCO", "J_IPRCHO", "J_C5HPALD1"}


def test_photolysis_off_and_constant(j_names):
    assert not JProvider(j_names, PhotolysisSettings("off")).at(0.0).any()
    pytest.importorskip("tuvx_photolysis")
    jp = JProvider(j_names, PhotolysisSettings("constant", jno2=5e-3))
    j = dict(zip(j_names, jp.at(0.0)))
    assert j["J_NO2"] == pytest.approx(5e-3)
    assert np.all(jp.at(0.0) == jp.at(43200.0))


def test_photolysis_mcm_mode_matches_parameterization(j_names):
    pytest.importorskip("tuvx_photolysis")
    jp = JProvider(j_names, PhotolysisSettings("mcm", 42.9, -78.9, BUFFALO_JULY.start_utc))
    t = 7 * 3600.0
    ref = mcm_photolysis(np.deg2rad(jp.sza_deg(t)), j_names)
    np.testing.assert_allclose(jp.at(t), [ref[n] for n in j_names], rtol=1e-12)


def test_photolysis_tuvx_matches_direct_tuvx(j_names):
    pytest.importorskip("tuvx_photolysis")
    from tuvx_photolysis import PhotolysisCalculator
    from tomas_jax.gaschem.config import TUVX_CONFIG
    root = tuvx_root()
    calc = PhotolysisCalculator.from_tuvx_json(root / TUVX_CONFIG, data_root=root)
    jp = JProvider(j_names, BUFFALO_JULY)
    for local_hour in (7.0, 12.0, 17.5):                # Buffalo local = UTC - 5 h
        assert jp.sza_deg(local_hour * 3600) < 80
        ref = calc.rate_constants(42.9, -78.9, 2026, 7, 1, 5 + local_hour, altitude_km=0.0)
        got = dict(zip(j_names, jp.at(local_hour * 3600)))
        for name, key in TUVX_DIRECT.items():
            assert got[name] == pytest.approx(ref[key], rel=1e-2), name
    assert not jp.at(-4 * 3600.0).any()                 # 01 local: night


# --- volatility ------------------------------------------------------------------
from tomas_jax.gaschem.volatility import (  # noqa: E402
    assign_classes, is_closed_shell, log10_cstar_li2016)


def test_li2016_examples():
    # CH class: (23.80 - 10) * 0.4861 for C10H16
    assert log10_cstar_li2016({"C": 10, "H": 16}) == pytest.approx(13.8 * 0.4861)
    # CHO: (22.66-10)*0.4481 - 3*1.656 - 2*30/13*(-0.7790)
    assert log10_cstar_li2016({"C": 10, "H": 16, "O": 3}) == pytest.approx(
        12.66 * 0.4481 - 3 * 1.656 + 2 * 30 / 13 * 0.7790)
    with pytest.raises(ValueError):
        log10_cstar_li2016({"H": 2, "O": 2})


def test_closed_shell_rule():
    assert is_closed_shell({"C": 10, "H": 16, "O": 3})        # pinonic acid
    assert not is_closed_shell({"C": 10, "H": 17, "O": 3})    # a C10 RO2
    assert is_closed_shell({"C": 10, "H": 17, "N": 1, "O": 5})  # an organic nitrate


def test_assign_classes(parsed):
    cm = assign_classes(parsed)
    assert len(cm.species) == 245 and "LIMONENE" not in cm.species
    assert all(is_closed_shell(parsed.atoms[s]) for s in cm.species)
    assert cm.class_cstar.size == 15 and np.all(np.diff(cm.class_dhvap) < 0)
    # a table entry overrides the estimate and is reflected in the provenance string
    name = cm.species[0]
    cm2 = assign_classes(parsed, {name: 2.0})
    assert cm2.log10_cstar[cm2.species.index(name)] == 2.0
    assert cm2.source.startswith("table (1 of")


def test_photolysis_constant_mode_validates_and_utc_time_base(j_names):
    pytest.importorskip("tuvx_photolysis")
    with pytest.raises(ValueError, match="constant light"):
        JProvider(j_names, PhotolysisSettings("constant", constant_sza_deg=95.0))
    from datetime import timedelta
    ny = timezone(timedelta(hours=-4))
    utc = JProvider(j_names, BUFFALO_JULY)
    local = JProvider(j_names, PhotolysisSettings(
        "tuvx", 42.9, -78.9, BUFFALO_JULY.start_utc.astimezone(ny)))
    assert local.sza_deg(3600.0) == pytest.approx(utc.sza_deg(3600.0))
    np.testing.assert_allclose(local.at(3600.0), utc.at(3600.0))


def test_cstar_table_validation(parsed, tmp_path):
    from tomas_jax.gaschem.volatility import load_cstar_table
    bad = tmp_path / "bad.csv"
    bad.write_text("species,cstar_ugm3\nLIMONENE,0\n")
    with pytest.raises(ValueError, match="positive"):
        load_cstar_table(bad)
    cm = assign_classes(parsed, {"NOT_A_SPECIES": 1.0, "LIMAO2": 1.0})   # unknown + radical
    assert "2 table rows not condensable or not in the mechanism" in cm.source
