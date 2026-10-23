"""The MCM toluene subset merged into the limonene mechanism (2026-10-23, ASSUMPTIONS A27).

Mechanism level: the website export parses, the merge keeps the 326 shared reactions
once with no conflicting duplicate, the RO2 sum is the union, every photolysis name has
a source, atom balance holds with the documented MCM conventions, and the rate
expressions decompose like the limonene ones. Driver level: a scenario with every
toluene species at zero runs the limonene mechanism alone (bit-identical, see
tests/test_aaar_presets_unchanged.py), a non-zero toluene selects the merged one,
TOLUENE is reported and consumed by OH in daylight.
"""
import hashlib

import numpy as np
import pytest

from tomas_jax.box.coupling import (toluene_merge, toluene_only_species, tutorial_mechanism,
                                    uses_toluene)
from tomas_jax.box.driver import _chemistry, _class_map, run, warmup  # noqa: F401
from tomas_jax.box.scenario import load_preset
from tomas_jax.gaschem.config import LIMONENE_EQN, TOLUENE_EQN
from tomas_jax.gaschem.environment import air_environment
from tomas_jax.gaschem.kpp_parser import parse_eqn
from tomas_jax.gaschem.mechanism import RateModel
from tomas_jax.gaschem.merge import merge_mechanisms
from tomas_jax.gaschem.photolysis import SOURCES

TOLUENE_SHA256 = "186f1032fa482282381d2a4739c48ee296d4cf2801972f0f1caa74efc69f5612"


@pytest.fixture(scope="module")
def toluene():
    return parse_eqn(TOLUENE_EQN)


@pytest.fixture(scope="module")
def merged():
    return toluene_merge()


def test_toluene_export_is_the_file_received(toluene):
    assert hashlib.sha256(TOLUENE_EQN.read_bytes()).hexdigest() == TOLUENE_SHA256
    assert len(toluene.species) == 291
    assert len(toluene.reactions) == 831
    assert len(toluene.ro2) == 56
    assert toluene.unused_declared == ["H2O"]
    assert toluene.molecular_weight("TOLUENE") == pytest.approx(92.141, abs=1e-3)
    assert "TOLUENE" in toluene.species and "CRESOL" in toluene.species


def test_merge_counts_and_no_conflicts(merged):
    mech, rep = merged
    assert (rep.base_species, rep.extra_species, rep.merged_species) == (725, 291, 906)
    assert (rep.base_reactions, rep.extra_reactions, rep.merged_reactions) == (2135, 831, 2640)
    assert (rep.base_ro2, rep.extra_ro2, rep.merged_ro2) == (146, 56, 183)
    assert len(rep.shared_species) == 110 and len(rep.new_species) == 181
    assert len(rep.shared_reactions) == 326 and len(rep.new_reactions) == 505
    assert rep.new_j == ("J_CH3CHO", "J_MVK_CO")
    # shared reactions are kept once under their limonene label; added ones are prefixed
    labels = [r.label for r in mech.reactions]
    assert len(set(labels)) == len(labels)
    assert all(lab.startswith("T") for lab in rep.new_reactions)
    assert mech.species[:725] == parse_eqn(LIMONENE_EQN).species
    assert mech.ro2[:146] == parse_eqn(LIMONENE_EQN).ro2
    # every toluene reaction is in the merged mechanism exactly once
    assert len(rep.shared_reactions) + len(rep.new_reactions) == rep.extra_reactions


def test_merge_rejects_conflicting_duplicates(tmp_path):
    base = tmp_path / "a.eqn"
    base.write_text("A = C ;\nB = C ;\n<1> A = B : 1e-12 ;\n")
    extra = tmp_path / "b.eqn"
    extra.write_text("A = C ;\nB = C ;\n<1> A = B : 2e-12 ;\n")
    with pytest.raises(ValueError, match="different rate expressions"):
        merge_mechanisms(parse_eqn(base), parse_eqn(extra))
    extra.write_text("A = 2C ;\nB = C ;\n<1> A = B : 1e-12 ;\n")
    with pytest.raises(ValueError, match="different atoms"):
        merge_mechanisms(parse_eqn(base), parse_eqn(extra))
    extra.write_text("A = C ;\nB = C ;\nD = C ;\n<1> A = B : 1e-12 ;\n<2> A = D : 1e-12 ;\n")
    m, rep = merge_mechanisms(parse_eqn(base), parse_eqn(extra))
    assert [r.label for r in m.reactions] == ["1", "T2"] and rep.new_species == ("D",)


def test_merged_j_names_all_have_a_source(merged):
    rm = RateModel(merged[0], external=("34",))
    assert len(rm.j_names) == 30
    assert set(rm.j_names) == set(SOURCES)
    assert SOURCES["J_CH3CHO"] == "TUV-x"
    assert SOURCES["J_MVK_CO"] == "TUV-x total, MCM branching"


def test_merged_atom_balance(merged):
    """Same conventions as the limonene file (tests/test_gaschem.py): N balances except in
    <13917>; C is only lost (MCM omits CO2) except in four reactions. The toluene export's
    two C-gaining reactions (<420>, <422>: C4MDIAL photolysis writes CO on top of a C5 acyl
    peroxy radical) are the limonene file's <7727>, <7729> and are kept once, so the
    merged mechanism adds no new exception."""
    mech = merged[0]

    def n_atoms(el, terms):
        return sum(mech.atoms[n].get(el, 0) * c for n, c in terms)

    ignore_involving, n_bad, c_gain = [], [], []
    for r in mech.reactions:
        names = list(r.reactants) + [n for n, _ in r.products]
        if any(not mech.atoms[n] for n in names):
            ignore_involving.append(r.label)
            continue
        lhs = [(n, 1) for n in r.reactants]
        if n_atoms("N", lhs) != n_atoms("N", r.products):
            n_bad.append(r.label)
        if n_atoms("C", lhs) < n_atoms("C", r.products):
            c_gain.append(r.label)
    assert ignore_involving == ["33", "34", "35"]
    assert n_bad == ["13917"]
    assert sorted(c_gain) == sorted(["14551", "13369", "7727", "7729"])


def test_merged_rate_decomposition(merged):
    env = air_environment(298.15, 101325.0, 0.5)
    rm = RateModel(merged[0], external=("34",))
    k_thermal, ro2_dep, j_coef = rm.at_env(env)
    assert len(rm.labels) == 2640
    assert int(ro2_dep.sum()) == 280 + 65           # limonene + toluene RO2 + RO2 reactions
    photo = (j_coef != 0).any(axis=1)
    assert int(photo.sum()) == 342 + 74             # photolysis reactions
    assert int((photo & (k_thermal != 0)).sum()) == 7   # the KBPAN + J(...) ones, all limonene
    i34 = rm.labels.index("34")
    assert np.all(k_thermal[~photo & (np.arange(len(rm.labels)) != i34)] > 0)


def test_tutorial_mechanism_variants():
    lim, tol = tutorial_mechanism(False), tutorial_mechanism(True)
    assert len(lim.species) == 726 and len(tol.species) == 907     # + NA_HET
    assert lim.species[-1] == "NA_HET" and "NA_HET" in tol.species
    assert "TOLUENE" not in lim.species and "TOLUENE" in tol.species
    assert len(toluene_only_species()) == 181 and "TOLUENE" in toluene_only_species()


def test_uses_toluene_rules():
    assert not uses_toluene({"TOLUENE": 0.0, "O3": 30.0}, {})
    assert uses_toluene({"TOLUENE": 1.0}, {})
    assert uses_toluene({}, {"TOLUENE": 0.1})
    assert uses_toluene({"CRESOL": 0.5}, {})
    assert uses_toluene({}, {}, hold_constant=("TOLUENE",))
    assert uses_toluene({}, {}, outdoor_gases_ppb={"BENZAL": 0.2})
    assert not uses_toluene({"LIMONENE": 5.0}, {"NO": 1.0}, hold_constant=("NO2",))


def test_class_map_with_toluene_falls_back_to_li2016_for_its_products():
    lim = _class_map("mcm_limonene_cstar_298K.csv", False)
    tol = _class_map("mcm_limonene_cstar_298K.csv", True)
    assert len(lim.species) == 228 and len(tol.species) == 289
    assert set(tol.fallback) >= set(toluene_only_species()) & set(tol.species)
    assert lim.fallback == ("LIMONENE", "D9", "CO", "CH4") or set(lim.fallback) == {"LIMONENE", "D9", "CO", "CH4"}


def _urban(hours=1.0, **gases):
    s = load_preset("level1_urban")
    return s.replace(duration_h=hours, gases_ppb={**s.gases_ppb, **gases})


def test_urban_preset_has_toluene_off_and_runs_the_limonene_mechanism():
    s = load_preset("level1_urban")
    assert s.gases_ppb["TOLUENE"] == 0.0
    assert not uses_toluene(s.gases_ppb, s.emissions_ppb_per_h, s.hold_constant)
    r = run(_urban())
    assert "TOLUENE" not in r.gases_ppb
    assert r.diagnostics["warnings"] == []


def test_toluene_on_is_reported_and_consumed_in_daylight():
    # 11:00-13:00 local: OH is up, toluene + OH (k ~ 5.6e-12) takes about 10 % in 2 h
    s = _urban(2.0, TOLUENE=2.0).replace(start_utc="2026-07-01T15:00:00+00:00")
    r = run(s)
    tol = np.asarray(r.gases_ppb["TOLUENE"])
    assert tol[0] == pytest.approx(2.0, rel=1e-6)
    assert 0.8 < tol[-1] / tol[0] < 0.98
    assert r.diagnostics["warnings"] == []          # no c* warning: SOA and nucleation off
    assert r.diagnostics["j_sources"]["J_CH3CHO"] == "TUV-x"
    # the same run with toluene at zero does not report it and has fewer species to solve
    r0 = run(s.replace(gases_ppb={**s.gases_ppb, "TOLUENE": 0.0}))
    assert "TOLUENE" not in r0.gases_ppb


def test_toluene_products_are_valid_names_everywhere():
    s = load_preset("level2_limonene_soa").replace(duration_h=0.5, gases_ppb={"CRESOL": 0.0})
    run(s)                                          # zero: limonene mechanism, name accepted
    with pytest.raises(ValueError, match="unknown mechanism species"):
        run(s.replace(gases_ppb={"NAPHTHALENE": 1.0}))
