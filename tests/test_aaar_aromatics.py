"""The MCM aromatic subsets (benzene, toluene, m-xylene) merged into the limonene mechanism
per run (2026-10-23, ASSUMPTIONS A27, the Urban VOC selector).

Mechanism level: each website export parses with its sha256, the merge keeps the shared
reactions once with no conflicting duplicate, the RO2 sum is the union, every photolysis
name has a source, atom balance holds with the documented MCM conventions, and the rate
expressions decompose like the limonene ones. Driver level: a scenario with every aromatic
species at zero runs the limonene mechanism alone (bit-identical, see
tests/test_aaar_presets_unchanged.py), a non-zero VOC selects only its own subset, the VOC is
reported and consumed by OH in daylight in the order of its k_OH.
"""
import hashlib

import numpy as np
import pytest

from tomas_jax.box.coupling import (AROMATICS, all_aromatic_species, aromatic_merge, aromatic_species,
                                    aromatics_used, merged_mechanism, tutorial_mechanism, uses_toluene)
from tomas_jax.box.driver import _chemistry, _class_map, run
from tomas_jax.box.scenario import URBAN_VOC_PPB_ON, URBAN_VOC_SPECIES, load_preset
from tomas_jax.gaschem.config import AROMATIC_EQN, LIMONENE_EQN
from tomas_jax.gaschem.environment import air_environment
from tomas_jax.gaschem.kpp_parser import parse_eqn
from tomas_jax.gaschem.mechanism import RateModel
from tomas_jax.gaschem.merge import merge_mechanisms
from tomas_jax.gaschem.photolysis import SOURCES

# the website exports as received (provenance files beside them)
SHA256 = {
    "BENZENE": "24bdc02f065838f5ccb365e70e09bc389b4eaee7b074c8606274465aa3776146",
    "TOLUENE": "186f1032fa482282381d2a4739c48ee296d4cf2801972f0f1caa74efc69f5612",
    "MXYL": "a4b33df178b265bc9b7f124744dbc1b1c31028073550c7c7c5d95018b43c35ec",
}
# (species used, reactions, RO2) of the export; (shared species, new species, shared reactions,
# new reactions) of the merge with the limonene subset; new J names
EXPORT = {"BENZENE": (157, 436, 28), "TOLUENE": (291, 831, 56), "MXYL": (301, 860, 60)}
MERGE = {"BENZENE": (36, 121, 105, 331), "TOLUENE": (110, 181, 326, 505), "MXYL": (110, 191, 326, 534)}
NEW_J = {"BENZENE": (), "TOLUENE": ("J_CH3CHO", "J_MVK_CO"), "MXYL": ("J_CH3CHO", "J_MVK_CO")}
MW = {"BENZENE": 78.114, "TOLUENE": 92.141, "MXYL": 106.168}


@pytest.mark.parametrize("name", AROMATICS)
def test_export_is_the_file_received(name):
    path = AROMATIC_EQN[name]
    assert hashlib.sha256(path.read_bytes()).hexdigest() == SHA256[name]
    p = parse_eqn(path)
    assert (len(p.species), len(p.reactions), len(p.ro2)) == EXPORT[name]
    assert p.unused_declared == ["H2O"]
    assert p.molecular_weight(name) == pytest.approx(MW[name], abs=1e-3)
    assert name in p.species


@pytest.mark.parametrize("name", AROMATICS)
def test_merge_counts_and_no_conflicts(name):
    mech, rep = aromatic_merge(name)
    shared_sp, new_sp, shared_rx, new_rx = MERGE[name]
    assert rep.base_species == 725 and rep.base_reactions == 2135 and rep.base_ro2 == 146
    assert (len(rep.shared_species), len(rep.new_species)) == (shared_sp, new_sp)
    assert (len(rep.shared_reactions), len(rep.new_reactions)) == (shared_rx, new_rx)
    assert rep.merged_species == 725 + new_sp and rep.merged_reactions == 2135 + new_rx
    assert len(rep.shared_reactions) + len(rep.new_reactions) == rep.extra_reactions
    assert rep.new_j == NEW_J[name]
    labels = [r.label for r in mech.reactions]
    assert len(set(labels)) == len(labels)
    prefix = {"BENZENE": "B", "TOLUENE": "T", "MXYL": "X"}[name]
    assert all(lab.startswith(prefix) for lab in rep.new_reactions)
    assert mech.species[:725] == parse_eqn(LIMONENE_EQN).species
    assert mech.ro2[:146] == parse_eqn(LIMONENE_EQN).ro2


def test_subsets_share_products_and_chain_merge_counts():
    b, t, x = (aromatic_species(n) for n in AROMATICS)
    assert len(b & t) == 61 and len(t & x) == 60 and len(b & x) == 10 and len(b & t & x) == 10
    assert len(all_aromatic_species()) == len(b | t | x)
    m = merged_mechanism(("BENZENE", "TOLUENE", "MXYL"))
    assert (len(m.species), len(m.reactions), len(m.ro2)) == (1097, 3173, 225)
    assert merged_mechanism(()).species == parse_eqn(LIMONENE_EQN).species
    # a subset's own merge equals the chain of one
    assert merged_mechanism(("MXYL",)).species == aromatic_merge("MXYL")[0].species
    with pytest.raises(ValueError, match="unknown aromatic"):
        merged_mechanism(("NAPHTHALENE",))


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


def test_every_j_of_every_merge_has_a_source():
    for aromatics in (("BENZENE",), ("TOLUENE",), ("MXYL",), ("BENZENE", "TOLUENE", "MXYL")):
        rm = RateModel(merged_mechanism(aromatics), external=("34",))
        assert set(rm.j_names) <= set(SOURCES)
    assert set(RateModel(merged_mechanism(("MXYL",)), external=("34",)).j_names) == set(SOURCES)
    assert SOURCES["J_CH3CHO"] == "TUV-x" and SOURCES["J_MVK_CO"] == "TUV-x total, MCM branching"


def test_merged_atom_balance():
    """Same conventions as the limonene file (tests/test_gaschem.py): N balances except in
    <13917>; C is only lost (MCM omits CO2) except in four reactions. The toluene and m-xylene
    exports' two C-gaining reactions (C4MDIAL photolysis) are the limonene file's <7727>,
    <7729> and are kept once; benzene adds none."""
    mech = merged_mechanism(("BENZENE", "TOLUENE", "MXYL"))

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


@pytest.mark.parametrize("aromatics,n_rxn,n_ro2_dep,n_photo", [
    (("TOLUENE",), 2640, 345, 416),
    (("BENZENE",), 2466, 326, 393),
    (("MXYL",), 2669, 347, 423),
])
def test_merged_rate_decomposition(aromatics, n_rxn, n_ro2_dep, n_photo):
    env = air_environment(298.15, 101325.0, 0.5)
    rm = RateModel(merged_mechanism(aromatics), external=("34",))
    k_thermal, ro2_dep, j_coef = rm.at_env(env)
    assert len(rm.labels) == n_rxn
    assert int(ro2_dep.sum()) == n_ro2_dep
    photo = (j_coef != 0).any(axis=1)
    assert int(photo.sum()) == n_photo
    assert int((photo & (k_thermal != 0)).sum()) == 7     # the KBPAN + J(...) ones, all limonene
    i34 = rm.labels.index("34")
    assert np.all(k_thermal[~photo & (np.arange(len(rm.labels)) != i34)] > 0)


def test_tutorial_mechanism_variants():
    lim = tutorial_mechanism(())
    assert len(lim.species) == 726 and lim.species[-1] == "NA_HET"
    assert len(tutorial_mechanism(("TOLUENE",)).species) == 907
    assert len(tutorial_mechanism(True).species) == 907          # the toluene-only form
    assert len(tutorial_mechanism(("BENZENE",)).species) == 847
    assert len(tutorial_mechanism(("MXYL",)).species) == 917
    assert len(tutorial_mechanism(("MXYL", "BENZENE")).species) == 725 + 121 + 191 - 10 + 1
    assert "TOLUENE" not in lim.species and "NA_HET" in tutorial_mechanism(("MXYL",)).species


def test_aromatics_used_rules():
    assert aromatics_used({"TOLUENE": 0.0, "BENZENE": 0.0, "O3": 30.0}, {}) == ()
    assert aromatics_used({"TOLUENE": 1.0}, {}) == ("TOLUENE",)
    assert aromatics_used({"MXYL": 1.0}, {"BENZENE": 0.1}) == ("BENZENE", "MXYL")
    assert aromatics_used({"CRESOL": 0.5}, {}) == ("TOLUENE",)          # a toluene-only product
    assert aromatics_used({"BZFUONE": 0.5}, {}) == ("BENZENE", "TOLUENE")  # shared by both
    assert aromatics_used({"MALANHY": 0.5}, {}) == ("BENZENE", "TOLUENE", "MXYL")   # by all three
    assert aromatics_used({}, {}, hold_constant=("MXYL",)) == ("MXYL",)
    assert aromatics_used({}, {}, outdoor_gases_ppb={"BENZAL": 0.2}) == ("TOLUENE",)
    assert aromatics_used({"LIMONENE": 5.0}, {"NO": 1.0}, hold_constant=("NO2",)) == ()
    assert uses_toluene({"TOLUENE": 1.0}, {}) and not uses_toluene({"BENZENE": 1.0}, {})


def test_class_map_with_an_aromatic_falls_back_to_li2016_for_its_products():
    lim = _class_map("mcm_limonene_cstar_298K.csv", ())
    assert len(lim.species) == 228
    for name, n_cond in (("BENZENE", 271), ("TOLUENE", 289), ("MXYL", 297)):
        cm = _class_map("mcm_limonene_cstar_298K.csv", (name,))
        assert len(cm.species) == n_cond
        assert set(cm.fallback) >= aromatic_species(name) & set(cm.species)


def _urban(hours=1.0, **gases):
    s = load_preset("level1_urban")
    return s.replace(duration_h=hours, gases_ppb={**s.gases_ppb, **gases})


def test_urban_preset_has_no_aromatic_and_runs_the_limonene_mechanism():
    s = load_preset("level1_urban")
    assert not set(URBAN_VOC_SPECIES) & set(s.gases_ppb)
    assert aromatics_used(s.gases_ppb, s.emissions_ppb_per_h, s.hold_constant) == ()
    r = run(_urban())
    assert not set(URBAN_VOC_SPECIES) & set(r.gases_ppb)
    assert r.diagnostics["warnings"] == []


@pytest.mark.parametrize("voc,lo,hi", [("BENZENE", 0.95, 0.995), ("TOLUENE", 0.8, 0.98), ("MXYL", 0.3, 0.75)])
def test_each_voc_is_reported_and_consumed_in_the_order_of_its_k_oh(voc, lo, hi):
    # 11:00-13:00 local at URBAN_VOC_PPB_ON; k_OH(298 K): benzene 1.2e-12, toluene 5.6e-12,
    # m-xylene 2.3e-11 cm3 s-1 (MCM), so the fraction left after 2 h falls in that order
    s = _urban(2.0, **{voc: URBAN_VOC_PPB_ON}).replace(start_utc="2026-07-01T15:00:00+00:00")
    r = run(s)
    v = np.asarray(r.gases_ppb[voc])
    assert v[0] == pytest.approx(URBAN_VOC_PPB_ON, rel=1e-6)
    assert lo < v[-1] / v[0] < hi
    assert r.diagnostics["warnings"] == []                   # no c* warning: SOA and nucleation off
    assert not (set(URBAN_VOC_SPECIES) - {voc}) & set(r.gases_ppb)   # the others are not integrated


def test_aromatic_products_are_valid_names_everywhere():
    s = load_preset("level2_limonene_soa").replace(duration_h=0.5, gases_ppb={"CRESOL": 0.0, "MXYLOL": 0.0})
    run(s)                                          # zero: limonene mechanism, names accepted
    with pytest.raises(ValueError, match="unknown mechanism species"):
        run(s.replace(gases_ppb={"NAPHTHALENE": 1.0}))
