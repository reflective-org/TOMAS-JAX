"""Case S2: Level 2 limonene SOA (definition shared by inputs.py, run.py and analyse.py).

TOMAS-JAX: preset ``level2_limonene_soa`` (6 h, constant light, seed 2000 cm-3,
GMD 80 nm, GSD 1.6, 30 % non-volatile absorbing OA + 70 % sulfate) with coagulation and
nucleation OFF; H2SO4 condensation and SOA partitioning ON; gamma(N2O5) = 0.
PyCHAM: same gases/T/P/RH, J prescribed from TOMAS-JAX, the TOMAS-JAX size grid and seed
(per-bin number and dry mass), the TOMAS-JAX volatility of every condensable species
(its class-centre c*), everything else non-partitioning. See inputs.py for every
setting and README.md for the list of differences that cannot be removed.

The case settings (``S2``) live in ``benchmarks/pycham/config.py`` with the other cases.
"""
from __future__ import annotations

from pathlib import Path

from tomas_jax.box.scenario import Scenario, load_preset

from ...config import S2
from ...metrics import signed_pct


CASE = S2["name"]
CASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CASE_DIR / "results"
FIG_DIR = CASE_DIR / "figures"

# result files (small, kept in the repo)
OURS_NPZ = {300: RESULTS_DIR / "ours_dt300.npz", 60: RESULTS_DIR / "ours_dt60.npz"}
PYCHAM_ALL_VARIANTS = (S2["pycham_variants"] + (S2["pycham_gas_only_variant"],)
                       + S2["pycham_attribution_variants"] + S2["pycham_volatility_variants"])
PYCHAM_NPZ = {v: RESULTS_DIR / f"pycham_{v}.npz" for v in PYCHAM_ALL_VARIANTS}
# TOMAS-JAX with SOA partitioning and H2SO4 condensation off (gas-phase feedback check)
OURS_NOPART_NPZ = RESULTS_DIR / "ours_dt300_nopart.npz"
# TOMAS-JAX attribution runs: 60 s coupling + organic CS scaled to PyCHAM's transfer
# rate, with and without the Kelvin term
OURS_ATTR_NPZ = {tag: RESULTS_DIR / f"ours_{tag}.npz" for tag in S2["ours_attribution_runs"]}
TRANSFER_JSON = RESULTS_DIR / "transfer_rates.json"
VOLATILITY_CSV = RESULTS_DIR / "pycham_volatility.csv"
SETUP_JSON = RESULTS_DIR / "setup.json"
RUN_INFO_JSON = RESULTS_DIR / "run_info.json"
SUMMARY_JSON = RESULTS_DIR / "summary.json"
GAS_STATS_CSV = RESULTS_DIR / "gas_species_stats.csv"


def scenario(coupling_dt_s: float | None = None, partitioning: bool = True) -> Scenario:
    """The S2 TOMAS-JAX scenario (preset coupling unless ``coupling_dt_s`` is given).

    ``partitioning=False`` switches SOA partitioning and H2SO4 condensation off (the
    seed stays a spectator): the gas-only counterpart used to size the feedback of
    partitioning on the gas phase.
    """
    s = load_preset(S2["preset"])
    dt = coupling_dt_s or S2["coupling_dt_s"]
    processes = dict(S2["processes"], chemistry=True)
    if not partitioning:
        processes.update(soa_partitioning=False, h2so4_condensation=False)
    return s.replace(name=f"benchmark {CASE}", processes=processes,
                     gamma_n2o5=0.0, numerics={"coupling_dt_s": dt, "output_dt_s": dt})


def mismatches(setup: dict, m: dict) -> list[dict]:
    """Every place the two S2 set-ups differ (what, why, measured effect).

    ``setup`` is run.py's setup record (what PyCHAM was given), ``m`` the measured numbers
    from analyse.measured, so the text in summary.json follows the runs."""
    pct = signed_pct
    seed = setup["seed"]
    w, wr = m["water"], m["water_rel"]
    st = m["steps"]
    zo, zp = m["size_6h"]["o300"], m["size_6h"]["py_class"]
    rng = lambda dct, keys: (f"{pct(min(dct[k] for k in keys))} to "    # noqa: E731
                             f"{pct(max(dct[k] for k in keys))}")
    late = ("0.5h", "1h", "3h", "6h")
    dens = m["density"]
    nit = m["nitrogen"]
    return [
        {"what": "Operator split. TOMAS-JAX integrates chemistry for one coupling interval, "
                 "then partitions the vapour formed over the same interval (A12). PyCHAM solves "
                 "chemistry and partitioning in one ODE, rates refreshed every 60 s.",
         "why": "Driver design; the interval is a model setting.",
         "effect": "Vapour formed during an interval gets the whole interval to condense, so SOA "
                   f"runs ahead early: {pct(m['soa_rel_o300']['0.5h'])} at 30 min, "
                   f"{pct(m['soa_rel_o300']['1h'])} at 1 h with 300 s coupling; "
                   f"{pct(m['soa_rel_o60']['0.5h'])} and {pct(m['soa_rel_o60']['1h'])} at 60 s. "
                   f"60 s vs 300 s coupling differ by {pct(m['coupling_60_vs_300']['6h'])} at 6 h."},
        {"what": "Gas-to-particle transfer rate. vbs-jax: Fuller D with one diffusion volume "
                 "(120) and the class-mean MW, mfp = 2D/c, Zaveri et al. (2014) Fuchs-Sutugin. "
                 "PyCHAM: each species' own diffusion volume from SMILES (~200-235), mfp = 3D/c, "
                 "Jacobson (2005) form.",
         "why": "Hard-coded in each model.",
         "effect": f"On the same size distribution PyCHAM's organic condensation sink is "
                   f"{m['cs_factor']:.2f}x ours at t = 0 ({m['cs_ratio_6h']:.2f}x at 6 h sizes; "
                   f"diffusivity alone {m['d_ratio']:.2f}x). With our sink scaled by "
                   f"{m['cs_factor']:.2f} (and 60 s coupling) the SOA difference is "
                   f"{pct(st['transfer']['0.5h'])} at 30 min and {pct(st['transfer']['1h'])} at 1 h."},
        {"what": "Raoult convention: organic-phase mass fraction (TOMAS-JAX, soacond) vs mole "
                 "fraction (PyCHAM). The seed-OA MW (232.3 g/mol) is close to the number-mean "
                 "SOA MW to keep them close.",
         "why": "Each model's formulation.",
         "effect": "PyCHAM with mass-fraction Raoult ('massfrac': one MW in the c* -> Pa "
                   f"conversion) has {rng(m['raoult_alone'], late)} more SOA than with mole fraction."},
        {"what": "Kelvin term. vbs-jax: sigma 0.025 N/m, rho_org 1200 kg/m3, class MW, wet Dp. "
                 "PyCHAM: sigma 0.072 N/m, mass-weighted mean particle MW (incl. water and "
                 "sulfate), particle density; it also applies Kelvin to water.",
         "why": "Hard-coded in both (PyCHAM partit_var_prep.py surfT).",
         "effect": f"Kelvin lowers SOA by {pct(-m['kelvin_ours']['6h'])[1:]} in TOMAS-JAX and "
                   f"{pct(-m['kelvin_pycham']['6h'])[1:]} in PyCHAM at 6 h "
                   f"({pct(-m['kelvin_ours']['0.5h'])[1:]} / {pct(-m['kelvin_pycham']['0.5h'])[1:]} "
                   "at 30 min). With transfer and Raoult matched the SOA difference is "
                   f"{rng(st['raoult'], late)} with Kelvin on and {rng(st['kelvin'], late)} with "
                   "Kelvin off in both (patched PyCHAM copy, vbs-jax sigma = 0)."},
        {"what": "Within-class species identity. TOMAS-JAX partitions c* classes and hands the "
                 "class change to species with a tracker (ClassBridge.scatter); PyCHAM is "
                 "explicit per species.",
         "why": "Lumping design (A6).",
         "effect": "Members of one class should have one particle fraction; class log c* = 0.5 "
                   f"at 6 h: PyCHAM {min(m['within_class_F']['pycham']):.2f}-"
                   f"{max(m['within_class_F']['pycham']):.2f}, TOMAS-JAX "
                   f"{min(m['within_class_F']['ours']):.2f}-{max(m['within_class_F']['ours']):.2f}. "
                   "Per-species particle and gas masses of condensables differ accordingly; class "
                   "totals are robust."},
        {"what": "Sectional numerics: TOMAS two-moment fixed bins (SOA growth and evaporation "
                 "move particles with the PPM log-mass remap, AerosolStepConfig."
                 "growth_bin_transfer = 'ppm') vs PyCHAM moving centre on the same 40-bin grid.",
         "why": "Each model's scheme.",
         "effect": "At 6 h, from each bin's mean dry size: number-mean dry diameter "
                   f"{zo['number_mean_nm']:.1f} (TOMAS-JAX) vs {zp['number_mean_nm']:.1f} nm "
                   f"(PyCHAM, {pct(zo['number_mean_nm'] / zp['number_mean_nm'] - 1)}), "
                   f"GSD {zo['gsd']:.3f} vs {zp['gsd']:.3f} ({pct(zo['gsd'] / zp['gsd'] - 1)}), "
                   f"dry-mass-weighted mean {zo['dry_mass_mean_nm']:.1f} vs "
                   f"{zp['dry_mass_mean_nm']:.1f} nm; particles below the seed's 1st-percentile "
                   f"diameter {100 * zo['frac_below_seed_p01']:.2f} % vs "
                   f"{100 * zp['frac_below_seed_p01']:.2f} %. "
                   "PyCHAM bins merge, so its curve is spiky. Number and mass are conserved in "
                   "both."},
        {"what": "Particle water. TOMAS-JAX: ezwatereqm, "
                 f"{seed['water_per_mass']['sulfate']:.4f} kg water per kg SO4 (NH4HSO4 ratio "
                 f"x 1.2) and {seed['water_per_mass']['organic']:.4f} per kg of any organic (A9); "
                 "no Kelvin term. PyCHAM: ideal solution with per-component dissociation "
                 f"constants (SA {seed['water_diss_wrtw']['SA']:.4f}, SEEDOA "
                 f"{seed['water_diss_wrtw']['SEEDOA']:.4f}, condensables "
                 f"{seed['water_diss_wrtw']['condensable_min']:.3f}-"
                 f"{seed['water_diss_wrtw']['condensable_max']:.3f}) that give TOMAS-JAX's water "
                 "per unit mass of each component when Kelvin = 1, plus its Kelvin term on water.",
         "why": "PyCHAM has no fixed hygroscopic-growth input; its water always feels Kelvin.",
         "effect": "PyCHAM water relative to TOMAS-JAX: "
                   + ", ".join(f"{pct(wr['py_class_vs_o300'][k])} at {k}" for k in ("0h", "1h", "3h", "6h"))
                   + " (Kelvin on water). With Kelvin off in both (attribution pair): "
                   + ", ".join(f"{pct(wr['py_massfrac_nokelvin_vs_ocs_nk'][k])} at {k}"
                               for k in ("0h", "1h", "6h")) + "."},
        {"what": "Densities. TOMAS-JAX: Tang (1997) H2SO4-solution density, organics counted as "
                 f"sulfate (wet seed {seed['rho_wet_tomas']:.0f} kg/m3). PyCHAM: additive "
                 f"volumes, {seed['rho_dry_pycham']:.0f} kg/m3 for every non-water component "
                 "(chosen so the wet seed volume matches at t = 0) and 1000 for water.",
         "why": "PyCHAM has no solution-density model.",
         "effect": "Same wet volume at t = 0. On TOMAS-JAX's bulk composition at 6 h the two "
                   f"rules give {dens['6h']['tomas_kg_m3']:.0f} vs {dens['6h']['pycham_rule_kg_m3']:.0f} "
                   f"kg/m3 ({pct(dens['6h']['rel'])}, about {pct(dens['6h']['rel'] / 3)} in "
                   "diameter for the same mass)."},
        {"what": "Seed sulfate identity. PyCHAM's seed component SA is H2SO4 (SMILES MW "
                 f"{m['seed_sa_mw']:.2f}) and receives TOMAS-JAX's SO4-slot mass; TOMAS-JAX "
                 "treats that mass as SO4 (MW 96) and as NH4HSO4 (x 1.2) for water.",
         "why": "PyCHAM needs a SMILES; there is no bisulfate component.",
         "effect": "Mass identical; moles differ by 2 %. Enters PyCHAM's mass-mean MW in its "
                   "Kelvin term (sulfate is not in the absorbing phase in either model). "
                   "Water is matched per unit mass. Expected effect on SOA: negligible."},
        {"what": "Volatility input. PyCHAM gets volP = the TOMAS-JAX class-centre c*(298.15 K) "
                 "of every condensable species, converted with PyCHAM's own MW, constant in T. "
                 "Non-condensables get 1e5 Pa (ppartit_cutoff 100 Pa); SA and the seed OA "
                 "1e-20 Pa.",
         "why": "Like-for-like test of the partitioning numerics.",
         "effect": "Exact by construction (checked to 4e-11). S2 therefore tests partitioning and "
                   "transport for identical volatility; it does not test the Li et al. (2016) c* "
                   f"(A8). With PyCHAM's own vapour pressures the SOA is "
                   f"{m['native']['6h']:.2f}x the class-centre run at 6 h and its N-containing "
                   f"share {100 * nit['py_native']['n_containing_share']:.0f} % (class-centre "
                   f"{100 * nit['py_class']['n_containing_share']:.0f} %), so the NOx-sink "
                   "feedback is an A8 consequence."},
        {"what": "c* lumping: TOMAS-JAX half-decade classes vs PyCHAM explicit species at the "
                 "class-centre c* (main run) or at each species' own Li c* ('exact').",
         "why": "Lumping design (A6).",
         "effect": f"PyCHAM own c* vs class centre: {pct(m['lumping']['0.5h'])} at 30 min, "
                   f"{pct(m['lumping']['1h'])} at 1 h, {pct(m['lumping']['6h'])} at 6 h."},
        {"what": "Seed and bins. TOMAS-JAX bins 4-32 go to PyCHAM (every bin with N0 >= 1e-6 "
                 "cm-3 plus 4 empty bins on top), bounds at the wet-volume radii of the dry-mass "
                 "edges; SEEDOA (6-hydroxyundecanedioic acid) is a stand-in molecule added "
                 "through a dead reaction.",
         "why": "PyCHAM needs a component with a SMILES; tiny bins only add stiffness.",
         "effect": f"{seed['seed_number_left_out_cm3']:.1e} cm-3 of seed left out; t = 0 number "
                   "and dry mass identical to 1e-6."},
        {"what": "H2SO4: TOMAS-JAX turns the SA counter into an H2SO4 source for TOMAS "
                 "condensation; PyCHAM SA gas condenses in its ODE with vp 1e-20 Pa.",
         "why": "Driver design.",
         "effect": f"Sulfate added by 6 h: {m['sulfate_6h'][0]:.4f} vs {m['sulfate_6h'][1]:.4f} ug/m3."},
        {"what": "Particle-phase diffusion: two-film with D_b = 1e-10 m2/s (vbs-jax) vs well "
                 "mixed (PyCHAM).",
         "why": "vbs-jax default.",
         "effect": "Negligible (< 4e-5 of the gas-side resistance at c* = 1e4 and 100 nm)."},
        {"what": "Gas phase as in G2: rates/J/RO2 frozen per interval, solver tolerances, Buck "
                 "vs Hyland-Wexler water vapour, SA/PROD/NA bookkeeping, <34> rate 0.",
         "why": "See case G2.", "effect": "Without particles the S2 set-ups agree like G2."},
        {"what": "Processes off in both: coagulation, nucleation, N2O5 uptake (gamma = 0).",
         "why": "Isolate partitioning.",
         "effect": "The preset has coagulation on; S2 does not test it."},
    ]
