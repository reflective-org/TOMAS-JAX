"""Single source of truth for the TOMAS-JAX vs PyCHAM benchmark.

Paths, pinned PyCHAM versions, timeouts, comparison thresholds, plot colours and the
case settings. Every other module in ``benchmarks/pycham`` imports from here.
"""
from __future__ import annotations

import os
from pathlib import Path

from tomas_jax.box.coupling import RHO_REPORT

# --- Paths -----------------------------------------------------------------------------
BENCH_DIR = Path(__file__).resolve().parent                 # benchmarks/pycham
TOMAS_JAX_ROOT = BENCH_DIR.parents[1]
CASES_DIR = BENCH_DIR / "cases"
DOC_FIG_DIR = TOMAS_JAX_ROOT / "docs" / "aaar" / "figures" / "pycham"

# PyCHAM install (outside the repo, see README.md). Override with env PYCHAM_BENCH.
PYCHAM_BENCH = Path(os.environ.get("PYCHAM_BENCH", TOMAS_JAX_ROOT.parent / "pycham-bench"))
PYCHAM_TEMPLATE = PYCHAM_BENCH / "PyCHAM"     # pinned + patched checkout; copied per case
PYCHAM_PYTHON = PYCHAM_BENCH / ".venv" / "bin" / "python"
PYCHAM_INPUTS = PYCHAM_BENCH / "inputs"       # generated model_var / eqn / xml / J.xlsx per case
PYCHAM_OUTPUTS = PYCHAM_BENCH / "outputs"     # raw PyCHAM results per case (large)
RUN_PYCHAM = BENCH_DIR / "run_pycham.py"      # headless driver (copied from pycham-bench)
WRITE_J_XLSX = BENCH_DIR / "write_j_xlsx.py"  # runs in the PyCHAM venv (needs openpyxl)

# Pinned upstream versions (README.md explains how to recreate the install).
PYCHAM_REPO = "https://github.com/simonom/PyCHAM.git"
PYCHAM_SHA = "7880d81991542ebbcd6597e01ddcf9d93d997a72"        # v5.8.0
UMANSYSPROP_REPO = "https://github.com/loftytopping/UManSysProp_public.git"
UMANSYSPROP_SHA = "f6d8645d40f47e57111d4ec77c5427502eeb5d86"
PYCHAM_PATCHES = ("single_seed_dry_fix.patch",)                 # applied; see patches/
# mcm331_kpp_photolysis.patch is kept for the record but NOT applied: the benchmark
# prescribes J from TOMAS-JAX through light_status = J.xlsx instead.

# Species data shipped with the pinned PyCHAM checkout.
PYCHAM_MCM_XML = PYCHAM_TEMPLATE / "PyCHAM" / "prop_store" / "mcm_v3p3p1_all_species.xml"
PYCHAM_MCM_CONSTANTS = PYCHAM_TEMPLATE / "PyCHAM" / "prop_store" / "constants_mcm.f90"
# Species of the limonene .eqn that are not in the MCM xml (SMILES for PyCHAM only;
# D9 = 1-butanol-d9 OH tracer, initialised to 0 in every case).
EXTRA_XML_SPECIES = {"D9": "CCCCO", "D9PROD": "CCCCO[O]"}

# --- Timeouts (every run is bounded) -----------------------------------------------------
PYCHAM_TIMEOUT_S = 3600.0     # one PyCHAM simulation
OURS_TIMEOUT_S = 1800.0       # one TOMAS-JAX run (enforced through driver.run(cancel=...))
COPY_TIMEOUT_S = 600.0        # cp -R of the PyCHAM checkout
XLSX_TIMEOUT_S = 120.0        # writing J.xlsx in the PyCHAM venv
# run_all.py: cases in order and the wall-clock limit of each step (whole process group)
RUN_ALL_CASES = ("g2", "G3", "S1", "S2")
RUN_ALL_TIMEOUTS_S = {
    "g2": {"run": 1800.0, "analyse": 600.0},
    "G3": {"run": 5400.0, "analyse": 900.0},
    "S1": {"run": 3600.0, "analyse": 600.0},
    "S2": {"run": 5400.0, "analyse": 900.0},
}
RESULT_SIZE_LIMIT_MB = 2.5    # per case results/ (guideline ~2 MB); run_all warns above

# --- PyCHAM photolysis indexing ----------------------------------------------------------
# PyCHAM indexes J by MCM photolysis number (J[4] = J_NO2); the KPP names map through
# constants_mcm.f90. J.xlsx has row 1 = times [s] and one row per index 0..61.
N_PYCHAM_J = 62
MCM_J_NUMBER = {
    "J_O3_O1D": 1, "J_O3_O3P": 2, "J_H2O2": 3, "J_NO2": 4, "J_NO3_NO": 5, "J_NO3_NO2": 6,
    "J_HONO": 7, "J_HNO3": 8, "J_HCHO_H": 11, "J_HCHO_H2": 12, "J_CH3CHO": 13,
    "J_C2H5CHO": 14, "J_C3H7CHO_HCO": 15, "J_C3H7CHO_C2H4": 16, "J_IPRCHO": 17,
    "J_MACR_HCO": 18, "J_MACR_H": 19, "J_C5HPALD1": 20, "J_CH3COCH3": 21, "J_MEK": 22,
    "J_MVK_CO": 23, "J_MVK_C2H3": 24, "J_GLYOX_H2": 31, "J_GLYOX_HCHO": 32,
    "J_GLYOX_HCO": 33, "J_MGLYOX": 34, "J_BIACET": 35, "J_CH3OOH": 41, "J_CH3NO3": 51,
    "J_C2H5NO3": 52, "J_NC3H7NO3": 53, "J_IC3H7NO3": 54, "J_TC4H9NO3": 55, "J_NOA": 56,
}

# --- PyCHAM numerics ----------------------------------------------------------------------
PYCHAM_UPDATE_STEP_S = 60.0       # rates, J and RO2 refresh interval (PyCHAM default 1 s)
PYCHAM_RECORDING_STEP_S = 60.0
PYCHAM_INT_TOL = (1.0e-3, 1.0e-4)  # PyCHAM default (atol [molec/cm3], rtol), BDF

# --- Comparison ---------------------------------------------------------------------------
# A species enters a statistic at a time when PyCHAM (the reference) is at least this
# abundant. 1e-5 ppb ~ 2.5e5 molec/cm3 keeps OH (~3e-5 ppb) and stays 25x above the
# TOMAS-JAX solver atol (1e4 molec/cm3 ~ 4e-7 ppb).
ABUNDANCE_THRESHOLD_PPB = 1.0e-5
STRICT_THRESHOLD_PPB = 1.0e-3      # secondary statistic: "abundant" species only (>= 1 ppt)
AGREEMENT_BAND = 0.10              # +-10 % lines and "within 10 %" fractions
COMPARE_DT_S = 300.0               # common time grid for all-species statistics
SNAPSHOT_TIMES_H = (1.0, 3.0, 6.0)

# Species that are bookkeeping, not chemistry (excluded from the all-species statistics):
# SA is reset every coupling interval in TOMAS-JAX (it becomes the H2SO4 source of the
# aerosol step; compared separately), NA_HET is a TOMAS-JAX-only counter, NA is the
# <33>/<34> sink product, PROD is dropped by the TOMAS-JAX parser, H2O is set from RH.
NON_CHEMICAL_SPECIES = ("SA", "NA_HET", "NA", "PROD", "H2O")

# --- Plotting -----------------------------------------------------------------------------
LABEL_OURS = "TOMAS-JAX"
LABEL_PYCHAM = "PyCHAM"
# One colour = one meaning in every figure of every case (validated with the dataviz
# palette checker: within each figure every pair is >= 15 OKLab dE apart for normal
# vision and >= 6 under protan/deutan simulation; line style is the secondary encoding).
# Attribution curves "run A vs run B" take the colour of the difference that the pair
# isolates or removes (cumulative steps keep the colour of the last step added).
COLORS = {
    # model identity
    "ours": "#0072B2",               # TOMAS-JAX at the preset settings (solid)
    "ours_alt": "#56B4E9",           # TOMAS-JAX numerical sensitivity: 60 s coupling, 160 bins (dashed)
    "pycham": "#D55E00",             # PyCHAM reference run (solid, thick)
    "pycham_alt": "#E69F00",         # PyCHAM with another input: own Li 2016 c*, J every 900 s (dashed/hatched)
    "pycham_native": "#000000",      # PyCHAM with its own (SMILES-based) vapour pressures (S2)
    # attribution: which difference a curve isolates or removes
    "attr_matched": "#009E73",       # every difference the case attributes removed (final residual)
    "attr_solver_both": "#882255",   # both models at tight solver tolerances
    "attr_solver_ours": "#A6761D",   # TOMAS-JAX default vs tight numerics (its own error)
    "attr_solver_pycham": "#CC79A7", # PyCHAM default vs tight numerics (its own error)
    "attr_hold": "#C51B7D",          # held-species convention (G3: NO2 held inside the solver)
    "attr_frozen_rates": "#762A83",  # rates/RO2 frozen per coupling interval, all else matched (G3)
    "attr_kelvin_off": "#661100",    # Kelvin term off in both models (S1, S2)
    "attr_transfer": "#332288",      # gas-to-particle transfer coefficient matched (S1, S2)
    "attr_raoult": "#AA4499",        # Raoult convention matched: mass- instead of mole-fraction (S2)
    # neutral, non-data
    "reference": "#4D4D4D",          # 1:1 and zero lines
    "band": "#9E9E9E",               # +-10 % lines / shading
    "night": "#E6E6E6",              # night shading (G3)
    "sun_event": "#7F7F7F",          # sunset / sunrise markers (G3)
    "annotation": "#333333",         # annotation text
    "annotation_box": "#CCCCCC",     # annotation box edge
    "grid": "#E0E0E0",               # axes grid
}
# snapshot times are ordered -> one sequential hue, light (first) to dark (last)
SNAPSHOT_COLORS = ("#B08FD8", "#7B4FBF", "#3F1D7A")
FIG_DPI = 150
FONT_SIZE = 11
# Every dry diameter in every particle figure: dry mass per particle at this density, the
# density the tutorial uses for reported dry diameters (tomas_jax.box.coupling.RHO_REPORT).
DISPLAY_DRY_DENSITY_KG_M3 = RHO_REPORT
DRY_DIAMETER_LABEL = f"dry diameter at {DISPLAY_DRY_DENSITY_KG_M3:.0f} kg m⁻³ [nm]"

# --- Case G2: gas-only, Level 2 conditions -----------------------------------------------
G2 = {
    "name": "g2",
    "preset": "level2_limonene_soa",
    "description": "Gas-phase only, Level 2 conditions (6 h, constant light, J prescribed)",
    # all aerosol processes off in TOMAS-JAX; PyCHAM has number_size_bins = 0
    "processes": {"coagulation": False, "h2so4_condensation": False,
                  "soa_partitioning": False, "n2o5_uptake": False, "nucleation": False},
    "coupling_dt_s": 300.0,               # preset value
    "coupling_dt_sensitivity_s": 60.0,    # = PyCHAM update_step
    # <34> N2O5 = NA + NA : 4.0E-4 is replaced in TOMAS-JAX by gamma uptake (gamma = 0 here)
    "eqn_rate_overrides": {"34": "0.0"},
    "key_species": ("LIMONENE", "O3", "NO", "NO2", "OH", "HO2", "NO3", "N2O5", "HNO3",
                    "HCHO", "H2O2", "CO"),
    # attribution runs (numerical sensitivity, not tuning): solver tolerances / water
    "ours_tight_tol": {"rtol": 1e-5, "atol": 1.0},          # TOMAS-JAX default 1e-3 / 1e4
    "pycham_tight_int_tol": (1.0e-3, 1.0e-6),               # PyCHAM default (1e-3, 1e-4)
    "attribution_dt_s": 900.0,            # output grid of the attribution runs (file size)
    "n_pan_species": 4,                   # PAN + the most abundant other PAN-type species
    "n_particle_relevant": 6,             # top particle-phase contributors, full Level 2 run
}

# --- Case G3: gas-only, Level 3 day/night ---------------------------------------------
G3 = {
    "name": "G3",
    "preset": "level3_day_night",
    "description": "Gas-phase only, Level 3 conditions (24 h day/night, TUV-x J prescribed, "
                   "NO2 held at 1 ppb)",
    # all aerosol processes off in TOMAS-JAX; PyCHAM has number_size_bins = 0
    "processes": {"coagulation": False, "h2so4_condensation": False,
                  "soa_partitioning": False, "n2o5_uptake": False, "nucleation": False},
    "gamma_n2o5": 0.0,                    # preset 0.02; uptake is off in a gas-only case
    "coupling_dt_s": 900.0,               # preset value
    "coupling_dt_sensitivity_s": 60.0,    # = PyCHAM update_step
    # <34> N2O5 = NA + NA : 4.0E-4 is replaced in TOMAS-JAX by gamma uptake (gamma = 0 here)
    "eqn_rate_overrides": {"34": "0.0"},
    # TOMAS-JAX hold_constant (reset after every coupling interval) -> PyCHAM const_comp
    # (d[NO2]/dt = 0 inside the ODE). The preset list is checked against this.
    "held_species": ("NO2",),
    # common output grid of all-species statistics: the preset run writes every 900 s
    "compare_dt_s": 900.0,
    # snapshots for scatter / ratio figures: first 900 s output after geometric sunset
    # (20:53 EDT), local midnight, local noon (end of run). Hours since start (12:00 EDT).
    "snapshots": (("sunset", 9.0), ("midnight", 12.0), ("noon", 24.0)),
    "utc_offset_h": -4.0,                 # EDT, for local-time axes
    "transition_window_h": 1.5,           # +- window of the sunset/sunrise zoom figure
    "key_species": ("LIMONENE", "O3", "NO", "NO2", "OH", "HO2", "NO3", "N2O5", "HNO3",
                    "PAN", "HCHO", "H2O2"),
    "product_species": ("LIMAL", "LIMKET", "LIMONONIC", "LIMANO3", "LIMALNO3"),
    "transition_species": ("NO", "O3", "OH", "HO2", "NO3", "N2O5"),
    "n_particle_relevant": 5,             # top particle contributors, full Level 3 run
    # attribution runs (numerical sensitivity, not tuning)
    "ours_tight_tol": {"rtol": 1e-5, "atol": 1.0},          # TOMAS-JAX default 1e-3 / 1e4
    "pycham_tight_int_tol": (1.0e-3, 1.0e-6),               # PyCHAM default (1e-3, 1e-4)
    # output grid of the attribution-only runs and of all attribution pairs (file size;
    # the sunset/sunrise-affected outputs 21:00 and 06:00 EDT are on it)
    "attribution_dt_s": 1800.0,
    "key_dt_s": 180.0,                    # key-species time series of the 60 s runs
    "no2_substep_s": 60.0,                # in-interval NO2 of the 900 s run (sawtooth)
    # rate-constant check times [h]: day, the interval holding sunset, midnight, the
    # interval holding sunrise, morning
    "rate_check_times_h": (1.0, 8.875, 12.0, 17.775, 21.0),
    # result-file size: species never above this in any run are not stored (10x below
    # the comparison threshold, so no compared or "TOMAS-JAX only" species is lost)
    "store_floor_ppb": 1.0e-6,
}

# --- Case S1: Level 1 surrogate condensation/evaporation, no chemistry ----------------
S1 = {
    "name": "S1",
    "preset": "level1_volatility",
    "description": "Level 1 surrogate condensation/evaporation (no chemistry), 12 h",
    "cstar_ugm3": (1.0, 100.0),
    # TOMAS-JAX processes: only SOA partitioning (+ water equilibrium, always on)
    "processes": {"chemistry": False, "nucleation": False, "coagulation": False,
                  "h2so4_condensation": False, "soa_partitioning": True,
                  "n2o5_uptake": False},
    "coupling_dt_s": 300.0,            # preset value (no gas chemistry: only sets the
    "output_dt_s": 300.0,              # observer interval; aerosol micro-step is 60 s)
    "compare_dt_s": 300.0,             # common time grid for the statistics
    "snapshot_times_h": (0.0, 3.0, 6.0, 12.0),
    # PyCHAM species. Both organics share one SMILES (MCM C923CO3H, C10H16O4) so that
    # PyCHAM's mole-fraction Raoult term equals TOMAS-JAX's mass-fraction one.
    "surrogate_name": "SURR",
    "poa_name": "POA",
    "organic_smiles": "OOC(=O)CC(CCC(=O)C)C(=C)C",
    "pycham_organic_mw": 200.23164,    # openbabel molwt of the SMILES (checked after the run)
    "pycham_water_mw": 18.0,           # PyCHAM's H2O molar mass (y_mw.npy)
    "sulfate_mw": 96.06,               # PyCHAM 'core' = particle sulfate (TOMAS SO4 slot)
    # TOMAS bins with at least this many particles get the same number in PyCHAM
    # (smaller tails are left empty); 1e-6 cm-3 loses < 1e-5 cm-3 in total
    "pycham_min_bin_number_cm3": 1.0e-6,
    "pycham_update_step_s": 60.0,
    "pycham_recording_step_s": 60.0,
    # numerical sensitivity runs (not tuning): PyCHAM update step / tolerance, ours micro-step
    "pycham_tight": {"update_step_s": 10.0, "int_tol": (1.0e-3, 1.0e-6)},
    "ours_micro_dt_sensitivity_s": 10.0,
    # bin-resolution runs (same 1.7 nm - 17.5 um range, mass ratio 2^(40/nbins))
    "nbins_sensitivity": (80, 160),
    # agreement band used in the summary (fraction)
    "band": 0.10,
}

# --- Case S2: Level 2 limonene SOA ----------------------------------------------------
S2 = {
    "name": "S2",
    "preset": "level2_limonene_soa",
    "description": "Level 2 limonene SOA, coagulation and nucleation off in both models",
    "processes": {"coagulation": False, "nucleation": False, "h2so4_condensation": True,
                  "soa_partitioning": True, "n2o5_uptake": False},
    "coupling_dt_s": 300.0,               # preset value
    "coupling_dt_sensitivity_s": 60.0,    # = PyCHAM update_step
    # <34> N2O5 = NA + NA : 4.0E-4 is replaced in TOMAS-JAX by gamma uptake (gamma = 0)
    "eqn_rate_overrides": {"34": "0.0"},
    # PyCHAM output every 300 s (= COMPARE_DT_S); the 60 s default would give ~0.5 GB of
    # text for 40 size bins x 728 components over 6 h. The update step stays 60 s.
    "pycham_recording_step_s": 300.0,
    # --- volatility given to PyCHAM (vol_Comp / volP, Pa, constant in T) -------------
    # condensable species: c* of their TOMAS-JAX class centre (or their own Li 2016 c*
    # in the sensitivity run) at the run temperature, converted with PyCHAM's MW.
    # Every other component gets GAS_ONLY_VP_PA and partitioning is switched off for
    # anything above PPARTIT_CUTOFF_PA (water is exempt inside PyCHAM).
    "gas_only_vp_pa": 1.0e5,
    "ppartit_cutoff_pa": 100.0,
    # H2SO4 (SA) and the seed organic are non-volatile in TOMAS-JAX
    "nonvolatile_vp_pa": 1.0e-20,
    # --- seed ---------------------------------------------------------------------------
    # absorbing non-volatile seed organic for PyCHAM: added to its .eqn copy through a
    # dead reaction (k = 0) so PyCHAM knows the component. The SMILES only sets its MW
    # (density and vapour pressure are overridden): 6-hydroxyundecanedioic acid,
    # C11H20O5, 232.3 g/mol, close to the number-mean MW of the TOMAS-JAX particle-phase
    # SOA (233 g/mol at 3 and 6 h; checked in run_info.json "seed_oa_mw_check"), so that
    # PyCHAM's mole fractions over seed OA + SOA are close to TOMAS-JAX's mass fractions.
    "seed_oa_name": "SEEDOA",
    "seed_oa_smiles": "OC(=O)CCCCC(O)CCCCC(=O)O",
    "seed_dead_reaction": "<990001> SEEDOA = PROD : 0.0 ;",
    # --- runs ---------------------------------------------------------------------------
    "pycham_variants": ("class", "exact"),   # class-centre c* (main) / own Li 2016 c*
    "pycham_gas_only_variant": "nopart",      # same inputs, number_size_bins = 0
    # attribution (numerical experiment, not tuning): class-centre c* converted to Pa
    # with ONE molar mass (the seed-OA MW) for every species. PyCHAM's mole-fraction
    # Raoult term then equals a mass-fraction Raoult term (TOMAS-JAX convention) up to
    # the ratio (number-mean MW of the organic phase) / (seed-OA MW).
    # "massfrac_nokelvin": the same with the Kelvin term off (surface tension 0 in a
    # patched copy of the checkout, PyCHAM_S2_nokelvin), which also removes PyCHAM's
    # Kelvin term on particle water.
    "pycham_attribution_variants": ("massfrac", "massfrac_nokelvin"),
    "pycham_kelvin_off_variants": ("massfrac_nokelvin",),
    # volatility sensitivity (PyCHAM only, not a like-for-like comparison): the 245
    # condensable species keep PyCHAM's own SMILES-based vapour pressures (its default
    # Nannoolal et al. 2008 method, T-dependent) instead of the TOMAS-JAX c*; every other
    # input as "class". Shows how much SOA and the NOx-sink feedback depend on A8.
    "pycham_volatility_variants": ("native",),
    # attribution: TOMAS-JAX (coupling 60 s) with its organic condensation sink scaled
    # by the PyCHAM/TOMAS-JAX transfer-rate ratio of the condensing species (transfer.py),
    # and the same with the vbs-jax Kelvin term off (sigma = 0)
    "ours_attribution_dt_s": 60.0,
    "ours_attribution_runs": {"dt60_cs_pycham": {"kelvin_off": False},
                              "dt60_cs_pycham_nokelvin": {"kelvin_off": True}},
    "snapshot_times_h": (1.0, 3.0, 6.0),
    "psd_times_h": (0.0, 3.0, 6.0),
    "key_species": ("LIMONENE", "O3", "NO", "NO2", "OH", "HO2", "NO3", "HNO3", "HCHO",
                    "H2O2", "LIMONONIC", "LIMAL"),
    # gas-phase response to partitioning (each model against its own run without particles)
    "feedback_species": ("LIMONENE", "O3", "NO", "NO2", "OH", "HO2", "NO3", "N2O5", "HNO3",
                         "HCHO", "H2O2", "PAN", "SA"),
    "n_top_species": 10,
    # result-file size: gas species below this in EVERY stored run (both models) are not
    # stored (100x below ABUNDANCE_THRESHOLD_PPB, so no compared species is lost)
    "store_floor_ppb": 1.0e-7,
}
