"""Case S1: Level 1 surrogate condensation/evaporation, no chemistry.

TOMAS-JAX: preset ``level1_volatility`` (seed 2000 cm-3, 50 nm, gsd 1.6, 30 % non-volatile
absorbing OA by mass, rest sulfate; one surrogate organic, MW 200 g/mol, produced at
0.5 ug m-3 h-1 for 12 h; T = 298.15 K, RH = 0.5) with coagulation, nucleation and
H2SO4 condensation off, run for c* = 1 and c* = 100 ug/m3.

PyCHAM: a one-reaction dummy scheme (``SURR = SURR : 0.0``), the surrogate as a continuous
influx, its vapour pressure from c*, a seed of ``core`` (sulfate, non-absorbing) + ``POA``
(non-volatile, absorbing), bin by bin the same initial number and dry mass as TOMAS-JAX,
and densities / water uptake set to reproduce the TOMAS-JAX particle (see
``pycham_setup``). Everything that cannot be made identical is listed in ``MISMATCHES``
and in the README.

The case settings (``S1``) live in ``benchmarks/pycham/config.py`` with the other cases.
"""
from __future__ import annotations

import functools
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tomas_jax.box.config import AerosolStepConfig
from tomas_jax.box.coupling import mass_grid, seed_distribution, kg_cell_to_ugm3
from tomas_jax.box.scenario import Scenario, load_preset
from tomas_jax.core.config import SRTH2O, SRTORG_NUC
from tomas_jax.gaschem.config import AVOGADRO, DHVAP_A_KJ, DHVAP_B_KJ
from tomas_jax.gaschem.environment import air_environment
from tomas_jax.physics.condensation_sink import NEPS_CONDSINK
from tomas_jax.physics.density import calc_density
from tomas_jax.physics.water_equilibrium import equilibrate_water

from vbs_jax.config import T_REF as VBS_T_REF, calc_cstar_T
from vbs_jax.constants import R_GAS

from ...config import S1
from ...metrics import signed_pct
from ...particles import (PYCHAM_WATER_DENSITY, effective_dry_density, pycham_diss_wrtw,
                          water_mass_ratios)

CASE = S1["name"]
CASE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = CASE_DIR / "results"
FIG_DIR = CASE_DIR / "figures"


# result files (small, kept in the repo)
def ours_npz(cstar: float, variant: str = "base") -> Path:
    return RESULTS_DIR / f"ours_{variant}_cstar{cstar:g}.npz"


def pycham_npz(cstar: float, variant: str = "base") -> Path:
    return RESULTS_DIR / f"pycham_{variant}_cstar{cstar:g}.npz"


RUN_INFO_JSON = RESULTS_DIR / "run_info.json"
SETUP_JSON = RESULTS_DIR / "pycham_setup.json"
SUMMARY_JSON = RESULTS_DIR / "summary.json"
STATS_CSV = RESULTS_DIR / "s1_stats.csv"


def scenario(cstar: float, nbins: int | None = None) -> Scenario:
    """The S1 TOMAS-JAX scenario for one surrogate c* [ug/m3 at 298 K].

    ``nbins`` (default: preset, 40) is only changed for the bin-resolution runs.
    """
    s = load_preset(S1["preset"])
    numerics = {"coupling_dt_s": S1["coupling_dt_s"], "output_dt_s": S1["output_dt_s"]}
    if nbins is not None:
        numerics["nbins"] = int(nbins)
    return s.replace(name=f"benchmark {CASE} c*={cstar:g}",
                     processes=dict(S1["processes"]),
                     h2so4_production_cm3_s=0.0,
                     surrogate={"cstar_ugm3": float(cstar)},
                     numerics=numerics)


def dhvap_kj(cstar: float) -> float:
    """Driver rule for the Level 1 surrogate (box/driver.py)."""
    return DHVAP_A_KJ - DHVAP_B_KJ * np.log10(cstar)


def cstar_at_run_temperature(s: Scenario) -> float:
    """c* [ug/m3] TOMAS-JAX actually uses at the run temperature.

    The preset c* is defined at vbs-jax's T_REF = 298.0 K; the run is at 298.15 K, so
    vbs-jax applies Clausius-Clapeyron with the driver's dHvap (+2.7 % for c* = 1).
    PyCHAM gets this value (its volP is constant in T).
    """
    c = s.surrogate.cstar_ugm3
    return float(calc_cstar_T(c, dhvap_kj(c), s.environment.temp_k, VBS_T_REF, r_gas=R_GAS))


@functools.lru_cache(maxsize=4)
def _initial_state(nbins: int, rh: float, temp: float, gmd: float, gsd: float,
                   number: float, org_frac: float):
    cfg = AerosolStepConfig()
    xk = mass_grid(nbins)
    from tomas_jax.box.scenario import SeedMode
    nk, mk = seed_distribution((SeedMode(number, gmd, gsd),), org_frac, xk, cfg.boxvol_cm3)
    mk = np.asarray(equilibrate_water(jnp.asarray(mk), rh, temp, cfg.water_scheme))
    return xk, nk, mk, cfg.boxvol_cm3


def initial_state(s: Scenario):
    """TOMAS-JAX initial aerosol exactly as driver.run builds it: (xk, Nk, Mk, boxvol)."""
    (mode,) = s.seed.modes
    return _initial_state(s.numerics.nbins, s.environment.rh, s.environment.temp_k,
                          mode.gmd_nm, mode.gsd, mode.number_cm3, s.seed.organic_mass_fraction)


def pycham_setup(s: Scenario) -> dict:
    """Every PyCHAM physical input derived from the TOMAS-JAX scenario (documented).

    * density: PyCHAM mixes volumes ideally and keeps water at 1.0 g/cm3 (a dens_Comp
      entry for H2O is ignored). The dry components (core, POA, SURR) get the effective
      dry density rho_d that makes m_dry/rho_d + m_w/1.0 equal TOMAS-JAX's wet volume
      (m_dry + m_w)/rho_Tang for the initial seed (Tang 1997 H2SO4-solution fit with
      organics counted as sulfate, TOMAS-JAX's water), i.e. the same wet volume per
      particle for the same dry and water mass. Note PyCHAM reads ``dens`` in kg/m3
      (its README says g/cm3) and ``seed_dens`` in g/cm3.
    * water: PyCHAM's water Raoult term uses per-component dissociation constants with
      respect to water; diss_i = w_i * MW_i / MW_H2O makes the water mass at RH = 0.5
      equal to TOMAS-JAX's ezwatereqm (w_i = water per unit mass of i) when Kelvin = 1.
      PyCHAM keeps its own Kelvin term on water (72 mN/m), which lowers water in small
      particles (recorded as a mismatch).
    * organics' Raoult term: core (sulfate) seed_diss = 0 and water H2O_diss_wrtorg = 0,
      so only POA + SURR (same molar mass) form the absorbing phase, as in TOMAS-JAX.
    * vapour pressure: volP = c*(T) R T / MW with TOMAS-JAX's c* at the run temperature;
      POA volP = 0 (non-volatile, like TOMAS-JAX's slot 41).
    * production: the same mass rate, converted to ppb/s with PyCHAM's molar mass and
      TOMAS-JAX's air number density (identical to PyCHAM's ppb factor, see G2).
    """
    env = s.environment
    xk, nk, mk, boxvol = initial_state(s)
    populated = nk / boxvol > S1["pycham_min_bin_number_cm3"]
    rho_wet = np.asarray(calc_density(jnp.asarray(mk)))
    # composition is the same in every bin -> one density (bins TOMAS treats as
    # populated, Nk > NEPS_CONDSINK; aerodens returns 1000 for near-empty bins)
    dense = nk > NEPS_CONDSINK
    rho_wet0 = float(np.median(rho_wet[dense]))
    if np.max(np.abs(rho_wet[dense] / rho_wet0 - 1.0)) > 1e-12:
        raise RuntimeError("initial wet density varies across bins")
    dry = mk[:, :SRTH2O].sum(axis=1)
    w0 = float(np.median(mk[dense, SRTH2O] / dry[dense]))        # water / dry mass
    rho_d = effective_dry_density(rho_wet0, w0)                 # kg/m3
    w = water_mass_ratios(s)
    mw_org, mw_s = S1["pycham_organic_mw"], S1["sulfate_mw"]
    cstar_t = cstar_at_run_temperature(s)
    temp = env.temp_k
    volp_pa = cstar_t * 1e-9 / (mw_org * 1e-3) * R_GAS * temp
    m_air = air_environment(env.temp_k, env.pres_pa, env.rh)["M"]
    prod_molec = s.surrogate.production_ugm3_per_h / 3600.0 * 1e-12 / mw_org * AVOGADRO
    cinfl_ppb_s = prod_molec / m_air * 1e9
    f_org = s.seed.organic_mass_fraction
    n_s, n_o = (1.0 - f_org) / mw_s, f_org / mw_org
    seedx = (n_s / (n_s + n_o), n_o / (n_s + n_o))

    # bins: TOMAS-JAX's own, dry mass per particle = bin midpoint mass (as the seed)
    rho_cgs = rho_d * 1e-3                                      # g/cm3
    m_mid = np.sqrt(xk[:-1] * xk[1:])                           # kg
    r_dry_um = (3.0 * m_mid * 1e3 / (4.0 * np.pi * rho_cgs)) ** (1 / 3) * 1e4
    edges_dry_um = (3.0 * xk * 1e3 / (4.0 * np.pi * rho_cgs)) ** (1 / 3) * 1e4
    # PyCHAM moves particles between bins on their total (wet) volume: wet bounds with
    # TOMAS-JAX's initial water
    wet_over_dry_vol = 1.0 + w0 * rho_d / PYCHAM_WATER_DENSITY
    edges_wet_um = edges_dry_um * wet_over_dry_vol ** (1 / 3)
    n_bin = np.where(populated, nk / boxvol, 0.0)
    return {
        "rho_dry_g_cm3": rho_cgs,
        "rho_wet_tomas_initial_g_cm3": rho_wet0 * 1e-3,
        "water_over_dry_tomas_initial": w0,
        "water_per_mass": w,
        "diss_wrtw_sulfate": pycham_diss_wrtw(w["sulfate"], mw_s, env.rh),
        "diss_wrtw_organic": pycham_diss_wrtw(w["organic"], mw_org, env.rh),
        "cstar_nominal_ugm3": s.surrogate.cstar_ugm3,
        "cstar_run_T_ugm3": cstar_t,
        "dhvap_kj": dhvap_kj(s.surrogate.cstar_ugm3),
        "volP_surr_pa": volp_pa,
        "cinfl_ppb_s": cinfl_ppb_s,
        "production_molec_cm3_s": prod_molec,
        "m_air_cm3": float(m_air),
        "seedx_core_poa": seedx,
        "r_dry_um": r_dry_um,
        "edges_dry_um": edges_dry_um,
        "edges_wet_um": edges_wet_um,
        "wet_over_dry_volume": wet_over_dry_vol,
        "n_bin_cm3": n_bin,
        "n_total_ours_cm3": float(nk.sum() / boxvol),
        "n_total_pycham_cm3": float(n_bin.sum()),
        "dry_mass_ours_ugm3": float(kg_cell_to_ugm3(dry.sum(), boxvol)),
        "dry_mass_pycham_ugm3": float(np.sum(n_bin * m_mid) * 1e15),   # #/cm3 * kg -> ug/m3
        "poa_ugm3": float(kg_cell_to_ugm3(mk[:, SRTORG_NUC].sum(), boxvol)),
    }


# Every place the two set-ups cannot be made identical (what, why, measured effect).
# Built by analyse.py from the PyCHAM set-up actually written (``pycham_setup``) and the
# measured results, so the numbers in summary.json cannot drift from the runs.
def mismatches(setup: dict, m: dict) -> list[dict]:
    """``setup``: pycham_setup() of c* = 1; ``m``: measured numbers (analyse.measured)."""
    pct = signed_pct
    k0, k12, k12b = m["kelvin_50nm_0h"], m["kelvin_50nm_12h"], m["kelvin_100nm_12h"]
    tr = m["transfer"]
    w0, w12, wm0 = m["water_c1_0h"], m["water_c1_12h"], m["water_mode_c1_0h"]
    return [
        {"what": "Kelvin term. TOMAS-JAX (vbs-jax): Ke = exp(4 sigma MW_i / (R T rho_org Dp)) "
                 "with sigma = 0.025 N/m, rho_org = 1200 kg/m3 and the condensing species' "
                 "MW = 200. PyCHAM: Ke = exp(2 avMW sigma / (R T r rho_p)) with sigma = 72 mN/m "
                 "(water), the particle's mass-weighted mean molar mass avMW (sulfate 96, "
                 "organics 200, water 18) and the particle density.",
         "why": "Hard-coded in both models (vbs-jax defaults; PyCHAM partit_var_prep.surfT = 72).",
         "effect": f"Nearly identical at t = 0 (Ke at 50 nm wet: {k0[0]:.3f} TOMAS-JAX vs "
                   f"{k0[1]:.3f} PyCHAM; PyCHAM's avMW of ~100 g/mol offsets its larger sigma). "
                   f"As SOA replaces sulfate PyCHAM's Ke grows (c* = 1, 12 h: {k12[0]:.3f} vs "
                   f"{k12[1]:.3f} at 50 nm, {k12b[0]:.3f} vs {k12b[1]:.3f} at 100 nm). Kelvin "
                   f"lowers c* = 100 SOA by {pct(-m['kelvin_effect_c100'][0])[1:]} (TOMAS-JAX) "
                   f"and {pct(-m['kelvin_effect_c100'][1])[1:]} (PyCHAM) at 12 h; the c* = 100 "
                   f"SOA difference of {pct(m['soa_c100_12h_full'])} becomes "
                   f"{pct(m['soa_c100_12h_kelvin_off'])} with Kelvin off in both (patched PyCHAM "
                   "copy with surfT = 0, vbs-jax sigma = 0)."},
        {"what": "Gas-to-particle transfer coefficient. TOMAS-JAX: Fuller diffusivity with "
                 "diffusion volume 120 (air 20.1), mean free path 2D/c, Fuchs-Sutugin in the "
                 "Zaveri et al. (2014) form (0.283 coefficient). PyCHAM: Fuller with the "
                 "diffusion volume from the SMILES (220.4, air 19.7), mean free path 3D/c, "
                 "Fuchs-Sutugin in the Jacobson form (1.33 + 0.71/Kn).",
         "why": "Each model's own formulation; PyCHAM has no input for the diffusion volume.",
         "effect": f"Diffusivity {tr['d_ours']:.3g} vs {tr['d_pycham']:.3g} m2/s. Per particle "
                   f"TOMAS-JAX's coefficient is {tr['ratio_1000nm']:.2f}x PyCHAM's at 1 um, "
                   f"{tr['ratio_54nm']:.2f}x at 54 nm and {tr['ratio_1nm']:.2f}x at 1 nm, where it "
                   f"exceeds the gas-kinetic collision rate pi r^2 c N by "
                   f"{tr['ours_over_kinetic_1nm']:.2f}x (vbs-jax pairs TOMAS's mean free path "
                   "2D/c with a Fuchs-Sutugin form defined for 3D/c). Condensation sink at t = 0: "
                   f"{m['cs0'][0]:.3g} vs {m['cs0'][1]:.3g} 1/s ({m['cs0'][2]:.3g} with PyCHAM's "
                   f"formula on the TOMAS-JAX state). c* = 1 SOA {pct(m['soa_c1_1to3h_full'])} "
                   f"(largest difference at 1-3 h) with Kelvin on; with Kelvin off in both it is "
                   f"{pct(m['soa_c1_1to3h_kelvin_off'])} and with PyCHAM's coefficient in "
                   f"TOMAS-JAX as well {pct(m['soa_c1_1to3h_matched'])}."},
        {"what": "Raoult term: mass fraction over the organic phase (TOMAS-JAX) vs mole fraction "
                 "(PyCHAM). Made identical by giving POA and SURR the same SMILES (molar mass "
                 f"{S1['pycham_organic_mw']:.2f}) and excluding sulfate (seed_diss = 0) and water "
                 "(H2O_diss_wrtorg = 0) from PyCHAM's absorbing phase.",
         "why": "-", "effect": "None by construction (TOMAS-JAX's 1e-4 absorbing floor adds "
                               "1e-4 x seed dry mass, < 0.03 % of the POA)."},
        {"what": f"Molar mass: 200 g/mol in TOMAS-JAX, {S1['pycham_organic_mw']:.2f} g/mol in "
                 "PyCHAM (from the SMILES; PyCHAM has no molar-mass input).",
         "why": "PyCHAM derives molar masses from SMILES.",
         "effect": "Production and c* are converted with PyCHAM's molar mass, so the mass "
                   "production and the mass-based c* are identical; the Kelvin exponent and mean "
                   "molecular speed differ by 0.1 %."},
        {"what": "Particle water. TOMAS-JAX: ezwatereqm ammonium-bisulfate ratio, "
                 f"{setup['water_per_mass']['sulfate']:.4f} kg water per kg SO4 (x1.2 for "
                 f"NH4HSO4) and {setup['water_per_mass']['organic']:.4f} per kg of organics (A9), "
                 "no Kelvin term. PyCHAM: water partitions with Raoult, with per-component "
                 "dissociation constants w.r.t. water (core "
                 f"{setup['diss_wrtw_sulfate']:.4f}, POA/SURR {setup['diss_wrtw_organic']:.4f}) "
                 "that give TOMAS-JAX's water mass when Kelvin = 1, plus its own Kelvin term "
                 "(72 mN/m).",
         "why": "PyCHAM has no fixed hygroscopic-growth input; its water always feels Kelvin.",
         "effect": f"c* = 1: total particle water {pct(w0[1] / w0[0] - 1)} in PyCHAM at t = 0 "
                   f"({w0[1]:.3f} vs {w0[0]:.3f} ug/m3) and {pct(w12[1] / w12[0] - 1)} at 12 h "
                   f"({w12[1]:.3f} vs {w12[0]:.3f}); in the most populated bin at t = 0 (the "
                   f"50 nm seed mode, where PyCHAM's Kelvin term on water is large) water/dry "
                   f"is {wm0[1]:.3f} vs {wm0[0]:.3f} "
                   f"({pct(wm0[1] / wm0[0] - 1)}). c* = 100: TOMAS-JAX's wet number-mean diameter is "
                   f"{pct(m['d_wet_c100_12h'])} vs PyCHAM at 12 h. With Kelvin off in both, "
                   f"TOMAS-JAX water is {pct(m['water_c1_12h_kelvin_off'])} vs PyCHAM (c* = 1, "
                   "12 h). Water does not absorb "
                   "organics in either model, so it acts only through size."},
        {"what": "Density. TOMAS-JAX: Tang (1997) H2SO4-solution density of the wet particle with "
                 "organics counted as sulfate "
                 f"({setup['rho_wet_tomas_initial_g_cm3']:.3f} g/cm3 for the initial wet seed). "
                 "PyCHAM: volume-additive mixture with water fixed at 1.0 g/cm3 (a dens_Comp "
                 "entry for H2O is ignored). The dry components (core, POA, SURR) get an "
                 f"effective {setup['rho_dry_g_cm3']:.3f} g/cm3, chosen so that "
                 "m_dry/rho_d + m_w/1.0 equals TOMAS-JAX's Tang wet volume at t = 0 "
                 f"(water/dry {setup['water_over_dry_tomas_initial']:.4f}).",
         "why": "PyCHAM has no non-ideal (solution) density.",
         "effect": "Identical wet volume at t = 0. Later the composition changes: applying the two "
                   "density rules to the TOMAS-JAX particles at 12 h (c* = 1) gives wet densities "
                   f"that differ by up to {100 * m['density_max_rel_12h']:.1f} % "
                   f"({100 * m['density_max_rel_12h'] / 3:.1f} % in diameter)."},
        {"what": "Size structure. TOMAS-JAX: two-moment sectional on 40 fixed mass-doubling bins "
                 "(MNFIX after sources, nucleation and coagulation). PyCHAM: moving-centre on the "
                 "same bins (manual bounds in wet "
                 "radius), same number per bin at t = 0, bins with < "
                 f"{S1['pycham_min_bin_number_cm3']:g} cm-3 left empty "
                 f"({setup['n_total_ours_cm3'] - setup['n_total_pycham_cm3']:.1e} cm-3 in total).",
         "why": "Each model's own size scheme.",
         "effect": "Identical initial number and dry mass per bin (checked to 1e-9)."},
        {"what": "Growth across bins. TOMAS-JAX's SOA step changes each bin's particle mass in "
                 "place and then moves the particles between its fixed bins with a PPM remap in "
                 "log-mass (physics/ppm_remap.py, AerosolStepConfig.growth_bin_transfer = 'ppm': "
                 "each bin's PPM sub-bin profile, as in H2SO4 condensation, carried by the bin's "
                 "own displacement and deposited onto the bins), once per 60 s micro-step. "
                 "PyCHAM's moving-centre bins keep each population's exact size.",
         "why": "Size-structure numerics of each model (two-moment fixed sectional vs "
                "moving-centre).",
         "effect": "c* = 1, 12 h: number-mean dry diameter "
                   f"{pct(m['d_dry_c1_12h']['40'])} vs PyCHAM with 40 bins and "
                   f"{pct(m['d_dry_c1_12h']['160'])} with 160 bins in both, GSD "
                   f"{pct(m['gsd_c1_12h']['40'])} / {pct(m['gsd_c1_12h']['160'])}; PyCHAM itself "
                   f"changes by {pct(m['d_dry_c1_12h']['pycham_160_vs_40'])} from 40 to 160 bins. "
                   "The rest of the diameter difference is the transfer coefficient and Kelvin "
                   "term (attribution). The remap conserves number and mass. (The previous MNFIX "
                   "transfer, still selectable as growth_bin_transfer = 'mnfix', spread the "
                   "distribution: -12.6 % at 40 bins.)"},
        {"what": "Time stepping. TOMAS-JAX: source added every 60 s micro-step, then vbs-jax "
                 "adaptive analytic sub-steps (operator split), Kelvin/CS frozen per sub-step. "
                 "PyCHAM: continuous influx inside the BDF solve (rtol 1e-4), partitioning "
                 "coefficients and Kelvin refreshed every 60 s update step.",
         "why": "Each model's own numerics.",
         "effect": f"TOMAS-JAX micro-step 10 s vs 60 s: SOA within "
                   f"{100 * m['ours_numerics_soa_after_1h']:.1f} % after 1 h; PyCHAM update step "
                   f"10 s and rtol 1e-6 vs default: SOA within "
                   f"{100 * m['pycham_numerics_soa_after_1h']:.2f} % after 1 h."},
        {"what": "c* at the run temperature: TOMAS-JAX's c* is defined at 298.0 K (vbs-jax "
                 "T_REF) and corrected to 298.15 K with dHvap = 131 - 11 log10 c* kJ/mol "
                 f"(c* = 1 -> {setup['cstar_run_T_ugm3']:.4f}). PyCHAM gets volP from this "
                 f"corrected c* ({setup['volP_surr_pa']:.4e} Pa for c* = 1; volP is constant "
                 "in T).",
         "why": "PyCHAM's volP is a fixed pressure.", "effect": "None by construction."},
        {"what": "Chemistry. PyCHAM needs at least one reaction, so it gets the dummy "
                 "`<1> SURR = SURR : 0.0`; TOMAS-JAX has chemistry off. The surrogate is a "
                 f"continuous influx (Cinfl = {setup['cinfl_ppb_s']:.4e} ppb/s with TOMAS-JAX's "
                 "air number density).",
         "why": "PyCHAM input requirements.", "effect": "None (mass closure checked in both)."},
    ]
