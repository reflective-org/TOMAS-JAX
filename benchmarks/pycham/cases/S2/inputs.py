"""PyCHAM inputs for case S2 that mirror the TOMAS-JAX particle set-up.

Everything PyCHAM is told about particles is derived here from the TOMAS-JAX model
objects (never from TOMAS-JAX results):

* **Volatility** (``volatility_table``): every closed-shell organic that TOMAS-JAX lumps
  into a c* class gets the c* of its class centre (variant ``class``) or its own
  Li et al. (2016) c* (variant ``exact``), or keeps PyCHAM's own SMILES-based vapour
  pressure (variant ``native``, a PyCHAM-only volatility sensitivity), corrected to the
  run temperature exactly as
  vbs-jax does (Clausius-Clapeyron, dHvap = 131 - 11 log10 c*, T_ref = 298 K) and
  converted to Pa with PyCHAM's own molar mass, so that PyCHAM's equilibrium gas
  concentration in ug/m3 is that c*. Radicals, species above the top class edge and
  inorganics (incl. HNO3 and its sink product NA) get a vapour pressure far above
  ``ppartit_cutoff`` so PyCHAM never partitions them. H2SO4 (SA) and the seed organic
  are non-volatile, as in TOMAS-JAX.
* **Seed and size grid** (``particle_keys``): the TOMAS-JAX mass grid and seed. PyCHAM
  gets the per-bin number concentration and dry mass per particle of the TOMAS-JAX seed
  (manual bin bounds, ``space_mode = man``), 70 % SA + 30 % absorbing non-volatile
  organic by mass.
* **Water**: PyCHAM's per-component "dissociation constants with respect to water" are
  set so that its ideal-solution water (Kelvin factor 1) equals TOMAS-JAX's ezwatereqm
  water component by component: SA gets TOMAS-JAX's water per unit SO4 mass (bisulfate
  ratio x 1.2), the seed organic and every condensable organic the water per unit
  organic mass (``particles.water_mass_ratios``). Water is excluded from the organic
  Raoult denominator (``H2O_diss_wrtorg = 0``) and so is sulfate (``seed_diss`` = 0 for
  SA). PyCHAM keeps its own Kelvin term on water (a recorded difference).
* **Densities**: one density for every non-water component, chosen so that PyCHAM's
  additive wet volume (water at 1 g/cm3) equals the TOMAS-JAX wet volume (Tang 1997
  H2SO4-solution density with organics counted as sulfate, ASSUMPTIONS A9).
"""
from __future__ import annotations

import csv
import json
import subprocess
from pathlib import Path

import jax.numpy as jnp
import numpy as np
from vbs_jax.config import T_REF as VBS_T_REF
from vbs_jax.constants import R_GAS

from tomas_jax.box.config import AerosolStepConfig
from tomas_jax.box.coupling import mass_grid, seed_distribution
from tomas_jax.box.driver import _chemistry, _class_map
from tomas_jax.core.config import SRTH2O
from tomas_jax.gaschem.config import (DHVAP_A_KJ, DHVAP_B_KJ, VOLATILITY_CLASS_HALF_WIDTH,
                                      VOLATILITY_LOG10_CSTAR_CENTERS)
from tomas_jax.gaschem.volatility import is_closed_shell, log10_cstar_li2016
from tomas_jax.core.config import SRTORG_NUC, SRTSO4
from tomas_jax.physics.condensation_sink import calc_wet_diameters
from tomas_jax.physics.water_equilibrium import equilibrate_water

from ...config import PYCHAM_PYTHON
from ...particles import (PYCHAM_WATER_DENSITY, effective_dry_density, pycham_diss_wrtw,
                          water_mass_ratios)
from .case import S2, VOLATILITY_CSV

SEED_N_MIN_CM3 = 1.0e-6         # bins of the TOMAS-JAX seed below this are left out of PyCHAM
N_EXTRA_TOP_BINS = 4            # empty bins above the seed for growth (x2 in diameter = 3 bins)
MW_TIMEOUT_S = 120.0


def pycham_molar_masses(xml: Path, out_json: Path) -> dict[str, float]:
    """PyCHAM's molar mass of every xml species (pybel, in the PyCHAM venv)."""
    proc = subprocess.run([str(PYCHAM_PYTHON), "-I", str(Path(__file__).with_name("pycham_mw.py")),
                           str(xml), str(out_json)], capture_output=True, text=True,
                          timeout=MW_TIMEOUT_S)
    if proc.returncode != 0:
        raise RuntimeError(f"pycham_mw.py failed ({proc.returncode}): {proc.stderr[-2000:]}")
    return {k: v["mw"] for k, v in json.loads(Path(out_json).read_text()).items()}


def cstar_at_temp(log10_cstar_298: np.ndarray, temp_k: float) -> np.ndarray:
    """c* [ug/m3] at ``temp_k`` as vbs-jax computes it (calc_cstar_T) with the tutorial
    dHvap rule (gaschem/config.py: dHvap = 131 - 11 log10 c*)."""
    lc = np.asarray(log10_cstar_298, float)
    dhvap_j = (DHVAP_A_KJ - DHVAP_B_KJ * lc) * 1e3
    return 10.0 ** lc * np.exp(-(dhvap_j / R_GAS) * (1.0 / temp_k - 1.0 / VBS_T_REF))


def cstar_to_pa(cstar_ugm3: np.ndarray, mw_g_mol: np.ndarray, temp_k: float) -> np.ndarray:
    """vbs_jax.config.cstar_ug_to_Pa; PyCHAM converts Pa back with the same R (scipy)."""
    return np.asarray(cstar_ugm3) * 1e-9 / (np.asarray(mw_g_mol) * 1e-3) * R_GAS * temp_k


def pycham_species() -> list[str]:
    """Components PyCHAM will have from the S2 .eqn (reaction species + PROD + SEEDOA).

    H2O and 'core' are added by PyCHAM itself and are never overridden here.
    """
    parsed = _chemistry()[0]
    names = [s for s in parsed.species if s != "NA_HET"]     # NA_HET: TOMAS-JAX-only counter
    return names + ["PROD", S2["seed_oa_name"]]


def volatility_table(variant: str, mw_pycham: dict[str, float], temp_k: float) -> list[dict]:
    """One row per PyCHAM component: category, c* used, vapour pressure given [Pa]."""
    known = S2["pycham_variants"] + S2["pycham_attribution_variants"] + S2["pycham_volatility_variants"]
    if variant not in known:
        raise ValueError(f"unknown variant {variant!r}")
    # "massfrac*": class-centre c*, converted with one common MW (attribution runs)
    mw_ref = mw_pycham[S2["seed_oa_name"]] if variant.startswith("massfrac") else None
    native = variant in S2["pycham_volatility_variants"]
    parsed = _chemistry()[0]
    cmap = _class_map(None)
    cls_of = {s: int(c) for s, c in zip(cmap.species, cmap.class_idx)}
    own_of = {s: float(v) for s, v in zip(cmap.species, cmap.log10_cstar)}
    centers = np.asarray(VOLATILITY_LOG10_CSTAR_CENTERS)
    top_edge = centers[-1] + VOLATILITY_CLASS_HALF_WIDTH
    rows = []
    for name in pycham_species():
        mw = mw_pycham[name]
        row = {"species": name, "mw_pycham": mw,
               "mw_ours": parsed.molecular_weight(name) if name in parsed.atoms else np.nan,
               "class": -1, "log10_cstar_own_298": np.nan, "log10_cstar_used_298": np.nan,
               "cstar_used_ugm3_at_T": np.nan}
        atoms = parsed.atoms.get(name, {})
        if name in cls_of:
            k = cls_of[name]
            lc = own_of[name] if variant == "exact" else centers[k]
            cst = float(cstar_at_temp(lc, temp_k))
            row.update(category="condensable", **{"class": k}, log10_cstar_own_298=own_of[name],
                       log10_cstar_used_298=float(lc), cstar_used_ugm3_at_T=cst,
                       vp_pa=float(cstar_to_pa(cst, mw if mw_ref is None else mw_ref, temp_k)))
            if native:              # no override: PyCHAM's own vapour pressure
                row.update(log10_cstar_used_298=np.nan, cstar_used_ugm3_at_T=np.nan,
                           vp_pa=np.nan)
        elif name in ("SA", S2["seed_oa_name"]):
            row.update(category="nonvolatile" if name == "SA" else "seed_oa",
                       vp_pa=S2["nonvolatile_vp_pa"])
        else:
            if not atoms.get("C"):
                cat = "gas_only_inorganic"
            elif not is_closed_shell(atoms):
                cat = "gas_only_radical"
            else:
                lc = log10_cstar_li2016(atoms)
                if lc <= top_edge:
                    raise RuntimeError(f"{name}: closed-shell organic below the top edge but "
                                       "not in the TOMAS-JAX class map")
                cat = "gas_only_volatile"
                row["log10_cstar_own_298"] = lc
            row.update(category=cat, vp_pa=S2["gas_only_vp_pa"])
        rows.append(row)
    cond = [r for r in rows if r["category"] == "condensable"]
    if len(cond) != len(cmap.species):
        raise RuntimeError(f"{len(cond)} condensable rows vs {len(cmap.species)} in the class map")
    worst = -np.inf if native else max(r["vp_pa"] for r in cond)
    if worst >= S2["ppartit_cutoff_pa"]:
        raise RuntimeError(f"a condensable species has vp {worst:g} Pa >= the cutoff")
    return rows


def write_volatility_csv(rows_by_variant: dict[str, list[dict]], path: Path = VOLATILITY_CSV) -> Path:
    keys = ("species", "category", "class", "mw_pycham", "mw_ours", "log10_cstar_own_298")
    variants = list(rows_by_variant)
    base = rows_by_variant[variants[0]]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(list(keys) + [f"{c}_{v}" for v in variants
                                 for c in ("log10_cstar_used_298", "cstar_used_ugm3_at_T", "vp_pa")])
        for k, r in enumerate(base):
            line = [r[c] if not isinstance(r[c], float) else f"{r[c]:.6g}" for c in keys]
            for v in variants:
                rv = rows_by_variant[v][k]
                line += [f"{rv['log10_cstar_used_298']:.6g}", f"{rv['cstar_used_ugm3_at_T']:.6g}",
                         f"{rv['vp_pa']:.6e}"]
            w.writerow(line)
    return Path(path)


def seed_state(scenario) -> dict:
    """TOMAS-JAX t = 0 aerosol (driver.run initial state), its per-component water and the
    effective dry density PyCHAM needs for the same wet volume."""
    cfg = AerosolStepConfig()
    xk = mass_grid(scenario.numerics.nbins)
    nk, mk = seed_distribution(scenario.seed.modes, scenario.seed.organic_mass_fraction, xk,
                               cfg.boxvol_cm3)
    env = scenario.environment
    mkw = np.asarray(equilibrate_water(jnp.asarray(mk), env.rh, env.temp_k, cfg.water_scheme))
    dry = mkw[:, :SRTH2O].sum(axis=1)
    _, rho = calc_wet_diameters(jnp.asarray(nk), jnp.asarray(mkw), jnp.asarray(xk))
    populated = nk > 1e10                                   # NEPS_CONDSINK [#/cell]
    ratio = mkw[populated, SRTH2O] / dry[populated]
    rho_wet = np.asarray(rho)[populated]
    if np.ptp(ratio) > 1e-10 * ratio.mean() or np.ptp(rho_wet) > 1e-8 * rho_wet.mean():
        raise RuntimeError("TOMAS-JAX water ratio / wet density not uniform over the seed bins")
    f, rho_w = float(ratio.mean()), float(rho_wet.mean())
    # per-component water (ezwatereqm): must reproduce the seed's water exactly
    w = water_mass_ratios(scenario)
    so4, org = mkw[populated, SRTSO4].sum(), mkw[populated, SRTORG_NUC].sum()
    f_parts = (w["sulfate"] * so4 + w["organic"] * org) / dry[populated].sum()
    if abs(f_parts / f - 1.0) > 1e-9:
        raise RuntimeError(f"per-component water {f_parts} != TOMAS-JAX seed water {f}")
    return {"xk": xk, "n_cm3": nk / cfg.boxvol_cm3, "m_mid_kg": np.sqrt(xk[:-1] * xk[1:]),
            "water_to_dry": f, "water_per_mass": w, "rho_wet_tomas": rho_w,
            "rho_dry_pycham": effective_dry_density(rho_w, f),
            "dry_mass_ugm3": float(dry.sum() * 1e15 / cfg.boxvol_cm3)}


def _r_um(volume_m3: np.ndarray) -> np.ndarray:
    return (3.0 * np.asarray(volume_m3) / (4.0 * np.pi)) ** (1.0 / 3.0) * 1e6


def particle_keys(scenario, mw_pycham: dict[str, float]) -> tuple[dict[str, str], dict]:
    """PyCHAM particle model variables + a record of the derived numbers."""
    seed = seed_state(scenario)
    n0, xk = seed["n_cm3"], seed["xk"]
    inside = np.nonzero(n0 >= SEED_N_MIN_CM3)[0]
    k_lo, k_top = int(inside[0]), int(inside[-1])
    k_hi = min(n0.size - 1, k_top + N_EXTRA_TOP_BINS)
    bins = np.arange(k_lo, k_hi + 1)
    f, rho_d = seed["water_to_dry"], seed["rho_dry_pycham"]
    # bin bounds in wet volume (PyCHAM moves particles by their total volume): a
    # particle at a TOMAS-JAX dry-mass edge with the TOMAS-JAX water ratio
    v_edges = xk[k_lo:k_hi + 2] * (1.0 / rho_d + f / PYCHAM_WATER_DENSITY)
    r_bounds = _r_um(v_edges)
    r_dry_mid = _r_um(seed["m_mid_kg"][bins] / rho_d)        # PyCHAM: Varr = dry volume
    pconc = np.where(n0[bins] >= SEED_N_MIN_CM3, n0[bins], 0.0)
    ofrac = scenario.seed.organic_mass_fraction
    oa, sa = S2["seed_oa_name"], "SA"
    x_sa = (1.0 - ofrac) / mw_pycham[sa]
    x_oa = ofrac / mw_pycham[oa]
    x_sa, x_oa = x_sa / (x_sa + x_oa), x_oa / (x_sa + x_oa)
    env = scenario.environment
    w = seed["water_per_mass"]

    def d_wrtw(name):         # TOMAS-JAX water per unit mass: SO4 ratio for SA, organic otherwise
        per_mass = w["sulfate"] if name == sa else w["organic"]
        return pycham_diss_wrtw(per_mass, mw_pycham[name], env.rh)

    cmap = _class_map(None)
    cond = list(cmap.species)
    dens_names = [sa, oa] + cond
    keys = {
        "number_size_bins": str(bins.size),
        "size_structure": "0",                       # moving centre (PyCHAM default)
        "space_mode": "man",
        "lower_part_size": ", ".join(f"{r:.10e}" for r in r_bounds),
        "upper_part_size": f"{r_bounds[-1]:.10e}",
        "pconc": ", ".join(f"{n:.10e}" for n in pconc),
        "pconct": "0.",
        "mean_rad": " : ".join(f"{r:.10e}" for r in r_dry_mid),
        "std": "1.6",                                # unused (per-bin number given)
        "seed_name": f"{sa}, {oa}",
        "seedx": f"{x_sa:.12f} ; {x_oa:.12f}",
        "seed_diss": "0., 1.",                       # sulfate does not dissolve organics
        "seed_diss_wrtw": f"{d_wrtw(sa):.10g}, {d_wrtw(oa):.10g}",
        "indiv_comp_diss_wrtw_name": ", ".join(cond),
        "indiv_comp_diss_wrtw": ", ".join(f"{d_wrtw(s):.10g}" for s in cond),
        "H2O_diss_wrtorg": "0.",                     # water does not dissolve organics
        "Vwat_inc": "0",                             # given sizes are dry
        "seed_eq_wat": "1",                          # water equilibrated before t = 0
        "nucv1": "0.",
        "dens_Comp": ", ".join(dens_names),
        # kg/m3, NOT the g/cm3 the PyCHAM README states: v5.8.0 prop_calc.py stores
        # ``dens`` unconverted into y_dens [kg/m3] (g/cm3 input gives particle densities
        # of ~2 kg/m3 and an overflowing Kelvin term). Checked after the run via MV.npy.
        "dens": ", ".join([f"{rho_d:.10g}"] * len(dens_names)),
    }
    record = {k: v for k, v in seed.items() if k not in ("xk", "n_cm3", "m_mid_kg")}
    record.update({
        "tomas_bins_in_pycham": [int(k_lo), int(k_hi)], "n_pycham_bins": int(bins.size),
        "seed_n_min_cm3": SEED_N_MIN_CM3,
        "seed_number_left_out_cm3": float(n0.sum() - pconc.sum()),
        "seed_number_cm3": float(n0.sum()),
        "seed_mole_fractions": {sa: x_sa, oa: x_oa},
        "water_diss_wrtw": {sa: d_wrtw(sa), oa: d_wrtw(oa),
                            "condensable_min": min(d_wrtw(s) for s in cond),
                            "condensable_max": max(d_wrtw(s) for s in cond)},
        "pycham_radius_bounds_um": r_bounds.tolist(),
        "pycham_dry_radius_mid_um": r_dry_mid.tolist(),
        "pycham_pconc_cm3": pconc.tolist(),
    })
    return keys, record


def volatility_keys(rows: list[dict]) -> dict[str, str]:
    """vol_Comp / volP for every row with a vapour pressure (NaN = PyCHAM's own)."""
    given = [r for r in rows if np.isfinite(r["vp_pa"])]
    return {"vol_Comp": ", ".join(r["species"] for r in given),
            "volP": ", ".join(f"{r['vp_pa']:.10e}" for r in given),
            "ppartit_cutoff": f"{S2['ppartit_cutoff_pa']:g}"}
