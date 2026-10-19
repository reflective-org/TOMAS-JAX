"""PyCHAM inputs and output reduction for case S1 (Level 1 surrogate, no chemistry).

* ``write_scheme`` / ``write_xml``: a one-reaction dummy scheme and an xml with SMILES for
  the surrogate (SURR), the non-volatile seed organic (POA) and water.
* ``model_var``: the model-variables dict (harness ``scenario_model_var`` + particle keys
  from ``case.pycham_setup``).
* ``reduce_output``: PyCHAM output folder -> small dict of arrays (ug/m3, #/cm3, nm).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from tomas_jax.box.scenario import Scenario

from ... import pycham_inputs
from ...load_pycham import load_pycham
from ...particles import dry_diameter_nm
from . import case
from .case import S1

DENSITY_UNIT_FACTOR = 1.0e3   # PyCHAM stores dens_Comp/dens as given into y_dens [kg/m3]
AVOGADRO = 6.02214076e23


def write_scheme(dst: Path) -> Path:
    """Dummy chemistry: PyCHAM needs at least one reaction (rate 0, no effect)."""
    sur = S1["surrogate_name"]
    Path(dst).write_text(
        "// S1 benchmark: no gas-phase chemistry; the surrogate only partitions\n"
        f"<1> {sur} = {sur} : 0.0 ;\n")
    return Path(dst)


def write_xml(dst: Path) -> Path:
    smiles = S1["organic_smiles"]
    species = {S1["surrogate_name"]: smiles, S1["poa_name"]: smiles, "H2O": "O"}
    body = "".join(f'<species species_number="s{k + 1}" species_name="{name}">\n'
                   f"<smiles>{smi}</smiles>\n</species>\n"
                   for k, (name, smi) in enumerate(species.items()))
    Path(dst).write_text('<?xml version="1.0" encoding="UTF-8"?>\n'
                         '<mechanism xmlns="http://mcm.leeds.ac.uk/MCM/mechanism.xsd">\n'
                         f"<species_defs>\n{body}</species_defs>\n</mechanism>\n")
    return Path(dst)


def _fmt(values, sep: str) -> str:
    return sep.join(f"{float(v):.10g}" for v in values)


def model_var(s: Scenario, *, res_file: Path, scheme: Path, xml: Path, constants: Path,
              surface_tension_off: bool = False, update_step_s: float | None = None,
              int_tol: tuple[float, float] | None = None,
              total_time_s: float | None = None) -> dict[str, str]:
    """Model variables for one S1 run (see case.pycham_setup for every choice)."""
    p = case.pycham_setup(s)
    sur, poa = S1["surrogate_name"], S1["poa_name"]
    rho_si = p["rho_dry_g_cm3"] * DENSITY_UNIT_FACTOR
    nb = p["n_bin_cm3"].size
    particles = {
        "number_size_bins": str(nb),
        "size_structure": "0",                         # moving-centre
        "space_mode": "man",
        "lower_part_size": _fmt(p["edges_wet_um"], ", "),
        "upper_part_size": f"{p['edges_wet_um'][-1]:.10g}",
        "mean_rad": _fmt(p["r_dry_um"], " : "),        # dry radius of each bin's seed [um]
        "pconc": _fmt(p["n_bin_cm3"], ", "),
        "pconct": "0.",
        "seed_name": f"core, {poa}",
        "seed_mw": f"{S1['sulfate_mw']}",
        "seed_dens": f"{p['rho_dry_g_cm3']:.10g}",
        "seedx": _fmt(p["seedx_core_poa"], " ; "),
        "Vwat_inc": "0",                               # sizes above are dry
        "seed_eq_wat": "1",                            # equilibrate water before t = 0
        # Raoult term for organics: sulfate and water do not absorb (as TOMAS-JAX)
        "seed_diss": "0., 1.",
        "H2O_diss_wrtorg": "0.",
        # water uptake: TOMAS-JAX's ezwatereqm water mass (when Kelvin = 1)
        "seed_diss_wrtw": f"{p['diss_wrtw_sulfate']:.10g}, {p['diss_wrtw_organic']:.10g}",
        "nonseed_diss_wrtw": f"{p['diss_wrtw_organic']:.10g}",
        "dens_Comp": f"{sur}, {poa}",
        "dens": _fmt([rho_si] * 2, ", "),
        "vol_Comp": f"{sur}, {poa}",
        "volP": f"{p['volP_surr_pa']:.10g}, 0.",
        "nucv1": "0.",
        "cont_infl": sur,
        "Cinfl": f"{p['cinfl_ppb_s']:.10g}",
        "cont_infl_t": "0.",
    }
    mv = pycham_inputs.scenario_model_var(s, res_file=res_file, eqn=scheme, xml=xml,
                                          constants=constants, j_xlsx=None,
                                          particles=particles)
    mv["update_step"] = f"{update_step_s or S1['pycham_update_step_s']:.1f}"
    mv["recording_time_step"] = f"{S1['pycham_recording_step_s']:.1f}"
    if int_tol is not None:
        mv["int_tol"] = f"{int_tol[0]:g}, {int_tol[1]:g}"
    if total_time_s is not None:
        mv["total_model_time"] = f"{total_time_s:.1f}"
    mv.pop("Comp0", None)
    mv.pop("C0", None)
    return mv


def reduce_output(out_dir: Path, s: Scenario) -> dict:
    """PyCHAM output -> arrays for the comparison (ug/m3, #/cm3, nm, all at 60 s)."""
    r = load_pycham(str(out_dir))
    names = r["names"]
    sur, poa = names.index(S1["surrogate_name"]), names.index(S1["poa_name"])
    h2o, core = names.index("H2O"), names.index("core")
    mw = r["mw_g_mol"]
    molar_vol = r["mv_cm3_mol"]
    part = r["part_ug_m3"]                                  # [t, bin, comp]
    gas_ugm3 = r["gas_molec_cm3"][:, sur] / AVOGADRO * mw[sur] * 1e12
    n = r["N_bin"]                                          # number of each (moving-centre) bin
    dry = part[:, :, [sur, poa, core]].sum(axis=2)          # ug/m3 per bin
    # wet volume from the composition (size_bin_radius at t = 0 is the dry input radius)
    vol = (r["part_molec_cm3"] / AVOGADRO * molar_vol[None, None, :]).sum(axis=2)  # cm3/cm3
    with np.errstate(divide="ignore", invalid="ignore"):
        m_dry_pp = np.where(n > 0, dry / n, np.nan) * 1e-12      # g per particle
        v_wet_pp = np.where(n > 0, vol / n, np.nan)              # cm3 per particle
    d_dry_nm = dry_diameter_nm(m_dry_pp * 1e-3)
    d_wet_nm = (6.0 * v_wet_pp / np.pi) ** (1.0 / 3.0) * 1e7
    return {
        "t_s": r["t_s"],
        "gas_surr_ugm3": gas_ugm3,
        "soa_ugm3": part[:, :, sur].sum(axis=1),
        "poa_ugm3": part[:, :, poa].sum(axis=1),
        "sulfate_ugm3": part[:, :, core].sum(axis=1),
        "water_ugm3": part[:, :, h2o].sum(axis=1),
        "n_bin_cm3": n,
        "dry_ugm3_bin": dry,
        "soa_ugm3_bin": part[:, :, sur],
        "poa_ugm3_bin": part[:, :, poa],
        "water_ugm3_bin": part[:, :, h2o],
        "d_dry_nm_bin": d_dry_nm,
        "d_wet_nm_bin": d_wet_nm,
        "d_wet_output_nm_bin": 2.0 * r["rad_um"] * 1e3,     # PyCHAM size_bin_radius
        "molar_volume_cm3_mol": molar_vol,
        "mw_g_mol": mw,
        "names": np.array(names),
        "psat_298_pa": r["psat_298_Pa"],
        "ppb_to_molec_cm3": r["ppb_to_molec_cm3"],
        "env": r["env"],
    }
