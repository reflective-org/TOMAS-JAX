"""Output contract of ``box.driver.run`` (JSON-ready via :meth:`RunResult.to_dict`).

Time axes are hours since the scenario start; every ``[t]`` list has one entry per
output time, every ``[t][bin]`` list one row per output time.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np

# Gas species reported as time series. Mixing ratios in ppb, radicals in molec/cm3.
REPORTED_GASES_PPB = ("LIMONENE", "O3", "NO", "NO2", "HNO3", "N2O5", "HCHO", "CO", "SO2", "H2O2")
REPORTED_RADICALS_CM3 = ("OH", "HO2", "NO3")
REPORTED_J = ("J_NO2", "J_O3_O1D", "J_HCHO_H", "J_NO3_NO2")
N_TOP_CONTRIBUTORS = 15


@dataclass
class RunResult:
    scenario: dict                                   # echo of the Scenario (JSON-ready)
    time_h: list[float]
    diameter_nm: list[float]                         # dry bin-centre diameters (rho = 1400 kg/m3)
    diameter_edges_nm: list[float]
    dndlog10dp_cm3: list[list[float]]                # [t][bin] dN/dlog10(Dp)
    dmdlog10dp_ugm3: list[list[float]]               # [t][bin] dry mass distribution
    totals: dict[str, list[float]]                   # [t]: N_cm3, SO4/NIT/SOA/POA/NH4/H2O/dry_ugm3,
                                                     #      S_um2cm3, V_um3cm3 (dry, rho 1400)
    dsdlog10dp_um2cm3: list[list[float]] = field(default_factory=list)  # [t][bin] pi Dp^2 dN/dlogDp, Dp = bin mean-mass diameter
    dvdlog10dp_um3cm3: list[list[float]] = field(default_factory=list)  # [t][bin] (dM/dlogDp) / 1.4 g/cm3
    # [t]: o_to_c (particle organics), kappa, kappa_organic, kappa_inorganic,
    # organic_volume_fraction (box/properties.py; NaN where there is no such mass)
    properties: dict[str, list[float]] = field(default_factory=dict)
    # cumulative amounts removed since the start, [t] (box/losses.py; all zero when no
    # loss is on): wall_vapour_ugm3 (condensable organic vapour to walls), o3_surface_ppb,
    # ventilated_vapour_ugm3 (net out, condensable organic vapour), ventilated_gases_ppb
    # {species: net out}, deposited_{dry,SOA,POA,inorganic}_ugm3, deposited_N_cm3,
    # ventilated_{dry,SOA,POA,inorganic}_ugm3 (net out), ventilated_N_cm3 (net out),
    # emitted_ppb {species}
    losses: dict = field(default_factory=dict)
    gases_ppb: dict[str, list[float]] = field(default_factory=dict)
    radicals_cm3: dict[str, list[float]] = field(default_factory=dict)
    h2so4_cm3: list[float] = field(default_factory=list)
    photolysis: dict[str, list[float]] = field(default_factory=dict)  # J [1/s] and "sza_deg"
    volatility: dict = field(default_factory=dict)   # log10_cstar [K], gas_ugm3/particle_ugm3 [t][K]
    top_contributors: list[dict] = field(default_factory=list)       # final particle-phase species
    diagnostics: dict = field(default_factory=dict)  # cs_per_s [t], warnings, c* source, timings

    def to_dict(self) -> dict:
        return _jsonable(dataclasses.asdict(self))


def _jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _jsonable(obj.tolist())
    if isinstance(obj, (np.floating, np.integer)):
        return _jsonable(obj.item())
    if isinstance(obj, float) and not np.isfinite(obj):
        return None
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj
