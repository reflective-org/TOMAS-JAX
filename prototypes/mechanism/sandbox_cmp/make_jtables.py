"""Shared photolysis for the MCM vs SANDBOX comparison: one TUV-x J(t) table per case.

Both models read the same table, so photolysis differences reduce to which channels
each mechanism has (not to two radiation codes). J comes from the TUV-x JAX port
(tuvx-jax e5d9032, config examples/tuv_5_4_no_aerosol.json, branching=True for the
HNO4 product channels) at the box altitude z, where z is the US Standard Atmosphere
height of the box pressure (the same rule SANDBOX's tuvx_photolysis_adapter uses):

    J_i(t) = J_i^TUV-x(SZA(t), r, z)

with SZA from tuvx_photolysis.geometry for the case latitude/longitude/UTC time and r
the Earth-Sun distance [AU] on the start date. J is tabulated on a 0.25 deg SZA grid
and linearly interpolated; J = 0 for SZA >= 90 deg. Output every 150 s.

Also resolves each case (air number density M, H2O, altitude) into cases_resolved.json
so the SANDBOX venv script uses identical numbers.

Run: TUVX_JAX_ROOT=<tuvx-jax> PYTHONPATH=<mech-wt> <mech-wt>/.venv/bin/python make_jtables.py
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
from tuvx_photolysis import PhotolysisCalculator, data, geometry

from tomas_jax.gaschem.config import TUVX_CONFIG
from tomas_jax.gaschem.environment import air_environment
from tomas_jax.gaschem.photolysis import TUVX_DIRECT, tuvx_root

HERE = Path(__file__).parent
OUT = HERE / "results"
DT_TABLE = 150.0                      # [s]
SZA_GRID = np.arange(0.0, 90.0, 0.25)  # [deg]

# SANDBOX photolysis equation -> TUV-x key (copied from SANDBOX
# stratchem-jax/tuvx_photolysis_adapter.py REACTION_MAP, halogen-free entries only;
# halogen species are zero in every run, so their J do not matter).
SANDBOX_MAP = {
    "HNO3 -> OH + NO2": "HNO3+hv->OH+NO2",
    "NO3 -> NO2 + O": "NO3+hv->NO2+O(3P)",
    "NO3 -> NO + O2": "NO3+hv->NO+O2",
    "NO2 -> NO + O": "NO2+hv->NO+O(3P)",
    "N2O5 -> NO2 + NO3": "N2O5+hv->NO2+NO3",
    "HNO4 -> NO2 + HO2": "HNO4+hv->HO2+NO2",
    "HNO4 -> NO3 + OH": "HNO4+hv->OH+NO3",
    "HONO -> OH + NO": "HNO2+hv->OH+NO",
    "H2O2 -> 2 OH": "H2O2+hv->OH+OH",
    "O3 -> O2 + O1D": "O3+hv->O2+O(1D)",
}
MCM_J = ["J_CH3NO3", "J_CH3OOH", "J_H2O2", "J_HCHO_H", "J_HCHO_H2", "J_HNO3", "J_HONO",
         "J_NO2", "J_NO3_NO", "J_NO3_NO2", "J_O3_O1D", "J_O3_O3P"]


def main() -> None:
    cases = json.loads((HERE / "cases.json").read_text())
    root = tuvx_root()
    calc = PhotolysisCalculator.from_tuvx_json(root / TUVX_CONFIG, data_root=root)
    names = set(calc.reaction_names(branching=True))
    keys = sorted({TUVX_DIRECT[j] for j in MCM_J} | set(SANDBOX_MAP.values()))
    missing = [k for k in keys if k not in names]
    if missing:
        raise SystemExit(f"TUV-x config lacks {missing}")
    dens = data.load_profile_csv(root / "data/profiles/atmosphere/ussa.dens")
    temp = data.load_profile_csv(root / "data/profiles/atmosphere/ussa.temp")
    resolved = {}
    for cname, c in cases.items():
        if cname.startswith("_"):
            continue
        t0 = time.perf_counter()
        start = datetime.fromisoformat(c["start_utc"]).replace(tzinfo=timezone.utc)
        z = geometry.pressure_to_altitude_km(c["P_hPa"], dens, temp)
        esd = geometry.earth_sun_distance(start.year, start.month, start.day, 0.0)
        table = {k: np.zeros(SZA_GRID.size) for k in keys}
        for i, sza in enumerate(SZA_GRID):
            prof = calc.rate_constants_profile(float(sza), esd, branching=True)
            for k in keys:
                table[k][i] = float(np.interp(z, calc.height_edges_km, prof[k]))
        n = int(round(c["days"] * 86400 / DT_TABLE)) + 1
        t = np.arange(n) * DT_TABLE
        sza_t = np.empty(n)
        for i, ts in enumerate(t):
            now = start + timedelta(seconds=float(ts))
            hour = now.hour + now.minute / 60 + now.second / 3600
            sza_t[i] = geometry.solar_zenith_angle(now.year, now.month, now.day, hour,
                                                   c["lat"], c["lon"])
        j = {k: np.where(sza_t < 90.0, np.interp(np.minimum(sza_t, 89.75), SZA_GRID, table[k]),
                         0.0).tolist() for k in keys}
        # air and water: tomas_jax rule (ideal gas, Buck saturation) for both models
        if c["RH"] is not None:
            air = air_environment(c["T"], c["P_hPa"] * 100.0, c["RH"])
        else:
            air = air_environment(c["T"], c["P_hPa"] * 100.0, 0.0)
            air["H2O"] = c["H2O_ppm"] * 1e-6 * air["M"]
        resolved[cname] = {**c, "M": air["M"], "O2": air["O2"], "N2": air["N2"],
                           "H2O": air["H2O"], "H2O_ppm": air["H2O"] / air["M"] * 1e6,
                           "altitude_km": z, "earth_sun_distance_au": esd}
        (OUT / f"jtable_{cname}.json").write_text(json.dumps(
            {"case": cname, "dt_s": DT_TABLE, "t_s": t.tolist(), "sza_deg": sza_t.tolist(),
             "altitude_km": z, "tuvx_config": TUVX_CONFIG, "J": j,
             "mcm_map": {m: TUVX_DIRECT[m] for m in MCM_J}, "sandbox_map": SANDBOX_MAP}))
        noon = int(np.argmin(sza_t[: int(86400 / DT_TABLE)]))
        print(f"{cname}: z = {z:.2f} km, min SZA {sza_t[noon]:.1f} deg, "
              f"J_NO2 noon {j['NO2+hv->NO+O(3P)'][noon]:.3e}, J_O1D noon "
              f"{j['O3+hv->O2+O(1D)'][noon]:.3e} s-1, H2O {resolved[cname]['H2O_ppm']:.0f} ppm "
              f"({time.perf_counter() - t0:.1f} s)")
    (HERE / "cases_resolved.json").write_text(json.dumps(resolved, indent=1))


if __name__ == "__main__":
    OUT.mkdir(exist_ok=True)
    main()
