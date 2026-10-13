"""Photolysis frequencies for the MCM mechanism (the 28 ``J_*`` names it uses).

Modes:

* ``"tuvx"`` — TUV-x (``tuvx_photolysis``, the JAX/Python port) at the surface for the
  site, date and UTC time. Surface J depends on the solar zenith angle (SZA) and the
  Earth-Sun distance only (TUV-x uses fixed standard-atmosphere O3/T profiles), so a
  J(SZA) table is computed once per process on a 1-degree grid and interpolated.
  Channels TUV-x lumps are split with MCM branching ratios at the same SZA; J's with
  no TUV-x recipe use the MCM clear-sky parameterization at the same SZA (see
  ``SOURCES``). The table is scaled by the Earth-Sun distance r [AU]
  exactly as TUV-x does (``apply_scale_factor(earth_sun_distance)`` multiplies the
  radiation field by r, not 1/r^2 — a TUV-x convention we reproduce; ASSUMPTIONS A11).
* ``"mcm"`` — MCM clear-sky parameterization for every J (Saunders et al., 2003).
* ``"constant"`` — sun fixed at ``constant_sza_deg``; the TUV-x spectral shape is
  rescaled so that J_NO2 equals ``jno2`` (chamber-like constant light).
* ``"off"`` — all zero (dark).

The solar geometry is the TUV-x one (``tuvx_photolysis.geometry``) in every mode.
"""
from __future__ import annotations

import functools
import os
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

from .config import TUVX_CONFIG, TUVX_SZA_GRID_DEG
from .mcm_rates import mcm_photolysis

# MCM J name -> TUV-x reaction key (one-to-one).
TUVX_DIRECT = {
    "J_O3_O1D": "O3+hv->O2+O(1D)",
    "J_O3_O3P": "O3+hv->O2+O(3P)",
    "J_H2O2": "H2O2+hv->OH+OH",
    "J_NO2": "NO2+hv->NO+O(3P)",
    "J_NO3_NO": "NO3+hv->NO+O2",
    "J_NO3_NO2": "NO3+hv->NO2+O(3P)",
    "J_HONO": "HNO2+hv->OH+NO",
    "J_HNO3": "HNO3+hv->OH+NO2",
    "J_HCHO_H": "CH2O+hv->H+HCO",
    "J_HCHO_H2": "CH2O+hv->H2+CO",
    "J_CH3NO3": "CH3ONO2+hv->CH3O+NO2",
    "J_CH3OOH": "CH3OOH+hv->CH3O+OH",
    "J_CH3COCH3": "CH3COCH3+hv->CH3CO+CH3",
    "J_MEK": "CH3COCH2CH3+hv->CH3CO+CH2CH3",
    "J_GLYOX_H2": "CHOCHO+hv->H2+CO+CO",
    "J_GLYOX_HCHO": "CHOCHO+hv->CH2O+CO",
    "J_GLYOX_HCO": "CHOCHO+hv->HCO+HCO",
    "J_MGLYOX": "CH3COCHO+hv->CH3CO+HCO",
    "J_BIACET": "CH3COCOCH3+hv->Products",
    "J_NOA": "nitro_acetone+hv->Products",
    "J_IC3H7NO3": "CH3CHONO2CH3+hv->CH3CHOCH3+NO2",
    "J_NC3H7NO3": "nC3H7ONO2+hv->C3H7O+NO2",
}
# MCM channel J name -> (TUV-x total key, MCM channels the total is split over).
TUVX_SPLIT = {
    "J_MACR_HCO": ("CH2=C(CH3)CHO+hv->Products", ("J_MACR_HCO", "J_MACR_H")),
    "J_MACR_H": ("CH2=C(CH3)CHO+hv->Products", ("J_MACR_HCO", "J_MACR_H")),
    "J_MVK_C2H3": ("MVK+hv->Products", ("J_MVK_CO", "J_MVK_C2H3")),
}
# No TUV-x recipe exists: MCM clear-sky parameterization at the same SZA.
MCM_ONLY = ("J_C3H7CHO_HCO", "J_IPRCHO", "J_C5HPALD1")

SOURCES = {
    **{n: "TUV-x" for n in TUVX_DIRECT},
    **{n: "TUV-x total, MCM branching" for n in TUVX_SPLIT},
    **{n: "MCM parameterization" for n in MCM_ONLY},
}


@dataclass(frozen=True)
class PhotolysisSettings:
    mode: str = "tuvx"                     # "tuvx" | "mcm" | "constant" | "off"
    latitude: float = 0.0                  # [deg N]
    longitude: float = 0.0                 # [deg E]
    start_utc: datetime = datetime(2026, 7, 1, tzinfo=timezone.utc)
    jno2: float = 8e-3                     # [1/s], "constant" mode only
    constant_sza_deg: float = 30.0         # "constant" mode only


def tuvx_root() -> Path:
    """tuvx-jax checkout holding ``data/`` (env ``TUVX_JAX_ROOT`` or the installed package)."""
    env = os.environ.get("TUVX_JAX_ROOT")
    if env:
        return Path(env)
    import tuvx_photolysis
    root = Path(tuvx_photolysis.__file__).resolve().parents[1]
    if not (root / "data").is_dir():
        raise RuntimeError("tuvx-jax data/ not found; set TUVX_JAX_ROOT to the tuvx-jax checkout")
    return root


@functools.lru_cache(maxsize=1)
def _tuvx_sza_table() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Surface J [1/s] for every needed TUV-x reaction on the SZA grid (ESD = 1 AU)."""
    from tuvx_photolysis import PhotolysisCalculator

    root = tuvx_root()
    calc = PhotolysisCalculator.from_tuvx_json(root / TUVX_CONFIG, data_root=root)
    needed = set(TUVX_DIRECT.values()) | {key for key, _ in TUVX_SPLIT.values()}
    missing = needed - set(calc.reaction_names())
    if missing:
        raise RuntimeError(f"TUV-x config lacks reactions needed by the MCM mechanism: {sorted(missing)}")
    grid = np.asarray(TUVX_SZA_GRID_DEG, dtype=float)
    table = {key: np.zeros(grid.size) for key in needed}
    for i, sza in enumerate(grid):
        if sza >= 90.0:
            continue                                   # sun below the horizon: J = 0
        profile = calc.rate_constants_profile(float(sza), 1.0)
        for key in needed:
            table[key][i] = float(np.interp(0.0, calc.height_edges_km, profile[key]))
    return grid, table


class JProvider:
    """J values in the order of ``j_names`` at elapsed time ``t`` [s] after ``start_utc``."""

    def __init__(self, j_names: list[str], settings: PhotolysisSettings):
        unknown = [n for n in j_names if n not in SOURCES]
        if unknown:
            raise ValueError(f"no photolysis source for {unknown}")
        if settings.mode not in ("tuvx", "mcm", "constant", "off"):
            raise ValueError(f"unknown photolysis mode {settings.mode!r}")
        self.j_names = list(j_names)
        self.settings = settings
        start = settings.start_utc
        self._start = start.replace(tzinfo=timezone.utc) if start.tzinfo is None else start.astimezone(timezone.utc)
        if settings.mode == "constant":
            if not (0.0 <= settings.constant_sza_deg < 90.0) or settings.jno2 < 0.0:
                raise ValueError("constant light needs 0 <= constant_sza_deg < 90 and jno2 >= 0")
            j = self._tuvx_at_sza(settings.constant_sza_deg, 1.0)
            self._constant = j * (settings.jno2 / j[self.j_names.index("J_NO2")])

    def sza_deg(self, t: float) -> float:
        from tuvx_photolysis.geometry import solar_zenith_angle

        s = self.settings
        if s.mode == "constant":
            return s.constant_sza_deg
        now = self._start + timedelta(seconds=float(t))
        hour = now.hour + now.minute / 60 + (now.second + now.microsecond * 1e-6) / 3600
        return solar_zenith_angle(now.year, now.month, now.day, hour, s.latitude, s.longitude)

    def _tuvx_at_sza(self, sza: float, esd: float) -> np.ndarray:
        grid, table = _tuvx_sza_table()
        mcm = mcm_photolysis(np.deg2rad(sza))
        out = np.zeros(len(self.j_names))
        if sza >= 90.0:
            return out
        for i, name in enumerate(self.j_names):
            if name in TUVX_DIRECT:
                out[i] = np.interp(sza, grid, table[TUVX_DIRECT[name]]) * esd
            elif name in TUVX_SPLIT:
                key, channels = TUVX_SPLIT[name]
                total_mcm = sum(float(mcm[c]) for c in channels)
                frac = float(mcm[name]) / total_mcm if total_mcm > 0 else 0.0
                out[i] = frac * np.interp(sza, grid, table[key]) * esd
            else:
                out[i] = float(mcm[name])
        return out

    def at(self, t: float) -> np.ndarray:
        mode = self.settings.mode
        if mode == "off":
            return np.zeros(len(self.j_names))
        if mode == "constant":
            return self._constant.copy()
        sza = self.sza_deg(t)
        if mode == "mcm":
            j = mcm_photolysis(np.deg2rad(sza), self.j_names)
            return np.array([float(j[n]) for n in self.j_names])
        from tuvx_photolysis.geometry import earth_sun_distance

        now = self._start + timedelta(seconds=float(t))
        esd = earth_sun_distance(now.year, now.month, now.day,
                                 now.hour + now.minute / 60 + now.second / 3600)
        return self._tuvx_at_sza(sza, esd)
