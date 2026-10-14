"""Run TOMAS-JAX for a benchmark case and record everything the comparison needs.

``run_ours(scenario)`` calls ``tomas_jax.box.driver.run`` with its opt-in ``observer``
hook (the run itself is unchanged) and returns, at t = 0 and after every coupling
interval:

* every gas species of the tutorial mechanism [ppb] (``gas_ppb``, ``species``),
* the J values used for each interval (``j``, ``j_names``),
* SA / NA_HET formed per interval (the driver resets these counters),
* size distributions dN/dlog10Dp and dry dM/dlog10Dp,
* particle mass per TOMAS component and per c* class [ug/m3], and the per-species
  particle tracker of the condensable organics [ug/m3].

``save_npz``/``load_npz`` store a run compactly (float32, optional time decimation).
"""
from __future__ import annotations

import contextlib
import functools
import time

import numpy as np

from tomas_jax.box import driver as _driver
from tomas_jax.box.coupling import bin_diameters_nm, kg_cell_to_ugm3, mass_grid
from tomas_jax.box.driver import _chemistry, _class_map, run
from tomas_jax.box.config import AerosolStepConfig
from tomas_jax.box.scenario import Scenario
from tomas_jax.core.config import SRTH2O, SRTNH4, SRTNIT, SRTORG_NUC, SRTSO4, SRTSOA1
from tomas_jax.gaschem.environment import air_environment

from .config import OURS_TIMEOUT_S


def h2o_hyland_wexler_cm3(temp_k: float, rh: float) -> float:
    """Water vapour [molec/cm3] as PyCHAM computes it (water_calc.py, Hyland-Wexler)."""
    t = temp_k
    psat = np.exp(-0.58002206e4 / t + 0.13914993e1 - 0.48640239e-1 * t + 0.41764768e-4 * t**2
                  - 0.14452093e-7 * t**3 + 0.65459673e1 * np.log(t))
    return float(rh * psat * 6.02214076e23 / (8.3144598e6 * t))


@contextlib.contextmanager
def numerics_override(rtol: float | None = None, atol: float | None = None,
                      pycham_water: bool = False):
    """Attribution runs only: patch the driver's gas solver tolerances and/or its water
    vapour (Buck -> PyCHAM's Hyland-Wexler) for the duration of the block.

    Benchmark-side monkeypatch of names imported into ``tomas_jax.box.driver``; the
    model code and its defaults are untouched and restored on exit.
    """
    orig_int, orig_air = _driver.integrate_interval, _driver.air_environment
    kw = {k: v for k, v in (("rtol", rtol), ("atol", atol)) if v is not None}
    try:
        if kw:
            _driver.integrate_interval = functools.partial(orig_int, **kw)
        if pycham_water:
            def air(temp, pres, rh):
                env = orig_air(temp, pres, rh)
                return dict(env, H2O=h2o_hyland_wexler_cm3(temp, rh))
            _driver.air_environment = air
        yield
    finally:
        _driver.integrate_interval, _driver.air_environment = orig_int, orig_air


def _component_names(k_cls: int, ncomp: int) -> list[str]:
    names = [f"comp{c}" for c in range(ncomp)]
    names[SRTSO4] = "SO4"
    for k in range(k_cls):
        names[SRTSOA1 + k] = f"SOA_class{k}"
    names[SRTNIT] = "NIT"
    names[SRTORG_NUC] = "POA"
    names[SRTNH4] = "NH4"
    names[SRTH2O] = "H2O"
    return names


def run_ours(scenario: Scenario, timeout_s: float = OURS_TIMEOUT_S) -> dict:
    """Run one scenario and return a dict of numpy arrays (see module docstring).

    Raises RuntimeError if the run exceeds ``timeout_s`` (checked between intervals).
    """
    if not scenario.processes.chemistry:
        raise ValueError("benchmark cases need chemistry = true")
    snaps: list[tuple[float, dict]] = []
    deadline = time.monotonic() + timeout_s
    t_wall = time.perf_counter()
    result = run(scenario, cancel=lambda: time.monotonic() > deadline,
                 observer=lambda t, snap: snaps.append((t, snap)))
    wall = time.perf_counter() - t_wall

    parsed, rate_model, _ = _chemistry()
    env = scenario.environment
    m_air = air_environment(env.temp_k, env.pres_pa, env.rh)["M"]
    boxvol = AerosolStepConfig().boxvol_cm3
    t_s = np.array([t for t, _ in snaps])
    y = np.array([s["y"] for _, s in snaps])
    nk = np.array([s["Nk"] for _, s in snaps]) / boxvol
    mk = kg_cell_to_ugm3(np.array([s["Mk"] for _, s in snaps]), boxvol)   # [t, bin, comp]
    xk = mass_grid(scenario.numerics.nbins)
    d_mid, d_edges = bin_diameters_nm(xk)
    dlog10 = np.log10(xk[1:] / xk[:-1]) / 3.0
    class_map = _class_map(scenario.cstar_table)
    k_cls = class_map.class_cstar.size

    out = {
        "t_s": t_s,
        "species": np.array(parsed.species),
        "gas_ppb": y / m_air * 1e9,
        "m_air_cm3": np.float64(m_air),
        "j_names": np.array(rate_model.j_names),
        # J used over (t_{i-1}, t_i]; row 0 (t = 0) is NaN
        "j": np.array([np.full(len(rate_model.j_names), np.nan) if s["j"] is None else s["j"]
                       for _, s in snaps]),
        "sa_formed_cm3": np.array([s["counters"]["SA"] for _, s in snaps]),
        "na_het_formed_cm3": np.array([s["counters"]["NA_HET"] for _, s in snaps]),
        "h2so4_gas_ugm3": kg_cell_to_ugm3(np.array([s["Gc"][SRTSO4] for _, s in snaps]), boxvol),
        "diameter_nm": d_mid, "diameter_edges_nm": d_edges,
        "dndlog10dp_cm3": nk / dlog10,
        "dmdlog10dp_ugm3": mk[:, :, :SRTH2O].sum(axis=2) / dlog10,
        "component_names": np.array(_component_names(k_cls, mk.shape[2])),
        "particle_component_ugm3": mk.sum(axis=1),                       # [t, comp]
        "class_log10_cstar": np.log10(class_map.class_cstar),
        "class_particle_ugm3": mk[:, :, SRTSOA1:SRTSOA1 + k_cls].sum(axis=1),
        "condensable_species": np.array(class_map.species),
        "condensable_particle_ugm3": kg_cell_to_ugm3(
            np.array([s["particle_tracker"] for _, s in snaps]), boxvol),
        "runtime_s": np.float64(wall),
        "driver_runtime_s": np.float64(result.diagnostics["runtime_s"]),
        "gas_solver_steps": np.int64(result.diagnostics["gas_solver_steps"]),
        "coupling_dt_s": np.float64(scenario.numerics.coupling_dt_s),
    }
    out["sa_cum_cm3"] = np.cumsum(out["sa_formed_cm3"])        # SA formed since t = 0
    out["result"] = result
    return out


# Arrays whose first axis is time (decimated by save_npz).
TIME_KEYS = ("t_s", "gas_ppb", "j", "sa_formed_cm3", "sa_cum_cm3", "na_het_formed_cm3",
             "h2so4_gas_ugm3",
             "dndlog10dp_cm3", "dmdlog10dp_ugm3", "particle_component_ugm3",
             "class_particle_ugm3", "condensable_particle_ugm3")


def on_grid(t_s: np.ndarray, every_s: float) -> np.ndarray:
    """Mask of times that are (numerically) multiples of ``every_s``."""
    q = np.asarray(t_s) / every_s
    return np.abs(q - np.round(q)) < 1e-6


def species_index(run_data: dict) -> dict[str, int]:
    return {str(s): i for i, s in enumerate(run_data["species"])}


def save_npz(path, run_data: dict, every_s: float | None = None) -> None:
    """Save a run (float32 arrays; keep only times that are multiples of ``every_s``)."""
    t = run_data["t_s"]
    keep = np.ones(t.size, bool) if every_s is None else on_grid(t, every_s)
    arrays = {}
    for key, val in run_data.items():
        if key == "result":
            continue
        val = np.asarray(val)
        if key in TIME_KEYS:
            val = val[keep]
        if val.dtype == np.float64 and val.ndim >= 1 and key not in ("t_s",):
            val = val.astype(np.float32)
        arrays[key] = val
    np.savez_compressed(path, **arrays)


def load_npz(path) -> dict:
    with np.load(path, allow_pickle=False) as f:
        return {k: f[k] for k in f.files}
