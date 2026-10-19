"""Export TOMAS-JAX photolysis frequencies to PyCHAM's J.xlsx format.

TOMAS-JAX freezes J for each coupling interval at its value at the interval midpoint
(``driver.run``: ``jp.at(t0 + 0.5 * dt_c)``). PyCHAM applies a J.xlsx file as a step
function (``photolysisRates.py``: column = last time <= now). So writing one column
per coupling interval, at the interval start, with our midpoint J makes both models
see identical photolysis.

J.xlsx layout (``J_value_file_open.py``): sheet "J", row 1 = times [s], then one row
per MCM J index 0..61 (``config.N_PYCHAM_J``); indices the mechanism does not use are 0.
"""
from __future__ import annotations

import csv
import subprocess
from pathlib import Path

import numpy as np

from tomas_jax.box.driver import _chemistry
from tomas_jax.box.scenario import Scenario
from tomas_jax.gaschem.photolysis import JProvider, PhotolysisSettings

from .config import MCM_J_NUMBER, N_PYCHAM_J, PYCHAM_PYTHON, WRITE_J_XLSX, XLSX_TIMEOUT_S


def j_per_interval(scenario: Scenario) -> tuple[np.ndarray, list[str], np.ndarray]:
    """(interval start times [s], J names, J [n_intervals, nJ]) exactly as driver.run uses them."""
    j_names = _chemistry()[1].j_names
    p = scenario.photolysis
    jp = JProvider(j_names, PhotolysisSettings(p.mode, p.latitude_deg, p.longitude_deg,
                                               scenario.start_utc, p.jno2_per_s,
                                               p.constant_sza_deg))
    dt_c = float(scenario.numerics.coupling_dt_s)
    n_int = int(round(scenario.duration_h * 3600.0 / dt_c))
    t0 = np.arange(n_int) * dt_c
    return t0, list(j_names), np.array([jp.at(t + 0.5 * dt_c) for t in t0])


def to_pycham_rows(j_names: list[str], j: np.ndarray) -> np.ndarray:
    """[n_intervals, nJ] in KPP-name order -> [N_PYCHAM_J, n_intervals] by MCM number."""
    unknown = [n for n in j_names if n not in MCM_J_NUMBER]
    if unknown:
        raise ValueError(f"no MCM J number for {unknown}")
    rows = np.zeros((N_PYCHAM_J, j.shape[0]))
    for col, name in enumerate(j_names):
        rows[MCM_J_NUMBER[name], :] = j[:, col]
    return rows


def write_j_files(out_dir: Path, t_start_s: np.ndarray, rows: np.ndarray,
                  stem: str = "J") -> tuple[Path, Path]:
    """Write ``<stem>.csv`` (same layout, for the record) and ``<stem>.xlsx`` for PyCHAM.

    The xlsx is written by ``write_j_xlsx.py`` in the PyCHAM venv (openpyxl is not a
    TOMAS-JAX dependency). Raises on a non-zero exit or timeout.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path, xlsx_path = out_dir / f"{stem}.csv", out_dir / f"{stem}.xlsx"
    with open(csv_path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([repr(float(t)) for t in t_start_s])
        for row in rows:
            w.writerow([repr(float(v)) for v in row])
    proc = subprocess.run([str(PYCHAM_PYTHON), "-I", str(WRITE_J_XLSX), str(csv_path),
                           str(xlsx_path)], capture_output=True, text=True,
                          timeout=XLSX_TIMEOUT_S)
    if proc.returncode != 0:
        raise RuntimeError(f"write_j_xlsx failed ({proc.returncode}): {proc.stderr[-2000:]}")
    return csv_path, xlsx_path
