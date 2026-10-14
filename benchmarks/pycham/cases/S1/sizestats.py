"""Size statistics computed the same way for both models (per-bin -> time series).

Each bin is represented by its particles' mean size (TOMAS-JAX: the bin's mean dry mass
per particle; PyCHAM moving-centre: the bins' exact particle size), so the statistics do
not depend on how either model draws bin edges.
"""
from __future__ import annotations

import numpy as np

PER_BIN_EVERY_S = 1800.0   # per-bin arrays are stored on this grid (snapshots 0/3/6/12 h on it)
PER_BIN_KEYS = ("n_bin_cm3", "dry_ugm3_bin", "soa_ugm3_bin", "poa_ugm3_bin", "water_ugm3_bin",
                "d_dry_nm_bin", "d_wet_nm_bin")


def _weights(n, d):
    n = np.asarray(n, float)
    d = np.asarray(d, float)
    ok = (n > 0) & np.isfinite(d) & (d > 0)
    return np.where(ok, n, 0.0), np.where(ok, d, 1.0)


def number_mean(n, d):
    """Number-weighted mean diameter over bins, per time [same unit as d]."""
    w, d = _weights(n, d)
    return (w * d).sum(axis=1) / w.sum(axis=1)


def geometric_std(n, d):
    """Number-weighted geometric standard deviation of d over bins, per time [-]."""
    w, d = _weights(n, d)
    lg = np.log(d)
    mu = (w * lg).sum(axis=1) / w.sum(axis=1)
    return np.exp(np.sqrt((w * (lg - mu[:, None]) ** 2).sum(axis=1) / w.sum(axis=1)))


def derived(d: dict) -> dict:
    """Time series from the full-resolution per-bin arrays of one run."""
    n = d["n_bin_cm3"]
    return {"d_dry_mean_nm": number_mean(n, d["d_dry_nm_bin"]),
            "d_wet_mean_nm": number_mean(n, d["d_wet_nm_bin"]),
            "gsd_dry": geometric_std(n, d["d_dry_nm_bin"]),
            "n_total_cm3": np.asarray(n, float).sum(axis=1)}


def decimate_per_bin(d: dict) -> dict:
    """Derived series (full resolution) + per-bin arrays on PER_BIN_EVERY_S (float32)."""
    t = np.asarray(d["t_s"])
    keep = np.abs(t / PER_BIN_EVERY_S - np.round(t / PER_BIN_EVERY_S)) < 1e-6
    out = dict(d)
    out.update(derived(d))
    out["t_bin_s"] = t[keep]
    for k in PER_BIN_KEYS:
        out[k] = np.asarray(d[k])[keep].astype(np.float32)
    return out
