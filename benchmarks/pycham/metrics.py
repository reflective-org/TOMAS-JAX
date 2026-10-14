"""Comparison metrics: TOMAS-JAX ("ours") against PyCHAM (the reference, "ref").

Conventions (all mixing ratios in ppb, arrays shaped [time, species]):

* A (time, species) point is *valid* when ref >= threshold (``ABUNDANCE_THRESHOLD_PPB``).
  t = 0 is excluded (both models start from the same initial state).
* relative difference  r = (ours - ref) / ref           (valid points, NaN elsewhere)
* normalised mean bias NMB = sum_t(ours - ref) / sum_t(ref)   over valid times
* max relative deviation  = max_t |r|                    over valid times
* log10 ratio          = log10(ours / ref); ours = 0 is floored at ``RATIO_FLOOR_PPB``
  (and counted) so missing species show up at the extreme of a CDF instead of vanishing.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from .config import ABUNDANCE_THRESHOLD_PPB, AGREEMENT_BAND, NON_CHEMICAL_SPECIES

RATIO_FLOOR_PPB = 1.0e-12


def signed_pct(x: float) -> str:
    """'+1.2 %' (two decimals below 1 %) for the generated mismatch texts."""
    return f"{100.0 * x:+.2f} %" if abs(x) < 0.01 else f"{100.0 * x:+.1f} %"


def common_times(t_a: np.ndarray, t_b: np.ndarray, tol_s: float = 1e-3) -> tuple[np.ndarray, np.ndarray]:
    """Index pairs (ia, ib) of times present in both axes [s]."""
    ia, ib = [], []
    for i, t in enumerate(t_a):
        j = int(np.argmin(np.abs(t_b - t)))
        if abs(t_b[j] - t) <= tol_s:
            ia.append(i)
            ib.append(j)
    if not ia:
        raise ValueError("no common output times")
    return np.array(ia), np.array(ib)


def common_species(names_ours, names_ref, exclude=NON_CHEMICAL_SPECIES) -> list[str]:
    """Species in both models (order of ``names_ours``) minus bookkeeping species."""
    ref = set(map(str, names_ref))
    return [str(n) for n in names_ours if str(n) in ref and str(n) not in exclude]


def columns(names, wanted) -> np.ndarray:
    index = {str(n): i for i, n in enumerate(names)}
    missing = [w for w in wanted if w not in index]
    if missing:
        raise KeyError(f"species not found: {missing[:10]}")
    return np.array([index[w] for w in wanted])


def valid_mask(ref: np.ndarray, threshold: float = ABUNDANCE_THRESHOLD_PPB,
               skip_first: bool = True) -> np.ndarray:
    mask = np.asarray(ref) >= threshold
    if skip_first:
        mask = mask.copy()
        mask[0] = False
    return mask


def relative_difference(ours: np.ndarray, ref: np.ndarray,
                        threshold: float = ABUNDANCE_THRESHOLD_PPB) -> np.ndarray:
    """(ours - ref)/ref where ref >= threshold, NaN elsewhere (t = 0 included)."""
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    out = np.full(np.broadcast(ours, ref).shape, np.nan)
    ok = ref >= threshold
    np.divide(ours - ref, ref, out=out, where=ok)
    return out


def normalised_mean_bias(ours: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per species (axis 0 = time): sum(ours - ref)/sum(ref) over ``mask``; NaN if empty."""
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    num = np.where(mask, ours - ref, 0.0).sum(axis=0)
    den = np.where(mask, ref, 0.0).sum(axis=0)
    return np.divide(num, den, out=np.full(num.shape, np.nan), where=den > 0)


def max_relative_deviation(ours: np.ndarray, ref: np.ndarray, mask: np.ndarray) -> np.ndarray:
    ours, ref = np.asarray(ours, float), np.asarray(ref, float)
    rel = np.abs(ours - ref) / np.where(mask, ref, 1.0)
    rel = np.where(mask, rel, -np.inf).max(axis=0)
    return np.where(np.isfinite(rel), rel, np.nan)


def log10_ratio(ours: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return np.log10(np.maximum(np.asarray(ours, float), RATIO_FLOOR_PPB)
                    / np.asarray(ref, float))


def species_stats(names: list[str], ours: np.ndarray, ref: np.ndarray,
                  threshold: float = ABUNDANCE_THRESHOLD_PPB) -> dict[str, np.ndarray]:
    """Per-species NMB, max relative deviation, number of valid times, peak ref [ppb]."""
    mask = valid_mask(ref, threshold)
    return {
        "species": np.array(names),
        "nmb": normalised_mean_bias(ours, ref, mask),
        "max_rel_dev": max_relative_deviation(ours, ref, mask),
        "n_valid": mask.sum(axis=0),
        "ref_max_ppb": np.asarray(ref).max(axis=0),
        "ours_max_ppb": np.asarray(ours).max(axis=0),
    }


def snapshot(names: list[str], t_s: np.ndarray, ours: np.ndarray, ref: np.ndarray, t_h: float,
             threshold: float = ABUNDANCE_THRESHOLD_PPB) -> dict[str, np.ndarray]:
    """Species above threshold in ref at the output time closest to ``t_h``."""
    i = int(np.argmin(np.abs(t_s - t_h * 3600.0)))
    if abs(t_s[i] - t_h * 3600.0) > 1.0:
        raise ValueError(f"no output at {t_h} h")
    ok = ref[i] >= threshold
    return {"t_h": t_h, "species": np.array(names)[ok], "ours": ours[i, ok], "ref": ref[i, ok],
            "log10_ratio": log10_ratio(ours[i, ok], ref[i, ok]),
            "n_ours_zero": int(np.sum(ours[i, ok] <= 0)),
            # species TOMAS-JAX has above threshold but PyCHAM does not
            "ours_only": np.array(names)[(ours[i] >= threshold) & ~ok]}


def fraction_within(values: np.ndarray, band: float = AGREEMENT_BAND) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    return float(np.mean(np.abs(v) <= band)) if v.size else float("nan")


def summary(stats: dict, snaps: list[dict], band: float = AGREEMENT_BAND) -> dict:
    """Headline numbers for a case."""
    ok = stats["n_valid"] > 0
    out = {
        "n_species_compared": int(ok.sum()),
        "frac_max_rel_dev_within_band": fraction_within(stats["max_rel_dev"][ok], band),
        "frac_abs_nmb_within_band": fraction_within(stats["nmb"][ok], band),
        "median_abs_nmb": float(np.nanmedian(np.abs(stats["nmb"][ok]))),
        "p95_abs_nmb": float(np.nanpercentile(np.abs(stats["nmb"][ok]), 95)),
    }
    for s in snaps:
        rel = 10.0 ** s["log10_ratio"] - 1.0
        key = f"{s['t_h']:g}h"
        out[f"n_species_{key}"] = int(s["species"].size)
        out[f"frac_within_band_{key}"] = fraction_within(rel, band)
        out[f"median_abs_log10_ratio_{key}"] = float(np.median(np.abs(s["log10_ratio"])))
        out[f"n_ours_only_{key}"] = int(s["ours_only"].size)
    return out


def write_stats_csv(path: Path, stats_by_run: dict[str, dict]) -> Path:
    """One row per species: NMB / max rel dev for each TOMAS-JAX run vs PyCHAM."""
    runs = list(stats_by_run)
    base = stats_by_run[runs[0]]
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["species", "pycham_max_ppb", "n_valid_times"]
                   + [f"{c}_{r}" for r in runs for c in ("nmb", "max_rel_dev", "max_ppb")])
        order = np.argsort(-base["ref_max_ppb"])
        for k in order:
            row = [base["species"][k], f"{base['ref_max_ppb'][k]:.6g}", int(base["n_valid"][k])]
            for r in runs:
                st = stats_by_run[r]
                row += [f"{st['nmb'][k]:.5g}", f"{st['max_rel_dev'][k]:.5g}",
                        f"{st['ours_max_ppb'][k]:.6g}"]
            w.writerow(row)
    return Path(path)
