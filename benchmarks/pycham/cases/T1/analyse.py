"""Case T1 analysis: metrics and a figure from the saved results (no model runs).

    cd tomas-jax && .venv/bin/python -m benchmarks.pycham.cases.T1.analyse

Writes ``results/species_stats.csv``, ``results/summary.json`` and ``figures/T1_key_species.png``
(copied to docs/aaar/figures/pycham/). Raises if a result file is missing.
"""
from __future__ import annotations

import json
import shutil

import numpy as np

from ... import metrics, ours, plotting
from ...config import (ABUNDANCE_THRESHOLD_PPB, COLORS, DOC_FIG_DIR, LABEL_OURS, LABEL_PYCHAM,
                       SNAPSHOT_TIMES_H, STRICT_THRESHOLD_PPB, T1)
from . import case

RADICALS_CM3 = ("OH", "HO2", "NO3")


def _need(path):
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run benchmarks.pycham.cases.T1.run first")
    return path


def _run_labels() -> dict[int, str]:
    return {int(T1["coupling_dt_s"]): f"{LABEL_OURS} (coupling {T1['coupling_dt_s']:.0f} s)",
            int(T1["coupling_dt_sensitivity_s"]):
                f"{LABEL_OURS} (coupling {T1['coupling_dt_sensitivity_s']:.0f} s)"}


def all_species_comparison() -> tuple[dict, dict]:
    py = dict(np.load(_need(case.PYCHAM_NPZ)))
    stats, summ = {}, {}
    for dt, label in _run_labels().items():
        o = ours.load_npz(_need(case.OURS_NPZ[dt]))
        names = metrics.common_species(o["species"], py["species"])
        ia, ib = metrics.common_times(o["t_s"], py["t_s"])
        x = o["gas_ppb"][ia][:, metrics.columns(o["species"], names)].astype(float)
        r = py["gas_ppb"][ib][:, metrics.columns(py["species"], names)].astype(float)
        t = o["t_s"][ia]
        stats[label] = metrics.species_stats(names, x, r)
        snaps = [metrics.snapshot(names, t, x, r, th) for th in SNAPSHOT_TIMES_H]
        summ[label] = metrics.summary(stats[label], snaps)
        strict = metrics.species_stats(names, x, r, STRICT_THRESHOLD_PPB)
        summ[label]["strict_threshold"] = metrics.summary(
            strict, [metrics.snapshot(names, t, x, r, th, STRICT_THRESHOLD_PPB) for th in SNAPSHOT_TIMES_H])
        summ[label]["worst_nmb_species"] = [
            {"species": str(stats[label]["species"][k]), "nmb": float(stats[label]["nmb"][k]),
             "pycham_max_ppb": float(stats[label]["ref_max_ppb"][k])}
            for k in np.argsort(-np.nan_to_num(np.abs(stats[label]["nmb"]), nan=-1))[:15]]
    return stats, summ


def key_species_comparison() -> dict:
    """Max relative difference and NMB of each key species (PyCHAM as reference), per run."""
    k = dict(np.load(_need(case.KEY_NPZ)))
    keys = [str(s) for s in k["species"]]
    out = {}
    for dt, label in _run_labels().items():
        ia, ib = metrics.common_times(k[f"ours_dt{dt}_t_s"], k["pycham_t_s"])
        x = k[f"ours_dt{dt}_ppb"][ia].astype(float)
        r = k["pycham_ppb"][ib].astype(float)
        st = metrics.species_stats(keys, x, r)
        out[label] = {name: {"max_rel_dev": float(st["max_rel_dev"][i]), "nmb": float(st["nmb"][i]),
                             "pycham_max_ppb": float(st["ref_max_ppb"][i]),
                             "ours_end_ppb": float(x[-1, i]), "pycham_end_ppb": float(r[-1, i])}
                      for i, name in enumerate(keys)}
    return out


def figure(summ_key: dict) -> None:
    k = dict(np.load(_need(case.KEY_NPZ)))
    py = dict(np.load(_need(case.PYCHAM_NPZ)))
    keys = [str(s) for s in k["species"]]
    m_air = float(py["ppb_to_molec_cm3"][0]) * 1e9
    units = {}
    series = []

    def values(t_key, ppb_key, every):
        vals = {}
        for i, name in enumerate(keys):
            v = k[ppb_key][:, i].astype(float)
            if name in RADICALS_CM3:
                v = v * 1e-9 * m_air
                units[name] = "molec cm⁻³"
            else:
                units[name] = "ppb"
            vals[name] = v
        return vals

    series.append(plotting.Series(LABEL_PYCHAM, COLORS["pycham"], k["pycham_t_s"],
                                  values("pycham_t_s", "pycham_ppb", 60), is_reference=True))
    for dt, label in _run_labels().items():
        series.append(plotting.Series(label, COLORS["ours"] if dt == 300 else COLORS["ours_alt"],
                                      k[f"ours_dt{dt}_t_s"], values(f"ours_dt{dt}_t_s", f"ours_dt{dt}_ppb", dt),
                                      linestyle="-" if dt == 300 else "--"))
    case.FIG_DIR.mkdir(parents=True, exist_ok=True)
    path = plotting.timeseries_grid(keys, series, case.FIG_DIR / "T1_key_species.png", units=units,
                                    title="T1: toluene + NOx + O3 in sunlight, gas phase only",
                                    log_species=("OH", "HO2", "NO3"))
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(path, DOC_FIG_DIR / path.name)


def main() -> int:
    stats, summ = all_species_comparison()
    key = key_species_comparison()
    metrics.write_stats_csv(case.RESULTS_DIR / "species_stats.csv", stats)
    out = {"all_species": summ, "key_species": key, "threshold_ppb": ABUNDANCE_THRESHOLD_PPB}
    case.SUMMARY_JSON.write_text(json.dumps(out, indent=2, default=float) + "\n")
    figure(key)
    for label, d in key.items():
        print(label)
        for name, v in d.items():
            print(f"  {name:8s} max rel dev {v['max_rel_dev']:.3e}  NMB {v['nmb']:+.3e}  "
                  f"end ours {v['ours_end_ppb']:.4g} pycham {v['pycham_end_ppb']:.4g} ppb")
    for label, s in summ.items():
        print(label, {k: v for k, v in s.items() if not isinstance(v, (list, dict))})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
