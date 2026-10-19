"""Case G2 analysis: metrics and figures from the saved results (no model runs).

    cd tomas-jax && .venv/bin/python -m benchmarks.pycham.cases.g2.analyse

Writes ``results/species_stats.csv``, ``results/summary.json`` and the figures in
``figures/`` (copied to docs/aaar/figures/pycham/). Raises if a result file is missing.
"""
from __future__ import annotations

import json
import shutil
import sys

import numpy as np

from ... import metrics, ours, plotting
from ...config import (ABUNDANCE_THRESHOLD_PPB, COLORS, DOC_FIG_DIR, G2, LABEL_OURS,
                       LABEL_PYCHAM, SNAPSHOT_TIMES_H, STRICT_THRESHOLD_PPB)
from . import case

RADICALS_CM3 = ("OH", "HO2", "NO3")       # plotted in molec/cm3, everything else in ppb
SA_LABEL = "SA"


def _need(path):
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run benchmarks.pycham.cases.g2.run first")
    return path


def _run_labels() -> dict[int, str]:
    return {int(G2["coupling_dt_s"]): f"{LABEL_OURS} (coupling {G2['coupling_dt_s']:.0f} s)",
            int(G2["coupling_dt_sensitivity_s"]):
                f"{LABEL_OURS} (coupling {G2['coupling_dt_sensitivity_s']:.0f} s)"}


def all_species_comparison() -> tuple[dict, dict, dict]:
    """Per-run stats, snapshots and summaries on the common 300 s grid."""
    py = dict(np.load(_need(case.PYCHAM_NPZ)))
    stats, snaps, summ = {}, {}, {}
    for dt, label in _run_labels().items():
        o = ours.load_npz(_need(case.OURS_NPZ[dt]))
        names = metrics.common_species(o["species"], py["species"])
        ia, ib = metrics.common_times(o["t_s"], py["t_s"])
        x = o["gas_ppb"][ia][:, metrics.columns(o["species"], names)].astype(float)
        r = py["gas_ppb"][ib][:, metrics.columns(py["species"], names)].astype(float)
        t = o["t_s"][ia]
        stats[label] = metrics.species_stats(names, x, r)
        snaps[label] = [metrics.snapshot(names, t, x, r, th) for th in SNAPSHOT_TIMES_H]
        summ[label] = metrics.summary(stats[label], snaps[label])
        strict = metrics.species_stats(names, x, r, STRICT_THRESHOLD_PPB)
        strict_snaps = [metrics.snapshot(names, t, x, r, th, STRICT_THRESHOLD_PPB)
                        for th in SNAPSHOT_TIMES_H]
        summ[label]["strict_threshold"] = metrics.summary(strict, strict_snaps)
        summ[label]["worst_nmb_species"] = [
            {"species": str(stats[label]["species"][k]), "nmb": float(stats[label]["nmb"][k]),
             "pycham_max_ppb": float(stats[label]["ref_max_ppb"][k])}
            for k in np.argsort(-np.nan_to_num(np.abs(stats[label]["nmb"]), nan=-1))[:15]]
    return stats, snaps, summ


def pair_stats(a: dict, b: dict, every_s: float) -> dict:
    """Per-species stats of run ``a`` against reference ``b`` on their common ``every_s`` grid."""
    names = metrics.common_species(a["species"], b["species"])
    ka, kb = ours.on_grid(a["t_s"], every_s), ours.on_grid(b["t_s"], every_s)
    ia, ib = metrics.common_times(a["t_s"][ka], b["t_s"][kb])
    x = a["gas_ppb"][ka][ia][:, metrics.columns(a["species"], names)].astype(float)
    r = b["gas_ppb"][kb][ib][:, metrics.columns(b["species"], names)].astype(float)
    return metrics.species_stats(names, x, r)


def attribution() -> tuple[dict, list]:
    """Which numerical difference explains what: pairwise comparisons on the 900 s grid."""
    every = G2["attribution_dt_s"]
    py = dict(np.load(_need(case.PYCHAM_NPZ)))
    py_t = dict(np.load(_need(case.PYCHAM_TIGHT_NPZ)))
    o300, o60 = (ours.load_npz(_need(case.OURS_NPZ[d])) for d in (300, 60))
    tight = ours.load_npz(_need(case.OURS_ATTR_NPZ["dt60_tight"]))
    tight_w = ours.load_npz(_need(case.OURS_ATTR_NPZ["dt60_tight_pychamH2O"]))
    tol = G2["ours_tight_tol"]
    p_atol, p_rtol = G2["pycham_tight_int_tol"]
    pairs = [
        ("TOMAS-JAX 300 s vs PyCHAM", o300, py, COLORS["ours"], "-"),
        ("TOMAS-JAX 60 s vs PyCHAM", o60, py, COLORS["ours_alt"], "--"),
        (f"TOMAS-JAX 60 s, rtol {tol['rtol']:g} vs PyCHAM, rtol {p_rtol:g}", tight, py_t,
         COLORS["attr_solver_both"], "-."),
        ("same + PyCHAM water vapour in TOMAS-JAX (all known differences removed)", tight_w, py_t,
         COLORS["attr_matched"], "-"),
        (f"PyCHAM default vs PyCHAM rtol {p_rtol:g} (PyCHAM solver error)", py, py_t,
         COLORS["attr_solver_pycham"], ":"),
        (f"TOMAS-JAX 60 s default vs rtol {tol['rtol']:g} (TOMAS-JAX solver error)", o60, tight,
         COLORS["attr_solver_ours"], ":"),
    ]
    out, panels = {}, {"|normalised mean bias|": [], "max |relative deviation|": []}
    for label, a, b, color, ls in pairs:
        st = pair_stats(a, b, every)
        ok = st["n_valid"] > 0
        out[label] = {"n_species": int(ok.sum()),
                      "median_abs_nmb": float(np.nanmedian(np.abs(st["nmb"][ok]))),
                      "p95_abs_nmb": float(np.nanpercentile(np.abs(st["nmb"][ok]), 95)),
                      "max_abs_nmb": float(np.nanmax(np.abs(st["nmb"][ok]))),
                      "median_max_rel_dev": float(np.nanmedian(st["max_rel_dev"][ok])),
                      "max_max_rel_dev": float(np.nanmax(st["max_rel_dev"][ok]))}
        panels["|normalised mean bias|"].append(
            plotting.Curve(label, st["nmb"][ok], color, ls, 2.4 if ls == "-" else 1.8))
        panels["max |relative deviation|"].append(
            plotting.Curve(label, st["max_rel_dev"][ok], color, ls, 2.4 if ls == "-" else 1.8))
    fig = plotting.stat_cdfs(
        panels, case.FIG_DIR / "g2_attribution.png",
        xlabels={k: f"{k} over 6 h, 900 s grid [-] (< 1e-6 drawn at 1e-6)" for k in panels},
        title="G2: attribution of the TOMAS-JAX vs PyCHAM differences (all species above threshold)")
    return out, [fig]


def _key_series(key: dict, species: list[str], m_air: float, py_cfac: np.ndarray):
    """plotting.Series for PyCHAM and both TOMAS-JAX runs (radicals in molec/cm3)."""
    names = [str(n) for n in key["species"]]

    def pick(arr, sp, conv):
        return arr[:, names.index(sp)].astype(float) * conv

    labels = _run_labels()
    series = []
    vals = {sp: pick(key["pycham_ppb"], sp, py_cfac if sp in RADICALS_CM3 else 1.0)
            for sp in species if sp != SA_LABEL}
    if SA_LABEL in species:
        vals[SA_LABEL] = pick(key["pycham_ppb"], SA_LABEL, 1.0)
    series.append(plotting.pycham_series(key["pycham_t_s"],
                                         vals, label=f"{LABEL_PYCHAM} (update step 60 s)"))

    for alt, dt in enumerate(labels):
        arr = key[f"ours_dt{dt}_ppb"]
        vals = {sp: pick(arr, sp, m_air * 1e-9 if sp in RADICALS_CM3 else 1.0)
                for sp in species if sp != SA_LABEL}
        if SA_LABEL in species:
            vals[SA_LABEL] = key[f"ours_dt{dt}_sa_cum_ppb"].astype(float)
        series.append(plotting.ours_series(key[f"ours_dt{dt}_t_s"], vals, label=labels[dt],
                                           alt=bool(alt)))
    # PyCHAM's own solver error: the same run at rtol 1e-6 (its diff panel line is
    # tight vs default). Drawn last so it stays visible where it coincides with ours.
    vals_t = {sp: pick(key["pycham_tight_ppb"], sp, py_cfac if sp in RADICALS_CM3 else 1.0)
              for sp in species}
    series.append(plotting.Series(
        f"{LABEL_PYCHAM} (rtol {G2['pycham_tight_int_tol'][1]:g})", COLORS["attr_solver_pycham"],
        key["pycham_tight_t_s"], vals_t, linestyle=":", linewidth=1.8))
    return series


def _units(species):
    return {sp: ("molec cm$^{-3}$" if sp in RADICALS_CM3 else "ppb") for sp in species}


def key_species_table(key: dict, species: list[str], m_air: float) -> list[dict]:
    """NMB / max relative deviation of the plotted species on the 300 s grid."""
    names = [str(n) for n in key["species"]]
    t_py = key["pycham_t_s"]
    rows = []
    for sp in species:
        row = {"species": sp}
        for dt in _run_labels():
            t_o = key[f"ours_dt{dt}_t_s"]
            keep = ours.on_grid(t_o, 300.0)
            ia, ib = metrics.common_times(t_o[keep], t_py)
            if sp == SA_LABEL:
                x = key[f"ours_dt{dt}_sa_cum_ppb"][keep][ia].astype(float)
            else:
                x = key[f"ours_dt{dt}_ppb"][keep][ia, names.index(sp)].astype(float)
            r = key["pycham_ppb"][ib, names.index(sp)].astype(float)
            mask = metrics.valid_mask(r[:, None])
            row[f"nmb_dt{dt}"] = float(metrics.normalised_mean_bias(x[:, None], r[:, None], mask)[0])
            row[f"max_rel_dev_dt{dt}"] = float(
                metrics.max_relative_deviation(x[:, None], r[:, None], mask)[0])
            row[f"ours_dt{dt}_6h_ppb"] = float(x[-1])
        row["pycham_6h_ppb"] = float(r[-1])
        row["pycham_max_ppb"] = float(r.max())
        rows.append(row)
    return rows


def main() -> int:
    case.FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    stats, snaps, summ = all_species_comparison()
    metrics.write_stats_csv(case.RESULTS_DIR / "species_stats.csv", stats)

    key = dict(np.load(_need(case.KEY_NPZ)))
    o300 = ours.load_npz(case.OURS_NPZ[300])
    m_air = float(o300["m_air_cm3"])
    py = dict(np.load(case.PYCHAM_NPZ))
    py_cfac = float(py["ppb_to_molec_cm3"][0])
    top = json.loads(_need(case.TOP_CONTRIBUTORS_JSON).read_text())["top_contributors"]

    key_sp = [sp for sp in G2["key_species"] if sp != "CO"] + [SA_LABEL]
    names = [str(n) for n in key["species"]]
    part_sp = [c["species"] for c in top[:G2["n_particle_relevant"]]]
    part_names = {c["species"]:
                  f"{c['species']} ({c['formula']}, log c* {round(c['log10_cstar'], 1) + 0.0:.1f})"
                  for c in top}
    # PAN plus the most abundant other PAN-type species not already in the SOA figure
    pan_all = [n for n in names if "PAN" in n and n != "PAN" and n not in part_sp]
    pan_all.sort(key=lambda n: -float(key["pycham_ppb"][:, names.index(n)].max()))
    pan_sp = ["PAN"] + pan_all[:G2["n_pan_species"] - 1]

    figs = []
    title = "G2 gas phase, Level 2 conditions (J from TOMAS-JAX in both models)"
    figs.append(plotting.timeseries_grid(
        key_sp, _key_series(key, key_sp, m_air, py_cfac), case.FIG_DIR / "g2_key_species.png",
        units=_units(key_sp), ncols=4, title=title,
        display_names={SA_LABEL: "SA (H2SO4 formed, cumulative)"}))
    figs.append(plotting.timeseries_grid(
        pan_sp, _key_series(key, pan_sp, m_air, py_cfac), case.FIG_DIR / "g2_pan_species.png",
        units=_units(pan_sp), ncols=4, title=f"{title}: PAN-type species"))
    figs.append(plotting.timeseries_grid(
        part_sp, _key_series(key, part_sp, m_air, py_cfac),
        case.FIG_DIR / "g2_particle_relevant.png", units=_units(part_sp), ncols=3,
        title=f"{title}: top particle-phase contributors of the Level 2 run",
        display_names=part_names))
    figs.append(plotting.scatter_snapshots(
        snaps, case.FIG_DIR / "g2_scatter_all_species.png", threshold_ppb=ABUNDANCE_THRESHOLD_PPB,
        title="G2: every gas species above threshold, TOMAS-JAX vs PyCHAM"))
    figs.append(plotting.ratio_distribution(
        snaps, case.FIG_DIR / "g2_ratio_cdf.png",
        title="G2: distribution of log10(TOMAS-JAX / PyCHAM) over all species above threshold"))
    attr, attr_figs = attribution()
    figs += attr_figs
    for f in figs:
        shutil.copyfile(f, DOC_FIG_DIR / f.name)

    out = {"threshold_ppb": ABUNDANCE_THRESHOLD_PPB, "strict_threshold_ppb": STRICT_THRESHOLD_PPB,
           "runs": summ, "attribution": attr,
           "key_species": key_species_table(key, key_sp + pan_sp + part_sp + ["CO"], m_air),
           "pan_species": pan_sp, "particle_relevant_species": part_sp,
           "figures": [str(f) for f in figs]}
    (case.RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    for label, s in summ.items():
        print(label, json.dumps({k: v for k, v in s.items() if k != "worst_nmb_species"}, indent=1))
    print("attribution", json.dumps(attr, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
