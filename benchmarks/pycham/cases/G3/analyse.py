"""Case G3 analysis: metrics and figures from the saved results (no model runs).

    cd tomas-jax && .venv/bin/python -m benchmarks.pycham.cases.G3.analyse

Writes ``results/species_stats.csv``, ``results/summary.json`` and the figures in
``figures/`` (copied to docs/aaar/figures/pycham/ with the G3_ prefix). Raises if a
result file is missing.
"""
from __future__ import annotations

import json
import shutil
import sys
from datetime import datetime

import numpy as np

from ... import metrics, ours, plotting
from ...config import (ABUNDANCE_THRESHOLD_PPB, AGREEMENT_BAND, COLORS, DOC_FIG_DIR,
                       LABEL_OURS, LABEL_PYCHAM, NON_CHEMICAL_SPECIES, STRICT_THRESHOLD_PPB)
from . import case, plots
from .case import G3

RADICALS_CM3 = ("OH", "HO2", "NO3")       # plotted in molec/cm3, everything else in ppb
SA_LABEL = "SA"
TRANSITION_STATS_WINDOW_H = 1.0           # +- around sunset/sunrise for the window stats
RUN_LABELS = {"dt900": f"{LABEL_OURS} (coupling 900 s)",
              "dt60": f"{LABEL_OURS} (coupling 60 s)"}
PY_LABELS = {"j60": f"{LABEL_PYCHAM} (J every 60 s)",
             "j900": f"{LABEL_PYCHAM} (J every 900 s)"}


def _need(path):
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run benchmarks.pycham.cases.G3.run first")
    return path


def load_sun() -> dict:
    sun = json.loads(_need(case.SUN_JSON).read_text())
    start = datetime.fromisoformat(sun["start_utc"])
    sun["start_local_h"] = (start.hour + start.minute / 60.0 + sun["utc_offset_h"]) % 24.0
    return sun


def _load_ours(tag):
    return ours.load_npz(_need(case.OURS_NPZ[tag]))


def _load_py(variant):
    return dict(np.load(_need(case.PYCHAM_NPZ[variant])))


def _mechanism_species(a, b) -> list[str]:
    """Species in both runs, chemistry only: no bookkeeping, no PyCHAM-only 'H2O'/'core',
    and not the held species (NO2 is 1 ppb at every output in both models by construction;
    it is analysed separately in held_analysis)."""
    from tomas_jax.box.driver import _chemistry
    mech = set(_chemistry()[0].species) - set(G3["held_species"])
    return [n for n in metrics.common_species(a["species"], b["species"]) if n in mech]


def aligned(a: dict, b: dict, every_s: float):
    """(names, t_s, a [t, sp], b [t, sp]) on the common ``every_s`` grid."""
    names = _mechanism_species(a, b)
    ka, kb = ours.on_grid(a["t_s"], every_s), ours.on_grid(b["t_s"], every_s)
    ia, ib = metrics.common_times(a["t_s"][ka], b["t_s"][kb])
    x = a["gas_ppb"][ka][ia][:, metrics.columns(a["species"], names)].astype(float)
    r = b["gas_ppb"][kb][ib][:, metrics.columns(b["species"], names)].astype(float)
    return names, a["t_s"][ka][ia], x, r


def masked_stats(x, r, time_mask, threshold=ABUNDANCE_THRESHOLD_PPB) -> dict:
    mask = (r >= threshold) & time_mask[:, None]
    ok = mask.any(axis=0)
    nmb = metrics.normalised_mean_bias(x, r, mask)
    mrd = metrics.max_relative_deviation(x, r, mask)
    if not ok.any():
        return {"n_species": 0}
    return {"n_species": int(ok.sum()),
            "median_abs_nmb": float(np.nanmedian(np.abs(nmb[ok]))),
            "p95_abs_nmb": float(np.nanpercentile(np.abs(nmb[ok]), 95)),
            "max_abs_nmb": float(np.nanmax(np.abs(nmb[ok]))),
            "median_max_rel_dev": float(np.nanmedian(mrd[ok])),
            "frac_max_rel_dev_within_band": metrics.fraction_within(mrd[ok]),
            "max_max_rel_dev": float(np.nanmax(mrd[ok]))}


def time_masks(t_s: np.ndarray, sun: dict) -> dict[str, np.ndarray]:
    t_h = np.asarray(t_s) / 3600.0
    night = (t_h > sun["t_sunset_h"]) & (t_h <= sun["t_sunrise_h"])
    near = np.zeros(t_h.size, bool)
    for ev in ("t_sunset_h", "t_sunrise_h"):
        near |= np.abs(t_h - sun[ev]) <= TRANSITION_STATS_WINDOW_H
    pos = t_h > 0
    return {"all": pos, "day": pos & ~night, "night": pos & night,
            f"within_{TRANSITION_STATS_WINDOW_H:g}h_of_sunset_or_sunrise": pos & near,
            "away_from_transitions": pos & ~near}


def all_species_comparison(sun: dict) -> tuple[dict, dict, dict]:
    """Per-run stats, named snapshots and summaries vs PyCHAM (J every 60 s), 900 s grid."""
    py = _load_py("j60")
    stats, snaps, summ = {}, {}, {}
    for tag, label in RUN_LABELS.items():
        names, t, x, r = aligned(_load_ours(tag), py, G3["compare_dt_s"])
        stats[label] = metrics.species_stats(names, x, r)
        snaps[label] = [metrics.snapshot(names, t, x, r, th) for _, th in G3["snapshots"]]
        s = metrics.summary(stats[label], snaps[label])
        s["snapshot_names"] = {f"{th:g}h": name for name, th in G3["snapshots"]}
        strict = metrics.species_stats(names, x, r, STRICT_THRESHOLD_PPB)
        strict_snaps = [metrics.snapshot(names, t, x, r, th, STRICT_THRESHOLD_PPB)
                        for _, th in G3["snapshots"]]
        s["strict_threshold"] = metrics.summary(strict, strict_snaps)
        s["by_period"] = {k: masked_stats(x, r, m) for k, m in time_masks(t, sun).items()}
        s["worst_nmb_species"] = [
            {"species": str(stats[label]["species"][k]), "nmb": float(stats[label]["nmb"][k]),
             "max_rel_dev": float(stats[label]["max_rel_dev"][k]),
             "pycham_max_ppb": float(stats[label]["ref_max_ppb"][k])}
            for k in np.argsort(-np.nan_to_num(np.abs(stats[label]["nmb"]), nan=-1))[:15]]
        s["worst_max_rel_dev_species"] = [
            {"species": str(stats[label]["species"][k]),
             "max_rel_dev": float(stats[label]["max_rel_dev"][k]),
             "t_of_max_h": float(t[int(np.nanargmax(np.where(
                 r[:, k] >= ABUNDANCE_THRESHOLD_PPB, np.abs(x[:, k] - r[:, k]) / r[:, k], -1)
                 * (t > 0)))] / 3600.0),
             "pycham_max_ppb": float(stats[label]["ref_max_ppb"][k])}
            for k in np.argsort(-np.nan_to_num(stats[label]["max_rel_dev"], nan=-1))[:10]]
        summ[label] = s
    return stats, snaps, summ


# --- attribution -------------------------------------------------------------------------
def attribution_pairs():
    """(label, run a, run b, colour, linestyle, in_figure) - a is compared against b."""
    tol, (_, p_rtol) = G3["ours_tight_tol"], G3["pycham_tight_int_tol"]
    o = {tag: _load_ours(tag) for tag in case.OURS_RUNS}
    p = {v: _load_py(v) for v in case.PYCHAM_VARIANTS}
    return [
        ("TOMAS-JAX 900 s vs PyCHAM (J every 60 s): as run", o["dt900"], p["j60"],
         COLORS["ours"], "-", True),
        ("TOMAS-JAX 60 s vs PyCHAM (J every 60 s): as run", o["dt60"], p["j60"],
         COLORS["ours_alt"], "--", True),
        ("PyCHAM J every 900 s vs every 60 s (J staircase alone)", p["j900"], p["j60"],
         COLORS["pycham_alt"], "-.", True),
        ("TOMAS-JAX 900 s: NO2 reset per interval vs held in solver (hold convention alone)",
         o["dt900"], o["dt900_heldNO2"], COLORS["attr_hold"], "-.", True),
        ("TOMAS-JAX 900 s, NO2 held in solver vs PyCHAM, same 900 s J (rates/RO2 frozen 900 s)",
         o["dt900_heldNO2"], p["j900"], COLORS["attr_frozen_rates"], "--", True),
        (f"TOMAS-JAX 60 s, rtol {tol['rtol']:g}, PyCHAM water, NO2 held in solver vs PyCHAM, "
         f"rtol {p_rtol:g} (all known differences removed)",
         o["dt60_tight_pychamH2O_heldNO2"], p["j60_tight"], COLORS["attr_matched"], "-", True),
        (f"PyCHAM default vs rtol {p_rtol:g} (PyCHAM solver error)", p["j60"], p["j60_tight"],
         COLORS["attr_solver_pycham"], ":", True),
        (f"TOMAS-JAX 60 s default vs rtol {tol['rtol']:g} (TOMAS-JAX solver error)", o["dt60"],
         o["dt60_tight"], COLORS["attr_solver_ours"], ":", True),
        # table only (intermediate steps)
        ("TOMAS-JAX 900 s vs PyCHAM, same 900 s J", o["dt900"], p["j900"], None, None, False),
        (f"TOMAS-JAX 60 s rtol {tol['rtol']:g} vs PyCHAM rtol {p_rtol:g}", o["dt60_tight"],
         p["j60_tight"], None, None, False),
        (f"TOMAS-JAX 60 s rtol {tol['rtol']:g} + PyCHAM water vs PyCHAM rtol {p_rtol:g}",
         o["dt60_tight_pychamH2O"], p["j60_tight"], None, None, False),
        ("TOMAS-JAX 60 s (tight, PyCHAM water): NO2 reset vs held in solver",
         o["dt60_tight_pychamH2O"], o["dt60_tight_pychamH2O_heldNO2"], None, None, False),
    ]


def attribution(sun: dict) -> tuple[dict, list]:
    every = G3["attribution_dt_s"]
    out, panels = {}, {"|normalised mean bias|": [], "max |relative deviation|": []}
    for label, a, b, color, ls, in_fig in attribution_pairs():
        names, t, x, r = aligned(a, b, every)
        st = metrics.species_stats(names, x, r)
        ok = st["n_valid"] > 0
        masks = time_masks(t, sun)
        out[label] = {"n_species": int(ok.sum()),
                      "median_abs_nmb": float(np.nanmedian(np.abs(st["nmb"][ok]))),
                      "p95_abs_nmb": float(np.nanpercentile(np.abs(st["nmb"][ok]), 95)),
                      "max_abs_nmb": float(np.nanmax(np.abs(st["nmb"][ok]))),
                      "species_max_abs_nmb": str(st["species"][ok][np.nanargmax(
                          np.abs(st["nmb"][ok]))]),
                      "median_max_rel_dev": float(np.nanmedian(st["max_rel_dev"][ok])),
                      "max_max_rel_dev": float(np.nanmax(st["max_rel_dev"][ok])),
                      "frac_max_rel_dev_within_band": metrics.fraction_within(
                          st["max_rel_dev"][ok]),
                      "by_period": {k: masked_stats(x, r, m) for k, m in masks.items()
                                    if k in ("day", "night")},
                      "key_species_nmb": {sp: float(st["nmb"][names.index(sp)])
                                          for sp in G3["key_species"] if sp in names}}
        if in_fig:
            lw = 2.4 if ls == "-" else 1.8
            panels["|normalised mean bias|"].append(plotting.Curve(label, st["nmb"][ok], color, ls, lw))
            panels["max |relative deviation|"].append(
                plotting.Curve(label, st["max_rel_dev"][ok], color, ls, lw))
    fig = plotting.stat_cdfs(
        panels, case.FIG_DIR / f"{case.FIG_PREFIX}attribution.png",
        xlabels={k: f"{k} over 24 h, {every:.0f} s grid [-]  (< 1e-6 drawn at 1e-6)"
                 for k in panels},
        title="G3: attribution of the TOMAS-JAX vs PyCHAM differences "
              "(all species above 1e-5 ppb in the second run of each pair)")
    return out, [fig]


# --- the held species (NO2) -------------------------------------------------------------
def noy_ppb(run: dict) -> np.ndarray:
    """Total reactive nitrogen [ppb N] over all stored species (NA/NA_HET count 1 N)."""
    from tomas_jax.box.driver import _chemistry
    atoms = _chemistry()[0].atoms
    n_n = np.array([1.0 if str(sp) in ("NA", "NA_HET")
                    else float(atoms.get(str(sp), {}).get("N", 0)) for sp in run["species"]])
    return run["gas_ppb"].astype(float) @ n_n


def sawtooth(key: dict, tag: str, sp: str) -> tuple[np.ndarray, np.ndarray]:
    """Held species of a TOMAS-JAX run inside its intervals: the reset value at each
    interval start followed by the re-integrated sub-step values (before the reset)."""
    dt = float(case.OURS_RUNS[tag][0])
    sub = min(G3["no2_substep_s"], dt)
    n_sub = int(round(dt / sub))
    k = [str(h) for h in key["held_species"]].index(sp)
    t_sub = key[f"held_{tag}_t_s"].reshape(-1, n_sub)
    v_sub = key[f"held_{tag}_ppb"][:, k].astype(float).reshape(-1, n_sub)
    t0 = t_sub[:, :1] - sub                                   # interval starts
    held_value = float(case.scenario().gases_ppb[sp])
    t = np.concatenate([t0, t_sub], axis=1).ravel()
    v = np.concatenate([np.full(t0.shape, held_value), v_sub], axis=1).ravel()
    return t, v


def interval_means(t_s, v, sun) -> dict:
    """Time means [ppb] (trapezoid) over 24 h, the night (sunset-sunrise), the day (rest)."""
    t_h = np.asarray(t_s) / 3600.0

    def integral(a, b):
        m = (t_h >= a) & (t_h <= b)
        return float(np.trapezoid(v[m], t_h[m])), float(t_h[m][-1] - t_h[m][0])

    i_all, s_all = integral(0.0, t_h.max())
    i_night, s_night = integral(sun["t_sunset_h"], sun["t_sunrise_h"])
    return {"24h": i_all / s_all, "night": i_night / s_night,
            "day": (i_all - i_night) / (s_all - s_night), "min": float(np.min(v))}


def held_analysis(key: dict, sun: dict) -> tuple[dict, list]:
    sp = G3["held_species"][0]
    py = _load_py("j60")
    o900, o60, held900 = _load_ours("dt900"), _load_ours("dt60"), _load_ours("dt900_heldNO2")
    t900, v900 = sawtooth(key, "dt900", sp)
    t60, v60 = sawtooth(key, "dt60", sp)
    py_idx = [str(n) for n in key["species"]].index(sp)
    lines = [
        {"label": PY_LABELS["j60"], "t_s": key["pycham_j60_t_s"],
         "ppb": key["pycham_j60_ppb"][:, py_idx].astype(float), "color": COLORS["pycham"],
         "linestyle": "-", "linewidth": 2.6},
        {"label": RUN_LABELS["dt900"] + ", inside the intervals", "t_s": t900, "ppb": v900,
         "color": COLORS["ours"], "linestyle": "-", "linewidth": 1.4},
        {"label": RUN_LABELS["dt60"] + ", inside the intervals", "t_s": t60, "ppb": v60,
         "color": COLORS["ours_alt"], "linestyle": "--", "linewidth": 1.2}]
    noy = []
    stats = {"mean_ppb": {"pycham_j60": 1.0, "ours_dt900": interval_means(t900, v900, sun),
                          "ours_dt60": interval_means(t60, v60, sun)},
             "noy_added_24h_ppb": {}}
    for label, run, color, ls, marker, tag in (
            (PY_LABELS["j60"], py, COLORS["pycham"], "-", None, "pycham_j60"),
            (RUN_LABELS["dt900"], o900, COLORS["ours"], "-", "o", "ours_dt900"),
            (RUN_LABELS["dt60"], o60, COLORS["ours_alt"], "--", None, "ours_dt60"),
            (f"{LABEL_OURS} 900 s, NO2 held in the solver (attribution run)", held900,
             COLORS["attr_hold"], "-.", None, "ours_dt900_heldNO2")):
        n = noy_ppb(run)
        noy.append({"label": label, "t_s": run["t_s"], "ppb": n - n[0], "color": color,
                    "linestyle": ls, "linewidth": 2.6 if tag == "pycham_j60" else 1.6,
                    "marker": marker})
        stats["noy_added_24h_ppb"][tag] = float(n[-1] - n[0])
    ref = stats["noy_added_24h_ppb"]["pycham_j60"]
    stats["noy_added_24h_rel_to_pycham"] = {k: v / ref - 1.0
                                            for k, v in stats["noy_added_24h_ppb"].items()}
    m9, m6 = stats["mean_ppb"]["ours_dt900"], stats["mean_ppb"]["ours_dt60"]
    text = (f"time-mean NO2 [ppb]   day / night / 24 h\n"
            f"PyCHAM            {1.0:.3f} / {1.0:.3f} / {1.0:.3f}\n"
            f"TOMAS-JAX 900 s   {m9['day']:.3f} / {m9['night']:.3f} / {m9['24h']:.3f}\n"
            f"TOMAS-JAX 60 s    {m6['day']:.3f} / {m6['night']:.3f} / {m6['24h']:.3f}")
    zooms = [("first two hours (12:00-14:00)", 0.0, 2.0),
             ("afternoon (15:00-17:00)", 3.0, 5.0),
             ("night (23:00-01:00)", 11.0, 13.0)]
    fig = plots.held_species_figure(
        lines, noy, case.FIG_DIR / f"{case.FIG_PREFIX}no2_hold.png", sun=sun, zooms=zooms,
        species=sp, text=text,
        title="G3: what \"NO2 held at 1 ppb\" means in each model (dominant difference "
              "at the preset 900 s coupling)")
    return stats, [fig]


# --- key species --------------------------------------------------------------------------
def _key_series(key: dict, species: list[str], include_py900: bool = True):
    names = [str(n) for n in key["species"]]
    m_air = float(key["m_air_cm3"])

    def vals(arr, conv, sa_cum=None):
        out = {}
        for sp in species:
            if sp == SA_LABEL and sa_cum is not None:
                out[sp] = sa_cum.astype(float)
                continue
            c = conv if sp in RADICALS_CM3 else 1.0
            out[sp] = arr[:, names.index(sp)].astype(float) * c
        return out

    series = [plotting.pycham_series(
        key["pycham_j60_t_s"], vals(key["pycham_j60_ppb"], float(key["pycham_j60_cfac"][0])),
        label=PY_LABELS["j60"])]
    if include_py900:
        series.append(plotting.Series(PY_LABELS["j900"], COLORS["pycham_alt"], key["pycham_j900_t_s"],
                                      vals(key["pycham_j900_ppb"], float(key["pycham_j900_cfac"][0])),
                                      linestyle=":", linewidth=1.8))
    s900 = plotting.ours_series(key["ours_dt900_t_s"],
                                vals(key["ours_dt900_ppb"], m_air * 1e-9, key["ours_dt900_sa_cum_ppb"]),
                                label=RUN_LABELS["dt900"])
    s900.marker = "o"
    series.append(s900)
    series.append(plotting.ours_series(key["ours_dt60_t_s"],
                                       vals(key["ours_dt60_ppb"], m_air * 1e-9,
                                            key["ours_dt60_sa_cum_ppb"]),
                                       label=RUN_LABELS["dt60"], alt=True))
    return series


def _units(species):
    return {sp: ("molec cm$^{-3}$" if sp in RADICALS_CM3 else "ppb") for sp in species}


def _floors(species, key):
    cfac = float(key["pycham_j60_cfac"][0])
    return {sp: ABUNDANCE_THRESHOLD_PPB * (cfac if sp in RADICALS_CM3 else 1.0) for sp in species}


def key_species_table(key: dict, species: list[str], sun: dict) -> list[dict]:
    """Per species: NMB / max rel. dev. (900 s grid) of each run vs PyCHAM (J every 60 s),
    day and night NMB, and the values at the named snapshots."""
    names = [str(n) for n in key["species"]]
    rows = []
    py_t = key["pycham_j60_t_s"]
    keep_py = ours.on_grid(py_t, G3["compare_dt_s"])
    runs = {"ours_dt900": "dt900", "ours_dt60": "dt60", "ours_dt900_heldNO2": "dt900_heldNO2",
            "ours_dt60_all_matched": "dt60_tight_pychamH2O_heldNO2"}
    for sp in species:
        row = {"species": sp}
        for col, tag in list(runs.items()) + [("pycham_j900", None)]:
            if tag is None:
                t_o, arr = key["pycham_j900_t_s"], key["pycham_j900_ppb"][:, names.index(sp)]
                ref_key = "pycham_j60_ppb"
            else:
                t_o = key[f"ours_{tag}_t_s"]
                arr = (key[f"ours_{tag}_sa_cum_ppb"] if sp == SA_LABEL
                       else key[f"ours_{tag}_ppb"][:, names.index(sp)])
                ref_key = "pycham_j60_ppb"
            keep = ours.on_grid(t_o, G3["compare_dt_s"])
            ia, ib = metrics.common_times(t_o[keep], py_t[keep_py])
            if ia.size != keep.sum():
                raise ValueError(f"{col}: {keep.sum() - ia.size} times without a PyCHAM match")
            x = arr[keep][ia].astype(float)[:, None]
            r = key[ref_key][keep_py][ib, names.index(sp)].astype(float)[:, None]
            t = t_o[keep][ia]
            masks = time_masks(t, sun)
            for period in ("all", "day", "night"):
                m = (r >= ABUNDANCE_THRESHOLD_PPB) & masks[period][:, None]
                row[f"nmb_{period}_{col}"] = float(metrics.normalised_mean_bias(x, r, m)[0])
            m = (r >= ABUNDANCE_THRESHOLD_PPB) & masks["all"][:, None]
            row[f"max_rel_dev_{col}"] = float(metrics.max_relative_deviation(x, r, m)[0])
            for name, th in G3["snapshots"]:
                i = int(np.argmin(np.abs(t - th * 3600.0)))
                row[f"{col}_{name}_ppb"] = float(x[i, 0])
                row[f"pycham_j60_{name}_ppb"] = float(r[i, 0])
        rows.append(row)
    return rows


def main() -> int:
    case.FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    sun = load_sun()
    stats, snaps, summ = all_species_comparison(sun)
    metrics.write_stats_csv(case.RESULTS_DIR / "species_stats.csv", stats)

    key = dict(np.load(_need(case.KEY_NPZ)))
    top = json.loads(_need(case.TOP_CONTRIBUTORS_JSON).read_text())["top_contributors"]
    snap_names = [n for n, _ in G3["snapshots"]]
    figs = []
    title = ("G3 gas phase, Level 3 day/night (Buffalo, 1-2 July 2026; NO2 held at 1 ppb; "
             "J from TOMAS-JAX/TUV-x in both models)")
    note = ("Difference panels: (run − PyCHAM J every 60 s) / PyCHAM J every 60 s, drawn where "
            "PyCHAM ≥ 1e-5 ppb. TOMAS-JAX 900 s outputs only every 900 s (markers).")
    key_sp = list(G3["key_species"]) + [SA_LABEL]
    figs.append(plots.diurnal_grid(
        key_sp, _key_series(key, key_sp), case.FIG_DIR / f"{case.FIG_PREFIX}key_species.png",
        sun=sun, units=_units(key_sp), ref_floor=_floors(key_sp, key), ncols=4, title=title,
        display_names={SA_LABEL: "SA (H2SO4 formed, cumulative)",
                       "NO2": "NO2 (held; outputs after the reset)"},
        legend_note=note, rel_cap_pct=100.0))
    held_stats, held_figs = held_analysis(key, sun)
    figs += held_figs
    top_sp = [c["species"] for c in top[:G3["n_particle_relevant"]]]
    prod_sp = list(dict.fromkeys(list(G3["product_species"]) + top_sp))
    part_names = {c["species"]: f"{c['species']} ({c['formula']}, log c* "
                                f"{round(c['log10_cstar'], 1) + 0.0:.1f})" for c in top}
    figs.append(plots.diurnal_grid(
        prod_sp, _key_series(key, prod_sp), case.FIG_DIR / f"{case.FIG_PREFIX}products.png",
        sun=sun, units=_units(prod_sp), ref_floor=_floors(prod_sp, key), ncols=4,
        title=f"{title}\nlimonene products and the top particle-phase contributors of the "
              "full Level 3 run", display_names=part_names, legend_note=note, rel_cap_pct=100.0))
    tr_rows = [{"species": sp, "label": sp,
                "unit": "molec cm$^{-3}$" if sp in RADICALS_CM3 else "ppb"}
               for sp in G3["transition_species"]]
    j_series = [
        {"label": "J every 60 s (PyCHAM reference and TOMAS-JAX 60 s)",
         "t0_s": key["j_dt60_t0_s"], "j": key["j_dt60"][:, 0], "color": COLORS["pycham"],
         "linestyle": "-", "linewidth": 2.4},
        {"label": "J every 900 s (TOMAS-JAX 900 s and PyCHAM J every 900 s)",
         "t0_s": key["j_dt900_t0_s"], "j": key["j_dt900"][:, 0], "color": COLORS["ours"],
         "linestyle": "-", "linewidth": 1.6}]
    if str(key["j_dt60_names"][0]) != "J_NO2":
        raise RuntimeError("expected J_NO2 first in the stored J subset")
    tr_series = _key_series(key, list(G3["transition_species"]))
    figs.append(plots.transitions(
        tr_rows, tr_series, j_series, case.FIG_DIR / f"{case.FIG_PREFIX}sunset_sunrise.png",
        sun=sun, window_h=G3["transition_window_h"],
        title="G3: sunset and sunrise. J frozen per coupling interval (step functions, "
              "J at each interval midpoint)"))
    figs.append(plots.scatter_named(
        snaps, snap_names, case.FIG_DIR / f"{case.FIG_PREFIX}scatter_all_species.png", sun=sun,
        threshold_ppb=ABUNDANCE_THRESHOLD_PPB,
        title="G3: every gas species above 1e-5 ppb, TOMAS-JAX vs PyCHAM (J every 60 s)"))
    figs.append(plots.ratio_named(
        snaps, snap_names, case.FIG_DIR / f"{case.FIG_PREFIX}ratio_cdf.png", sun=sun,
        title="G3: distribution of log10(TOMAS-JAX / PyCHAM) over all species above 1e-5 ppb"))
    attr, attr_figs = attribution(sun)
    figs += attr_figs
    for f in figs:
        shutil.copyfile(f, DOC_FIG_DIR / f.name)

    run_info = json.loads(_need(case.RUN_INFO_JSON).read_text())
    out = {"threshold_ppb": ABUNDANCE_THRESHOLD_PPB, "strict_threshold_ppb": STRICT_THRESHOLD_PPB,
           "agreement_band": AGREEMENT_BAND, "compare_dt_s": G3["compare_dt_s"],
           "excluded_species": list(NON_CHEMICAL_SPECIES),
           "held_species_reported_separately": list(G3["held_species"]), "sun": sun,
           "snapshots": {n: th for n, th in G3["snapshots"]},
           "runs": summ, "attribution": attr, "held_no2": held_stats,
           "key_species": key_species_table(key, key_sp + prod_sp, sun),
           "product_species": prod_sp, "particle_relevant_species": top_sp,
           "wall_s": {k: v["wall_s"] for k, v in run_info.items()
                      if isinstance(v, dict) and "wall_s" in v},
           "figures": [str(f) for f in figs]}
    (case.RESULTS_DIR / "summary.json").write_text(json.dumps(out, indent=2) + "\n")
    for label, s in summ.items():
        print(label, json.dumps({k: v for k, v in s.items()
                                 if k not in ("worst_nmb_species", "worst_max_rel_dev_species")},
                                indent=1))
    print("attribution", json.dumps({k: {kk: vv for kk, vv in v.items()
                                         if kk not in ("by_period", "key_species_nmb")}
                                     for k, v in attr.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
