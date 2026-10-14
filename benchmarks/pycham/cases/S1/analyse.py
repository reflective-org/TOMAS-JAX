"""Case S1 analysis: metrics, attribution and figures from the saved results (no runs).

    cd tomas-jax && .venv/bin/python -m benchmarks.pycham.cases.S1.analyse

Writes ``results/summary.json``, ``results/s1_stats.csv`` and five figures in
``figures/`` (copied to docs/aaar/figures/pycham/, names start with S1_). Raises if a
result file is missing.

Quantities (both c*): particle SOA (condensed surrogate), particle organic (SOA + POA),
gas-phase surrogate, particle water [ug/m3]; number-mean dry diameter (dry mass per
particle converted at one display density for both models), number-mean wet diameter
(each model's own) [nm] and the number-weighted geometric standard deviation of the dry
diameter [-]. Statistics use the harness metrics (NMB, max relative deviation) on the
common times (300 s grid), t > 0.
"""
from __future__ import annotations

import csv
import json
import shutil
import sys

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tomas_jax.box.config import AerosolStepConfig
from tomas_jax.box.coupling import mass_grid, ugm3_to_kg_cell
from tomas_jax.core.config import ICOMP, SRTH2O, SRTORG_NUC, SRTSO4, SRTSOA1
from tomas_jax.physics.density import calc_density

from ... import metrics, plotting
from ...config import (COLORS, DISPLAY_DRY_DENSITY_KG_M3, DOC_FIG_DIR, DRY_DIAMETER_LABEL,
                       FIG_DPI, FONT_SIZE, LABEL_OURS, LABEL_PYCHAM)
from ...particles import PYCHAM_WATER_DENSITY, dry_diameter_nm
from . import case, transfer
from .case import S1

MASS_THRESHOLD_UGM3 = 1.0e-5       # a time counts for a mass statistic when PyCHAM >= this
LATE_FROM_H = 1.0                  # "after 1 h" statistics (kinetic start-up excluded)
OURS_VARIANTS = ("base", "micro10", "nokelvin", "nokelvin_pytransfer", "bins80", "bins160")
PYCHAM_VARIANTS = ("base", "tight", "nokelvin", "bins160")
QUANTITIES = {                     # key: (label, unit)
    "soa": ("Particle SOA (condensed surrogate)", "µg m⁻³"),
    "org": ("Particle organic (SOA + POA)", "µg m⁻³"),
    "gas": ("Gas-phase surrogate", "µg m⁻³"),
    "water": ("Particle water", "µg m⁻³"),
    "d_dry": ("Number-mean dry diameter", "nm"),
    "d_wet": ("Number-mean wet diameter", "nm"),
    "gsd": ("Geometric std. dev. of dry diameter", "-"),
}
PAIRS = {   # name: (label, run a, run b) with runs as (model, variant); stats of a vs b
    "full": ("TOMAS-JAX vs PyCHAM (as configured, 40 bins)", ("ours", "base"), ("py", "base")),
    "kelvin_off": ("Kelvin off in both", ("ours", "nokelvin"), ("py", "nokelvin")),
    "kelvin_off_transfer": ("Kelvin off in both + PyCHAM transfer coefficient in TOMAS-JAX",
                            ("ours", "nokelvin_pytransfer"), ("py", "nokelvin")),
    "bins160": ("160 bins in both", ("ours", "bins160"), ("py", "bins160")),
    "ours_numerics": ("TOMAS-JAX micro-step 10 s vs 60 s", ("ours", "micro10"), ("ours", "base")),
    "pycham_numerics": ("PyCHAM update step 10 s + rtol 1e-6 vs default", ("py", "tight"),
                        ("py", "base")),
    "ours_resolution": ("TOMAS-JAX 160 vs 40 bins", ("ours", "bins160"), ("ours", "base")),
    "ours_resolution80": ("TOMAS-JAX 80 vs 40 bins", ("ours", "bins80"), ("ours", "base")),
    "pycham_resolution": ("PyCHAM 160 vs 40 bins", ("py", "bins160"), ("py", "base")),
    "kelvin_effect_ours": ("TOMAS-JAX with vs without Kelvin", ("ours", "base"),
                           ("ours", "nokelvin")),
    "kelvin_effect_pycham": ("PyCHAM with vs without Kelvin", ("py", "base"), ("py", "nokelvin")),
}
PAIR_STYLE = {   # pairs drawn in the attribution figure: colour, linestyle, width
    "full": (COLORS["ours"], "-", 2.4),
    "kelvin_off": (COLORS["attr_kelvin_off"], "-.", 1.8),
    "kelvin_off_transfer": (COLORS["attr_matched"], "-", 1.8),
    "bins160": (COLORS["ours_alt"], "--", 1.7),
    "ours_numerics": (COLORS["attr_solver_ours"], ":", 1.7),
    "pycham_numerics": (COLORS["attr_solver_pycham"], ":", 1.7),
}
SNAP_STYLE = {   # size-distribution lines: (model, variant) -> label, colour, ls, lw, z
    ("py", "base"): (f"{LABEL_PYCHAM} (moving-centre, 40 bins)", COLORS["pycham"], "-", 2.6, 1),
    ("ours", "base"): (f"{LABEL_OURS} (40 fixed bins, preset)", COLORS["ours"], "-", 1.7, 3),
    ("ours", "bins160"): (f"{LABEL_OURS} (160 fixed bins, summed onto the 40-bin grid)",
                          COLORS["ours_alt"], "--", 1.6, 2),
}
SHORT_NAMES = {"soa": "SOA", "org": "SOA + POA", "gas": "Gas-phase surrogate",
               "d_dry": "Number-mean dry diameter", "d_wet": "Number-mean wet diameter",
               "gsd": "GSD of dry diameter", "water": "Particle water"}
DIAMETER_TICKS = (10, 20, 50, 100, 200, 500)


def _need(path):
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run benchmarks.pycham.cases.S1.run first")
    return path


def load_runs() -> dict:
    runs = {}
    for c in S1["cstar_ugm3"]:
        for v in OURS_VARIANTS:
            runs[(c, "ours", v)] = dict(np.load(_need(case.ours_npz(c, v))))
        for v in PYCHAM_VARIANTS:
            runs[(c, "py", v)] = dict(np.load(_need(case.pycham_npz(c, v))))
    return runs


def series(d: dict) -> dict:
    """Quantities of one run -> {key: (t_s, values)} (full time resolution)."""
    t = d["t_s"]
    return {"soa": (t, d["soa_ugm3"]), "org": (t, d["soa_ugm3"] + d["poa_ugm3"]),
            "gas": (t, d["gas_surr_ugm3"]), "water": (t, d["water_ugm3"]),
            "d_dry": (t, d["d_dry_mean_nm"]), "d_wet": (t, d["d_wet_mean_nm"]),
            "gsd": (t, d["gsd_dry"]), "n_total": (t, d["n_total_cm3"])}


def pair_stats(sa: dict, sb: dict) -> dict:
    """Per-quantity stats of run a against reference b on their common times (t > 0)."""
    out = {}
    for q in QUANTITIES:
        (ta, a), (tb, b) = sa[q], sb[q]
        ia, ib = metrics.common_times(ta, tb)
        t, x, r = ta[ia], np.asarray(a[ia], float)[:, None], np.asarray(b[ib], float)[:, None]
        thr = MASS_THRESHOLD_UGM3 if QUANTITIES[q][1] == "µg m⁻³" else 0.0
        mask = metrics.valid_mask(r, thr)
        late = mask & (t[:, None] >= LATE_FROM_H * 3600.0)
        rel = metrics.relative_difference(x, r, thr)[:, 0]
        out[q] = {
            "nmb": float(metrics.normalised_mean_bias(x, r, mask)[0]),
            "max_rel_dev": float(metrics.max_relative_deviation(x, r, mask)[0]),
            "max_rel_dev_after_1h": float(metrics.max_relative_deviation(x, r, late)[0]),
            "rel_diff_at": {f"{h:g}h": float(rel[np.argmin(np.abs(t - h * 3600.0))])
                            for h in S1["snapshot_times_h"] if h > 0},
            "a_at": {f"{h:g}h": float(x[np.argmin(np.abs(t - h * 3600.0)), 0])
                     for h in S1["snapshot_times_h"]},
            "b_at": {f"{h:g}h": float(r[np.argmin(np.abs(t - h * 3600.0)), 0])
                     for h in S1["snapshot_times_h"]},
            "t_s": t, "rel": rel,
        }
    return out


def _bin_index(d: dict, t_h: float) -> int:
    k = int(np.argmin(np.abs(d["t_bin_s"] - t_h * 3600.0)))
    if abs(d["t_bin_s"][k] - t_h * 3600.0) > 1.0:
        raise ValueError(f"no per-bin output at {t_h} h")
    return k


def fixed_bin_density(d: dict, k: int, nbins_out: int | None = None
                      ) -> tuple[np.ndarray, np.ndarray]:
    """TOMAS-JAX fixed bins: (dry-diameter edges [nm], dN/dlog10D per bin [cm-3]).

    Edges are the model's own dry-mass bin bounds at the display density; checks that
    every bin's mean dry mass lies inside its bounds. ``nbins_out`` sums nested bins onto
    a coarser grid of the same range (160 -> 40: four bins each, exact)."""
    n = d["n_bin_cm3"][k].astype(float)
    edges = mass_grid(n.size)
    with np.errstate(divide="ignore", invalid="ignore"):
        m = d["dry_ugm3_bin"][k].astype(float) / n * 1e-15      # kg per particle
    ok = n > 1e-6
    if np.any((m[ok] < edges[:-1][ok] * (1 - 1e-5)) | (m[ok] > edges[1:][ok] * (1 + 1e-5))):
        raise ValueError("TOMAS-JAX bin mean mass outside its bounds")
    if nbins_out is not None and nbins_out != n.size:
        f = n.size // nbins_out
        coarse = mass_grid(nbins_out)
        if f * nbins_out != n.size or not np.allclose(edges[::f], coarse, rtol=1e-9):
            raise ValueError(f"{n.size} bins do not nest in {nbins_out}")
        n, edges = n.reshape(nbins_out, f).sum(axis=1), coarse
    d_edges = dry_diameter_nm(edges)
    return d_edges, n / np.diff(np.log10(d_edges))


def moving_centre_density(d: dict, k: int) -> tuple[np.ndarray, np.ndarray]:
    """PyCHAM moving-centre bins: (dry diameter of each populated bin [nm], dN/dlog10D).

    Each population sits at its exact size; its log-width reaches halfway to the
    neighbouring populations (the outermost widths mirror their neighbour's)."""
    n = d["n_bin_cm3"][k].astype(float)
    dd = d["d_dry_nm_bin"][k].astype(float)
    ok = (n > 1e-6) & np.isfinite(dd)
    order = np.argsort(dd[ok])
    x, nn = np.log10(dd[ok][order]), n[ok][order]
    mid = 0.5 * (x[1:] + x[:-1])
    edges = np.concatenate([[2 * x[0] - mid[0]], mid, [2 * x[-1] - mid[-1]]])
    return 10.0 ** x, nn / np.diff(edges)


def _diameter_axis(ax, axis: str = "x") -> None:
    from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
    a = ax.xaxis if axis == "x" else ax.yaxis
    a.set_major_locator(FixedLocator(DIAMETER_TICKS))
    a.set_major_formatter(ScalarFormatter())
    a.set_minor_formatter(NullFormatter())


# --- diagnostics: water, Kelvin, condensation sink ---------------------------------------
def _ours_mk_cell(d: dict, k: int) -> np.ndarray:
    """Per-bin composition of one TOMAS-JAX per-bin output time, Mk layout [kg/cell]."""
    mk = np.zeros((d["n_bin_cm3"].shape[1], ICOMP))
    soa, poa = d["soa_ugm3_bin"][k].astype(float), d["poa_ugm3_bin"][k].astype(float)
    mk[:, SRTSOA1], mk[:, SRTORG_NUC] = soa, poa
    mk[:, SRTSO4] = d["dry_ugm3_bin"][k].astype(float) - soa - poa
    mk[:, SRTH2O] = d["water_ugm3_bin"][k].astype(float)
    return ugm3_to_kg_cell(mk, AerosolStepConfig().boxvol_cm3)


def condensation_sinks(d_ours: dict, d_py: dict, cstar: float, temp: float, pres: float) -> dict:
    """Organic CS [1/s] at the per-bin output times: each model's own formula on its own
    state, and PyCHAM's formula on the TOMAS-JAX state (isolates the formula)."""
    cs_ours, cs_cross = [], []
    for k in range(d_ours["t_bin_s"].size):
        n = d_ours["n_bin_cm3"][k].astype(float) * 1e6
        dp = d_ours["d_wet_nm_bin"][k].astype(float) * 1e-9
        rho = np.asarray(calc_density(jnp.asarray(_ours_mk_cell(d_ours, k))))
        cs_ours.append(transfer.tomas_sink(n, dp, rho, temp, pres, cstar).sum())
        cs_cross.append(float(np.sum(transfer.pycham_sink(n, dp, temp, pres))))
    n_py = np.nan_to_num(d_py["n_bin_cm3"].astype(float)) * 1e6
    dp_py = np.nan_to_num(d_py["d_wet_nm_bin"].astype(float)) * 1e-9
    cs_py = np.sum(np.where(n_py > 0, transfer.pycham_sink(n_py, np.maximum(dp_py, 1e-12),
                                                         temp, pres), 0.0), axis=1)
    return {"t_ours_s": d_ours["t_bin_s"], "ours": np.array(cs_ours),
            "pycham_formula_on_ours": np.array(cs_cross), "t_py_s": d_py["t_bin_s"],
            "pycham": cs_py}


def kelvin_and_water(d_ours: dict, d_py: dict, t_h: float, temp: float) -> dict:
    """Per-bin water/dry ratio and surrogate Kelvin factor at one time in both models."""
    ko, kp = _bin_index(d_ours, t_h), _bin_index(d_py, t_h)
    n_o = d_ours["n_bin_cm3"][ko].astype(float)
    ok_o = n_o > 1.0
    dw_o = d_ours["d_wet_nm_bin"][ko].astype(float)
    n_p = d_py["n_bin_cm3"][kp].astype(float)
    ok_p = n_p > 1.0
    dry_p = d_py["dry_ugm3_bin"][kp].astype(float)
    wat_p = d_py["water_ugm3_bin"][kp].astype(float)
    soa_p, poa_p = d_py["soa_ugm3_bin"][kp].astype(float), d_py["poa_ugm3_bin"][kp].astype(float)
    names = [str(x) for x in d_py["names"]]
    mw = d_py["mw_g_mol"]
    rho = mw / d_py["molar_volume_cm3_mol"] * 1e3                # kg/m3
    order = [names.index(x) for x in (S1["surrogate_name"], S1["poa_name"], "core", "H2O")]
    masses = np.stack([soa_p, poa_p, dry_p - soa_p - poa_p, wat_p])
    tot = np.maximum(masses.sum(axis=0), 1e-300)
    av_mw = (masses * mw[order][:, None]).sum(axis=0) / tot      # as PyCHAM part_prop
    rho_p = (masses * rho[order][:, None]).sum(axis=0) / tot
    dw_p = d_py["d_wet_nm_bin"][kp].astype(float)
    return {
        "ours_d_dry_nm": d_ours["d_dry_nm_bin"][ko][ok_o].astype(float),
        "ours_d_wet_nm": dw_o[ok_o],
        "ours_water_over_dry": (d_ours["water_ugm3_bin"][ko][ok_o].astype(float)
                                / d_ours["dry_ugm3_bin"][ko][ok_o].astype(float)),
        "ours_kelvin": transfer.kelvin_tomas(dw_o[ok_o] * 1e-9, temp),
        "ours_n": n_o[ok_o],
        "py_d_dry_nm": d_py["d_dry_nm_bin"][kp][ok_p].astype(float),
        "py_d_wet_nm": dw_p[ok_p],
        "py_water_over_dry": wat_p[ok_p] / dry_p[ok_p],
        "py_kelvin": transfer.kelvin_pycham(dw_p[ok_p] * 1e-9, temp, av_mw[ok_p], rho_p[ok_p]),
        "py_n": n_p[ok_p],
    }


def density_rule_difference(d_ours: dict, t_h: float, rho_dry_kg_m3: float) -> float:
    """Max |rho_PyCHAM-rule / rho_TOMAS - 1| over populated bins of the TOMAS-JAX state at
    ``t_h``: TOMAS-JAX's Tang density vs PyCHAM's additive volumes (dry at the effective
    dry density of the set-up, water 1000 kg/m3) on the same composition."""
    k = _bin_index(d_ours, t_h)
    mk = _ours_mk_cell(d_ours, k)
    rho_t = np.asarray(calc_density(jnp.asarray(mk)))
    m_dry, m_w = mk[:, :SRTH2O].sum(axis=1), mk[:, SRTH2O]
    with np.errstate(divide="ignore", invalid="ignore"):
        rho_p = (m_dry + m_w) / (m_dry / rho_dry_kg_m3 + m_w / PYCHAM_WATER_DENSITY)
    ok = d_ours["n_bin_cm3"][k].astype(float) > 1.0
    return float(np.max(np.abs(rho_p[ok] / rho_t[ok] - 1.0)))


def _window_extreme(st: dict, h0: float, h1: float) -> float:
    """Signed relative difference with the largest magnitude for h0 <= t <= h1."""
    t_h = st["t_s"] / 3600.0
    sel = (t_h >= h0) & (t_h <= h1) & np.isfinite(st["rel"])
    r = st["rel"][sel]
    return float(r[np.argmax(np.abs(r))])


def measured(stats: dict, diag: dict, tc: dict, runs: dict, setup1: dict) -> dict:
    """The measured numbers the mismatch list (case.mismatches) quotes."""
    c1, c100 = S1["cstar_ugm3"]
    kel = lambda th, dw: tuple(diag[f"cstar1_{th:g}h_kelvin_at_wet_diameter"][f"{dw:g}nm"][m]  # noqa: E731
                               for m in ("ours", "pycham"))
    by_d = tc["by_wet_diameter_nm"]
    at = lambda c, pair, q, h: stats[c][pair][q]["rel_diff_at"][f"{h:g}h"]  # noqa: E731
    water = lambda h: (stats[c1]["full"]["water"]["a_at"][f"{h:g}h"],       # noqa: E731
                       stats[c1]["full"]["water"]["b_at"][f"{h:g}h"])
    mode0 = diag["cstar1_0h_at_number_mode"]
    cs0 = diag["cstar1"]["cs_per_s"]["0h"]
    return {
        "kelvin_50nm_0h": kel(0, 50), "kelvin_50nm_12h": kel(12, 50),
        "kelvin_100nm_12h": kel(12, 100),
        "kelvin_effect_c100": (at(c100, "kelvin_effect_ours", "soa", 12),
                               at(c100, "kelvin_effect_pycham", "soa", 12)),
        "soa_c100_12h_full": at(c100, "full", "soa", 12),
        "soa_c100_12h_kelvin_off": at(c100, "kelvin_off", "soa", 12),
        "transfer": {"d_ours": tc["diffusivity_m2_s"]["tomas_jax"],
                     "d_pycham": tc["diffusivity_m2_s"]["pycham"],
                     "ratio_1000nm": by_d["1000"]["tomas_over_pycham"],
                     "ratio_54nm": by_d["54"]["tomas_over_pycham"],
                     "ratio_1nm": by_d["1"]["tomas_over_pycham"],
                     "ours_over_kinetic_1nm": by_d["1"]["tomas_over_kinetic_limit"]},
        "cs0": (cs0["ours"], cs0["pycham"], cs0["pycham_formula_on_ours_state"]),
        "soa_c1_1to3h_full": _window_extreme(stats[c1]["full"]["soa"], 1, 3),
        "soa_c1_1to3h_kelvin_off": _window_extreme(stats[c1]["kelvin_off"]["soa"], 1, 3),
        "soa_c1_1to3h_matched": _window_extreme(stats[c1]["kelvin_off_transfer"]["soa"], 1, 3),
        "water_c1_0h": water(0), "water_c1_12h": water(12),
        "water_mode_c1_0h": (mode0["ours"]["water_over_dry"], mode0["pycham"]["water_over_dry"]),
        "d_wet_c100_12h": at(c100, "full", "d_wet", 12),
        "water_c1_12h_kelvin_off": at(c1, "kelvin_off", "water", 12),
        "density_max_rel_12h": density_rule_difference(runs[(c1, "ours", "base")], 12.0,
                                                       setup1["rho_dry_g_cm3"] * 1e3),
        "d_dry_c1_12h": {"40": at(c1, "full", "d_dry", 12), "160": at(c1, "bins160", "d_dry", 12),
                         "pycham_160_vs_40": at(c1, "pycham_resolution", "d_dry", 12)},
        "ours_numerics_soa_after_1h": max(stats[c]["ours_numerics"]["soa"]["max_rel_dev_after_1h"]
                                          for c in S1["cstar_ugm3"]),
        "pycham_numerics_soa_after_1h": max(
            stats[c]["pycham_numerics"]["soa"]["max_rel_dev_after_1h"] for c in S1["cstar_ugm3"]),
    }


def _at_mode(kw: dict, model: str, key: str) -> float:
    """Value at the bin with the most particles."""
    return float(kw[f"{model}_{key}"][np.argmax(kw[f"{model}_n"])])


# --- figures ----------------------------------------------------------------------------
def fig_size_distributions(runs: dict, path) -> None:
    plotting.style()
    times, cs = S1["snapshot_times_h"], S1["cstar_ugm3"]
    fig, axes = plt.subplots(len(cs), len(times), figsize=(3.7 * len(times), 3.4 * len(cs)),
                             sharex=True, sharey=True, constrained_layout=True, squeeze=False)
    for row, c in enumerate(cs):
        for col, th in enumerate(times):
            ax = axes[row, col]
            for (model, variant), (label, color, ls, lw, z) in SNAP_STYLE.items():
                d = runs[(c, model, variant)]
                k = _bin_index(d, th)
                if model == "py":
                    x, y = moving_centre_density(d, k)
                    ax.plot(x, y, color=color, ls=ls, lw=lw, marker="o", ms=3.5, label=label,
                            zorder=z)
                else:
                    x, y = fixed_bin_density(d, k, nbins_out=case.scenario(c).numerics.nbins)
                    ax.step(x, np.append(y, 0.0), where="post", color=color, ls=ls, lw=lw,
                            label=label, zorder=z)
            ax.set_xscale("log")
            ax.set_xlim(10, 600)
            _diameter_axis(ax)
            ax.set_title(f"c* = {c:g} µg m⁻³, t = {th:g} h", loc="left")
            if row == len(cs) - 1:
                ax.set_xlabel(DRY_DIAMETER_LABEL)
            if col == 0:
                ax.set_ylabel("dN/dlog₁₀D [cm⁻³]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    fig.suptitle("S1 number size distributions (dry diameter = dry mass per particle at "
                 f"{DISPLAY_DRY_DENSITY_KG_M3:.0f} kg m⁻³ for both models; TOMAS-JAX as "
                 "fixed-bin averages, PyCHAM at each population's exact size)",
                 fontsize=FONT_SIZE)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def fig_timeseries(runs: dict, keys: list[str], path, title: str, ncols: int) -> None:
    """plotting.timeseries_grid over (quantity, c*) panels, base runs."""
    names, units, disp = [], {}, {}
    vals = {"ours": {}, "py": {}}
    for c in S1["cstar_ugm3"]:
        so, sp = series(runs[(c, "ours", "base")]), series(runs[(c, "py", "base")])
        for q in keys:
            key = f"{q}_c{c:g}"
            names.append(key)
            units[key] = f"[{QUANTITIES[q][1]}]"
            disp[key] = f"{SHORT_NAMES[q]}, c* = {c:g}"
            for m, s in (("ours", so), ("py", sp)):
                vals[m][key] = s[q]
    axes_t = {}
    for m in ("ours", "py"):          # timeseries_grid takes one time axis per series
        ts = [t for t, _ in vals[m].values()]
        if any(len(t) != len(ts[0]) or np.any(t != ts[0]) for t in ts):
            raise ValueError(f"{m}: keys of one figure must share a time axis")
        axes_t[m] = ts[0]
    ser = [plotting.pycham_series(axes_t["py"], {k: np.asarray(v, float)
                                                 for k, (_, v) in vals["py"].items()}),
           plotting.ours_series(axes_t["ours"], {k: np.asarray(v, float)
                                                 for k, (_, v) in vals["ours"].items()})]
    plotting.timeseries_grid(names, ser, path, units=units, ncols=ncols, title=title,
                             display_names=disp)


def fig_attribution(stats: dict, path) -> None:
    plotting.style()
    cs, qs = S1["cstar_ugm3"], ("soa", "gas", "d_dry", "gsd")
    fig, axes = plt.subplots(len(cs), len(qs), figsize=(4.6 * len(qs), 3.7 * len(cs)),
                             sharex=True, constrained_layout=True, squeeze=False)
    for row, c in enumerate(cs):
        for col, q in enumerate(qs):
            ax = axes[row, col]
            peak = 0.0
            for name, (color, ls, lw) in PAIR_STYLE.items():
                st = stats[c][name][q]
                ax.plot(st["t_s"] / 3600.0, 100.0 * st["rel"], color=color, ls=ls, lw=lw,
                        label=PAIRS[name][0])
                ok = st["t_s"] >= 0.25 * 3600.0
                peak = max(peak, float(np.nanmax(np.abs(st["rel"][ok]))) * 100.0)
            ax.axhline(0.0, color=COLORS["reference"], lw=0.8)
            lim = max(0.05, 1.15 * peak)
            ax.set_ylim(-lim, lim)
            ax.set_title(f"{QUANTITIES[q][0]}, c* = {c:g}", loc="left", fontsize=FONT_SIZE - 1)
            ax.set_ylabel("(A − B) / B [%]")
            if row == len(cs) - 1:
                ax.set_xlabel("time [h]")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=3, frameon=False)
    fig.suptitle("S1 attribution: relative difference of run A against run B "
                 "(y range set by t ≥ 15 min)", fontsize=FONT_SIZE)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def fig_diagnostics(kw0: dict, kw12: dict, cs_by_c: dict, path) -> None:
    plotting.style()
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.8), constrained_layout=True)
    for ax, key, xkey, ylabel, title, logy in (
            (axes[0], "water_over_dry", "d_dry_nm", "particle water / dry mass [-]",
             "Particle water (c* = 1, RH 0.5)", False),
            (axes[1], "kelvin", "d_wet_nm", "Kelvin factor of the surrogate [-]",
             "Kelvin factor (c* = 1)", True)):
        for kw, ls, mk, tag in ((kw0, "-", "o", "0 h"), (kw12, "--", "^", "12 h")):
            ax.plot(kw[f"ours_{xkey}"], kw[f"ours_{key}"], color=COLORS["ours"], ls=ls,
                    marker=mk, ms=3, lw=1.6, label=f"{LABEL_OURS}, {tag}")
            ax.plot(kw[f"py_{xkey}"], kw[f"py_{key}"], color=COLORS["pycham"], ls=ls,
                    marker=mk, ms=3, lw=2.2, label=f"{LABEL_PYCHAM}, {tag}")
        ax.set_xscale("log")
        _diameter_axis(ax)
        if logy:
            ax.set_yscale("log")
            from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
            ax.yaxis.set_major_locator(FixedLocator((1.0, 1.05, 1.1, 1.2, 1.3, 1.5, 1.7)))
            ax.yaxis.set_major_formatter(ScalarFormatter())
            ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel(DRY_DIAMETER_LABEL if "dry" in xkey
                      else "wet diameter (each model's own) [nm]")
        ax.set_ylabel(ylabel)
        ax.set_title(title, loc="left")
        ax.legend(frameon=False, fontsize=FONT_SIZE - 2)
    ax = axes[2]
    for c, ls in zip(S1["cstar_ugm3"], ("-", "--")):
        cs = cs_by_c[c]
        ax.plot(cs["t_ours_s"] / 3600.0, cs["ours"] * 1e3, color=COLORS["ours"], ls=ls, lw=1.8,
                label=f"{LABEL_OURS} (own formula), c* = {c:g}")
        ax.plot(cs["t_py_s"] / 3600.0, cs["pycham"] * 1e3, color=COLORS["pycham"], ls=ls, lw=2.4,
                label=f"{LABEL_PYCHAM} (own formula), c* = {c:g}")
        ax.plot(cs["t_ours_s"] / 3600.0, cs["pycham_formula_on_ours"] * 1e3,
                color=COLORS["attr_transfer"], ls=ls, lw=1.4,
                label=f"PyCHAM formula on {LABEL_OURS} state, c* = {c:g}")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("organic condensation sink [10⁻³ s⁻¹]")
    ax.set_title("Gas-to-particle transfer", loc="left")
    ax.legend(frameon=False, fontsize=FONT_SIZE - 3)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)


def _clean(obj):
    """Drop arrays from the stats for JSON."""
    if isinstance(obj, dict):
        return {k: _clean(v) for k, v in obj.items() if k not in ("t_s", "rel")}
    return obj


def write_csv(stats: dict, path) -> None:
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["cstar_ugm3", "pair", "quantity", "nmb", "max_rel_dev", "max_rel_dev_after_1h",
                    "rel_diff_3h", "rel_diff_6h", "rel_diff_12h", "a_12h", "b_12h", "unit"])
        for c, by_pair in stats.items():
            for pair, by_q in by_pair.items():
                for q, st in by_q.items():
                    w.writerow([f"{c:g}", pair, q, f"{st['nmb']:.4g}", f"{st['max_rel_dev']:.4g}",
                                f"{st['max_rel_dev_after_1h']:.4g}",
                                *(f"{st['rel_diff_at'][h]:.4g}" for h in ("3h", "6h", "12h")),
                                f"{st['a_at']['12h']:.6g}", f"{st['b_at']['12h']:.6g}",
                                QUANTITIES[q][1]])


def transfer_comparison(temp: float, pres: float) -> dict:
    """Per-particle transfer coefficient of the two formulas for one particle per m3,
    against each other and against the gas-kinetic collision limit pi r^2 c N (alpha = 1)."""
    from vbs_jax import DEFAULT_VBS_CONFIG as cfg
    from vbs_jax.gas_properties import calc_gas_diffusivity, calc_mean_molecular_speed
    c_bar = float(calc_mean_molecular_speed(temp, 200.0))
    out = {"diffusivity_m2_s": {"tomas_jax": float(calc_gas_diffusivity(temp, pres, 200.0,
                                                                        cfg.sv)),
                                "pycham": float(transfer.pycham_diffusivity(temp, pres))},
           "by_wet_diameter_nm": {}}
    for dp_nm in (1.0, 10.0, 54.0, 100.0, 200.0, 1000.0):
        n, dp = np.array([1.0]), np.array([dp_nm * 1e-9])
        k_o = float(transfer.tomas_sink(n, dp, np.array([1700.0]), temp, pres, 1.0)[0])
        k_p = float(transfer.pycham_sink(n, dp, temp, pres)[0])
        kin = np.pi * (dp[0] / 2.0) ** 2 * c_bar
        out["by_wet_diameter_nm"][f"{dp_nm:g}"] = {
            "tomas_over_pycham": k_o / k_p, "tomas_over_kinetic_limit": k_o / kin,
            "pycham_over_kinetic_limit": k_p / kin}
    return out


def diagnostics(runs: dict, sers: dict, cs_by_c: dict, kw: dict) -> dict:
    out = {}
    for c in S1["cstar_ugm3"]:
        cs = cs_by_c[c]
        pick = lambda t, v, h: float(v[np.argmin(np.abs(t - h * 3600.0))])  # noqa: E731
        out[f"cstar{c:g}"] = {
            "cs_per_s": {f"{h:g}h": {
                "ours": pick(cs["t_ours_s"], cs["ours"], h),
                "pycham": pick(cs["t_py_s"], cs["pycham"], h),
                "pycham_formula_on_ours_state": pick(cs["t_ours_s"], cs["pycham_formula_on_ours"],
                                                     h)} for h in (0.0, 3.0, 12.0)},
            "n_total_cm3_end": {m: float(sers[(c, m, "base")]["n_total"][1][-1])
                                for m in ("ours", "py")},
            "size_12h_by_resolution": {
                f"{m}_{v}": {"d_dry_mean_nm": float(sers[(c, m, v)]["d_dry"][1][-1]),
                             "gsd_dry": float(sers[(c, m, v)]["gsd"][1][-1]),
                             "soa_ugm3": float(sers[(c, m, v)]["soa"][1][-1])}
                for m, v in (("ours", "base"), ("ours", "bins80"), ("ours", "bins160"),
                             ("py", "base"), ("py", "bins160"))},
            "soa_12h_by_variant_ugm3": {f"{m}_{v}": float(sers[(c, m, v)]["soa"][1][-1])
                                        for (cc, m, v) in runs if cc == c},
            "gas_12h_by_variant_ugm3": {f"{m}_{v}": float(sers[(c, m, v)]["gas"][1][-1])
                                        for (cc, m, v) in runs if cc == c},
        }
    for th, k in kw.items():
        out[f"cstar1_{th:g}h_kelvin_at_wet_diameter"] = {
            f"{dw:g}nm": {m: float(np.exp(np.interp(np.log(dw), np.log(k[f"{s}_d_wet_nm"]),
                                                    np.log(k[f"{s}_kelvin"]))))
                          for m, s in (("ours", "ours"), ("pycham", "py"))}
            for dw in (30.0, 50.0, 100.0, 200.0)}
        out[f"cstar1_{th:g}h_at_number_mode"] = {
            m: {key: _at_mode(k, short, key) for key in
                ("d_dry_nm", "d_wet_nm", "water_over_dry", "kelvin")}
            for m, short in (("ours", "ours"), ("pycham", "py"))}
    return out


def main() -> int:
    runs = load_runs()
    case.FIG_DIR.mkdir(parents=True, exist_ok=True)
    env = case.scenario(S1["cstar_ugm3"][0]).environment
    temp, pres = env.temp_k, env.pres_pa
    sers = {k: series(v) for k, v in runs.items()}
    stats = {c: {name: pair_stats(sers[(c,) + a], sers[(c,) + b])
                 for name, (_, a, b) in PAIRS.items()} for c in S1["cstar_ugm3"]}
    cs_by_c = {c: condensation_sinks(runs[(c, "ours", "base")], runs[(c, "py", "base")], c,
                                     temp, pres) for c in S1["cstar_ugm3"]}
    c1 = S1["cstar_ugm3"][0]
    kw = {th: kelvin_and_water(runs[(c1, "ours", "base")], runs[(c1, "py", "base")], th, temp)
          for th in (0.0, 12.0)}

    figs = {
        "S1_size_distributions.png": lambda p: fig_size_distributions(runs, p),
        "S1_mass_timeseries.png": lambda p: fig_timeseries(
            runs, ["soa", "org", "gas"], p, "S1: organic particle mass and gas-phase surrogate "
            "(lower panels: (TOMAS-JAX − PyCHAM) / PyCHAM)", ncols=3),
        "S1_mean_diameter.png": lambda p: fig_timeseries(
            runs, ["d_dry", "d_wet", "gsd"], p, "S1: number-mean diameter and width (dry: same "
            "mass-to-size conversion for both; wet: each model's own)", ncols=3),
        "S1_attribution.png": lambda p: fig_attribution(stats, p),
        "S1_diagnostics.png": lambda p: fig_diagnostics(kw[0.0], kw[12.0], cs_by_c, p),
    }
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, make in figs.items():
        path = case.FIG_DIR / name
        make(path)
        shutil.copyfile(path, DOC_FIG_DIR / name)

    setup = json.loads(_need(case.SETUP_JSON).read_text())
    info = json.loads(_need(case.RUN_INFO_JSON).read_text())
    diag = diagnostics(runs, sers, cs_by_c, kw)
    tcomp = transfer_comparison(temp, pres)
    meas = measured(stats, diag, tcomp, runs, setup[f"cstar{c1:g}"])
    summary = {
        "case": S1["description"],
        "quantities": {k: {"label": v[0], "unit": v[1]} for k, v in QUANTITIES.items()},
        "pairs": {k: v[0] for k, v in PAIRS.items()},
        "stats": {f"cstar{c:g}": _clean(stats[c]) for c in S1["cstar_ugm3"]},
        "diagnostics": diag,
        "transfer_coefficient": tcomp,
        "measured_for_mismatches": meas,
        "pycham_setup": {c: {k: v for k, v in d.items() if not isinstance(v, list)}
                         for c, d in setup.items()},
        "runtime_s": {k: v["wall_s"] for k, v in info.items()
                      if isinstance(v, dict) and "wall_s" in v},
        "mismatches": case.mismatches(setup[f"cstar{c1:g}"], meas),
    }
    case.SUMMARY_JSON.write_text(json.dumps(summary, indent=2, default=float) + "\n")
    write_csv(stats, case.STATS_CSV)
    for c in S1["cstar_ugm3"]:
        for pair in ("full", "kelvin_off", "kelvin_off_transfer", "bins160"):
            st = stats[c][pair]
            print(f"c* = {c:g} {pair}: " + "; ".join(
                f"{q} NMB {100 * st[q]['nmb']:+.2f}% (12 h {100 * st[q]['rel_diff_at']['12h']:+.2f}%)"
                for q in QUANTITIES))
    print(f"figures: {case.FIG_DIR} (copied to {DOC_FIG_DIR})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
