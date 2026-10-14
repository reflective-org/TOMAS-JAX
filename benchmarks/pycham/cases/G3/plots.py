"""G3 figures that the shared ``plotting`` module cannot draw as is.

* night shading (SZA >= 90 deg) and a local-time axis on diurnal time series
  (``plotting.timeseries_grid`` has no background/axis hook),
* the difference panel masked where PyCHAM is below the abundance threshold (night-time
  OH/NO are ~0 and their relative difference is meaningless),
* scatter / ratio figures at *named* times (sunset, midnight, noon) instead of "t = 9 h",
* a sunset/sunrise zoom with the prescribed J staircases.

Colours, labels and the base style come from ``config`` / ``plotting``.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from ...config import (AGREEMENT_BAND, COLORS, FIG_DPI, FONT_SIZE, LABEL_OURS,  # noqa: E402
                       LABEL_PYCHAM, SNAPSHOT_COLORS)
from ...metrics import common_times  # noqa: E402
from ...plotting import SNAPSHOT_LINESTYLES, Series, style  # noqa: E402

NIGHT_COLOR = COLORS["night"]            # neutral background, not a data colour
SUN_LINE_COLOR = COLORS["sun_event"]


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def shade_night(ax, sun: dict) -> None:
    ax.axvspan(sun["t_sunset_h"], sun["t_sunrise_h"], color=NIGHT_COLOR, lw=0, zorder=0)


def local_time_axis(ax, sun: dict, t_max_h: float, step_h: float = 3.0) -> None:
    ticks = np.arange(0.0, t_max_h + 1e-9, step_h)
    ax.set_xticks(ticks)
    t0_local = sun["start_local_h"]
    ax.set_xticklabels([f"{int((t0_local + t) % 24):02d}" for t in ticks])
    ax.set_xlim(0.0, t_max_h)


def _rel_diff(s: Series, ref: Series, sp: str, ref_floor: float):
    ia, ib = common_times(s.t_s, ref.t_s)
    if ia.size != s.t_s.size:
        raise ValueError(f"{s.label}: {s.t_s.size - ia.size} output times have no "
                         f"{ref.label} counterpart (grids must nest)")
    r = ref.values[sp][ib].astype(float)
    x = s.values[sp][ia].astype(float)
    rel = np.full(ia.size, np.nan)
    ok = r >= ref_floor
    rel[ok] = (x[ok] - r[ok]) / r[ok]
    return s.t_s[ia] / 3600.0, 100.0 * rel


def diurnal_grid(species: list[str], series: list[Series], path: Path, *, sun: dict,
                 units: dict[str, str], ref_floor: dict[str, float], ncols: int = 4,
                 title: str = "", display_names: dict[str, str] | None = None,
                 rel_cap_pct: float | None = None, legend_note: str = "") -> Path:
    """Values (top) and 100*(run - PyCHAM)/PyCHAM (bottom) per species, night shaded.

    The difference is drawn where PyCHAM >= ``ref_floor[sp]`` (same units as the
    values). ``rel_cap_pct`` caps the symmetric difference range; points beyond it are
    drawn at the edge as open triangles (so a capped spike is visible, not hidden).
    """
    refs = [s for s in series if s.is_reference]
    if len(refs) != 1:
        raise ValueError("exactly one reference (PyCHAM) series is required")
    ref = refs[0]
    for s in series:
        missing = [sp for sp in species if sp not in s.values]
        if missing:
            raise KeyError(f"{s.label} lacks {missing}")
    style()
    t_max = max(float(s.t_s.max()) for s in series) / 3600.0
    nrows = int(np.ceil(len(species) / ncols))
    fig = plt.figure(figsize=(3.7 * ncols, 3.6 * nrows + 0.6), constrained_layout=True)
    outer = fig.add_gridspec(nrows, ncols)
    handles = None
    for k, sp in enumerate(species):
        inner = outer[k // ncols, k % ncols].subgridspec(2, 1, height_ratios=(2.4, 1), hspace=0.05)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)
        for a in (ax, axr):
            shade_night(a, sun)
        for s in series:
            kw = {"marker": s.marker, "markersize": 3.5} if s.marker else {}
            ax.plot(s.t_s / 3600.0, s.values[sp], color=s.color, ls=s.linestyle,
                    lw=s.linewidth, label=s.label, zorder=1 if s.is_reference else 2, **kw)
        peak = 0.0
        diffs = []
        for s in series:
            if s.is_reference:
                continue
            th, rel = _rel_diff(s, ref, sp, ref_floor[sp])
            diffs.append((s, th, rel))
            if np.any(np.isfinite(rel)):
                peak = max(peak, float(np.nanmax(np.abs(rel))))
        lim = max(1.0, 1.15 * peak)
        if rel_cap_pct is not None:
            lim = min(lim, rel_cap_pct)
        for s, th, rel in diffs:
            kw = {"marker": s.marker, "markersize": 2.5} if s.marker else {}
            axr.plot(th, np.clip(rel, -lim, lim), color=s.color, ls=s.linestyle,
                     lw=s.linewidth * 0.8, **kw)
            over = np.abs(rel) > lim
            if np.any(over):
                axr.plot(th[over], np.sign(rel[over]) * lim * 0.97, ls="none",
                         marker="^", mfc="none", mec=s.color, ms=4)
        axr.axhline(0.0, color=COLORS["reference"], lw=0.8)
        if lim >= 100 * AGREEMENT_BAND:
            for b in (-AGREEMENT_BAND, AGREEMENT_BAND):
                axr.axhline(100 * b, color=COLORS["band"], lw=0.8, ls=":")
        axr.set_ylim(-lim, lim)
        name = (display_names or {}).get(sp, sp)
        ax.set_title(name, loc="left", fontweight="bold")
        ax.set_ylabel(units[sp])
        ax.set_ylim(bottom=min(0.0, ax.get_ylim()[0]))
        axr.set_ylabel("diff. [%]")
        axr.set_xlabel("local time, EDT [h]")
        local_time_axis(axr, sun, t_max)
        plt.setp(ax.get_xticklabels(), visible=False)
        if handles is None:
            handles = ax.get_legend_handles_labels()
    hl, ll = handles
    hl = list(hl) + [Patch(color=NIGHT_COLOR, label="night (SZA ≥ 90°)")]
    ll = list(ll) + ["night (SZA ≥ 90°)"]
    fig.legend(hl, ll, loc="outside lower center", ncols=min(len(ll), 5), frameon=False)
    if title or legend_note:
        fig.suptitle("\n".join(x for x in (title, legend_note) if x), fontsize=FONT_SIZE + 1)
    return _save(fig, path)


def transitions(rows: list[dict], series: list[Series], j_series: list[dict], path: Path, *,
                sun: dict, window_h: float, title: str = "") -> Path:
    """Sunset (left) and sunrise (right) zoom: J_NO2 staircases on top, species below.

    ``rows`` = [{"species", "label", "unit"}]; ``j_series`` = [{"label", "t0_s", "j",
    "color", "linestyle", "linewidth"}] drawn as step functions from each interval start.
    """
    style()
    events = (("sunset", sun["t_sunset_h"]), ("sunrise", sun["t_sunrise_h"]))
    nrows = 1 + len(rows)
    fig, axes = plt.subplots(nrows, 2, figsize=(11.5, 2.15 * nrows + 0.8), sharex="col",
                             constrained_layout=True, squeeze=False)
    for c, (ev, t_ev) in enumerate(events):
        lo, hi = t_ev - window_h, t_ev + window_h
        ax = axes[0][c]
        j_top = 0.0
        for js in j_series:
            t = np.append(js["t0_s"], js["t0_s"][-1] + (js["t0_s"][1] - js["t0_s"][0])) / 3600.0
            ax.step(t, np.append(js["j"], js["j"][-1]), where="post", color=js["color"],
                    ls=js["linestyle"], lw=js["linewidth"], label=js["label"])
            win = (t[:-1] <= hi) & (t[1:] >= lo)
            j_top = max(j_top, float(np.max(js["j"][win])))
        ax.set_ylim(-0.03 * j_top, 1.08 * j_top)
        ax.set_ylabel("J$_{NO2}$ [s$^{-1}$]")
        ax.set_title(f"{ev}: geometric {ev} at "
                     f"{_clock(sun, t_ev)} EDT (t = {t_ev:.2f} h)", loc="left", fontweight="bold")
        for r, row in enumerate(rows, start=1):
            ax = axes[r][c]
            for s in series:
                t = s.t_s / 3600.0
                m = (t >= lo - 0.3) & (t <= hi + 0.3)
                kw = {"marker": s.marker, "markersize": 4} if s.marker else {}
                ax.plot(t[m], s.values[row["species"]][m], color=s.color, ls=s.linestyle,
                        lw=s.linewidth, label=s.label, zorder=1 if s.is_reference else 2, **kw)
            ax.set_ylabel(f"{row['label']} [{row['unit']}]")
        for r in range(nrows):
            ax = axes[r][c]
            shade_night(ax, sun)
            ax.axvline(t_ev, color=SUN_LINE_COLOR, lw=1.0, ls="--", zorder=1)
            ax.set_xlim(lo, hi)
        ticks = np.arange(np.ceil(lo * 4) / 4, hi + 1e-9, 0.5)
        axes[-1][c].set_xticks(ticks)
        axes[-1][c].set_xticklabels([_clock(sun, t) for t in ticks])
        axes[-1][c].set_xlabel("local time, EDT [hh:mm]")
    h1, l1 = axes[0][0].get_legend_handles_labels()
    h2, l2 = axes[1][0].get_legend_handles_labels()
    hs = list(h1) + list(h2) + [Patch(color=NIGHT_COLOR)]
    ls = list(l1) + list(l2) + ["night (SZA ≥ 90°)"]
    fig.legend(hs, ls, loc="outside lower center", ncols=3, frameon=False)
    if title:
        fig.suptitle(title, fontsize=FONT_SIZE + 1)
    return _save(fig, path)


def held_species_figure(lines: list[dict], noy: list[dict], path: Path, *, sun: dict,
                        zooms: list[tuple[str, float, float]], species: str = "NO2",
                        title: str = "", text: str = "") -> Path:
    """Held species inside the coupling intervals (full day + zooms) and the NOy it adds.

    ``lines``: [{"label", "t_s", "ppb", "color", "linestyle", "linewidth"}] (the TOMAS-JAX
    lines are sawtooth arrays: reset value at each interval start, then sub-steps);
    ``noy``: [{"label", "t_s", "ppb", "color", "linestyle", "linewidth", "marker"}];
    ``zooms``: [(name, t0_h, t1_h)].
    """
    style()
    fig = plt.figure(figsize=(12.5, 10.0), constrained_layout=True)
    gs = fig.add_gridspec(3, len(zooms))
    ax_full = fig.add_subplot(gs[0, :])
    ax_z = [fig.add_subplot(gs[1, k]) for k in range(len(zooms))]
    ax_n = fig.add_subplot(gs[2, :])
    t_max = max(float(np.max(d["t_s"])) for d in lines) / 3600.0
    for ax in [ax_full] + ax_z:
        shade_night(ax, sun)
        for d in lines:
            ax.plot(np.asarray(d["t_s"]) / 3600.0, d["ppb"], color=d["color"], ls=d["linestyle"],
                    lw=d["linewidth"], label=d["label"])
        ax.set_ylabel(f"{species} [ppb]")
    local_time_axis(ax_full, sun, t_max)
    ax_full.set_xlabel("local time, EDT [h]")
    ax_full.set_title(f"{species} inside the coupling intervals (TOMAS-JAX resets it to 1 ppb "
                      "after every interval; PyCHAM holds d[NO2]/dt = 0)", loc="left",
                      fontweight="bold")
    for ax, (name, t0, t1) in zip(ax_z, zooms):
        ax.set_xlim(t0, t1)
        ticks = np.arange(np.ceil(t0 * 2) / 2, t1 + 1e-9, 0.5)
        ax.set_xticks(ticks)
        ax.set_xticklabels([_clock(sun, t) for t in ticks])
        ax.set_xlabel("local time, EDT [hh:mm]")
        ax.set_title(f"zoom: {name}", loc="left", fontweight="bold")
    shade_night(ax_n, sun)
    for d in noy:
        kw = {"marker": d["marker"], "markersize": 3.5} if d.get("marker") else {}
        ax_n.plot(np.asarray(d["t_s"]) / 3600.0, d["ppb"], color=d["color"], ls=d["linestyle"],
                  lw=d["linewidth"], label=d["label"], **kw)
    local_time_axis(ax_n, sun, t_max)
    ax_n.set_xlabel("local time, EDT [h]")
    ax_n.set_ylabel("NOy − NOy(t = 0) [ppb]")
    ax_n.set_title("NOx added by holding NO2 (increase of total reactive nitrogen, all species)",
                   loc="left", fontweight="bold")
    ax_n.legend(frameon=False, loc="upper left")
    if text:
        ax_full.text(0.01, 0.97, text, transform=ax_full.transAxes, ha="left", va="top",
                     fontsize=FONT_SIZE - 2, color=COLORS["annotation"], family="monospace",
                     bbox={"facecolor": "white", "edgecolor": COLORS["annotation_box"], "alpha": 0.9})
    h, lab = ax_full.get_legend_handles_labels()
    fig.legend(list(h) + [Patch(color=NIGHT_COLOR)], list(lab) + ["night (SZA ≥ 90°)"],
               loc="outside lower center", ncols=4, frameon=False)
    if title:
        fig.suptitle(title, fontsize=FONT_SIZE + 1)
    return _save(fig, path)


def _clock(sun: dict, t_h: float) -> str:
    x = (sun["start_local_h"] + t_h) % 24.0
    h = int(x)
    m = int(round((x - h) * 60.0))
    if m == 60:
        h, m = (h + 1) % 24, 0
    return f"{h:02d}:{m:02d}"


def snapshot_title(sun: dict, name: str, t_h: float) -> str:
    return f"{name}, {_clock(sun, t_h)} EDT (t = {t_h:g} h)"


def scatter_named(snaps_by_run: dict[str, list[dict]], names: list[str], path: Path, *,
                  sun: dict, threshold_ppb: float, title: str = "", label_top: int = 4) -> Path:
    """``plotting.scatter_snapshots`` with named panels (sunset, midnight, noon)."""
    style()
    runs = list(snaps_by_run)
    times = [s["t_h"] for s in snaps_by_run[runs[0]]]
    fig, axes = plt.subplots(2, len(times), figsize=(4.9 * len(times), 8.8),
                             constrained_layout=True, squeeze=False,
                             gridspec_kw={"height_ratios": (1.25, 1)})
    lo = threshold_ppb / 3.0
    hi = max(float(np.max(s["ref"])) for r in runs for s in snaps_by_run[r]) * 3.0
    xx = np.array([lo, hi])
    for k, (t_h, name) in enumerate(zip(times, names)):
        ax = axes[0][k]
        ax.plot(xx, xx, color=COLORS["reference"], lw=1.0, label="1:1")
        ax.fill_between(xx, xx * (1 - AGREEMENT_BAND), xx * (1 + AGREEMENT_BAND),
                        color=COLORS["band"], alpha=0.25, lw=0, label="±10 %")
        lines = []
        for j, run in enumerate(runs):
            s = snaps_by_run[run][k]
            y = np.maximum(s["ours"], lo)
            within = np.mean(np.abs(10.0 ** s["log10_ratio"] - 1.0) <= AGREEMENT_BAND)
            if j == 0:
                ax.scatter(s["ref"], y, s=14, color=COLORS["ours"], edgecolor="white",
                           linewidth=0.3, zorder=3, label=run)
            else:
                ax.scatter(s["ref"], y, s=26, facecolor="none", edgecolor=COLORS["ours_alt"],
                           linewidth=0.8, zorder=2, label=run)
            lines.append(f"{run.split(' (')[-1].rstrip(')')}: {100 * within:.0f} % within ±10 %")
            if j == 0 and label_top:
                for w in np.argsort(-np.abs(s["log10_ratio"]))[:label_top]:
                    if abs(s["log10_ratio"][w]) < np.log10(1 + AGREEMENT_BAND):
                        continue
                    ax.annotate(s["species"][w], (s["ref"][w], y[w]), fontsize=FONT_SIZE - 3,
                                xytext=(3, 3), textcoords="offset points", color=COLORS["annotation"])
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal")
        n = snaps_by_run[runs[0]][k]["species"].size
        ax.set_title(f"{snapshot_title(sun, name, t_h)}\n{n} species ≥ {threshold_ppb:g} ppb "
                     f"in {LABEL_PYCHAM}", loc="left")
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top",
                fontsize=FONT_SIZE - 2)
        ax.set_xlabel(f"{LABEL_PYCHAM} [ppb]")
        ax.set_ylabel(f"{LABEL_OURS} [ppb]")
        axd = axes[1][k]
        axd.axhline(0.0, color=COLORS["reference"], lw=0.8)
        peak = 0.0
        for j, run in enumerate(runs):
            s = snaps_by_run[run][k]
            rel = 100.0 * (10.0 ** s["log10_ratio"] - 1.0)
            peak = max(peak, float(np.max(np.abs(rel))))
            if j == 0:
                axd.scatter(s["ref"], rel, s=14, color=COLORS["ours"], edgecolor="white",
                            linewidth=0.3, zorder=3)
            else:
                axd.scatter(s["ref"], rel, s=26, facecolor="none", edgecolor=COLORS["ours_alt"],
                            linewidth=0.8, zorder=2)
        lim = max(1.0, 1.15 * peak)
        if lim >= 100 * AGREEMENT_BAND:
            for b in (-AGREEMENT_BAND, AGREEMENT_BAND):
                axd.axhline(100 * b, color=COLORS["band"], lw=0.8, ls=":")
        axd.set_ylim(-lim, lim)
        axd.set_xscale("log")
        axd.set_xlim(lo, hi)
        axd.set_xlabel(f"{LABEL_PYCHAM} [ppb]")
        axd.set_ylabel(f"({LABEL_OURS} − {LABEL_PYCHAM}) / {LABEL_PYCHAM} [%]")
        axd.set_title(f"{name}: relative difference", loc="left")
    h, lab = axes[0][0].get_legend_handles_labels()
    fig.legend(h, lab, loc="outside lower center", ncols=len(lab), frameon=False)
    if title:
        fig.suptitle(title, fontsize=FONT_SIZE + 1)
    return _save(fig, path)


def ratio_named(snaps_by_run: dict[str, list[dict]], names: list[str], path: Path, *,
                sun: dict, title: str = "") -> Path:
    """``plotting.ratio_distribution`` with named snapshot labels."""
    style()
    runs = list(snaps_by_run)
    peak = max(float(np.max(np.abs(s["log10_ratio"]))) for r in runs for s in snaps_by_run[r])
    xlim = max(1.5 * np.log10(1 + AGREEMENT_BAND), 1.1 * peak)
    fig, axes = plt.subplots(len(runs), 2, figsize=(11.5, 3.7 * len(runs)),
                             constrained_layout=True, squeeze=False)
    band_log = np.log10(1 + AGREEMENT_BAND)
    bins = np.linspace(-xlim, xlim, 81)
    for row, run in enumerate(runs):
        axh, axc = axes[row]
        for k, s in enumerate(snaps_by_run[run]):
            col, ls = SNAPSHOT_COLORS[k % 3], SNAPSHOT_LINESTYLES[k % 3]
            lab = f"{names[k]} {_clock(sun, s['t_h'])}"
            lr = np.clip(s["log10_ratio"], -xlim, xlim)
            axh.hist(lr, bins=bins, histtype="step", color=col, ls=ls, lw=1.8,
                     label=f"{lab} (n = {lr.size})")
            dev = np.maximum(np.sort(np.abs(10.0 ** s["log10_ratio"] - 1.0)), 1e-6)
            axc.step(dev * 100.0, np.arange(1, dev.size + 1) / dev.size, where="post",
                     color=col, ls=ls, lw=1.8, label=lab)
        for x in (-band_log, band_log):
            axh.axvline(x, color=COLORS["band"], ls=":", lw=1.0)
        axh.axvline(0.0, color=COLORS["reference"], lw=0.8)
        axh.set_xlabel(f"log10({LABEL_OURS} / {LABEL_PYCHAM})  [-]")
        axh.set_ylabel("species [count]")
        axh.set_title(run, loc="left", fontweight="bold")
        axh.legend(frameon=False, loc="upper left")
        axc.axvline(100 * AGREEMENT_BAND, color=COLORS["band"], ls=":", lw=1.0)
        axc.set_xscale("log")
        axc.set_xlim(1e-4, 1e2)
        axc.set_ylim(0, 1.01)
        axc.set_xlabel(f"|{LABEL_OURS}/{LABEL_PYCHAM} − 1|  [%]")
        axc.set_ylabel("cumulative fraction of species [-]")
        axc.set_title(run, loc="left", fontweight="bold")
        axc.legend(frameon=False, loc="lower right")
    if title:
        fig.suptitle(title, fontsize=FONT_SIZE + 1)
    return _save(fig, path)
