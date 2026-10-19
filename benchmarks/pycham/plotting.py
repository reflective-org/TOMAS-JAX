"""Shared figures for the TOMAS-JAX vs PyCHAM benchmark (matplotlib, PNG at FIG_DPI).

Colours and labels come from ``config`` so every figure uses the same identity:
TOMAS-JAX blue (solid; sensitivity runs sky blue, dashed), PyCHAM vermillion.

* ``timeseries_grid``  - one small panel pair per species: values (top) and the
  relative difference to PyCHAM in % (bottom)
* ``scatter_snapshots`` - all species, PyCHAM (x) vs TOMAS-JAX (y), log-log, 1:1 and
  +-10 % lines, one panel per snapshot time
* ``ratio_distribution`` - histogram and CDF of log10(TOMAS-JAX / PyCHAM)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from .config import (AGREEMENT_BAND, COLORS, FIG_DPI, FONT_SIZE, LABEL_OURS,  # noqa: E402
                     LABEL_PYCHAM, SNAPSHOT_COLORS)
from .metrics import common_times  # noqa: E402

SNAPSHOT_MARKERS = ("o", "s", "^")
SNAPSHOT_LINESTYLES = ("-", "--", ":")


def style() -> None:
    plt.rcParams.update({
        "font.size": FONT_SIZE, "axes.titlesize": FONT_SIZE, "axes.labelsize": FONT_SIZE,
        "legend.fontsize": FONT_SIZE - 1, "xtick.labelsize": FONT_SIZE - 1,
        "ytick.labelsize": FONT_SIZE - 1, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.color": COLORS["grid"],
        "grid.linewidth": 0.6, "savefig.dpi": FIG_DPI, "figure.dpi": FIG_DPI,
    })


@dataclass
class Series:
    """One model run for time-series plots. ``values`` maps species -> array over t_s."""
    label: str
    color: str
    t_s: np.ndarray
    values: dict[str, np.ndarray]
    linestyle: str = "-"
    linewidth: float = 1.8
    is_reference: bool = False
    marker: str | None = None
    extra: dict = field(default_factory=dict)


def ours_series(t_s, values, label=LABEL_OURS, alt=False) -> Series:
    return Series(label, COLORS["ours_alt" if alt else "ours"], np.asarray(t_s), values,
                  linestyle="--" if alt else "-", linewidth=1.6 if alt else 1.8)


def pycham_series(t_s, values, label=LABEL_PYCHAM) -> Series:
    return Series(label, COLORS["pycham"], np.asarray(t_s), values, linestyle="-",
                  linewidth=2.6, is_reference=True)


def darker(color: str, factor: float = 0.55) -> str:
    """A darker shade of ``color`` (hatch / edge lines of hatched bars)."""
    from matplotlib.colors import to_hex, to_rgb
    return to_hex(tuple(c * factor for c in to_rgb(color)))


def _save(fig, path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=FIG_DPI, bbox_inches="tight")
    plt.close(fig)
    return path


def timeseries_grid(species: list[str], series: list[Series], path: Path, *,
                    units: dict[str, str], ncols: int = 4, title: str = "",
                    log_species: tuple[str, ...] = (), rel_ylim_pct: float | None = None,
                    display_names: dict[str, str] | None = None) -> Path:
    """Values (top) and 100*(model - PyCHAM)/PyCHAM (bottom) for each species."""
    refs = [s for s in series if s.is_reference]
    if len(refs) != 1:
        raise ValueError("exactly one reference (PyCHAM) series is required")
    ref = refs[0]
    for s in series:
        missing = [sp for sp in species if sp not in s.values]
        if missing:
            raise KeyError(f"{s.label} lacks {missing}")
    style()
    nrows = int(np.ceil(len(species) / ncols))
    fig = plt.figure(figsize=(3.6 * ncols, 3.5 * nrows), constrained_layout=True)
    outer = fig.add_gridspec(nrows, ncols)
    handles = None
    for k, sp in enumerate(species):
        inner = outer[k // ncols, k % ncols].subgridspec(2, 1, height_ratios=(2.4, 1), hspace=0.05)
        ax = fig.add_subplot(inner[0])
        axr = fig.add_subplot(inner[1], sharex=ax)
        for s in series:
            ax.plot(s.t_s / 3600.0, s.values[sp], color=s.color, ls=s.linestyle,
                    lw=s.linewidth, label=s.label, zorder=1 if s.is_reference else 2)
        peak = 0.0
        for s in series:
            if s.is_reference:
                continue
            ia, ib = common_times(s.t_s, ref.t_s)
            r = ref.values[sp][ib]
            rel = np.divide(s.values[sp][ia] - r, r, out=np.full(ia.size, np.nan), where=r > 0)
            axr.plot(s.t_s[ia] / 3600.0, 100.0 * rel, color=s.color, ls=s.linestyle,
                     lw=s.linewidth * 0.8)
            if np.any(np.isfinite(rel)):
                peak = max(peak, float(np.nanmax(np.abs(rel))) * 100.0)
        axr.axhline(0.0, color=COLORS["reference"], lw=0.8)
        # symmetric range that shows every point; the +-10 % lines only when in range
        lim = rel_ylim_pct if rel_ylim_pct is not None else max(1.0, 1.15 * peak)
        if lim >= 100 * AGREEMENT_BAND:
            for b in (-AGREEMENT_BAND, AGREEMENT_BAND):
                axr.axhline(100 * b, color=COLORS["band"], lw=0.8, ls=":")
        axr.set_ylim(-lim, lim)
        if sp in log_species:
            ax.set_yscale("log")
        name = (display_names or {}).get(sp, sp)
        ax.set_title(name, loc="left", fontweight="bold")
        ax.set_ylabel(units[sp])
        axr.set_ylabel("diff. [%]")
        axr.set_xlabel("time [h]")
        plt.setp(ax.get_xticklabels(), visible=False)
        if handles is None:
            handles = ax.get_legend_handles_labels()
    fig.legend(*handles, loc="outside lower center", ncols=len(series), frameon=False)
    if title:
        fig.suptitle(title, fontsize=FONT_SIZE + 1)
    return _save(fig, path)


def scatter_snapshots(snaps_by_run: dict[str, list[dict]], path: Path, *, threshold_ppb: float,
                      title: str = "", label_top: int = 4) -> Path:
    """All-species scatter, PyCHAM (x) vs TOMAS-JAX (y); one panel per snapshot time.

    ``snaps_by_run`` maps a run label to the ``metrics.snapshot`` dicts (same times for
    every run). The first run is drawn in the TOMAS-JAX colour, a second one (the
    sensitivity run) in the sky-blue alternative with open markers.
    """
    style()
    runs = list(snaps_by_run)
    times = [s["t_h"] for s in snaps_by_run[runs[0]]]
    fig, axes = plt.subplots(2, len(times), figsize=(4.8 * len(times), 8.6),
                             constrained_layout=True, squeeze=False,
                             gridspec_kw={"height_ratios": (1.25, 1)})
    lo = threshold_ppb / 3.0
    hi = max(float(np.max(s["ref"])) for r in runs for s in snaps_by_run[r]) * 3.0
    xx = np.array([lo, hi])
    for ax, t_h, k in zip(axes[0], times, range(len(times))):
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
            lines.append(f"{run}: {100 * within:.0f} % within ±10 %")
            if j == 0 and label_top:
                worst = np.argsort(-np.abs(s["log10_ratio"]))[:label_top]
                for w in worst:
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
        ax.set_title(f"t = {t_h:g} h  ({n} species ≥ {threshold_ppb:g} ppb in {LABEL_PYCHAM})",
                     loc="left")
        ax.text(0.03, 0.97, "\n".join(lines), transform=ax.transAxes, va="top",
                fontsize=FONT_SIZE - 2)
        ax.set_xlabel(f"{LABEL_PYCHAM} [ppb]")
        ax.set_ylabel(f"{LABEL_OURS} [ppb]")
        # bottom row: relative difference vs abundance (resolves what the 1:1 plot cannot)
        axd = axes[1][k]
        axd.axhline(0.0, color=COLORS["reference"], lw=0.8)
        peak = 0.0
        for j, run in enumerate(runs):
            s = snaps_by_run[run][k]
            rel = 100.0 * (10.0 ** s["log10_ratio"] - 1.0)
            peak = max(peak, float(np.max(np.abs(rel))))
            if j == 0:
                axd.scatter(s["ref"], rel, s=14, color=COLORS["ours"], edgecolor="white",
                            linewidth=0.3, zorder=3, label=run)
            else:
                axd.scatter(s["ref"], rel, s=26, facecolor="none", edgecolor=COLORS["ours_alt"],
                            linewidth=0.8, zorder=2, label=run)
        lim = max(1.0, 1.15 * peak)
        if lim >= 100 * AGREEMENT_BAND:
            for b in (-AGREEMENT_BAND, AGREEMENT_BAND):
                axd.axhline(100 * b, color=COLORS["band"], lw=0.8, ls=":")
        axd.set_ylim(-lim, lim)
        axd.set_xscale("log")
        axd.set_xlim(lo, hi)
        axd.set_xlabel(f"{LABEL_PYCHAM} [ppb]")
        axd.set_ylabel(f"({LABEL_OURS} − {LABEL_PYCHAM}) / {LABEL_PYCHAM} [%]")
        axd.set_title(f"t = {t_h:g} h: relative difference", loc="left")
    axes[0][0].legend(loc="lower right", frameon=False)
    if title:
        fig.suptitle(title, fontsize=FONT_SIZE + 1)
    return _save(fig, path)


@dataclass
class Curve:
    label: str
    values: np.ndarray
    color: str
    linestyle: str = "-"
    linewidth: float = 1.8


def stat_cdfs(panels: dict[str, list[Curve]], path: Path, *, xlabels: dict[str, str],
              title: str = "", xlim: tuple[float, float] = (1e-6, 1e1)) -> Path:
    """Cumulative distributions over species of per-species statistics (log x).

    ``panels`` maps a panel title to curves; each curve's values are one number per
    species (NaN dropped, zeros drawn at the left edge). A dotted line marks 10 %.
    """
    style()
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 4.8),
                             constrained_layout=True, squeeze=False)
    for ax, (name, curves) in zip(axes[0], panels.items()):
        for c in curves:
            v = np.abs(np.asarray(c.values, float))
            v = np.sort(np.maximum(v[np.isfinite(v)], xlim[0]))
            if v.size == 0:
                raise ValueError(f"no finite values for {c.label}")
            ax.step(v, np.arange(1, v.size + 1) / v.size, where="post", color=c.color,
                    ls=c.linestyle, lw=c.linewidth, label=c.label)
        ax.axvline(AGREEMENT_BAND, color=COLORS["band"], ls=":", lw=1.0)
        ax.set_xscale("log")
        ax.set_xlim(*xlim)
        ax.set_ylim(0, 1.01)
        ax.set_xlabel(xlabels[name])
        ax.set_ylabel("cumulative fraction of species [-]")
        ax.set_title(name, loc="left", fontweight="bold")
    handles, labels = axes[0][0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="outside lower center", ncols=2, frameon=False)
    if title:
        fig.suptitle(title, fontsize=FONT_SIZE + 1)
    return _save(fig, path)


def ratio_distribution(snaps_by_run: dict[str, list[dict]], path: Path, *, title: str = "",
                       xlim: float | None = None) -> Path:
    """Histogram (left) and CDF of |TOMAS-JAX/PyCHAM - 1| (right); one row per run.

    ``xlim`` (histogram range in log10 units) defaults to the data range, at least
    1.5x the +-10 % band; values beyond it are clipped into the edge bins.
    """
    style()
    runs = list(snaps_by_run)
    if xlim is None:
        peak = max(float(np.max(np.abs(s["log10_ratio"]))) for r in runs for s in snaps_by_run[r])
        xlim = max(1.5 * np.log10(1 + AGREEMENT_BAND), 1.1 * peak)
    fig, axes = plt.subplots(len(runs), 2, figsize=(11, 3.6 * len(runs)),
                             constrained_layout=True, squeeze=False)
    band_log = np.log10(1 + AGREEMENT_BAND)
    bins = np.linspace(-xlim, xlim, 81)
    for row, run in enumerate(runs):
        axh, axc = axes[row]
        for k, s in enumerate(snaps_by_run[run]):
            col, ls = SNAPSHOT_COLORS[k % 3], SNAPSHOT_LINESTYLES[k % 3]
            lr = np.clip(s["log10_ratio"], -xlim, xlim)
            axh.hist(lr, bins=bins, histtype="step", color=col, ls=ls, lw=1.8,
                     label=f"{s['t_h']:g} h (n = {lr.size})")
            dev = np.sort(np.abs(10.0 ** s["log10_ratio"] - 1.0))
            dev = np.maximum(dev, 1e-6)
            cdf = np.arange(1, dev.size + 1) / dev.size
            axc.step(dev * 100.0, cdf, where="post", color=col, ls=ls, lw=1.8,
                     label=f"{s['t_h']:g} h")
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
