"""Case S2 analysis: metrics and figures from the saved results (no model runs).

    cd tomas-jax && .venv/bin/python -m benchmarks.pycham.cases.S2.analyse

Writes ``results/summary.json``, ``results/gas_species_stats.csv`` and the figures in
``figures/`` (copied to docs/aaar/figures/pycham/ with an ``S2_`` prefix). Raises if a
result file is missing.
"""
from __future__ import annotations

import json
import shutil
import sys

import matplotlib.pyplot as plt
import numpy as np

from tomas_jax.box.driver import _chemistry, _class_map
from tomas_jax.gaschem.config import AVOGADRO, VOLATILITY_LOG10_CSTAR_CENTERS
from tomas_jax.physics.density import calc_density

from ... import metrics, ours, plotting
from ...config import (ABUNDANCE_THRESHOLD_PPB, AGREEMENT_BAND, COLORS, DOC_FIG_DIR,
                       DRY_DIAMETER_LABEL, FONT_SIZE, LABEL_OURS, LABEL_PYCHAM,
                       STRICT_THRESHOLD_PPB)
from ...particles import PYCHAM_WATER_DENSITY, dry_diameter_nm
from . import case
from .case import S2
from .inputs import cstar_at_temp

RADICALS = ("OH", "HO2", "NO3")
FEEDBACK_SPECIES = S2["feedback_species"]
N_FEEDBACK_CONDENSABLES = 8
LABEL_OURS300 = f"{LABEL_OURS} (coupling 300 s)"
LABEL_OURS60 = f"{LABEL_OURS} (coupling 60 s)"
LABEL_PY_CLASS = f"{LABEL_PYCHAM} (class-centre c*)"
LABEL_PY_EXACT = f"{LABEL_PYCHAM} (own Li 2016 c*)"
LABEL_PY_NATIVE = f"{LABEL_PYCHAM} (own SMILES-based vapour pressures)"


def _need(path):
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run benchmarks.pycham.cases.S2.run first")
    return path


def load_all() -> dict:
    d = {"o300": ours.load_npz(_need(case.OURS_NPZ[300])),
         "o60": ours.load_npz(_need(case.OURS_NPZ[60])),
         "onop": ours.load_npz(_need(case.OURS_NOPART_NPZ)),
         "ocs": ours.load_npz(_need(case.OURS_ATTR_NPZ["dt60_cs_pycham"])),
         "ocs_nk": ours.load_npz(_need(case.OURS_ATTR_NPZ["dt60_cs_pycham_nokelvin"]))}
    for v, p in case.PYCHAM_NPZ.items():
        d[f"py_{v}"] = dict(np.load(_need(p)))
    d["info"] = json.loads(_need(case.RUN_INFO_JSON).read_text())
    d["transfer"] = json.loads(_need(case.TRANSFER_JSON).read_text())
    return d


# --- per-model quantities -------------------------------------------------------------
def _cmap():
    return _class_map(None)


def _cols(names, wanted):
    return metrics.columns([str(n) for n in names], wanted)


def ours_particle_by_class(o):
    return o["class_particle_ugm3"].astype(float)


def py_particle_species(p):
    """[t, condensable] particle mass in class-map order [ug/m3]."""
    return p["particle_species_ugm3"][:, _cols(p["particle_names"], _cmap().species)].astype(float)


def py_particle_by_class(p):
    cm = _cmap()
    ps = py_particle_species(p)
    return np.stack([np.bincount(cm.class_idx, weights=row, minlength=cm.class_cstar.size)
                     for row in ps])


def gas_condensable_ugm3(run, kind):
    """[t, condensable] gas-phase mass [ug/m3] with each model's own molar masses."""
    cm = _cmap()
    ppb = run["gas_ppb"][:, _cols(run["species"], cm.species)].astype(float)
    if kind == "ours":
        cm3 = ppb * 1e-9 * float(run["m_air_cm3"])
        mw = cm.mw
    else:
        cm3 = ppb * run["ppb_to_molec_cm3"].reshape(-1, 1)
        mw = run["mw_g_mol"][_cols(run["species"], cm.species)]
    return cm3 * mw / AVOGADRO * 1e12


def by_class(per_species):
    cm = _cmap()
    return np.stack([np.bincount(cm.class_idx, weights=row, minlength=cm.class_cstar.size)
                     for row in per_species])


def water_series(run, kind):
    """Particle water [ug/m3] over time."""
    if kind == "ours":
        return ours_component(run, "H2O")
    return run["bin_water_ugm3"].astype(float).sum(axis=1)


def ours_component(o, name):
    return o["particle_component_ugm3"][:, list(map(str, o["component_names"])).index(name)]


def pycham_nominal_diameters(d) -> np.ndarray:
    """Nominal dry diameter [nm] of each PyCHAM bin: the TOMAS-JAX bin it was built from."""
    k_lo, k_hi = d["info"]["setup"]["seed"]["tomas_bins_in_pycham"]
    return d["o300"]["diameter_nm"][k_lo:k_hi + 1]


def size_distribution(run, kind, i, nominal_nm=None):
    """Per bin: (dry diameter [nm] at the display density, dN/dlog10Dp [cm-3], dM/dlog10Dp
    [ug m-3], N [cm-3]). The diameter is the bin's mean dry mass per particle; empty bins
    sit at their nominal diameter. Both grids double the dry mass per bin, so every bin
    is log10(2)/3 wide in diameter (PyCHAM got the TOMAS-JAX bounds, inputs.py)."""
    if kind == "ours":
        dlog = np.log10(run["diameter_edges_nm"][1:] / run["diameter_edges_nm"][:-1])
        n = run["dndlog10dp_cm3"][i] * dlog
        m = run["dmdlog10dp_ugm3"][i] * dlog
        nominal_nm = run["diameter_nm"]
    else:
        n = run["n_bin_cm3"][i].astype(float)
        m = run["bin_dry_ugm3"][i].astype(float)
        dlog = np.full(n.size, np.log10(2.0) / 3.0)
        if nominal_nm is None or len(nominal_nm) != n.size:
            raise ValueError("PyCHAM size distribution needs the nominal bin diameters")
    with np.errstate(divide="ignore", invalid="ignore"):
        mp = m * 1e-9 / np.where(n > 0, n * 1e6, np.nan)          # kg per particle
        dd = dry_diameter_nm(mp)
    dd = np.where(np.isfinite(dd) & (n > 1e-6), dd, nominal_nm)
    return dd, n / dlog, m / dlog, n


def _idx(t_s, t_h):
    i = int(np.argmin(np.abs(np.asarray(t_s) - t_h * 3600.0)))
    if abs(t_s[i] - t_h * 3600.0) > 1.0:
        raise ValueError(f"no output at {t_h} h")
    return i


# --- equilibrium reference (bulk, no Kelvin) ----------------------------------------------
def pankow(c_tot, cstar, c_nv, mw=None, mw_nv=None, tol=1e-12):
    """Bulk absorptive equilibrium; ``mw`` given -> Raoult by mole fraction (PyCHAM form).

    Returns the particle-phase mass per entry [ug/m3].
    """
    c_tot, cstar = np.asarray(c_tot, float), np.asarray(cstar, float)

    def particle(x):              # x = absorbing moles-equivalent or mass [same units]
        if mw is None:
            return c_tot / (1.0 + cstar / x)
        # n_i in particle: c_p,i/MW_i; gas c_g,i = (c_p,i/MW_i)/x * c*_i
        return c_tot / (1.0 + cstar / (x * mw))

    def resid(x):
        cp = particle(x)
        total = c_nv + cp.sum() if mw is None else c_nv / mw_nv + (cp / mw).sum()
        return total - x

    lo, hi = (c_nv, c_nv + c_tot.sum()) if mw is None else \
        (c_nv / mw_nv, c_nv / mw_nv + (c_tot / mw).sum())
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if resid(mid) > 0:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol * hi:
            break
    return particle(0.5 * (lo + hi))


# --- metrics --------------------------------------------------------------------------------
def gas_phase_stats(d) -> tuple[dict, dict, dict]:
    py = d["py_class"]
    stats, snaps, summ = {}, {}, {}
    cond = set(_cmap().species)
    for key, label in (("o300", LABEL_OURS300), ("o60", LABEL_OURS60)):
        o = d[key]
        names = metrics.common_species(o["species"], py["species"])
        ia, ib = metrics.common_times(o["t_s"], py["t_s"])
        x = o["gas_ppb"][ia][:, _cols(o["species"], names)].astype(float)
        r = py["gas_ppb"][ib][:, _cols(py["species"], names)].astype(float)
        t = o["t_s"][ia]
        stats[label] = metrics.species_stats(names, x, r)
        snaps[label] = [metrics.snapshot(names, t, x, r, th) for th in S2["snapshot_times_h"]]
        summ[label] = metrics.summary(stats[label], snaps[label])
        strict = metrics.species_stats(names, x, r, STRICT_THRESHOLD_PPB)
        summ[label]["strict_threshold"] = metrics.summary(
            strict, [metrics.snapshot(names, t, x, r, th, STRICT_THRESHOLD_PPB)
                     for th in S2["snapshot_times_h"]])
        st = stats[label]
        ok = st["n_valid"] > 0
        is_c = np.array([n in cond for n in st["species"]])
        for tag, sel in (("condensable", ok & is_c), ("non_condensable", ok & ~is_c)):
            summ[label][tag] = {
                "n_species": int(sel.sum()),
                "median_abs_nmb": float(np.nanmedian(np.abs(st["nmb"][sel]))),
                "p95_abs_nmb": float(np.nanpercentile(np.abs(st["nmb"][sel]), 95)),
                "frac_abs_nmb_within_band": metrics.fraction_within(st["nmb"][sel])}
        summ[label]["worst_nmb_species"] = [
            {"species": str(st["species"][k]), "nmb": float(st["nmb"][k]),
             "condensable": bool(is_c[k]), "pycham_max_ppb": float(st["ref_max_ppb"][k])}
            for k in np.argsort(-np.nan_to_num(np.abs(st["nmb"]), nan=-1))[:15]]
    return stats, snaps, summ


def particle_metrics(d) -> dict:
    cm = _cmap()
    o300, o60, pyc, pye = d["o300"], d["o60"], d["py_class"], d["py_exact"]
    soa = {"o300": ours_particle_by_class(o300).sum(1), "o60": ours_particle_by_class(o60).sum(1),
           "py_class": py_particle_species(pyc).sum(1), "py_exact": py_particle_species(pye).sum(1)}
    t = {"o300": o300["t_s"], "o60": o60["t_s"], "py_class": pyc["t_s"], "py_exact": pye["t_s"]}
    out = {"soa_ugm3": {}, "soa_rel_diff_vs_pycham_class": {}, "soa_nmb_vs_pycham_class": {}}
    for k in soa:
        out["soa_ugm3"][k] = {f"{th:g}h": float(soa[k][_idx(t[k], th)])
                              for th in (0.5, 1, 2, 3, 4, 5, 6)}
    ref_t, ref = t["py_class"], soa["py_class"]
    for k in ("o300", "o60", "py_exact"):
        ia, ib = metrics.common_times(t[k], ref_t)
        ok = ref[ib] > 0
        out["soa_nmb_vs_pycham_class"][k] = float((soa[k][ia][ok] - ref[ib][ok]).sum()
                                                  / ref[ib][ok].sum())
        out["soa_rel_diff_vs_pycham_class"][k] = {
            f"{th:g}h": float(soa[k][_idx(t[k], th)] / ref[_idx(ref_t, th)] - 1.0)
            for th in (0.5, 1, 2, 3, 4, 5, 6)}
    # lumping and coupling effects within each model
    out["lumping_effect_pycham_exact_vs_class"] = out["soa_rel_diff_vs_pycham_class"]["py_exact"]
    out["coupling_effect_ours_60_vs_300"] = {
        f"{th:g}h": float(soa["o60"][_idx(t["o60"], th)] / soa["o300"][_idx(t["o300"], th)] - 1)
        for th in (0.5, 1, 3, 6)}
    # per-class particle mass at the snapshots
    cls = {"o300": ours_particle_by_class(o300), "o60": ours_particle_by_class(o60),
           "py_class": py_particle_by_class(pyc), "py_exact": py_particle_by_class(pye),
           "py_massfrac": py_particle_by_class(d["py_massfrac"])}
    t["py_massfrac"] = d["py_massfrac"]["t_s"]
    out["class_particle_ugm3"] = {
        f"{th:g}h": {k: cls[k][_idx(t[k], th)].round(5).tolist() for k in cls}
        for th in S2["snapshot_times_h"]}
    out["class_log10_cstar"] = list(VOLATILITY_LOG10_CSTAR_CENTERS)
    # condensable totals (gas + particle): does the chemistry make the same material?
    gas = {"o300": gas_condensable_ugm3(o300, "ours"), "o60": gas_condensable_ugm3(o60, "ours"),
           "py_class": gas_condensable_ugm3(pyc, "py"), "py_exact": gas_condensable_ugm3(pye, "py")}
    out["condensable_gas_ugm3"] = {k: {f"{th:g}h": float(gas[k][_idx(t[k], th)].sum())
                                       for th in (1, 3, 6)} for k in gas}
    out["condensable_total_ugm3"] = {
        k: {f"{th:g}h": float(gas[k][_idx(t[k], th)].sum() + soa[k][_idx(t[k], th)])
            for th in (1, 3, 6)} for k in gas}
    # bulk equilibrium (no Kelvin) from each model's own gas + particle totals
    temp = d["info"]["setup"]["temperature_k"]
    c_nv = float(ours_component(o300, "POA")[0])
    eq = {}
    lc_cls = np.asarray(VOLATILITY_LOG10_CSTAR_CENTERS)[cm.class_idx]
    mw_py = pyc["mw_g_mol"][_cols(pyc["species"], cm.species)]
    mw_nv = d["info"]["setup"]["seed_oa_mw_pycham"]
    for k, run, kind, lc in (("o300", o300, "ours", lc_cls), ("py_class", pyc, "py", lc_cls),
                             ("py_exact", pye, "py", cm.log10_cstar)):
        part = (py_particle_species(run) if kind == "py"
                else o300["condensable_particle_ugm3"].astype(float))
        cst = cstar_at_temp(lc, temp)
        eq[k] = {}
        for th in S2["snapshot_times_h"]:
            i = _idx(run["t_s"], th)
            tot = gas[k][i] + part[i]
            eq[k][f"{th:g}h"] = {
                "model_soa": float(soa[k][i]),
                "eq_mass_fraction_raoult": float(pankow(tot, cst, c_nv).sum()),
                "eq_mole_fraction_raoult": float(pankow(tot, cst, c_nv, mw_py, mw_nv).sum())}
    out["equilibrium_from_own_totals"] = eq
    # sulfate, water, number, mean diameter
    names = list(map(str, pyc["species"]))
    out["sulfate_condensed_ugm3"] = {
        "o300": {f"{th:g}h": float(ours_component(o300, "SO4")[_idx(t["o300"], th)]
                                   - ours_component(o300, "SO4")[0]) for th in (1, 3, 6)},
        "py_class": {f"{th:g}h": float(pyc["bin_sa_ugm3"][_idx(t["py_class"], th)].sum()
                                       - pyc["bin_sa_ugm3"][0].sum()) for th in (1, 3, 6)}}
    water_runs = {"o300": ("ours", o300), "py_class": ("py", pyc),
                  "ocs": ("ours", d["ocs"]), "py_massfrac": ("py", d["py_massfrac"]),
                  "ocs_nk": ("ours", d["ocs_nk"]),
                  "py_massfrac_nokelvin": ("py", d["py_massfrac_nokelvin"])}
    out["particle_water_ugm3"] = {
        k: {f"{th:g}h": float(water_series(run, kind)[_idx(run["t_s"], th)])
            for th in (0, 0.5, 1, 3, 6)} for k, (kind, run) in water_runs.items()}
    out["particle_water_rel_diff"] = {
        f"{a}_vs_{b}": {th: out["particle_water_ugm3"][a][th] / out["particle_water_ugm3"][b][th] - 1
                        for th in out["particle_water_ugm3"][a]}
        for a, b in (("py_class", "o300"), ("py_massfrac", "ocs"),
                     ("py_massfrac_nokelvin", "ocs_nk"))}
    out["h2so4_gas_6h_ugm3"] = {
        "o300": float(o300["h2so4_gas_ugm3"][-1]),
        "py_class": float(pyc["gas_ppb"][-1, names.index("SA")] * pyc["ppb_to_molec_cm3"][-1]
                          * pyc["mw_g_mol"][names.index("SA")] / AVOGADRO * 1e12)}
    out["number_cm3"] = {}
    out["dry_size_stats"] = {}
    dd0, _, _, n0 = size_distribution(o300, "ours", 0)
    d_p01 = seed_percentile_diameter(dd0, n0, 0.01)
    out["seed_p01_dry_diameter_nm"] = d_p01
    for k, run, kind in (("o300", o300, "ours"), ("py_class", pyc, "py"),
                         ("py_exact", pye, "py")):
        out["number_cm3"][k] = {}
        out["dry_size_stats"][k] = {}
        for th in S2["psd_times_h"]:
            dd, _, dm, n = size_distribution(run, kind, _idx(run["t_s"], th),
                                             pycham_nominal_diameters(d))
            out["number_cm3"][k][f"{th:g}h"] = float(n.sum())
            out["dry_size_stats"][k][f"{th:g}h"] = dry_size_stats(dd, n, dm, d_p01)
    return out


def dry_size_stats(dd, n, dm, d_p01) -> dict:
    """Size statistics from each bin's mean dry diameter, the same way for both models
    (as S1's sizestats): number-mean [nm], number GSD [-], dry-mass-weighted mean [nm],
    number and dry-mass medians [nm] (mid-rank, see midrank_median) and the number
    fraction below the seed's 1st-percentile diameter."""
    w = np.where((n > 0) & np.isfinite(dd), n, 0.0)
    lg = np.log(np.where(w > 0, dd, 1.0))
    mu = (w * lg).sum() / w.sum()
    return {"number_mean_nm": float((w * dd).sum() / w.sum()),
            "gsd": float(np.exp(np.sqrt((w * (lg - mu) ** 2).sum() / w.sum()))),
            "dry_mass_mean_nm": float((dm * dd).sum() / dm.sum()),
            "number_median_nm": midrank_median(dd, n),
            "dry_mass_median_nm": midrank_median(dd, dm),
            "frac_below_seed_p01": float(w[dd < d_p01].sum() / w.sum())}


def midrank_median(dd, weights) -> float:
    """Diameter [nm] at half of ``weights``, each bin at its mean dry diameter, from the
    mid-rank cumulative (weight below the bin + half its own), log-interpolated over
    the bins with weight. The plain cumulative (weight up to and including the bin)
    reads the median about half a bin low and drifts with resolution; bins without
    weight (PyCHAM keeps empty ones) would pull it as well."""
    w = np.asarray(weights, float)
    keep = (w > 0) & np.isfinite(dd)
    d, w = np.asarray(dd, float)[keep], w[keep]
    order = np.argsort(d)
    d, w = d[order], w[order]
    c = (np.cumsum(w) - 0.5 * w) / w.sum()
    return float(10.0 ** np.interp(0.5, c, np.log10(d)))


def seed_percentile_diameter(dd, n, q) -> float:
    """Dry diameter [nm] below which a fraction q of the particles lies at t = 0."""
    order = np.argsort(dd)
    c = np.cumsum(np.asarray(n, float)[order])
    return float(10.0 ** np.interp(q, c / c[-1], np.log10(np.asarray(dd)[order])))


def feedback(d) -> dict:
    """Gas-phase change due to partitioning: S2 / no-particle run - 1, at 6 h."""
    cm = _cmap()
    o300, onop, pyc, pyn = d["o300"], d["onop"], d["py_class"], d["py_nopart"]
    top = np.argsort(-d["o300"]["condensable_particle_ugm3"][-1])[:N_FEEDBACK_CONDENSABLES]
    species = list(FEEDBACK_SPECIES) + [cm.species[k] for k in top]
    out = {}
    for sp in species:
        ip = list(map(str, pyc["species"])).index(sp)
        b = float(pyc["gas_ppb"][-1, ip]) / float(pyn["gas_ppb"][-1, ip]) - 1.0
        if sp == "SA":
            # TOMAS-JAX: SA is a per-interval counter; gas H2SO4 lives in Gc[SRTSO4]
            a = float(o300["h2so4_gas_ugm3"][-1]) / float(onop["h2so4_gas_ugm3"][-1]) - 1.0
            sp = "H2SO4 (SA)"
        else:
            io = list(map(str, o300["species"])).index(sp)
            a = float(o300["gas_ppb"][-1, io]) / float(onop["gas_ppb"][-1, io]) - 1.0
        if not (np.isfinite(a) and np.isfinite(b)):
            raise RuntimeError(f"feedback of {sp} is not finite")
        out[sp] = {"ours": a, "pycham": b, "condensable": sp in set(cm.species)}
    return out


# --- figures --------------------------------------------------------------------------------
def _legend_below(fig, axes, ncol=4):
    handles, labels = [], []
    for ax in np.ravel(axes):
        for h, lab in zip(*ax.get_legend_handles_labels()):
            if lab not in labels:
                handles.append(h)
                labels.append(lab)
    fig.legend(handles, labels, loc="outside lower center", ncols=ncol, frameon=False)


def fig_soa_timeseries(d, path):
    plotting.style()
    o300, o60, pyc, pye = d["o300"], d["o60"], d["py_class"], d["py_exact"]
    runs = [  # key, label, colour, ls, lw, kind
        ("py_class", LABEL_PY_CLASS, COLORS["pycham"], "-", 2.6, "py"),
        ("py_exact", LABEL_PY_EXACT, COLORS["pycham_alt"], "--", 1.8, "py"),
        ("o300", LABEL_OURS300, COLORS["ours"], "-", 1.8, "ours"),
        ("o60", LABEL_OURS60, COLORS["ours_alt"], "--", 1.6, "ours"),
    ]
    soa = {"o300": ours_particle_by_class(o300).sum(1), "o60": ours_particle_by_class(o60).sum(1),
           "py_class": py_particle_species(pyc).sum(1), "py_exact": py_particle_species(pye).sum(1)}
    gas = {"o300": gas_condensable_ugm3(o300, "ours").sum(1),
           "o60": gas_condensable_ugm3(o60, "ours").sum(1),
           "py_class": gas_condensable_ugm3(pyc, "py").sum(1),
           "py_exact": gas_condensable_ugm3(pye, "py").sum(1)}
    so4 = {"o300": ours_component(o300, "SO4") - ours_component(o300, "SO4")[0],
           "o60": ours_component(o60, "SO4") - ours_component(o60, "SO4")[0],
           "py_class": pyc["bin_sa_ugm3"].sum(1) - pyc["bin_sa_ugm3"][0].sum(),
           "py_exact": pye["bin_sa_ugm3"].sum(1) - pye["bin_sa_ugm3"][0].sum()}
    water = {"o300": ours_component(o300, "H2O"), "o60": ours_component(o60, "H2O"),
             "py_class": pyc["bin_water_ugm3"].sum(1), "py_exact": pye["bin_water_ugm3"].sum(1)}
    fig, axes = plt.subplots(2, 3, figsize=(15.5, 8.4), constrained_layout=True)
    panels = [(axes[0, 0], soa, "SOA (particle phase) [µg m⁻³]", "a) SOA mass"),
              (axes[0, 2], gas, "gas-phase mass [µg m⁻³]",
               "c) Condensable species, gas phase"),
              (axes[1, 0], {k: soa[k] + gas[k] for k in soa}, "gas + particle [µg m⁻³]",
               "d) Condensable species, gas + particle"),
              (axes[1, 1], so4, "sulfate added [µg m⁻³]", "e) Particle sulfate from H₂SO₄"),
              (axes[1, 2], water, "particle water [µg m⁻³]", "f) Particle water")]
    for ax, data, ylab, title in panels:
        for key, label, col, ls, lw, _ in runs:
            ax.plot(d[key]["t_s"] / 3600.0, data[key], color=col, ls=ls, lw=lw, label=label)
        ax.set_ylabel(ylab)
        ax.set_xlabel("time [h]")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.set_xlim(0, 6)
        ax.set_ylim(bottom=0)
    ax = axes[0, 1]
    ref_t, ref = pyc["t_s"], soa["py_class"]
    peak = 0.0
    for key, label, col, ls, lw, _ in runs[1:]:
        ia, ib = metrics.common_times(d[key]["t_s"], ref_t)
        ok = ref[ib] > 0.05                                   # skip the first minutes
        rel = 100.0 * (soa[key][ia][ok] / ref[ib][ok] - 1.0)
        ax.plot(d[key]["t_s"][ia][ok] / 3600.0, rel, color=col, ls=ls, lw=lw, label=label)
        peak = max(peak, float(np.max(np.abs(rel))))
    ax.axhline(0.0, color=COLORS["reference"], lw=0.8)
    for b in (-AGREEMENT_BAND, AGREEMENT_BAND):
        ax.axhline(100 * b, color=COLORS["band"], lw=0.8, ls=":")
    top = min(40.0, max(12.0, 1.15 * peak))
    ax.set_ylim(-15.0, top)
    if peak > top:
        ax.text(0.98, 0.97, f"y axis clipped; maximum {peak:+.0f} % at the first output (5 min)",
                transform=ax.transAxes, ha="right", va="top", fontsize=FONT_SIZE - 2)
    ax.set_xlim(0, 6)
    ax.set_xlabel("time [h]")
    ax.set_ylabel(f"(run − {LABEL_PYCHAM} class-centre) / {LABEL_PYCHAM} [%]")
    ax.set_title("b) SOA relative to PyCHAM (class-centre c*)", loc="left", fontweight="bold")
    _legend_below(fig, axes)
    fig.suptitle("S2: Level 2 limonene SOA, coagulation and nucleation off "
                 "(SOA = 245 condensable species; seed OA and sulfate excluded)",
                 fontsize=FONT_SIZE + 1)
    return plotting._save(fig, path)


def fig_class_bars(d, path):
    plotting.style()
    cm = _cmap()
    centers = np.asarray(VOLATILITY_LOG10_CSTAR_CENTERS)
    keys = ("o300", "py_class", "py_massfrac", "py_exact")
    part = {k: (ours_particle_by_class(d[k]) if k == "o300" else py_particle_by_class(d[k]))
            for k in keys}
    gas = {k: by_class(gas_condensable_ugm3(d[k], "ours" if k == "o300" else "py")) for k in keys}
    bars = [("o300", LABEL_OURS300, COLORS["ours"], None, "white"),
            ("py_class", LABEL_PY_CLASS, COLORS["pycham"], None, "white"),
            ("py_massfrac", "PyCHAM (class-centre c*, mass-fraction Raoult; attribution)",
             COLORS["attr_raoult"], "...", plotting.darker(COLORS["attr_raoult"])),
            ("py_exact", LABEL_PY_EXACT + ", grouped by its TOMAS-JAX class",
             COLORS["pycham_alt"], "///", plotting.darker(COLORS["pycham_alt"]))]
    times = S2["snapshot_times_h"]
    fig, axes = plt.subplots(2, len(times), figsize=(5.6 * len(times), 8.4),
                             constrained_layout=True, sharey="row")
    w = 0.21
    x = np.arange(centers.size)
    for col, th in enumerate(times):
        for row, (data, what) in enumerate(((part, "particle phase"), (gas, "gas phase"))):
            ax = axes[row, col]
            for j, (key, label, color, hatch, edge) in enumerate(bars):
                i = _idx(d[key]["t_s"], th)
                ax.bar(x + (j - 1.5) * w, data[key][i], width=w * 0.92, color=color,
                       hatch=hatch, edgecolor=edge, linewidth=0.0 if hatch is None else 0.4,
                       label=label)
            ax.set_xticks(x[::2])
            ax.set_xticklabels([f"{c:g}" for c in centers[::2]])
            ax.set_xlabel("class centre log₁₀ c* (298 K) [log₁₀ µg m⁻³]")
            ax.set_title(f"{'abc'[col] if row == 0 else 'def'[col]}) {what}, t = {th:g} h",
                         loc="left", fontweight="bold")
            ax.grid(axis="x", visible=False)
            ax.tick_params(labelleft=True)
        if col == 0:
            axes[0, col].set_ylabel("particle-phase mass per class [µg m⁻³]")
            axes[1, col].set_ylabel("gas-phase mass per class [µg m⁻³]")
    n_cls = np.bincount(cm.class_idx, minlength=centers.size)
    axes[1, 0].text(0.02, 0.97, "species per class: " + " ".join(str(n) for n in n_cls),
                    transform=axes[1, 0].transAxes, va="top", fontsize=FONT_SIZE - 3)
    _legend_below(fig, axes[0], ncol=2)
    fig.suptitle("S2: condensable organics per TOMAS-JAX c* class (half-decade classes)",
                 fontsize=FONT_SIZE + 1)
    return plotting._save(fig, path)


def fig_size_distributions(d, path):
    plotting.style()
    runs = [("py_class", LABEL_PY_CLASS, COLORS["pycham"], 2.6, "py", "s"),
            ("o300", LABEL_OURS300, COLORS["ours"], 1.8, "ours", "o")]
    times = S2["psd_times_h"]
    nominal = pycham_nominal_diameters(d)
    fig, axes = plt.subplots(3, len(times), figsize=(5.3 * len(times), 11.5),
                             constrained_layout=True, sharex=True)
    for col, th in enumerate(times):
        for key, label, color, lw, kind, mk in runs:
            dd, dn, dm, n = size_distribution(d[key], kind, _idx(d[key]["t_s"], th), nominal)
            ok = (dd > 5) & (dd < 2000)
            axes[0, col].plot(dd[ok], dn[ok], color=color, lw=lw, marker=mk, ms=4, label=label)
            axes[1, col].plot(dd[ok], dm[ok], color=color, lw=lw, marker=mk, ms=4, label=label)
            order = np.argsort(dd)
            for vals, ls, what in ((n, "-", "number"), (dm, "--", "dry mass")):
                c = np.cumsum(vals[order]) / vals.sum()
                axes[2, col].step(dd[order], c, where="post", color=color, ls=ls, lw=lw * 0.8,
                                  label=f"{label}: {what}")
        titles = (("dN/dlog₁₀Dp [cm⁻³]", "number"), ("dM/dlog₁₀Dp, dry [µg m⁻³]", "dry mass"),
                  ("cumulative fraction [-]", "cumulative number (solid) and dry mass (dashed)"))
        for row, (ylab, what) in enumerate(titles):
            ax = axes[row, col]
            ax.set_xscale("log")
            ax.set_xlim(10, 1000)
            ax.set_ylim(bottom=0, top=1.02 if row == 2 else None)
            ax.set_title(f"{'abcdefghi'[3 * row + col]}) {what}, t = {th:g} h", loc="left",
                         fontweight="bold", fontsize=FONT_SIZE - (1 if row == 2 else 0))
            ax.set_ylabel(ylab)
            if row == 2:
                ax.set_xlabel(DRY_DIAMETER_LABEL)
    _legend_below(fig, axes[2], ncol=2)
    fig.suptitle("S2: size distributions on the same 40-bin grid (markers: per-bin mean dry "
                 "diameter; PyCHAM moving-centre bins merge as the distribution narrows)",
                 fontsize=FONT_SIZE + 1)
    return plotting._save(fig, path)


def fig_gas_key_species(d, path):
    o300, o60, pyc = d["o300"], d["o60"], d["py_class"]
    species = list(S2["key_species"])
    m_air = float(o300["m_air_cm3"])

    def vals(run, kind):
        names = list(map(str, run["species"]))
        out = {}
        for sp in species:
            v = run["gas_ppb"][:, names.index(sp)].astype(float)
            if sp in RADICALS:
                v = v * (1e-9 * m_air if kind == "ours" else run["ppb_to_molec_cm3"])
            out[sp] = v
        return out

    series = [plotting.pycham_series(pyc["t_s"], vals(pyc, "py"), label=LABEL_PY_CLASS),
              plotting.ours_series(o300["t_s"], vals(o300, "ours"), label=LABEL_OURS300),
              plotting.ours_series(o60["t_s"], vals(o60, "ours"), label=LABEL_OURS60, alt=True)]
    units = {sp: ("[molec cm⁻³]" if sp in RADICALS else "[ppb]") for sp in species}
    return plotting.timeseries_grid(
        species, series, path, units=units, ncols=4,
        title="S2: gas-phase key species with particles present (value and difference to PyCHAM)")


def fig_feedback(fb, path):
    plotting.style()
    groups = [("a) Oxidants, inorganics, small products",
               [s for s in fb if not fb[s]["condensable"]]),
              ("b) Condensable species, gas phase (PAN: class 3.5; then the largest\n"
               "    particle-phase contributors)",
               [s for s in fb if fb[s]["condensable"]])]
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 6.0), constrained_layout=True)
    for ax, (title, sp) in zip(axes, groups):
        y = np.arange(len(sp))
        a = np.array([100 * fb[s]["ours"] for s in sp])
        b = np.array([100 * fb[s]["pycham"] for s in sp])
        ax.barh(y - 0.2, b, height=0.38, color=COLORS["pycham"], label=LABEL_PY_CLASS)
        ax.barh(y + 0.2, a, height=0.38, color=COLORS["ours"], label=LABEL_OURS300)
        ax.set_yticks(y)
        ax.set_yticklabels(sp)
        ax.invert_yaxis()
        ax.axvline(0, color=COLORS["reference"], lw=0.8)
        ax.set_xlabel("gas phase at 6 h: (with particles / without − 1) [%]")
        ax.set_title(title, loc="left", fontweight="bold")
        ax.grid(axis="y", visible=False)
        lim = 1.15 * max(np.max(np.abs(a)), np.max(np.abs(b)), 0.1)
        ax.set_xlim(-lim, lim if title.startswith("a") else 0.15 * lim)
    _legend_below(fig, axes, ncol=2)
    fig.suptitle("S2: gas phase with particles vs without, each model against its own run "
                 "without particles (SOA and H2SO4 uptake)", fontsize=FONT_SIZE + 1)
    return plotting._save(fig, path)


def fig_top_species(d, path):
    plotting.style()
    cm = _cmap()
    o300, pyc, pye = d["o300"], d["py_class"], d["py_exact"]
    n = S2["n_top_species"]
    a = o300["condensable_particle_ugm3"][-1].astype(float)
    b = py_particle_species(pyc)[-1]
    c = py_particle_species(pye)[-1]
    sel = list(dict.fromkeys(list(np.argsort(-b)[:n]) + list(np.argsort(-a)[:n])))
    sel.sort(key=lambda k: -(a[k] + b[k]))
    y = np.arange(len(sel))
    fig, ax = plt.subplots(figsize=(10.5, 0.42 * len(sel) + 2.2), constrained_layout=True)
    ax.barh(y - 0.27, b[sel], height=0.26, color=COLORS["pycham"], label=LABEL_PY_CLASS)
    ax.barh(y, a[sel], height=0.26, color=COLORS["ours"],
            label=f"{LABEL_OURS300} (per-species tracker)")
    ax.barh(y + 0.27, c[sel], height=0.26, color=COLORS["pycham_alt"], hatch="///",
            edgecolor=plotting.darker(COLORS["pycham_alt"]), linewidth=0.4, label=LABEL_PY_EXACT)
    rank_a = {k: r + 1 for r, k in enumerate(np.argsort(-a)[:n])}
    rank_b = {k: r + 1 for r, k in enumerate(np.argsort(-b)[:n])}
    labels = [f"{cm.species[k]}  (log c* {cm.log10_cstar[k]:.2f} → class "
              f"{VOLATILITY_LOG10_CSTAR_CENTERS[cm.class_idx[k]]:g}; rank P{rank_b.get(k, '–')}"
              f"/T{rank_a.get(k, '–')})" for k in sel]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=FONT_SIZE - 2)
    ax.invert_yaxis()
    ax.set_xlabel("particle-phase mass at 6 h [µg m⁻³]")
    ax.grid(axis="y", visible=False)
    ax.legend(loc="lower right", frameon=False)
    ax.set_title(f"S2: top-{n} particle-phase species of each model at 6 h "
                 "(rank P = PyCHAM class-centre, T = TOMAS-JAX)", loc="left")
    return plotting._save(fig, path)


def fig_gas_scatter(snaps, path):
    return plotting.scatter_snapshots(
        snaps, path, threshold_ppb=ABUNDANCE_THRESHOLD_PPB,
        title="S2: all gas-phase species, TOMAS-JAX vs PyCHAM (class-centre c*), with particles")


# --- attribution -----------------------------------------------------------------------------
LABEL_OURS_CS = f"{LABEL_OURS} (60 s, organic CS × PyCHAM/TOMAS-JAX transfer ratio)"
LABEL_PY_MASSFRAC = f"{LABEL_PYCHAM} (class-centre c*, mass-fraction Raoult)"


def soa_series(d, key):
    run = d[key]
    if key.startswith("py_"):
        return run["t_s"], py_particle_species(run).sum(1)
    return run["t_s"], ours_particle_by_class(run).sum(1)


def own_equilibrium_series(d, key):
    """SOA / (bulk equilibrium SOA from the run's own gas + particle totals) over time.

    Mass-fraction Raoult for TOMAS-JAX and PyCHAM 'massfrac', mole-fraction Raoult for
    PyCHAM 'class' (each model's own convention). No Kelvin term."""
    cm = _cmap()
    run = d[key]
    temp = d["info"]["setup"]["temperature_k"]
    c_nv = float(ours_component(d["o300"], "POA")[0])
    cst = cstar_at_temp(np.asarray(VOLATILITY_LOG10_CSTAR_CENTERS)[cm.class_idx], temp)
    if key.startswith("py_"):
        part, gas = py_particle_species(run), gas_condensable_ugm3(run, "py")
        mw = run["mw_g_mol"][_cols(run["species"], cm.species)]
    else:
        part, gas, mw = run["condensable_particle_ugm3"].astype(float), \
            gas_condensable_ugm3(run, "ours"), None
    mole = key == "py_class"
    mw_nv = d["info"]["setup"]["seed_oa_mw_pycham"]
    t, soa = soa_series(d, key)
    ratio = np.full(t.size, np.nan)
    for i in range(1, t.size):
        tot = part[i] + gas[i]
        eq = pankow(tot, cst, c_nv, mw if mole else None, mw_nv if mole else None).sum()
        if eq > 0.05:
            ratio[i] = soa[i] / eq - 1.0
    return t, ratio


LABEL_STEP = {   # cumulative attribution steps, each against PyCHAM
    "as_configured": f"{LABEL_OURS300} vs {LABEL_PY_CLASS} (as configured)",
    "coupling": f"{LABEL_OURS60} vs {LABEL_PY_CLASS}: coupling = PyCHAM update step",
    "transfer": "+ transfer rate matched (TOMAS-JAX organic CS × PyCHAM/TOMAS-JAX ratio)",
    "raoult": "+ Raoult matched (PyCHAM with mass-fraction Raoult)",
    "kelvin": "+ Kelvin term off in both (every attributed difference matched)",
}
ATTRIBUTION_PAIRS = (  # (run, reference, label, colour, linestyle)
    ("o300", "py_class", LABEL_STEP["as_configured"], COLORS["ours"], "-"),
    ("o60", "py_class", LABEL_STEP["coupling"], COLORS["ours_alt"], "--"),
    ("ocs", "py_class", LABEL_STEP["transfer"], COLORS["attr_transfer"], "-."),
    ("ocs", "py_massfrac", LABEL_STEP["raoult"], COLORS["attr_raoult"], "-"),
    ("ocs_nk", "py_massfrac_nokelvin", LABEL_STEP["kelvin"], COLORS["attr_matched"], "-"),
    ("py_massfrac", "py_class", "PyCHAM alone: mass- vs mole-fraction Raoult",
     COLORS["attr_raoult"], ":"),
    ("py_exact", "py_class", "PyCHAM alone: own Li 2016 c* vs class-centre c* (lumping)",
     COLORS["pycham_alt"], "--"),
)
TABLE_ONLY_PAIRS = (  # Kelvin effect inside each model (with / without - 1)
    ("ocs", "ocs_nk", "TOMAS-JAX alone: Kelvin on vs off (60 s, CS matched)"),
    ("py_massfrac", "py_massfrac_nokelvin", "PyCHAM alone: Kelvin on vs off (mass-fraction Raoult)"),
)


def attribution(d) -> dict:
    out = {}
    pairs = [(a, b, label) for a, b, label, _, _ in ATTRIBUTION_PAIRS] + list(TABLE_ONLY_PAIRS)
    for a, b, label in pairs:
        ta, sa = soa_series(d, a)
        tb, sb = soa_series(d, b)
        ia, ib = metrics.common_times(ta, tb)
        ok = sb[ib] > 0
        out[label] = {f"{th:g}h": float(sa[_idx(ta, th)] / sb[_idx(tb, th)] - 1.0)
                      for th in (0.25, 0.5, 1, 2, 3, 6)}
        out[label]["nmb"] = float((sa[ia][ok] - sb[ib][ok]).sum() / sb[ib][ok].sum())
    out["cs_factor"] = d["transfer"]["attribution_factor"]
    out["distance_from_own_bulk_equilibrium"] = {}
    for key in ("o300", "o60", "ocs", "py_class", "py_massfrac"):
        t, r = own_equilibrium_series(d, key)
        out["distance_from_own_bulk_equilibrium"][key] = {
            f"{th:g}h": float(r[_idx(t, th)]) for th in (0.5, 1, 3, 6)}
    return out


def fig_attribution(d, path):
    plotting.style()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2), constrained_layout=True)
    ax = axes[0]
    for a, b, label, color, ls in ATTRIBUTION_PAIRS:
        ta, sa = soa_series(d, a)
        tb, sb = soa_series(d, b)
        ia, ib = metrics.common_times(ta, tb)
        ok = sb[ib] > 0.05
        ax.plot(ta[ia][ok] / 3600.0, 100.0 * (sa[ia][ok] / sb[ib][ok] - 1.0), color=color,
                ls=ls, lw=2.2 if ls == "-" else 1.8, label=label)
    ax.axhline(0.0, color=COLORS["reference"], lw=0.8)
    for bnd in (-AGREEMENT_BAND, AGREEMENT_BAND):
        ax.axhline(100 * bnd, color=COLORS["band"], lw=0.8, ls=":")
    top = 16.0
    ta, sa = soa_series(d, "o300")
    tb, sb = soa_series(d, "py_class")
    ia, ib = metrics.common_times(ta, tb)
    ok = sb[ib] > 0.05
    first = 100.0 * (sa[ia][ok][0] / sb[ib][ok][0] - 1.0)
    ax.set_ylim(-4, top)
    ax.set_xlim(0, 6)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("SOA: (run / reference − 1) [%]")
    ax.set_title(f"a) SOA differences, run vs reference (y axis clipped at +{top:.0f} %; "
                 f"300 s coupling: {first:+.0f} % at {ta[ia][ok][0] / 60:.0f} min)", loc="left",
                 fontweight="bold", fontsize=FONT_SIZE - 1)
    ax.legend(frameon=False, fontsize=FONT_SIZE - 2, loc="upper right")
    ax = axes[1]
    for key, label, color, ls in (
            ("o300", LABEL_OURS300, COLORS["ours"], "-"),
            ("o60", LABEL_OURS60, COLORS["ours_alt"], "--"),
            ("ocs", LABEL_OURS_CS, COLORS["attr_transfer"], "-."),
            ("py_class", LABEL_PY_CLASS + ", mole-fraction eq.", COLORS["pycham"], "-"),
            ("py_massfrac", LABEL_PY_MASSFRAC + ", mass-fraction eq.", COLORS["attr_raoult"],
             ":")):
        t, r = own_equilibrium_series(d, key)
        ax.plot(t / 3600.0, 100.0 * r, color=color, ls=ls, lw=2.2 if ls == "-" else 1.8,
                label=label)
    ax.axhline(0.0, color=COLORS["reference"], lw=0.8)
    ax.set_ylim(-30, 5)
    ax.set_xlim(0, 6)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("SOA / bulk equilibrium SOA − 1 [%]")
    ax.set_title("b) Distance from each run's own bulk equilibrium (no Kelvin)", loc="left",
                 fontweight="bold")
    ax.legend(frameon=False, fontsize=FONT_SIZE - 2, loc="lower right")
    fig.suptitle("S2 attribution: coupling, mass-transfer rate, Raoult convention, Kelvin term "
                 f"and c* lumping (organic CS factor {d['transfer']['attribution_factor']:.3f})",
                 fontsize=FONT_SIZE + 1)
    return plotting._save(fig, path)


# --- within-class apportionment ----------------------------------------------------------------
def particle_fraction_species(d, key, i):
    """Per condensable species at output i: particle mass, gas mass, F = p / (g + p)."""
    run = d[key]
    if key.startswith("py_"):
        p, g = py_particle_species(run)[i], gas_condensable_ugm3(run, "py")[i]
    else:
        p, g = run["condensable_particle_ugm3"][i].astype(float), gas_condensable_ugm3(run, "ours")[i]
    with np.errstate(invalid="ignore", divide="ignore"):
        return p, g, p / (p + g)


def within_class(d, th=6.0, n_per_class=6, n_classes=4) -> dict:
    cm = _cmap()
    i = {k: _idx(d[k]["t_s"], th) for k in ("o300", "py_class", "py_massfrac")}
    po, go, fo = particle_fraction_species(d, "o300", i["o300"])
    pc, gc, fc = particle_fraction_species(d, "py_class", i["py_class"])
    pm, gm, fm = particle_fraction_species(d, "py_massfrac", i["py_massfrac"])
    cls_mass = np.bincount(cm.class_idx, weights=pc, minlength=cm.class_cstar.size)
    out = {}
    for k in np.argsort(-cls_mass)[:n_classes]:
        members = np.nonzero(cm.class_idx == k)[0]
        members = members[np.argsort(-pc[members])][:n_per_class]
        cp, cg = po[cm.class_idx == k].sum(), go[cm.class_idx == k].sum()
        out[f"{VOLATILITY_LOG10_CSTAR_CENTERS[k]:g}"] = {
            "class_particle_fraction_ours": float(cp / (cp + cg)),
            "members": [{"species": cm.species[m], "mw": float(cm.mw[m]),
                         "F_ours_tracker": float(fo[m]), "F_pycham_class": float(fc[m]),
                         "F_pycham_massfrac": float(fm[m]),
                         "particle_ours": float(po[m]), "particle_pycham_class": float(pc[m]),
                         "gas_ours": float(go[m]), "gas_pycham_class": float(gc[m])}
                        for m in members]}
    return out


def fig_within_class(wc, path, th=6.0):
    plotting.style()
    keys = list(wc)
    fig, axes = plt.subplots(1, len(keys), figsize=(4.4 * len(keys), 5.6),
                             constrained_layout=True, sharey=True)
    for ax, k in zip(np.atleast_1d(axes), keys):
        mem = wc[k]["members"]
        x = np.arange(len(mem))
        ax.axhline(wc[k]["class_particle_fraction_ours"], color=COLORS["ours"], lw=1.0, ls=":",
                   label=f"{LABEL_OURS} class total")
        ax.plot(x, [m["F_pycham_class"] for m in mem], "s", color=COLORS["pycham"], ms=9,
                label=LABEL_PY_CLASS)
        ax.plot(x, [m["F_pycham_massfrac"] for m in mem], "^", color=COLORS["attr_raoult"],
                ms=9, mfc="none", mew=1.8, label=LABEL_PY_MASSFRAC)
        ax.plot(x, [m["F_ours_tracker"] for m in mem], "o", color=COLORS["ours"], ms=8,
                label=f"{LABEL_OURS300} (per-species tracker)")
        ax.set_xticks(x)
        ax.set_xticklabels([f"{m['species']}\n{m['mw']:.0f} g/mol" for m in mem], rotation=60,
                           ha="right", fontsize=FONT_SIZE - 3)
        ax.set_ylim(0, 1.02)
        ax.set_title(f"class log₁₀ c* = {k}", loc="left", fontweight="bold")
        ax.grid(axis="x", visible=False)
    np.atleast_1d(axes)[0].set_ylabel(f"particle fraction p / (g + p) at {th:g} h [-]")
    _legend_below(fig, np.atleast_1d(axes), ncol=2)
    fig.suptitle("S2: members of the four most massive c* classes (same c* within a class in "
                 "every run shown)", fontsize=FONT_SIZE + 1)
    return plotting._save(fig, path)


# --- volatility sensitivity (PyCHAM's own vapour pressures) ------------------------------
VOLATILITY_FEEDBACK_SPECIES = ("LIMONENE", "O3", "NO", "NO2", "OH", "HO2", "HNO3", "PAN")


def nitrogen_share(d, key, th=6.0) -> dict:
    """Particle-phase SOA at ``th``: total, N-containing mass, its share and N [umol/m3]."""
    cm = _cmap()
    atoms = _chemistry()[0].atoms
    run = d[key]
    i = _idx(run["t_s"], th)
    if key.startswith("py_"):
        p = py_particle_species(run)[i]
        mw = run["mw_g_mol"][_cols(run["species"], cm.species)].astype(float)
    else:
        p, mw = run["condensable_particle_ugm3"][i].astype(float), cm.mw
    n_n = np.array([atoms[s].get("N", 0) for s in cm.species], float)
    has_n = n_n > 0
    return {"soa_ugm3": float(p.sum()), "n_containing_ugm3": float(p[has_n].sum()),
            "n_containing_share": float(p[has_n].sum() / p.sum()) if p.sum() > 0 else float("nan"),
            "n_umol_m3": float(np.sum(p / mw * n_n)),
            "top_species": [cm.species[k] for k in np.argsort(-p)[:5]]}


def volatility_sensitivity(d, fb) -> dict:
    """PyCHAM 'native' (own vapour pressures for the 245 condensables) vs 'class'."""
    pn, pc, pnop = d["py_native"], d["py_class"], d["py_nopart"]
    soa_n, soa_c = py_particle_species(pn).sum(1), py_particle_species(pc).sum(1)
    out = {"soa_ugm3": {k: {f"{th:g}h": float(s[_idx(r["t_s"], th)]) for th in (1, 3, 6)}
                        for k, s, r in (("py_native", soa_n, pn), ("py_class", soa_c, pc))}}
    out["soa_native_over_class"] = {th: out["soa_ugm3"]["py_native"][th]
                                    / out["soa_ugm3"]["py_class"][th] for th in ("1h", "3h", "6h")}
    out["nitrogen_6h"] = {k: nitrogen_share(d, k) for k in ("o300", "py_class", "py_native")}
    names_n, names_nop = list(map(str, pn["species"])), list(map(str, pnop["species"]))
    out["gas_feedback_6h"] = {}
    for sp in VOLATILITY_FEEDBACK_SPECIES:
        nat = float(pn["gas_ppb"][-1, names_n.index(sp)]) / float(pnop["gas_ppb"][-1, names_nop.index(sp)]) - 1
        out["gas_feedback_6h"][sp] = {"ours": fb[sp]["ours"], "pycham_class": fb[sp]["pycham"],
                                      "pycham_native": nat}
    return out


def fig_volatility(d, vs, path):
    plotting.style()
    fig, axes = plt.subplots(1, 3, figsize=(17.0, 5.6), constrained_layout=True,
                             gridspec_kw={"width_ratios": (1.25, 0.9, 1.25)})
    ax = axes[0]
    for key, label, col, ls, lw in (
            ("py_class", LABEL_PY_CLASS, COLORS["pycham"], "-", 2.6),
            ("py_exact", LABEL_PY_EXACT, COLORS["pycham_alt"], "--", 1.8),
            ("py_native", LABEL_PY_NATIVE, COLORS["pycham_native"], "-.", 1.8),
            ("o300", LABEL_OURS300, COLORS["ours"], "-", 1.8)):
        ts, s = soa_series(d, key)
        ax.plot(ts / 3600.0, s, color=col, ls=ls, lw=lw, label=label)
    ax.set_xlim(0, 6)
    ax.set_ylim(bottom=0)
    ax.set_xlabel("time [h]")
    ax.set_ylabel("SOA (particle phase) [µg m⁻³]")
    ax.set_title("a) SOA with three volatility inputs", loc="left", fontweight="bold")
    ax.legend(frameon=False, fontsize=FONT_SIZE - 2, loc="upper left")
    ax = axes[1]
    runs = (("o300", LABEL_OURS, COLORS["ours"]), ("py_class", "PyCHAM\nclass c*", COLORS["pycham"]),
            ("py_native", "PyCHAM\nown vp", COLORS["pycham_native"]))
    x = np.arange(len(runs))
    for j, (key, label, col) in enumerate(runs):
        ns = vs["nitrogen_6h"][key]
        ax.bar(x[j], ns["n_containing_ugm3"], color=col, width=0.6)
        ax.bar(x[j], ns["soa_ugm3"] - ns["n_containing_ugm3"], bottom=ns["n_containing_ugm3"],
               color="white", edgecolor=col, hatch="///", width=0.6, linewidth=1.0)
        ax.text(x[j], ns["soa_ugm3"], f"{100 * ns['n_containing_share']:.0f} % N", ha="center",
                va="bottom", fontsize=FONT_SIZE - 2)
    ax.set_xticks(x)
    ax.set_xticklabels([r[1] for r in runs])
    ax.set_ylabel("particle-phase SOA at 6 h [µg m⁻³]")
    ax.set_title("b) N-containing share of the SOA", loc="left", fontweight="bold")
    ax.grid(axis="x", visible=False)
    ax.set_ylim(0, 1.3 * max(vs["nitrogen_6h"][k]["soa_ugm3"] for k, _, _ in runs))
    from matplotlib.patches import Patch
    neutral = COLORS["reference"]
    ax.legend(handles=[Patch(facecolor=neutral, label="N-containing species (filled)"),
                       Patch(facecolor="white", edgecolor=neutral, hatch="///",
                             label="N-free species (hatched)")],
              frameon=False, fontsize=FONT_SIZE - 2, loc="upper right")
    ax = axes[2]
    sp = list(vs["gas_feedback_6h"])
    y = np.arange(len(sp))
    for j, (key, label, col) in enumerate((("ours", f"{LABEL_OURS300} (Li c* classes)", COLORS["ours"]),
                                           ("pycham_class", LABEL_PY_CLASS, COLORS["pycham"]),
                                           ("pycham_native", LABEL_PY_NATIVE, COLORS["pycham_native"]))):
        ax.barh(y + (j - 1) * 0.27, [100 * vs["gas_feedback_6h"][s][key] for s in sp], height=0.25,
                color=col, label=label)
    ax.set_yticks(y)
    ax.set_yticklabels(sp)
    ax.invert_yaxis()
    ax.axvline(0, color=COLORS["reference"], lw=0.8)
    ax.set_xlabel("gas phase at 6 h: (with particles / without − 1) [%]")
    ax.set_title("c) Gas-phase response to partitioning", loc="left", fontweight="bold")
    ax.grid(axis="y", visible=False)
    ax.legend(frameon=False, fontsize=FONT_SIZE - 2, loc="lower right")
    fig.suptitle("S2 volatility sensitivity (PyCHAM only): the NOx-sink feedback and the SOA amount "
                 "follow the c* placeholder (A8)", fontsize=FONT_SIZE + 1)
    return plotting._save(fig, path)


# --- measured numbers for the mismatch list ----------------------------------------------
def density_rule_difference(d, th=6.0) -> dict:
    """TOMAS-JAX Tang density vs PyCHAM's additive rule on TOMAS-JAX's bulk particle
    composition (dry at the effective dry density of the set-up, water 1000 kg/m3)."""
    o = d["o300"]
    rho_d = d["info"]["setup"]["seed"]["rho_dry_pycham"]
    out = {}
    for tt in (0.0, th):
        comp = o["particle_component_ugm3"][_idx(o["t_s"], tt)].astype(float)
        rho_t = float(np.asarray(calc_density(np.asarray(comp)[None, :]))[0])
        names = list(map(str, o["component_names"]))
        m_w = comp[names.index("H2O")]
        m_dry = comp.sum() - m_w
        rho_p = (m_dry + m_w) / (m_dry / rho_d + m_w / PYCHAM_WATER_DENSITY)
        out[f"{tt:g}h"] = {"tomas_kg_m3": rho_t, "pycham_rule_kg_m3": rho_p,
                           "rel": rho_p / rho_t - 1.0}
    return out


def measured(d, part, attr, wc, vs) -> dict:
    cm = _cmap()
    rd = part["soa_rel_diff_vs_pycham_class"]
    tr = d["transfer"]
    wcls = wc["0.5"]["members"] if "0.5" in wc else next(iter(wc.values()))["members"]
    return {
        "soa_rel_o300": {k: rd["o300"][k] for k in ("0.5h", "1h", "3h", "6h")},
        "soa_rel_o60": {k: rd["o60"][k] for k in ("0.5h", "1h", "3h", "6h")},
        "soa_6h": (part["soa_ugm3"]["o300"]["6h"], part["soa_ugm3"]["py_class"]["6h"]),
        "coupling_60_vs_300": part["coupling_effect_ours_60_vs_300"],
        "cs_factor": tr["attribution_factor"],
        "d_ratio": tr["t0_tomas_seed"]["d_ratio_weighted"],
        "cs_ratio_6h": tr["6h_pycham_sizes"]["cs_ratio_weighted"],
        "steps": {k: attr[LABEL_STEP[k]] for k in LABEL_STEP},
        "raoult_alone": attr["PyCHAM alone: mass- vs mole-fraction Raoult"],
        "kelvin_ours": attr[TABLE_ONLY_PAIRS[0][2]], "kelvin_pycham": attr[TABLE_ONLY_PAIRS[1][2]],
        "lumping": attr["PyCHAM alone: own Li 2016 c* vs class-centre c* (lumping)"],
        "water": part["particle_water_ugm3"], "water_rel": part["particle_water_rel_diff"],
        "within_class_F": {"ours": [m["F_ours_tracker"] for m in wcls],
                           "pycham": [m["F_pycham_class"] for m in wcls]},
        "size_6h": {k: v["6h"] for k, v in part["dry_size_stats"].items()},
        "density": density_rule_difference(d),
        "nitrogen": vs["nitrogen_6h"], "native": vs["soa_native_over_class"],
        "native_feedback": vs["gas_feedback_6h"],
        "sulfate_6h": (part["sulfate_condensed_ugm3"]["o300"]["6h"],
                       part["sulfate_condensed_ugm3"]["py_class"]["6h"]),
        "seed_sa_mw": d["py_class"]["mw_g_mol"][list(map(str, d["py_class"]["species"])).index("SA")],
        "n_condensable": len(cm.species),
    }


def gas_only_baseline(d) -> dict:
    """Gas phase of the two no-particle runs: the G2-like agreement S2 starts from."""
    o, py = d["onop"], d["py_nopart"]
    names = metrics.common_species(o["species"], py["species"])
    ia, ib = metrics.common_times(o["t_s"], py["t_s"])
    x = o["gas_ppb"][ia][:, _cols(o["species"], names)].astype(float)
    r = py["gas_ppb"][ib][:, _cols(py["species"], names)].astype(float)
    st = metrics.species_stats(names, x, r)
    ok = st["n_valid"] > 0
    return {"n_species": int(ok.sum()), "median_abs_nmb": float(np.nanmedian(np.abs(st["nmb"][ok]))),
            "p95_abs_nmb": float(np.nanpercentile(np.abs(st["nmb"][ok]), 95)),
            "frac_abs_nmb_within_band": metrics.fraction_within(st["nmb"][ok]),
            "max_abs_nmb": float(np.nanmax(np.abs(st["nmb"][ok])))}


def main(argv=None) -> int:
    d = load_all()
    case.FIG_DIR.mkdir(parents=True, exist_ok=True)
    stats, snaps, gas_summ = gas_phase_stats(d)
    metrics.write_stats_csv(case.GAS_STATS_CSV, stats)
    part = particle_metrics(d)
    fb = feedback(d)
    attr = attribution(d)
    wc = within_class(d)
    baseline = gas_only_baseline(d)
    vs = volatility_sensitivity(d, fb)
    meas = measured(d, part, attr, wc, vs)
    figs = [
        fig_soa_timeseries(d, case.FIG_DIR / "S2_soa_timeseries.png"),
        fig_class_bars(d, case.FIG_DIR / "S2_cstar_class_bars.png"),
        fig_size_distributions(d, case.FIG_DIR / "S2_size_distributions.png"),
        fig_gas_key_species(d, case.FIG_DIR / "S2_gas_key_species.png"),
        fig_feedback(fb, case.FIG_DIR / "S2_partitioning_feedback.png"),
        fig_top_species(d, case.FIG_DIR / "S2_top_species.png"),
        fig_within_class(wc, case.FIG_DIR / "S2_within_class.png"),
        fig_attribution(d, case.FIG_DIR / "S2_attribution.png"),
        fig_gas_scatter(snaps, case.FIG_DIR / "S2_gas_scatter.png"),
        fig_volatility(d, vs, case.FIG_DIR / "S2_volatility_sensitivity.png"),
    ]
    summary = {"particles": part, "gas_phase": gas_summ, "gas_phase_no_particles": baseline,
               "partitioning_feedback_6h": fb, "attribution": attr, "within_class_6h": wc,
               "transfer_rates": d["transfer"], "volatility_sensitivity": vs,
               "measured_for_mismatches": meas,
               "mismatches": case.mismatches(d["info"]["setup"], meas),
               "threshold_ppb": ABUNDANCE_THRESHOLD_PPB, "figures": [p.name for p in figs]}
    case.SUMMARY_JSON.write_text(json.dumps(summary, indent=2) + "\n")
    DOC_FIG_DIR.mkdir(parents=True, exist_ok=True)
    for p in figs:
        shutil.copyfile(p, DOC_FIG_DIR / p.name)
    print(json.dumps({"soa": part["soa_ugm3"], "attribution": attr, "baseline": baseline},
                     indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
