"""Diagnostics, comparison table and plots for the MCM-subset vs SANDBOX runs.

Definitions (analysis window W = day `analysis_day` of the run, 24 h; noon = time of
minimum SZA in W; brackets [X] = number density, molec cm-3):

  OH_noon, HO2_noon      [OH], [HO2] at noon
  <OH>_24h                mean [OH] over W
  P_HOx(O1D)              primary HOx source 2 k(O1D + H2O) [O1D] (+ k(O1D + CH4)[O1D][CH4]
                          in SANDBOX, which makes one OH and a CH3 sink) at noon
  dO3                     O3(end of W) - O3(start of W)  [ppb per day]
  NO/NO2                  ratio at noon;  NOx/NOy at noon, NOy = NO + NO2 + NO3 + 2 N2O5 +
                          HNO3 + HONO + HNO4 (+ CH3NO3 + CH3O2NO2 in MCM)
  tau_SO2(OH)             1 / (k_OH+SO2 <OH>_24h)                      [days]
  tau_SO2(model)          24 h / ln(SO2_start / SO2_end) over W        [days]
  P_H2SO4 noon            d[H2SO4]/dt at noon (centred difference of the 300 s output;
                          H2SO4 = SA in MCM) [molec cm-3 s-1]
  P_H2SO4 24h             H2SO4 made in W [ppt per day] (= SO2 lost, since SO3 and HSO3
                          are short-lived)
  SO2 loss not via OH     share of the SO2 loss over W through O + SO2 (MCM <30>) or
                          HO2 + SO2 (SANDBOX, JPL upper limit with assumed products)
  O3 budget over W        [ppb/day], integrals of the rate over W:
     P(O3)   = k(HO2+NO)[HO2][NO] (+ k(CH3O2+NO->NO2)[CH3O2][NO] in MCM)
     L(HOx)  = k(HO2+O3)[HO2][O3] + k(OH+O3)[OH][O3]
     L(O1D)  = k(O1D+H2O)[O1D] (+ k(O1D+CH4)[O1D][CH4] in SANDBOX)
     L(NOx)  = 2 k(O+NO2->NO)[O][NO2];   L(Ox) = 2 k(O+O3)[O][O3]

Run: <mech-wt>/.venv/bin/python compare.py   (numpy + matplotlib only)
Writes results/comparison.json, results/comparison.md and three PNGs.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

HERE = Path(__file__).parent
RES = HERE / "results"
RUNS = [("mcm", "matched"), ("mcm", "background"), ("mcm", "jplphot"), ("sandbox", "opt0"),
        ("sandbox", "opt1"), ("sandbox", "opt0_fill")]
LABEL = {("mcm", "matched"): "MCM subset, matched start",
         ("mcm", "background"): "MCM subset + CO, H2",
         ("mcm", "jplphot"): "MCM subset + J(N2O5), J(HNO4)",
         ("sandbox", "opt0"): "SANDBOX JPL 19-5 (opt 0)",
         ("sandbox", "opt1"): "SANDBOX JPL 19-5 (opt 1)",
         ("sandbox", "opt0_fill"): "SANDBOX (opt 0) + J(O3->O3P), HNO4/N2O5 decomp."}
# reference categorical palette, slots 1-5 (light mode); opt 1 reuses slot 2 dashed;
# the two dotted sensitivity runs carry linestyle as a second encoding
STYLE = {("mcm", "matched"): dict(color="#2a78d6", ls="-"),
         ("mcm", "background"): dict(color="#1baf7a", ls="-"),
         ("mcm", "jplphot"): dict(color="#eda100", ls=(0, (1, 1.5))),
         ("sandbox", "opt0"): dict(color="#eb6834", ls="-"),
         ("sandbox", "opt1"): dict(color="#eb6834", ls=(0, (4, 2))),
         ("sandbox", "opt0_fill"): dict(color="#e87ba4", ls=(0, (1, 1.5)))}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load(model, variant, case):
    return json.loads((RES / f"{model}_{case}_{variant}.json").read_text())


def series(r, s):
    alias = {"mcm": {"H2SO4": "SA", "HNO4": "HO2NO2"}, "sandbox": {}}
    key = alias["mcm" if r["model"].startswith("MCM") else "sandbox"].get(s, s)
    return np.asarray(r["conc"][key]) if key in r["conc"] else None


def diagnostics(r, case, jt):
    is_mcm = r["model"].startswith("MCM")
    t = np.asarray(r["t_s"])
    M = r["M"]
    d = case["analysis_day"]
    w = (t >= (d - 1) * 86400 - 1) & (t <= d * 86400 + 1)
    sza = np.interp(t, jt["t_s"], jt["sza_deg"])
    iw = np.flatnonzero(w)
    inoon = iw[np.argmin(sza[iw])]
    c = {s: series(r, s) for s in ["OH", "HO2", "O3", "NO", "NO2", "NO3", "N2O5", "HNO3",
                                   "HONO", "HNO4", "CH3NO3", "CH3O2NO2", "SO2", "H2SO4",
                                   "O1D", "CH4", "O", "CH3O2"]}
    k = r["k"]
    k_so2 = k["OH+SO2 (KMT12)"] if is_mcm else k["OH+SO2 (JPL I4)"]
    oh_mean = float(np.mean(c["OH"][iw]))
    noy = sum(c[s][inoon] * f for s, f in [("NO", 1), ("NO2", 1), ("NO3", 1), ("N2O5", 2),
                                          ("HNO3", 1), ("HONO", 1), ("HNO4", 1),
                                          ("CH3NO3", 1), ("CH3O2NO2", 1)] if c[s] is not None)
    if is_mcm:
        phox = 2 * k["O1D -> 2OH (<13>, 1/s)"] * c["O1D"][inoon]
    else:
        phox = (2 * k["O1D -> 2OH (1/s)"] + k["O1D+CH4"] * c["CH4"][inoon]) * c["O1D"][inoon]
    h2 = c["H2SO4"]
    p_noon = (h2[inoon + 1] - h2[inoon - 1]) / (t[inoon + 1] - t[inoon - 1])
    so2_0, so2_1 = c["SO2"][iw[0]], c["SO2"][iw[-1]]
    tw = t[iw]

    def integ(x):                                   # integral over W, in ppb
        return float(np.trapezoid(x[iw], tw) / M * 1e9)

    def kk(mcm_key, sb_key):
        return k[mcm_key] if is_mcm else k[sb_key]

    so2_oh = integ(k_so2 * c["OH"] * c["SO2"])
    so2_other = (integ(k["O+SO2 (<30>)"] * c["O"] * c["SO2"]) if is_mcm
                 else integ(k["HO2+SO2 (JPL I34 upper limit)"] * c["HO2"] * c["SO2"]))
    p_o3 = integ(kk("HO2+NO (<24>)", "HO2+NO (C10)") * c["HO2"] * c["NO"])
    if is_mcm:
        p_o3 += integ(k["CH3O2+NO -> NO2 (<47>)"] * c["CH3O2"] * c["NO"])
        l_o1d = integ(k["O1D -> 2OH (<13>, 1/s)"] * c["O1D"])
    else:
        l_o1d = integ((k["O1D -> 2OH (1/s)"] + k["O1D+CH4"] * c["CH4"]) * c["O1D"])
    l_hox = integ((kk("HO2+O3 (<18>)", "HO2+O3") * c["HO2"]
                   + kk("OH+O3 (<14>)", "OH+O3") * c["OH"]) * c["O3"])
    l_nox = integ(2 * kk("O+NO2 -> NO (<4>)", "O+NO2 -> NO (K1 activation)") * c["O"] * c["NO2"])
    l_ox = integ(2 * kk("O+O3 (<2>)", "O+O3") * c["O"] * c["O3"])
    return {
        "OH_noon": float(c["OH"][inoon]), "HO2_noon": float(c["HO2"][inoon]),
        "OH_24h_mean": oh_mean, "HO2_OH_noon": float(c["HO2"][inoon] / c["OH"][inoon]),
        "P_HOx_O1D_noon": float(phox),
        "O3_start_ppb": float(c["O3"][iw[0]] / M * 1e9),
        "dO3_ppb_per_day": float((c["O3"][iw[-1]] - c["O3"][iw[0]]) / M * 1e9),
        "NO_NO2_noon": float(c["NO"][inoon] / c["NO2"][inoon]),
        "NOx_ppt_noon": float((c["NO"][inoon] + c["NO2"][inoon]) / M * 1e12),
        "NOx_NOy_noon": float((c["NO"][inoon] + c["NO2"][inoon]) / noy),
        "HNO4_ppt_noon": float(c["HNO4"][inoon] / M * 1e12),
        "N2O5_ppt_noon": float(c["N2O5"][inoon] / M * 1e12),
        "k_OH_SO2": float(k_so2),
        "tau_SO2_OH_days": float(1.0 / (k_so2 * oh_mean) / 86400),
        "tau_SO2_model_days": float(1.0 / np.log(so2_0 / so2_1)),
        "P_H2SO4_noon": float(p_noon),
        "P_H2SO4_ppt_per_day": float((h2[iw[-1]] - h2[iw[0]]) / M * 1e12),
        "noon_utc_h": float(t[inoon] / 3600 % 24),
        "SO2_loss_not_OH_frac": so2_other / (so2_oh + so2_other),
        "P_O3": p_o3, "L_HOx": l_hox, "L_O1D": l_o1d, "L_NOx": l_nox, "L_Ox": l_ox,
    }


ROWS = [  # key, label, format
    ("OH_noon", "OH at noon [cm-3]", "{:.2e}"),
    ("OH_24h_mean", "OH 24 h mean [cm-3]", "{:.2e}"),
    ("HO2_noon", "HO2 at noon [cm-3]", "{:.2e}"),
    ("HO2_OH_noon", "HO2/OH at noon", "{:.0f}"),
    ("P_HOx_O1D_noon", "2 k[O1D][H2O] (+O1D+CH4) at noon [cm-3 s-1]", "{:.2e}"),
    ("O3_start_ppb", "O3 at start of analysis day [ppb]", "{:.1f}"),
    ("dO3_ppb_per_day", "O3 change over analysis day [ppb/day]", "{:+.2f}"),
    ("NO_NO2_noon", "NO/NO2 at noon", "{:.2f}"),
    ("NOx_ppt_noon", "NOx at noon [ppt]", "{:.0f}"),
    ("NOx_NOy_noon", "NOx/NOy at noon", "{:.2f}"),
    ("HNO4_ppt_noon", "HNO4 at noon [ppt]", "{:.1f}"),
    ("N2O5_ppt_noon", "N2O5 at noon [ppt]", "{:.1f}"),
    ("k_OH_SO2", "k(OH+SO2) [cm3 s-1]", "{:.2e}"),
    ("tau_SO2_OH_days", "SO2 lifetime 1/(k<OH>) [days]", "{:.1f}"),
    ("tau_SO2_model_days", "SO2 lifetime from model decay [days]", "{:.1f}"),
    ("P_H2SO4_noon", "H2SO4 production at noon [cm-3 s-1]", "{:.2e}"),
    ("P_H2SO4_ppt_per_day", "H2SO4 produced over analysis day [ppt/day]", "{:.0f}"),
    ("SO2_loss_not_OH_frac", "SO2 loss not via OH (MCM: O+SO2; SANDBOX: HO2+SO2)", "{:.1e}"),
    ("P_O3", "O3 budget: P = HO2+NO (+CH3O2+NO) [ppb/day]", "{:.2f}"),
    ("L_HOx", "O3 budget: L = HO2+O3, OH+O3 [ppb/day]", "{:.2f}"),
    ("L_O1D", "O3 budget: L = O1D+H2O (+O1D+CH4) [ppb/day]", "{:.2f}"),
    ("L_NOx", "O3 budget: L = 2 O+NO2 [ppb/day]", "{:.2f}"),
    ("L_Ox", "O3 budget: L = 2 O+O3 [ppb/day]", "{:.2f}"),
]


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.8)
    for sp in ("top", "right"):
        ax.spines[sp].set_visible(False)
    for sp in ("left", "bottom"):
        ax.spines[sp].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=8)


def timeseries_plot(case_name, case, runs, path):
    nox_panel = (("RES", "HNO4 + 2 N2O5 [ppt] (NOx held at 1 ppb)", 1e12)
                 if case.get("hold_NOx_ppb") else ("NOx", "NOx = NO + NO2 [ppt]", 1e12))
    panels = [("OH", "OH [molec cm-3]", 1.0), ("HO2", "HO2 [molec cm-3]", 1.0),
              ("O3", "O3 [ppb]", 1e9), nox_panel,
              ("SO2", "SO2 [ppb]", 1e9), ("H2SO4", "H2SO4 made [ppt]", 1e12)]
    fig, axes = plt.subplots(3, 2, figsize=(10, 8.5), sharex=True)
    for ax, (s, lab, f) in zip(axes.flat, panels):
        for key, r in runs.items():
            if key == ("sandbox", "opt1"):           # table only, keeps the plot readable
                continue
            t = np.asarray(r["t_s"]) / 3600
            if s == "NOx":
                y = series(r, "NO") + series(r, "NO2")
            elif s == "RES":
                y = series(r, "HNO4") + 2 * series(r, "N2O5")
            else:
                y = series(r, s)
            ax.plot(t, y / r["M"] * f if f != 1.0 else y, lw=2,
                    label=LABEL[key], **STYLE[key])
        ax.set_ylabel(lab, color=INK, fontsize=9)
        style_axes(ax)
    for ax in axes[-1]:
        ax.set_xlabel("hours since 00 UTC", color=INK, fontsize=9)
    d = case["analysis_day"]
    for ax in axes.flat:
        ax.axvspan((d - 1) * 24, d * 24, color="#f1f0ec", zorder=0)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False, fontsize=8, labelcolor=INK)
    fig.suptitle(f"{case['label']}  (shaded: analysis day)", color=INK, fontsize=10)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(path, dpi=130)
    plt.close(fig)


def rate_plot(rates, path):
    """JPL 19-5 (SANDBOX) / MCM v3.3.1 rate-coefficient ratio for shared reactions."""
    so3 = ("SO3+H2O (1st order)", "SO3+H2O (<35>, 1/s)", "SO3+H2O (I79, 1/s)")
    pairs = [("OH+SO2", "OH+SO2 (KMT12)", "OH+SO2 (JPL I4)"),
             ("OH+NO2+M", "OH+NO2 (KMT08)", "OH+NO2 (C4)"),
             ("HO2+NO2+M", "HO2+NO2+M (KMT09)", "HO2+NO2+M (C6)"),
             ("HO2+NO", "HO2+NO (<24>)", "HO2+NO (C10)"),
             ("NO+O3", "NO+O3 (<7>)", "NO+O3 (C19)"),
             ("OH+CH4", "OH+CH4 (<46>)", "OH+CH4 (D14)"),
             ("HO2+HO2", "HO2+HO2 (<20>)", "HO2+HO2 (B13)"),
             ("OH+HO2", "OH+HO2 (<19>)", "OH+HO2"),
             ("OH+O3", "OH+O3 (<14>)", "OH+O3"),
             ("HO2+O3", "HO2+O3 (<18>)", "HO2+O3"),
             ("O1D+H2O (1st order)", "O1D -> 2OH (<13>, 1/s)", "O1D -> 2OH (1/s)"),
             ("O1D quenching (1st order)", "O1D -> O (<6>, 1/s)", "O1D -> O (1/s, opt=0)")]
    fig, ax = plt.subplots(figsize=(7.5, 5.2))
    yy = np.arange(len(pairs))
    out = {}
    for off, (cname, col, mk) in zip((-0.17, 0.17), (("trop", "#2a78d6", "o"),
                                                       ("strat", "#eb6834", "s"))):
        km, ks = rates[cname]["mcm"], rates[cname]["sandbox"]
        ratio = np.array([ks[b] / km[a] for _, a, b in pairs])
        out[cname] = {p[0]: float(v) for p, v in zip(pairs, ratio)}
        out[cname][so3[0]] = float(ks[so3[2]] / km[so3[1]])
        ax.hlines(yy + off, 1.0, ratio, color=col, lw=2)
        ax.plot(ratio, yy + off, mk, color=col, ms=7, mec="white", mew=1.5,
                label={"trop": "298 K, 1013 hPa", "strat": "220 K, 50 hPa"}[cname])
    ax.axvline(1.0, color=INK2, lw=1)
    ax.set_xscale("log")
    ax.set_xlim(0.5, 2.5)
    ax.set_xticks([0.5, 0.75, 1, 1.5, 2, 2.5], ["0.5", "0.75", "1", "1.5", "2", "2.5"])
    ax.minorticks_off()
    for cname, col in (("trop", "#2a78d6"), ("strat", "#eb6834")):
        v = out[cname]["OH+SO2"]
        ax.annotate(f"{v:.2f}", (v, -0.17 if cname == "trop" else 0.17), xytext=(6, 0),
                    textcoords="offset points", va="center", fontsize=8, color=INK2)
    ax.text(0.0, -0.16, f"Not shown: SO3 + H2O, ratio {out['trop'][so3[0]]:.0f} at 298 K and "
            f"{out['strat'][so3[0]]:.1f} at 220 K\n(MCM first order in H2O, JPL second order); "
            "in both models far faster than SO2 loss.", transform=ax.transAxes, fontsize=7,
            color=INK2, va="top")
    ax.set_yticks(yy, [p[0] for p in pairs], fontsize=8, color=INK)
    ax.invert_yaxis()
    ax.set_xlabel("k(JPL 19-5, SANDBOX) / k(MCM v3.3.1)", color=INK, fontsize=9)
    style_axes(ax)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    ax.set_title("Rate coefficients of shared reactions at the two box conditions",
                 color=INK, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return out


def main() -> None:
    cases = json.loads((HERE / "cases_resolved.json").read_text())
    table, rates = {}, {}
    for cname, case in cases.items():
        jt = json.loads((RES / f"jtable_{cname}.json").read_text())
        runs = {key: load(key[0], key[1], cname) for key in RUNS}
        table[cname] = {LABEL[key]: diagnostics(r, case, jt) for key, r in runs.items()}
        rates[cname] = {"mcm": runs[("mcm", "matched")]["k"],
                        "sandbox": runs[("sandbox", "opt0")]["k"]}
        timeseries_plot(cname, case, runs, HERE / f"fig_{cname}_timeseries.png")
    ratios = rate_plot(rates, HERE / "fig_rate_ratios.png")
    (RES / "comparison.json").write_text(json.dumps(
        {"diagnostics": table, "rate_constants": rates, "rate_ratio_jpl_over_mcm": ratios},
        indent=1))
    md = []
    for cname, case in cases.items():
        cols = list(table[cname])
        md.append(f"\n### {case['label']} (analysis day {case['analysis_day']})\n")
        md.append("| Quantity | " + " | ".join(cols) + " |")
        md.append("|---|" + "---|" * len(cols))
        for key, lab, fmt in ROWS:
            md.append(f"| {lab} | " + " | ".join(fmt.format(table[cname][c][key]) for c in cols)
                      + " |")
    (RES / "comparison.md").write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print("\nJPL/MCM rate ratios:", json.dumps(ratios, indent=1))


if __name__ == "__main__":
    main()
