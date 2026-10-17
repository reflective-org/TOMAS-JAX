"""Markdown tables for the JAX sparse spike from the result files.

    python make_report.py [--final-tag final] [--nominal-tag nominal] [--sweep-prefix sw_r]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
sys.path.insert(0, str(HERE.parent))
from spike.paths import BENCH  # noqa: E402
from bench_common import REFERENCE  # noqa: E402
from compare import score  # noqa: E402

CASES = [("lim726_L2", 900), ("lim726_L3", 900), ("syn1353_L3", 900), ("syn4488_L3", 900),
         ("syn5742_L3", 900), ("lim726_L2", 60), ("lim726_L3", 60), ("syn1353_L3", 60),
         ("syn4488_L3", 60), ("syn5742_L3", 60)]
SCIPY = [("scipysparse_a", "1e-3, 1e4"), ("scipysparse_rtol1e-4", "1e-4, 1e4"),
         ("scipysparse_rtol1e-5", "1e-5, 1e4"), ("scipysparse_rtol1e-3_atol1e2", "1e-3, 1e2"),
         ("scipysparse_rtol1e-4_atol1e2", "1e-4, 1e2"), ("scipysparse_rtol1e-5_atol1e2", "1e-5, 1e2"),
         ("scipysparse_rtol1e-6_atol1e2", "1e-6, 1e2")]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def ld(u: str) -> str:
    return u.split("load averages:")[1].split()[0] if u and "load averages:" in u else "?"


def dense(p, c):
    t = load(BENCH / "results" / f"jaxdense_{p}_dt{c}_timing.json")
    if t is None or "accuracy" not in t:
        return None, 5e-2, None
    e = t["accuracy"]["key"]["max"]
    tw = t.get("t_warm_median_s") or t["t_first_s"]
    return e, min(5e-2, max(2 * e, 1e-2)), (tw, bool(t.get("t_warm_median_s")))


def scipy_best(p, c, gate):
    best = None
    for tag, lab in SCIPY:
        f = BENCH / "results" / f"{tag}_{p}_dt{c}.json"
        if not f.exists():
            continue
        e = score(f, REFERENCE / f"ref_{p}_dt{c}.json")["key"]["max"]
        w = load(f)["stats"]["wall_s"]
        if e <= gate and (best is None or w < best[1]):
            best = (lab, w, e)
    return best


def sweep_table(prefix, rtols):
    head = " | ".join(f"rtol {r}: warm [s], steps, E_key" for r in rtols)
    rows = [f"| Case | gate | {head} |", "|---|---|" + "---|" * len(rtols)]
    for p, c in CASES:
        _, gate, _ = dense(p, c)
        cells = []
        for r in rtols:
            t = load(RES / f"{prefix}{r}_{p}_dt{c}_timing.json")
            if t is None or "accuracy" not in t:
                cells.append("-")
                continue
            e = t["accuracy"]["key"]["max"]
            w = t.get("t_warm_median_s") or t["t_first_s"]
            ok = "**pass**" if e <= gate else "fail"
            cells.append(f"{w:.2f}, {t['accepted_steps']}, {e:.1e} {ok}")
        rows.append(f"| {p} {c} s | {gate:.1e} | " + " | ".join(cells) + " |")
    return "\n".join(rows)


def run_table(tag):
    rows = ["| Case | rtol, atol | t_setup [s] (symbolic) | t_first [s] | t_first_interval [s] | "
            "t_warm median [s] (runs) | steps (rejected) | ms per step | peak RSS [MB] | "
            "E_key (worst) | E_all | gate | 1-min load |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, c in CASES:
        t = load(RES / f"{tag}_{p}_dt{c}_timing.json")
        if t is None:
            continue
        s = t["settings"]
        _, gate, _ = dense(p, c)
        if t["status"] != "complete":
            rows.append(f"| {p} {c} s | {s['rtol']:g}, {s['atol']:g} | {t['status']} |")
            continue
        warm = [r["wall_s"] for r in t["runs"] if r["kind"] == "warm" and "wall_s" in r]
        loads = ", ".join(ld(r["uptime_before"]) for r in t["runs"])
        acc = t["accuracy"]
        e = acc["key"]["max"]
        rows.append(
            f"| {p} {c} s | {s['rtol']:g}, {s['atol']:g} | {t['t_setup_s']:.1f} ({t['t_symbolic_s']:.1f}) | "
            f"{t['t_first_s']:.2f} | {t['t_first_interval_s']:.2f} | "
            f"{t.get('t_warm_median_s', float('nan')):.2f} ({', '.join(f'{w:.2f}' for w in warm)}) | "
            f"{t['accepted_steps']} ({t['work']['rejected']}) | {t.get('ms_per_step_warm_median', float('nan')):.2f} | "
            f"{t['peak_rss_mb']:.0f} | {e:.1e} ({acc['key']['worst_species']}) | "
            f"{acc.get('all_species_hourly_max', float('nan')):.1e} | {gate:.1e} {'pass' if e <= gate else 'FAIL'} | {loads} |")
    return "\n".join(rows)


def spike_cell(tag, p, c, gate):
    t = load(RES / f"{tag}_{p}_dt{c}_timing.json")
    if not t or "accuracy" not in t:
        return "-", None
    w = t.get("t_warm_median_s") or t["t_first_s"]
    e = t["accuracy"]["key"]["max"]
    s = t["settings"]
    return (f"{s['rtol']:g}, {s['atol']:g}: {w:.2f}, {e:.1e}" + ("" if e <= gate else " (FAILS gate)"),
            w if e <= gate else None)


def ab_cells(p, c):
    """JAX dense and scipy runs repeated in the interleaved session (if any)."""
    d = load(BENCH / "results" / f"jaxdense_ab_{p}_dt{c}_timing.json")
    dc = "-"
    if d and "accuracy" in d:
        dc = f"{d.get('t_warm_median_s') or d['t_first_s']:.1f}"
    sc = load(RES / f"scipy_ab_{p}_dt{c}.json")
    scc = "-"
    if sc:
        e = score(RES / f"scipy_ab_{p}_dt{c}.json", REFERENCE / f"ref_{p}_dt{c}.json")["key"]["max"]
        scc = f"{sc['stats']['wall_s']:.1f} ({e:.1e})"
    return dc, scc


def compare_table(tag_a, tag_b, nominal_tag):
    rows = ["| Case | gate | JAX dense: 24 h [s], E_key | scipy sparse, matched: rtol, atol, [s], E_key | "
            "spike A at nominal 1e-3, 1e4: [s], E_key | spike A matched: rtol, atol: [s], E_key | "
            "spike B (max norm, carried h): rtol, atol: [s], E_key | dense / A | dense / B | scipy / A | scipy / B | "
            "same session: dense [s], scipy [s] (E_key) |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, c in CASES:
        ed, gate, dt = dense(p, c)
        dcell = (f"{dt[0]:.1f}{'' if dt[1] else ' (one run)'}, {ed:.1e}" if dt else "not run (over 20 min)")
        sb = scipy_best(p, c, gate)
        scell = f"{sb[0]}, {sb[1]:.1f}, {sb[2]:.1e}" if sb else "-"
        ncell, _ = spike_cell(nominal_tag, p, c, gate)
        acell, wa = spike_cell(tag_a, p, c, gate)
        bcell, wb = spike_cell(tag_b, p, c, gate)
        r = lambda x, y, f: (f.format(x / y) if (x and y) else "-")
        dab, sab = ab_cells(p, c)
        rows.append(f"| {p} {c} s | {gate:.1e} | {dcell} | {scell} | {ncell} | {acell} | {bcell} | "
                    f"{r(dt[0] if dt else None, wa, '{:.0f}')} | {r(dt[0] if dt else None, wb, '{:.0f}')} | "
                    f"{r(sb[1] if sb else None, wa, '{:.1f}')} | {r(sb[1] if sb else None, wb, '{:.1f}')} | "
                    f"{dab}, {sab} |")
    return "\n".join(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--final-tag", default="final")
    ap.add_argument("--nominal-tag", default="nominal")
    ap.add_argument("--sweep-prefix", default="sw_r")
    ap.add_argument("--rtols", default="1e-3,5e-4,2e-4,1e-4")
    a = ap.parse_args()
    print("## sweep (warm 1 run each, exploratory timings)\n")
    print(sweep_table(a.sweep_prefix, a.rtols.split(",")))
    for tag in (a.nominal_tag, a.final_tag, "best", "best_scan", "final_repeat", "best_repeat"):
        print(f"\n## runs: {tag}\n")
        print(run_table(tag))
    print("\n## comparison\n")
    print(compare_table(a.final_tag, "best", a.nominal_tag))


if __name__ == "__main__":
    main()
