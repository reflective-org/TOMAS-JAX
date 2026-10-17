"""Markdown tables of the Julia spike from bench/results/*_timing.json.

    python julia_tables.py [tag ...]      (default tags: nom loose tight)

Table 1: every Julia run (one fresh process per row). Table 2: per case, the cheapest
setting of each solver that passes the matched-accuracy gate of BENCHMARK.md section 6,
against JAX dense and the cheapest passing scipy sparse setting.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

BENCH = Path(__file__).resolve().parents[2] / "bench"
sys.path.insert(0, str(BENCH / "scripts"))
from make_tables import CASES, SCIPY_SETTINGS, load  # noqa: E402
from compare import score  # noqa: E402

RESULTS = BENCH / "results"
REFERENCE = BENCH / "reference"
TAGS = sys.argv[1:] or ["nom", "loose", "tight"]


def jax(p, c):
    t = load(RESULTS / f"jaxdense_{p}_dt{c}_timing.json")
    if t is None or "accuracy" not in t:
        return None, None, 5e-2
    ej = t["accuracy"]["key"]["max"]
    return (t.get("t_warm_median_s") or t["t_first_s"]), ej, min(5e-2, max(2 * ej, 1e-2))


def scipy_best(p, c, gate):
    best = None
    for tag, lab in SCIPY_SETTINGS:
        f = RESULTS / f"{tag}_{p}_dt{c}.json"
        if not f.exists():
            continue
        e = score(f, REFERENCE / f"ref_{p}_dt{c}.json")["key"]["max"]
        w = load(f)["stats"]["wall_s"]
        if e <= gate and (best is None or w < best[1]):
            best = (lab, w, e)
    return best


def julia_runs(p, c):
    out = []
    for tag in TAGS:
        for f in sorted(RESULTS.glob(f"{tag}_*_{p}_dt{c}_timing.json")):
            t = json.loads(f.read_text())
            if t["problem"] != p or t["coupling_s"] != c:
                continue
            out.append((tag, t))
    return out


def fmt_t(t):
    w = t["t_warm_s"]
    if w:
        return f"{statistics.median(w):.2f} ({', '.join(f'{x:.2f}' for x in w)})"
    return f"{t['t_first_s']:.2f} (first only)"


def main():
    rows1 = ["| Case | Solver | rtol, atol | t_warm median [s] (runs) | steps (rejected) | f evals | "
             "Jacobians | LU factorisations | t_setup [s] | t_first [s] | t_first_interval [s] | "
             "peak RSS [MB] | E_key (worst species) | E_all | gate | 1-min load |",
             "|" + "---|" * 16]
    rows2 = ["| Case | gate | JAX dense [s] | scipy sparse, cheapest passing (rtol, atol): [s] | "
             "Julia, cheapest passing per solver: rtol/atol: [s] (E_key) | best Julia [s] | "
             "JAX / best Julia | scipy / best Julia |", "|" + "---|" * 8]
    for p, c in CASES:
        tj, ej, gate = jax(p, c)
        best_per = {}
        for tag, t in julia_runs(p, c):
            w = t["t_warm_s"]
            wm = statistics.median(w) if w else t["t_first_s"]
            st = t["work_per_24h"]
            ok = t["E_key"] <= gate
            loads = ", ".join(f"{r['load_before'][0]:.1f}" for r in t["runs"])
            rows1.append(
                f"| {p} {c} s | {t['solver']}{' carry-dt' if t.get('carry_dt') else ''} | {t['rtol']:.0e}, {t['atol']:.0e} | {fmt_t(t)} | "
                f"{st.get('naccept')} ({st.get('nreject')}) | {st.get('nf')} | {st.get('njacs')} | "
                f"{st.get('nw')} | {t['t_setup_s']:.1f} | {t['t_first_s']:.2f} | "
                f"{t['t_first_interval_s']:.3f} | {t['peak_rss_mb']:.0f} | "
                f"{t['E_key']:.1e} ({t['worst_species']}) | {t['E_all_hourly']:.1e} | "
                f"{'pass' if ok else 'FAIL'} | {loads} |")
            label = t["solver"] + (" carry-dt" if t.get("carry_dt") else "")
            if ok and t["solver"] != "cvode_colamd":
                b = best_per.get(label)
                if b is None or wm < b[1]:
                    best_per[label] = (f"{t['rtol']:.0e}/{t['atol']:.0e}", wm, t["E_key"])
        sb = scipy_best(p, c, gate)
        cells = "; ".join(f"{s} {r}: {w:.2f} ({e:.0e})" for s, (r, w, e) in sorted(best_per.items()))
        bj = min((v[1] for v in best_per.values()), default=None)
        rows2.append(
            f"| {p} {c} s | {gate:.1e} | {f'{tj:.1f}' if tj else 'not run'} | "
            f"{f'{sb[0]}: {sb[1]:.1f}' if sb else '-'} | {cells or 'none passes'} | "
            f"{f'{bj:.2f}' if bj else '-'} | {f'{tj / bj:.0f}' if (tj and bj) else '-'} | "
            f"{f'{sb[1] / bj:.0f}' if (sb and bj) else '-'} |")
    print("## All Julia runs\n\n" + "\n".join(rows1) + "\n\n## Matched accuracy\n\n" + "\n".join(rows2))


if __name__ == "__main__":
    main()
