"""Print the markdown tables of BENCHMARK.md from the result and reference files."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import PROBLEMS, REFERENCE, RESULTS  # noqa: E402
from compare import score  # noqa: E402

CASES = [("lim726_L2", 900), ("lim726_L3", 900), ("syn1353_L3", 900), ("syn4488_L3", 900),
         ("syn5742_L3", 900), ("lim726_L2", 60), ("lim726_L3", 60), ("syn1353_L3", 60),
         ("syn4488_L3", 60), ("syn5742_L3", 60)]


def load(p: Path):
    return json.loads(p.read_text()) if p.exists() else None


def load_str(u: str) -> str:
    # "17:28  up 8 days, ..., load averages: 6.87 3.98 4.18" -> "6.87"
    return u.split("load averages:")[1].split()[0] if "load averages:" in u else "?"


def jax_table():
    rows = ["| Case | n | t_setup [s] | t_first [s] | t_first_interval [s] | t_warm median [s] (runs) | "
            "steps | ms/step | peak RSS [MB] | E_key (worst) | E_all | 1-min load |",
            "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for p, c in CASES:
        for tag in ("jaxdense", "jaxdense_repeat"):
            t = load(RESULTS / f"{tag}_{p}_dt{c}_timing.json")
            if t is None:
                continue
            name = f"{p} {c} s" + (" (repeat)" if tag.endswith("repeat") else "")
            loads = ", ".join(load_str(r["uptime_before"]) for r in t.get("runs", []))
            if t.get("status") != "complete":
                q = t["partial"]
                rows.append(f"| {name} | {t['n_species']} | {t['t_setup_s']:.1f} | stopped at the "
                            f"{t['cap_s']:.0f} s cap after {q['intervals_done']} of "
                            f"{q['n_intervals']} intervals | {q['t_first_interval_s']:.1f} | "
                            f"about {q['extrapolated_24h_s'] / 60:.0f} min extrapolated | "
                            f"{q['steps']} (partial) | {1e3 * q['wall_s'] / q['steps']:.0f} | "
                            f"{t['peak_rss_mb']:.0f} | - | - | {load_str(t['uptime_before'])} |")
                continue
            warm = [r["wall_s"] for r in t["runs"] if r["kind"] == "warm" and "wall_s" in r]
            acc = t.get("accuracy", {})
            ek = acc.get("key", {})
            rows.append(
                f"| {name} | {t['n_species']} | {t['t_setup_s']:.1f} | {t['t_first_s']:.1f} | "
                f"{t['t_first_interval_s']:.2f} | "
                + (f"{np.median(warm):.1f} ({', '.join(f'{w:.1f}' for w in warm)})" if warm else "-")
                + f" | {t['accepted_steps']} | "
                + (f"{t['ms_per_step_warm_median']:.1f}" if warm else
                   f"{1e3 * t['t_first_s'] / t['accepted_steps']:.1f} (first run)")
                + f" | {t['peak_rss_mb']:.0f} | "
                + (f"{ek['max']:.1e} ({ek['worst_species']})" if ek else "-")
                + f" | {acc.get('all_species_hourly_max', float('nan')):.1e} | {loads} |")
    return "\n".join(rows)


def scipy_table():
    rows = ["| Case | wall pass a [s] | wall pass b [s] | steps | LU factorisations | "
            "RHS evaluations | peak RSS [MB] | E_key (worst) | E_all | 1-min load a, b |",
            "|---|---|---|---|---|---|---|---|---|---|"]
    for p, c in CASES:
        a = load(RESULTS / f"scipysparse_a_{p}_dt{c}.json")
        b = load(RESULTS / f"scipysparse_b_{p}_dt{c}.json")
        if a is None:
            continue
        s = score(RESULTS / f"scipysparse_a_{p}_dt{c}.json", REFERENCE / f"ref_{p}_dt{c}.json")
        st = a["stats"]
        rows.append(f"| {p} {c} s | {st['wall_s']:.1f} | "
                    + (f"{b['stats']['wall_s']:.1f}" if b else "-")
                    + f" | {st['steps']} | {st['nlu']} | {st['nfev']} | {st['peak_rss_mb']:.0f} | "
                    f"{s['key']['max']:.1e} ({s['key']['worst_species']}) | "
                    f"{s['all_species_hourly_max']:.1e} | {st['load_before'][0]:.1f}"
                    + (f", {b['stats']['load_before'][0]:.1f}" if b else "") + " |")
    return "\n".join(rows)


SCIPY_SETTINGS = [("scipysparse_a", "1e-3, 1e4"), ("scipysparse_rtol1e-4", "1e-4, 1e4"),
                  ("scipysparse_rtol1e-5", "1e-5, 1e4"),
                  ("scipysparse_rtol1e-3_atol1e2", "1e-3, 1e2"),
                  ("scipysparse_rtol1e-4_atol1e2", "1e-4, 1e2"),
                  ("scipysparse_rtol1e-5_atol1e2", "1e-5, 1e2"),
                  ("scipysparse_rtol1e-6_atol1e2", "1e-6, 1e2")]


def best_table():
    """Cheapest scipy setting that passes the matched-accuracy gate, against JAX dense."""
    rows = ["| Case | JAX dense: 24 h [s], E_key | scipy sparse, nominal 1e-3/1e4: [s], E_key | "
            "scipy sparse, cheapest setting passing the gate: rtol/atol, [s], E_key | ratio JAX/scipy at matched accuracy |",
            "|---|---|---|---|---|"]
    for p, c in CASES:
        t = load(RESULTS / f"jaxdense_{p}_dt{c}_timing.json")
        if t is not None and "accuracy" in t:
            ej = t["accuracy"]["key"]["max"]
            gate = min(5e-2, max(2 * ej, 1e-2))
            tj = t.get("t_warm_median_s") or t["t_first_s"]
            jcell = f"{tj:.1f}{'' if t.get('t_warm_median_s') else ' (one run)'}, {ej:.1e}"
        else:
            gate, tj, jcell = 5e-2, None, "not run (over 20 min)"
        nominal, best = "-", None
        for tag, lab in SCIPY_SETTINGS:
            f = RESULTS / f"{tag}_{p}_dt{c}.json"
            if not f.exists():
                continue
            e = score(f, REFERENCE / f"ref_{p}_dt{c}.json")["key"]["max"]
            w = load(f)["stats"]["wall_s"]
            if tag == "scipysparse_a":
                nominal = f"{w:.1f}, {e:.1e}"
            if e <= gate and (best is None or w < best[1]):
                best = (lab, w, e)
        bcell = f"{best[0]}, {best[1]:.1f}, {best[2]:.1e}" if best else "none tried passes"
        ratio = f"{tj / best[1]:.0f}" if (best and tj) else "-"
        rows.append(f"| {p} {c} s | {jcell} | {nominal} | {bcell} | {ratio} |")
    return "\n".join(rows)


def matched_table():
    head = " | ".join(f"rtol, atol = {lab}: wall [s], E_key" for _, lab in SCIPY_SETTINGS)
    rows = [f"| Case | JAX dense E_key | gate | {head} | JAX dense 24 h [s] |",
            "|---|---|---|" + "---|" * len(SCIPY_SETTINGS) + "---|"]
    for p, c in CASES:
        t = load(RESULTS / f"jaxdense_{p}_dt{c}_timing.json")
        if t is not None and "accuracy" in t:
            ej = t["accuracy"]["key"]["max"]
            gate = min(5e-2, max(2 * ej, 1e-2))
            jcell, tj = f"{ej:.1e}", f"{t.get('t_warm_median_s') or t['t_first_s']:.1f}"
        else:
            gate, jcell, tj = 5e-2, "not run", "-"
        cells = []
        for tag, _ in SCIPY_SETTINGS:
            f = RESULTS / f"{tag}_{p}_dt{c}.json"
            if not f.exists():
                cells.append("-")
                continue
            e = score(f, REFERENCE / f"ref_{p}_dt{c}.json")["key"]["max"]
            ok = "**pass**" if e <= gate else "fail"
            cells.append(f"{load(f)['stats']['wall_s']:.1f}, {e:.1e} {ok}")
        rows.append(f"| {p} {c} s | {jcell} | {gate:.1e} | " + " | ".join(cells) + f" | {tj} |")
    return "\n".join(rows)


def ref_cost():
    parts = []
    for p, c in CASES:
        r = load(REFERENCE / f"ref_{p}_dt{c}.json")
        if r:
            st = r["stats"]
            parts.append(f"{p} {c} s: {st['wall_s']:.0f} s, {st['steps']} steps, "
                         f"{st['peak_rss_mb']:.0f} MB")
    return "; ".join(parts) + "."


def file_sizes():
    man = load(PROBLEMS / "MANIFEST.json")
    rows = [f"| `{f}` | {d['bytes'] / 1e6:.2f} MB |" for f, d in man["files"].items()]
    total = sum(d["bytes"] for d in man["files"].values()) / 1e6
    ref_total = sum(p.stat().st_size for p in REFERENCE.glob("ref_*")) / 1e6
    return "\n".join(rows), total, ref_total


if __name__ == "__main__":
    rows, total, ref_total = file_sizes()
    print("## files\n" + rows + f"\nproblems total {total:.1f} MB; ref_* total {ref_total:.1f} MB\n")
    print("## jax\n" + jax_table() + "\n")
    print("## scipy\n" + scipy_table() + "\n")
    print("## matched\n" + matched_table() + "\n")
    print("## best\n" + best_table() + "\n")
    print("## ref cost\n" + ref_cost())
