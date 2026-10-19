"""Regenerate every TOMAS-JAX vs PyCHAM benchmark case and figure.

    cd tomas-jax
    .venv/bin/python -m benchmarks.pycham.run_all                    # everything (~35 min)
    .venv/bin/python -m benchmarks.pycham.run_all --cases S1,S2      # some cases
    .venv/bin/python -m benchmarks.pycham.run_all --skip-pycham      # reuse raw PyCHAM output
    .venv/bin/python -m benchmarks.pycham.run_all --analyse-only     # metrics + figures only

For each case, in order: ``cases/<case>/run.py`` (PyCHAM inputs, PyCHAM runs, TOMAS-JAX
runs, built-in checks, small results) and then ``cases/<case>/analyse.py`` (metrics,
summary.json, figures copied to docs/aaar/figures/pycham/). Every step runs in its own
process group with a wall-clock limit (config.RUN_ALL_TIMEOUTS_S); on timeout the whole
group (including PyCHAM) is killed. Cases run one after the other (each has its own
PyCHAM checkout, but parallel runs distort the reported runtimes). Logs go to
``$PYCHAM_BENCH/logs/run_all/``. The exit code is non-zero if any step failed; the other
cases still run. After each case, local paths in its text results are replaced by
placeholders (``$PYCHAM_BENCH``, ``<tomas-jax>``, ``~``), since the results are committed.
"""
from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

from .config import (CASES_DIR, DOC_FIG_DIR, PYCHAM_BENCH, RESULT_SIZE_LIMIT_MB,
                     RUN_ALL_CASES, RUN_ALL_TIMEOUTS_S, TOMAS_JAX_ROOT)

LOG_DIR = PYCHAM_BENCH / "logs" / "run_all"


def run_step(case: str, step: str, extra: list[str], timeout_s: float) -> dict:
    """``python -m benchmarks.pycham.cases.<case>.<step>`` in its own process group."""
    log = LOG_DIR / f"{case}_{step}.log"
    cmd = [sys.executable, "-m", f"benchmarks.pycham.cases.{case}.{step}", *extra]
    t0 = time.monotonic()
    with open(log, "w") as fh:
        proc = subprocess.Popen(cmd, cwd=TOMAS_JAX_ROOT, stdout=fh, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            rc = proc.wait(timeout=timeout_s)
            status = "ok" if rc == 0 else f"failed (exit {rc})"
        except subprocess.TimeoutExpired:
            os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=60)
            rc, status = None, f"timeout after {timeout_s:.0f} s"
    wall = time.monotonic() - t0
    print(f"  {case}.{step}: {status} in {wall:.0f} s (log {log})", flush=True)
    if rc != 0:
        tail = log.read_text()[-2500:]
        print("  ---- last lines ----\n" + tail + "\n  --------------------", flush=True)
    return {"case": case, "step": step, "status": status, "wall_s": round(wall, 1),
            "log": str(log), "cmd": " ".join(cmd[1:])}


def dir_size_mb(path: Path) -> float:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) / 1e6


# Local paths in the committed text results -> placeholders (the repository is public).
# PyCHAM needs absolute paths, so model_var copies, run_info.json and summary.json hold them.
PATH_PLACEHOLDERS = ((str(PYCHAM_BENCH), "$PYCHAM_BENCH"), (str(TOMAS_JAX_ROOT), "<tomas-jax>"),
                     (str(Path.home()), "~"))
TEXT_RESULTS = (".json", ".txt", ".csv")


def redact_local_paths(results_dir: Path) -> list[str]:
    """Replace local paths in the case's text results; return the files changed."""
    changed = []
    for f in sorted(results_dir.rglob("*")):
        if f.suffix not in TEXT_RESULTS or not f.is_file():
            continue
        text = new = f.read_text()
        for path, placeholder in PATH_PLACEHOLDERS:
            new = new.replace(path, placeholder)
        if new != text:
            f.write_text(new)
            changed.append(f.name)
    return changed


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--cases", default=",".join(RUN_ALL_CASES),
                    help=f"comma list (default {','.join(RUN_ALL_CASES)})")
    ap.add_argument("--skip-pycham", action="store_true",
                    help="pass --skip-pycham to every run.py (reuse raw PyCHAM output)")
    ap.add_argument("--analyse-only", action="store_true", help="only the analyse steps")
    args = ap.parse_args(argv)
    cases = [c for c in args.cases.split(",") if c]
    unknown = [c for c in cases if c not in RUN_ALL_CASES]
    if unknown:
        ap.error(f"unknown case(s) {unknown}; known: {list(RUN_ALL_CASES)}")
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    t0 = time.monotonic()
    steps = []
    for case in cases:
        print(f"== {case}", flush=True)
        if not args.analyse_only:
            res = run_step(case, "run", ["--skip-pycham"] if args.skip_pycham else [],
                           RUN_ALL_TIMEOUTS_S[case]["run"])
            steps.append(res)
            if res["status"] != "ok":
                print(f"  {case}: run failed, analyse skipped", flush=True)
                continue
        steps.append(run_step(case, "analyse", [], RUN_ALL_TIMEOUTS_S[case]["analyse"]))
        redacted = redact_local_paths(CASES_DIR / case / "results")
        if redacted:
            print(f"  {case}: local paths replaced in {len(redacted)} result files", flush=True)
    sizes = {c: round(dir_size_mb(CASES_DIR / c / "results"), 2) for c in cases
             if (CASES_DIR / c / "results").is_dir()}
    big = {c: s for c, s in sizes.items() if s > RESULT_SIZE_LIMIT_MB}
    n_fig = len(list(DOC_FIG_DIR.glob("*.png")))
    report = {"started": started, "wall_s": round(time.monotonic() - t0, 1), "steps": steps,
              "results_mb": sizes, "doc_figures": n_fig}
    name = "run_all_analyse_only.json" if args.analyse_only else "run_all.json"
    (LOG_DIR / name).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"wall_s": report["wall_s"], "results_mb": sizes, "doc_figures": n_fig,
                      "failed": [f"{s['case']}.{s['step']}" for s in steps if s["status"] != "ok"]},
                     indent=1))
    if big:
        print(f"WARNING: results above {RESULT_SIZE_LIMIT_MB} MB: {big}")
    return 0 if all(s["status"] == "ok" for s in steps) else 1


if __name__ == "__main__":
    sys.exit(main())
