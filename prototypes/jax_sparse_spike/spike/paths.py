"""Where the shared benchmark lives (problem files, references, bench_common.py).

Order: $TOMAS_GAS_BENCH, then ../bench next to this spike (the _work/julia layout), then
../julia_bench (the layout committed in TOMAS-JAX prototypes/)."""
from __future__ import annotations

import os
import sys
from pathlib import Path

SPIKE_ROOT = Path(__file__).resolve().parents[1]


def bench_dir() -> Path:
    cands = [Path(os.environ["TOMAS_GAS_BENCH"])] if os.environ.get("TOMAS_GAS_BENCH") else []
    cands += [SPIKE_ROOT.parent / "bench", SPIKE_ROOT.parent / "julia_bench"]
    for c in cands:
        has_problems = any((c / "problems").glob("problem_*.json"))
        if (c / "scripts" / "bench_common.py").exists() and has_problems:
            return c
    raise FileNotFoundError("benchmark not found; set TOMAS_GAS_BENCH to the bench folder "
                            f"(tried {', '.join(map(str, cands))})")


BENCH = bench_dir()
if str(BENCH / "scripts") not in sys.path:
    sys.path.insert(0, str(BENCH / "scripts"))
