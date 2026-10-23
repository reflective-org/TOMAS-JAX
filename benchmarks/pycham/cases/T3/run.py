"""Case T3: see cases/T1/run.py (--case T3)."""
import sys

from ..T1.run import main

if __name__ == "__main__":
    raise SystemExit(main(["--case", "T3", *sys.argv[1:]]))
