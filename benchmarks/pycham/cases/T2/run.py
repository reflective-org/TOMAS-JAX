"""Case T2: see cases/T1/run.py (--case T2)."""
import sys

from ..T1.run import main

if __name__ == "__main__":
    raise SystemExit(main(["--case", "T2", *sys.argv[1:]]))
