"""Case T2: see cases/T1/analyse.py (--case T2)."""
import sys

from ..T1.analyse import main

if __name__ == "__main__":
    raise SystemExit(main(["--case", "T2", *sys.argv[1:]]))
