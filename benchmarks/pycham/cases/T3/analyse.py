"""Case T3: see cases/T1/analyse.py (--case T3)."""
import sys

from ..T1.analyse import main

if __name__ == "__main__":
    raise SystemExit(main(["--case", "T3", *sys.argv[1:]]))
