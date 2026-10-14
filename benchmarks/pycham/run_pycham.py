"""Headless PyCHAM driver.

Usage:
    .venv/bin/python run_pycham.py /abs/path/to/model_var.txt [more_model_var.txt ...]

Runs each model-variables file through PyCHAM's own automated path
(gui.PyCHAM(param_const) -> autorun -> on_click81sing -> middle()) with
Qt's offscreen platform, so no window opens. Exits non-zero if PyCHAM
reports an error (PyCHAM only shows errors in GUI labels, so this script
reads those labels back and prints them).

PyCHAM must run with cwd == the PyCHAM repo root: MCM photolysis is only
switched on when photo_path == cwd + '/PyCHAM/photofiles/MCMv3.2', and it
writes rate_coeffs.py, ode_solv.py, pickle.pkl etc. into PyCHAM/PyCHAM/.
So one checkout can only run one simulation at a time.
"""
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
PYCHAM_ROOT = os.environ.get("PYCHAM_ROOT", os.path.join(HERE, "PyCHAM"))


def main(argv):
    if len(argv) < 2:
        print(__doc__)
        return 2
    mod_var_files = [os.path.abspath(p) for p in argv[1:]]
    for p in mod_var_files:
        if not os.path.isfile(p):
            print(f"model variables file not found: {p}")
            return 2

    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    os.chdir(PYCHAM_ROOT)
    sys.path.insert(0, os.path.join(PYCHAM_ROOT, "PyCHAM"))

    from PyQt6.QtWidgets import QApplication
    import gui
    import middle as _middle_mod

    # gui.act_81 does `from middle import middle` at call time, so wrapping the
    # module attribute lets us echo PyCHAM's progress and note/error messages
    # (normally only shown in GUI labels) to stdout.
    _orig_middle = _middle_mod.middle

    def _verbose_middle(self):
        last = -10.0
        for prog in _orig_middle(self):
            if isinstance(prog, str):
                print("[PyCHAM message]", prog[:3000], flush=True)
            else:
                pct = float(prog)
                if pct >= last + 10.0:
                    print(f"[PyCHAM progress] {pct:.0f}% ({time.time() - t0:.1f} s)", flush=True)
                    last = pct
            yield prog

    _middle_mod.middle = _verbose_middle

    app = QApplication(sys.argv)
    param_const = {
        "mod_var_name": mod_var_files,
        # autorun() compares this substring across successive file names to
        # decide whether the chemistry changed; any string not in the names works
        "wo_str": "__no_such_substring__",
    }
    t0 = time.time()
    ex = gui.PyCHAM(param_const)
    wall = time.time() - t0

    status = ex.l81b.text() if hasattr(ex, "l81b") else ""
    setup = ex.l80.text() if hasattr(ex, "l80") else ""
    print("---- PyCHAM setup label ----")
    print(setup)
    print("---- PyCHAM status label ----")
    print(status)
    print(f"---- wall time {wall:.1f} s ----")
    failed = ("Error" in status) or ("error" in status) or ("Simulation complete" not in status)
    app.quit()
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
