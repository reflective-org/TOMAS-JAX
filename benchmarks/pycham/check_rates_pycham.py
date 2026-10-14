"""Evaluate PyCHAM's generated rate coefficients at a given environment (PyCHAM venv).

Usage: python -I check_rates_pycham.py PYCHAM_ROOT J.xlsx env.json out.npy

Imports the ``rate_coeffs.py`` that PyCHAM generated for the last run in PYCHAM_ROOT
(so it reflects that run's .eqn and rate_constant_file), loads J from the J.xlsx the
way PyCHAM does, and saves the rate-coefficient vector (PyCHAM reaction order).
env.json holds TEMP [K], M, N2, O2, H2O, RO2, NO, HO2, NO3 [molec/cm3], RH [0-1] and
sumt [s] (time used to pick the J column).
"""
import json
import os
import sys
import types

import numpy as np


def main(root, xlsx, env_path, out):
    os.chdir(root)
    sys.path.insert(0, os.path.join(root, "PyCHAM"))
    import rate_coeffs
    from J_value_file_open import J_value_file_open

    env = json.load(open(env_path))
    self = types.SimpleNamespace(inname=os.path.abspath(env_path))
    J_value_file_open(os.path.abspath(xlsx), self)
    self.light_stat = (np.ones(len(self.light_time)) * 3).astype(int)
    self.light_stat_now = 3
    self.tf_range = 0
    self.tf = np.ones(1)
    self.tf_UVC = np.ones(1)
    rv, erf, msg = rate_coeffs.evaluate_rates(
        env["RO2"], env["H2O"], env["RH"], env["TEMP"], 0.0, env["M"], env["N2"], env["O2"],
        62, env["NO"], env["HO2"], env["NO3"], env["sumt"], self)
    if erf:
        raise SystemExit(f"PyCHAM rate evaluation failed: {msg}")
    np.save(out, np.asarray(rv, dtype=float))
    return 0


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:5]))
