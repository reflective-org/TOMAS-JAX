"""Fuller diffusion volumes PyCHAM computes from SMILES (run in the PyCHAM venv).

Usage: python -I pycham_diffvol.py PYCHAM_ROOT pycham_mw.json out.json NAME [NAME ...]

Uses PyCHAM's own ``diff_vol_est.py`` (atomic increments + ring correction) on the
SMILES that ``pycham_mw.py`` recorded, so the gas diffusivities of the two models can be
compared outside a run (PyCHAM does not write them to its output).
"""
import json
import os
import sys

import openbabel.pybel as pybel


def main(root, mw_json, out, names):
    sys.path.insert(0, os.path.join(root, "PyCHAM"))
    import diff_vol_est
    smiles = json.load(open(mw_json))
    objs = [pybel.readstring("smi", smiles[n]["smiles"]) for n in names]
    vols = diff_vol_est.diff_vol_est(objs)
    json.dump({n: float(v) for n, v in zip(names, vols)}, open(out, "w"))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:]))
