"""Gas-to-particle transfer rates of the two models, evaluated outside a run.

Both models use a Fuchs-Sutugin-corrected continuum flux, CS = 4 pi sum_k r_k N_k D f(Kn_k),
with alpha = 1, but with different ingredients:

* TOMAS-JAX (vbs-jax ``organic_condensation_sink``): Fuller D with ONE diffusion volume
  for every organic (``SV_ORG`` = 120) and the class-mean molar mass, air volume 20.1 and
  M_air 28.9; mfp = 2 D / c; Kn = 2 mfp / Dp; f = 0.75 (1 + Kn) / (Kn (1 + Kn) +
  0.283 Kn + 0.75) (Zaveri et al. 2014 Eq. 14); two-film particle side (D_b = 1e-10,
  negligible here).
* PyCHAM (``partit_var_prep.py``, ``partit_var.py``): Fuller D with each species' own
  diffusion volume from its SMILES (``diff_vol_est``) and own molar mass, air volume
  19.7, M_air 28.966; mfp = 3 D / c; Kn = mfp / r; f = 1 / (1 + Kn (1.33 + 0.71/Kn) /
  (1 + 1/Kn)) (Jacobson 2005 Eq. 16.19).

``transfer_ratios`` evaluates both on the SAME size distribution so only the formula and
property differences remain.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import scipy.constants as si
from vbs_jax.config import SV_ORG
from vbs_jax.gas_properties import calc_gas_diffusivity, calc_mean_free_path

from tomas_jax.box.driver import _class_map

from ...config import PYCHAM_PYTHON
from ...pycham_inputs import ensure_pycham_root

DIFFVOL_TIMEOUT_S = 300.0


def pycham_diffusion_volumes(mw_json: Path, out_json: Path, names: list[str]) -> dict[str, float]:
    root = ensure_pycham_root("S2")
    proc = subprocess.run([str(PYCHAM_PYTHON), "-I", str(Path(__file__).with_name("pycham_diffvol.py")),
                           str(root), str(mw_json), str(out_json), *names],
                          capture_output=True, text=True, timeout=DIFFVOL_TIMEOUT_S)
    if proc.returncode != 0:
        raise RuntimeError(f"pycham_diffvol.py failed ({proc.returncode}): {proc.stderr[-2000:]}")
    return json.loads(Path(out_json).read_text())


def cs_ours(r_m, n_m3, mw_class, temp, pres):
    """[species] condensation sink [1/s], vbs-jax formulation (gas side, alpha = 1)."""
    d = np.asarray(calc_gas_diffusivity(temp, pres, mw_class, SV_ORG))
    mfp = np.asarray(calc_mean_free_path(temp, pres, mw_class, SV_ORG))
    kn = 2.0 * mfp[None, :] / (2.0 * r_m[:, None])
    fc = 0.75 * (1.0 + kn) / (kn * (1.0 + kn) + 0.283 * kn + 0.75)
    return 4.0 * np.pi * np.sum(r_m[:, None] * n_m3[:, None] * d[None, :] * fc, axis=0), d


def cs_pycham(r_m, n_m3, mw, diff_vol, temp, pres):
    """[species] condensation sink [1/s], PyCHAM formulation (alpha = 1)."""
    ma = 28.966e-3
    d = (1.013e-2 * temp ** 1.75 * np.sqrt((mw + ma * 1e3) / (mw * ma * 1e3))
         / (pres * (diff_vol ** (1 / 3) + 19.7 ** (1 / 3)) ** 2))
    therm = np.sqrt(8.0 * si.k * temp / (np.pi * (mw / si.N_A) * 1e-3))
    mfp = 3.0 * d / therm
    kn = mfp[None, :] / r_m[:, None]
    inv = 1.0 / kn
    corr = 1.0 / (1.0 + ((1.33 + 0.71 * inv) / (1.0 + inv)) * kn)
    return 4.0 * np.pi * np.sum(r_m[:, None] * n_m3[:, None] * d[None, :] * corr, axis=0), d


def transfer_ratios(r_m, n_cm3, mw_pycham: dict, diff_vol: dict, temp, pres) -> dict:
    """Per condensable species: CS_PyCHAM / CS_TOMAS-JAX and D_PyCHAM / D_TOMAS-JAX."""
    cm = _class_map(None)
    n_m3 = np.asarray(n_cm3) * 1e6
    mw_cls = cm.class_mw[cm.class_idx]
    cs_o, d_o = cs_ours(np.asarray(r_m), n_m3, mw_cls, temp, pres)
    mw = np.array([mw_pycham[s] for s in cm.species])
    dv = np.array([diff_vol[s] for s in cm.species])
    cs_p, d_p = cs_pycham(np.asarray(r_m), n_m3, mw, dv, temp, pres)
    return {"species": list(cm.species), "cs_ours": cs_o, "cs_pycham": cs_p,
            "cs_ratio": cs_p / cs_o, "d_ratio": d_p / d_o, "diff_vol_pycham": dv,
            "d_ours": d_o, "d_pycham": d_p}
