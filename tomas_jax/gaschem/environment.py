"""Air number densities for rate-constant evaluation (MCM/KPP environment variables)."""
from __future__ import annotations

import numpy as np

from .config import BUCK_A, BUCK_B, BUCK_C, BUCK_D, K_BOLTZMANN, N2_FRACTION, O2_FRACTION


def air_environment(temp: float, pres: float, rh: float) -> dict[str, float]:
    """TEMP [K], M, O2, N2, H2O [molec/cm3] from temperature [K], pressure [Pa], RH [0-1]."""
    m = pres / (K_BOLTZMANN * temp) * 1e-6
    tc = temp - 273.15
    e_sat = BUCK_A * np.exp((BUCK_B - tc / BUCK_D) * tc / (tc + BUCK_C))
    h2o = rh * e_sat / (K_BOLTZMANN * temp) * 1e-6
    return {"TEMP": float(temp), "M": float(m), "O2": O2_FRACTION * m,
            "N2": N2_FRACTION * m, "H2O": float(h2o)}
