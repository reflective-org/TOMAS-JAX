"""Compare MCM v3.3.1 rate coefficients (as coded in tomas_jax.gaschem.mcm_rates and the
inline expressions of limonene_mcm.eqn) with the JPL 19-5 forms used by SANDBOX
(stratchem-jax/reactions.py) for the reactions both mechanisms share.

Planning prototype for docs/pro/mechanism/mcm_coverage.md. Prints k_MCM, k_JPL and the
ratio at two conditions: near-surface air and the lower stratosphere. Pure numpy.

The JPL expressions are transcribed from SANDBOX stratchem-jax/reactions.py and
mechanism.py (falloff with Fc = 0.6, 298 K reference); they are not imported so this
script does not depend on the SANDBOX package.

Run (from the worktree root):
    python prototypes/mechanism/mcm_vs_jpl_rates.py
"""
from __future__ import annotations

import importlib.util
import math
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
KB = 1.380649e-23


def _mcm_rates():
    path = ROOT / "tomas_jax" / "gaschem" / "mcm_rates.py"
    spec = importlib.util.spec_from_file_location("mcm_rates_ro", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def jpl_falloff(T, M, k0_298, n0, kinf_298, ninf, fc=0.6):
    """SANDBOX mechanism.falloff: JPL 19-5 termolecular form, returns (k_f, kinf)."""
    k0 = k0_298 * (T / 298.0) ** n0
    kinf = kinf_298 * (T / 298.0) ** ninf
    ratio = k0 * M / kinf
    kf = (k0 * M / (1.0 + ratio)) * fc ** (1.0 / (1.0 + math.log10(ratio) ** 2))
    return kf, kinf


def conditions():
    out = {}
    # Near-surface: 298 K, 1013.25 hPa, 50 % RH (Buck e_sat as in gaschem.config).
    T, P = 298.15, 101325.0
    M = P / (KB * T) * 1e-6
    tc = T - 273.15
    esat = 611.21 * math.exp((18.678 - tc / 234.5) * tc / (tc + 257.14))
    out["surface 298 K 1013 hPa RH50"] = (T, M, 0.5 * esat / P * M)
    # Lower stratosphere: 220 K, 50 hPa, 5 ppmv H2O.
    T, P = 220.0, 5000.0
    M = P / (KB * T) * 1e-6
    out["strat 220 K 50 hPa 5 ppmv H2O"] = (T, M, 5e-6 * M)
    return out


def main() -> None:
    mr = _mcm_rates()
    exp = math.exp
    for label, (T, M, H2O) in conditions().items():
        O2, N2 = 0.2095 * M, 0.7809 * M
        K = {k: float(v) for k, v in mr.complex_rates(np.array(T), np.array(M), np.array(O2),
                                                       np.array(N2), np.array(H2O)).items()}
        f = lambda *a: jpl_falloff(T, M, *a)  # noqa: E731
        kf_no2, kinf_no2 = f(3.4e-31, -1.6, 2.3e-11, -0.2)
        kf_hno3, kinf_hno3 = f(3.9e-31, -7.2, 1.5e-13, -4.8)
        rows = [
            # name, k_MCM, k_JPL, unit
            ("OH + SO2 (+M)", K["KMT12"], f(2.9e-31, -4.1, 1.7e-12, 0.2)[0], "cm3/s"),
            ("SO3 + H2O, first-order loss", 1.2e-15 * H2O, 8.5e-41 * exp(6540 / T) * H2O ** 2, "1/s"),
            ("OH + NO2 (+M) -> HNO3", K["KMT08"], f(1.8e-30, -3.0, 2.8e-11, 0.0)[0], "cm3/s"),
            ("HO2 + NO2 (+M) -> HNO4", K["KMT09"], f(1.9e-31, -3.4, 4.0e-12, -0.3)[0], "cm3/s"),
            ("NO2 + NO3 (+M) -> N2O5", K["KMT03"], f(2.4e-30, -3.0, 1.6e-12, 0.1)[0], "cm3/s"),
            ("OH + NO (+M) -> HONO", K["KMT07"], f(7.1e-31, -2.6, 3.6e-11, -0.1)[0], "cm3/s"),
            ("O + NO (+M) -> NO2", K["KMT01"], f(9.1e-32, -1.5, 3.0e-11, 0.0)[0], "cm3/s"),
            ("O + NO2 (+M) -> NO3", K["KMT02"], kf_no2, "cm3/s"),
            ("O + NO2 -> NO + O2", 5.5e-12 * exp(188 / T),
             5.3e-12 * exp(200 / T) * (1 - kf_no2 / kinf_no2), "cm3/s"),
            ("OH + HNO3 -> NO3", K["KMT11"],
             kf_hno3 + 3.7e-14 * exp(240 / T) * (1 - kf_hno3 / kinf_hno3), "cm3/s"),
            ("NO + O3", 1.4e-12 * exp(-1310 / T), 3.0e-12 * exp(-1500 / T), "cm3/s"),
            ("NO2 + O3", 1.4e-13 * exp(-2470 / T), 1.2e-13 * exp(-2450 / T), "cm3/s"),
            ("HO2 + NO", 3.45e-12 * exp(270 / T), 3.44e-12 * exp(260 / T), "cm3/s"),
            ("HO2 + O3", 2.03e-16 * (T / 300) ** 4.57 * exp(693 / T), 1.0e-14 * exp(-490 / T), "cm3/s"),
            ("OH + O3", 1.70e-12 * exp(-940 / T), 1.7e-12 * exp(-940 / T), "cm3/s"),
            ("OH + HO2", 4.8e-11 * exp(250 / T), 4.8e-11 * exp(250 / T), "cm3/s"),
            ("OH + HONO", 2.5e-12 * exp(260 / T), 3.0e-12 * exp(250 / T), "cm3/s"),
            ("NO + NO3", 1.8e-11 * exp(110 / T), 1.7e-11 * exp(125 / T), "cm3/s"),
            ("OH + HNO4", 3.2e-13 * exp(690 / T), 4.5e-13 * exp(610 / T), "cm3/s"),
            ("HO2 + HO2 (incl. M, H2O)",
             2.20e-13 * K["KMT06"] * exp(600 / T) + 1.90e-33 * M * K["KMT06"] * exp(980 / T),
             (3.0e-13 * exp(460 / T) + 2.1e-33 * M * exp(920 / T))
             * (1 + 1.4e-21 * H2O * exp(2200 / T)), "cm3/s"),
            ("O + O2 + M -> O3, first-order in O",
             5.6e-34 * N2 * (T / 300) ** -2.6 * O2 + 6.0e-34 * O2 * (T / 300) ** -2.6 * O2,
             6.1e-34 * (T / 298) ** -2.4 * M * O2, "1/s"),
            ("O + O3", 8.0e-12 * exp(-2060 / T), 8.0e-12 * exp(-2060 / T), "cm3/s"),
            ("O1D + H2O", 2.14e-10, 1.63e-10 * exp(60 / T), "cm3/s"),
            ("O1D quenching (O2 + N2), first-order",
             3.2e-11 * exp(67 / T) * O2 + 2.0e-11 * exp(130 / T) * N2,
             3.3e-11 * exp(55 / T) * O2 + 2.15e-11 * exp(110 / T) * 0.79 * M, "1/s"),
            ("CH4 + OH", 1.85e-12 * exp(-1690 / T), 2.45e-12 * exp(-1775 / T), "cm3/s"),
            ("Cl + CH4", 6.6e-12 * exp(-1240 / T), 7.1e-12 * exp(-1270 / T), "cm3/s"),
        ]
        print(f"\n{label}: T = {T:.2f} K, M = {M:.3e}, H2O = {H2O:.3e} molec/cm3")
        print(f"{'reaction':38s} {'k_MCM':>11s} {'k_JPL19-5':>11s} {'MCM/JPL':>8s}  unit")
        for name, km, kj, unit in rows:
            print(f"{name:38s} {km:11.3e} {kj:11.3e} {km / kj:8.3f}  {unit}")


if __name__ == "__main__":
    main()
