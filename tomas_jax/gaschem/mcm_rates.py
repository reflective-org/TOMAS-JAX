"""MCM v3.3.1 generic/complex rate coefficients and clear-sky photolysis parameters.

Transcribed from ``constants_mcm.f90`` as distributed with KPP
(KineticPreProcessor/KPP, ``examples/mcm/constants_mcm.f90``), which is the module
the MCM KPP export ``USE``s. Definitions are those of the MCM v3.3.1 "complex rate
coefficients" and "generic rate coefficients" (Jenkin et al., 1997; Saunders et al.,
2003; Jenkin et al., 2015, http://mcm.york.ac.uk). Names match the Fortran so the
mechanism's rate expressions can reference them directly.

All functions are plain numpy (scalars or arrays). Units: TEMP [K]; M, O2, N2, H2O
[molec/cm3]; returned bimolecular coefficients in cm3/molec/s, unimolecular in 1/s.
"""
from __future__ import annotations

import numpy as np


def _troe(k0: np.ndarray, kinf: np.ndarray, fc: float | np.ndarray) -> np.ndarray:
    """MCM Troe fall-off form: k = k0*kinf*F/(k0+kinf), F = 10**(log10 Fc/(1+(log10(k0/kinf)/Nc)^2))."""
    kr = k0 / kinf
    nc = 0.75 - 1.27 * np.log10(fc)
    f = 10.0 ** (np.log10(fc) / (1.0 + (np.log10(kr) / nc) ** 2))
    return k0 * kinf * f / (k0 + kinf)


def complex_rates(TEMP, M, O2, N2, H2O) -> dict[str, np.ndarray]:  # noqa: N803 (Fortran names)
    """Every named coefficient defined by ``define_constants_mcm`` (excluding J)."""
    del N2  # not used by any MCM v3.3.1 complex coefficient; kept for a uniform signature
    T = TEMP
    K: dict[str, np.ndarray] = {}
    K["K14ISOM1"] = 3.00e7 * np.exp(-5300.0 / T)
    K["K298CH3O2"] = 3.5e-13 + 0.0 * T
    K["KAPHO2"] = 5.2e-13 * np.exp(980.0 / T)
    K["KAPNO"] = 7.5e-12 * np.exp(290.0 / T)
    K["KCH3O2"] = 1.03e-13 * np.exp(365.0 / T)
    K["KDEC"] = 1.00e6 + 0.0 * T
    K["KMT05"] = 1.44e-13 * (1.0 + M / 4.2e19)
    K["KMT06"] = 1.0 + 1.40e-21 * np.exp(2200.0 / T) * H2O
    K["KMT18"] = 9.5e-39 * O2 * np.exp(5270.0 / T) / (1.0 + 7.5e-29 * O2 * np.exp(5610.0 / T))
    K["KNO3AL"] = 1.44e-12 * np.exp(-1862.0 / T)
    K["KRO2HO2"] = 2.91e-13 * np.exp(1300.0 / T)
    K["KRO2NO"] = 2.7e-12 * np.exp(360.0 / T)
    K["KRO2NO3"] = 2.3e-12 + 0.0 * T
    K["KROPRIM"] = 2.50e-14 * np.exp(-300.0 / T)
    K["KROSEC"] = 2.50e-14 * np.exp(-300.0 / T)

    # PAN decomposition (KBPAN) and formation (KFPAN), PPN decomposition (KBPPN)
    K["KBPAN"] = _troe(1.10e-5 * M * np.exp(-10100.0 / T), 1.90e17 * np.exp(-14100.0 / T), 0.30)
    K["KBPPN"] = _troe(1.7e-3 * np.exp(-11280.0 / T) * M, 8.3e16 * np.exp(-13940.0 / T), 0.36)
    K["KFPAN"] = _troe(3.28e-28 * M * (T / 300.0) ** -6.87, 1.125e-11 * (T / 300.0) ** -1.105, 0.30)

    K["KMT01"] = _troe(1.0e-31 * M * (T / 300.0) ** -1.6, 5.0e-11 * (T / 300.0) ** -0.3, 0.85)
    K["KMT02"] = _troe(1.3e-31 * M * (T / 300.0) ** -1.5, 2.3e-11 * (T / 300.0) ** 0.24, 0.6)
    K["KMT03"] = _troe(3.6e-30 * M * (T / 300.0) ** -4.1, 1.9e-12 * (T / 300.0) ** 0.2, 0.35)
    K["KMT04"] = _troe(1.3e-3 * M * (T / 300.0) ** -3.5 * np.exp(-11000.0 / T),
                       9.7e14 * (T / 300.0) ** 0.1 * np.exp(-11080.0 / T), 0.35)
    K["KMT07"] = _troe(7.4e-31 * M * (T / 300.0) ** -2.4, 3.3e-11 * (T / 300.0) ** -0.3, 0.81)
    K["KMT08"] = _troe(3.2e-30 * M * (T / 300.0) ** -4.5, 3.0e-11 + 0.0 * T, 0.41)
    K["KMT09"] = _troe(1.4e-31 * M * (T / 300.0) ** -3.1, 4.0e-12 + 0.0 * T, 0.4)
    K["KMT10"] = _troe(4.10e-5 * M * np.exp(-10650.0 / T), 6.0e15 * np.exp(-11170.0 / T), 0.4)
    k1 = 2.40e-14 * np.exp(460.0 / T)
    k3 = 6.50e-34 * np.exp(1335.0 / T)
    k4 = 2.70e-17 * np.exp(2199.0 / T)
    K["KMT11"] = k1 + (k3 * M) / (1.0 + k3 * M / k4)
    K["KMT12"] = _troe(2.5e-31 * M * (T / 300.0) ** -2.6, 2.0e-12 + 0.0 * T, 0.53)
    K["KMT13"] = _troe(2.5e-30 * M * (T / 300.0) ** -5.5, 1.8e-11 + 0.0 * T, 0.36)
    K["KMT14"] = _troe(9.0e-5 * np.exp(-9690.0 / T) * M, 1.1e16 * np.exp(-10560.0 / T), 0.36)
    K["KMT15"] = _troe(8.6e-29 * M * (T / 300.0) ** -3.1, 9.0e-12 * (T / 300.0) ** -0.85, 0.48)
    K["KMT16"] = _troe(8.0e-27 * M * (T / 300.0) ** -3.5, 3.0e-11 * (T / 300.0) ** -1.0, 0.5)
    K["KMT17"] = _troe(5.0e-30 * M * (T / 300.0) ** -1.5, 1.0e-12 + 0.0 * T,
                       0.17 * np.exp(-51.0 / T) + np.exp(-T / 204.0))
    return K


# MCM clear-sky photolysis parameterization J = l * cos(chi)^m * exp(-n * sec(chi))
# (Saunders et al., 2003), keyed by the KPP J_* names. Values from constants_mcm.f90.
MCM_J_PARAMS: dict[str, tuple[float, float, float]] = {
    "J_O3_O1D": (6.073e-05, 1.743, 0.474),       # MCM J1
    "J_O3_O3P": (4.775e-04, 0.298, 0.080),       # J2
    "J_H2O2": (1.041e-05, 0.723, 0.279),         # J3
    "J_NO2": (1.165e-02, 0.244, 0.267),          # J4
    "J_NO3_NO": (2.485e-02, 0.168, 0.108),       # J5
    "J_NO3_NO2": (1.747e-01, 0.155, 0.125),      # J6
    "J_HONO": (2.644e-03, 0.261, 0.288),         # J7
    "J_HNO3": (9.312e-07, 1.230, 0.307),         # J8
    "J_HCHO_H": (4.642e-05, 0.762, 0.353),       # J11
    "J_HCHO_H2": (6.853e-05, 0.477, 0.323),      # J12
    "J_CH3CHO": (7.344e-06, 1.202, 0.417),       # J13
    "J_C2H5CHO": (2.879e-05, 1.067, 0.358),      # J14
    "J_C3H7CHO_HCO": (2.792e-05, 0.805, 0.338),  # J15
    "J_C3H7CHO_C2H4": (1.675e-05, 0.805, 0.338), # J16
    "J_IPRCHO": (7.914e-05, 0.764, 0.364),       # J17
    "J_MACR_HCO": (1.482e-06, 0.396, 0.298),     # J18
    "J_MACR_H": (1.482e-06, 0.396, 0.298),       # J19
    "J_C5HPALD1": (7.600e-04, 0.396, 0.298),     # J20
    "J_CH3COCH3": (7.992e-07, 1.578, 0.271),     # J21
    "J_MEK": (5.804e-06, 1.092, 0.377),          # J22
    "J_MVK_CO": (2.4246e-06, 0.395, 0.296),      # J23
    "J_MVK_C2H3": (2.424e-06, 0.395, 0.296),     # J24
    "J_GLYOX_H2": (6.845e-05, 0.130, 0.201),     # J31
    "J_GLYOX_HCHO": (1.032e-05, 0.130, 0.201),   # J32
    "J_GLYOX_HCO": (3.802e-05, 0.644, 0.312),    # J33
    "J_MGLYOX": (1.537e-04, 0.170, 0.208),       # J34
    "J_BIACET": (3.326e-04, 0.148, 0.215),       # J35
    "J_CH3OOH": (7.649e-06, 0.682, 0.279),       # J41
    "J_CH3NO3": (1.588e-06, 1.154, 0.318),       # J51
    "J_C2H5NO3": (1.907e-06, 1.244, 0.335),      # J52
    "J_NC3H7NO3": (2.485e-06, 1.196, 0.328),     # J53
    "J_IC3H7NO3": (4.095e-06, 1.111, 0.316),     # J54
    "J_TC4H9NO3": (1.135e-05, 0.974, 0.309),     # J55
    "J_NOA": (4.365e-05, 1.089, 0.323),          # J56
}


def mcm_photolysis(zenith_rad, names=None) -> dict[str, np.ndarray]:
    """MCM clear-sky J [1/s] at solar zenith angle ``zenith_rad``; zero when the sun is down."""
    cosz = np.cos(zenith_rad)
    up = cosz > 0.0
    safe = np.where(up, cosz, 1.0)
    out = {}
    for name in names if names is not None else MCM_J_PARAMS:
        l, m, n = MCM_J_PARAMS[name]
        out[name] = np.where(up, l * safe**m * np.exp(-n / safe), 0.0)
    return out
