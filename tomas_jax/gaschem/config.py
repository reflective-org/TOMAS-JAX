"""Constants and solver defaults for the gas-phase chemistry package.

Single source of truth for every number ``tomas_jax.gaschem`` uses that is not
part of a mechanism file. ``gaschem`` deliberately does not import TOMAS
(``tomas_jax.core``) so it can be split out later; TOMAS-side constants stay in
``tomas_jax/core/config.py``.
"""
from pathlib import Path

# --- Physical constants --------------------------------------------------------
AVOGADRO = 6.02214076e23        # [1/mol]
K_BOLTZMANN = 1.380649e-23      # [J/K]
O2_FRACTION = 0.2095            # O2 mole fraction in dry air (MCM/KPP convention)
N2_FRACTION = 0.7809            # N2 mole fraction in dry air

# Saturation vapour pressure of water, Buck (1981) — same form and constants as
# tomas_jax/physics/so2_chemistry.py: e_sat = A*exp((B - Tc/D)*Tc/(Tc + C)) [Pa]
BUCK_A = 611.21                 # [Pa]
BUCK_B = 18.678
BUCK_C = 257.14                 # [degC]
BUCK_D = 234.5                  # [degC]

# Standard atomic weights [g/mol] (IUPAC 2021 conventional values), used to get
# molecular weights from the atom compositions declared in the .eqn file.
ATOMIC_WEIGHTS = {
    "C": 12.011,
    "H": 1.008,
    "N": 14.007,
    "O": 15.999,
    "S": 32.06,
    "Cl": 35.45,
    "Br": 79.904,
}

# --- Mechanism files ------------------------------------------------------------
MECHANISM_DIR = Path(__file__).parent / "mechanisms"
# MCM v3.3.1 limonene subset (SCENTS 04272022) supplied by S. Jathar, kept
# byte-identical; tutorial-specific edits are applied in code (see ASSUMPTIONS.md).
LIMONENE_EQN = MECHANISM_DIR / "limonene_mcm.eqn"
# MCM v3.3.1 aromatic subsets exported from the MCM website (KPP, inorganic + generic
# rates), kept byte-identical (provenance and sha256 in mechanisms/<name>_mcm.provenance.md).
# Each is merged with the limonene subset in code (gaschem.merge) for the runs that use it
# (box.coupling.aromatics_used); the Urban level's VOC selector (ASSUMPTIONS A27). Keyed by the
# parent VOC's MCM name; the label prefix keeps the added reactions' labels unique.
AROMATIC_EQN = {
    "BENZENE": MECHANISM_DIR / "benzene_mcm.eqn",
    "TOLUENE": MECHANISM_DIR / "toluene_mcm.eqn",
    "MXYL": MECHANISM_DIR / "mxyl_mcm.eqn",
}
AROMATIC_LABEL_PREFIX = {"BENZENE": "B", "TOLUENE": "T", "MXYL": "X"}
TOLUENE_EQN = AROMATIC_EQN["TOLUENE"]
# Per-species c* (298.15 K) of the limonene mechanism supplied by S. Jathar, kept
# byte-identical (provenance and sha256 in mechanisms/*.provenance.md; ASSUMPTIONS A8).
# Default `cstar_table` of the Level 2 and 3 presets.
DEFAULT_CSTAR_TABLE = "mcm_limonene_cstar_298K.csv"
# Provenance label reported in ClassMap.source for known tables (file name -> label).
CSTAR_TABLE_SOURCES = {
    DEFAULT_CSTAR_TABLE: ("Shantanu Jathar (2026-10-16) table: Nannoolal et al. (2008) "
                          "vapour pressures via PyCHAM 5.8.0, c* at 298.15 K"),
}

# Product names that carry no chemical meaning and are dropped at parse time.
DROPPED_PRODUCTS = frozenset({"PROD"})

# --- Stiff solver defaults (diffrax) -------------------------------------------------
# Concentrations are integrated in molec/cm3. Chosen by the 24 h limonene benchmark
# (docs/aaar/gaschem.md): Kvaerno3 with KPP-like tolerances; atol = 1e4 molec/cm3
# (~0.4 ppq at 1 atm) is far below any species of interest and removes the step-size
# collapse that atol ~ 1 molec/cm3 causes for near-zero species.
SOLVER = "kvaerno3"
SOLVER_RTOL = 1e-3
SOLVER_ATOL = 1e4               # [molec/cm3]
# Step budget (accepted + rejected) per coupling interval; when it runs out,
# integrate_interval raises and the run fails. One step costs ~9 ms (725 species,
# laptop CPU) and cannot be interrupted, so this caps one interval at ~20 s. Measured
# need: <= 23 steps per interval for the presets and 161 for the most extreme
# tomas-api input tried (Level 3 gases x1000, capped at 1e5 ppb, 900 s intervals).
SOLVER_MAX_STEPS = 2_000

# --- Photolysis (TUV-x) --------------------------------------------------------------
# TUV-x configuration (relative to the tuvx-jax checkout, see photolysis.tuvx_root()).
TUVX_CONFIG = "examples/tuv_5_4_no_aerosol.json"
# Surface J is tabulated once per process on this SZA grid [deg] and interpolated;
# J = 0 at and beyond 90 deg (same convention as PhotolysisCalculator.rate_constants).
TUVX_SZA_GRID_DEG = tuple(float(a) for a in range(0, 91))

# --- Volatility classes (SOA partitioning) --------------------------------------------
# Condensable organics are lumped into half-decade c* classes (log10 c* at 298 K, ug/m3)
# that map onto TOMAS organic slots 1..N (ASSUMPTIONS A6). Species more volatile than the
# top class edge stay gas-only; less volatile than the bottom edge join the bottom class.
VOLATILITY_LOG10_CSTAR_CENTERS = tuple(-3.0 + 0.5 * i for i in range(15))   # -3 ... 4
VOLATILITY_CLASS_HALF_WIDTH = 0.25
# Enthalpy of vaporization [kJ/mol] = A - B*log10(c*_298) (Epstein et al., 2010, as used
# in SOM-TOMAS soacond; c* in ug/m3).
DHVAP_A_KJ = 131.0
DHVAP_B_KJ = 11.0
# Molecular weight assigned to a class with no member species [g/mol].
DEFAULT_CLASS_MW = 200.0
