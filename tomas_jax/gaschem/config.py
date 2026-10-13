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
SOLVER_MAX_STEPS = 20_000       # per coupling interval
