"""Configuration for the AAAR tutorial box model.

Static settings (hashable frozen dataclasses): changing a field recompiles
the jitted step that receives it.
"""
from dataclasses import dataclass

from ..physics.ppm_remap import BIN_TRANSFER_PPM, BIN_TRANSFERS
from ..physics.soa_partitioning import SOA_N_SUBSTEPS
from ..physics.water_equilibrium import WATER_SCHEME_BISULFATE


@dataclass(frozen=True)
class AerosolStepConfig:
    """Settings of box.aerosol_step.aerosol_outer_step.

    Number-threshold policy: TOMAS thresholds are per grid cell (e.g.
    NEPS_CONDSINK = 1e10 #/cell decides whether a bin is populated for
    sink geometry, SOA geometry and nitrate placement). The tutorial box
    volume of 1e10 cm^3 makes that 1 #/cm^3; smaller boxes treat real
    bins as empty.
    """
    micro_dt: float = 60.0             # [s] micro-step inside an outer step
    boxvol_cm3: float = 1.0e10         # [cm^3] tutorial box volume
    alpha_h2so4: float = 1.0           # [-] H2SO4 mass accommodation
    water_scheme: float = WATER_SCHEME_BISULFATE   # see equilibrate_water
    water_every_process: bool = False  # make_step option (TOMAS box.f order)
    soa_n_substeps: int = SOA_N_SUBSTEPS  # vbs-jax adaptive sub-step budget
    # How SOA growth/evaporation and N2O5-uptake nitrate (mass added to or
    # removed from bins in place) move particles between the fixed bins:
    # 'ppm' (PPM log-mass remap, as H2SO4 condensation) or 'mnfix' (the
    # previous MNFIX split, kept for comparison). docs/aaar/aerosol_step.md
    growth_bin_transfer: str = BIN_TRANSFER_PPM

    def __post_init__(self):
        if self.growth_bin_transfer not in BIN_TRANSFERS:
            raise ValueError(
                f"growth_bin_transfer={self.growth_bin_transfer!r}; "
                f"valid: {BIN_TRANSFERS}")
