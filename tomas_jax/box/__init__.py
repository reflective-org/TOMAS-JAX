"""Coupled tutorial box model for the AAAR tutorial (docs/aaar/): MCM gas chemistry +
TUV-x photolysis + TOMAS microphysics with c*-class SOA partitioning.

    scenario     - run description (Scenario) and TOML presets
    result       - output contract (RunResult)
    aerosol_step - aerosol side of one coupled step (aerosol_outer_step)
    config       - box-model configuration (AerosolStepConfig)
"""

from ..core.config import require_organics as _require_organics
_require_organics('The box model (SOA/nitrate/organic species)')
