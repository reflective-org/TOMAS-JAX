"""Run description for the coupled box model (the input contract of ``box.driver.run``).

Plain frozen dataclasses with physical units in the field names so the web API can
mirror them 1:1. Presets live in ``box/presets/*.toml`` and load with
:func:`load_preset`; ``Scenario.replace`` applies user overrides.
"""
from __future__ import annotations

import dataclasses
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

PRESET_DIR = Path(__file__).parent / "presets"


@dataclass(frozen=True)
class SeedMode:
    number_cm3: float                      # total number concentration [#/cm3]
    gmd_nm: float                          # geometric mean (dry) diameter [nm]
    gsd: float                             # geometric standard deviation [-]


@dataclass(frozen=True)
class Seed:
    modes: tuple[SeedMode, ...] = (SeedMode(2000.0, 80.0, 1.6),)
    organic_mass_fraction: float = 0.3     # non-volatile, absorbing OA; the rest is sulfate


@dataclass(frozen=True)
class Environment:
    temp_k: float = 298.15
    pres_pa: float = 101325.0
    rh: float = 0.5                        # [0-1]


@dataclass(frozen=True)
class Photolysis:
    mode: str = "off"                      # "tuvx" | "mcm" | "constant" | "off"
    latitude_deg: float = 42.9
    longitude_deg: float = -78.9
    jno2_per_s: float = 5e-3               # "constant" mode only
    constant_sza_deg: float = 30.0         # "constant" mode only


@dataclass(frozen=True)
class Surrogate:
    """Level 1: one condensing organic of user-chosen volatility (no chemistry)."""
    cstar_ugm3: float = 1.0                # saturation concentration at 298 K
    production_ugm3_per_h: float = 0.5     # gas-phase production; 0 = off
    initial_gas_ugm3: float = 0.0
    mw_g_mol: float = 200.0


@dataclass(frozen=True)
class Processes:
    chemistry: bool = False                # MCM limonene gas-phase chemistry
    nucleation: bool = False
    coagulation: bool = True
    h2so4_condensation: bool = True
    soa_partitioning: bool = True
    n2o5_uptake: bool = False


@dataclass(frozen=True)
class Numerics:
    coupling_dt_s: float = 300.0           # gas <-> aerosol operator-split interval
    output_dt_s: float = 600.0
    nbins: int = 40


@dataclass(frozen=True)
class Scenario:
    name: str = "custom"
    level: int = 1                         # tutorial level (1, 2, 3); informational
    duration_h: float = 12.0
    start_utc: datetime = datetime(2026, 7, 1, 16, 0, tzinfo=timezone.utc)  # naive = UTC
    environment: Environment = field(default_factory=Environment)
    seed: Seed = field(default_factory=Seed)
    photolysis: Photolysis = field(default_factory=Photolysis)
    gases_ppb: dict[str, float] = field(default_factory=dict)   # initial mixing ratios
    hold_constant: tuple[str, ...] = ()    # species reset to their initial value each interval
    h2so4_production_cm3_s: float = 0.0    # Level 1 (no chemistry) H2SO4 source [molec/cm3/s]
    surrogate: Surrogate | None = None
    gamma_n2o5: float = 0.0                # N2O5 reactive uptake coefficient
    # c* table: CSV file name in gaschem/mechanisms; closed-shell organics it does not list
    # get the Li et al. (2016) estimate. None = Li et al. (2016) for all. The Level 2/3
    # presets use gaschem.config.DEFAULT_CSTAR_TABLE (ASSUMPTIONS A8).
    cstar_table: str | None = None
    processes: Processes = field(default_factory=Processes)
    numerics: Numerics = field(default_factory=Numerics)

    def __post_init__(self):
        object.__setattr__(self, "start_utc", _to_utc(self.start_utc))
        object.__setattr__(self, "hold_constant", tuple(self.hold_constant))

    def replace(self, **changes) -> "Scenario":
        """Copy with top-level or nested changes, e.g. ``environment={"rh": 0.8}``."""
        updated = {}
        for key, value in changes.items():
            if key in _NESTED and isinstance(value, dict):
                current = getattr(self, key)
                base = dataclasses.asdict(current) if current is not None else {}
                value = _build(_NESTED[key], {**base, **value})
            updated[key] = value
        return dataclasses.replace(self, **updated)


def _to_utc(value: datetime | str) -> datetime:
    """ISO string or datetime -> aware UTC datetime; naive values are taken as UTC."""
    if isinstance(value, str):
        value = datetime.fromisoformat(value)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def _build(cls, data: dict):
    """Construct a (nested) scenario dataclass from plain dicts/lists."""
    if cls is Seed:
        unknown = set(data) - {f.name for f in dataclasses.fields(Seed)}
        if unknown:
            raise TypeError(f"Seed got unexpected keys {sorted(unknown)}")
        data = {**{"modes": Seed().modes, "organic_mass_fraction": Seed().organic_mass_fraction}, **data}
        modes = tuple(m if isinstance(m, SeedMode) else SeedMode(**m) for m in data["modes"])
        return Seed(modes=modes, organic_mass_fraction=data["organic_mass_fraction"])
    return cls(**data)


_NESTED = {"environment": Environment, "seed": Seed, "photolysis": Photolysis,
           "surrogate": Surrogate, "processes": Processes, "numerics": Numerics}


def from_dict(data: dict) -> Scenario:
    kwargs = dict(data)
    for key, cls in _NESTED.items():
        if isinstance(kwargs.get(key), dict):
            kwargs[key] = _build(cls, kwargs[key])
    return Scenario(**kwargs)


def load_preset(name: str) -> Scenario:
    with open(PRESET_DIR / f"{name}.toml", "rb") as f:
        return from_dict(tomllib.load(f))


def list_presets() -> list[str]:
    return sorted(p.stem for p in PRESET_DIR.glob("*.toml"))
