"""Run description for the coupled box model (the input contract of ``box.driver.run``).

Plain frozen dataclasses with physical units in the field names so the web API can
mirror them 1:1. Presets live in ``box/presets/*.toml`` and load with
:func:`load_preset`; ``Scenario.replace`` applies user overrides.

Each class checks its own values when it is built (wrong type or outside the range the
model is valid for -> ValueError naming the field, with a hint for common unit slips
such as RH in percent). The checks only raise; a valid scenario is unchanged. A single
species name for ``hold_constant`` is taken as one species.
"""
from __future__ import annotations

import dataclasses
import math
import numbers
import tomllib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PRESET_DIR = Path(__file__).parent / "presets"
PHOTOLYSIS_MODES = ("tuvx", "mcm", "constant", "off")


def _number(where: str, value, lo: float | None = None, hi: float | None = None, *,
            lo_open: bool = False, hint: str = "") -> None:
    """Raise ValueError unless ``value`` is a finite real number in [lo, hi] ((lo, hi] if lo_open)."""
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise ValueError(f"{where} must be a number, got {value!r}")
    v = float(value)
    below = lo is not None and (v <= lo if lo_open else v < lo)
    if not math.isfinite(v) or below or (hi is not None and v > hi):
        if lo is not None and hi is not None:
            allowed = f"from {lo:g} to {hi:g}" + (f" (above {lo:g})" if lo_open else "")
        elif lo is not None:
            allowed = f"above {lo:g}" if lo_open else f"at least {lo:g}"
        else:
            allowed = f"at most {hi:g}"
        raise ValueError(f"{where} must be {allowed}, got {value!r}" + (f"; {hint}" if hint else ""))


def _switch(where: str, value) -> None:
    """Process switches are True or False only (a string such as "off" would count as on)."""
    if not isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{where} must be True or False, got {value!r}")


@dataclass(frozen=True)
class SeedMode:
    number_cm3: float                      # total number concentration [#/cm3]
    gmd_nm: float                          # geometric mean (dry) diameter [nm]
    gsd: float                             # geometric standard deviation [-]

    def __post_init__(self):
        _number("seed mode number_cm3", self.number_cm3, 0.0)
        _number("seed mode gmd_nm", self.gmd_nm, 0.0, lo_open=True)
        _number("seed mode gsd", self.gsd, 1.0, lo_open=True,
                hint="the geometric standard deviation is a ratio, e.g. 1.6")


@dataclass(frozen=True)
class Seed:
    modes: tuple[SeedMode, ...] = (SeedMode(2000.0, 80.0, 1.6),)
    organic_mass_fraction: float = 0.3     # non-volatile, absorbing OA; the rest is sulfate

    def __post_init__(self):
        if isinstance(self.modes, (str, bytes)) or not all(isinstance(m, SeedMode) for m in self.modes):
            raise ValueError(f"seed.modes must be a list of seed modes, got {self.modes!r}")
        _number("seed.organic_mass_fraction", self.organic_mass_fraction, 0.0, 1.0,
                hint=_fraction_hint(self.organic_mass_fraction))


@dataclass(frozen=True)
class Environment:
    temp_k: float = 298.15
    pres_pa: float = 101325.0
    rh: float = 0.5                        # [0-1]

    def __post_init__(self):
        _number("environment.temp_k", self.temp_k, 100.0, 400.0,
                hint=f"temperature is in kelvin; did you mean {float(self.temp_k) + 273.15:g} K?"
                if isinstance(self.temp_k, numbers.Real) and -100 <= self.temp_k < 100 else "")
        _number("environment.pres_pa", self.pres_pa, 0.0, 120_000.0, lo_open=True,
                hint="pressure is in pascal; 1 atm = 101325 Pa")
        _number("environment.rh", self.rh, 0.0, 1.0, hint=_fraction_hint(self.rh))


@dataclass(frozen=True)
class Photolysis:
    mode: str = "off"                      # "tuvx" | "mcm" | "constant" | "off"
    latitude_deg: float = 42.9
    longitude_deg: float = -78.9
    jno2_per_s: float = 5e-3               # "constant" mode only
    constant_sza_deg: float = 30.0         # "constant" mode only

    def __post_init__(self):
        if self.mode not in PHOTOLYSIS_MODES:
            raise ValueError(f"photolysis.mode must be one of {list(PHOTOLYSIS_MODES)}, got {self.mode!r}")
        _number("photolysis.latitude_deg", self.latitude_deg, -90.0, 90.0)
        _number("photolysis.longitude_deg", self.longitude_deg, -180.0, 360.0)
        _number("photolysis.jno2_per_s", self.jno2_per_s, 0.0)
        _number("photolysis.constant_sza_deg", self.constant_sza_deg, 0.0, 90.0)


@dataclass(frozen=True)
class Surrogate:
    """Level 1: one condensing organic of user-chosen volatility (no chemistry)."""
    cstar_ugm3: float = 1.0                # saturation concentration at 298 K
    production_ugm3_per_h: float = 0.5     # gas-phase production; 0 = off
    initial_gas_ugm3: float = 0.0
    mw_g_mol: float = 200.0

    def __post_init__(self):
        _number("surrogate.cstar_ugm3", self.cstar_ugm3, 0.0, lo_open=True)
        _number("surrogate.production_ugm3_per_h", self.production_ugm3_per_h, 0.0)
        _number("surrogate.initial_gas_ugm3", self.initial_gas_ugm3, 0.0)
        _number("surrogate.mw_g_mol", self.mw_g_mol, 0.0, lo_open=True)


@dataclass(frozen=True)
class Processes:
    chemistry: bool = False                # MCM limonene gas-phase chemistry
    nucleation: bool = False
    coagulation: bool = True
    h2so4_condensation: bool = True
    soa_partitioning: bool = True
    n2o5_uptake: bool = False

    def __post_init__(self):
        for f in dataclasses.fields(self):
            _switch(f"processes.{f.name}", getattr(self, f.name))


@dataclass(frozen=True)
class Numerics:
    coupling_dt_s: float = 300.0           # gas <-> aerosol operator-split interval
    output_dt_s: float = 600.0
    nbins: int = 40

    def __post_init__(self):
        _number("numerics.coupling_dt_s", self.coupling_dt_s, 0.0, lo_open=True)
        _number("numerics.output_dt_s", self.output_dt_s, 0.0, lo_open=True)
        if isinstance(self.nbins, bool) or not isinstance(self.nbins, numbers.Integral) or self.nbins < 2:
            raise ValueError(f"numerics.nbins must be a whole number of bins (e.g. 40), got {self.nbins!r}")


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
        held = self.hold_constant
        # one name, not its letters: hold_constant="NO2" holds NO2, not N, O and 2
        held = (held,) if isinstance(held, str) else tuple(held)
        if not all(isinstance(sp, str) for sp in held):
            raise ValueError(f"hold_constant must be species names, got {self.hold_constant!r}")
        object.__setattr__(self, "hold_constant", held)
        _number("duration_h", self.duration_h, 0.0, lo_open=True)
        if not isinstance(self.gases_ppb, dict):
            raise ValueError(f"gases_ppb must map species names to mixing ratios, got {self.gases_ppb!r}")
        for sp, ppb in self.gases_ppb.items():
            if not isinstance(sp, str):
                raise ValueError(f"gases_ppb keys must be species names, got {sp!r}")
            _number(f"gases_ppb[{sp!r}]", ppb, 0.0)
        _number("h2so4_production_cm3_s", self.h2so4_production_cm3_s, 0.0)
        _number("gamma_n2o5", self.gamma_n2o5, 0.0, 1.0)
        for name, cls in _NESTED.items():
            value = getattr(self, name)
            if not (isinstance(value, cls) or (value is None and name == "surrogate")):
                raise ValueError(f"{name} must be a scenario.{cls.__name__} object, got {value!r} (for a dict "
                                 "of its fields use Scenario.replace or from_dict)")

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


def _fraction_hint(value) -> str:
    """'did you mean 0.8?' for a fraction given in percent (1 < value <= 100)."""
    if isinstance(value, numbers.Real) and not isinstance(value, bool) and 1.0 < value <= 100.0:
        return f"it is a fraction, not a percentage: did you mean {float(value) / 100.0:g}?"
    return ""


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
