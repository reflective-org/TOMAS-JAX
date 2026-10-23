"""Glue between the gas-phase chemistry (molec/cm3) and TOMAS (kg/cell, #/cell).

* unit conversions and the initial seed aerosol,
* the tutorial edits applied to the MCM mechanism (ASSUMPTIONS A1-A4), and
* gather/scatter of condensable species to/from the c*-class gas slots
  ``Gc[SRTSOA1:SRTSOA1+K]``, with a per-species particle tracker P_i [kg/cell] that
  keeps the explicit identity of the particle-phase organics (for "top contributors").

Scatter rule (mass-conserving per class): net condensation Δg_k < 0 is taken from each
member species in proportion to its share of the class gas; net evaporation Δg_k > 0
is returned in proportion to its share of the class particle tracker. Afterwards P is
rescaled so that sum_{i in k} P_i equals the size-resolved class mass sum_bins Mk[:, k]
(this absorbs coagulation top-bin overflow).
"""
from __future__ import annotations

import dataclasses
import functools

import numpy as np

from ..core.config import (ICOMP, MW_NH4, MW_NO3, MW_SO4, SRTNH4, SRTNIT, SRTORG_NUC, SRTSO4,
                           make_grid, XK0)
from ..gaschem.config import AVOGADRO, LIMONENE_EQN, TOLUENE_EQN
from ..gaschem.kpp_parser import ParsedMechanism, Reaction, parse_eqn
from ..gaschem.merge import MergeReport, merge_mechanisms

RHO_SEED_INIT = 1770.0     # [kg/m3] density used to place the seed on the mass grid (as tomas-api)
RHO_REPORT = 1400.0        # [kg/m3] density for reported dry diameters (as tomas-api)
MW_AMMONIUM_SULFATE = MW_SO4 + 2.0 * MW_NH4   # 132 g/mol
MW_AMMONIUM_NITRATE = MW_NO3 + MW_NH4         # 80 g/mol

# --- Tutorial edits to the supplied mechanism (documented in ASSUMPTIONS.md) -----------
N2O5_HET_LABEL = "34"      # <34> N2O5 = NA + NA : 4e-4 (fixed chamber sink) -> gamma-based uptake
NA_HET = "NA_HET"          # counter for particle nitrate from <34> (A1); <33>'s NA stays a gas sink (A2)


@functools.lru_cache(maxsize=1)
def toluene_merge() -> tuple[ParsedMechanism, MergeReport]:
    """The limonene subset with the MCM toluene subset merged in (gaschem.merge), as
    exported: no tutorial edits yet."""
    return merge_mechanisms(parse_eqn(LIMONENE_EQN), parse_eqn(TOLUENE_EQN))


@functools.lru_cache(maxsize=1)
def toluene_only_species() -> frozenset[str]:
    """Species the toluene subset adds to the limonene mechanism (TOLUENE and its products)."""
    return frozenset(toluene_merge()[1].new_species)


def uses_toluene(gases_ppb: dict, emissions_ppb_per_h: dict, hold_constant=(),
                 outdoor_gases_ppb: dict | None = None) -> bool:
    """True when a scenario gives a toluene-subset species a non-zero amount or holds one.

    The toluene chemistry (181 species, 505 reactions) is integrated only then; with every
    toluene species at zero the run uses the limonene mechanism alone and gives the same
    numbers as before the toluene subset existed (ASSUMPTIONS A27).
    """
    extra = toluene_only_species()
    named = {sp for sp, v in gases_ppb.items() if v > 0}
    named |= {sp for sp, v in emissions_ppb_per_h.items() if v > 0}
    named |= {sp for sp, v in (outdoor_gases_ppb or {}).items() if v > 0}
    named |= set(hold_constant)
    return bool(named & extra)


@functools.lru_cache(maxsize=2)
def tutorial_mechanism(toluene: bool = False) -> ParsedMechanism:
    """The limonene mechanism (with the toluene subset merged in when ``toluene``) with
    <34>'s product renamed to the NA_HET counter."""
    parsed = toluene_merge()[0] if toluene else parse_eqn(LIMONENE_EQN)
    reactions = []
    for r in parsed.reactions:
        if r.label == N2O5_HET_LABEL:
            if r.reactants != ("N2O5",) or r.products != (("NA", 2.0),):
                raise ValueError(f"unexpected reaction <{r.label}>: {r}")
            r = Reaction(r.label, r.reactants, ((NA_HET, 2.0),), r.rate_expr, r.photolysis)
        reactions.append(r)
    return dataclasses.replace(
        parsed, reactions=reactions, species=parsed.species + [NA_HET],
        atoms={**parsed.atoms, NA_HET: {}})


# --- Units ------------------------------------------------------------------------------
def cm3_to_kg_cell(conc_cm3, mw_g_mol, boxvol_cm3):
    """molec/cm3 -> kg/cell."""
    return np.asarray(conc_cm3) * boxvol_cm3 * np.asarray(mw_g_mol) * 1e-3 / AVOGADRO


def kg_cell_to_cm3(mass_kg, mw_g_mol, boxvol_cm3):
    """kg/cell -> molec/cm3."""
    return np.asarray(mass_kg) * AVOGADRO / (np.asarray(mw_g_mol) * 1e-3 * boxvol_cm3)


def kg_cell_to_ugm3(mass_kg, boxvol_cm3):
    """kg/cell -> ug/m3."""
    return np.asarray(mass_kg) * 1e15 / boxvol_cm3


def ugm3_to_kg_cell(conc_ugm3, boxvol_cm3):
    return np.asarray(conc_ugm3) * boxvol_cm3 * 1e-15


def mass_grid(nbins: int) -> np.ndarray:
    """Bin-edge masses [kg]: 1.7 nm start, same diameter range for any nbins (as tomas-api)."""
    return np.asarray(make_grid(nbins, XK0, 2.0 ** (40.0 / nbins)))


def bin_diameters_nm(xk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(centres, edges) dry diameters [nm] at RHO_REPORT."""
    def d(m):
        return (6.0 * m / (np.pi * RHO_REPORT)) ** (1.0 / 3.0) * 1e9
    return d(np.sqrt(xk[:-1] * xk[1:])), d(xk)


def seed_distribution(modes, organic_mass_fraction: float, xk: np.ndarray, boxvol_cm3: float, *,
                      ammonium_nitrate_mass_fraction: float = 0.0, ammonium: bool = False):
    """Lognormal seed (sum of modes) -> (Nk [#/cell], Mk [kg/cell]); OA in SRTORG_NUC.

    Same placement as tomas-api ``_create_lognormal_dist``: dN/dlnDp at the bin
    midpoint (sulfate density) times the bin width in ln Dp.

    Dry composition (ASSUMPTIONS A15): organic_mass_fraction in SRTORG_NUC,
    ammonium_nitrate_mass_fraction as NH4NO3 (NO3 in SRTNIT), the rest sulfate. With
    ``ammonium`` the salts carry their ammonium in SRTNH4: (NH4)2SO4 = 96/132 SO4 +
    36/132 NH4, NH4NO3 = 62/80 NO3 + 18/80 NH4 by mass. Without it (the default and the
    Level 1-3 presets) SO4 and NO3 are placed alone.
    """
    m_mid = np.sqrt(xk[:-1] * xk[1:])
    dp_mid = (6.0 * m_mid / (np.pi * RHO_SEED_INIT)) ** (1.0 / 3.0)
    dlndp = np.log(xk[1:] / xk[:-1]) / 3.0
    nk = np.zeros(m_mid.size)
    for mode in modes:
        ln_gsd = np.log(mode.gsd)
        pdf = mode.number_cm3 / (np.sqrt(2 * np.pi) * ln_gsd) * np.exp(
            -(np.log(dp_mid) - np.log(mode.gmd_nm * 1e-9)) ** 2 / (2 * ln_gsd**2))
        nk += pdf * dlndp * boxvol_cm3
    mk = np.zeros((m_mid.size, ICOMP))
    dry = nk * m_mid
    f_an = ammonium_nitrate_mass_fraction
    mk[:, SRTORG_NUC] = dry * organic_mass_fraction
    if not ammonium and f_an == 0.0:
        mk[:, SRTSO4] = dry * (1.0 - organic_mass_fraction)      # as before 2026-10-19
        return nk, mk
    f_as = 1.0 - organic_mass_fraction - f_an
    if ammonium:
        mk[:, SRTSO4] = dry * f_as * (MW_SO4 / MW_AMMONIUM_SULFATE)
        mk[:, SRTNIT] = dry * f_an * (MW_NO3 / MW_AMMONIUM_NITRATE)
        mk[:, SRTNH4] = dry * (f_as * (2.0 * MW_NH4 / MW_AMMONIUM_SULFATE)
                               + f_an * (MW_NH4 / MW_AMMONIUM_NITRATE))
    else:
        mk[:, SRTSO4] = dry * f_as
        mk[:, SRTNIT] = dry * f_an
    return nk, mk


# --- Gather / scatter of condensable organics ------------------------------------------
class ClassBridge:
    """Moves condensable species between the gas state y [molec/cm3] and class slots."""

    def __init__(self, class_map, boxvol_cm3: float):
        self.idx = np.asarray(class_map.species_idx)
        self.cls = np.asarray(class_map.class_idx)
        self.k = class_map.class_cstar.size
        self.mw = np.asarray(class_map.mw)
        self.boxvol = boxvol_cm3
        self.particle = np.zeros(self.idx.size)          # P_i [kg/cell]

    def gas_mass(self, y: np.ndarray) -> np.ndarray:
        """Per-species condensable gas mass [kg/cell]."""
        return cm3_to_kg_cell(y[self.idx], self.mw, self.boxvol)

    def class_sum(self, per_species: np.ndarray) -> np.ndarray:
        return np.bincount(self.cls, weights=per_species, minlength=self.k)

    def scatter(self, y: np.ndarray, gas_before: np.ndarray, class_gas_after: np.ndarray,
                class_particle: np.ndarray) -> np.ndarray:
        """Distribute the class gas change back to species; returns the updated y.

        Args:
            gas_before: per-species gas mass at gather time [kg/cell]
            class_gas_after: class gas mass after the aerosol step [kg/cell], shape (K,)
            class_particle: size-integrated class particle mass after the step [kg/cell]
        """
        class_before = self.class_sum(gas_before)
        delta = class_gas_after - class_before                       # (K,)
        gas_share = np.divide(gas_before, class_before[self.cls],
                              out=np.zeros_like(gas_before), where=class_before[self.cls] > 0)
        class_p = self.class_sum(self.particle)
        part_share = np.divide(self.particle, class_p[self.cls],
                               out=gas_share.copy(), where=class_p[self.cls] > 0)
        d = delta[self.cls]
        transfer = np.where(d < 0, d * gas_share, d * part_share)     # gas change per species
        gas_after = np.maximum(gas_before + transfer, 0.0)
        self.particle = np.maximum(self.particle - transfer, 0.0)
        # keep the tracker consistent with the size-resolved TOMAS class mass
        class_p = self.class_sum(self.particle)
        scale = np.divide(class_particle, class_p, out=np.zeros(self.k), where=class_p > 0)
        self.particle = self.particle * scale[self.cls]
        y = y.copy()
        y[self.idx] = kg_cell_to_cm3(gas_after, self.mw, self.boxvol)
        return y
