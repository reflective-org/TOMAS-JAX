"""Volatility (c*) of mechanism species and their lumping into c* classes.

c* (saturation mass concentration at 298 K, ug/m3) comes from a per-species table
(the Level 2/3 presets use ``config.DEFAULT_CSTAR_TABLE``: Nannoolal et al. (2008)
vapour pressures via PyCHAM, supplied by S. Jathar) and, for closed-shell organics
the table does not list (or when no table is given), from the elemental composition
with the Li et al. (2016) parameterization:

    log10 c* = (n0C - nC) bC - nO bO - 2 nC nO/(nC + nO) bCO - nN bN - nS bS

Li, Y., Pöschl, U., and Shiraiwa, M.: Molecular corridors and parameterizations of
volatility in the chemical evolution of organic aerosols, ACP 16, 3327-3344, 2016
(Table 1, 298 K).

Only closed-shell organics partition: radicals (RO2, RO, acyl peroxy) are excluded by
the valence parity rule — a closed-shell CHONS(Cl,Br) molecule has an even number of
H + N + Cl + Br atoms.
"""
from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .config import (DEFAULT_CLASS_MW, DHVAP_A_KJ, DHVAP_B_KJ, VOLATILITY_CLASS_HALF_WIDTH,
                     VOLATILITY_LOG10_CSTAR_CENTERS)
from .kpp_parser import ParsedMechanism

# composition class -> (n0C, bC, bO, bCO, bN, bS), Li et al. (2016) Table 1
LI2016_PARAMS = {
    "CH": (23.80, 0.4861, 0.0, 0.0, 0.0, 0.0),
    "CHO": (22.66, 0.4481, 1.656, -0.7790, 0.0, 0.0),
    "CHN": (24.59, 0.4066, 0.0, 0.0, 0.9619, 0.0),
    "CHON": (24.13, 0.3667, 0.7732, -0.07790, 1.114, 0.0),
    "CHOS": (24.06, 0.3637, 1.327, -0.3988, 0.0, 0.7579),
    "CHONS": (28.50, 0.3848, 1.011, 0.2921, 1.053, 1.316),
}


def log10_cstar_li2016(atoms: dict[str, int]) -> float:
    """log10 c* [ug/m3, 298 K] of an organic molecule from its C/O/N/S counts."""
    nc, no, nn, ns = (atoms.get(a, 0) for a in ("C", "O", "N", "S"))
    if nc == 0:
        raise ValueError("Li et al. (2016) applies to organic (carbon-containing) species")
    key = "CH" + ("O" if no else "") + ("N" if nn else "") + ("S" if ns else "")
    if key not in LI2016_PARAMS:          # e.g. CHNS: no fit published
        raise ValueError(f"no Li et al. (2016) fit for composition class {key}")
    n0c, bc, bo, bco, bn, bs = LI2016_PARAMS[key]
    co = 2.0 * nc * no / (nc + no) if (nc + no) else 0.0
    return (n0c - nc) * bc - no * bo - co * bco - nn * bn - ns * bs


def is_closed_shell(atoms: dict[str, int]) -> bool:
    return (atoms.get("H", 0) + atoms.get("N", 0) + atoms.get("Cl", 0) + atoms.get("Br", 0)) % 2 == 0


# accepted c* columns [ug/m3 at 298 K]: our own format and the PyCHAM-derived table
CSTAR_COLUMNS = ("cstar_ugm3", "cstar_ug_m3_298.15K")


def load_cstar_table(path: str | Path) -> dict[str, float]:
    """c* table CSV -> {species: log10 c*} (c* in ug/m3 at 298 K).

    Columns ``species`` and one of :data:`CSTAR_COLUMNS`; other columns are ignored.
    Lines starting with ``#`` are comments.
    """
    with open(path, newline="") as f:
        lines = [line for line in f if not line.lstrip().startswith("#")]
    reader = csv.DictReader(lines)
    fields = [name.strip() for name in reader.fieldnames or ()]
    reader.fieldnames = fields
    column = next((c for c in CSTAR_COLUMNS if c in fields), None)
    if "species" not in fields or column is None:
        raise ValueError(f"{path}: need a 'species' column and one of {CSTAR_COLUMNS}; "
                         f"got {fields}")
    out = {}
    for row in reader:
        name = (row["species"] or "").strip()
        if not name:
            raise ValueError(f"{path}: row without a species name: {row}")
        if name in out:
            raise ValueError(f"{path}: species {name!r} listed twice")
        raw = (row[column] or "").strip()
        if not raw:
            raise ValueError(f"{path}: no c* for {name!r}")
        value = float(raw)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"c* must be positive and finite: {row}")
        out[name] = float(np.log10(value))
    return out


@dataclass(frozen=True)
class ClassMap:
    """Which mechanism species condense, into which class, and the class properties."""
    species: tuple[str, ...]          # condensable species names
    species_idx: np.ndarray           # their indices in the mechanism state vector
    class_idx: np.ndarray             # class of each condensable species (0 = least volatile)
    log10_cstar: np.ndarray           # per condensable species [log10 ug/m3, 298 K]
    mw: np.ndarray                    # per condensable species [g/mol]
    class_cstar: np.ndarray           # class c* [ug/m3, 298 K]
    class_dhvap: np.ndarray           # class enthalpy of vaporization [kJ/mol]
    class_mw: np.ndarray              # mean member MW [g/mol]
    source: str                       # provenance of the c* values
    # closed-shell organics missing from the c* table whose c* is the Li et al. (2016)
    # estimate (condensable or not); empty when no table is given (then all are Li 2016)
    fallback: tuple[str, ...] = ()


def assign_classes(parsed: ParsedMechanism, log10_cstar: dict[str, float] | None = None,
                   table_source: str = "table") -> ClassMap:
    """Lump the closed-shell organics of ``parsed`` into the configured c* classes.

    ``log10_cstar`` (a c* table) gives the c* of the species it lists; closed-shell
    organics it does not list fall back to the Li et al. (2016) estimate and are recorded
    in ``ClassMap.fallback``. Table rows for radicals or non-organics are ignored.
    ``table_source`` labels the table in ``ClassMap.source``.
    """
    centers = np.asarray(VOLATILITY_LOG10_CSTAR_CENTERS)
    top_edge = centers[-1] + VOLATILITY_CLASS_HALF_WIDTH
    table = log10_cstar or {}
    names, idx, cls, lc, mw, fallback = [], [], [], [], [], []
    for i, name in enumerate(parsed.species):
        atoms = parsed.atoms[name]
        if not atoms.get("C") or not is_closed_shell(atoms):
            continue
        if name in table:
            value = table[name]
        else:
            value = log10_cstar_li2016(atoms)
            if table:
                fallback.append(name)
        if value > top_edge:
            continue
        names.append(name)
        idx.append(i)
        cls.append(int(np.argmin(np.abs(centers - value))))
        lc.append(value)
        mw.append(parsed.molecular_weight(name))
    cls_arr, mw_arr = np.asarray(cls, dtype=int), np.asarray(mw)
    class_mw = np.array([mw_arr[cls_arr == k].mean() if np.any(cls_arr == k) else DEFAULT_CLASS_MW
                         for k in range(centers.size)])
    if table:
        n_table = sum(1 for n in names if n in table)
        n_fb_cond = len(names) - n_table
        unmatched = len(set(table) - set(names))
        source = (f"{table_source} ({n_table} of {len(names)} condensable species; "
                  f"Li et al. 2016 for {len(fallback)} closed-shell organics not in the table, "
                  f"{n_fb_cond} of them condensable; "
                  f"{unmatched} table rows not condensable or not in the mechanism)")
    else:
        source = "Li et al. (2016) composition-based estimate"
    return ClassMap(
        species=tuple(names), species_idx=np.asarray(idx, dtype=int), class_idx=cls_arr,
        log10_cstar=np.asarray(lc), mw=mw_arr, class_cstar=10.0 ** centers,
        class_dhvap=DHVAP_A_KJ - DHVAP_B_KJ * centers, class_mw=class_mw, source=source,
        fallback=tuple(fallback),
    )
