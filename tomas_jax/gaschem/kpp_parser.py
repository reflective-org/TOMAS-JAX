"""Parser for MCM-generated KPP equation files (``*.eqn``).

Reads the three kinds of content an MCM KPP export contains:

* species declarations ``NAME = 2C + 4H + O ;`` (atom composition, or ``IGNORE``),
* the ``#INLINE F90_RCONST ... #ENDINLINE`` block, from which only the peroxy-radical
  sum ``RO2 = C(ind_X) + C(ind_Y) + ...`` is extracted (everything else in the block,
  e.g. ``USE constants_mcm``, is provided by :mod:`tomas_jax.gaschem.mcm_rates`),
* reactions ``<label> A + B = C + D : <Fortran rate expression> ;``.

Stoichiometry: MCM KPP files encode a coefficient of 2 by repeating the species
(``NO2 + NO2``); explicit numeric coefficients are also accepted (``0.5OH``).
``hv`` on the reactant side marks photolysis and is not a species.

Pure Python, no JAX: the result is a plain description that
:mod:`tomas_jax.gaschem.mechanism` turns into arrays.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from .config import ATOMIC_WEIGHTS, DROPPED_PRODUCTS

_SPECIES_RE = re.compile(r"^\s*([A-Za-z][A-Za-z0-9_]*)\s*=\s*(.*?)\s*;\s*$")
_REACTION_RE = re.compile(r"^\s*<\s*([^>]+?)\s*>\s*(.*?)=(.*?):(.*);\s*$")
_TERM_RE = re.compile(r"^([0-9]*\.?[0-9]*(?:[eE][-+]?[0-9]+)?)\s*([A-Za-z][A-Za-z0-9_]*)$")
_ATOM_RE = re.compile(r"([0-9]*)\s*([A-Z][a-z]?)")
_RO2_TERM_RE = re.compile(r"C\(\s*ind_([A-Za-z0-9_]+)\s*\)")


@dataclass(frozen=True)
class Reaction:
    label: str
    reactants: tuple[str, ...]                 # repeated names encode stoichiometry
    products: tuple[tuple[str, float], ...]    # (species, coefficient)
    rate_expr: str                             # Fortran expression, verbatim
    photolysis: bool


@dataclass
class ParsedMechanism:
    species: list[str]                          # species that appear in reactions, file order
    atoms: dict[str, dict[str, int]]            # composition per declared species ({} for IGNORE)
    reactions: list[Reaction]
    ro2: list[str]                              # members of the inline RO2 sum
    unused_declared: list[str] = field(default_factory=list)

    def molecular_weight(self, name: str) -> float:
        """Molecular weight [g/mol] from the declared composition (NaN if IGNORE)."""
        comp = self.atoms.get(name, {})
        if not comp:
            return float("nan")
        return sum(ATOMIC_WEIGHTS[a] * n for a, n in comp.items())


def _parse_atoms(text: str) -> dict[str, int]:
    if text.strip().upper() == "IGNORE":
        return {}
    atoms: dict[str, int] = {}
    for part in text.split("+"):
        m = _ATOM_RE.fullmatch(part.strip())
        if not m:
            raise ValueError(f"cannot parse atom term {part!r} in {text!r}")
        count, symbol = m.groups()
        if symbol not in ATOMIC_WEIGHTS:
            raise ValueError(f"unknown element {symbol!r} in {text!r}")
        atoms[symbol] = atoms.get(symbol, 0) + (int(count) if count else 1)
    return atoms


def _parse_side(text: str) -> list[tuple[str, float]]:
    terms = []
    for raw in text.split("+"):
        raw = raw.strip()
        m = _TERM_RE.fullmatch(raw)
        if not m:
            raise ValueError(f"cannot parse term {raw!r}")
        coef, name = m.groups()
        terms.append((name, float(coef) if coef else 1.0))
    return terms


def parse_eqn(path: str | Path) -> ParsedMechanism:
    """Parse an MCM KPP ``.eqn`` file. Raises ``ValueError`` on anything unrecognised."""
    atoms: dict[str, dict[str, int]] = {}
    reactions: list[Reaction] = []
    inline: list[str] = []
    in_inline = False

    for lineno, line in enumerate(Path(path).read_text().splitlines(), start=1):
        stripped = line.strip()
        if not stripped or stripped.startswith("//"):
            continue
        if stripped.startswith("#INLINE"):
            in_inline = True
            continue
        if stripped.startswith("#ENDINLINE"):
            in_inline = False
            continue
        if in_inline:
            inline.append(stripped)
            continue
        if stripped.startswith("<"):
            m = _REACTION_RE.match(stripped)
            if not m:
                raise ValueError(f"line {lineno}: cannot parse reaction: {stripped}")
            label, lhs, rhs, rate = m.groups()
            lhs_terms = _parse_side(lhs)
            photolysis = any(name == "hv" for name, _ in lhs_terms)
            reactants: list[str] = []
            for name, coef in lhs_terms:
                if name == "hv":
                    continue
                if coef != int(coef):
                    raise ValueError(f"line {lineno}: non-integer reactant coefficient")
                reactants.extend([name] * int(coef))
            products: dict[str, float] = {}
            for name, coef in _parse_side(rhs):
                if name in DROPPED_PRODUCTS:
                    continue
                products[name] = products.get(name, 0.0) + coef
            reactions.append(Reaction(label, tuple(reactants), tuple(products.items()),
                                      rate.strip(), photolysis))
            continue
        m = _SPECIES_RE.match(stripped)
        if m:
            atoms[m.group(1)] = _parse_atoms(m.group(2))
            continue
        raise ValueError(f"line {lineno}: unrecognised content: {stripped}")

    ro2 = _RO2_TERM_RE.findall(" ".join(inline))

    species: list[str] = []
    seen: set[str] = set()
    for rxn in reactions:
        for name in (*rxn.reactants, *(p for p, _ in rxn.products)):
            if name not in seen:
                seen.add(name)
                species.append(name)
    undeclared = [s for s in species if s not in atoms]
    if undeclared:
        raise ValueError(f"species used in reactions but not declared: {undeclared}")
    missing_ro2 = [s for s in ro2 if s not in seen]
    if missing_ro2:
        raise ValueError(f"RO2 members not used in any reaction: {missing_ro2}")

    return ParsedMechanism(
        species=species,
        atoms=atoms,
        reactions=reactions,
        ro2=ro2,
        unused_declared=[s for s in atoms if s not in seen],
    )
