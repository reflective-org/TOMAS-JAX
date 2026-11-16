"""Merge two MCM KPP subsets (e.g. limonene + toluene) into one mechanism, in code.

The MCM website exports every subset with the same inorganic chemistry, the same
generic rate coefficients and the same reactions of the small shared oxidation products
(CH3O2, HCHO, GLYOX, ...). Merging means:

* species: the union, base order first, then the species only the extra file has;
  a shared species must have the same atom composition in both files;
* reactions: a reaction of the extra file that the base already has — same reactants
  (as a multiset), same products with the same coefficients and the same rate
  expression (whitespace ignored) — is kept once, under its base label; a reaction
  with the same reactants and products but a different rate expression is a conflict
  and raises (none exist between the limonene and toluene exports, see
  :class:`MergeReport`); every other reaction is added with its label prefixed by
  ``label_prefix`` so labels stay unique;
* the RO2 sum: the union of both files' peroxy-radical lists, base order first.

The extra file's rate expressions must evaluate with the same generic coefficients as
the base; :class:`tomas_jax.gaschem.mechanism.RateModel` checks every name.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from .kpp_parser import ParsedMechanism, Reaction

_WS_RE = re.compile(r"\s+")


@dataclass(frozen=True)
class MergeReport:
    """What :func:`merge_mechanisms` did (counts before and after, what was shared)."""
    base_species: int
    extra_species: int
    merged_species: int
    base_reactions: int
    extra_reactions: int
    merged_reactions: int
    base_ro2: int
    extra_ro2: int
    merged_ro2: int
    shared_species: tuple[str, ...]          # in both files (same atoms)
    new_species: tuple[str, ...]             # only in the extra file, file order
    shared_reactions: tuple[tuple[str, str], ...]   # (extra label, base label) kept once
    new_reactions: tuple[str, ...]           # merged labels of the added reactions
    new_j: tuple[str, ...]                   # photolysis names the base does not use


def _reaction_key(r: Reaction) -> tuple:
    return (tuple(sorted(r.reactants)), tuple(sorted(r.products)), _WS_RE.sub("", r.rate_expr))


def _rp_key(r: Reaction) -> tuple:
    return (tuple(sorted(r.reactants)), tuple(sorted(r.products)))


_J_RE = re.compile(r"\bJ\(\s*(J_[A-Za-z0-9_]+)\s*\)")


def _j_names(reactions) -> set[str]:
    return {j for r in reactions for j in _J_RE.findall(r.rate_expr)}


def merge_mechanisms(base: ParsedMechanism, extra: ParsedMechanism,
                     label_prefix: str = "T") -> tuple[ParsedMechanism, MergeReport]:
    """Merge ``extra`` into ``base`` (see the module docstring).

    Raises ``ValueError`` on a shared species with different atom compositions, on a
    reaction with the same reactants and products but a different rate expression (a
    conflicting duplicate; which version to keep is a decision, not a default), and on
    a prefixed label that collides with a base label.
    """
    base_set = set(base.species)
    atom_conflicts = [s for s in extra.species if s in base_set and base.atoms[s] != extra.atoms[s]]
    if atom_conflicts:
        raise ValueError(f"species declared with different atoms in the two files: {atom_conflicts}")
    shared_species = tuple(s for s in extra.species if s in base_set)
    new_species = tuple(s for s in extra.species if s not in base_set)

    base_by_key = {_reaction_key(r): r for r in base.reactions}
    base_by_rp: dict[tuple, list[Reaction]] = {}
    for r in base.reactions:
        base_by_rp.setdefault(_rp_key(r), []).append(r)
    base_labels = {r.label for r in base.reactions}

    reactions = list(base.reactions)
    shared, added, conflicts = [], [], []
    for r in extra.reactions:
        same = base_by_key.get(_reaction_key(r))
        if same is not None:
            shared.append((r.label, same.label))
            continue
        if _rp_key(r) in base_by_rp:
            conflicts.append((r.label, r.rate_expr, [(b.label, b.rate_expr) for b in base_by_rp[_rp_key(r)]]))
            continue
        label = f"{label_prefix}{r.label}"
        if label in base_labels:
            raise ValueError(f"merged label {label!r} collides with a base reaction label")
        reactions.append(Reaction(label, r.reactants, r.products, r.rate_expr, r.photolysis))
        added.append(label)
    if conflicts:
        lines = [f"extra <{lab}> {rate!r} vs base {bases}" for lab, rate, bases in conflicts[:10]]
        raise ValueError("reactions with the same reactants and products but different rate "
                         f"expressions ({len(conflicts)}): " + "; ".join(lines))

    ro2 = list(base.ro2) + [s for s in extra.ro2 if s not in set(base.ro2)]
    merged = ParsedMechanism(
        species=list(base.species) + list(new_species),
        atoms={**extra.atoms, **base.atoms},
        reactions=reactions,
        ro2=ro2,
        unused_declared=list(base.unused_declared),
    )
    report = MergeReport(
        base_species=len(base.species), extra_species=len(extra.species),
        merged_species=len(merged.species),
        base_reactions=len(base.reactions), extra_reactions=len(extra.reactions),
        merged_reactions=len(merged.reactions),
        base_ro2=len(base.ro2), extra_ro2=len(extra.ro2), merged_ro2=len(ro2),
        shared_species=shared_species, new_species=new_species,
        shared_reactions=tuple(shared), new_reactions=tuple(added),
        new_j=tuple(sorted(_j_names(extra.reactions) - _j_names(base.reactions))),
    )
    return merged, report
