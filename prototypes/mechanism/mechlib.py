"""Prototype helpers for multi-mechanism support (planning only, not model code).

* ``extract_closure``: the subset of a parsed MCM mechanism reachable from a set of
  seed species (what the MCM website exports for a marklist, when the parent file
  already contains every reaction of those species).
* ``write_eqn``: write a ParsedMechanism back as an MCM-style KPP .eqn module.
* ``compose``: merge several parsed modules with explicit duplicate / conflict rules.
* ``atom_imbalance``: per-reaction element balance.

Uses only the public parser of tomas_jax.gaschem; changes nothing in the package.
"""
from __future__ import annotations

import dataclasses
import re
from collections import defaultdict
from dataclasses import dataclass, field

from tomas_jax.gaschem.kpp_parser import ParsedMechanism, Reaction

# Species that MCM writes into rate expressions or omits from products; never state species.
BATH = frozenset({"M", "O2", "N2", "H2O", "hv"})


# ----------------------------------------------------------------------- extraction
def extract_closure(parsed: ParsedMechanism, seeds: set[str],
                    drop_labels: frozenset[str] = frozenset()) -> ParsedMechanism:
    """Reactions whose reactants are all reachable from ``seeds`` (fixed point)."""
    have = set(seeds)
    chosen: list[Reaction] = []
    changed = True
    while changed:
        changed = False
        for r in parsed.reactions:
            if r.label in drop_labels or r in chosen:
                continue
            if all(s in have for s in r.reactants):
                chosen.append(r)
                for p, _ in r.products:
                    if p not in have:
                        have.add(p)
                        changed = True
    order = {id(r): i for i, r in enumerate(parsed.reactions)}
    chosen.sort(key=lambda r: order[id(r)])
    species: list[str] = []
    for r in chosen:
        for s in (*r.reactants, *(p for p, _ in r.products)):
            if s not in species:
                species.append(s)
    return ParsedMechanism(species=species, atoms={s: parsed.atoms[s] for s in species},
                           reactions=chosen, ro2=[s for s in parsed.ro2 if s in species])


def _atoms_text(atoms: dict[str, int]) -> str:
    if not atoms:
        return "IGNORE"
    return " + ".join(f"{n if n > 1 else ''}{el}" for el, n in atoms.items())


def write_eqn(parsed: ParsedMechanism, path, header: list[str]) -> None:
    lines = [f"// {h}" for h in header] + [""]
    lines += [f"{s} = {_atoms_text(parsed.atoms[s])}  ;" for s in parsed.species]
    lines += ["", "#INLINE F90_RCONST", "  USE constants_mcm"]
    if parsed.ro2:
        lines.append("  RO2 = " + " + ".join(f"C(ind_{s})" for s in parsed.ro2))
    else:
        lines.append("  RO2 = 0")
    lines += ["#ENDINLINE", ""]
    for r in parsed.reactions:
        lhs = " + ".join(r.reactants) + (" + hv" if r.photolysis else "")
        rhs = " + ".join(n if c == 1.0 else f"{c:g}{n}" for n, c in r.products) or "PROD"
        lines.append(f"<{r.label}> {lhs} = {rhs} : {r.rate_expr} ;")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")


# ----------------------------------------------------------------------- balance
def atom_imbalance(parsed: ParsedMechanism, r: Reaction) -> dict[str, float]:
    """products minus reactants per element (IGNORE species count as empty)."""
    out: dict[str, float] = defaultdict(float)
    for s in r.reactants:
        for el, n in parsed.atoms.get(s, {}).items():
            out[el] -= n
    for s, c in r.products:
        for el, n in parsed.atoms.get(s, {}).items():
            out[el] += n * c
    return {el: v for el, v in out.items() if abs(v) > 1e-9}


# ----------------------------------------------------------------------- composition
_WS = re.compile(r"\s+")
_FEXP = re.compile(r"(\d\.?\d*)[dD]([-+]?\d)")
_NUM = re.compile(r"(?<![A-Za-z_0-9.])(\d+\.?\d*(?:E[-+]?\d+)?|\.\d+(?:E[-+]?\d+)?)")


def normalise_rate(expr: str) -> str:
    """Whitespace-free, upper-case, D exponents -> E, every literal as repr(float)
    (so 4.50E-14 == 4.5E-14 == 4.5D-14). Not algebraic equivalence: 2*K and K*2 differ."""
    e = _FEXP.sub(r"\1E\2", _WS.sub("", expr)).upper()
    return _NUM.sub(lambda m: repr(float(m.group(1))), e)


def reaction_key(r: Reaction) -> tuple:
    """Identity of a reaction independent of label and of omitted bath species."""
    reac = tuple(sorted(s for s in r.reactants if s not in BATH))
    prod = tuple(sorted((s, round(c, 6)) for s, c in r.products if s not in BATH))
    return (reac, r.photolysis, prod)


@dataclass
class Module:
    name: str                                   # registry id of the module, e.g. "mcm_inorganic"
    parsed: ParsedMechanism
    source: str                                 # "mcm" | "jpl" | "chamber" | ...
    species_map: dict[str, str] = field(default_factory=dict)   # module name -> canonical


@dataclass
class ComposeReport:
    duplicates: list[tuple[str, str, str]] = field(default_factory=list)   # (kept, dropped, key)
    rate_conflicts: list[tuple[str, str]] = field(default_factory=list)
    overlap_pairs: list[tuple[tuple[str, ...], tuple[str, ...]]] = field(default_factory=list)
    atom_conflicts: list[str] = field(default_factory=list)


class ComposeError(ValueError):
    pass


def _mapped(m: Module) -> ParsedMechanism:
    mp = m.species_map
    if not mp:
        return m.parsed
    rn = lambda s: mp.get(s, s)  # noqa: E731
    reactions = [dataclasses.replace(
        r, reactants=tuple(rn(s) for s in r.reactants),
        products=tuple((rn(s), c) for s, c in r.products)) for r in m.parsed.reactions]
    atoms = {rn(s): a for s, a in m.parsed.atoms.items()}
    return ParsedMechanism(species=[rn(s) for s in m.parsed.species], atoms=atoms,
                           reactions=reactions, ro2=[rn(s) for s in m.parsed.ro2])


def compose(modules: list[Module], allow_overlap: dict[tuple[str, ...], str] | None = None
            ) -> tuple[ParsedMechanism, ComposeReport]:
    """Merge modules in order. Rules (each violation raises ComposeError):

    1. a species name means one composition in every module (atoms must agree);
    2. a reaction identical in reactants, products (bath species ignored) and normalised
       rate is a duplicate: the first copy is kept, labels recorded;
    3. same reactants and products but a different rate: conflict (error);
    4. the same reactant set appearing in modules of *different sources* (e.g. MCM and
       JPL both describing OH + SO2, possibly with different product lumping) is an
       overlap; it is an error unless ``allow_overlap[reactants]`` names the module whose
       channels win (the other module's channels for that reactant set are dropped);
    5. RO2 is the union of all modules' RO2 lists;
    6. labels are namespaced ``module:label`` so they never collide.
    """
    allow_overlap = allow_overlap or {}
    report = ComposeReport()
    atoms: dict[str, dict[str, int]] = {}
    owner_of_pair: dict[tuple[str, ...], tuple[str, str]] = {}   # reactants -> (source, module)
    seen: dict[tuple, tuple[str, str]] = {}                       # key -> (label, rate)
    reactions: list[Reaction] = []
    ro2: list[str] = []
    mapped = [(m, _mapped(m)) for m in modules]

    for m, p in mapped:
        for s, a in p.atoms.items():
            if s in atoms and atoms[s] != a and a and atoms[s]:
                report.atom_conflicts.append(f"{s}: {atoms[s]} vs {a} ({m.name})")
            atoms.setdefault(s, a)
    if report.atom_conflicts:
        raise ComposeError(f"species compositions disagree: {report.atom_conflicts[:5]}")

    # rule 4 first: which module owns each reactant set
    for m, p in mapped:
        for r in p.reactions:
            pair = tuple(sorted(s for s in r.reactants if s not in BATH)) + (("hv",) if r.photolysis else ())
            if pair not in owner_of_pair:
                owner_of_pair[pair] = (m.source, m.name)
            elif owner_of_pair[pair][0] != m.source:
                winner = allow_overlap.get(pair)
                if winner is None:
                    report.overlap_pairs.append((pair, (owner_of_pair[pair][1], m.name)))
                else:
                    owner_of_pair[pair] = (dict((mm.name, mm.source) for mm in modules)[winner], winner)
    if report.overlap_pairs:
        raise ComposeError(f"reactant sets described by more than one source: {report.overlap_pairs[:8]}")

    for m, p in mapped:
        for r in p.reactions:
            pair = tuple(sorted(s for s in r.reactants if s not in BATH)) + (("hv",) if r.photolysis else ())
            if pair in allow_overlap and allow_overlap[pair] != m.name:
                continue
            key = reaction_key(r)
            rate = normalise_rate(r.rate_expr)
            label = f"{m.name}:{r.label}"
            if key in seen:
                if seen[key][1] == rate:
                    report.duplicates.append((seen[key][0], label, str(key)))
                    continue
                report.rate_conflicts.append((seen[key][0], label))
                continue
            seen[key] = (label, rate)
            reactions.append(dataclasses.replace(r, label=label))
        for s in p.ro2:
            if s not in ro2:
                ro2.append(s)
    if report.rate_conflicts:
        raise ComposeError(f"same reaction, different rate: {report.rate_conflicts[:5]}")

    species: list[str] = []
    for r in reactions:
        for s in (*r.reactants, *(q for q, _ in r.products)):
            if s not in species:
                species.append(s)
    return (ParsedMechanism(species=species, atoms={s: atoms[s] for s in species},
                            reactions=reactions, ro2=[s for s in ro2 if s in species]), report)


def ro2_consistency(parsed: ParsedMechanism, ro2_dependent_labels: set[str]) -> list[str]:
    """RO2-dependent reactions whose reactant is not in the RO2 sum (should be empty)."""
    ro2 = set(parsed.ro2)
    return [r.label for r in parsed.reactions
            if r.label in ro2_dependent_labels and not all(s in ro2 for s in r.reactants)]


def explain_residual(imb: dict[str, float]) -> dict[str, float] | None:
    """Write a reaction's residual (products - reactants) as omitted CO2, H2O and O2.

    MCM omits O2, H2O and CO2 on either side (they are in the rate or not tracked).
    Returns {"CO2": c, "H2O": w, "O2": o} (integer CO2 and H2O, O2 in halves, as in
    RO2 -> RO + 1/2 O2), or None if the residual
    involves other elements or is not an integer combination of the three.
    """
    if set(imb) - {"C", "H", "O"}:
        return None
    dc, dh, do = imb.get("C", 0.0), imb.get("H", 0.0), imb.get("O", 0.0)
    w = dh / 2.0
    o = (do - 2.0 * dc - w) / 2.0
    if dc > 1e-9:                         # MCM never omits CO2 as a reactant: a C gain is a defect
        return None
    if any(abs(v - round(v)) > 1e-9 for v in (dc, w)) or abs(2 * o - round(2 * o)) > 1e-9:
        return None
    return {"CO2": dc, "H2O": w, "O2": o}
