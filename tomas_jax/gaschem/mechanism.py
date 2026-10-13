"""Turn a parsed KPP mechanism into arrays for a vectorised JAX right-hand side.

Rate constants follow KPP ``UPDATE_RCONST`` semantics: they are evaluated once per
coupling interval and held fixed while the stiff solver integrates that interval.
Every MCM rate expression is a function of the environment (TEMP, M, O2, N2, H2O),
the peroxy-radical sum RO2 and the photolysis frequencies J. :meth:`RateModel.at_env`
decomposes them, at a given environment, into

    k = k_thermal * (RO2 if ro2_dependent else 1) + J_coef @ J

and checks that decomposition at a random (RO2, J) point, so the per-interval update
in JAX is a multiply-add. Reactions listed in ``external`` get their k from the caller
instead (e.g. heterogeneous uptake computed from the aerosol).

RHS (gather/scatter, no dense [n_rxn x n_species] work):
    r_j = k_j * y[a_j] * y[b_j]          (missing reactant -> index of a constant 1.0)
    dy/dt = sum_products coef * r  -  sum_reactants r
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

# float64 enforced by tomas_jax.core.config (imported by the tomas_jax package)
from .kpp_parser import ParsedMechanism
from .mcm_rates import MCM_J_PARAMS, complex_rates

_ENV_NAMES = ("TEMP", "M", "O2", "N2", "H2O", "RO2")
_IDENT_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_JCALL_RE = re.compile(r"\bJ\(\s*(J_[A-Za-z0-9_]+)\s*\)")
_FORTRAN_EXP_RE = re.compile(r"(\d\.?)[dD]([-+]?\d)")
_NUMBER_EXP_RE = re.compile(r"\d\.?[eE][-+]?\d+")


def _translate(expr: str, k_names: set[str]) -> tuple[str, list[str]]:
    """Fortran rate expression -> numpy expression; returns (code, J names used)."""
    py = _FORTRAN_EXP_RE.sub(r"\1e\2", expr)
    j_used = _JCALL_RE.findall(py)
    py = _JCALL_RE.sub(lambda m: f'J["{m.group(1)}"]', py)
    py = re.sub(r"\bEXP\(", "exp(", py)
    py = re.sub(r"\bLOG10\(", "log10(", py)
    allowed = set(_ENV_NAMES) | k_names | {"exp", "log10", "J"} | set(j_used)
    bare = _NUMBER_EXP_RE.sub("0", re.sub(r'"[^"]*"', "", py))
    unknown = sorted(set(_IDENT_RE.findall(bare)) - allowed)
    if unknown:
        raise ValueError(f"rate expression {expr!r} uses unknown names {unknown}")
    return py, j_used


class RateModel:
    """Numpy evaluator for all rate expressions of a mechanism (built once)."""

    def __init__(self, parsed: ParsedMechanism, external: tuple[str, ...] = ()):
        k_names = set(complex_rates(298.0, 2.46e19, 5.2e18, 1.9e19, 4e17))
        self.labels = [r.label for r in parsed.reactions]
        missing = [lab for lab in external if lab not in self.labels]
        if missing:
            raise ValueError(f"external reaction labels not in mechanism: {missing}")
        self.external_idx = np.array([self.labels.index(lab) for lab in external], dtype=int)

        codes, j_names = [], set()
        for i, rxn in enumerate(parsed.reactions):
            code, j_used = _translate(rxn.rate_expr, k_names)
            j_names.update(j_used)
            codes.append("0.0" if i in set(self.external_idx.tolist()) else code)
        self.j_names = sorted(j_names)
        unknown_j = [j for j in self.j_names if j not in MCM_J_PARAMS]
        if unknown_j:
            raise ValueError(f"photolysis names without MCM definition: {unknown_j}")
        unpack = "".join(f"    {n} = K['{n}']\n" for n in sorted(k_names))
        fn_src = (f"def _rates(TEMP, M, O2, N2, H2O, RO2, J, K):\n{unpack}"
                  f"    return [{', '.join(codes)}]\n")
        namespace = {"exp": np.exp, "log10": np.log10}
        exec(compile(fn_src, "<mechanism rates>", "exec"), {"__builtins__": {}, **namespace},
             namespace)
        self._rates = namespace["_rates"]

    def _eval(self, env: dict, ro2: float, jvals: dict) -> np.ndarray:
        K = complex_rates(env["TEMP"], env["M"], env["O2"], env["N2"], env["H2O"])
        vals = self._rates(env["TEMP"], env["M"], env["O2"], env["N2"], env["H2O"],
                           ro2, jvals, K)
        return np.asarray(vals, dtype=np.float64)

    def at_env(self, env: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """(k_thermal [nr], ro2_dependent [nr] bool, j_coef [nr, nJ]) at this environment."""
        zero_j = {j: 0.0 for j in self.j_names}
        k1 = self._eval(env, 1.0, zero_j)
        k0 = self._eval(env, 0.0, zero_j)
        ro2_dep = (k0 == 0.0) & (k1 != 0.0)
        k_thermal = k1
        j_coef = np.zeros((len(self.labels), len(self.j_names)))
        for col, j in enumerate(self.j_names):
            unit = dict(zero_j, **{j: 1.0})
            j_coef[:, col] = self._eval(env, 1.0, unit) - k1

        # verify the linear decomposition at a random point
        rng = np.random.default_rng(0)
        ro2_test = 10.0 ** rng.uniform(6, 10)
        j_test = {j: 10.0 ** rng.uniform(-7, -2) for j in self.j_names}
        exact = self._eval(env, ro2_test, j_test)
        approx = (k_thermal * np.where(ro2_dep, ro2_test, 1.0)
                  + j_coef @ np.array([j_test[j] for j in self.j_names]))
        bad = ~np.isclose(exact, approx, rtol=1e-10, atol=0.0)
        if bad.any():
            labels = [self.labels[i] for i in np.flatnonzero(bad)[:5]]
            raise ValueError(f"rate expressions not of the form a*RO2^0|1 + c.J: {labels}")
        return k_thermal, ro2_dep, j_coef


class Mechanism(eqx.Module):
    """Static index arrays for the vectorised RHS (an equinox pytree)."""
    n_species: int = eqx.field(static=True)
    reactant_idx: jax.Array      # [nr, 2] int; n_species = padding -> constant 1.0
    prod_rxn: jax.Array          # [nP] reaction index of each product entry
    prod_sp: jax.Array           # [nP] species index
    prod_coef: jax.Array         # [nP] stoichiometric coefficient
    loss_rxn: jax.Array          # [nL] one entry per reactant occurrence
    loss_sp: jax.Array           # [nL]
    ro2_idx: jax.Array           # [nRO2] species in the RO2 sum

    @staticmethod
    def build(parsed: ParsedMechanism) -> "Mechanism":
        index = {s: i for i, s in enumerate(parsed.species)}
        n = len(parsed.species)
        reactant_idx = np.full((len(parsed.reactions), 2), n, dtype=np.int32)
        prod_rxn, prod_sp, prod_coef, loss_rxn, loss_sp = [], [], [], [], []
        for j, rxn in enumerate(parsed.reactions):
            if len(rxn.reactants) > 2:
                raise ValueError(f"reaction <{rxn.label}> has >2 reactants")
            for slot, name in enumerate(rxn.reactants):
                reactant_idx[j, slot] = index[name]
                loss_rxn.append(j)
                loss_sp.append(index[name])
            for name, coef in rxn.products:
                prod_rxn.append(j)
                prod_sp.append(index[name])
                prod_coef.append(coef)
        return Mechanism(
            n_species=n,
            reactant_idx=jnp.asarray(reactant_idx),
            prod_rxn=jnp.asarray(prod_rxn, dtype=jnp.int32),
            prod_sp=jnp.asarray(prod_sp, dtype=jnp.int32),
            prod_coef=jnp.asarray(prod_coef, dtype=jnp.float64),
            loss_rxn=jnp.asarray(loss_rxn, dtype=jnp.int32),
            loss_sp=jnp.asarray(loss_sp, dtype=jnp.int32),
            ro2_idx=jnp.asarray([index[s] for s in parsed.ro2], dtype=jnp.int32),
        )

    def rates(self, y: jax.Array, k: jax.Array) -> jax.Array:
        y_ext = jnp.concatenate([y, jnp.ones((1,), y.dtype)])
        return k * y_ext[self.reactant_idx[:, 0]] * y_ext[self.reactant_idx[:, 1]]

    def rhs(self, y: jax.Array, k: jax.Array) -> jax.Array:
        r = self.rates(y, k)
        gain = jax.ops.segment_sum(self.prod_coef * r[self.prod_rxn], self.prod_sp,
                                   num_segments=self.n_species)
        loss = jax.ops.segment_sum(r[self.loss_rxn], self.loss_sp,
                                   num_segments=self.n_species)
        return gain - loss

    def ro2(self, y: jax.Array) -> jax.Array:
        return jnp.sum(y[self.ro2_idx])


@dataclass(frozen=True)
class EnvRates:
    """Rate pieces at one environment, as JAX arrays (see RateModel.at_env)."""
    k_thermal: jax.Array
    ro2_dependent: jax.Array
    j_coef: jax.Array

    def k(self, ro2: jax.Array, j: jax.Array) -> jax.Array:
        return self.k_thermal * jnp.where(self.ro2_dependent, ro2, 1.0) + self.j_coef @ j
