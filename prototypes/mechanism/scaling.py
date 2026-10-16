"""Gas-solver cost against mechanism size: Jacobian sparsity, LU fill, and timings of the
pieces of one stiff step, dense (today) and sparse (the Pro prototype's levelised LU).

Mechanisms: the 29-species inorganic module, the 725-species limonene file, and
synthetic "multi-VOC" mechanisms made of N copies of limonene's organic chemistry
(species with 4 or more C renamed per copy) sharing the inorganic and C1-C3 chemistry.
A synthetic copy has the same rate constants, so its stiffness is like limonene's. Real
multi-VOC MCM mechanisms share more intermediates, so expect more fill than here.

Usage: python scaling.py <name> <chemlib_dir> <out.json>
  name: inorganic | c3 | limonene | copies<N> (N copies of the C>=4 chemistry)
  chemlib_dir: prototypes/pro/perf of the Pro worktree (read-only import of chemlib.py)
"""
from __future__ import annotations

import dataclasses
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
name, chemlib_dir, out_path = sys.argv[1], sys.argv[2], sys.argv[3]
sys.path.insert(0, chemlib_dir)

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import jax.scipy.linalg as jsl  # noqa: E402

import chemlib  # noqa: E402
from tomas_jax.gaschem.config import LIMONENE_EQN  # noqa: E402
from tomas_jax.gaschem.kpp_parser import ParsedMechanism, parse_eqn  # noqa: E402
from tomas_jax.gaschem.mechanism import Mechanism, RateModel  # noqa: E402

ENV = {"TEMP": 298.15, "M": 2.46e19, "O2": 0.2095 * 2.46e19, "N2": 0.7809 * 2.46e19,
       "H2O": 3.9e17}


def copies(parsed: ParsedMechanism, n: int) -> tuple[ParsedMechanism, np.ndarray]:
    """n copies of the C>=4 chemistry; returns the mechanism and, per reaction, the index
    of the original reaction (to reuse its rate constant)."""
    big = {s for s, a in parsed.atoms.items() if a.get("C", 0) >= 4}
    reactions, origin, atoms, ro2 = [], [], dict(parsed.atoms), []
    seen = set()
    for c in range(n):
        ren = (lambda s, c=c: f"{s}_{c}" if (s in big and c > 0) else s)
        for i, r in enumerate(parsed.reactions):
            involves_big = any(s in big for s in (*r.reactants, *(p for p, _ in r.products)))
            if c > 0 and not involves_big:
                continue                         # shared chemistry appears once
            rr = dataclasses.replace(r, label=f"{r.label}_{c}",
                                     reactants=tuple(map(ren, r.reactants)),
                                     products=tuple((ren(p), k) for p, k in r.products))
            reactions.append(rr)
            origin.append(i)
        for s in parsed.ro2:
            t = ren(s)
            if t not in seen:
                seen.add(t)
                ro2.append(t)
        for s in big:
            atoms[ren(s)] = parsed.atoms[s]
    species = []
    sp_seen = set()
    for r in reactions:
        for s in (*r.reactants, *(p for p, _ in r.products)):
            if s not in sp_seen:
                sp_seen.add(s)
                species.append(s)
    return (ParsedMechanism(species=species, atoms={s: atoms[s] for s in species},
                            reactions=reactions, ro2=ro2), np.asarray(origin))


def best(fn, *args, reps=5):
    jax.block_until_ready(fn(*args))
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        jax.block_until_ready(fn(*args))
        ts.append(time.perf_counter() - t0)
    return min(ts)


def compile_time(fn, *args):
    t0 = time.perf_counter()
    jax.block_until_ready(fn(*args))
    return time.perf_counter() - t0


lim = parse_eqn(LIMONENE_EQN)
k_lim, _, _ = RateModel(lim, external=("34",)).at_env(ENV)
if name == "inorganic":
    parsed = parse_eqn(HERE / "modules" / "mcm_inorganic_c1_sulfur.eqn")
    k = RateModel(parsed).at_env(ENV)[0]
elif name == "c3":
    sys.path.insert(0, str(HERE))
    from mechlib import extract_closure
    seeds = {s for s, a in lim.atoms.items() if a.get("C", 0) <= 3}
    parsed = extract_closure(lim, seeds, drop_labels=frozenset({"34"}))
    k = RateModel(parsed).at_env(ENV)[0]
elif name == "limonene":
    parsed, k = lim, k_lim
else:
    parsed, origin = copies(lim, int(name.replace("copies", "")))
    k = k_lim[origin]

mech = Mechanism.build(parsed)
m = chemlib.Mech(mech.n_species, np.asarray(mech.reactant_idx), np.asarray(mech.prod_rxn),
                 np.asarray(mech.prod_sp), np.asarray(mech.prod_coef), np.asarray(mech.loss_rxn),
                 np.asarray(mech.loss_sp), list(parsed.species))
n = m.n
res = {"name": name, "species": n, "reactions": len(parsed.reactions), "ro2": len(parsed.ro2)}
rng = np.random.default_rng(1)
y = jnp.asarray(10.0 ** rng.uniform(4, 12, n))
kk = jnp.asarray(np.maximum(k, 1e-30))
h = 10.0

rhs = jax.jit(chemlib.make_rhs(m))
res["t_rhs_ms"] = 1e3 * best(rhs, y, kk)

pat = chemlib.jac_pattern(m)
res["jac_nnz"] = int(pat["nnz"])
res["jac_density"] = pat["nnz"] / n**2

# dense path (today: autodiff Jacobian + dense LU)
if n <= 6000:
    jfwd = jax.jit(jax.jacfwd(chemlib.make_rhs(m)))
    res["compile_jacfwd_s"] = compile_time(jfwd, y, kk)
    res["t_jacfwd_ms"] = 1e3 * best(jfwd, y, kk, reps=3)
    J = jfwd(y, kk)
    A = jnp.eye(n) / h - J
    lu = jax.jit(jsl.lu_factor)
    res["compile_dense_lu_s"] = compile_time(lu, A)
    res["t_dense_lu_ms"] = 1e3 * best(lu, A, reps=3)
    fac = lu(A)
    sol = jax.jit(lambda f, b: jsl.lu_solve(f, b))
    res["t_dense_solve_ms"] = 1e3 * best(sol, fac, y, reps=3)
    res["dense_matrix_mb"] = n * n * 8 / 1e6
    del J, A, fac

# sparse path (Pro prototype: analytic Jacobian on the pattern + levelised LU)
t0 = time.perf_counter()
perm = chemlib.min_degree_order(pat["prow"], pat["pcol"], n)
res["t_ordering_s"] = time.perf_counter() - t0
t0 = time.perf_counter()
slu = chemlib.SparseLU(pat["prow"], pat["pcol"], n, perm)
res["t_symbolic_s"] = time.perf_counter() - t0
res["lu_nnz"] = int(slu.nnz_lu)
res["lu_fill_ratio"] = slu.nnz_lu / pat["nnz"]
res["lu_flops"] = int(slu.flops)
res["factor_levels"] = int(slu.n_levels)
res["solve_levels"] = len(slu.fwd) + len(slu.bwd)
jv = jax.jit(chemlib.make_jac_values(m, pat))
res["t_jacvals_ms"] = 1e3 * best(jv, y, kk)
vals = jv(y, kk)
fac_s = jax.jit(lambda v: slu.factor(v, 1.0 / h))
res["compile_sparse_factor_s"] = compile_time(fac_s, vals)
res["t_sparse_factor_ms"] = 1e3 * best(fac_s, vals)
lu_v = fac_s(vals)
sol_s = jax.jit(lambda v, b: slu.solve(v, b))
res["compile_sparse_solve_s"] = compile_time(sol_s, lu_v, y)
res["t_sparse_solve_ms"] = 1e3 * best(sol_s, lu_v, y)
# correctness of the sparse LU against a dense solve (when the dense one was built)
if n <= 6000:
    Jd = chemlib.make_dense_jac(m, pat)(y, kk)
    xd = jnp.linalg.solve(jnp.eye(n) / h - Jd, y)
    res["sparse_vs_dense_solve_maxrel"] = float(jnp.max(jnp.abs(sol_s(lu_v, y) - xd)
                                                        / jnp.maximum(jnp.abs(xd), 1e-300)))
res["jax"] = jax.__version__
Path(out_path).write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
