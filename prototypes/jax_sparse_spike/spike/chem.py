"""JAX right-hand side, analytic Jacobian (written straight into the sparse LU layout) and
interval rate constants for a benchmark problem (bench/scripts/bench_common.py)."""
from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from .paths import BENCH  # noqa: F401  (puts bench/scripts on sys.path)
from bench_common import NumpyChem, Problem  # noqa: E402

from .ordering import markowitz_order  # noqa: E402
from .sparse_lu import SparseLUPlan, min_degree_order, symbolic  # noqa: E402


def _i32(a):
    return jnp.asarray(np.asarray(a), dtype=jnp.int32)


@dataclass
class JaxChem:
    n: int
    rhs: callable          # rhs(y, k) -> dy/dt
    neg_jac_lu: callable   # neg_jac_lu(y, k) -> -J values in the LU layout (zeros at fill)
    diag_mask: jax.Array   # 1 at the diagonal positions of the LU layout, else 0
    rate_constants: callable   # rate_constants(y_start, j_row) -> k
    post_interval: callable    # post_interval(y_end_clipped) -> next start state
    plan: SparseLUPlan
    numpy_chem: NumpyChem


def build(P: Problem, dense_tail: int = 0, perm: np.ndarray | None = None,
          order: str = "markowitz") -> JaxChem:
    m = P.mech
    n, nr = m.n, m.nr
    nc = NumpyChem(m)
    reac = np.asarray(m.reactants)               # padded with n (constant 1.0)
    # net stoichiometry as one sorted list of (species, reaction, coefficient)
    sp_ = np.concatenate([m.prod_sp, m.loss_sp])
    rx_ = np.concatenate([m.prod_rxn, m.loss_rxn])
    cf_ = np.concatenate([m.prod_coef, -np.ones(m.loss_sp.size)])
    o = np.lexsort((rx_, sp_))
    sp_, rx_, cf_ = sp_[o], rx_[o], cf_[o]
    r0, r1 = _i32(reac[:, 0]), _i32(reac[:, 1])
    j_sp, j_rx, j_cf = _i32(sp_), _i32(rx_), jnp.asarray(cf_)

    def rates(y, k):
        ye = jnp.concatenate([y, jnp.ones((1,), y.dtype)])
        return k * ye[r0] * ye[r1]

    def rhs(y, k):
        r = rates(y, k)
        return jax.ops.segment_sum(j_cf * r[j_rx], j_sp, num_segments=n,
                                   indices_are_sorted=True)

    # Jacobian pattern (original numbering) -> sparse LU plan
    key = np.unique(nc.e_col * n + nc.e_row)
    rows, cols = key % n, key // n
    if perm is None:
        perm = (markowitz_order(rows, cols, n) if order == "markowitz"
                else min_degree_order(rows, cols, n))
    plan = symbolic(rows, cols, n, perm=perm, dense_tail=dense_tail)
    # each Jacobian term e: d f_row / d y_col += nu * k[rxn] * y[other]
    e_lu = np.array([plan.pos[(int(plan.iperm[r]), int(plan.iperm[c]))]
                     for r, c in zip(nc.e_row, nc.e_col)], dtype=np.int64)
    o = np.argsort(e_lu, kind="stable")
    e_lu_s, e_rxn, e_oth, e_nu = (_i32(e_lu[o]), _i32(nc.e_rxn[o]), _i32(nc.e_other[o]),
                                  jnp.asarray(-nc.e_nu[o]))
    nnz = plan.nnz

    def neg_jac_lu(y, k):
        ye = jnp.concatenate([y, jnp.ones((1,), y.dtype)])
        return jax.ops.segment_sum(e_nu * k[e_rxn] * ye[e_oth], e_lu_s, num_segments=nnz,
                                   indices_are_sorted=True)

    diag_mask = jnp.zeros(nnz).at[plan.diag_pos].set(1.0)

    # rate constants of an interval
    k_th = jnp.asarray(P.k_thermal)
    dep = jnp.asarray(P.ro2_dependent)
    jc = P.j_coef.tocoo()
    jc_r, jc_c, jc_v = _i32(jc.row), _i32(jc.col), jnp.asarray(jc.data)
    ro2_idx = _i32(m.ro2)
    ext_idx = _i32([e[0] for e in P.external])
    ext_val = jnp.asarray([e[1] for e in P.external], dtype=jnp.float64)

    def rate_constants(y_start, j_row):
        ro2 = jnp.sum(y_start[ro2_idx])
        kj = jax.ops.segment_sum(jc_v * j_row[jc_c], jc_r, num_segments=nr)
        k = k_th * jnp.where(dep, ro2, 1.0) + kj
        if ext_idx.size:
            k = k.at[ext_idx].set(ext_val)
        return k

    rz, ri = _i32(P.reset_zero), _i32(P.reset_initial)
    y0_ri = jnp.asarray(P.y0[P.reset_initial])

    def post_interval(y_end):
        y = y_end
        if rz.size:
            y = y.at[rz].set(0.0)
        if ri.size:
            y = y.at[ri].set(y0_ri)
        return y

    return JaxChem(n=n, rhs=rhs, neg_jac_lu=neg_jac_lu, diag_mask=diag_mask,
                   rate_constants=rate_constants, post_interval=post_interval, plan=plan,
                   numpy_chem=nc)
