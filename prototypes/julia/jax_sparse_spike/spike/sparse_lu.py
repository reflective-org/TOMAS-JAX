"""Sparse LU without pivoting for (c I - J), with a fixed elimination order, in JAX.

Symbolic part (numpy, once per mechanism):
  1. order: greedy minimum degree on the pattern of A + A^T;
  2. fill: exact no-pivot fill of the permuted unsymmetric pattern;
  3. schedule: pivots grouped into levels; the pivots of one level do not touch each
     other's rows or columns, so a level is one vectorised update;
  4. triangular solves grouped into levels the same way.
Numeric part (JAX): every level is a few gathers, a sorted segment sum and a scatter
on one flat value vector, unrolled at trace time (the level count sets compile time).

Optionally the last `dense_tail` pivots (the densely coupled inorganic core that minimum
degree leaves to the end) are factored and solved as one dense block with LAPACK.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import jax.scipy.linalg as jsl
import numpy as np
from jax import lax


def min_degree_order(rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    """Greedy minimum degree on the symmetrised pattern (explicit elimination graph)."""
    adj = [set() for _ in range(n)]
    for r, c in zip(rows.tolist(), cols.tolist()):
        if r != c:
            adj[r].add(c)
            adj[c].add(r)
    heap = [(len(adj[i]), i) for i in range(n)]
    heapq.heapify(heap)
    done = np.zeros(n, dtype=bool)
    order = []
    while heap:
        d, i = heapq.heappop(heap)
        if done[i] or d != len(adj[i]):
            continue
        done[i] = True
        order.append(i)
        nb = adj[i]
        for a in nb:
            s = adj[a]
            s.discard(i)
            s |= nb
            s.discard(a)
            heapq.heappush(heap, (len(s), a))
        adj[i] = set()
    assert len(order) == n
    return np.asarray(order, dtype=np.int64)


@dataclass
class Level:
    # step (a): scale the L columns of this level's pivots
    l_pos: np.ndarray
    l_diag: np.ndarray
    # step (b): a_ij -= l_ik u_kj, contributions sorted by target
    u_l: np.ndarray
    u_u: np.ndarray
    u_seg: np.ndarray
    u_tgt: np.ndarray


@dataclass
class SolveLevel:
    rows: np.ndarray        # rows solved in this level (unique, sorted)
    e_pos: np.ndarray       # value positions of the off-diagonal entries of these rows
    e_col: np.ndarray       # their columns (permuted numbering)
    e_seg: np.ndarray       # index into rows


@dataclass
class SparseLUPlan:
    n: int
    perm: np.ndarray            # permuted index p -> original index perm[p]
    iperm: np.ndarray
    nnz: int                    # entries of L + U (diagonal included)
    pos: dict                   # (i, j) permuted -> value position
    diag_pos: np.ndarray        # [n] positions of the diagonal (permuted order)
    levels: list = field(default_factory=list)
    fwd: list = field(default_factory=list)
    bwd: list = field(default_factory=list)
    bwd_diag_only: np.ndarray = None
    flops: int = 0
    n_sparse: int = 0           # pivots handled by the sparse levels (the rest is dense)
    tail_pos: np.ndarray = None  # [m, m] positions of the dense tail block (-1 if absent)


def _sorted_by(key, *arrays):
    o = np.argsort(key, kind="stable")
    return [a[o] for a in arrays]


def symbolic(rows: np.ndarray, cols: np.ndarray, n: int, perm: np.ndarray | None = None,
             dense_tail: int = 0) -> SparseLUPlan:
    """rows, cols: the nonzero pattern of J (original numbering). Diagonal is added."""
    if perm is None:
        perm = min_degree_order(rows, cols, n)
    iperm = np.empty(n, dtype=np.int64)
    iperm[perm] = np.arange(n)
    pr, pc = iperm[rows], iperm[cols]
    urow = [set() for _ in range(n)]    # j > i
    lcol = [set() for _ in range(n)]    # i > j
    for i, j in zip(pr.tolist(), pc.tolist()):
        if i < j:
            urow[i].add(j)
        elif i > j:
            lcol[j].add(i)
    m = int(dense_tail)
    ns = n - m                           # pivots 0..ns-1 sparse; ns..n-1 dense tail
    # fill (exact, no pivoting); the dense tail block is made full
    flops = 0
    for k in range(n):
        lk = lcol[k]
        uk = urow[k]
        flops += len(lk) * (1 + 2 * len(uk))
        for i in lk:
            for j in uk:
                if i < j:
                    urow[i].add(j)
                elif i > j:
                    lcol[j].add(i)
    if m:
        for i in range(ns, n):
            for j in range(ns, n):
                if i < j:
                    urow[i].add(j)
                elif i > j:
                    lcol[j].add(i)
    # value layout: row-major over the permuted matrix
    ent = [(i, i) for i in range(n)]
    for i in range(n):
        ent.extend((i, j) for j in urow[i])
    for j in range(n):
        ent.extend((i, j) for i in lcol[j])
    ent.sort()
    pos = {e: q for q, e in enumerate(ent)}
    plan = SparseLUPlan(n=n, perm=perm, iperm=iperm, nnz=len(ent), pos=pos,
                        diag_pos=np.array([pos[(i, i)] for i in range(n)], dtype=np.int64),
                        flops=flops, n_sparse=ns)
    # factor levels for the sparse pivots
    level = np.zeros(n, dtype=np.int64)
    for k in range(ns):
        lk, uk = lcol[k], urow[k]
        if uk:
            for i in lk:
                level[i] = max(level[i], level[k] + 1)
        if lk:
            for j in uk:
                level[j] = max(level[j], level[k] + 1)
    nlev = int(level[:ns].max()) + 1 if ns else 0
    by_level = [[] for _ in range(nlev)]
    for k in range(ns):
        by_level[level[k]].append(k)
    for piv in by_level:
        l_pos, l_diag, ul, uu, ut = [], [], [], [], []
        for k in piv:
            dk = pos[(k, k)]
            for i in sorted(lcol[k]):
                l_pos.append(pos[(i, k)])
                l_diag.append(dk)
            for i in lcol[k]:
                pik = pos[(i, k)]
                for j in urow[k]:
                    ul.append(pik)
                    uu.append(pos[(k, j)])
                    ut.append(pos[(i, j)])
        l_pos, l_diag = _sorted_by(np.asarray(l_pos, dtype=np.int64),
                                   np.asarray(l_pos, dtype=np.int64),
                                   np.asarray(l_diag, dtype=np.int64))
        ut = np.asarray(ut, dtype=np.int64)
        ul, uu, ut = _sorted_by(ut, np.asarray(ul, dtype=np.int64),
                                np.asarray(uu, dtype=np.int64), ut)
        tgt, seg = np.unique(ut, return_inverse=True)
        plan.levels.append(Level(l_pos, l_diag, ul, uu, seg.astype(np.int64), tgt))
    # dense tail block positions
    if m:
        tp = np.empty((m, m), dtype=np.int64)
        for a in range(m):
            for b in range(m):
                tp[a, b] = pos[(ns + a, ns + b)]
        plan.tail_pos = tp
    # forward solve (unit L), rows 0..ns-1 levelised; the tail rows read the sparse part
    lrow = [[] for _ in range(n)]
    for j in range(n):
        for i in lcol[j]:
            lrow[i].append(j)
    flev = np.zeros(n, dtype=np.int64)
    for i in range(n):
        if lrow[i]:
            flev[i] = 1 + max(flev[j] for j in lrow[i])
    plan.fwd = _solve_levels(n, ns, lrow, flev, pos, tail_rows=m > 0)
    # backward solve (U), rows ns-1..0; the tail block is solved first (dense)
    blev = np.zeros(n, dtype=np.int64)
    urow_l = [sorted(urow[i]) for i in range(n)]
    for i in range(n - 1, -1, -1):
        if i < ns and urow_l[i]:      # tail rows are known (dense solve) before these
            blev[i] = 1 + max(blev[j] for j in urow_l[i])
    plan.bwd = _solve_levels(n, ns, urow_l, blev, pos, tail_rows=m > 0)
    plan.bwd_diag_only = np.array([i for i in range(ns) if not urow_l[i]], dtype=np.int64)
    plan.lrow, plan.urow_l, plan.flev, plan.blev = lrow, urow_l, flev, blev
    return plan


def _solve_levels(n, ns, offd, lev, pos, tail_rows):
    """Group rows with off-diagonal entries by level. With a dense tail, the tail rows'
    entries that reach into the sparse part are put in one extra group (`tail`)."""
    out = []
    rows_sparse = [i for i in range(ns) if offd[i]]
    if rows_sparse:
        levs = sorted({int(lev[i]) for i in rows_sparse})
        for lv in levs:
            rows = [i for i in rows_sparse if lev[i] == lv]
            out.append(_solve_group(rows, offd, pos, only_cols_below=None))
    if tail_rows:
        rows = [i for i in range(ns, n)]
        g = _solve_group(rows, offd, pos, only_cols_below=ns)
        g.rows = np.asarray(rows, dtype=np.int64)
        out.append(("tail", g))
    return out


def _solve_group(rows, offd, pos, only_cols_below):
    e_pos, e_col, e_seg = [], [], []
    for r, i in enumerate(rows):
        for j in offd[i]:
            if only_cols_below is not None and j >= only_cols_below:
                continue
            e_pos.append(pos[(i, j)])
            e_col.append(j)
            e_seg.append(r)
    return SolveLevel(np.asarray(rows, dtype=np.int64), np.asarray(e_pos, dtype=np.int64),
                      np.asarray(e_col, dtype=np.int64), np.asarray(e_seg, dtype=np.int64))


# ---------------------------------------------------------------------------------------
# numeric part (JAX)

def _i32(a):
    return jnp.asarray(a, dtype=jnp.int32)


def make_factor(plan: SparseLUPlan):
    """Returns factor(vals) -> (lu_vals, tail_lu, tail_piv); vals in the plan's layout."""
    levels = [(_i32(L.l_pos), _i32(L.l_diag), _i32(L.u_l), _i32(L.u_u), _i32(L.u_seg),
               _i32(L.u_tgt), int(L.u_tgt.size)) for L in plan.levels]
    tail = plan.tail_pos is not None
    tail_pos = _i32(plan.tail_pos) if tail else None

    def factor(vals):
        for l_pos, l_diag, u_l, u_u, u_seg, u_tgt, nt in levels:
            if l_pos.size:
                vals = vals.at[l_pos].set(vals[l_pos] / vals[l_diag], unique_indices=True,
                                          indices_are_sorted=True)
            if nt:
                s = jax.ops.segment_sum(vals[u_l] * vals[u_u], u_seg, num_segments=nt,
                                        indices_are_sorted=True)
                vals = vals.at[u_tgt].add(-s, unique_indices=True, indices_are_sorted=True)
        if tail:
            lu, piv = jsl.lu_factor(vals[tail_pos])
            return vals, lu, piv
        return vals, None, None

    return factor


def make_solve(plan: SparseLUPlan):
    """Returns solve(fac, b) for fac = factor(vals); b and x in the original numbering."""
    perm, iperm = _i32(plan.perm), _i32(plan.iperm)
    diag = _i32(plan.diag_pos)
    ns, n = plan.n_sparse, plan.n

    def conv(group):
        if isinstance(group, tuple):
            g = group[1]
            return ("tail", _i32(g.rows), _i32(g.e_pos), _i32(g.e_col), _i32(g.e_seg),
                    int(g.rows.size))
        return ("lev", _i32(group.rows), _i32(group.e_pos), _i32(group.e_col),
                _i32(group.e_seg), int(group.rows.size))

    fwd = [conv(g) for g in plan.fwd]
    bwd = [conv(g) for g in plan.bwd]
    tail = plan.tail_pos is not None

    tail_grp = [g for g in fwd if g[0] == "tail"]
    fwd_lev = [g for g in fwd if g[0] == "lev"]
    bwd_lev = [g for g in bwd if g[0] == "lev"]
    diag_only = _i32(plan.bwd_diag_only)

    def solve(fac, b):
        vals, tlu, tpiv = fac
        x = b[perm]
        # forward, L y = b (unit diagonal): sparse rows by level
        for _, rows, e_pos, e_col, e_seg, nrow in fwd_lev:
            s = jax.ops.segment_sum(vals[e_pos] * x[e_col], e_seg, num_segments=nrow,
                                    indices_are_sorted=True)
            x = x.at[rows].add(-s, unique_indices=True, indices_are_sorted=True)
        if tail:
            # tail rows: subtract L21 y1, then the dense Schur-complement solve
            for _, rows, e_pos, e_col, e_seg, nrow in tail_grp:
                if e_pos.size:
                    s = jax.ops.segment_sum(vals[e_pos] * x[e_col], e_seg, num_segments=nrow,
                                            indices_are_sorted=True)
                    x = x.at[rows].add(-s, unique_indices=True, indices_are_sorted=True)
            x = x.at[ns:].set(jsl.lu_solve((tlu, tpiv), x[ns:]))
        # backward, U x = y: rows without off-diagonal entries, then the levels
        inv_d = 1.0 / vals[diag[:ns]]
        if diag_only.size:
            x = x.at[diag_only].multiply(inv_d[diag_only], unique_indices=True,
                                         indices_are_sorted=True)
        for _, rows, e_pos, e_col, e_seg, nrow in bwd_lev:
            s = jax.ops.segment_sum(vals[e_pos] * x[e_col], e_seg, num_segments=nrow,
                                    indices_are_sorted=True)
            x = x.at[rows].set((x[rows] - s) * inv_d[rows], unique_indices=True,
                               indices_are_sorted=True)
        return x[iperm]

    return solve


# ---------------------------------------------------------------------------------------
# solve v2: rows of a level are contiguous (separate numberings for the forward and the
# backward phase), so a level is one fused gather-multiply-reduce plus one in-place
# dynamic_update_slice. The L and U values are gathered into solve order once per
# factorisation (prepare), in padded [rows, width] blocks where padding is cheap.

def _phase_blocks(rows_by_level, offd, pos, colmap, nnz, pad_ratio=2.0):
    """rows_by_level: list of row lists (permuted numbering), in processing order.
    Returns (blocks, gather) where gather indexes the extended value vector (nnz -> 0)."""
    blocks, gather, off = [], [], 0
    for rows in rows_by_level:
        cnt = [len(offd[i]) for i in rows]
        ent = sum(cnt)
        w = max(cnt) if cnt else 0
        if ent == 0:
            blocks.append(("none", len(rows)))
            continue
        if len(rows) * w <= pad_ratio * ent + 64:
            vidx = np.full((len(rows), w), nnz, dtype=np.int64)
            cidx = np.zeros((len(rows), w), dtype=np.int64)
            for r, i in enumerate(rows):
                js = offd[i]
                vidx[r, :len(js)] = [pos[(i, j)] for j in js]
                cidx[r, :len(js)] = [colmap[j] for j in js]
            gather.append(vidx.ravel())
            blocks.append(("pad", len(rows), w, off, cidx))
            off += vidx.size
        else:
            vidx, cidx, seg = [], [], []
            for r, i in enumerate(rows):
                for j in offd[i]:
                    vidx.append(pos[(i, j)])
                    cidx.append(colmap[j])
                    seg.append(r)
            gather.append(np.asarray(vidx, dtype=np.int64))
            blocks.append(("seg", len(rows), len(vidx), off, np.asarray(cidx, dtype=np.int64),
                           np.asarray(seg, dtype=np.int64)))
            off += len(vidx)
    g = np.concatenate(gather) if gather else np.zeros(0, dtype=np.int64)
    return blocks, g


def make_solve_v2(plan: SparseLUPlan, lrow, urow_l, flev, blev):
    """Returns (prepare, solve): fac2 = prepare(factor(vals)); x = solve(fac2, b)."""
    n, ns, nnz, pos = plan.n, plan.n_sparse, plan.nnz, plan.pos
    tail = plan.tail_pos is not None
    # forward numbering: sparse rows by forward level, then the tail rows
    sp_rows = list(range(ns))
    F = sorted(sp_rows, key=lambda i: (flev[i], i)) + list(range(ns, n))
    posF = np.empty(n, dtype=np.int64)
    posF[F] = np.arange(n)
    f_levels = []
    for lv in sorted({int(flev[i]) for i in sp_rows}):
        f_levels.append([i for i in F[:ns] if flev[i] == lv])
    f_starts = np.cumsum([0] + [len(r) for r in f_levels])
    lsparse = [[j for j in lrow[i] if j < ns] for i in range(n)]
    f_blocks, f_gather = _phase_blocks(f_levels + ([list(range(ns, n))] if tail else []),
                                       lsparse, pos, posF, nnz)
    # backward numbering: tail rows first (known), then sparse rows by backward level
    B = list(range(ns, n)) + sorted(sp_rows, key=lambda i: (blev[i], i))
    posB = np.empty(n, dtype=np.int64)
    posB[B] = np.arange(n)
    b_levels = []
    for lv in sorted({int(blev[i]) for i in sp_rows}):
        b_levels.append([i for i in B[n - ns:] if blev[i] == lv])
    b_starts = (n - ns) + np.cumsum([0] + [len(r) for r in b_levels])
    b_blocks, b_gather = _phase_blocks(b_levels, urow_l, pos, posB, nnz)
    diag_B = plan.diag_pos[np.asarray(B[n - ns:], dtype=np.int64)]

    in_F = _i32(plan.perm[np.asarray(F)])          # b (original) -> x_F
    F_to_B = _i32(posF[np.asarray(B)])              # x_F -> x_B
    B_to_out = _i32(posB[plan.iperm])               # x_B -> x (original)
    fg, bg, dg = _i32(f_gather), _i32(b_gather), _i32(diag_B)
    conv = []
    for blocks in (f_blocks, b_blocks):
        conv.append([(b[0],) + tuple(_i32(v) if isinstance(v, np.ndarray) else v for v in b[1:])
                     for b in blocks])
    f_blocks, b_blocks = conv

    def prepare(fac):
        vals, tlu, tpiv = fac
        ve = jnp.concatenate([vals, jnp.zeros((1,), vals.dtype)])
        return ve[fg], ve[bg], 1.0 / vals[dg], tlu, tpiv

    def _level(x, vv, blk, start):
        kind = blk[0]
        if kind == "none":
            return x, None
        if kind == "pad":
            _, nrow, w, off, cidx = blk
            v = lax.slice(vv, (off,), (off + nrow * w,)).reshape(nrow, w)
            s = jnp.sum(v * x[cidx], axis=1)
        else:
            _, nrow, ne, off, cidx, seg = blk
            v = lax.slice(vv, (off,), (off + ne,))
            s = jax.ops.segment_sum(v * x[cidx], seg, num_segments=nrow,
                                    indices_are_sorted=True)
        return x, s

    def solve(fac2, b):
        vf, vb, invd, tlu, tpiv = fac2
        x = b[in_F]
        nf = len(f_levels)
        for q, blk in enumerate(f_blocks):
            start = int(f_starts[q]) if q < nf else ns
            x, s = _level(x, vf, blk, start)
            if s is not None:
                cur = lax.slice(x, (start,), (start + s.shape[0],))
                x = lax.dynamic_update_slice(x, cur - s, (start,))
        if tail:
            x = lax.dynamic_update_slice(x, jsl.lu_solve((tlu, tpiv), x[ns:]), (ns,))
        x = x[F_to_B]
        for q, blk in enumerate(b_blocks):
            start = int(b_starts[q])
            nrow = blk[1]
            x, s = _level(x, vb, blk, start)
            cur = lax.slice(x, (start,), (start + nrow,))
            d = lax.slice(invd, (start - (n - ns),), (start - (n - ns) + nrow,))
            new = (cur - s) * d if s is not None else cur * d
            x = lax.dynamic_update_slice(x, new, (start,))
        return x[B_to_out]

    return prepare, solve


# ---------------------------------------------------------------------------------------
# factor v2: L is left unscaled during the elimination; each update uses
# a_ij -= a_ik a_kj / a_kk with the three values final at pivot k's level, and the
# accumulation is one scatter-add with repeated (sorted) targets. L is scaled by 1/a_kk
# afterwards, in the gather that puts the values into solve order.

def make_factor_v2(plan: SparseLUPlan, dup_scatter: bool = True):
    """factor(vals) -> (vals with U final and L unscaled, None, None, l_scale) where
    l_scale[q] = 1/a_kk for an L entry at position q (1 elsewhere)."""
    assert plan.tail_pos is None, "factor v2 has no dense tail"
    lev = []
    for L in plan.levels:
        if L.u_tgt.size == 0:
            continue
        piv_of = {}
        # pivot diagonal position for each contribution: the column k of u_l
        lev.append(L)
    n, nnz = plan.n, plan.nnz
    # pivot (diag position) of every L entry, for the final scaling
    col_of_pos = np.zeros(nnz, dtype=np.int64)
    is_l = np.zeros(nnz, dtype=bool)
    for (i, j), q in plan.pos.items():
        col_of_pos[q] = j
        is_l[q] = i > j
    l_pos = np.nonzero(is_l)[0]
    l_diag = plan.diag_pos[col_of_pos[l_pos]]
    levels = []
    for L in lev:
        # diag position of the pivot of each contribution = diag of u_l's column
        k_diag = plan.diag_pos[col_of_pos[L.u_l]]
        if dup_scatter:
            levels.append((_i32(L.u_l), _i32(L.u_u), _i32(k_diag), _i32(L.u_tgt[L.u_seg]), None, 0))
        else:
            levels.append((_i32(L.u_l), _i32(L.u_u), _i32(k_diag), _i32(L.u_tgt), _i32(L.u_seg),
                           int(L.u_tgt.size)))
    l_pos_j, l_diag_j = _i32(l_pos), _i32(l_diag)

    def factor(vals):
        for u_l, u_u, k_diag, tgt, seg, nt in levels:
            c = vals[u_l] * vals[u_u] / vals[k_diag]
            if seg is None:
                vals = vals.at[tgt].add(-c, indices_are_sorted=True)
            else:
                s = jax.ops.segment_sum(c, seg, num_segments=nt, indices_are_sorted=True)
                vals = vals.at[tgt].add(-s, unique_indices=True, indices_are_sorted=True)
        return vals.at[l_pos_j].divide(vals[l_diag_j], unique_indices=True,
                                       indices_are_sorted=True), None, None

    return factor
