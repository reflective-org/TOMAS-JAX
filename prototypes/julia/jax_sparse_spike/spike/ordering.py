"""Markowitz elimination order for the sparse LU (diagonal pivots only)."""
from __future__ import annotations

import heapq

import numpy as np


def markowitz_order(rows: np.ndarray, cols: np.ndarray, n: int) -> np.ndarray:
    """Greedy Markowitz on the unsymmetric pattern: pick the diagonal pivot with the
    smallest (r - 1)(c - 1), r and c the active row and column counts (as KPP does)."""
    R = [set() for _ in range(n)]
    Cc = [set() for _ in range(n)]
    for i, j in zip(rows.tolist(), cols.tolist()):
        R[i].add(j)
        Cc[j].add(i)
    for i in range(n):
        R[i].add(i)
        Cc[i].add(i)

    def cost(k):
        return (len(R[k]) - 1) * (len(Cc[k]) - 1)

    heap = [(cost(i), len(R[i]) + len(Cc[i]), i) for i in range(n)]
    heapq.heapify(heap)
    done = np.zeros(n, dtype=bool)
    order = []
    while heap:
        c, _, k = heapq.heappop(heap)
        if done[k] or c != cost(k):
            continue
        done[k] = True
        order.append(k)
        rk = [j for j in R[k] if j != k]
        ck = [i for i in Cc[k] if i != k]
        for i in ck:
            R[i].discard(k)
            for j in rk:
                if j not in R[i]:
                    R[i].add(j)
                    Cc[j].add(i)
        for j in rk:
            Cc[j].discard(k)
        for v in set(ck) | set(rk):
            heapq.heappush(heap, (cost(v), len(R[v]) + len(Cc[v]), v))
        R[k] = set()
        Cc[k] = set()
    return np.asarray(order, dtype=np.int64)
