"""Reader for the shared gas-chemistry benchmark files, plus a numpy right-hand side,
an analytic sparse Jacobian and the coupling-interval loop.

Plain numpy/scipy, no JAX and no TOMAS-JAX import, so it doubles as the executable
specification that a Julia (or any other) implementation must reproduce.

Problem definition (see BENCHMARK.md):
    reaction j:   r_j = k_j * prod_{reactant occurrences a} y_a     (A + A gives y_A^2)
    species s:    dy_s/dt = sum_j nu_sj r_j,  nu_sj = product coef - reactant count
    interval i (length dt, i = 0..N-1, t_i = i dt):
        k_i = k_thermal * (RO2(y(t_i)) if ro2_dependent else 1) + J_coef @ J_i
        k_i[external reaction] = k_external
        J_i = photolysis frequencies at the interval midpoint t_i + dt/2 (tabulated)
        solve dy/dt = f(y; k_i) from t_i to t_i + dt, then
        clip negatives to 0 -> this is the interval-end output that is scored,
        then reset counters to 0 and held species to their initial values.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.sparse as sp

BENCH = Path(__file__).resolve().parents[1]
PROBLEMS = BENCH / "problems"
REFERENCE = BENCH / "reference"
RESULTS = BENCH / "results"


@dataclass
class Mech:
    name: str
    species: list[str]
    reactants: np.ndarray        # [nr, 2] int, padded with n (index of a constant 1.0)
    prod_rxn: np.ndarray
    prod_sp: np.ndarray
    prod_coef: np.ndarray
    loss_rxn: np.ndarray
    loss_sp: np.ndarray
    ro2: np.ndarray

    @property
    def n(self) -> int:
        return len(self.species)

    @property
    def nr(self) -> int:
        return self.reactants.shape[0]

    def index(self, name: str) -> int:
        return self.species.index(name)


def load_mechanism(path: str | Path) -> Mech:
    d = json.loads(Path(path).read_text())
    n, nr = d["n_species"], d["n_reactions"]
    reac = np.full((nr, 2), n, dtype=np.int64)
    prod_rxn, prod_sp, prod_coef, loss_rxn, loss_sp = [], [], [], [], []
    for j, (rs, ps) in enumerate(zip(d["reactants"], d["products"])):
        for q, a in enumerate(rs):
            reac[j, q] = a
            loss_rxn.append(j)
            loss_sp.append(a)
        for s, c in ps:
            prod_rxn.append(j)
            prod_sp.append(s)
            prod_coef.append(c)
    return Mech(d["name"], d["species"], reac, np.array(prod_rxn), np.array(prod_sp),
                np.array(prod_coef, dtype=float), np.array(loss_rxn), np.array(loss_sp),
                np.array(d["ro2"], dtype=np.int64))


@dataclass
class Problem:
    name: str
    meta: dict
    mech: Mech
    k_thermal: np.ndarray
    ro2_dependent: np.ndarray    # bool [nr]
    j_coef: sp.csr_matrix        # [nr, nJ]
    j_names: list[str]
    external: list[tuple[int, float]]
    y0: np.ndarray
    reset_zero: np.ndarray
    reset_initial: np.ndarray
    key_species: list[str]
    rtol: float
    atol: float
    duration_s: float

    def j_table(self, coupling_s: int) -> np.ndarray:
        return np.asarray(self.meta["photolysis"][str(coupling_s)], dtype=float)

    def n_intervals(self, coupling_s: int) -> int:
        return int(round(self.duration_s / coupling_s))

    def k(self, y: np.ndarray, j: np.ndarray) -> np.ndarray:
        ro2 = y[self.mech.ro2].sum()
        k = self.k_thermal * np.where(self.ro2_dependent, ro2, 1.0) + self.j_coef @ j
        for idx, val in self.external:
            k[idx] = val
        return k

    def post_interval(self, y: np.ndarray) -> np.ndarray:
        """Clip, then reset counters and held species; returns the next start state."""
        y = np.maximum(y, 0.0)
        y[self.reset_zero] = 0.0
        y[self.reset_initial] = self.y0[self.reset_initial]
        return y


def load_problem(name: str, problems_dir: Path = PROBLEMS) -> Problem:
    meta = json.loads((problems_dir / f"problem_{name}.json").read_text())
    mech = load_mechanism(problems_dir / meta["mechanism_file"])
    rates = meta["rates"]
    nr = mech.nr
    jr = np.array([e[0] for e in rates["j_coef"]], dtype=np.int64)
    jc = np.array([e[1] for e in rates["j_coef"]], dtype=np.int64)
    jv = np.array([e[2] for e in rates["j_coef"]], dtype=float)
    j_coef = sp.csr_matrix((jv, (jr, jc)), shape=(nr, len(rates["j_names"])))
    ro2_dep = np.zeros(nr, dtype=bool)
    ro2_dep[np.asarray(rates["ro2_dependent"], dtype=np.int64)] = True
    y0 = np.zeros(mech.n)
    for idx, val in meta["initial"]["y0_nonzero"]:
        y0[idx] = val
    post = meta["post_interval"]
    return Problem(
        name=meta["name"], meta=meta, mech=mech,
        k_thermal=np.asarray(rates["k_thermal"], dtype=float), ro2_dependent=ro2_dep,
        j_coef=j_coef, j_names=list(rates["j_names"]),
        external=[(int(e["reaction"]), float(e["k"])) for e in rates["external"]],
        y0=y0, reset_zero=np.asarray(post["reset_to_zero"], dtype=np.int64),
        reset_initial=np.asarray(post["reset_to_initial"], dtype=np.int64),
        key_species=list(meta["key_species"]), rtol=meta["tolerances"]["rtol"],
        atol=meta["tolerances"]["atol"], duration_s=float(meta["duration_s"]))


class NumpyChem:
    """f(y; k) and the analytic sparse Jacobian df/dy (CSC) of a mechanism."""

    def __init__(self, mech: Mech):
        self.m = mech
        n, nr = mech.n, mech.nr
        # net stoichiometry nu[j, s] as COO entries
        nu = sp.coo_matrix(
            (np.concatenate([mech.prod_coef, -np.ones(mech.loss_rxn.size)]),
             (np.concatenate([mech.prod_rxn, mech.loss_rxn]),
              np.concatenate([mech.prod_sp, mech.loss_sp]))), shape=(nr, n)).tocsr()
        nu.sum_duplicates()
        nu.eliminate_zeros()
        rows, cols, rxn, slot, val = [], [], [], [], []
        nu_coo = nu.tocoo()
        for q in (0, 1):
            col = mech.reactants[nu_coo.row, q]
            ok = col < n
            rows.append(nu_coo.col[ok])
            cols.append(col[ok])
            rxn.append(nu_coo.row[ok])
            slot.append(np.full(ok.sum(), q))
            val.append(nu_coo.data[ok])
        self.e_row = np.concatenate(rows)
        self.e_col = np.concatenate(cols)
        self.e_rxn = np.concatenate(rxn)
        self.e_other = mech.reactants[self.e_rxn, 1 - np.concatenate(slot)]
        self.e_nu = np.concatenate(val)
        # CSC structure of the Jacobian and the map entry -> stored position
        key = self.e_col * n + self.e_row
        uniq, self.e_pos = np.unique(key, return_inverse=True)
        self.indices = (uniq % n).astype(np.int32)
        cols_u = uniq // n
        self.indptr = np.searchsorted(cols_u, np.arange(n + 1)).astype(np.int32)
        self.nnz = uniq.size

    def rates(self, y, k):
        ye = np.append(y, 1.0)
        return k * ye[self.m.reactants[:, 0]] * ye[self.m.reactants[:, 1]]

    def rhs(self, y, k):
        r = self.rates(y, k)
        n = self.m.n
        return (np.bincount(self.m.prod_sp, weights=self.m.prod_coef * r[self.m.prod_rxn],
                            minlength=n)
                - np.bincount(self.m.loss_sp, weights=r[self.m.loss_rxn], minlength=n))

    def jac(self, y, k):
        ye = np.append(y, 1.0)
        vals = self.e_nu * k[self.e_rxn] * ye[self.e_other]
        data = np.bincount(self.e_pos, weights=vals, minlength=self.nnz)
        n = self.m.n
        return sp.csc_matrix((data, self.indices, self.indptr), shape=(n, n))


def run_intervals(prob: Problem, coupling_s: int, solve, on_interval=None):
    """Drive the coupling-interval loop.

    solve(y_start, k, dt) -> y_end (unclipped; clipping is done here).
    on_interval(i, y_start, k, y_end_clipped) is called after every interval.
    Returns (outputs [N, n] interval-end states after clipping, ro2_start [N]).
    """
    jtab = prob.j_table(coupling_s)
    n_int = prob.n_intervals(coupling_s)
    assert jtab.shape[0] == n_int
    y = prob.y0.copy()
    out = np.zeros((n_int, prob.mech.n))
    ro2 = np.zeros(n_int)
    for i in range(n_int):
        ro2[i] = y[prob.mech.ro2].sum()
        k = prob.k(y, jtab[i])
        y_end = np.maximum(np.asarray(solve(y, k, float(coupling_s)), dtype=float), 0.0)
        out[i] = y_end
        if on_interval is not None:
            on_interval(i, y, k, y_end)
        y = prob.post_interval(y_end.copy())
    return out, ro2


# --- accuracy metric -------------------------------------------------------------------
METRIC_FLOOR = 1e6      # [molec/cm3] values below this are compared in absolute terms


def key_errors(key_out: dict[str, np.ndarray], key_ref: dict[str, np.ndarray],
               floor: float = METRIC_FLOOR) -> dict:
    """e = |y - y_ref| / max(|y_ref|, floor) over key species and interval ends."""
    per = {}
    allv = []
    for s, ref in key_ref.items():
        ref = np.asarray(ref, dtype=float)
        e = np.abs(np.asarray(key_out[s], dtype=float) - ref) / np.maximum(np.abs(ref), floor)
        per[s] = float(e.max())
        allv.append(e)
    allv = np.concatenate(allv)
    worst = max(per, key=per.get)
    return {"max": float(allv.max()), "median": float(np.median(allv)),
            "p95": float(np.percentile(allv, 95)), "worst_species": worst, "per_species": per}


def all_species_error(y: np.ndarray, y_ref: np.ndarray, floor: float = METRIC_FLOOR) -> float:
    """Max over all species of |y - y_ref| / max(|y_ref|, floor) (hourly full states)."""
    return float((np.abs(y - y_ref) / np.maximum(np.abs(y_ref), floor)).max())


def load_reference(problem: str, coupling_s: int, ref_dir: Path = REFERENCE):
    meta = json.loads((ref_dir / f"ref_{problem}_dt{coupling_s}.json").read_text())
    full = np.load(ref_dir / f"ref_{problem}_dt{coupling_s}_hourly.npz")
    return meta, full
