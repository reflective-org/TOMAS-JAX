"""Size-distribution parameters of a binned number distribution: moments from
the bins (no fit) and fits of 1-3 lognormal modes, the mode count chosen by BIC.

Moments, from each bin's dry diameter (its mean dry mass per particle at
RHO_GRID; the bin centre where the bin is empty): number geometric mean
diameter Dg and geometric standard deviation sigma_g over all particles, and
the effective radius r_eff = sum N r^3 / sum N r^2.

Fits: N_k = sum_m N_m [Phi(z_m(D_k+1)) - Phi(z_m(D_k))], z_m(D) =
ln(D/Dg_m)/ln(sigma_m), D_k the bin edges at RHO_GRID (each mode integrated
over the bins, not sampled at bin centres). Residuals are in ln(N + floor),
floor = FLOOR_FRAC * sum N, so a mode with a small share of the number (the
grown particles beside a nucleation mode) counts as much as a large one, and
bins far below the floor do not count. The mode count K in 1..MAX_MODES
minimises BIC = n ln(RSS/n) + 3K ln n (n bins). Modes are returned sorted by Dg.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.signal import find_peaks
from scipy.special import ndtr

from benchmarks.python.gas_substep_convergence import RHO_GRID, edges_nm

MAX_MODES = 3
FLOOR_FRAC = 1e-6
LN_SIGMA = (0.02, np.log(4.0))      # bounds on ln sigma_m (an 80-bin bin is 0.058 wide)


def moments(N, M, xk, boxvol):
    """Dg [nm], sigma_g and r_eff [nm] of the number distribution N [#/cm3]
    with dry mass M [kg/cell] per bin."""
    e = edges_nm(xk)
    centre = np.sqrt(e[:-1] * e[1:])
    n_cell = N * boxvol
    full = n_cell > 1e-6
    m = np.where(full, M / np.where(full, n_cell, 1.0), 0.0)
    D = np.where(full, (6 * m / (np.pi * RHO_GRID)) ** (1 / 3) * 1e9, centre)
    w = N / N.sum()
    lnD = np.log(D)
    mu = np.sum(w * lnD)
    return dict(Dg=float(np.exp(mu)),
                sigma=float(np.exp(np.sqrt(np.sum(w * (lnD - mu) ** 2)))),
                reff=float(0.5 * np.sum(N * D ** 3) / np.sum(N * D ** 2)))


def model(x, le):
    """Bin numbers of the modes x = [ln N_m, ln Dg_m, ln sigma_m] * K."""
    lnN, lnDg, lns = np.reshape(x, (-1, 3)).T
    z = (le[None, :] - lnDg[:, None]) / lns[:, None]
    return (np.exp(lnN)[:, None] * np.diff(ndtr(z), axis=1)).sum(0)


def _starts(N, le, K):
    """Initial guesses: the K most prominent peaks of ln(N + floor), then the
    number quantiles for any modes left over."""
    floor = FLOOR_FRAC * N.sum()
    y = np.log(N + floor)
    lc = 0.5 * (le[:-1] + le[1:])
    peaks, prop = find_peaks(np.concatenate(([y.min()], y, [y.min()])),
                             prominence=0.5)
    peaks = list(peaks[np.argsort(prop['prominences'])[::-1]] - 1)[:K]
    q = np.searchsorted(np.cumsum(N) / N.sum(), (np.arange(K) + 0.5) / K)
    starts = []
    for extra in (q, q[::-1]):
        at = peaks + [int(i) for i in extra if int(i) not in peaks][:K - len(peaks)]
        at = (at + list(range(len(N))))[:K]
        x = []
        for i in at:
            x += [np.log(max(N[i], floor) * 4.0), lc[i], 0.3]
        starts.append(np.array(x))
    return starts


def fit(N, xk, K, x0=None):
    """Best fit with K modes: (x sorted by Dg, RSS)."""
    le = np.log(edges_nm(xk))
    floor = FLOOR_FRAC * N.sum()
    y = np.log(N + floor)
    lo = np.tile([np.log(floor) - 5, le[0] - 2, LN_SIGMA[0]], K)
    hi = np.tile([np.log(N.sum()) + 2, le[-1] + 2, LN_SIGMA[1]], K)
    best = None
    for x in _starts(N, le, K) + ([] if x0 is None else [x0]):
        r = least_squares(lambda x: np.log(model(x, le) + floor) - y,
                          np.clip(x, lo + 1e-9, hi - 1e-9), bounds=(lo, hi),
                          x_scale=np.tile([1.0, 0.1, 0.1], K))
        if best is None or r.cost < best.cost:
            best = r
    x = np.reshape(best.x, (-1, 3))
    x = x[np.argsort(x[:, 1])].ravel()
    return x, 2.0 * best.cost


def bic(rss, n, K):
    return n * np.log(max(rss, 1e-300) / n) + 3 * K * np.log(n)


def best_fit(N, xk):
    """(K, x, {K: (x, rss)}): the BIC-best fit over K = 1..MAX_MODES."""
    fits = {K: fit(N, xk, K) for K in range(1, MAX_MODES + 1)}
    K = min(fits, key=lambda K: bic(fits[K][1], len(N), K))
    return K, fits[K][0], fits


def modes(x):
    """x -> dict of arrays N [#/cm3], Dg [nm], sigma per mode."""
    lnN, lnDg, lns = np.reshape(x, (-1, 3)).T
    return dict(N=np.exp(lnN), Dg=np.exp(lnDg), sigma=np.exp(lns))
