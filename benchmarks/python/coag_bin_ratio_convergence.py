"""Coagulation convergence across bin resolutions: TFL vs linear sub-bin scheme.

Two coag-only cases, each run with both rate schemes at 40/80/160/320 bins
(mass ratio 2^(40/nbins), 1.7 nm start, as in tomas-api):

  analytic  - constant kernel, exponential initial distribution; compared
              with the exact Smoluchowski solution (N falls 6x in 24 h).
  brownian  - Fuchs Brownian kernel, lognormal N=1e5 cm-3, GMD 100 nm,
              GSD 1.6, 24 h (the tomas-web case that exposed TFL's
              oscillations at 80/160 bins).

TFL at p != 2 is what calc_coagulation_rates did before it started routing
non-doubling grids to the linear scheme; it is shown for reference.

Usage::

    python -m benchmarks.python.coag_bin_ratio_convergence
    python -m benchmarks.python.coag_bin_ratio_convergence --no-plots
"""
import argparse
import os
import time

import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.core.config import ICOMP, ICOMP_NODIAG, SRTSO4, make_grid
from tomas_jax.core.mnfix_jax import mnfix_jax
from tomas_jax.physics.properties import calc_particle_properties
from tomas_jax.physics.coagulation_kernel import calc_coagulation_kernel
from tomas_jax.physics.coagulation_rates import calc_coagulation_rates_tfl
from tomas_jax.physics.coagulation_rates_linear import calc_coagulation_rates_linear

XK0_1_7NM = 3.4728e-24
BOXVOL = 1.0e6
TEMP, PRES = 273.0, 1.0e5
DT, NSUB, HOURS = 60.0, 3, 24.0
NBINS = (40, 80, 160, 320)
SCHEMES = {"tfl": calc_coagulation_rates_tfl, "linear": calc_coagulation_rates_linear}

# Constant-kernel case
N0 = 1.0e4
X0 = np.pi / 6.0 * 1770.0 * (100e-9) ** 3
TAU_END = 10.0
KCONST = TAU_END / (N0 * HOURS * 3600.0)

OUTDIR = os.path.join("benchmarks", "results", "coag_bin_ratio")


def grid(nbins):
    return make_grid(nbins, XK0_1_7NM, 2.0 ** (40.0 / nbins))


def exact_bins(xk, tau):
    s = 1.0 + tau / 2.0
    xs = X0 * s
    a, b = np.asarray(xk[:-1]), np.asarray(xk[1:])
    Nk = N0 / s * (np.exp(-a / xs) - np.exp(-b / xs))
    Mk = N0 / s * xs * ((a / xs + 1) * np.exp(-a / xs) - (b / xs + 1) * np.exp(-b / xs))
    return Nk, Mk


def lognormal(xk, n_total=1e5, gmd_nm=100.0, gsd=1.6):
    m_mid = jnp.sqrt(xk[:-1] * xk[1:])
    dp = jnp.cbrt(m_mid / 1770.0 * 6.0 / np.pi)
    dndlog = n_total / (np.sqrt(2 * np.pi) * np.log(gsd)) * jnp.exp(
        -(jnp.log(dp) - np.log(gmd_nm * 1e-9)) ** 2 / (2 * np.log(gsd) ** 2))
    Nk = dndlog * jnp.log(xk[1:] / xk[:-1]) / 3.0 * BOXVOL
    return Nk, jnp.zeros((xk.shape[0] - 1, ICOMP)).at[:, SRTSO4].set(Nk * m_mid)


def integrate(Nk, Mk, xk, kernel_fn, rates_fn):
    """Forward Euler + MNFIX, kernel refreshed each step (as coag_euler_step)."""
    def step(carry, _):
        N, M, ov = carry
        kij = kernel_fn(N, M)
        for _ in range(NSUB):
            dN, dM, dov = rates_fn(N, M, kij, xk, ICOMP_NODIAG)
            N = jnp.maximum(N + DT / NSUB * dN, 0.0)
            M = jnp.maximum(M + DT / NSUB * dM, 0.0)
            ov = ov + DT / NSUB * dov
            N, M = mnfix_jax(N, M, xk, ICOMP_NODIAG)
        return (N, M, ov), None

    run = jax.jit(lambda N, M: jax.lax.scan(
        step, (N, M, jnp.zeros(ICOMP)), None, length=int(HOURS * 3600 / DT))[0])
    t0 = time.perf_counter()
    out = jax.block_until_ready(run(Nk, Mk))
    return out, time.perf_counter() - t0


def dndlogdp(xk, Nk_cm3):
    xk = np.asarray(xk)
    return np.asarray(Nk_cm3) / (np.log10(xk[1:] / xk[:-1]) / 3.0)


def n_extrema(y):
    y = y[y > 1e-3 * y.max()]
    return int(np.sum(np.diff(np.sign(np.diff(y))) != 0))


def run_case(case, scheme, nbins):
    xk = grid(nbins)
    if case == "analytic":
        Nk0, Mk0 = exact_bins(xk, 0.0)
        Nk = jnp.asarray(Nk0) * BOXVOL
        Mk = jnp.zeros((nbins, ICOMP)).at[:, SRTSO4].set(jnp.asarray(Mk0) * BOXVOL)
        kij = jnp.full((nbins, nbins), KCONST / BOXVOL)
        kernel_fn = lambda N, M: kij
    else:
        Nk, Mk = lognormal(xk)
        kernel_fn = lambda N, M: calc_coagulation_kernel(
            *calc_particle_properties(N, M, TEMP, PRES), BOXVOL)
    M0 = float(jnp.sum(Mk[:, :ICOMP_NODIAG]))
    (N, M, ov), wall = integrate(Nk, Mk, xk, kernel_fn, SCHEMES[scheme])
    if not bool(jnp.all(jnp.isfinite(N))):
        raise RuntimeError(f"Non-finite result: {case} {scheme} {nbins} bins")
    y = dndlogdp(xk, N / BOXVOL)
    res = dict(xk=np.asarray(xk), y=y, wall=wall,
               N=float(jnp.sum(N)) / BOXVOL, extrema=n_extrema(y),
               mass_budget=(float(jnp.sum(M[:, :ICOMP_NODIAG]) + jnp.sum(ov)) - M0) / M0)
    if case == "analytic":
        Ne, _ = exact_bins(xk, TAU_END)
        ye = dndlogdp(xk, Ne)
        w = np.log10(np.asarray(xk[1:] / xk[:-1])) / 3.0
        m = ye > 1e-3 * ye.max()
        res.update(y_exact=ye,
                   l1=float(np.sum(np.abs(y - ye)[m] * w[m]) / np.sum(ye[m] * w[m])),
                   peak_ratio=float(y.max() / ye.max()),
                   N_exact=float(Ne.sum()))
    return res


def print_table(results):
    print(f"\n{'case':9s} {'scheme':7s} {'bins':>4s} {'N_final':>10s} {'L1 err':>8s} "
          f"{'peak/ex':>8s} {'extrema':>7s} {'mass budget':>11s} {'wall s':>7s}")
    for (case, scheme, nb), r in results.items():
        l1 = f"{r['l1']:.4f}" if "l1" in r else "-"
        pk = f"{r['peak_ratio']:.3f}" if "peak_ratio" in r else "-"
        print(f"{case:9s} {scheme:7s} {nb:4d} {r['N']:10.4e} {l1:>8s} {pk:>8s} "
              f"{r['extrema']:7d} {r['mass_budget']:+11.1e} {r['wall']:7.1f}")


def plot(results, outdir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(outdir, exist_ok=True)
    for case in ("analytic", "brownian"):
        fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
        for ax, scheme in zip(axes, SCHEMES):
            for nb in NBINS:
                key = (case, scheme, nb)
                if key not in results:
                    raise KeyError(f"Missing result {key}; run all cases before plotting")
                r = results[key]
                dp = np.cbrt(np.sqrt(r["xk"][:-1] * r["xk"][1:]) / 1770.0 * 6 / np.pi) * 1e9
                ax.plot(dp, r["y"], label=f"{nb} bins")
                if case == "analytic" and nb == NBINS[-1]:
                    ax.plot(dp, r["y_exact"], "k--", lw=1, label="exact")
            ax.set_xscale("log")
            ax.set_xlim(10, 3000)
            ax.set_xlabel("Dp [nm]")
            ax.set_title(f"{case}: {scheme}")
            ax.legend(fontsize=8)
        axes[0].set_ylabel("dN/dlogDp [cm$^{-3}$] after 24 h")
        fig.tight_layout()
        path = os.path.join(outdir, f"coag_bin_ratio_{case}.png")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        print(f"wrote {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--outdir", default=OUTDIR)
    args = parser.parse_args()

    results = {}
    for case in ("analytic", "brownian"):
        for scheme in SCHEMES:
            for nb in NBINS:
                results[(case, scheme, nb)] = run_case(case, scheme, nb)
                print(f"done {case} {scheme} {nb}", flush=True)
    print_table(results)
    if not args.no_plots:
        plot(results, args.outdir)


if __name__ == "__main__":
    main()
