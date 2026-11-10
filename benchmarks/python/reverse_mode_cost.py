"""Wall time and peak memory: 24 h jax.grad (ad_mode='reverse') vs jacfwd.

One run per process (peak RSS is per process):
    python -m benchmarks.python.reverse_mode_cost grad
    python -m benchmarks.python.reverse_mode_cost jacfwd
Study settings, 40 bins, 217 K, 1 ppmv SO2, gas_substep='adaptive'.
Output: scalar ln N(>3 nm) (NMASS) at 24 h, derivative w.r.t. all 25
NucleationParams fields.
"""
import sys
import time
import resource

import numpy as np
import jax
import jax.numpy as jnp

from tomas_jax.core.config import make_grid, XK0
from sensitivities.dunne_nuc import common as c
from sensitivities.dunne_nuc.sabre_bins import nmass_cumulative

HOURS, TEMP, SO2_PPBV = 24, 217.0, 1e3


def main(mode):
    xk = jnp.asarray(make_grid(40, XK0, 2.0))
    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(SO2_PPBV, TEMP,
                                                      c.STRAT['pres']))
    ad = 'reverse' if mode == 'grad' else 'forward'
    run = c.make_runner(xk, HOURS * 60, cond=dict(c.STRAT, temp=TEMP),
                        gas_substep='adaptive', ad_mode=ad)

    def f(p):
        Nk, Mk, _, _ = run(p, *state)
        return jnp.log(nmass_cumulative(Nk, Mk, xk, c.BOXVOL)[0])

    P = jax.tree.map(jnp.float64, c.NUC_PARAMS)
    d = jax.grad(f) if mode == 'grad' else jax.jacfwd(f)
    t0 = time.time()
    out = jax.block_until_ready(jax.jit(d)(P))
    t = time.time() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**30  # macOS: bytes
    vec = np.array([float(getattr(out, k)) for k in P._fields])
    np.save(f'benchmarks/results/gas_substep/reverse_cost_{mode}.npy', vec)
    print(f"{mode}: {t:.0f} s (incl. compile), peak RSS {rss:.2f} GB, "
          f"|d|_max {np.max(np.abs(vec)):.4g}")


if __name__ == '__main__':
    main(sys.argv[1])
