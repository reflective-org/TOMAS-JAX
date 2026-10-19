"""Heterogeneous N2O5 uptake on the aerosol population (JIT-compilable).

The first-order loss rate of N2O5 to particles is the condensation sink
(getCondSink.f) with the mass accommodation coefficient replaced by the
reactive uptake coefficient gamma:

    k_het = 2*pi*D_g * sum_k(Dp_k * N_k * beta_k(Kn_k, gamma)) / V

which reduces to the free-molecular limit k_het = gamma * c_bar * A / 4
(A = particle surface area per volume) for Kn >> 1. Sizes are the TOMAS
wet diameters (calc_wet_diameters). The per-bin fractions of k_het tell
the caller where the N2O5 -> 2 NO3-(p) product goes (Mk[:, SRTNIT]).

See docs/aaar/heterogeneous.md.

References:
    - getCondSink.f (Jeff Pierce, May 2007)
    - Fuchs & Sutugin (1971), Topics in Current Aerosol Research
    - Burkholder et al. (2019), JPL Evaluation No. 19-5 (N2O5 gamma)
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp
from typing import Tuple

from .condensation_sink import calc_condensation_sink
from ..core.config import MW_N2O5, SV_N2O5

# The bin weights are computed with max(gamma, GAMMA_WEIGHT_FLOOR) so they
# stay a well-defined partition (sum 1) when gamma = 0. At this gamma the
# Fuchs-Sutugin factor is linear in gamma, so k_het = CS * gamma/gamma_w is
# exact below the floor too.
GAMMA_WEIGHT_FLOOR = 1.0e-8


def n2o5_uptake(
    Nk: jnp.ndarray,
    Mk: jnp.ndarray,
    xk: jnp.ndarray,
    temp,
    pres,
    boxvol,
    gamma,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """First-order N2O5 uptake rate and its per-bin distribution.

    Args:
        Nk: Number per bin [#/grid cell], shape (ibins,)
        Mk: Mass per bin [kg/grid cell], shape (ibins, icomp) (wet)
        xk: Bin boundaries [kg], shape (ibins+1,)
        temp: Temperature [K]
        pres: Pressure [Pa]
        boxvol: Grid cell volume [cm^3]
        gamma: N2O5 reactive uptake coefficient [-], >= 0

    Returns:
        k_het: N2O5 loss rate to particles [s^-1]
        weights: Fraction of k_het per bin, shape (ibins,). Sums to 1
            (all zeros only if the population has no particles at all).
    """
    gamma = jnp.asarray(gamma, dtype=jnp.float64)
    gamma_w = jnp.maximum(gamma, GAMMA_WEIGHT_FLOOR)
    cs, weights = calc_condensation_sink(
        Nk, Mk, temp, pres, boxvol,
        molecular_weight=MW_N2O5, diffusion_volume=SV_N2O5,
        alpha=gamma_w, xk=xk,
    )
    k_het = cs * (gamma / gamma_w)
    return k_het, weights
