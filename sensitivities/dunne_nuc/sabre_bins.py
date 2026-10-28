"""Map TOMAS-JAX size distributions onto SABRE measurement bins.

Source: Lyu et al. (2026), Science, doi:10.1126/science.adw8939, SuppMat
"Aerosol size distribution instruments" and "Data processing for particle
size distributions" (references/sabre/):

  - Inverted product: dN/dlogDp with 20 log-spaced intervals per decade,
    Dp from 0.003 to ~4 um (NMASS + UHSAS + CMASS combined).
  - NMASS: 5 cumulative CPC channels, minimum Dp 3, 8, 12, 27, 53 nm.
  - Sizes are dry (sample heated from ~220 K to >273 K; <10% water by mass,
    not corrected). Sizing is DES-calibrated (n = 1.44, close to H2SO4).
  - Inversion uncertainty in number: standard relative error < 8%,
    relative bias < +16% (Table S2).

ASSUMPTION: the paper does not list exact bin edges. SABRE_EDGES_M starts
at 3 nm and steps 10^(1/20) (63 bins, top edge 4.24 um). Replace with the
edges from the SABRE data files when available; every function takes
``edges`` explicitly.

NMASS channels are modelled as sharp cutoffs. The real channels have
smooth counting-efficiency curves (SuppMat Fig. S19).

All functions are pure JAX, linear in Nk, and differentiable w.r.t. Nk
and Mk.
"""
import numpy as np
import jax.numpy as jnp

from tomas_jax.core.config import SRTSO4, SRTNH4, SRTH2O, SRTORG1, SRTORGLAST, PI
from tomas_jax.physics.density import calc_density
from tomas_jax.physics.radiative_forcing import h2so4_solution_density

SABRE_BINS_PER_DECADE = 20
SABRE_EDGES_M = 3e-9 * 10.0 ** (np.arange(64) / SABRE_BINS_PER_DECADE)
SABRE_MID_M = np.sqrt(SABRE_EDGES_M[:-1] * SABRE_EDGES_M[1:])
NMASS_CUTOFFS_M = np.array([3.0, 8.0, 12.0, 27.0, 53.0]) * 1e-9
SABRE_NUMBER_REL_ERR = 0.08
RHO_WATER = 1000.0  # kg/m3


def dry_density(Mk):
    """Per-bin dry particle density [kg/m3] from composition.

    NH4-free bins (stratospheric sulfate): pure H2SO4 from the CRC table
    (h2so4_solution_density at 100 wt%, 1836 kg/m3). calc_density would give
    1902 here: it counts SO4 mass as H2SO4 (x 98/96), so dry particles sit
    at 102 wt%, and its Tang (1997) fit is ~3% above CRC near 100 wt%.

    Bins with NH4: calc_density, with the acid protons added as solute mass
    so the dry weight percent is 100, not above. Mass fractions are
    normalised per bin first, so calc_density's absolute-mass empty-bin
    fallback (< 1e-15 kg -> 1000 kg/m3, boxvol-dependent) does not fire for
    sparse but real bins. Organics are lumped with sulfate, as in the model.
    """
    Mk_dry = Mk.at[:, SRTH2O].set(0.0)
    tot = jnp.sum(Mk_dry, axis=1, keepdims=True)
    frac = Mk_dry / jnp.where(tot > 0.0, tot, 1.0)

    n_so4 = (frac[:, SRTSO4]
             + jnp.sum(frac[:, SRTORG1:SRTORGLAST + 1], axis=1)) / 96.0
    n_nh4 = frac[:, SRTNH4] / 18.0
    protons = jnp.maximum(2.0 * n_so4 - n_nh4, 0.0) * 1.0   # g/mol of H+
    rho_nh4 = calc_density(frac.at[:, SRTH2O].set(protons))

    rho_acid = h2so4_solution_density(100.0)
    return jnp.where(frac[:, SRTNH4] > 0.0, rho_nh4, rho_acid)


def model_bin_diameters(Mk, xk, water_frac=0.0):
    """Lower/upper Dp [m] of each model bin.

    Uses dry mass edges xk and dry density. water_frac > 0 adds that mass
    fraction of water (volume-additive), to test the residual water left
    in SABRE's heated sample.
    """
    rho = dry_density(Mk)
    grow = (1.0 + water_frac / (1.0 - water_frac) * rho / RHO_WATER) ** (1.0 / 3.0)
    d_lo = (6.0 * xk[:-1] / (PI * rho)) ** (1.0 / 3.0) * grow
    d_hi = (6.0 * xk[1:] / (PI * rho)) ** (1.0 / 3.0) * grow
    return d_lo, d_hi


def rebin_to_sabre(Nk, Mk, xk, boxvol, edges=SABRE_EDGES_M, water_frac=0.0):
    """dN/dlogDp [cm-3] on the SABRE bins.

    Number is spread uniformly in log Dp within each model bin, and the
    overlap fraction with each SABRE bin is summed.
    """
    d_lo, d_hi = model_bin_diameters(Mk, xk, water_frac)
    lo, hi = jnp.log(d_lo)[:, None], jnp.log(d_hi)[:, None]
    e = jnp.log(jnp.asarray(edges))
    overlap = jnp.clip(jnp.minimum(hi, e[None, 1:]) - jnp.maximum(lo, e[None, :-1]),
                       0.0, None) / (hi - lo)
    N = (Nk / boxvol) @ overlap
    return N / jnp.diff(jnp.log10(jnp.asarray(edges)))


def nmass_cumulative(Nk, Mk, xk, boxvol, cutoffs=NMASS_CUTOFFS_M, water_frac=0.0):
    """N(Dp > cutoff) [cm-3] for each NMASS channel (sharp cutoffs)."""
    d_lo, d_hi = model_bin_diameters(Mk, xk, water_frac)
    lo, hi = jnp.log(d_lo)[:, None], jnp.log(d_hi)[:, None]
    c = jnp.log(jnp.asarray(cutoffs))[None, :]
    above = jnp.clip((hi - c) / (hi - lo), 0.0, 1.0)
    return (Nk / boxvol) @ above
