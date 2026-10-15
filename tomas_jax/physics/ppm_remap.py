"""PPM remap of an in-place particle mass change onto the fixed bins (JIT).

Some processes change the mass of the particles in each bin without moving
them: SOA partitioning (vbs-jax adds or removes class mass per bin) and the
N2O5-uptake nitrate source. Afterwards a bin's mean mass can sit outside
the bin. ``remap_mass_change`` moves the particles to the bins they now
belong to with the flux-form PPM advection in log-mass that H2SO4
condensation uses (physics/condensation_ppm.py), instead of MNFIX's
two-packet split.

Algorithm (one call per in-place change, e.g. per 60 s micro-step):

1. Bin displacement. Each bin with particles (Nk > n_min, dry mass > 0)
   moved its mean dry mass from m to m' = m + dm, a log-mass displacement
   s_k = ln(m'/m) at xi_k = ln(m). dm < 0 (evaporation) gives s_k < 0.
2. Edge displacement (upwind). Within each bin, s is taken as linear
   in ln(m) around the bin mean, with the smaller (in magnitude) of the
   two one-sided slopes to the neighbouring bin means (zero next to a
   neighbour without particles or with the same s); the edge values are
   clipped to lie between the bin's s and the neighbour's. Particles
   cross an edge upwards with the lower bin's value at that edge (if > 0)
   and downwards with the upper bin's value (if < 0); the domain edges
   carry no flux. So particles whose mass did not change (s = 0 next to
   a jump in s) never move, even when a neighbour grows or shrinks a lot;
   interpolating between the two neighbouring means instead (an earlier
   version) dragged them along. Where s varies smoothly and falls with
   size (growth), the upward edge value equals that interpolation. A bin
   loses particles through both of its edges only where its own s
   changes sign inside it; the sub-step count keeps those two departure
   regions apart.
   Why not dmdt_int from each bin's own mean (as ezcond_ppm_jax): the
   dmdt_int law (d m^(2/3)/dt constant) extrapolated from the bin mean to
   the bin's upper edge slows the edge more than the vbs-jax rates (Fuchs-
   Sutugin transition regime) do, so number lags mass, every bin's mean
   creeps to its upper edge (Level 1 at 80 bins: from 0.50 to 0.96 of the
   bin width in 12 h) and MNFIX then splits the bins again.
3. Sub-steps. n_sub = ceil(max_k(out-flux displacement of bin k) /
   (C_max dxi)), dxi = ln(x_{k+1}/x_k); each sub-step moves 1/n_sub of
   the displacement (Courant <= C_max = 0.8).
4. Fluxes per sub-step (condensation_ppm building blocks):
   - number: PPM reconstruction of dN/dln m (ppm_reconstruct, monotone
     and positive) integrated over the departure region of each edge;
   - mass: the mass-weighted integral of the same reconstruction over the
     departure region, as a fraction of the reconstruction's bin total.
     Every species of the donor bin (the post-change masses) moves with
     that fraction, so all species share one consistent flux, the change
     travels with the particles that carry it, and no bin goes negative.
   Updates are flux differences: number and every species' mass are
   conserved to round-off.
5. keep_means_in_bins: PPM carries no information on where the particles
   sit inside a bin, so at a steep edge of the distribution a bin's mean
   can still creep past its edge. Such a bin keeps its particles just
   inside the edge and hands the fewest possible particles to the
   neighbour at the neighbour's mean mass (conservative; a few percent of
   that bin, unlike MNFIX's reset to the bin centre).
6. cap_top_bin (called by the caller, before MNFIX): no flux leaves
   the grid, so growth in the top bin stays there. Once its mean passes
   the top edge MNFIX would scale down every species of that bin
   (deleting seed mass); instead the caller takes back just enough of the
   mass it added in place (SOA -> vapour, nitrate -> reported).

For a uniform log-mass shift, number and mass move consistently: each
bin's mean stays at the same place in its bin step after step (the mode
itself spreads only by PPM's small numerical diffusion). Bins with
Nk <= n_min are not moved; a change they received stays in place.

References:
    - Colella & Woodward (1984), J. Comput. Phys. 54, 174-201 (PPM)
"""
import jax
# float64 enforced by core/config.py
import jax.numpy as jnp

from ..core.config import ICOMP_NODIAG
from .condensation_ppm import (
    _integrate_mass_parabola_left, _integrate_mass_parabola_right,
    _integrate_parabola_left, _integrate_parabola_right, advect_totals,
    dry_mass_from_ppm_number, ppm_reconstruct,
)

# How an in-place mass change moves particles between the fixed bins:
# this module's PPM remap, or MNFIX (the TOMAS split; kept for comparison)
BIN_TRANSFER_PPM = 'ppm'
BIN_TRANSFER_MNFIX = 'mnfix'
BIN_TRANSFERS = (BIN_TRANSFER_PPM, BIN_TRANSFER_MNFIX)

# Max Courant number per sub-step (same as ppm_condensation_step)
REMAP_C_MAX = 0.8

# Bins with Nk at or below this [#/cell] are not moved
REMAP_N_MIN = 1.0e-10

# keep_means_in_bins: a corrected mean sits this fraction of a bin width
# inside the edge it crossed
REMAP_MARGIN = 0.02

_TINY = 1.0e-30


def mean_log_shift(Nk, Mk_old, Mk_new, xk, icomp_nodiag=ICOMP_NODIAG,
                   n_min=REMAP_N_MIN):
    """Log-mass displacement of each bin's mean dry mass over the step.

    Returns:
        (shift, xi, active): shift = ln(m_new/m_old) of the bin mean
        (clipped to the grid span), xi = ln(m_old) of the bin mean (the
        log centre for inactive bins), active = Nk > n_min with dry mass.
    """
    dry_old = jnp.sum(Mk_old[:, :icomp_nodiag], axis=1)
    dry_new = jnp.sum(Mk_new[:, :icomp_nodiag], axis=1)
    active = (Nk > n_min) & (dry_old > 0.0)
    span = jnp.log(xk[-1] / xk[0])
    old_safe = jnp.where(active, dry_old, 1.0)
    shift = jnp.log(jnp.maximum(dry_new, _TINY) / old_safe)
    shift = jnp.where(active, jnp.clip(shift, -span, span), 0.0)
    xi_centre = 0.5 * (jnp.log(xk[:-1]) + jnp.log(xk[1:]))
    xi = jnp.where(active, jnp.log(old_safe / jnp.where(active, Nk, 1.0)),
                   xi_centre)
    return shift, xi, active


def edge_shifts(xk, shift, xi, active):
    """Upward and downward log-mass displacement at each bin edge.

    Upwind, from a slope-limited linear reconstruction of each bin's
    displacement in ln(m) around its mean (module doc, step 2).

    Args:
        xk: Bin boundaries [kg], shape (nbins+1,)
        shift, xi, active: from mean_log_shift

    Returns:
        (s_up, s_down), each shape (nbins+1,) and >= 0: s_up moves the
        lower bin's particles up, s_down the upper bin's particles down.
        Both are 0 at the domain edges.
    """
    ln_x = jnp.log(xk)
    dx = xi[1:] - xi[:-1]
    pair = active[:-1] & active[1:] & (dx > 0.0)
    d = jnp.where(pair, (shift[1:] - shift[:-1]) / jnp.where(pair, dx, 1.0),
                  0.0)
    zero = jnp.zeros(1)
    d_lo = jnp.concatenate([zero, d])       # to the bin below
    d_hi = jnp.concatenate([d, zero])       # to the bin above
    # the smaller one-sided slope: 0 next to a bin whose s did not change
    # or that has no particles, the interpolation slope where s is smooth
    slope = jnp.where(jnp.abs(d_lo) <= jnp.abs(d_hi), d_lo, d_hi)

    s_above = jnp.concatenate([shift[1:], shift[-1:]])
    s_below = jnp.concatenate([shift[:1], shift[:-1]])
    s_top = jnp.clip(shift + slope * (ln_x[1:] - xi),
                     jnp.minimum(shift, s_above), jnp.maximum(shift, s_above))
    s_bot = jnp.clip(shift + slope * (ln_x[:-1] - xi),
                     jnp.minimum(shift, s_below), jnp.maximum(shift, s_below))
    s_top = jnp.where(active, s_top, 0.0)
    s_bot = jnp.where(active, s_bot, 0.0)
    s_up = jnp.concatenate([zero, jnp.maximum(s_top[:-1], 0.0), zero])
    s_down = jnp.concatenate([zero, jnp.maximum(-s_bot[1:], 0.0), zero])
    return s_up, s_down


def remap_mass_change(Nk, Mk_old, Mk_new, xk, icomp_nodiag=ICOMP_NODIAG,
                      n_min=REMAP_N_MIN, c_max=REMAP_C_MAX):
    """Move particles whose mass changed in place to the right fixed bins.

    Args:
        Nk: Number per bin [#/cell], shape (nbins,) -- unchanged by the
            in-place process
        Mk_old: Mass per bin before the change [kg/cell], (nbins, icomp)
        Mk_new: Mass per bin after the change [kg/cell], (nbins, icomp)
        xk: Bin boundaries [kg], shape (nbins+1,), uniform in ln(m)
        icomp_nodiag: Dry (bin-defining) species count, as MNFIX
        n_min: Bins with Nk <= n_min are not moved [#/cell]
        c_max: Max Courant number per sub-step

    Returns:
        (Nk, Mk): remapped. sum(Nk) and sum(Mk, axis=0) equal the inputs'
        (Nk, Mk_new) to round-off.
    """
    nbins = Nk.shape[0]
    dxi = jnp.log(xk[1] / xk[0])

    shift, xi, active = mean_log_shift(Nk, Mk_old, Mk_new, xk, icomp_nodiag,
                                       n_min)
    s_up, s_down = edge_shifts(xk, shift, xi, active)
    # A bin can lose through its top (growth) and bottom (shrinking) edge
    # at once; keep the two departure regions apart: sum <= c_max.
    s_out = s_up[1:] + s_down[:-1]
    n_sub = jnp.maximum(jnp.ceil(jnp.max(s_out) / (c_max * dxi)), 1.0)
    c_up = s_up / (n_sub * dxi)          # per-sub-step Courant numbers
    c_down = s_down / (n_sub * dxi)

    # Donor of each edge: left bin for upward, right bin for downward flux
    edge = jnp.arange(nbins + 1)
    left = jnp.clip(edge - 1, 0, nbins - 1)
    right = jnp.clip(edge, 0, nbins - 1)
    m_left = xk[:-1][left]               # lower mass bound of the donor
    m_right = xk[:-1][right]

    def substep(_, carry):
        N, M = carry
        n_L, n_R, n_6 = ppm_reconstruct(N / dxi)
        m_tot = dry_mass_from_ppm_number(n_L, n_R, n_6, xk, dxi)
        has_m = m_tot > _TINY
        m_tot = jnp.where(has_m, m_tot, 1.0)

        cl = (n_L[left], n_R[left], n_6[left])
        cr = (n_L[right], n_R[right], n_6[right])
        fn_up = _integrate_parabola_right(*cl, c_up) * dxi
        fn_down = _integrate_parabola_left(*cr, c_down) * dxi
        fm_up = (_integrate_mass_parabola_right(*cl, m_left, c_up, dxi)
                 / m_tot[left])
        fm_down = (_integrate_mass_parabola_left(*cr, m_right, c_down, dxi)
                   / m_tot[right])
        # exact zeros where nothing moves (the integrals leave ~1 ulp)
        go_up = has_m[left] & (c_up > 0.0)
        go_down = has_m[right] & (c_down > 0.0)
        fn_up = jnp.where(c_up > 0.0, fn_up, 0.0)
        fn_down = jnp.where(c_down > 0.0, fn_down, 0.0)
        fm_up = jnp.where(go_up, jnp.clip(fm_up, 0.0, 1.0), 0.0)
        fm_down = jnp.where(go_down, jnp.clip(fm_down, 0.0, 1.0), 0.0)

        F_N = fn_up - fn_down
        F_M = fm_up[:, None] * M[left] - fm_down[:, None] * M[right]
        N = advect_totals(N, F_N, 1.0)
        M = advect_totals(M, F_M, 1.0)
        return jnp.maximum(N, 0.0), jnp.maximum(M, 0.0)

    Nk, Mk = jax.lax.fori_loop(0, n_sub.astype(jnp.int32), substep,
                               (Nk, Mk_new))
    return keep_means_in_bins(Nk, Mk, xk, icomp_nodiag, n_min)


def keep_means_in_bins(Nk, Mk, xk, icomp_nodiag=ICOMP_NODIAG,
                       n_min=REMAP_N_MIN, margin=REMAP_MARGIN):
    """Pull any bin mean that left its bin back inside, moving as few
    particles as possible to the neighbour (one vectorised pass).

    PPM reconstructs number only, so a bin's mean mass can creep past an
    edge where the number density is steep (the leading edge of a growing
    mode). MNFIX would then reset the bin to its centre and drop a large
    packet near the far side of the next bin. Here, a bin whose mean m is
    above x_{k+1} keeps its particles at mean t = x_{k+1} exp(-margin dxi)
    and sends n = N (m - t)/(u - t) particles up, each with the
    neighbour's mean mass u (clipped to at least margin inside the
    neighbour bin), so the neighbour's mean barely changes; the same
    downwards. If m is beyond u the whole bin moves. The moved
    particles carry the donor's composition. Number and every species are
    conserved exactly. Top-bin means above the grid and bottom-bin means
    below it are left (see cap_top_bin; MNFIX trims the number of a
    bottom bin whose mean is below the grid).
    """
    nbins = Nk.shape[0]
    dxi = jnp.log(xk[1] / xk[0])
    lo, hi = xk[:-1], xk[1:]
    dry = jnp.sum(Mk[:, :icomp_nodiag], axis=1)
    active = (Nk > n_min) & (dry > 0.0)
    m = jnp.where(active, dry / jnp.where(active, Nk, 1.0), jnp.sqrt(lo * hi))
    m_in = jnp.clip(m, lo, hi)               # a mean usable as a target

    k = jnp.arange(nbins)
    up = active & (m > hi) & (k < nbins - 1)
    down = active & (m < lo) & (k > 0)
    shrink = jnp.exp(-margin * dxi)
    t = jnp.where(up, hi * shrink, lo / shrink)
    u = jnp.where(up, jnp.maximum(jnp.roll(m_in, -1), hi / shrink),
                  jnp.minimum(jnp.roll(m_in, 1), lo * shrink))
    # u beyond m (overshoot past the neighbour's mean): move the whole bin
    frac_n = jnp.clip((m - t) / jnp.where(u != t, u - t, 1.0), 0.0, 1.0)
    frac_m = jnp.where(frac_n < 1.0, frac_n * u / m, 1.0)
    frac_n = jnp.where(up | down, frac_n, 0.0)
    frac_m = jnp.where(up | down, jnp.clip(frac_m, 0.0, 1.0), 0.0)

    dN = frac_n * Nk
    dM = frac_m[:, None] * Mk
    up_f, down_f = up[:, None], down[:, None]
    Nk = (Nk - dN + jnp.roll(jnp.where(up, dN, 0.0), 1)
          + jnp.roll(jnp.where(down, dN, 0.0), -1))
    Mk = (Mk - dM + jnp.roll(jnp.where(up_f, dM, 0.0), 1, axis=0)
          + jnp.roll(jnp.where(down_f, dM, 0.0), -1, axis=0))
    return jnp.maximum(Nk, 0.0), jnp.maximum(Mk, 0.0)


def cap_top_bin(Nk, Mk, xk, cols, icomp_nodiag=ICOMP_NODIAG,
                n_min=REMAP_N_MIN, margin=REMAP_MARGIN):
    """Take mass of the columns ``cols`` out of the top bin until its mean
    dry mass is back inside the grid.

    The remap has no flux through the top edge of the grid, so particles
    growing in the top bin stay there. When the top bin's mean passes
    xk[-1], MNFIX would scale every species of that bin down to a mean of
    xk[-1]/1.2, deleting seed mass as well. This removes only the given
    in-place species (proportionally, at most all of them), down to a mean
    margin inside the top edge; the caller decides where it goes (SOA
    returns to the vapour, nitrate is reported as not added).

    Args:
        Nk, Mk, xk: as remap_mass_change
        cols: Static (start, stop) column range of the in-place species,
            all dry (< icomp_nodiag)
        icomp_nodiag, n_min, margin: as keep_means_in_bins

    Returns:
        (Mk, removed): removed [kg/cell] has shape (stop - start,).
    """
    c0, c1 = cols
    dxi = jnp.log(xk[1] / xk[0])
    n_top = Nk[-1]
    dry = jnp.sum(Mk[-1, :icomp_nodiag])
    over = (n_top > n_min) & (dry > n_top * xk[-1])
    excess = jnp.where(over, dry - n_top * xk[-1] * jnp.exp(-margin * dxi),
                       0.0)
    avail = Mk[-1, c0:c1]
    tot = jnp.sum(avail)
    frac = jnp.where(tot > 0.0,
                     jnp.minimum(excess / jnp.where(tot > 0.0, tot, 1.0), 1.0),
                     0.0)
    removed = frac * avail
    return Mk.at[-1, c0:c1].add(-removed), removed
