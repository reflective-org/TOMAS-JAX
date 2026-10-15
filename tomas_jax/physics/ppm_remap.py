"""PPM remap of an in-place particle mass change onto the fixed bins (JIT).

Some processes change the mass of the particles in each bin without moving
them: SOA partitioning (vbs-jax adds or removes class mass per bin) and the
N2O5-uptake nitrate source. Afterwards a bin's mean mass can sit outside
the bin. ``remap_mass_change`` moves the particles to the bins they now
belong to, using the PPM sub-bin profile of H2SO4 condensation
(physics/condensation_ppm.py::ppm_reconstruct) instead of MNFIX's
two-packet split.

Algorithm (one call per in-place change, e.g. per 60 s micro-step):

1. Bin displacement. Each bin with particles (Nk > n_min, dry mass > 0)
   moved its mean dry mass from m to m' = m + dm: a log-mass displacement
   s_k = ln(m'/m) at xi_k = ln(m). dm < 0 (evaporation) gives s_k < 0.
2. Displacement inside a bin. s is taken as linear in ln(m) through the
   bin mean, with the smaller (in magnitude) of the two one-sided slopes
   to the neighbouring bin means (0 next to a neighbour without particles
   or with the same s), and the bin-edge values are clipped between the
   bin's s and the neighbour's. The bin [xi_lo, xi_hi] therefore maps onto
   its image [xi_lo + s_bot, xi_hi + s_top]: where s varies smoothly the
   images of neighbouring bins join (stretched for evaporation, squeezed
   for growth); at a jump in s (two particle populations side by side)
   each bin keeps its own s, so particles whose mass did not change stay
   exactly where they are while others grow or shrink past them. An image
   narrower than REMAP_MIN_STRETCH bin widths is replaced by a rigid
   shift by s_k (it would otherwise fold).
3. Deposit (Lagrangian remap). Number inside each source bin follows its
   PPM reconstruction (ppm_reconstruct: monotone and positive parabola in
   ln m), carried linearly onto the image. Target bin j receives from
   source bin k the integral of that profile over the part of the image
   inside j (polynomial integral) and the mass share with weight
   exp(xi') (the particle's new mass; closed forms phi_j below). Each
   source bin moves with its own displacement however many bins that is
   (no Courant limit, no sub-steps); for a uniform shift smaller than one
   bin this is exactly the flux-form PPM step of condensation_ppm (the
   departure-region integrals). Particles mapped below/above the grid go
   into the bottom/top bin.
4. Composition. Every species of a source bin moves with its mass share.
   With composition slopes (default) each species' mass fraction is also
   linear inside the source bin (monotonized-central slope against the
   neighbouring bins' fractions, slopes summing to 0, one common limiter
   so no fraction goes negative), so particles keep the composition that
   goes with their size: a grow-then-evaporate round trip comes back to
   the moving-section answer instead of mixing compositions bin by bin.
   Each source bin's transfer sums to 1 and its slope term to 0, so number
   and every species are conserved to round-off; nothing goes negative.
5. keep_means_in_bins (safety net): PPM reconstructs number only, so a
   bin's mean mass can still land past an edge of its target bin. Such a
   bin hands the fewest possible particles to the neighbour (see there).
   In the growth/evaporation test cases the particles it moves add up to
   at most ~1e-3 of the total over a whole run.
6. cap_top_bin (called by the caller, before MNFIX): no mass leaves the
   grid, so growth in the top bin stays there. Once its mean passes the
   top edge MNFIX would scale down every species of that bin (deleting
   seed mass); instead the caller takes back just enough of the mass it
   added in place (SOA -> vapour, nitrate -> reported).

Each bin's image is kept between REMAP_MIN_STRETCH and REMAP_MAX_STRETCH
bin widths (a rigid shift if it would fold, squeezed around the image of
its mean if wider), so it touches at most _BAND target bins: the work is
a (nbins, _BAND) band per call, no loops, and the call is skipped when no
bin changed. Bins with Nk <= n_min are not moved; a change they received
stays in place.

References:
    - Colella & Woodward (1984), J. Comput. Phys. 54, 174-201 (PPM)
    - van Leer (1977), J. Comput. Phys. 23, 276-299 (MC limiter)
"""
import math

import jax
# float64 enforced by core/config.py
import jax.numpy as jnp

from ..core.config import ICOMP_NODIAG
from .condensation_ppm import ppm_reconstruct

# How an in-place mass change moves particles between the fixed bins:
# this module's PPM remap, or MNFIX (the TOMAS split; kept for comparison)
BIN_TRANSFER_PPM = 'ppm'
BIN_TRANSFER_MNFIX = 'mnfix'
BIN_TRANSFERS = (BIN_TRANSFER_PPM, BIN_TRANSFER_MNFIX)

# Bins with Nk at or below this [#/cell] are not moved
REMAP_N_MIN = 1.0e-10

# keep_means_in_bins: a corrected mean sits this fraction of a bin width
# inside the edge it crossed
REMAP_MARGIN = 0.02

# A bin's image is at least REMAP_MIN_STRETCH bin widths wide (else a
# rigid shift) and at most REMAP_MAX_STRETCH (else squeezed to it), so it
# touches at most _BAND target bins
REMAP_MIN_STRETCH = 0.25
REMAP_MAX_STRETCH = 4
_BAND = REMAP_MAX_STRETCH + 2

# phi_j(z): Taylor series below this z (closed forms above); 17 terms
# leave < 1e-16 relative error at z = 1
_PHI_SERIES_Z = 1.0
_PHI_COEF = [[1.0 / (math.factorial(n) * (n + j + 1)) for n in range(17)]
             for j in range(4)]

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


def bin_edge_shifts(xk, shift, xi, active):
    """Log-mass displacement at each bin's lower and upper edge (module
    doc, step 2).

    Args:
        xk: Bin boundaries [kg], shape (nbins+1,)
        shift, xi, active: from mean_log_shift

    Returns:
        (s_bot, s_top), each shape (nbins,); 0 for inactive bins.
    """
    ln_x = jnp.log(xk)
    dx = xi[1:] - xi[:-1]
    pair = active[:-1] & active[1:] & (dx > 0.0)
    d = jnp.where(pair, (shift[1:] - shift[:-1]) / jnp.where(pair, dx, 1.0),
                  0.0)
    zero = jnp.zeros(1)
    d_lo = jnp.concatenate([zero, d])       # to the bin below
    d_hi = jnp.concatenate([d, zero])       # to the bin above
    # minmod: the smaller one-sided slope where both sides agree in sign
    # (s smooth), 0 where they disagree (the bin's s is a local extremum,
    # e.g. unchanged particles between growing neighbours, which must stay
    # in place) or next to a bin with the same s or no particles
    slope = jnp.where(d_lo * d_hi > 0.0,
                      jnp.where(jnp.abs(d_lo) <= jnp.abs(d_hi), d_lo, d_hi),
                      0.0)

    s_above = jnp.concatenate([shift[1:], shift[-1:]])
    s_below = jnp.concatenate([shift[:1], shift[:-1]])
    s_top = jnp.clip(shift + slope * (ln_x[1:] - xi),
                     jnp.minimum(shift, s_above), jnp.maximum(shift, s_above))
    s_bot = jnp.clip(shift + slope * (ln_x[:-1] - xi),
                     jnp.minimum(shift, s_below), jnp.maximum(shift, s_below))
    return jnp.where(active, s_bot, 0.0), jnp.where(active, s_top, 0.0)


def _phi(z):
    """phi_j(z) = int_0^1 v^j exp(z v) dv for j = 0..3, z >= 0.

    Taylor series sum_n z^n / (n! (n + j + 1)) (Horner) for
    z < _PHI_SERIES_Z, where the closed forms cancel; else
    phi_0 = expm1(z)/z and phi_j = (exp(z) - j phi_{j-1}) / z.
    """
    small = z < _PHI_SERIES_Z
    zs = jnp.where(small, z, 0.0)
    ser = []
    for j in range(4):
        acc = jnp.full_like(z, _PHI_COEF[j][-1])
        for c in _PHI_COEF[j][-2::-1]:
            acc = acc * zs + c
        ser.append(acc)
    zl = jnp.where(small, 1.0, z)
    ez = jnp.exp(zl)
    inv = 1.0 / zl
    closed = [jnp.expm1(zl) * inv]
    for j in range(1, 4):
        closed.append((ez - j * closed[-1]) * inv)
    return [jnp.where(small, a, c) for a, c in zip(ser, closed)]


def _minmod3(a, b, c):
    """Smallest-magnitude argument if all three share a sign, else 0."""
    same = ((a > 0) & (b > 0) & (c > 0)) | ((a < 0) & (b < 0) & (c < 0))
    m = jnp.minimum(jnp.minimum(jnp.abs(a), jnp.abs(b)), jnp.abs(c))
    return jnp.where(same, jnp.sign(a) * m, 0.0)


def composition_slopes(Mk, active, u_c):
    """Slope of every species' mass fraction inside each bin (module doc,
    step 4), per bin width.

    Monotonized-central limited against the neighbouring bins' fractions
    (0 at the edge of a plateau, next to an empty bin and at the ends of
    the grid), shifted so the slopes of a bin sum to 0 (the fractions keep
    summing to 1), then scaled by one factor per bin so that no fraction
    is negative at either bin edge, u = 0 or 1, for a fraction that is
    the bin mean at u = u_c.

    Args:
        Mk: Mass per bin [kg/cell], shape (nbins, icomp)
        active: Bins that move, shape (nbins,)
        u_c: Mass centroid of each bin in [0, 1], shape (nbins,)

    Returns:
        Slopes, shape (nbins, icomp).
    """
    tot = jnp.sum(Mk, axis=1, keepdims=True)
    f = Mk / jnp.where(tot > 0.0, tot, 1.0)
    act = active[:, None]
    f_lo = jnp.concatenate([f[:1], f[:-1]])
    f_hi = jnp.concatenate([f[1:], f[-1:]])
    no = jnp.zeros((1, 1), dtype=bool)
    both = (act & jnp.concatenate([no, act[:-1]])
            & jnp.concatenate([act[1:], no]))
    sig = jnp.where(both, _minmod3(0.5 * (f_hi - f_lo), 2.0 * (f - f_lo),
                                   2.0 * (f_hi - f)), 0.0)
    sig = sig - f * jnp.sum(sig, axis=1, keepdims=True)
    uc = u_c[:, None]
    room = jnp.where(sig > 0.0, f / jnp.maximum(sig * uc, _TINY),
                     jnp.where(sig < 0.0,
                               f / jnp.maximum(-sig * (1.0 - uc), _TINY),
                               jnp.inf))
    theta = jnp.clip(jnp.min(room, axis=1, keepdims=True), 0.0, 1.0)
    return theta * sig


def remap_mass_change(Nk, Mk_old, Mk_new, xk, icomp_nodiag=ICOMP_NODIAG,
                      n_min=REMAP_N_MIN, slopes=True):
    """Move particles whose mass changed in place to the right fixed bins.

    Args:
        Nk: Number per bin [#/cell], shape (nbins,) -- unchanged by the
            in-place process
        Mk_old: Mass per bin before the change [kg/cell], (nbins, icomp)
        Mk_new: Mass per bin after the change [kg/cell], (nbins, icomp)
        xk: Bin boundaries [kg], shape (nbins+1,), uniform in ln(m)
        icomp_nodiag: Dry (bin-defining) species count, as MNFIX
        n_min: Bins with Nk <= n_min are not moved [#/cell]
        slopes: Static; linear composition inside each source bin (module
            doc, step 4). False moves every species with the source bin's
            mean composition.

    Returns:
        (Nk, Mk): remapped. sum(Nk) and sum(Mk, axis=0) equal the inputs'
        (Nk, Mk_new) to round-off.
    """
    shift, xi, active = mean_log_shift(Nk, Mk_old, Mk_new, xk, icomp_nodiag,
                                       n_min)
    # nothing to move (e.g. no nitrate source this step): skip the work
    return jax.lax.cond(
        jnp.any(active & (shift != 0.0)),
        lambda: _remap(Nk, Mk_new, xk, shift, xi, active, icomp_nodiag,
                       n_min, slopes),
        lambda: (Nk, Mk_new))


def _remap(Nk, Mk_new, xk, shift, xi, active, icomp_nodiag, n_min, slopes):
    """remap_mass_change after mean_log_shift (module doc, steps 2-5)."""
    nbins = Nk.shape[0]
    ln_x = jnp.log(xk)
    dxi = ln_x[1] - ln_x[0]
    s_bot, s_top = bin_edge_shifts(xk, shift, xi, active)
    # image [a, b] of each bin in ln m: a rigid shift where it would fold,
    # squeezed to REMAP_MAX_STRETCH bin widths around the image of the bin
    # mean where wider
    a = ln_x[:-1] + s_bot
    b = ln_x[1:] + s_top
    stretch = (b - a) / dxi
    rigid = stretch < REMAP_MIN_STRETCH
    wide = stretch > REMAP_MAX_STRETCH
    mid = xi + shift
    a = jnp.where(rigid, ln_x[:-1] + shift,
                  jnp.where(wide, mid - REMAP_MAX_STRETCH * (xi - ln_x[:-1]),
                            a))
    b = jnp.where(rigid, ln_x[1:] + shift,
                  jnp.where(wide, mid + REMAP_MAX_STRETCH * (ln_x[1:] - xi),
                            b))
    width = b - a

    # The image lies within _BAND consecutive bins of the (extended) grid,
    # starting with the one holding a; bins beyond the grid map onto the
    # bottom/top bin. u = pre-image in [0, 1] of the band's inner edges
    # (the grid's own edge values inside the grid, so a bin that does not
    # move maps onto itself exactly).
    e0 = jnp.floor((a - ln_x[0]) / dxi).astype(jnp.int32)
    e = e0[:, None] + jnp.arange(1, _BAND)[None, :]
    x_e = jnp.where(e < 0, ln_x[0] + e * dxi,
                    jnp.where(e > nbins, ln_x[-1] + (e - nbins) * dxi,
                              ln_x[jnp.clip(e, 0, nbins)]))
    u = jnp.clip((x_e - a[:, None]) / width[:, None], 0.0, 1.0)
    one = jnp.ones((nbins, 1))
    u = jnp.concatenate([0.0 * one, u, one], axis=1)       # (nbins, _BAND+1)
    k = jnp.arange(nbins)[:, None]
    target = jnp.where(active[:, None],
                       jnp.clip(e0[:, None] + jnp.arange(_BAND)[None, :],
                                0, nbins - 1),
                       k)

    # PPM number profile p(v) = n_L + n_b v - n_6 v^2 of each source bin;
    # cumulative number int_0^u p, mass int_0^u p e^{w v} (w = width: the
    # particle at v has new mass e^{a + w v}) and its first moment in v
    n_L, n_R, n_6 = ppm_reconstruct(Nk / dxi)
    n_L, n_b, n_6 = n_L[:, None], (n_R - n_L + n_6)[:, None], n_6[:, None]
    cum_n = u * (n_L + u * (0.5 * n_b - u * n_6 / 3.0))
    f0, f1, f2, f3 = _phi(width[:, None] * u)
    cum_m = u * (n_L * f0 + u * (n_b * f1 - u * n_6 * f2))

    first = jnp.zeros(_BAND).at[0].set(1.0)

    def shares(cum):
        t = jnp.maximum(jnp.diff(cum, axis=1), 0.0)
        tot = jnp.sum(t, axis=1, keepdims=True)
        t = t / jnp.where(tot > 0.0, tot, 1.0)
        return jnp.where(active[:, None], t, first)    # inactive: stay

    rows = target.ravel()
    icomp = Mk_new.shape[1]
    Nk = jnp.zeros(nbins).at[rows].add((Nk[:, None] * shares(cum_n)).ravel())
    dM = shares(cum_m)[:, :, None] * Mk_new[:, None, :]
    if slopes:
        # first moment of each target's share about the source's mass
        # centroid u_c (sums to 0 per source): carries the composition
        # slopes
        cum_v = u * u * (n_L * f1 + u * (n_b * f2 - u * n_6 * f3))
        w_tot = jnp.where(cum_m[:, -1] > 0.0, cum_m[:, -1], 1.0)
        u_c = jnp.clip(cum_v[:, -1] / w_tot, 0.0, 1.0)
        t_v = (jnp.diff(cum_v, axis=1)
               - u_c[:, None] * jnp.diff(cum_m, axis=1)) / w_tot[:, None]
        t_v = jnp.where(active[:, None], t_v, 0.0)
        sig = composition_slopes(Mk_new, active, u_c)
        dM = dM + t_v[:, :, None] * (jnp.sum(Mk_new, axis=1)[:, None]
                                     * sig)[:, None, :]
    Mk = jnp.maximum(
        jnp.zeros_like(Mk_new).at[rows].add(dM.reshape(-1, icomp)), 0.0)
    return keep_means_in_bins(Nk, Mk, xk, icomp_nodiag, n_min)


def keep_means_in_bins(Nk, Mk, xk, icomp_nodiag=ICOMP_NODIAG,
                       n_min=REMAP_N_MIN, margin=REMAP_MARGIN):
    """Pull any bin mean that left its bin back inside, moving as few
    particles as possible to the neighbour (one vectorised pass).

    PPM reconstructs number only, so a bin's mean mass can land past an
    edge where the number density is steep. MNFIX would then reset the bin
    to its centre and drop a large packet near the far side of the next
    bin. Here, a bin whose mean m is
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
