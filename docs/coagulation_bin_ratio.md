# Coagulation on Non-Doubling Bin Grids

## Problem

TOMAS coagulation (`multicoag.f`, ported as `calc_coagulation_rates_tfl`) uses the
Tzivion, Feingold & Levin (1987) two-moment algorithm. Its transfer terms assume
**mass-doubling bins** (`xk[k+1] = 2·xk[k]`, the 40-bin default):

1. Two particles from bin k form a particle in bin k+1. True only for p = 2: the
   product mass lies in [2·xk[k], 2·xk[k+1]), which is bin k+2 at p = √2 (80 bins)
   and bin k+4 at p = 2^(1/4) (160 bins).
2. A collision with a smaller particle moves a particle at most one bin. At p < 2,
   partners from the few bins just below k push a particle beyond k+1.
3. The φ/η (`phi`/`eff`) endpoint densities and the `1/(2·xk[k])` terms use x_k as
   the bin width. The width is (p−1)·x_k, so for p ≠ 2 the number crossing into the
   next bin is undercounted by a factor of (p−1): 0.41× at 80 bins, 0.19× at 160.

Commit `950d151` (March 2026) generalized ζ and added a `/(p−1)` to φ/η, but it
could not fix (1) and (2) and only half-fixed (3). The result was size
distributions that oscillate more as resolution increases. Total number and mass
stay correct, which is why budget checks never flagged it. Correcting the width
alone (3) smooths 80 bins but leaves 160 bins broken.

## Fix: linear sub-bin scheme

`tomas_jax/physics/coagulation_rates_linear.py::calc_coagulation_rates_linear`
makes no assumption about the bin ratio. For each pair of bins (j, i) with i ≤ j:

- **Reconstruction.** Particles in the larger bin j get a non-negative linear
  number density on [xk[j], xk[j+1]] that reproduces N_j and the mean mass x̄_j
  (`linear_subbin_reconstruction`; the "linear discrete method" of Simmel et al.,
  2002):
  - x̄ within w/6 of the bin centre: linear across the whole bin;
  - x̄ closer to an edge: a ramp to zero on a sub-interval;
  - x̄ at or outside an edge: a narrow uniform around x̄.
- **Products.** Partner particles are taken at x̄_i. The products x + x̄_i fill bin
  j's support shifted up by x̄_i. Bin widths grow with mass, so this interval spans
  **at most two** destination bins. The number and the bin-j mass landing in each
  are found by integrating the linear density exactly (Simpson's rule is exact for
  x·n(x)). The partner's mass goes along with each product.
- **Losses** are exact: K_ij N_i N_j collisions remove one particle (at the bin's
  mean composition) from each partner bin, and ½ K_jj N_j² for self-coagulation.

Per-species mass is conserved to round-off (products above the grid go to
`dM_overflow`), and each collision removes exactly one particle. The function has
the same signature and returns as the TFL version.

## Dispatch

`calc_coagulation_rates` chooses with `jax.lax.cond` on `xk[1]/xk[0]`:

| Grid | Ratio p | Scheme |
|------|---------|--------|
| 40 bins (default) | 2 | TFL, bit-identical to before and to Fortran `multicoag.f` |
| 80 / 160 / 320 bins | √2, 2^(1/4), 2^(1/8) | linear sub-bin |
| fewer than 40 bins | > 2 | linear sub-bin (TFL is also invalid for p > 2) |

Every solver path (`coag_euler_step`, `diffrax_step`, `make_step`, scan loops,
`utils/diagnostics.py`) calls `calc_coagulation_rates`, so no call sites changed.
Under `jit` the dispatcher's output is bit-identical to calling the selected scheme
directly.

## Validation

`python -m benchmarks.python.coag_bin_ratio_convergence`. Coag-only, 24 h, dt = 60 s,
3 Euler substeps + MNFIX, grid 1.7 nm start with ratio 2^(40/nbins) (tomas-api setup).

**Constant kernel vs exact Smoluchowski solution** (exponential start, N falls 6×).
L1 error of dN/dlogDp:

| bins | TFL L1 | TFL peak/exact | TFL extrema | linear L1 | linear peak/exact | linear extrema |
|------|--------|----------------|-------------|-----------|-------------------|----------------|
| 40   | 4.6%   | 0.955          | 1           | 0.47%     | 1.002             | 1              |
| 80   | 18.6%  | 1.277          | 7           | 0.13%     | 1.001             | 1              |
| 160  | 28.8%  | 1.633          | 21          | 0.06%     | 1.000             | 1              |
| 320  | 27.2%  | 1.565          | 32          | 0.05%     | 1.000             | 1              |

**Brownian kernel, tomas-web case** (N = 1e5 cm⁻³, GMD 100 nm, GSD 1.6):

| bins | TFL N (cm⁻³) | TFL extrema | linear N (cm⁻³) | linear peak | linear extrema |
|------|--------------|-------------|-----------------|-------------|----------------|
| 40   | 1.6569e4     | 1           | 1.6580e4        | 4.03e4      | 1              |
| 80   | 1.6645e4     | 3           | 1.6559e4        | 4.09e4      | 1              |
| 160  | 1.6731e4     | 23          | 1.6554e4        | 4.11e4      | 1              |
| 320  | 1.6770e4     | 37          | 1.6553e4        | 4.11e4      | 1              |

Only the TFL entries at 40 bins reflect current behavior. TFL at p ≠ 2 is what the
code did before the dispatch.

Tests: `tests/test_coagulation_bin_ratio.py` covers the reconstruction, conservation,
analytic convergence, dispatch, and a Brownian regression that fails if TFL is forced
at 80/160 bins.

## Known limitations

- **Cost.** The linear scheme takes about 1.4× TFL's wall time (CPU, 24 h coag-only:
  2.2 s vs 1.6 s at 80 bins, 6.9 s vs 4.6 s at 160). Destination bins come from
  `jnp.searchsorted`; a closed-form log index for geometric grids would be faster.
- **Partner treated at its mean mass.** The scheme is first order in the partner bin
  width, which is negligible for fine grids. At p = 2 it is still 10× more accurate
  than TFL on the constant-kernel test, but 40 bins stays on TFL for Fortran parity.
- **Bin widths must not decrease with size** (true for any constant-ratio grid);
  that is what limits each pair's products to two destination bins.
- **Mass budget creep from empty-bin placeholders** (pre-existing, scheme-independent).
  `_preprocess_concentrations` gives empty bins NEPS = 1e-3 placeholder particles.
  Their coagulation gains count but their losses are clipped by the positivity
  clamp, so `M(t) + overflow` drifts upward in proportion to the number of empty
  bins. Both schemes give identical budgets: 6e-9 (40 bins) to 5e-8 (320 bins) in
  the constant-kernel case, ≤ 2e-9 in the Brownian case.
- **GPU-fast model** (`gpu-fast` branch, not on dev). `fast/coagulation.py` calls
  `calc_coagulation_rates` and gets the dispatch on merge. The fused Pallas kernel
  (`fast/coagulation_pallas.py`) hard-codes TFL and a base-2 MNFIX, so it stays
  valid only for 40 bins.
