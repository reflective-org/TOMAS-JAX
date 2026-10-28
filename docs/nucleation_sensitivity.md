# Nucleation Coefficient Sensitivity (forward-mode AD)

Gradient-based sensitivity of stratospheric aerosol to the `ricco_dunne` fit
coefficients. It uses JAX forward-mode differentiation instead of a large
perturbed-parameter ensemble.

Work plan, by phase:

| Phase | What | Status |
|---|---|---|
| 0 | Warm-up: rate-level and 1 h full-model gradients, checked against analytic forms and finite differences | **done** |
| 1 | SABRE-binned dry N(Dp) output operator (Lyu et al. 2026) | **done** |
| 2 | σ-scaled sensitivity map over T × SO₂ (10 pptv to 3000 ppmv), chemistry on, 40 and 80 bins | planned |
| 3 | Background and SAI-plume scenarios (48 h), with FD checks | planned |
| 4 | Linearization check with a small ensemble | planned |

Code: `sensitivities/dunne_nuc/`. Results: `sensitivities/dunne_nuc/results/`.

## Differentiable coefficients

`tomas_jax.physics.nucleation.NucleationParams` is a NamedTuple pytree with 25 fields:

- Dunne 2016 (Table S1): `p_bn u_bn v_bn w_bn`, `p_tn u_tn v_tn w_tn p_an a_n`,
  `p_bi u_bi v_bi w_bi`, `p_ti u_ti v_ti w_ti p_ai a_i`. The rate constant is
  `k = exp(u − exp(v·(T/1000 − w)))`.
- Riccobono 2014 / Yu 2017: `k_m`, `delta_h`.
- Nucleated cluster: `r_nuc_nm`, `rho_nuc`, `org_frac`. The SO₄ fraction is
  `1 − org_frac`.

```python
import jax, jax.numpy as jnp
from tomas_jax.physics.nucleation import DEFAULT_NUCLEATION_PARAMS
from tomas_jax.solvers.condensation import make_step

step = make_step(['so2_chemistry', 'nucleation', 'coagulation', 'condensation'],
                 jit=False)
P = jax.tree.map(jnp.float64, DEFAULT_NUCLEATION_PARAMS)

def n_total(p):
    def body(c, _):
        return step(*c, xk, temp, pres, boxvol, rh, alpha, dt, oh_conc=1e6,
                    org_conc=0.0, nh3_conc=0.0, fion=10.0, nucl_params=p), None
    (Nk, _, _), _ = jax.lax.scan(body, (Nk0, Mk0, Gc0), None, length=60)
    return jnp.sum(Nk)

dN_dP = jax.jit(jax.jacfwd(n_total))(P)   # a NucleationParams of derivatives
```

## Forward mode only

The nucleation sub-step loop in `make_step` is `lax.fori_loop(0, n_nuc, ...)`,
and `n_nuc` is a traced value. JAX lowers that to a `while_loop`, so reverse mode
fails:

```
ValueError: Reverse-mode differentiation does not work for lax.while_loop or
lax.fori_loop with dynamic start/stop values.
```

Use `jax.jacfwd` or `jax.jvp`. Forward mode costs about one forward run per
parameter (25 here), whatever the number of outputs. Reverse mode would cost about
one run per output, and Phases 1–4 have many outputs (SABRE bins × time). Forward
mode also avoids storing the scan history.

### Enabling reverse mode later

Reverse mode pays off only for a scalar objective with many parameters, such as
calibrating coefficients (or time-varying or per-bin inputs) against SABRE data
through one misfit. Two dynamic loops would then need converting:

1. **Nucleation sub-steps** ([condensation.py](../tomas_jax/solvers/condensation.py),
   `make_step`, `_full_step_core`, `condensation_step_with_nucleation_jax`, and
   the scan-fused loops). Replace `fori_loop(0, n_nuc, ...)` with a fixed-length
   `lax.scan` over `max_nuc_substeps`, and keep the result of iterations with
   `i >= n_nuc` via `jnp.where`. Results are unchanged. Every step then always
   runs `max_nuc_substeps` nucleation + MNFIX passes (instead of usually 1), so
   the forward run gets slower.
2. **PPM condensation CFL sub-steps**
   ([condensation_ppm.py](../tomas_jax/physics/condensation_ppm.py), around line 937
   and the number-only variant). `n_sub` is traced and **uncapped**, so a
   fixed-length scan needs a hard maximum. Wherever the CFL condition needs more
   substeps than that cap allows, results change. This is a numerics change: it
   must be validated against the current solver (convergence benchmark,
   PPM ≠ TFL check) before use.

Static-bound loops (MNFIX over bins, TFL remap, the scan-based coagulation Euler
step) are already reverse-differentiable. Over a 48 h run, wrap each step in
`jax.checkpoint` to bound memory: the stored history would otherwise be
steps × substeps × state.

## What gradients cannot see

- **Integer and branch decisions.** These include the nucleation sub-step count
  (`ceil`), the gas-exhaustion clamp in `nucleation_step`, and MNFIX target-bin
  selection. Inside a branch the derivative is exact; across a switch it misses
  the jump. `sensitivities/dunne_nuc/common.make_runner` records `n_sub` and the
  clamp flag every step, so every sensitivity can be tagged with its regime.
- **No RH dependence in Dunne 2016.** The parameterization as implemented ignores
  RH, so binary H₂SO₄–H₂O sensitivity to humidity is structurally absent.
- **Correlated coefficients.** u, v and w for one mechanism come from a joint fit.
  Without a published covariance, per-parameter σ-scaled sensitivities must not
  be summed into a total variance.

## Phase 0 results (2026-10-23)

Run with `python -m sensitivities.dunne_nuc.phase0_warmup [--part 0a|0b|0c]`.
Conditions: T = 217 K, p = 55 hPa, fion = 10 cm⁻³ s⁻¹, NH₃ = 0.

**0a: input gradients.**
- `jax.grad` of ln Jbn and ln Jbi with respect to T and ln[H₂SO₄] matches the
  closed forms to machine precision and finite differences to about 1e-10.
- At 217 K: ∂lnJbn/∂T = −0.21 K⁻¹, so a 10 K cooling raises Jbn about 8×.
  ∂lnJbi/∂T = −0.056 K⁻¹, about 1.7× per 10 K.
- The binary-neutral T-sensitivity grows with temperature across 190–240 K
  (`results/phase0/part0a_dlnJ_dT.png`).

**0b: coefficient gradients.**
- `jax.jacfwd` with respect to `NucleationParams` matches the analytic identities
  (∂lnJbn/∂u = 1, ∂/∂p = ln cna, ∂/∂v, ∂/∂w).
- Every rate coefficient matches finite differences of the log of its own
  mechanism to within 2e-9.
- Tangents are finite at NH₃ = fion = org = 0. This needed double-`where` guards
  on the NH₃ branch and the ion sqrt.
- Compare FD against each mechanism's own rate, not ln(J_total). Minor mechanisms
  move ln(J_total) by about 1e-8, which is below the float64 FD round-off floor.
- Finding: at 217 K the Yu 2017 temperature correction to Riccobono, extrapolated
  from 278 K, scales the rate by about 3×10⁸. Even 1e5 cm⁻³ of organics then makes
  Riccobono account for essentially all of J. Stratospheric runs should set
  `org_conc = 0`, or treat this extrapolation as a separate uncertainty.

**0c: 1 h through the full model** (SO₂ chemistry + nucleation + coagulation +
condensation, 40 bins, SO₂ 1 ppbv, background 10 cm⁻³ at 100 nm).
- `jacfwd` with respect to all 25 fields took about 2 s including compile.
- d ln N/dθ for `u_bn`, `p_bn` and `u_bi` matches central FD to 2e-9.
- No sub-stepping or clamping occurred in this regime.
- Elasticities d ln N_total/d ln θ, top four: `v_bn` −13.0, `w_bi` 7.9,
  `u_bi` −7.9, `r_nuc_nm` 2.8.
- Elasticities overstate the importance of coefficients sitting inside an
  exponent, so the ranking needs Phase 2's σ-scaling before it means anything.
- Finding: ∂ln N/∂ln r_nuc ≈ 2.8 and ∂ln N/∂ln ρ_nuc ≈ 0.94, close to 3 and 1.
  The nucleated cluster (3.47e-24 kg) is lighter than the bin-0 lower edge
  (4.55e-24 kg; ratio 0.76). MNFIX keeps the added mass and resets the number,
  so only **64% of J·V·dt survives as new particles**, measured on the 40-bin
  grid at 217 K. New-particle number is therefore mass-limited, N ∝ m_nuc, and
  the cluster size and density act as a direct multiplier on the particle
  formation rate.

## Phase 1: SABRE-binned output (2026-10-23)

`sensitivities/dunne_nuc/sabre_bins.py` maps the model distribution onto the
SABRE 2023 measurement product described in Lyu et al. (2026, *Science*,
doi:10.1126/science.adw8939; PDFs in `references/sabre/`):

- `rebin_to_sabre(Nk, Mk, xk, boxvol)` gives dry dN/dlogDp [cm⁻³] on 20
  log-spaced bins per decade from 3 nm, 63 bins up to 4.24 µm.
  - Each model bin's number is spread uniformly in log Dp between its dry-mass
    edges, converted to Dp with the bin's dry density.
  - The result is linear in Nk and differentiable with respect to Nk and Mk.
- `nmass_cumulative(...)` gives N(Dp > 3, 8, 12, 27, 53 nm), the NMASS channels.
- `water_frac` adds residual water by mass. SABRE heats the sample and leaves
  less than 10% water, uncorrected.
- `SABRE_NUMBER_REL_ERR = 0.08` is the inversion number uncertainty (SuppMat
  Table S2).

Assumptions and limitations:
- **Bin edges are assumed.** The paper says "20 logarithmically spaced size
  intervals per decade" from 0.003 to about 4 µm but lists no edges. Replace
  `SABRE_EDGES_M` with the edges from the SABRE data files when available; all
  functions accept `edges=`.
- **NMASS channels are sharp cutoffs.** The real channels have smooth
  counting-efficiency curves (SuppMat Fig. S19, not tabulated in the paper).
- **Dry density for sizing.**
  - NH₄-free bins (the stratospheric case) use pure H₂SO₄ from the CRC table,
    `h2so4_solution_density(100)` = 1836 kg/m³.
  - `calc_density` would give 1902 here, for two reasons. It counts SO₄ mass as
    H₂SO₄ (×98/96), which puts dry particles at 102 wt%. And its Tang (1997)
    fit is about 3% above CRC at 100 wt% (1890 vs 1836).
  - Bins containing NH₄ use `calc_density`, with the acid protons added as
    solute so the dry weight percent is exactly 100.
  - Mass fractions are normalised per bin, so `calc_density`'s
    `mtot < 1e-15 kg` → 1000 kg/m³ fallback, which depends on box volume, never
    fires. For the model physics, see "Sparse-bin density" below.

Check (`python -m sensitivities.dunne_nuc.phase1_sabre_check`): 6 h at 217 K,
SO₂ 1 ppbv, background aerosol, on 40 and 80 bins.
- `jacfwd` of all 63 SABRE bins plus 5 NMASS channels with respect to the 25
  coefficients is finite. It takes 14 s (40 bins) and 39 s (80 bins), including
  compile. Figure: `results/phase1/sabre_jacobian.png`.
- About 89% of the particles formed sit between 1.7 and 3 nm, below SABRE's
  lower limit. N_total is 1.36e5 vs N(>3 nm) 1.55e4 cm⁻³ (40 bins).
- **N(>3 nm) depends on resolution: 1.55e4 (40 bins) vs 1.15e4 (80 bins), 35%.**
  N_total differs by only 7% and MNFIX survival is identical (0.636), so the gap
  comes from rebinning. The nucleation mode is steep at 3 nm, and spreading it
  uniformly within a coarse 40-bin bin pushes number across the cutoff. 80 bins
  (about 20 per decade, matching SABRE) is the primary grid. Treat the 3 nm
  channel as the least robust.
- Sensitivities are large in the nucleation mode (3–8 nm): d ln(dN/dlogDp)/d u_bi
  is up to 1.5 and d/d r_nuc_nm up to 25 per nm. They are below the 8% SABRE
  uncertainty per unit coefficient change for Dp > 15 nm after 6 h. Whether
  coefficients are detectable in the accumulation mode is a question for the
  24–48 h runs in Phases 2–3.

## Sparse-bin density (`make_step(sparse_density=...)`)

Two Fortran fallbacks assign a default density to low-occupancy bins. Both
thresholds are per grid cell, so their meaning depends on `boxvol`:

| Where | Trigger | Fortran default |
|---|---|---|
| `calc_density` (aerodens.f), used in the coagulation kernel | bin mass < 1e-15 kg/cell | 1000 kg/m³ |
| `calc_wet_diameters` (getCondSink.f), used in the condensation sink | Nk ≤ 1e10 #/cell (1e4 cm⁻³ in a 1e6 cm³ box) | 1500 kg/m³, mp = 1.4·xk |

In the stratosphere and in SAI plumes, nearly every bin falls below the
condensation-sink threshold. Sparse nucleation-mode bins also fall below the
calc_density threshold. `make_step(..., sparse_density=rho)` replaces both
defaults with `rho`. The study uses
`radiative_forcing.DENSITY_H2SO4_SOLUTION = 1700 kg/m³` (about 75 wt% H₂SO₄ at
220 K); see `common.SPARSE_DENSITY`. Details:
- The `mp = 1.4·xk` default mass is unchanged.
- Populated bins are unchanged.
- `None` (the default) is bit-identical to the Fortran behaviour.
- The option is threaded through `coag_euler_step`, `calc_particle_properties`,
  `calc_condensation_sink`/`calc_wet_diameters`, `_condensation_step_core`,
  `ezcond_ppm_jax` and `ezcond_tfl_jax`. Other public wrappers (`run_full_scan`,
  `full_step_jax`, …) keep the Fortran defaults.

Impact: 217 K, SO₂ 1 ppbv, 10 cm⁻³ background at 100 nm, 6 h.

| | 40 bins | 80 bins |
|---|---|---|
| Initial condensation sink | −8.0% | −8.0% |
| N_total after 6 h | +1.9% | +4.1% |
| N(>3 nm) | +6.6% | −1.2% |
| N(>12 nm) | 0 | 0 |
| H₂SO₄ after 6 h | +0.5% | +1.0% |

These effects are of the same order as SABRE's 8% number uncertainty in the
nucleation mode, so the choice matters for comparisons there. The Phase 0/1
numbers above were rerun with `sparse_density=1700`: 0c N_total is 271.5 cm⁻³,
the elasticities change by less than 0.5%, and Phase 1 N(>3 nm) is 1.65e4
(40 bins) and 1.14e4 (80 bins).

## Water uptake in the stratosphere (2026-10-27)

The study runner (`common.make_runner`) now uses `WATER_SCHEME_H2SO4`
(Tabazadeh 1997 binary H₂SO₄/H₂O) instead of the TOMAS ammonium-bisulfate
default. At 217 K and RH 2%, the bisulfate fit leaves particles almost dry
(2.4% water by mass). Tabazadeh gives 68.6 wt% H₂SO₄, which is 32% water and
correct for stratospheric sulfuric acid. Sizing for SABRE still uses dry
diameters; the water changes only the growth physics (condensation sink and
coagulation kernel use wet sizes).

Effect at 217 K, SO₂ 1 ppbv, 6 h (`sparse_density` = 1700 in all runs):

| | 40 bins | 80 bins |
|---|---|---|
| SABRE dry density 1902 → 1836 (sizing only): N(>3 nm) | +7.7% | +7.0% |
| Bisulfate → H₂SO₄ water scheme: N(>3 nm) | +55% | +83% |
| Bisulfate → H₂SO₄ water scheme: N_total | −20% | −18% |
| Bisulfate → H₂SO₄ water scheme: H₂SO₄ | −16% | −15% |

Wetter particles give a larger condensation sink, which means less H₂SO₄ and
fewer new particles. The new particles also grow faster, so more of them cross
3 nm.

The water scheme is a structural choice whose effect on N(>3 nm) is far larger
than SABRE's 8% uncertainty and larger than most per-coefficient
sensitivities. It belongs in the uncertainty discussion alongside the
nucleation coefficients.

Updated Phase 1 values with the H₂SO₄ scheme and 1836 kg/m³:
- N(>3 nm) is 2.72e4 (40 bins) and 2.37e4 (80 bins). The 40/80-bin gap shrinks
  from 35% to 14%.
- Phase 0c N_total is 271.3 cm⁻³; the elasticities are unchanged to 3 digits.
