# Nucleation Coefficient Sensitivity (forward-mode AD)

Gradient-based sensitivity of stratospheric aerosol to the `ricco_dunne` fit
coefficients. It uses JAX forward-mode differentiation instead of a large
perturbed-parameter ensemble.

Work plan, by phase:

| Phase | What | Status |
|---|---|---|
| 0 | Warm-up: rate-level and 1 h full-model gradients, checked against analytic forms and finite differences | **done** |
| 1 | SABRE-binned dry N(Dp) output operator (Lyu et al. 2026) | planned |
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
