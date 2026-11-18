# Prescribed nucleation rate: response of the size distribution to J

How much does the particle number SABRE would measure change when the nucleation rate J changes, whatever scheme produced it? This analysis bypasses the nucleation scheme, prescribes J directly, and differentiates the size distribution with respect to J. It complements the coefficient sensitivity in [nucleation_sensitivity.md](nucleation_sensitivity.md), which asks how N responds to the Dunne 2016 fit coefficients.

Code: `sensitivities/dunne_nuc/common.py` (`make_prescribed_j_runner`) and `sensitivities/dunne_nuc/prescribed_j.py`. Results (git-ignored): `sensitivities/dunne_nuc/results/prescribed_j/`.

Status (2026-11-18): 217 K / 55 hPa, 40 bins done, with pure sulfate clusters (`org_frac = 0`). The AFGL tropical altitudes (13–25 km) with adaptive gas sub-steps are done on 40 bins ([below](#results-tropical-altitudes-40-bins)); they show that a constant J cannot probe the high-J end at high SO₂, and a J that scales with H₂SO₄ is being developed on a separate branch. Other temperatures, 80 bins and the HiLa conditions are not run.

## Quantities

- **Elasticity** d ln N / d ln J: the fractional change in N per fractional change in J, at fixed growth conditions. 1 means N ∝ J; below 1, extra nucleation is offset by coagulation of the new particles and by competition for H₂SO₄; 0 means N does not depend on J.
- **Absolute sensitivity** dN/dJ [s]: saved, but less useful on its own because J spans many orders of magnitude.
- **Outputs:** N(>3, >8, >12, >27, >53 nm) (NMASS channels), N(3–10.65 nm) (first 11 SABRE bins, the nucleation mode) and SABRE dN/dlogDp, hourly over 24 h.

This differs from the "J_all ×2.5" direction of the coefficient study. That direction scales the Dunne rate by a common factor, so J keeps its dependence on H₂SO₄, T and ions and the derivative is taken about the scheme's own J(t). Here J is a free input, held constant in time.

## Method

### Runner

`common.make_prescribed_j_runner(xk, nsteps, ...)` returns `run(J, Nk, Mk, Gc)`. Each time step is the Phase 2 step with the rate replaced:

1. SO₂ + OH chemistry (Sun et al. 2022, constant OH).
2. Nucleation at the prescribed J [cm⁻³ s⁻¹], as in `nucleation.nucleation_step`:
   - clusters go to bin 0 with mass `_cluster_mass(NUC_PARAMS)` (1.7 nm at 1770 kg/m³, equal to `XK0`), pure sulfate (`org_frac = 0`, see [nucleation_sensitivity.md](nucleation_sensitivity.md));
   - their sulfate mass is taken from gas-phase H₂SO₄, with the same gas clamp;
   - sub-steps from `compute_nucleation_substeps` (driven by J), MNFIX after each.
3. Coagulation and condensation from `make_step` (PPM condensation, fixed Euler coagulation, sparse density 1770 kg/m³, Tabazadeh water).

The runner also records the Dunne rate at the current H₂SO₄ (`J_scheme`), the sub-step count and whether the clamp fired. The library is unchanged.

### Conditions

Those of the Phase 2 map: 55 hPa (about 20 km), RH 2%, OH 1e6 cm⁻³ not titrated by SO₂, no NH₃, nucleating organics or dilution, 10 cm⁻³ background sulfate (GMD 100 nm, GSD 1.6), 24 h. Time step from `run_map.dt_for(SO₂)`: 60 s (≤ 10 ppbv), 10 s (100 ppbv), 2 s (1–10 ppmv), 0.5 s (100 ppmv).

### J grid: a window around the scheme

At each temperature T and SO₂ level:

J = J_ref × 10^(−1 … +1), 9 levels in quarter-decade steps,

where J_ref is the 24 h mean Dunne J from the Phase 2 map at the same T, SO₂ and grid (`run_map --resolved-dt`). A constant J_ref makes the same number of particles over 24 h as the scheme. The window covers the Dunne measurement uncertainty (×2.5) and the Table S2 alternative fit (0.3–0.7× in N), with room either side.

The window replaced a first version with fixed absolute levels, J = 1e-4 … 1e6 cm⁻³ s⁻¹ for every SO₂ level (kept in `results/prescribed_j/absolute_J_grid/`). That grid put very low J with very high H₂SO₄, a state the scheme does not produce, and there the AD derivative is wrong (next section).

### Derivatives: AD with a finite-difference check in every cell

- **AD:** one forward-mode tangent (`jax.jvp`) per cell gives dY/dJ for every output.
- **FD:** central difference at J(1 ± 2%), two extra forward runs per cell. J is one number, so this is affordable, and it checks AD where AD is blind: the nucleation sub-step count, the gas clamp, and MNFIX bin selection.
- Cells where the two differ by more than 0.05 + 10% (in elasticity) are flagged, as are cells where the clamp fires on more than 1% of steps.

Each dt group is one `jit(vmap(...))` over (SO₂ levels in the group) × J levels.

## The H₂SO₄ ceiling on J

Prescribed J does not depend on H₂SO₄, but H₂SO₄ caps it. Each cluster takes 28 H₂SO₄ molecules from the gas (the cluster mass is 28 H₂SO₄ masses, all sulfate; the gas is debited the sulfate mass as H₂SO₄ mass). Gas H₂SO₄ starts at zero and comes only from SO₂ + OH, at P = k₁[OH][SO₂]. Within a sub-step, nucleation cannot use more than the gas present (the clamp), and the step order (chemistry → nucleation → coagulation → condensation) gives nucleation first claim on each step's production. The sustained ceiling is therefore

J_max ≈ P / 28,

before condensation takes any. At 217 K and 55 hPa (k₁ = 5.9e-13 cm³ s⁻¹), that is about 39 cm⁻³ s⁻¹ per ppbv of SO₂. J_ref is from the 40-bin Phase 2 map:

| SO₂ (ppbv) | P (cm⁻³ s⁻¹) | J_max | J_ref | J_ref / J_max |
|---|---|---|---|---|
| 0.1 | 1.1e2 | 3.9 | 0.13 | 0.03 |
| 1 | 1.1e3 | 39 | 2.1 | 0.05 |
| 10 | 1.1e4 | 3.9e2 | 21 | 0.05 |
| 100 | 1.1e5 | 3.9e3 | 3.2e2 | 0.08 |
| 1e3 | 1.1e6 | 3.9e4 | 6.1e3 | 0.16 |
| 1e4 | 1.1e7 | 3.9e5 | 1.2e5 | 0.30 |
| 1e5 | 1.1e8 | 3.9e6 | 1.6e6 | 0.42 |

- Up to 100 ppbv the whole window is below the ceiling.
- At 1e3, 1e4 and 1e5 ppbv the ceiling is at about 6×, 3× and 2.4× J_ref. Above it, nucleation is limited by the H₂SO₄ supply and the elasticity is 0. That is physical, but AD cannot see the switch, so these cells are flagged. In the runs the clamp fired at 10× J_ref (1e3 ppbv) and from 3.2× (1e4 and 1e5 ppbv).
- The Dunne scheme itself uses up to about 42% of the H₂SO₄ production at 100 ppmv, one reason its response flattens there.
- With constant OH this ceiling is an upper bound. At ≥ 1 ppmv real OH would be titrated by SO₂, and both P and the ceiling would be much lower.

## Why not far below the scheme: PPM's derivative is unstable there

With few new particles and fast growth (low J, high H₂SO₄), the AD derivative grows without bound while the model state stays smooth. dN(>3 nm)/dJ at 100 ppbv, J = 0.01 cm⁻³ s⁻¹ (3e4 below J_ref), 217 K, 40 bins, in seconds:

| Hour | 3 | 6 | 9 | 12 | 15 | 18 |
|---|---|---|---|---|---|---|
| AD | 1.1e4 | 2.1e4 | 3.8e4 | 4.1e5 | 3.6e6 | −9.8e6 |
| FD | 1.1e4 | 2.1e4 | 2.9e4 | 3.6e4 | 4.1e4 | 4.3e4 |

- **Source: PPM condensation.** At the 9 h state, one 10 s step of the linearised PPM condensation multiplies the dominant tangent mode by 1.0086 (about ×22 per hour). Chemistry, coagulation and MNFIX do not amplify it. The mode sits in bins 0–3, where a few new particles grow quickly.
- **Why:** PPM's slope limiters keep the scheme itself stable. Differentiating with the limiter branches held fixed gives a linear scheme that can grow, a known problem with tangent-linear models of limiter-based advection.
- **Top-hat (TFL) condensation is no alternative:** its AD derivative is NaN here and its FD is noisy (bin assignment jumps).
- **Inside the window AD agrees with FD up to 1 ppmv** (Results). At 10–100 ppmv the 2% FD is erratic, so neither derivative is confirmed there.
- **The coefficient study also differentiates through PPM.** Its Phase 3 FD check passed in the plume's growth channels, but no FD check has been run on the Phase 2 map, and some Phase 4 nonlinearity could be this artefact. Not yet checked.

## Frozen-shape tangent (`ppm_tangent='frozen_shape'`)

`make_step(..., ppm_tangent='frozen_shape')` (also `common.make_runner`, `make_prescribed_j_runner`, `--ppm-tangent frozen_shape` in `run_map` and `prescribed_j`, which write to `*_frozen` directories) leaves every value unchanged, bit for bit, and changes only the derivative of the PPM reconstruction (`condensation_ppm._freeze_shape`).

- **What it does.** Each bin's parabola (n_L, n_R, n_6) is held at its current shape, s_X = n_X/n̄ under `stop_gradient` (flat for empty bins), and scaled by that bin's own tangent: δn_X = s_X·δn̄. Written as `stop_gradient(n_X) + s_X·(n̄ − stop_gradient(n̄))`, the added term is exactly 0 in value.
- **Why it is stable.** The exact derivative is the 4th-order edge interpolation with the limiter branches held fixed, which has negative weights and can amplify. With the shape frozen, a bin's flux derivative is its swept fraction (in [0, 1]) times its own δN: an upwind scheme with non-negative weights. Tangents through the growth velocity (H₂SO₄, coefficients) are unchanged.
- **It is not the exact derivative.** It drops the response of the sub-bin shape to neighbouring bins.

Checks (`benchmarks/python/ppm_tangent_stability.py`, 217 K, 40 bins, 18 h, d N(>3 nm)/dJ against central FD at ±2%; 2026-11-09):

| case | exact vs FD | frozen vs FD |
|---|---|---|
| 100 ppbv, J = 0.01 (the table above) | diverges from 9 h (×10–300 off) | within 1% throughout |
| 100 ppbv, J_ref = 317 | within 1–3% (8% at 9 h) | 2–7% high (12% at 9 h) |
| 1 ppmv, J_ref = 6070 | within 0.2–4% | within 0.4–5% |
| 1 ppbv, J_ref = 2.1 | FD itself erratic (negative in hours 1–2); exact and frozen within 16% of each other | |

At the 9 h state of the J = 0.01 case, the leading tangent mode of one hour's linearised step grows ×2–11 per hour with the exact derivative and ×0.99 with the frozen one.

**Not fixed: 80 bins at 100 ppmv.** In the Phase 2 cells at 223 K and 240 K / 100 ppmv (adaptive f = 0.1), exact, frozen and FD (±0.05 along the H₂SO₄-calibration direction) agree at 6 h (L1 within 0.01), but at 24 h the frozen tangent also blows up (Σ|δN|/ΣN 9e4 and 3e27, against ~1 for FD). From the 240 K state at 14 h, the leading mode grows ×170 per 10 min with either derivative (×800 at 20 h). Measured in Σ|δN| it sits in bins 31–33 (35–36 at 20 h): the small-size tail of the seed mode, 0.04–1 particle per cell (1e-8–1e-6 cm⁻³).

The source is the **linear sub-bin coagulation scheme** (`coagulation_rates_linear.py`, used on every non-doubling grid, so 80 bins and not 40). Power iteration on the 14 h state, one change at a time:

| Tangent change (values unchanged) | Growth per 10 min |
|---|---|
| none (frozen PPM shape) | ×171 |
| TAU's per-particle terms (mpw/N, sink share/N) frozen | ×169 |
| `species_flux` composition ratio frozen | ×189 |
| coagulation kernel frozen | ×171 |
| coagulation tangent replaced by the identity | ×1.2 (×0.97 with TAU frozen too) |
| absolute cut: bins < 1 #/cell enter condensation without a tangent | ×0.98 |

Scavenging of the nucleated particles pushes these sparse bins' mean mass x̄ to within 1–2% of the bin width of the top edge b. There the scheme's ramp profile has n_hi = 2/(3(b − x̄)), so the fraction moved up a bin per collision has a derivative ∝ 1/(b − x̄)², capped only by `_NARROW` = 1e-6. Holding the profile, or x̄, fixed in the derivative is worse (tangent ~1e279 within one step): number and mass then move with inconsistent tangents. No fix yet. Cutting the tangent below a relative occupancy (1e-5·ΣN) bounds it but loses the growth front (N(>12 nm) off by up to 1000× at 40 bins), so it was removed.

## Checks

- **Same step as the scheme runner.** With one nucleation sub-step and J set to the scheme's rate, one step of `make_prescribed_j_runner` matches `make_runner` to round-off, with and without the clamp (N to 8e-13 relative, gas exactly). With more sub-steps they differ on purpose: the scheme recomputes J as nucleation draws down H₂SO₄ within a step, while the prescribed J stays fixed.
- **AD against FD** in every cell (above).
- **Finite outputs:** the run fails on any non-finite value or derivative.

## Limitations

- **Constant J.** The scheme's J varies through the day as H₂SO₄ builds up; a constant J with the same 24 h total front-loads or back-loads nucleation differently. The response is to the level of J, not its timing.
- **J_ref is grid-specific.** It comes from the Phase 2 map on the same grid; the 80-bin runs need the 80-bin map.
- **Conditions are the Phase 2 ones** (55 hPa, untitrated OH). They do not represent the high-latitude lowermost stratosphere; see the HiLa plan in [nucleation_sensitivity.md](nucleation_sensitivity.md#planned-hila-high-latitude-lowermost-stratosphere-scenario).
- **Cluster size fixed** at 1.7 nm and 1770 kg/m³; the coefficient study found N sensitive to the cluster mass where particles grow.

## Usage

```bash
python -m sensitivities.dunne_nuc.prescribed_j                     # 217 K, 40 bins
python -m sensitivities.dunne_nuc.prescribed_j --temps 208 217 223
python -m sensitivities.dunne_nuc.prescribed_j --grids 80          # needs the 80-bin Phase 2 map
python -m sensitivities.dunne_nuc.prescribed_j --plot-only
python -m sensitivities.dunne_nuc.prescribed_j --smoke             # 2 h, 2 SO2 x 2 J
```

Run time: 217 K on 40 bins, 43 min sharing the CPU with two other runs (9 J levels × 8 SO₂ levels, one tangent and two FD runs each).

Output NPZ (`pj_{nbins}bin_T{T}.npz`): `dndlog`, `nmass` (SO₂, J, hour, channel), `dJ_*` (AD), `fdJ_*` (FD), `J`, `j_ref`, `j_factors`, diagnostics `aux_*` (mean H₂SO₄, scheme J, sub-step and clamp fractions). `prescribed_j.load` appends N(3–10.65 nm) as channel 5.

## Results

217 K, 40 bins, 24 h (`results/prescribed_j/pj_40bin_T217.npz`; 43 min, sharing the CPU). Elasticity d ln N / d ln J by AD at J = J_ref, with the range over the window (0.1–10× J_ref) where it is not clamped:

| SO₂ (ppbv) | N(>3 nm) at J_ref (cm⁻³) | N(>3 nm) | N(3–10.65 nm) | N(>12 nm) |
|---|---|---|---|---|
| 0.1 | 2.4e3 | 0.83 (0.95 → 0.28) | 0.83 (0.98 → 0.28) | 0 |
| 1 | 2.5e4 | 0.26 (0.62 → 0.22) | 0.29 (0.78 → 0.24) | −0.03 (−0.14 to 0.06) |
| 10 | 3.3e4 | 0.33 (0.26–0.42) | 0.50 (0.35–0.51) | 0.06 (0.05–0.20) |
| 100 | 5.3e4 | 0.60 (0.39–0.61) | 0.72 (0.63–0.81) | 0.10 (0.06–0.24) |
| 1e3 | 9.8e4 | 1.06 (0.54–1.06; 0 at 10×) | 1.13 (0.82–1.16; 0 at 10×) | 0.18 (0.01–0.63) |
| 1e4 | 1.3e5 | 1.8 (0.59–1.8; 0 from 3.2×) | 1.9 (1.2–1.9) | 0.02 (0–0.08) |
| 1e5 | 8.3e4 | 1.95 (0.30–1.95; 0 from 3.2×) | 2.1 (1.35–2.15) | 0.01 |

At 0.01 ppbv N(>3 nm) is the background (10 cm⁻³) and N(3–10.65) is below 0.01 cm⁻³ at J_ref.

- **N is far from proportional to J below 1 ppmv.** At 1–100 ppbv the elasticity of N(>3 nm) is 0.26–0.60: extra particles are offset by coagulation and by sharing the H₂SO₄. At 0.1 ppbv it falls from 0.95 to 0.28 across the window as the new particles start to coagulate with each other.
- **Above 1 ppmv N rises faster than J** (elasticity 1.1–2.1 for N(>3 nm) and N(3–10.65) near J_ref), up to the H₂SO₄ ceiling, where it drops to 0. The mechanism is not separated, and at 10–100 ppmv the FD check does not confirm it (below).
- **N(>12 nm) responds little** (≤ 0.25), except at 1 ppmv and 3–5.6× J_ref (0.5–0.63). **N(>53 nm)** is within ±0.06, except at 10 ppbv and 0.1× J_ref (−0.28, where FD disagrees).
- **Prescribed J gives higher elasticities than the scheme.** The Phase 2 J_all elasticity of N(>3 nm) at 217 K is 0.07, 0.25, 0.40, 0.60, 0.87 and 0.75 at 1 ppbv to 100 ppmv, against 0.26, 0.33, 0.60, 1.06, 1.8 and 1.95 here. With the scheme, extra nucleation lowers H₂SO₄ and with it J; a prescribed J has no such feedback.
- **The states differ too.** At J = J_ref the Dunne rate at the run's own H₂SO₄ is 1.9–7.6× J_ref (24 h mean) at ≥ 1 ppbv, and N(>3 nm) is 1.2–2.3× the Phase 2 value: a constant J nucleates early, before H₂SO₄ builds up, where the scheme nucleates later.

**AD against FD (24 h, N(>3 nm)): 16 of 72 cells disagree.**
- **Up to 1 ppmv** AD and FD agree in all but three cells: two at 0.1 ppbv and 5.6–10× J_ref (AD 0.37 and 0.28, FD 0.57 and 0.41) and one FD outlier at 10 ppbv.
- **At 10 and 100 ppmv** 13 cells disagree. There FD jumps between −1.0 and 9.6 from one J level to the next while AD varies smoothly. With 43 000–173 000 steps (2 s and 0.5 s), a 2% change in J moves particles across bin edges and sub-step counts, so the FD is not a reliable check either. Neither number is confirmed at ≥ 10 ppmv.
- N(>12 nm) and N(>53 nm) disagree in 3 and 4 cells, all small values.

## Tropical altitudes with adaptive gas sub-steps (`prescribed_j_tropical.py`)

The study above on the `gas_substep_psd80` grid: altitude 13–25 km (`common.TROPICAL_AFGL`, T and P together) × SO₂ × 9 J factors, 24 h, `make_step(gas_substep='adaptive')` (f = 0.1) instead of `dt_for`.

- **`fn_prescribed`.** `make_step`'s step function and `nucleation_step` take `fn_prescribed` [cm⁻³ s⁻¹] (ricco_dunne only; `None` = the scheme, bit-identical). It replaces the rate in the nucleation step, its sub-step count and the gas sub-step criterion: J does not depend on H₂SO₄, so n_eff = 1 and k_nuc = (1 − org_frac) m_nuc boxvol J / C. `common.make_prescribed_j_runner(..., gas_substep='adaptive', gas_min_dt=...)` uses it with the chemistry split around nucleation, as the scheme runner does.
- **J above the H₂SO₄ supply.** The gas empties, k_nuc = J m / C grows as C → 0 and the clamp fires on every sub-step whatever its length; with the default 1 ms floor the 16384-sub-step cap then binds (`GasSubstepCapWarning`, wrong answer). The study uses a floor of `MIN_DT` = 0.1 s (600 sub-steps per 60 s step at most). The clamped result (nucleation takes the production, condensation gets none) does not depend on the sub-step. Those cells are marked by `frac_clamp` in the figures.
- **J_ref** = 24 h mean of the scheme's J from an adaptive-sub-step scheme run in the same cell (`make_runner`, 60 s end-of-step values, as in the Phase 2 map).
- **One file per run.** `jref_z<km>_<so2>.npz` and `pj_z<km>_<so2>_j<i>.npz` (value, AD tangent, FD ±2%), run in separate processes (`--all`, `NJOBS` at a time, resumable): an adaptive `while_loop` under `vmap` runs at the pace of its slowest lane, so cells and J factors are not batched. Output: `results/prescribed_j_tropical/<bins>bin/`.
- **Run time (40 bins, one J = J_ref point, one process):** 1 ppmv 90 s, 10 ppmv 400 s at 15 km. 80 bins costs about 5×, and at ≥ 100 ppmv the 80-bin AD tangent blows up (above), so 80 bins is for a subset.
- **Tests:** `tests/test_prescribed_j_adaptive.py` (default bit-identical, scheme-rate equivalence, the criterion's k_nuc, adaptive against fixed 1 s, AD against FD).

```bash
python -m sensitivities.dunne_nuc.prescribed_j_tropical --smoke                 # 2 h, 15 km, 10 ppbv
python -m sensitivities.dunne_nuc.prescribed_j_tropical --all                   # SO2 <= 10 ppmv, 40 bins
python -m sensitivities.dunne_nuc.prescribed_j_tropical --all --so2 1e5         # 100 ppmv
python -m sensitivities.dunne_nuc.prescribed_j_tropical --all --bins 80 --alts 15 20 --so2 0.01 1 100 1e4
sensitivities/dunne_nuc/run_prescribed_j_overnight.sh [40|80] [main|hi|all]     # overnight, resumable
```

### Results: tropical altitudes, 40 bins

First pass, 2026-11-17: 6 altitudes × 8 SO₂ levels (0.01 ppbv–100 ppmv) × 9 J factors, 432 runs, 26 CPU-h. Per run, 9 s at ≤ 1 ppbv, 26–135 s at 1 ppmv, 83–555 s at 10 ppmv and 245–2296 s at 100 ppmv. Figures: `results/prescribed_j_tropical/40bin/pj_40bin_z<km>.png` and `pj_40bin_elasticity_vs_so2.png`.

**Where the window sits against the H₂SO₄ ceiling.** The table gives J_ref / (P/28), where P = k₁[OH][SO₂] at t = 0 (see [The H₂SO₄ ceiling on J](#the-h₂so₄-ceiling-on-j)):

| SO₂ (ppbv) | 13 km | 15 km | 17 km | 20 km | 23 km | 25 km |
|---|---|---|---|---|---|---|
| 0.01 | 0.01 | 0.02 | 0.03 | 0.00 | 0.00 | 0.00 |
| 0.1 | 0.05 | 0.10 | 0.15 | 0.09 | 0.01 | 0.00 |
| 1 | 0.04 | 0.10 | 0.16 | 0.08 | 0.06 | 0.06 |
| 10 | 0.05 | 0.15 | 0.24 | 0.10 | 0.05 | 0.04 |
| 100 | 0.10 | 0.26 | 0.37 | 0.16 | 0.06 | 0.04 |
| 1e3 | 0.17 | 0.38 | 0.50 | 0.29 | 0.12 | 0.07 |
| 1e4 | 0.24 | 0.48 | 0.60 | 0.41 | 0.22 | 0.15 |
| 1e5 | 0.30 | 0.54 | 0.65 | 0.51 | 0.32 | 0.24 |

- The scheme uses an increasing share of the supply as SO₂ rises, most at 15–17 km (cold).
- **The clamp fires on > 1% of steps at the first J level that is 1.3–2.4× P/28 or more,** in every cell where the window reaches that far. The level below it is under 1.4× P/28 (quarter-decade grid). So P/28 is the ceiling, within the grid spacing.
- **45 of 432 runs are clamped,** all at J / J_ref ≥ 1.8: 0–2 per SO₂ level up to 10 ppbv, then 5, 8, 13 and 15 at 0.1, 1, 10 and 100 ppmv. From 1 ppmv up, the top 2–4 J factors are clamped at most altitudes. There d ln N / d ln J = 0 by AD, FD and physics alike: N is set by the supply, not by J.

**Elasticity d ln N(>3 nm) / d ln J at J = J_ref, AD, 24 h** (* = AD and FD ±2% differ by more than 0.05 + 10%):

| SO₂ (ppbv) | 13 km | 15 km | 17 km | 20 km | 23 km | 25 km |
|---|---|---|---|---|---|---|
| 0.01 | 0.90 | 0.93 | 0.86* | 0.01 | 0.00 | 0.00 |
| 0.1 | 0.35 | 0.28 | 0.25* | 0.57* | 0.90* | 0.20 |
| 1 | 0.28 | 0.28 | 0.27 | 0.23 | 0.33* | 0.40 |
| 10 | 0.51 | 0.51 | 0.48 | 0.37 | 0.28 | 0.25 |
| 100 | 0.96 | 0.88 | 0.76 | 0.63 | 0.50 | 0.43 |
| 1e3 | 1.28* | 1.58 | 1.31 | 1.05 | 0.87 | 0.76 |
| 1e4 | 1.35* | 1.73* | 1.73 | 1.80 | 1.50 | 1.30 |
| 1e5 | 1.46* | 2.01* | 2.04 | 1.89* | 1.86* | 1.58* |

- **At 20–25 km and 0.01 ppbv (and 0.1 ppbv at 25 km) N(>3 nm) is the background seed.** New particles do not reach 3 nm in 24 h, so the elasticity is ~0. Where they do grow at the lowest SO₂ (13–17 km at 0.01 ppbv, 23 km at 0.1 ppbv), N is nearly proportional to J (0.86–0.93).
- **0.1–10 ppbv: 0.2–0.6.** As in the 217 K study, extra particles are offset by self-coagulation and by sharing the H₂SO₄.
- **Above 100 ppbv the elasticity exceeds 1 and rises with SO₂, to 1.3–2.0 at 10–100 ppmv.** Across the window it peaks just below the clamp: for example 2.17 at 13 km / 100 ppmv, 3.2× J_ref. One step further up it falls to 0. Near the ceiling, nucleation draws H₂SO₄ down. A scheme's J would fall with it, but the prescribed J does not, so this end of the window is where a constant J is least representative.

**AD against FD.** Of the 387 unclamped runs, 91 differ: 4–11 per SO₂ level up to 1 ppmv (mostly single isolated cells where FD jumps), then 18 at 10 ppmv and 30 at 100 ppmv. At 10–100 ppmv the ±2% FD swings between about −5 and +11 from one J level to the next while AD varies smoothly (e.g. 13 km / 10 ppmv: AD 1.35, FD 9.21 at J_ref). This is the same bin-edge and sub-step noise as in the 217 K study, so **AD is not confirmed at ≥ 10 ppmv**, except at 17 and 20 km near J_ref where the two agree. Where FD is smooth (most cells ≤ 1 ppmv), AD agrees to a few %.

**Conclusion of the first pass.** At high SO₂, J_ref is already 0.15–0.65 of the H₂SO₄ ceiling, so the upper half of the ×10 window is either clamped (elasticity 0) or in the drawdown regime that a constant J misrepresents. A constant prescribed J therefore cannot explore the sensitivity at high J. Two things are needed:

- **A prescribed J that responds to H₂SO₄,** such as J = A·J_ref·(C/C_ref)^n. It falls as nucleation draws down H₂SO₄, so it never clamps; n = 0 recovers the runs above. It is being developed on a separate branch. The gas sub-step criterion already derives n_eff from J(C) when it is given a rate function; `fn_prescribed` currently fixes n_eff = 1.
- **A more robust FD check at ≥ 1 ppmv:** a larger step, or the Phase 3 regression stencil.
