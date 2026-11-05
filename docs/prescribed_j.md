# Prescribed nucleation rate: response of the size distribution to J

How much does the particle number SABRE would measure change when the nucleation rate J changes, whatever scheme produced it? This analysis bypasses the nucleation scheme, prescribes J directly, and differentiates the size distribution with respect to J. It complements the coefficient sensitivity in [nucleation_sensitivity.md](nucleation_sensitivity.md), which asks how N responds to the Dunne 2016 fit coefficients.

Code: `sensitivities/dunne_nuc/common.py` (`make_prescribed_j_runner`) and `sensitivities/dunne_nuc/prescribed_j.py`. Results (git-ignored): `sensitivities/dunne_nuc/results/prescribed_j/`.

Status (2026-11-05): 217 K, 40 bins done, with pure sulfate clusters (`org_frac = 0`). Other temperatures, 80 bins and the HiLa conditions are not run.

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
