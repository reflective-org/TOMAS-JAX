# Nucleation Coefficient Sensitivity (forward-mode AD)

Sensitivity of stratospheric aerosol, as SABRE would measure it, to the 25 `ricco_dunne` fit coefficients. It uses JAX forward-mode differentiation through the full model instead of a large perturbed-parameter ensemble.

Code: `sensitivities/dunne_nuc/`. Results (git-ignored): `sensitivities/dunne_nuc/results/`.

**Model setup.** The runs use the stratospheric settings in [stratosphere_setup.md](stratosphere_setup.md): cluster and sparse-bin density 1770 kg/m³ (the grid density) and pure sulfate clusters, set in `common.NUC_PARAMS` / `NUC_PARAMS_S2` and `common.SPARSE_DENSITY`. All numbers below are from the runs of 2026-11-05 unless marked "at 1350" or "with 10% organic". Earlier results: `results/rho1350/` (before the density change of 2026-11-04) and `results/orgfrac0.1/` (before the org_frac change of 2026-11-05).
- **Density change:** with 1350 the cluster was lighter than the bin-0 lower edge, so MNFIX put the nucleated mass into fewer, larger particles (finding 8).
- **org_frac change:** the Phase 3 Jacobian predicted plume N(>3 nm) +12–15% and bg_ttl N(>12 nm) −5%. The reruns gave plume N(>3 nm) +11.5% (40 bins) and +15% (80), bg_ttl N(>12 nm) −4% and −5%; in the Phase 2 map N(>3 nm) changed by −13% to +19% (largest at ≥ 1 ppmv) and N(>12 nm) by at most 6%.

| Phase | What | Script | Status |
|---|---|---|---|
| 0 | Rate-level and 1 h full-model gradients against analytic forms and FD | `phase0_warmup.py` | done |
| 1 | SABRE-binned dry N(Dp) output operator | `sabre_bins.py`, `phase1_sabre_check.py` | done |
| 2 | T × SO₂ map along the published uncertainty directions | `run_map.py` | done, 40 and 80 bins at resolved dt (the 60 s maps only at 1350) |
| 3 | Three 48 h scenarios: full Jacobian, FD check | `scenarios.py`, `phase3_scenarios.py` | done, 40 and 80 bins |
| 4 | Linearity at published-size perturbations | `phase4_linearity.py` | done, 40 and 80 bins |
| J | Response to a prescribed nucleation rate (scheme bypassed) | `prescribed_j.py` | done, 217 K, 40 bins ([prescribed_j.md](prescribed_j.md)) |
| HiLa | High-latitude lowermost stratosphere (13 km, 60°S), 1–100 ppmv SO₂ | — | planned ([adjustments](#planned-hila-high-latitude-lowermost-stratosphere-scenario)) |

## Main findings

1. **The Dunne binary-neutral channel dominates** wherever nucleation matters (TTL, plume, Phase 2 below 223 K): v_bn (the temperature dependence of J_bn), then p_bn and u_bn. The ion-induced channel matters only where there is little nucleation (bg_midlat).
2. **The cluster mass is the leading non-Dunne coefficient where particles grow.** In the plume the elasticity of N(>3 nm) to r_nuc is +5.8 (40 bins) and +8.1 (80), and to ρ_nuc exactly a third of that, so it acts through the cluster mass. In bg_ttl, N(>12 nm) has +0.49 to +0.51. Where few particles form (bg_midlat, Phase 0) it is −0.01 to −0.2.
3. **N(>53 nm) is insensitive everywhere.** Changes along the published directions stay within SABRE's ±8% at the published size (α = ±1). At twice that size, only the plume's joint direction reaches it (α = +2: Δln N = +0.11 on 40 bins, +0.09 on 80). The signal is in N(>3–12 nm) and the SABRE bins below about 30 nm.
4. **Structural choices outweigh the published coefficient uncertainty:**
   - **grid:** 40 vs 80 bins changes the bg_ttl N(>12 nm) ×6.5, the plume N(>3 nm) ×2.6, and the Phase 2 N(>3 nm) at ≥ 10 ppmv ×0.27–0.6;
   - **Dunne Table S2 vs S1:** plume N(>3 nm) ×0.37–0.45;
   - **water scheme:** Phase 1 N(>3 nm) +55–83% (at 1350);
   - **cluster density vs grid** (finding 8, [stratosphere_setup.md](stratosphere_setup.md#cluster-density-matched-to-the-grid)).
5. **AD and FD agree in the plume's growth channels, not in bg_ttl's.** In the plume every N(>8 nm) entry at 24–48 h and N(>12 nm) at 48 h agree; the remaining failures are at 6 h, N(>12 nm) at 24 h and N(>27–53 nm), with FD/AD 0.93–1.20. In bg_ttl FD differs from AD by a median 13% (mostly FD > AD) in N(>8 nm) at 24–48 h and N(>27–53 nm), attributed to AD's blind spots (bin selection, sub-step count). Separately, PPM's tangent can blow up where few new particles meet high H₂SO₄: one 80-bin Phase 2 cell (240 K, 100 ppmv), see [Phase 2](#phase-2-t--so-map-2026-10-27-to-10-29). At 1350 the plume showed the same 20–30% gap: that part came from the MNFIX number reset switching on and off.
6. **The response is nonlinear at the published size.** Most detectable SABRE-bin changes at α = ±1 differ from the linear prediction by more than 20%. Only plume N(>3 nm) along J_bn is linear to within about 10%. An uncertainty budget needs forward runs at α = ±1 (as in Phase 4), not a Jacobian alone.
7. **A model step (operator-splitting dt) of 60 s is wrong at ≥ 1 ppmv SO₂.** The gas clamp sets the sensitivity to exactly 0 and N(>3 nm) is too high. High-SO₂ runs need dt ≈ 0.5–2 s.
   - Nucleation's own sub-stepping (up to 20 sub-steps, each limited to +50% of N_total) does not fix this. It splits only the nucleation process, after chemistry has made the whole step's H₂SO₄ and before condensation runs, so nucleation is still the only sink for that H₂SO₄ within the step.
8. **The cluster density must match the grid density.** With the library default (1350 kg/m³) the 1.7 nm cluster is 0.76 of the bin-0 lower edge, and MNFIX conserves the mass by putting it into fewer, larger particles (1.2·xk[0] each, 1.58× the cluster mass). Where that reset fires, the particle count is about 64% of J·V·dt and N scales with the cluster mass. Matching the densities changed:
   - Phase 0 (1 h): N +50%; r_nuc and ρ_nuc elasticities +2.8 and +0.94 → about −0.01;
   - the plume: N(>3 nm) +52% (40 bins) and +64% (80), N(>12 nm) sensitivities about doubled, and the AD/FD gap closed (finding 5);
   - Phase 2: N(>3 nm) up to 1.5× at ≥ 100 ppbv and 217–223 K;
   - bg_ttl: N within about 4% (the reset rarely fired there).

## Setup

### Differentiable coefficients

`tomas_jax.physics.nucleation.NucleationParams` is a NamedTuple pytree with 25 fields:
- **Dunne 2016 Table S1:** `p_bn u_bn v_bn w_bn`, `p_tn u_tn v_tn w_tn p_an a_n`, `p_bi u_bi v_bi w_bi`, `p_ti u_ti v_ti w_ti p_ai a_i`, with `k = exp(u − exp(v·(T/1000 − w)))`.
- **Riccobono 2014 / Yu 2017:** `k_m`, `delta_h`.
- **Nucleated cluster:** `r_nuc_nm`, `rho_nuc`, `org_frac`.

```python
step = make_step(['so2_chemistry', 'nucleation', 'coagulation', 'condensation'], jit=False)
P = jax.tree.map(jnp.float64, DEFAULT_NUCLEATION_PARAMS)

def n_total(p):
    def body(c, _):
        return step(*c, xk, temp, pres, boxvol, rh, alpha, dt, oh_conc=1e6,
                    org_conc=0.0, nh3_conc=0.0, fion=10.0, nucl_params=p), None
    (Nk, _, _), _ = jax.lax.scan(body, (Nk0, Mk0, Gc0), None, length=60)
    return jnp.sum(Nk)

dN_dP = jax.jit(jax.jacfwd(n_total))(P)   # a NucleationParams of derivatives
```

`common.make_runner` is the study's version of this loop. It adds per-step diagnostics (`n_sub`, clamp, H₂SO₄, J, SO₂) and optional dilution. The study starts from `common.NUC_PARAMS` (`rho_nuc = 1770`), not `DEFAULT_NUCLEATION_PARAMS`.

### Forward mode only

**Why reverse mode fails.** The nucleation sub-step loop is `lax.fori_loop(0, n_nuc, ...)` with a traced `n_nuc`, so reverse mode raises. The same goes for the PPM CFL sub-steps and the adaptive coagulation solver.

**Why forward mode is the right choice here anyway.** It costs about one run per parameter (25), whatever the number of outputs (63 SABRE bins × 48 h). It also stores no scan history.

**Enabling reverse mode later.** It only pays for a scalar objective, such as calibrating against SABRE data.
- The nucleation `fori_loop` would need to become a fixed-length `scan` over `max_nuc_substeps`, masked with `jnp.where`. Results stay the same, but the forward run is slower.
- PPM's uncapped `n_sub` would need a hard cap, which is a numerics change that needs validating.
- `jax.checkpoint` per step would bound memory.

### What gradients cannot see

- **Integer and branch decisions:** the nucleation sub-step count, the gas-exhaustion clamp, MNFIX target-bin selection, and the adaptive coagulation sub-step count. The derivative is exact inside a branch and misses the jump across one. The runner records `n_sub` and the clamp, so each sensitivity can be tagged with its regime. Phase 3's FD check shows the size of this effect (finding 5).
- **PPM condensation's derivative can be unstable.** With few new particles growing fast (low J, high H₂SO₄), the linearised PPM step amplifies the tangent (about ×22 per hour at 100 ppbv and J = 0.01, 217 K) while the model state stays smooth; AD then diverges from FD after a few hours. Found in the prescribed-J analysis ([prescribed_j.md](prescribed_j.md)). It also hit the 80-bin Phase 2 map at 240 K and 100 ppmv (tangent ~1e38 after 24 h). Phase 3's FD check passed in the plume's growth channels; the Phase 2 map has no FD check yet.
- **No RH term in Dunne 2016.** The binary sensitivity to humidity is structurally absent.
- **Correlated coefficients.** The 20 Dunne coefficients were fitted jointly and no covariance was published, so per-coefficient sensitivities must not be summed into a variance. The study uses published uncertainty directions instead (below).

### Published uncertainty directions

From Dunne et al. 2016 SM Sect. 19 and Riccobono et al. 2014 (PDFs in `references/sabre/`). Dunne gives no per-coefficient σ, so the uncertainty is expressed as directions in coefficient space (`run_map.directions`). α = 1 is the published size.

| Direction | Published uncertainty | Δθ |
|---|---|---|
| `J_bn ×2.5`, `J_bi ×2.5`, `J_all ×2.5` | J at 1.7 nm uncertain by a factor 2.5 | Δu = ln 2.5 on the mechanism(s), so J ×2.5^α exactly |
| `H2SO4 cal` | [H₂SO₄] ×1.5 (≥ 223 K) to ×2 (≤ 208 K, low-T clustering) | Δu_x = p_x·ln f(T) for every mechanism, log-linear in T between |
| `joint` (Phase 4) | `J_all ×2.5` + `H2SO4 cal` | |
| Table S2 | alternative single-exponential T form (R² 0.83) | structural, so a forward run (`dunne_form='single_exp'`), not a derivative |

**Limits of the CLOUD measurements:**
- T from 208 to 292 K, so 190–208 K is an extrapolation;
- RH mostly 38%, against about 1–10% in the stratosphere;
- the J_bn = 0 and NH₃ refits are unpublished.

Riccobono's uncertainties (k_m up to ×4 or ÷10) do not enter, because the study uses org_conc = 0. At 217 K the Yu 2017 correction, extrapolated from 278 K, scales the organic rate by about 3×10⁸, so stratospheric runs must keep org_conc at 0.

### Model choices for the stratosphere

Cluster density and composition, sparse-bin density, water uptake, SABRE sizing density, ions, NH₃, organics and OH: see [stratosphere_setup.md](stratosphere_setup.md).

### SABRE output operator (Phase 1, `sabre_bins.py`)

**What it computes.** SABRE 2023 measurements (Lyu et al. 2026, *Science*, doi:10.1126/science.adw8939):
- `rebin_to_sabre` gives dry dN/dlogDp on 63 log bins (20 per decade from 3 nm to 4.24 µm). It is linear in Nk and differentiable.
- `nmass_cumulative` gives the NMASS channels N(Dp > 3, 8, 12, 27, 53 nm).
- `SABRE_NUMBER_REL_ERR = 0.08`.

**Assumed, not published:**
- **The bin edges** (the paper gives none). All functions accept `edges=` to replace them.
- **Sharp NMASS cutoffs.** The real channels have counting-efficiency curves.

## Phase 0: gradient checks (2026-10-23; rerun at 1770 on 2026-11-04 and with pure sulfate clusters on 2026-11-05)

Conditions: 217 K, 55 hPa, fion = 10. Rate-level results (0a, 0b) don't involve the cluster and are unchanged.

- **Rate level.** Gradients of ln J_bn and ln J_bi match the closed forms to machine precision and FD to 1e-10. ∂ln J_bn/∂T = −0.21 K⁻¹, so J_bn rises ×8 per 10 K of cooling.
- **Coefficients.** Every coefficient matches FD to 2e-9 on its own mechanism's log rate.
  - Tangents are finite at NH₃ = fion = org = 0, which needed double-`where` guards.
  - Compare against each mechanism's own rate, not ln J_total: minor mechanisms sit below the FD round-off floor of the total.
- **1 h full model** (40 bins, SO₂ 1 ppbv):
  - `jacfwd` over all 25 fields takes about 2 s and matches FD to 2e-9.

  | | ρ_nuc 1350, sparse 1700, 10% organic | ρ_nuc 1770, sparse 1770, 10% organic | ρ_nuc 1770, sparse 1770, pure sulfate (current) |
  |---|---|---|---|
  | N_total after 1 h [cm⁻³] | 271 | 408 | 408 |
  | new particles per nucleated cluster (ΔN / ΣJ·dt) | 0.655 | 0.996 | not recomputed |
  | elasticity to v_bn, w_bi, u_bi | −13.0, +7.92, −7.88 | −13.2, +8.00, −7.96 | −13.2, +8.00, −7.96 |
  | elasticity to r_nuc, ρ_nuc | +2.83, +0.94 | −0.009, −0.003 | −0.011, −0.004 |

- **Pure sulfate clusters change little after 1 h.** N_total falls by 0.05% and the Dunne elasticities by about 0.1%; AD still matches FD to 2.1e-9. The r_nuc and ρ_nuc elasticities grow by about 20% in magnitude, consistent with each cluster now taking all of its mass (28 H₂SO₄ rather than 25) from the gas.

- **At 1350, new particles were mass-limited.** The cluster was lighter than bin 0's lower edge, so MNFIX kept the mass and reset the number (fewer particles, born larger), and cluster size and density acted as a direct multiplier on N (see [Cluster density](stratosphere_setup.md#cluster-density-matched-to-the-grid)). At 1770 that is gone. The small negative r_nuc and ρ_nuc elasticities that remain are plausibly heavier clusters taking more of the H₂SO₄ per particle.

## Phase 1: SABRE output (2026-10-23; rerun at 1770 on 2026-11-04 and with pure sulfate clusters on 2026-11-05)

6 h at 217 K, SO₂ 1 ppbv, with the stratospheric model choices above.

- **Run time.** The Jacobian of all 63 SABRE bins plus 5 NMASS channels takes 4 s on 40 bins and 6 s on 80 bins (14 s and 39 s before the MNFIX speed-up).
- **Most particles are below SABRE's range.** After 6 h, 73% (40 bins) and 74% (80 bins) of the 9.8e4 cm⁻³ are between 1.7 and 3 nm. At 1350 it was about 89%.
- **N(>3 nm)** is 2.68e4 on 40 bins and 2.51e4 on 80 bins (7% apart). With 10% organic clusters it was 2.74e4 and 2.56e4 (2.1% and 1.7% higher); at 1350, 2.72e4 and 2.37e4 (14% apart). N(>3 nm) rises much less than N_total (+50% after 1 h in Phase 0) because the extra particles share the same H₂SO₄ and fewer grow past 3 nm in 6 h. The steep nucleation mode at the 3 nm cutoff makes this the least robust channel. 80 bins (about 20 per decade, like SABRE) is the primary grid.
- **Sensitivities are large in 3–8 nm and small above 15 nm after 6 h.** In 3–8 nm, d ln(dN/dlogDp)/du_bi reaches 0.77 (40 bins) and 0.96 (80 bins); it was up to 1.5 at 1350. Above 15 nm they are below SABRE's 8% per unit coefficient change.
- **Pure sulfate clusters lower the N(>3 nm) sensitivities by 5–10%.** d ln N(>3 nm)/du_bi is 0.117 (40 bins) and 0.096 (80), against 0.123 and 0.105 with 10% organic; u_bn and p_bi drop by similar amounts. The r_nuc sensitivity changes more: −0.52 → −0.57 on 40 bins and +0.17 → +0.09 on 80, where the two grids already disagree in sign.

## Phase 2: T × SO₂ map (2026-10-27 to 10-29)

`python -m sensitivities.dunne_nuc.run_map [--resolved-dt] [--coag-solver euler_adaptive] [--smoke] [--plot-only]`

**Configuration.** The resolved-dt maps (`results/phase2_dt/`, 40 and 80 bins) are the 1770, pure-sulfate reruns (2026-11-05). The 60 s maps (`phase2`, `phase2_adaptive`) exist only at 1350 (`results/rho1350/`); the resolved-dt map supersedes them, and comparisons with them below are at 1350.

**Setup.**
- **Grid:** T ∈ {190, 200, 208, 217, 223, 240} K × SO₂ from 10 pptv to 100 ppmv (8 decades).
- **Run:** 24 h at 55 hPa with constant OH, starting from 10 cm⁻³ background aerosol (100 nm, GSD 1.6).
- **Fixed conditions:** RH 2%, OH 1e6 cm⁻³ not titrated by SO₂, 10 ion pairs cm⁻³ s⁻¹, no NH₃ or nucleating organics, no dilution.
- **SO₂ is a mixing ratio at 55 hPa.** At higher pressure the same ppbv makes more H₂SO₄, and at ≥ 1 ppmv real OH would be titrated, so the map's H₂SO₄ production at high SO₂ is an upper bound. The map does not stand in for lower altitudes (see [HiLa](#planned-hila-high-latitude-lowermost-stratosphere-scenario)).
- **Hourly outputs:** SABRE dN/dlogDp, NMASS, the Jacobian along the 4 published directions, a Table S2 forward run, and diagnostics.

**Outputs:**

| Directory | dt | Coag | Grids | Status |
|---|---|---|---|---|
| `results/rho1350/phase2` | 60 s | fixed | 40, 80 | **invalid at ≥ 1 ppmv** (gas clamp); at 1350 only |
| `results/rho1350/phase2_adaptive` | 60 s | adaptive | 40 | same dt problem; shows the coag solver matters only at 100 ppmv; at 1350 only |
| `results/phase2_dt` | `run_map.dt_for`: 60 s (≤ 10 ppbv), 10 s (100 ppbv), 2 s (1–10 ppmv), 0.5 s (100 ppmv) | fixed | 40, 80 | **the valid map** (40 bins 56 min alone, 4.4 h before the MNFIX speed-up; 80 bins 4.0 h sharing the CPU) |

**dt matters most.** At 60 s with constant OH, one step makes more H₂SO₄ than nucleation can use, and operator splitting lets nucleation take all of it (the gas clamp).
- Nucleated number then equals the available gas whatever J is, so the sensitivity is exactly 0.
- N(>3 nm) is 1.8× too high at 1 ppmv and 2.4× at 10 ppmv.
- Resolved, δ ln N(>3 nm) ≈ 0.5.

The SO₂ levels that share a dt run as one vmapped batch. The Jacobian is taken along the 4 directions only (4 tangents; it matches the projected full Jacobian to 4e-11).

**Resolved against 60 s (both at 1350).** At ≤ 10 ppbv the two maps are identical. At 100 ppbv, N(>3 nm) is 0.74–0.95× the 60 s value; at ≥ 1 ppmv it is down to 1–80% (2–7% at 10–100 ppmv and 223–240 K).

**Results (resolved dt, 40 bins, 24 h).**
- **Against 1350 (both with 10% organic):** N(>3 nm) is within 4% at ≤ 1 ppbv and 0.76–1.52× at ≥ 10 ppbv (largest at 217–223 K and 1–10 ppmv). N(>12 nm) is within 6%.
- **Against 10% organic clusters:** N(>3 nm) within 3% at ≤ 100 ppbv, 0.87–1.19× at ≥ 1 ppmv (+14–19% at 217–223 K and 10–100 ppmv; −13% at 190 K and 100 ppmv); N(>12 nm) within 6%.
- **≥ 1 ppmv:** along the H₂SO₄ calibration, δ ln N(>3 nm) is 1.1–1.7 at 208–223 K and 10–100 ppmv, where the 60 s map gave 0. These are the largest values on the map.
  - J_bn dominates below 223 K.
  - N(>12 nm) responds only at 0.1–1 ppmv (δ ≤ 0.3); N(>53 nm) hardly responds (|δ| ≤ 0.12, the largest at 10 ppbv and 190–208 K).
- **Table S2** gives 0.28–0.67× the N(>3 nm) at ≥ 10 ppmv and 208–223 K (0.6–0.9× at 240 K).
- **Unresolved:** at 190 K and 100 ppmv the clamp still fires on 50% of steps at 0.5 s.

**Elasticity to J, d ln N / d ln J_all** (`run_map.dlnY_dlnJall`). All four Dunne rates are scaled by one common factor, so this is the J_all ×2.5 result divided by ln 2.5 (Riccobono is not scaled; there are no organics). It needs one tangent instead of 25. 1 means N ∝ J; below 1, more nucleation is offset by coagulation and competition for H₂SO₄. For N(3–10.65 nm) at 24 h (40 bins):
- **0.01 ppbv:** 1.0 everywhere, but N(3–10.65) ≤ 0.7 cm⁻³.
- **1 ppbv:** 0.08–0.14 at every T; N(3–10.65) stays at 2–3e4 cm⁻³ whatever T and J are.
- **Cold (190–200 K), ≥ 10 ppbv:** 0.16–0.32 (the 190 K, 100 ppmv value is biased low by the clamp).
- **Warm (217–240 K), ≥ 1 ppmv:** 0.66–1.18, so N(3–10.65) there tracks J directly.
- **N(>12 nm) and N(>53 nm):** ≤ 0.11 everywhere.
- A prescribed J (scheme bypassed) gives higher elasticities at ≥ 1 ppbv, because it removes the feedback of H₂SO₄ on J ([prescribed_j.md](prescribed_j.md#results)).

**80 bins against 40 (24 h).**
- **N(>3 nm) falls with SO₂ on 80 bins:** within 12% at ≤ 1 ppbv (except 0.1 ppbv at 223 and 240 K: 0.77 and 0.55, at 550 and 40 cm⁻³), 0.5–0.87× at 10 ppbv–1 ppmv (up to 0.99 at 240 K), and 0.27–0.61× at 10–100 ppmv (0.61–0.9× at 240 K). N(3–10.65 nm) falls further, to 0–0.1× at 100 ppmv and 217–240 K. The same grid dependence as the plume (Phase 3).
- **N(>12 nm)** agrees within 10% except at 1 ppbv, where 80 bins give 0.24–0.76× (the growing mode reaches 12 nm later, as in bg_ttl).
- **Sensitivities are larger on 80 bins at ≥ 1 ppmv** and below 223 K: δ ln N(>3 nm) along the H₂SO₄ calibration reaches 1.9 (208–217 K, 10–100 ppmv) against 1.4–1.7 on 40 bins. Table S2 gives 0.31–0.61× at 10 ppmv and 208–223 K.
- **Invalid cell: 240 K, 100 ppmv on 80 bins.** The forward run is fine, but the tangent grows ×1e7 per hour after 12 h, once N(3–10.65) falls below about 100 cm⁻³ with H₂SO₄ still high, and reaches 1e38. This is the PPM tangent instability (see [What gradients cannot see](#what-gradients-cannot-see)). The 223 K, 100 ppmv cell shows the tangent starting to grow in the last 2 h, so its 24 h sensitivities (e.g. elasticity to J of N(3–10.65) = 3.6, with N(3–10.65) = 180 cm⁻³) are suspect too. No FD check has been run on the map.

**Coagulation solver** (`make_step(coag_solver='euler_adaptive')`: the Fortran multicoag.f sub-step limits, forward mode only). The fixed 10-sub-step Euler solver goes NaN above about 3e9 cm⁻³ of clusters (≥ 1000 ppmv at 60 s); the adaptive one stays stable. Up to 10 ppmv the two agree within 7%. At 100 ppmv and 60 s, the fixed solver gives 1.5–2× the adaptive N(>3 nm), but the sensitivities agree within 0.09.

**Possible replacement for `dt_for`:** an adaptive inner step for gas → nucleation → condensation, keyed to the H₂SO₄ lifetime (`docs/future_features.md` §7). Not implemented.

## Phase 3: scenarios, Jacobian and FD check (2026-11-02 to 11-04)

`python -m sensitivities.dunne_nuc.phase3_scenarios [--smoke] [--plot-only] [--grids ...] [--scenarios ...]`, writing to `results/phase3/`.

**Configuration.** Results are from the pure-sulfate rerun (2026-11-05), except the plume dt ladder, which is at 1350 and 10% organic. The older results are in `results/orgfrac0.1/phase3` and `results/rho1350/phase3`.

### Scenarios (`scenarios.py`)

All run 48 h from the Phase 2 background aerosol, with constant OH = 1e6 and the stratospheric model choices. RH comes from the H₂O mixing ratio (Buck formula, as in `calc_k1_so2_oh`).

| name | T | p | H₂O | SO₂ (initial) | dilution | dt |
|---|---|---|---|---|---|---|
| `bg_midlat` | 217 K | 55 hPa | 5 ppmv | 20 pptv | none | 60 s (ladder) |
| `bg_ttl` | 195 K | 100 hPa | 4 ppmv | 50 pptv | none; below CLOUD's 208 K | 60 s (ladder) |
| `sai_plume` | 217 K | 55 hPa | 5 ppmv | 10 ppmv | kdil = ln 2/86400 s⁻¹ toward the `bg_midlat` initial state | 0.5 s (fixed) |

**Why OH is constant.** A reduced HOx scheme with OH titrated by SO₂ was built, checked against SANDBOX, and then reverted (user decision). It is kept on branch `hox-oh`. With constant OH, the plume's H₂SO₄ production is an upper bound, about 70× SANDBOX's full chemistry at 10 ppmv.

### Method

1. **dt ladder** (40 bins): 60 → 10 → 2 → 0.5 s, using a forward run plus one `jvp` along the H₂SO₄ calibration.
   - Pick the coarsest dt whose 48 h N(>3 nm) and N(>12 nm) (5%) and sensitivity (±0.05) agree with the next finer dt.
   - At that dt, compare fixed and adaptive coagulation, and use adaptive if they differ by more than 5%. Fixed won in every scenario (difference ≤ 0.7%).
   - **The plume is fixed at 0.5 s** (user decision). Its ladder (at 1350, not rerun) does not converge in N(>3 nm):

     | dt | 40 bins: N(>3) / N(>12) / δ ln N(>3) | 80 bins: N(>3) / N(>12) / δ ln N(>3) |
     |---|---|---|
     | 60 s | 3.97e5 (clamp 55%) / 3.79e3 / 0.68 | 2.94e5 (clamp 55%) / 3.28e3 / 0.87 |
     | 10 s | 7.94e4 / 1.49e3 / 1.67 | 3.55e4 / 1.45e3 / 2.12 |
     | 2 s | 4.75e4 / 1.42e3 / 1.75 | 1.73e4 / 1.41e3 / 2.19 |
     | 0.5 s | 4.30e4 / 1.41e3 / 1.76 | 1.51e4 / 1.41e3 / 2.18 |

     N(>12 nm) and the sensitivity are converged. N(>3 nm) is still about 10% (40 bins) to 13% (80 bins) from converged. That is small next to the ×2.8 grid difference.
2. **Full Jacobian:** `jacfwd` over all 25 coefficients of the hourly SABRE dN/dlogDp and NMASS, on 40 and 80 bins, plus a Table S2 forward run.
3. **FD check** (40 bins): the 4 directions (step α = 0.1) and the 4 coefficients with the largest elasticity (step 1e-2·|θ|), compared with AD at 6, 24 and 48 h.
   - **Why not two-point FD.** Run-to-run jitter from MNFIX bin selection and PPM sub-steps is about 1% in N(>12 nm), which swamps a two-point difference (the error grew as h shrank).
   - **What is done instead.** The slope is a least-squares fit through 9 points (±0.25…±1 step plus the base run), with σ from the residuals.
   - **Pass criteria.** AD agrees if |slope − AD| ≤ max(1e-3|AD|, 3σ). Entries with 3σ > |AD| count as noise-limited.
4. **Figures:**
   - `phase3_evolution.png`: dN/dlogDp(t), NMASS with the SABRE band and ±1-direction envelopes, H₂SO₄ and SO₂;
   - `phase3_spectra.png`: δ ln dN/dlogDp at 6, 24 and 48 h;
   - `phase3_elasticity.png`;
   - `phase3_fd_check.png`.

### Results

**Baseline after 48 h** (cm⁻³):

| scenario | grid | N(>3 nm) | N(>12 nm) | N(>53 nm) | Table S2 / S1, N(>3 nm) |
|---|---|---|---|---|---|
| bg_midlat | 40 | 14.7 | 9.98 | 9.13 | 3.73 |
| bg_midlat | 80 | 10.2 | 9.96 | 9.11 | 1.35 |
| bg_ttl | 40 | 1.10e4 | 96.7 | 9.27 | 1.29 |
| bg_ttl | 80 | 1.14e4 | 14.8 | 9.28 | 1.16 |
| sai_plume | 40 | 7.27e4 | 1.46e3 | 1.40e3 | 0.45 |
| sai_plume | 80 | 2.85e4 | 1.42e3 | 1.41e3 | 0.37 |

With 10% organic clusters, plume N(>3 nm) was 6.52e4 (40 bins) and 2.48e4 (80), and bg_ttl N(>12 nm) 101 and 15.6; the other entries are within 1%. At 1350, plume N(>3 nm) was 4.29e4 and 1.51e4.

- **bg_midlat** hardly nucleates: about 5 cm⁻³ of new particles on 40 bins and almost none on 80.
- **bg_ttl** makes 1e4 cm⁻³ of new particles, but only the leading edge reaches 12 nm in 48 h. The 40-bin grid moves that edge out faster (N(>12 nm) ×6.5).
- **The plume's ×2.6 grid difference in N(>3 nm) is not MNFIX at nucleation:** it was ×2.8 with the reset active and is ×2.6 without it. Untested candidates:
  - the coagulation scheme (TFL at mass ratio 2, linear sub-bin at √2);
  - bin-0 width (1.7–2.1 nm on 40 bins against 1.7–1.9 nm on 80);
  - PPM resolution of early growth.

**Largest elasticities θ·∂ln N/∂θ at 48 h:**

| scenario, grid | N(>3 nm) | N(>12 nm) |
|---|---|---|
| bg_midlat 80 | w_bi +0.27, u_bi −0.27, v_bn −0.14, p_bi −0.06 | ≈ 0 |
| bg_midlat 40 | w_bi +3.2, u_bi −3.2, v_bn −2.2 | ≈ 0 |
| bg_ttl 80 | v_bn −1.4, u_bn +0.41, r_nuc −0.13 | v_bn −3.3, u_bn +0.98, r_nuc +0.51, p_bn +0.40 |
| bg_ttl 40 | v_bn −2.3, u_bn +0.69, r_nuc −0.16 | v_bn −1.8, u_bn +0.54, r_nuc +0.49, p_bn +0.24 |
| sai_plume 80 | v_bn −33, p_bn +11, r_nuc +8.1, u_bn +6.8 | v_bn −0.81, p_bn +0.28, r_nuc +0.20 |
| sai_plume 40 | v_bn −25, p_bn +8.4, r_nuc +5.8, u_bn +5.1 | v_bn −2.5, p_bn +0.86, r_nuc +0.76 |

ρ_nuc has a third of r_nuc's elasticity everywhere (it enters only through the cluster mass ∝ r³ρ).

**Along the published directions (plume, 48 h, δ ln N, 80 / 40 bins):**

| channel | J_bn ×2.5 | J_bi ×2.5 | H₂SO₄ cal |
|---|---|---|---|
| N(>3 nm) | 0.64 / 0.48 | 0.29 / 0.21 | 1.99 / 1.48 |
| N(>12 nm) | 0.016 / 0.048 | 0.007 / 0.020 | 0.048 / 0.147 |

- **Same ranking on both grids.** On 80 bins, N(>3 nm) is about 35% more sensitive and N(>12 nm) about 3× less sensitive than on 40.
- **Against 10% organic clusters:** N(>3 nm) within 3%; N(>12 nm) about 35% more sensitive.
- **Against 1350:** N(>3 nm) is about 10% less sensitive and N(>12 nm) more than twice as sensitive.
- **The 40-bin grid moves the growing mode past 12 nm earlier.** That is the same grid effect as in bg_ttl.

**FD check** (40 bins; agree = of 15 entries per direction or coefficient):

| scenario | agree | pattern |
|---|---|---|
| bg_midlat | 12–15 | 6 failures that are not noise-limited, in the tiny N(>3 nm) signal (6 and 48 h) and N(>27–53 nm) at 48 h |
| bg_ttl | 5–10 | FD/AD median 1.13 (0.03–2.12) in N(>8 nm) at 24–48 h, N(>27–53 nm) at 6–48 h and N(>12 nm) at 6 and 48 h |
| sai_plume | 6–9 | N(>8 nm) at 24–48 h and N(>12 nm) at 48 h all agree (H₂SO₄ cal, N(>8 nm), 48 h: AD 103.5, FD 103.6 ± 2.6; r_nuc: AD 57.3, FD 57.8 ± 0.8); failures at 6 h, N(>12 nm) at 24 h and N(>27–53 nm), FD/AD 0.93–1.20 |

**At 1350 the plume agreed in only 0–8 entries,** with FD > AD by 20–30% in N(>8–12 nm) at 24–48 h (H₂SO₄ cal, N(>8 nm), 48 h: AD 38.8, FD 49.6 ± 1.5; r_nuc: AD 12.4, FD 5.0 ± 0.1). The FD steps switched the MNFIX number reset on and off, which AD cannot see. bg_ttl changed little, because the reset rarely fired there.

**The remaining disagreement is systematic, not noise.** σ is about 1% of the slope, and FD > AD in most failures (48 of 60 in bg_ttl, 55 of 64 in the plume).
- **Curvature doesn't explain it:** the symmetric stencil cancels curvature, so Phase 4's nonlinearity is not the cause.
- **The likely cause is AD's blind spots.** FD sees particles crossing bin edges earlier or later; AD differentiates with the bin assignment, sub-step count and clamp frozen.
- **Not yet separated by cause.** One test would be to freeze MNFIX bin selection in the FD runs.

**Elasticity to J at 48 h** (sum of the u_bn, u_tn, u_bi and u_ti columns; matches the J_all direction ÷ ln 2.5 to 2e-16):

| Scenario | Bins | N(>3 nm) | N(3–10.65 nm) | N(>12 nm) | N(>53 nm) |
|---|---|---|---|---|---|
| bg_midlat | 40 / 80 | 0.32 / 0.03 | 1.00 / 1.07 | 0.00 / 0.00 | 0.00 / 0.00 |
| bg_ttl | 40 / 80 | 0.08 / 0.05 | 0.08 / 0.05 | 0.06 / 0.11 | −0.002 / −0.002 |
| sai_plume | 40 / 80 | 0.75 / 1.02 | 0.76 / 1.07 | 0.08 / 0.02 | 0.009 / 0.006 |

- **bg_midlat:** N(3–10.65) is 4.7 cm⁻³ (40 bins) and 0.25 (80), so the elasticity of 1 is not measurable.
- **bg_ttl:** strongly self-limited; N(3–10.65) ≈ 1.1e4 cm⁻³ barely responds to J.
- **sai_plume:** close to N ∝ J for N(>3 nm) and N(3–10.65); the 40/80 difference (0.75 vs 1.02) is another case of the grid dependence in finding 4.

## Phase 4: linearity (2026-11-03 to 11-04)

`python -m sensitivities.dunne_nuc.phase4_linearity [--smoke] [--plot-only] [--grids ...] [--scenarios ...]`, which needs the Phase 3 NPZs and writes to `results/phase4/`.

**Configuration.** Results are from the pure-sulfate rerun (2026-11-05). The older results are in `results/orgfrac0.1/phase4` and `results/rho1350/phase4`; against 10% organic, the counts moved by up to 10% and the plume N(>12 nm) changes grew (joint, α = +1: +0.45 against +0.36 on 80 bins), but the pattern is the same.

**Method.**
- **Runs:** α ∈ {−2, −1, +1, +2} along J_bn ×2.5, J_bi ×2.5, H₂SO₄ cal and joint. That is 16 forward runs per scenario and grid, at the Phase 3 dt and coag solver.
- **Comparison:** the nonlinear ln(Y_α/Y₀) against the linear α·(J·d)/Y₀, for NMASS and SABRE bins at 6, 24 and 48 h.
- **"Nonlinear":** at α = ±1, |nonlinear − linear| > 20% of |nonlinear|, counted only where the change is above SABRE's 8%.
- **Outputs:** `linearity_summary.md` (the full table), `phase4_alpha.png`, `phase4_spectra.png`.

**SABRE bins, all hours, α = ±1:**

| scenario | nonlinear / detectable, 40 bins | 80 bins |
|---|---|---|
| bg_midlat | 216 / 565 | 68 / 133 |
| bg_ttl | 3161 / 3895 | 2341 / 3056 |
| sai_plume | 8733 / 12 296 | 7483 / 12 428 |

Most detectable changes sit at the edges of the growing mode, where a small size shift makes a large relative change in a bin.

**NMASS at 48 h** (Δln N, nonlinear against linear; plume figures given as 80 / 40 bins):

| | behaviour | example |
|---|---|---|
| plume N(>3 nm), J_bn | linear within about 10% | α = ±1: −0.58/+0.67 against ∓0.64 (80) |
| plume N(>3 nm), J_bi | convex | α = ±1: −0.21/+0.39 against ∓0.29 (80) |
| plume N(>3 nm), H₂SO₄ cal and joint | saturates | H₂SO₄ cal α = +2: 2.23 against 3.99 (80); 1.65 against 2.95 (40) |
| plume N(>12 nm), H₂SO₄ cal and joint | strongly convex | joint α = +1: +0.45 against +0.07 (80); +0.70 against +0.22 (40) |
| bg_ttl N(>12 nm) | below the line at α = +2 | H₂SO₄ cal: 0.28 against 0.59 (80) |
| bg_midlat N(>3 nm) | detectable only at α ≥ +1, far above linear | joint α = +2: +3.2 against +0.14 (80) |
| N(>53 nm), all | within ±8% at α = ±1 | at α = +2, the plume joint direction reaches +0.11 (40) and +0.088 (80) |

The table and figures also include N(3–10.65 nm) (`linearity_summary.md`).

**Why N(>12 nm) is convex:** more H₂SO₄ pushes more particles past 12 nm within 48 h, while less H₂SO₄ cannot push the count below the background.

## Planned: HiLa (high-latitude lowermost stratosphere) scenario

Target: 13 km at 60°S with 1–100 ppmv SO₂ (a fresh injection plume). Noted 2026-11-04; not implemented.

**Conditions** (approximate climatology, from memory; check against ERA5 or MERRA-2 before use):

| | Summer (DJF) | Winter–spring (JJA–SON) |
|---|---|---|
| Pressure | ~165 hPa | ~150–160 hPa |
| T | ~225–230 K | ~210–218 K; ~200–205 K at the polar vortex edge |
| H₂O | ~5 ppmv (RH over liquid ~1%) | ~5 ppmv (RH ~3–12%) |
| Ion production | ~20–40 pairs cm⁻³ s⁻¹ | same |

**Nearest Phase 2 cells:** T rows 223 K (summer), 217 and 208 K (winter–spring), 200 K (vortex edge); SO₂ columns 1e3–1e5 ppbv. The map cannot be read off for HiLa:
- **Pressure.** At 160 hPa the same mixing ratio is 2.9× more molecules, and k(SO₂+OH) is 1.6× higher (9.8e-13 vs 6.1e-13 cm³ s⁻¹ at 215 K). With constant OH that is ≈ 4.7× the H₂SO₄ production: 1 ppmv at 13 km behaves like ~5 ppmv in the map, 100 ppmv like ~470 ppmv, above the map's top.
- **Pressure also changes** coagulation, condensation (mean free path, diffusivity) and ion recombination; no map cell covers this.
- **OH titration.** At 1 ppmv, k₁·[SO₂] ≈ 5 s⁻¹, so OH is removed in ~0.2 s. With OH held at 1e6 cm⁻³, H₂SO₄ production is likely orders of magnitude too high at 1–100 ppmv. This also affects the existing map's ≥ 1 ppmv cells.

**Adjustments for the HiLa scenario** (a Phase 3-style scenario in `scenarios.py`):
1. **Pressure 160 hPa**, T 215 K and 227 K (add 205 K for the vortex edge), H₂O 5 ppmv via `rh_from_h2o`.
2. **SO₂ 1, 10, 100 ppmv** as mixing ratios at 160 hPa.
3. **Ion production ~30 pairs cm⁻³ s⁻¹** (`fion`), not 10.
4. **OH titration**, or a fixed H₂SO₄ production rate limited by OH production. This matters most: it sets how much H₂SO₄ there is at all.
5. **dt convergence below 0.5 s** at 100 ppmv: with ~4.7× the production per ppmv (before titration), the 0.5 s step that already fails at 190 K and 100 ppmv in the map will not be enough. Re-check fixed vs adaptive coagulation at the converged dt.
6. **Plume dilution** as in `sai_plume`, toward a high-latitude background.

## Run time and performance

**Wall time.** Old MNFIX on one device → now (bin-skipping MNFIX; the FD and Phase 4 forward batches split over 6 virtual CPU devices). Times include compilation.

| simulation | 40 bins | 80 bins |
|---|---|---|
| Phase 3 Jacobian, bg_midlat / bg_ttl | 91 → 17 s | 294 → 36 s |
| Phase 3 Jacobian, sai_plume (0.5 s) | 10 979 → 2703 s (3.0 h → 45 min) | ≈ 11 h (estimated) → 4033 s (67 min) |
| FD batch (64 runs), bg_midlat / bg_ttl | 111 → 8 s | — |
| FD batch, sai_plume | not finished with the old MNFIX; 2746 s with the new MNFIX on 1 device → 861 s on 6 (3.2×) | — |
| Phase 4 (16 runs), bg_midlat / bg_ttl | 9 → 3 s (new MNFIX in both) | 55 → 21 s |
| Phase 4, sai_plume | 919 → 214 s (4.3×; new MNFIX in both) | 6446 → 2388 s (2.7×; new MNFIX in both) |

The 1770 reruns took 2.1 h (Phase 3), 44 min (Phase 4) and 56 min (Phase 2 resolved-dt map, 4.4 h with the old MNFIX). The pure-sulfate reruns ran as three parallel chains and took 2.4 h (Phase 3), 48 min (Phase 4), 1.1 h (Phase 2, 40 bins) and 4.0 h (Phase 2, 80 bins). A full Phase 3 + 4 run of all scenarios and grids now takes about 3 h (2.9 h for the rows above, plus the dt ladders and coagulation checks), With the old MNFIX, the plume's Jacobians alone took 3.0 h (40 bins, measured) and an estimated 11 h (80 bins). The reruns with the new code are bit-identical to the old results in every forward output, including the plume FD check and all Phase 4 batches. The Jacobians agree to 3e-13 of each column's maximum, because `jacfwd` batches its tangents like `vmap`.

**Per-step profile** (`profile_step.py`, plume at dt = 2 s, µs; original → now):

| | 40 bins | 80 bins |
|---|---|---|
| full step | 1106 → 168 | 4520 → 1829 |
| jvp (1 tangent) | 3133 → 692 | 16 203 → 4541 |
| jacfwd (25 tangents) | 32 919 → 5971 | 115 214 → 12 812 |

**MNFIX speed-up.** Before it, MNFIX was 85% of a 40-bin step: it is called about 13 times per step (10 times inside coagulation), and it looped over every bin three times. Now (`core/mnfix_jax.py`):
- **Vectorised.** Phases 1–2 run on all bins at once.
- **Bin-skipping.** Phase 3 stays sequential in Fortran order, because an upward shift feeds later bins. It now runs only from the first out-of-range bin to `hi`, the last out-of-range bin extended by each upward target, and writes only rows k and kk.
- **Bit-identical unbatched.** `tests/test_mnfix.py` checks against the original port in `tests/reference/mnfix_sequential.py`, including edge cases (no-op, 12-bin jumps, cascades, grid ends).
- **Not bit-identical under `vmap`.** Batched results agree to 1e-14; the original was not batch-invariant either.
- **Up and down shift arithmetic is kept as separate expressions.** XLA's FMA contraction changes merged expressions by ulps, and cancellation amplifies that.
- **Cost per call:** 70 → 1.6 µs (all bins in range) and 70 → 24 µs (worst case) on 40 bins.

**Remaining cost.** On 80 bins, coagulation is now about 98% of a step, mostly the linear sub-bin rates × 10 sub-steps. `n_coag_substeps = 10` is fixed whatever dt is, so at dt ≤ 2 s most sub-steps are unnecessary. Scaling it with dt is the next speed-up (untested).

**Multi-core** (6 performance + 12 efficiency cores; plume, per step, 80 bins):

| | 1 job | 3 at once | 6 at once |
|---|---|---|---|
| forward | 1.80 ms | 1.81 ms | 1.97–2.14 ms |
| `jacfwd` (25 tangents) | 11.8 ms | 18.2 ms | 30.4 ms |

- **Forward runs use about one core.** `common.vmap_over_devices` splits the FD and Phase 4 batches over 6 virtual CPU devices, as `pmap(vmap(f))` with padding.
  - `phase3_scenarios.py` sets `--xla_force_host_platform_device_count=6` before JAX starts; an explicit count in `XLA_FLAGS` wins. `phase4_linearity.py` imports it first.
  - On 80 bins this gives 3.0× the throughput (18 runs in 6.9 ms per step, against 16 in 18.6 ms).
  - Measured on the full batches: plume FD 3.2×, plume Phase 4 4.3× (40 bins) and 2.7× (80 bins).
  - Every rerun so far was bit-identical to the single-batch results. That isn't guaranteed: batch size can change multiply-add fusion by ulps.
- **`jacfwd` already uses several threads** (about 620% CPU). Splitting its tangents gains only 1.26×, so it stays on one device. Separate Jacobians can run as separate processes, at about 2× the throughput with 3 at once.
- **XLA's single-thread flags** change nothing measurable.
- **No GPU:** jax-metal has no float64.
