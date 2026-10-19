# Case G2: gas phase only, Level 2 conditions

```bash
cd tomas-jax
.venv/bin/python -m benchmarks.pycham.cases.g2.run       # inputs, PyCHAM x2, TOMAS-JAX x5 (~3 min)
.venv/bin/python -m benchmarks.pycham.cases.g2.analyse   # metrics + figures (~10 s)
```

`run.py --skip-pycham` reuses the PyCHAM output in `pycham-bench/outputs/g2*`. The
write-up for scientists is `docs/aaar/pycham_comparison.md`.

Inputs: the `level2_limonene_soa` gases, T = 298.15 K, P = 101325 Pa, RH = 0.5, 6 h,
constant light (TUV-x at SZA 30° scaled to J_NO2 = 5e-3 s⁻¹). Our J is prescribed to
PyCHAM. All TOMAS-JAX aerosol processes are off. PyCHAM has `number_size_bins = 0`, no
walls and no dilution. PyCHAM uses `update_step = 60 s`. TOMAS-JAX runs at the preset
coupling of 300 s and at 60 s.

Checks built into `run.py` (`results/run_info.json`, `rate_constant_check.json`):

- All 2135 rate coefficients, evaluated by PyCHAM's generated `rate_coeffs.py` with our
  J file, match ours to 4e-16 relative at the same T, M, H2O and RO2. This includes the
  KPP-name → MCM-number J mapping. With PyCHAM's own water vapour the largest
  difference is 2.2e-4.
- PyCHAM's RO2 pool equals ours (146 members). The air number density agrees to 2e-16.
- The gas phase of TOMAS-JAX is bit-identical with the 2000 cm⁻³ seed and with a
  1e-6 cm⁻³ seed, so the spectator aerosol does not touch the gas phase.

## Every remaining difference between the two G2 set-ups

| # | What | Why it cannot be identical | Expected / measured effect |
|---|---|---|---|
| 1 | Rate constants, J and the RO2 sum are frozen per coupling interval in TOMAS-JAX (300 s preset). PyCHAM refreshes them every `update_step` (60 s). | KPP semantics of our operator split. The coupling interval is a model setting (A12). | In the first interval RO2 is frozen at its t = 0 value (0), so RO2+RO2 products (e.g. LIMAOH, CH3OH, C924OH) are exactly 0 at 5 min instead of 1e-5–1e-3 ppb. Max \|rel. dev.\| is 100 % at 5 min, ≤ 26 % at 15 min and ≤ 6 % after 1 h. Max \|NMB\| is 3 % (LIMAOH) and the median \|NMB\| is 1.6e-3. The effect disappears at 60 s coupling. |
| 2 | ODE solver: diffrax Kvaerno3, rtol 1e-3, atol 1e4 molec/cm³ (ours) vs scipy BDF, rtol 1e-4, atol 1e-3 molec/cm³ (PyCHAM `int_tol` default). Each restarts at every interval/update step. | Each model's own numerics. | PyCHAM at its default tolerance differs from itself at rtol 1e-6 by up to 1.5 % (median \|NMB\| 2.9e-4), e.g. LIMONENE at 6 h (0.003 ppb, after 8 e-foldings). Our default differs from our rtol 1e-5 by ≤ 1.5e-4. Almost all of the 60 s coupling residual is PyCHAM's solver error. |
| 3 | Water vapour: Buck (1981) in ours vs Hyland–Wexler in PyCHAM (`water_calc.py`). | Different saturation-pressure formulas. | H2O is 2.2e-4 higher in PyCHAM, which changes H2O-dependent k (KMT06, O1D+H2O, …) by the same amount. Species change ≤ 2e-4 (median 3e-5). With PyCHAM's water in TOMAS-JAX (attribution run) and tight tolerances in both, the median \|NMB\| is 5e-7. |
| 4 | `<34> N2O5 = NA + NA`: TOMAS-JAX replaces the 4e-4 s⁻¹ chamber sink by γ-uptake (γ = 0 here → k = 0). PyCHAM gets a copy of the .eqn with the rate set to `0.0`. | Tutorial edit A1 lives in code, and PyCHAM reads the file. | None (both k = 0, verified in the rate check). |
| 5 | SA (from SO3 + H2O) is a counter in TOMAS-JAX that is reset every interval and handed to the aerosol step as an H2SO4 source. PyCHAM accumulates it as a gas. | Driver design. | It is compared as cumulative H2SO4 formed and agrees within the 300 s/60 s coupling effect (`summary.json`, key species). It is excluded from the all-species statistics. |
| 6 | `PROD` (undeclared product name) is dropped by our parser. PyCHAM carries it. `NA` is the sink product of `<33>`. | Parser convention (A4). | No feedback on any species. Both are excluded from the statistics. |
| 7 | D9/D9PROD are not in the MCM xml. The writer adds SMILES (1-butanol, butylperoxy). | PyCHAM needs a SMILES for every species. | None (D9 = 0 initially in both). PyCHAM's own classifier calls D9PROD a peroxy radical, but it is not in the RO2 pool. |
| 8 | Air number density: ours P/(k_B T). PyCHAM uses R = 8.3144621 (AtChem2) in `rrc_calc` and scipy R for the ppb factor. | Constants. | 6e-8 relative in M inside PyCHAM's rate evaluation. The ppb factor agrees to 2e-16. Negligible. |
| 9 | J timing: ours freezes J at the interval midpoint. PyCHAM applies J.xlsx as a step function from each interval start, filled with our midpoint values. | — | Identical by construction (J is constant in G2). For diurnal cases the J file must be written for the coupling interval of the run being compared. |
| 10 | TOMAS-JAX carries a spectator seed (2000 cm⁻³, water equilibrium only). PyCHAM has no particles. | The driver always has an aerosol state. | None (bit-identical gas phase without the seed). |

## Results (`results/summary.json`)

391 species exceed 1e-5 ppb in PyCHAM at some time (301/354/389 at 1/3/6 h).

| | coupling 300 s | coupling 60 s |
|---|---|---|
| median \|NMB\| | 1.7e-3 | 3.1e-4 |
| 95th percentile \|NMB\| | 6.8e-3 | 2.2e-3 |
| species with \|NMB\| ≤ 10 % | 100 % | 100 % |
| species with max \|rel. dev.\| ≤ 10 % over 6 h | 87 % (the rest only in the first 20 min) | 100 % (max 1.5 %) |
| species within ±10 % at 1 / 3 / 6 h | 100 / 100 / 100 % | 100 / 100 / 100 % |
| wall time (laptop CPU, incl. JIT of the step) | 4.8 s | 19 s |

PyCHAM takes ~16 s (update step 60 s; 17 s at rtol 1e-6). The attribution runs
(TOMAS-JAX 60 s at rtol 1e-5 / atol 1 molec/cm³) take ~27 s. Figures: `g2_key_species.png`,
`g2_pan_species.png`, `g2_particle_relevant.png`, `g2_scatter_all_species.png`,
`g2_ratio_cdf.png`, `g2_attribution.png`.
