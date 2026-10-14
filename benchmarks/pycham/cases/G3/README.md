# Case G3: gas phase only, Level 3 day/night

Inputs: the `level3_day_night` gases, T = 293.15 K, P = 101325 Pa, RH = 0.6, 24 h from
2026-07-01T16:00Z (12:00 EDT) at Buffalo (42.89 N, 78.88 W), and NO2 held at 1 ppb.
Geometric sunset is at 20:53 EDT (t = 8.88 h) and sunrise at 05:47 EDT (t = 17.78 h).

- **TOMAS-JAX:** TUV-x photolysis, every aerosol process off, γ(N2O5) = 0.
- **PyCHAM:** J is prescribed from TOMAS-JAX (`J.xlsx`) and NO2 is held with
  `const_comp = NO2`. `<34>` has rate `0.0`. There are no particles, walls or dilution,
  and `update_step` is 60 s.

```bash
cd tomas-jax
.venv/bin/python -m benchmarks.pycham.cases.G3.run       # PyCHAM x3, TOMAS-JAX x8, checks (~20 min)
.venv/bin/python -m benchmarks.pycham.cases.G3.analyse   # metrics + 7 figures (~30 s)
```

`run.py --skip-pycham` reuses `pycham-bench/outputs/G3_*`. The case uses its own PyCHAM
copy, `pycham-bench/PyCHAM_G3`.

## Runs

| Run | What it is for |
|---|---|
| PyCHAM `j60` | **Reference.** J.xlsx on the 60 s grid, i.e. the J that TOMAS-JAX uses at 60 s coupling. |
| PyCHAM `j900` | J.xlsx on the 900 s grid, i.e. the same J staircase as TOMAS-JAX at the preset coupling. |
| PyCHAM `j60_tight` | As `j60` with `int_tol` rtol 1e-6, to measure PyCHAM's own solver error. |
| TOMAS-JAX `dt900` | Preset coupling of 900 s. |
| TOMAS-JAX `dt60` | Coupling 60 s (= PyCHAM `update_step`). |
| TOMAS-JAX `dt900_heldNO2` | Attribution: NO2 held *inside* the solver, as PyCHAM does (`hold.py`). |
| TOMAS-JAX `dt60_tight*` | Attribution: rtol 1e-5 / atol 1, then + PyCHAM water vapour, then + NO2 held in the solver ("all known differences removed"). |

All attribution switches live in benchmark code (`ours.numerics_override`, `hold.held_in_solver`).
The model and its defaults are unchanged.

## Checks built into `run.py`

Results are in `results/run_info.json` and `rate_constant_check.json`.

- **J:** the J that TOMAS-JAX used in every interval equals the exported J exactly (max
  difference 0) at 900 s and at 60 s.
- **Rate constants:** all 2135 rate coefficients from PyCHAM's generated `rate_coeffs.py`
  and our J.xlsx match ours to 6.7e-16 at 5 times: 13:00, the interval holding sunset,
  00:00, the interval holding sunrise, and 09:00, on both J grids. With PyCHAM's own
  water vapour the largest difference is 2.0e-4.
- **Photolysis is applied in PyCHAM:** daytime Leighton ratio J_NO2[NO2]/(k[NO][O3])
  has a median of 1.188 in PyCHAM and 1.195 in TOMAS-JAX at 60 s.
- **RO2 pool:** identical, 146 members.
- **Air number density:** agrees to 4e-16.
- **Seed:** the gas phase is bit-identical with the 3000 cm⁻³ seed and with a
  1e-6 cm⁻³ seed.
- **NO2 hold:** NO2 = 1 ppb at every PyCHAM output and every TOMAS-JAX output.
- **In-interval NO2:** re-integrating each TOMAS-JAX interval from its recorded start
  state reproduces the run to 1.5e-14. This yields NO2 inside the intervals
  (`G3_no2_hold.png`).

## Every difference between the two G3 set-ups

| # | What | Why it cannot be identical | Measured effect |
|---|---|---|---|
| 1 | **Held NO2.** TOMAS-JAX resets NO2 to 1 ppb after each coupling interval and lets it evolve freely inside the interval. PyCHAM `const_comp` sets d[NO2]/dt = 0 (and its Jacobian row) inside the ODE. | Driver design: the hold is applied in operator-split step 5 (`driver.run`, `y[held] = held_values`). | **Dominant at 900 s.** Inside the intervals NO2 drops to 0.84–0.91 ppb in daylight and to 0.78 ppb in the first hour. Time-mean NO2 is 0.956 by day, 0.992 by night and 0.969 over 24 h. The hold therefore adds 8.4 % less NOx over 24 h (NOy +4.95 vs +5.41 ppb). Effect on the 900 s run: HNO3 −8.7 %, NO −10.6 % by day, OH −3.8 %, HO2 +4.3 %, H2O2 +4.9 %, O3 −1.4 %, SA −2.9 % (NMB). On its own it gives a median \|NMB\| of 10.4 % over 358 species. It scales with the interval: at 60 s the NOx deficit is −0.54 % and the median \|NMB\| 0.58 %. In the sunrise interval (05:45–06:00) NO2 rises to 1.35 ppb as the N2O5/NO3 reservoir is released, and the reset discards it. PyCHAM discards the same NOx through d[NO2]/dt = 0, so both NOy curves step down at 06:00. |
| 2 | **J staircase.** TOMAS-JAX freezes J at each interval midpoint (900 s preset). The reference PyCHAM run has J every 60 s. | KPP-style operator split (A12). The J file must sit on the coupling grid of the run being compared. | Measured in PyCHAM alone (`j900` vs `j60`): median \|NMB\| 4e-4, but spikes up to 49 % in the interval holding sunset (20:45–21:00 still has J from 20:52) and the one holding sunrise (05:45–06:00 has J from 05:52, so light comes on 2 min early). Affected species include NO (×28 at 21:00), NO3, N2O5 and OH. TOMAS-JAX at 900 s shows the same spikes. Largest single deviation of the 900 s run: N2O5 +120 % at 06:00. |
| 3 | **Rate constants and the RO2 sum frozen per interval** (900 s vs 60 s). | As 2. | With J and the NO2 hold matched (`dt900_heldNO2` vs PyCHAM `j900`): median \|NMB\| 3.9e-4, p95 4.3 %, 93 % of species always within ±10 %. The tail is the RO2+RO2 products: LIMALBOH −61 %, LIMAOH −57 %, LIMALAOH −56 %, LIMCOH −54 %, C824OH/C823OH −49 % (NMB). The RO2 sum is frozen at its t = 0 value (0) for the first 900 s, while 55 % of the limonene (2.0 → 0.89 ppb) reacts. This is G2 lesson 1, much larger here because limonene is consumed faster relative to the interval. |
| 4 | **ODE solver:** diffrax Kvaerno3, rtol 1e-3, atol 1e4 molec/cm³ (ours) vs scipy BDF, rtol 1e-4, atol 1e-3 (PyCHAM default). | Each model's own numerics. | PyCHAM default vs rtol 1e-6: median \|NMB\| 2.2e-4, max \|rel. dev.\| 5 %. Ours default vs rtol 1e-5: 2.3e-5 and 5e-4. |
| 5 | **Water vapour:** Buck (ours) vs Hyland–Wexler (PyCHAM). | Different saturation-pressure formulas. | H2O 2.0e-4 higher in PyCHAM at 293.15 K / RH 0.6, so H2O-dependent k differ by up to 2.0e-4. Species effect is below the NO2-hold effect at 60 s. |
| 6 | `<34> N2O5 = NA + NA`: γ uptake in TOMAS-JAX (γ = 0 here; the preset uses 0.02). PyCHAM gets an .eqn copy with rate `0.0`. | Tutorial edit A1 lives in code. | None (k = 0 in both, verified). |
| 7 | **SA counter** (reset per interval in TOMAS-JAX) vs accumulated gas (PyCHAM); PROD, NA, NA_HET, H2O are bookkeeping. | Driver and parser conventions (G2 lessons 5, 6). | SA is compared as cumulative H2SO4: −2.9 % at 900 s (OH −3.8 %), −0.2 % at 60 s. The other species are excluded from the statistics. |
| 8 | D9/D9PROD SMILES, air number density, spectator seed. | As G2 lessons 7, 8, 10. | None (verified). |

With every known difference removed (TOMAS-JAX 60 s, rtol 1e-5, PyCHAM water, NO2 held in
the solver vs PyCHAM `j60_tight`), the two models agree to a median \|NMB\| of 1.5e-6
(max 3.3e-5, max \|rel. dev.\| 4.9e-4) over 361 species.

## Results (`results/summary.json`, 900 s grid, species ≥ 1e-5 ppb in PyCHAM)

361 species are compared over 24 h (NO2, held at 1 ppb in both, is excluded): 178 at
sunset (21:00), 178 at midnight and 123 at noon.

| | coupling 900 s (preset) | coupling 60 s |
|---|---|---|
| median \|NMB\| | 0.104 | 5.7e-3 |
| 95th percentile \|NMB\| | 0.29 | 0.019 |
| species with \|NMB\| ≤ 10 % | 49 % | 100 % |
| species with max \|rel. dev.\| ≤ 10 % over 24 h | 6 % | 100 % (max 2.6 %) |
| within ±10 % at sunset / midnight / noon | 53 / 56 / 38 % | 100 / 100 / 100 % |
| median \|NMB\|, day / night | 0.104 / 0.085 | 5.8e-3 / 4.6e-3 |
| species ≥ 1 ppt only (162): median \|NMB\|, within ±10 % always | 0.036, 20 % | 2.6e-3, 100 % |
| wall time (laptop CPU) | 11 s | 77–90 s |

PyCHAM takes 37–64 s per 24 h run (last full run).

| Species | NMB 900 s (day / night) | max dev. 900 s | NMB 60 s | max dev. 60 s | 900 s, NO2 held in solver | PyCHAM J900 vs J60 |
|---|---|---|---|---|---|---|
| LIMONENE | +5.4 % (+5.4 % / –) | 36 % | +0.4 % | 2.4 % | −0.1 % | +0.0 % |
| O3 | −1.4 % (−1.4 % / −1.4 %) | 2 % | −0.1 % | 0.1 % | −0.0 % | −0.0 % |
| NO | −10.6 % (−10.6 % / −1.3 %) | 46 % | −0.5 % | 1.6 % | +0.0 % | +0.0 % |
| OH | −3.8 % (−3.8 % / −3.9 %) | 24 % | −0.1 % | 0.5 % | −0.4 % | −0.4 % |
| HO2 | +4.3 % (+4.9 % / −1.3 %) | 16 % | +0.2 % | 0.8 % | −0.2 % | −0.2 % |
| NO3 | −1.6 % (+1.7 % / −1.6 %) | 84 % | −0.1 % | 1.7 % | +0.6 % | +0.6 % |
| N2O5 | −2.8 % (+4.2 % / −2.8 %) | 120 % | −0.2 % | 2.5 % | +0.6 % | +0.6 % |
| HNO3 | −8.7 % (−8.9 % / −8.5 %) | 17 % | −0.6 % | 1.7 % | −0.0 % | +0.0 % |
| PAN | −0.9 % (−1.1 % / −0.6 %) | 4 % | −0.0 % | 0.2 % | +0.0 % | −0.0 % |
| HCHO | −0.1 % (−0.3 % / +0.1 %) | 5 % | −0.0 % | 0.5 % | +0.0 % | +0.0 % |
| H2O2 | +4.9 % (+5.4 % / +4.0 %) | 10 % | +0.3 % | 0.6 % | −0.0 % | −0.0 % |
| SA (cumulative) | −2.9 % (−2.8 % / −3.0 %) | 5 % | −0.2 % | 0.5 % | −0.0 % | −0.0 % |
| LIMALNO3 | +1.6 % (+1.2 % / +27.5 %) | 32 % | +0.0 % | 1.1 % | +0.0 % | +0.0 % |
| C517PAN | +4.8 % (+2.4 % / +12.9 %) | 15 % | +0.2 % | 1.7 % | −0.0 % | +0.0 % |

LIMALNO3 and the C*PAN species are the top particle-phase contributors of the full Level 3
run (`results/level3_top_contributors.json`).

## Figures

All figures are in `figures/` and copied to `docs/aaar/figures/pycham/`:

- `G3_key_species.png`: diurnal series and % difference, night shaded.
- `G3_products.png`: limonene products and the particle-relevant species.
- `G3_sunset_sunrise.png`: J staircases and radicals around the transitions.
- `G3_no2_hold.png`: NO2 inside the intervals and the NOx added by the hold.
- `G3_scatter_all_species.png`: all species at sunset, midnight and noon.
- `G3_ratio_cdf.png`: distribution of log10(TOMAS-JAX / PyCHAM).
- `G3_attribution.png`: the attribution CDFs.

## Notes

- `pycham_inputs.scenario_model_var` writes the preset's `hold_constant` as PyCHAM
  `const_comp`; `run.py` checks that it is exactly `NO2`.
- The `G3` settings live in `benchmarks/pycham/config.py` with the other cases.
- `cases/G3/plots.py` holds local variants of `timeseries_grid`, `scatter_snapshots` and
  `ratio_distribution`. They add night shading, a local-time axis, difference masked
  below the threshold, and named snapshots. Colours come from `config.COLORS`.
- The held species (NO2) is left out of the all-species statistics: it is exactly
  1 ppb at every output in both models by construction. It is analysed on its own
  (`G3_no2_hold.png`, `summary.json` → `held_no2`).
