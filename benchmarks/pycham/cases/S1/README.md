# Case S1: Level 1 surrogate condensation/evaporation, TOMAS-JAX vs PyCHAM

Preset `level1_volatility`: seed 2000 cm⁻³, GMD 50 nm (placed at 1770 kg m⁻³), GSD 1.6,
30 % non-volatile absorbing organic + 70 % sulfate by mass; one surrogate organic
(MW 200 g/mol) produced at 0.5 µg m⁻³ h⁻¹ for 12 h; 298.15 K, RH 0.5. Run at
c* = 1 and 100 µg m⁻³. No chemistry; coagulation, nucleation and H2SO4 condensation off
(the preset has coagulation on; it is not tested here).

```bash
cd tomas-jax
.venv/bin/python -m benchmarks.pycham.cases.S1.run       # PyCHAM x8, TOMAS-JAX x12 (~6 min)
.venv/bin/python -m benchmarks.pycham.cases.S1.analyse   # metrics + 5 figures (~20 s)
```

`run.py --skip-pycham` re-reduces the raw PyCHAM output in `pycham-bench/outputs/S1_*`;
`--skip-ours` reuses the TOMAS-JAX npz. Settings: `config.S1`. Every PyCHAM input is
derived from the TOMAS-JAX scenario in `case.pycham_setup` and recorded in
`results/pycham_setup.json`. The write-up is in `docs/aaar/pycham_comparison.md`.

## Runs (each for c* = 1 and 100)

| Run | Purpose |
|---|---|
| PyCHAM `base` | Reference: 40 bins (TOMAS-JAX's), update step 60 s, default tolerances. |
| PyCHAM `tight` | Numerics: update step 10 s, rtol 1e-6. |
| PyCHAM `nokelvin` | Attribution: surface tension 0 in the copy `PyCHAM_S1_nokelvin` (no Kelvin term for the surrogate or water). |
| PyCHAM `bins160` | Resolution: TOMAS-JAX's 160-bin initial state. |
| TOMAS-JAX `base` | Preset, 40 bins. |
| TOMAS-JAX `micro10` | Numerics: aerosol micro-step 10 s. |
| TOMAS-JAX `nokelvin` | Attribution: vbs-jax σ = 0. |
| TOMAS-JAX `nokelvin_pytransfer` | Attribution: σ = 0 and PyCHAM's transfer coefficient (`transfer.pycham_sink_patch`). |
| TOMAS-JAX `bins80`, `bins160` | Resolution. |

Each TOMAS-JAX run is a separate process (`ours_s1.py`), so a benchmark-side patch
cannot leak into another run's jit trace.

## How PyCHAM is set up

- **Seed and bins**: TOMAS-JAX's bins with the same per-bin number and dry mass
  (manual bounds in wet radius, `space_mode = man`); bins below 1e-6 cm⁻³ left empty.
  Seed components `core` (sulfate, MW 96.06, does not absorb organics) and `POA`
  (non-volatile, absorbing), mole fractions from the 70/30 mass split.
- **Surrogate**: continuous influx `Cinfl` (ppb/s with TOMAS-JAX's air number
  density); vapour pressure from TOMAS-JAX's c* at the run temperature (298.15 K, with
  ΔHvap = 131 − 11 log c*), converted with PyCHAM's molar mass. POA and SURR share one
  SMILES (C10H16O4), so PyCHAM's mole-fraction Raoult term equals TOMAS-JAX's
  mass-fraction one.
- **Water**: dissociation constants (core 1.8516, POA/SURR 3.2163) that give
  TOMAS-JAX's ezwatereqm water (0.347 kg per kg SO4, 0.289 per kg organics) when
  PyCHAM's Kelvin factor is 1 (`particles.pycham_diss_wrtw`); water does not absorb
  organics.
- **Density**: dry components at an effective 2.170 g cm⁻³ so that
  m_dry/ρ_d + m_w/1.0 equals TOMAS-JAX's Tang wet volume at t = 0; PyCHAM fixes water at
  1.0 g cm⁻³.

Checks in `run.py` (raise on failure): PyCHAM read back the molar masses, densities and
vapour pressure; initial number and dry mass per bin equal TOMAS-JAX's to 1e-9; the
surrogate mass (gas + particle) equals production × time to 1e-6 in both models;
TOMAS-JAX raised no warnings.

## Every remaining difference

Generated with measured numbers into `results/summary.json` → `mismatches`
(`case.mismatches`): Kelvin term, transfer coefficient, Raoult term (matched), molar
mass, particle water (PyCHAM's Kelvin term on water: −13 % total water at t = 0, −9 % at
12 h, −22 % in the 50 nm mode bin), density, size structure, growth across bins (PPM
remap since 2026-10-15; MNFIX before), time stepping, c* temperature, dummy chemistry.

## Results (`results/summary.json`)

| TOMAS-JAX vs PyCHAM | c* = 1 | c* = 100 |
|---|---|---|
| particle SOA at 12 h | +0.65 % (4.944 vs 4.911 µg m⁻³) | −0.26 % |
| SOA max deviation after 1 h | 4.1 % (1–3 h) | 0.28 % |
| gas-phase surrogate at 12 h | −2.9 % | 5e-6 |
| particle water at 12 h | +10 % | +13 % |
| number-mean dry diameter at 12 h | +2.3 % (123.3 vs 120.5 nm at 1400 kg m⁻³) | 0.00 % |
| GSD of dry diameter at 12 h | −3.2 % (1.817 vs 1.878) | 0.00 % |

(TOMAS-JAX with the PPM bin transfer for SOA, re-run 2026-10-15 after the remap was
changed to carry each bin with its own displacement. The first PPM version gave +1.5 %
/ −1.7 %, but moved by +0.5 % / −1.3 % from 40 to 160 bins; the previous MNFIX transfer
gave −12.6 % in diameter and +4.8 % in GSD at 40 bins, SOA +0.44 %.)

Attribution (`S1_attribution.png`): with Kelvin off in both, c* = 100 SOA agrees to
+0.02 % and water to 0.4 %; with PyCHAM's transfer coefficient in TOMAS-JAX as well, the
c* = 1 SOA gap at 1–3 h goes from +4.1 % to −0.9 % and the 12 h mean dry diameter and
GSD to +0.1 % and −0.4 % (−0.2 % / +0.1 % at 160 bins). TOMAS-JAX's mean dry diameter is
123.3 / 123.2 / 123.2 nm and its GSD 1.817 / 1.816 / 1.815 at 40 / 80 / 160 bins (PyCHAM
120.5 nm and 1.878 at either resolution), so the bins no longer matter: the +2.3 % /
−3.2 % are the Kelvin term (PyCHAM's σ is 3× larger, which broadens its distribution)
and the transfer coefficient. Numerics are small in both
models (TOMAS-JAX micro-step: SOA ≤ 0.8 % after 1 h; PyCHAM tight: ≤ 0.03 %).

Figures: `S1_mass_timeseries.png`, `S1_size_distributions.png`,
`S1_mean_diameter.png`, `S1_attribution.png`, `S1_diagnostics.png` (water/dry and
Kelvin factor by size, condensation sinks). Dry diameters in every figure are the dry
mass per particle at 1400 kg m⁻³, so the 50 nm seed peaks at 54 nm.
