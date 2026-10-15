# Case S2: Level 2 limonene SOA, TOMAS-JAX vs PyCHAM

Preset `level2_limonene_soa`: 6 h, constant light (J prescribed to PyCHAM), 10 ppb
limonene, 40 ppb O3, 0.5 ppb NO, 2 ppb NO2, 1 ppb SO2, 298.15 K, RH 0.5. Seed:
2000 cm⁻³, GMD 80 nm, GSD 1.6, 30 % non-volatile absorbing organic + 70 % sulfate by
mass. Coagulation, nucleation and N2O5 uptake are off in both models (the preset has
coagulation on; it is not tested here). H2SO4 condensation and SOA partitioning are on.
γ(N2O5) = 0, and `<34>` has rate 0 in PyCHAM's .eqn copy.

```bash
cd tomas-jax
.venv/bin/python -m benchmarks.pycham.cases.S2.run       # PyCHAM x6 (~45 s each), TOMAS-JAX x5 (~2 min)
.venv/bin/python -m benchmarks.pycham.cases.S2.analyse   # metrics + 10 figures (~30 s)
```

`run.py --skip-pycham` / `--skip-ours` reuse existing output; `--variants a,b` runs
selected PyCHAM variants; `--smoke` runs PyCHAM for 30 min only. Settings: `config.S2`.
The PyCHAM inputs are derived from TOMAS-JAX objects in `inputs.py`, never from TOMAS-JAX
results. `results/` holds the npz files (300 s grid, float32; gas species below 1e-7 ppb
in every run are not stored), `pycham_volatility.csv` (the vapour pressure given to every
PyCHAM component in every variant), `setup.json`, `transfer_rates.json`, `run_info.json`,
`summary.json` (incl. the generated `mismatches` list with measured effects) and
`gas_species_stats.csv`. The write-up is in `docs/aaar/pycham_comparison.md`.

## Runs

| Run | Purpose |
|---|---|
| PyCHAM `class` | **Reference.** Every condensable species at the c* of its TOMAS-JAX class centre. |
| PyCHAM `exact` | Lumping sensitivity: each species at its own Li et al. (2016) c*. |
| PyCHAM `nopart` | No particles: gas-phase counterpart for the partitioning feedback. |
| PyCHAM `massfrac` | Attribution: class-centre c* converted to Pa with one molar mass, which turns PyCHAM's mole-fraction Raoult term into a mass-fraction one. |
| PyCHAM `massfrac_nokelvin` | Attribution: as `massfrac` in the copy `PyCHAM_S2_nokelvin` (surface tension 0: no Kelvin term for organics or water). |
| PyCHAM `native` | Volatility sensitivity (PyCHAM only): the 245 condensables keep PyCHAM's own SMILES-based vapour pressures (Nannoolal et al. 2008). |
| TOMAS-JAX `dt300`, `dt60` | Preset coupling and coupling = PyCHAM update step. |
| TOMAS-JAX `dt300_nopart` | SOA partitioning and H2SO4 condensation off. |
| TOMAS-JAX `dt60_cs_pycham` | Attribution: organic condensation sink × the PyCHAM/TOMAS-JAX transfer ratio (0.69), `attribution.py`. |
| TOMAS-JAX `dt60_cs_pycham_nokelvin` | Attribution: as above with vbs-jax σ = 0. |

## How PyCHAM is set up to mirror TOMAS-JAX

| Item | TOMAS-JAX | PyCHAM input (how it is matched) |
|---|---|---|
| Gas chemistry, J | as in G2 | as in G2: same .eqn, KPP constants, our J per interval in `J.xlsx`, update step 60 s |
| Condensable set | 245 closed-shell organics with Li et al. (2016) log c* ≤ 4.25, lumped into 15 half-decade classes | the same 245 species, each at the c* of its class centre (`class`) or its own Li c* (`exact`) |
| c* → vapour pressure | class c*(298 K), Clausius–Clapeyron with ΔHvap = 131 − 11 log c* to 298.15 K (vbs-jax) | `volP` = the same c*(298.15 K) converted to Pa with PyCHAM's own molar mass (pybel, `pycham_mw.py`), so PyCHAM's equilibrium gas mass equals that c* |
| Never partitioning | radicals (243), closed-shell organics above the top class edge (217), inorganics (20, incl. HNO3 and NA) | `volP` = 1e5 Pa and `ppartit_cutoff` = 100 Pa. All other components carry ≤ 1e-6 µg m⁻³ in the particles (checked) |
| H2SO4 | SA counter → H2SO4 source → TOMAS condensation (α = 1, irreversible) | SA gas condenses in PyCHAM's ODE with `volP` = 1e-20 Pa |
| Seed | per bin: N_k, dry mass; 70 % SO4 + 30 % absorbing non-volatile OA (slot 41) | the same per-bin N_k and dry mass (manual bounds, `space_mode = man`); seed components SA (H2SO4, `seed_diss` 0) and SEEDOA (`seed_diss` 1) |
| Seed OA molar mass | not used (Raoult by mass) | SEEDOA = 6-hydroxyundecanedioic acid, 232.3 g/mol ≈ number-mean MW of the TOMAS-JAX particle SOA (232–233 g/mol at 3 and 6 h), added through a dead reaction `<990001> SEEDOA = PROD : 0.0` |
| Size grid | 40 mass-doubling bins, 1.7 nm – 17.5 µm | TOMAS-JAX bins 4–32 (every bin with N₀ ≥ 1e-6 cm⁻³ plus 4 empty bins on top); bounds at the wet-volume radii of our dry-mass edges |
| Water | ezwatereqm: 0.3470 kg water per kg SO4 (ammonium-bisulfate ratio × 1.2) and 0.2891 kg per kg of any organic at RH 0.5 (A9), no Kelvin term | per-component dissociation constants w.r.t. water: SA 0.347 × MW_SA/18, SEEDOA and every condensable 0.289 × MW_i/18 (`particles.pycham_diss_wrtw`), so PyCHAM's ideal-solution water equals ours component by component when its Kelvin factor is 1; `H2O_diss_wrtorg = 0` (water does not dissolve SOA). PyCHAM keeps its Kelvin term on water |
| Densities | Tang (1997) H2SO4-solution density, organics counted as sulfate (A9): wet seed 1682 kg m⁻³ | 2170 kg m⁻³ for every non-water component (water 1000), chosen so the wet seed volume equals ours at t = 0 |
| Output | every coupling interval | `recording_time_step = 300 s` |

Checks in `run.py` (raise on failure): PyCHAM's molar masses equal the ones the vapour
pressures were converted with; its recorded vapour pressures equal `volP` (4e-11); its
RO2 pool equals ours (146); the t = 0 particle state matches (N 2000.0 vs 2000.0 cm⁻³;
dry mass 2.5644 vs 2.5644 µg m⁻³; SO4 1.7951 vs 1.7951; seed OA 0.7693 vs 0.7693); the
per-component water reproduces TOMAS-JAX's seed water exactly; no non-condensable
component enters the particles.

## Every remaining difference

The full list with measured effects is generated into `results/summary.json` →
`mismatches` (`case.mismatches`). In short:

| # | What | Measured effect |
|---|---|---|
| 1 | **Operator split** (chemistry, then partitioning, per coupling interval) vs one ODE | SOA +13 % at 30 min, +4.2 % at 1 h (300 s); +8.6 %, +3.9 % at 60 s |
| 2 | **Transfer rate**: one diffusion volume (120), mfp 2D/c̄, Zaveri form vs per-species volume (~200–235), 3D/c̄, Jacobson form | PyCHAM's organic sink 0.69× ours (diffusivity alone 0.79×); with ours × 0.69 the 30 min excess falls from +8.6 % to +3.1 % |
| 3 | **Raoult**: mass fraction vs mole fraction | PyCHAM with mass-fraction Raoult: +1.2 to +1.6 % SOA |
| 4 | **Kelvin**: σ 0.025, ρ 1200, class MW vs σ 0.072, mass-mean particle MW incl. water and sulfate | lowers SOA by 0.8 % (ours) vs 1.3 % (PyCHAM) at 6 h, 2.4 % vs 3.8 % at 30 min |
| 5 | **Within-class species identity** (tracker vs explicit species) | particle fractions in class log c* = 0.5: 0.29–0.94 vs 0.81–0.83; top contributors off up to 4× |
| 6 | **Sectional numerics** (fixed bins, SOA moved by the PPM remap, vs moving centre) | at 6 h number-mean dry diameter 213.8 vs 210.1 nm (+1.8 %), GSD 1.362 vs 1.411 (−3.5 %), dry-mass-weighted mean 276.5 vs 281.7 nm; 0.16 % of particles below the seed's 1st-percentile size (PyCHAM 0.05 %; 0.06 / 0.03 % at 80 / 160 bins). The previous MNFIX transfer: 187.4 nm, GSD 1.85, 2.1 % below the seed |
| 7 | **Water**: PyCHAM's Kelvin term on water (bulk water matched per component) | PyCHAM water −9 % at t = 0, −12 % at 1 h, −10 % at 3 h, −9 % at 6 h; with Kelvin off in both: within 2.5 % at 30 min, ≤ 0.6 % after 1 h |
| 8 | **Density**: Tang vs additive (matched at t = 0) | +0.5 % on the 6 h bulk composition (+0.2 % in diameter) |
| 9 | **Seed sulfate identity**: SA is H2SO4 (98.08 g/mol) in PyCHAM, SO4 (96) / NH4HSO4 for water in TOMAS-JAX | mass identical, moles 2 % different; enters PyCHAM's mean MW in its Kelvin term; negligible for SOA (sulfate is not in the absorbing phase) |
| 10 | **Volatility input**: PyCHAM gets our class-centre c* | exact by construction, so S2 tests partitioning and transport for identical volatility, **not** the Li c* (A8); see the `native` sensitivity |
| 11 | Seed and bins, H2SO4 route, particle-phase diffusion (D_b = 1e-10 vs well mixed), gas phase as in G2 | ≤ 1e-6 of seed left out; sulfate 0.0523 vs 0.0522 µg m⁻³; negligible; without particles median \|NMB\| 1.7e-3 |

## Results

SOA [µg m⁻³] (245 condensable species; seed OA and sulfate excluded):

| t | TOMAS-JAX 300 s | TOMAS-JAX 60 s | PyCHAM class-centre c* | PyCHAM own Li c* |
|---|---|---|---|---|
| 0.5 h | 5.65 (+13.3 %) | 5.42 (+8.6 %) | 4.98 | 5.52 (+10.7 %) |
| 1 h | 9.07 (+4.2 %) | 9.04 (+3.9 %) | 8.70 | 9.30 (+7.0 %) |
| 3 h | 13.41 (+2.0 %) | 13.45 (+2.3 %) | 13.15 | 13.61 (+3.5 %) |
| 6 h | 15.86 (+1.1 %) | 15.90 (+1.4 %) | 15.68 | 15.87 (+1.2 %) |
| NMB over 6 h | +2.4 % | +2.4 % | — | +3.4 % |

(TOMAS-JAX with the PPM bin transfer for SOA, re-run 2026-10-15; with the previous
MNFIX transfer the 300 s run was +12.1 / +3.9 / +1.8 / +1.0 % at 0.5 / 1 / 3 / 6 h.)

The water correction (per-component instead of one seed-average ratio, 2026-10-14)
changed PyCHAM's SOA by −0.3 % at 30 min, −0.15 % at 1 h and −0.05 % at 6 h; its water
went from "within 2.5 % after 1 h" (two errors cancelling) to −9 to −12 % (its Kelvin
term on water, the only remaining water difference).

- Gas + particle mass of the 245 species: 39.50 (ours) vs 39.69 µg m⁻³ at 6 h (−0.5 %).
- Per class: within 11 % for every class above 0.3 µg m⁻³ at 1, 3 and 6 h
  (`S2_cstar_class_bars.png`).
- Gas phase with particles: median \|NMB\| 4.2 % (300 s) and 3.7 % (60 s) over 374
  species, 72 % within ±10 %; condensables 5.5 %, the rest 3.6 %. Without particles the
  same set-ups agree to 0.17 %, so the extra spread comes from items 1–5, mostly 5.
- Partitioning feedback (each model against its own run without particles, 6 h): NO
  −78 %, NO2 −79 %, NO3 −78 %, N2O5 −95 %, OH −35 %, HO2 −30 %, O3 −15 %, HNO3 −42 %,
  PAN −71 %, limonene +168 %, the two models within 1 point. 98 % of the SOA is
  N-containing (0.067 µmol m⁻³ of N, two thirds of the initial NOx).
- Volatility sensitivity (`S2_volatility_sensitivity.png`): with PyCHAM's own vapour
  pressures SOA is 0.08× at 1 h and 0.51× at 6 h (7.96 µg m⁻³), 5 % N-containing, and
  the feedback vanishes (NO +1 %, OH +6 %, limonene −4 %). The NOx sink follows A8.

## Attribution (`S2_attribution.png`)

| SOA difference, cumulative | 15 min | 30 min | 1 h | 3 h | 6 h |
|---|---|---|---|---|---|
| TOMAS-JAX 300 s vs PyCHAM | +34 % | +13 % | +4.2 % | +2.0 % | +1.1 % |
| 60 s coupling | +16 % | +8.6 % | +3.9 % | +2.3 % | +1.4 % |
| + transfer rate matched (CS × 0.69) | +2.4 % | +3.1 % | +2.4 % | +1.7 % | +1.1 % |
| + Raoult matched (PyCHAM `massfrac`) | +1.0 % | +1.5 % | +0.8 % | +0.3 % | −0.07 % |
| + Kelvin off in both | −0.75 % | −0.01 % | −0.09 % | −0.35 % | −0.54 % |

With the three largest differences matched (coupling, transfer, Raoult) SOA agrees
within 1.5 % from 10 min on (largest at 25 min). PyCHAM's Kelvin term is stronger than
ours; with Kelvin off in both TOMAS-JAX is 0.01–0.75 % below PyCHAM (largest at
15 min). The remaining differences are not matched: one scalar transfer factor for 245
species (species-dependent through diffusion volumes 200–235), water and density rules,
moving-centre vs fixed bins, the within-class tracker and the seed sulfate identity.
The attribution runs are numerical experiments in benchmark code (`attribution.py`,
separate processes; `pycham_inputs.ensure_pycham_root(kelvin_off=True)`); the model is
unchanged.

## Findings for the tutorial

1. Total SOA agrees with PyCHAM to 1–4.2 % after 1 h (+1.1 % at 6 h) and the class
   totals to ≤ 11 %. Early SOA is faster in TOMAS-JAX because of the operator split and
   the vbs-jax transfer coefficient (items 1–2).
2. Per-species "top contributors" are a bookkeeping estimate: within a class they can
   be off by up to 4× (item 5). Present class totals.
3. Since 2026-10-15 SOA growth moves particles with a PPM remap; the size distribution
   is within 2 % (mean) and 3.5 % (GSD) of PyCHAM and barely depends on resolution. The
   previous MNFIX transfer broadened it and left ~2 % of the particles below the seed
   size (item 6).
4. The NOx-sink feedback (NO2 −79 % at 6 h) and most of the SOA amount follow the Li
   (2016) placeholder c* (A8); S2 cannot test the volatility itself.

## PyCHAM pitfalls found here

- `dens` must be given in **kg m⁻³** (v5.8.0 stores it unconverted; its README says
  g cm⁻³). g cm⁻³ values give ~2 kg m⁻³ particles, an overflowing Kelvin term and a
  "Factor is exactly singular" crash in the water solver.
- `particle_number_concentration_dry` is re-binned by dry volume and does not line up
  with the per-bin composition; use `load_pycham(...)["N_bin"]`.
- Seed components other than `core` must appear in the chemical scheme; a dead reaction
  (`k = 0`) and an xml SMILES are enough.
