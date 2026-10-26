# TOMAS-JAX Development Progress Report

This file tracks all significant changes to the TOMAS-JAX codebase. Entries are in reverse chronological order (newest first).

---

## 2026-10-25 (Sun) — Indoor losses resolved by particle size and by vapour species

**Branch**: `feat/wall-loss-outputs` (from `dev` dd928b2). Ali (2026-10-25, verbatim): "can you show the
size dependant particle wall losses over time, total amount of loss to the walls and each species as
vapor loss in mass over time? Can you make a PR so it does not interfere with the the current version?"

- `box/driver.py`: `_Budget` also accumulates the deposited number and dry mass per size bin and the
  vapour wall loss per condensable species (per c* class without chemistry), from the same
  per-interval sinks as the totals. `_loss_detail` adds to `RunResult.losses`, only when a loss is on:
  `deposition_rate_per_h`, `deposited_dndlog10dp_cm3`, `deposited_dmdlog10dp_ugm3`,
  `wall_vapour_by_species` (largest 15) and `wall_vapour_other_ugm3` (docs/aaar/coupling.md).
- Nothing else changes: the run itself is untouched, and runs without losses (Urban air, Day and
  night) have the same output as before; `tests/test_aaar_presets_unchanged.py` passes.
- Tests (`tests/test_aaar_series_driver.py`): per-bin deposition adds up to the totals, β(D) dips
  between 50 nm and 2 µm, the species and the rest add up to the wall loss (chemistry and the
  surrogate), runs without losses report no resolved losses.
- Indoor preset: 13.4 µg/m³ of vapour on the walls after 4 h (LIMAOOH 2.3, LIMCOOH 2.0, LIMALAOH
  1.2 µg/m³ lead); 206 cm⁻³ particles (0.39 µg/m³) deposited; β from 17 h⁻¹ at 1.7 nm to 0.008 h⁻¹.
- `code_version` changes with any model source, so the website library has to be stamped again
  after the API is deployed (tomas-api `scripts/precompute_library.py stamp`).

---

## 2026-10-23 (Fri) — Urban air: VOC selector benzene / toluene / m-xylene (MCM v3.3.1), stacked on the full-day Urban (B1)

**Branch**: `urban-toluene` (from `urban-full-day`, PR #30); PR #31. Shantanu (2026-10-22, verbatim):
"Another idea here would be to add a toggle for a VOC like naphthalene that can react to convert
NO to NO2 and boost O3. This would demonstrate concepts around O3-NOx cycle, VOC effect on O3
production, and aerosol coagulation (separately)." Naphthalene is not in MCM v3.3.1; Ali: "OK,
let's do toluene if naphthalene does not exist". Then Shantanu: "May be use one for m-xylene? I
don't want to use benzene or toluene since they react too slowly. 0.5 ppb/h is good enough."
(the 0.5 ppb/h confirms the A26 NO emission); Ali: "yes. let's do it" — a four-way selector.

- `gaschem/mechanisms/{benzene,toluene,mxyl}_mcm.eqn`: the MCM website's BENZENE, TOLUENE and
  MXYL exports in KPP format (inorganic + generic rates), byte-identical, each with a
  `.provenance.md` (URL, 2026-10-23, sha256, MCM citation); `config.AROMATIC_EQN`. `kpp_parser`
  skips the exports' section markers.
- `gaschem/merge.py`: `merge_mechanisms(base, extra, label_prefix)` keeps a reaction once when
  reactants, products and rate expression agree, raises on a conflicting duplicate or atom
  conflict (none in any subset), adds the others with the prefix, RO2 = union. Against
  limonene: benzene 105 shared / 331 new reactions (121 new species), toluene 326 / 505 (181),
  m-xylene 326 / 534 (191); all three chained 1097 species, 3173 reactions, 225 RO2.
- `gaschem/photolysis.py`: J_CH3CHO (TUV-x acetaldehyde) and J_MVK_CO (TUV-x MVK total with the
  MCM branching); every J of every merge has a source (a missing one raises).
- `box/coupling.py`: `aromatic_merge`, `aromatic_species`, `all_aromatic_species`,
  `aromatics_used`, `merged_mechanism`, `tutorial_mechanism(aromatics)`; `box/driver.py` picks
  the subsets per run (a run with one VOC pays for its subset alone). Levels 1–3, Indoor and
  the Urban preset are bit-identical (fingerprints kept; merging everywhere was measured:
  1e-7–1e-5 on gases, 1e-3 on Level 2 SOA, class MW +52 %). The VOC is reported when present;
  the c* fallback warning only where SOA partitioning or nucleation is on; `warmup()` compiles
  the three variants of the urban preset.
- `level1_urban` preset: no aromatic; `scenario.URBAN_VOC_SPECIES`, `URBAN_VOC_PPB_ON` = 5 ppb
  for all three (the same level isolates k_OH: 1.2e-12 / 5.6e-12 / 2.3e-11 cm³ s⁻¹ at 298 K;
  A27). Full day: O3 maximum none 30.9 ppb at 10:00, benzene 31.0, toluene 41.8 at 17:30,
  m-xylene 94.5 at 19:00 (the VOC used up by 16:00); tables in `docs/aaar/coupling.md` and A27.
- PyCHAM checks, cases T1 (toluene), T2 (benzene), T3 (m-xylene) (`benchmarks/pycham/cases/T1`,
  `pycham_inputs.write_merged_eqn(aromatics)`): same RO2 pools; max relative difference over 6 h
  of sunlight at 300 s coupling: toluene 3.8e-4 / O3 2.4e-4 / NO 7.1e-4 / NO2 4.4e-4 / OH 5.3e-4
  / HO2 1.0e-3; benzene 4.2e-5 / 7.2e-5 / 2.1e-4 / 1.5e-4 / 5.1e-4 / 6.8e-4; m-xylene O3 1.3e-4
  / NO 9.1e-4 / NO2 7.5e-4 / OH 5.1e-4 / HO2 9.4e-4 (the VOC 1.7e-2 on its 0.007 ppb remainder);
  131 / 54 / 175 species all within 10 %.
- Cost: 10.0 → 14.0 / 15.7 / 16.0 ms per solver step (benzene / toluene / m-xylene), the Urban
  day 27 → 38 / 44 / 48 s (1.4 / 1.6 / 1.75×); zero when no VOC is chosen.
- Tests: `tests/test_aaar_aromatics.py` (21), fingerprints `level1_urban+benzene/+toluene/
  +mxylene@1h/@24h` added (nothing regenerated), `test_gaschem` adjusted for the two extra J
  sources.
- Docs: `docs/aaar/gaschem.md`, `coupling.md`, `pycham_comparison.md` (T1–T3), ASSUMPTIONS A27
  and question 12, CLAUDE.md.
- Open: a Nannoolal c* table for the aromatic products (they fall back to Li et al. 2016); the
  5 ppb level for all three (m-xylene gives O3 above 90 ppb in the undiluted layer) and the
  night-time HO2 with a VOC on, for Shantanu.

## 2026-10-20 (Tue) — Nucleation inputs: ions, NH3, organic vapour; Level 1 H2SO4 reaches the particles

**Branch**: `aaar-nucleation-inputs` (from `dev` 617b52b). Ali (2026-10-20, verbatim):
"If I unlock all inputs both in level one and level two, when I have nucleation on with
large amount of SO2, the size distribution shows no nucleation. ... We should see
nucleation without ions, NH3, and organic as well. Right? The nucleation should work in
different setup. ... We can document the assumptions".

- Cause: with nucleation on the Riccobono/Dunne scheme got NH3 = ions = organic = 0 (pure
  binary neutral nucleation, J 1e-8 cm⁻³ s⁻¹ at the preset temperatures), and in the
  urban level the chemistry's H2SO4 never reached the particles (condensation off, no SO2).
- `box/scenario.py`: new `Nucleation` block (`Scenario.nucleation`, TOML `[nucleation]`):
  `ion_pairs_cm3_s` (q, 0–1e4, default 4), `nh3_ppb` (0–1e4, default 3; 30 in the indoor
  preset; hint for molec/cm³ values), `organic` (True/False) and
  `organic_log10_cstar_max` (−15 to 10, default −0.5 = c* ≤ 0.3 µg/m³). Checked in
  `__post_init__` like the other inputs; read only with `processes.nucleation` on.
- `box/driver.py`: passes `fion`, `nh3_conc` [molec/cm³] and `org_conc` through
  `AerosolForcing` only when nucleation is on (runs without it build the same forcing as
  before; fingerprints unchanged). BioOxOrg = Σ gas of the mechanism's closed-shell
  organics with log10 c* ≤ the threshold, from the gas state handed to the aerosol step
  each interval (13 LVOC species with the Jathar table; the MCM has no ELVOC). With
  nucleation on `_processes` adds `h2so4_condensation` (urban level) with a warning line
  so the new particles grow. New diagnostics `nucleation_inputs` and
  `nucleating_organic_cm3`.
- Presets: `[nucleation]` sections in `level1_urban`, `level2_indoor` (NH3 30 ppb),
  `level3_day_night`; nucleation stays off everywhere.
- Measured (laptop, defaults): Level 3 24 h bursts in daylight, N 3000 → 23345 cm⁻³ at
  16:30 (H2SO4 5.9e7, J 10.6 at 15:30), mode grown to 42 nm by 24 h, 118 s; inputs off
  one by one: J 1.7e-8 / 4.6e-4 / 9.0 / 10.6. Level 1 urban + SO2 30 ppb: H2SO4 4.4e7,
  J 1.0 at noon, 222 cm⁻³ below 10 nm, N still falls (12609 vs 11533 without SO2); 50
  ppb: 940 cm⁻³ below 10 nm. Level 2 indoor + SO2 30 ppb: ozonolysis OH gives H2SO4
  9.1e7 in 10 min, J 167 (NH3 30 ppb), N 3000 → 74105 at 20 min → 4599 at 4 h. Wall
  times unchanged (28–33 s for 4–6 h). Table: `docs/aaar/coupling.md` "Nucleation inputs".
- Docs: ASSUMPTIONS A13 rewritten, A21 (ions, Laakso 2004 / Hirsikko 2011), A22 (NH3,
  Ampollini 2019 / Behera 2013), A23 (BioOxOrg, Riccobono 2014 / Yu 2017 / Donahue 2012 /
  Schervish & Donahue 2020), A24 (Level 1 coupling), question 9 for Shantanu; coupling.md,
  aerosol_step.md, CLAUDE.md. DOIs verified via Crossref; links only.
- Tests: `tests/test_aaar_nucleation_inputs.py` (checks, rate order binary < ions < NH3 <
  organic at the defaults and inside a Level 3 run, urban + SO2 new particles, indoor + SO2
  burst, Level 3 daylight burst; slow: the full 6 h / 4 h / 24 h runs). Fingerprints not
  regenerated.
- Open: 30 ppb of SO2 indoors is a stress test; the organic path is an LVOC stand-in for
  HOM the mechanism lacks (A23); tomas-api/tomas-fe mirror of the new inputs is piece 2/3.

---

## 2026-10-19 (Sun) — Indoor air holds O3 at 30 ppb; "Urban air"

- `level2_indoor` now holds O3 at 30 ppb (`hold_constant = ["O3"]`; Shantanu: "limonene
  ozonolysis (O3 = 30 ppbv)"). With O3 free the closed room pulled O3 to 5.6 ppb within
  40 minutes and the SOA peaked at 0.23 µg/m³; logged for Shantanu in ASSUMPTIONS A20.
  New default numbers: limonene 20 → 0.006 ppb (13.3 reacted, 6.7 out), OH peak 7.6e5,
  SOA peak 0.96 µg/m³ at 1.8 h (0.45 at 4 h), walls 13.4 µg/m³, ventilated 18.5, SOA
  ventilated 1.36 and deposited 0.03; N 3000 → 4542 cm⁻³, dry 3.8 → 11.4 µg/m³, O:C 0.503,
  κ 0.333; 9.1 µg/m³ SOA with the losses off; window open (3 h⁻¹) SOA peak 0.08; 60 s
  coupling: SOA peak 1.01 (300 s −4 %), reacted 13.1 (+1 %). Wall about 8 s.
- Level 1 of the series is "Urban air" (Ali; it was "Urban warm-up"): preset name,
  comments, tests, A19.
- Tests: `test_aaar_series_presets.py` checks the held O3, the free-O3 variant (2 h fast,
  4 h slow: balance 5.7 ppb, SOA peak < 0.3), the no-loss and window-open variants.
  Levels 1–3 fingerprints unchanged. Full suite (`pytest tests`, tuvx-jax 655668c):
  703 passed, 650 skipped, exit 0, 343 s. Tag `aaar-stable-2026-10-19b`.

---

## 2026-10-19 (Sun) — Series review: volume from the dry mass

- Review of the series work (aaar-stable-2026-10-19 = c8d9794: full suite 702 passed, 650
  skipped; Level 3 2 h identical to a631e80 to the last bit). The reported volume used the
  bin-centre diameter, so in the coagulation-only urban run V fell 0.23 % in 6 h while the
  dry mass stayed to 3e-10. Now dV/dlog10Dp = (dM/dlog10Dp)/ρ and S uses each bin's
  mean-mass diameter (`box/properties.py`); V is constant to 3e-10, S 704 → 669 µm² cm⁻³
  (was 663). Tests tightened (V to 1e-6 in the urban preset test, V × 1.4 = dry mass to
  1e-12). Fingerprints unchanged (S and V are not part of them).

---

## 2026-10-19 (Sun) — Series levels: urban air and indoor air (model side)

**Branch**: `aaar-series` (from `feat/aaar-tutorial` a631e80)

- Shantanu's series (2026-10-18): Level 1 "O3-NO-NO2 under midlatitude mid-day conditions
  and aerosol coagulation, happening together but none affecting the other", Level 2
  "limonene ozonolysis (O3 = 30 ppbv) and some OH under indoor conditions to form SOA",
  Level 3 Day & Night kept. Ali: "Yes: Urban, Indoor, keep Day & Night".
- Scenario fields (`box/scenario.py`): `emissions_ppb_per_h`, `photolysis.cloud_fraction`,
  `losses` (ventilation λ with outdoor gases and an outdoor seed, O3 surface loss k_s,
  condensable vapour wall loss k_w, Lai–Nazaroff particle deposition with room
  parameters), seed salts (`ammonium_nitrate_mass_fraction`, `ammonium`, `organic_o_to_c`).
  All default to off; `tests/test_aaar_presets_unchanged.py` holds bit-level fingerprints
  of Levels 1–3 made at a631e80 (1 h runs fast, full presets slow) and they are unchanged.
- Driver (`box/driver.py`, `box/losses.py`): per coupling interval emissions → chemistry
  (J × (1 − 0.75 CF)) → gas losses → aerosol step → particle losses, each loss solved
  exactly over the interval with its time integral so the budgets close (equations in
  `docs/aaar/coupling.md`, "Emissions and losses"; ASSUMPTIONS A16, A17, A19).
- Outputs (`box/properties.py`, `box/result.py`): dS/dlog10Dp, dV/dlog10Dp and S, V
  totals; O:C of the particle organics (tracker atom counts + seed O:C); κ (Lambe 2011
  for organics, Petters–Kreidenweis salts, volume mixing); cumulative loss budgets
  (`RunResult.losses`): walls, surfaces, ventilation (net), deposition, emissions.
- Presets `level1_urban` (6 h; 40 °N, 1 July, 10:00–16:00 EDT; O3 40, NO 5, NO2 15 ppb,
  1 ppb/h NO; 2e4 cm⁻³ in three modes, 50 % organic, 30 % AS, 20 % AN; coagulation only)
  and `level2_indoor` (4 h, dark; limonene 20, O3 30 ppb; λ 0.5 h⁻¹, k_s 3 h⁻¹,
  k_w 1e-4 s⁻¹, deposition on). Measured: urban photostationary ratio 0.99–1.00, O3 40 →
  36.9, NO 5 → 7.2, NO2 15 → 12.7 ppb, N 20003 → 11757 cm⁻³, dry mass constant, wall
  5.5 s; indoor O3 30 → 5.6 ppb (= λ O3_out/(λ + k_s)), limonene 20 → 1.0 (11.1 out,
  7.9 reacted), SOA peak 0.23 µg/m³ with 7.6 µg/m³ of vapour on the walls and 10.5 out
  (7 µg/m³ SOA with the losses off), κ 0.34, wall 7.5 s. Open point for Shantanu (A20):
  the dark-room default lets O3 fall to 15 % of outdoor and the walls take the vapours.
- Tests: `test_aaar_series_scenario.py`, `test_aaar_losses.py`, `test_aaar_series_driver.py`
  (exact closures), `test_aaar_series_presets.py`, `test_aaar_presets_unchanged.py`;
  `test_presets_load` lists five presets; the nucleation-cap tests run on all five.
- `warmup()` now compiles five presets (two more 300 s shapes and their 60 s variants).

---

## 2026-10-17 (Sat) — Level 3 coupling interval 900 s → 300 s

**Branch**: `aaar-l3-300s` (from `feat/aaar-tutorial` 44b6db2; tag `aaar-stable-2026-10-17d`)

- Prompted by Ali (Figma comments 4 and 5): "15mins seems to be a bit long in my opinion,
  it really depends how fast the chemistry is and how much emissions exist. I think that
  should be defined in the numerics/solver." and "change "numerics" to "numerical
  solver" or something that is clear." Decision (Ali): 300 s now, an interval chosen
  automatically by the numerical solver after the tutorial (planned, not built).
- `box/presets/level3_day_night.toml`: `numerics.coupling_dt_s` 900 → 300 (output every
  1800 s, a multiple of 300, unchanged). All three presets now request 300 s.
- Measured (Level 3 preset, 24 h, 40 bins, nucleation off; final values, deviation from a
  30 s coupling / 2.5 s micro-step reference): 900 s SOA 0.05982 (+38 %), nitrate 3.1373
  (−6.4 %), sulfate 5.5004 (−1.0 %) µg/m³, N 2551.8 cm⁻³ (−0.07 %), GMD 149.57 nm
  (−1.0 %), peak OH −5.2 %, peak NO3 −2.0 %; 300 s SOA 0.04781 (+10.3 %), nitrate
  3.2840 (−2.0 %), sulfate 5.5420 (−0.27 %), N 2553.1 (−0.02 %), GMD 150.62 (−0.31 %),
  OH −1.5 %, NO3 −0.6 %; 60 s SOA +1.8 %, nitrate −0.2 %. The error is first order in
  the interval, X(Δt_c) ≈ X₀ + a·Δt_c. Against 900 s, 300 s gives SOA −20.1 %, nitrate
  +4.7 % (as measured 2026-10-16).
- Nucleation on vs off in the Level 3 preset (no new particles at 293 K): SOA −8.6 %,
  nitrate +1.8 %, sulfate +0.24 %, GMD +0.27 % (was −27 %, +6.6 % against 900 s). On and
  off with the same 60 s / 5 s steps agree to 1e-6: the whole shift is the time steps.
- New preset numbers (300 s): γ 0.002 / 0.02 / 0.1 → nitrate 0.40 / 3.28 / 7.15 µg/m³
  (900 s: 0.39 / 3.14 / 6.81); nitrate 3.284 / 3.281 / 3.280 at 40 / 80 / 160 bins;
  SOA peak 0.36 µg/m³ at 2 h (0.39 at 900 s).
- Wall time, 24 h, warm, laptop with other work running: 13 s (900 s), 18 s (600 s),
  27 s (300 s), 90 s (60 s), 157 s (30 s), 213 s (reference); one thread (as the API
  workers) 13 / 27 / 85 s at 900 / 300 / 60 s. With nucleation on Level 3 still takes
  112–125 s (60 s / 5 s steps whatever the request).
- `driver.warmup()` compiles each preset at its own interval, so it now covers Level 3 at
  300 s (and the nucleation 60 s variant as before); a run without nucleation at 600 or
  900 s compiles once on its first call in a process (about 2–3 s). For tomas-api:
  `code_version()` changes (it hashes the preset TOML), so every cache key changes and
  the prewarm has to run again after deploying; the Level 3 prewarm runs (preset and
  γ grid) take about 27 s each instead of 13 s; `limits.py` may keep {300, 600, 900}.
- Docs: `docs/aaar/coupling.md` new section "Coupling interval (numerical solver
  setting)" (equation, table, wall times, planned automatic interval), the nucleation
  side-effect paragraph, presets table, Level 3 SOA paragraph and the compile note;
  ASSUMPTIONS A8 (Level 3 numbers), A12 (choice of interval, sensitivity table), A13
  (side effect, cost); `gaschem.md`, `aerosol_step.md`, `pycham_comparison.md` and the G3
  README say the PyCHAM G3 runs used the 900 s preset of 2026-10-14 (the case keeps
  900 s); CLAUDE.md.
- Tests: `test_presets_load` checks that every preset requests 300 s;
  `test_warmup_covers_capped_preset_runs` now also runs each preset with nucleation off
  and checks the gas solver's compile count; the slow 80-bin Level 3 test expects nitrate
  3.281 (was 3.134 at 900 s; band 1.22 cm⁻³ < 1.5, 0.1 % quantile 78.5 nm > 75).
  Full suite (`pytest tests`, TOMAS-JAX venv with tuvx-jax 655668c): 667 passed, 650
  skipped (the Fortran 24 h comparisons) in 255 s, exit code 0.
- Not changed here (other repositories still say Level 3 = 900 s or 15 min): the
  tutorial's `docs/INSTRUCTORS.md` and flowchart, tomas-web `src/content/assumptions.ts`
  and `src/constants/tutorial.ts` (nucleation note: "every 15 minutes", SOA "about a
  quarter lower", nitrate "about 7% higher", and a run time), tomas-api
  `docs/aaar/INSTRUCTOR_GUIDE.md`.

## 2026-10-17 (Sat) — tuvx-jax pin moved to 655668c (Windows folder names)

- Why: the laptop app's Windows test run showed that the netCDF/HDF5 C library cannot
  open a file whose path has non-ASCII characters (a cache folder named
  "cache-José-データ"). tuvx-jax 655668c reads each data file's bytes in Python and
  opens them from memory, so the folder name no longer matters. Its data files and
  arrays are unchanged.
- `pyproject.toml` (photolysis extra): tuvx-jax `e5d9032` -> `655668c`. No model change.
  `docs/aaar/pycham_comparison.md` still names `e5d9032`: that is the version the
  2026-10-15 comparison runs used.
- Level 3 preset, 2 h, old and new tuvx-jax in otherwise identical environments: all
  937 values in the run result are identical except `code_version` (it hashes the
  tuvx-jax sources) and the run time; J for all 28 channels in every coupling interval,
  the full gas vector and the particle number and mass in every bin are bit-identical.
- Tests: `pytest tests` with tuvx-jax 655668c installed: 667 passed, 650 skipped
  (the Fortran 24 h comparisons) in 269 s.

## 2026-10-17 (Sat) — Measured 48 h nucleation run times

- ASSUMPTIONS A13 and docs/aaar/coupling.md: the 48 h nucleation run times are now
  measured (Level 3 235 s, Level 2 204 s, one thread through the tomas-api worker job
  path) instead of estimated, and the API's tutorial job limit is 480 s (was 120 s).
  Docs only; `code_version()` is unchanged.

## 2026-10-17 (Sat) — Input checks on the scenario

**Branch**: `aaar-input-checks` (from `feat/aaar-tutorial` 49011b4)

- Prompted by Ali's go-ahead ("Fix now, small patch") after a check through the tutorial
  helpers showed three silent mistakes: RH 80 ran as RH = 80, `hold_constant="NO2"` held
  N, O and 2, and `coagulation="off"` stayed on.
- `box/scenario.py`: each class (`SeedMode`, `Seed`, `Environment`, `Photolysis`,
  `Surrogate`, `Processes`, `Numerics`, `Scenario`) checks its fields in `__post_init__`
  and raises `ValueError` naming the field, with hints for a percentage given as a
  fraction ("did you mean 0.8?") and a Celsius temperature. Process switches must be
  `True`/`False` (numpy bools allowed). A string `hold_constant` is one species.
- Valid inputs are unchanged: short Level 1 and Level 2 runs are bit-identical to 49011b4
  (only `code_version` and the runtime differ).
- Tests: `tests/test_aaar_scenario_checks.py` (15).
- Docs: `docs/aaar/coupling.md` (Inputs and outputs), CLAUDE.md.

## 2026-10-16 (Fri) — Short time steps whenever nucleation is on

**Branch**: `aaar-nucleation-dt` (from `feat/aaar-tutorial` 5227ddd; committed as 49011b4,
tag `aaar-stable-2026-10-17`)

- Prompted by Ali: a 15 min interval is far too long for nucleation; 1–2 min at most.
  A convergence study (Levels 1–3 with bursts at 265–270 K, 40 bins, reference 30 s
  coupling with a 1 s micro-step) found that the aerosol micro-step matters more than
  the coupling interval: at 60 s coupling and the 60 s micro-step, peak N is +30 % in
  Level 2 (265 K, SO2 50 ppb) and +78 % in Level 3 (265 K, SO2 20 ppb). 60 s coupling
  with a 5 s micro-step is within 5 % on peak N, final N and GMD in every nucleating
  case (largest +3.5 %). Details: ASSUMPTIONS A13, `docs/aaar/coupling.md` ("Time
  steps with nucleation").
- `box/driver.py`: `NUCLEATION_MAX_COUPLING_DT_S = 60.0`, `NUCLEATION_MAX_MICRO_DT_S =
  5.0` and `time_steps(scenario, aerosol_cfg, nucleation_caps)`, used once at the start
  of `run`. With nucleation on, the requested interval is split into
  m = ⌈Δt_req / 60 s⌉ equal parts (300/600/900 s → 60 s) and the micro-step is
  min(requested, 5 s); output times and the end time are unchanged. One warning line
  says what was shortened; `diagnostics` gains `requested_coupling_dt_s`,
  `requested_micro_dt_s` and `micro_dt_s` (`coupling_dt_s` is now the value used).
  `run(..., nucleation_caps=False)` turns the caps off for convergence studies.
  `warmup(nucleation=True)` also compiles each preset with nucleation on at 60 s
  (warmup 8 → 11 s).
- Runs without nucleation are bit-identical: all three presets at full length give
  the same `RunResult` (every field and diagnostic except the three new keys,
  runtime and code version) as 5227ddd.
- Cost with nucleation on (warm, laptop): Level 1 1.3 → 14 s, Level 2 5.4 → 25 s,
  Level 3 11.5 → 112 s for the preset durations, whether or not a burst happens. The
  API allows runs up to 48 h, which would take about 225 s (Level 3) and 200 s
  (Level 2; both estimated in proportion to duration). Decision for the API: raise the
  tutorial job limit (`JOB_TIMEOUT_S`, now 120 s) to about 400 s, or limit the run
  length when nucleation is on. The 60 s cap stays: the 120 s fallback is within 5 %
  only because its −4 % coupling error (Level 2) cancels part of the micro-step error.
- The 5 % accuracy is for 40 bins. At 80 bins (notebook 01): Level 1 at 265 K with 60 s / 5 s is
  +5.9 % on peak N (final N +2.8 %, GMD −1.0 %) against a 30 s / 0.5 s reference that
  is itself not converged (1 s differs by 2 %); 60 s / 2 s is +4.5 %. So the cap is
  not within 5 % at 80 bins; the web API uses only 40 bins. Level 1 takes 52 s at
  80 bins (5 s) and 138 s (2 s) for 12 h.
- Level 3 with nucleation on but no burst still changes SOA (−27 %) and nitrate
  (+6.6 %) at 24 h, because the coupling interval drops from 900 s to 60 s; the 60 s
  values are the accurate ones (A12, A13). Noted in A13 and `docs/aaar/coupling.md`.
- The warning names no repository file, since the web app shows it to students.
  `micro_dt_s` is documented as the micro-step target.
- Tests: `tests/test_aaar_nucleation_dt.py`, 14 tests (caps per preset, splitting
  rule, warning and diagnostics, observer sees 60 s intervals, nucleation off
  unchanged, Level 1 265 K burst within 5 % of a 30 s / 1 s reference and the uncapped
  run more than twice as far off, and no new aerosol-step compile for a capped preset
  run after `warmup()`).

---

## 2026-10-16 (Fri) — Per-species c* table from Shantanu Jathar is the Level 2/3 default

**Branch**: `aaar-cstar` (from `feat/aaar-tutorial` 2faa391; not committed)

- New input: `tomas_jax/gaschem/mechanisms/mcm_limonene_cstar_298K.csv`, sent by Shantanu
  Jathar on 2026-10-16, kept byte-identical (sha256 `598273949d8d…2b49`; provenance in
  `mcm_limonene_cstar_298K.provenance.md`). Nannoolal et al. (2008) vapour pressures from
  PyCHAM 5.8.0 at 298.15 K, C* = 1e6·MW·Psat/(R·298.15). `.gitignore` ignored `*.csv`
  (it would have been left out of git and of the wheel): `!tomas_jax/gaschem/mechanisms/*.csv`
  added; `.gitattributes` marks those files `-text` (the file mixes LF and CRLF).
- `gaschem/volatility.py`: `load_cstar_table` skips `#` lines, accepts `cstar_ugm3` or
  `cstar_ug_m3_298.15K`, rejects duplicates, missing columns and non-positive values
  (return type unchanged). `assign_classes(..., table_source=)` records closed-shell
  organics missing from the table in `ClassMap.fallback` (Li et al. 2016 c*); the driver
  reports them as `diagnostics["cstar_fallback"]` and labels the default table
  "Shantanu Jathar (2026-10-16) table: Nannoolal et al. (2008) vapour pressures via
  PyCHAM 5.8.0, c* at 298.15 K" in `cstar_source` (`gaschem/config.py`:
  `DEFAULT_CSTAR_TABLE`, `CSTAR_TABLE_SOURCES`).
- Presets `level2_limonene_soa` and `level3_day_night` set `cstar_table`; `Scenario`
  still defaults to `None` (Li et al. 2016). S2 benchmark pins `cstar_table=None` (its
  PyCHAM inputs are built from Li c*; it would raise with the table).
- Coverage: the table has 700 rows, all mechanism species (726 with NA_HET): 458 of the
  462 closed-shell organics and 242 of the 243 organic radicals (radicals ignored). Missing
  closed-shell organics: LIMONENE, D9, CO, CH4 (excluded by the table on purpose; Li
  log10 c* 6.7–11.1, gas-only). Missing radical: D9PROD. 228 species condense (Li: 245);
  per class 0, 1, 2, 2, 5, 7, 14, 11, 23, 16, 23, 31, 15, 35, 43.
- Partial c* tables: when a table leaves condensable species on Li et al. (2016), the
  run adds a `diagnostics["warnings"]` entry (`driver._cstar_fallback_warnings`); the
  shipped table leaves none, Li runs (`cstar_table=None`) never warn. Prompted by the
  tutorial's `run(cstar_ugm3=...)`, which writes only the user rows, so all other
  species silently fell back to Li (2-row table: 243 of 244 condensable species Li).
- Measured impact (ASSUMPTIONS A8, A12; 40 bins, each preset run with the table and
  with Li):
  - Level 2 at 6 h: SOA 8.66 µg/m³ (Li 15.86), yield 0.156 (0.285), GMD 159 nm (203),
    5.0 % N-containing (98.1 %); -OOH species 90 % of the SOA, PANs 1 % (Li 3 % / 84 %);
    top contributor LIMALOOH 3.71 µg/m³ (Li: C822PAN, now gas-only at 10^4.85).
  - NOx dependence reversed: NOx 0 / 0.5 / 1 / 2.5 / 5 / 10 / 20 ppb → SOA 8.54 / 8.15 /
    8.19 / 8.66 / 6.54 / 1.02 / 0.16 µg/m³ (Li 0.01 → 65). No NOx-sink feedback (NO
    +1.3 %, OH +6.4 % vs no partitioning; Li −78 %, −35 %).
  - Limonene 1 / 2 / 5 / 10 / 20 / 40 ppb → SOA 0.04 / 0.23 / 2.99 / 8.66 / 17.84 /
    38.4 µg/m³, yield 0.008 → 0.175 (Li saturated near 16–17). O3 10 → 160 ppb → SOA
    6.4 → 17.1 (Li 13.7 → 18.0). One decade up in c* removes 73 % of the SOA (Li 64 %).
  - Level 3 at 24 h: SOA 0.060 µg/m³ (Li 7.85; peak 0.39 at 2 h, then decline as OH
    and photolysis destroy the gas-phase hydroperoxides), nitrate 3.137 µg/m³ (Li 5.160)
    at 40 bins, 3.134/3.134 at 80/160; γ 0.002 / 0.02 / 0.1 → 0.39 / 3.14 / 6.81
    (Li 0.98 / 5.16 / 7.71). k_het ×1.22 over the first 4 h (Li ×2.76), ends 23 % below
    4e-4 s⁻¹ (Li +20 %). Day-1 OH peak 1.18e7 (Li 1.72e7). Coupling 600/300 s: nitrate
    +2.6/+4.7 %, SOA −10.5/−20 % (Li +2.4/+4.7 %).
  - Level 1 and the Li runs of Levels 2 and 3 are bit-identical to 2faa391 (67 arrays).
    Run times unchanged (Level 2 median 5.8 s, Level 3 12.8 s).
  - vs PyCHAM `native` (S2 set-up, same Nannoolal volatilities): 8.66 vs 7.96 µg/m³ at
    6 h (+9 %), +39 % at 30 min, mostly from lumping into class centres;
    N share 5.0 vs 4.9 %, same leading species.
  - Full report: the session scratch `cstar/impact.md` (not in the repo).
- Tests: loader (comments, CRLF, both columns, duplicates, missing column), sha256 of the
  shipped table, coverage and fallback, class counts, presets use the table,
  `ClassBridge` mass conservation with both c* sources, Level 2 `cstar_source` and
  `cstar_fallback`, partial-table warning, Level 2 6-h science check (SOA 6–12 µg/m³,
  N share of the top contributors < 0.2, LIMALOOH first, no warnings). Changed expected
  values: Level 3 80-bin nitrate 5.151 → 3.134
  (same rel 2e-3; band 0.93 → 1.35 cm⁻³ still < 1.5); Level 2 `cstar_source` prefix
  "Li et al." → "Shantanu Jathar (2026-10-16) table". Li-based tests (245 species, Li
  examples) unchanged.
- Docs: ASSUMPTIONS A8 rewritten, A6/A12 numbers annotated or re-measured, question 1
  for Shantanu closed with follow-ups (PAN decomposition, Level 3 SOA, NOx dependence,
  boiling-point method); `docs/aaar/gaschem.md`, `soa_partitioning.md`, `coupling.md`
  (Level 3 SOA peak/decline and its coupling sensitivity), `aerosol_step.md`,
  `pycham_comparison.md` (early-time gap from lumping), S2 README, CLAUDE.md.
- Next (aaar-tutorial repo, after the pin moves to this commit): notebooks 02–04,
  README and INSTRUCTORS.md describe the Li placeholder (98 % nitrates, C822PAN as top
  contributor, "wait for the curated table"); `write_cstar_table` / `run(cstar_ugm3=...)`
  must merge the user values into the Jathar table (`load_cstar_table(MECHANISM_DIR /
  DEFAULT_CSTAR_TABLE)`, update, write all rows as 10**v), otherwise the run now warns;
  nb02 cells 22/24/38 and nb04 cell 43 must load the table; the PAN experiment
  (−1 % merged) needs replacing (e.g. hydroperoxides → 1e5: 8.66 → 0.16 µg/m³);
  `tests/test_results.py:93` (`nitrogen_share > 0.95`) must change; cached examples
  need regenerating.

## 2026-10-16 (Fri) — Nitrate goes to thin tail bins too (Level 3 band at 40–90 nm)

**Branch**: `feat/aaar-tutorial-ppm-soa` (not committed)

- Audit finding: in Level 3 a band of a few particles cm⁻³ stayed behind at 40–90 nm
  after sunset and grew with resolution (N at 40–70 nm at 24 h: 3.1 / 3.2 / 4.3 / 6.6 cm⁻³
  at 40 / 80 / 160 / 320 bins). Cause: `aerosol_step` put N₂O₅-uptake nitrate only on
  bins with Nk > NEPS_CONDSINK = 1e10 #/cell (1 cm⁻³ in the tutorial box), so thin tail
  bins stopped growing at night while their neighbours moved on.
- Fix (`box/aerosol_step.py`): gate `NITRATE_N_MIN` = 2e-5 #/cell, just above MNFIX's
  empty-bin floor (1e-5), and each bin's share scaled by Nk/Nk(start of the outer step).
  The driver's weights are frozen per coupling interval; with only a lower gate, a bin
  that empties inside the outer step piles its start-of-step share onto MNFIX's floor
  particles (MNFIX then trims it: −0.4 % nitrate in
  `test_nitrate_injection_conserved[mnfix]`, 1 h step). The remap's 1e-10 was tried first
  and failed that test and `test_nitrate_injection_mass_and_slot`.
- Level 3 after the fix: band gone (N at 40–70 nm 2.7 / 0.93 / 0.35 / 0.10 cm⁻³; 0.1 %
  quantile diameter 68 / 69 / 62 / 60 → 73 / 83 / 82 / 83 nm). Nitrate total −0.01 to
  −0.03 % (5.160 / 5.151 / 5.148 / 5.148 µg m⁻³), N ≤ +0.02 %, median diameter −0.05 to
  −0.17 %. MNFIX setting: nitrate −0.02 to −0.04 % (40 to 320 bins). Particle nitrate vs `nitrate_added`
  ≤ 3e-15 every outer step, no driver warnings. Level 1 and 2 bit-identical (no nitrate).
- Tests: `tests/test_aaar_nitrate_gate.py` (unit test, both bin-transfer settings: a
  0.5 cm⁻³ bin gets its share, N = 0 / MNFIX-floor / below-gate bins get none, exact
  nitrate budget; slow Level 3 test at 80 bins for the band).
- Re-measured for ASSUMPTIONS: A12 coupling sensitivity (Level 3, 40 bins, vs 900 s):
  600 s nitrate +3.5 %, SOA +2.4 %; 300 s +6.6 %, +4.7 % (was "≤ 3 % vs 600 s"). A8: 98.1 %
  of the Level 2 SOA is nitrogen-containing with all 245 condensed species tracked (was
  "99 %").

## 2026-10-15 (Thu) — PPM bin transfer: final fixes after the last verification round

**Time**: ~03:00 PDT 2026-10-15
**Branch**: `feat/aaar-tutorial-ppm-soa` (pushed to `feat/aaar-tutorial`)

- `ppm_remap.bin_edge_shifts`: minmod edge slope. A bin whose displacement is a local
  extremum (e.g. one unchanged sulfate bin between growing SOA bins) gets no slope, so its
  particles stay in place; before, 43 % of such a bin's sulfate could move into the next bin
  (adversarial review, final round). Test `test_unchanged_bin_between_changing_neighbours_stays`.
- `driver.run` passes `AerosolStepConfig.micro_dt` to `aerosol_outer_step` (it was silently
  ignored). Default 60 s is bit-identical. Test `test_driver_passes_micro_dt`.
- Decision on A5: totals are judged against the converged 320-bin value, not the previous
  code (Ali, 2026-10-15). Level 3 nitrate at 40 bins is +4.0 % against the old code but within
  0.21 % of the converged 5.149 µg m⁻³; the old 40-bin value was 3.6 % low.
- Documented limitation: an abrupt +35 K step after strong growth leaves ripples at 60 s
  micro-steps (smooth at 10 s; +25/+30 K smooth). Not reachable in the presets.
- Final verification round (independent agents): PyCHAM sizes pass (numerics converged; the
  remaining +2.3 % mean diameter / −3.2 % GSD in S1 is the Kelvin term and the vbs-jax
  transfer coefficient, +0.1 % / −0.4 % with those matched); runtime pass (PPM faster than
  MNFIX at 80 bins in clean A/B pairs, 18.9 vs 22.7 s for Level 3); conservation to 5e-15.

## 2026-10-15 (Thu) — PPM bin transfer, round 3: each bin moves with its own displacement; composition slopes

**Time**: ~01:40 PDT 2026-10-15 (08:40 UTC)
**Branch**: `feat/aaar-tutorial-ppm-soa`

### Summary
Independent verifiers of the previous round reported that the flux-form remap moved
particles wrongly when a bin had to move more than about one bin in one call, piled
particles up next to unchanged ones, and was not smooth at 160 bins in three hard cases
(+30 K after growth, slow Kelvin ripening after a zero-OA bootstrap, bimodal
evaporation). All reproduced. `remap_mass_change` is now a Lagrangian remap of the same
PPM sub-bin profile, with a composition slope inside each bin. It fixes all three and
also the documented round-trip limitation (grow, then evaporate).

### Verifier findings and what was done
- **Multi-bin moves / pile-up (real, fixed).** The flux-form step froze one displacement
  per bin edge for all sub-steps, so a bin moving several bins took the displacement of
  the bins it crossed. Reproduced (80 bins, SOA-only bins 30–39 ×10 next to unchanged
  sulfate bins 40–49): 8.25e12 #/cell in bin 40 against 2e12, 2.96 % of the sulfate
  moved, SOA mode centre at bin 40.0 instead of 41.1; shrinking SOA (keep 1 %) stopped at
  mean bin 48.0 instead of ~41. Now: 2.0e12, sulfate unchanged to 1e-16, SOA centre 41.15
  (exact 41.14) and 41.21 (exact 41.21). New tests
  `test_remap_moves_particles_past_unchanged_ones` (5 cases, growing and shrinking 3–13
  bins) and `test_remap_multi_bin_shift_keeps_the_mode`.
- **"Unchanged particles never move" (was only half true, now true).** It held next to a
  shrinking neighbour only. Now unchanged bins map onto themselves exactly in both
  directions (tests above); docs and CLAUDE.md say so with the tests that show it.
- **Not smooth at 160 bins (real, fixed; part of it was physics).** Cause: number lagged
  mass in the flux-form step, bin means crept to their edges (0.93–0.999 of the width),
  and `keep_means_in_bins` + MNFIX turned that into alternating bins; switching
  `keep_means_in_bins` off did not help (the verifier's diagnosis is right). With the new
  remap (scenarios of the verifier's `evap_lag.py`, final dN/dlog10Dp extrema at 40 / 80 /
  160 bins, before → now): E2 (+30 K after 6 h growth) 1/1/7 → 1/1/1; E3 (bimodal
  evaporation) 2/2/5 → 2/2/2 (two modes); B2 (5 µg m⁻³ c* = 1 vapour on pure sulfate,
  6 h) 1/3/11 → 3/3/3. B2's 3 are physical: moving sections (no bin transfer, 320
  sections) form the same second mode at 45 nm (Kelvin ripening sends the smallest
  particles back to their cores) and PPM overlays it at all three resolutions; B1 (same
  seed, 0.5 µg m⁻³ h⁻¹ source) likewise. The 40/80/160-bin results match the moving
  sections: +30 K case number-mean 59.0/59.9/59.5 nm vs 59.2 nm (flux-form 67.6/66.4,
  MNFIX 39.7/59.3), +20 K case 115.3/120.4/120.1 vs 120.2 nm (flux-form 138.5/138.9). New
  test `test_soa_matches_moving_sections` (round trip and bootstrap/ripening at 80 bins,
  ≤ 0.3 % in mean diameter, ≤ 0.5 % in GSD) and the +30 K test at 160 bins.
- **Dirty worktree (not ours).** The 21 modified result files (S1/S2 npz, run_info,
  summary, one PNG) differed from HEAD only in runtimes and `code_version`; they were
  written by a parallel verifier. They were reset to HEAD, then S1 and S2 were
  regenerated with the final code (below), so no unreviewed file is committed.
- **Provenance (fixed).** Regenerated S1/S2 `run_info.json` carry `code_version`
  749346461e5f72e2 = HEAD's tomas_jax. Run with a scratch `PYCHAM_BENCH` that links the
  PyCHAM outputs read-only; the PyCHAM model_var records were kept as committed.
- **S2 medians (restored).** `analyse.dry_size_stats` reports number and dry-mass medians
  again, from a mid-rank cumulative over bins with particles (resolution-stable; the old
  plain cumulative read half a bin low and PyCHAM's empty bins inflated its value).
- **S1 wording (fixed).** "Size distribution within 2 %" replaced by what holds: +2.3 %
  mean / −3.2 % GSD at 40 and 160 bins; with Kelvin and transfer matched the models
  agree within 0.2 % / 0.4 % at 40, 80 and 160 bins.
- **A5 nitrate at 40 bins (still fails as written; decision needed).** See measurements.

### Changes
- `physics/ppm_remap.py`: `remap_mass_change` → Lagrangian remap. Displacement linear
  inside each bin (same upwind/limited slope as before, `bin_edge_shifts`); each bin's
  PPM profile is carried onto its image and deposited by exact overlap integrals (number:
  polynomial; mass weighted by the new particle mass: φ_j closed forms with a Taylor
  branch, `_phi`); MC-limited, sum-zero, positivity-limited composition slopes
  (`composition_slopes`, `slopes=True`); images kept between 0.25 and 4 bin widths so the
  work is a (nbins, 6) band (scatter-add); no-op calls skipped with `lax.cond`.
  `keep_means_in_bins`, `cap_top_bin` unchanged. Tried and dropped: an exponential tilt of
  each profile to match the bin mean (pile-ups at steep fronts, 3 extrema in the nitrate
  test).
- Tests (`tests/test_aaar_soa_ppm.py`, 28 → 38): see above; the old
  unchanged-particles test is replaced by the five pass-through cases.
- PyCHAM harness: S2 medians (mid-rank), S1/S2 mismatch texts describe the new remap;
  S1/S2 TOMAS-JAX results and figures regenerated (PyCHAM output reused).
- Docs: aerosol_step.md (algorithm, dropped alternatives, moving-section comparison,
  measurements, limitations, verified behaviour), soa_partitioning.md, pycham_comparison.md
  and the S1/S2 READMEs (new numbers), ASSUMPTIONS A6, CLAUDE.md.

### Measurements (laptop CPU; other users' jobs kept the load at 4–5)
- A1: per remap call in the presets (1440–2880 calls each, L1soa 80, L1 160, L2 80, L3 40,
  L3 80): number ≤ 1.4e-15, every species ≤ 1.2e-15, min N/M ≥ 0, all finite. Per outer
  step (40/80/160 bins, all presets): gas + particle per SOA class ≤ 3.7e-15, nitrate vs
  `nitrate_added` ≤ 1.3e-15, other species ≤ 3.6e-14, sulfate ≤ 6.5e-11 (MNFIX floor, as
  with 'mnfix'), number in SOA-only runs ≤ 4.4e-13, whole-run class budget (Level 1) ≤
  7.8e-15, tracker vs Mk ≤ 5.7e-16, no populated bin mean outside its bin. Remap tests:
  number and every species ≤ 1e-12 / 1e-13.
- A2 final-extrema count, MNFIX → PPM: Level 1 SOA only 3/13/30 → 1/1/1 (40/80/160);
  Level 1 preset 3/9/28 → 1/1/1; Level 2 3/5/18 → 1/1/1; Level 3 3/7/17 → 1/1/1 (and 1 at
  320 bins); PPM is 1 at every output time of every preset run.
- A3: c* = 10 evaporation test passes (mode back to the seed, 1 extremum per step);
  +30 K after growth: 1 extremum at every step at 40/80/160 bins, class budget ≤ 3.4e-16
  and number ≤ 1.4e-15 per step, min Mk = 0; +25 K: 1 at every step; +20 K: 1 at 80 and
  160 bins, 2–7 at 40 bins (a physical shoulder the moving sections show at 45–70 nm,
  which mass-doubling bins resolve as a plateau). MNFIX +30 K: 13/17/22.
- A4: `test_zero_oa_bootstrap_both_transfers` passes with both settings; bootstrap +
  ripening matches moving sections (B2 for 4 h at 80 bins: 139.07 vs 138.96 nm, GSD
  1.505 vs 1.507).
- A5 vs the old code (= 'mnfix'): SOA within 0.21 % in every preset at 40/80/160 bins
  (L1 +0.20/+0.07/+0.01 %, L2 +0.05/+0.02 %, L3 −0.06/−0.01 % at 40/80). Level 3 nitrate
  5.160/5.152/5.149/5.149 µg m⁻³ at 40/80/160/320 bins vs old 4.962/5.075/5.139/5.149:
  +4.0 % at 40 bins (fails the 2 % limit), +1.5 % at 80, +0.20 % at 160. The new 40-bin
  value is 0.21 % from the 320-bin one, the old one −3.6 %.
- A6 warm, median of 3, interleaved, MNFIX → PPM: 80 bins L1 6.41 → 4.48 s, L2 8.54 →
  7.56 s, L3 23.27 → 19.75 s; 40 bins 1.77 → 1.15, 6.15 → 5.74, 13.50 → 12.42 s. One
  remap call 0.05/0.09/0.14 ms at 40/80/160 bins. `test_cstar_change_does_not_recompile`
  passes.
- A7 `pytest tests -q -x`: 622 passed, 650 skipped in 183 s (before: 612 passed, 650
  skipped; +10 net new tests).
- A8 `AerosolStepConfig.growth_bin_transfer` = 'ppm' (default) | 'mnfix'; both tested.
- S1 (c* = 1, 12 h) number-mean dry diameter 123.3/123.2/123.2 nm at 40/80/160 bins vs
  PyCHAM 120.5 (+2.3 %), GSD 1.817/1.816/1.815 vs 1.878 (−3.2 %); with Kelvin off and
  PyCHAM's transfer coefficient 138.2/137.8/137.7 vs 138.1 nm, GSD 1.423/1.429/1.430 vs
  1.428. S2 at 6 h: number-mean 212.3/211.6/211.4 nm vs 210.1, number median
  202.6/202.7/202.4 vs 200.9, GSD 1.370/1.374/1.375 vs 1.411; SOA +1.1 % at 6 h.

### Known issues / limitations
- A5 nitrate at 40 bins: +4.0 % against the old code (limit 2 %). The new value is the
  converged one; the user has to decide whether A5 compares against the old code or a
  converged reference.
- Displacement and composition are linear inside each bin: structure narrower than about
  two bins is not kept (40-bin plateau in the +20 K case). Bins holding two kinds of
  particle move with the bin's mean change.
- An image wider than 4 bin widths is squeezed (not reached in any test).
- S1 GSD differs by −3.2 % from PyCHAM (was −1.7 % with the first PPM version at 40 bins):
  the first version was 1.3 % broader at 40 than at 160 bins; the new one does not move
  with resolution, and with the physics matched the GSDs agree within 0.4 %. The
  remaining gap is PyCHAM's stronger Kelvin term.

### Next steps
- Decide the A5 nitrate reference.
- Consider the vbs-jax transfer-coefficient and Kelvin choices (A6, S1/S2), which now
  carry all of the size-distribution difference to PyCHAM.

---

## 2026-10-15 (Thu) — PPM bin transfer: verifier fixes (upwind edges, top bin), A1–A8 re-measured, S1/S2 regenerated

**Time**: ~23:30 PDT 2026-10-14 (06:30 UTC 2026-10-15)
**Branch**: `feat/aaar-tutorial-ppm-soa`

### Summary
Independent verifiers of the PPM bin transfer (previous entry) reported two code defects
and stale benchmark text. Both defects were reproduced and fixed with tests; the S1/S2
PyCHAM comparison was regenerated with the fixed code and its text updated.

### Changes
- `physics/ppm_remap.py`: the edge displacement is now **upwind** (`edge_shifts`): each
  bin's log shift is reconstructed linearly with the smaller one-sided slope, clipped
  between neighbour values, and an edge is crossed with the donor bin's own value.
  Before, the edge value was interpolated between the two neighbouring bin means, which
  dragged unchanged particles along with a shrinking neighbour (80 bins, SOA-only bins
  30–39 losing 99 %, unchanged sulfate bins 40–49: bin 40 went from 1e12 to 1.1e9
  #/cell, 10 % of the unchanged particles and 1.3 % of the sulfate ended below bin 40;
  now exactly 0). For growth with s falling with size the upward edge value is
  unchanged, so the preset results barely move (SOA ≤ 0.001 %, nitrate ≤ 0.035 %,
  mean diameter ≤ 0.9 %).
- `ppm_remap.cap_top_bin` (new): no flux leaves the grid, so particles growing in the top
  bin stay there; when its mean passes 17.5 µm, the in-place species are taken back
  instead of letting MNFIX trim every species of the bin. `soa_partition_step('ppm')`
  returns that SOA to the vapour and also returns any class mass the post-remap MNFIX
  deletes (empty-bin floor), as vbs-jax does after the 'mnfix' hook. Before, the SOA
  MNFIX trimmed was deleted: −1.1e-4 to −1.2e-4 of the class total per 600 s step
  (2000 µg/m³ of c* = 0.01 vapour onto 15 µm particles); now ≤ 2.2e-16. The nitrate
  remap in `aerosol_outer_step` uses the same cap; nitrate that does not fit is counted
  out of `nitrate_added` and the driver warns (new wording).
- Tests (`tests/test_aaar_soa_ppm.py`, +7): unchanged particles stay in place; cap unit
  test; per-class budget with particles at the grid top (both settings); nitrate at the
  grid top; temperature-rise evaporation (both settings).
- PyCHAM harness: S1/S2 mismatch text describes the PPM remap; S2 size metric is now
  number-mean dry diameter, GSD, dry-mass-weighted mean and the fraction below the
  seed's 1st-percentile size (the fixed-bin median was biased low by about half a bin;
  the verifier measured 184/193/198 nm at 40/80/160 bins with the first PPM version while
  the mean was flat); S1 reports the GSD
  difference. S1/S2 TOMAS-JAX results and figures regenerated (PyCHAM output reused).
- Docs: aerosol_step.md (algorithm, measurements, limitations), soa_partitioning.md,
  heterogeneous.md, pycham_comparison.md (S1/S2 tables and text), S1/S2 READMEs,
  ASSUMPTIONS A6, CLAUDE.md.

### Measurements (laptop CPU)
- A1 conservation (preset runs, checked after every outer step, 40/80/160 bins): gas +
  particle per SOA class ≤ 4.1e-15, particle nitrate vs `nitrate_added` ≤ 1.6e-15,
  nucleated organic ≤ 6e-14, sulfate ≤ 7e-11 (MNFIX floor, same with 'mnfix'), number in
  SOA-only runs ≤ 4.4e-13; min Mk = 0, no NaN/inf, no populated bin mean outside its bin.
  Remap alone (tests): number and every species ≤ 1e-12. Top-bin case: class budget
  ≤ 2.2e-16 per step (was −1.2e-4).
- A2 interior extrema of the final dN/dlog10Dp, MNFIX → PPM: Level 1 SOA only 3/13/30 → 1/1/1
  (40/80/160 bins); Level 1 preset 3/9/28 → 1/1/1; Level 2 3/5/18 → 1/1/1; Level 3
  3/7/17 → 1/1/1; PPM is 1 at every output time (MNFIX up to 36).
- A3 evaporation: c* = 10 SOA, no vapour (80 bins): mode returns to the seed size, 1
  extremum every step. +30 K after 6 h of growth (c* = 1): class budget ≤ 5.6e-16 and
  number ≤ 4e-16 per step, mean diameter falls monotonically, 1 extremum at 40 and 80
  bins (5 small ones at 160 bins; MNFIX 13/17/22). No negative mass anywhere.
- A4 zero-OA bootstrap: `test_zero_oa_bootstrap_both_transfers` passes with both settings
  (c* = 0.01 > 50 % condensed, c* = 1000 < 1 %, no spike).
- A5 end-of-run totals vs the old code (= 'mnfix', bit-identical: L1 40/80, L1 SOA-only
  80, L2 40, L3 40/80): SOA within 0.22 % in all presets at 40, 80 and 160 bins. Level 3
  nitrate +4.39 % at 40 bins (5.180 vs 4.962 µg/m³), +1.75 % at 80, +0.33 % at 160,
  +0.07 % at 320 (5.153 vs 5.149). The 2 % criterion fails at 40 bins only; the old
  40-bin value is 3.6 % below the converged ~5.15, the new one 0.6 % above.
- A6 runtime (warm, median of 3, interleaved, load 3–4), MNFIX → PPM: 40 bins L1 1.75 →
  1.09 s, L2 6.05 → 5.75 s, L3 13.46 → 12.26 s; 80 bins L1 6.17 → 4.26 s, L2 8.35 →
  7.51 s, L3 22.77 → 18.74 s. `test_cstar_change_does_not_recompile` passes.
- A7 `pytest tests -q -x`: 612 passed, 650 skipped in 169 s (before: 605 passed, 650 skipped;
  +7 new tests).
- A8 `AerosolStepConfig.growth_bin_transfer` = 'ppm' (default) | 'mnfix'; both settings are
  tested (growth, evaporation, temperature rise, top bin, bootstrap, Pankow, class budget).
- S1/S2 vs PyCHAM (TOMAS-JAX side regenerated, PyCHAM output reused, load ~4 with nothing
  else running; TOMAS-JAX run times within 4 % of the committed ones for S2): S1 c* = 1 at
  12 h number-mean dry diameter 122.3/122.7/123.0 nm at 40/80/160 bins vs PyCHAM 120.5
  (+1.5 %), GSD −1.7 %, SOA +0.65 %, max +4.1 % at 1–3 h, gas −2.9 %. S2 at 6 h:
  number-mean 213.8 vs 210.1 nm (+1.8 %; 212.6/212.0 at 80/160 bins), GSD −3.5 %, 0.16 %
  of the particles below the seed's 1st-percentile size (0.06/0.03 % at 80/160; PyCHAM
  0.05 %; MNFIX 2.1 %); SOA +1.1 % at 6 h, +13 % at 30 min; with coupling, transfer and
  Raoult matched within 1.5 % from 10 min on, with Kelvin off as well −0.01 to −0.75 %.

### Known issues / limitations
- A5 nitrate: Level 3 at 40 bins is +4.4 % against the old code (limit 2 %). The new
  value is the converged one (see A5 above); a decision is needed: accept on the
  convergence evidence, or compare against a 160/320-bin reference instead of the old
  40-bin value.
- Grow-then-evaporate round trips do not return to the seed shape: one composition per
  bin, and species move with the donor bin's mean composition (as in H2SO4 PPM). After
  6 h of growth and an instant +30 K: number-mean 71.4/70.2 nm at 40/160 bins vs ~63 nm
  for kept cores (MNFIX 42.0/62.7). Not seen in the presets (1 extremum, ≤ 1 % change
  40 → 160 bins), but resolution does not cure it.
- One remap per call: particles moving several bins in one 60 s step take the
  displacement of the bins they pass through (c* = 100 pure-SOA mode next to a sulfate
  mode: 0.40 % of the sulfate below bin 40 vs 0.17 % for an exact internal-mixture
  remap; MNFIX 0.21 %). Largest populated-bin displacement per call in the presets: 0.03
  / 1.6 / 1.9 bins (L1/L2/L3 at 160 bins).
- Bottom of the grid: a bin-0 mean below 1.7 nm is handled by MNFIX (number trimmed, mass
  kept), as before.

### Next steps
- Decide on the A5 nitrate criterion (above).
- If round trips matter for the tutorial (day/night SOA), transport composition with a
  higher-order scheme too (per-species PPM instead of the donor's mean composition), or
  remap per vbs-jax sub-step for large one-step changes.

---

## 2026-10-15 (Thu) — SOA and nitrate move between bins by PPM remap instead of MNFIX

**Time**: ~20:59 PDT 2026-10-14 (03:59 UTC 2026-10-15)
**Branch**: `feat/aaar-tutorial-ppm-soa` (from `feat/aaar-tutorial`)

### Summary
SOA partitioning (vbs-jax) and the N2O5-uptake nitrate source change particle mass in
place. Until now MNFIX then moved the particles between the fixed bins (after every
vbs-jax sub-step), and its two-packet split combed a growing mode: the final
dN/dlog10Dp of Level 1 with SOA only had 3 / 13 / 30 interior extrema at 40 / 80 / 160
bins and the number-mean diameter depended on resolution (S1 vs PyCHAM: −12.6 % at 40
bins). They now move with a flux-form PPM remap in log-mass built from the H2SO4
condensation pieces. The partitioning physics is unchanged.

### Changes
- `tomas_jax/physics/ppm_remap.py` (new): `remap_mass_change(Nk, Mk_old, Mk_new, xk)`.
  Bin displacement = log change of each bin's mean dry mass, interpolated in ln m to the
  edges; number by PPM (`ppm_reconstruct`), every species with the PPM mass-weighted
  departing fraction of the donor (one consistent flux, conservative, never negative);
  sub-steps at Courant ≤ 0.8 with growth and evaporation regions kept apart;
  `keep_means_in_bins` hands the fewest possible particles to a neighbour when a bin
  mean still crosses an edge. `BIN_TRANSFER_PPM` / `BIN_TRANSFER_MNFIX`.
- `physics/soa_partitioning.py`: static `bin_transfer` ('ppm' default, 'mnfix' = old
  hook). With 'ppm' the vbs-jax `redistribute` hook only writes the class masses; the
  remap runs once per call, then MNFIX restores its floor in emptied bins.
- `box/aerosol_step.py`: the nitrate source is remapped the same way;
  `box/config.py`: `AerosolStepConfig.growth_bin_transfer` ('ppm' | 'mnfix', validated).
- `box/driver.py`: `run(..., aerosol_cfg=None)`; non-default fields reported in
  `diagnostics["aerosol_cfg"]` (empty dict for default runs).
- Tests: `tests/test_aaar_soa_ppm.py` (new, 21 tests: remap conservation per species at
  40/80/160 bins, mode shift, identity, limiter, growth and evaporation with both
  settings, zero-OA bootstrap, nitrate, settings, driver); `tests/test_aaar_aerosol.py`
  runs Pankow, vbs-jax equilibrium and the all-process class budget with both settings.
- Docs: `docs/aaar/aerosol_step.md` (new "Bin transfer" section with algorithm,
  measurements, limitations), `soa_partitioning.md`, `heterogeneous.md`, `coupling.md`,
  dated notes in `pycham_comparison.md`, ASSUMPTIONS A6, CLAUDE.md, README.md.

### Measurements (laptop CPU)
- Interior extrema of the final dN/dlog10Dp (> 5 % of peak), MNFIX → PPM: Level 1 SOA
  only 3/13/30 → 1/1/1 (40/80/160 bins); Level 1 preset 3/9/28 → 1/1/1; Level 2 3/5 → 1/1
  (40/80); Level 3 3/7/17 → 1/1/1 (40/80/160). PPM: 1 at every output time.
- Conservation of the remap per call inside the preset runs: number ≤ 1e-15, every
  species ≤ 1.3e-15 relative; no negative or non-finite values (1440–2880 calls/run).
- S1 (TOMAS-JAX side re-run, c* = 1, 12 h): number-mean dry diameter 122.4 / 122.7 /
  123.0 nm at 40/80/160 bins vs PyCHAM 120.5 (old 105.3 / 115.1 / 121.3). With Kelvin
  off and PyCHAM's transfer coefficient: 138.9 vs 138.1 nm. S2 (6 h, 40 bins):
  number-median dry diameter 184 vs PyCHAM 181 (old 156).
- Evaporation (c* = 10 SOA, no vapour, 80 bins): the mode shrinks back to the seed
  size, one extremum at every step, class budget 1e-14 (MNFIX: 6–13 extrema).
- Totals vs the MNFIX scheme: SOA within 0.21 % in all presets at 40 and 80 bins.
  Level 3 nitrate +4.4 % (40 bins) and +1.7 % (80 bins): the corrected distribution has
  15–23 % more surface for N2O5 uptake at 40 bins; the old scheme's nitrate drifted with
  resolution (4.96/5.08/5.14 µg/m³), the new one does not (5.18/5.16/5.16).
- Runtime (warm, 80 bins): Level 1 6.14 → 4.28 s, Level 2 8.41 → 7.50 s, Level 3
  22.4 → 18.8 s (40 bins: 1.75 → 1.20, 5.91 → 5.64, 13.3 → 12.3 s).
- `pytest tests`: 605 passed, 650 skipped (baseline 580 passed, 650 skipped).
- `growth_bin_transfer='mnfix'` reproduces the previous results exactly (Level 1 and
  Level 3 at 80 bins, S2 medians).

### Known issues / limitations
- Level 3 nitrate at 40 bins differs from the old code by 4.4 % (> 2 %); the old value
  was the resolution-dependent one (see above).
- PPM reconstructs number only; `keep_means_in_bins` acts where a bin mean crosses an
  edge (mostly sparse tails), moving a few percent of that bin.
- No mass leaves the grid in the remap; beyond the top/bottom bin MNFIX trims as before.
- The figures and tables of `docs/aaar/pycham_comparison.md` (S1/S2) still show the
  MNFIX scheme; only the TOMAS-JAX side was re-run (into a scratch directory).

### Next steps
- Re-run `benchmarks/pycham/run_all.py` to regenerate the S1/S2 figures and tables.
- Check whether H2SO4 condensation (`ezcond_ppm_jax` adds the condensed mass to the
  source bins after the PPM transport) would gain from the same mass-with-particles
  flux; it is smooth today (1 extremum at 40/80/160 bins).

---

## 2026-10-14 (Wed) — PyCHAM comparison: cases G3, S1, S2, audit fixes, run_all, write-up

**Time**: ~23:30 PDT 2026-10-13 (06:30 UTC 2026-10-14)
**Branch**: `feat/aaar-tutorial`

### Summary
The TOMAS-JAX vs PyCHAM benchmark now covers four cases: G2 (gas, Level 2), G3 (gas,
Level 3 day/night), S1 (Level 1 surrogate condensation/evaporation) and S2 (Level 2
limonene SOA). An audit of the case results was addressed, the harness was
consolidated, one command regenerates everything, and the write-up for scientists is
`docs/aaar/pycham_comparison.md`. Nothing in `tomas_jax/` was changed in this step and
nothing was tuned to PyCHAM.

### Changes
- `benchmarks/pycham/run_all.py` (new): runs every case's `run.py` and `analyse.py` in
  turn, each in its own process group with a wall-clock limit (`config.RUN_ALL_TIMEOUTS_S`),
  logs and a JSON report in `pycham-bench/logs/run_all/`, non-zero exit on any failure.
- `config.py`: the G3/S1/S2 settings moved in (single source of truth); semantic colour
  keys (one colour = one meaning in every figure, validated per figure with the dataviz
  palette checker); one display density for dry diameters (`RHO_REPORT`, 1400 kg m⁻³).
- `particles.py` (new): TOMAS-JAX water per unit mass of sulfate/organics, the matching
  PyCHAM water dissociation constant, the effective dry density, display diameters.
- `load_pycham.py`: per-bin number is now `N_bin` (the wet file, which lines up with the
  composition); the dry re-binned file is `N_dry_rebinned`; `mv_cm3_mol` exposes MV.npy;
  the t = 0 dry radius is documented.
- `pycham_inputs.py`: `hold_constant` → PyCHAM `const_comp`; `ensure_pycham_root(case,
  kelvin_off=True)` makes the surface-tension-0 attribution copy (was S1-local).
- S2: PyCHAM water constants per component (SA 0.347, organics 0.289 kg/kg) instead of
  one seed-average ratio, which had made SOA too hygroscopic and hid PyCHAM's Kelvin
  term on water; seed sulfate identity (H2SO4 vs SO4) recorded; Kelvin-off attribution
  pair (PyCHAM `massfrac_nokelvin`, TOMAS-JAX σ = 0); PyCHAM volatility sensitivity with
  its own vapour pressures (`native`); generated mismatch list with measured effects;
  gas species below 1e-7 ppb in every run are not stored (results 2.4 MB).
- S1: mismatch list generated from `pycham_setup()` and the measured results (the
  density entry now states the 2.170 g/cm³ effective dry density actually used; water
  reported as totals and at the mode); dry diameters at 1400 kg m⁻³.
- G3: the held species (NO2) is excluded from the all-species statistics (361 species).
- READMEs: shared harness README rewritten (run_all, layout, conventions); per-case
  READMEs for g2 (moved), G3, S1 (new), S2 (rewritten).
- `docs/aaar/pycham_comparison.md` (new), linked from README.md; ASSUMPTIONS A6, A8,
  A9, A12, A14 and question 6 updated with the measured caveats.

### Findings
- Gas chemistry: same ODE system (rate constants to 4e-16; median |NMB| 5e-7 with
  matched numerics). Preset couplings cost 1.7e-3 (Level 2, 300 s) and 0.10 (Level 3,
  900 s) median |NMB|; at 60 s 3e-4 and 6e-3.
- Level 3: the NO2 hold is a reset per interval (8.4 % less NOx than a continuous hold,
  HNO3 −9 %); J step at sunset/sunrise and RO2 = 0 in the first 900 s add to it.
- S1: SOA within 0.4 % at 12 h (max 3.7 % at 1–3 h); vbs-jax transfer coefficient up
  to 1.5× PyCHAM's and above the kinetic limit; 40-bin MNFIX spreading (mean dry
  diameter −12.6 %, converges at 160 bins).
- S2: total SOA +12 % at 30 min, +3.9 % at 1 h, +1.0 % at 6 h. With coupling, transfer
  and Raoult matched: within 0.5 % from 10 min on; with Kelvin off as well: 0.5–2.2 %
  below PyCHAM (offsetting terms). The water fix changed PyCHAM SOA by ≤ 0.3 %. Per-species tracker
  off by up to 4× within a class. The NOx-sink feedback (NO2 −79 %) is reproduced by
  PyCHAM with the same c* and disappears with PyCHAM's own vapour pressures (SOA 0.51×).
- `driver.run` never passes `AerosolStepConfig.micro_dt` to `aerosol_outer_step`.

### Known limitations
- Coagulation, nucleation, N2O5 uptake and the Level 3 aerosol are not tested by any
  case (switched off). S2 does not test the volatility estimate (identical c* given).

### Next steps
- Decisions for Ali (ASSUMPTIONS question 6): NO2 hold inside the solver / shorter
  Level 3 coupling; vbs-jax transfer coefficient; reporting SOA by class.
- A coagulation-only seed case (PyCHAM `coag_on = 1`).

## 2026-10-13 (Tue) — PyCHAM benchmark harness + case G2 (gas phase, Level 2)

**Time**: ~19:30 PDT
**Branch**: `feat/aaar-tutorial`

### Summary
Independent check of the AAAR box model against PyCHAM v5.8.0 (pinned, headless) with
the same limonene mechanism and identical prescribed J. Case G2 is gas only, Level 2
conditions, 6 h. Differences are reported and attributed. Nothing is tuned.

### Changes
- `box/driver.py`: opt-in `run(..., observer=fn)` hook. It hands every gas species, the
  aerosol state, the J used and the SA/NA_HET counters to the caller after each
  coupling interval. Defaults and RunResult are unchanged. Test:
  `test_observer_is_opt_in_and_sees_every_interval`.
- `benchmarks/pycham/` (new): config, `ours.py`, `jvalues.py` (our J → PyCHAM
  J.xlsx), `pycham_inputs.py`, `metrics.py`, `plotting.py`, `check_rates_pycham.py`,
  copies of `run_pycham.py`/`load_pycham.py`/patches/lock, README with the install
  recipe and the per-case list of model differences. Case `cases/g2/`
  (`run.py`, `analyse.py`, results < 2 MB, 6 figures → `docs/aaar/figures/pycham/`).
- `.gitignore` negations in `benchmarks/pycham/` and `docs/aaar/` so the small
  results and figures can be committed (the root ignores figures/, *.png, *.npz, *.csv).

### Findings (G2)
- All 2135 rate coefficients match PyCHAM's generated `rate_coeffs.py` to 4e-16
  (same T, M, H2O, RO2, J), and the RO2 pools are identical (146).
- 391 species above 1e-5 ppb. At the preset 300 s coupling the median |NMB| is
  1.7e-3 and the max |NMB| is 3 % (LIMAOH). 87 % of species stay within 10 % at all
  times; the rest deviate only in the first 20 min, because RO2 is frozen at its
  t = 0 value (0) for the first 300 s, so RO2+RO2 alcohols are 0 at 5 min. At 60 s
  coupling the median |NMB| is 3e-4 and the max deviation 1.5 %.
- The 60 s residual is PyCHAM's own BDF error at its default rtol 1e-4 (PyCHAM vs
  PyCHAM at rtol 1e-6 reproduces it). With tight tolerances in both and PyCHAM's
  water vapour (Hyland–Wexler, +2.2e-4 vs our Buck) the models agree to a median
  |NMB| of 5e-7.

### Next steps
- Cases with particles (SOA partitioning, seed, N2O5 uptake) and the Level 3 diurnal
  case. These need J files written for the coupling interval of each compared run.

## 2026-10-13 (Tue) — AAAR tutorial: MCM gas chemistry, TUV-x photolysis, coupled driver

**Time**: ~17:30 PDT
**Branch**: `feat/aaar-tutorial` (merges `feat/aaar-tutorial-gaschem` and `-aerosol`)

### Summary
Three-level teaching box model for the AAAR photochemical box-modelling tutorial
(S. Jathar): (1) volatility of one surrogate organic, (2) MCM limonene chemistry →
SOA by c*-class absorptive partitioning, (3) TUV-x diurnal photolysis + N2O5 uptake
→ particle nitrate. Entry point `tomas_jax.box.driver.run(Scenario) -> RunResult`.

### Changes
- `gaschem/` (new): KPP/MCM parser, MCM v3.3.1 complex rates and J(χ) (from KPP
  `constants_mcm.f90`; 31/33 identical to F0AM), exact k decomposition with frozen
  rates per interval, gather/segment_sum RHS, diffrax Kvaerno3 solver, TUV-x J
  provider (J(SZA) table, MCM branching/parameterization fallbacks), Li et al. (2016)
  c* + 15 volatility classes. Mechanism file byte-identical to the one supplied.
- `box/`: `scenario.py` (contract + TOML presets), `result.py`, `coupling.py`
  (units, seed, tutorial mechanism edits, class gather/scatter + particle tracker),
  `driver.py` (`run`, `warmup`, `code_version`).
- Tests: `tests/test_gaschem.py` (16), `tests/test_aaar_driver.py` (9).
- Docs: `docs/aaar/gaschem.md`, `docs/aaar/coupling.md`; `ASSUMPTIONS.md` A1–A4,
  A7, A8, A10–A14 + questions for Shantanu.
- Dependencies: tuvx-jax `feat/aaar-tutorial` (all photolysis recipes the mechanism
  needs ported and validated against the Fortran TUV-x; extra `[photolysis]`).

### Findings
- Solver: Kvaerno5/atol 1 took 53 s per 24 h chemistry-only; Kvaerno3, rtol 1e-3,
  atol 1e4 molec/cm³ and 900 s coupling take ~9 s (O3 within 0.6 %).
- TUV-x scales the radiation field by r, not 1/r² (Fortran `apply_scale_factor`);
  reproduced, logged as A11.
- Li et al. (2016) gives PAN-type nitrates very low c*; they dominate "top
  contributors" until a curated c* table replaces the placeholder (A8).
- Level 3 needs a NOx source for nighttime N2O5; the preset holds NO2 at 1 ppb (A14).
- Runtime (warm): Level 1 ~1.7 s, Level 2 ~6 s, Level 3 ~14 s.

### Next steps
- Shantanu's c* table, confirmation of A1–A4/A10/A14.
- tomas-api tutorial endpoints + workers, tomas-web guided levels, Render/Vercel.

---

## 2026-10-13 (Tue) — AAAR tutorial aerosol step: SOA partitioning, N2O5 uptake, outer step

**Time**: ~00:30 PDT
**Branch**: `feat/aaar-tutorial-aerosol` (stream C of the AAAR tutorial build)

### Summary
This is the aerosol side shared by the three AAAR photochemical box-model
tutorial levels. The levels are (1) H2SO4 plus one surrogate organic
vapour of user-chosen c\*, (2) MCM limonene products lumped into c\*
classes, and (3) Level 2 plus nitrate from N2O5 uptake. SOA partitioning
uses the vbs-jax kinetic solver (pinned dependency) through a TOMAS
adapter. The coupling driver (another stream) calls one jitted
`aerosol_outer_step`. Defaults of all existing code paths are unchanged.

### Changes
- `core/config.py`: organic slot carve. `SRTSOA1=1`, `N_SOA_CLASSES=15`,
  `SRTNIT=40` (nitrate as NO3), `SRTORG_NUC=41` (== `SRTORGLAST`), with
  disjointness asserts. `MW_N2O5`, `MW_NO3`, and `SV_N2O5=38.78` (Fuller
  1966 atomic volumes, as for `SV_H2SO4`).
- `physics/condensation_sink.py`: `calc_wet_diameters` extracted from
  `calc_condensation_sink` (results unchanged) so SOA geometry matches the
  H2SO4 sink.
- `physics/soa_partitioning.py` (new): `soa_partition_step` adapter over
  `vbs_jax.vbs_partition_step`. Per-cell ↔ per-m³; MNFIX redistribute
  hook; TOMAS geometry hook; per-cell (Nk, Mk) carried as aux. Absorbing
  floor of 1e-4 × non-SOA dry mass in the Raoult denominator.
- `physics/heterogeneous.py` (new): `n2o5_uptake` gives k_het and per-bin
  weights (condensation sink with α = γ).
- `box/` (new): `aerosol_step.py` (`aerosol_outer_step`, `AerosolState`,
  `AerosolForcing`, `SOAParams`, `AerosolDiag`) and `config.py`
  (`AerosolStepConfig`). SOA class arrays (c\*, ΔHvap in kJ/mol, MW)
  come from the caller (`gaschem/volatility.py`). The aerosol side has no
  c\*/ΔHvap rule.
- `tests/test_aaar_aerosol.py` (15 tests, ~13 s).
- Docs: `docs/aaar/{aerosol_step,soa_partitioning,heterogeneous}.md`,
  `ASSUMPTIONS.md` (A5, A6, A9, AC1, AC2), `docs/references.md`,
  `CLAUDE.md`, `README.md`.

### Verification
- Full suite: 496 passed (481 before + 15 new).
- Single c\*-class equilibrium within 0.35% of Pankow from either side.
  Six Donahue classes within 0.5% per class of `vbs_jax.vbs_equilibrium`.
  The residual is the Kelvin term at 1.5 µm.
- Per-class gas + particle SOA conserved to 1e-10 over a 600 s step with
  all processes and sources on. 12 h with no sources conserves dry mass to
  ~5e-12, and N(t) is identical to a coagulation-only run.
- Free-molecular N2O5 limit within 0.04% of γ c̄ A/4. γ = 0 gives k_het = 0
  and finite weights.
- Changing c\* does not recompile (jit cache size unchanged). A Level-1-like
  12 h run takes ~2–3 s after a ~1 s compile.

### Known issues / limitations
- Without the absorbing floor, zero-OA bins have psat = 0 in vbs-jax and
  even c\* = 1000 vapour condenses, then re-evaporates in a sawtooth. The
  floor delays the start of supersaturated condensation by a few minutes
  (ASSUMPTIONS.md AC1).
- Nitrate is irreversible and non-volatile, with no NH4NO3 thermodynamics.
  It takes the bisulfate water ratio and sulfate-like density, as all slots
  1–41 do. SOA is too hygroscopic for the same reason (A5, A9).
- The nitrate weights are computed once per outer step by the driver.

### Next steps
- Driver stream: call `aerosol_outer_step` from `box/driver.py`. Map gas
  products to `org_rate`/class vapour, N2O5 to `nitrate_rate` with
  `n2o5_uptake`, and compute H2SO4 production.
- Confirm A5/A6/A9/AC1/AC2 with Shantanu.

---

## 2026-10-06 (Tue) — Selectable water uptake (Tabazadeh) and water after every process

**Time**: ~14:00 PDT
**Branch**: `feature/water-uptake`

### Summary
tomas-web runs ignored RH in coag-only mode: water was only set at the end
of condensation, so coagulation in coag-only runs used dry sizes. Water
uptake was also always the TOMAS ammonium-bisulfate fit, which
underestimates water for pure sulfuric acid by 2–10×. Added a Tabazadeh (1997)
H2SO4/H2O scheme and an option to re-equilibrate water after every process,
as the TOMAS box model does. Defaults are unchanged.

### Changes
- `physics/water_equilibrium.py`: `tabazadeh_h2so4_wt` (JIT-safe, matches
  the numpy reference exactly in 185–260 K, smooth extrapolation outside),
  `calc_equilibrium_water_h2so4`, and the `equilibrate_water` dispatcher
  (`WATER_SCHEME_BISULFATE` / `WATER_SCHEME_H2SO4`). The Tabazadeh table now
  lives here; `radiative_forcing.py` imports it.
- `solvers/condensation.py`: `water_scheme` and `water_every_process` on
  `make_step`, the step functions and all scan loops. Both are static, like
  `use_tfl`.
- `tests/test_water_uptake_schemes.py` (44 tests); `docs/water_uptake.md`; `CLAUDE.md`.

### Verification
- Defaults are bit-identical to origin/dev for `run_condensation_scan`,
  `run_combined_scan_ppm`, `run_nucleation_condensation_scan`,
  `run_full_scan` and a 4-process `make_step`.
- Coag-only, 100 nm mode, 24 h, RH 90%: final N is 1.5676e4 (dry), 1.8655e4
  (bisulfate, every process) and 1.9451e4 (H2SO4, every process).

### Known limitations
- No deliquescence hysteresis; organics use the bisulfate ratio under the
  H2SO4 scheme; NH4 is ignored by the H2SO4 scheme.
- The bundled Fortran harness still only equilibrates water in
  condensation modes (by design; the option defaults off).

### Next steps
- tomas-api: water-uptake option (auto/H2SO4/bisulfate), initial water,
  `water_every_process=True`, flags in results. tomas-web: selector,
  flags, and an assumptions/limitations view.

---

## 2026-10-06 (Tue) — `estimate_nucleation_rate` alias deprecated

**Time**: ~10:30 PDT
**Branch**: `release/dev-to-main` (review on #17)

### Summary
Following review on #17, the old name from the #14 rename is no longer used
anywhere. `CLAUDE.md` and `docs/nucleation.md` now use
`ricco_dunne_nucleation_rate`. The `estimate_nucleation_rate` alias now
emits a `DeprecationWarning` (same pattern as `coag_rk4_step`) instead of
silently aliasing. It is kept because v0.3.0 on `main` exports it as the real
function name. No library or tomas-api code calls it.

### Changes
- `tomas_jax/physics/nucleation.py`: the alias is a warning shim.
- `tests/test_nucleation.py`: the alias test now asserts the warning and an identical result.

### Next steps
- Remove the alias in a later release once downstream users have moved.

---

## 2026-10-06 (Tue) — Coagulation on non-doubling bin grids (80/160 bins)

**Time**: ~09:30 PDT
**Branch**: `feat/coag-any-bin-ratio`

### Summary
Coag-only runs on tomas-web showed oscillating size distributions at 80 and
160 bins (fine at 40). Root cause: TFL (`multicoag.f`) assumes mass-doubling
bins. It sends self-coagulation products to bin k+1 and lets a collision move a
particle at most one bin, and its φ/η and `1/(2·xk)` terms take the bin width
to be x_k. The March generalization (`950d151`) only adjusted ζ and φ/η, so the
transfer to the next bin was undercounted by (p−1), and the one-bin-hop
structure was still wrong for p ≠ 2. Against the exact constant-kernel
solution, TFL error grew with resolution: 4.6% → 18.6% → 28.8% L1 at
40/80/160 bins.

Added a linear sub-bin two-moment scheme that is valid for any bin ratio, and
made `calc_coagulation_rates` dispatch to it whenever p ≠ 2. 40-bin results
are bit-identical to before.

### Changes
- `tomas_jax/physics/coagulation_rates_linear.py` (new): `calc_coagulation_rates_linear`
  and `linear_subbin_reconstruction`. Non-negative linear density in the larger
  bin of each pair (Simmel et al. 2002 LDM), shifted by the partner's mean mass
  and integrated exactly over at most two destination bins. Exact losses; mass
  conserved to round-off.
- `tomas_jax/physics/coagulation_rates.py`: TFL renamed `calc_coagulation_rates_tfl`
  (body unchanged). New `calc_coagulation_rates` dispatches with `lax.cond`
  (TFL iff `xk[1]/xk[0] == 2`). No call-site changes.
- `tests/test_coagulation_bin_ratio.py` (new, 55 tests): reconstruction validity,
  conservation, constant-kernel convergence, dispatch bit-identity, and a
  Brownian 80/160-bin regression (fails if TFL is forced).
- `benchmarks/python/coag_bin_ratio_convergence.py` (new): TFL vs linear at
  40/80/160/320 bins, exact constant-kernel + Brownian cases, 2 figures.
- `docs/coagulation_bin_ratio.md` (new), `CLAUDE.md` (dispatch rule, layout, testing).

### Results (coag-only, 24 h)
- Constant kernel, L1 vs exact. Linear: 0.47% / 0.13% / 0.06% / 0.05% at
  40/80/160/320 bins. TFL: 4.6% / 18.6% / 28.8% / 27.2%.
- Brownian, N=1e5, GMD 100 nm, GSD 1.6. Linear keeps a single mode at every
  resolution (N 1.6580e4 → 1.6553e4, peak 4.03e4 → 4.11e4). TFL had 1/3/23/37
  local extrema.
- 40-bin `coag_euler_step` over 24 h is bit-identical to the pre-change TFL.
- Full suite: 487 passed, 650 skipped. The skips are the Fortran 24 h comparison
  tests, which need `benchmarks/run_24h.sh` output that wasn't generated locally;
  they only exercise the unchanged 40-bin path.

### Known limitations
- The linear scheme takes ~1.4× TFL's wall time (6.9 s vs 4.6 s, 160 bins, 24 h, CPU).
- Empty-bin NEPS placeholders make `M + overflow` drift upward in proportion
  to the number of empty bins: up to 5e-8 at 320 bins in the constant-kernel
  case. This is pre-existing and identical for both schemes.
- `gpu-fast` branch: `fast/coagulation_pallas.py` hard-codes TFL and a base-2
  MNFIX, so it stays valid only for 40 bins.

### Next steps
- After merge into dev: bump the tomas-api `tomas-jax` submodule.
- Optional: closed-form destination index for geometric grids, to close the speed gap.

---

## 2026-05-07 (Wed) — Nucleation rate helpers refactored

**Time**: evening PST
**Branch**: `feature/zhao2024-rate-fixes`

### Summary
Factored the rate computation out of `zhao2024_nucleation_step` into a
new `zhao2024_nucleation_rate(...) -> (total_J, per_mech_(11,))` helper
and renamed `estimate_nucleation_rate` to `ricco_dunne_nucleation_rate`
for symmetry. Both helpers are JIT-safe and compute the actual
instantaneous rate without state mutation. The "estimate" terminology
was misleading — the function never approximated; it just didn't apply
gas-clamping. Driven by tomas-api needing a way to report the correct
rate per scheme in its diagnostic snapshots and per-step history.

### Changes
- `tomas_jax/physics/nucleation.py`:
  - New `zhao2024_nucleation_rate`. Returns total J and per-mechanism
    array in enable_masks order (bn, tn, bi, ti, syn, porg_n, porg_i,
    org_sa, amine, iod_n, iod_i).
  - `zhao2024_nucleation_step` now delegates rate computation to the
    helper before applying gas-clamping and bin-0 deposition.
  - `ricco_dunne_nucleation_rate` (renamed). `estimate_nucleation_rate`
    kept as a deprecated alias for backward compat.
- `tomas_jax/solvers/condensation.py`: internal callers (4 call sites
  in `_full_step_core`, `full_step_jax`, `_run_scan` body, `make_step`)
  switched to `ricco_dunne_nucleation_rate`.
- `tests/test_nucleation.py`: added `TestRiccoDunneNucleationRate` and
  `TestZhao2024NucleationRate` (8 tests) including a refactor regression
  asserting `dN_step == total_J * boxvol * dt`.

### Known limitations
- The new helper isn't yet re-exported from `tomas_jax/__init__.py`;
  callers still import from `tomas_jax.physics.nucleation`.

---

## 2026-04-18 (Fri) — 1000-Scenario Tropo→Strato Benchmark (PPM-only)

**Time**: late afternoon PST

### Summary
Scaled the Fortran-vs-JAX PPM benchmark from 49 tropospheric scenarios to 1000 scenarios spanning troposphere → lower stratosphere. Three strata: tropo (600, P ≥ 25 kPa), UTLS (250, 5 kPa ≤ P < 25 kPa), strato (150, 100 Pa ≤ P < 5 kPa). JAX PPM_JIT vs Fortran PPM on all 3 modes (coag_only, cond_only, combined).

### New infrastructure
- `benchmarks/python/scenarios_atmos.py` — stratified LHC generator (3 regimes × regime-realistic ranges)
- `tomas_fortran/harness/benchmark_24h_ppm_atmos.f` — 1000-scenario Fortran harness with 4-digit scenario IDs, crash-tolerant timing CSV
- `benchmarks/python/run_atmos_scenarios.py` — PPM-only JAX runner; fast combined path via `run_combined_scan_ppm` (avoids the 900 s/scenario Python fallback)
- `benchmarks/python/export_atmos_metrics.py` — regime-classified skill metrics export (1000 × 3 modes × 4 hours = 12k rows)
- `benchmarks/python/plot_atmos_suite.py` — 12 comprehensive figures (coverage, pairplot, regime Taylor, violin, CDF, error-vs-P, error contour, per-bin heatmap, error evolution, mass conservation, rep scenarios, walltime)
- `benchmarks/python/build_atmos_tables.py` — regime-stratified markdown + LaTeX skill tables
- `benchmarks/run_atmos.sh` — orchestration driver
- `tomas_fortran/Makefile` — added `benchmark_24h_ppm_atmos` + `run_24h_ppm_atmos` targets

### Runtime (measured)
- Fortran 1000-run: 1000 scenarios × 3 modes = 3000 sims, 216 001 CSV files, 0 crashes
- JAX 1000-run: 2993/3000 runs in 3.23 h wall (coag_only ~3.9 s/scen, cond_only ~0.2 s/scen, combined ~0.6 s/scen)
- Results: `benchmarks/results/atmos/` (3000 NPZs) + `tomas_fortran/output/atmos/` (216k CSVs) + `benchmarks/results/atmos/stats/` (12 figures + 2 tables + 12k-row metrics CSV)

### Key discovery
The old `run_24h_scenarios.run_scenario` combined-mode path falls through to a Python for-loop with 1440 per-step dispatches (~920 s/scenario). Bypassed by calling `run_combined_scan_ppm` directly from `run_atmos_scenarios.run_combined_fast`, which uses the existing XLA scan-fused kernel. Per-scenario wall time dropped from 920 s → 0.6 s (1500× speedup).

### Ensemble skill (h24, all 1000 scenarios)
| Regime | Mode | n | Nk R² med | Nk KGE med | \|N_tot relerr\| med | \|N_tot relerr\| p95 |
|---|---|---:|---:|---:|---:|---:|
| tropo  | coag_only | 600 | 0.9997 | 0.9967 | 2.3e-03 | 1.5e-02 |
| tropo  | cond_only | 600 | 0.9585 | 0.9401 | 1.7e-07 | 2.4e-04 |
| tropo  | combined  | 600 | 0.9802 | 0.9679 | 3.3e-03 | 2.8e-02 |
| utls   | coag_only | 250 | 0.9997 | 0.9990 | 6.4e-04 | 1.4e-02 |
| utls   | cond_only | 250 | 0.9325 | 0.8804 | 2.0e-07 | 2.1e-04 |
| utls   | combined  | 250 | 0.9371 | 0.8849 | 1.4e-03 | 2.7e-02 |
| strato | coag_only | 150 | 0.9998 | 0.9996 | 2.7e-04 | 5.8e-03 |
| strato | cond_only | 150 | 0.9097 | 0.8571 | 5.5e-11 | 1.3e-07 |
| strato | combined  | 150 | 0.9120 | 0.8585 | 6.6e-04 | 9.7e-03 |

### Findings
1. **Coagulation**: JAX matches Fortran to within 0.03-0.2% median on N_tot across the entire tropo→strato range. Nk R² > 0.9997 in all regimes.
2. **Condensation**: per-bin R² degrades from 0.96 (tropo) → 0.91 (strato) as conditions get colder/drier/sparser. But total N is essentially unchanged (N_tot_relerr median 1e-7 to 5e-11) — the two codes redistribute the mass slightly differently but conserve the totals to machine precision.
3. **Stratosphere**: N_total rel err is smallest in strato (5.5e-11 for cond_only) because condensation events are tiny when gas and N are low. But per-bin log-bias is larger (regime has wider Nk dynamic range).
4. **No Fortran crashes** at any scenario across the 1000-point atmospheric envelope — Fortran PPM is robust stratospheric-regime conditions (surprising given the code has no stratospheric test coverage in the original distribution).

---

## 2026-04-18 (Fri) — Fortran-vs-JAX Statistical Benchmark Suite + Audit

**Time**: afternoon PST

### Summary
Added a thin statistical-reporting layer on top of the existing 49-scenario LHC × 5-mode × 2-method benchmark so Fortran-vs-JAX parity can be presented to the group with quantitative skill metrics. Zero new simulations; all new code consumes existing NPZ/CSV artifacts. Also shipped a consolidated audit document covering JIT status and precomputed-and-recycled compromises.

### New files
- `benchmarks/python/metrics.py` — skill metrics (bias, rmse, nrmse, mape, r2, pearson_r, kge, log_bias, max_relerr)
- `benchmarks/python/export_metrics_csv.py` — 3750-row per-(scenario,mode,pair,hour) CSV + 75-row summary
- `benchmarks/python/plot_statistical_suite.py` — 7 figures: Taylor, Q-Q, CDF, per-bin heatmap, error-vs-params, walltime, convergence
- `benchmarks/python/build_presentation_tables.py` — markdown + LaTeX ensemble tables
- `benchmarks/run_stats.sh` — single entrypoint, runs zero simulations
- `docs/AUDIT_JAX_VS_FORTRAN.md` — consolidated audit: JIT status, compromises, GPU readiness

### Key numbers (h24, 49 LHC scenarios)
- coag_only TFL vs Fortran: Nk R² = 0.9998, |N_tot relerr| median = 1.2e-3, p95 = 9.0e-3
- cond_only TFL vs Fortran: Nk R² = 0.9986, |N_tot relerr| median = 1.4e-7
- combined TFL vs Fortran: Nk R² = 0.9996, |N_tot relerr| median = 1.5e-3
- `full` mode has the intentional nucleation-clamping tail (p95 ≈ 1); expected.

### Infrastructure
- Symlinked `tomas_fortran/output/24h` → nested actual location for compare_24h path compatibility.
- `manifest.json` records md5 of scenarios.csv so stats export exits loudly if LHC regenerates.
- `--scenarios` flag plumbed through for incremental runs.

### Files Modified
- `PROGRESS.md` — this entry
- `docs/porting_status.md` — cross-link to audit doc

---

## 2026-04-17 (Thu) — Radiative Forcing JAX Port: Phase 5 (Gauss-Legendre Upscatter)

**Time**: morning PST

### Summary
Replaced scipy.quad-based upscatter integration with 32-point Gauss-Legendre quadrature, making the entire precomputation pipeline JIT-compilable. Test suite speedup: 36s → 3s. All 84 RF tests pass.

### Phase 5: Gauss-Legendre Upscatter
- Added `_upscatter_fraction_gl(g, sza_rad)` — JIT-compilable upscatter via 32-point GL quadrature
- Added `_avg_solar_power_gl(lat_rad, sda, So)` — JIT-compilable 24h solar power average
- Rewrote `_compute_global_avg_upscatter` — vectorized with 2D vmap over (g, sza) pairs
- Vectorized `scattering_efficiency_vs_radius` — single-wavelength uses vmap + batch upscatter; spectral uses 2D vmap
- Module-level GL constants: `_GL_NODES`, `_GL_WEIGHTS` (32-point, from `np.polynomial.legendre.leggauss`)
- GL vs scipy accuracy: atol < 1e-4 across 16 (g, sza) combinations
- Plot validation: fig2/fig3 byte-identical; fig1 visually identical (121-byte diff from GL approximation)

### Tests Added
- `TestGaussLegendreUpscatter` class (22 tests): GL vs scipy match (4g × 4sza = 16 parametrized), bounds, JIT, vmap, grad, isotropic
- Total: 84 RF tests (37 existing + 47 new)

### Performance
- Test suite: 36s → 3.3s (11x speedup from eliminating scipy.quad in precomputation)
- `scattering_efficiency_vs_radius`: fully vectorized, no Python loops

### Files Modified
- `tomas_jax/physics/radiative_forcing.py` — GL quadrature, vectorized upscatter/scattering_efficiency
- `tests/test_radiative_forcing.py` — `TestGaussLegendreUpscatter` class

---

## 2026-04-16 (Wed) — Radiative Forcing JAX Port (Phases 1–4)

**Time**: evening PST

### Summary
Ported the radiative forcing pipeline to pure JAX with JIT compilation, vmap vectorization, and autodiff support. Three files modified: `bhmie.py` (new JAX Mie functions), `radiative_forcing.py` (runtime functions ported to jnp), `test_radiative_forcing.py` (26 new tests).

### Phase 1: bhmie → JAX
- Added `bhmie_jax(x, refrel, nang)` — full Mie with S1/S2, JIT-compilable via `@partial(jax.jit, static_argnums=(2,))`
- Added `bhmie_qsca_jax(x, refrel)` — lightweight (Qext, Qsca, gsca only), `@jax.jit`
- Uses `jax.lax.fori_loop` with fixed upper bounds (NSTOP_MAX=250, NMX_MAX=300) and traced masks
- Fixed S1/S2 backward sign convention: `p = -p` must happen **before** backward amplitude accumulation
- Matches numpy `bhmie` to rtol=1e-10 for x ∈ {0.01, 0.1, 1.0, 5.0, 20.0, 50.0, 100.0}

### Phase 2: Runtime RF → jnp
- `compute_optical_depth`, `compute_rf`, `compute_mass_scattering_efficiency`, `compute_rf_efficiency`: `np.*` → `jnp.*`, `if/else` → `jnp.where`, `@jax.jit`
- `h2so4_solution_density`, `_interpolate_upscatter`: `np.interp` → `jnp.interp`, `@jax.jit`
- `_planck`: `np.exp` → `jnp.exp`; `_solar_spectral_weights`: `np.linspace` → `jnp.linspace`
- Precomputation functions (scipy.quad-based) remain as numpy fallbacks

### Phase 3: vmap Mie precomputation
- Single-wavelength: `jax.vmap(bhmie_qsca_jax, (0, None))` replaces 40-bin for-loop
- Spectral: 2D vmap `jax.vmap(jax.vmap(...))` replaces 30-wavelength × 40-bin double loop
- Vectorized upscatter interpolation and weighted accumulation via broadcasting

### Phase 4: Tests
- 26 new tests: `TestBhmieJax` (14 tests) + `TestJITRuntime` (12 tests)
- bhmie_jax: numpy reference match, JIT, vmap (1D + 2D), grad through Qsca
- Runtime: JIT correctness, grad w.r.t. Nk and column_area, vmap over distributions
- Total: 63 RF tests (37 existing + 26 new)

### Files Modified
- `tomas_jax/physics/bhmie.py` — Added `bhmie_jax`, `bhmie_qsca_jax`, helpers
- `tomas_jax/physics/radiative_forcing.py` — JAX imports, jnp runtime, vmap precomputation
- `tests/test_radiative_forcing.py` — `TestBhmieJax`, `TestJITRuntime` classes

### Known Limitations
- Grad through Qsca at integer nstop boundaries has local accuracy only (jnp.int32 truncation)

---

## 2026-04-16 (Wed) — Phase A Hardening: Unit Tests for Foundation Modules

**Time**: evening PST

### Summary
Added 90 unit tests across 4 previously untested foundation modules: `nh3_equilibrium.py`, `water_equilibrium.py`, `condensation_sink.py`, and `mnfix_jax.py`. All 677 tests pass (up from 579). Updated `docs/porting_status.md` with test counts.

### New Test Files

1. **`tests/test_nh3_equilibrium.py`** — 20 tests
   - Ammonia-limited case (all NH3 → particles, distributed proportional to SO4)
   - Ammonia-excess case (2:1 NH4:SO4 stoichiometry, excess stays gas)
   - Total nitrogen conservation across 5 NH3/SO4 ratios
   - Edge cases: zero NH3, zero SO4, single bin, species isolation
   - JIT compilation and eager/JIT equivalence

2. **`tests/test_water_equilibrium.py`** — 30 tests
   - Sulfate WR: known references (50%→1.29, 80%→1.96, 95%→5.05), monotonicity, clipping
   - Piecewise polynomial continuity at 5 breakpoints (41%, 61%, 81%, 91%, 96%)
   - Sea salt WR: low RH baseline, higher hygroscopicity than sulfate
   - calc_equilibrium_water: SO4/organic contributions, proportionality, species isolation
   - JIT compilation

3. **`tests/test_condensation_sink.py`** — 18 tests
   - CS physical range, formula verification, scaling with Nk and particle size
   - sinkfrac normalization, non-negativity, single-bin dominance
   - NEPS threshold (1e10), sparse bin default density handling
   - NaN guard for Dpk=0, accommodation coefficient effect
   - JIT compilation

4. **`tests/test_mnfix.py`** — 22 tests
   - Phase 1: empty bins get NEPS=1e-5, SO4 at geometric mean
   - Phase 2: extreme avg mass trimming (above grid, below grid)
   - Phase 3: partial transfer (upward, downward, partial not total, in-range unchanged)
   - Conservation: number, mass, multi-species, composition fractions
   - Multi-bin jumps (10+ bins), positivity, no-NaN stress test
   - JIT compilation

### Updated Documentation
- `docs/porting_status.md`: Updated test counts in tables, marked 4 modules as tested, reduced untested count from 8 to 4

---

## 2026-04-15 (Tue) — Radiative Forcing with Tabazadeh H₂SO₄/H₂O Equilibrium

**Time**: afternoon PST

### Summary
Added direct shortwave radiative forcing module (Chylek & Wong 1995 / Pierce et al. 2010) with Bohren-Huffman Mie scattering and Tabazadeh et al. (1997) binary H₂SO₄/H₂O equilibrium composition parameterization.

### Changes

1. **Mie scattering** (`tomas_jax/physics/bhmie.py`)
   - Bohren-Huffman Mie code (numpy). Computes Qsca, Qext, gsca for any size parameter and refractive index.

2. **Radiative forcing** (`tomas_jax/physics/radiative_forcing.py`)
   - `precompute_mie_properties()`: Mie + global-avg upscatter (Wiscombe & Grams 1976). Single-wavelength and spectral integration modes.
   - `compute_rf()`: Chylek & Wong RF with Pierce SI cloud-fraction extension.
   - `scattering_efficiency_vs_radius()`: Reproduces Pierce et al. (2010) Figure 1.
   - `h2so4_equilibrium_wt(temp, rh)`: Tabazadeh et al. (1997) Table 1 vapor pressure interpolation.
   - `h2so4_solution_density(wt)`: CRC Handbook density interpolation.

3. **Benchmark plots** (`benchmarks/python/plot_radiative_forcing.py`) — 5 figures
4. **Tests** (`tests/test_radiative_forcing.py`) — 8 Tabazadeh composition tests + RF tests
5. **Documentation** (`docs/radiative_forcing.md`)

### Key Finding
Pierce assumed 75 wt% H₂SO₄ (peak 0.84 W/m² per Mt-S). Tabazadeh at T=220K, RH=5% gives 63.8 wt% and peak 1.09 — 30% higher due to water dilution.

---

## 2026-04-10 (Thu) — JAX Performance Optimization

**Time**: evening PST

### Summary
Comprehensive JAX/JIT performance audit and optimization. Identified and fixed redundant computation in hot loops, unnecessary memory allocations, and unused scan arrays.

### Changes

1. **PPM substep loop-invariant hoisting** (`tomas_jax/physics/condensation_ppm.py`)
   - Moved `compute_edge_velocity()` and `_compute_moment_integrals()` out of `fori_loop` in both `ppm_condensation_step` and `ppm_advect_number_only`
   - All inputs are constant across CFL substeps — explicit hoisting guarantees no redundant computation regardless of XLA LICM behavior

2. **Removed `.copy()` on immutable JAX arrays** (`tomas_jax/physics/condensation_tfl_jax.py`)
   - `ANKD.copy()` / `AMKD.copy()` replaced with direct references — JAX arrays are immutable

3. **Eliminated unused scan carry allocation** (`tomas_jax/solvers/condensation.py`)
   - `jnp.arange(nsteps)` passed to `jax.lax.scan` but never read by body → replaced with `None`

### Deferred (not worth the churn)
- `static_argnums` → `static_argnames`: cleaner but requires updating all call sites
- `compute_zeta(xk)` caching: just a few scalar FLOPs, XLA LICM handles this
- kwargs validation restructure: current trace-time behavior is correct

### Verification
All 550 tests pass (0 failures). PPM, TFL JIT, and smoke tests verified.

---

## 2026-04-10 (Thu) — Radiative Forcing Module (Phase 9)

**Time**: afternoon PST

### Summary
Added direct shortwave radiative forcing calculation based on Chylek & Wong (1995), following Pierce et al. (2010).

### Files created
- `tomas_jax/physics/bhmie.py` — Bohren-Huffman Mie scattering (numpy, cleaned from reference)
- `tomas_jax/physics/radiative_forcing.py` — RF module: Mie precomputation per TOMAS bin, upscatter fraction (Wiscombe & Grams 1976), optical depth, Chylek & Wong RF equation
- `tests/test_radiative_forcing.py` — 29 tests (all passing), covers Mie, upscatter, RF, MSE, Pierce Fig 1 validation
- `docs/radiative_forcing.md` — Full documentation with usage examples

### Key features
- `precompute_mie_properties(xk)`: Qsca, Qext, gsca, global-avg upscatter for each bin
- `compute_rf(Nk, mie, column_area)`: RF [W/m²] from Chylek & Wong 1995
- `compute_rf_efficiency(Nk, Mk, mie)`: RF per unit burden [W/m² per g/m²]
- `compute_mass_scattering_efficiency(Nk, Mk, mie)`: MSE [m²/g]
- `scattering_efficiency_vs_radius()`: Reproduces Pierce et al. 2010 Figure 1 curve
- `compute_rf_latitude_resolved()`: Full lat/month calculation as in getRF_monthly.py

### References
- Chylek & Wong (1995), GRL 22, 929-931
- Pierce et al. (2010), GRL 37, L18805
- Wiscombe & Grams (1976), J. Atmos. Sci. 33, 2440-2451
- Bohren & Huffman (1983), Absorption and Scattering of Light by Small Particles

### Next steps
- Multi-wavelength integration across solar spectrum
- Gravitational settling efficiency (Pierce Fig 1 right axis)

---

## 2026-04-10 (Thu) — Code Review Round 2: Robustness, Tests, Documentation

**Time**: morning PST

### Summary
Second round of code review hardening. Focus on numerical robustness, test quality, test coverage, and physics documentation.

### Changes
1. **Euler solver robustness** — replaced hardcoded `1e-15`/`1e-25` thresholds with `TINY_N`/`TINY_M` from config. Added `return_info` parameter to detect when `max_substeps` truncates integration.
2. **Tightened PPM flux test** — `rtol=0.1` (10%) → `rtol=1e-10` for test where analytical answer is exact.
3. **Shared test fixtures** — created `tests/conftest.py` with `xk` fixture and `make_lognormal` helper. Removed duplicate grid construction from 3 test files.
4. **Deleted stale `pyproj.toml`** (typo'd filename, outdated metadata). Removed dead `ZETA_DEFAULT` constant. Fixed misleading Dpk=0 comment in coagulation kernel.
5. **Literature references** — added citations for Riccobono 2014, Dunne 2016, Tang 1997, Sutherland viscosity, and TOMAS MFP formula.
6. **Smoke tests** — 8 lightweight tests (~5s) exercising coagulation, condensation, and full pipeline. Always run, no Fortran data needed.
7. **Physics foundation tests** — 22 unit tests for `density.py`, `properties.py`, `gas_properties.py` (previously zero coverage). Tests against reference values and physical limits.

### Files Modified/Created
- `tomas_jax/solvers/euler.py` (thresholds, return_info)
- `tomas_jax/physics/coagulation_rates.py` (removed ZETA_DEFAULT)
- `tomas_jax/physics/coagulation_kernel.py` (comment fix)
- `tomas_jax/physics/nucleation.py` (citations)
- `tomas_jax/physics/density.py` (citation)
- `tomas_jax/physics/properties.py` (citation)
- `tomas_jax/physics/gas_properties.py` (MFP explanation)
- `tests/conftest.py` (new — shared fixtures)
- `tests/test_smoke.py` (new — 8 smoke tests)
- `tests/test_physics_foundations.py` (new — 22 unit tests)
- `tests/test_ppm_condensation.py` (tightened tolerance)
- `tests/test_nucleation.py`, `test_tfl_jit_condensation.py`, `test_ppm_jit_condensation.py` (use shared fixtures)
- Deleted: `pyproj.toml`

---

## 2026-04-09 (Wed) — Code Review Fixes

**Time**: evening PST

### Summary
Addressed issues found during comprehensive dev branch code review. Focus on API safety, solver robustness, naming consistency, and test infrastructure.

### Changes
1. **kwargs validation in `make_step()`** — warns on unknown kwargs (catches typos that silently default to 0.0). Validates against set of 17 known parameter names.
2. **Process order validation** — warns when process order deviates from canonical physical ordering (so2_chemistry → nucleation → coagulation → condensation → dilution).
3. **diffrax solver status checking** — checks `solution.result` after each Tsit5 substep; reverts to previous state on failure instead of silently using partial results. Optional `return_status=True` flag for callers.
4. **`icomp_nodiag` constant cleanup** — replaced all hardcoded `=42` defaults with `ICOMP_NODIAG` from config across 6 files.
5. **Renamed `accommodation_coeff` → `alpha`** — standardized parameter name across `condensation_sink.py`, `gas_properties.py`, and 10 call sites to match TomasState convention.
6. **Stale README reference** — removed deleted `run_sensitivity_analysis.py` from project structure.
7. **`coag_rk4_step` deprecation** — now emits `DeprecationWarning` instead of being a silent alias.
8. **Top-bin overflow tracking for coagulation mass budget** — `calc_coagulation_rates` now returns a third value `dM_overflow` (mass rate that would enter a hypothetical bin above the grid, i.e., the terms `shift_right` truncates). Both `coag_euler_step` and `diffrax_step` accumulate overflow via `return_overflow=True`. Root cause of coag mass conservation test failures: top-bin self-coagulation produces particles exceeding the grid boundary — this mass is physically lost. Now tracked explicitly so `M(0) = M(24h) + overflow` closes to <1e-8 relative.
9. **Rewrote `test_coag_mass_conservation`** — no longer loads stale NPZ data; runs solver inline with overflow tracking and verifies mass budget closure across all 50 scenarios.
10. **`--force` flag for benchmark runner** — `run_24h_scenarios.py` now supports `--force` to overwrite existing NPZ files.

### Files Modified
- `tomas_jax/solvers/condensation.py` (kwargs validation, process order, ICOMP_NODIAG)
- `tomas_jax/solvers/diffrax.py` (solver status, ICOMP_NODIAG, coag_rk4_step deprecation, overflow tracking)
- `tomas_jax/solvers/euler.py` (ICOMP_NODIAG, overflow unpacking)
- `tomas_jax/solvers/coagulation_jax.py` (overflow unpacking)
- `tomas_jax/physics/coagulation_rates.py` (ICOMP_NODIAG, dM_overflow return)
- `tomas_jax/physics/condensation_sink.py` (alpha rename)
- `tomas_jax/physics/gas_properties.py` (alpha rename)
- `tomas_jax/physics/ezcond.py`, `ezcond_ppm.py`, `ezcond_ppm_jax.py`, `condensation_tfl_jax.py` (alpha rename)
- `tomas_jax/core/mnfix_jax.py`, `mnfix_fortran.py` (ICOMP_NODIAG)
- `tomas_jax/utils/diagnostics.py` (overflow unpacking)
- `tests/test_24h_scenarios.py` (rewrote coag mass conservation test)
- `benchmarks/python/run_24h_scenarios.py` (--force flag)
- `benchmarks/python/level07_rates.py` (overflow unpacking)
- `README.md` (stale reference)

---

## 2026-03-24 (Mon) — JAX/GPU-Readiness Audit

**Time**: ~evening PST

### Summary
Comprehensive audit of all ~25 tomas_jax modules for GPU deployment readiness. The JIT-compiled fast paths were already excellent pure JAX. This audit fixed patterns that would cause issues on GPU: `None` default args breaking JIT tracing, missing kwarg defaults, scattered `jax.config.update` calls, and undocumented legacy numpy paths.

### Changes
1. **Moved coag import to module level** in `solvers/condensation.py` — was inside 3 function bodies, now at top with other imports.
2. **Removed 12 redundant `jax.config.update("jax_enable_x64", True)`** calls from library modules (`core/mnfix_jax.py`, `core/mnfix_fortran.py`, `physics/condensation_sink.py`, `physics/coagulation_rates.py`, `physics/coagulation_kernel.py`, `physics/density.py`, `physics/gas_properties.py`, `physics/nh3_equilibrium.py`, `physics/water_equilibrium.py`, `physics/properties.py`, `solvers/euler.py`, `solvers/diffrax.py`). Only `core/config.py` retains the call.
3. **Fixed `dilution_step` None defaults** — removed `None` default args that break JIT when call patterns change. Now requires explicit arrays. Updated callers: `solvers/condensation.py`, `tests/test_dilution.py`, `run_box_model.py`.
4. **Fixed `make_step` kwargs** — replaced bare `kwargs['nh3_conc']`, `kwargs['fion']`, `kwargs['org_conc']` with `kwargs.get('key', 0.0)` to prevent KeyError crashes.
5. **Added deprecation warnings** to legacy numpy paths (`method='tfl'`, `method='ppm'`) in `condensation_step()`. Updated docstrings in `condensation.py`, `ezcond.py`, `ezcond_ppm.py`.
6. **Auto-JIT `make_step`** — added `jit=True` parameter; returns JIT-compiled function by default. Removed redundant `jax.jit()` wrapping from `run_box_model.py`, `benchmark_dilution.py`, `benchmark_so2_sensitivity.py`, `run_sai_simulation.py`.
7. **Created `docs/gpu_deployment.md`** — float64 by GPU type, `JAX_PLATFORMS` usage, code path selection, `jax.vmap` batch processing, memory estimates, JIT stability tips.
8. **Updated CLAUDE.md** — added GPU deployment section, centralized x64 config note, dilution_step change, file layout entry.

### Files Modified
- `tomas_jax/solvers/condensation.py` (import, kwargs, dilution, deprecation, auto-JIT)
- `tomas_jax/physics/dilution.py` (removed None defaults)
- `tomas_jax/physics/condensation.py`, `ezcond.py`, `ezcond_ppm.py` (deprecation docstrings)
- 12 files: removed redundant `jax.config.update` calls
- `tests/test_dilution.py` (explicit zero backgrounds)
- `run_box_model.py` (auto-JIT, explicit backgrounds)
- `benchmarks/python/benchmark_dilution.py`, `benchmark_so2_sensitivity.py` (auto-JIT)
- `experimental_case/run_sai_simulation.py` (auto-JIT)

### Files Created
- `docs/gpu_deployment.md`

### Known Issues
- Tests and benchmarks retain their own `jax.config.update` calls (they may be run as standalone scripts)
- No actual GPU testing performed (audit is structural/code-level only)

---

## 2026-03-14 (Fri) — SAI 96h Box Model Simulation

**Time**: ~10:15 PM PST

### Summary
Added a 96-hour SAI box model simulation that evolves a fresh SO2 plume (10 tonnes, 1 ton/min, 250 m/s, r=100m → ~2×10¹³ molec/cm³) through all microphysical processes at stratospheric conditions (216K, 55 hPa). Three dilution cases: baseline (closed box), volume 2× in 24h, and volume 1.78× in 24h.

### Files Created
- `experimental_case/run_sai_simulation.py` — 96h simulation: 3 cases × 5760 steps, 6 figures
- `experimental_case/__init__.py` — Package init for module imports

### Files Modified
- `experimental_case/README.md` — Added simulation description, usage, output table

### Key Results
- Massive nucleation burst in first ~2h: N goes from 5/cm³ background to ~2×10⁷/cm³
- SO2 decay: baseline τ≈11 days (OH-limited), dilution cases much faster
- H2SO4 consumed rapidly by condensation sink + nucleation
- Banana plots show classic nucleation → growth → coagulation evolution
- Runtime: ~15s per case (~45s total for 3 cases)

### Figures (in `experimental_case/results/`)
1. `fig1_ntotal.png` — N_total timeseries (absolute + ratio with passive tracer)
2. `fig2_mdry.png` — M_dry timeseries
3. `fig3_dndlogdp.png` — dN/dlogDp snapshots at 12h, 24h, 48h, 72h, 96h
4. `fig4_dmdlogdp.png` — dM/dlogDp snapshots
5. `fig5_banana.png` — Banana plots (3 panels)
6. `fig6_gas.png` — SO2 + H2SO4 gas-phase timeseries

---

## 2026-03-13 (Thu) — Dilution / Entrainment Process

**Time**: ~PST

### Summary
Added dilution/entrainment as the 5th physics process. First-order relaxation of Nk, Mk, Gc toward a background state: `C(t+dt) = Cbg + (C - Cbg) * exp(-kdil * dt)`. Analytical, unconditionally stable, JIT-compilable. Includes a passive scalar tracer (`dilute_tracer()`) that decays by dilution only — provides a clean reference to compare reactive species against pure dilution.

### Files Created
- `tomas_jax/physics/dilution.py` — `dilution_step()` + `dilute_tracer()` functions
- `tests/test_dilution.py` — 10 unit tests (6 dilution_step + 4 passive tracer)
- `docs/dilution.md` — Algorithm docs, typical kdil values, tracer usage

### Files Modified
- `tomas_jax/solvers/condensation.py` — Added `'dilution'` to `make_step()` valid set + process dispatch
- `run_box_model.py` — Added `--dilution-rate`, `--dilution-bg` CLI flags; passive tracer tracking + diagnostics
- `CLAUDE.md` — Added dilution.py to file layout, updated architecture rules
- `docs/missing_physics.md` — Marked §5 (Dilution) as implemented

### Benchmark
- `benchmarks/python/benchmark_dilution.py` — 3 cases (baseline/moderate/strong) × 24h, all processes + SO2
- 6 figures: N_total, M_dry, gas-phase SO2+H2SO4, size distributions, dilution attribution, banana plots
- Passive tracer overlay shows chemistry/nucleation partially compensating dilution losses
- Runtime: ~1.8s per scenario (3 × 1.8s = 5.4s total)

### Usage
```bash
python run_box_model.py --make-step --dilution-rate 1e-4                       # clean-air
python run_box_model.py --make-step --dilution-rate 1e-4 --dilution-bg ambient # relax to initial
python -m benchmarks.python.benchmark_dilution                                 # full benchmark + plots
python -m benchmarks.python.benchmark_dilution --plot-only                     # replot from NPZ
```

---

## 2026-03-12 (Thu) — SO2 Lifetime Heatmaps (Surface to Stratosphere)

**Time**: ~6:00 PM PST

### Summary
Added Figure G to SO2 chemistry validation: 3-panel 2D heatmap showing SO2+OH kinetics from surface to stratosphere (0–50 km). Uses US Standard Atmosphere 1976 (all 5 layers).

### Key Results
- Panel (a): k1(T,P) — rate constant drops ~130× from surface to 50km due to pressure decrease
- Panel (b): SO2 lifetime at [OH]=10⁶ — ranges from ~10 days (surface) to ~1400 days (50km)
- Panel (c): Lifetime vs altitude & [OH] — practical reference chart for atmospheric modelers
- Stratospheric temperature inversion (T rising above 20km) visible in standard atmosphere overlay

### Files Modified
- `benchmarks/python/validate_so2_chemistry.py` — Added `_std_atmosphere()` (US Std Atm 1976, 0–50km), `figure_g()` (3-panel heatmap); now 7 figures total

---

## 2026-03-12 (Thu) — SO2 Sensitivity Benchmark

**Time**: ~5:00 PM PST

### Summary
Added SO2 sensitivity benchmark: 5 SO2 concentrations (1e8–1e11 molec/cm³) × 4 process modes × 3 altitudes × 2 grid resolutions over 48 hours. Produces 12 figures showing SO2 decay (3 units), H2SO4 gas, particle number/mass evolution, banana plots, and size distributions.

### Files Created
- `benchmarks/python/benchmark_so2_sensitivity.py` — Full benchmark runner + 12-figure plotting suite

### Key Results
- 20 scenarios (surface, 40-bin) run in ~40s total
- SO2 decay ratio at 48h: ~0.82 (surface, cond_only, SO2=1e10), consistent with k1×[OH]=1e6 lifetime ~11 days
- Full 120-scenario suite (all altitudes + 80-bin): ~15-30 min estimated
- All 12 figures generate correctly from saved NPZ files

### Usage
```bash
python -m benchmarks.python.benchmark_so2_sensitivity           # run all + plot
python -m benchmarks.python.benchmark_so2_sensitivity --plot-only  # replot from NPZ
python -m benchmarks.python.benchmark_so2_sensitivity --grids 40   # 40-bin only
python -m benchmarks.python.benchmark_so2_sensitivity --altitudes sfc  # surface only
```

---

## 2026-03-12 (Thu) — SO2 + OH Gas-Phase Chemistry

**Time**: ~2:30 PM PST

### Summary
Implemented SO2 + OH → H2SO4 gas-phase chemistry using the Sun et al. (2022) Troe formalism with H2O vapor enhancement. This replaces the constant H2SO4 production rate with physically-based SO2 oxidation. All 4 functions are JIT-compilable.

### Key Details
- **Rate constant**: k1(298K, 1atm) = 1.05×10⁻¹² cm³/molec/s, SO2 lifetime ~11 days at [OH]=10⁶
- **H2O enhancement**: 2-5% at 298K, 50% RH (Sun et al. Eqs. 5-6)
- **OH modes**: Constant (default) and diurnal (proportional to cos(SZA))
- **Gc extended**: 43→44 elements. SRTSO2=43, N_GAS_SPECIES=44
- **Backward compatible**: No SO2 flags → legacy constant H2SO4 production path

### Files Created
- `tomas_jax/physics/so2_chemistry.py` — 4 JIT functions: calc_k1_so2_oh, calc_solar_zenith_angle, calc_oh_concentration, so2_oxidation_step
- `tests/test_so2_chemistry.py` — 28 tests (Troe rate, SZA, OH modes, oxidation step, conservation)
- `benchmarks/python/validate_so2_chemistry.py` — 6 validation figures reproducing Sun et al. (2022)
- `docs/so2_chemistry.md` — Algorithm documentation, usage, validation

### Files Modified
- `tomas_jax/core/config.py` — Added SRTSO2=43, MW_SO2=64.066, MW_OH=17.007, SV_SO2=41.73; N_GAS_SPECIES=44
- `tomas_jax/solvers/condensation.py` — Added 'so2_chemistry' to make_step valid processes; _run_scan supports so2_prod_rate
- `run_box_model.py` — New CLI: --so2-init, --so2-emission, --oh-conc, --oh-diurnal, --lat, --lon, --day-of-year
- `docs/references.md` — Added Sun et al. (2022), Buck (1981)
- `README.md` — SO2 chemistry features, CLI options, state variables, file tree
- `CLAUDE.md` — SO2 chemistry architecture rules, file layout, species indices

### Reference
- Sun, W., et al.: Kinetics of OH + SO2 + M. *Atmos. Chem. Phys.*, 22, 4969-4984, 2022.
- Supplemental: `references/so2_chemistry/acp-22-4969-2022-supplement.pdf`

### Verification
- 160 tests pass (28 new + 132 existing, all backward-compatible)
- Box model runs with SO2 chemistry: `python run_box_model.py --so2-init 5e10 --oh-conc 1e6 --make-step`
- 6 validation figures in `benchmarks/results/so2_chemistry/`

---

## 2026-03-12 (Thu) — Default Grid: 40 Bins, 1.7nm Start

**Time**: ~12:00 PM PST

### Summary
Changed default bin grid from 36 bins (3.2nm start) to 40 bins (1.7nm start, Dunne 2016 nucleation cluster size). Added 80-bin high-resolution preset. All physics/solver code is shape-agnostic — no changes needed there.

### Files Modified
- `tomas_jax/core/config.py` — NBINS=40, XK0=(pi/6)*(1.7e-9)^3*1770, added `XK0_LEGACY`, `NBINS_LEGACY`, `make_grid_80bin()`
- `tests/test_ppm_condensation.py` — Import NBINS from config instead of hardcoded
- `tests/test_tfl_jit_condensation.py` — Use XK0 from config
- `tests/test_ppm_jit_condensation.py` — Use XK0 from config
- 13 benchmark scripts — Import `NBINS_LEGACY as NBINS` and `XK0_LEGACY` for Fortran 36-bin comparison
- `benchmarks/python/test_cases.py` — Derive nbins from xk shape instead of config constant
- `CLAUDE.md` — Updated dimensions and grid documentation

### Grid Configurations
| Config | Bins | Start | End | Mass ratio | Usage |
|--------|------|-------|-----|------------|-------|
| Default (40-bin) | 40 | 1.7nm | 17.5μm | ×2 | Production, nucleation studies |
| High-res (80-bin) | 80 | 1.7nm | 17.5μm | ×√2 | Convergence studies |
| Legacy (36-bin) | 36 | 2.6nm | 10.6μm | ×2 | Fortran comparison benchmarks |

### Verification
- All 132 core tests pass (nucleation, PPM, TFL, coagulation)
- 24 pre-existing test_24h_scenarios failures (load stored 36-bin data — unrelated)
- Nucleation cluster (mnuc=3.47e-24 kg) still fits in bin 0 of 40-bin grid

---

## 2026-03-11 (Wed) — Zhao 2024 11-Mechanism NPF Scheme

**Time**: ~5:00 PM PST

### Summary

Implemented the complete Zhao et al. 2024 (Nature 631, 98–105) 11-mechanism new particle formation scheme as a selectable alternative to the existing Riccobono 2014 + Dunne 2016 scheme.

### Files Modified
- `tomas_jax/physics/nucleation.py` — Added 7 new mechanism functions + driver:
  - `zhao2024_synergistic_hno3_rate()` — Mechanism 5 (Wang 2022 + Zhao 2024 cutoff)
  - `kirkby2016_pure_organic_rate()` — Mechanisms 6–7 (Kirkby 2016 + Zhao 2024 T-factor)
  - `zhao2024_organic_h2so4_rate()` — Mechanism 8 (Lehtipalo 2018 refit)
  - `zhao2024_amine_h2so4_rate()` — Mechanism 9 (Hanson 2017 + Cai 2021 T-function)
  - `zhao2024_iodine_oxoacid_rate()` — Mechanisms 10–11 (He 2021 fits)
  - `zhao2024_nucleation_step()` — 11-mechanism driver with enable masks
  - `_compute_ionc()` — Shared ion concentration helper
  - Fixed `estimate_nucleation_rate()` missing parameters (Gc, temp, pres, boxvol)
- `tomas_jax/solvers/condensation.py` — Added `nucl_scheme` parameter to `make_step()`:
  - `nucl_scheme='ricco_dunne'` (default) or `'zhao2024'`
  - Python-level dispatch, no recompilation between schemes
  - Imported `zhao2024_nucleation_step`, `ZHAO2024_ALL_ENABLED`
- `run_box_model.py` — Added `--nucl-scheme` CLI option:
  - Both manual and `make_step` paths support Zhao 2024
  - Added Zhao 2024 gas precursor inputs (hno3, ulvoc, dma, hio3)
- `tests/test_nucleation.py` — Added 35 new tests (58 total, all passing):
  - `TestSynergisticHNO3` (4 tests): scaling, temperature cutoff
  - `TestKirkby2016` (5 tests): neutral/ion rates, temperature, zero input
  - `TestOrganicH2SO4` (3 tests): bilinear form, scaling
  - `TestAmineH2SO4` (5 tests): H₂SO₄^3, DMA^1.5 scaling, Cai 2021 f_T
  - `TestIodineOxoacids` (6 tests): rates, T-clamps at 263K/283K
  - `TestZhao2024Step` (8 tests): driver mass balance, enable masks, clamping, JIT
  - `TestComputeIonc` (3 tests): ion concentration helper
- `docs/zhao2024_nucleation.md` — Added driver function docs, usage examples, CLI
- `docs/future_features.md` — Added item #8: Explicit ELVOC/HOM Gas-Phase Species
- `CLAUDE.md` — Updated nucleation architecture rule and file layout

### Key Design Decisions
- `ulvoc` (Zhao 2024) and `org_conc` (Riccobono) are separate inputs
- Enable masks are tuples of 11 floats to avoid JIT recompilation
- ULVOC used for both mechs 6-7 (ULVOC-only) and mech 8 (ULVOC+ELVOC) — noted as approximation
- Kirkby 2016 [n±] units: back-calculated as 10³ cm⁻³ from Fig. 3 data
- Mechanism 9 base rate from Hanson 2017 formula (≡ Almeida × 10 at 278K)
- Iodine T-clamps: 263K neutral, 283K ion-induced (He 2021 experimental range)

### Known Limitations
- `estimate_nucleation_rate()` always uses Ricco+Dunne for adaptive sub-stepping (conservative estimate; Zhao 2024 rates may be higher)
- ELVOC/HOM as explicit gas species is a future feature (#8 in future_features.md)
- Scan-fused paths (`run_full_scan`, `run_nucleation_condensation_scan`) still use `ricco_dunne` only

---

## 2026-03-11 (Wed) — Web Interface UX Design Decisions

**Time**: ~PST

### Summary

Completed structured Q&A session to define all UX and architecture decisions for the tomas-app web interface. All decisions documented in `tomas-app/tomas-web/DECISIONS.md` (ADR-008 through ADR-016).

### Key Decisions
- **Bins**: 40 default, 80/160 as options (dropdown)
- **Simulation time**: 6–48h, default 12h
- **Timestep**: dropdown (1/5/10/20/30/40/60/120s), default 60s
- **Process modes**: 6 named modes (coag-only → full model), not independent toggles
- **Condensation method**: PPM only, no user choice
- **Gas-phase modes**: 4 modes per gas (H₂SO₄, NH₃, organics) — depleting, prescribed trend, constant, constant production
- **Prescribed trend profiles**: 6 profiles (diurnal, morning burst, linear ramp up/down, exponential decay, step change) with preview graph
- **Presets**: 7 scenarios populating all fields, all editable after selection
- **Plots**: 9 panels — number/mass size dist (hourly lines + hover), banana plot, N/M/Gc time series, nucleation rate, condensation sink, mass conservation
- **Audience**: atmospheric researchers, students, climate scientists (SAI/CCN)

### Next Steps
- Implement new form design with named process modes
- Implement 4-mode gas-phase forcing with profile previews
- Add bin resolution selector
- Add simulation time/timestep controls
- Implement 9-panel results visualization with hover interactions

---

## 2026-03-10 (Tue) — Adaptive Nucleation Sub-Stepping

**Time**: ~PST

### Summary

Implemented adaptive nucleation sub-stepping to handle high nucleation rates (J > 100 cm⁻³s⁻¹) that can create dN comparable to the entire existing particle population in a single 60s timestep.

### Files Modified
- `tomas_jax/physics/nucleation.py` — Added `estimate_nucleation_rate()` and `compute_nucleation_substeps()` (two new JIT-compilable functions)
- `tomas_jax/solvers/condensation.py` — Modified `_full_step_core()`, `condensation_step_with_nucleation_jax()`, `full_step_jax()`, `run_nucleation_condensation_scan()`, `run_full_scan()`, and `make_step()` to use adaptive nucleation sub-stepping via `fori_loop`
- `docs/nucleation.md` — Documented adaptive sub-stepping algorithm and parameters

### Algorithm
- Before nucleation, estimate total rate J and compute `n_sub = ceil(dN / (max_frac * N_total))`, clamped to [1, max_substeps]
- Loop `n_sub` times with `dt_nuc = dt / n_sub`, calling `nucleation_step` + `mnfix_jax` per substep
- Dynamic `n_nuc` compiles to XLA `while_loop` via `jax.lax.fori_loop`
- Default: `max_nucleation_frac=0.5` (50% per substep), `max_nuc_substeps=20`

### Verification
- All 83 core tests pass (nucleation: 24, PPM: 31, TFL: 15, coag: 13)
- Smoke test: J=363 at H2SO4=1e8 gives n_sub=5 (N_total=1e10) or n_sub=20 (N_total=1e8)
- With sub-stepping disabled (max_frac=1e10), matches original single-step output

---

## 2026-03-10 (Tue) — Web Interface Architecture Specification

**Time**: ~PST

### Summary

Created comprehensive architecture and implementation specification for a public web interface to the TOMAS-JAX box model. Document covers full-stack design: FastAPI backend wrapping existing JAX solvers, React frontend with scientific visualizations, deployment strategy, security, and a phased development roadmap.

### Files Created
- `docs/web_interface_spec.md` — Full specification covering:
  - Two-repository architecture (tomas-api + tomas-web)
  - REST API design (4 endpoints: simulate, results, presets, health)
  - Async job queue for long-running simulations (full mode: 5–13s)
  - JIT warmup strategy to avoid 60s cold starts
  - Frontend layout with 4 visualization types (size distribution, banana plot, time series, conservation)
  - 6 curated preset scenarios for researchers
  - Deployment options (Fly.io/Cloud Run + Vercel, ~$8–16/month)
  - Security: rate limiting, input validation, DoS mitigation
  - 4-phase development roadmap (MVP in 2–3 weeks)
  - Risk matrix and complete parameter reference

### Next Steps
- Decide on monorepo vs two-repo approach
- Scaffold backend and frontend repositories
- Implement Phase 1 MVP

---

## 2026-03-10 (Mon) — Constant-Gas Full-Mode Benchmark + Fixed-Production Mode

**Time**: ~15:00 PST

### Summary

Added dedicated nucleation+coagulation+condensation benchmark (`benchmark_nucleation_constgc.py`) comparing JAX PPM (40/80 bins) vs Fortran TFL (36 bins). Discovered and explained mass resolution dependence in constant-gas mode, then implemented fixed-production mode that eliminates it.

### New Files

- **`benchmarks/python/benchmark_nucleation_constgc.py`** — Full-mode benchmark with two gas modes:
  - *Constant-gas* (`--h2so4 <conc>`): H2SO4 reset each step. Good for comparing against Fortran.
  - *Fixed-production* (`--prod-rate <rate>`): H2SO4 produced at constant rate, consumed by condensation. Resolution-independent.
  - Features: 4 figures (dN, dM, timeseries with Gc panel, 3-panel banana plot), Fortran overlay, `--no-organic-nuc` flag
- **Fortran constgc harness updates**: Added nucleation support (do_nucl toggle, nucleation_driver call, hourly Nk snapshots, dt=10s option)
- **Makefile**: constgc target now links NUC_OBJS + nucleation_driver.o
- **convergence_test.py**: Added full-mode scan functions and `--mode full` flag

### Key Finding: Constant-Gas vs Fixed-Production

With **constant gas** (H2SO4 = 1e8, Dunne-only, dt=10s):
- 40-bin M_dry = 11.35 μg/m³, 80-bin M_dry = 14.70 μg/m³ (**30% divergence**)
- Cause: Different bin resolutions → different CS → different mcond per step. With gas held constant, CS differences compound without self-correction.

With **fixed production** (prod = 1e7 molec/cm³/s, Dunne-only, dt=10s):
- 40-bin M_dry = 140.7943 μg/m³, 80-bin M_dry = 140.7947 μg/m³ (**0.0003% match**)
- Why: Higher CS → faster gas depletion → lower equilibrium gas → same total condensation. The CS cancels: `mcond ≈ CS × (prod/CS) × dt = prod × dt`.

### Organic Nucleation Parameters

Investigated how `org_conc` is handled in original models:
- **TOMAS box.f**: Uses user-entered constant nucleation rate (no Riccobono/Dunne)
- **SOM-TOMAS**: Organic concentration computed dynamically from gas-phase chemistry (SOMGC array → ELVOC sum)
- **Our code**: `org_conc` is a subroutine argument — correct design. Hardcoded `1e7` in benchmarks is a test convenience, not model behavior.

### Performance Audit (not yet implemented)

Identified key bottlenecks for future optimization:
1. MNFIX called ~12-15 times per timestep (Phases 1 & 2 could be vectorized)
2. Redundant condensation_sink computation in ezcond (already computed in core)
3. `n_coag_substeps=10` in full mode (Fortran uses 1; could reduce to 3)

### Known Issues

- Fortran constgc harness currently hardcoded at H2SO4 = 1e8 with organic nucleation disabled (temporary for benchmarking)
- 160-bin grid commented out in convergence_test.py (coag jaggedness at fine resolution)

---

## 2026-03-06 (Thu) — Code Review Fixes, PPM Threshold Bug Fix, Branch Merge

**Time**: ~09:00 PST

### Summary

Applied code review fixes across 5 source files, identified and fixed a critical PPM condensation bug where `ezcond_ppm_jax` was falling through to `simple_add_path` instead of the PPM advection algorithm. Cleaned up 10 stale files. Merged `configurable-nbins` into `dev`.

### Bug Fix: PPM Condensation Threshold

**Root cause**: `ezcond_ppm_jax.py` line 131 had `mcond > tot_m * 1e-3` (should be `mcond > 0.0`). For scenarios where total particle mass was large relative to condensed mass, this threshold evaluated False, routing to `simple_add_path` (proportional mass distribution identical to TFL) instead of the actual PPM advection. Result: PPM produced TFL-identical jagged distributions instead of its own smooth, resolution-stable output.

**Fix**: Reverted threshold to `mcond > 0.0` — any positive condensation triggers PPM advection.

### Code Review Changes

1. **`tomas_jax/physics/properties.py`**: Safe division fix (`jnp.maximum(Nk, 1e-20)` before division to prevent NaN in XLA select), removed dead `Kn2` variable, Horner form for denominator
2. **`tomas_jax/core/config.py`**: Vectorized `make_grid()` — geometric progression via `jnp.power` instead of sequential loop
3. **`tomas_jax/solvers/diffrax.py`**: Removed TomasState from ODE state (was polluting PID error norm), replaced `try/except TypeError` tracing trap with explicit `n_substeps: int` parameter
4. **`tomas_jax/solvers/euler.py`**: Empty bin mask fix in adaptive dt — bins with `Nk < 1e-15` or `Mk < 1e-25` get `dt=inf` instead of near-zero dt (prevented death spiral)
5. **`tomas_jax/physics/ezcond_ppm_jax.py`**: PPM threshold fix (see above)

### Cleanup

Deleted 10 orphaned files: `diffrax_original.py`, `tomas_jax/config.py`, `tomas_jax/plotting.py`, `tomas_jax/production_driver.py`, `main.py`, `diagnose_errors.py`, `diagnose_ppm_n_loss.py`, `run_sensitivity_analysis.py`, `run_temp_sensitivity.py`, `run_mode_dp_sensitivity.py`

### Branch Merge

Merged `configurable-nbins` into `dev`. Resolved add/add conflicts in Fortran source files and PROGRESS.md by taking `configurable-nbins` versions.

### Verification

- 28 condensation tests pass (TFL JIT + PPM JIT)
- Convergence benchmark (constant Gc, H2SO4=1e7, N=1e4, GMD=0.02, GSD=1.6): PPM smooth and resolution-stable at 40/80 bins, TFL shows expected bin-doubling artifacts
- PPM ≠ TFL confirmed numerically (PPM N=4.892e9 vs TFL N=4.757e9 at 40 bins combined)

---

## 2026-03-05 (Wed) — Modular Process Orchestrator for condensation.py

**Time**: ~16:00 PST

### Summary

Refactored `tomas_jax/solvers/condensation.py` from 769 lines of duplicated code to ~430 lines using layered core helpers + thin wrappers. Added `make_step()` composable API for easy process reordering.

### Changes

1. **`tomas_jax/solvers/condensation.py`** (primary — 769 → ~430 lines):
   - Added `_condensation_step_core(ezcond_fn)` — single implementation for both PPM and TFL JIT paths
   - Added `_combined_step_core()` — coag + cond parameterized by ezcond_fn
   - Added `_full_step_core()` — nucl + coag + cond parameterized by ezcond_fn
   - Added `_run_scan()` — single scan implementation replacing 6 copy-pasted loops
   - Added `make_step(processes, cond_method)` — public composable API
   - Fixed `condensation_step_with_nucleation_jax` — was running BOTH TFL and PPM then selecting via `jnp.where`; now uses Python-level dispatch (no double compute)
   - All 15+ existing function names preserved as thin wrappers for backward compatibility

2. **`benchmarks/python/run_24h_scenarios.py`**: Changed `use_tfl=jnp.asarray(val)` → `use_tfl=val` (Python float for static dispatch)

3. **`benchmarks/python/time_single_scenario.py`**: Changed `use_tfl=jnp.float64(1.0)` → `use_tfl=1.0`

4. **`run_box_model.py`**: Added `--make-step` flag to demo `make_step()` composable API

### Verification
- All 28 condensation tests pass (TFL JIT + PPM JIT)
- All 24 nucleation tests pass
- `make_step()` import and creation verified

### Known Issues
- `use_tfl` parameter is now Python float (not JAX array) — minor breaking change for callers that pass `jnp.asarray()`. All in-repo callers updated.

---

## 2026-03-05 (Wed) — Add Coag+Cond Combined Mode to Convergence Benchmark

**Time**: ~12:00 PST

### Summary

Extended the convergence benchmark to support combined coagulation+condensation at multiple resolutions (36/72/144 bins). Added `--mode combined` CLI flag to `convergence_test.py` and updated the Fortran harness with a `do_coag` toggle.

### Changes

1. **`benchmarks/python/convergence_test.py`**:
   - Added `--mode {cond_only,combined}` CLI argument (default: cond_only)
   - Imported `run_combined_scan_tfl`, `run_combined_scan_ppm`, `combined_step_tfl_jax`, `combined_step_ppm_jax` from `solvers/condensation.py`
   - Added `_run_constant_gc_combined_scan_tfl()` and `_run_constant_gc_combined_scan_ppm()` for constant-gas mode with coagulation
   - Mode branching in `_run_tfl()` and `_run_ppm()` inner functions
   - Plot titles include process label ("Coag+Cond" vs "Condensation")
   - Fortran overlay labels update to "Fortran Coag+Cond" in combined mode
   - Output filenames get `_combined` suffix in combined mode
   - `load_fortran_results()` accepts `mode` parameter for mode-dependent filenames

2. **`tomas_fortran/harness/benchmark_constgc.f`**:
   - Added `do_coag` logical parameter (set to `.true.` for combined mode)
   - Calls `multicoag(dt)` + `mnfix(Nk, Mk)` before condensation when `do_coag=.true.`
   - Output filenames use `constgc_combined_` prefix when `do_coag=.true.`

### Results

- Combined mode (dt=60s): Coagulation reduces N by ~10% (1.0e9 → 9.0e8) over 24h
- JAX TFL 36-bin vs Fortran 36-bin: N agreement ~0.2%
- PPM converges well across resolutions: N varies <0.1% from 72→144 bins
- Fortran combined: 0.28s, JAX TFL combined: 1.32s, JAX PPM combined: 1.05s (36 bins)
- Backward compatibility: `--mode cond_only` produces identical results to before

### Output

- 7 plots per mode in `benchmarks/results/convergence/` (with `_combined` suffix)
- Fortran output in `tomas_fortran/output/constgc/` (constgc_combined_* and constgc_* files)

---

## 2026-03-03 (Mon) — Configurable NBINS + TFL vs PPM Convergence Test

**Time**: ~15:00 PST

### Summary

Made the number of size bins (NBINS) a runtime parameter instead of a hardcoded constant. Both TFL and PPM condensation now work at arbitrary bin resolutions (36, 72, 144+). Created a multi-resolution convergence benchmark that runs both methods at 3 grid configurations.

### Changes

1. **`make_grid()` function** (`core/config.py`): Creates bin boundaries for arbitrary resolution. Accepts `nbins`, `xk0` (lower boundary), and `doubling_factor`. Refactored existing `xk_boundaries()` to delegate to it.

2. **Removed NBINS hardcoding** (`core/state.py`): Replaced `assert Nk.shape[0] == NBINS` with shape consistency checks (Nk 1D, Mk bins == Nk bins).

3. **Cleaned NBINS imports from 8 physics files**: All JIT-compiled functions now derive bin count from array shapes (`Nk.shape[0]`) instead of importing the config constant. Removed unused NBINS imports from `condensation_sink.py`, `condensation_ppm.py`, `condensation_tfl_jax.py`, `ezcond_ppm_jax.py`, `nucleation.py`.

4. **`condensation_sink.py` xk parameter**: Added explicit `xk` parameter (default None, falls back to `xk_boundaries()`). Updated all 8 call sites to pass `xk=xk` explicitly.

5. **PPM adaptive delta_xi** (`condensation_ppm.py`): Parameterized all PPM internals by `delta_xi = ln(doubling_factor)` instead of hardcoded `ln(2)`. Added `_compute_moment_integrals(a)` for dynamic I0/I1/I2 computation. PPM now works correctly at any bin resolution.

6. **Convergence benchmark** (`benchmarks/python/convergence_test.py`): New script running condensation-only at 3 resolutions (36×2, 72×√2, 144×2^¼) starting from 1.7nm. Generates 4 figures: size distributions, zoomed comparison, integral convergence, timing.

### Key Results

| Config | TFL time | TFL M_dry | PPM time | PPM M_dry |
|--------|----------|-----------|----------|-----------|
| 36 bins (×2) | 0.48s | 6.32e-7 | 0.20s | 6.32e-7 |
| 72 bins (×√2) | 1.14s | 1.06e-6 | 0.59s | 6.31e-7 |
| 144 bins (×2^¼) | 3.72s | 1.09e-6 | 2.07s | 6.29e-7 |

- **PPM M_dry is resolution-stable** (0.5% variation) while TFL M_dry diverges 72% at finer grids (S01)
- **PPM is ~1.8x faster** than TFL at every resolution
- TFL shows oscillation artifacts at 144 bins; PPM stays smooth
- For well-contained distributions (S05), both methods converge identically

### Files Modified
- `tomas_jax/core/config.py` — Added `make_grid()`
- `tomas_jax/core/state.py` — Shape consistency checks
- `tomas_jax/physics/ezcond.py`, `ezcond_ppm.py`, `condensation.py` — `ibins = Nk.shape[0]`
- `tomas_jax/physics/condensation_sink.py` — Added `xk` parameter
- `tomas_jax/physics/condensation_ppm.py` — Adaptive `delta_xi`
- `tomas_jax/physics/condensation_tfl_jax.py`, `ezcond_ppm_jax.py`, `nucleation.py` — Removed unused imports
- `tomas_jax/solvers/condensation.py` — Pass `xk=xk` to condensation_sink calls
- `benchmarks/python/convergence_test.py` — **New**

### Plots
- `benchmarks/results/convergence/convergence_sizedist_S{01,05}.png`
- `benchmarks/results/convergence/convergence_zoomed_S{01,05}.png`
- `benchmarks/results/convergence/convergence_totals_S{01,05}.png`
- `benchmarks/results/convergence/convergence_timing_S{01,05}.png`

---

## 2026-03-03 (Mon) — Solver Cleanup + Clean Timing Benchmark

**Time**: ~22:00 PST

### Summary

Cleaned up solver code accumulated during nucleation debugging: restored degraded Tsit5 parameters, renamed misnamed `coag_rk4_step` → `coag_euler_step`, documented all 6 scan-fused functions, and wrote a clean single-scenario timing benchmark script.

### Changes

1. **Restored `diffrax_step` parameters** (`solvers/diffrax.py`): rtol 1e-3→1e-4, atol 1.0→1e-10, max_steps 500→5000. These were loosened during debugging and never restored.

2. **Renamed `coag_rk4_step` → `coag_euler_step`** (`solvers/diffrax.py`, `solvers/condensation.py`): The function was always forward Euler, not RK4. Added deprecated alias for backward compatibility. Updated all callers and docstrings.

3. **Documented scan-fused functions** (`solvers/condensation.py`): Added table at top of module docstring listing all 6 scan functions with their coag/cond/nucl capabilities.

4. **New timing benchmark** (`benchmarks/python/time_single_scenario.py`): Clean script that times all 8 solver combinations for a single scenario with JIT warmup, median of 3 runs, and Fortran comparison table.

### Fresh Timing Results (S01, 24h, M1 Pro)

| Test | JAX (s) | Fortran (s) | Ratio |
|------|---------|-------------|-------|
| Tsit5 coag-only | 4.144 | 0.256 | 16.2x |
| Euler coag-only | 0.346 | 0.256 | 1.35x |
| PPM cond-only | 0.213 | 0.086 | 2.47x |
| TFL cond-only | 0.488 | 0.086 | 5.66x |
| Euler+PPM combined | 0.579 | 0.328 | 1.76x |
| Euler+TFL combined | 0.828 | 0.328 | 2.52x |
| Nucl+TFL cond | 0.701 | — | — |
| Full (nucl+coag+cond) | 1.673 | — | — |

### Key Observations

- Tsit5 with restored tolerances (atol=1e-10) is 16x slower than Fortran — the loose atol=1.0 was what made it fast before
- Euler coag is only 1.35x slower than Fortran — competitive
- PPM cond is the fastest JAX condensation path (2.5x Fortran)
- Euler+PPM combined is the best full JAX path (1.76x Fortran)

### Files Modified
- `tomas_jax/solvers/diffrax.py` — parameters + rename
- `tomas_jax/solvers/condensation.py` — updated imports/calls + docstring table
- `benchmarks/python/time_single_scenario.py` — new timing script

---

## 2026-03-03 (Mon) — Full-Mode Stability Fixes + Benchmark Plots

**Time**: ~14:00 PST

### Summary

Fixed critical stability issues in full-mode (nucleation+coagulation+condensation) that caused NaN divergence in 13/50 scenarios. Root cause: unclamped organic mass in nucleation when H2SO4 gas is exhausted. Also replaced RK4 coagulation solver with forward Euler (matches Fortran approach). Generated comprehensive 8-figure benchmark comparison for all 5 modes.

### Bugs Fixed

1. **Nucleation organic mass clamping** (`nucleation.py`): When H2SO4 gas supply is exhausted (`need_clamp=True`), number and SO4 mass were clamped but organic mass was left at the full nucleation rate. This created particles with avg mass 80,000x above bin 0 boundary (e.g., mass=3.08e-19 in bin 0 where xk[0]=1.6e-23). The misalignment caused catastrophic cascading rates in TFL coagulation. Fix: `dM_org_clamped = dN_clamped * 0.1 * _MNUC` — clamp organic proportionally.

2. **RK4 intermediate-stage amplification** (`diffrax.py`): RK4 intermediate stages (k2/k3/k4) amplify N^2 coagulation rates through positive feedback when particle counts are high (cold scenarios with strong nucleation). Replaced with forward Euler (single RHS evaluation per substep + positivity clipping + MNFIX). This matches Fortran's explicit approach and is more stable.

3. **Fortran CSV missing-E exponent** (`utils.py`): Fortran writes very small numbers like `0.1024186408109169-238` (missing 'E' before exponent). Added `_fix_fortran_float()` regex to insert 'E'.

### Results (All 5 modes, 50 scenarios, TFL vs Fortran at hour 24)

| Mode | N Scenarios | N<1% | N<5% | N Med Err | N Max Err | M Med Err | M Max Err |
|------|-------------|------|------|-----------|-----------|-----------|-----------|
| Coag Only | 42 | 39 | 42 | 1.37e-3 | 2.59e-2 | 3.15e-7 | 4.82e-3 |
| Cond Only | 42 | 42 | 42 | 9.97e-7 | 1.01e-3 | 2.64e-6 | 1.00e+0 |
| Coag+Cond | 41 | 34 | 40 | 1.30e-3 | 7.74e-2 | 3.47e-6 | 4.82e-3 |
| Nucl+Cond | 50 | 33 | 41 | 2.14e-3 | 2.48e-1 | 1.22e-5 | 4.33e-3 |
| Full | 50 | 16 | 24 | 8.06e-2 | 9.95e-1 | 8.21e-4 | 1.00e+0 |

Note: Coag/Cond/Combined have 42/41 scenarios because Fortran S50 crashes (known ezcond STOP). Full mode has higher errors due to organic clamping divergence from Fortran (intentional fix for JAX stability).

### Files Modified
- `tomas_jax/physics/nucleation.py` — Added organic mass clamping in gas-limited path
- `tomas_jax/solvers/diffrax.py` — Replaced RK4 with forward Euler coagulation solver
- `tomas_jax/solvers/condensation.py` — n_substeps=10, removed dual TFL/PPM computation in full_step_jax
- `benchmarks/python/utils.py` — Fixed Fortran CSV parser for missing-E exponents

### Files Created
- `benchmarks/python/plot_all_modes.py` — 8-figure comprehensive benchmark comparison
- `benchmarks/results/nucleation_benchmark/` — 8 PNG figures (timeseries, error evolution, scatter, heatmap, summary)

---

## 2026-03-02 (Sun) — Nucleation Benchmark: Fortran vs JAX Verification

**Time**: ~21:00 PST

### Summary

Two-phase nucleation benchmark comparing JAX vs Fortran for Riccobono 2014 and Dunne 2016 parameterizations.

### Phase 1: Parameterization-Level (Option A)

20 test cases (10 Riccobono + 10 Dunne) with varying T, H2SO4, org, NH3, fion. All match Fortran to machine precision (max relative error < 1e-12).

**Files created:**
- `tomas_fortran/src/ricco_nucl.f`, `tomas_fortran/src/dunne_inorg_nucl.f` — copied from TRACER_SOM-TOMAS (prints stripped)
- `tomas_fortran/harness/benchmark_nucleation.f` — standalone Fortran benchmark
- `benchmarks/python/compare_nucleation.py` — JAX vs Fortran comparison

### Phase 2: Full 24h Driver (Option B)

50 scenarios × nucl_cond and full modes, comparing hourly output over 24 hours.

**Key bugs found and fixed:**
1. **Gas depletion MW correction**: JAX had `gas_depleted = dM_so4 * (98/96)`, Fortran depletes Gc by SO4 mass directly (no correction). Fixed.
2. **Organic mass clamping**: When H2SO4 gas is exhausted, Fortran does NOT revert the organic mass to bin 0. JAX was reverting it. Fixed to match Fortran.
3. **MNFIX multi-bin shift**: JAX MNFIX only handled 1-2 bin shifts. Nucleated particles with large organic mass need to jump 12+ bins. Fixed with analytical log2 computation: `kk = ceil(log2(avg*1.1/xk[0])) - 1`.

**Results (nucl_cond mode, 50 scenarios):**
- N_tot median error: 0.2%, max 25% (33/50 < 1%, 41/50 < 5%)
- M_tot max error: 0.43%, median 1.2e-5
- Residual N errors from MNFIX partial-transfer differences compounding over 1440 steps

**Files created:**
- `tomas_fortran/src/nucleation_driver.f` — 24h nucleation driver subroutine
- Fortran output: `tomas_fortran/output/24h/` (50 scenarios × 5 modes × 24 hours)

**Files modified:**
- `tomas_jax/physics/nucleation.py` — Fixed gas depletion and organic clamping
- `tomas_jax/core/mnfix_jax.py` — Fixed multi-bin shift with analytical computation
- `tomas_jax/solvers/condensation.py` — Added `full_step_jax`, `run_full_scan` (scan-fused nucleation+coagulation+condensation)
- `tomas_fortran/harness/benchmark_24h.f` — Added modes 4 (nucl_cond) and 5 (full)
- `tomas_fortran/Makefile` — Added nucleation objects
- `benchmarks/python/run_24h_scenarios.py` — Added nucl_cond and full modes with scan-fused paths, hourly snapshots
- `benchmarks/python/compare_24h.py` — Added nucl_cond and full modes, fixed FORTRAN_DIR path
- `tests/test_nucleation.py` — Updated mass balance test

### Performance
- JAX nucl_cond (scan-fused): ~0.5s/scenario (24×60-step scans)
- JAX full (scan-fused with diffrax): ~5-13s/scenario (adaptive ODE stepping)
- Fortran combined: ~0.33s/scenario

---

## 2026-03-02 (Sun) — Add Nucleation (Riccobono 2014 + Dunne 2016)

**Time**: ~22:00 PST

### Summary

Implemented nucleation parameterizations ported from TRACER_SOM-TOMAS Fortran. Two schemes: Riccobono 2014 organic nucleation (with Yu 2017 temperature correction) and Dunne 2016 inorganic nucleation (4 mechanisms: binary/ternary x neutral/ion-induced). All functions are pure-JAX and JIT-compilable.

### Files Created
- `tomas_jax/physics/nucleation.py` — 3 functions: `ricco_nucleation_rate`, `dunne_nucleation_rate`, `nucleation_step`
- `tests/test_nucleation.py` — 24 unit tests (all passing)
- `docs/nucleation.md` — Algorithm documentation

### Files Modified
- `tomas_jax/solvers/condensation.py` — Added `condensation_step_with_nucleation_jax`, `run_nucleation_condensation_scan`
- `run_box_model.py` — Nucleation in time loop (after H2SO4 production, before coagulation), `--no-nucleation` CLI flag
- `CLAUDE.md` — Updated file layout, operator-split order, Fortran source mapping
- `PROGRESS.md` — This entry

### Key Details
- Nucleation cluster: r=0.85nm, rho=1350 kg/m3, composition 90% SO4 + 10% organic
- Cluster mass (~3.47e-24 kg) < XK0 (1.6e-23 kg), always placed in bin 0
- Gas depletion: SO4 mass subtracted directly from Gc[SRTSO4] (no 98/96 MW correction), with clamping
- In clamped path (gas exhausted): SO4 mass = Gc * 96/98, organic mass NOT reverted (matches Fortran)
- Enable/disable flags use multiplicative float masks (0.0/1.0) to avoid JIT recompilation
- Operator splitting order: H2SO4 production -> nucleation -> coagulation -> condensation

### Verification
- Riccobono at T=278K, h2so4=1e7, org=1e7: J = 3.27 cm^-3 s^-1 (exact)
- Dunne with nh3=0, fion=0: only Jbn > 0 (correct)
- Mass balance: aerosol SO4 gained * 98/96 = gas H2SO4 lost (within 1e-6 relative)
- All 24 tests passing, JIT compilation verified

### Known Limitations
- Coagulation-sink survival fraction (Kerminen-Kulmala) not implemented (commented out in Fortran too)
- Organic vapor (org_conc) is an external input, not coupled to the gas-phase chemistry
- No nuc_bin search — always bin 0 (valid since mnuc < XK0)

### Next Steps
- Run 24h benchmark with nucleation enabled
- Compare nucleation event banana plots with observations
- Consider coupling organic vapor to gas-phase chemistry

---

## 2026-03-02 (Sun) — Fortran TFL vs PPM 24h Benchmark Comparison

**Time**: ~14:30 PST

### Summary

Ran 49 scenarios (S01-S49) x 24 hours x 2 modes (cond_only, combined) for both Fortran TFL and Fortran PPM, then generated 8 comparison plots and a detailed summary.

### Key Results

- **22 of 49 scenarios** exercise PPM transport (Path 1 where mcond > 1e-3 * total_dry_mass)
- **Cond-only (active only)**: N_tot error median=1.5e-6, max=1.2e-3; M_dry error median=5.1e-6, max=1.9e-4
- **Combined (active only)**: N_tot error median=6.0e-3, max=4.9e-2; M_dry error median=5.2e-6, max=1.7e-4
- **N conservation**: TFL perfect (1.0000); PPM median=0.9999, worst=0.9996 (cond-only)
- **Per-bin differences**: Up to 100% in individual bins — expected algorithmic difference (PPM is less diffusive, produces sharper distributions; TFL smears the growth front across bins)
- **Timing**: PPM ~0.93x TFL for cond-only (69ms vs 74ms); equal for combined (~0.32s, dominated by coagulation)
- **Scenario 50**: Crashes both methods (known extreme condition issue)

### Files Created
- `benchmarks/python/compare_fortran_tfl_ppm.py` — Comparison script (8 plots + summary)
- `benchmarks/results/fortran_tfl_vs_ppm/` — 8 PNG figures + summary.txt

### Next Steps
- Investigate PPM N conservation loss (worst case 0.04% at 24h) — likely from positivity clamping
- Consider running with `-O2` for more realistic timing comparison
- Compare Fortran PPM against JAX PPM to verify cross-language consistency

---

## 2026-03-02 (Sun) — Port PPM Condensation to Fortran

**Time**: ~14:00 PST

### Summary

Ported the PPM (Piecewise Parabolic Method) condensation algorithm from JAX/Python (`tomas_jax/physics/condensation_ppm.py`) to Fortran. The PPM Fortran code lives in a separate `tomas_fortran/src_ppm/` directory so both TFL and PPM methods coexist and can be benchmarked head-to-head in pure Fortran.

### Files Created
- `tomas_fortran/src_ppm/tmcond_ppm.f` — 8 PPM subroutines (~590 lines):
  - `PPM_RECONSTRUCT` — 4th-order interface interpolation + Colella-Woodward limiting + positivity
  - `PPM_EDGE_VELOCITY` — Upwind velocity at bin edges via DMDT_INT
  - `PPM_NUMBER_FLUX` — Departure-point parabola integration for number flux
  - `PPM_MASS_FLUX` — Analytical mass-weighted flux (∫m(η)n(η)dη with exp antiderivatives)
  - `PPM_DRY_MASS_ANALYTICAL` — Exact dry mass from PPM coefficients (moment integrals I0, I1, I2)
  - `PPM_SPECIES_FLUX` — Upwind donor ratio × dry mass flux for all 44 species
  - `PPM_COMPUTE_SUBSTEPS` — CFL-limited substep count (C_max = 0.8)
  - `PPM_CONDENSATION_STEP` — Main orchestrator (freeze WR, substep loop, call all above)
- `tomas_fortran/src_ppm/ezcond_ppm.f` — PPM-aware ezcond driver (~170 lines):
  - Same 3-path decision tree as ezcond.f (CS check, significance thresholds)
  - Path 1 calls PPM_CONDENSATION_STEP with dt=1.0, then adds condensed mass via sinkfrac
  - Mass conservation check and correction (same as ezcond.f)
- `tomas_fortran/harness/benchmark_24h_ppm.f` — PPM benchmark harness (~230 lines):
  - Same as benchmark_24h.f but calls ezcond_ppm instead of ezcond
  - Output files prefixed with `ppm_` (e.g., `ppm_s01_cond_hour01_Nk.csv`)
  - Timing CSV: `output/24h/timing_fortran_ppm.csv`

### Files Modified
- `tomas_fortran/Makefile` — Added `SRC_PPM`, `PPM_OBJS`, compile rules, `benchmark_24h_ppm` target, `benchmark_both` convenience target
- `tomas_fortran/README.md` — Documented src_ppm/ and new build targets
- `CLAUDE.md` — Added src_ppm/ files to file layout, updated Fortran source mapping

### Verification
- `make benchmark_24h_ppm` compiles cleanly (no warnings)
- `make benchmark_24h` still compiles (TFL unchanged)
- Both executables produced: `benchmark_24h_ppm.exe` (121KB), `benchmark_24h.exe` (103KB)

### Next Steps
- Run single scenario cond-only and compare against JAX PPM JIT output
- Run full 50-scenario benchmark for timing comparison
- Add Python comparison script for TFL vs PPM Fortran output

---

## 2026-03-02 (Sun) — Bring TOMAS Fortran into tomas_fortran/

**Time**: ~22:00 PST

### Summary

Made the repository self-contained by copying the original TOMAS Fortran source into `tomas_fortran/` with a proper build system. Previously, the Fortran source was referenced via fragile relative paths to `../../original-models/TOMAS/backup/src/`.

### Files Created
- `tomas_fortran/src/` — 14 core Fortran source files (multicoag.f, initbounds.f, mnfix.f, aerodens.f, loginit.f, ezcond.f, tmcond.f, dmdt_int.f, getCondSink.f, gasdiff.f, eznh3eqm.f, ezwatereqm.f, waterso4.f, waternacl.f)
- `tomas_fortran/include/sizecode.COM` — Common block definitions
- `tomas_fortran/harness/benchmark_24h.f` — 24h benchmark driver (copied from benchmarks/fortran/)
- `tomas_fortran/harness/benchmark_harness.f` — Single-scenario driver (copied from benchmarks/fortran/)
- `tomas_fortran/Makefile` — Self-contained build system (targets: all, benchmark_24h, run_24h, clean)
- `tomas_fortran/README.md` — Build instructions and file descriptions

### Files Modified
- `benchmarks/fortran/Makefile` — Updated TOMAS_SRC and TOMAS_INC to point to `../../tomas_fortran/src/` and `../../tomas_fortran/include/`
- `CLAUDE.md` — Added tomas_fortran/ to file layout, updated Fortran source path reference

### Next Steps
- Remove symlink dependency from benchmarks/fortran/ (sizecode.COM still symlinked for its local compile)
- Consider adding `.gitignore` for `tomas_fortran/output/` and `*.o` files

---

## 2026-03-02 (Sun) — GMD Paper Draft Populated

**Time**: ~18:00 PST

### Summary

Populated the full GMD-style paper draft at `docs/paper/draft.md` (~5000 words) with:
- Complete prose for all 8 sections + 4 appendices
- LaTeX-style equations extracted from source code (dmdt_int, PPM reconstruction, analytical mass-weighted flux, Fuchs-Sutugin, condensation sink, MNFIX partial transfer)
- 6 validation/timing tables with exact numbers from benchmark summary
- 9 planned figure captions and 6 table references
- 30+ references in GMD bibliography style
- Cross-verified all formulas against source code and all benchmark numbers against `summary.txt` — no inconsistencies found

### Files Modified
- `docs/paper/draft.md` — complete rewrite from outline to full paper

### Next Steps
- Generate actual figures from benchmark data
- Convert to LaTeX for GMD submission
- Add nucleation and organic condensation sections when implemented

---

## 2026-03-02 (Sun) — PPM Analytical Mass-Weighted Flux Fix + Vectorized Species Transport

**Time**: ~11:00 PST

### Problem

PPM condensation had two critical issues:
1. **81% N loss over 24h** — The `passive_flux` function used `F_M = F_N * r_avg_donor`, assigning average mass to all departing particles. But PPM selectively removes particles from the bin edge (where they're 2x heavier), causing systematic mass underestimate → M/N mismatch → MNFIX clips → N loss death spiral.
2. **~150x slower than Fortran** — Species transport used `jax.lax.fori_loop` over 44 species, creating convoluted XLA control flow instead of matrix math.

### Fix: Analytical Mass-Weighted Integrals

Replaced naive `F_M = F_N * r_avg` with exact analytical integral of `m(η)*n(η)` over the departure region:
- `m(η) = m_L * exp(a*η)` is the dry mass at position η within a bin (mass-doubling: 2x variation)
- `n(η) = n_L + b*η - n_6*η²` is the PPM number density parabola
- Antiderivatives: `∫ η^k * exp(aη) dη` for k=0,1,2

New functions in `condensation_ppm.py`:
- `_mass_antideriv(eta)` — evaluates antiderivatives of η^k * exp(aη)
- `_integrate_mass_parabola_right/left()` — mass integrals over departure regions
- `ppm_mass_flux()` — computes F_M_dry at each edge using mass-weighted integrals
- `species_flux()` — vectorized, all 44 species in one matrix multiply (no loops)

Key normalization insight: `species_flux` uses `M_dry_analytical` from `dry_mass_from_ppm_number()` (not tracked mass) to ensure exact conservation when C=1.

### Results (49 scenarios, 24h)

| Mode | N_tot error (median) | Mass conservation | JAX wall time | JAX/Fortran |
|------|---------------------|-------------------|---------------|-------------|
| Cond PPM_JIT | 2.08e-7 | 8.6e-16 | 0.26s | 3.4x |
| Cond TFL_JIT | 2.08e-7 | 5.9e-15 | 0.47s | 6.2x |
| Combined PPM_JIT | 2.18e-3 | 1.9e-10 | 0.41s | 1.3x |
| Coag-Only | 1.83e-3 | 1.9e-10 | 0.15s | 0.57x |

PPM_JIT is **1.8x faster than TFL_JIT** for condensation, with machine-precision mass conservation and perfect N conservation.

### Files Modified

- `tomas_jax/physics/condensation_ppm.py` — Added analytical mass-weighted flux functions, replaced `passive_flux` with vectorized `species_flux`, removed `closure_update`, eliminated `fori_loop` over species
- `tests/test_ppm_condensation.py` — Updated imports and tests for new API (removed `passive_flux`/`closure_update`, added `species_flux`/`ppm_mass_flux` tests)
- `benchmarks/python/run_ppm_analytical_benchmark.py` — **NEW**: 49 scenarios x 5 modes benchmark

### Output

- `benchmarks/results/2026-03-02-ppm-analytical/` — 9 figures + summary.txt

---

## 2026-03-01 (Sat) — Fix TFL Condensation Oscillation (Root Cause: MNFIX + Condensation Sink)

**Time**: ~00:30 PST

### Root Cause

TFL condensation produced oscillating/pulsing size distributions (bins emptying completely then all particles jumping to the next bin). Two bugs were identified:

1. **MNFIX was doing "move-all" instead of Fortran's "partial-transfer"** (THE ROOT CAUSE)
   - Our MNFIX: when avg mass > upper bin boundary → move ALL particles to next bin → source bin empties → pulse
   - Fortran MNFIX: split the bin — some particles stay at geometric mean mass, excess moves to neighbor → smooth gradual redistribution
   - Fortran uses `nshift = (drymass - xold*number) / (xnew - xold)` for the partial transfer, keeping remaining particles at `xold = sqrt(xk[k]*xk[k+1])` and shifting `nshift` particles at `xnew = xk[kk+1]/1.1`

2. **Condensation sink Neps threshold was wrong** (SECONDARY)
   - Fortran getCondSink.f: `Neps = 1e10` — bins with < 1e10 particles use default `density=1500`, `mp=1.4*xk[k]`
   - Our code: `Neps = 1e-20` — computed actual density/mp for ALL bins, giving different `sinkfrac` distributions

3. **Mass conservation correction threshold was too permissive** (MINOR)
   - Fortran ezcond.f: `abs(1-ratio) < 1.0` → ratio in (0, 2)
   - Our ezcond.py: `abs(ratio) < 100.0` → allowed extreme amplification

### Files Modified

- `tomas_jax/core/mnfix_jax.py` — **Complete rewrite** to match Fortran mnfix.f:
  - Phase 1: Fix empty bins (Neps=1e-5, matching Fortran)
  - Phase 2: Fix extreme out-of-range (avg > max grid or avg < min grid)
  - Phase 3: Partial transfer using Fortran's split algorithm (nshift formula)
  - Uses `jax.lax.fori_loop` for sequential processing matching Fortran order
  - Multi-bin drift handled for 1-2 bin shifts (common case)

- `tomas_jax/physics/condensation_sink.py` — Match Fortran getCondSink.f Neps=1e10:
  - Bins with Nk > 1e10: compute actual density and mp from composition
  - Bins with Nk <= 1e10: use default density=1500, mp=1.4*xk[k]
  - Now directly computes Dpk instead of calling calc_particle_properties

- `tomas_jax/physics/ezcond.py` — Fixed mass correction threshold:
  - Changed from `abs(ratio) < 100.0` to `ratio > 0.0 and ratio < 2.0` (matching Fortran)

- `tomas_jax/core/config.py` — Added `xk_boundaries()` utility function and `XK0` constant

### Results (1-hour, 60 steps)

| Scenario | Before Fix | After Fix |
|----------|-----------|-----------|
| S10 (peak bins) | 2-7% error | 0.02-12% error |
| S20 (all bins) | **100% error (empty bins!)** | **0.3-1.9% error** |
| S39 (peak bins) | **93% error** | **0.2-2.4% error** |

### Tests

All 73 non-cached tests pass (15 TFL JIT + 43 PPM + 13 PPM JIT + 2 coagulation).

### Known Remaining Issues

- S39 lower-tail bins (6-8) still have 14-39% error — these have very few particles and are affected by the Neps threshold boundary
- PPM still has fundamental N conservation issues (separate from MNFIX) due to passive_flux using average donor mass instead of position-dependent mass
- The 24h benchmark NPZ files need regeneration with the fixed code

---

## 2026-02-28 (Fri) — JIT-Compile TFL Condensation (Fortran-Matching, 43x Speedup)

**Time**: ~00:00 PST

### Summary
Created a pure-JAX, JIT-compilable version of the TFL (Tzivion-Feingold-Levin) condensation algorithm. The sequential TFL matches Fortran output well (unlike PPM which produces overly narrow distributions), so JIT-compiling TFL gives us both accuracy and speed.

### Approach
- Vectorized top-hat construction and dmdt_int translation across all bins
- `jax.lax.fori_loop` over 36 source bins with vectorized overlap-fraction scatter inside
- Condensing species uses inverse dmdt_int at bin boundaries for trapezoidal interpolation, matching Fortran's 1.5*YM for full middle bins
- Three-way branch via `jax.lax.cond` (tmcond / simple-add / no-op)
- `jax.lax.scan` fuses the 1440-step time loop into a single XLA program

### Files Created
- `tomas_jax/physics/condensation_tfl_jax.py` — Pure-JAX TFL: `tmcond_jax()` (vectorized bin remapping) + `ezcond_tfl_jax()` (condensation driver). ~370 lines.
- `tests/test_tfl_jit_condensation.py` — 15 tests: tmcond zero-tau/shift/JIT, ezcond zero-mcond/JIT, full-step no-nan/gas-depletion/mass-conservation/JIT/1-step-match/60-step-match, scan runs/scan-vs-loop/N-conservation, dispatcher. All passing.

### Files Modified
- `tomas_jax/solvers/condensation.py` — Added `condensation_step_tfl_jax()`, `condensation_step_tfl_jit`, `run_condensation_scan_tfl()`. New `method='tfl_jit'` dispatcher.
- `run_box_model.py` — Added `--method tfl_jit` option.
- `benchmarks/python/run_24h_scenarios.py` — Added `tfl_jit` method with JIT warmup and scan-fused fast path.
- `benchmarks/python/compare_methods.py` — Added TFL_JIT to 5-way comparison (Fortran/TFL/TFL_JIT/PPM/PPM_JIT).
- `tomas_jax/solvers/__init__.py` — Updated docstring.

### Performance Results (S01 cond-only, 24h)
| Method | Wall Time | vs Fortran | vs Sequential TFL |
|--------|-----------|------------|-------------------|
| Fortran | 0.091s | 1.0x | — |
| TFL (sequential) | 13.942s | 153x slower | 1.0x |
| **TFL JIT** | **0.323s** | **3.5x slower** | **43x faster** |
| PPM (sequential) | 131.327s | 1443x slower | — |
| PPM JIT | 0.321s | 3.5x slower | — |

### Validation
- TFL JIT matches sequential TFL exactly after 1 step (N_tot rel_diff=0.00e+00, M_dry rel_diff=4.79e-16)
- TFL JIT matches sequential TFL exactly after 60 steps (N_tot rel_diff=0.00e+00, M_dry rel_diff=3.02e-16, all significant bins identical)
- TFL JIT matches sequential TFL after 6 hours in full comparison (N_ratio=0.999846 for both)
- Mass conservation error < 1e-4 relative at hour 6
- All 15 TFL JIT tests pass, all 43 PPM tests pass, all 13 PPM JIT tests pass

### PPM Accuracy Issue Identified
PPM produces overly narrow size distributions (1-bin spike) that don't match Fortran. TFL matches Fortran well. This is a pre-existing algorithmic issue with PPM (not a JIT bug — PPM JIT matches PPM exactly). The TFL JIT approach bypasses this entirely.

### Known Limitations
- TFL JIT is 3.5x slower than Fortran (Fortran uses hand-optimized sequential loops; JAX pays XLA dispatch + fori_loop overhead)
- JIT compilation takes ~30-60s on first call (subsequent calls reuse cached XLA)
- PPM accuracy issue remains open (not addressed in this change)

---

## 2026-02-27 (Thu) — JIT-Compile PPM Condensation Pipeline

**Time**: ~22:00 PST

### Summary
Rewrote the PPM condensation wrappers as pure JAX to enable full JIT compilation of the condensation pipeline. The PPM algorithm itself was already JIT-compiled, but it was wrapped in numpy code with 36+ array conversions and Python for-loops per call, executed 1440 times per 24h run.

### Files Created
- `tomas_jax/physics/ezcond_ppm_jax.py` — Pure-JAX ezcond replacement (~120 lines). Vectorized TAU computation, `jax.lax.cond` for three-way branch (PPM/simple-add/no-op), no internal MNFIX.

### Files Modified
- `tomas_jax/solvers/condensation.py` — Added `condensation_step_jax()` (pure-JAX single step), `condensation_step_jit` (pre-compiled JIT wrapper), `run_condensation_scan()` (scan-fused 1440-step time loop). New `method='ppm_jit'` dispatcher.
- `run_box_model.py` — Added `--method ppm_jit` option.
- `benchmarks/python/run_24h_scenarios.py` — Added `ppm_jit` method with JIT warmup and scan-fused fast path for cond-only mode.
- `tomas_jax/physics/__init__.py`, `tomas_jax/solvers/__init__.py` — Updated docstrings.

### Files Created (Tests)
- `tests/test_ppm_jit_condensation.py` — 13 tests: ezcond_ppm_jax equivalence, JIT compilation, mass/N conservation, scan vs loop equivalence, dispatcher routing. All passing.

### Performance Results
| Config | Old PPM | JIT PPM | Fortran |
|--------|---------|---------|---------|
| Per-step | 21.7ms | 0.1ms | — |
| 24h cond-only | ~12s | 0.29s | 0.08s |
| Speedup | — | ~40x | — |

### Existing Tests
- All 43 existing PPM tests still pass
- All 13 new JIT tests pass
- 89 pre-existing 24h scenario test failures unchanged

---

## 2026-02-27 (Thu) — Fix PPM Number Conservation: Replace Closure with Direct Transport

**Time**: ~17:00 PST

### Root Cause

The PPM closure step (`M_cs = M_dry_integral - M_noncond`) systematically lost ~2% of SO4 mass per step because the PPM integral uses a reconstructed parabola whose effective bin-average mass differs from the discrete geometric mean. Over 1440 steps, this destroyed mass and caused the mass conservation correction to produce absurd ratios (e.g., -192,379), which were then skipped, leaving MNFIX to aggressively clip bins and destroy N.

### Fix: Transport All Species + Add Mass Separately

1. **`condensation_ppm.py`**: Modified `ppm_condensation_step` to transport ALL species (including condensing species) using upwind flux. Removed the closure step entirely (no more `dry_mass_from_ppm_number` + `closure_update` inside substeps).

2. **`ezcond_ppm.py`**: After PPM transport, add condensed mass proportionally to sinkfrac (`Mk2[k, spec] += mcond * sinkfrac[k] / totsinkfrac`). Removed the broken ratio-based mass conservation correction. Reduced MNFIX calls from 4+ to 2 (input + final).

### Why It Works
- Upwind transport preserves total mass (flux telescoping: `sum(F[k+1]-F[k]) = 0`)
- Adding `mcond * sinkfrac` adds exactly `mcond`
- N-M consistency: same `F_N` used for both N and M transport, so `xbar = M/N` stays in bounds, no MNFIX clipping needed

### Results
- **N loss per step**: 0.0000e+00 (was ~2% before)
- **Projected 24h N loss**: 0.0% (was 33% before)
- **Mass conservation**: exact (mcond gained = mcond added)
- **All 43 PPM tests pass** (one test updated to reflect new design)

### Files Modified
- `tomas_jax/physics/condensation_ppm.py` — Transport all species, remove closure
- `tomas_jax/physics/ezcond_ppm.py` — Add mass after PPM, remove broken correction, reduce MNFIX
- `tests/test_ppm_condensation.py` — Updated `test_cspecies_increases_with_growth` → `test_cspecies_conserved_by_transport`
- `diagnose_ppm_n_loss.py` — Updated for new code path

---

## 2026-02-27 (Thu) — 24h Benchmark: Bug Fixes, Fortran Timing, Full 50-Scenario Run

**Time**: ~09:00–15:00 PST

### Bug Fixes

1. **NaN propagation in condensation sink** (`tomas_jax/physics/condensation_sink.py`)
   - **Root cause**: Empty bins have `Dpk = 0` → `Kn = 2*mfp/0 = inf` → `beta = NaN` → `CS = NaN` → gas-phase mass dumped to bin 0 → mass/number blowup
   - **Fix**: Added `safe_beta = jnp.where(Dpk > 0.0, beta, 0.0)` at line 71. Bins with no diameter contribute zero to condensation sink.
   - **Impact**: S01 PPM N_tot went from 746,000x blowup to 0.66 ratio at 24h.

2. **MNFIX empty-bin detection** (`tomas_jax/core/mnfix_jax.py`)
   - **Root cause**: `mask_empty` only checked `Nk_new < TINY_N`, missing bins with number but zero dry mass (created by PPM mass conservation rescaling zeroing out mass)
   - **Fix**: Changed to `mask_empty = (Nk_new < TINY_N) | ((Nk_new >= TINY_N) & (drymass_new < NEPS))` at line 129
   - **Impact**: Prevents NaN propagation from bins with number but no mass

### Fortran Timing

- **Modified**: `benchmarks/fortran/benchmark_24h.f` — Changed timing CSV write from end-of-program to incremental after each scenario (survives crashes)
- **Result**: Fortran timing captured for 49/50 scenarios (S50 crashes in ezcond due to mass conservation STOP)
- **Average times**: coag 0.27s, cond 0.08s, combined 0.33s per 24h scenario

### 50-Scenario JAX Run

- Deleted all old NPZ files and re-ran with both bug fixes applied
- Run covers 50 scenarios × 3 modes × 2 methods (TFL + PPM) = 250 NPZ files
- Small-GMD scenarios (S45 GMD=0.018um, S46 GMD=0.012um) are very slow for PPM due to CFL substep overhead

### Plots Generated (with 41 scenarios)

- All 8 summary figures + 1 timing bar chart generated
- Key findings:
  - TFL mass conservation: excellent (1e-15 to 1e-8 relative error)
  - PPM mass conservation: problematic (10-100% error for many scenarios)
  - TFL vs Fortran N_tot: ~1e-3 median relative error
  - PPM vs Fortran N_tot: ~1e-2 median relative error

### Files Modified
- `tomas_jax/physics/condensation_sink.py` — NaN guard on empty bins
- `tomas_jax/core/mnfix_jax.py` — Empty-bin detection includes zero-dry-mass case
- `benchmarks/fortran/benchmark_24h.f` — Incremental timing CSV write
- `benchmarks/python/compare_24h.py` — Added M_dry keys to save_comparison_npz
- `benchmarks/python/plot_24h_summary.py` — Figure 2 styling (distinct line styles for Fortran/TFL/PPM)

### Known Issues
- **PPM N conservation**: 33% drift for small-GMD scenarios. Eulerian advection + MNFIX clipping breaks 0th moment conservation. Needs moment-preserving scheme (Moving Center or similar).
- **PPM mass conservation**: Many scenarios show >1% dry mass error. Related to N drift issue.
- **JAX speed**: TFL is ~30-100x slower than Fortran (sequential Python loops). PPM varies widely (fast for large GMD, very slow for small GMD due to CFL substeps).
- **Fortran S50**: ezcond calls STOP on mass conservation error, killing the benchmark.
- **Bin-0 mass dump**: When CS is too small, condensation dumps mass to bin 0 — this is nucleation, not condensation. Should be removed before nucleation is implemented.

### Next Steps
- Complete 50-scenario run and regenerate all plots
- Generate timing comparison (Fortran vs JAX-TFL vs JAX-PPM)
- Fix PPM N conservation (Moving Center method or moment-preserving advection)
- Remove bin-0 mass dump from condensation pathway

---

## 2026-02-26 (Wed) — 24h Benchmark Suite: Implementation

**Time**: ~10:00–22:00 PST

### New Files Created

1. **Scenario Generation** (`benchmarks/python/scenarios.py`)
   - 50 Latin Hypercube scenarios via pyDOE2
   - Parameters: N_total, GMD, GSD, temp, pres, RH, Gc_SO4, H2SO4_prod
   - Seed=42 for reproducibility
   - Writes `benchmarks/fortran/scenarios.csv`

2. **JAX 24h Runner** (`benchmarks/python/run_24h_scenarios.py`)
   - Runs all 50 scenarios × 3 modes × 2 methods (TFL + PPM)
   - Saves hourly snapshots to NPZ (Nk, Mk, Gc, N_tot, M_tot, M_dry)
   - Records wall-clock timing per run
   - Writes `benchmarks/results/24h/timing_jax.csv`

3. **3-Way Comparison Engine** (`benchmarks/python/compare_24h.py`)
   - Loads Fortran CSV + JAX NPZ results
   - Computes per-hour relative errors for N_tot, M_tot, M_dry
   - Saves comparison NPZ files per mode

4. **Summary Plot Generator** (`benchmarks/python/plot_24h_summary.py`)
   - Figure 1: Total N timeseries (coag/cond/combined panels)
   - Figure 2: Total dry mass timeseries
   - Figure 3: Error evolution (TFL/PPM vs Fortran, 2×3 grid)
   - Figure 4: PPM vs TFL scatter at hour 24
   - Figure 5: Mass conservation error bar chart
   - Figure 6: Size distribution evolution (3 representative scenarios)
   - Figure 7: All 50 scenarios hour-24 size distribution (10×5 grid)
   - Figure 8: Per-bin error heatmap (50 scenarios × 36 bins)

5. **Timing Plot Generator** (`benchmarks/python/plot_24h_timing.py`)
   - Bar chart: average wall-clock time per mode (Fortran/TFL/PPM)
   - Scatter: per-scenario Fortran vs JAX timing

6. **Fortran Benchmark Harness** (`benchmarks/fortran/benchmark_24h.f`)
   - Reads 50 scenarios from CSV
   - Runs 24h simulation with hourly CSV output
   - Three modes: coag_only, cond_only, combined
   - CPU timing per scenario/mode

7. **Test Suite** (`tests/test_24h_scenarios.py`)
   - Parametrized over all 50 scenarios
   - Tests: TFL vs Fortran (N, M), PPM vs Fortran (N, M), PPM vs TFL, mass conservation
   - Tolerance table: 50% N (different integrators), 5% M (coag), 1% conservation

8. **Shell Runner** (`benchmarks/run_24h.sh`)
   - End-to-end: scenarios → Fortran → JAX → compare → plots → timing

9. **Documentation** (`docs/24h_benchmark.md`)
   - Full benchmark documentation (overview, scenarios, conservation, running, output, plots, tests, FAQ)

### Mass Conservation Fix
- Changed `M_tot` tracking to use dry mass: `M_dry = sum(Mk[:, :SRTH2O])` where SRTH2O=43 (water index)
- Water is diagnostic (set by equilibrium), not conserved — including it creates spurious sources/sinks

### Files Modified
- `benchmarks/python/compare_24h.py` — Full rewrite for 3-way comparison
- `benchmarks/python/plot_24h_summary.py` — Full rewrite with 8 figures
- `tests/test_24h_scenarios.py` — Full rewrite with parametrized tests

---

## 2026-02-25 (Tue) — PPM Condensation: Phase 2 Complete

**Time**: Full day

### New Files Created

1. **PPM Algorithm** (`tomas_jax/physics/condensation_ppm.py`)
   - Piecewise Parabolic Method (Colella & Woodward 1984) in log-mass space
   - JIT-compatible via `jax.lax.fori_loop`
   - 3rd-order accuracy, CFL-based automatic substepping (C_max=0.8)
   - Monotonicity + positivity limiters

2. **PPM ezcond Driver** (`tomas_jax/physics/ezcond_ppm.py`)
   - Same interface as TFL ezcond
   - Computes TAU for full mcond (PPM handles its own CFL substepping)
   - Post-hoc mass conservation correction

3. **Method Selection** (`tomas_jax/solvers/condensation.py`)
   - Added `method='tfl'|'ppm'` parameter to `condensation_step()`
   - TFL: sequential numpy (Fortran-faithful)
   - PPM: JIT-compiled JAX

4. **PPM Tests** (`tests/test_ppm_condensation.py`)
   - 43 tests covering PPM reconstruction, flux, advection, species transport
   - All passing

5. **PPM Documentation** (`docs/ppm_condensation.md`)
   - Full algorithm description with mathematical formulation
   - Edge velocity, species transport, dry mass closure

### Performance
- PPM is ~100-180x faster than TFL for condensation portion
- JIT compilation overhead on first call

---

## 2026-02-24 (Mon) — Condensation: Phase 1 Complete (TFL Port)

**Time**: Full day

### New Files Created

1. **Gas Properties** (`tomas_jax/physics/gas_properties.py`)
   - Fuller-Schettler-Giddings diffusivity
   - Mean molecular speed, mean free path
   - Knudsen number, Fuchs-Sutugin correction

2. **Condensation Sink** (`tomas_jax/physics/condensation_sink.py`)
   - First-order gas loss rate CS [s^-1]
   - Per-bin sink fractions

3. **TFL Condensation** (`tomas_jax/physics/condensation.py`)
   - `dmdt_int` — analytic growth (Stevens et al. 1996)
   - `tmcond` — TFL bin redistribution (Tzivion/Feingold/Levin 1989)
   - Sequential Python loops matching Fortran line-by-line

4. **ezcond Driver** (`tomas_jax/physics/ezcond.py`)
   - Simple condensation driver (Jeff Pierce 2007)
   - Adaptive sub-stepping, mass conservation correction

5. **NH3 Equilibrium** (`tomas_jax/physics/nh3_equilibrium.py`)
   - Stoichiometric NH4:SO4 = 2:1 partitioning

6. **Water Equilibrium** (`tomas_jax/physics/water_equilibrium.py`)
   - Piecewise polynomial fits to ISORROPIA (273K)
   - Ammonium bisulfate + sea salt (disabled)

7. **Architecture Docs** (`docs/architecture.md`)
   - Full condensation pipeline documentation

### Validation
- H2SO4 diffusivity at 298K: 1.15e-5 m²/s (correct)
- dmdt_int: verified growth/shrink/neutral behavior
- Mass conservation: aerosol gained = gas lost within correction

---

## 2025-11-29 (Fri) — Initial Coagulation Implementation

**Commits**: 592ee01 → f41202e

### Features
- Working TOMAS coagulation in JAX
- JIT-compiled via diffrax (Tsit5 adaptive solver)
- Brownian coagulation kernel with Fuchs correction
- TFL mass-conserving algorithm
- MNFIX vectorized in JAX
- Sensitivity analysis via automatic differentiation
- README and basic project structure

### Files
- `tomas_jax/core/config.py` — Constants, dimensions
- `tomas_jax/core/state.py` — TomasState NamedTuple
- `tomas_jax/core/mnfix_jax.py` — Mass-number consistency
- `tomas_jax/physics/coagulation_kernel.py` — Brownian kernel
- `tomas_jax/physics/coagulation_rates.py` — TFL solver
- `tomas_jax/solvers/diffrax.py` — ODE integrator
- `run_box_model.py` — Main driver
- `run_sensitivity_analysis.py` — AD examples

### Validation
- Mass conservation < 1e-13 relative error for coagulation
