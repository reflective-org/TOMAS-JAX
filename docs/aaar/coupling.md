# Coupled tutorial box model (`tomas_jax/box`)

`box.driver.run(scenario, progress_cb=None, cancel=None) -> RunResult` couples the
MCM gas chemistry and TUV-x photolysis (`gaschem`, see `gaschem.md`) to TOMAS with
c*-class SOA partitioning and N2O5 uptake (`aerosol_step.md`, `soa_partitioning.md`,
`heterogeneous.md`). Modelling assumptions are in `ASSUMPTIONS.md` (A1–A24).

## Inputs and outputs

- `box/scenario.py`: `Scenario` (frozen dataclasses, units in field names), TOML
  presets in `box/presets/` (`load_preset`, `list_presets`), nested overrides via
  `Scenario.replace(environment={"rh": 0.8})`. Every class checks its values when it is
  built and raises `ValueError` naming the field: wrong types (a process switch must be
  `True` or `False`, so `"off"` is refused instead of counting as on; `numerics.nbins` a
  whole number) and values outside what the model accepts (RH and the seed organic
  fraction 0–1, with "did you mean 0.8?" for a percentage; temperature 100–400 K, with
  the kelvin value for a Celsius slip; pressure up to 120 kPa; seed GSD above 1;
  photolysis mode, latitude, solar zenith angle; non-negative amounts and rates; γ 0–1;
  positive duration and time steps). A single species name for `hold_constant` counts as
  one species (`"NO2"`, not N, O and 2). The checks only raise, so valid scenarios give
  bit-identical results. The web API's narrower limits (`tomas-api` `limits.py`) apply
  on top.
- `box/result.py`: `RunResult` (JSON via `to_dict()`): size distributions
  (dN/dlog10Dp, dry dM/dlog10Dp; diameters at ρ = 1400 kg/m³ as in tomas-api),
  composition totals (N, SO4, NIT, SOA, POA, NH4, H2O, dry), gases [ppb], radicals
  [molec/cm³], H2SO4, J and SZA, gas/particle mass per c* class, top particle-phase
  species, diagnostics (condensation sink, nucleation rate, c* source, J sources,
  warnings, solver steps, runtime, `code_version()`). Since 2026-10-19 also the surface
  and volume distributions with their totals, the particle properties (O:C, κ) and the
  cumulative loss budgets ([Reported properties and budgets](#reported-properties-and-budgets)).
- Series fields (2026-10-19, all off by default so Levels 1–3 are bit-identical,
  `tests/test_aaar_presets_unchanged.py`): `emissions_ppb_per_h` {species: E},
  `photolysis.cloud_fraction` (0–1), `losses` (`ventilation_per_h` λ, `outdoor_gases_ppb`,
  `outdoor_seed`, `o3_surface_per_h` k_s, `organic_vapour_wall_per_s` k_w,
  `particle_deposition` "off" | "lai_nazaroff", `room` with u*, S/V, height and particle
  density) and the seed salts (`seed.ammonium_nitrate_mass_fraction`, `seed.ammonium`,
  `seed.organic_o_to_c`). Equations below; sources in ASSUMPTIONS A15–A20.
- Nucleation inputs (2026-10-20, read only with `processes.nucleation` on, so every
  preset is still bit-identical): `nucleation.ion_pairs_cm3_s` (q, 0–1e4),
  `nucleation.nh3_ppb` (0–1e4, with a hint when a molec/cm³ value is given),
  `nucleation.organic` (True/False) and `nucleation.organic_log10_cstar_max` (−15 to 10).
  TOML table `[nucleation]`. Equations and defaults in
  [Nucleation inputs](#nucleation-inputs); sources in ASSUMPTIONS A21–A24.

`run(scenario, progress_cb=None, cancel=None, observer=None, aerosol_cfg=None,
nucleation_caps=True)`: the optional
`observer(t_s, snapshot)` is called at t = 0 and after every coupling interval with
numpy copies of every gas species (`y`, molec/cm³), `Nk`, `Mk`, `Gc`, the J used for
the interval, the SA/NA_HET counters before reset and the per-species particle
tracker. It is used by the PyCHAM benchmark (`benchmarks/pycham/`) and does not change
the run.

## Operator split (one coupling interval Δt_c)

1. J at the interval midpoint; k_het(N2O5) from the aerosol at the interval start;
   all k frozen for the interval (RO2 from the starting state).
2. Gas chemistry over Δt_c (`integrate_interval`). `SA` (from SO3 + H2O) and
   `NA_HET` (from `<34>`, renamed — A1) are counters: read, converted to rates
   (kg/cell/s) for the aerosol step, reset to zero.
3. Gather: condensable species (ClassMap) → class gas mass in `Gc[1:1+K]`. With
   nucleation on, the same gas state gives BioOxOrg, the sum of the lowest-c* species
   ([Nucleation inputs](#nucleation-inputs)), held constant over the interval.
4. `aerosol_outer_step` over Δt_c in 60 s micro-steps (5 s with nucleation, see
   below): sources (H2SO4, nitrate by
   uptake weights, PPM remap) → MNFIX → nucleation → coagulation → H2SO4 condensation
   → SOA partitioning (PPM remap) → water. H2SO4 condensation is on whenever
   nucleation is, even in a preset that switches it off (Level 1 urban air), so the
   new particles grow; the run says so in a warning line.
5. Scatter (`ClassBridge`): the class gas change goes back to species — condensation
   by gas share, evaporation by share of the per-species particle tracker P_i — then
   P_i is rescaled to the size-resolved class mass. "Hold constant" species are reset.

Level 1 has no chemistry: one class (K = 1) holds the user's surrogate with a
production rate, and H2SO4 has an optional production rate.

## Coupling interval (numerical solver setting)

The coupling interval Δt_c (`numerics.coupling_dt_s`) is a setting of the numerical
solver, not part of the chemistry or the aerosol physics. Within one interval the gas
chemistry runs with its rate constants k, photolysis rates J (at the interval midpoint)
and RO2 sum (at its start) held fixed, and the aerosol receives the H2SO4 and nitrate
formed in the interval, S, as a constant production rate P = S / Δt_c. Splitting the
processes this way is a first-order method: a result X behaves as

    X(Δt_c) ≈ X₀ + a·Δt_c,

where X₀ is the answer with continuous coupling and a is how strongly X depends on the
interval. Halving Δt_c halves the error and makes the chemistry about 1.7 times as
expensive (gas solver steps over 24 h: 1047 at 900 s, 2446 at 300 s, 8664 at 60 s).
The error is largest where the chemistry changes quickly within one interval (after
sunrise and sunset, and in the first interval, where the RO2 sum is held at its
starting value 0) and where much is emitted or held fixed (Level 3 resets NO2 to 1 ppb
after every interval, ASSUMPTIONS A14).

Every preset uses Δt_c = 300 s. Level 3 used 900 s until 2026-10-17, when Ali chose
300 s for the tutorial ("15 mins seems to be a bit long ... it really depends how fast
the chemistry is and how much emissions exist"). Planned for after the tutorial: the
numerical solver chooses Δt_c itself from how fast the chemistry changes and how much is
emitted, instead of one fixed value per preset.

Level 3 preset (24 h, 40 bins, nucleation off), measured 2026-10-17: final values at
24 h and peaks over the 30 min output times, with the deviation from a reference run
(30 s coupling with a 2.5 s aerosol micro-step) in brackets. All other runs use the
60 s micro-step (30 s for the 30 s run).

| Δt_c | SOA [µg/m³] | nitrate [µg/m³] | sulfate [µg/m³] | N [cm⁻³] | GMD [nm] | peak OH [cm⁻³] | peak NO3 [cm⁻³] |
|---|---|---|---|---|---|---|---|
| 900 s (old) | 0.05982 (+38 %) | 3.1373 (−6.4 %) | 5.5004 (−1.0 %) | 2551.8 (−0.07 %) | 149.57 (−1.0 %) | 2.375e7 (−5.2 %) | 1.723e9 (−2.0 %) |
| 600 s | 0.05353 (+23 %) | 3.2177 (−4.0 %) | 5.5215 (−0.6 %) | 2552.4 (−0.05 %) | 150.14 (−0.6 %) | 2.422e7 (−3.3 %) | 1.735e9 (−1.3 %) |
| **300 s (preset)** | **0.04781 (+10.3 %)** | **3.2840 (−2.0 %)** | **5.5420 (−0.27 %)** | **2553.1 (−0.02 %)** | **150.62 (−0.31 %)** | **2.468e7 (−1.5 %)** | **1.747e9 (−0.6 %)** |
| 60 s | 0.04414 (+1.8 %) | 3.3444 (−0.2 %) | 5.5581 (+0.02 %) | 2553.7 (0.00 %) | 151.05 (−0.02 %) | 2.503e7 (−0.1 %) | 1.756e9 (−0.07 %) |
| 30 s | 0.04357 (+0.5 %) | 3.3516 (0.00 %) | 5.5584 (+0.03 %) | 2553.7 (0.00 %) | 151.09 (0.00 %) | 2.506e7 (0.00 %) | 1.757e9 (0.00 %) |
| 30 s, 2.5 s micro-step | 0.04335 | 3.3514 | 5.5569 | 2553.6 | 151.08 | 2.506e7 | 1.757e9 |

These Level 3 numbers (and the figures in the text below) were measured from the 12:00 EDT
start; since 2026-10-20 the preset starts at 00:00 EDT (ASSUMPTIONS A25: SOA 0.070,
nitrate 1.39 µg/m³ at 24 h at 300 s), so the coupling errors are quoted for the old start.

GMD is the number geometric mean diameter, exp(Σ N_i ln D_i / Σ N_i) over the bins.
Peak OH is at noon of the second day (24 h; 1.19e7 cm⁻³ on day 1, at 4 h) and peak
NO3 at 00:30 local time (12.5 h). The SOA error is about +0.035 to +0.04 % per second
of interval (+10 % at 300 s, +38 % at 900 s) and the nitrate error about −0.007 % per
second, as the first-order equation says. SOA is this sensitive because the Level 3 SOA
at 24 h is a small residual: it peaks at 0.36 µg/m³ 2 h in, and OH and photolysis then
destroy most of the semivolatile gases it is in equilibrium with (see
[Presets and runtime](#presets-and-runtime-warm-process-laptop-cpu)).

Wall time for the 24 h run (warm process, laptop CPU with other work running, load
average about 6 on 14 cores): 13 s at 900 s, 18 s at 600 s, 27 s at 300 s, 90 s at
60 s, 157 s at 30 s and 213 s for the reference. On one thread, as in the API workers:
13 s at 900 s, 27 s at 300 s and 85 s at 60 s.

## Time steps with nucleation

With nucleation on, the driver shortens both time steps (`driver.time_steps`, constants
`NUCLEATION_MAX_COUPLING_DT_S = 60` s and `NUCLEATION_MAX_MICRO_DT_S = 5` s;
ASSUMPTIONS A12, A13). Runs without nucleation are bit-identical to before.

- Coupling interval: a requested interval Δt_req (`numerics.coupling_dt_s`) is split
  into m = ⌈Δt_req / 60 s⌉ equal parts, and each part Δt_c = Δt_req / m is one
  coupling interval (300, 600 and 900 s give 60 s; 90 s gives 45 s). Splitting into
  equal parts, rather than using 60 s whatever the request, keeps every boundary of
  the requested grid, so the duration and the output interval stay whole multiples of
  Δt_c and the output times are the same as without the cap. The smallest such m gives
  the longest interval within the cap.
- Micro-step: Δt_m = min(`AerosolStepConfig.micro_dt`, 5 s), so a finer requested
  micro-step is kept. This is a target length: the aerosol step splits each coupling
  interval into n = ⌈Δt_c / Δt_m⌉ equal micro-steps of Δt_c / n, which can be slightly
  shorter (a requested 130 s gives Δt_c = 43.3 s and micro-steps of 4.81 s).
- The run adds one warning line, e.g. "Nucleation is on, so the coupling interval is
  60 s instead of 300 s and the aerosol micro-step is 5 s instead of 60 s, to follow
  the nucleation burst. Output times are unchanged." The web app shows it to students,
  so it names no repository file. The diagnostics always hold
  `requested_coupling_dt_s`, `coupling_dt_s`, `requested_micro_dt_s` and `micro_dt_s`
  (the micro-step target).
- Code that calls `aerosol_outer_step` in its own loop, as tutorial notebook 04 does,
  gets no caps; it should take both values from `driver.time_steps`.

Why both. In each coupling interval the gas chemistry runs with rate constants,
photolysis rates and the RO2 sum frozen, and the H2SO4 formed, S, becomes a constant
production rate P = S / Δt_c. Each aerosol micro-step first raises the H2SO4
concentration C at once, C ← C + P·Δt_m, then nucleates at the rate J_nuc(C, T), then
coagulates and condenses. Binary nucleation goes roughly as J_nuc ∝ C^k with k about 3
to 8, so a long micro-step overshoots the burst; this is the larger error. A long
coupling interval holds OH, and so P, at one value while it changes within minutes.
In Level 1 there is no chemistry and the coupling interval has no effect on its own.

Convergence (study of 2026-10-16, 40 bins, against 30 s coupling with a 1 s
micro-step; the reference is about 1 % from converged):

| case | 300–900 s / 60 s (old) | 60 s / 60 s | 60 s / 10 s | 60 s / 5 s (now) | 120 s / 5 s |
|---|---|---|---|---|---|
| Level 1, 265 K, peak N | +5.5 % | +5.5 % | +3.7 % | +1.6 % | – |
| Level 2, 265 K, SO2 50 ppb, peak N | +15.7 % (300 s) | +30.0 % | +8.2 % | +2.6 % | −1.2 % |
| Level 3, 265 K, SO2 20 ppb, peak N | +57.6 % (900 s) | +77.8 % | +8.8 % | +3.5 % | +2.6 % |

Final N and GMD move the same way by less. With the 5 s micro-step, every nucleating
case is within 5 % on peak N, final N and GMD (largest +3.5 %); 2 s would be needed
for 2 %. The Level 1 regression is `tests/test_aaar_nucleation_dt.py`.

The 120 s / 5 s column passes only because two errors of opposite sign partly cancel:
with a 2 s micro-step, 120 s coupling alone is −4.3 % in Level 2 (SO2 50 ppb), against
−0.7 % at 60 s, and the +3 % micro-step error at 5 s brings it to −1.2 %. This is one
reason the cap is 60 s.

The table is for 40 bins, the only bin count the web API allows. At 80 bins (tutorial
notebook 01), micro-step convergence is slower: Level 1 at 265 K with 60 s / 5 s is
+5.9 % on peak N, +2.8 % on final N and −1.0 % on GMD against a 30 s / 0.5 s
reference, 60 s / 2 s is +4.5 % on peak N, and the reference itself still moves 2 %
between 1 s and 0.5 s. At 80 bins the cap is therefore outside 5 % on peak N. The
12 h Level 1 run takes 52 s at 80 bins with 5 s micro-steps and 138 s with 2 s.

Turning nucleation on in Level 3 also changes SOA and nitrate when nothing nucleates,
because the coupling interval drops from 300 s to 60 s and the aerosol micro-step from
60 s to 5 s. In the Level 3 preset (293 K, no new particles: peak N 3000 cm⁻³ in both
runs, largest nucleation rate 2e-8 cm⁻³ s⁻¹) the final values at 24 h with nucleation
on, against off, are SOA −8.6 %, nitrate +1.8 %, sulfate +0.24 % and GMD +0.27 %
(measured 2026-10-17; against the old 900 s interval it was SOA −27 %, nitrate +6.6 %).
With the same time steps (60 s / 5 s) nucleation on and off agree to 1e-6, so the whole
shift comes from the time steps, not from new particles. The nucleation-on run is the
more accurate one: it is within 0.8 % of the 30 s / 2.5 s reference (SOA +0.8 %,
nitrate −0.2 %), against +10 % and −2 % for the 300 s run
([Coupling interval](#coupling-interval-numerical-solver-setting)). In Levels 1 and 2
only GMD moves (−1.3 % and −1.2 %). A comparison of nucleation on and off in Level 3
therefore needs the off run at 60 s coupling too (or `nucleation_caps=False` on both
runs, at the cost of an inaccurate burst).

Cost (warm process, laptop): with nucleation on, Level 1 takes about 14 s (1.3 s
without the caps), Level 2 about 25 s (5.4 s), Level 3 about 112–125 s (27 s without
nucleation at the 300 s preset interval, 11.5 s at the old 900 s; the same on one
thread, as in the API workers). The cost is the same whether or not a burst
happens, and it grows in proportion to the run length: at the API's longest allowed
run, 48 h, Level 3 takes 235 s and Level 2 204 s (measured 2026-10-17 through the API
worker's job path, one thread). The API's tutorial job limit is 480 s for this reason
(raised from 120 s; tomas-api ADR-15).

Units: gas state molec/cm³; hand-off kg/cell; TOMAS #/cell and kg/cell with
boxvol = 1e10 cm³ (so per-cell thresholds mean 1 #/cm³); outputs µg/m³, ppb, cm⁻³.

## Nucleation inputs

Until 2026-10-20 the nucleation scheme saw H2SO4 and temperature only (NH3, ions and
the organic vapour were 0, so only the binary neutral path of Dunne et al. 2016 was
active, and at the preset temperatures nothing nucleated), and in the urban preset the
chemistry's H2SO4 never reached the particles. `Scenario.nucleation` now supplies the
other inputs; they are read only when `processes.nucleation` is on, so every run
without nucleation is unchanged (ASSUMPTIONS A13, A21–A24).

The rate the aerosol step uses (`physics/nucleation.py::ricco_dunne_nucleation_rate`),
with C the H2SO4 concentration, A the NH3 concentration (both in units of 1e6 cm⁻³ in
the Dunne terms), T the temperature and n± the small-ion concentration:

- Dunne et al. (2016), four paths, J = J_bn + J_tn + J_bi + J_ti [cm⁻³ s⁻¹]:
  J_bn = k_bn(T) C^3.95 (binary neutral), J_bi = k_bi(T) n± C^3.37 (binary ion-induced),
  J_tn = k_tn(T) f_n(C, A) and J_ti = k_ti(T) n± f_i(C, A) (ternary with NH3), where
  f_n = A C^2.89 / (1.6e-6 + C^2.89 / A^8), f_i = A C^3.14 / (4.85e-3 + C^3.14 / A^3.07)
  and each k(T) = exp(u − exp(v (T/1000 − w))) with the paper's u, v, w per path.
- Small ions from the ion-pair production rate q [pairs cm⁻³ s⁻¹] in steady state with
  recombination: n± = √(q / α), α = 6e-8 √(300/T) + 6e-26 M (300/T)^4 cm³ s⁻¹ (M the
  air number density). q = 0 switches both ion paths off; A = 0 both ternary paths.
- Riccobono et al. (2014) with the Yu et al. (2017) temperature factor:
  J_org = k_m [H2SO4]² [BioOxOrg] exp(ΔH/k_B (1/T − 1/T₀)), k_m = 3.27e-21 cm⁶ s⁻¹,
  ΔH = 2.66e-19 J, T₀ = 278 K.

Inputs (`box/scenario.py::Nucleation`, TOML `[nucleation]`):

| field | meaning | unit | default | presets |
|---|---|---|---|---|
| `ion_pairs_cm3_s` | q, ion-pair production | pairs cm⁻³ s⁻¹ | 4 | 4 in all three (A21) |
| `nh3_ppb` | NH3 mixing ratio, constant | ppb | 3 | 3 urban and day & night, 30 indoor (A22) |
| `organic` | Riccobono path from the mechanism's vapours | True/False | True | True |
| `organic_log10_cstar_max` | BioOxOrg = gas-phase sum of the closed-shell organics with log10 c* ≤ this | log10 µg/m³ at 298 K | −0.5 (c* ≤ 0.3 µg/m³) | −0.5 (A23) |

The driver converts NH3 to molec/cm³ once (A × 1e-9 × M), takes BioOxOrg from the gas
state it hands to the aerosol step (after the interval's chemistry and gas losses, the
same state the SOA partitioning sees) as Σ_i y_i over the species with
log10 c*_i ≤ `organic_log10_cstar_max` from the scenario's c* table, and passes all
three through `AerosolForcing` (`fion`, `nh3_conc`, `org_conc`), constant over the
interval. Without chemistry (Level 1 volatility) BioOxOrg is 0. The organic path does
not consume the vapour (the nucleated organic mass, 10 % of the cluster, is created in
`SRTORG_NUC` as before, AC2). `diagnostics["nucleation_inputs"]` records q, NH3
[molec/cm³], the threshold and the species BioOxOrg sums (None with nucleation off),
and `diagnostics["nucleating_organic_cm3"]` BioOxOrg at every output time.

Why c* ≤ 0.3 µg/m³: Riccobono's BioOxOrg were the highly oxygenated pinanediol
oxidation products seen by a nitrate chemical-ionisation mass spectrometer, i.e. the
ELVOC/LVOC end of the volatility scale. The MCM limonene mechanism has no autoxidation,
so its lowest c* in the Jathar table is 2.3e-3 µg/m³ (HPC52CO3H) and the ELVOC boundary
of Donahue et al. (2012), 3e-4 µg/m³, selects nothing. The default therefore takes the
LVOC and below (13 species at −0.5: C925OOH, C825OOH, C734OOH, INDOOH, INDHPCO3H,
HPC52PAN, HPC52CO3H, C537OOH and five nitro-acids); it is an upper-bound stand-in, and
the threshold is an input so it can be tightened (−3.5 gives the ELVOC definition and
no organic nucleation with this mechanism).

Level 1 urban air keeps gases and particles apart by switching H2SO4 condensation and
SOA partitioning off. With nucleation on, `_processes` adds `h2so4_condensation`, so
the H2SO4 the chemistry makes from SO2 both nucleates and condenses, dC/dt = P − CS·C
with CS the condensation sink of the particles, and the new particles grow. The run
warns "H2SO4 condensation is on as well, because nucleation is on: the new particles
need it to grow." Without SO2 in the urban preset the H2SO4 source is zero, so nothing
changes but the time steps; add SO2 (20–50 ppb) for a daytime burst (A24).

Measured 2026-10-20 (laptop, 40 bins, nucleation on with the default inputs; N in cm⁻³,
H2SO4 in cm⁻³, J in cm⁻³ s⁻¹, "< 10 nm" the number below 10 nm):

| run | peak H2SO4 (when) | peak J (when) | N start / peak (when) / end | < 10 nm peak / end | new mode grew to | wall |
|---|---|---|---|---|---|---|
| Level 1 urban, 6 h, no SO2 (preset) | 0 | 0 | 20003 / 20003 / 11533 | 10 / 0 | – | 33 s |
| Level 1 urban, 6 h, SO2 30 ppb | 4.4e7 (2.8 h) | 1.0 (3.0 h, solar noon) | 20003 / 20003 / 12609 | 222 (3 h) / 28 | stays at 2 nm | 28 s |
| Level 1 urban, 6 h, SO2 50 ppb | 6.8e7 (2.7 h) | 3.8 (2.8 h) | 20003 / 20003 / 13124 | 940 (2.7 h) / 105 | stays at 2 nm | 28 s |
| Level 2 indoor, 4 h, no SO2 (preset) | 0 | 0 | 3000 / 4540 / 4540 (outdoor air) | 0 / 0 | – | 29 s |
| Level 2 indoor, 4 h, SO2 30 ppb | 9.1e7 (10 min) | 167 (10 min) | 3000 / 74105 (20 min) / 4599 | 70822 (20 min) / 54 | 8 nm | 28 s |
| Level 3 day & night, 24 h (preset) | 5.9e7 (3.5 h, 15:30) | 10.6 (3.5 h) | 3000 / 23345 (4.5 h) / 9775 | 18615 (4 h) / 5209 (second morning) | 42 nm by 24 h | 118 s |

- Level 1 urban: the chemistry's H2SO4 now reaches the particles; with 30 ppb SO2 it
  reaches 4.4e7 cm⁻³ against a condensation sink of 2e4 cm⁻³ of seed, J peaks at 1
  cm⁻³ s⁻¹ at solar noon and a few hundred new particles per cm³ sit below 10 nm; they
  are scavenged before they grow (total N still falls by coagulation, ending 1100
  cm⁻³ above the no-SO2 run). With 50 ppb the sub-10 nm number reaches 940 cm⁻³.
- Level 2 indoor: no sunlight, but the ozonolysis OH (peak 8e5 cm⁻³) oxidises 30 ppb
  of SO2 onto only 3000 cm⁻³ of particles, so H2SO4 reaches 9.1e7 cm⁻³ within 10
  minutes and with 30 ppb NH3 the ternary rate reaches 167 cm⁻³ s⁻¹: N jumps to 7.4e4
  cm⁻³ at 20 minutes and falls back to 4.6e3 by 4 h as the limonene (and so OH and
  H2SO4, down to 1.5e5 cm⁻³) runs out and the 2 nm particles coagulate; some grow to
  8 nm. (30 ppb of SO2 indoors is a stress test, not a typical room.) Without SO2
  there is no H2SO4 and no nucleation; BioOxOrg is 4e6 cm⁻³.
- Level 3: with the default inputs the preset bursts in daylight (N 3000 → 2.3e4 cm⁻³
  at 16:30 local), the mode grows past 25 nm overnight by H2SO4, SOA and nitrate (to
  the 42 nm bin by 24 h), N falls to 5.0e3 cm⁻³ by 20 h and the second morning starts
  a new burst (9.8e3 at 24 h, J 6.8). J at night is below 3e-4. Switching the inputs
  off one by one in the first 6 h: binary alone J 1.7e-8 (no new particles), with ions
  4.6e-4 (about 1 cm⁻³), with ions and NH3 9.0 (N peak 2.0e4), with the organic path
  as well 10.6 (2.3e4); BioOxOrg peaks at 5.1e6 cm⁻³.
- Wall time is unchanged by the inputs (about 28–33 s for 4–6 h, 118 s for 24 h).

## Presets and runtime (warm process, laptop CPU)

| Preset | What it shows | Duration | Δt_c | Wall time |
|---|---|---|---|---|
| `level1_volatility` | growth vs c* (0.01: 54→133 nm; 100: 54→57 nm GMD) | 12 h | 300 s | ~1.7 s |
| `level2_limonene_soa` | limonene oxidation → ~9 µg/m³ SOA (Jathar c* table, A8), top contributors | 6 h | 300 s | ~6 s |
| `level3_day_night` | diurnal J/OH/NO3; nitrate 0 → 3.3 → 7.1 µg/m³ for γ = 0, 0.02, 0.1; SOA 0.05 µg/m³ | 24 h | 300 s | ~27 s |
| `level1_urban` | O3/NO/NO2 in midday sun with 1 ppb/h NO (photostationary state to 1 %); 2e4 → 1.2e4 cm⁻³ by coagulation only, dry mass conserved | 6 h | 300 s | ~5.5 s |
| `level2_indoor` | dark limonene ozonolysis at O3 held at 30 ppb, with ventilation, O3 surface loss, vapour wall loss and deposition; SOA peak 1.0 µg/m³ (9 with the losses off; 0.2 with O3 free to fall to 5.6 ppb) | 4 h | 300 s | ~8 s |

With nucleation on, every preset runs at 60 s coupling with 5 s micro-steps and takes
about 14 s, 25 s and 112–125 s ([Time steps with nucleation](#time-steps-with-nucleation));
since 2026-10-20 the nucleation inputs make the Level 3 preset burst in daylight
(3000 → 2.3e4 cm⁻³), and Levels 1–2 burst with 20–50 ppb SO2
([Nucleation inputs](#nucleation-inputs)).
Δt_c is the coupling interval, a numerical solver setting
([Coupling interval](#coupling-interval-numerical-solver-setting)); Level 3 used 900 s
(~13 s) until 2026-10-17.

Level 3 SOA with the Jathar c* table (A8) is small and not monotonic: there is no
dilution, yet it peaks at 0.36 µg/m³ about 2 h in, falls to ~0.12 µg/m³ by sunset,
stays flat at night and falls to 0.048 µg/m³ on the second morning, as OH and
photolysis destroy the gas-phase semivolatile hydroperoxides and the particles
evaporate to follow. Because it is a small residual, it depends on the coupling
interval much more than nitrate does: at the preset 300 s it is 10 % above the 30 s
reference and nitrate 2 % below (900 s: +38 % and −6.4 %; A12). Quote the Level 3 SOA
as "a few tenths of a µg/m³ or less", not to two figures.

First call in a process adds JIT compilation and the TUV-x table (~3–10 s);
`warmup()` runs every preset for two intervals at its own coupling interval (300 s in
all three), and again with nucleation on at 60 s (`warmup(nucleation=False)` skips that;
about 8 s against 11 s in total). `coupling_dt_s` is a static argument of the jitted
aerosol step and of the gas solver: each distinct value compiles separately, so the API
allows only {300, 600, 900} s. A run without nucleation at 600 or 900 s is not covered
by `warmup()` and compiles on its first call in a process (about 2–3 s; before
2026-10-17 this was true of Level 3 at 300 and 600 s). With nucleation on all three
become 60 s intervals with 5 s micro-steps, one more compile per preset, which
`warmup()` covers (`tests/test_aaar_nucleation_dt.py`).

Validation in `tests/test_aaar_driver.py`: every preset requests a 300 s coupling
interval; Level 1 organic budget closes
(gas + particle = produced, 1e-3) and low vs high c* bracket the Pankow limits;
zero vapour gives constant dry mass and number decreasing only by coagulation;
class gather/scatter conserves mass per class (1e-12); Level 2 consumes limonene and
forms SOA, and over 6 h gives 6–12 µg/m³ with < 20 % nitrogen-containing top
contributors (Jathar c*); a c* table that leaves condensable species on Li et al.
(2016) adds a run warning; Level 3 nitrate is zero for γ = 0 and > 1 µg/m³ for
γ = 0.02; bad species and surrogate+chemistry are rejected.

## Emissions and losses (series of 2026-10-19)

The urban and indoor levels add constant emissions and first-order losses. They are
operator-split like everything else: each acts once per coupling interval Δt_c, in this
order within the interval (A17, A19):

1. Emissions: y_s ← y_s + E_s Δt_c for every emitted species, before the gas chemistry
   (`emissions_ppb_per_h`, converted with E [molec cm⁻³ s⁻¹] = E [ppb/h] × 10⁻⁹ M / 3600).
   Adding the whole interval's emission up front is first order in Δt_c, as the rest of
   the split (at 300 s with 1 ppb/h of NO this is 0.08 ppb per interval).
2. Cloud fraction: every photolysis rate used in the interval (and reported) is
   J = J_clear × (1 − 0.75 CF). This is a tutorial simplification, not a radiative
   transfer result (A16).
3. Gas chemistry over Δt_c (unchanged).
4. Gas losses (after the chemistry, before the organics are gathered into the c* classes):
   for every mechanism species, dC/dt = λ (C_out − C) − k C, with λ the air-exchange rate,
   C_out the outdoor mixing ratio (`losses.outdoor_gases_ppb`, 0 when unlisted) and k an
   extra sink: k_s for O3 (surface loss), k_w for every condensable organic vapour
   (irreversible wall loss), 0 otherwise. The interval is solved exactly:
   C(Δt_c) = C* + (C₀ − C*) e^{−(λ+k) Δt_c}, with C* = λ C_out / (λ + k).
5. Aerosol step over Δt_c (unchanged).
6. Particle losses, per bin i and species: dN_i/dt = λ (N_out,i − N_i) − β(Dp_i) N_i, the
   same for every mass component, solved exactly as in 4. N_out is the outdoor seed
   (`losses.outdoor_seed`, water-equilibrated like the initial state; zero without one).
   β(Dp) is the Lai and Nazaroff (2000) deposition rate of a room (below). The vapours
   in the TOMAS gas array are treated in the same step: H2SO4 vapour is ventilated; the
   Level 1 surrogate is ventilated and wall-lost with k_w; with chemistry on the c*-class
   slots are not touched here because their species were already treated in step 4 and
   the class change is scattered back to them. Water is re-equilibrated afterwards.
7. Scatter and "hold constant" reset (unchanged).

Budgets. For each first-order step the time integral ∫C dt over the interval is known in
closed form (C* Δt_c + (C₀ − C*)(1 − e^{−rΔt_c})/r with r = λ + k), so each sink removes
exactly (its rate) × ∫C dt and ventilation brings in λ C_out Δt_c; in − out − removed equals
the change to rounding (`tests/test_aaar_losses.py`). The cumulative amounts since the start
are reported at every output time in `RunResult.losses`.

Lai and Nazaroff (2000) deposition (`box/losses.py::lai_nazaroff_beta`;
https://doi.org/10.1016/S0021-8502(99)00536-4). The loss rate of a room is

    β(Dp) = (S_v v_dv + S_u v_du + S_d v_dd) / V,

with deposition velocities to vertical (v_dv), upward-facing (v_du, floor) and
downward-facing (v_dd, ceiling) smooth surfaces:

    v_dv = u* / I,    v_du = v_s / (1 − e^{−v_s I / u*}),    v_dd = v_s / (e^{v_s I / u*} − 1),

where u* is the near-surface friction velocity, v_s the settling velocity (Stokes with
the Cunningham slip correction) and I the dimensionless integral over the near-wall eddy
diffusivity, I = 3.64 Sc^{2/3} (a − b) + 39 for r⁺ ≤ 4.3 (Sc = ν/D the particle Schmidt
number, r⁺ = (Dp/2) u*/ν; the a and b terms are the closed-form integral of the paper's
eq. 11 over 0 ≤ y⁺ ≤ 4.3). Room: u* = 1 cm/s, S/V = 3 m⁻¹, height 2.5 m (so floor and
ceiling are 0.4 m⁻¹ each and the walls 2.2 m⁻¹), particle density 1.4 g/cm³ (A17,
placeholders). At these values β is 0.19 h⁻¹ at 10 nm, 0.008 h⁻¹ at 200 nm (the minimum)
and 1.6 h⁻¹ at 5 µm.

## Reported properties and budgets

- Surface and volume distributions from each bin's dry mass M and number N (ρ = 1.4 g/cm³):
  dV/dlog10Dp = (dM/dlog10Dp) / ρ = (π/6) Dp³ dN/dlog10Dp [µm³ cm⁻³] and
  dS/dlog10Dp = π Dp² dN/dlog10Dp [µm² cm⁻³], with Dp = (6 m / (π ρ))^{1/3} the bin's
  mean-mass diameter, m = M/N (1 µg m⁻³ at 1 g cm⁻³ is 1 µm³ cm⁻³). `totals["S_um2cm3"]`
  and `totals["V_um3cm3"]` are their sums over the bins. V × ρ is the dry mass at every
  time, so coagulation (which adds volumes) keeps V to rounding (Level 1 urban: 3e-10 over
  6 h). The first version used the bin-centre diameter for both, which let V drift by
  0.2 % in 6 h as particles moved inside their bins (fixed 2026-10-19 in review).
- `properties["o_to_c"]`: O:C of the particle organics from the per-species tracker
  and the mechanism's atom counts, n_O = Σ P_i O_i / MW_i, n_C = Σ P_i C_i / MW_i, plus
  the seed organic at its own O:C (`seed.organic_o_to_c`, 0.5 by default) with mass per
  mole of carbon 12.011 + 1.008 (2 − O:C) + 15.999 O:C g/mol (H:C = 2 − O:C) (A18).
- `properties["kappa"]`: κ_org = 0.18 O:C + 0.03 (Lambe et al. 2011), κ = 0.61 for
  ammonium sulfate and 0.67 for ammonium nitrate (Petters and Kreidenweis 2007), mixed by
  dry volume (densities 1.4, 1.77, 1.72 g/cm³); sulfate counts as ammonium sulfate and
  nitrate as ammonium nitrate whether or not ammonium is present, so the ammonium mass is
  not counted twice. `kappa_organic`, `kappa_inorganic` and `organic_volume_fraction` are
  reported with it; NaN (null in JSON) where there is no such mass.
- `losses`: cumulative since the start, at the output times: `wall_vapour_ugm3` and
  `ventilated_vapour_ugm3` (condensable organic vapour; ventilation net out),
  `o3_surface_ppb`, `ventilated_gases_ppb` {species: net out} for the reported gases,
  `deposited_{dry,SOA,POA,inorganic}_ugm3`, `deposited_N_cm3`, the same `ventilated_*`
  (net out; negative when outdoor air brings more in), `emitted_ppb` {species}. For the
  indoor level, "where did the limonene go" is initial + ventilated in − ventilated out −
  remaining = reacted, and the reacted carbon is split between SOA (`totals`), vapour on
  the walls, vapour ventilated out, and SOA deposited or ventilated.

Validation (`tests/test_aaar_series_driver.py`, `tests/test_aaar_series_presets.py`):
with every other process off, particle ventilation follows the exact exponential toward
the outdoor distribution, deposition and ventilation budgets close to 1e-9, O3 in the dark
follows C* + (C₀ − C*) e^{−(λ+k_s)t} to 1e-6, NO emission accumulates linearly, the
surrogate's wall loss closes to 1e-9, a cloud fraction of 1 scales every J by 0.25, and
the seed salts carry the stated stoichiometry. Urban preset: photostationary state
k[O3][NO] / (J(NO2)[NO2]) = 1 ± 0.05 at every daytime output, N falls monotonically with
dry mass conserved to 1e-6 and V to 1 %. Indoor preset: O3 reaches λ O3_out/(λ + k_s)
within 15 %, the limonene budget closes, the cumulative losses are monotonic, SOA is
0.05–1 µg/m³ with the losses on and > 5 µg/m³ with them off, and opening the window
(λ = 3 h⁻¹) more than doubles the indoor O3.
