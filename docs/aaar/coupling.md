# Coupled tutorial box model (`tomas_jax/box`)

`box.driver.run(scenario, progress_cb=None, cancel=None) -> RunResult` couples the
MCM gas chemistry and TUV-x photolysis (`gaschem`, see `gaschem.md`) to TOMAS with
c*-class SOA partitioning and N2O5 uptake (`aerosol_step.md`, `soa_partitioning.md`,
`heterogeneous.md`). Modelling assumptions are in `ASSUMPTIONS.md` (A1–A14).

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
  warnings, solver steps, runtime, `code_version()`).

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
3. Gather: condensable species (ClassMap) → class gas mass in `Gc[1:1+K]`.
4. `aerosol_outer_step` over Δt_c in 60 s micro-steps (5 s with nucleation, see
   below): sources (H2SO4, nitrate by
   uptake weights, PPM remap) → MNFIX → nucleation → coagulation → H2SO4 condensation
   → SOA partitioning (PPM remap) → water.
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

## Presets and runtime (warm process, laptop CPU)

| Preset | What it shows | Duration | Δt_c | Wall time |
|---|---|---|---|---|
| `level1_volatility` | growth vs c* (0.01: 54→133 nm; 100: 54→57 nm GMD) | 12 h | 300 s | ~1.7 s |
| `level2_limonene_soa` | limonene oxidation → ~9 µg/m³ SOA (Jathar c* table, A8), top contributors | 6 h | 300 s | ~6 s |
| `level3_day_night` | diurnal J/OH/NO3; nitrate 0 → 3.3 → 7.1 µg/m³ for γ = 0, 0.02, 0.1; SOA 0.05 µg/m³ | 24 h | 300 s | ~27 s |

With nucleation on, every preset runs at 60 s coupling with 5 s micro-steps and takes
about 14 s, 25 s and 112–125 s ([Time steps with nucleation](#time-steps-with-nucleation)).
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
