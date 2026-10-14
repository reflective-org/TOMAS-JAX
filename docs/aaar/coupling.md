# Coupled tutorial box model (`tomas_jax/box`)

`box.driver.run(scenario, progress_cb=None, cancel=None) -> RunResult` couples the
MCM gas chemistry and TUV-x photolysis (`gaschem`, see `gaschem.md`) to TOMAS with
c*-class SOA partitioning and N2O5 uptake (`aerosol_step.md`, `soa_partitioning.md`,
`heterogeneous.md`). Modelling assumptions are in `ASSUMPTIONS.md` (A1–A14).

## Inputs and outputs

- `box/scenario.py`: `Scenario` (frozen dataclasses, units in field names), TOML
  presets in `box/presets/` (`load_preset`, `list_presets`), nested overrides via
  `Scenario.replace(environment={"rh": 0.8})`.
- `box/result.py`: `RunResult` (JSON via `to_dict()`): size distributions
  (dN/dlog10Dp, dry dM/dlog10Dp; diameters at ρ = 1400 kg/m³ as in tomas-api),
  composition totals (N, SO4, NIT, SOA, POA, NH4, H2O, dry), gases [ppb], radicals
  [molec/cm³], H2SO4, J and SZA, gas/particle mass per c* class, top particle-phase
  species, diagnostics (condensation sink, nucleation rate, c* source, J sources,
  warnings, solver steps, runtime, `code_version()`).

`run(scenario, progress_cb=None, cancel=None, observer=None)`: the optional
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
4. `aerosol_outer_step` over Δt_c in 60 s micro-steps: sources (H2SO4, nitrate by
   uptake weights) → MNFIX → nucleation → coagulation → H2SO4 condensation → SOA
   partitioning → water.
5. Scatter (`ClassBridge`): the class gas change goes back to species — condensation
   by gas share, evaporation by share of the per-species particle tracker P_i — then
   P_i is rescaled to the size-resolved class mass. "Hold constant" species are reset.

Level 1 has no chemistry: one class (K = 1) holds the user's surrogate with a
production rate, and H2SO4 has an optional production rate.

Units: gas state molec/cm³; hand-off kg/cell; TOMAS #/cell and kg/cell with
boxvol = 1e10 cm³ (so per-cell thresholds mean 1 #/cm³); outputs µg/m³, ppb, cm⁻³.

## Presets and runtime (warm process, laptop CPU)

| Preset | What it shows | Duration | Δt_c | Wall time |
|---|---|---|---|---|
| `level1_volatility` | growth vs c* (0.01: 54→133 nm; 100: 54→57 nm GMD) | 12 h | 300 s | ~1.7 s |
| `level2_limonene_soa` | limonene oxidation → ~16 µg/m³ SOA, top contributors | 6 h | 300 s | ~6 s |
| `level3_day_night` | diurnal J/OH/NO3; nitrate 0 → 5 → 8 µg/m³ for γ = 0, 0.02, 0.1 | 24 h | 900 s | ~14 s |

First call in a process adds JIT compilation and the TUV-x table (~3–10 s);
`warmup()` runs every preset for two intervals. `coupling_dt_s` is a static argument
of the jitted aerosol step: each distinct value compiles separately, so the API allows
only {300, 600, 900} s.

Validation in `tests/test_aaar_driver.py`: Level 1 organic budget closes
(gas + particle = produced, 1e-3) and low vs high c* bracket the Pankow limits;
zero vapour gives constant dry mass and number decreasing only by coagulation;
class gather/scatter conserves mass per class (1e-12); Level 2 consumes limonene and
forms SOA; Level 3 nitrate is zero for γ = 0 and > 1 µg/m³ for γ = 0.02; bad species
and surrogate+chemistry are rejected.
