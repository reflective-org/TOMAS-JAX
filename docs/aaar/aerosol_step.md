# Aerosol outer step (AAAR tutorial)

`tomas_jax/box/aerosol_step.py::aerosol_outer_step` is the aerosol side
that all three tutorial levels share. The coupling driver (`box/driver.py`)
calls it once per outer (gas-chemistry) step and holds the sources constant
over that step.

## Call contract

```python
from tomas_jax.box.aerosol_step import (
    AerosolState, AerosolForcing, SOAParams, aerosol_outer_step,
    AEROSOL_PROCESSES)
from tomas_jax.box.config import AerosolStepConfig

state, diag = aerosol_outer_step(
    state, forcing, soa_params, dt_outer,           # dt_outer [s], static
    micro_dt=60.0,                                  # [s], static
    processes=AEROSOL_PROCESSES,                    # static tuple
    cfg=AerosolStepConfig())                        # static
```

| Type | Field | Shape | Units |
|---|---|---|---|
| `AerosolState` | `Nk` | (nbins,) | #/cell |
| | `Mk` | (nbins, 44) | kg/cell |
| | `Gc` | (44,) | kg/cell |
| | `xk` | (nbins+1,) | kg |
| | `temp`, `pres`, `rh`, `boxvol` | scalar | K, Pa, 0–1, cm³ |
| `AerosolForcing` | `h2so4_rate` | scalar | kg/cell/s into `Gc[SRTSO4]` |
| | `org_rate` | (K,) | kg/cell/s into `Gc[SRTSOA1:SRTSOA1+K]` |
| | `nitrate_rate` | scalar | kg/cell/s of NO₃ into `Mk[:, SRTNIT]` |
| | `nitrate_weights` | (nbins,) | – (e.g. `n2o5_uptake` weights) |
| | `nh3_conc`, `fion`, `org_conc` | scalar, default 0 | molec/cm³, pairs/cm³/s, molec/cm³ (nucleation only) |
| `SOAParams` | `cstar_ref` | (K,) | µg/m³ at 298 K |
| | `dhvap` | (K,) | kJ/mol |
| | `mw` | (K,) | g/mol |
| | `alpha` | scalar | – |
| `AerosolDiag` | `cs_h2so4` | scalar | s⁻¹, H₂SO₄ sink at step end |
| | `j_nuc` | scalar | cm⁻³ s⁻¹, mean Ricco+Dunne J at micro-step starts (0 if off) |
| | `soa_cs` | (K,) | s⁻¹, last vbs-jax sub-step |
| | `soa_time_integrated` | bool | every SOA call advanced its full micro-step |
| | `nitrate_added` | scalar | kg/cell actually added |

Unit conversions: kg/cell = µg/m³ × 1e-9 × boxvol·1e-6;
molec/cm³/s of H₂SO₄ → kg/cell/s: × boxvol × MW·1e-3 / N_A.

**SOAParams arrays** come from the caller. For the tutorial,
`tomas_jax/gaschem/volatility.py` (gas-chemistry stream) defines the 15
half-decade classes (log10 c\*(298 K) = −3, −2.5, …, 4) and their ΔH_vap.
This module applies no c\* or ΔH_vap rule of its own. ΔH_vap is in
**kJ/mol** (vbs-jax `delta_hvap_kj`).

**Levels.** Level 1: K = 1 (the surrogate in slot 1, `cstar_ref = [c_star]`),
`org_rate = [P_org]`. Either source can be zero. K ≥ 1 always:
for an H₂SO₄-only run, pass `org_rate = [0.0]` or leave
`'soa_partitioning'` out of `processes`. Level 2: K ≤ 15
lumped classes from the gas mechanism. Level 3: Level 2 plus
`nitrate_rate`/`nitrate_weights` from `physics/heterogeneous.py`
([heterogeneous.md](heterogeneous.md)).

**Compilation.** One compile per (nbins, K, dt_outer, micro_dt, processes,
cfg). Values in `state`, `forcing` and `soa_params` are traced: moving the
c\* slider or changing T, RH or rates does not recompile
(`test_cstar_change_does_not_recompile`). Pass scalars with consistent
types (e.g. always `jnp.float64`), because jit specialises on weak
vs. strong types.

## Micro-step

The outer step is split into n = ceil(dt_outer/micro_dt) equal micro-steps
(`lax.fori_loop`). Each one runs:

1. **Sources.** `Gc[SRTSO4] += h2so4_rate·dt` and `Gc[classes] += org_rate·dt`.
   `Mk[:, SRTNIT] += nitrate_rate·dt·w`, where w is `nitrate_weights` over
   bins with Nk > 1e10 #/cell, renormalised. The particles that gained
   nitrate then move between bins by the PPM remap (see
   [Bin transfer](#bin-transfer-of-soa-and-nitrate-growth_bin_transfer)).
   Then MNFIX.
2. **Nucleation → coagulation → H₂SO₄ condensation**:
   `solvers.condensation.make_step` with the enabled subset, `jit=False`,
   PPM-JIT condensation and `cfg.water_scheme`/`cfg.water_every_process`.
   H₂SO₄ condensation also does NH₃ equilibrium (`Gc[SRTNH4]`).
3. **SOA partitioning**: `soa_partition_step` ([soa_partitioning.md](soa_partitioning.md)),
   followed by the PPM remap of the particles whose SOA changed.
4. **Water**: `equilibrate_water(Mk, rh, temp, cfg.water_scheme)`.

`processes` toggles steps 2–3. The order is fixed. With all processes off,
only sources, MNFIX and water run. SO₂ chemistry and dilution are not part
of this step; the driver owns them.

## Configuration (`box/config.py::AerosolStepConfig`)

| Field | Default | Meaning |
|---|---|---|
| `micro_dt` | 60 s | default micro-step |
| `boxvol_cm3` | 1e10 | tutorial box volume. TOMAS thresholds are per cell (NEPS_CONDSINK = 1e10 #/cell), so this makes "populated" mean ≥ 1 cm⁻³ |
| `alpha_h2so4` | 1.0 | H₂SO₄ accommodation |
| `water_scheme` | bisulfate | `equilibrate_water` scheme |
| `water_every_process` | False | `make_step` option |
| `soa_n_substeps` | 10 | vbs-jax sub-step budget (raise if `soa_time_integrated` is False) |
| `growth_bin_transfer` | `'ppm'` | how SOA growth/evaporation and nitrate move particles between bins: `'ppm'` (PPM remap) or `'mnfix'` (the previous MNFIX split, kept for comparison). Any other value raises `ValueError` |

The driver stream may add its own classes to `box/config.py`.
`driver.run(scenario, aerosol_cfg=AerosolStepConfig(...))` runs a scenario
with non-default settings; the fields that differ from the defaults are
listed in `RunResult.diagnostics["aerosol_cfg"]` (empty for a default run).

## Bin transfer of SOA and nitrate (`growth_bin_transfer`)

SOA partitioning (vbs-jax) and the N₂O₅-uptake nitrate source add mass to
(or remove it from) the particles of each bin in place. The bins are
fixed, so the particles then have to move to the bins their new mass
belongs to. H₂SO₄ condensation already does this with PPM advection in
log-mass (`physics/condensation_ppm.py`). SOA and nitrate now use the same
machinery through `physics/ppm_remap.py::remap_mass_change`.

**Previous scheme (`'mnfix'`).** MNFIX ran after every vbs-jax sub-step
(and after the nitrate source). When a bin's mean mass leaves the bin,
MNFIX keeps part of the particles at the bin centre and moves the rest to
1/1.1 of the next bin's upper edge. Repeated every 60 s this combs the
size distribution: the final dN/dlog10Dp of the Level 1 preset had 3 / 9 /
28 interior extrema at 40 / 80 / 160 bins (count of local maxima and
minima above 5 % of the peak), more at higher resolution, and the
number-mean diameter depended on resolution.

**PPM remap (`'ppm'`, default).** vbs-jax changes the class masses in
place during its sub-steps (its `redistribute` hook only copies them into
Mk; geometry comes from each bin's actual mean mass). After the call:

1. Each bin with particles moved its mean dry mass m by
   s = ln(m_new/m_old) in log-mass.
2. Edge displacement, upwind. Inside each bin s is taken as linear in
   ln m around the bin mean, with the smaller of the two one-sided slopes
   to the neighbouring bin means (zero next to a neighbour without
   particles or with the same s), and clipped so that an edge value lies
   between the bin's s and the neighbour's. Particles cross an edge
   upwards with the lower bin's value at that edge (if > 0) and downwards
   with the upper bin's value (if < 0). No flux passes the two grid ends.
3. Number moves by flux-form PPM in ln m (`ppm_reconstruct`, the same
   monotone, positive reconstruction as H₂SO₄ condensation), sub-stepped
   at Courant ≤ 0.8. Every species moves with the PPM mass-weighted
   fraction of the donor bin, so all species share one flux and the
   gained (or lost) mass travels with the particles that carry it.
4. A bin whose mean still ended up past an edge (possible at steep edges
   of the distribution, because PPM reconstructs number only) keeps its
   particles just inside the edge and hands the fewest possible particles
   to the neighbour at the neighbour's mean mass (`keep_means_in_bins`).
5. Top bin (`cap_top_bin`): particles that grow in the top bin stay
   there. If its mean passes the top edge, just enough of the mass added
   in place is taken back (SOA returns to the vapour; nitrate is not
   added and the driver warns), so MNFIX never trims the bin, which would
   delete seed mass as well.

Number and the mass of every species are conserved to round-off (flux
differences); no bin can go negative. Growth (s > 0) moves particles up,
evaporation (s < 0) moves them down. MNFIX still runs where it did before
(after the sources, inside nucleation/coagulation/condensation) and once
after the SOA remap, where it restores its floor (1e-5 #/cell) in bins
the remap emptied. Any SOA it deletes there goes back to the vapour, so
gas + particle mass closes per class with either setting. In the presets
no populated bin is left outside its bounds after the remap (checked at
every outer step), so MNFIX has nothing to split there; after very large
one-step changes (see Limitations) it can still split a few bins.

Why the edge displacement comes from the donor's side: an edge value
interpolated between the two neighbouring bin means (the first version)
drags particles that did not change. Example: SOA-only particles in bins
30–39 lose 99 % of their mass, sulfate particles in bins 40–49 do not
change (80 bins). Interpolation gave the edge between bins 39 and 40 half
of bin 39's displacement, pointing down, so bin 40's sulfate particles
moved down as well: 10 % of the unchanged particles and 1.3 % of the
sulfate ended below bin 40. Upwind, they stay exactly where they are
(`test_remap_leaves_unchanged_particles_in_place`). Where s varies
smoothly the edge value is the same as the interpolation for growth (the
usual case: s falls with size).

Why not TOMAS's `dmdt_int` law around each bin's own mean (as
`ezcond_ppm_jax` does for H₂SO₄): `dmdt_int` assumes d(m^2/3)/dt is
constant, which slows the upper edge of a bin more than the vbs-jax rates
do. Number then lags mass, every bin's mean drifts to its upper edge
(Level 1, 80 bins: from 0.50 to 0.96 of the bin width in 12 h) and MNFIX
starts splitting bins again.

**Measured (2026-10-15, laptop CPU, upwind version).** Interior extrema of
the final dN/dlog10Dp (local maxima and minima above 5 % of the peak),
MNFIX → PPM:

| Run | 40 bins | 80 bins | 160 bins |
|---|---|---|---|
| Level 1, SOA only | 3 → 1 | 13 → 1 | 30 → 1 |
| Level 1 preset (SOA + coagulation) | 3 → 1 | 9 → 1 | 28 → 1 |
| Level 2 preset | 3 → 1 | 5 → 1 | 18 → 1 |
| Level 3 preset | 3 → 1 | 7 → 1 | 17 → 1 |

With PPM the count is 1 at every output time of these runs (MNFIX: up to
36). Number-weighted mean diameter of the final dN/dlog10Dp (bin-centre
Dp, display density), PPM / MNFIX:

| Preset | 40 bins | 80 bins | 160 bins |
|---|---|---|---|
| Level 1, SOA only (12 h) | 122.0 / 100.0 nm | 122.7 / 111.5 | 123.0 / 119.9 |
| Level 1 preset | 127.9 / 105.8 | 128.5 / 117.4 | 128.7 / 125.7 |
| Level 2 preset (6 h) | 214.0 / 178.9 | 213.5 / 197.1 | 212.9 / 209.2 |
| Level 3 preset (24 h) | 196.2 / 177.7 | 194.7 / 188.8 | 194.2 / 196.4 |

PPM changes by ≤ 1 % between 40 and 160 bins; MNFIX by 10–20 %. Against
PyCHAM, see [pycham_comparison.md](pycham_comparison.md) (S1 and S2).

Conservation in the preset runs (checked after every outer step, 40/80/160
bins): gas + particle per SOA class ≤ 4e-15 relative, particle nitrate vs
`nitrate_added` ≤ 2e-15, nucleated organic ≤ 6e-14, sulfate (gas +
particle + source) ≤ 7e-11 (MNFIX's empty-bin floor; the same with
`'mnfix'`); number in the SOA-only runs ≤ 5e-13; no negative or
non-finite values; no populated bin (Nk > 1e-6 of the peak) with its mean
outside its bin. The largest displacement of a populated bin in one
remap is 0.03 / 1.6 / 1.9 bins (Level 1 / 2 / 3, 160 bins).

End-of-run totals, PPM vs MNFIX: SOA within 0.22 % in all presets at 40,
80 and 160 bins (largest: Level 1 at 40 bins, +0.21 %). Level 3 nitrate:

| | 40 bins | 80 bins | 160 bins | 320 bins |
|---|---|---|---|---|
| PPM | 5.180 | 5.164 | 5.156 | 5.153 µg m⁻³ |
| MNFIX (the previous code) | 4.962 | 5.075 | 5.139 | 5.149 |
| PPM vs MNFIX | +4.4 % | +1.7 % | +0.33 % | +0.07 % |

Both converge to about 5.15 µg m⁻³. At 40 bins PPM is 0.6 % above the
320-bin MNFIX value and the previous code 3.6 % below it. N₂O₅ uptake
follows the particle surface, which the bin transfer changes; the +4.4 %
at 40 bins is the old scheme's resolution error, not a new one. The 2 %
acceptance limit against the old code is therefore not met at 40 bins
(it is at 80 bins and above); see PROGRESS.md for the open decision.

Runtime: the PPM path is not slower, because MNFIX (a sequential loop
over the bins) no longer runs after every vbs-jax sub-step. Warm runs on a
laptop CPU, median of 3, PPM and MNFIX interleaved in one process (load
average 3–4):

| Preset | 40 bins | 80 bins |
|---|---|---|
| Level 1 (12 h) | 1.75 → 1.09 s | 6.17 → 4.26 s |
| Level 2 (6 h) | 6.05 → 5.75 s | 8.35 → 7.51 s |
| Level 3 (24 h) | 13.46 → 12.26 s | 22.77 → 18.74 s |

(MNFIX → PPM; MNFIX is bit-identical to the code before the PPM remap.)

The step still compiles once per (nbins, K, dt_outer, micro_dt, processes,
cfg); the remap's sub-step count is a traced loop bound.

**Limitations.**
- PPM is a one-moment reconstruction; the bin means are kept inside their
  bins by `keep_means_in_bins`, which moves a few percent of a bin when it
  acts. It acts mostly in the sparse tails.
- Grow-then-evaporate round trips do not return to the seed shape.
  TOMAS keeps one composition per bin, and the remap moves species with
  the donor bin's mean composition (as the H₂SO₄ PPM scheme does), so
  particles with different cores that grow into the same bins share
  their composition. Test: Level-1-like growth (c\* = 1, 2 µg m⁻³ h⁻¹,
  6 h at 288 K), then +30 K at once and 3 h of evaporation. Kept cores
  would give a number-mean dry diameter of ~63 nm and GSD 1.6 at the
  end. PPM gives 71.4 / 70.2 nm and GSD 1.35 / 1.30 at 40 / 160 bins
  (the core-size spread carried by the bins is already 1.32–1.36 after
  the growth, at either resolution); MNFIX gives 42.0 / 62.7 nm and GSD
  2.66 / 1.57. PPM conserves everything and moves the mode down
  monotonically; it keeps one extremum at 40 and 80 bins, while at 160
  bins this abrupt +30 K step leaves 5 small extrema (MNFIX 22). The
  presets (gradual temperature change in Level 3) keep one extremum.
- One remap per call moves particles with the displacement of the bin
  they are in at each sub-step, so a bin that has to move several bins in
  one 60 s step passes through bins with other displacements. Example: a
  pure c\* = 100 SOA mode (no vapour) next to a sulfate mode loses 75 % of
  its mass in the first step; 0.40 % of the sulfate ends below bin 40 (80
  bins), against 0.17 % for an exact remap of the bins' new mean masses
  and 0.21 % with MNFIX. Where a bin holds both kinds of particle, all of
  them move with the bin's mean change (TOMAS keeps one composition per
  bin).
- Nothing leaves the grid. At the top, the in-place mass that would push
  the top bin past 17.5 µm is taken back (`cap_top_bin`: SOA stays in the
  gas, nitrate is not added and the driver warns), so MNFIX never trims
  the top bin's sulfate. At the bottom, a bin-0 mean below 1.7 nm is
  handled by MNFIX as before (it trims number, keeps mass). The tutorial
  presets never reach either end.
- Bins with Nk ≤ 1e-10 #/cell are not moved; a change they received stays
  in place.
- One remap per 60 s micro-step: within a micro-step vbs-jax sees the
  particles at their in-place mean mass, which is the bin they started in.

## Verified behaviour (`tests/test_aaar_aerosol.py`, `tests/test_aaar_soa_ppm.py`)

- 12 h with no sources: no NaN, dry mass (particles + gas) conserved to
  1e-10, and N(t) identical to a coagulation-only run.
- Per-class gas + particle mass is conserved to 1e-10 over a 600 s step
  with all processes and sources on.
- Level-1-like 12 h run (1e5 cm⁻³ s⁻¹ H₂SO₄, 1 µg m⁻³ h⁻¹ surrogate): C\* = 1
  gives ~11 µg m⁻³ SOA, C\* = 1000 stays in the gas phase. About 2–3 s per
  12 h on a laptop CPU after a ~1 s compile.
- PPM remap alone (40/80/160 bins, growth by +150 % and evaporation of 90 %
  of the SOA): number and every species conserved to 1e-12, no negative
  values, every populated bin's mean inside its bin, one extremum, each
  particle keeps its composition; a uniform ×1.7 mass change moves the
  mode by ln 1.7 within 2 %; no change gives the input back.
- Both settings: Level-1-like growth (80 bins, 12 h) conserves the class
  budget (1e-10) and number (1e-10); PPM keeps one extremum at every step,
  MNFIX ends with more than two. SOA evaporation (c\* = 10, no vapour) moves
  the mode down to the seed size (within 5 %), smoothly with PPM, with
  conservation every step. Zero-OA bootstrap and the c\* = 1000 no-spike
  check behave the same with both. Nitrate injection conserves nitrate,
  number and sulfate. Pankow / vbs-jax equilibrium and the all-process
  class budget are tested with both settings.
- Unchanged particles stay put: SOA-only bins losing 99 % next to
  unchanged sulfate bins move down, the sulfate bins keep their number
  and mass to 1e-12 and no sulfate ends below them.
- Top of the grid: particles in the top three bins with 2000 µg m⁻³ of
  c\* = 0.01 vapour (12 × 600 s) close the class budget to 1e-12 every
  step with both settings; with PPM the top-bin mean stays inside the
  grid and no sulfate is lost. Nitrate pushed onto top-bin particles is
  partly not added, and `nitrate_added` equals the particle nitrate.
- +30 K after 6 h of growth (40 bins): class budget 1e-12 every step,
  mean diameter falls monotonically with both settings; PPM conserves
  number and keeps one extremum.
