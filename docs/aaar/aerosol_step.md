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
   the bins with particles (Nk > `NITRATE_N_MIN` = 2e-5 #/cell, just
   above MNFIX's empty-bin floor of 1e-5 #/cell), each scaled by
   Nk/Nk(start of the outer step), renormalised: the weights are shares
   of the outer step's initial number, so a bin keeps its weight per
   particle as particles move in or out during the outer step. An empty
   bin (N = 0 or the MNFIX floor) gets none. The particles that gained
   nitrate then move between bins by the PPM remap (see
   [Bin transfer](#bin-transfer-of-soa-and-nitrate-growth_bin_transfer)).
   Then MNFIX. Until 2026-10-16 the gate was 1e10 #/cell (1 cm⁻³) and
   the shares were not scaled: tail bins below 1 cm⁻³ got no nitrate,
   stopped growing at night while their neighbours moved on, and left a
   band of a few cm⁻³ at 40–90 nm in Level 3 that grew with resolution
   (N at 40–70 nm at 24 h: 3.1 / 3.2 / 4.3 / 6.6 cm⁻³ at 40 / 80 / 160 /
   320 bins; now 2.7 / 0.93 / 0.35 / 0.10, and the 0.1 % quantile
   diameter 68 / 69 / 62 / 60 nm → 73 / 83 / 82 / 83 nm). Nitrate total,
   N and median diameter changed by ≤ 0.03 %, ≤ 0.02 % and ≤ 0.17 %. A
   lower gate alone (e.g. the remap's 1e-10) also removes the band, but
   then MNFIX floor bins take nitrate, and a bin that empties inside a
   long outer step piles its start-of-step share onto the floor particles
   (MNFIX then trims it: −0.4 % nitrate in
   `test_nitrate_injection_conserved[mnfix]`, 1 h outer step); the
   floor-aware gate and the per-particle scaling prevent both.
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
| `boxvol_cm3` | 1e10 | tutorial box volume. TOMAS thresholds are per cell (NEPS_CONDSINK = 1e10 #/cell picks a bin's own or a default size for sink geometry), so this makes that threshold 1 cm⁻³. Nitrate placement uses 2e-5 #/cell, SOA partitioning and the remap 1e-10 #/cell |
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
belongs to. `physics/ppm_remap.py::remap_mass_change` does this with the
PPM sub-bin profile that H₂SO₄ condensation uses
(`physics/condensation_ppm.py::ppm_reconstruct`).

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
   s = ln(m_new/m_old) in log-mass (s < 0 for evaporation).
2. Inside each bin, s is taken as linear in ln m through the bin mean,
   with the smaller of the two one-sided slopes to the neighbouring bin
   means (zero next to a neighbour without particles or with the same s),
   and the bin-edge values are clipped between the bin's s and the
   neighbour's. The bin then maps onto an image interval in ln m. Where s
   varies smoothly the images of neighbouring bins join (stretched for
   evaporation, squeezed for growth). At a jump in s (two particle
   populations side by side) each bin keeps its own s, so particles whose
   mass did not change stay exactly where they are while others grow or
   shrink past them.
3. Number inside each source bin follows its PPM reconstruction (monotone
   and positive parabola in ln m), carried onto the image. Each target
   bin receives the integral of that profile over the part of the image
   inside it (exact polynomial integrals); the mass share is the same
   integral weighted by the particle's new mass. Every bin moves with its
   own displacement however many bins that is: there is no Courant limit
   and no sub-stepping. For a uniform shift smaller than one bin this is
   exactly the flux-form PPM step of H₂SO₄ condensation. Particles mapped
   below or above the grid go into the bottom or top bin.
4. Composition. Every species of a source bin moves with that bin's mass
   share, and in addition each species' mass fraction is linear inside the
   source bin (monotonized-central slope against the neighbouring bins'
   fractions, slopes summing to zero, one common limiter so no fraction
   goes negative). Particles therefore keep the composition that goes with
   their size, which is what makes a grow-then-evaporate round trip come
   back to the right sizes (below).
5. Safety net (`keep_means_in_bins`). PPM reconstructs number only, so a
   bin mean can still land past an edge of its target bin. Such a bin
   hands the fewest possible particles to the neighbour at the
   neighbour's mean mass. In the growth and evaporation tests below the
   particles it moves add up to at most ~1e-3 of the total over a run.
6. Top bin (`cap_top_bin`): particles that grow in the top bin stay
   there. If its mean passes the top edge, just enough of the mass added
   in place is taken back (SOA returns to the vapour; nitrate is not
   added and the driver warns), so MNFIX never trims the bin, which would
   delete seed mass as well.

Each source bin's shares sum to one and its composition-slope term to
zero, so number and the mass of every species are conserved to
round-off, and nothing goes negative. MNFIX still runs where it did
before (after the sources, inside nucleation/coagulation/condensation) and
once after the SOA remap, where it restores its floor (1e-5 #/cell) in
bins the remap emptied. Any SOA it deletes there goes back to the vapour,
so gas + particle mass closes per class with either setting. An image is
kept between 0.25 and 4 bin widths (a rigid shift if it would fold,
squeezed around the image of the bin mean if wider), so each bin touches
at most 6 target bins and the work per call is a small band; a call with
nothing to move (no nitrate this step) is skipped.

Choices that were tried and dropped:

- *Flux-form PPM with a fixed velocity per edge* (the first versions).
  Each edge kept the donor bin's displacement for all sub-steps, so a bin
  that had to move several bins took the displacement of the bins it
  passed through. Next to unchanged particles it piled up: SOA-only
  particles in bins 30–39 growing ×10 next to unchanged sulfate particles
  in bins 40–49 (80 bins) reached 8.25e12 #/cell in bin 40 against 2e12,
  and 3 % of the sulfate moved. In slow Kelvin ripening and after a +30 K
  step at 160 bins, bin means crept to their edges and the safety net and
  MNFIX left 5–11 extrema. The remap gives 2.0e12, no sulfate moved, and
  the cases below.
- *Edge displacement interpolated between neighbouring means.* Drags
  unchanged particles next to a shrinking neighbour (10 % of them moved in
  the example above with 99 % mass loss).
- *TOMAS's `dmdt_int` law around each bin's own mean* (as `ezcond_ppm_jax`
  does for H₂SO₄). It slows the upper edge of a bin more than the vbs-jax
  rates do, so number lags mass and every bin mean drifts to its upper
  edge (Level 1, 80 bins: 0.50 → 0.96 of the bin width in 12 h).
- *Tilting each bin's PPM profile so its mean matches the bin mean.* Made
  the number and mass shares consistent but piled particles up at steep
  fronts (nitrate injection at 80 bins: 3 extrema instead of 1).
- *One mean composition per source bin* (no composition slopes). Smooth,
  but a grow-then-evaporate round trip mixes compositions bin by bin and
  comes back too narrow (see the round-trip numbers below).

**Measured (2026-10-15, laptop CPU).** Interior extrema of the final
dN/dlog10Dp (local maxima and minima above 5 % of the peak), MNFIX → PPM:

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
| Level 1, SOA only (12 h) | 123.0 / 100.0 nm | 123.2 / 111.5 | 123.2 / 119.9 |
| Level 1 preset | 128.7 / 105.8 | 128.9 / 117.4 | 128.9 / 125.7 |
| Level 2 preset (6 h) | 213.3 / 178.9 | 213.2 / 197.1 | 213.1 / 209.2 |
| Level 3 preset (24 h) | 195.4 / 177.7 | 194.3 / 188.8 | 194.2 / 196.4 |

PPM changes by ≤ 0.6 % between 40 and 160 bins; MNFIX by 10–20 %. Against
PyCHAM, see [pycham_comparison.md](pycham_comparison.md) (S1 and S2).

Growth and evaporation against moving sections. The reference runs the
same vbs-jax physics with no bin transfer at all (every section keeps its
particles; 320 sections). Number-mean dry diameter at the end:

| Case | Moving sections | PPM 40 / 80 / 160 bins | flux-form PPM (before) | MNFIX |
|---|---|---|---|---|
| 6 h growth at 288 K (c\* = 1, 2 µg m⁻³ h⁻¹), then +30 K for 3 h | 59.2 nm, GSD 1.61 | 59.0 / 59.9 / 59.5, GSD 1.71 / 1.59 / 1.58 | 67.6 / – / 66.4 (40 / 160) | 39.7 / – / 59.3 |
| the same with +20 K | 120.2, GSD 1.91 | 115.3 / 120.4 / 120.1, GSD 2.46 / 1.92 / 1.87 | 138.5 / 138.9 (40 / 80) | 80.9 / 105.6 (40 / 80) |

(1770 kg m⁻³ for the diameter.) With +30 K the PPM count is 1 at every
step at all three resolutions (the flux-form version had 5 at 160 bins,
MNFIX 13–22). With +20 K, 80 and 160 bins keep 1; at 40 bins the
mass-doubling bins cannot resolve the shoulder the moving sections form
at 45–70 nm (small particles that Kelvin drives back to their cores) and
show it as a plateau with 2–7 small extrema. Zero-OA bootstrap followed
by Kelvin ripening (pure sulfate seed, 2000 cm⁻³ at 80 nm, 5 µg m⁻³ of
c\* = 1 vapour, 6 h at 298 K) forms a real second mode at 45 nm in the
moving sections; PPM reproduces it with the same 3 extrema at 40, 80 and
160 bins (the flux-form version had 1 / 3 / 11, with alternating ±10–33 %
bins).

Conservation in the preset runs (checked after every outer step, 40/80/160
bins): gas + particle per SOA class ≤ 4e-15 relative, particle nitrate vs
`nitrate_added` ≤ 2e-15, every other species except sulfate ≤ 4e-14,
sulfate (gas + particle + source) ≤ 7e-11 (MNFIX's empty-bin floor; the
same with `'mnfix'`); number in the SOA-only runs ≤ 5e-13; no negative or
non-finite values; no populated bin (Nk > 1e-6 of the peak) with its mean
outside its bin. Per remap call inside the presets (1440–2880 calls):
number ≤ 1.4e-15 and every species ≤ 1.2e-15 relative.

End-of-run totals, PPM vs MNFIX: SOA within 0.21 % in all presets at 40,
80 and 160 bins (largest: Level 1 at 40 bins, +0.21 %). Level 3 nitrate:

| | 40 bins | 80 bins | 160 bins | 320 bins |
|---|---|---|---|---|
| PPM | 5.160 | 5.152 | 5.149 | 5.149 µg m⁻³ |
| MNFIX (the previous code) | 4.962 | 5.075 | 5.139 | 5.149 |
| PPM vs MNFIX | +4.0 % | +1.5 % | +0.20 % | −0.01 % |
| PPM, nitrate gate fix (2026-10-16) | 5.160 | 5.151 | 5.148 | 5.148 |
| MNFIX, nitrate gate fix | 4.961 | 5.075 | 5.138 | 5.148 |

Both converge to 5.149 µg m⁻³ (5.148 with the 2026-10-16 nitrate gate,
see Micro-step). At 40 bins PPM was 0.21 % above the 320-bin
value before the gate fix (0.23 % after) and the previous code 3.6 % below it. N₂O₅ uptake follows the
particle surface, which the bin transfer changes; the +4.0 % at 40 bins
is the old scheme's resolution error, not a new one. The 2 % acceptance
limit against the old code is not met at 40 bins (it is at 80 bins and
above). **Decision (2026-10-15):** totals are judged against the converged
(320-bin) value, not against the previous code, whose accuracy was never
established for nitrate (N₂O₅ uptake was off in every PyCHAM case). PPM is
within 0.21 % of it at 40 bins.

**Abrupt evaporation.** A sudden +35 K step after strong growth (c* about
0.3 → 100 at once) leaves ripples of up to 14 % of the peak (3 / 5 / 12
extrema at 40 / 80 / 160 bins) in the first hot step, although mean, width
and conservation are right. One 60 s remap cannot carry such a large,
strongly size-dependent shrink in one go; `micro_dt = 10` s removes the
ripples (+25 K and +30 K steps are smooth at 60 s). The presets never do
this (temperature is constant within a run). `driver.run(...,
aerosol_cfg=AerosolStepConfig(micro_dt=...))` sets the micro-step.

Runtime: the PPM path is faster, because MNFIX (a sequential loop over the
bins) no longer runs after every vbs-jax sub-step. Warm runs on a laptop
CPU, median of 3, PPM and MNFIX interleaved in one process (load average
4–5 from other work):

| Preset | 40 bins | 80 bins |
|---|---|---|
| Level 1 (12 h) | 1.77 → 1.15 s | 6.41 → 4.48 s |
| Level 2 (6 h) | 6.15 → 5.74 s | 8.54 → 7.56 s |
| Level 3 (24 h) | 13.50 → 12.42 s | 23.27 → 19.75 s |

(MNFIX → PPM; MNFIX is bit-identical to the code before the PPM remap.)
One remap call costs about 0.05 / 0.09 / 0.14 ms at 40 / 80 / 160 bins
(timed on its own).
The step still compiles once per (nbins, K, dt_outer, micro_dt,
processes, cfg).

**Limitations.**
- PPM reconstructs number only. A bin mean that lands outside its target
  bin is pulled back by `keep_means_in_bins` (it moves few particles, see
  above). Displacement and composition are linear inside each source bin,
  so structure finer than a bin is not kept: on the 40-bin mass-doubling
  grid a shoulder narrower than about two bins shows as a plateau with
  small wiggles (+20 K case above).
- Where a bin holds two kinds of particle, all of them move with the
  bin's mean change (TOMAS keeps one composition per bin; the slopes only
  carry a linear trend across the bin).
- Nothing leaves the grid. At the top, the in-place mass that would push
  the top bin past 17.5 µm is taken back (`cap_top_bin`: SOA stays in the
  gas, nitrate is not added and the driver warns), so MNFIX never trims
  the top bin's sulfate. At the bottom, a bin-0 mean below 1.7 nm is
  handled by MNFIX as before (it trims number, keeps mass). The tutorial
  presets never reach either end.
- A bin's image is limited to 4 bin widths: a bin whose particles spread
  over more than four times its width in one call (not seen in any test)
  is squeezed around the image of its mean.
- Bins with Nk ≤ 1e-10 #/cell are not moved; a change they received stays
  in place.
- One remap per 60 s micro-step: within a micro-step vbs-jax sees the
  particles at their in-place mean mass, which is the bin they started in.

## Verified behaviour (`tests/test_aaar_aerosol.py`, `tests/test_aaar_soa_ppm.py`, `tests/test_aaar_nitrate_gate.py`)

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
- Particles move past unchanged ones: SOA-only particles in bins 30–39
  (80 bins) grow ×3 or ×10, or SOA-only particles in bins 50–59 keep 25 %
  or 1 % of their mass, next to unchanged sulfate particles in bins
  40–49. The sulfate keeps its number and mass per bin to 1e-14, the SOA
  particles' mean bin lands within 0.02 bins of the exact position (3 to
  13 bins away) and no bin exceeds the two populations' 2e12 #/cell by
  more than a few percent (`test_remap_moves_particles_past_unchanged_ones`).
- Multi-bin shifts: ×10 and ÷10 at once at 160 bins (13.3 bins) move the
  mode by ln 10 with its width kept to 1e-6 and one extremum. The mass
  integrals φ_j agree with an exact series to 1e-14.
- Top of the grid: particles in the top three bins with 2000 µg m⁻³ of
  c\* = 0.01 vapour (12 × 600 s) close the class budget to 1e-12 every
  step with both settings; with PPM the top-bin mean stays inside the
  grid and no sulfate is lost. Nitrate pushed onto top-bin particles is
  partly not added, and `nitrate_added` equals the particle nitrate.
- Nitrate placement (`tests/test_aaar_nitrate_gate.py`): one micro-step
  with both settings; a 0.5 cm⁻³ bin and a 1e-13 cm⁻³ bin get their
  shares to 1e-10, bins with N = 0, at the MNFIX floor or below the gate
  get none whatever their weight, `nitrate_added` equals the particle
  nitrate to 1e-12, no non-finite values. Slow: Level 3 at 80 bins has
  < 1.5 cm⁻³ at 40–70 nm at 24 h (3.2 with the old gate).
- +30 K after 6 h of growth (40 and 160 bins with PPM, 40 with MNFIX):
  class budget 1e-12 every step, mean diameter falls monotonically with
  both settings; PPM conserves number and keeps one extremum at every step.
- Against moving sections at 80 bins (`test_soa_matches_moving_sections`):
  grow 3 h at 283 K then 1 h at 313 K, and zero-OA bootstrap with Kelvin
  ripening (4 h); number-mean dry diameter within 0.3 % and GSD within
  0.5 %, one extremum every hour in the round trip.
