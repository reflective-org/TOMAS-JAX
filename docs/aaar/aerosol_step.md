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
2. s is interpolated linearly in ln m between neighbouring bin means to
   the bin edges (no flux through the two grid ends).
3. Number moves by flux-form PPM in ln m (`ppm_reconstruct`, the same
   monotone, positive reconstruction as H₂SO₄ condensation), sub-stepped
   at Courant ≤ 0.8. Every species moves with the PPM mass-weighted
   fraction of the donor bin, so all species share one flux and the
   gained (or lost) mass travels with the particles that carry it.
4. A bin whose mean still ended up past an edge (possible at steep edges
   of the distribution, because PPM reconstructs number only) keeps its
   particles just inside the edge and hands the fewest possible particles
   to the neighbour at the neighbour's mean mass (`keep_means_in_bins`).

Number and the mass of every species are conserved to round-off (flux
differences); no bin can go negative. Growth (s > 0) moves particles up,
evaporation (s < 0) moves them down. MNFIX still runs where it did before
(after the sources, inside nucleation/coagulation/condensation); after the
remap it finds nothing to split in populated bins.

Why the edge displacement is interpolated rather than taken from TOMAS's
`dmdt_int` law around each bin's own mean (as `ezcond_ppm_jax` does for
H₂SO₄): `dmdt_int` assumes d(m^2/3)/dt is constant, which slows the
upper edge of a bin more than the vbs-jax rates do. Number then lags mass,
every bin's mean drifts to its upper edge (Level 1, 80 bins: from 0.50 to
0.96 of the bin width in 12 h) and MNFIX starts splitting bins again.

**Measured (2026-10-15, laptop CPU).** Interior extrema of the final
dN/dlog10Dp, MNFIX → PPM:

| Run | 40 bins | 80 bins | 160 bins |
|---|---|---|---|
| Level 1, SOA only | 3 → 1 | 13 → 1 | 30 → 1 |
| Level 1 preset (SOA + coagulation) | 3 → 1 | 9 → 1 | 28 → 1 |
| Level 2 preset | 3 → 1 | 5 → 1 | |
| Level 3 preset | 3 → 1 | 7 → 1 | 17 → 1 |

With PPM the count is 1 at every output time of these runs. Level 1
number-mean diameter at 12 h (SOA only, display density): 121.7 / 122.6 /
122.9 nm at 40 / 80 / 160 bins (MNFIX: 100.0 / 111.5 / 119.9). Against
PyCHAM (case S1, c\* = 1, [pycham_comparison.md](pycham_comparison.md)):
number-mean dry diameter 122.4 / 122.7 / 123.0 nm at 40 / 80 / 160 bins vs
PyCHAM 120.5 nm (MNFIX: 105.3 / 115.1 / 121.3); with Kelvin off in both and
PyCHAM's transfer coefficient, 138.9 vs 138.1 nm. Level 2 (S2) at 6 h,
40 bins: number-median dry diameter 184 nm (MNFIX 156, PyCHAM 181),
dry-mass median 237 nm (MNFIX 272, PyCHAM 233); particles below the
seed's 1st-percentile diameter 0.2 % (MNFIX 2.1 %).

End-of-run totals change little: SOA within 0.25 % of the MNFIX scheme in
all three presets at 40 and 80 bins (largest: Level 1 at 40 bins, +0.21 %).
Level 3 nitrate is +4.4 % at 40 bins and +1.7 % at 80 bins, because the
corrected distribution has 15–23 % more dry surface area at 40 bins (more
N₂O₅ uptake; with the SOA remap alone and nitrate still placed by MNFIX it
is +4.1 %). The MNFIX scheme's nitrate moved with resolution (4.96 / 5.08 /
5.14 µg m⁻³ at 40 / 80 / 160 bins); the PPM scheme gives 5.18 / 5.16 /
5.16.

Runtime: the PPM path is faster, because MNFIX (a sequential loop over the
bins) no longer runs after every vbs-jax sub-step. Warm runs of the
presets on a laptop CPU, MNFIX → PPM:

| Preset | 40 bins | 80 bins |
|---|---|---|
| Level 1 (12 h) | 1.75 → 1.20 s | 6.14 → 4.28 s |
| Level 2 (6 h) | 5.91 → 5.64 s | 8.41 → 7.50 s |
| Level 3 (24 h) | 13.3 → 12.3 s | 22.4 → 18.8 s |

The step still compiles once per (nbins, K, dt_outer, micro_dt, processes,
cfg); the remap's sub-step count is a traced loop bound.

**Limitations.**
- PPM is a one-moment reconstruction; the bin means are kept inside their
  bins by `keep_means_in_bins`, which moves a few percent of a bin when it
  acts. It acts mostly in the sparse tails.
- Nothing leaves the grid: particles growing past the top bin or
  shrinking below the bottom bin stay in it. MNFIX then trims them as
  before (top: mass; bottom: number). The tutorial grid (1.7 nm–17.5 µm)
  is far from both ends.
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
