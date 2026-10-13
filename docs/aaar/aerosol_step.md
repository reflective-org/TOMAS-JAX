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
   bins with Nk > 1e10 #/cell, renormalised. Then MNFIX.
2. **Nucleation → coagulation → H₂SO₄ condensation**:
   `solvers.condensation.make_step` with the enabled subset, `jit=False`,
   PPM-JIT condensation and `cfg.water_scheme`/`cfg.water_every_process`.
   H₂SO₄ condensation also does NH₃ equilibrium (`Gc[SRTNH4]`).
3. **SOA partitioning**: `soa_partition_step` ([soa_partitioning.md](soa_partitioning.md)).
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

The driver stream may add its own classes to `box/config.py`.

## Verified behaviour (`tests/test_aaar_aerosol.py`)

- 12 h with no sources: no NaN, dry mass (particles + gas) conserved to
  1e-10, and N(t) identical to a coagulation-only run.
- Per-class gas + particle mass is conserved to 1e-10 over a 600 s step
  with all processes and sources on.
- Level-1-like 12 h run (1e5 cm⁻³ s⁻¹ H₂SO₄, 1 µg m⁻³ h⁻¹ surrogate): C\* = 1
  gives ~11 µg m⁻³ SOA, C\* = 1000 stays in the gas phase. About 2–3 s per
  12 h on a laptop CPU after a ~1 s compile.
