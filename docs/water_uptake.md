# Aerosol Water Uptake

Aerosol water is a diagnostic species (`Mk[:, SRTH2O]`). It is not moved by
coagulation or condensation; it is recomputed from the dry composition and RH.
Wet mass and density set the particle sizes used by coagulation
(`calc_particle_properties`), the condensation sink (`calc_condensation_sink`)
and condensational growth (PPM/TFL wet/dry ratio), so water uptake affects all
three.

## Schemes

Selected with `equilibrate_water(Mk, rh, temp, water_scheme)` in
`tomas_jax/physics/water_equilibrium.py`.

| `water_scheme` | Function | Physics | Use for |
|---|---|---|---|
| `WATER_SCHEME_BISULFATE` (0.0, default) | `calc_equilibrium_water` | TOMAS `ezwatereqm.f`/`waterso4.f`: polynomial fits to ISORROPIA results for ammonium bisulfate at 273 K. SO4 × 1.2 (as NH4HSO4) and organics share one wet/dry mass ratio. No temperature dependence. | Tropospheric aerosol with ammonia (TOMAS default) |
| `WATER_SCHEME_H2SO4` (1.0) | `calc_equilibrium_water_h2so4` | Tabazadeh et al. (1997) binary H2SO4/H2O equilibrium. SO4 × 98/96 (as H2SO4) takes the water of a solution at the equilibrium wt%(T, RH). Organics keep the bisulfate-fit ratio; NH4 takes no water. | Pure sulfuric acid aerosol (no ammonia), e.g. stratospheric sulfate |

Water per kg SO4 for comparison:

| | RH 10% | RH 50% | RH 90% |
|---|---|---|---|
| Bisulfate fit (any T) | 0.06 | 0.35 | 2.40 |
| Tabazadeh, 260 K | 0.63 | 1.48 | 4.87 |
| Tabazadeh, 215 K | 0.72 | 1.66 | 4.84 |

### Tabazadeh weight percent

`tabazadeh_h2so4_wt(temp, rh_percent)` finds the wt% at which the solution
water vapour pressure (Table 1, `ln P = a + b/T + c/T²` per wt% row) equals
`RH/100 × P_sat(T)` (Eq. 1), by linear interpolation in `-ln P`. It is
JIT-safe and matches the numpy `radiative_forcing.h2so4_equilibrium_wt`
exactly inside the fitted range. The table (`TABAZADEH_TABLE1`,
`TABAZADEH_EQ1`) lives in `water_equilibrium.py` and is shared with
`radiative_forcing.py`.

- Fitted for **T = 185–260 K** and **10–80 wt%**. Outside that temperature
  range the row formula is extrapolated, not clamped. At 298 K / RH 50% it
  gives 43.9 wt%, close to 25 °C water-activity data (~43 wt%).
- RH is clipped to [1e-3, 99] %, so wt% stays in [10, 80] and water is at
  most 9× the acid mass.

## When water is computed

By default water is set only at the end of each condensation step (matching
the bundled Fortran harness `benchmark_24h.f`). In that mode:

- coag-only runs never have water, so RH has no effect on them;
- in coag+cond and full runs, coagulation uses the previous step's water and
  the first step is dry;
- after coagulation or nucleation, the water in each bin is stale until
  condensation recomputes it.

`water_every_process=True` re-equilibrates water after nucleation,
coagulation and dilution as well as condensation, the order used by the
TOMAS box model (`TRACER_SOM-TOMAS/src/box.f` calls `ezwatereqm` after
`multicoag`, after nucleation and after condensation). Callers should also
equilibrate the initial distribution so the first step is wet.

Both options are available on `make_step` (factory arguments) and on the
step and scan functions (static arguments, like `use_tfl`):

```python
from tomas_jax.physics.water_equilibrium import WATER_SCHEME_H2SO4, equilibrate_water
from tomas_jax.solvers.condensation import make_step

Mk = equilibrate_water(Mk, rh, temp, WATER_SCHEME_H2SO4)   # wet initial state
step = make_step(['coagulation', 'condensation'],
                 water_scheme=WATER_SCHEME_H2SO4, water_every_process=True)
Nk, Mk, Gc = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt)
```

The defaults (`WATER_SCHEME_BISULFATE`, `water_every_process=False`) leave
every path bit-identical to before these options existed, so Fortran-harness
comparisons are unaffected. tomas-api turns `water_every_process` on.

## Effect

Single 100 nm sulfate mode (N = 1e5 cm⁻³, GSD 1.6), coag-only, 298 K,
24 h, 40 bins, final N in cm⁻³ (initial state equilibrated when
`water_every_process=True`):

| RH | Default (dry) | `water_every_process`, bisulfate | `water_every_process`, H2SO4 |
|---|---|---|---|
| 10% | 1.5676e4 | 1.5806e4 | 1.6841e4 |
| 50% | 1.5676e4 | 1.6446e4 | 1.7810e4 |
| 90% | 1.5676e4 | 1.8655e4 | 1.9451e4 |

At 90% RH the mode grows to 1.75× its dry diameter. For two similar ~100 nm
particles the Brownian kernel falls as they grow, so wet particles coagulate
more slowly here. With a nucleation mode present, larger wet particles
scavenge small ones faster, so the sign of the effect depends on the
distribution.

## Assumptions and limitations

- Instantaneous equilibrium; no deliquescence/efflorescence hysteresis.
  Particles are always on the wet branch.
- Bisulfate scheme: one ISORROPIA fit at 273 K for all temperatures; SO4 is
  always treated as NH4HSO4, whatever the actual NH4 content.
- H2SO4 scheme: assumes binary H2SO4/H2O. NH4 present (NH3 > 0) is not
  accounted for. Organics use the bisulfate-fit ratio, not a
  composition-specific hygroscopicity.
- Nucleated particles are 10% organic by mass (nucleation.py), so organics
  appear even when organic vapour is zero.
- Sea salt water uptake (`water_uptake_seasalt`) is ported but not used.
- The legacy numpy `condensation_step(method='tfl'|'ppm')` always uses the
  bisulfate scheme.
- The GPU-fast model (`gpu-fast` branch, `fast/water.py`) has its own
  batched Tabazadeh implementation; it can switch to
  `tabazadeh_h2so4_wt` when that branch merges.

## References

- Tabazadeh, A., O. B. Toon, S. L. Clegg, and P. Hamill (1997), A new
  parameterization of H2SO4/H2O aerosol composition: Atmospheric
  implications, GRL, 24(15), 1931–1934.
- Nenes, A., S. N. Pandis, and C. Pilinis (1998), ISORROPIA, Aquatic
  Geochemistry, 4, 123–152.
- TOMAS `ezwatereqm.f` (P. Adams, 2000) and `waterso4.f` (P. Adams, 2001).
