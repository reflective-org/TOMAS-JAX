# Missing Physics & Chemistry — Gap Analysis

This document catalogs all physics and chemistry not yet implemented in TOMAS-JAX, organized by priority for building a complete atmospheric aerosol box model.

---

## What's Already Implemented

| Process | Status | Notes |
|---------|--------|-------|
| Brownian coagulation (Fuchs correction) | Done | Two JIT solvers (Tsit5 adaptive, forward Euler) |
| H₂SO₄ condensation (top-hat (TFL) + PPM) | Done | JIT-compiled, machine-precision conservation |
| Nucleation (Riccobono 2014 + Dunne 2016) | Done | Adaptive sub-stepping, JIT |
| Nucleation (Zhao 2024, 11 mechanisms) | Done | HNO₃, organic, amine, iodine pathways |
| NH₃/NH₄⁺ equilibrium | Done | Stoichiometric (eznh3eqm.f port) |
| Water uptake | Done | ISORROPIA piecewise polynomial fits |
| MNFIX mass-number correction | Done | Fortran-faithful, analytical multi-bin shift |
| Condensation sink | Done | First-order gas loss rate, bin-resolved |
| 40-bin / 80-bin grids | Done | 1.7nm start, configurable resolution |
| Dilution / entrainment | Done | First-order relaxation toward background, JIT |

---

## Tier 1 — Essential for Realistic Box Model

### 1. SO₂ → H₂SO₄ Gas-Phase Chemistry

**What:** Oxidation of SO₂ by OH to produce H₂SO₄ vapor. Currently TOMAS-JAX uses a constant H₂SO₄ production rate — no actual chemistry.

**Minimum viable implementation:**
```
d[H₂SO₄]/dt = k_OH+SO₂ × [OH] × [SO₂] − CS × [H₂SO₄]
```
where `k_OH+SO₂ ≈ 1.5e-12 cm³/s` at 298K (Atkinson et al.), and OH can be prescribed (diurnal cycle) or computed from a simplified mechanism.

**Fortran reference:** `saprc.f` has full SAPRC-99 (100+ reactions), but a simplified 3-reaction scheme (OH+SO₂, OH+DMS, photolysis) covers most box model needs.

**Effort:** 1–3 days for simplified scheme; 2–4 weeks for full SAPRC port.

---

### 2. VBS Organic Condensation / SOA Formation

**What:** Semi-volatile organic compounds (SVOCs) partition between gas and particle phase based on saturation concentration C*. The Volatility Basis Set (VBS) framework bins organics by C* (0.01, 0.1, 1, 10, 100, 1000 μg/m³).

**Key equations:**
```
ξᵢ = (1 + Cᵢ* / COA)⁻¹                          (partitioning coefficient)
Cᵢ* = Cᵢ*,ref × exp[ΔHvap/R × (1/T - 1/Tref)]   (T-dependence)
COA = Σ ξᵢ × Cᵢ,total                             (iterative: COA depends on ξ)
```

**Stiffness concern:** The partitioning equation is implicit (COA appears on both sides). Fortran uses iterative Newton-Raphson. In JAX, options include:
- `jax.lax.while_loop` Newton iteration (JIT-compatible)
- Implicit ODE solver via diffrax (treats partitioning as a stiff ODE)
- Analytical solution for 1-bin VBS (explicit formula exists)

**Fortran reference:** `soacond.f` (32KB) — VBS with Kelvin effect, equilibrium + kinetic partitioning.

**Effort:** 1–2 weeks.

---

### 3. Kelvin Effect (Curvature Correction)

**What:** Saturation vapor pressure over a curved droplet surface is higher than over a flat surface:
```
Cᵢ*(Dp) = Cᵢ*,flat × exp(4σv / RT Dp)
```
where σ is surface tension and v is molar volume. Growth is proportional to C − C*(Dp), so relative to a nonvolatile vapour (C* = 0) the growth rate is 1 − 1/S, with S = C / C*(Dp). The Kelvin term matters only where it brings S down towards 1.

**Status (2026-11-13):**

- **SOA (organic c*-classes): included.** vbs-jax applies the Kelvin factor (`vbs_jax/kelvin.py`, σ = 0.025 N/m) in the partitioning that `physics/soa_partitioning.py` calls.
- **H2SO4: deliberately not included.** `ezcond` / `ezcond_ppm_jax` / `condensation_tfl_jax` condense H2SO4 with C* = 0, as TOMAS Fortran does (Fortran parity). For the SAI plume below ~22 km this is justified; in background air it starts to matter at the top of that range:
  - **Method:** `benchmarks/python/h2so4_kelvin_driving_force.py` computes the equilibrium H2SO4 over sulfate–water droplets the way CARMA/WACCM does:
    - pure acid: Ayers et al. (1980) with the Kulmala & Laaksonen (1990) low-temperature extrapolation;
    - activity: a fit to Giauque et al. (1960) enthalpies;
    - composition (wt% H2SO4 in the droplet): Tabazadeh et al. (1997), from the RH;
    - Kelvin factor: σ = 0.08 N/m, the conservative end. At 1.7 nm (wet) it is ≈ 400–600.

    It plots the growth lost, 1/S = C*/C, against particle size, against altitude (AFGL tropical T, p and H2O, 13–22 km) and against H2SO4 at fixed altitudes. H2SO4 is the hourly mean of the Phase 3 runs, hours 2–48 (hour 1 is spin-up). `benchmarks/python/h2so4_mass_rate.py` plots the net condensation/evaporation rate (%/h) against H2SO4 for r_dry = 1–1024 nm.
  - **sai_plume** (Phase 3, 217 K, 5.5 kPa, 5 ppmv H2O, RH 0.9 %, 72 wt%; H2SO4 6e7–1e8 cm⁻³): growth lost ≤ 6e-4 at 1.7 nm, ~2e-5 at 3 nm, ~5e-7 for a flat surface.
  - **bg_midlat** (same conditions, 0.02 ppbv SO2; H2SO4 1.1e5–7.5e5 cm⁻³): ~3 % at 1.7 nm at the median, ~20 % at the lowest hour, < 1.3 % at 3 nm.
  - **Across 13–22 km** (AFGL tropical with its own H2O: 9.9 ppmv at 13 km, 2.6–2.9 ppmv at 17–22 km), at each run's lowest H2SO4:
    - C* at 1.7 nm is ~250 cm⁻³ at 13 km (RH 5.8 %, 63 wt%), ~1e-3 at the 17 km cold point (195 K, RH 18 %, 52 wt%) and ~3e4 at 22 km (215 K, RH 0.5 %, 74 wt%). Warmer, drier air means more concentrated droplets and a much higher C*.
    - Plume: ≤ 5e-4 at 1.7 nm anywhere in 13–22 km.
    - Background: up to 28 % at 1.7 nm and 2 % at 3 nm at 22 km. The margin shrinks quickly above ~20 km.
  - **Above ~22 km, or above ~240 K:** in background air (1.1e5 cm⁻³) 1.7–3 nm particles stop growing at ~23–27 km and flat-surface evaporation (the top of the Junge layer) sets in at ~28–32 km. Plume 1.7 nm particles keep growing to ~28–33 km. (These come from a 13–40 km version of the altitude panel, 4 ppmv H2O, NRLMSISE-00 above 25 km, which is not kept in the script.) Above ~28 km the RH falls below the Tabazadeh table (it clamps at 80 wt%), so C* there is a lower bound and composition needs a better treatment as well. A model run in that regime needs C* and the Kelvin term.
  - **250 K, 50 hPa** (Kleinschmitt 2017, Fig. 3.4): that temperature belongs to ~40 km, not 50 hPa. There C* is first order: at r_dry = 1 nm the zero-growth point is ~1.5e8 cm⁻³ and at 1e5 cm⁻³ the particles evaporate at ~1000 %/h. Her figure shows ~10× smaller rates and a 1 nm zero-growth point at ~1e7 cm⁻³. The difference is mainly her α = 0.1 (10× on every rate) and her Kelvin term, which uses the molecular mass of water instead of the H2SO4 molar volume (about 5× too small in the exponent). `h2so4_mass_rate.py --kleinschmitt` reproduces her figure and changes one setting at a time.
  - **Caveat:** the bulk Kelvin formula is rough at 1.7 nm. Cluster survival there is already part of the Dunne 2016 rate, which is defined at 1.7 nm.
- **Organics or other low-volatility vapours added later: the H2SO4 conclusion does not carry over.** This covers LVOC/ELVOC treated outside the c*-class partitioning, HIO3, NH3/HNO3 co-condensation and similar vapours. Their C* is many orders of magnitude above H2SO4 over sulfate solution, and the Kelvin factor applies to all of them.
  - **Example:** C* = 1e-3 µg m⁻³ (M ≈ 300 g/mol) is ≈ 2e6 molec cm⁻³. With a Kelvin factor of ~100 at 1.7 nm, S < 1 at typical ELVOC levels (1e6–1e7 cm⁻³).
  - **Consequence:** for these vapours the Kelvin term decides whether the smallest particles grow at all. This is the 10–100× effect on 1–5 nm growth seen in NPF studies, and it must be included.
  - **How:** multiply each bin's C* by the Kelvin factor in the condensation driving force. The factor depends only on bin diameter (from xk) and composition.

**Effort (if needed for H2SO4 above ~22 km in background air or above ~240 K, or for new low-volatility species):** 2–3 days, as an opt-in flag (default off keeps Fortran parity).

---

### 4. Equilibrium Partitioning (Implicit Solver)

**What:** At each timestep, determine how much of each semi-volatile species is in the gas vs. particle phase at thermodynamic equilibrium. This is the core of the VBS scheme (#2) but also applies to HNO₃, NH₃, and other inorganic semi-volatiles.

**Stiffness in Fortran:** The Fortran solver uses fixed-point iteration or bisection, which is slow and sometimes fails to converge. JAX alternatives:
- **Newton-Raphson in `lax.while_loop`**: Quadratic convergence, JIT-compatible
- **Implicit diffrax solver**: Treat partitioning as a stiff ODE, use `Kvaerno5` or `TRBDF2`
- **Hybrid kinetic approach**: Solve condensation/evaporation kinetically at fine sub-timesteps instead of assuming equilibrium (avoids the implicit solve entirely)

**The hybrid kinetic approach may be superior in JAX:** Since JIT-compiled condensation is already fast, running kinetic condensation at 1-second substeps may converge to equilibrium naturally without needing an implicit solver. This sidesteps the stiffness problem entirely.

**Effort:** 3–5 days for Newton solver; 1 week for hybrid kinetic approach.

---

### 5. Dilution / Entrainment ✅ IMPLEMENTED

**Status:** Done. See `docs/dilution.md`.

**Implementation:** `tomas_jax/physics/dilution.py` — `dilution_step()` (JIT-compiled). Integrated into `make_step()` as `'dilution'` process. CLI: `--dilution-rate`, `--dilution-bg`.

**Equation:**
```
dC/dt = −kdil × (C − Cbg)
C(t+dt) = Cbg + (C(t) − Cbg) × exp(−kdil × dt)
```

---

### 6. Dry Deposition / Wall Loss

**What:** Size-dependent removal of particles by:
- **Atmospheric dry deposition**: gravitational settling + Brownian diffusion to surface
- **Chamber wall loss**: diffusion to chamber walls (size-dependent, typically ~10⁻⁴ to 10⁻³ s⁻¹)

**Equations:**
```
dNk/dt = −βk × Nk
dMk/dt = −βk × Mk
```
where `βk` is the size-dependent loss rate. For atmospheric deposition:
```
vdep(Dp) = vg(Dp) + 1/(Ra + Rb + Ra×Rb×vg)    (resistance model)
```
For chamber wall loss:
```
βwall(Dp) = (π²D/L² + vg/L) × ke              (Crump & Seinfeld)
```

**Fortran reference:** `wallloss.f` (6.3KB) — vapor + particle wall losses for chamber simulations.

**Effort:** 3–5 days.

---

## Tier 2 — Full Atmospheric Model

### 7. Emissions (Primary Particles + Gases)

**What:** Size-resolved source terms for:
- Primary sulfate (industrial, volcanic)
- Sea salt (wind-speed dependent, 0.01–10 μm)
- Mineral dust (size-distributed, episodic)
- Primary organic aerosol (biomass burning, cooking)
- Gas-phase: SO₂, DMS, NH₃, VOCs

**Implementation:** Add `'emission'` process to `make_step()`. Each source provides `dNk/dt` and `dMk/dt` per bin.

**Effort:** 1 week (framework + 2–3 source types).

---

### 8. Wet Scavenging

**What:** Removal of aerosol by cloud and rain:
- **In-cloud (nucleation) scavenging**: activated CCN are removed when cloud precipitates
- **Below-cloud (impaction) scavenging**: falling rain drops collect aerosol by inertial impaction and Brownian diffusion
- **Washout ratio**: empirical relationship between aerosol and rain concentrations

**Key equation (below-cloud):**
```
dNk/dt = −Λ(Dp) × Nk
Λ(Dp) = (3/2) × (P/Drain) × E(Dp, Drain)      (collection efficiency)
```

**Effort:** 1–2 weeks.

---

### 9. DMS Chemistry

**What:** Dimethyl sulfide (DMS) oxidation — the primary natural source of SO₂ over oceans:
```
DMS + OH → SO₂ + products     (abstraction pathway)
DMS + OH → DMSO → ... → MSA   (addition pathway)
DMS + NO₃ → SO₂ + products    (nighttime)
```

**Why important:** DMS is the dominant sulfur source in marine environments. Controls baseline CCN concentrations and the "CLAW hypothesis" feedback loop.

**Effort:** 2–3 days (3–5 reactions).

---

### 10. Organic Film / Variable Mass Accommodation

**What:** The accommodation coefficient α controls the probability that a gas molecule hitting a particle sticks. Currently α = 1.0 (every collision sticks). In reality:
- Clean inorganic particles: α ≈ 0.5–1.0
- Organic-coated particles: α ≈ 0.01–0.1 (viscous organic films resist uptake)
- Temperature-dependent: glassy state at low T reduces α further

**Impact:** Reduces H₂SO₄ condensation rate by 10–100× when organic coatings are thick. Critical for Arctic and upper troposphere conditions.

**Implementation:** Make α a function of organic mass fraction per bin: `α(k) = f(Morg(k) / Mtot(k))`.

**Effort:** 1–2 days.

---

### 11. Aqueous-Phase Chemistry (In-Cloud SO₄ Production)

**What:** Sulfate production inside cloud droplets:
```
SO₂(aq) + H₂O₂(aq) → SO₄²⁻ + H₂O     (dominant, pH-independent)
SO₂(aq) + O₃(aq) → SO₄²⁻ + O₂          (important at high pH)
```

**Why important:** Aqueous-phase production often exceeds gas-phase H₂SO₄ production. It adds mass preferentially to CCN-active bins (those that activated into cloud droplets), modifying the size distribution shape.

**Effort:** 1–2 weeks (includes Henry's law, dissociation equilibria, bin-resolved activation).

---

### 12. Surface Tension (Composition-Dependent)

**What:** Surface tension affects both the Kelvin effect (#3) and CCN activation (#13). Currently would use a fixed value. Should depend on:
- Organic mass fraction (organics lower σ from ~0.072 to ~0.03 N/m)
- Surfactant partitioning (Szyszkowski equation)

**Effort:** 2–3 days.

---

## Tier 3 — Extended Capabilities

### 13. CCN Activation (Köhler Theory)

**What:** Predict cloud droplet number concentration from the aerosol size distribution and supersaturation:
```
Sc(Dp) = exp(4Aσ / 3RTDp) / (1 + κDd³/(Dp³ − Dd³))^(1/2)
```
where κ is the hygroscopicity parameter. Particles with Sc < Smax activate into cloud droplets.

**Outputs:** CDNC (cloud droplet number), activated fraction per bin, maximum supersaturation.

**Effort:** 3–5 days.

---

### 14. Ice Nucleation (INP Parameterizations)

**What:** Parameterize the number of ice nucleating particles as a function of temperature, aerosol composition, and supersaturation:
- Immersion freezing (DeMott et al. 2010, 2015)
- Deposition nucleation
- Contact freezing

**Effort:** 1 week.

---

### 15. Optical Properties

**What:** Compute aerosol optical properties from Mie theory:
- Extinction, scattering, absorption coefficients
- Single scattering albedo
- Asymmetry parameter
- Refractive indices from composition (volume mixing rule)

**Can be post-processed:** Does not need to be inside the time loop. Compute from final Nk + Mk.

**Effort:** 3–5 days.

---

### 16. Gravitational Settling

**What:** Size-dependent sedimentation velocity:
```
vg = ρp × Dp² × g × Cc / (18μ)    (Stokes settling)
```
Important for particles > 1 μm (dust, sea salt). The 40-bin grid extends to 17.5 μm, so settling matters for the upper bins.

**Effort:** 1–2 days.

---

## Tier 4 — JAX-Specific Enhancements

These are already documented in `docs/future_features.md`:

| # | Feature | Effort | Unlocks |
|---|---------|--------|---------|
| 1 | Reverse-mode autodiff | 1 day | Parameter fitting, ML hybrids, adjoint sensitivity |
| 2 | GPU/TPU acceleration | 2–3 days | 10–100× speedup for ensembles |
| 3 | Batched vmap | 2 days | 50–1000× parallel scenarios |
| 5 | Learned coagulation surrogate | 1–2 weeks | Amortized O(N²) cost |
| 7 | Adaptive timestepping | 3–5 days | Handle nucleation bursts efficiently |

---

## SOA Solver Strategy — JAX Advantages

The stiffness of organic equilibrium partitioning is a known problem in Fortran TOMAS. JAX offers several strategies that could outperform the Fortran approach:

1. **Kinetic condensation at fine sub-timesteps:** Since JIT-compiled condensation is fast (~0.26s for 1440 steps), running 10–100 kinetic sub-timesteps per operator-split step may converge to equilibrium naturally. No implicit solver needed.

2. **Newton-Raphson in `lax.while_loop`:** Quadratic convergence, JIT-compilable. The Jacobian of the VBS system is analytically available (it's a sum of rational functions).

3. **Implicit ODE via diffrax:** `Kvaerno5` or `TRBDF2` handle stiff systems natively. The VBS partitioning ODE can be solved directly.

4. **Autodiff through the solver:** Once implemented, `jax.grad` can differentiate through the entire SOA partitioning, enabling gradient-based fitting of C* values and ΔHvap to chamber data.

---

## Implementation Roadmap (Suggested Order)

```
Phase 1 — Minimal realistic box model (2–4 weeks)
  ├── #3  Kelvin effect (done for SOA via vbs-jax; not needed for H2SO4 ≤ 240 K — see #3)
  ├── #1  SO₂ + OH → H₂SO₄ chemistry      (1–3 days)
  ├── #5  Dilution                         (1–2 days)
  ├── #6  Dry deposition / wall loss       (3–5 days)
  └── #2  VBS organic condensation         (1–2 weeks)

Phase 2 — Organic aerosol complete (2–4 weeks)
  ├── #4  Equilibrium partitioning solver  (3–5 days)
  ├── #10 Variable accommodation coeff     (1–2 days)
  ├── #12 Composition-dependent σ          (2–3 days)
  └── #9  DMS chemistry                    (2–3 days)

Phase 3 — Atmospheric processes (4–6 weeks)
  ├── #7  Emissions                        (1 week)
  ├── #8  Wet scavenging                   (1–2 weeks)
  ├── #11 Aqueous-phase SO₄ production     (1–2 weeks)
  └── #16 Gravitational settling           (1–2 days)

Phase 4 — Cloud and radiation (2–4 weeks)
  ├── #13 CCN activation                   (3–5 days)
  ├── #14 Ice nucleation                   (1 week)
  └── #15 Optical properties               (3–5 days)

Phase 5 — JAX power features (ongoing)
  ├── Reverse-mode autodiff
  ├── GPU/TPU acceleration
  ├── vmap batching
  └── Learned surrogates
```

---

## Fortran Source Reference

Key unported Fortran files in `tomas_fortran/src/` and `TRACER_SOM-TOMAS/src/`:

| File | Size | Physics |
|------|------|---------|
| `saprc.f` | 52 KB | Full SAPRC-99 gas-phase mechanism (100+ reactions) |
| `soacond.f` | 32 KB | VBS organic condensation with Kelvin effect |
| `wallloss.f` | 6.3 KB | Chamber wall loss (vapor + particles) |
| `layer_diff.f` | — | Height-dependent dilution |
| `HOMCOND.f` | — | HOM condensation (alternative to soacond) |

---

*Last updated: 2026-03-12*
