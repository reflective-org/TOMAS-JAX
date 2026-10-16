# Modelling assumptions

Physics simplifications that change model results. Each entry covers what
is assumed, why, where it is implemented and how to change it.

## AAAR tutorial assumptions — confirm with Shantanu

### A1 — N2O5 heterogeneous loss is γ-based uptake (replaces `<34>`)

- The supplied file's `<34> N2O5 = NA + NA : 4.0E-4` is a fixed first-order chamber
  sink. In the tutorial its rate is set every coupling interval to
  k_het = condensation sink with α = γ on the current TOMAS aerosol (A5), and its
  product is renamed to the counter `NA_HET`, so **only** this channel becomes
  particle nitrate. γ = 0 means no N2O5 heterogeneous loss at all.
- Where: `box/coupling.py::tutorial_mechanism` (`N2O5_HET_LABEL`, `NA_HET`),
  `box/driver.py`. The .eqn file itself is unchanged (byte-identical copy).

### A2 — `<33> HNO3 = NA : 6.0E-6` stays a gas-phase sink

- Kept as supplied (an HNO3 loss with a ~46 h lifetime, chamber wall/deposition-like).
  Its `NA` product is not tracked as particle nitrate.

### A3 — D9 (chamber OH tracer) is inert

- `D9` and `<900001> D9 + OH = D9PROD` are kept but D9 is initialised to 0, so it
  has no effect unless a user sets it (it can then serve as an OH-exposure tracer).

### A4 — Mechanism kept as supplied otherwise

- Limonene + OH branches stay scaled to the chamber k_OH = 1.47e-10 cm³/s, as in the
  file header. `CL` is never produced and starts at 0 (inert). `PROD` (an undeclared
  product name) is dropped at parse time. `H2O`, `BR`, `HCL` are declared but unused;
  H2O enters only through rate expressions (from RH).
- Mechanism observations from the parser checks (`tests/test_gaschem.py`), left
  as supplied: `<13917> CO2N3PAN + OH = CO2N3CO3` does not conserve N (no NO2
  product); `<14551>`, `<13369>`, `<7727>`, `<7729>` gain one carbon; 346
  reactions lose carbon (MCM omits CO2).
- Rate constants: MCM v3.3.1 `constants_mcm` (KPP distribution). KNO3AL there is
  1.44e-12·exp(−1862/T); F0AM uses 1.4e-12·exp(−1860/T) (2 % lower). We follow the
  KPP/MCM export the .eqn was generated with.

### A5 — Particle nitrate from N2O5 uptake

- N₂O₅(g) → 2 NO₃⁻(p) with the full yield (no ClNO₂ channel), at the rate
  k_het = condensation sink with α = γ (`physics/heterogeneous.py`).
- Nitrate is irreversible and non-volatile. There is no NH₄NO₃ / HNO₃
  thermodynamics, and `eznh3eqm` neutralises sulfate only.
- It is stored as NO₃ mass (MW 62) in slot 40 (`SRTNIT`) and added to every
  bin with particles (Nk > `NITRATE_N_MIN` = 2e-5 #/cell, twice MNFIX's
  empty-bin floor) in proportion to the per-bin uptake weights. The driver
  computes the weights once per coupling interval; within it each bin keeps
  its weight per particle (share scaled by Nk / Nk at the interval start).
  Until 2026-10-16 only bins with ≥ 1e10 #/cell (1 cm⁻³ in the tutorial box)
  got nitrate, which left a band of a few cm⁻³ at 40–90 nm in Level 3 that
  grew with resolution.
- Existing TOMAS routines treat slot 40 like the organic block: sulfate-like
  density (`density.py`) and the ammonium-bisulfate water ratio
  (`water_equilibrium.py`). For SOA partitioning it is inorganic (it does
  not dissolve SOA).
- γ is a constant user input (no RH/composition dependence).
- Docs: `docs/aaar/heterogeneous.md`.

### A6 — SOA volatility classes

- Condensable organics are lumped into up to 15 c\* classes (slots 1–15) at
  half-decade spacing, log10 c\*(298 K) = −3, −2.5, …, 4. Each class spans
  ±0.25 decade around its c\*. Level 1 uses a single class (K = 1, slot 1)
  for the user's surrogate.
- Partitioning uses vbs-jax (Zaveri et al. 2014 kinetics, soacond.f
  driving force). Raoult's law uses the **organic-phase mass fraction**
  (soacond convention), not mole fraction. Only organics absorb; SO₄,
  NH₄, nitrate and water do not.
- ΔH_vap = 131 − 11·log10 c\*(298 K) kJ/mol (Epstein et al. 2010, as used
  in SOM-TOMAS). The classes and this rule are defined in
  `tomas_jax/gaschem/volatility.py` (gas-chemistry stream). The aerosol
  side only takes the arrays (`SOAParams`).
- Other class properties are the vbs-jax defaults: σ = 0.025 N/m
  (Kelvin), ρ_org = 1200 kg/m³, Fuller volume 120, D_b = 1e-10 m²/s, no
  particle-phase reaction (k_c = 0). MW and α come with the arrays.
- Docs: `docs/aaar/soa_partitioning.md`.
- **Measured against PyCHAM (2026-10-14, `docs/aaar/pycham_comparison.md`, S1/S2) —
  for Shantanu:**
  - Transfer rate: vbs-jax pairs TOMAS's mean free path 2D/c̄ with a Fuchs–Sutugin
    form written for 3D/c̄, so its per-particle coefficient is up to 1.5× PyCHAM's and
    1.5× the gas-kinetic limit for small particles; the single Fuller volume of 120
    gives D 29 % above Fuller's value for C10H16O4. Effect: SOA forms faster early
    (+3–4 % at 1–3 h in Level 1 with c* = 1; part of +12 % at 30 min in Level 2);
    equilibrium SOA is unaffected.
  - Size distribution (changed 2026-10-15): SOA growth/evaporation and N2O5-uptake
    nitrate move particles between the fixed bins with a PPM remap in log-mass
    (`physics/ppm_remap.py`; `AerosolStepConfig.growth_bin_transfer = 'ppm'`,
    docs/aaar/aerosol_step.md): each bin's PPM sub-bin profile (the one H2SO4
    condensation uses) is carried by that bin's own displacement, however many bins,
    and deposited onto the fixed bins, with a limited composition slope inside each
    bin. The previous MNFIX split (still selectable as `'mnfix'`, bit-identical to the
    old code) combed the distribution (3/9/28 interior extrema in Level 1 at 40/80/160
    bins), spread it (number-mean dry diameter −13 % at 40 bins in S1, converging only
    at 160 bins) and put ~2 % of the particles below the seed size in Level 2. What
    holds now: one mode with no comb at every output time of all presets at 40/80/160
    bins; preset mean diameters change by ≤ 0.6 % from 40 to 160 bins; particles that
    do not change stay exactly in place while others grow or shrink past them; a
    grow-then-evaporate round trip (6 h growth, then +30 K) returns to the
    moving-section answer (number-mean 59.0/59.9/59.5 nm at 40/80/160 bins vs 59.2 nm);
    zero-OA bootstrap followed by Kelvin ripening reproduces the second mode the moving
    sections form. S1 (c* = 1, 12 h): number-mean dry diameter 123.3/123.2/123.2 nm at
    40/80/160 bins vs PyCHAM 120.5 (+2.3 %), GSD −3.2 %; with Kelvin off in both and
    PyCHAM's transfer coefficient the two agree within 0.2 % in diameter and 0.4 % in
    GSD at every resolution, so the rest is the Kelvin/transfer difference above, not
    the bins. S2 at 6 h: number-mean 212.3 vs 210.1 nm (+1.1 %), number median 202.6 vs
    200.9 nm, GSD −2.9 %, 0.11 % of the particles below the seed's 1st-percentile size
    (PyCHAM 0.05 %, MNFIX 2.1 %). SOA totals change by ≤ 0.21 %. (These numbers use the
    Li et al. 2016 c*; with the Jathar table Level 3 nitrate is 3.137 / 3.134 / 3.134
    µg/m³ at 40 / 80 / 160 bins.) Level 3 nitrate is
    4.0 % above the old code at 40 bins (5.16 vs 4.96 µg/m³): N2O5 uptake follows the
    particle surface, and the old 40-bin value carried the old scheme's resolution
    error. Both converge to 5.149 µg/m³ at 320 bins (PPM 5.160/5.152/5.149/5.149, MNFIX
    4.962/5.075/5.139/5.149 at 40/80/160/320). Remaining numerical limits: one remap per
    60 s micro-step; displacement and composition are linear inside each bin, so
    structure narrower than about two bins is not kept (on the 40-bin grid a Kelvin
    shoulder shows as a plateau with small wiggles); PPM reconstructs number only, so a
    small limiter keeps each bin's mean inside its bin (it moves at most ~1e-3 of the
    particles over a run in the tests); where a bin holds two kinds of particle they
    move with the bin's mean change. At the top of the grid (17.5 µm) the in-place
    SOA/nitrate that does not fit is taken back (SOA to the gas, nitrate reported)
    instead of MNFIX trimming the bin.
  - Per-species tracker: members of one class should keep one particle fraction (same
    c*); the tracker gives 0.29–0.94 where PyCHAM gives 0.81–0.83, so individual "top
    contributors" can be off by up to 4×. Class totals agree within ~11 %.

### A7 — Photolysis sources

- 25 of the 28 MCM J values come from TUV-x at the surface (`gaschem/photolysis.py`,
  `SOURCES`), with MACR (J18/J19) and MVK (J24) totals split by the MCM branching
  ratios at the same SZA. J_C3H7CHO_HCO (J15), J_IPRCHO (J17) and J_C5HPALD1 (J20) have
  no TUV-x recipe and use the MCM clear-sky parameterization (Saunders et al. 2003)
  at the same SZA.
- TUV-x uses its standard-atmosphere O3/T profiles, surface albedo 0.1, clear sky,
  no aerosol radiative effects; cross sections use profile, not box, temperature.
  The surface J(SZA) table is interpolated on a 1° grid (≤ 0.4 % vs direct calls).

### A8 — c* of the mechanism species: per-species table (Jathar, Nannoolal via PyCHAM)

- **Source (since 2026-10-16).** The Level 2 and 3 presets set
  `cstar_table = "mcm_limonene_cstar_298K.csv"` (`gaschem/config.py::DEFAULT_CSTAR_TABLE`),
  a table sent by Shantanu Jathar on 2026-10-16 and kept byte-identical in
  `tomas_jax/gaschem/mechanisms/` (sha256 and provenance in
  `mcm_limonene_cstar_298K.provenance.md`). Method, from its header: pure-component
  vapour pressure at 298.15 K from PyCHAM 5.8.0 with Nannoolal et al. (2008), converted
  as C* = 10⁶ · MW · Psat / (8.314 · 298.15) µg m⁻³ (O'Meara et al. 2014, eq. 1, as in
  PyCHAM `cham_up.py`). `ClassMap.source` / `diagnostics["cstar_source"]` name it.
- **Coverage.** 700 rows, every one a mechanism species: 458 closed-shell organics and
  242 radicals. The radicals are ignored (closed-shell rule below); 26 of them have
  c* ≤ 10^4.25 µg/m³ in the table and would condense otherwise. The mechanism's
  other closed-shell carbon species — LIMONENE, D9, CO and CH4, which the table excludes
  on purpose — get the Li et al. (2016) estimate and are listed in `ClassMap.fallback`
  and `diagnostics["cstar_fallback"]`; their Li log10 c* is 6.7–11.1, so none of them
  condenses. With the table 228 species condense (Li et al. 2016: 245); the other 230
  closed-shell table species have c* > 10^4.25 µg/m³ and stay gas-only. Per class
  (log10 c* = −3 … 4): 0, 1, 2, 2, 5, 7, 14, 11, 23, 16, 23, 31, 15, 35, 43 species;
  the least volatile is 10^−2.64 µg/m³, so the bottom class is empty.
- **What is still assumed.**
  - One estimation method (Nannoolal vapour pressures) at one temperature; its
    uncertainty is not carried. Other group-contribution methods can differ by a decade
    or more for multifunctional species.
  - Lumping into 15 half-decade classes (A6): each species condenses with the c* of its
    class centre (±0.25 decade), with ΔHvap = 131 − 11·log10 c* kJ/mol (Epstein et al.
    2010) for the temperature dependence; the table itself has no ΔHvap.
  - The table's 298.15 K values are used as the model's 298 K reference c* (vbs-jax
    `T_REF`), so the c* applied at 298.15 K is 1.8–3.3 % above the table.
  - Raoult's law by organic mass fraction, one absorbing organic phase (A6).
  - PAN-type species are treated like any other product: no particle-phase
    decomposition. The table makes most of them gas-only (23 of 54 PANs condense,
    Li et al. 2016: all 54); C822PAN, the top contributor with Li, has 10^4.85 µg/m³.
- **Li et al. (2016) stays available**: `cstar_table = None` (the `Scenario` default,
  used by custom scenarios that do not name a table) gives the composition-only
  estimate for every closed-shell organic. A user table (`species,cstar_ugm3` or the
  PyCHAM-style `cstar_ug_m3_298.15K` column, `#` comment lines allowed) replaces the
  default table as a whole; species it does not list fall back to Li et al. (2016),
  not to the Jathar table. When that leaves condensable species on Li, the run says so
  in `diagnostics["warnings"]` (the shipped table leaves none). To change a few c*
  values, merge them into the full table (`load_cstar_table` of the default, update,
  write all rows).
- The vapour-pressure method needs a normal boiling point; which estimate PyCHAM 5.8.0
  used with Nannoolal for this table is not stated in the file (question 1).
- **Measured consequence (2026-10-16, Level 2 preset, 6 h; all particle-phase species
  tracked):** SOA 8.66 µg/m³ (Li: 15.86), mass yield 0.156 (Li: 0.285), GMD 159 nm
  (Li: 203), 5.0 % nitrogen-containing (Li: 98.1 %; 2.8 % vs 98.7 % of the top 15).
  Species named -OOH hold 90 % of the SOA, PANs 1 % (Li: 3 % and 84 %). The largest
  contributors are hydroperoxides and an acid (LIMALOOH 3.71, C826OOH 1.09,
  C728OOH 0.85, C825OOH 0.75 µg/m³; LIMONIC 0.12). SOA grows slowly and almost
  linearly (0.25 / 0.89 / 4.45 µg/m³ at 0.5 / 1 / 3 h; Li reached 57 % of its 6-h value
  at 1 h).
  - **NOx dependence reversed:** SOA 8.54 / 8.15 / 8.19 / 8.66 / 6.54 / 1.02 /
    0.16 µg/m³ at NOx 0 / 0.5 / 1 / 2.5 / 5 / 10 / 20 ppb (Li: rose from 0.01 to 65).
    High NOx suppresses SOA (RO2 + NO competes with RO2 + HO2, fewer hydroperoxides).
  - **Yield rises with loading, no saturation:** limonene 1 / 2 / 5 / 10 / 20 / 40 ppb →
    SOA 0.04 / 0.23 / 2.99 / 8.66 / 17.84 / 38.4 µg/m³, yield 0.008 → 0.175 (Li:
    saturated near 16–17, yield falling 0.72 → 0.08).
  - O3 10 → 160 ppb: SOA 6.4 → 17.1 µg/m³ (Li: 13.7 → 18.0). Shifting every c* up one
    decade removes 73 % of the equilibrium SOA (Li: 64 %). Oxidant shares of limonene
    loss are unchanged (O3 60 %, OH 38 %, NO3 1 %). The NOx-sink feedback of the Li run is gone: relative to the
  run without particles NO +1.3 %, NO2 +0.5 %, O3 −0.1 %, limonene −5.3 %, OH +6.4 %
  (Li: −78 %, −79 %, −15 %, +167 %, −35 %). This matches the PyCHAM comparison, where
  PyCHAM's own Nannoolal vapour pressures gave 0.51× the Li SOA and 5 % N-containing
  (S2 `native`, `docs/aaar/pycham_comparison.md`).
- **Level 3 preset (24 h):** SOA 0.060 µg/m³ at 24 h (Li: 7.85) and particle nitrate
  3.14 µg/m³ (Li: 5.16). There is no dilution, yet SOA peaks at 0.39 µg/m³ about 2 h in
  and then falls: 0.28 at 4 h, 0.18 at 6 h, about 0.14 through the night, 0.093 at 22 h
  and 0.060 at 24 h. 2 ppb limonene gives little organic mass to absorb into, and OH
  (up to 1.2×10⁷ cm⁻³ on day 1, 2.4×10⁷ on day 2) oxidises and photolysis destroys the
  gas-phase semivolatile hydroperoxides, so the particles evaporate to follow the gas.
  The table SOA is 2 % N-containing (Li: ~100 %). Nitrate drops because SOA and its
  water no longer add surface for N2O5 uptake: k_het rises ×1.22 over the first 4 h
  (Li ×2.76) and ends 23 % below the chamber's 4×10⁻⁴ s⁻¹ (Li +20 %); γ 0.002 / 0.02 /
  0.1 → 0.39 / 3.14 / 6.81 µg/m³ (Li 0.98 / 5.16 / 7.71). Because the SOA is so small,
  it is far more sensitive to the coupling interval than nitrate (A12: −10.5 % at 600 s,
  −20 % at 300 s, vs nitrate +2.6 / +4.7 %).
- Level 1, and Levels 2 and 3 with `cstar_table=None`, are bit-identical to 2faa391
  (67 arrays compared). Run times are unchanged (Level 2 median 5.8 s, Level 3 12.8 s).
- Radicals (odd H + N + Cl + Br) never partition. Species more volatile than
  10^4.25 µg/m³ stay gas-only; less volatile than 10^−3.25 would join the lowest class.
- Earlier results with the Li et al. (2016) placeholder (98 % nitrogen-containing SOA,
  PAN partitioning as a strong NOx sink, confirmed with PyCHAM in S2 on 2026-10-14) are
  artefacts of that estimate; S2 is kept with `cstar_table=None` as a test of the
  partitioning code at fixed c*.

### A9 — Organic water uptake

- All organic slots (1–41, including SOA and nitrate) take the TOMAS
  ammonium-bisulfate wet/dry mass ratio (`water_equilibrium.py`, both
  water schemes) and count as sulfate in `density.py`. That makes SOA far
  too hygroscopic (κ_SOA ≈ 0.1 vs. ≈ 0.5–0.6 for ammonium bisulfate), so
  wet sizes, condensation sinks and N₂O₅ uptake are overestimated for
  organic-rich particles.
- There is no Kelvin term on particle water. PyCHAM (ideal solution with Kelvin,
  72 mN/m) holds 22 % less water in 50 nm particles and 9–13 % less in total for the
  Level 1 seed (2026-10-14, S1).

### A10 — Level 2 light is constant

- Level 2 uses constant light: the TUV-x spectrum at SZA 30° rescaled so J_NO2
  equals the user's value (default 5e-3 s⁻¹). Level 3 uses the real diurnal cycle.
  Open question: constant "chamber-like" light vs. noon sun at a chosen site.

### A11 — Earth–Sun distance as in TUV-x

- TUV-x (Fortran and the port) multiplies the radiation field by the Earth–Sun
  distance r [AU] (`apply_scale_factor(earth_sun_distance)`), not by 1/r². We
  reproduce TUV-x exactly; physically the flux scales as 1/r², a seasonal
  difference of ≤ ~5 %. Worth reporting upstream.

### A12 — Rate constants frozen per coupling interval (KPP semantics)

- All k (MCM complex rates, J at the interval midpoint, RO2 sum, k_het) are held
  fixed while the stiff solver integrates one coupling interval (300 s Level 1–2,
  900 s Level 3). Gas and aerosol are operator-split at the same interval; H2SO4
  (from the SA counter) and nitrate (from NA_HET) are passed to TOMAS as constant
  rates over the interval.
- With nucleation on, the interval used is at most 60 s and the aerosol micro-step
  at most 5 s, whatever the scenario asks for (A13). The requested interval is split
  into equal parts, so output times and the run's end do not move. Runs without
  nucleation keep the requested interval and the 60 s micro-step.
- Coupling-interval sensitivity, Level 3 preset at 40 bins with the Jathar c* table
  (measured 2026-10-16, final values at 24 h, against the preset 900 s: nitrate
  3.137 µg/m³, SOA 0.060 µg/m³, N 2552 cm⁻³): 600 s gives nitrate +2.6 %, SOA −10.5 %,
  N +0.03 %; 300 s gives nitrate +4.7 %, SOA −20.1 %, N +0.05 %. The totals are not
  converged in the coupling interval. With the Li et al. (2016) c* (same day: nitrate
  5.160, SOA 7.850 µg/m³, N 2620 cm⁻³) the changes were nitrate +3.5 % / +6.6 %, SOA
  +2.4 % / +4.7 %, N +0.09 % / +0.19 %.
- Small negative concentrations from the implicit solver are clipped to zero.
- Measured against PyCHAM (benchmark case G2, Level 2 gas phase,
  `benchmarks/pycham/README.md`): at 300 s coupling the RO2 sum is frozen at its t = 0
  value (0) for the first interval, so RO2+RO2 products (e.g. LIMAOH) are 0 at 5 min.
  Deviations are ≤ 6 % after 1 h and |NMB| ≤ 3 % over 6 h (median 1.7e-3). At 60 s
  coupling the median |NMB| is 3e-4.
- **Level 3 against PyCHAM (G3, 2026-10-14) — for Shantanu:** at the preset 900 s the
  median |NMB| over 361 species is 10 % (60 s: 0.6 %). Most of it is the NO2 reset
  convention (A14). The rest: the interval that contains sunset or sunrise uses the J
  of its midpoint (light 2 min early at 05:45–06:00; N2O5 +120 % at 06:00), and RO2 is
  0 for the first 900 s while half of the limonene reacts, so RO2+RO2 products are
  50–60 % low early on. The 600 s and 300 s numbers above show the same: 900 s is
  several percent away from a continuous solution.

### A13 — Nucleation uses H2SO4 only, with short time steps

- When nucleation is on, the Riccobono/Dunne scheme gets NH3 = 0, ions = 0 and
  organic vapour = 0, so it is pure binary/neutral H2SO4–H2O nucleation; organic
  (HOM) nucleation is not represented. Off by default in all presets. At the preset
  temperatures (298 K Levels 1–2, 293 K Level 3) the presets do not nucleate even with
  it on; bursts need colder air (265–270 K) or much more SO2.
- **Time steps (2026-10-16, for Shantanu).** With nucleation on the driver uses a
  coupling interval of at most 60 s and an aerosol micro-step of at most 5 s
  (`box/driver.py`: `NUCLEATION_MAX_COUPLING_DT_S`, `NUCLEATION_MAX_MICRO_DT_S`,
  applied in `time_steps`). A requested interval Δt_req is split into
  m = ⌈Δt_req / 60 s⌉ equal parts of Δt_req / m (300, 600 and 900 s all give 60 s), so
  output times stay the same. The run adds a warning line and records
  `requested_coupling_dt_s`, `coupling_dt_s`, `requested_micro_dt_s` and `micro_dt_s`
  in its diagnostics.
- Why: each micro-step of length Δt_m first adds the H2SO4 produced over it,
  C ← C + P·Δt_m (C the H2SO4 concentration, P its production rate), and then
  nucleates at J_nuc(C, T), which rises roughly as C^k with k about 3 to 8. A long
  micro-step therefore lifts C above its smooth path and overshoots the burst. The
  coupling interval matters through the chemistry: OH, and so P, changes within
  minutes but is frozen per interval (A12).
- Measured (40 bins; reference 30 s coupling with a 1 s micro-step, itself about 1 %
  from converged; peak N / final N / GMD deviation):

  | case | old steps (coupling / micro-step) | old | 60 s / 5 s |
  |---|---|---|---|
  | Level 1, 265 K, P = 1e6 cm⁻³ s⁻¹ | 300 s / 60 s | +5.5 / +3.0 / −1.1 % | +1.6 / +0.8 / −0.3 % |
  | Level 1, 270 K, P = 1e6 cm⁻³ s⁻¹ | 300 s / 60 s | +6.7 / +6.0 / −3.1 % | +1.5 / +1.2 / −1.2 % |
  | Level 2, 265 K, SO2 50 ppb | 300 s / 60 s | +15.7 / +8.6 / −3.4 % | +2.6 / +1.6 / −0.6 % |
  | Level 2, 265 K, SO2 20 ppb | 300 s / 60 s | +19.4 / +17.5 / −8.8 % | +0.8 / +0.7 / −0.2 % |
  | Level 3, 265 K, SO2 20 ppb | 900 s / 60 s | +57.6 / +28.1 / −14.5 % | +3.5 / +2.0 / −1.1 % |

  A 60 s coupling interval alone does not help (Level 3 at 60 s / 60 s: +77.8 %
  peak N); a 10 s micro-step still gives +8 to +9 % in Levels 2 and 3; 2 s is needed
  for 2 %. 120 s / 5 s is also within 5 % (largest +2.6 %) and is the fallback if run
  time matters, but only because errors cancel: with a 2 s micro-step, 120 s coupling
  alone is −4.3 % in Level 2 (SO2 50 ppb), against −0.7 % at 60 s. Long coupling and
  long micro-steps err in opposite directions in Levels 2 and 3, so the old defaults
  could look better than either change alone.
- Bins: the table is for 40 bins. At 80 bins (tutorial notebook 01), Level 1 at 265 K
  with 60 s / 5 s is peak N +5.9 %, final N
  +2.8 % and GMD −1.0 % against 30 s / 0.5 s (60 s / 2 s: +4.5 %, +2.1 %, −0.7 %;
  30 s / 1 s: +2.0 %, +0.9 %, −0.3 %). The 80-bin reference is not converged (0.5 s
  and 1 s still differ by 2 %), so at 80 bins 5 s is outside 5 % on peak N and 2 s
  is marginal. The web API allows only 40 bins. Run time at 80 bins: 52 s (5 s) and
  138 s (2 s) for the 12 h Level 1 run, against 14 s at 40 bins.
- Level 3 side effect: turning nucleation on also moves the coupling interval from
  900 s to 60 s, so SOA and nitrate change even without new particles. Level 3 preset
  at 293 K (no burst, peak N 3000 cm⁻³ either way), final values at 24 h, on against
  off: SOA −27 %, nitrate +6.6 %, sulfate +1.0 %, GMD +1.0 %. The 60 s run agrees with
  30 s / 2.5 s within 0.8 %, so this is the A12 coupling error of the 900 s run, not
  nucleation. Levels 1 and 2 move only in GMD (−1.3 %, −1.2 %). To isolate nucleation
  in Level 3, run the off case at 60 s coupling as well.
- Cost (warm, laptop): Level 1 with nucleation 1.3 → 14 s, Level 2 5.4 → 25 s,
  Level 3 11.5 → 112 s for the preset durations (also 112 s on one thread, as in the
  API workers, whose job limit is 120 s). The cost does not depend on whether a burst
  happens and grows in proportion to the run length, so the longest run the API
  allows, 48 h, would take about 225 s in Level 3 and 200 s in Level 2 (estimated).
- Not tried: adding the H2SO4 production gradually within the micro-step instead of
  all at its start might remove most of the micro-step error at lower cost.
- `driver.run(..., nucleation_caps=False)` turns the caps off, for convergence
  studies only.

### A14 — Seed and box conventions

- The seed's organic fraction is non-volatile, absorbing OA in slot 41 (with
  nucleated organic; reported as "POA"); the rest is sulfate. No dilution,
  deposition or wall loss. Level 3 holds NO2 constant as a stand-in for continuous
  NOx emissions (otherwise NOx is titrated in a few hours and no N2O5 forms at
  night); O3 accumulates without deposition.
- How the hold works matters (G3 against PyCHAM, 2026-10-14) — for Shantanu: the
  driver resets NO2 to 1 ppb after each coupling interval and lets it evolve inside
  the interval (it falls to 0.78–0.91 ppb in daylight at 900 s). The time-mean NO2 is
  0.969 ppb, so 8.4 % less NOx is supplied over 24 h than with NO2 fixed inside the
  solver (PyCHAM `const_comp`), giving HNO3 −9 %, NO −11 % by day and night-time
  nitrates up to +28 %. Holding NO2 inside the gas solver removes this (prototype:
  `benchmarks/pycham/cases/G3/hold.py`); at 60 s coupling the effect is 0.5 %.

### AC1 — SOA absorbing floor (stream C)

- In SOA partitioning, the Raoult denominator of each bin includes
  1e-4 × (bin dry mass outside the SOA classes)
  (`SOA_ABSORBING_FLOOR`, `physics/soa_partitioning.py`). This mass is not
  stored in Mk.
- Why: with zero organic mass vbs-jax sets psat = 0. Any vapour, even
  c\* = 1000 µg/m³, then condenses (~12% of it in the first 60 s on a pure
  sulfate seed), evaporates, and re-condenses in a sawtooth.
- Effect: a subsaturated vapour stays in the gas phase, as Pankow predicts.
  Equilibrium C_OA is biased by ≤ 1e-4 × seed dry mass. Supersaturated
  vapour starts condensing a few minutes later than without the floor.

### AC2 — Nucleated organic is absorbing (stream C)

- The 10% organic mass that nucleation puts in slot 41 (`SRTORG_NUC`)
  counts as non-volatile absorbing OA for SOA partitioning. New particles
  therefore have an absorbing phase from birth. All other organic slots
  outside the active classes (except nitrate) are treated the same way.

## Questions for Shantanu

1. ~~c* table for the mechanism species~~ — received 2026-10-16 and used by Levels 2
   and 3 (A8). Open: is ignoring particle-phase PAN decomposition acceptable now that
   23 PANs still condense? Is Level 3 SOA of 0.06 µg/m³ (peak 0.39 at 2 h) acceptable
   for the tutorial, or should the Level 3 preset change (more limonene, lower NO2
   hold)? With the table, Level 2 SOA falls with NOx (8.5 µg/m³ at NOx = 0, 1.0 at
   10 ppb) and is 90 % hydroperoxides: does that match your expectation and your own
   PyCHAM run (PyCHAM's Nannoolal run in S2 gives the same species ranking)? Which
   boiling-point method did PyCHAM 5.8.0 use with Nannoolal for this table?
2. A1–A4: confirm the chamber-to-ambient edits.
3. A10: constant light in Level 2, or noon sun at a site?
4. Level 3 preset: NO2 held at 1 ppb as a NOx source (A14) — acceptable, or
   prefer explicit emissions?
5. γ(N2O5) default 0.02 and the slider values {0, 0.002, 0.02, 0.1}.
6. From the PyCHAM comparison (2026-10-14, `docs/aaar/pycham_comparison.md`): hold NO2
   inside the gas solver instead of resetting it per interval (A14), and/or a shorter
   Level 3 coupling (A12)? Use mfp = 3D/c̄ (or a matching Fuchs–Sutugin form) and a
   per-class diffusion volume in vbs-jax (A6)? Report SOA by class rather than by
   species (A6 tracker)?
7. Nucleation time steps (A13): with nucleation on, runs now use 60 s coupling and a
   5 s aerosol micro-step (5 % accuracy on the burst at 40 bins; Level 3 takes about
   2 minutes for 24 h, about 4 minutes for the API's longest run of 48 h, so the API
   job limit has to rise to about 400 s or the run length be limited when nucleation
   is on). Acceptable, or would you rather have a cheaper scheme (production spread
   through the micro-step) or ternary/organic nucleation for the tutorial's
   nucleation case?
