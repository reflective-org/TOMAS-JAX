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
- It is stored as NO₃ mass (MW 62) in slot 40 (`SRTNIT`) and added to
  populated bins (≥ 1e10 #/cell) in proportion to the per-bin uptake
  weights. The driver computes the weights once per outer step.
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
  - Size distribution: SOA growth moves particles between the 40 fixed bins only
    through MNFIX, which spreads the distribution numerically (number-mean dry diameter
    −13 % after a ×9 mass growth; converges at 160 bins) and puts ~2 % of the particles
    below the seed size in Level 2. Mass is unaffected.
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

### A8 — c* placeholder until a curated table arrives

- c* (298 K) of every closed-shell organic comes from Li et al. (2016) using its
  C/H/O/N/S counts (`gaschem/volatility.py`). This is a composition-only estimate
  (±1–2 decades typical). It assigns very low c* to multifunctional PAN-type
  nitrates, which then dominate "top contributors" — a known artefact; PANs are
  thermally labile gases. A per-species table (`cstar_table`, CSV `species,cstar_ugm3`)
  overrides the estimate species by species; the result reports its provenance.
- Measured consequence (Level 2 preset, 2026-10-13): 99 % of the particle SOA is
  nitrogen-containing; limononic acid, limonaldehyde and most hydroperoxides get
  c* > 10^4.25 µg/m³ and never condense. NOx = 0 gives ~0 SOA (0.01 µg/m³), and SOA
  saturates near 16 µg/m³ for limonene 5 → 20 ppb (NOx-limited). These are artefacts
  of the placeholder, not limonene chemistry; Level 2 conclusions about yields or
  low-NOx SOA need the curated table.
- Radicals (odd H + N + Cl + Br) never partition. Species more volatile than
  10^4.25 µg/m³ stay gas-only; less volatile than 10^−3.25 join the lowest class.
- **Measured consequence, confirmed with PyCHAM (2026-10-14, S2) — for Shantanu:**
  partitioning of the PAN-type nitrates is a strong NOx sink. At 6 h of Level 2, NO
  −78 %, NO2 −79 %, OH −35 %, O3 −15 % and limonene +167 % relative to the same run
  without particles; PyCHAM given the same c* agrees within 1 point. With PyCHAM's own
  SMILES-based vapour pressures (Nannoolal) for the same species, SOA is 0.51× at 6 h,
  5 % N-containing instead of 98 %, and the gas-phase feedback disappears. The NOx sink
  is a consequence of the placeholder c*, not of the partitioning code.

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
  900 s Level 3; ≤ 3 % difference vs 600 s in the Level 3 benchmark). Gas and
  aerosol are operator-split at the same interval; H2SO4 (from the SA counter) and
  nitrate (from NA_HET) are passed to TOMAS as constant rates over the interval.
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
  50–60 % low early on. The "≤ 3 % vs 600 s" figure above understates the effect of
  900 s against a continuous solution.

### A13 — Nucleation uses H2SO4 only

- When nucleation is on, the Riccobono/Dunne scheme gets NH3 = 0, ions = 0 and
  organic vapour = 0, so it is pure binary/neutral H2SO4–H2O nucleation; organic
  (HOM) nucleation is not represented. Off by default in all presets.

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

1. **Blocking for Level 2:** c* table for the mechanism species (CSV
   `species,cstar_ugm3` at 298 K) — replaces the Li et al. (2016) placeholder, which
   makes SOA almost entirely organic nitrates (A8). Also: how to treat PAN-type species.
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
