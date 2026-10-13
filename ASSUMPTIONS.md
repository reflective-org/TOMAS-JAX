# Modelling assumptions

Physics simplifications that change model results. Each entry covers what
is assumed, why, where it is implemented and how to change it.

## AAAR tutorial assumptions — confirm with Shantanu

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
  half-decade spacing; each class spans ±0.25 decade around its c\*.
  Level 1 uses a single class (K = 1, slot 1) for the user's surrogate.
- Partitioning uses vbs-jax (Zaveri et al. 2014 kinetics, soacond.f
  driving force). Raoult's law uses the **organic-phase mass fraction**
  (soacond convention), not mole fraction. Only organics absorb; SO₄,
  NH₄, nitrate and water do not.
- ΔH_vap = 129 − 11·log10 C\* kJ/mol (Epstein et al. 2010; fitted at 300 K,
  applied with T_ref = 298 K) in `box/aerosol_step.py::soa_params_from_cstar`;
  constants in `AerosolStepConfig`.
- Class properties are the vbs-jax defaults: MW 200 g/mol, σ = 0.025 N/m
  (Kelvin), ρ_org = 1200 kg/m³, D_b = 1e-10 m²/s, no particle-phase
  reaction (k_c = 0), α = 1.
- Docs: `docs/aaar/soa_partitioning.md`.

### A9 — Organic water uptake

- All organic slots (1–41, including SOA and nitrate) take the TOMAS
  ammonium-bisulfate wet/dry mass ratio (`water_equilibrium.py`, both
  water schemes) and count as sulfate in `density.py`. That makes SOA far
  too hygroscopic (κ_SOA ≈ 0.1 vs. ≈ 0.5–0.6 for ammonium bisulfate), so
  wet sizes, condensation sinks and N₂O₅ uptake are overestimated for
  organic-rich particles.

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
