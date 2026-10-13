# Gas-phase chemistry and photolysis (`tomas_jax/gaschem`)

Explicit MCM chemistry for the AAAR tutorial: the MCM v3.3.1 limonene subset supplied
by S. Jathar (`gaschem/mechanisms/limonene_mcm.eqn`, byte-identical to the original;
728 declared species, 725 used, 2135 reactions, 146-member RO2 sum, 28 photolysis
frequencies). `gaschem` does not import TOMAS so it can be split out later.

| Module | Role |
|---|---|
| `config.py` | constants, atomic weights, solver tolerances, TUV-x and volatility settings |
| `kpp_parser.py` | `parse_eqn(path) -> ParsedMechanism` (species atoms, reactions, RO2 list) |
| `mcm_rates.py` | `complex_rates(T, M, O2, N2, H2O)` and the MCM clear-sky J(χ) parameters |
| `mechanism.py` | `RateModel` (rate expressions -> k decomposition), `Mechanism` (index arrays + RHS) |
| `solver.py` | `integrate_interval(mech, y, k, dt)` — diffrax Kvaerno3 |
| `environment.py` | `air_environment(T, P, RH)` -> M, O2, N2, H2O [molec/cm³] |
| `photolysis.py` | `JProvider` — TUV-x / MCM / constant / off |
| `volatility.py` | Li et al. (2016) c*, closed-shell filter, `assign_classes` |

## Rate constants (KPP semantics)

MCM rate expressions are Fortran strings referencing `TEMP, M, O2, N2, H2O, RO2`, the
complex coefficients (`KMT01`…`KMT17`, `KRO2NO`, `KAPHO2`, `KBPAN`, …) and `J(J_*)`.
`RateModel` translates each one (identifiers checked against an allowlist before
evaluation) and, at a given environment, decomposes all 2135 expressions exactly into

    k = k_thermal * (RO2 if RO2-dependent else 1) + J_coef @ J

checked at a random (RO2, J) point. 280 reactions depend on RO2, 342 on J (7 of them
`KBPAN + J(...)`, i.e. thermal + photolysis). Like KPP's `UPDATE_RCONST`, k is
evaluated once per coupling interval (RO2 from the state at the interval start, J at
the midpoint) and frozen while the stiff solver integrates the interval.

The complex coefficients are transcribed from KPP's `examples/mcm/constants_mcm.f90`.
Cross-check against the independent F0AM `MCMv331_K.m` at three (T, P, RH) points:
31 of 33 identical to machine precision; KNO3AL differs by 2 % (F0AM uses an older
expression), KMT18 by 0.2 % (O2 fraction; not used by this mechanism).

## RHS and solver

`Mechanism.rhs(y, k)`: reactions have ≤ 2 reactants, so `r = k * y[a] * y[b]` with a
padding index pointing at a constant 1.0, and `dy/dt` is two `segment_sum`s over the
product and reactant entries. Nothing scales with n_rxn × n_species, so compile time
does not depend on mechanism size and the Jacobian (diffrax forms it with forward-mode
AD) costs ~2.6 ms; one dense 725² LU is ~1.6 ms.

Solver choice (24 h chemistry-only benchmark, Buffalo July, MCM J, 300 s intervals):

| Solver | rtol / atol [molec/cm³] | coupling | wall time |
|---|---|---|---|
| Kvaerno5 | 1e-4 / 1 | 300 s | 53.5 s |
| Kvaerno5 | 1e-3 / 1 | 300 s | 36.5 s |
| Kvaerno3 | 1e-3 / 1 | 300 s | 31.9 s |
| Kvaerno3 | 1e-3 / 1e4 | 300 s | 20.0 s |
| Kvaerno3 | 1e-3 / 1e4 | 600 s | 11.8 s |
| **Kvaerno3** | **1e-3 / 1e4** | **900 s** | **8.8 s** |

atol = 1e4 molec/cm³ (~0.4 ppq) removes the step-size collapse caused by near-zero
species; O3 after 24 h changes by < 0.6 % across these settings. Carrying the step size
across intervals was tried and rejected (fewer accepted, many more rejected steps).
Defaults: Kvaerno3, rtol 1e-3, atol 1e4 (`config.py`).

## Photolysis

`JProvider(j_names, PhotolysisSettings(mode, latitude, longitude, start_utc, jno2,
constant_sza_deg)).at(t)` returns the 28 J values at `t` seconds after `start_utc`.

- `tuvx`: surface J from the TUV-x port (`tuvx-jax`, branch `feat/aaar-tutorial`, in
  which the CH2O, CH3ONO2, RONO2, nitroxy-acetone, MEK, CH3COCHO, MVK and acetone
  recipes were ported and validated against the Fortran). Surface J depends only on
  SZA and Earth–Sun distance, so a 1° J(SZA) table is built once per process (~3 s)
  and interpolated (≤ 0.4 % vs direct calls up to SZA 78°). See `SOURCES` for which J
  comes from where (25 TUV-x, of which MACR/MVK channels by MCM branching; 3 MCM
  parameterization). Earth–Sun scaling reproduces TUV-x (× r, ASSUMPTIONS A11).
- `mcm`: MCM clear-sky parameterization (Saunders et al., 2003) for all J.
- `constant`: TUV-x spectrum at a fixed SZA rescaled to a target J_NO2 (Level 2).
- `off`: dark.

The tuvx-jax `data/` directory is found via `TUVX_JAX_ROOT` or the installed
(editable) package.

## Volatility

`assign_classes(parsed, log10_cstar=None)` keeps closed-shell organics (even
H + N + Cl + Br) with c*(298 K) ≤ 10^4.25 µg/m³ (245 species with the placeholder
c*) and lumps them into 15 half-decade classes (log10 c* = −3 … 4). Class c* = centre,
ΔHvap = 131 − 11·log10 c* kJ/mol (Epstein et al., 2010), class MW = mean member MW.
c* comes from a per-species table when given (`cstar_table`, CSV
`species,cstar_ugm3`) and otherwise from Li et al. (2016), Table 1 coefficients
transcribed from the paper.

## Tests

`tests/test_gaschem.py`: parser counts and atom balance (with the documented
mechanism quirks), K values at 298 K, J parameterization, k decomposition and the
expression allowlist, RHS vs a naive loop (1e-12), dark-ozonolysis initial rate,
a 1 h daytime integration, photolysis modes vs direct TUV-x (1 %), Li et al. examples,
closed-shell rule, class assignment and table override.

## References

- Jenkin et al. (1997), Saunders et al. (2003), Jenkin et al. (2015): MCM v3.3.1.
- Li, Pöschl & Shiraiwa (2016), ACP 16, 3327–3344.
- Epstein, Riipinen & Donahue (2010), ES&T 44, 743–748.
- Damian et al. (2002) / KPP; Wolfe et al. (2016) F0AM (cross-check).
