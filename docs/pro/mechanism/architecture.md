# Several gas-phase mechanisms in TOMAS-JAX: architecture plan

Status: plan, written 2026-10-16 on branch `feat/pro-mechanism` (from `feat/aaar-tutorial`
5227ddd). Build after 2026-10-27. Nothing in the model code, the presets or the tutorial is
changed. Prototypes are in [`prototypes/mechanism/`](../../../prototypes/mechanism/README.md);
every number marked **[measured]** comes from one of them, on an Apple M4 Pro shared with
other jobs (load average 6-19, so timings are upper bounds). **[MCM site]** is read off a
public page at [mcm.york.ac.uk](https://mcm.york.ac.uk/MCM). **[estimate]** is my judgement
and says what it rests on.

Companion note: [`mcm_coverage.md`](mcm_coverage.md) (same folder, written in parallel)
covers which SANDBOX species and reactions MCM has, the MCM export tool, CRI and the MCM
licence in detail. This note covers how the code should carry more than one mechanism.

Why (Ali, this session): "Can we have the MCM version of the SANDBOX species in the
chemistry part? Having SO2 comprehensive chemistry and other species." and "I want more
comprehensive gas-phase mechanism. It seems like for now it is only limonene from MCM. I meant
to bring more MCM reactions for species that we have in sandbox not just bringing sandbox
chemistry."

## 1. Summary

1. **Most of the SANDBOX species are already in the model, with MCM chemistry.** The
   limonene file contains MCM's whole inorganic block, the methane chain and MCM's SO2
   chemistry. Extracted on its own it is a 29-species, 68-reaction module
   (`mcm_inorganic_c1_sulfur.eqn`) that covers 18 of the 36 SANDBOX species directly
   **[measured]**. Of the rest, O2 and H2O are environment variables, CH3 is folded into CH3O2
   by MCM, C2H6 needs the MCM ethane chain, HNO3aq is an aerosol counter and 14 halogen species
   have no MCM chemistry (MCM has only Cl + alkanes; no ClO, ClONO2 or bromine).
2. **"Comprehensive SO2" in MCM terms means three things**: OH + SO2 (KMT12) and O(3P) + SO2,
   which the module has; Criegee intermediate + SO2, which arrives with every alkene or
   terpene subset (10 reactions in the limonene file, 49 in the full MCM per
   [`mcm_coverage.md`](mcm_coverage.md)); and DMS oxidation, the only MCM SO2 *source*, from
   the MCM DMS chain. SO2 + HO2 is not in MCM (JPL gives only an upper limit) and stays out.
3. **Make the mechanism a registry entry, not a constant.** Today one file is hard-wired in
   about ten places (section 2). A registry (section 3) names each selectable mechanism, the
   `.eqn` modules it is built from, the edits applied in code, its counters and heterogeneous
   reactions, its c\* table, its photolysis map and its solver. The tutorial keeps
   `mcm_limonene` unchanged and bit-identical.
4. **Prefer one MCM export per mechanism; compose only across sources.** MCM's exporter
   already merges several VOCs into one file with one inorganic block and one RO2 sum. The
   composer (section 4) is for joining an MCM export with non-MCM modules (JPL halogen,
   heterogeneous, chamber) and for appending a delta to a file that must stay byte-identical.
   The prototype composer enforces six rules; each was shown to fire on a deliberate
   violation, and inorganic + limonene composes back to exactly the limonene file (68
   duplicates removed, RO2 lists equal) **[measured]**.
5. **The dense gas solver stops being usable at about 1000 species.** Measured per-step parts
   scale like n² (autodiff Jacobian) and n³ (dense LU). Estimated gas-only time for 24 h at
   900 s coupling: 12 s at 725 species, 44 s at 1352, 14 min at 4487 and 32 min at 5741
   (full-MCM size); with the sparse KPP-style solver from the Pro plan: 0.6 s, 1.0 s, 3.6 s
   and 6.7 s **[estimate from measured parts, section 8]**. The sparse solver is a
   prerequisite for any multi-VOC mechanism, and its LU fill stays at 1.46-1.57 times the
   Jacobian on synthetic mechanisms up to 5741 species **[measured]**.
6. **c\* for products of new VOCs: reuse Shantanu's own method.** Nannoolal vapour pressure
   with the Nannoolal boiling point, through UManSysProp as PyCHAM calls it, reproduces all
   700 rows of Shantanu's table (median |Δlog10 c\*| 1e-7, worst 0.028 dex) and runs on all
   5834 species of PyCHAM's MCM species xml in 13 s with no failures **[measured]**. So a
   c\* table for any MCM mechanism can be built offline with the same method. Li et al.
   (2016) stays the fallback, but it has no halogen term, so it must not be used for
   chlorinated products. Ask Shantanu to confirm.
7. **Rate sets differ.** For the 24 reactant sets MCM and SANDBOX's JPL 19-5 share, MCM/JPL
   ratios at 298 K range from 0.58 (HO2 + NO2) to 1.36 (NO + O); SO3 + H2O differs by 91x
   because MCM is first order in water and JPL second order **[measured]**. A mechanism uses
   one source per reactant set, chosen explicitly in the registry (rule 4).
8. **API and cache key: add `chemistry.mechanism` (id) and a mechanism hash.** Stop hashing
   mechanism data files into `code_version()`; otherwise adding any new mechanism re-keys
   every cached tutorial result.

### Decisions for Ali

| # | Decision | Options | Recommended |
|---|---|---|---|
| M1 | First non-tutorial mechanisms | (a) `mcm_inorganic` + DMS + ethane/propane only; (b) (a) plus a biogenic set (isoprene, alpha- and beta-pinene, limonene); (c) the full MCM | (a) first, as the test bed for the registry, then (b) once the sparse solver is in. (c) only as a stress test |
| M2 | Halogens | (a) none yet; (b) Xue et al. (2015) tropospheric Cl module; (c) SANDBOX's JPL 19-5 Cl/Br + heterogeneous, as an optional module | (c) as an optional module for stratospheric/volcanic scenarios, (b) later if a coastal or urban Cl case is wanted |
| M3 | Rate set for the shared inorganic reactions | (a) MCM (IUPAC-based) everywhere; (b) JPL 19-5 everywhere; (c) per mechanism | (c): MCM for tropospheric mechanisms; a separate `jpl195_core` module for stratospheric ones, never mixed reaction by reaction |
| M4 | c\* for new VOC products | (a) Nannoolal via UManSysProp, same as Shantanu; (b) Li et al. (2016); (c) ask Shantanu to produce the tables | (a), built offline per mechanism, then sent to Shantanu to check; (b) only as the documented fallback |
| M5 | Where multi-VOC files come from | (a) one MCM export per mechanism; (b) compose separate exports | (a); (b) only to append a delta to the byte-identical tutorial file |

## 2. What is hard-wired to one mechanism today

| Where | What assumes the limonene file | Change |
|---|---|---|
| `gaschem/config.py` `LIMONENE_EQN`, `DEFAULT_CSTAR_TABLE`, `CSTAR_TABLE_SOURCES` | one mechanism file and one table | registry entries |
| `box/coupling.py::tutorial_mechanism` (`lru_cache(maxsize=1)`) | one parsed mechanism, the `<34>` rename by label | registry patch keyed on content (reactants and products), not label |
| `box/driver.py::_chemistry` (`lru_cache(maxsize=1)`), `_class_map` | one RateModel/Mechanism per process | cache keyed by mechanism hash |
| `box/driver.py` `COUNTER_SPECIES = ("SA", NA_HET)`, `index["SA"]`, `index[NA_HET]`, `index["N2O5"]`, `N2O5_HET_LABEL` | these species and that label exist | counters and external reactions declared per mechanism |
| `box/result.py` `REPORTED_GASES_PPB` (includes LIMONENE), `REPORTED_J` | these species and J names exist | per-mechanism reported lists |
| `gaschem/mechanism.py::RateModel` | every J name is in `MCM_J_PARAMS`; only MCM K names and `exp`, `log10` in rate expressions | J map from the registry; vetted extra rate functions per rate set |
| `gaschem/photolysis.py` `SOURCES` | 28 J names | all 34 MCM names plus halogen names (section 6) |
| `gaschem/volatility.py::log10_cstar_li2016` | composition classes CH, CHO, CHN, CHON, CHOS, CHONS; Cl and Br atoms are ignored | refuse halogenated organics instead of silently treating them as CHO |
| `box/driver.py::code_version` | hashes every `.eqn`, `.csv`, `.toml` in the package | exclude mechanism data, hash the resolved mechanism per run (section 11) |
| `gaschem/solver.py` | dense Kvaerno3, Jacobian by forward-mode autodiff | sparse Ros3 for large mechanisms (section 8) |

## 3. Mechanism registry

### 3.1 Concepts

- **Module**: one `.eqn` file plus its metadata: id, format (`mcm_kpp`, or `tomas_kpp` for
  KPP syntax with vetted non-MCM rate functions), source (`mcm`, `jpl`, `chamber`, ...),
  provenance and sha256, an optional species name map, the labels of MCM's lumped
  gas-particle sinks (`HNO3 = NA`, `N2O5 = NA + NA`, `SO3 = SA`).
- **Mechanism**: what a user picks. A list of modules, optional extra modules a user may
  toggle, patches applied in code, reactions to drop, the winner for each cross-source
  overlap, counters, external (heterogeneous) reactions, c\* table, J map overrides, the
  reported species, the solver it needs, and a status (`validated`, `prototype`, `planned`).
- **Patch**: a named, content-keyed edit in Python, like today's `tutorial_mechanism()`
  (rename the product of `N2O5 = NA + NA` to `NA_HET` after checking the reactants and
  products, not just label 34). Files stay byte-identical.
- **Resolved mechanism**: the output of resolution: species, atoms, reactions (namespaced
  labels), RO2 list, J names with their sources, counters, external reactions, c\* map. It
  has a hash (section 11) and is what the driver, the cache and the chemistry tab use.

A format sketch is in
[`prototypes/mechanism/registry_example.toml`](../../../prototypes/mechanism/registry_example.toml).

### 3.2 Proposed entries

| id | Modules | Species / reactions | Solver | Status |
|---|---|---|---|---|
| `mcm_limonene` | the tutorial file + `tutorial_n2o5_het` patch | 726 / 2135 (725 + NA_HET) **[measured]** | dense Kvaerno3 (frozen for the tutorial) | validated |
| `mcm_inorganic` | inorganic + CH4 + SO2 (prototype extraction now; a direct MCM export later) | 29 / 68 **[measured]** | either | prototype |
| `mcm_background` | inorganic + CH4, C2H6, C3H8, DMS from one export | about 60-120 **[estimate]**; the C1-C3 closure of the limonene file is 98 / 280 **[measured]** | either | planned |
| `mcm_bvoc` | one export: isoprene, alpha-pinene, beta-pinene, limonene, plus the background roots | about 2000-3000 **[estimate, mcm_coverage.md]**; isoprene alone 602 / 1926 per that note | sparse only | planned |
| `mcm_full` | the full MCM v3.3.1 | 5832 / 17224 **[MCM site]** | sparse only | stress test |
| module `+jpl_halogen` | SANDBOX's Cl/Br gas reactions (only those with a Cl or Br species) + 4 gamma-based heterogeneous reactions | 14 halogen species; 19 gas, 4 heterogeneous and 12 photolysis reactions (active ones with a Cl or Br species, counted in REACTIONS.md) | either | planned |
| module `+xue2015_cl` | tropospheric Cl module, 205 reactions ([Xue et al., 2015](https://gmd.copernicus.org/articles/8/3151/2015/)) | not counted | either | later |
| `jpl_strat` | `jpl195_core` + `+jpl_halogen`: the SANDBOX chemistry in TOMAS-JAX | 36 SANDBOX species | either | planned, for comparison with SANDBOX |

Sizes of `mcm_background` and `mcm_bvoc` must be replaced by counts from a real export.

## 4. Composing modules safely

Prototype: `mechlib.compose`, exercised by `compose_demo.py` **[measured]**.

Let a reaction be r = (R, P, k), with R the multiset of reactants, P the products with
coefficients and k the rate expression. Let B = {M, O2, N2, H2O, hv} be the bath species MCM
writes into rates or omits. The composer uses two keys:

- reaction key K(r) = (sorted(R \ B), photolysis flag, sorted(P \ B) with coefficients),
- reactant-set key S(r) = sorted(R \ B) plus "hv" if photolysis.

Rules, applied in module order (each raises unless stated):

1. **One name, one composition.** A species name has the same atoms in every module.
2. **Duplicates are dropped.** Same K and same normalised rate text (whitespace removed,
   upper case, Fortran `D` exponents and every numeric literal rewritten as a float, so
   `4.50E-14` equals `4.5D-14`): the first copy is kept and the dropped label is logged.
   Inorganic + limonene: 68 duplicates removed, result identical to the limonene file.
3. **Same reaction, different rate is an error.** Shown by patching `<31>` to 2.0e-12.
4. **Cross-source overlaps need an explicit winner.** If two modules with different
   `source` both have reactions with the same S, the composer stops, because summing them
   double counts. This catches lumped versus explicit descriptions that rule 3 cannot: JPL's
   `SO2 + OH = SO3 + HO2` (SANDBOX R79) against MCM's `OH + SO2 = HSO3` followed by
   `HSO3 = HO2 + SO3`. With `allow_overlap = {("OH", "SO2"): "jpl_sulfur"}` only the JPL
   channel stays. Within one source, branches with the same S are normal (MCM splits
   channels) and are kept.
5. **RO2 is the union** of the modules' RO2 lists, and every reaction whose rate is
   proportional to RO2 must have its reactant in the list. Limonene: 280 RO2-dependent
   reactions, none with a reactant outside the 146-member sum.
6. **Labels are namespaced** `module:label` (MCM exports renumber; never key code on labels).

What the rules cannot catch, so a reviewer must:

- algebraically equal rate expressions written differently (`2*K` vs `K*2`): reported as a
  conflict, resolved by hand;
- the same chemistry with different reactant sets (SANDBOX `O1D + CH4 = CH3 + OH` versus
  nothing in MCM; CH3 versus CH3O2): handled by the species map and by review;
- orphans left by an overlap decision (after JPL wins OH + SO2, `HSO3 = HO2 + SO3` is never
  fed). The composer should report species that are consumed but never produced and have
  no initial value.

Species name maps: MCM names are canonical (upper case, `HO2NO2` not HNO4, `SA` for
H2SO4, `CL`, `HCL`, `BR`). A non-MCM module carries a map from its names to MCM names; any
species not in MCM keeps a mapped upper-case name (`CLO`, `CLONO2`, `BRO`). Rule 1 then
checks the compositions.

## 5. Rate expressions and rate sets

- **MCM coverage is complete in the code.** The MCM site lists 12 generic coefficients and
  18 complex ones (KFPAN, KBPAN, KBPPN, KMT01-KMT18) **[MCM site]**
  ([generic](https://mcm.york.ac.uk/MCM/rates/generic),
  [complex](https://mcm.york.ac.uk/MCM/rates/complex)); `mcm_rates.complex_rates` defines
  all of them. A full MCM export should translate without new names; the allowlist in
  `RateModel` will say so if not.
- **Non-MCM modules** need JPL forms. Add a small, vetted function set with KPP-compatible
  names, evaluated in numpy like `constants_mcm`: the JPL termolecular form
  k(T, M) = k₀M / (1 + k₀M/k∞) · 0.6^{1/(1+(log₁₀(k₀M/k∞))²)}, with k₀ = k₀(298)(T/298)^(−n)
  and k∞ = k∞(298)(T/298)^(−m) (T temperature [K], M air number density [cm⁻³]; JPL 19-5
  Table 2-1 convention as in SANDBOX `mechanism.falloff`); the chemical-activation form
  k_f + k_int(1 − k_f/k∞); and equilibrium-constant reverse rates k_f/K_eq. They must keep
  the RateModel decomposition k = a·RO2^{0 or 1} + c·J (they do: thermal only).
- **MCM and JPL disagree on shared reactions** (`rate_crosscheck.py`, ratio MCM/JPL of the
  summed channels; the sibling note has the same comparison inverted, with matching numbers):

| Reactant set | 298 K, 1 atm | 240 K, 300 hPa | 220 K, 50 hPa |
|---|---|---|---|
| HO2 + NO2 | 0.58 | 0.54 | 0.53 |
| NO + O | 1.36 | 1.32 | 1.24 |
| NO + OH | 1.32 | 1.28 | 1.15 |
| HO2 + HO2 | 1.14 | 1.27 | 1.37 |
| NO2 + OH | 0.94 | 1.10 | 1.30 |
| OH + SO2 | 0.94 | 0.84 | 0.72 |
| SO3 (+ H2O) | 0.011 | 0.023 | 0.21 |
| other 17 sets | 0.86-1.09 | 0.87-1.08 | 0.87-1.11 |

  SO3 + H2O matters little near the surface (SO3 lifetime 2 ms in MCM, 0.02 ms in JPL), but
  at 220 K it is 100 s against 21 s. MCM's inorganic chemistry is a tropospheric set; for
  stratospheric runs the registry should offer `jpl195_core`, not a reaction-by-reaction mix.

## 6. Photolysis names

MCM v3.3.1 uses 34 J numbers (J1-J8, J11-J24, J31-J35, J41, J51-J56) **[MCM site]**
([photolysis](https://mcm.york.ac.uk/MCM/rates/photolysis)); `MCM_J_PARAMS` holds all 34, but
`photolysis.SOURCES` maps only the 28 the limonene file uses. A larger MCM mechanism will
fail at `JProvider` until these six are mapped:

| MCM name | Proposed source |
|---|---|
| J_CH3CHO (J13) | TUV-x `CH3CHO+hv->CH3+HCO` (in the port) |
| J_C2H5CHO (J14) | TUV-x `C2H5CHO+hv->C2H5+HCO` (in the port) |
| J_C2H5NO3 (J52) | TUV-x `C2H5ONO2+hv->C2H5O+NO2` (in the port) |
| J_MVK_CO (J23) | add to the existing MVK split (`TUVX_SPLIT` lists the channel but not the key) |
| J_C3H7CHO_C2H4 (J16) | MCM parameterisation, as J15 today |
| J_TC4H9NO3 (J55) | MCM parameterisation; TUV-x lists `C(CH3)3(ONO2)` but the port does not compute it |

Halogen modules need Cl2, ClONO2 (2 channels), ClOOCl, HOCl, OClO, ClNO2, HCl, BrO,
BrONO2 (2), BrCl, HOBr and Br2: 20 of the 21 TUV-x reactions checked (all of these, plus
N2O5, HNO4 and PAN) are computed by the tuvx-jax port (94 reactions) **[measured]**. The J
map becomes a registry table: J name → (TUV-x key, or TUV-x total with a branching rule,
or MCM l, m, n). For the `mcm` photolysis mode, names without MCM parameters (all halogen
J's) get l, m, n fitted once to the TUV-x J(SZA) table, J = l·cos(χ)^m·exp(−n·sec χ) with χ
the solar zenith angle, and are labelled "fitted to TUV-x" in `diagnostics["j_sources"]`.

## 7. Heterogeneous reactions and counters

- External reactions (k from the aerosol, like `<34>` today) are declared per mechanism.
  The JPL halogen module adds ClONO2 + HCl, ClONO2 + H2O, HOCl + HCl and BrONO2 + H2O, with
  k_het = ¼·γ·v̄·A (γ uptake coefficient, v̄ mean molecular speed, A aerosol surface area per
  volume from TOMAS).
- These are first order in the gas taken up but consume a second species (HCl).
  `Mechanism` today computes r = k·y_a·y_b over all reactants. Add a per-reaction
  "rate reactants" list separate from "consumed species" (SANDBOX's `rate_law`), with the
  consumed species entered as a loss term and k_het limited when HCl runs out.
- Counters (SA, NA, NA_HET, HNO3aq) are declared per mechanism with the aerosol sink they
  feed (sulfate, nitrate). The driver reads and resets them as today.

## 8. Solver cost against mechanism size

### 8.1 Model

The gas time of a run is T_gas = N_s · c_step, with N_s the number of solver steps over the
run and c_step the cost of one step. For the dense Kvaerno3 solver used today and the sparse
Ros3 solver of the Pro plan (`prototypes/pro/perf/chemlib.py`):

  c_dense ≈ a_d (t_J + t_LU + 8 (t_sv + t_f)),   c_sparse ≈ a_s (t_Jv + t_fac + 3 (t_ssv + t_f))

with t_J the forward-mode autodiff Jacobian, t_LU the dense LU of (I/h − J), t_sv one dense
triangular solve pair, t_f one right-hand side, t_Jv the analytic Jacobian on its sparsity
pattern, t_fac the levelised sparse LU and t_ssv one sparse solve pair, all measured per
call. The factor 8 counts the Newton solves of three implicit stages (an assumption), and
a_d = 1.40 and a_s = 0.52 calibrate the sums to the measured step costs at 725 species
(11.3 ms Kvaerno3 in the model, `run_inorganic.py`; 0.39 ms Ros3, Pro plan section 3.2).
N_s is taken from Level 3 (Pro plan): Kvaerno3 1061 steps at 900 s coupling and 4855 at
about 120 s; Ros3 1576 and 8029. It is assumed not to depend on mechanism size, which holds
for the synthetic mechanisms (same rate constants) but must be measured for real ones.

Dense LU work grows as (2/3)·n³ flops (n species); the autodiff Jacobian as about n·(n_r +
n_P + n_L) gathers (n_r reactions, n_P product entries, n_L reactant entries). The sparse LU
work is set by the fill: 88 thousand flops at 725 species.

### 8.2 Measured parts

`scaling.py`. "copies N": N copies of limonene's chemistry of species with 4 or more C,
renamed per copy, sharing the inorganic and C1-C3 chemistry. A real multi-VOC MCM shares more
intermediates, so its fill will be somewhat higher.

| Mechanism | n | n_r | Jacobian nnz | LU nnz (fill) | LU flops | factor levels | t_f | t_J | t_LU | t_Jv | t_fac | t_ssv |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| inorganic | 29 | 68 | 165 | 182 (1.10) | 597 | 20 | 0.006 | 0.014 | 0.008 | 0.004 | 0.038 | 0.033 |
| C1-C3 closure | 98 | 280 | 753 | 917 (1.22) | 5.1e3 | 36 | 0.009 | 0.081 | 0.146 | 0.010 | 0.057 | 0.060 |
| limonene | 725 | 2135 | 6142 | 8970 (1.46) | 8.8e4 | 96 | 0.028 | 2.89 | 1.79 | 0.037 | 0.261 | 0.121 |
| copies 2 | 1352 | 3989 | 11490 | 16980 (1.48) | 1.9e5 | 123 | 0.039 | 11.2 | 6.33 | 0.056 | 0.422 | 0.219 |
| copies 4 | 2606 | 7697 | 22190 | 33870 (1.53) | 5.1e5 | 174 | 0.060 | 52.8 | 44.0 | 0.085 | 1.16 | 0.287 |
| copies 7 | 4487 | 13259 | 38230 | 59810 (1.56) | 1.2e6 | 232 | 0.087 | 269 | 161 | 0.109 | 2.50 | 0.504 |
| copies 9 | 5741 | 16967 | 48926 | 76676 (1.57) | 1.6e6 | 260 | 0.084 | 709 | 348 | 0.165 | 4.29 | 1.13 |

Times in ms, best of 3-5, each piece jitted separately **[measured]**. The sparse solve
matches a dense solve to 1.3e-11 (725) and 1.2e-9 (4487) relative. The 725-species row
reproduces the Pro plan's numbers (6142 nonzeros, 8970 after fill, 96 levels).

### 8.3 Estimated 24 h gas-solve times

`estimate.py` **[estimate from measured parts]**; laptop, one process, gas chemistry only:

| Species (case) | Dense step | Sparse step | Dense, 900 s | Dense, 120 s | Sparse, 900 s | Sparse, 120 s |
|---|---|---|---|---|---|---|
| 29 (inorganic) | 0.18 ms (0.054 ms measured) | 0.08 ms | 0.2 s | 0.9 s | 0.1 s | 0.7 s |
| 98 (about 100) | 0.68 ms | 0.14 ms | 0.7 s | 3.3 s | 0.2 s | 1.2 s |
| 725 (today) | 11.3 ms | 0.39 ms | 12 s | 55 s | 0.6 s | 3.1 s |
| 1352 (about 1000-1500) | 42 ms | 0.66 ms | 44 s | 3.4 min | 1.0 s | 5.3 s |
| 2606 | 215 ms | 1.2 ms | 3.8 min | 17 min | 1.9 s | 9.6 s |
| 4487 (about 5000) | 790 ms | 2.3 ms | 14 min | 64 min | 3.6 s | 18 s |
| 5741 (full-MCM size) | 1.8 s | 4.2 ms | 32 min | 2.5 h | 6.7 s | 34 s |

At small n the model overestimates (dispatch overhead dominates each separately jitted
piece; the measured 29-species Kvaerno3 step is 0.054 ms). On a Render core multiply by
2.1-2.9 (Pro plan). Two conclusions: the dense solver is fine up to a few hundred species
and unusable past about 1000; with the sparse solver a full-MCM-size 24 h run costs a few
seconds to half a minute of gas chemistry, so the aerosol step stays the larger cost.

### 8.4 Compile time and memory

- Compile **[measured]**: the autodiff Jacobian and dense LU compile in 0.1-1 s at any size.
  The levelised sparse LU unrolls one XLA op group per level, so its compile grows with the
  level count: factor 0.70 s / solve 0.57 s at 725 species, 1.8 / 1.1 s at 4487, 2.3 / 1.5 s
  at 5741. The symbolic analysis is pure Python, once per mechanism: ordering + symbolic
  0.18 s at 725, 4.1 s at 4487, 8.1 s at 5741; cache it on disk keyed by the mechanism hash.
- Each mechanism is a separate compiled program (all index arrays have mechanism-specific
  shapes). Warm only the tutorial mechanism and the Pro default at worker start; others go
  through `JAX_COMPILATION_CACHE_DIR`.
- Peak resident memory of the scaling run (dense and sparse pieces in one process; the dense
  Jacobian dominates): 0.52 GB at 725 species, 1.6 GB at 2606 and 5.9 GB at 5741
  **[measured, /usr/bin/time, `results/peak_memory.json`]**. The dense matrix alone is 8n² bytes (4 MB at 725, 264 MB at 5741),
  and forward-mode autodiff carries n tangents through every gather. The sparse solver's
  arrays are under 20 MB at 5741 species **[estimate from the entry counts]**. Render's
  2 GB worker cannot hold a dense multi-VOC run; `vmap` batching multiplies all of this.
- `RateModel` and `Mechanism.build` take 0.01-0.05 s at 725 species and scale linearly
  **[measured]**; not a concern at 17 000 reactions.

### 8.5 Rule

Registry entries declare the solver. Dense Kvaerno3 is allowed up to 800 species (the
tutorial, inorganic, background); above that only the sparse solver, and the API refuses a
dense request.

## 9. Coupling to TOMAS

- **Volatility classes do not grow with the mechanism.** `assign_classes` lumps every
  closed-shell organic into the same 15 half-decade classes (TOMAS slots 1-15). Gather and
  scatter in `ClassBridge` are numpy over the condensable indices; at about 2000
  condensable species they are still well under 1 ms per interval **[estimate]**.
- **c\* table per mechanism.** The registry names one table (or a list merged with a
  duplicate check: two tables giving different c\* for one species is an error). The run's
  `cstar_fallback` diagnostic and warning stay as they are.
- **Sulfur.** SA (from SO3 + H2O) already feeds the TOMAS H2SO4 source. A DMS chain adds
  MSA (CH3SO3H), a closed-shell CHOS species that `assign_classes` would put in a c\* class
  with the Li et al. (2016) estimate. MSA is a strong acid whose uptake is not ideal
  partitioning; until Shantanu advises, mark it non-condensable in the registry (open
  question).
- **Halogens.** HNO3aq from the JPL heterogeneous reactions joins particle nitrate like
  NA_HET. HCl uptake, chloride in particles and halogenated SOA are out of scope.
- `tomas_jax/physics/so2_chemistry.py` (the Simulator's prescribed-OH SO2 + OH, Sun et al.
  2022) stays separate: it is for runs without gas chemistry.

## 10. c\* for products of new VOCs

| Option | What it is | Cost | Notes |
|---|---|---|---|
| A. Nannoolal via UManSysProp (as Shantanu) | SMILES → Nannoolal T_b → Nannoolal p_sat(298.15 K) → C\* = 10⁶ · MW · p_sat / (R · 298.15) | 13 s for 5834 species **[measured]** | needs SMILES: PyCHAM's MCM v3.3.1 species xml (5834 entries in the benchmark copy `pycham-bench/inputs/G3/limonene_mcm.xml`, which appends the limonene file's extra species); MCM's export page also offers a species list; openbabel is in the pycham-bench venv |
| B. Li et al. (2016) | composition only | free, already in the code | ignores structure; no halogen term; gave PAN-dominated SOA in Level 3 (ASSUMPTIONS A8) |
| C. Ask Shantanu | his own PyCHAM run per mechanism | his time | best provenance; slow loop |

Symbols: T_b normal boiling point [K], p_sat pure-component saturation vapour pressure [Pa],
MW molar mass [g mol⁻¹], R = 8.314 J mol⁻¹ K⁻¹, C\* [µg m⁻³].

Prototype `cstar_nannoolal.py` [measured]: against Shantanu's 700-row table, median
|Δlog10 C\*| = 1.0e-7, 99th percentile 0.025, maximum 0.028 dex; 530 rows within 0.01 dex;
the largest deviations are all hydroperoxides (+0.025 to +0.028 dex, cause not checked;
possibly an openbabel or MW difference). This also answers ASSUMPTIONS question 1 as far as
the numbers can: the table is consistent with the Nannoolal boiling point, PyCHAM's default
for the `nannoolal` option (`prop_calc.py`).

Recommendation: A, as an offline build step that writes one table per mechanism with a
provenance file like today's; B only for species without SMILES; then C as a review (send
Shantanu the table and the top contributors of one run). Before any halogen module produces
organics, make `log10_cstar_li2016` raise for Cl or Br.

## 11. API, scenario and cache key

- Scenario (the next-phase configuration tree): `[chemistry] mechanism = "mcm_limonene"`
  (default, the only value the tutorial accepts) and `modules = ["jpl_halogen"]` (optional
  toggles the entry allows). `gases_ppb` and `hold_constant` are validated against the
  resolved species list, as today.
- Mechanism hash = SHA-256 of the canonical JSON of the resolved mechanism (species and
  atoms, reactions with normalised rate text, RO2 list, J map, counters, external reactions,
  patches, c\* table contents). It goes into the run id next to `code_version()`.
- `code_version()` should hash code only: today it adds every `.eqn` and `.csv` under
  `tomas_jax/`, so adding `mcm_bvoc.eqn` would re-key every cached tutorial result even
  though no tutorial number changes. With the mechanism hash in the key, data files can be
  left out of `code_version()`.
- Endpoints (extending the next-phase plan's `/api/v2/pro/mechanism`):
  `GET /api/v2/pro/mechanisms` (id, title, sizes, solver, status, measured run-time class,
  c\* coverage), `GET /api/v2/pro/mechanisms/{id}` (species with formula, MW, c\*, module of
  origin; about 40 kB at 725 species, about 0.3 MB at 5800 **[estimate]**),
  `GET /api/v2/pro/mechanisms/{id}/reactions?species=...` (paged; the full table is too big
  to send at 17 000 reactions).
- Limits per mechanism: maximum duration and finest coupling interval, so a full-MCM run
  cannot be submitted with Pro "Accurate" numerics until it has been timed on Render.

## 12. UI

- **Mechanism picker** (Pro Build, section "Gases and chemistry"): one card per registry
  entry with species and reaction counts, primary VOCs, solver, measured run time for the
  Pro default, c\* source and coverage, validation status, and the optional modules as
  toggles. Incompatible choices are disabled with the reason ("JPL halogens need a rate set
  for CH4 + Cl: JPL"). Changing the mechanism keeps gas inputs that still exist and lists
  the ones dropped.
- **Gas table**: species search over the resolved list (5800 names need search, not a
  list), with the mechanism's primary VOCs as quick picks.
- **Chemistry network tab** (Pro plan section 5): the species-centred flux view works for
  any mechanism. Additions: each reaction carries its module and source (MCM label or JPL
  table id) in the reaction table; the carbon Sankey runs per primary VOC (species tagged
  by which roots they are reachable from); halogen families get a Cl/Br budget view.
  Flux payload grows with reactions: 0.71 MB gzipped for 2135 reactions × 96 frames
  (Pro plan), so about 6 MB at 17 000 reactions **[estimate, linear]**. For large
  mechanisms serve per-species flux queries from the worker instead of one blob.

## 13. Tests

| Test | Gate | Prototype evidence |
|---|---|---|
| Composer rules 1-6 | each rule raises on a crafted violation; inorganic + limonene equals the limonene file | all six shown in `compose_demo.py` |
| Atom balance per reaction | residual (products − reactants) is an integer combination of CO2 omitted from the products, H2O on either side and a half-integer of O2, or the reaction is an RO2 permutation; N, S, Cl and Br balance exactly; everything else in a per-mechanism exception list | limonene: 533 balanced, 1297 explained by omitted CO2/H2O/O2, 280 RO2 permutations, 25 exceptions: the 3 NA/SA sinks, CH4 + CL without HCl, `<13917>` CO2N3PAN + OH (N lost) and the 4 carbon gains already in ASSUMPTIONS A4, plus 16 with an odd hydrogen residual, to review |
| Element conservation in a run | total S (SO2 + HSO3 + SO3 + SA) and N (with counters) conserved | S closes to 4.6e-16 over 24 h |
| Sub-mechanism equivalence | a mechanism and a superset with the extra species at zero give the same run | inorganic vs limonene file (no VOC): O3 1.6e-6, SO2 1.2e-6, OH 3.2e-3 max relative difference (solver noise at rtol 1e-3) |
| Rate constants vs MCM reference | every K and every reaction k at three (T, p, RH) points against an independent implementation (F0AM, as already done for 33 K values; or `constants_mcm.f90` compiled) | today's F0AM check; extend to the full list |
| MCM vs JPL table | documented, not a gate | `rate_crosscheck.py` |
| J names | every J of every registry entry resolves; TUV-x vs MCM within today's tolerance | `compose_demo.py` J check |
| Sparse vs dense | same solve to 1e-9; coupled 24 h end values within 0.1 % (Pro plan gate) | 1.3e-11 / 1.2e-9 |
| c\* pipeline | reproduces Shantanu's table: median below 1e-4 dex, maximum below 0.05 dex | 1e-7 / 0.028 |
| Regression vs PyCHAM, same mechanism | PyCHAM reads the same `.eqn` and its MCM xml; extend `benchmarks/pycham` (`write_eqn(src=...)`) to `mcm_inorganic` (gas only: OH, SO2, SA) and one multi-VOC mechanism; no worse than G2 (median \|NMB\| 1.7e-3 at 300 s) | harness exists |
| Tutorial invariance | the three presets bit-identical with the registry merged | to add |
| SANDBOX comparison | `jpl_strat` against a SANDBOX scenario | to add |

## 14. Phases (after 2026-10-27)

Effort is focused person-days **[estimate]**.

| Phase | Work | Effort | Depends on |
|---|---|---|---|
| R1 | Registry, resolver, composer (from the prototype), content-keyed patches, per-mechanism counters and reported species, `code_version` split; tutorial bit-identical | 4-6 | nothing |
| R2 | `mcm_inorganic` and `mcm_background` (one MCM export with methane, ethane, propane, DMS); parser adapted to the York KPP export; the six J names | 3-4 | R1, MCM export |
| R3 | Sparse Ros3 solver (Pro plan Phase 1) | in the Pro plan | none |
| R4 | c\* build step (UManSysProp), tables for R2 and R5 | 2-3 | R1 |
| R5 | `mcm_bvoc` export, timings, PyCHAM regression | 3-5 | R3, R4 |
| R6 | JPL rate functions, `rate reactants` vs consumed species, `+jpl_halogen` generated from SANDBOX, `jpl_strat`, SANDBOX comparison | 5-8 | R1 |
| R7 | API (mechanism id, hash, endpoints, limits), Pro picker, chemistry-tab additions | 4-6 | R1, Pro Phases 2-4 |

Total about 21-32 person-days on top of the Pro plan.

## 15. Open questions

For Shantanu:
1. Is Nannoolal (T_b and p_sat) via UManSysProp the method he wants for other VOCs, and is
   the hydroperoxide offset (+0.025 dex) something PyCHAM adds on purpose?
2. Which VOCs matter for his students: biogenic only, or also toluene and alkanes?
3. MSA from DMS: condensable with an estimated c\*, non-volatile, or gas only?
4. Keep MCM's lumped sinks `HNO3 = NA` (6e-6 s⁻¹) and `SO3 = SA` in new mechanisms, or
   replace HNO3 loss by deposition/uptake as N2O5 already is?
5. Should multi-VOC mechanisms keep his chamber edit (LIMONENE + OH scaled to 1.47e-10)
   or use stock MCM limonene?

For Ali:
6. M1-M5 above.
7. Is the stratospheric case (`jpl_strat`, SANDBOX in TOMAS-JAX) a Pro goal or a research
   tool only? It decides whether the JPL rate set gets UI.

For the MCM team: redistribution of exported subsets in a web app and in notebook exports
(see [`mcm_coverage.md`](mcm_coverage.md) section 5.5).

## 16. Sources

- MCM v3.3.1 site: [home](https://mcm.york.ac.uk/MCM) (17224 reactions, 5832 species, methane
  and 142 non-methane VOCs), [export](https://mcm.york.ac.uk/MCM/export) (FACSIMILE and KPP;
  "Include inorganic reactions?", "Include generic rate coefficients?"),
  [browse](https://mcm.york.ac.uk/MCM/browse) (DMS, ethane, isoprene, pinenes, toluene listed),
  [photolysis](https://mcm.york.ac.uk/MCM/rates/photolysis),
  [generic](https://mcm.york.ac.uk/MCM/rates/generic) and
  [complex](https://mcm.york.ac.uk/MCM/rates/complex) rate coefficients,
  [construction](https://mcm.york.ac.uk/MCM/about/construction).
- Saunders et al. (2003), MCM v3 protocol, [ACP 3, 161](https://acp.copernicus.org/articles/3/161/2003/).
- Xue et al. (2015), chlorine module for MCM, [GMD 8, 3151](https://gmd.copernicus.org/articles/8/3151/2015/).
- Li, Pöschl and Shiraiwa (2016), [ACP 16, 3327](https://acp.copernicus.org/articles/16/3327/2016/).
- Nannoolal et al. (2008), vapour pressure by group contribution, [Fluid Phase Equilib. 269, 117](https://doi.org/10.1016/j.fluid.2008.04.020).
- Burkholder et al. (2020), JPL Publication 19-5, [jpldataeval.jpl.nasa.gov](https://jpldataeval.jpl.nasa.gov/).
- KPP: [kpp.readthedocs.io](https://kpp.readthedocs.io/en/stable/). PyCHAM: [github.com/simonom/PyCHAM](https://github.com/simonom/PyCHAM). UManSysProp: [github.com/loftytopping/UManSysProp_public](https://github.com/loftytopping/UManSysProp_public).
- In this repo and its neighbours: `tomas_jax/gaschem/*`, `tomas_jax/box/{coupling,driver,result,scenario}.py`,
  the Pro plan, the next-phase plan and `performance_cloud.md` (tomas-api `feat/aaar-pro` worktree,
  `docs/aaar/pro/`, `docs/aaar/next/`), SANDBOX `stratchem-jax/REACTIONS.md` and `reactions.py`.
