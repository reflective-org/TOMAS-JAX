# MCM coverage of the SANDBOX species, and what to add for a broader Pro mechanism

Status: planning note for branch `feat/pro-mechanism` (from `feat/aaar-tutorial` 5227ddd). Build
after 2026-10-27. Nothing in the model code, presets or tutorial is changed by this note.

Sources used: the mechanism file in this repo
(`tomas_jax/gaschem/mechanisms/limonene_mcm.eqn`), SANDBOX
(`/Users/ali/Documents/GitHub/gas-phase-chemistry/SANDBOX/stratchem-jax/REACTIONS.md` and
`reactions.py`), the public MCM web pages at [mcm.york.ac.uk](https://mcm.york.ac.uk/MCM), and
the source code of the MCM website ([wacl-york/mcm-web](https://github.com/wacl-york/mcm-web)).
No MCM mechanism file was downloaded. Every count marked **[measured]** comes from the two
scripts below; **[MCM site]** is read off a public MCM page; **[published]** is a number from a
paper abstract or the paper's landing page; **[estimate]** is my judgement and must be replaced
by a count from a real export before it is used for planning.

Scripts (read-only, pure Python, no JAX):

- `prototypes/mechanism/mcm_coverage_scan.py` parses `limonene_mcm.eqn` with the package's own
  `kpp_parser` and prints, for every SANDBOX species, its MCM name, how many reactions use it,
  and its carbon-free reactions. `python prototypes/mechanism/mcm_coverage_scan.py --json out.json`.
- `prototypes/mechanism/mcm_vs_jpl_rates.py` evaluates the rate coefficients of the reactions
  the two mechanisms share, MCM form (from `tomas_jax/gaschem/mcm_rates.py` and the `.eqn`
  text) against the JPL 19-5 form SANDBOX uses, at a surface and a lower-stratosphere point.

## 1. Summary

1. **The current file already holds all of MCM v3.3.1's inorganic chemistry.** The MCM site
   lists 48 inorganic reactions: 37 thermal, 3 gas-particle and 8 photolysis **[MCM site]**
   ([thermal](https://mcm.york.ac.uk/MCM/inorganic?category=Thermal%20gas-phase),
   [gas-particle](https://mcm.york.ac.uk/MCM/inorganic?category=Gas-particle),
   [photolysis](https://mcm.york.ac.uk/MCM/inorganic?category=Photolysis)). The KPP export
   merges three pairs with the same reactants and products into one line each (O → O3,
   O1D → O, HO2 + HO2 → H2O2), so they appear as labels `<1>` to `<45>` in the file. That is
   44 carbon-free lines plus CO + OH **[measured]**.
2. **Of the 36 SANDBOX species, 21 have an MCM v3.3.1 species counterpart, 2 are MCM
   environment variables, and 13 have no counterpart.**
   - 19 of the 21 are in the file. HNO3aq and H2SO4 count only approximately: they map onto
     MCM's lumped sinks NA and SA.
   - The two counterparts not in the file: HCl (MCM `HCL`) is declared but no reaction in the
     subset uses it, and C2H6 is in MCM but not in the limonene subset.
   - O2 and H2O enter MCM only inside rate expressions.
   - No counterpart: CH3 (MCM folds CH3 + O2 into CH3O2), and every ClOx and BrOx species
     except the Cl atom and HCl. MCM has the Cl atom, but only as Cl + alkanes.
3. **MCM's sulfur chemistry is broader than SANDBOX's in the troposphere, and the subset
   already contains it.** SO2 is a reactant in 12 reactions in the file: OH + SO2 with KMT12
   fall-off, then the explicit HSO3 + O2 step; O(3P) + SO2; and 10 stabilised Criegee
   intermediate + SO2 reactions. SO3 + H2O gives SA. The full MCM has 51 SO2 sinks, 49 of
   them Criegee reactions, plus 5 SO2 sources from the DMS scheme **[MCM site]**
   ([SO2 page](https://mcm.york.ac.uk/MCM/species/SO2)). Adding a VOC subset brings in its
   Criegee + SO2 reactions automatically. DMS is the single addition that gives a sulfur
   source.
4. **SANDBOX has chemistry that MCM does not have and never will.** That covers ClOx, BrOx,
   heterogeneous halogen and N2O5 chemistry on sulfate, HNO3aq re-evaporation, N2O5 and HNO4
   photolysis, O(1D) + CH4, and the SO2 + HO2 upper limit. MCM's Cl chemistry is the Cl atom
   reacting with 41 alkanes, with no ClO, Cl2, HOCl or ClONO2 and no HCl as a product **[MCM
   site]** ([CL page](https://mcm.york.ac.uk/MCM/species/CL)). MCM has no bromine chemistry
   apart from the two bromocarbon VOCs. Tropospheric Cl chemistry needs the Xue et al. (2015)
   module. Stratospheric halogen and heterogeneous chemistry has to come from JPL 19-5, as in
   SANDBOX.
5. **For "more comprehensive", the most useful additions are:**
   - ethane and propane (small; background air);
   - DMS (marine sulfur);
   - isoprene (602 species, 1926 reactions on its own **[published]**);
   - alpha-pinene and beta-pinene;
   - toluene, and possibly benzene and the xylenes.

   CO, HCHO, methanol, acetone, formic and acetic acid, glyoxal and methylglyoxal are already
   in the file with their full downstream chemistry. The MCM site does not publish per-VOC
   subset sizes, so apart from isoprene every size below is an estimate. A "background +
   biogenic + toluene" mechanism is likely to have 2000 to 3000 species
   **[estimate]**. That is 3 to 4 times today's 725, and the dense gas solver cannot carry
   it. The sparse Ros3 solver in the Pro plan is a prerequisite.
6. **Combine VOCs in one MCM export, not by stitching files.** The MCM exporter follows every
   product from all selected root VOCs, adds the inorganic block once, and writes a single
   RO2 sum. That is how this file was made (roots LIMONENE and CH4; section 5.2). Shantanu's
   file carries two edits of its own: the LIMONENE + OH branches are scaled to the chamber kOH,
   and a D9 OH tracer is added. A merge must re-apply those edits to the new export, or add
   only the new reactions to the existing file (section 5.4).
7. **Licence:** the MCM site states no licence. The University of York legal statement linked
   from every page says content "may not be reproduced without prior permission" and allows
   downloading "for personal, non-commercial reference only"
   ([York legal statements](https://www.york.ac.uk/about/legal-statements/)). The tomas-jax repo
   already ships an MCM subset. Written permission from the MCM team should cover the repo
   copy, the Pro web display and Pro exports (section 5.5).

## 2. The 36 SANDBOX species in MCM v3.3.1

Columns:
- **MCM name:** the MCM v3.3.1 species name; "none" means MCM has no such species.
- **In file:** the species occurs in at least one reaction of `limonene_mcm.eqn`.
- **Rxns:** distinct reactions of the file that use the species as reactant or product
  **[measured]**. A reaction where it is on both sides counts once.
- **Loss / prod:** reactions with the species on the reactant / product side.
- **Inorg:** of those, the carbon-free ones, given by label. Their equations are in section
  3.1.

| SANDBOX | MCM name | In MCM v3.3.1 | In file | Rxns | Loss / prod | Inorg (labels) | Notes |
|---|---|---|---|---|---|---|---|
| Cl | `CL` | yes, Cl + 41 alkanes, 32 sources from chlorinated VOCs | yes | 1 | 1 / 0 | 0 | only `<2676>` CH4 + CL = CH3O2 (no HCl product). Nothing in the subset makes CL, so it stays at its initial value |
| ClO | none | no | no | 0 | | | JPL only |
| ClOOCl | none | no | no | 0 | | | JPL only |
| ClONO2 | none | no | no | 0 | | | JPL only |
| HCl | `HCL` | yes, only as a product of CHCLOOA; no sinks | declared, unused | 0 | | | MCM keeps no HCl budget |
| Cl2 | none | no | no | 0 | | | |
| NO | `NO` | yes | yes | 203 | 198 / 5 | 11: 3, 4, 7, 9, 10, 11, 21, 24, 39, 40, 42 | 192 organic (mostly RO2 + NO) |
| NO2 | `NO2` | yes | yes | 602 | 75 / 528 | 21: 3, 4, 5, 7, 8, 9, 10, 11, 12, 22, 23, 24, 25, 26, 27, 28, 39, 41, 43, 44, 45 | |
| O | `O` (O(3P)) | yes | yes | 10 | 6 / 4 | 10: 1, 2, 3, 4, 5, 6, 30, 37, 39, 41 | all inorganic |
| O2 | (auxiliary `O2`) | constant, 0.2095 M | in 18 rate expressions | | | | not a species in MCM; SANDBOX tracks it |
| O3 | `O3` | yes | yes | 81 | 42 / 39 | 8: 1, 2, 7, 8, 14, 18, 36, 37 | 35 VOC + O3 reactions in the subset |
| CH3 | none | no | no | | | | MCM writes CH4 + OH = CH3O2 (CH3 + O2 is instantaneous) |
| CH4 | `CH4` | yes (a root VOC) | yes | 2 | 2 / 0 | 0 | `<46>` CH4 + OH, `<2676>` CH4 + CL; no O1D + CH4 in MCM |
| HNO3 | `HNO3` | yes | yes | 40 | 3 / 37 | 4: 22, 29, 33, 43 | 36 organic sources (NO3 + aldehyde H-abstraction) |
| H2O | (auxiliary `H2O`) | from RH | in 17 rate expressions | | | | `H2O` is declared but not used as a species |
| HOCl | none | no | no | | | | |
| N2O5 | `N2O5` | yes | yes | 3 | 2 / 1 | 3: 12, 34, 44 | `<34>` is replaced in code by gamma-based uptake (NA_HET, tutorial A1) |
| NO3 | `NO3` | yes | yes | 226 | 197 / 29 | 11: 5, 8, 10, 11, 12, 23, 27, 29, 40, 41, 44 | |
| OH | `OH` | yes | yes | 785 | 502 / 342 | 18: 13, 14, 15, 17, 18, 19, 21, 22, 23, 24, 26, 27, 28, 29, 31, 38, 42, 43 | |
| HO2 | `HO2` | yes | yes | 573 | 247 / 340 | 12: 14, 15, 17, 18, 19, 20, 23, 24, 25, 27, 32, 45 | |
| O1D | `O1D` | yes | yes | 3 | 2 / 1 | 3: 6, 13, 36 | |
| HONO | `HONO` | yes | yes | 3 | 2 / 1 | 3: 21, 28, 42 | |
| HNO4 | `HO2NO2` | yes | yes | 3 | 2 / 1 | 3: 25, 26, 45 | MCM has thermal decomposition (KMT10), no photolysis |
| OClO | none | no | no | | | | |
| Br | none (`BR` is declared in the file header but unused) | no Br-atom chemistry | no | | | | MCM's only bromine species are CH3BR and DIBRET (VOCs) |
| BrO | none | no | no | | | | |
| BrONO2 | none | no | no | | | | |
| BrCl | none | no | no | | | | |
| HBr | none | no | no | | | | |
| HOBr | none | no | no | | | | |
| HNO3aq | `NA` (closest) | yes, `IGNORE` sink | yes | 2 | 0 / 2 | 2: 33, 34 | MCM's NA is a terminal "nitrate aerosol" sink with fixed first-order rates. SANDBOX's HNO3aq re-evaporates |
| C2H6 | `C2H6` | yes (root VOC) | no | 0 | | | ethane chain not in the subset |
| SO2 | `SO2` | yes | yes | 12 | 12 / 0 | 2: 30, 31 | plus 10 Criegee + SO2 (section 3.2) |
| H2O2 | `H2O2` | yes | yes | 13 | 2 / 11 | 3: 17, 20, 38 | 10 organic sources: Criegee + H2O |
| SO3 | `SO3` | yes | yes | 13 | 1 / 12 | 3: 30, 32, 35 | |
| H2SO4 | `SA` | yes, `IGNORE` sink | yes | 1 | 0 / 1 | 1: 35 | tomas-jax reads SA as the H2SO4 production counter for TOMAS (`box/driver.py`, `COUNTER_SPECIES`) |

Tally (21 + 2 + 13 = 36):
- **21 with an MCM species counterpart:** Cl, HCl, NO, NO2, O, O3, CH4, HNO3, N2O5, NO3, OH,
  HO2, O1D, HONO, HNO4, HNO3aq (as NA, approximate), C2H6, SO2, H2O2, SO3 and H2SO4 (as SA,
  approximate). All but HCl and C2H6 are in the file.
- **2 auxiliary variables:** O2 and H2O.
- **13 with no counterpart:** ClO, ClOOCl, ClONO2, Cl2, HOCl, OClO, Br, BrO, BrONO2, BrCl,
  HBr, HOBr and CH3. MCM has Cl and HCl as species, but none of SANDBOX's Cl reactions
  except Cl + CH4, and that with different products.

The file has 728 declared species, 725 used in reactions, 2135 reactions and 146 RO2 sum
members **[measured]**. The species only consumed and never produced are CH4, CL, D9,
LIMONENE and SO2. The sinks are D9PROD, NA and SA.

## 3. What the file already has

### 3.1 Inorganic reactions (all of MCM v3.3.1)

Symbols: T is temperature [K]; M, O2, N2 and H2O are number densities [molec cm⁻³]. The KMTxx
fall-off coefficients use the IUPAC Troe form

  k = k0·k∞·F / (k0 + k∞),  log10 F = log10 Fc / (1 + (log10(k0/k∞) / Nc)²),  Nc = 0.75 − 1.27·log10 Fc,

with k0 = k0'·M the low-pressure limit [cm³ s⁻¹] and k∞ the high-pressure limit [cm³ s⁻¹].
The parameters are in `tomas_jax/gaschem/mcm_rates.py`. J(·) is a photolysis frequency [s⁻¹]
from TUV-x or the MCM parameterisation J = l·cos(χ)^m·exp(−n·sec χ), with χ the solar zenith
angle. Rates are verbatim from the file; "(nothing)" means the products are not tracked.

| Label | Reaction | Rate coefficient |
|---|---|---|
| 1 | O = O3 | 5.6E-34·N2·(T/300)^-2.6·O2 + 6.0E-34·O2·(T/300)^-2.6·O2 |
| 2 | O + O3 = (nothing) | 8.0E-12·exp(−2060/T) |
| 3 | NO + O = NO2 | KMT01 |
| 4 | NO2 + O = NO | 5.5E-12·exp(188/T) |
| 5 | NO2 + O = NO3 | KMT02 |
| 6 | O1D = O | 3.2E-11·exp(67/T)·O2 + 2.0E-11·exp(130/T)·N2 |
| 7 | NO + O3 = NO2 | 1.4E-12·exp(−1310/T) |
| 8 | NO2 + O3 = NO3 | 1.4E-13·exp(−2470/T) |
| 9 | NO + NO = 2 NO2 | 3.3E-39·exp(530/T)·O2 |
| 10 | NO + NO3 = 2 NO2 | 1.8E-11·exp(110/T) |
| 11 | NO2 + NO3 = NO + NO2 | 4.50E-14·exp(−1260/T) |
| 12 | NO2 + NO3 = N2O5 | KMT03 |
| 13 | O1D = 2 OH | 2.14E-10·H2O |
| 14 | O3 + OH = HO2 | 1.70E-12·exp(−940/T) |
| 15 | H2 + OH = HO2 | 7.7E-12·exp(−2100/T) |
| 16 | CO + OH = HO2 | KMT05 = 1.44E-13·(1 + M/4.2E19) |
| 17 | H2O2 + OH = HO2 | 2.9E-12·exp(−160/T) |
| 18 | HO2 + O3 = OH | 2.03E-16·(T/300)^4.57·exp(693/T) |
| 19 | HO2 + OH = (nothing) | 4.8E-11·exp(250/T) |
| 20 | HO2 + HO2 = H2O2 | 2.20E-13·KMT06·exp(600/T) + 1.90E-33·M·KMT06·exp(980/T), KMT06 = 1 + 1.40E-21·exp(2200/T)·H2O |
| 21 | NO + OH = HONO | KMT07 |
| 22 | NO2 + OH = HNO3 | KMT08 |
| 23 | NO3 + OH = HO2 + NO2 | 2.0E-11 |
| 24 | HO2 + NO = NO2 + OH | 3.45E-12·exp(270/T) |
| 25 | HO2 + NO2 = HO2NO2 | KMT09 |
| 26 | HO2NO2 + OH = NO2 | 3.2E-13·exp(690/T) |
| 27 | HO2 + NO3 = NO2 + OH | 4.0E-12 |
| 28 | HONO + OH = NO2 | 2.5E-12·exp(260/T) |
| 29 | HNO3 + OH = NO3 | KMT11 |
| 30 | O + SO2 = SO3 | 4.0E-32·exp(−1000/T)·M |
| 31 | OH + SO2 = HSO3 | KMT12: k0 = 2.5E-31·M·(T/300)^-2.6, k∞ = 2.0E-12, Fc = 0.53 |
| 32 | HSO3 = HO2 + SO3 | 1.3E-12·exp(−330/T)·O2 |
| 33 | HNO3 = NA | 6.00E-06 (fixed first-order loss) |
| 34 | N2O5 = 2 NA | 4.00E-04 (fixed; replaced in code by gamma uptake) |
| 35 | SO3 = SA | 1.20E-15·H2O |
| 36 | O3 + hv = O1D | J(J_O3_O1D) (MCM J1) |
| 37 | O3 + hv = O | J(J_O3_O3P) (J2) |
| 38 | H2O2 + hv = 2 OH | J(J_H2O2) (J3) |
| 39 | NO2 + hv = NO + O | J(J_NO2) (J4) |
| 40 | NO3 + hv = NO | J(J_NO3_NO) (J5) |
| 41 | NO3 + hv = NO2 + O | J(J_NO3_NO2) (J6) |
| 42 | HONO + hv = NO + OH | J(J_HONO) (J7) |
| 43 | HNO3 + hv = NO2 + OH | J(J_HNO3) (J8) |
| 44 | N2O5 = NO2 + NO3 | KMT04 |
| 45 | HO2NO2 = HO2 + NO2 | KMT10 |

### 3.2 Sulfur reactions in the file (12 SO2 sinks, 13 SO3 reactions)

SO2 sinks: `<30>` O + SO2, `<31>` OH + SO2 (KMT12) and these 10 Criegee reactions, all with
k = 7.0E-14 cm³ s⁻¹:

| Label | Reaction | Parent chemistry |
|---|---|---|
| 2752 | CH2OO + SO2 = HCHO + SO3 | CH2OO from limonene, MACR, ISOPDNO3 and C622CHO ozonolysis |
| 14334 | LIMBOO + SO2 = LIMAL + SO3 | limonene + O3 |
| 14644 | LMKBOO + SO2 = LMLKET + SO3 | limonaketone + O3 |
| 13924 | NC4OO + SO2 = MVKNO3 + SO3 | isoprene-nitrate fragments |
| 16538 | M3FOO + SO2 = C532CO + SO3 | |
| 4223 | GAOO + SO2 = HOCH2CHO + SO3 | |
| 4229 | MGLYOO + SO2 = MGLYOX + SO3 | |
| 4254 | ACLOO + SO2 = ACETOL + SO3 | |
| 4259 | GLYOO + SO2 = GLYOX + SO3 | |
| 13864 | NC3OO + SO2 = NOA + SO3 | |

The SO2 oxidation path is the two-step OH + SO2 → HSO3 and HSO3 + O2 → HO2 + SO3. At 298 K
and 1 atm the second step runs at about 2×10⁶ s⁻¹, so HSO3 lives under a microsecond. That is
the same net reaction as SANDBOX's single "SO2 + OH → SO3 + HO2" (R79). It adds one stiff
species, which the implicit solver handles.

### 3.3 Methane chain (C1) in the file

39 reactions in which every species has at most one carbon **[measured]**:
- CO + OH;
- the 16 reactions labelled 46 to 61 (CH4 + OH to CH3O2; CH3O2 + NO, NO2, NO3, HO2 and the
  RO2 pool; CH3O; CH3NO3; CH3O2NO2; CH3OOH);
- `<3778>`, the second CH3O2 + HO2 channel to HCHO;
- HCHO + OH, NO3 and hv (two channels);
- CH3OH + OH, HCOOH + OH and CH4 + CL;
- the CH2OO Criegee chemistry.

SANDBOX has two methane reactions. CH4 + OH gives HO2 directly (R75, product lumped), and
O(1D) + CH4 gives CH3 + OH, with CH3 a dead end.

## 4. Differences between the two mechanisms

### 4.1 In MCM (and in the file), not in SANDBOX

- **Sulfur:**
  - O(3P) + SO2 (`<30>`);
  - the explicit HSO3 step (`<31>`, `<32>`);
  - KMT12 (IUPAC) instead of JPL I4;
  - 10 Criegee + SO2 reactions here, 49 in the full MCM;
  - SO3 + H2O written as first order in H2O (`<35>`), where JPL is second order;
  - the DMS scheme (not in the subset; five SO2 sources:
    [SO2 page](https://mcm.york.ac.uk/MCM/species/SO2),
    [DMS page](https://mcm.york.ac.uk/MCM/species/DMS)).
- **Reactive nitrogen:**
  - N2O5 thermal decomposition (`<44>` KMT04);
  - HO2NO2 thermal decomposition (`<45>` KMT10), which matters in the warm troposphere;
  - NO + NO + O2 (`<9>`);
  - NO2 + NO3 → NO + NO2 (`<11>`);
  - NO3 + OH (`<23>`) and HO2 + NO3 (`<27>`);
  - fixed HNO3 and N2O5 sinks to NA (`<33>`, `<34>`).
- **HOx:** H2 + OH (`<15>`), CO + OH (`<16>`, KMT05), H2O2 + OH (`<17>`). SANDBOX makes H2O2
  but has no OH + H2O2 loss.
- **Methane and carbon:** the explicit CH4 chain to CH3O2, CH3OOH, CH3NO3, CH3O2NO2, HCHO,
  CH3OH, CO and HCOOH, with the RO2 self- and cross-reaction pool through the RO2 sum.
- **Ozonolysis, Criegee and VOC chemistry:** about 2090 organic reactions in this subset.

12 of MCM's inorganic lines have no SANDBOX counterpart: 9, 11, 15, 16, 17, 23, 27, 30, 32,
33, 44 and 45.

### 4.2 In SANDBOX, not in MCM

- **ClOx (gas):**
  - ClO + NO, ClO + ClO (+M) and its reverse;
  - ClO + NO2 (+M) and the reverse (Fahey);
  - Cl + O3; O + ClO; OH + ClO (two channels); OH + HCl; HO2 + ClO;
  - photolysis of ClONO2 (two channels), ClOOCl (two), Cl2, HOCl and OClO.
- **BrOx (gas):**
  - O + BrO, Br + O3, BrO + NO;
  - BrO + ClO (three channels);
  - BrO + NO2 (+M);
  - photolysis of BrO, BrONO2 (two), BrCl and HOBr.
- **Heterogeneous (on sulfate, γ-based):** ClONO2 + HCl, ClONO2 + H2O, HOCl + HCl,
  N2O5 + H2O and BrONO2 + H2O. The products are tracked as HNO3aq, which re-evaporates
  (R70, 1e-5 s⁻¹). MCM has only the fixed N2O5 → NA sink. tomas-jax already replaces it with
  γ uptake on the TOMAS particles.
- **Photolysis MCM does not have:** N2O5 (R22), HNO4 to NO2 + HO2 and to NO3 + OH (R61,
  R62), and O2 (R68, disabled in SANDBOX).
- **Other gas reactions:**
  - O(1D) + CH4 (R47);
  - SO2 + HO2 (R78 null sink and R80 to SO3 + OH). JPL 19-5 gives only an upper limit,
    1e-18 cm³ s⁻¹, and no products. SANDBOX's products are an assumption;
  - Cl + C2H6 (R71, disabled).
- **Chlorine chemistry where MCM has a version but SANDBOX writes it differently:**
  - Cl + CH4: MCM gives CH3O2 with no HCl; JPL gives HCl + CH3.
  - Cl + C2H6: MCM gives C2H5O2.

Of SANDBOX's 81 reactions, 37 involve a halogen species and 5 are heterogeneous. 36
non-halogen SANDBOX reactions have an MCM counterpart:

| SANDBOX | MCM label |
|---|---|
| R11 | 34 |
| R18 | 43 |
| R19 | 41 |
| R20 | 40 |
| R21 | 39 |
| R23 | 37 |
| R24 | 36 |
| R25 | 12 |
| R26 | 29 |
| R27 | 22 |
| R28 | 3 |
| R29 | 5 |
| R30 | 4 |
| R31 | 7 |
| R32 | 8 |
| R33 | 21 |
| R34 | 28 |
| R35 | 10 |
| R36 | 26 |
| R37 | 24 |
| R38 | 25 |
| R39 | 1 |
| R40 | 2 |
| R41 | 14 |
| R42 | 19 |
| R43 | 18 |
| R44, R45 | 6 |
| R46 | 13 |
| R69 | 42 |
| R74, R79 | 31 + 32 |
| R75 | 46 |
| R76 | 20 |
| R77 | 38 |
| R81 | 35 |

The two halogen reactions with an MCM counterpart are R7 ↔ 2676 and R71 ↔ CL + C2H6, which
is in the full MCM but not in this subset.

### 4.3 Rate differences on the shared reactions [measured]

From `mcm_vs_jpl_rates.py`. Ratio = k_MCM / k_JPL19-5, with the JPL forms transcribed from
SANDBOX (JPL fall-off with Fc = 0.6, 298 K reference). Surface: 298.15 K, 1013 hPa,
50 % RH. Lower stratosphere: 220 K, 50 hPa, 5 ppmv H2O.

| Reaction | Surface ratio | Stratosphere ratio |
|---|---|---|
| OH + SO2 (+M) | 0.94 | 0.72 |
| SO3 + H2O, first-order SO3 loss | 0.011 | 0.21 |
| HO2 + NO2 (+M) → HNO4 | 0.58 | 0.53 |
| O + NO2 (+M) → NO3 | 0.52 | 0.40 |
| O + NO (+M) → NO2 | 1.36 | 1.24 |
| OH + NO (+M) → HONO | 1.32 | 1.15 |
| OH + NO2 (+M) → HNO3 | 0.94 | 1.30 |
| HO2 + HO2 (with M and H2O terms) | 1.14 | 1.37 |
| CH4 + OH | 1.00 | 1.11 |
| all other shared reactions | 0.86 to 1.21 | 0.87 to 1.11 |

How to read this:
- SO3 + H2O is fast either way at the surface. The MCM SO3 lifetime is about 2 ms, the JPL
  one about 24 µs, so the H2SO4 production rate is set by OH + SO2 in both.
- In the cold, dry stratosphere the MCM form gives an SO3 lifetime of about 100 s against
  21 s for JPL. A stratospheric Pro mode should use the JPL [H2O]² form.
- Most of the HO2 + NO2 and O + NO2 gap comes from Fc: IUPAC uses 0.4 and 0.6, JPL uses 0.6
  with N = 1. I have not checked which values the current IUPAC datasheets give.
- For a tropospheric Pro mechanism the plan is to keep the MCM values, so the mechanism stays
  internally consistent and traceable to IUPAC.

## 5. A broader Pro mechanism

### 5.1 What to add

MCM v3.3.1 holds methane and 142 non-methane VOCs: 17,224 reactions of 5,832 species
**[MCM site]** ([home](https://mcm.york.ac.uk/MCM)). The primary VOCs are listed by class on
the [browse page](https://mcm.york.ac.uk/MCM/browse).

Already in the file with their full downstream chemistry (the exporter follows every product,
section 5.2): CH4, CO, HCHO, CH3OH, CH3COCH3 (acetone), HCOOH, CH3CO2H, GLYOX, MGLYOX, MACR,
and the CH3CO3 and PAN chain. Missing small species: CH3CHO, C2H5CHO, MEK, MVK (MVKNO3 is in,
MVK is not), C2H4, C3H6, C2H2, C2H5OH and BIACET.

Candidate additions in priority order. MCM names are the root species to mark for export.

| Add | MCM root | Why | Size, alone (species / reactions) | Basis |
|---|---|---|---|---|
| Ethane | C2H6 | SANDBOX species; background OH sink; brings CH3CHO, C2H5O2, PAN chemistry | ~25-35 / ~80-120 incl. methane chain | [estimate] |
| Propane | C3H8 | background, marine and continental; acetone source | ~45-70 / ~150-250 | [estimate] |
| CO, methanol, acetone, HCHO | CO, CH3OH, CH3COCH3, HCHO | already present; only need initial and emission inputs in the scenario | 0 new | [measured] |
| DMS | DMS | marine sulfur: the only MCM source of SO2 (through CH3SO, CH3SO2, MSIA) and of MSA | ~20-30 / ~40-70 | [estimate]. First steps from the [DMS page](https://mcm.york.ac.uk/MCM/species/DMS): DMS + OH (abstraction 1.12E-11·exp(−250/T); addition KMT18), DMS + NO3. KMT18 is already in `mcm_rates.py` |
| Isoprene | C5H8 | dominant global biogenic VOC; Criegee + SO2; isoprene SOA (IEPOX path) | 602 / 1926 | [published], [Jenkin et al. 2015](https://acp.copernicus.org/articles/15/11433/2015/). Some products are already in the file (MACR, ISOPCO2, ISOPDO2, IEPOXB, MPAN, HMML, MAE and 28 MACR/ISOP-named species), so the net addition is smaller |
| alpha-pinene | APINENE | most common monoterpene; SOA | ~300-500 / ~900-1600 | [estimate] |
| beta-pinene | BPINENE | second monoterpene; NOPINONE path | ~250-450 / ~800-1400; overlaps alpha-pinene and limonene at C≤5 | [estimate] |
| Toluene | TOLUENE | main anthropogenic aromatic; SOA; furanone and glyoxal chemistry | ~300-450 / ~900-1400 | [estimate]. Scheme: [Bloss et al. 2005](https://acp.copernicus.org/articles/5/641/2005/) |
| Benzene, m-, o- and p-xylene, 1,2,4-TMB | BENZENE, MXYL, OXYL, PXYL, TM124B | urban mixes; each adds a few hundred species | ~150-400 / ~500-1300 each | [estimate] |
| Ethene, propene | C2H4, C3H6 | small alkenes; Criegee + SO2; KMT15 and KMT16 already coded | ~30-60 / ~100-200 each | [estimate] |
| beta-caryophyllene | BCARY | sesquiterpene SOA; 9 Criegee + SO2 entries on the SO2 page | large, ~800-1500 / ~2500-4500 | [estimate]. Scheme: [Jenkin et al. 2012](https://doi.org/10.5194/acp-12-5275-2012) |

Size of the merged mechanism. Subsets overlap in their small products (C1 to C3, PAN, glyoxal,
methylglyoxal, acetone), so the size of a merge is less than the sum of the parts. Write
n_s for the number of species and n_r for the number of reactions.

| Package | Content | n_s | n_r | Basis |
|---|---|---|---|---|
| P0 (today) | limonene + CH4 + inorganic | 725 | 2135 | [measured] |
| P1 | P0 + ethane + propane + DMS | ~800-850 | ~2400-2600 | [estimate] |
| P2 | P1 + isoprene + alpha-pinene + beta-pinene | ~1700-2200 | ~5500-7000 | [estimate] |
| P3 | P2 + toluene (+ xylenes) | ~2000-3000 | ~6500-9500 | [estimate] |
| Upper bound | full MCM v3.3.1 | 5832 | 17,224 | [MCM site] |

What the size does to the solver. Write n_J for the Jacobian nonzeros. Today n_J = 6142 for
726 species, about 8.5 per species **[measured]** (performance_cloud.md, section 3.2).

- **Dense LU (the Kvaerno3 path):** cost ∝ n_s³. Going from 726 to 2500 species multiplies
  the 1.5 ms factorization by (2500/726)³ ≈ 41, to about 60 ms per factorization, before the
  AD Jacobian, which also grows **[estimate]**.
- **Sparse LU (the Pro Ros3 path):** cost scales with the nonzeros after fill. Today that is
  8971 nonzeros and about 88 thousand flops. If the fill per species stays similar, cost grows
  close to linearly, about 3.5x for P3 **[estimate]**. The sparse gas solver is therefore a
  prerequisite for P2 and P3, and the mechanism work should land after it (Pro plan, Phase 1).
- **Memory:** a dense Jacobian needs 8·n_s² bytes, 50 MB at n_s = 2500. That is acceptable
  for one run but multiplies under vmapped sweeps.

Other knock-on work for each addition:
- **RO2 sum:** grows with the new peroxy radicals. 146 members today; each RO2 has 2 or 3
  RO2-pool reactions.
- **c* table:** Shantanu's table covers only limonene species. New condensable species need
  c* values, or they stay gas-only. The MCM species export (TSV with SMILES, InChI and mass,
  section 5.2) gives the SMILES that the Nannoolal/PyCHAM route needs.
- **Photolysis:** `mcm_rates.MCM_J_PARAMS` already has all 34 MCM J values
  ([photolysis rates](https://mcm.york.ac.uk/MCM/rates/photolysis)). The TUV-x map in
  `gaschem/photolysis.py` covers the 28 J names the file uses. New subsets will need
  J_CH3CHO, J_C2H5CHO, J_C3H7CHO_C2H4, J_C2H5NO3 and J_TC4H9NO3 mapped to TUV-x reactions,
  or added to the MCM-only list.
- **Complex rates:** `mcm_rates.complex_rates` already defines every coefficient in MCM's
  `constants_mcm`, including KMT15 to KMT18 and KBPPN. The current file does not use those
  five, but ethene, propene, ethyne, DMS and PPN chemistry need them.

### 5.2 How the MCM extraction tool works

- **Where:** [mcm.york.ac.uk/MCM/export](https://mcm.york.ac.uk/MCM/export). Older documents
  point to the Leeds site (mcm.leeds.ac.uk) and its `extract.htt` page; the York site
  replaces it.
- **Choosing roots:** open a species page (for example
  [C5H8](https://mcm.york.ac.uk/MCM/species/C5H8)), press "Add to marklist", repeat for each
  VOC, then "Export". The marklist is the root set.
- **Form options, verbatim:** "Include inorganic reactions?", "Include generic rate
  coefficients?", "Choose an output format:" FACSIMILE or KPP. A "Species List
  (Tab-Separated Values)" download is also offered.
- **KPP constants:** KPP output needs `constants_mcm.f90` from
  `/MCM/export/kpp_constants`, last updated 2023-12-19 **[MCM site]**. tomas-jax does not
  need it, because `mcm_rates.py` carries those constants. A new export is still the time to
  check them against the file.
- **What the exporter does** (source:
  [routes/export.rb](https://github.com/wacl-york/mcm-web/blob/main/routes/export.rb) and
  [lib/mcm/db.rb](https://github.com/wacl-york/mcm-web/blob/main/lib/mcm/db.rb)):
  1. Starts from every reaction whose reactant is a selected root. The root must be in
     species category "VOC".
  2. Adds, recursively, every reaction consuming a product of a reaction already in the set,
     as long as that product is a "VOC"-category species. Inorganic products such as NO2 or
     SO3 do not pull in further reactions. Organic radicals, Criegee intermediates and
     stable products do, so Criegee + SO2 comes in with any alkene.
  3. If "inorganic" is ticked, adds the fixed set of reactions that carry an inorganic
     category: the 48 listed in section 1.
  4. Collects every complex-rate token used, and the tokens they depend on, plus every
     photolysis rate used.
  5. Writes the RO2 sum from the species flagged as peroxy radicals. It warns about any
     species without SMILES, because the sum may then be wrong.
  6. Prepends the MCM citation text as comments.
- **The v3.3.1 full-mechanism archive** holds Fortran, KPP `.eqn`, KPP constants, FACSIMILE
  and a species `.tsv` with SMILES, InChI, InChIKey and mass **[MCM site]**
  ([archive](https://mcm.york.ac.uk/MCM/about/archive)).
- **How this file was made (inference):** the species only ever consumed are LIMONENE, CH4,
  CL, D9 and SO2. D9 is Shantanu's OH tracer, SO2 is inorganic, and CL is reached only through
  CH4. CH4 + OH is not a product of limonene chemistry, so the export roots were most likely
  LIMONENE and CH4 with inorganic reactions ticked. The header lines (declared but unused HCL,
  BR and H2O) and the non-sequential labels show the format differs from the current York
  KPP exporter, which writes `#DEFVAR`/`#EQUATIONS` sections and the citation block. The file
  says "SCENTS 04272022", so it came through Shantanu's toolchain (likely PyCHAM-style).
  Confirm with Shantanu before re-exporting.

### 5.3 CRI as the compact alternative

The same site serves CRI v2.2 (the "Select Mechanism" menu): 1268 reactions of 445 species
for all its VOCs **[MCM site, via search]** ([CRI](https://mcm.york.ac.uk/CRI)). Its isoprene
subset is 186 reactions of 56 species **[published]**, against 1926 / 602 in MCM
([Jenkin et al. 2019](https://eprints.whiterose.ac.uk/146669/)). CRI keeps O3 and OH fidelity
but lumps the intermediates, so it loses the explicit species the Pro chemistry tab shows. It
also has no per-species c*. A possible Pro option is "explicit MCM for one chosen VOC, CRI for
the rest". That would still be one parser and one solver, but it means two mechanism sources
to cite and keep consistent.

### 5.4 Combining several VOC subsets into one mechanism

Preferred: **one export with every root in the marklist**, inorganic and generic rates ticked,
KPP format. The exporter returns the union of the chains: each reaction once, one inorganic
block and one RO2 sum. Steps:

1. Marklist: LIMONENE, CH4 and the new roots, for example C2H6, C3H8, DMS, C5H8, APINENE,
   BPINENE and TOLUENE. Export KPP and the species TSV.
2. Re-apply Shantanu's two edits to the export, in code like the tutorial's N2O5 edit, not by
   editing the file:
   - scale the LIMONENE + OH branches to kOH = 1.47e-10;
   - add the D9 tracer.

   Then diff the export's limonene reactions against `limonene_mcm.eqn`. Every difference
   other than these two edits is either a website update after 2022-04-27 or a toolchain
   difference, and is logged for Shantanu.
3. Check that the parser accepts the York format: the `#DEFVAR`/`#EQUATIONS` headers, the
   `#INCLUDE atoms` line and the comment block. `kpp_parser.parse_eqn` raises on unknown
   content today, so it fails loudly. That is good.
4. Keep tomas-jax's code edits keyed on content, not label. `box/coupling.py` already checks
   that label 34 is "N2O5 = 2 NA" and raises otherwise. A new export renumbers labels, so
   switch the lookup to match by reactants and products.

Fallback, if the limonene file must stay byte-identical: **append a delta**. Export the new
roots alone, then merge:
- Species: the union by MCM name. MCM names are global, so the same name means the same
  species.
- Reactions: drop any whose (sorted reactants, products with coefficients, normalised rate
  text) already exists, and log any pair with the same reactants and products but a
  different rate.
- The inorganic block: taken once, from the existing file.
- RO2 sum: the union of both inline lists. Every RO2 that reacts with "RO2" must be in it, or
  the self-reaction rates are wrong.

Both routes end with the same checks:
- every reacting species declared;
- each atom conserved, apart from IGNORE species and known MCM sinks;
- no duplicate reaction keys;
- J names covered (section 5.1);
- the merged mechanism reproduces today's Level 3 run to 0.1 % when the new VOCs start at
  zero. This is the same gate as the solver swap in the Pro plan.

### 5.5 Licence, terms of use and citation

- **Licence:** the MCM pages state none. The website code
  ([wacl-york/mcm-web](https://github.com/wacl-york/mcm-web)) has no licence file (the GitHub
  API reports `license: null`). Every page links the
  [University of York legal statements](https://www.york.ac.uk/about/legal-statements/):
  content is "copyright of the University of York" unless stated otherwise, "may not be
  reproduced without prior permission", and downloading is "for personal, non-commercial
  reference only". Permission queries go to communications-support@york.ac.uk. MCM contact:
  Andrew Rickard (andrew.rickard@york.ac.uk, linked on every page).
- **Practical reading:** the research community uses and redistributes MCM subsets with
  citation, and the exporter adds the citation text to every file. Written permission is
  still the safe step before:
  - the tomas-jax repo ships a bigger subset (it already ships one);
  - the Pro tab shows MCM rate text to the public;
  - Pro exports bundle mechanism files.

  The Pro plan (section 8.4) and the next-phase plan (D12) already list this. It needs an
  owner.
- **How to cite** (the site's own wording, [cite page](https://mcm.york.ac.uk/MCM/about/cite)):
  "The chemical mechanistic information was taken from the Master Chemical Mechanism, MCM
  v3.3.1 (reference), via website: http://mcm.york.ac.uk/". Then cite the protocol papers for
  each scheme used:
  - Non-aromatic: [Jenkin et al. 1997](https://doi.org/10.1016/S1352-2310(96)00105-7) and
    [Saunders et al. 2003](https://doi.org/10.5194/acp-3-161-2003).
  - Aromatic (toluene, benzene, xylenes): [Jenkin et al. 2003](https://doi.org/10.5194/acp-3-181-2003)
    and [Bloss et al. 2005](https://doi.org/10.5194/acp-5-641-2005).
  - Isoprene: [Jenkin et al. 2015](https://doi.org/10.5194/acp-15-11433-2015).
  - beta-caryophyllene: [Jenkin et al. 2012](https://doi.org/10.5194/acp-12-5275-2012).
  - If used: the Cl module, [Xue et al. 2015](https://doi.org/10.5194/gmd-8-3151-2015) (205
    reactions: Cl-initiated VOC degradation plus inorganic Cl chemistry), and JPL 19-5
    ([JPL data evaluation](https://jpldataeval.jpl.nasa.gov/)) for halogen, heterogeneous or
    stratospheric reactions taken from SANDBOX.

## 6. Recommendation for the Pro mechanism (for review)

1. **Keep MCM's inorganic block exactly as is.** It is already complete. For the
   tropospheric default, take nothing from SANDBOX except what MCM lacks and what matters
   near the surface: HNO4 photolysis (two channels) and N2O5 photolysis. Both are small and
   are rate-only additions. Put them in a separate `extensions` mechanism file with JPL
   citations, opt-in at first.
2. **Sulfur:** MCM already does more than SANDBOX in the troposphere. Add DMS for the marine
   source. Leave SO2 + HO2 out by default: JPL gives only an upper limit and no products. Add
   a stratospheric option that switches SO3 + H2O to the JPL [H2O]² form.
3. **Halogens:** not MCM. For coastal and polluted cases, the Xue et al. (2015) Cl module
   (ClNO2, Cl2, HOCl, Cl + VOC). For stratospheric or volcanic cases, a "SANDBOX halogen"
   module carrying the 37 JPL halogen reactions and the 5 heterogeneous ones, using TOMAS
   particle surface area for γ uptake as tomas-jax already does for N2O5. Each is a separate
   file merged at build time, with its own provenance.
4. **VOCs:** P1 (ethane, propane, DMS) first. It is small and fits even the dense solver.
   Then P2 (isoprene, alpha-pinene, beta-pinene) once the sparse Ros3 solver is in. P3
   (aromatics) last. Each package needs c* values for its condensable species before it
   can make SOA.
5. **Measure before planning further.** After 2026-10-27, export each package and run
   `mcm_coverage_scan.py`-style counts on it. Replace every [estimate] in section 5.1 with
   the measured n_s, n_r, n_J and LU fill.

## 7. Open questions

1. Shantanu: which roots and which tool produced `limonene_mcm.eqn` (SCENTS 04272022)? Was
   the CH4 root deliberate, and did he make other edits besides the kOH scaling and D9?
2. MCM team: is redistribution in tomas-jax and in a public web app, including exports,
   fine under the York terms? Should a specific attribution text be shown in the app?
3. Pro scope: is a stratospheric mode in scope (SANDBOX halogens, JPL rates, cold and dry
   conditions)? Or is Pro tropospheric only, with SANDBOX coupling a separate product?
4. Which IUPAC Fc values are current for HO2 + NO2 and O + NO2? The MCM and JPL forms differ
   by a factor of about 2 (section 4.3).
