# Halogen chemistry for the Pro Simulator: sources, SANDBOX mapping, options

Status: planning note, 2026-10-16. Branch `feat/pro-mechanism` (from `feat/aaar-tutorial` 5227ddd).
Build after 2026-10-27. No model code, preset or tutorial file is changed by this note.

Scope: chlorine, bromine and iodine gas-phase and heterogeneous chemistry for a more complete
gas mechanism in the Pro Simulator. Covers what MCM has, the published Cl add-ons (Xue et al.
2015, Riedel et al. 2014), MCM-compatible Br and I modules, and the stratospheric alternatives (JPL
19-5 as in SANDBOX, IUPAC, CAABA/MECCA, GEOS-Chem). The inorganic core and the sulfur chain are
covered in [mcm_coverage.md](mcm_coverage.md). This note covers the halogens only.

How the facts were collected: from public web pages and article landing pages (abstracts) only. I
did not open any paper or supplement PDF, and I did not fetch any MCM mechanism file. I did read
three kinds of public files to check them line by line:

- the GEOS-Chem KPP file `fullchem.eqn` at commit
  [4d4b290](https://github.com/geoschem/geos-chem/blob/4d4b2905d368d845d98f919b13707b6d47e59d89/KPP/fullchem/fullchem.eqn)
  (2026-08-06);
- the F0AM halogen sub-mechanisms at commit
  [1d69b05](https://github.com/AirChem/F0AM/tree/1d69b050b21a607daabe527455feb62826ae6ed0/Chem/MCMv331)
  (2025-11-12);
- the local copy of JPL 19-5 that SANDBOX already cites
  (`gas-phase-chemistry/frank-model/references/NASA-JPL_Evaluation_19-5.pdf`), to check its rate
  conventions.

These files sit in the scratch folder only. Nothing from them is copied into the repository.

---

## 1. Summary

1. **The current mechanism has almost no halogen chemistry.** `limonene_mcm.eqn` contains a
   single halogen reaction, `<2676> CH4 + CL = CH3O2`. It has no Cl source, and its product has no
   HCl. `BR` and `HCL` are declared in the file but used by no reaction (the parser lists them in
   `unused_declared`).
2. **MCM v3.3.1 has only organic Cl chemistry.** It has 41 Cl + alkane reactions (CH4 to C12) and
   32 reactions that release Cl when chlorinated VOCs degrade. It has no inorganic chlorine (no
   Cl + O3, ClO, ClONO2 or ClNO2; HCL appears once, as a product of CHCLOOA, with no sink), and
   no bromine or iodine atom chemistry
   ([MCM species page for CL](https://mcm.york.ac.uk/MCM/species/CL)). In MCM, Cl + alkane
   reactions do not produce HCl.
3. **The Xue et al. (2015) module adds 205 Cl reactions.** It covers Cl + alkenes, aromatics,
   alkynes, aldehydes, ketones, alcohols, some acids and nitrates, plus inorganic chlorine. As far
   as the article page shows, the reaction list exists only as tables in the supplement (CC BY
   3.0). I found no KPP or FACSIMILE file and no copy in F0AM.
   ([GMD 8, 3151](https://gmd.copernicus.org/articles/8/3151/2015/))
4. **Riedel et al. (2014) is smaller.** It adds Cl + alkenes and alcohols and a simplified
   multiphase N2O5/ClNO2/HOCl/ClONO2/Cl2 scheme. A machine-readable copy for MCM exists in F0AM
   (GPL-3.0), with the heterogeneous lines commented out.
   ([ACP 14, 3789](https://acp.copernicus.org/articles/14/3789/2014/))
5. **No Br or I module comes from the MCM group.** The search found three MCM-compatible
   third-party Br/I modules, all in F0AM (GPL-3.0):
   - `Halogens_Sherwen2016.m`: Cl/Br/I from GEOS-Chem, 118 reactions, plus 9 aerosol uptakes;
   - `Halogens_MECCA.m`: a Cl/Br subset of MECCA, 81 reactions, no heterogeneous chemistry, no
     iodine;
   - `Cl_VOC_Riedel2014.m` (the Riedel scheme, item 4).

   Some papers add Br and I to MCM but publish no file
   ([Stone et al. 2018](https://acp.copernicus.org/articles/18/3541/2018/),
   [Peng et al. 2020](https://doi.org/10.1093/nsr/nwaa304)).
6. **Stratospheric halogen and heterogeneous chemistry is not in MCM.** The usable sources:
   - JPL 19-5, as in SANDBOX. Evaluation 20 (JPL Publication 25-1, 2025) has now replaced it
     ([JPL data evaluation](https://science.jpl.nasa.gov/projects/data-evaluation));
   - IUPAC;
   - GEOS-Chem `fullchem.eqn` (MIT licence). It has all of SANDBOX's halogen gas reactions except
     ClONO2 thermal decomposition, plus the sulfate/PSC uptake reactions;
   - CAABA/MECCA (GPL, and the authors ask users to contact them first).
7. **SANDBOX's halogen set is small and lacks several reactions that every other source has.**
   These include BrO + HO2 → HOBr, Br + HO2 → HBr, HBr + OH, Cl + HO2, OH + HOCl,
   OH + ClONO2, BrO + BrO and HOBr + HCl (heterogeneous). SANDBOX lists HBr as a species but
   gives it no reaction.
8. **Recommendation.**
   - Tropospheric Pro mechanism: keep MCM's IUPAC-based inorganic core. Add an optional "Cl
     module" made of inorganic Cl gas chemistry (JPL 19-5 or IUPAC), HCl-corrected MCM Cl + alkane
     reactions, and the Xue (2015) Cl + VOC reactions for species already in the mechanism.
   - Stratospheric/SAI preset: replace the inorganic core with SANDBOX's JPL 19-5 set. Add
     SANDBOX's halogen gas and heterogeneous reactions, plus the missing JPL reactions in item 7
     behind a "SANDBOX parity" switch.
   - Use one owner per reactant set, MCM-style upper-case names, new named JPL rate functions, the
     existing γ-uptake path in tomas-jax, and new `J_*` names mapped to TUV-x keys that the TUV-x
     config already has.

---

## 2. What the current mechanism contains

Census of `tomas_jax/gaschem/mechanisms/limonene_mcm.eqn` using the package parser
(`prototypes/mechanism/halogen_overlap.py` and scratch `mechplan/halo_census.py`):

| Item | Value |
|---|---|
| species / reactions | 725 / 2135 |
| declared, never used | `H2O`, `BR`, `HCL` |
| reactions with `CL` | 1: `<2676> CH4 + CL = CH3O2 : 6.6E-12*EXP(-1240./TEMP)` |
| Cl sources | none |
| photolysis names used | 28 MCM `J_*` names, none for a halogen species |

`<2676>` loses one H and one Cl in the element balance: HCl is not written as a product. The
sibling prototype `compose_demo.py` reaches the same result (`atom_balance_limonene`, label 2676).

---

## 3. Sources

### 3.1 MCM v3.3.1

From the [MCM species page for CL](https://mcm.york.ac.uk/MCM/species/CL):

- **Cl as a reactant: 41 reactions**, all with alkanes: CH4, C2H6, C3H8 (two channels), n- and
  i-butane, the pentanes, neopentane, the hexanes and methylpentanes, the dimethylbutanes,
  cyclohexane, heptane, the methylhexanes, and octane to dodecane. The products are only the
  peroxy radicals (for example `CL + C2H6 → C2H5O2 : 8.3e-11*exp(-100/T)`). No HCl is written.
- **Cl as a product: 32 reactions.** These come from the degradation of chlorinated VOCs (CHOCL,
  COCL2, CCL3O, CH3COCL, chlorinated Criegee intermediates and similar) by OH, NO3, photolysis
  (`J<11>`, `J<15>`, `J<22>`, `J<34>`) and decomposition (`KDEC`).
- **Missing:** inorganic Cl (Cl + O3, ClO, HCl, HOCl, Cl2, ClONO2, ClNO2) and any Br or I chemistry.
  The [construction page](https://mcm.york.ac.uk/MCM/about/construction) describes the protocol
  ([Jenkin et al. 1997](http://dx.doi.org/10.1016/S1352-2310(96)00105-7),
  [Saunders et al. 2003](https://doi.org/10.5194/acp-3-161-2003)), which covers VOC oxidation by
  OH, O3 and NO3.

MCM's inorganic rates follow IUPAC ([Saunders et al. 2003](https://doi.org/10.5194/acp-3-161-2003)).
SANDBOX follows JPL. This difference is behind most of the conflicts in section 5.

### 3.2 Xue et al. (2015), GMD: chlorine module for MCM

[Xue, Saunders, Wang, Gao, Wang, Zhang and Wang (2015), Geosci. Model Dev. 8, 3151-3162](https://gmd.copernicus.org/articles/8/3151/2015/),
[doi:10.5194/gmd-8-3151-2015](https://doi.org/10.5194/gmd-8-3151-2015).

- **What it covers (from the abstract):** 205 reactions of Cl-initiated chemistry for alkenes,
  aromatics, alkynes, aldehydes, ketones, alcohols, some organic acids and nitrates, plus
  "inorganic chlorine chemistry and its precursors". It was tested in an MCM box model of ClNO2
  impacts at a coastal Hong Kong site. It does not include bromine.
- **Where the list is:** the article page links a supplement PDF (747 KB) with the reaction and
  rate tables. It shows no code or data statement, and no KPP, FACSIMILE or AtChem file. I did not
  open the supplement (rule for this step). F0AM, the box model most likely to carry it, has no
  Xue file (section 3.4).
- **Terms:** the article is under CC BY 3.0 (article page). Transcribing the rate parameters with
  attribution is allowed under that licence.
- **Unknown until the supplement is read:**
  - whether Cl + alkane reactions get an HCl product;
  - which of its inorganic reactions are gas-phase and which are parameterised uptake;
  - which photolysis numbers it uses for Cl2, ClNO2, HOCl and ClONO2. MCM's `J<n>` table has none
    of these.

### 3.3 Riedel et al. (2014), ACP: Cl additions for ClNO2 studies

[Riedel et al. (2014), Atmos. Chem. Phys. 14, 3789-3800](https://acp.copernicus.org/articles/14/3789/2014/),
CC BY 3.0. According to the abstract it adds:

- Cl + alkenes and alcohols, including Cl addition to alkenes, which forms chlorinated organics;
- ClNO2 production and photolysis;
- simplified multiphase chemistry of N2O5, ClNO2, HOCl, ClONO2 and Cl2.

A machine-readable copy is in F0AM as
[`Cl_VOC_Riedel2014.m`](https://github.com/AirChem/F0AM/blob/1d69b050b21a607daabe527455feb62826ae6ed0/Chem/MCMv331/SubMechanisms/Cl_VOC_Riedel2014.m):
47 active reactions, with the 3 heterogeneous ones commented out. Problems seen in that file:

- `CLO + NO2 = CLONO2` uses a constant `2.3399e-12`, a 1 atm value that is wrong in the
  stratosphere;
- several Cl + VOC lines drop HCl and CO (`CL + HCHO = HO2`);
- one line sets `fCH3COCH3(i)=fCH3CHO(i)+1`, a likely copy error;
- it names the nitrate `CLONO2`, while the other F0AM halogen files use `CLNO3`.

### 3.4 MCM-compatible bromine and iodine

No MCM-group module exists. The MCM site lists no halogen add-ons. These are the third-party
modules found:

| Module | What | Machine-readable | Terms |
|---|---|---|---|
| F0AM [`Halogens_Sherwen2016.m`](https://github.com/AirChem/F0AM/blob/1d69b050b21a607daabe527455feb62826ae6ed0/Chem/MCMv331/Halogens_Sherwen2016/Halogens_Sherwen2016.m) + [`HalogenAerosol_Sherwen2016.m`](https://github.com/AirChem/F0AM/blob/1d69b050b21a607daabe527455feb62826ae6ed0/Chem/MCMv331/Halogens_Sherwen2016/HalogenAerosol_Sherwen2016.m) | Cl/Br/I from GEOS-Chem ([Sherwen et al. 2016](https://doi.org/10.5194/acp-16-12239-2016)), as used by Romer et al. 2018, with MCM v3.3.1 names. 118 reactions, plus 9 uptake reactions with fixed "moderate" γ on a generic surface area | yes (MATLAB) | GPL-3.0 ([repo](https://github.com/AirChem/F0AM)) |
| F0AM [`Halogens_MECCA.m`](https://github.com/AirChem/F0AM/blob/1d69b050b21a607daabe527455feb62826ae6ed0/Chem/MCMv331/SubMechanisms/Halogens_MECCA.m) | Br and Cl subset of MECCA `gas.eqn` "provided by Rolf Sander", chosen for the tropical western Pacific. 81 reactions. The header says it is not the full mechanism and lacks heterogeneous chemistry and iodine | yes (MATLAB) | GPL-3.0, derived from GPL MECCA |
| [Stone et al. 2018, ACP 18, 3541](https://acp.copernicus.org/articles/18/3541/2018/) | MCM v3.2 box model plus Br and I. The abstract does not say where the halogen scheme comes from. A supplement zip exists (not opened) | unknown | CC BY 4.0 |
| [Peng et al. 2020, Natl Sci Rev 8, nwaa304](https://pmc.ncbi.nlm.nih.gov/articles/PMC8310770/) | MCM v3.3.1 (KPP/MATLAB) "modified to include chlorine- and bromine-related reactions". Kinetic data in Table S2 (docx) | table only | CC BY 4.0 |
| [BAS Antarctic boundary-layer study](https://www.bas.ac.uk/data/our-data/publication/coupling-of-hox-nox-and-halogen-chemistry-in-the-antarctic-boundary-layer/) | MCM inorganic and organic chemistry plus I and Br reactions | not found | not stated |
| [Sommariva, Bloss and von Glasow (2012)](https://ueaeprints.uea.ac.uk/39883/) | gas-phase iodine mechanism for tropospheric box models (not MCM-specific) | not checked | not checked |

Problems seen in the F0AM Sherwen file (do not import it without checking every line):

- `CLO + CLO = CL2O2` and its reverse use `k0 = 1.60e-21 (T/300)^-2`. That value does not look
  like the JPL termolecular k0, which is 1.9e-32.
- Two lines have typos in the equation text: `'I2 + hv = I + 1'` and `'HOI = hv = I + OH'`.
- The header warns that photolysis has no temperature dependence in the cross sections.
- The uptake γ values are fixed and, by the file's own note, "likely not valid everywhere".

### 3.5 Stratospheric alternatives

| Source | Halogen content | Rate conventions | Machine-readable | Terms |
|---|---|---|---|---|
| **JPL 19-5** ([Burkholder et al. 2019](https://ntrs.nasa.gov/citations/20210006316); [download page](https://science.jpl.nasa.gov/projects/data-evaluation)). **Evaluation 20, JPL 25-1 (2025), [doi:10.48577/jpl.KFQ0TZ](https://doi.org/10.48577/jpl.KFQ0TZ)**, is now current | All ClOx, BrOx and IOx gas kinetics (Section 1), termolecular (Section 2), equilibria (Section 3), photochemistry (Section 4), heterogeneous uptake on sulfuric acid, ice and NAT (Section 5) | Bimolecular `A exp(-E/RT)`, sometimes `(T/298)^n`. Termolecular with a 298 K reference and Fc = 0.6 (checked in the local 19-5 PDF, Section 2, Eq. 2.1) | PDF tables only | Front matter: "Copyright 2019, California Institute of Technology. U.S. Government sponsorship acknowledged." No reuse licence is stated. Models routinely transcribe the parameters with citation |
| **IUPAC Task Group** ([site](https://iupac.aeris-data.fr/)) | Inorganic halogen gas-phase datasheets ([Atkinson et al. 2007, Vol. III](https://doi.org/10.5194/acp-7-981-2007)), heterogeneous ([Crowley et al. 2010](https://doi.org/10.5194/acp-10-9059-2010), [Ammann et al. 2013](https://doi.org/10.5194/acp-13-8045-2013)) | 300 K falloff reference, Fc specific to each reaction (the same family as MCM) | datasheets on the website, no bulk download found | [Legal page](https://iupac.aeris-data.fr/legal-mentions-2/): "Unless authorized, any use of works other than individual and private reproduction and consultation is prohibited." The ACP summary papers are open access |
| **GEOS-Chem `fullchem.eqn`** ([file](https://github.com/geoschem/geos-chem/blob/4d4b2905d368d845d98f919b13707b6d47e59d89/KPP/fullchem/fullchem.eqn)) | Cl/Br/I, tropospheric and stratospheric (Eastham et al. 2014; Sherwen et al. 2016; [Wang et al. 2019](https://acp.copernicus.org/articles/19/3981/2019/)). Includes ClNO3 + HCl, ClNO3 + H2O, HOCl + HCl, BrNO3 + H2O, BrNO3 + HCl, HOBr + HCl, HOBr + HBr, N2O5 + HCl and N2O5 + H2O uptake | KPP. `GCARR_ac(A, C)`, `GCJPLPR_*` (JPL falloff), `GCJPLEQ_*` (reverse via equilibrium). Uptake rates are Fortran functions `<X>uptkBy<Y>(State_Het)` that are not in the `.eqn`. Photolysis is `PHOTOL(n)` | yes (KPP) | MIT ([LICENSE](https://github.com/geoschem/geos-chem/blob/main/LICENSE.txt)). The bundled KPP code is GPL-3.0, but the `.eqn` is not KPP code |
| **CAABA/MECCA 4.0** ([Sander et al. 2019](https://gmd.copernicus.org/articles/12/1365/2019/)) | Cl, Br, I, S, Hg. The MOM organic mechanism. Rates "updated to the latest JPL recommendations". Includes the tool `xmcm2mecca`, which converts MCM web exports | KPP (`gas.eqn`) | in the article supplement | GPL ([licence page](https://www.rolf-sander.net/messy/mecca/license.html)), version not stated. The authors ask users to contact them first, to discuss co-authorship, and to share derived code |
| **SANDBOX stratchem-jax** (local) | 37 halogen reactions plus 5 heterogeneous ones, JPL 19-5 (section 4) | Python lambdas. `troe()` and `falloff()` use the JPL form. `het()` uses free-molecular γ uptake | yes (Python) | Ali's code |

Stratospheric heterogeneous chemistry in MECCA: the 2019 paper describes it for the MOZART and
CB05BASCOE options (sulfate and PSC). I did not find it described for the default MECCA
mechanism.

---

## 4. SANDBOX halogen reactions mapped to each source

SANDBOX numbers are from `stratchem-jax/REACTIONS.md`. Key: **Y** = same reactants and channel.
**~** = same reactants with a different channel split, a different form or a suspect rate.
**-** = absent. **U** = uptake reaction. The matching was done by script
(scratch `mechplan/halo_map.py`) and checked by hand.

- Xue (2015) is not a column: its per-reaction list was not opened. Its abstract implies the
  tropospheric inorganic Cl rows (R1, R4, R6, R12, R13, R16, R17, R51, R52) are present, and
  rows with Br are absent.
- IUPAC Vol. III covers the gas-phase inorganic halogen rows. I did not check it row by row.

| R# | SANDBOX reaction | MCM v3.3.1 | Riedel 2014 (F0AM) | F0AM MECCA subset | F0AM Sherwen 2016 | GEOS-Chem fullchem | Rate in GEOS-Chem vs SANDBOX |
|---|---|---|---|---|---|---|---|
| R1 | ClO + NO → NO2 + Cl | - | Y | Y | Y | Y | same (6.4e-12 e^{290/T}) |
| R2 | ClO + ClO + M → ClOOCl | - | - | Y (IUPAC form) | ~ (k0 suspect) | Y, plus 3 bimolecular channels | same JPL parameters |
| R3 | ClOOCl + M → 2 ClO | - | - | Y (IUPAC Keq) | ~ | Y (`GCJPLEQ`) | same Keq 2.16e-27 e^{8537/T} |
| R4 | ClO + NO2 + M → ClONO2 | - | ~ (constant 1 atm) | Y (IUPAC) | Y | Y | same JPL parameters |
| R5 | ClONO2 + M → ClO + NO2 | - | - | Y (6.918e-7 e^{-10909/T} M, the same expression as SANDBOX) | - | - | absent |
| R6 | Cl + O3 → ClO + O2 | - | Y | Y (older 2.8e-11 e^{-250/T}) | Y | Y | same |
| R7 | Cl + CH4 → HCl + CH3 | ~ (CH3O2, no HCl, 6.6e-12 e^{-1240/T}) | - (uses MCM) | Y (+HCL) | - (uses MCM) | Y (HCl + MO2) | same (7.1e-12 e^{-1270/T}) |
| R8 | ClONO2 + HCl → Cl2 + HNO3aq (U) | - | ~ (on chloride, commented out) | - | - | U (`ClNO3uptkByHCl`) | different code |
| R9 | ClONO2 + H2O → HOCl + HNO3aq (U) | - | - | - | U (γ = 0.1 fixed) | U (`ClNO3uptkByH2O`) | different code |
| R10 | HOCl + HCl → Cl2 + H2O (U) | - | ~ (on chloride, commented out) | - | - | U (`HOClUptkByHCl`) | different code |
| R11 | N2O5 + H2O → 2 HNO3aq (U) | ~ (`<34>` constant 4e-4 s-1; tomas-jax replaces it with γ uptake) | ~ (N2O5 → ClNO2, commented out) | - | - | U | different code |
| R12, R13 | ClONO2 + hν (two channels) | - | Y | Y | Y | Y (`PHOTOL(19,20)`) | J |
| R14 | ClOOCl + hν → 2 Cl + O2 | - | - | - | Y (Cl + ClOO) | Y (Cl + ClOO) | J |
| R15 | ClOOCl + hν → 2 ClO | - | - | - | - | - | SANDBOX only (j45 fallback) |
| R16 | Cl2 + hν → 2 Cl | - | Y | Y | Y | Y | J |
| R17 | HOCl + hν → OH + Cl | - | Y | - | Y | Y | J |
| R48 | O + ClO → Cl + O2 | - | - | Y | - | Y | same |
| R49, R50 | OH + ClO (two channels) | - | - | ~ (one lumped line, 0.94/0.06) | Y | Y | same |
| R51 | OH + HCl → Cl + H2O | - | Y | Y | Y | Y | same |
| R52 | HO2 + ClO → HOCl + O2 | - | Y | Y | Y | Y | same |
| R53 | HO2 + ClO → HCl + O3 (disabled) | - | - | - | - | - | - |
| R54 | O + BrO → Br + O2 | - | - | Y | - | Y | same |
| R55 | Br + O3 → BrO + O2 | - | - | Y | Y | Y | same |
| R56 | BrO + NO → NO2 + Br | - | - | Y | Y | Y | same |
| R57-R59 | BrO + ClO (three channels) | - | - | Y (IUPAC) | - | ~ (Br + Cl + O2 written as Br + ClOO) | same |
| R60 | BrO + NO2 + M → BrONO2 | - | - | Y (IUPAC) | Y | Y | JPL 15-10 (5.4e-31) vs 19-5 (5.5e-31) |
| R63 | OClO + hν → O + ClO | - | - | - | Y | Y | J |
| R64 | BrO + hν → Br + O | - | - | Y | Y | Y | J |
| R65, R66 | BrONO2 + hν (two channels) | - | - | Y | Y | Y | J |
| R67 | BrCl + hν → Br + Cl | - | - | Y | Y | Y | J |
| R71 | Cl + C2H6 → HCl (disabled) | ~ (C2H5O2, no HCl) | - | Y | Y | Y | same |
| R72 | BrONO2 + H2O → HOBr + HNO3aq (U) | - | - | - | U (γ = 0.1) | U (`BrNO3uptkByH2O`) | different code |
| R73 | HOBr + hν → OH + Br | - | - | Y | Y | Y | J |

### 4.1 Reactions other sources have and SANDBOX lacks

All of the following are in GEOS-Chem `fullchem.eqn` (checked line by line), except Cl + NO2 → ClNO2, which is only in the F0AM Sherwen file. Most are also in the F0AM MECCA subset, and they are evaluated by JPL:

- **Bromine:** BrO + HO2 → HOBr + O2; Br + HO2 → HBr + O2; HBr + OH → Br + H2O;
  BrO + BrO (two channels); BrO + OH → Br + HO2; Br + HCHO → HBr + HO2 + CO; Br2 + OH;
  Br2 + hν; Br + NO3; Br + BrONO2.
- **Chlorine:** Cl + HO2 (→ HCl + O2 and → ClO + OH); ClO + ClO bimolecular (three channels);
  Cl + O2 ⇌ ClOO; Cl + H2O2; Cl + HCHO; Cl + H2; OH + HOCl; OH + OClO; OH + Cl2;
  OH + ClONO2; O + ClONO2; Cl + ClONO2; ClNO2 + hν; Cl + NO2 → ClNO2.
- **Uptake:** BrONO2 + HCl → BrCl + HNO3; HOBr + HCl → BrCl + H2O; HOBr + HBr → Br2;
  N2O5 + HCl → ClNO2 + HNO3.

Consequences in SANDBOX today:

- HBr is a species with no reaction.
- HOBr is made only by uptake (R72).
- Cl ends in HCl only through Cl + CH4 and OH + ClO.

In the stratosphere, BrO + HO2 is the main gas-phase path to HOBr. HOBr photolysis and
HOBr + HCl are part of the bromine-driven ozone loss that SAI sulfate would increase. Adding these
reactions changes SANDBOX results, so the additions belong behind a switch (section 6.2).

---

## 5. Consistency issues

### 5.1 Duplicate reactions

The prototype `prototypes/mechanism/halogen_overlap.py` (results in
`prototypes/mechanism/results/halogen_overlap.md`) evaluates both codes, without changing them:

- SANDBOX through its own `Reaction.coeff_fn`, with `opt = 0` and the sulfur chain on;
- the MCM file through `tomas_jax.gaschem.mechanism.RateModel`.

Each side sums every channel of the same reactant set. Bath species are folded in as
k_eff = k [O2]^a [H2O]^b [N2]^c, so the two ways of writing a reaction give comparable numbers.

Symbols: T is temperature [K]; [M] is the air number density [molec cm-3]; k is a rate
coefficient [cm3 molec-1 s-1, or s-1 for pseudo-first-order]. Conditions:

- surface: T = 298 K, [M] = 2.46e19, H2O 1 %;
- 20 km: T = 217 K, [M] = 1.84e18, H2O 5 ppm;
- 25 km: T = 222 K, [M] = 8.4e17, H2O 5 ppm.

| Reactants (MCM names) | SANDBOX | MCM | k_SANDBOX / k_MCM at surface | at 20 km | at 25 km |
|---|---|---|---|---|---|
| CH4 + CL | R7 | `<2676>` | 0.973 | 0.937 | 0.940 |
| NO2 + OH | R27 | `<22>` | 1.06 | 0.777 | 0.688 |
| HO2 + NO2 | R38 | `<25>` | 1.73 | 1.90 | 1.83 |
| HO2 + HO2 | R76 | `<20>` | 0.875 | 0.726 | 0.730 |
| OH + SO2 | R79 | `<31>` | 1.06 | 1.38 | 1.45 |
| NO + O | R28 | `<3>` | 0.736 | 0.804 | 0.822 |
| NO + OH | R33 | `<21>` | 0.756 | 0.867 | 0.900 |
| SO3 (+ H2O) | R81 | `<35>` | 59.2 | 8.0 | 1.85 |
| other 16 shared sets | | | 0.92 to 1.16 | 0.89 to 1.15 | 0.90 to 1.15 |

Only one halogen reaction overlaps with the current file (`CH4 + CL`), and its rate agrees within
6 %. The products differ: SANDBOX gives HCl + CH3, MCM gives CH3O2 without HCl. SO3 + H2O differs most: JPL's [H2O]^2 law against MCM's 1.2e-15 [H2O]. Both are fast enough near the surface that this matters little there. The other large
differences are in the shared inorganic core (owned by [mcm_coverage.md](mcm_coverage.md)). Up to
a factor of 1.9 at 20 km means the core cannot mix the two evaluations reaction by reaction
without a stated owner for each reactant set.

Duplicate rules for the halogen modules:

1. **One owner per reactant set.** The composer in `prototypes/mechanism/mechlib.py` already
   raises an error when two modules describe the same reactant set. Keep that behaviour, and
   resolve each case with an explicit `allow_overlap` entry naming the winning module.
2. **CH4 + CL.** Replace `<2676>` with `CH4 + CL = HCL + CH3O2 : 7.1e-12*exp(-1270/T)` (JPL 19-5).
   CH3 + O2 → CH3O2 is instantaneous, so MCM's CH3O2 is the right product.
3. **The 41 MCM Cl + alkane reactions.** If ethane and larger alkanes are exported later, a patch
   module adds `HCL` to each product list and keeps MCM's rates. Without it, every Cl atom that
   reacts with an alkane leaves Cl unaccounted for, and the Cly budget does not close.
4. **SANDBOX lumps that duplicate MCM chains.** Drop R75 (`CH4 + OH → HO2`) and R74/R78 (sulfur
   reference lumps), and keep MCM's explicit CH4 chain. SANDBOX's R47 (O1D + CH4) has no MCM
   counterpart, so add it with product CH3O2 + OH.
5. **N2O5 uptake.** tomas-jax already replaces MCM `<34>` with γ uptake (`box/coupling.py`,
   `physics/heterogeneous.py`). SANDBOX R11 is the same process. Use one implementation, the
   tomas-jax one, and pass it SANDBOX's γ when in stratospheric mode.

### 5.2 Species names

KPP writes Fortran, which ignores case: `ind_Cl` and `ind_CL` are the same name to a Fortran
compiler. The tomas-jax parser is Python and case-sensitive, so `Cl` and `CL` would silently become
two species. F0AM shows the risk of mixed names: its Riedel file uses `CLONO2`, while its MECCA and
Sherwen files use `CLNO3` for the same molecule.

Proposed canonical names (MCM style, upper case) and their aliases:

| Canonical | SANDBOX | GEOS-Chem | F0AM | Note |
|---|---|---|---|---|
| CL | Cl | Cl | CL | already declared in the MCM file |
| CLO | ClO | ClO | CLO | |
| CL2O2 | ClOOCl | Cl2O2 | CL2O2 | |
| CLONO2 | ClONO2 | ClNO3 | CLONO2 / CLNO3 | pick one. CLONO2 is closer to SANDBOX and JPL |
| CLNO2 | (none) | ClNO2 | CLNO2 | needed for tropospheric Cl |
| HCL | HCl | HCl | HCL | already declared in the MCM file |
| CL2, HOCL, OCLO, CLOO | Cl2, HOCl, OClO, (none) | Cl2, HOCl, OClO, ClOO | same | |
| BR, BRO, HBR, HOBR, BR2, BRCL | Br, BrO, HBr, HOBr, (none), BrCl | Br, BrO, HBr, HOBr, Br2, BrCl | same | BR is already declared |
| BRONO2 | BrONO2 | BrNO3 | BRNO3 | |
| HO2NO2 | HNO4 | HNO4 | HO2NO2 | MCM name wins |
| CH3O2 | CH3 | MO2 | CH3O2 | CH3 is folded into CH3O2 |
| NA_HET or a new HNO3AQ | HNO3aq | (gas HNO3) | (none) | see 5.4 |

Build rule: the parser should reject two species whose names differ only in case, and modules
should map their names through one alias table at load time.

### 5.3 Rate-expression conventions

Symbols: T is temperature [K]; [M] is the air number density [molec cm-3]; k0 is the
low-pressure limit [cm6 molec-2 s-1]; k∞ is the high-pressure limit [cm3 molec-1 s-1]; Fc is the
broadening factor.

**JPL 19-5 termolecular** (Section 2, Eq. 2.1; this is what SANDBOX `mechanism.falloff` does):

    k0(T)  = k0_298 (T/298)^(-n),    k∞(T) = k∞_298 (T/298)^(-m)
    k_f    = k0(T)[M] / (1 + k0(T)[M]/k∞(T)) · 0.6^{ 1 / (1 + [log10(k0(T)[M]/k∞(T))]^2) }

**MCM / IUPAC termolecular** (`tomas_jax/gaschem/mcm_rates._troe`):

    k0 = k0_300 [M] (T/300)^(-n),    k∞ = k∞_300 (T/300)^(-m)
    Nc = 0.75 - 1.27 log10(Fc)
    log10 F = log10(Fc) / (1 + [log10(k0/k∞)/Nc]^2)
    k = k0 k∞ F / (k0 + k∞)

Here Fc is specific to each reaction (mostly 0.3 to 0.85 in MCM), while JPL fixes Fc = 0.6 and Nc = 1.

**JPL chemical activation** (Table 2-2: OH + HNO3, O + NO2). The pressure-independent channel is
k_int (1 − k_f/k∞), with k_int = A exp(−B/T). MCM writes such reactions as separate channels with
their own expressions.

**JPL reverse reactions** (R3, ClOOCl → 2 ClO) are computed as k_rev = k_f / K_eq, with
K_eq = A exp(B/T) [cm3 molec-1] from Section 3.

**Bath species.** MCM writes O2, N2, H2O and M into the rate (for example
`<13> O1D = OH + OH : 2.14E-10*H2O`). SANDBOX writes H2O as a reactant species, and some SANDBOX
rates already contain [H2O] (R81 has [H2O]^2 in total). Every imported reaction must be rewritten
in MCM style. The prototype's k_eff ratios for O + O2 + M (1.06, 1.00, 1.00) and for the O1D sinks
(0.99 to 1.01) confirm that the folding works.

**What the tomas-jax parser accepts today.** A rate expression may contain arithmetic, `EXP`,
`LOG10`, `TEMP`, `M`, `O2`, `N2`, `H2O`, `RO2`, `J(...)` and the names returned by
`mcm_rates.complex_rates`. A JPL fall-off can be written inline (it needs only `**` and `LOG10`),
but that is long and easy to get wrong. Build-phase proposal: add a `jpl_rates.py` beside
`mcm_rates.py` that returns named coefficients (for example `KJPL_CLO_NO2`, `KJPL_CLOOCL_DEC`),
computed by `falloff()` and `troe()` copied from SANDBOX, and test it against SANDBOX's numbers.

**Heterogeneous rates cannot go in an `.eqn` line**, because they depend on the aerosol.

- tomas-jax already handles one such reaction. `RateModel(parsed, external=("34",))` sets the rate
  of `<34>` from `physics.heterogeneous.n2o5_uptake`, which is the condensation sink with α
  replaced by γ:

      k_het = 2π D_g Σ_k Dp_k N_k β(Kn_k, γ) / V   →   γ c̄ A / 4   for Kn ≫ 1

  Here D_g is the gas diffusivity [cm2 s-1], Dp_k the wet diameter of bin k [cm], N_k the number
  in bin k, β the Fuchs-Sutugin transition factor, V the box volume [cm3], c̄ = (8 k_B T / (π m))^(1/2)
  the mean molecular speed, m the molecular mass, k_B Boltzmann's constant, and A the aerosol
  surface area density [cm2 cm-3].
- SANDBOX uses only the free-molecular limit (`khet`) with SA in µm2 cm-3.
- Plan: generalise `n2o5_uptake` to a list of (species, γ function, product counter) entries and
  mark each uptake label as external.

Two uptake details need decisions:

- **Bimolecular uptake** (ClONO2 + HCl, HOCl + HCl, HOBr + HCl). SANDBOX makes the rate
  first-order in the gas taken up (`rate_law={driver: 1}`) but still consumes HCl. The tomas-jax
  `Mechanism` uses mass action over all reactants. The build must either (a) add a rate-law
  override to the mechanism arrays, or (b) divide the coefficient by [HCl], the "limiting reagent"
  trick of Parrella et al. 2012 noted in the F0AM uptake file. (a) is cleaner. (b) is singular as
  [HCl] → 0.
- **γ that depends on the gas state.** SANDBOX's `hetgammas_jpl00` γ for ClONO2 and HOCl depends
  on the HCl and ClONO2 partial pressures, the H2SO4 weight percent, T, the water activity and the
  radius, and SANDBOX recomputes it at every evaluation of the right-hand side. Holding γ fixed
  over each coupling step, as tomas-jax does for photolysis, is an approximation whose error must
  be measured against SANDBOX before it is accepted.

### 5.4 Products that are not gas species

- **SANDBOX HNO3aq.** It is the product of R8, R9, R11 and R72, and re-evaporates at 1e-5 s-1
  (R70). The MCM file has `NA` (a dummy sink in `<33>` and `<34>`), and tomas-jax has `NA_HET`
  (particle nitrate). For SAI parity, keep a counter species `HNO3AQ` with R70. For the
  tropospheric module, uptake gives `NA_HET` as now.
- **H2O.** It is a species in SANDBOX and fixed in MCM. Keep it fixed. The water consumed by
  hydrolysis is negligible next to [H2O].
- **The SANDBOX O3 photolysis "opt" switch** (`opt = 1` splits J into O and O1D by humidity) is a
  workaround for the original MATLAB. Do not port it. Use TUV-x `J_O3_O1D` and `J_O3_O3P`, and the
  MCM O1D sinks (`<6>`, `<13>`). Their summed rate matches SANDBOX R44-R46 with `opt = 0` to within 1.2 %.

### 5.5 Photolysis names

`RateModel` raises an error for any `J_*` name missing from `MCM_J_PARAMS`, and MCM has no
clear-sky parameters for halogen species. The TUV-x config tomas-jax uses
(`tuvx-jax/examples/tuv_5_4_no_aerosol.json`) already has every halogen J needed:

| Proposed tomas-jax name | Reaction | TUV-x key (present) | GEOS-Chem | F0AM | SANDBOX |
|---|---|---|---|---|---|
| J_CL2 | Cl2 → 2 Cl | `Cl2+hv->Cl+Cl` | PHOTOL(22) | Jn32 | R16 |
| J_CLNO2 | ClNO2 → Cl + NO2 | `ClNO2+hv->Cl+NO2` | PHOTOL(21) | Jn23 | (none) |
| J_HOCL | HOCl → OH + Cl | `HOCl+hv->HO+Cl` | PHOTOL(24) | Jn36 | R17 |
| J_CLONO2_CL / J_CLONO2_CLO | ClONO2 → Cl + NO3 / ClO + NO2 | `ClONO2+hv->Cl+NO3` / `ClONO2+hv->ClO+NO2` | PHOTOL(19/20) | Jn34/35 | R12/R13 |
| J_CL2O2 | ClOOCl → Cl + ClOO (→ 2 Cl + O2) | `ClOOCl+hv->Cl+ClOO` | PHOTOL(26) | Jn38 | R14 |
| (branch) | ClOOCl → 2 ClO | none (SANDBOX: j45 fallback) | (none) | (none) | R15 |
| J_OCLO | OClO → O + ClO | `OClO+hv->Products` | PHOTOL(25) | Jn37 | R63 |
| J_CLO | ClO → Cl + O | `ClO+hv->Cl+O(3P)` | PHOTOL(27) | Jn33 | (none) |
| J_BR2, J_BRO, J_HOBR, J_BRCL | | `Br2+hv->Br+Br`, `BrO+hv->Br+O`, `HOBr+hv->OH+Br`, `BrCl+hv->Br+Cl` | PHOTOL(23/28/32/33) | Jn24/25/26/31 | R64, R73, R67 |
| J_BRONO2_BR / J_BRONO2_BRO | | `BrONO2+hv->Br+NO3` / `BrONO2+hv->BrO+NO2` | PHOTOL(29/30) | Jn28/29 | R65/R66 |
| J_N2O5 | N2O5 → NO2 + NO3 | `N2O5+hv->NO2+NO3` | PHOTOL(14) | (n/a) | R22 (not in MCM) |
| J_HO2NO2 | HO2NO2 → HO2 + NO2 | `HNO4+hv->HO2+NO2` | PHOTOL(18); OH + NO3 branch PHOTOL(17) | (n/a) | R61 (not in MCM). R62 has no TUV-x branch |

Decisions for the build:

1. The halogen modules need `photolysis = "tuvx"`. In `"mcm"` mode they should raise an error,
   unless parameters l, m, n are fitted for J = l cos(χ)^m exp(−n sec χ) (χ the solar zenith
   angle; [Saunders et al. 2003](https://doi.org/10.5194/acp-3-161-2003)) to surface TUV-x for each
   new J, the way MCM did.
2. tomas-jax computes TUV-x J at the surface. A stratospheric preset needs J at the box altitude,
   which is what SANDBOX's `tuvx_photolysis_adapter._box_altitude_km` does.
3. The F0AM `Jn` and GEOS-Chem `PHOTOL(n)` numbers are local to those codes. Map by reaction, not
   by number.

---

## 6. Options for the Pro Simulator

### 6.1 Tropospheric Pro mechanism: MCM plus a Cl module

Contents of an optional module `cl_trop`, off by default:

1. **Inorganic Cl gas reactions** from JPL 19-5, or IUPAC to match the MCM core: Cl + O3,
   ClO + NO, ClO + HO2, OH + ClO (two channels), OH + HCl, Cl + HO2 (two channels), Cl + H2O2,
   Cl + H2, Cl + HCHO, ClO + NO2 + M ⇌ ClONO2, OH + HOCl, OH + Cl2, OH + ClNO2. Photolysis of Cl2,
   ClNO2, HOCl and ClONO2.
2. **The MCM Cl + alkane reactions with HCl added** (section 5.1, rule 3).
3. **Xue (2015) Cl + VOC reactions** for parent species already in the exported mechanism (for the
   limonene subset: the alkenes, aldehydes, ketones and alcohols it contains). Transcribe them
   from the CC BY supplement with a row-by-row check. The question to settle first: Cl + limonene
   itself is fast, of order 1e-10 cm3 molec-1 s-1 (estimate; to be checked against IUPAC), and it
   is probably not in Xue. Its products would need a protocol decision with Shantanu.
4. **Sources.** ClNO2 comes from N2O5 uptake on chloride-containing particles, but TOMAS-JAX does
   not carry chloride now (`SRTNACL` is commented out in `core/config.py`). Phase 1: initial or
   held ClNO2 and Cl2 as user inputs (as campaign box models do), plus an optional ClNO2 yield φ on
   the existing N2O5 uptake: N2O5 → φ ClNO2 + (2 − φ) NO3−(p). Chloride depletion waits until sea
   salt exists in TOMAS-JAX.
5. **Br and I:** not in the first release. If they are needed, transcribe from GEOS-Chem
   `fullchem.eqn` (MIT) with JPL/IUPAC checks. Do not copy F0AM (GPL-3.0, and the errors listed in
   section 3.4).

### 6.2 Stratospheric/SAI preset: JPL core plus SANDBOX halogens and uptake

1. **Core swap.** In this preset, JPL 19-5 owns every inorganic reactant set that SANDBOX
   describes (R18-R47, R61, R62, R69, R76, R77 and the sulfur chain). MCM's organic chemistry and
   its CH4 chain stay. This follows the one-owner rule, so the tropospheric preset keeps MCM/IUPAC.
2. **Halogen gas reactions.** SANDBOX R1-R7 and R48-R60 with JPL named rates (5.3). Add R5 (ClONO2
   thermal decomposition, Fahey) unchanged for parity.
3. **Reactions SANDBOX is missing** (section 4.1), behind `halogens.sandbox_parity = true|false`:
   - `true` reproduces SANDBOX exactly;
   - `false` adds at least BrO + HO2, Br + HO2, HBr + OH, Cl + HO2 (two channels), OH + HOCl,
     OH + ClONO2, O + ClONO2, Cl + HCHO, BrO + BrO, BrO + OH, and HOBr + HCl and BrONO2 + HCl
     uptake (JPL 19-5 Section 5).
4. **Uptake.** R8-R11 and R72 on the TOMAS surface area through the generalised γ uptake (5.3).
   The ClONO2 and HOCl γ come from SANDBOX's `hetgammas_jpl00`, with the H2SO4 weight percent from
   TOMAS composition, T and H2O (SANDBOX `aerosol.h2so4wp_at`), and BrONO2 γ = 0.8. This is what
   connects SAI aerosol to ozone in the box.
5. **Photolysis.** TUV-x at the box altitude with the new `J_*` names (5.5). R15 and R62 keep
   SANDBOX's j45 fallback until TUV-x has those branches.
6. **Evaluation version.** Check the halogen and uptake entries of JPL Evaluation 20 (2025) against
   19-5 before the build, and record any change as an assumption. SANDBOX stays on 19-5 unless Ali
   moves it.

### 6.3 Tests for the build

1. **SANDBOX parity.** The `strat` preset with `sandbox_parity = true`, run against stratchem-jax:
   same T, P, H2O, surface area, H2SO4 weight percent, fixed J values and initial state, 10 days,
   no aerosol feedback. Every shared species should agree within the solver tolerance. The
   threshold is to be set from a self-convergence run, not assumed.
2. **Rate unit tests.** Each `KJPL_*` against SANDBOX `coeff_fn` at the three conditions of
   section 5.1, to 1e-12 relative.
3. **Cly and Bry closure.** The sums Cly = Cl + ClO + 2 Cl2O2 + ClONO2 + HCl + HOCl + 2 Cl2 + OClO
   + BrCl + ClNO2 and Bry = Br + BrO + HBr + HOBr + BrONO2 + BrCl + 2 Br2 are conserved to the
   solver tolerance in every module when emissions and uptake are off. The MCM `<2676>` as shipped
   would fail this test, which is why it must be patched.
4. **Tutorial.** With all halogen modules off, Levels 1-3 end values are unchanged, compared bit
   for bit to the `aaar-stable` tag.

### 6.4 Effort (estimates, not measured)

| Item | Days |
|---|---|
| Alias table, case check, `jpl_rates.py` with tests | 2-3 |
| `strat` core swap and SANDBOX halogens, parity test | 3-5 |
| Generalised γ uptake (bimolecular rate law, γ from state), tests | 3-4 |
| New J names, TUV-x mapping, altitude J for `strat` | 2-3 |
| `cl_trop`: inorganic Cl, MCM HCl patch, Xue transcription for the exported species | 4-6 |
| Missing JPL halogen reactions (`sandbox_parity = false`) | 2-3 |

---

## 7. Licences and terms (not legal advice)

| Source | Terms | Implication for tomas-jax |
|---|---|---|
| MCM v3.3.1 | Website states no licence. The [cite page](https://mcm.york.ac.uk/MCM/about/cite) requires a citation. The [mcm-web repo](https://github.com/wacl-york/mcm-web) has no licence file | Already flagged in NEXT_PHASE_PLAN (D12). Ask Andrew Rickard before bundling exports |
| Xue 2015, Riedel 2014 | CC BY 3.0 (article pages) | Transcription with attribution is fine |
| JPL 19-5 / 20 | Caltech copyright, US Government sponsorship. No reuse licence stated | Transcribe parameters with citation, as all models do. Do not redistribute the PDF |
| IUPAC website | "All rights reserved", private use only | Cite the ACP papers. Do not scrape datasheets |
| GEOS-Chem `fullchem.eqn` | MIT | Can be adapted with the MIT notice |
| F0AM | GPL-3.0 | Copying files would bring GPL-3.0 into tomas-jax, whose licence is unresolved (D12). Use F0AM only as a cross-check |
| CAABA/MECCA | GPL, plus contact, co-authorship and code-sharing requests | Same as F0AM. Contact Rolf Sander if any use is planned |

---

## 8. Assumptions and open questions (for Shantanu and Ali)

Assumptions made in this note:

- **H-A1.** The canonical names are MCM style, upper case (`CL`, `CLO`, `CLONO2`, `BRONO2`), not
  SANDBOX mixed case.
- **H-A2.** HCl is added as a product to MCM Cl + alkane reactions, without changing the rates.
- **H-A3.** The stratospheric preset uses JPL 19-5 for the whole inorganic core, and the
  tropospheric preset uses MCM/IUPAC. The two are never mixed reaction by reaction.
- **H-A4.** SANDBOX R5 (ClONO2 thermal decomposition, not in JPL) is kept for parity.
- **H-A5.** The SANDBOX `opt = 1` O3 photolysis workaround is not ported.
- **H-A6.** The prototype conditions (T, [M], H2O at 0, 20 and 25 km) are representative, not
  taken from a standard atmosphere table.

Questions:

1. Is a stratospheric/SAI preset in Pro scope for the first release, or only for later?
2. Cl + limonene and other monoterpenes: add them with a Shantanu-approved product scheme, or leave
   Cl + monoterpene out?
3. For the tropospheric Cl module, should rates follow IUPAC (consistent with MCM) or JPL
   (consistent with SANDBOX)?
4. Should ClNO2 and Cl2 be user-set initial or held values only, or should TOMAS-JAX get a
   chloride component first?
5. Move SANDBOX to JPL Evaluation 20, or keep 19-5 as the shared reference?

---

## 9. Files

- `prototypes/mechanism/halogen_overlap.py`: SANDBOX vs MCM rate overlap, three steps (SANDBOX
  venv, a tomas-jax venv, then join). Outputs in `prototypes/mechanism/results/halogen_sandbox.json`,
  `halogen_mcm.json` and `halogen_overlap.md`. Run used
  `gas-phase-chemistry/SANDBOX/.venv/bin/python` and `aerosol-microphysics/acdc-jax/.venv/bin/python`
  (the worktree has no venv of its own).
- Scratch, not in the repo: `mechplan/halo_census.py` (census of the MCM file),
  `mechplan/halo_map.py` and `f0am_list.py` (source mapping), the pinned GEOS-Chem and F0AM files,
  and the text pages extracted from the local JPL 19-5 PDF.
