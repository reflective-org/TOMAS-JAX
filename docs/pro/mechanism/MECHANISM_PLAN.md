# Pro Simulator gas-phase mechanism: plan

Status: plan, written 2026-10-16 on branch `feat/pro-mechanism` (from `feat/aaar-tutorial`
5227ddd). Build after 2026-10-27. No model code, preset or tutorial file is changed, and nothing
is committed. This note pulls together three companion notes and two prototypes, all in this
worktree:

- [`mcm_coverage.md`](mcm_coverage.md): which SANDBOX species and reactions MCM has, the MCM
  export tool, sizes of candidate subsets, licence and citation.
- [`halogens.md`](halogens.md): chlorine, bromine and iodine sources, SANDBOX mapping, options.
- [`architecture.md`](architecture.md): registry, composer, rate sets, solver cost, c\*, API, UI.
- [`prototypes/mechanism/`](../../../prototypes/mechanism/README.md): the MCM inorganic + CH4 +
  sulfur subset cut from the limonene file, the composer, scaling and c\* scripts, and the
  MCM-vs-SANDBOX box comparison in
  [`sandbox_cmp/README.md`](../../../prototypes/mechanism/sandbox_cmp/README.md).

Labels: **[measured]** comes from a prototype run (laptop shared with other jobs, so timings are
upper bounds); **[MCM site]** is read off a public page at [mcm.york.ac.uk](https://mcm.york.ac.uk/MCM);
**[estimate]** is judgement and must be replaced by a measured count before it is used for
scheduling. No MCM mechanism file was downloaded and no paper was opened for this plan.

Why (Ali, this session): "Can we have the MCM version of the SANDBOX species in the chemistry
part? Having SO2 comprehensive chemistry and other species." and "I want more comprehensive
gas-phase mechanism. It seems like for now it is only limonene from MCM. I meant to bring more
MCM reactions for species that we have in sandbox not just bringing sandbox chemistry."

## 1. Summary

1. **The MCM version of the SANDBOX species is mostly in the model already.** The tutorial file
   `limonene_mcm.eqn` (725 species, 2135 reactions) contains all 48 of MCM v3.3.1's inorganic
   reactions (written as 45 lines), the methane chain and MCM's SO2 chemistry. Cut out on its
   own, that part is a 28-species, 67-reaction mechanism that parses and runs in
   `tomas_jax.gaschem` with no code change, with every atom imbalance explained by an MCM
   convention and sulfur conserved to 1e-15 **[measured]**. It covers 16 SANDBOX species
   explicitly, plus H2SO4 as MCM's lumped `SA`.
2. **What MCM cannot supply.** 12 SANDBOX halogen species (ClO, ClOOCl, ClONO2, Cl2, HOCl,
   OClO, Br, BrO, BrONO2, BrCl, HBr, HOBr) have no MCM chemistry; MCM's Cl chemistry is 41
   Cl + alkane reactions (no HCl product) and 32 Cl-releasing steps in chlorinated-VOC
   degradation, and it has no bromine or iodine atom chemistry. N2O5 and HNO4 photolysis, O2
   photolysis, the heterogeneous reactions and the stratospheric HOx-Ox terms in section 3.5
   are also not in MCM. These come from JPL 19-5, as in SANDBOX.
3. **"Comprehensive SO2" in MCM terms** is four pieces: OH + SO2 (KMT12) through HSO3; O(3P) +
   SO2; stabilised Criegee intermediate + SO2, which arrives with every alkene or terpene subset
   (10 reactions in the limonene file, 49 in the full MCM); and DMS oxidation, MCM's only SO2
   source (5 reactions) **[MCM site]**. SO2 + HO2 is not in MCM; JPL gives only an upper limit
   and no products, and it is below 1e-5 of SO2 loss in both test cases **[measured]**.
4. **More comprehensive means more MCM roots, from one MCM export per mechanism.** Proposed
   set: `mcm_background` (inorganic + CH4, ethane, propane, DMS; about 100-150 species
   **[estimate]**), then `mcm_bvoc` (+ isoprene, alpha- and beta-pinene, limonene; about
   1700-2200 species **[estimate]**), later `mcm_urban` (+ toluene, xylenes, small alkenes).
   The tutorial keeps `mcm_limonene` bit-identical.
5. **A stratospheric/SAI mechanism needs JPL for the inorganic core.** In a 220 K, 50 hPa box
   with the same TUV-x photolysis, MCM gives a 43-day SO2 lifetime and SANDBOX 33 days; almost
   all of it is one rate constant, OH + SO2, where JPL 19-5 is 38 % faster than MCM/IUPAC at
   that point **[measured]**. MCM also lacks N2O5 and HNO4 photolysis, which leaves NOx 2.6
   times too low there, and neither model has O2 photolysis, which at 20.6 km is a 28 ppb/day
   Ox source, ten times the net O3 tendency of either run (it barely moves SO2 or OH; section
   12) **[measured]**. Proposed: `strat_sai`, with JPL 19-5 owning every inorganic reactant
   set, MCM owning the organic chain, SANDBOX's halogen chemistry and γ uptake on TOMAS
   particle surface; and `sandbox_jpl`, SANDBOX's reaction set exactly, as the validation target.
6. **The sparse gas solver comes first.** Estimated gas-only time for a 24 h run at 900 s
   coupling: dense solver 44 s at 1352 species and 32 min at 5741; sparse solver 1.0 s and
   6.7 s **[estimate from measured parts]**. Anything past about 800 species is sparse-only.
7. **Plumbing.** A mechanism registry replaces the one hard-wired file (about ten places in the
   code), a composer joins MCM exports with JPL modules under six checked rules, c\* for new
   products comes from the same Nannoolal/UManSysProp method that reproduces Shantanu's table
   (median difference 1e-7 dex over 700 rows **[measured]**), and every run is keyed by a
   mechanism hash.
8. **Effort after 2026-10-27:** about 29-44 person-days for the first release (phases M0-M7),
   plus 6-10 for later additions, on top of the Pro plan's sparse solver **[estimate]**.
9. **Permission.** The MCM site states no licence; the University of York terms say content
   "may not be reproduced without prior permission". Written permission from the MCM team
   (Andrew Rickard) should come before a new export lands in a repository or the Pro tab shows
   MCM rate text.

## 2. Decisions for Ali

| # | Decision | Options | Recommendation |
|---|---|---|---|
| MP1 | Which MCM mechanisms, in what order | (a) `mcm_background` only; (b) then `mcm_bvoc`; (c) then `mcm_urban`; (d) full MCM | (a) first, as the test bed for the registry and the SANDBOX comparison; (b) once the sparse solver is in; (c) after the first release; (d) only as a stress test, never a user choice |
| MP2 | Stratospheric/SAI mechanism in the first Pro release | (a) yes; (b) later; (c) never, keep SANDBOX separate | (a): `strat_sai` labelled "research", released once the SANDBOX parity test and the SO2-lifetime check pass. If the schedule slips, it is the first item after the tropospheric set, not dropped |
| MP3 | Rate source for inorganic reactions both MCM and JPL have | (a) MCM/IUPAC everywhere; (b) JPL 19-5 everywhere; (c) per mechanism | (c): MCM/IUPAC in tropospheric mechanisms, JPL 19-5 for every inorganic reactant set in `strat_sai`. Never mix within one reactant set. One exception, exposed as a named option: `oh_so2_rate = "iupac" \| "jpl195"`, because it alone moves the stratospheric SO2 lifetime by about 30 % |
| MP4 | N2O5 and HNO4 photolysis (not in MCM) | (a) leave out; (b) add to every Pro mechanism; (c) only to `strat_sai` | (b), with J from TUV-x. It changed nothing in the tropospheric test case and fixes NOx at 20 km **[measured]**. O2 photolysis as a switch, on by default in `strat_sai` and in any run above the tropopause (28 ppb/day of Ox at 20.6 km, section 12 **[measured]**). Tutorial unchanged |
| MP5 | MCM's fixed aerosol sinks `<33> HNO3 = NA` and `<34> N2O5 = 2 NA` | keep, switch off, or replace | `<34>`: replace by γ uptake on TOMAS surface, as the tutorial does. `<33>`: off whenever TOMAS aerosol is coupled, pending Shantanu. `SO3 = SA` stays: it is the H2SO4 source for TOMAS |
| MP6 | Tropospheric chlorine | (a) none in the first release; (b) `cl_trop` module; (c) Br and I too | (a) for the first release, then (b): inorganic Cl, MCM Cl + alkanes with HCl added, and Xue et al. (2015) Cl + VOC for species in the export, with IUPAC rates to match MCM. ClNO2 and Cl2 as user inputs until TOMAS-JAX carries chloride. No Cl + monoterpene until Shantanu approves a product scheme. No Br or I |
| MP7 | SANDBOX's missing halogen reactions (BrO + HO2, HBr + OH, Cl + HO2, OH + HOCl, OH + ClONO2, BrO + BrO, HOBr + HCl uptake and others) | (a) parity with SANDBOX; (b) complete JPL set | Both, behind `halogens.sandbox_parity`: `true` only in `sandbox_jpl` (validation), `false` (complete) in `strat_sai`. Also add them to SANDBOX later, separately |
| MP8 | JPL version | (a) keep 19-5; (b) move to Evaluation 20 (JPL 25-1, 2025) | (a) for this build, so the SANDBOX comparison stays clean; check the Evaluation 20 entries for every reaction used and log changes; move SANDBOX and TOMAS-JAX together later |
| MP9 | Where multi-VOC files come from | (a) one MCM export per mechanism; (b) compose separate exports | (a). The composer is for joining MCM with JPL/heterogeneous modules, and for appending a delta to a file that must stay byte-identical |
| MP10 | c\* for products of new VOCs | (a) Nannoolal via UManSysProp offline; (b) Li et al. (2016); (c) Shantanu builds each table | (a), reviewed by Shantanu; (b) only as the documented fallback, and made to refuse halogenated organics. MSA from DMS stays non-condensable until Shantanu advises |
| MP11 | MCM permission | (a) ask the MCM team now; (b) rely on citation | (a): Ali writes to Andrew Rickard in the week of 2026-11-02, as part of decision D12 (licence) in the next-phase plan, covering the repository, the Pro web display and Pro exports |
| MP12 | Halogen source files | (a) transcribe parameters from primary sources; (b) adapt GEOS-Chem `fullchem.eqn` (MIT); (c) copy F0AM or MECCA (GPL) | (a), cross-checked against (b); never (c) while the TOMAS-JAX licence is open |

## 3. Target mechanism set

### 3.1 Rules

- **One source per reactant set.** Two sources describing the same reactants are never summed;
  the registry names the owner. Inorganic means carbon-free plus CO.
- **MCM names are canonical** (upper case: `HO2NO2` for HNO4, `SA` for H2SO4, `O` for O(3P),
  `CL`, `HCL`). Non-MCM modules carry an alias table; names that differ only in case are
  rejected, because KPP's Fortran output ignores case and the TOMAS-JAX parser does not.
- **One MCM export per mechanism**, with edits applied in code (content-keyed patches), never
  by editing the exported file.
- **What MCM lacks comes from JPL 19-5**, transcribed with citations, in separate modules.

### 3.2 Mechanisms and modules

| id | Contents | Species / reactions | Solver | Release |
|---|---|---|---|---|
| `mcm_limonene` | tutorial file + the tutorial N2O5 uptake patch | 726 / 2135 **[measured]** | dense (frozen) | tutorial, unchanged |
| `mcm_inorganic` | MCM inorganic + CH4 + SO2, the prototype cut | 28 / 67 **[measured]** | either | prototype now; test fixture |
| `mcm_background` | one export: roots CH4, C2H6, C3H8, DMS, inorganic ticked | about 100-150 / 300-450 **[estimate]**; the C1-C3 closure of the limonene file alone is 98 / 280 **[measured]** | either | first release |
| `mcm_bvoc` | one export: background roots + C5H8, APINENE, BPINENE, LIMONENE; Shantanu's limonene edits re-applied | about 1700-2200 / 5500-7000 **[estimate]**; isoprene alone 602 / 1926 ([Jenkin et al. 2015](https://doi.org/10.5194/acp-15-11433-2015)) | sparse only | first release |
| `mcm_urban` | `mcm_bvoc` roots + TOLUENE, BENZENE, xylenes, C2H4, C3H6 | about 2000-3000 / 6500-9500 **[estimate]** | sparse only | later |
| `mcm_full` | full MCM v3.3.1 | 5832 / 17224 **[MCM site]** | sparse only | stress test |
| `strat_sai` | `jpl195_core` (every inorganic reactant set, JPL 19-5) + MCM organic chain from `mcm_background` + `jpl_halogen` (complete) + `jpl_photolysis_ext` + γ uptake | about 120-170 **[estimate]** | either | first release, "research" |
| `sandbox_jpl` | SANDBOX's 81 reactions as written, `sandbox_parity = true` | 36 SANDBOX species | either | validation only |
| module `jpl_photolysis_ext` | N2O5 → NO2 + NO3; HO2NO2 → HO2 + NO2 and → OH + NO3; O2 → 2 O (switch) | 3-4 reactions | | added to every Pro MCM mechanism (MP4) |
| module `jpl_halogen` | SANDBOX Cl/Br gas reactions, the missing JPL reactions when `sandbox_parity = false`, 4 halogen uptake reactions | 12 halogen species + Cl, HCl | | in `strat_sai` and `sandbox_jpl` |
| module `cl_trop` | inorganic Cl, MCM Cl + alkanes with HCl, Xue (2015) Cl + VOC for exported species | not counted | | later (MP6) |

All [estimate] sizes are replaced by counts from the real export in phase M2 or M4.

### 3.3 Where each SANDBOX species comes from

| SANDBOX species | In the Pro mechanisms | Source |
|---|---|---|
| O3, NO, NO2, NO3, N2O5, HNO3, HONO, HNO4 (`HO2NO2`), OH, HO2, H2O2, O, O1D, CH4, SO2, SO3 (16) | explicit species | MCM inorganic block in tropospheric mechanisms; JPL 19-5 in `strat_sai` |
| H2SO4 | `SA`, a counter that feeds the TOMAS H2SO4 source | MCM `SO3 = SA` (first order in H2O); JPL [H2O]² form in `strat_sai` |
| HNO3aq | a counter added to particle nitrate, like `NA_HET` today | uptake reactions |
| O2, H2O | environment variables inside rate expressions | |
| CH3 | folded into CH3O2 by MCM | MCM CH4 chain |
| C2H6 | explicit | MCM ethane chain (`mcm_background`) |
| Cl, HCl | explicit | `jpl_halogen` or `cl_trop`; MCM `CH4 + CL` patched to give HCL |
| ClO, ClOOCl, ClONO2, Cl2, HOCl, OClO, Br, BrO, BrONO2, BrCl, HBr, HOBr (12) | explicit, only with a halogen module | JPL 19-5 via SANDBOX's table |

### 3.4 Sulfur

| Process | Reactions | In |
|---|---|---|
| OH + SO2 + M → HSO3; HSO3 + O2 → HO2 + SO3 | MCM `<31>` KMT12, `<32>` | every MCM mechanism |
| O(3P) + SO2 + M → SO3 | MCM `<30>`, 0.3 % of stratospheric SO2 loss **[measured]**. JPL 19-5 (I3, fall-off) is 1.7 times MCM at 220 K, 50 hPa and 0.37 times at 298 K, 1 atm; SANDBOX lacks it | every MCM mechanism; JPL form in `strat_sai` |
| SO3 + H2O → H2SO4 | MCM `<35>` 1.2e-15 [H2O]; JPL 8.5e-41 exp(6540/T) [H2O]² | MCM form in tropospheric, JPL in `strat_sai` |
| Criegee + SO2 | 10 in the limonene file, 49 in full MCM **[MCM site]**, every one at 7.0e-14 cm³ s⁻¹; direct measurements for CH2OO are about 500 times faster (section 12) | arrive with each alkene or terpene root; rate is a named option, pending Shantanu |
| DMS → SO2, MSA | DMS + OH (abstraction and addition, KMT18), DMS + NO3, then the CH3SO and MSIA chain | `mcm_background` and up |
| SO2 + HO2 | JPL upper limit only, no products | left out; at most a SANDBOX-parity term |

Out of scope for gas-phase chemistry: aqueous and cloud oxidation of SO2, and SO2 uptake on
particles. `tomas_jax/physics/so2_chemistry.py` (prescribed-OH SO2 + OH for runs without gas
chemistry) stays as it is.

The OH + SO2 rate. Both forms are Troe fall-off expressions. Write T for temperature [K], [M]
for air number density [cm⁻³], k₀ and k∞ for the low- and high-pressure limits, and
x = k₀[M]/k∞.

- MCM KMT12 (IUPAC): k₀ = 2.5e-31 (T/300)^−2.6, k∞ = 2.0e-12, F_c = 0.53,
  k = k₀[M] / (1 + x) · F_c^{1 / (1 + (log₁₀ x / N)²)}, with N = 0.75 − 1.27 log₁₀ F_c.
- JPL 19-5: k₀ = 2.9e-31 (T/298)^−4.1, k∞ = 1.7e-12 (T/298)^0.2,
  k = k₀[M] / (1 + x) · 0.6^{1 / (1 + (log₁₀ x)²)}.

At 298 K, 1 atm the two agree within 6 %; at 220 K, 50 hPa JPL is 1.38 times MCM, set mainly
by the steeper temperature exponent of k₀ **[measured]**. The SO2 lifetime against OH is
τ = 1 / (k ⟨[OH]⟩), with ⟨[OH]⟩ the 24 h mean OH number density.

### 3.5 Halogen options

- **Tropospheric (later, `cl_trop`).** Inorganic Cl gas chemistry (Cl + O3, ClO + NO, ClO + HO2,
  OH + ClO, OH + HCl, Cl + HO2, ClO + NO2 + M ⇌ ClONO2, OH + HOCl, photolysis of Cl2, ClNO2,
  HOCl and ClONO2); MCM's 41 Cl + alkane reactions with HCl added; Xue et al. (2015) Cl + VOC
  reactions for species in the export ([GMD 8, 3151](https://doi.org/10.5194/gmd-8-3151-2015),
  205 reactions, list in the CC BY 3.0 supplement only). Optional ClNO2 yield φ on N2O5
  uptake: N2O5 → φ ClNO2 + (2 − φ) NO3⁻(particle).
- **Stratospheric (`strat_sai`, `sandbox_jpl`).** SANDBOX's halogen gas reactions (R1-R7,
  R48-R60), its uptake reactions R8-R11 and R72 on TOMAS surface area, and with
  `sandbox_parity = false` the reactions SANDBOX lacks (list in [`halogens.md`](halogens.md)
  section 6.2).
- **Stratospheric HOx-Ox and cross terms that neither MCM nor SANDBOX has** (add to
  `jpl195_core` / `jpl_halogen` for `strat_sai`; checked against the
  [MCM inorganic list](https://mcm.york.ac.uk/MCM/inorganic?category=Thermal%20gas-phase) and
  SANDBOX `reactions.py`): O + OH → H + O2, O + HO2 → OH + O2, H + O2 + M → HO2,
  H + O3 → OH + O2, OH + OH → H2O + O and (+M) → H2O2, O1D + H2, and, if N2O is carried,
  O1D + N2O → 2 NO / N2 + O2 (the stratospheric NOy source). O1D + CH4 is in SANDBOX but not
  in MCM. At 20.6 km O + OH and O + HO2 are about 0.2 % and 1 % of OH and HO2 loss
  (**[estimate]** from the prototype's noon O, OH and HO2), so they matter mainly above about
  30 km. Halogen + MCM organic cross reactions (Cl + HCHO, Cl + CH3OOH, Cl + CH3OH, Cl + H2,
  Br + HCHO → HBr, ClO + CH3O2) are in neither source file; Br + HCHO is a main HBr source.
  Cl + CH4 and Cl + C2H6 exist in both (MCM without HCl): JPL owns them in `strat_sai`.
- **Not planned:** tropospheric bromine and iodine. If needed later, transcribe from GEOS-Chem
  `fullchem.eqn` (MIT) with JPL/IUPAC checks.

## 4. How the mechanism is obtained

### 4.1 MCM export

From the public tool at [mcm.york.ac.uk/MCM/export](https://mcm.york.ac.uk/MCM/export)
**[MCM site]**:

1. For each root VOC, open its species page (for example
   [C5H8](https://mcm.york.ac.uk/MCM/species/C5H8)) and press "Add to marklist". Roots per
   mechanism are in section 3.2; include CH4 in every one.
2. Export with "Include inorganic reactions?" and "Include generic rate coefficients?" ticked,
   format KPP. Also download the "Species List (Tab-Separated Values)", which carries SMILES,
   InChI and molar mass for the c\* build.
3. The exporter follows every organic product from all roots, adds the inorganic block once and
   writes one RO2 sum (from the website source,
   [wacl-york/mcm-web](https://github.com/wacl-york/mcm-web)). Store the file with its date,
   the marklist and a SHA-256 in the registry entry.
4. Check the York KPP layout (`#DEFVAR`, `#EQUATIONS`, `#INCLUDE atoms`, the citation block)
   against `kpp_parser.parse_eqn`, which raises on unknown content.
5. Re-apply edits in code: Shantanu's LIMONENE + OH branches scaled to kOH = 1.47e-10 cm³ s⁻¹
   and the D9 tracer (for mechanisms that include limonene, pending question S3); the N2O5
   uptake rename; the HCl product on Cl + alkane. All patches match reactions by reactants and
   products, never by label, because each export renumbers labels.
6. Diff the export's limonene and inorganic reactions against `limonene_mcm.eqn`. Every
   difference other than the known edits is a website update since 2022-04-27 or a toolchain
   difference; log each one for Shantanu.
7. Download the KPP constants file (`/MCM/export/kpp_constants`, updated 2023-12-19
   **[MCM site]**) once, only to check `mcm_rates.py` against it (section 7, V4).

How the current file was made (inference, to confirm with Shantanu): the only species consumed
but never produced are LIMONENE, CH4, CL, D9 and SO2, which points to roots LIMONENE and CH4
with inorganic reactions ticked, passed through Shantanu's toolchain ("SCENTS 04272022").

### 4.2 JPL and other non-MCM modules

- Transcribe rate parameters from JPL 19-5 ([JPL data evaluation](https://jpldataeval.jpl.nasa.gov/))
  as SANDBOX's `stratchem-jax/reactions.py` already does, with the table id per reaction.
- Generate `sandbox_jpl` and `jpl_halogen` from SANDBOX's reaction table by script, so the two
  codes cannot drift silently.
- Cross-check against GEOS-Chem `fullchem.eqn` (MIT) and F0AM; do not copy F0AM or MECCA files
  (GPL-3.0).
- For `cl_trop`, transcribe the Xue (2015) supplement tables with a row-by-row check.

### 4.3 Terms and citation

- **Terms.** The MCM site states no licence and its website code has no licence file. Every
  page links the [University of York legal statements](https://www.york.ac.uk/about/legal-statements/):
  content "may not be reproduced without prior permission" and downloads are "for personal,
  non-commercial reference only". TOMAS-JAX already ships an MCM subset. Ask Andrew Rickard
  (MCM contact) for written permission covering the repository copies, the Pro web display of
  reactions and rates, and Pro exports (MP11). JPL: transcribe parameters with citation, do not
  redistribute the PDF. IUPAC website: private use only; cite the ACP summary papers. Xue
  (2015): CC BY 3.0. None of this is legal advice.
- **Citation**, shown in the Pro mechanism card, in every export and in the docs. The MCM's own
  wording ([cite page](https://mcm.york.ac.uk/MCM/about/cite)): "The chemical mechanistic
  information was taken from the Master Chemical Mechanism, MCM v3.3.1 (reference), via
  website: http://mcm.york.ac.uk/". Then, per scheme used:
  - non-aromatic: [Jenkin et al. 1997](https://doi.org/10.1016/S1352-2310(96)00105-7),
    [Saunders et al. 2003](https://doi.org/10.5194/acp-3-161-2003);
  - aromatic: [Jenkin et al. 2003](https://doi.org/10.5194/acp-3-181-2003),
    [Bloss et al. 2005](https://doi.org/10.5194/acp-5-641-2005);
  - isoprene: [Jenkin et al. 2015](https://doi.org/10.5194/acp-15-11433-2015);
  - JPL 19-5 ([Burkholder et al. 2020](https://jpldataeval.jpl.nasa.gov/)) for `jpl195_core`,
    `jpl_halogen`, `jpl_photolysis_ext` and the uptake coefficients;
  - Xue et al. 2015 if `cl_trop` is on; SANDBOX as the source of the halogen table.

## 5. How it plugs into TOMAS-JAX and the Pro Simulator

### 5.1 Registry

Today one file is hard-wired in about ten places (`gaschem/config.py`, `box/coupling.py`,
`box/driver.py` counters and N2O5 label, `box/result.py` reported species, `RateModel`'s J
allowlist, `photolysis.SOURCES`, `volatility.py`, `code_version`, `solver.py`; table in
[`architecture.md`](architecture.md) section 2). A registry entry replaces them. It names the
modules, the content-keyed patches, the owner of each cross-source reactant set, the counters
and external (uptake) reactions, the c\* table, the J map, the reported species, the required
solver and a status (`validated`, `prototype`, `planned`). Format sketch:
[`registry_example.toml`](../../../prototypes/mechanism/registry_example.toml). Resolution gives
one resolved mechanism (species, atoms, namespaced reactions, RO2 list, J sources, counters,
external reactions, c\* map) that the driver, the cache and the chemistry tab all read.

### 5.2 Composition

Write a reaction as r = (R, P, k), with R the reactant multiset, P the products with
coefficients and k the rate expression; B = {M, O2, N2, H2O, hv} the bath species MCM writes
into rates or omits. Reaction key K(r) = (sorted(R \ B), photolysis flag, sorted(P \ B));
reactant-set key S(r) = sorted(R \ B) plus "hv" for photolysis. The composer applies six rules
(each shown to fire on a deliberate violation in `compose_demo.py` **[measured]**):

1. one name, one atomic composition across modules;
2. same K and same normalised rate text: duplicate, first copy kept, logged;
3. same K, different rate: error;
4. same S from two sources: error unless the registry names the owner;
5. RO2 is the union of the modules' lists, and every reaction with a rate proportional to RO2
   has its reactant in it;
6. labels are namespaced `module:label`.

Evidence: inorganic + limonene composes back to exactly the limonene file (68 duplicates
removed, RO2 lists equal, none of the 280 RO2-dependent reactions outside the 146-member sum)
**[measured]**. The composer also reports orphans: species consumed but never produced and
without an initial value (for example HSO3 if JPL owned OH + SO2 as a lumped channel).

### 5.3 Rate functions

- MCM: all 12 generic and 18 complex coefficients are in `mcm_rates.py` **[MCM site]**, so a
  full MCM export should translate without new names.
- JPL: a `jpl_rates.py` module with KPP-style names, evaluated like `constants_mcm`: the
  termolecular form above (JPL uses a 298 K reference and F_c = 0.6; IUPAC/MCM use 300 K and a
  per-reaction F_c); the chemical-activation form k = k_f + k_int (1 − k_f / k∞), with k_f the
  termolecular rate and k_int the bimolecular low-pressure limit; and equilibrium reverse rates
  k_r = k_f / K_eq, with K_eq the equilibrium constant [cm³]. All are thermal, so the
  `RateModel` split k = a · RO2^(0 or 1) + c · J still holds.

### 5.4 Photolysis

- `MCM_J_PARAMS` holds all 34 MCM J numbers, but `photolysis.SOURCES` maps only the 28 the
  limonene file uses. Six more are needed for `mcm_background` and up: J_CH3CHO, J_C2H5CHO and
  J_C2H5NO3 from TUV-x reactions the port already computes, J_MVK_CO from the existing MVK
  split, J_C3H7CHO_C2H4 and J_TC4H9NO3 from MCM's parameterisation.
- New `J_*` names for `jpl_photolysis_ext` and `jpl_halogen` (N2O5, HNO4 two channels, O2,
  Cl2, ClONO2 two channels, ClOOCl, HOCl, OClO, ClNO2, BrO, BrONO2 two channels, BrCl, HOBr)
  map to TUV-x keys; 20 of the 21 checked are computed by the tuvx-jax port **[measured]**.
- `strat_sai` must use TUV-x J at the box altitude. MCM's clear-sky J(χ) = l · cos(χ)^m ·
  exp(−n · sec χ), with χ the solar zenith angle and l, m, n fitted constants, is a surface fit
  and is refused above the boundary layer. This also applies to J_C3H7CHO_C2H4 and
  J_TC4H9NO3 above: in `strat_sai` they need a TUV-x analogue (for example the
  i-C3H7ONO2 cross section for J_TC4H9NO3), not MCM's l, m, n.
- TUV-x J at altitude must use the overhead O3 column and temperature of the case, not the
  US Standard Atmosphere profile the prototype used, and for SAI runs the sulfate optical
  depth from TOMAS (scattering changes J). Both models in the prototype share the same
  simplification, so the comparison is unaffected.

### 5.5 Uptake reactions and counters

- External reactions take k from the aerosol: k_het = ¼ γ v̄ A, with γ the uptake coefficient,
  v̄ = (8 R T / (π M_w))^½ the mean molecular speed (R gas constant, M_w molar mass of the gas
  [kg mol⁻¹]) and A the particle surface area per volume from TOMAS. This is the path the
  tutorial already uses for N2O5.
- ClONO2 + HCl and HOCl + HCl are first order in the gas taken up but also consume HCl. Add a
  per-reaction list of rate reactants separate from consumed species, and limit k_het when HCl
  runs out. γ for ClONO2 and HOCl depends on the H2SO4 weight percent, T and H2O (SANDBOX
  `hetgammas_jpl00`, `aerosol.h2so4wp_at`), computed from TOMAS composition. SANDBOX's γ
  routine divides by zero when HCl = 0; guard it in the port.
- Counters (`SA`, `NA`, `NA_HET`, HNO3aq) are declared per mechanism with the TOMAS species
  they feed (sulfate, nitrate).

### 5.6 c\* for new products

The tutorial table is Shantanu's (Nannoolal boiling point and vapour pressure through
UManSysProp, as PyCHAM calls it). The prototype `cstar_nannoolal.py` applies the same method:
C\* = 10⁶ · MW · p_sat / (R · 298.15 K), with C\* in µg m⁻³, MW the molar mass [g mol⁻¹] and
p_sat the pure-component saturation vapour pressure [Pa] from the SMILES. It reproduces all 700
rows of Shantanu's table (median |Δ log₁₀ C\*| = 1e-7, maximum 0.028, the largest all
hydroperoxides) and runs on all 5834 species of PyCHAM's MCM SMILES file in 13 s with no
failures **[measured]**. Plan: an offline build step writes one table per mechanism with a
provenance file; Shantanu reviews it; Li et al. (2016) only for species without SMILES, and it
raises for Cl or Br. Volatility classes do not grow with the mechanism: every closed-shell
organic still lumps into the 15 TOMAS classes.

### 5.7 API and cache

- Scenario: `[chemistry] mechanism = "mcm_limonene"` (default, the only value the tutorial
  accepts), `modules = [...]` (toggles the entry allows) and `options` (`oh_so2_rate`,
  `halogens.sandbox_parity`, `o2_photolysis`). Gas inputs and held species are validated
  against the resolved species list.
- Mechanism hash: SHA-256 of the canonical JSON of the resolved mechanism (including patches,
  options and c\* contents), part of the run id. `code_version()` stops hashing `.eqn` and
  `.csv` files; otherwise adding any mechanism re-keys every cached tutorial result.
- Endpoints, extending the next-phase plan's `/api/v2/pro/mechanism`:
  `GET /api/v2/pro/mechanisms` (id, title, sizes, solver, status, run-time class, c\* coverage,
  citation), `GET /api/v2/pro/mechanisms/{id}` (species with formula, molar mass, c\*, module),
  `GET /api/v2/pro/mechanisms/{id}/reactions?species=...` (paged).
- Per-mechanism limits on run length and the finest coupling interval, until each mechanism is
  timed on Render.

### 5.8 UI

- **Mechanism picker** in Pro Build, "Gases and chemistry": one card per entry with species and
  reaction counts, root VOCs, solver, expected run time, c\* source and coverage, validation
  status, citation, and the allowed modules and options as toggles. Disabled choices say why.
  Switching mechanism keeps the gas inputs that still exist and lists the ones dropped.
- **Gas table**: search over the resolved species (thousands of names), root VOCs as quick
  picks.
- **Chemistry tab**: each reaction shows its module and source (MCM label or JPL table id);
  carbon flows per root VOC; a Cl/Br family budget when a halogen module is on; an SO2 budget
  panel (OH, O(3P), Criegee and DMS terms). For large mechanisms, serve per-species flux
  queries from the worker instead of one payload (about 6 MB at 17 000 reactions
  **[estimate]**).

## 6. Performance

Gas time of a run: T_gas = N_s · c_step, with N_s the solver steps and c_step the cost of one
step. Dense LU grows as n³ and the autodiff Jacobian as about n² (n species); the sparse LU
grows with the nonzeros after fill, which stay at 1.46-1.57 times the Jacobian's nonzeros up
to 5741 species **[measured]**. Estimated 24 h gas-only times on the laptop, one process
(`estimate.py`, from measured parts; N_s from Level 3 and assumed independent of size, which
must be measured on real exports) **[estimate]**:

| Species | Case | Dense, 900 s | Sparse, 900 s | Sparse, 120 s |
|---|---|---|---|---|
| 28-170 | `mcm_inorganic`, `mcm_background`, `strat_sai` (98 measured as the C1-C3 closure) | 0.2-2 s | 0.1-0.3 s | 0.7-1.5 s |
| 725 | `mcm_limonene` | 12 s | 0.6 s | 3.1 s |
| 1352 | | 44 s | 1.0 s | 5.3 s |
| 1700-2200 | `mcm_bvoc` (interpolated) | 1.5-3 min | 1.2-1.7 s | 6-9 s |
| 2606 | `mcm_urban` range | 3.8 min | 1.9 s | 9.6 s |
| 5741 | full-MCM size | 32 min | 6.7 s | 34 s |

- On a Render core multiply by 2.1-2.9 (Pro plan).
- Peak memory with the dense path: 0.52 GB at 725 species, 1.6 GB at 2606, 5.9 GB at 5741
  **[measured]**, so a dense multi-VOC run does not fit Render's 2 GB worker. Sparse arrays
  stay under 20 MB at 5741 **[estimate]**.
- Compile: the levelised sparse LU compiles in 2.3 s (factor) + 1.5 s (solve) at 5741 species;
  the symbolic analysis takes 8 s there and is cached on disk by mechanism hash
  **[measured]**. Warm only `mcm_limonene` and the Pro default at worker start.
- Rule: dense allowed up to 800 species; above that the API refuses a dense request.

## 7. Validation

| # | Test | Gate | Evidence so far |
|---|---|---|---|
| V1 | Parse and resolve | every reacting species declared; every J name resolves to a source | `compose_demo.py` J check |
| V2 | Atom balance per reaction | residual Δ_e = Σ_products ν a_e − Σ_reactants ν a_e (ν stoichiometric coefficient, a_e atoms of element e) is zero, or explained by omitted O2, H2O or CO2, an IGNORE sink, or an RO2 permutation; N, S, Cl and Br exact; anything else in a reviewed exception list | inorganic subset: 26 exact, 41 explained, 0 unexplained; limonene file: 533 exact, 1297 explained, 280 RO2, 25 exceptions **[measured]** |
| V3 | Element conservation in runs | total S (SO2 + HSO3 + SO3 + SA + DMS chain); N with counters; Cly and Bry with uptake and emissions off | S to 1e-15 over multi-day runs **[measured]** |
| V4 | MCM reference rates | every K and every reaction k at three (T, p, RH) points against MCM's `constants_mcm.f90` compiled, or F0AM | 33 K values already checked against F0AM |
| V5 | JPL rates | every `jpl_rates` function against SANDBOX `coeff_fn` to 1e-12 relative | to add |
| V6 | Sub-mechanism equivalence | a mechanism and a superset with the extra species at zero give the same run | inorganic vs limonene file: O3 1.6e-6, SO2 1.2e-6, OH 3.2e-3 maximum relative difference (solver noise) **[measured]** |
| V7 | Tutorial invariance | the three tutorial presets bit-identical to tag `aaar-stable-2026-10-16c` (or the newest stable tag) | to add |
| V8a | SANDBOX parity | `sandbox_jpl` against stratchem-jax with the same T, p, H2O, surface area, J and start, 10 days; tolerance from a self-convergence run | to add |
| V8b | SANDBOX comparison, shared species | `mcm_background` and `strat_sai` against SANDBOX in the two prototype cases; every difference traced to a named reaction or rate | done for the prototype subset: SO2 lifetime 43 vs 33 days at 220 K (OH + SO2 rate), 1.2 vs 2.0 days at 298 K (HNO4/N2O5 decomposition, CH4 chain) **[measured]**; strat MCM and SANDBOX cases re-run independently in review, same numbers. Still to add: a case with sulfate surface area and N2O5 hydrolysis in both models, and the tropospheric case with CO and HCHO |
| V9 | PyCHAM, same mechanism | extend `benchmarks/pycham` (`write_eqn(src=...)`) to `mcm_background` gas-only and `mcm_bvoc`; no worse than G2 (median \|NMB\| 1.7e-3 at 300 s) | harness exists |
| V10 | AtChem2, same mechanism (one-off) | run the FACSIMILE export of `mcm_background` and `mcm_bvoc` gas-only in [AtChem2](https://github.com/AtChem/AtChem2) ([Sommariva et al. 2020](https://doi.org/10.5194/gmd-13-169-2020)), an MCM box model with its own rate code and CVODE solver; same targets as V9. Worth it because it is a second independent reading of the MCM rates and runs the full export quickly; not a CI gate (Fortran build) | to add |
| V11 | Sparse vs dense | same linear solve to 1e-9; coupled 24 h end values within 0.1 % | 1.3e-11 at 725, 1.2e-9 at 4487 **[measured]** |
| V12 | c\* pipeline | reproduces Shantanu's table: median below 1e-4 dex, maximum below 0.05 dex | 1e-7 / 0.028 **[measured]** |

## 8. Phases after 2026-10-27

Effort in focused person-days, all **[estimate]**. Earliest start 2026-11-02. The sparse solver
is the Pro plan's Phase 1 (9-14 days, target 2026-11-20) and is not counted here.

| Phase | Work | Effort | Depends on |
|---|---|---|---|
| M0 | Decisions MP1-MP12; questions to Shantanu; permission request to the MCM team | 0.5-1 | none |
| M1 | Registry, resolver, composer from the prototype, content-keyed patches, per-mechanism counters, reported species and J map, `code_version` split, mechanism hash; tutorial bit-identical (V7) | 4-6 | M0 |
| M2 | `mcm_background`: one MCM export, York KPP layout in the parser, six J names, `jpl_photolysis_ext`, HCl patch on Cl + alkanes, aerosol sinks per MP5; V1-V4, V6, V8b rerun | 4-6 | M1, MCM permission |
| M3 | c\* build step and tables for M2 and M4; Li et al. (2016) halogen refusal; MSA rule | 2-3 | M1 |
| M4 | `mcm_bvoc`: export, limonene edits re-applied, measured sizes, LU fill and timings (laptop and Render), V9; V10 optional (+1-2) | 3-5 | M2, M3, sparse solver |
| M5 | JPL infrastructure: `jpl_rates.py` (V5), rate reactants vs consumed species, γ from state, halogen J names, altitude J | 6-9 | M1 |
| M6 | `sandbox_jpl` generated from SANDBOX and V8a; `strat_sai` (JPL core, MCM organics, complete halogens, uptake on TOMAS surface, `oh_so2_rate` option); Evaluation 20 check | 5-8 | M5 |
| M7 | API (mechanism id, hash, endpoints, limits), Pro picker, chemistry-tab additions | 4-6 | M1, Pro plan Phases 2-4 |
| Later | `cl_trop` (inorganic Cl, Xue transcription) 4-6; `mcm_urban` export and timings 2-4 | 6-10 | M5, M4 |

First release (M0-M7): about 29-44 person-days, plus 1-2 for AtChem2. M1-M3 and M5 do not need
the sparse solver and can start in November; M4 follows the solver; M7 rides with the Pro
front-end phases. With one person on both plans, the Pro plan alone runs to about
January-February 2027 (Pro plan section 9), so this work moves the Pro beta to about March-April
2027 unless a second person takes M5-M6.

## 9. Questions for Shantanu

1. Which roots and which tool produced `limonene_mcm.eqn` ("SCENTS 04272022")? Was the CH4 root
   deliberate, and are there edits besides the LIMONENE + OH scaling and the D9 tracer?
2. Which VOCs matter for his students: biogenic only, or also aromatics and alkanes?
3. In multi-VOC mechanisms, keep his chamber edit (LIMONENE + OH scaled to 1.47e-10) and D9, or
   use stock MCM limonene?
4. Is Nannoolal (boiling point and vapour pressure) via UManSysProp the method he wants for other
   VOCs? Is the +0.025 dex offset on hydroperoxides something PyCHAM adds on purpose?
5. MSA from DMS: condensable with an estimated c\*, non-volatile, or gas only?
6. Keep MCM's `HNO3 = NA` (6e-6 s⁻¹) in Pro mechanisms, or replace HNO3 loss by uptake or
   deposition as N2O5 already is?
7. Cl + limonene and other monoterpenes: is there a product scheme he would accept, or leave them
   out?
8. Is adding HCl as the product of MCM's Cl + alkane reactions acceptable?
9. For tropospheric runs, is IUPAC (via MCM) the reference he wants for OH + SO2, with JPL as an
   option?
10. Stabilised Criegee intermediates: keep MCM v3.3.1's uniform sCI + SO2 = 7.0e-14 cm³ s⁻¹
   and its water-only sCI losses, or offer a literature option (faster sCI + SO2 with
   unimolecular decay)? This decides how much of "comprehensive SO2" the BVOC mechanisms add.

## 10. Assumptions to log

- A-M1. The prototype subset is cut from the limonene file, not a fresh MCM export; its inorganic
  block is assumed equal to MCM v3.3.1's and is checked in M2 step 6.
- A-M2. Per-VOC sizes are estimates until an export is counted (M2, M4).
- A-M3. Solver step counts N_s do not grow with mechanism size; to be measured on `mcm_bvoc`.
- A-M4. Tropospheric mechanisms use MCM/IUPAC for every inorganic reactant set, `strat_sai` uses
  JPL 19-5 for all of them, never mixed within a reactant set; `oh_so2_rate` is the one
  exposed exception.
- A-M5. N2O5 and HNO4 photolysis are added to every Pro MCM mechanism (no effect in the
  tropospheric test case, large at 20 km).
- A-M6. SANDBOX's ClONO2 thermal decomposition (R5, not in JPL) is kept for parity; its `opt = 1`
  O3 photolysis shortcut is not ported; its dropped O3 → O(3P) photolysis channel is included
  in `strat_sai` (it changes the stratospheric O3 tendency from −0.9 to −2.7 ppb/day in the
  prototype **[measured]**).
- A-M7. Prototype comparison conditions as in `sandbox_cmp/README.md`: NOx held at 1 ppb in the
  tropospheric case, SANDBOX's 60 hPa preset used at 50 hPa and 220 K, both runs from 00 UTC,
  no aerosol, no halogens, SANDBOX `opt = 0`.

## 11. Sources

- MCM v3.3.1: [home](https://mcm.york.ac.uk/MCM), [export](https://mcm.york.ac.uk/MCM/export),
  [browse](https://mcm.york.ac.uk/MCM/browse), [SO2](https://mcm.york.ac.uk/MCM/species/SO2),
  [CL](https://mcm.york.ac.uk/MCM/species/CL), [DMS](https://mcm.york.ac.uk/MCM/species/DMS),
  [inorganic reactions](https://mcm.york.ac.uk/MCM/inorganic?category=Thermal%20gas-phase),
  [photolysis](https://mcm.york.ac.uk/MCM/rates/photolysis),
  [generic](https://mcm.york.ac.uk/MCM/rates/generic) and
  [complex](https://mcm.york.ac.uk/MCM/rates/complex) rates,
  [cite](https://mcm.york.ac.uk/MCM/about/cite), [archive](https://mcm.york.ac.uk/MCM/about/archive),
  website source [wacl-york/mcm-web](https://github.com/wacl-york/mcm-web).
- [University of York legal statements](https://www.york.ac.uk/about/legal-statements/).
- Protocol papers: [Jenkin et al. 1997](https://doi.org/10.1016/S1352-2310(96)00105-7),
  [Saunders et al. 2003](https://doi.org/10.5194/acp-3-161-2003),
  [Jenkin et al. 2003](https://doi.org/10.5194/acp-3-181-2003),
  [Bloss et al. 2005](https://doi.org/10.5194/acp-5-641-2005),
  [Jenkin et al. 2015](https://doi.org/10.5194/acp-15-11433-2015).
- Halogens: [Xue et al. 2015](https://doi.org/10.5194/gmd-8-3151-2015),
  [Riedel et al. 2014](https://acp.copernicus.org/articles/14/3789/2014/),
  [GEOS-Chem fullchem.eqn](https://github.com/geoschem/geos-chem/blob/4d4b2905d368d845d98f919b13707b6d47e59d89/KPP/fullchem/fullchem.eqn),
  [F0AM](https://github.com/AirChem/F0AM).
- JPL 19-5 and Evaluation 20: [jpldataeval.jpl.nasa.gov](https://jpldataeval.jpl.nasa.gov/),
  [JPL data evaluation project](https://science.jpl.nasa.gov/projects/data-evaluation).
- c\*: [Nannoolal et al. 2008](https://doi.org/10.1016/j.fluid.2008.04.020),
  [UManSysProp](https://github.com/loftytopping/UManSysProp_public),
  [PyCHAM](https://github.com/simonom/PyCHAM),
  [Li, Pöschl and Shiraiwa 2016](https://acp.copernicus.org/articles/16/3327/2016/).
- Reference box models: [AtChem2](https://github.com/AtChem/AtChem2),
  [Sommariva et al. 2020](https://doi.org/10.5194/gmd-13-169-2020); PyCHAM (above).
- Local: SANDBOX `stratchem-jax/REACTIONS.md` and `reactions.py`; the Pro plan
  (`docs/aaar/pro/PRO_TAB_PLAN.md`), the next-phase plan and `performance_cloud.md`
  (`docs/aaar/next/`) in the `pro-wt` worktree.

## 12. Review notes (2026-10-16, skeptical chemistry review)

Scope: the MCM coverage claims, the halogen claims, the MCM-vs-SANDBOX prototype, and whether
MCM rate expressions and photolysis hold at stratospheric T and p. Checked against
`limonene_mcm.eqn`, SANDBOX `stratchem-jax/reactions.py`, the local JPL 19-5 PDF that SANDBOX
cites, and public MCM pages ([inorganic](https://mcm.york.ac.uk/MCM/inorganic?category=Thermal%20gas-phase),
[SO2](https://mcm.york.ac.uk/MCM/species/SO2), [CL](https://mcm.york.ac.uk/MCM/species/CL),
[HCL](https://mcm.york.ac.uk/MCM/species/HCL), [photolysis](https://mcm.york.ac.uk/MCM/rates/photolysis),
[home](https://mcm.york.ac.uk/MCM)). Review scripts and outputs are in
`prototypes/mechanism/review/` (`rerun_strat.py`, `rerun_sb.py`, `jo2_table.py`). No MCM file was
downloaded and no paper was opened.

### 12.1 Confirmed

- MCM v3.3.1 inorganic set: 37 thermal reactions on the MCM page, written as 34 lines in the
  `.eqn` (three rates are split into two terms on the site), plus 3 gas-particle and 8
  photolysis: 48 in 45 lines, all present in `limonene_mcm.eqn` with the same rates.
- Sulfur: O + SO2, OH + SO2 (KMT12), HSO3 + O2, SO3 + H2O; 49 sCI + SO2 in full MCM (10 in the
  limonene file); 5 SO2-forming steps, all in the DMS chain (CH3SOO, CH3SO + O3, CH3SO + NO2,
  CH3SO2, MSIA + OH). No SO2 + HO2.
- Cl: 41 Cl + alkane reactions (CH4 to C12 and cyclohexane), none with HCl as a product; 32
  Cl-releasing steps from chlorinated VOCs. `<2676> CH4 + CL` is the only Cl reaction in the
  limonene file. The Xue et al. (2015) module is 205 reactions including inorganic Cl
  ([GMD article page](https://gmd.copernicus.org/articles/8/3151/2015/), CC BY 3.0 for the
  article; the page states no separate licence for the supplement).
- MCM size 17 224 reactions, 5832 species, CH4 + 142 NMVOCs; 34 J numbers.
- JPL 19-5 termolecular convention: k₀(T) = k₀²⁹⁸ (T/298)^−n and k∞(T) = k∞²⁹⁸ (T/298)^−m
  (Table 2-1 column heads in the local PDF), so the 298 K reference used in SANDBOX and in
  section 5.3 is right (older JPL editions used 300 K). OH + SO2 (I4): k₀²⁹⁸ = 2.9e-31, n = 4.1,
  k∞²⁹⁸ = 1.7e-12, m = −0.2, as in SANDBOX. Recomputed: JPL/MCM = 1.06 at 298 K, 1 atm and 1.38
  at 220 K, 50 hPa; it is 1.00 at 220 K, 1 atm and 1.15 at 298 K, 50 hPa, so the gap needs
  both low T and the fall-off regime.
- Prototype re-run (strat case, bounded, about 5 s for MCM and 1 s for SANDBOX): MCM matched
  OH noon 2.29e6, N2O5 642 ppt, SO2 lifetime 43.0 days; MCM + J(N2O5, HNO4) NOx 1535 ppt, 44.7
  days; SANDBOX opt 0 OH noon 2.32e6, N2O5 81 ppt, k(OH + SO2) 4.88e-13, 32.9 days. All match
  the README tables.

### 12.2 Corrected in this plan or its notes

1. **O2 photolysis is not small at 20 km.** TUV-x port, 20.6 km, 30° N, 21 June: J(O2) =
   4.0e-12 s⁻¹ at noon, 7.8e-13 s⁻¹ as a 24 h mean, so P(Ox) = 2 J(O2) [O2] = 5.4e5
   cm⁻³ s⁻¹ = 28 ppb/day. Adding it to "MCM + J(N2O5), J(HNO4)" turns the O3 change from
   −2.65 to +25.1 ppb/day; OH changes by 1 % and the SO2 lifetime from 44.7 to 45.1 days
   **[measured]**. So the O3-tendency rows of the prototype compare mechanisms, not the real
   O3 budget. Fixed in sections 1 and 2 (MP4) and in `sandbox_cmp/README.md` sections 5 and 6
   (which had said "small over days" and "above ~25 km").
2. **MCM halogen wording.** Summary item 2 now names the 32 Cl-releasing steps and says "no
   Br or I atom chemistry". `halogens.md` said MCM has "no HCl"; HCL does appear, once, as a
   product of CHCLOOA, with no sink. Fixed there.
3. **O + SO2.** JPL 19-5 (I3 fall-off) is 1.7 times MCM at 220 K, 50 hPa and 0.37 times at
   298 K, 1 atm; SANDBOX has no O + SO2 at all. Small (under 1 % of stratospheric SO2 loss)
   but it is a reactant set with two sources, so `strat_sai` takes the JPL form. Added to 3.4.
4. **sCI + SO2.** MCM v3.3.1 gives all 49 sCI + SO2 the same 7.0e-14 cm³ s⁻¹, and the
   limonene sCI (LIMBOO, LMKBOO) are lost only to NO, NO2, SO2 and H2O, with no unimolecular
   decay. Direct measurements of CH2OO + SO2 give about 3.9e-11 cm³ s⁻¹
   ([Welz et al. 2012](https://doi.org/10.1126/science.1213229)), about 500 times MCM. Two
   consequences: the MCM sCI channel may understate SO2 oxidation in BVOC runs, and in dry air
   (4 ppm H2O, 10 ppb SO2 at 50 hPa) MCM's water loss, 1.6e-17 [H2O] = 1e-4 s⁻¹, is ten times
   slower than its SO2 loss, so almost every sCI would oxidise SO2, which no measured
   unimolecular lifetime supports. Marked as a named option in 3.4 and question 10.
5. **Stratospheric reactions missing from both sources** (O + OH, O + HO2, H-atom chemistry,
   OH + OH, O1D + H2, O1D + N2O; halogen + MCM-oxygenate cross terms such as Br + HCHO) added
   to 3.5. They are small at 20 km and needed above about 30 km and for any halogen run with
   MCM organics.
6. **J outside TUV-x.** Two of the six new J names come from MCM's surface fit, which 5.4
   refuses above the boundary layer; they now need TUV-x analogues in `strat_sai`. Overhead
   O3 column and SAI aerosol optical depth added as J inputs.

### 12.3 Applicability of MCM rate expressions at stratospheric T and p (not changed, for M5-M6)

- Pressure is handled correctly in the code path the prototype uses: `air_environment`
  gives M = p / (k_B T), O2 = 0.2095 M, N2 = 0.781 M, and every KMT fall-off and the explicit
  `M`, `O2`, `N2` factors take them from the environment. No fixed 1 atm value was found in
  `mcm_rates.py`.
- Fall-off forms: MCM uses the IUPAC Troe form with a per-reaction F_c and
  N = 0.75 − 1.27 log₁₀ F_c, reference 300 K; JPL uses F_c = 0.6, N = 1, reference 298 K.
  Both are valid fall-off forms at 50 hPa; the differences are in the evaluated parameters
  (OH + SO2 38 %, OH + NO2 23 %, HO2 + NO2 a factor 1.7-1.9), which is why JPL owns the
  inorganic sets in `strat_sai`.
- Temperature range: several MCM/IUPAC expressions are evaluated only down to about 250 K
  (for example HO2 + O3, (T/300)^4.57 exp(693/T)); some have no T dependence where JPL has one
  (KMT05 for CO + OH; O1D + H2O, 2.14e-10, against JPL 1.63e-10 exp(60/T), equal within 1 %
  at 220 K). Under MP3 this matters only for the MCM organic chain in `strat_sai`: CH4 + OH is
  10 % faster in MCM than JPL at 220 K (8.5e-16 vs 7.7e-16), and MCM's RO2 + NO alkyl nitrate
  yields are fixed values derived for about 298 K and 1 atm, while the real yields rise at
  low T and fall at low p. Log each MCM organic rate used at 220 K with its evaluated range
  before `strat_sai` is labelled anything but "research".
- Photolysis: MCM's J fit has no stated altitude, albedo or O3 column on the MCM page; use
  TUV-x only above the boundary layer (as planned).

### 12.4 Caveats on the prototype comparison (no change to its conclusions)

- The tropospheric case has no CO or VOC (OH 2-4e7 cm⁻³ at noon), so its 1.2-2.0 day SO2
  lifetimes are a mechanism test, not ambient values.
- The stratospheric case is not at NOy steady state on day 5: the HNO3 photolysis lifetime
  from the same J table is about 17 days. And it has no aerosol: under SAI, N2O5 hydrolysis on
  sulfate controls NOx, and it would shrink the photolysis-coverage gap that drives the
  NOx difference. Add a case with surface area and γ(N2O5) in both models before quoting NOx.
- Neither model carries N2O, so there is no NOy source; fine for 5 days, not for long runs.
