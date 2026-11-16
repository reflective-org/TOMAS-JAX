# mxyl_mcm.eqn — provenance

- **Source:** Master Chemical Mechanism, MCM v3.3.1, website export of the MXYL (m-xylene)
  subset in KPP format with the inorganic and generic rate coefficients included:
  `https://mcm.york.ac.uk/MCM/export/download?selected[]=MXYL&inorganic=on&generic=on&format=kpp`
- **Retrieved:** 2026-10-23 18:57 UTC (curl, HTTP 200, 54 553 bytes). Kept byte-identical;
  never edit it in place (tutorial edits are applied in code, `box/coupling.py`).
- **sha256:** `a4b33df178b265bc9b7f124744dbc1b1c31028073550c7c7c5d95018b43c35ec`
- **Contents:** the MCM citation header, `#INCLUDE atoms`, `#DEFVAR` with 301 species
  (300 used; `H2O` declared and unused), the `#INLINE F90_RCONST` block with the 60-member RO2
  sum, and `#EQUATIONS` with 860 reactions `<1>`…`<860>`. MXYL + OH: k = 2.31e-11 cm³ s⁻¹
  (no temperature dependence in the MCM), four channels `<384>`–`<387>`.
- **Use:** merged in code with the limonene subset (`gaschem/merge.py`,
  `box/coupling.tutorial_mechanism(("MXYL",))`) when a run sets m-xylene or one of its
  products (ASSUMPTIONS A27). 326 of its reactions are identical to limonene reactions and are
  kept once; 534 reactions and 191 species are new; its products use J_CH3CHO and J_MVK_CO
  (mapped in `photolysis.py`, as for toluene).
- **Citation (required by the MCM):** "The chemical mechanistic information was taken from the
  Master Chemical Mechanism, MCM v3.3.1, via website: http://mcm.york.ac.uk/"; aromatic
  schemes: Jenkin et al., Atmos. Chem. Phys. 3, 181 (2003), https://doi.org/10.5194/acp-3-181-2003;
  Bloss et al., Atmos. Chem. Phys. 5, 641 (2005), https://doi.org/10.5194/acp-5-641-2005.
