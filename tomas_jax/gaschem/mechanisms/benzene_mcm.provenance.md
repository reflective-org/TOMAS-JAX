# benzene_mcm.eqn — provenance

- **Source:** Master Chemical Mechanism, MCM v3.3.1, website export of the BENZENE subset in
  KPP format with the inorganic and generic rate coefficients included:
  `https://mcm.york.ac.uk/MCM/export/download?selected[]=BENZENE&inorganic=on&generic=on&format=kpp`
- **Retrieved:** 2026-10-23 18:57 UTC (curl, HTTP 200, 28 174 bytes). Kept byte-identical;
  never edit it in place (tutorial edits are applied in code, `box/coupling.py`).
- **sha256:** `24bdc02f065838f5ccb365e70e09bc389b4eaee7b074c8606274465aa3776146`
- **Contents:** the MCM citation header, `#INCLUDE atoms`, `#DEFVAR` with 157 species
  (156 used; `H2O` declared and unused), the `#INLINE F90_RCONST` block with the 28-member RO2
  sum, and `#EQUATIONS` with 436 reactions `<1>`…`<436>` (inorganic `<1>`–`<35>` and generic
  rate names as in the limonene export). BENZENE + OH: k = 2.3e-12 exp(−190/T) cm³ s⁻¹
  (1.22e-12 at 298 K), three channels `<101>`–`<103>`.
- **Use:** merged in code with the limonene subset (`gaschem/merge.py`,
  `box/coupling.tutorial_mechanism(("BENZENE",))`) when a run sets benzene or one of its
  products (ASSUMPTIONS A27). 105 of its reactions are identical to limonene reactions and are
  kept once; 331 reactions and 121 species are new; no photolysis frequency beyond the
  limonene subset's.
- **Citation (required by the MCM):** "The chemical mechanistic information was taken from the
  Master Chemical Mechanism, MCM v3.3.1, via website: http://mcm.york.ac.uk/"; aromatic
  schemes: Jenkin et al., Atmos. Chem. Phys. 3, 181 (2003), https://doi.org/10.5194/acp-3-181-2003;
  Bloss et al., Atmos. Chem. Phys. 5, 641 (2005), https://doi.org/10.5194/acp-5-641-2005.
