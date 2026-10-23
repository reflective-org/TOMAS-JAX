# toluene_mcm.eqn — provenance

- **Source:** Master Chemical Mechanism, MCM v3.3.1, website export of the TOLUENE
  subset in KPP format with the inorganic and generic rate coefficients included:
  `https://mcm.york.ac.uk/MCM/export/download?selected[]=TOLUENE&inorganic=on&generic=on&format=kpp`
- **Retrieved:** 2026-10-23 16:42 UTC (curl, HTTP 200, 52 522 bytes). Kept byte-identical;
  never edit it in place (tutorial edits are applied in code, `box/coupling.py`).
- **sha256:** `186f1032fa482282381d2a4739c48ee296d4cf2801972f0f1caa74efc69f5612`
- **Contents:** the MCM citation header, `#INCLUDE atoms`, `#DEFVAR` with 292 species
  (291 used in reactions; `H2O` is declared and unused), the `#INLINE F90_RCONST` block
  with the 56-member RO2 sum, and `#EQUATIONS` with 831 reactions `<1>`…`<831>`
  (inorganic `<1>`–`<35>` and generic rate names as in the limonene export). The
  aromatic ring-opening scheme is that of Jenkin et al. (2003) and Bloss et al. (2005).
- **Use:** merged in code with the limonene subset (`gaschem/merge.py`,
  `box/coupling.tutorial_mechanism(toluene=True)`) when a scenario sets toluene or one of
  its products (ASSUMPTIONS A27). The 326 reactions it shares with the limonene file are
  identical (reactants, products and rate expression) and are kept once; 505 reactions
  and 181 species are new.
- **Citation (required by the MCM):** "The chemical mechanistic information was taken
  from the Master Chemical Mechanism, MCM v3.3.1, via website: http://mcm.york.ac.uk/";
  aromatic schemes: Jenkin et al., Atmos. Chem. Phys. 3, 181 (2003),
  https://doi.org/10.5194/acp-3-181-2003; Bloss et al., Atmos. Chem. Phys. 5, 641 (2005),
  https://doi.org/10.5194/acp-5-641-2005; non-aromatic schemes: Jenkin et al., Atmos.
  Environ. 31, 81 (1997), https://doi.org/10.1016/S1352-2310(96)00105-7; Saunders et al.,
  Atmos. Chem. Phys. 3, 161 (2003), https://doi.org/10.5194/acp-3-161-2003.
