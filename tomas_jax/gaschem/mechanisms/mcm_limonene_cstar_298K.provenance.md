# mcm_limonene_cstar_298K.csv — provenance

- **From:** Shantanu Jathar (Colorado State University, AAAR 2026 tutorial co-instructor),
  received 2026-10-16. Kept byte-identical (comment lines use LF, data rows CRLF;
  `.gitattributes` marks it `-text` so git does not convert line endings).
- **sha256:** `598273949d8dc646e746929375ac397c45b97180f2820ffed61483d5e07d2b49`
- **Method (from the file's header):** pure-component vapour pressure Psat of each
  limonene MCM v3.3.1 species at 298.15 K from PyCHAM 5.8.0 with the Nannoolal et al.
  (2008) method (stored in `limonene/output/mcm_soa/PyCHAM_output.nc`), converted with
  C* [µg m⁻³] = 1e6 · MW [g mol⁻¹] · Psat [Pa] / (8.314 · 298.15) (O'Meara et al. 2014,
  eq. 1, as in PyCHAM `cham_up.py`). The file does not say which normal-boiling-point
  estimate was used with Nannoolal (asked of Shantanu, ASSUMPTIONS question 1).
- **Contents:** 4 `#` comment lines, header
  `species,formula,molar_mass_g_mol,psat_Pa_298.15K,cstar_ug_m3_298.15K,log10_cstar`,
  700 rows (458 closed-shell organics, 242 radicals). It excludes LIMONENE, the D9/D9PROD
  OH tracer, CO, CO2, CH4 and species without carbon. Every row is a species of
  `limonene_mcm.eqn`; molar masses agree with the parser's to 0.05 g mol⁻¹ and C* is
  reproduced from Psat and MW to 1e-4.
- **Use:** default `cstar_table` of the Level 2 and 3 presets (ASSUMPTIONS A8).
  `gaschem.volatility.load_cstar_table` reads the `cstar_ug_m3_298.15K` column; the other
  columns are informational. Radicals in the table are ignored (closed-shell rule).
