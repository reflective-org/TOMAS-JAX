# Case T1: toluene + NOx + O3 in sunlight, merged mechanism (gas phase only)

```bash
cd tomas-jax
PYCHAM_BENCH=../pycham-bench .venv/bin/python -m benchmarks.pycham.cases.T1.run       # inputs, PyCHAM, TOMAS-JAX x2 (~1.5 min)
PYCHAM_BENCH=../pycham-bench .venv/bin/python -m benchmarks.pycham.cases.T1.analyse   # metrics + figure (~10 s)
```

`run.py --skip-pycham` reuses the PyCHAM output in `pycham-bench/outputs/T1`. The write-up is
the "Case T1" section of `docs/aaar/pycham_comparison.md`.

Inputs: the `level1_urban` gases with 5 ppb toluene (the Urban toggle's "on" value,
`box.scenario.URBAN_TOLUENE_PPB_ON`), 6 h from 10:00 EDT on 1 July at 40 °N, 75 °W, T = 298.15 K,
P = 101325 Pa, RH = 0.5, no NO emission, every aerosol process off. TOMAS-JAX integrates the
limonene + toluene mechanism (`box.coupling.uses_toluene`); PyCHAM reads the same merged
mechanism as one KPP file written by `pycham_inputs.write_merged_eqn` (parsed back and checked
against the in-code merge), `number_size_bins = 0`, no walls, `update_step = 60 s`, our J per
300 s interval prescribed (`J.xlsx`, with J_CH3CHO = MCM 13 and J_MVK_CO = 23).

Checks built into `run.py` (`results/run_info.json`): PyCHAM's RO2 pool equals the merged
mechanism's (183 members); the run used the 907-species mechanism; the air number density
agrees to 2e-16.

Results (2026-10-23, `results/summary.json`): at 300 s coupling the max relative deviation over
6 h is 3.8e-4 for toluene, 2.4e-4 O3, 7.1e-4 NO, 4.4e-4 NO2, 5.3e-4 OH, 1.0e-3 HO2, 2.7e-3 H2O2;
131 species above 1e-5 ppb, all within 10 %, median |NMB| 4.9e-4. The 60 s run uses J written
for the 300 s intervals, so its early deviations (up to 1.4e-2) are J timing, not chemistry.

Differences from an identical set-up: the same as G2 (frozen rates per coupling interval,
solver tolerances, Buck vs Hyland–Wexler water, `<34>` at 0, SA as a counter, `PROD` dropped,
D9/D9PROD SMILES added to the xml and unused). Labels of the added toluene reactions are `T<n>`
in both files; PyCHAM orders reactions by position, not label.

Results kept in the repo: `ours_dt300.npz`, `ours_dt60.npz`, `pycham.npz` (all species, 300 s
grid, float32), `key_species.npz`, `species_stats.csv`, `summary.json`, `run_info.json`,
`j_prescribed.csv`, `pycham_model_var.txt`; figure `figures/T1_key_species.png`.
