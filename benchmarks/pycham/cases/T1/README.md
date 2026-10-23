# Cases T1–T3: an aromatic VOC + NOx + O3 in sunlight, merged mechanism (gas phase only)

T1 toluene, T2 benzene, T3 m-xylene. One module (`cases/T1/`) runs all three by `--case`;
`cases/T2/` and `cases/T3/` hold thin wrappers and their results.

```bash
cd tomas-jax
PYCHAM_BENCH=../pycham-bench .venv/bin/python -m benchmarks.pycham.cases.T1.run --case T2      # inputs, PyCHAM, TOMAS-JAX x2 (~1.5 min)
PYCHAM_BENCH=../pycham-bench .venv/bin/python -m benchmarks.pycham.cases.T1.analyse --case T2  # metrics + figure (~10 s)
PYCHAM_BENCH=../pycham-bench .venv/bin/python -m benchmarks.pycham.cases.T3.run                # the same for m-xylene
```

`run.py --skip-pycham` reuses the PyCHAM output in `pycham-bench/outputs/<case>`. The write-up is
the "Cases T1–T3" section of `docs/aaar/pycham_comparison.md`.

Inputs: the `level1_urban` gases with 5 ppb of the VOC (the Urban selector's level,
`box.scenario.URBAN_VOC_PPB_ON`), 6 h from 10:00 EDT on 1 July at 40 °N, 75 °W, T = 298.15 K,
P = 101325 Pa, RH = 0.5, no NO emission, every aerosol process off. TOMAS-JAX integrates the
limonene mechanism with that VOC's MCM subset merged in (`box.coupling.aromatics_used`); PyCHAM
reads the same merged mechanism as one KPP file written by `pycham_inputs.write_merged_eqn`
(parsed back and checked against the in-code merge), `number_size_bins = 0`, no walls,
`update_step = 60 s`, our J per 300 s interval prescribed (`J.xlsx`, with J_CH3CHO = MCM 13 and
J_MVK_CO = 23 for toluene and m-xylene).

Checks built into `run.py` (`results/run_info.json`): PyCHAM's RO2 pool equals the merged
mechanism's (183 / 171 / 187 members); the run used the merged mechanism (907 / 847 / 917
species); the air number density agrees to 2e-16.

Results (2026-10-23, `results/summary.json` of each case, 300 s coupling, max relative
deviation over 6 h): toluene 3.8e-4, O3 2.4e-4, NO 7.1e-4, NO2 4.4e-4, OH 5.3e-4, HO2 1.0e-3
(131 species above 1e-5 ppb, all within 10 %); benzene 4.2e-5, O3 7.2e-5, NO 2.1e-4, NO2 1.5e-4,
OH 5.1e-4, HO2 6.8e-4 (54 species); m-xylene O3 1.3e-4, NO 9.1e-4, NO2 7.5e-4, OH 5.1e-4,
HO2 9.4e-4, the VOC 1.7e-2 on the 0.007 ppb left after 6 h (175 species). The 60 s runs use J
written for the 300 s intervals, so their early deviations (up to 1.4e-2) are J timing, not
chemistry.

Differences from an identical set-up: the same as G2 (frozen rates per coupling interval,
solver tolerances, Buck vs Hyland–Wexler water, `<34>` at 0, SA as a counter, `PROD` dropped,
D9/D9PROD SMILES added to the xml and unused). Labels of the added reactions are `T<n>`, `B<n>`,
`X<n>` in both files; PyCHAM orders reactions by position, not label.

Results kept in the repo per case: `ours_dt300.npz`, `ours_dt60.npz`, `pycham.npz` (all species,
300 s grid, float32), `key_species.npz`, `species_stats.csv`, `summary.json`, `run_info.json`,
`j_prescribed.csv`, `pycham_model_var.txt`; figures `figures/<case>_key_species.png`.
