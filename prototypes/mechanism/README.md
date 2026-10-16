# Mechanism prototypes (planning, 2026-10-16)

Planning artefacts for [docs/pro/mechanism/architecture.md](../../docs/pro/mechanism/architecture.md).
Nothing here is imported by `tomas_jax`; the scripts only read the package and the files
named below. No MCM files were downloaded: everything starts from
`tomas_jax/gaschem/mechanisms/limonene_mcm.eqn`.

| File | What it does | Output |
|---|---|---|
| `mechlib.py` | closure extraction, `.eqn` writer, composer (duplicates, rate conflicts, cross-source overlaps, RO2 union, namespaced labels), atom-balance helpers | library |
| `build_inorganic.py` | extracts the MCM inorganic + CH4 + SO2 module (29 species, 68 reactions) | `modules/mcm_inorganic_c1_sulfur.eqn`, `results/inorganic_summary.json` |
| `compose_demo.py` | composes inorganic + limonene, shows each composer rule failing on purpose, RO2 and atom-balance checks, SANDBOX species coverage, J-name coverage incl. TUV-x port | `results/compose_summary.json` |
| `sandbox_dump.py` | SANDBOX JPL 19-5 gas rate coefficients at three conditions (run with the SANDBOX venv) | `results/sandbox_rates.json` |
| `rate_crosscheck.py` | MCM vs JPL 19-5 for the 24 shared reactant sets | `results/rate_crosscheck.json` |
| `run_inorganic.py` | 24 h gas-only run of the module vs the limonene file with no VOC | `results/run_inorganic.json` |
| `scaling.py` | Jacobian sparsity, LU fill, dense and sparse timings for 29 to 5741 species | `results/scaling_*.json` |
| `estimate.py` | per-step cost models and 24 h gas-solve estimates | `results/runtime_estimates.json` |
| `cstar_nannoolal.py` | Nannoolal c* from SMILES (UManSysProp as in PyCHAM) vs Shantanu's table; whole MCM SMILES file | `results/cstar_nannoolal.json` |
| `registry_example.toml` | sketch of the registry format | |
| (peak memory) | `/usr/bin/time -l` around `scaling.py` for 725, 2606, 5741 species | `results/peak_memory.json` |

`mcm_coverage_scan.py`, `mcm_vs_jpl_rates.py`, `halogen_overlap.py` and `results/halogen_*`
belong to the companion note [`mcm_coverage.md`](../../docs/pro/mechanism/mcm_coverage.md).

Run (bounded; paths as on Ali's laptop):

```bash
WT=<this worktree>; PY=<tomas-jax venv python>; export PYTHONPATH=$WT TUVX_JAX_ROOT=<tuvx-jax>
gtimeout 180 $PY build_inorganic.py
gtimeout 300 $PY compose_demo.py
gtimeout 120 <SANDBOX>/.venv/bin/python sandbox_dump.py <SANDBOX>/stratchem-jax results/sandbox_rates.json
gtimeout 120 $PY rate_crosscheck.py
gtimeout 500 $PY run_inorganic.py
for n in inorganic c3 limonene copies2 copies4 copies7 copies9; do
  gtimeout 540 $PY scaling.py $n <pro-wt>/prototypes/pro/perf results/scaling_$n.json; done
python3 estimate.py
gtimeout 560 <pycham-bench>/.venv/bin/python -I cstar_nannoolal.py <pycham-bench>/PyCHAM/umansysprop \
    <pycham-bench>/inputs/G3/limonene_mcm.xml $WT/tomas_jax/gaschem/mechanisms/mcm_limonene_cstar_298K.csv \
    results/cstar_nannoolal.json
```

Timings were taken on an Apple M4 Pro shared with other jobs (load average 6-19), so
treat them as upper bounds.

## MCM subset vs SANDBOX (JPL 19-5) box comparison

`build_mcm_inorganic_sulfur.py` writes `mcm_inorganic_sulfur.eqn` (closure of
{O3, NO, NO2, CO, CH4, SO2, H2O2, HONO, HNO3, N2O5, H2, H2O}: 28 species, 67 reactions,
atom balance explained for every reaction) and `results/mcm_inorganic_sulfur_build.json`.
`sandbox_cmp/` runs it in `tomas_jax.gaschem` and SANDBOX `stratchem-jax` (halogens off)
with one shared TUV-x J table, for a 298 K tropospheric case and a 220 K, 50 hPa SAI-like
case; scripts, results, three plots and the write-up are in
[`sandbox_cmp/README.md`](sandbox_cmp/README.md).
