# GasBench (Julia spike)

Julia solvers for the shared gas-chemistry benchmark in `../bench/BENCHMARK.md`: the problem
files are read as they are, and each run is scored with the benchmark's metric and written in
its reference format, so `bench/scripts/compare.py` scores it too.

Julia 1.12.7 (official Apple-silicon tarball) lives in `../opt`, and its packages in
`../depot`. Nothing is installed globally. `source env.sh` sets the paths.

## Layout

- `src/GasBench.jl`: the whole package.
  - Problem reader.
  - Right-hand side: rates r_j = k_j · y_a · y_b over the reactant entries, then
    dy/dt = S r with S the sparse net stoichiometry.
  - Analytic sparse Jacobian on a fixed pattern that includes the diagonal.
  - Interval loop with frozen k (KPP semantics).
  - Solver set-up through one integrator per run, restarted with `reinit!` every interval.
  - Metric and output writer.
  - PrecompileTools workload on a 4-species toy problem.
- `scripts/check.jl`: the unit-test vectors (f, J·v, k, RO2) for all five problems.
- `scripts/bench.jl`: one case in one fresh process, timed per BENCHMARK.md section 7.
- `scripts/run_case.sh`, `run_batch.sh`, `latency.sh`, `ab_session.sh`: bounded batch drivers.
- `scripts/build_sysimage.jl`, `precompile_exec.jl`: the PackageCompiler system image.
- `scripts/julia_tables.py`: the results tables from `bench/results/*_timing.json`.
- `sysimage/env`: the environment that the system image is built from and runs in.
- `sysimage/tools`: PackageCompiler.

## Solvers

All solvers use KLU on the sparse pattern:

| Name | Solver |
|---|---|
| `rodas5p`, `rodas4`, `rodas4p`, `rodas3`, `ros3p`, `ros34pw2`, `rosenbrock23` | OrdinaryDiffEq Rosenbrock methods; LinearSolve `KLUFactorization` |
| `fbdf`, `qndf` | OrdinaryDiffEq BDF and NDF; LinearSolve `KLUFactorization` |
| `cvode` | Sundials CVODE_BDF with SUNDIALS KLU, set to AMD ordering |
| `cvode_colamd` | the same with SUNDIALS' default ordering (COLAMD), which is about 10 times slower here |

## Reproduce

Paths are relative to this folder. The bench data (`../bench/problems`, `../bench/reference`)
come from `../bench/scripts` (BENCHMARK.md section 10).

```sh
source env.sh
$JULIA --startup-file=no scripts/install.jl          # once: packages into ../depot
$JULIA --startup-file=no scripts/check.jl            # unit-test vectors, all problems
scripts/run_case.sh 900 lim726_L3 900 fbdf 1e-3 1e4 3 nom 600   # one timed case
scripts/run_batch.sh logs/nominal.log nom 3 "rodas5p fbdf qndf cvode" "lim726_L3 900" "syn5742_L3 60"
$JULIA --startup-file=no --project=sysimage/tools scripts/build_sysimage.jl   # about 5 min
scripts/latency.sh logs/latency.log lat
scripts/ab_session.sh logs/ab.log ab
../venv/bin/python scripts/julia_tables.py nom loose tight
```

`run_case.sh` arguments, in order:

1. timeout [s]
2. problem
3. coupling [s]
4. solver
5. rtol
6. atol
7. number of warm runs
8. tag
9. in-run cap [s]

It writes these files to `../bench/results`:

- `<tag>_<solver>_rtol<r>_atol<a>_<problem>_dt<c>.json` and its `_hourly.npz`, in the
  reference format
- the matching `_timing.json`
