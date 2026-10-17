# Prototypes for the Julia decision

- `bench/`: the shared gas-chemistry benchmark (generators, scoring, baselines). Its document is
  `docs/pro/julia/BENCHMARK.md`.
- `jax_sparse_spike/`: the sparse JAX solver measured on that benchmark.

The Julia side, the plan and the review are in the private repository
[reflective-org/TOMAS.jl](https://github.com/reflective-org/TOMAS.jl). See
`docs/pro/julia/README.md`. Nothing here is used by the tutorial.
