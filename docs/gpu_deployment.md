# GPU Deployment Guide

TOMAS-JAX is designed for GPU execution via JAX's XLA compiler. This document covers hardware considerations, code path selection, and batch processing.

## float64 Requirement

TOMAS-JAX **requires float64 precision**. This is non-negotiable for aerosol microphysics:

- Mass conservation thresholds operate at 1e-15 relative error
- Condensation sink calculations involve exp(-CS*dt) where CS spans 6 orders of magnitude
- Coagulation kernels have N^2 scaling — rounding errors compound catastrophically
- MNFIX partial-transfer algorithm requires exact bin boundary comparisons

### float64 Performance by GPU Type

| GPU | float64 vs float32 | Notes |
|-----|---------------------|-------|
| NVIDIA A100 | 1/2 FP32 throughput | Best GPU choice for float64 workloads |
| NVIDIA H100 | 1/2 FP32 throughput | Excellent float64 support |
| NVIDIA V100 | 1/2 FP32 throughput | Good float64 support |
| NVIDIA consumer (RTX 30xx/40xx) | 1/32 FP32 throughput | **Avoid** — float64 is crippled |
| Google TPU v3/v4 | Native float64 | Excellent, but JAX TPU support varies |
| AMD MI250X/MI300X | 1/2 FP32 throughput | Good float64 support |

**Recommendation**: Use datacenter GPUs (A100, H100, MI250X) or TPUs. Consumer GPUs will be 16x slower than expected due to float64 penalties.

### Enabling float64

float64 is enforced centrally in `tomas_jax/core/config.py` (line 17):
```python
jax.config.update("jax_enable_x64", True)
```

All library modules import from config, inheriting this setting. Entry-point scripts (tests, benchmarks) may also set it for standalone execution.

## Selecting the GPU Platform

```bash
# Use GPU (default if available)
JAX_PLATFORMS=gpu python run_box_model.py

# Force CPU (for debugging)
JAX_PLATFORMS=cpu python run_box_model.py

# Verify platform
python -c "import jax; print(jax.devices())"
```

## Which Code Paths to Use

### GPU-ready (JIT-compiled, pure JAX)

These compile to a single XLA program with zero Python dispatch overhead:

| API | Description |
|-----|-------------|
| `make_step(processes, cond_method='ppm_jit')` | Composable process API (recommended) |
| `run_condensation_scan()` / `run_condensation_scan_tfl()` | Scan-fused condensation loops |
| `run_combined_scan_ppm()` / `run_combined_scan_tfl()` | Scan-fused coag + cond |
| `run_full_scan()` | Scan-fused nucl + coag + cond |
| `condensation_step_jit` / `condensation_step_tfl_jit` | Single condensation steps |
| `coag_euler_step()` | Forward Euler coagulation |
| `diffrax_step()` | Adaptive Tsit5 coagulation |

### Legacy paths to AVOID on GPU

These use numpy arrays and Python loops — they will silently transfer data CPU↔GPU every step:

| API | Replacement |
|-----|-------------|
| `condensation_step(..., method='tfl')` | Use `method='tfl_jit'` |
| `condensation_step(..., method='ppm')` | Use `method='ppm_jit'` |
| `ezcond()` (from `physics/ezcond.py`) | Use `ezcond_tfl_jax()` |
| `ezcond_ppm()` (from `physics/ezcond_ppm.py`) | Use `ezcond_ppm_jax()` |

A `DeprecationWarning` is raised when legacy methods are used.

## Batch Processing with vmap

For ensemble runs (e.g., Monte Carlo, parameter sweeps), use `jax.vmap` to batch across scenarios:

```python
from tomas_jax.solvers.condensation import make_step

step_fn = make_step(
    ['nucleation', 'coagulation', 'condensation'],
    cond_method='ppm_jit',
)

# Batch over different initial conditions
# Nk_batch shape: (n_scenarios, nbins)
# Mk_batch shape: (n_scenarios, nbins, icomp)
batched_step = jax.vmap(step_fn, in_axes=(0, 0, 0, None, None, None, None, None, None, None))

Nk_out, Mk_out, Gc_out = batched_step(
    Nk_batch, Mk_batch, Gc_batch,
    xk, temp, pres, boxvol, rh, alpha, dt,
)
```

`vmap` compiles into a single kernel that processes all scenarios in parallel on the GPU. This is dramatically faster than a Python loop over scenarios.

**Important**: `vmap` requires all batched arrays to have the same leading dimension. Scalar parameters (temp, pres, etc.) are broadcast automatically via `in_axes=None`.

## Large vmapped ensembles (adaptive gas sub-steps, heun_adaptive)

Measured 2026-11-23 on one H200 (JAX 0.6.2): `benchmarks/python/heun_cesm_gpu_batch.py` runs every
CESM cell at levels 43-45 (~43-62 hPa, 165,888 cells, one snapshot hour) as a `jax.vmap` batch in
chunks, for a sweep of SO2, with `make_step(coag_solver='heun_adaptive', gas_substep='adaptive')`.

- **`cap_warnings=False` is required.** The gas and coagulation cap warnings are `jax.debug.callback`s,
  and JAX unrolls a callback once per batch member: 8192 cells took 217 s to compile and 5.3 s per
  360 s step with them, 9 s and 0.06 s without. With `cap_warnings=False` the step reports nothing by
  itself: read `info['hit_cap']` and `info['coag_dt_left']` per cell (`return_substep_info=True`)
  and treat a binding cap as a failure, as the benchmark does.
- **A vmap of batch size exactly 1 aborts XLA** (`hlo_dataflow_analysis.cc` check failure). Sizes of 2 and up
  compile. Pad chunks to at least 2 cells.
- **`mnfix_trim=True` was 6-8x faster than `False` under vmap** here (4096 cells spanning the T range, 1 h
  of simulated time: 5.7 s against 33.7 s at 1e2 ppbv SO2, 129 s against 1030 s at 1e4 ppbv). This is
  the opposite of the older advice to pass `mnfix_trim=False` for vmapped ensembles; time both for a
  new workload (`--mnfix-trim`, the benchmark's calibration does).
- **Memory is not the limit, time is.** ~153 KiB per cell (peak, 40 bins), so ~130k cells fit in 35 GiB; the
  run time grows linearly with the number of cells above ~1k cells (the GPU is saturated), so a chunk of
  4096 loses nothing against a larger one and loses less to lockstep (a batched `while_loop` runs
  to its slowest cell: at 1000 ppbv the median cell takes 50 gas sub-steps per outer step, the worst 1321).
- Set `XLA_PYTHON_CLIENT_PREALLOCATE=false` and `XLA_PYTHON_CLIENT_MEM_FRACTION` before importing JAX
  to cap what the process may take on a shared GPU.

Results (one H200, GPU shared node, 6 h, dt = 360 s, 40 bins, cells sorted by temperature, chunks of
4096, uniform SO2 in all 165,888 cells; the time for 3.87M cells is the measured cells/s extrapolated
linearly):

| SO2 (ppbv) | cells/s | est. 3.87M cells | worst-cell n_sub/step | median-cell mean |
|---|---|---|---|---|
| 1e-2 | 20,860 | 3.1 min | 1 | 1.0 |
| 1e-1 | 18,985 | 3.4 min | 1 | 1.0 |
| 1 | 4,302 | 15 min | 8 | 1.0 |
| 10 | 1,004 | 64 min | 48 | 1.6 |
| 100 | 248 | 4.3 h | 249 | 10.0 |
| 1e3 | 54 | 19.9 h | 1486 | 45.0 |

Chunk ordering matters because of lockstep. Offline replay on the saved per-cell counts
(`benchmarks/python/heun_chunk_replay.py`, a lower-bound cost model): the 1e3 sub-step cost correlates with
k1(T,P,RH)*[OH]*[SO2] (Spearman 0.95) far better than with T (0.22). With a 0.1 ppbv background and
50 cells at 1e3, T-sorted chunks cost 7x the SO2-sorted ones (730 s against 104 s for 165,888 cells);
with a uniform 1e3 field k*OH*SO2 sorting is ~1.7x faster than T sorting (oracle ~3.4x). Sort chunks by
k*OH*SO2 (or SO2 where it varies), not by T, when SO2 differs between cells. The replay is not yet
confirmed on the GPU as below. Details: `PROGRESS.md` 2026-11-23, 2026-11-24.

GPU test of the orderings (`benchmarks/python/heun_sort_gpu.py`, 2026-11-24; sulfate layout,
`mnfix_trim=True`, `heun_adaptive` + adaptive gas sub-steps, 6 h, dt = 360 s, 40 bins, chunks of 4096, GPU 0
of the shared node, host CPU shared and loaded; timings are summed chunk run times, compile excluded).
Cells: the 24 levels PR 20 simulates (CESM levels 27-50, 1.2-143 hPa, 1,327,104 cells) with per-cell
T, P, RH, OH and the file's own SO2 as background (p50 0.003-0.04 ppbv); SO2 is replaced by one uniform
value in levels 43-45 (165,888 cells) only. Lockstep waste = sum of chunk-max sub-steps / total
sub-steps (1.0 = perfect packing).

| ordering | 100 ppbv: s | vs T | waste | 1 ppbv: s | vs T | waste |
|---|---|---|---|---|---|---|
| columns (levels mixed per chunk) | 1102 | 0.66x | 20.1 | 61.6 | 0.82x | 2.54 |
| T | 733 | 1.0x | 12.7 | 50.6 | 1.0x | 2.10 |
| lonlat (level-major) | 173 | 4.2x | 2.87 | 24.7 | 2.05x | 1.22 |
| SO2 | 177 | 4.1x | 2.88 | 28.7 | 1.76x | 1.26 |
| k1*OH*SO2 | 103 | 7.1x | 1.88 | 21.8 | 2.32x | 1.15 |
| k1*OH*SO2 tiers, T within tier | 91 | 8.0x | 1.52 | 20.0 | 2.53x | 1.07 |
| oracle (measured sub-steps; bound) | 72 | 10.1x | 1.10 | 19.8 | 2.55x | 1.01 |
| bucketed (512-cell chunks for hot cells) | 437 | 1.7x | 1.76 | 58.1 | 0.87x | 1.12 |

Sorting by T is the worst usable key when SO2 differs between levels (it scatters the hot band over
every chunk); k1*OH*SO2 with half-decade tiers and T inside a tier reaches ~80% of the oracle's speed.
Per-cell final N is identical in every ordering (pure permutation, max relative difference 0), no NaN,
no binding cap. Limits: SO2 is uniform inside levels 43-45, so the hot cells are a whole band, not a
localized plume; `bucketed` (hot = k1*OH*SO2 >= 10x the median, which picked 354,902 cells, twice the
elevated ones) lost to under-filled 512-cell chunks, so a tighter cut is untested; 1e3 ppbv was not run
(too slow and not realistic); PR 20 itself runs 6 x 60 s without `gas_substep`, so absolute times are
the benchmark's, not PR 20's.

Two further scenarios (`heun_sort_gpu.py --scenario injection|advected`; same cells, layout, MNFIX trim,
step settings and GPU 1 of the shared node, idle at launch, host CPU shared and loaded; 6 h; summed chunk run times):

*Advected patch*: SO2 peaks at 100 ppbv at +-30 deg lat with a Gaussian taper (sigma 7 deg, cut at +-15 deg) in
levels 43-45 (55,296 cells, 10-100 ppbv), the file's background elsewhere. Hot cells reach 10,287 sub-steps.

| ordering | time (s) | vs T | waste |
|---|---|---|---|
| T | 442 | 1.0x | 16.1 |
| columns | 373 | 1.2x | 13.4 |
| SO2 | 94 | 4.7x | 3.35 |
| lonlat | 90 | 4.9x | 3.07 |
| k1*OH*SO2 | 42.6 | 10.4x | 1.48 |
| tiers+T | 41.3 | 10.7x | 1.43 |
| oracle | 36.5 | 12.1x | 1.21 |

(`bucketed` with a hot cut of 10x the p99 production selected 0 cells, because the patch is 4% of all cells and the
p99 lies inside it, so it equals the production sort, 42.8 s.)

*Injection*: 100 ppmv in 8 adjacent cells (level 44, lat 29.7, lon 335-344, sunlit, OH 3e6 /cm3), background elsewhere.
One chunk holds the hot cells and costs 711 s of the 730 s for grid order and for the production sort; T sorting
spreads them over about three chunks (1968 s). 16-cell chunks for the 27 cells above the hot cut (the 8 plus 19
background cells) take 467 s (hot chunk 450 s): isolating the cells gives 1.6x on the hot chunk, so the cells' own
sub-step chain dominates and chunking cannot remove it. The `waste` column of this run is invalid (int32 overflow in
the script, fixed afterwards; times are unaffected). The same 100 ppmv cells at lon 180 (night, OH ~400 /cm3, fixed
per cell in this benchmark) needed ~2 sub-steps per step and the domain took 3 s for 1 h: where a plume sits relative to
the terminator decides its cost, so injection cost here depends on the snapshot's OH.

## JIT Stability Tips

1. **Always pass the same set of kwargs** to `make_step` step functions across calls. Changing which kwargs are present triggers recompilation.

2. **Use float masks instead of booleans** for enable/disable flags (e.g., `enable_organic=1.0` not `True`). This avoids JIT recompilation when flags change.

3. **Pass explicit zero arrays** for backgrounds instead of `None`. The dilution step requires explicit `Nk_bg`, `Mk_bg`, `Gc_bg` arrays.

4. **Avoid Python control flow** that depends on array values. Use `jnp.where()` and `jax.lax.cond()` instead.

## Memory Considerations

A single TOMAS state (40 bins, 44 components) is small:
- Nk: 40 floats = 320 bytes
- Mk: 40 x 44 = 1,760 floats = 14 KB
- Gc: 44 floats = 352 bytes
- xk: 41 floats = 328 bytes

The main memory consumer is the coagulation kernel matrix (40 x 40 = 1,600 entries). Even with 1000 vmap'd scenarios, total GPU memory is < 100 MB. Memory is not a constraint.

## Performance Optimization Opportunities

Current CPU performance baseline (single scenario, 24h at dt=60s, 1440 steps):

| Component | JAX (CPU) | Fortran | Ratio |
|-----------|-----------|---------|-------|
| Coagulation (Euler) | ~0.15s | ~0.27s | **0.57x** (JAX faster) |
| Condensation (PPM JIT) | ~0.26s | ~0.08s | 3.4x slower |
| Combined (Euler+PPM) | ~0.41s | ~0.33s | 1.27x slower |

### 1. GPU execution (highest impact)

The single biggest speedup. The N^2 coagulation kernel and vectorized condensation
map naturally to GPU parallelism. All JIT-compiled paths are GPU-ready after the
audit (centralized float64, no None defaults, no inline imports).

```bash
JAX_PLATFORMS=gpu python run_box_model.py --make-step
```

Use datacenter GPUs (A100, H100, MI250X) for float64 workloads. Consumer GPUs
throttle float64 to 1/32 of float32 throughput.

### 2. `jax.vmap` for ensemble/batch runs (high impact)

When running 50+ scenarios (Monte Carlo, parameter sweeps, LHC benchmarks),
`vmap` batches them into a single GPU kernel instead of sequential Python calls.
Expected 10–50x speedup for batch processing. See the [vmap section](#batch-processing-with-vmap) above.

### 3. Scan-fused time loops (moderate impact, already available)

The scan-fused functions (`run_full_scan`, `run_combined_scan_ppm`, etc.) compile
the entire 1440-step time loop into a single XLA program, eliminating all Python
dispatch overhead between timesteps. These are already implemented and should be
preferred over Python `while` loops calling `make_step` per timestep.

For custom process combinations, wrap `make_step` in `jax.lax.scan`:

```python
step_fn = make_step(['nucleation', 'coagulation', 'condensation'])

def scan_body(carry, _):
    Nk, Mk, Gc = carry
    Gc = Gc.at[SRTSO4].add(prod_rate * dt)
    Nk, Mk, Gc = step_fn(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, dt,
                           org_conc=org_conc, nh3_conc=nh3_conc, fion=fion)
    return (Nk, Mk, Gc), jnp.sum(Nk)

(Nk_f, Mk_f, Gc_f), N_history = jax.lax.scan(scan_body, (Nk, Mk, Gc), None, length=1440)
```

### Not recommended

- **Reducing MNFIX calls**: MNFIX enforces mass-number consistency after each process. Removing calls risks silent drift that compounds over 1440 steps, especially after nucleation creates particles far from bin centers. The cost of MNFIX is small compared to coagulation/condensation.
- **Mixed float32/float64 precision**: Aerosol microphysics requires float64 throughout. Condensation sink thresholds, mass conservation at 1e-15, and MNFIX bin boundary comparisons all depend on full precision. Float32 intermediates would introduce subtle mass errors that accumulate over 24h simulations and corrupt size distributions. The 2x memory savings is irrelevant (total state is <100 KB).

## Known Limitations

- **No multi-GPU**: TOMAS-JAX does not use `jax.pmap` or sharding. For multi-GPU, use `vmap` on a single GPU and `pmap` across GPUs for independent scenarios.
- **Compilation time**: First call to a JIT function compiles the XLA program (~10-60s depending on complexity). Subsequent calls are fast. Use `make_step` to compile once and reuse.
- **Diffrax on GPU**: The adaptive Tsit5 solver (`diffrax_step`) works on GPU but may be slower than forward Euler (`coag_euler_step`) due to variable-length stepping.
