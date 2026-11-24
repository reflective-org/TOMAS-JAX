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
confirmed on the GPU; a full-domain test (all 70 levels, 0.1 ppbv background, levels 43-45 at 1e2,
three orderings) is outlined in `PROGRESS.md` 2026-11-23. Details: `PROGRESS.md` 2026-11-23.

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
