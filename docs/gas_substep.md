# Adaptive gas sub-steps and reverse-mode AD

> `benchmarks/results/gas_substep/run_grid_parts.sh` is a local, git-ignored file.

Two opt-in `make_step` options (2026-11-05). With both at their defaults, `make_step` is bit-identical to before (Fortran parity).

- `gas_substep='adaptive'`: sub-steps SO₂ chemistry, nucleation and condensation, sized from the H₂SO₄ budget. One setting covers 10 pptv–100 ppmv SO₂ at 190–240 K.
- `ad_mode='reverse'`: every loop with a traced trip count gets a static length, so `jax.grad` and `jax.vjp` work.

## Why

With a 60 s step, `so2_oxidation_step` adds a whole step of H₂SO₄ before condensation can take any. At ≥ 1 ppmv with constant OH the nucleation gas clamp fires, N(>3 nm) is up to ×50 too high and every coefficient sensitivity is 0 ([nucleation_sensitivity.md](nucleation_sensitivity.md)). The workaround `run_map.dt_for` (60/10/2/0.5 s by SO₂) was tuned at 217 K, fails at 190 K / 100 ppmv, is 0.2–0.3 off in L1 number at ≥ 10 ppmv, and repeats coagulation every short step.

## The scheme

### Gas sub-steps (`solvers/gas_substep.py`)

At the start of each sub-step, for each candidate Δt:

| quantity | definition |
|---|---|
| C_seen(Δt) | C + ΔH₂SO₄ from chemistry over the whole candidate Δt, with no condensation or nucleation loss. Each candidate gets its own C_seen. Nucleation actually sees about C + ΔH₂SO₄(Δt/2) (see Split chemistry), so this is an upper bound |
| CS | condensation sink [s⁻¹] (`calc_condensation_sink`, with `sparse_density`) |
| k_nuc | (1 − org_frac)·J(C_seen)·m_nuc·V / C_seen, the loss frequency to nucleation. J is the rate of the scheme the step applies (`nucl_scheme`): Riccobono + Dunne, or the zhao2024 total over the enabled mechanisms with its inputs (`hno3`, `ulvoc`, `dma`, `hio3`, `enable_masks`); for zhao2024, (1 − org_frac)·m_nuc = 0.9·`_MNUC`, what `zhao2024_nucleation_step` removes |
| n_eff | d ln J / d ln C at C_seen, clipped to [1, 20], the effective order of J in H₂SO₄ |

Criterion: **(CS + n_eff·k_nuc)·Δt ≤ f**, f = `gas_substep_frac` (0.1). Candidates are Δt_rem/2^k, k = 0…23, where Δt_rem is the time left in the current coagulation interval (never past the end of the outer step). The ladder therefore moves with the position in the step: after a 7.5 s sub-step in a 60 s interval the candidates are 52.5, 26.25, … s. Candidates are floored at `gas_min_dt` (1e-3 s), and `gas_min_dt` itself is a 25th rung, so it is reachable even when Δt_rem/2^23 is larger (coagulation intervals longer than 2^23·`gas_min_dt` ≈ 8389 s; until 2026-11-15 such intervals floored at Δt_rem/2^23 instead); the largest passing one is taken, re-chosen every sub-step, under `stop_gradient`.

- **The clamp cannot fire:** n_eff ≥ 1, so k_nuc·Δt ≤ f < 1 (`info['knuc_dt_max']`).
- **zhao2024 mechanisms that do not depend on H₂SO₄** (iodine, pure organic): `zhao2024_nucleation_step` still takes 0.9·J·m_nuc of H₂SO₄ per cluster, so these mechanisms are an H₂SO₄ sink with n_eff = 1. Without enough chemistry to resupply the gas (e.g. hio3 = 1e8 cm⁻³, no SO₂), C goes to zero, k_nuc grows without bound and the sub-steps reach `gas_min_dt` (`n_floor` > 0, `knuc_dt_max` > f): the gas runs out, which is what that step's chemistry implies. Until 2026-11-13 the criterion always used Riccobono + Dunne, so with zhao2024 it ignored those mechanisms and the amine mechanism (one 60 s step at k_nuc·Δt = 680, reported as 5e-6).
- **CS alone is not enough:** the failure happens where CS is small, before the burst, where f/CS ≥ 60 s.
- **Low SO₂ costs nothing:** at ≤ 10 ppbv there is one sub-step per outer step.

### Split chemistry

If `so2_chemistry` precedes `nucleation`, each sub-step runs chemistry Δt/2 → nucleation → chemistry Δt/2 → condensation. With all chemistry first, nucleation sees the top of the H₂SO₄ sawtooth and J ∝ Cⁿ is n·CS·Δt/2 high (11% at 240 K / 10 ppmv, f = 0.1; 1.7–2.7 nm bins 4–8% off). The split makes the bias O((CS·Δt)²) at no measurable cost. The controller still uses the whole sub-step's chemistry, an upper bound.

The same split is available at fixed dt: `make_step(..., split_chemistry=True)` (needs `so2_chemistry` before `nucleation`, else `ValueError`; default `False` is bit-identical). The 80-bin references use it. Without it a fixed-dt run is first order in dt through the H₂SO₄ nucleation sees: at 13 km / 1 ppmv, 6 h, unsplit 0.5 s was 0.022 off (L1 number) a split 0.125 s run and nucleated 2.3% more, while split 0.5 s was 0.0014 off and split 0.25 s 0.0005 (2026-11-11).

### Coagulation intervals

Coagulation and dilution run on intervals, not on the outer 60 s step (that was 1.8× off in N(>3 nm) at 217 K / 100 ppmv) or every gas sub-step (too expensive).

- An interval ends at its target length, or when nucleation has added `coag_interval_frac` (0.01)·N. At 0.1 the 1.7 nm bin was ~7% low (60 s of coagulation on seconds-old clusters); at 0.01 it is within 0.6%, at 1.5–2× the cost.
- Every interval takes at least one gas sub-step before the end test (a do-while, in both modes). An interval can start closed: with N = 0 and no coagulation in `processes` to add TINY values back, dN < `coag_interval_frac`·N is 0 < 0. A loop that tested first then took no sub-step and never advanced t. `coag_interval_frac` must be > 0 (`ValueError`).
- If coagulation removed a fraction r > `coag_interval_frac` of N, the next target shrinks to t_int·frac/r, rounded down to a power of 2 s; otherwise it may double. Intervals never cross the end of the outer step, and the target resets to the outer dt at the start of every outer step, so the first interval of each step always tries the full dt. Rounding keeps the lengths discrete, so gradients do not see a continuously moving step.

### Caps, diagnostics, usage

- `gas_max_substeps` = 16384 per outer step; the last allowed sub-step takes the rest of dt. **The cap counts sub-steps per outer step, so the number needed grows with the outer dt:** the sub-steps are set by the H₂SO₄ relaxation time (~0.01 s at 100 ppmv), not by dt. 190 K / 100 ppmv needs ~5700 per 60 s; 217 K / 18200 Pa / 100 ppmv needs ~4,500 per 60 s and ~41,000 per 600 s. Scale the cap with dt (or keep dt at 60 s) for long outer steps.
- **A binding cap gives NaN or a wrong answer.** The forced last sub-step is far past the criterion (217 K / 18200 Pa / 100 ppmv, dt = 600 s: 369 s in one sub-step, k_nuc·Δt = 5e10), the nucleation clamp fires, and the burst it makes sends the fixed-Euler coagulation to NaN (without coagulation N was 5.2e9 #/cm³ against 1.25e8 uncapped). At 1 ppmv with cap 64, N was 7.58e6 against 7.83e6. There is no fallback: any finite answer after the cap binds is wrong, so the step warns instead (`GasSubstepCapWarning`, a `RuntimeWarning`, from a `jax.debug.callback` after every adaptive step; under `vmap` it counts the batch members that hit the cap). It fires whether or not `return_substep_info` is on. Silence it with `warnings.filterwarnings('ignore', category=GasSubstepCapWarning)` only if you check `info['hit_cap']` yourself.
- `return_substep_info=True` returns `(Nk, Mk, Gc, info)` with `n_sub`, `n_coag`, `dt_min`, `hit_cap` (the cap forced a sub-step longer than the criterion's choice), `dt_capped` (that sub-step's length [s], 0 if none), `n_floor` (sub-steps at `gas_min_dt`), `knuc_dt_max` (largest k_nuc·Δt over the sub-steps actually taken, the forced one included; until 2026-11-13 the forced step was missed and a binding cap still reported ≤ 0.1).
- Under `vmap` the slowest member sets the pace.

```python
step = make_step(['so2_chemistry', 'nucleation', 'coagulation', 'condensation'],
                 gas_substep='adaptive', sparse_density=..., water_scheme=...)
Nk, Mk, Gc = step(Nk, Mk, Gc, xk, temp, pres, boxvol, rh, alpha, 60.0,
                  oh_conc=1e6, fion=10.0, nucl_params=params)
```

`common.make_runner(..., gas_substep='adaptive')` and `run_map --gas-substep adaptive` (→ `results/phase2_gas`) pass it through.

## Validation

Grid: T 190–240 K × SO₂ 10 pptv–100 ppmv, study settings, 24 h. Per cell: adaptive f = 0.3, 0.1, 0.03 (60 s outer step), `dt_for`, and a fixed-dt reference. Each run is a forward pass plus a `jvp` along the H₂SO₄-calibration direction. At 6 h and 24 h, f = 0.1 is compared with the reference by L1 distances over the size distribution:

| metric | definition |
|---|---|
| number | Σ\|N_k − N_k,ref\| / Σ N_k,ref |
| dry mass | Σ\|M_k − M_k,ref\| / Σ M_k,ref |
| sensitivity | Σ\|dN_k − dN_k,ref\| / Σ N_k,ref (dN_k: the tangent) |

Pass: each ≤ 0.05, no clamp, no cap (the 80-bin check tests number and dry mass only, below). Failures exit non-zero.

### 80 bins (`benchmarks/python/gas_substep_psd80.py`), the test of record

- **Grid:** altitude × SO₂, T and P together from the AFGL tropical atmosphere (`common.TROPICAL_AFGL`: 13, 15, 17, 20, 23, 25 km; 217/203.7/194.8/206.7/217/221.4 K, 182/132/94/56/35/26 hPa). Until 2026-11-10 it was T × SO₂ at 5500 Pa.
- **Reference** (`gas_substep_psd80.reference(so2)`): fixed dt with split chemistry (`split_chemistry=True`), 0.25 s up to 10 ppbv; at 100 ppbv, 1, 10 and 100 ppmv the halved run (`ref_split_half`: 0.125, 0.25, 0.05, 0.01 s). The split 0.25/0.5/0.1/0.02 s run (`ref_split`) is within 0.003 in L1 number of the halved one (24 h). The old unsplit reference (`ref`, same steps as `ref_split`) was up to 0.05 off it (13 km / 1 ppmv, 24 h), which is why f = 0.1 failed L1 number there. 1 s was 0.17 off a 0.125 s run at 208 K / 0.1 ppbv (L1 sensitivity).
- **Hour means:** each run records Nk, dry Mk and dNk averaged over the 60 minutes before each report hour (fixed dt must divide 60 s). Snapshots catch bins 0–1 trading particles minute to minute: at 217 K / 10 ppmv, f = 0.1, snapshot L1 number reached 0.082, the hour mean ≤ 0.014.
- **Bin pairs merged** (`MERGE` = 2, the 40-bin grid's bins). At 200 K and low SO₂, new particles sit in bins 0–2 for hours, and when MNFIX moves them up one 80-bin bin depends on dt in no order. Fixed 0.125 s and 0.25 s have the same total N but are 0.160 apart in L1 sensitivity per 80-bin bin, 0.010 merged; no smaller dt_max fixes it (ladder 0.125–60 s, 2026-11-07). Per-80-bin distances are printed, not tested. Merging to 20 bins would hide real differences.
- **Sensitivity not tested (2026-11-12):** the L1 sensitivity is printed and plotted, not tested. At 15 km / 100 ppbv, 6 h, a central FD (±0.05) put every tangent, the references and f = 0.1 alike, 0.16–0.28 off in merged bins 1–2 (~2–4 nm), where the one-sided differences have opposite signs (−0.23, +0.50): MNFIX bin selection, which the tangent does not see. The FDs of split and unsplit chemistry agreed to 0.004 and the total sensitivity agreed (FD 0.548, jvp 0.553); the split tangent moved further off as dt shrank (0.24 at 0.25 s, 0.28 at 0.125 s), so f = 0.1 was 0.12 from the halved reference there. The adaptive tangent also blows up at 13 km / 10 ppmv, 24 h (3.6; the references agree to 0.1).
- **Invalid tangents:** where the reference's own Σ|δN|/ΣN > `INVALID_TANGENT` (1e3), a tangent instability has blown up every run of the cell (13 km / 100 ppmv at 24 h on the altitude grid). Valid cells are ≤ 2.5. On 80 bins this happens at 223 K and 240 K / 100 ppmv after ~12 h; the source is the linear sub-bin coagulation scheme's derivative, not PPM ([prescribed_j.md](prescribed_j.md#frozen-shape-tangent-ppm_tangentfrozen_shape)), so `ppm_tangent='frozen_shape'` does not help.
- **Finite differences in those cells** (`gas_substep_psd80.FD_CELLS`): every kind also runs forward only at α = ±`FD_H` (0.05) along the direction (runs `<kind>_fd+`, `<kind>_fd-`; 28 runs), and `load_run` replaces the jvp tangents by the central difference; the check, `plot_gas_substep_grid` (boxed points) and `plot_gas_substep_psd` read it. Coarser outputs do not avoid the blow-up: merging 2–80 bins, total N and NMASS N(>3 nm) are as far off as the per-bin tangent (2026-11-10). Over the first 2 h at 240 K, FD is within 0.4% of the jvp; `--check` prints FD against the jvp at every hour the jvp stays valid.
- **Result (2026-11-12, altitude grid, split reference):** f = 0.1 passes L1 number and dry mass in every cell at 6 h and 24 h (largest number 0.012, dry mass 0.002). Constant dt = 60 s goes NaN at 100 ppmv at 13, 15 and 17 km (fixed-step coagulation).
- **Result (2026-11-07, T grid at 5500 Pa, unsplit reference):** f = 0.1 passes every valid cell (largest: number 0.014, dry mass 0.004, sensitivity 0.018). The only failures are the invalid tangents at 223 K and 240 K / 100 ppmv, 24 h. f = 0.03 hits the cap at 190 K / 100 ppmv; `dt_for` is up to 0.31 off.
- **Cost** (24 h, mean over T, compile excluded):

| SO₂ | reference | f = 0.1 | `dt_for` |
|---|---|---|---|
| 0.01 ppbv | 1866 s | 9 s | 8 s |
| 1 ppmv | 1090 s | 241 s | 270 s |
| 100 ppmv | 25481 s | 4347 s | 1063 s |

- **Running:** `run_awake -m benchmarks.python.gas_substep_psd80 --all` (missing runs only, `NJOBS` at a time, default 11; the split references alone are ~190 CPU-h, ~15 h on 12 jobs, set by the six 100 ppmv runs at 0.01 s), then `--check`. Figures: `plot_gas_substep_grid --bins 80`, `plot_gas_substep_psd`.

### 40 bins (`benchmarks/python/gas_substep_convergence.py`)

Snapshots only; reference `dt_for`/4, 0.1 s at 10 ppmv, 0.02 s at 100 ppmv. Run in parallel with `benchmarks/results/gas_substep/run_grid_parts.sh` (~1.5 h on 11 cores; `--merge` checks completeness and judges). f = 0.1 passes 140 of 144 checks; the 4 failures (sensitivity 0.068–0.096 at 190/200 K / 0.01 ppbv, 208 K / 0.1 ppbv, 240 K / 10 ppmv) are MNFIX bin-shift noise: fixed-dt runs scatter by 0.01–0.07 among themselves in no order with dt, and all four pass at 80 bins. Before the chemistry split, f = 0.1 failed 13 checks (`results/gas_substep/chem_before_nucl/`).

### Tests

`tests/test_gas_substep.py`: argument validation; one sub-step at 10 ppbv; no clamp, no cap and closed sulfur at 100 ppmv (190, 217 K); a binding cap warns, sets `hit_cap`/`dt_capped` and reports k_nuc·Δt > 1 (forward, reverse, `vmap`); 1 h at 1 ppmv vs fixed 0.1 s (5%); `jvp` vs FD; `vmap`; zhao2024: k_nuc equals the H₂SO₄ `zhao2024_nucleation_step` removes (amine + iodine), and an iodine-dominated run at 1 ppmv stays within the criterion.

## Reverse-mode AD (`ad_mode='reverse'`)

`core/bounded_loop.py` replaces each traced-trip-count loop with a static-length `lax.scan` over `lax.cond(active, body, identity)`, optionally checkpointed:

| loop | bound |
|---|---|
| nucleation sub-steps | `max_nuc_substeps` (exact) |
| PPM CFL sub-steps | `ppm_max_substeps` = 64 (binding → Courant > 0.8) |
| `euler_adaptive` coagulation | `coag_max_substeps` = 1024, forward 10000 (binding → the rest of dt gets no coagulation; `CoagSubstepCapWarning`) |
| gas sub-steps | `gas_max_substeps`, one flat loop; intervals closed by `lax.cond` (nested bounded loops cost max_outer × max_inner) |

- **Threading:** a static `bounded` flag on `euler_step`, `ppm_condensation_step`, `ezcond_ppm_jax`, `ezcond_tfl_jax`, `_condensation_step_core`; `make_step(ad_mode='reverse')` sets them all. `ezcond_tfl_jax` accepts it only for the shared `ezcond_fn` interface. MNFIX needs no `bounded` flag: by default phase 3 is a `fori_loop` over every bin (static trip count), so `jax.grad` goes through it in every caller, including `diffrax_step` and `coag_euler_step`. `make_step(mnfix_trim=None)` switches to the faster `mnfix_jax(trim=True)` while_loop only with `ad_mode='forward'`.
- **Values** match forward mode to ≤ 1e-11 (XLA fuses differently inside `cond`). **Gradients** match `jacfwd` to ≤ 5e-12 (1 ppmv with and without gas sub-steps, 100 ppmv with `euler_adaptive`) and central FD (h = 1e-4) to 1e-3.
- **Memory:** checkpoint the caller's time loop (`common.make_runner(ad_mode='reverse')` does).
- **`coag_max_substeps` binds sooner in reverse mode** (1024 against 10000). `euler_step` stops at the cap and drops the rest of dt, so N comes out too high: 3e10 cm⁻³ in bin 0 at 280 K, dt = 60 s gave N = 2.58e7 in reverse mode against 1.76e7 in forward mode. `make_step` warns (`CoagSubstepCapWarning`, a `RuntimeWarning`, from a `jax.debug.callback` after the step, in both modes; under `vmap` it counts the batch members that hit the cap); with gas sub-steps `info['coag_dt_left']` is the longest time dropped in any coagulation interval. Raise `coag_max_substeps` or shorten dt.
- **`vmap`:** `cond` runs both branches, so every iteration up to the cap costs a full one, whether it is needed or not. With `gas_substep='adaptive'` that is `gas_max_substeps` (16384) gas sub-steps plus a coagulation interval each: about 61 s per step against 9 ms in forward mode. Do not combine `ad_mode='reverse'`, `gas_substep='adaptive'` and `vmap`; loop over the members, or use forward mode (`jacfwd`) for batched sensitivities.
- **When worth it:** few outputs, many more inputs than the 25 `NucleationParams` (e.g. initial-state sensitivities). For 25 inputs `jacfwd` is ~7× cheaper: d ln N(>3 nm) at 24 h, 217 K, 1 ppmv, 40 bins took 34 s / 0.67 GB with `jacfwd` and 238 s / 1.26 GB with `grad` (`benchmarks/python/reverse_mode_cost.py`), agreeing to 2.4e-12.

## coag_interval_frac ladder (benchmark)

`benchmarks/python/gas_substep_coag_interval.py`: `coag_solver='euler_adaptive'` with adaptive gas sub-steps (f = 0.1) at outer dt = 360 s, `coag_interval_frac` = 0.003, 0.03, 0.1, 0.3, 0.5 on the tropical grid (13–25 km × SO₂, 40 bins, 24 h). The 0.01 default is the existing `ea_f0.1_360` run (not recomputed; its wall time is from a single-job machine load, the new rungs ran 3 jobs at once, so time ratios are rough and, if biased, understate the savings of coarser rungs). Reference: the smallest-dt split-chemistry run of each cell, `ref_split_half` (SO₂ ≥ 100 ppbv) else `ref_split`. Valid: L1 N ≤ 0.05 at 6 h and 24 h, finite, no gas or coagulation cap. `--all`, then `--compare`; output `benchmarks/results/gas_substep/coag_interval_tropical/` (table in `compare.txt`). `build(coag_interval_frac=...)` in `gas_substep_convergence.py` passes it to `make_step`.

Results (48 cells per rung):

| frac | valid | wall time vs 0.01 | L1 N at SO₂ ≥ 100 ppbv |
|---|---|---|---|
| 0.003 | 47 | ~3× slower | 0.002–0.004 |
| 0.01 | 47 | 1 | ~0.006–0.008 |
| 0.03 | 48 | 1.2–1.5× faster | 0.02–0.03 |
| 0.1 | 2 (SO₂ ≤ 10 ppbv) | 1.3–1.7× faster | ~0.08 |
| 0.3, 0.5 | 2 | 1.3–1.8× faster | 0.12–0.25 |

- **0.03 is the coarsest rung that passes everywhere.** From 0.1 up the L1 N exceeds the tolerance at SO₂ ≥ 100 ppbv (accuracy failures, no cap) while time barely drops further (100,000 ppbv: 544, 538, 538 s summed over altitudes at 0.1, 0.3, 0.5). The error stops growing near 0.3–0.5, and the sub-steps inside each interval, not the interval count, carry most of the cost.
- **0.003 costs ~3× the default** (more intervals, more `euler_adaptive` and gas sub-steps) for an L1 already below 0.01 at 0.01.
- **The two invalid cells below 0.1** (`ci0.003` and `ci0.01`, 15 km / 100,000 ppbv) hit `CoagSubstepCapWarning` at `coag_max_substeps` = 10000, with L1 N ≤ 0.01. Rerun at `COAG_MAX_SUBSTEPS` = 100000 (the 0.01 file in `euler_adaptive_tropical/` was overwritten; the capped runs are in `bak_capped/`): the warning still fires, L1 N is unchanged (0.005 and 0.010), and wall time rises to 382 s and 1717 s (0.03: 175 s). Both cells stay flagged and their times are cap-dominated. Diagnosis below.

### Why `ci0.01` hits the coagulation cap and `ci0.03` does not (15 km, 100,000 ppbv)

`benchmarks/python/diag_coag_cap.py FRAC HOURS CAP` wraps `euler_step` and prints every coagulation interval that needed ≥ 1000 sub-steps, with the limiting bin and species of `_compute_adaptive_dt` at the start and end state (24 h, cap 10000; the first 6 h of `ci0.01` have none).

- `ci0.01`: 758 intervals needed ≥ 1000 sub-steps (the first eight printed are all at the cap); `ci0.03`: none.
- The capped intervals are the shrunk intervals of the burst, 0.03–0.07 s long. The limiter is the **number limit** `0.25 N/|dN/dt|` of **one bin (bin 6)** every time; the mass limit stays large (~48 s at the end state).
- Bin 6 enters each of them nearly empty (N ≈ 0.02–0.09 #/cell) with a source of ~2×10⁴ #/cell/s that does not depend on N, and ends at N ≈ 10⁻⁵ (MNFIX's floor), where the limit is ~10⁻¹⁰ s.
- Probable mechanism (not tested): each sub-step grows bin 6 by ≤ 25%; the new particles' mean mass lies outside the bin, so MNFIX moves them out and resets the bin to its 10⁻⁵ floor; the limit collapses again and the cycle runs to the cap. The mask in `_compute_adaptive_dt` excludes bins below `TINY_N` (10⁻¹⁰), but MNFIX leaves 10⁻⁵, so a floored bin that is not empty sets the step.
- Effect: each capped interval drops essentially its whole 0.03–0.07 s of coagulation; 758 of them are well under a minute of 24 h, which is why L1 N stays ≈ 0.01 (0.03's error of 0.029 is not from this).
- Why 0.03 avoids it is not established: its intervals (rounded down to powers of 2 s) apparently never put bin 6 in that state, and the time of the first capped interval was not traced for either rung. 0.003 is also capped, so the cap is not monotonic in `coag_interval_frac`.
- Masking floored bins in `_compute_adaptive_dt` would diverge from Fortran `multicoag.f`, so it is opt-in: `coag_n_floor` (next section).

### `coag_n_floor`: opt-in floor on the euler_adaptive number limit

`make_step(coag_solver='euler_adaptive', coag_n_floor=x)` (default `None`) passes `n_floor=x` to `euler_step`, which leaves bins with N ≤ x [#/cell] out of the number limit `0.25 N/|dN/dt|` (the mask is N > `TINY_N` = 1e-10 otherwise). The mass limit is unchanged. It needs `euler_adaptive` and x ≥ 0 (`ValueError`); `0.0` equals `None`.

Checks:
- **Off is bit-identical.** A 3-step `make_step` run (15 and 25 km × 100 and 10000 ppbv, with and without adaptive gas sub-steps) against an archived HEAD copy: 24 arrays, all equal. `tests/test_coag_adaptive.py`.
- **Mass and number.** Coagulation only (`euler_step`, `recompute_kernel=True`, 217 K, 10 × 360 s, 1e7 and 3e9 cm⁻³ 1.7 nm burst on a background): dry mass closes to ≤ 1e-9 of M₀ with the floor off, 1e-3 and 1e-2 (overflow estimated from the rate at each step start; the residual of 5e-10 is that estimate), no species moves more than the total, total mass differs between floor and no floor by ≤ 4e-11. Sub-steps fall 1.4–1.9× at 1e7 and 13× (12902 to ≈ 900) at 3e9. Total N differs by 0.4–1.3 % and bins with N > 1e-3 #/cell by up to 0.4 (1e7) and 0.9 (3e9): the bin-shift noise of thin bins, no truth reference in this test.
- **Grid** (`gas_substep_coag_interval.py`, kinds `nf0.01`, `nf0.001`: `coag_interval_frac` 0.01 with floor 1e-2 / 1e-3; same 48 cells, reference and criteria as the ladder): both pass 48/48 (the ladder's 0.01 passed 47, the capped cell excepted). Sub-step counts equal the 0.01 run in 46 of 48 cells (15 km / 100000 ppbv changes, 17 km / 1000 ppbv by 5 of 292999) and the max L1 N over the 48 cells is 0.047 for both floors and for 0.01.
- **15 km / 100000 ppbv** (the cell that hit `CoagSubstepCapWarning`): no cap, L1 N 0.010 / 0.009 (6 h / 24 h; 0.01 gave 0.010 / 0.010 with the cap), 24 h in 206 s against 377 s at `coag_max_substeps` = 10000 (the 100000-cap rerun took 1717 s and still warned). Plain 0.03 takes 175 s with L1 N 0.029 / 0.028. The floor therefore keeps the 0.01 accuracy at 1.8× less time; 0.03 is a further 15 % faster for three times the error.
- Elsewhere the floor does nothing: the wall times at the long runs are 0.92–0.94 of the 0.01 run's (the short ones are noise), but sub-step counts are equal, so that is load (3 jobs at once against one for the 0.01 run), not the floor.
- Floors 1e-2 and 1e-3 give the same result at every cell; 1e-2 is the suggested value. Bins that cause the collapse sit at 0.02–0.09 #/cell, far above MNFIX's 1e-5, so a floor of 2e-5 would not mask them. Not validated on 80 bins; stays opt-in.
- Mechanism: confirmed only by this effect (cap gone, same L1), not traced inside a run.

### Ideas that did not speed up the 0.01 default (workflow of 2026-11-20)

Four candidates were run on a 10-cell quick set (13/20/25 km × 10–1000 ppbv, 13 km at 10000 ppbv; `benchmarks/python/coag_speed_eval.py`) plus 15 km / 100000 ppbv, one reviewed independently. Wall times are ±10 % (shared machine).
- **Decoupling the two interval closers** (`coag_loss_frac`, `coag_target_growth`, not merged): the accuracy loss at `coag_interval_frac` ≥ 0.03 comes from the nucleation trigger, not the shrink rule. Raising the shrink threshold alone gave 1.06× over the 10 cells (max L1 N 0.035); slower target growth changed nothing; a tighter nucleation trigger (0.005) cut L1 N 2.5× at 1.3× the time. The 13 km / 10000 ppbv cell (65 % of the time) did not change.
- **Strang-style coagulation on the mid-interval state** (`coag_split='mid'`, not merged): second order in the interval, mid 0.03 has max L1 N 0.030 against 0.044 for plain 0.03 at about 1.3× over 0.01 (mid 0.05: 0.038, 1.3×; mid 0.3 fails). Review found no bug, but phase 2 has no nucleation-based closure, there are no tests, and it is forward-mode only.
- **Cost reduction at identical numerics** (not merged): the time per gas sub-step (31–46 µs at 13 km / 1000 ppbv) is thunk dispatch and control flow (nested `while`/`cond`), not arithmetic: coagulation 35–40 %, PPM condensation about 35 %. Dropping the MNFIX that follows nucleation at the start of condensation was 1.01–1.09× (bit-identical); a kernel computed inside the loop was 1e-12 off and gave nothing. Gains must come from fewer sub-steps, not faster ones.
- **Interval count is not the cost:** at 13 km / 1000–10000 ppbv there is about one `euler_adaptive` sub-step per coagulation interval and 0.6 intervals per gas sub-step, so no coagulation setting removes more than the 1.2–1.5× that 0.03 gives. The semi-implicit gas step under Limitations is the remaining way to cut the gas sub-steps.


## Gradient blind spots

In both modes: MNFIX bin selection, nucleation sub-step counts, the clamp (cannot fire with gas sub-steps), the gas sub-step choice. The Δt_rem/2^k ladder makes the objective piecewise smooth, so central FD with h ≳ 1e-3 can straddle a switch (1 ppmv: FD 0.056–0.061 vs AD 0.0563); at h = 1e-4 FD equals AD.

## Limitations

- **Stiffness at cold, high SO₂:** ~5700 sub-steps per 60 s at 190 K / 100 ppmv. A semi-implicit gas step (solve C* = C₀ + PΔt − Δt·(CS·C* + L_nuc(C*)) by bisection, nucleate J(C*)·Δt, condense the rest) would remove it; not implemented.
- **Coagulation intervals react after the fact:** no rejection; the nucleation trigger ends an interval after 0.01·N new particles.
- **Order:** gas processes run before coagulation in each interval, whatever the process list says.
- **Bin-shift noise:** MNFIX timing of slowly growing small bins makes per-bin distances noisy at 0.02–0.1 on 40 bins and at 0.16 per bin on 80 bins, independent of dt. Hence hour means and 40-bin resolution for the test.
