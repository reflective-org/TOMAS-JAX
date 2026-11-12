# Adaptive gas sub-steps and reverse-mode AD

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
| k_nuc | (1 − org_frac)·J(C_seen)·m_nuc·V / C_seen, the loss frequency to nucleation |
| n_eff | d ln J / d ln C at C_seen, clipped to [1, 20], the effective order of J in H₂SO₄ |

Criterion: **(CS + n_eff·k_nuc)·Δt ≤ f**, f = `gas_substep_frac` (0.1). Candidates are Δt_rem/2^k, k = 0…23, where Δt_rem is the time left in the current coagulation interval (never past the end of the outer step). The ladder therefore moves with the position in the step: after a 7.5 s sub-step in a 60 s interval the candidates are 52.5, 26.25, … s. Candidates are floored at `gas_min_dt` (1e-3 s); the largest passing one is taken, re-chosen every sub-step, under `stop_gradient`.

- **The clamp cannot fire:** n_eff ≥ 1, so k_nuc·Δt ≤ f < 1 (`info['knuc_dt_max']`).
- **CS alone is not enough:** the failure happens where CS is small, before the burst, where f/CS ≥ 60 s.
- **Low SO₂ costs nothing:** at ≤ 10 ppbv there is one sub-step per outer step.

### Split chemistry

If `so2_chemistry` precedes `nucleation`, each sub-step runs chemistry Δt/2 → nucleation → chemistry Δt/2 → condensation. With all chemistry first, nucleation sees the top of the H₂SO₄ sawtooth and J ∝ Cⁿ is n·CS·Δt/2 high (11% at 240 K / 10 ppmv, f = 0.1; 1.7–2.7 nm bins 4–8% off). The split makes the bias O((CS·Δt)²) at no measurable cost. The controller still uses the whole sub-step's chemistry, an upper bound.

The same split is available at fixed dt: `make_step(..., split_chemistry=True)` (needs `so2_chemistry` before `nucleation`, else `ValueError`; default `False` is bit-identical). The 80-bin references use it. Without it a fixed-dt run is first order in dt through the H₂SO₄ nucleation sees: at 13 km / 1 ppmv, 6 h, unsplit 0.5 s was 0.022 off (L1 number) a split 0.125 s run and nucleated 2.3% more, while split 0.5 s was 0.0014 off and split 0.25 s 0.0005 (2026-11-11).

### Coagulation intervals

Coagulation and dilution run on intervals, not on the outer 60 s step (that was 1.8× off in N(>3 nm) at 217 K / 100 ppmv) or every gas sub-step (too expensive).

- An interval ends at its target length, or when nucleation has added `coag_interval_frac` (0.01)·N. At 0.1 the 1.7 nm bin was ~7% low (60 s of coagulation on seconds-old clusters); at 0.01 it is within 0.6%, at 1.5–2× the cost.
- If coagulation removed a fraction r > `coag_interval_frac` of N, the next target shrinks to t_int·frac/r, rounded down to a power of 2 s; otherwise it may double. Intervals never cross the end of the outer step, and the target resets to the outer dt at the start of every outer step, so the first interval of each step always tries the full dt. Rounding keeps the lengths discrete, so gradients do not see a continuously moving step.

### Caps, diagnostics, usage

- `gas_max_substeps` = 16384 per outer step; the last allowed sub-step takes the rest of dt (`info['hit_cap']`). 190 K / 100 ppmv needs ~5700 at f = 0.1 (H₂SO₄ relaxes in ~0.01 s).
- `return_substep_info=True` returns `(Nk, Mk, Gc, info)` with `n_sub`, `n_coag`, `dt_min`, `hit_cap`, `n_floor` (sub-steps at `gas_min_dt`), `knuc_dt_max`.
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

`tests/test_gas_substep.py`: argument validation; one sub-step at 10 ppbv; no clamp, no cap and closed sulfur at 100 ppmv (190, 217 K); 1 h at 1 ppmv vs fixed 0.1 s (5%); `jvp` vs FD; `vmap`.

## Reverse-mode AD (`ad_mode='reverse'`)

`core/bounded_loop.py` replaces each traced-trip-count loop with a static-length `lax.scan` over `lax.cond(active, body, identity)`, optionally checkpointed:

| loop | bound |
|---|---|
| MNFIX phase 3 | nbins (exact) |
| nucleation sub-steps | `max_nuc_substeps` (exact) |
| PPM CFL sub-steps | `ppm_max_substeps` = 64 (binding → Courant > 0.8) |
| `euler_adaptive` coagulation | `coag_max_substeps` = 1024 (binding → truncated coagulation) |
| gas sub-steps | `gas_max_substeps`, one flat loop; intervals closed by `lax.cond` (nested bounded loops cost max_outer × max_inner) |

- **Threading:** a static `bounded` flag on `mnfix_jax`, `coag_euler_step`, `euler_step`, `ppm_condensation_step`, `ezcond_ppm_jax`, `ezcond_tfl_jax`, `_condensation_step_core`; `make_step(ad_mode='reverse')` sets them all.
- **Values** match forward mode to ≤ 1e-11 (XLA fuses differently inside `cond`). **Gradients** match `jacfwd` to ≤ 5e-12 (1 ppmv with and without gas sub-steps, 100 ppmv with `euler_adaptive`) and central FD (h = 1e-4) to 1e-3.
- **Memory:** checkpoint the caller's time loop (`common.make_runner(ad_mode='reverse')` does).
- **`vmap`:** `cond` runs both branches, so skipped iterations cost full ones.
- **When worth it:** few outputs, many more inputs than the 25 `NucleationParams` (e.g. initial-state sensitivities). For 25 inputs `jacfwd` is ~7× cheaper: d ln N(>3 nm) at 24 h, 217 K, 1 ppmv, 40 bins took 34 s / 0.67 GB with `jacfwd` and 238 s / 1.26 GB with `grad` (`benchmarks/python/reverse_mode_cost.py`), agreeing to 2.4e-12.

## Gradient blind spots

In both modes: MNFIX bin selection, nucleation sub-step counts, the clamp (cannot fire with gas sub-steps), the gas sub-step choice. The Δt_rem/2^k ladder makes the objective piecewise smooth, so central FD with h ≳ 1e-3 can straddle a switch (1 ppmv: FD 0.056–0.061 vs AD 0.0563); at h = 1e-4 FD equals AD.

## Limitations

- **Stiffness at cold, high SO₂:** ~5700 sub-steps per 60 s at 190 K / 100 ppmv. A semi-implicit gas step (solve C* = C₀ + PΔt − Δt·(CS·C* + L_nuc(C*)) by bisection, nucleate J(C*)·Δt, condense the rest) would remove it; not implemented.
- **Coagulation intervals react after the fact:** no rejection; the nucleation trigger ends an interval after 0.01·N new particles.
- **Order:** gas processes run before coagulation in each interval, whatever the process list says.
- **Bin-shift noise:** MNFIX timing of slowly growing small bins makes per-bin distances noisy at 0.02–0.1 on 40 bins and at 0.16 per bin on 80 bins, independent of dt. Hence hour means and 40-bin resolution for the test.
