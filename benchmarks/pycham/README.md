# TOMAS-JAX vs PyCHAM benchmark

An independent check of the AAAR tutorial box model (`tomas_jax.box`) against
[PyCHAM](https://github.com/simonom/PyCHAM) (O'Meara et al., 2021) with the same MCM
limonene mechanism. Differences are reported as they come out and explained. The model
is never tuned to PyCHAM.

The write-up for scientists, with every figure and the verdict per process, is
[`docs/aaar/pycham_comparison.md`](../../docs/aaar/pycham_comparison.md). This README
documents the harness.

## Regenerate everything

```bash
cd tomas-jax
.venv/bin/python -m benchmarks.pycham.run_all                  # all cases, ~35 min on a laptop
.venv/bin/python -m benchmarks.pycham.run_all --cases S1,S2    # selected cases
.venv/bin/python -m benchmarks.pycham.run_all --skip-pycham    # reuse raw PyCHAM output
.venv/bin/python -m benchmarks.pycham.run_all --analyse-only   # metrics + figures only (~20 s)
```

`run_all.py` runs, for each case in turn, `cases/<case>/run.py` (PyCHAM inputs, PyCHAM
runs, TOMAS-JAX runs, built-in checks that raise on failure, small results) and then
`cases/<case>/analyse.py` (metrics, `results/summary.json`, figures, copied to
`docs/aaar/figures/pycham/`). Each step runs in its own process group with a wall-clock
limit (`config.RUN_ALL_TIMEOUTS_S`); on a timeout the whole group, PyCHAM included, is
killed. Logs and a JSON report (exit status and wall time of every step, result sizes)
go to `pycham-bench/logs/run_all/`. The exit code is non-zero if any step failed. Cases
run one after the other: they have separate PyCHAM checkouts, but parallel runs distort
the runtimes the write-up reports.

## Cases

| Case | What it tests | Preset | Processes on | Case README |
|---|---|---|---|---|
| G2 | gas chemistry, constant light, 6 h | `level2_limonene_soa` | chemistry only | [cases/g2](cases/g2/README.md) |
| G3 | gas chemistry, diurnal photolysis, NO2 held, 24 h | `level3_day_night` | chemistry only | [cases/G3](cases/G3/README.md) |
| S1 | condensation/evaporation of one surrogate (c* = 1, 100), 12 h | `level1_volatility` | SOA partitioning only | [cases/S1](cases/S1/README.md) |
| S2 | limonene SOA: chemistry + 245 condensable species, 6 h | `level2_limonene_soa` | chemistry, H2SO4 condensation, SOA partitioning | [cases/S2](cases/S2/README.md) |
| T1 | toluene + NOx + O3 in sunlight, the merged limonene + toluene mechanism (2640 reactions), 6 h | `level1_urban` + 5 ppb toluene | chemistry only | [cases/T1](cases/T1/README.md) |

Not tested by any case (switched off although the presets use them): coagulation (on in
all three presets), nucleation (off in the presets as well), N2O5 uptake (Level 3,
γ = 0.02) and the Level 3 aerosol.

## Layout

| Path | Role |
|---|---|
| `config.py` | Single source of truth: paths, pinned PyCHAM SHAs, timeouts, thresholds, colours, display density, and the settings of every case (`G2`, `G3`, `S1`, `S2`). |
| `run_all.py` | Regenerates every case and figure (above). |
| `ours.py` | `run_ours(scenario)` runs TOMAS-JAX through `driver.run(observer=...)` and records every gas species [ppb], J per interval, SA/NA_HET counters, size distributions and particle mass per component, c* class and condensable species. `save_npz`/`load_npz` write float32, decimated files. `numerics_override(...)` patches solver tolerances or water vapour for attribution runs only. |
| `jvalues.py` | `j_per_interval(scenario)` gives our J exactly as the driver freezes it (interval midpoint). `to_pycham_rows` maps them to MCM J numbers 0..61, and `write_j_files` writes `J.csv` + PyCHAM's `J.xlsx`. |
| `write_j_xlsx.py` | CSV → `J.xlsx` (sheet "J"). Runs in the PyCHAM venv because openpyxl is not a TOMAS-JAX dependency. |
| `pycham_inputs.py` | `write_eqn` (the .eqn with rate overrides), `write_xml` (PyCHAM's MCM xml + D9/D9PROD), `write_constants`, `scenario_model_var` (also `hold_constant` → `const_comp`) / `write_model_var`, `ensure_pycham_root(case, kelvin_off=False)` (per-case `cp -R` of the checkout; `kelvin_off` makes the `surfT = 0` attribution copy), `run_pycham` (subprocess with timeout; raises on failure). |
| `particles.py` | Shared particle set-up: TOMAS-JAX water per unit mass of sulfate and organics (`water_mass_ratios`), the PyCHAM water dissociation constant that reproduces it (`pycham_diss_wrtw`), the effective dry density for equal wet volume (`effective_dry_density`), and the display dry diameter (`dry_diameter_nm`). |
| `load_pycham.py` | PyCHAM output loader. Per-bin number is `N_bin` (`particle_number_concentration_wet`, the number that goes with the per-bin composition). `N_dry_rebinned` is PyCHAM's dry re-binning, which does not line up with the composition. `rad_um` at t = 0 is the dry input radius when `Vwat_inc = 0`. `mv_cm3_mol` exposes `MV.npy`. |
| `check_rates_pycham.py` | Evaluates PyCHAM's generated `rate_coeffs.py` at a given environment (PyCHAM venv), to check every rate constant against ours. |
| `metrics.py` | Relative difference, normalised mean bias, max relative deviation, snapshots and summaries for species above an abundance threshold; CSV writer. |
| `plotting.py` | `timeseries_grid` (values + % difference panel), `scatter_snapshots`, `ratio_distribution`, `stat_cdfs`, `darker`. |
| `run_pycham.py` | Headless PyCHAM driver (copied from `pycham-bench/`). |
| `patches/` | `single_seed_dry_fix.patch` (applied). `mcm331_kpp_photolysis.patch` is kept for the record but **not applied**: the benchmark prescribes J from TOMAS-JAX instead. |
| `requirements-lock.txt` | Exact package versions of the PyCHAM venv (Python 3.10). |
| `cases/<case>/` | `case.py` (paths, scenario, mismatch list), `run.py`, `analyse.py`, `README.md`, `results/` (small npz/csv/json, ≤ ~2.5 MB), `figures/`. |

## Conventions

- **Colours** (`config.COLORS`): one colour has one meaning in every figure. TOMAS-JAX
  at the preset settings is blue (solid), a TOMAS-JAX numerical sensitivity run (60 s
  coupling, 160 bins) sky blue (dashed), PyCHAM vermillion (thick solid), PyCHAM with
  another input (own Li c*, J every 900 s) orange (dashed/hatched), PyCHAM with its own
  vapour pressures black. Attribution curves "run A vs run B" take the colour of the
  difference they isolate or remove: `attr_matched` (green) means every difference the
  case attributes is removed; `attr_solver_ours` / `attr_solver_pycham` are each model's
  own solver error; `attr_solver_both`, `attr_hold`, `attr_frozen_rates`,
  `attr_kelvin_off`, `attr_transfer` and `attr_raoult` name the difference. Within each
  figure every pair passes the dataviz palette check (≥ 15 OKLab ΔE for normal vision,
  ≥ 6 under protan/deutan simulation); line style is the second encoding.
- **Dry diameters** in every particle figure are the dry mass per particle at
  `config.DISPLAY_DRY_DENSITY_KG_M3` = 1400 kg m⁻³ (`coupling.RHO_REPORT`, what the
  tutorial reports), stated on the axis. TOMAS-JAX places the seed with 1770 kg m⁻³, so
  a "50 nm" seed shows at 54 nm.
- **Units**: gases in ppb (OH, HO2, NO3 in molec cm⁻³ in figures), particle mass in
  µg m⁻³, number in cm⁻³, times in hours since the start of the run.

## Metrics

A (time, species) point counts when PyCHAM (the reference) is at least
`ABUNDANCE_THRESHOLD_PPB = 1e-5` ppb (~2.5e5 molec/cm³, which keeps OH). t = 0 is
excluded. Per species:

- normalised mean bias NMB = Σ(ours − PyCHAM) / Σ PyCHAM
- max relative deviation = max |ours/PyCHAM − 1|

Snapshots give the scatter plots and the log10-ratio distributions. A secondary
threshold of 1e-3 ppb is reported in `summary.json`. SA, NA_HET, NA, PROD and H2O are
bookkeeping species and are left out of the all-species statistics (SA is compared
separately as cumulative H2SO4); so is a held species (G3: NO2), which is analysed on
its own.

## Recreating the PyCHAM install

The install lives next to the repo, in `tomas-aaar/pycham-bench/`. It is not part of
the repo. Set `PYCHAM_BENCH` to use another location.

```bash
cd tomas-aaar                                    # parent of tomas-jax
mkdir -p pycham-bench && cd pycham-bench
git clone https://github.com/simonom/PyCHAM.git PyCHAM
git -C PyCHAM checkout 7880d81991542ebbcd6597e01ddcf9d93d997a72          # v5.8.0
git -C PyCHAM apply ../../tomas-jax/benchmarks/pycham/patches/single_seed_dry_fix.patch
# UManSysProp must sit in the PyCHAM root (PyCHAM would otherwise clone it at run time)
git clone https://github.com/loftytopping/UManSysProp_public.git PyCHAM/umansysprop
git -C PyCHAM/umansysprop checkout f6d8645d40f47e57111d4ec77c5427502eeb5d86
uv venv --python 3.10 .venv
uv pip install --python .venv/bin/python -r ../tomas-jax/benchmarks/pycham/requirements-lock.txt
mkdir -p inputs outputs
```

PyCHAM is not pip-installed. `run_pycham.py` puts `$PYCHAM_ROOT/PyCHAM` on `sys.path` and
`chdir`s to `$PYCHAM_ROOT`. Running from the repo root matters: PyCHAM switches MCM
photolysis on only when `photo_path == cwd + '/PyCHAM/photofiles/MCMv3.2'`, and
otherwise J is silently zero.

**One simulation per checkout at a time.** PyCHAM rewrites `rate_coeffs.py`,
`ode_solv.py` and `pickle.pkl` in place. `pycham_inputs.ensure_pycham_root(case)`
therefore copies the pinned checkout once per case to `pycham-bench/PyCHAM_<case>`
(`cp -R`, ~400 MB each) and points the driver at the copy with `PYCHAM_ROOT`. Runs
within a case are sequential. The Kelvin-off attribution runs use
`PyCHAM_<case>_nokelvin`, a copy with `surfT = 0` in `partit_var_prep.py`.

Smoke test (exit code 0 and "Simulation complete" expected):

```bash
cd tomas-aaar/pycham-bench
PYCHAM_ROOT=$PWD/PyCHAM .venv/bin/python -I ../tomas-jax/benchmarks/pycham/run_pycham.py /abs/path/model_var.txt
```

PyCHAM input rules the writers enforce or follow:

- `res_file_name` is absolute (PyCHAM overwrites an existing folder).
- `p_init = 101325` is always written (the PyCHAM default is 9.8e4 Pa).
- Sizes are radii in µm. `lon` is positive west. `daytime_start` is seconds after 00:00 UTC.
- `light_status = /abs/J.xlsx` is written with no `light_time` key (it would override
  the times read from the file), and with `light_adapt = 0`.
- `dens` (with `dens_Comp`) is read in **kg m⁻³** (v5.8.0 stores it unconverted,
  although its README says g cm⁻³); `seed_dens` is in g cm⁻³. A `dens_Comp` entry for
  H2O is ignored (water is always 1000 kg m⁻³, MW 18.0).
- Seed components other than `core` must appear in the chemical scheme; a dead
  reaction (`k = 0`) and an xml SMILES are enough.
- An inert dry seed with no absorbing mass at t = 0 stalls the solver, so the seed
  always carries a non-volatile absorbing organic.
