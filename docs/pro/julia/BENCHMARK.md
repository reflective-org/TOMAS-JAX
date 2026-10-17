# Shared gas-chemistry benchmark: Julia against JAX

Status: written 2026-10-17 in `_work/julia/bench/`, from TOMAS-JAX branch
`feat/pro-mechanism` (7e7d19d), checked out as branch `pro-julia-bench` in `_work/julia/tj`.
It is for the Pro Simulator after 2026-10-27. The AAAR tutorial stays on JAX, and nothing
in TOMAS-JAX was changed.

**Where the files are now.** The paths below (`_work/julia/bench/...`) are those of the
workspace where the benchmark was built. In git:

- TOMAS-JAX, branch `feat/pro-julia`:
  - `prototypes/julia/bench/` holds the scripts, the problem `MANIFEST.json`, the reference
    `CHECKS.json`, the batch logs and every run's `_timing.json`;
  - `prototypes/julia/jax_sparse_spike/` holds the sparse JAX spike;
  - this file is `docs/pro/julia/BENCHMARK.md`.
- TOMAS.jl: this file is `docs/BENCHMARK.md`, and the Julia spike is
  `prototypes/julia-spike/`.

The problem, reference and full result files (about 200 MB) are not committed. Section 10
regenerates them.

The results of the Julia and sparse JAX spikes, and a review that re-ran the key timings, are
in TOMAS.jl `docs/TOMAS_JL_PLAN.md`, sections 2 and 9. In short, at matched accuracy
neither engine is clearly faster for the gas chemistry. Which one wins depends on the method,
the error measure and the coupling interval.

Why (Ali): "you should also deploy agents for Julia development since I think that has much
faster runtime if it is desiged properly to have it on laptops. That way we can tackle full
MCM. Do you agree? You should make a plan for it." The choice between Julia and JAX for
full-MCM gas chemistry will rest on a fair benchmark. This file defines that benchmark:

- the problems
- the files that carry them
- the reference solutions
- the accuracy metric
- the timing protocol
- the numbers for today's JAX dense solver

It also gives a sparse-solver baseline that a candidate has to beat.

## 1. Summary

1. **Five problems, each at two coupling intervals (10 cases).**
   - The tutorial limonene mechanism (726 species, 2135 reactions) under Level 2 and
     Level 3 conditions.
   - Synthetic mechanisms of 1353, 4488 and 5742 species built from it (the construction
     in `prototypes/mechanism/scaling.py`), under Level 3 conditions.

   Each case runs 24 h of gas chemistry only. Rate constants are held fixed over each
   coupling interval of 900 s or 60 s.
2. **One neutral file set (7.5 MB of JSON).** It holds:
   - the mechanism structure
   - everything that sets each interval's rate constants
   - the photolysis tables
   - the initial state
   - the rules applied between intervals

   A port needs nothing else. It can check itself before any timing with unit-test
   vectors: the right-hand side and a Jacobian-vector product at a test state.
3. **The export is exact.**
   - The right-hand side built from the files equals TOMAS-JAX's own, bit for bit.
   - The rate constants equal the TOMAS-JAX driver's to 3e-16.
   - The analytic sparse Jacobian equals the autodiff Jacobian entry by entry.
4. **References accurate to about 1e-7.** They come from scipy BDF with the analytic
   sparse Jacobian at rtol 1e-10 and atol 1e-4 molec/cm3.
   - A different method (Radau, rtol 1e-9) agrees to 1.4e-7 or better.
   - The synthetic problems are the limonene problem in disguise (section 2.3). Their
     references reproduce the limonene reference to 1.1e-8.
5. **Metric and gate.** The metric is the largest relative error over 18 to 20 key
   species at every interval end. Values below 1e6 molec/cm3 are compared in absolute
   terms. Speed comparisons count only at matched accuracy. The gate is no worse than
   twice the JAX dense error or 1e-2 (whichever is larger), and never above 5e-2.
6. **Today's JAX dense solver** (Kvaerno3, autodiff Jacobian, dense LU, rtol 1e-3,
   atol 1e4) takes the following for 24 h at 900 s coupling:

   | Species | Time for 24 h | Peak memory |
   |---|---|---|
   | 726 | 8 s | 0.4 GB |
   | 1353 | 30 s | 0.6 GB |
   | 4488 | 7.8 min (one run) | 3.4 GB |
   | 5742 | 13.5 min (one run) | 6.2 GB |

   At 60 s coupling it needs 59-69 s at 726 species and 253 s at 1353 species. At 4488
   and 5742 species the 60 s runs would take about 1 and 1.8 h and were not run (over the
   20 min cap). The step count barely changes with size: 1047, 995, 904 and 890 steps at
   900 s. Each step costs 8 ms, 30 ms, 0.52 s and 0.91 s. The key-species error is 2e-4
   to 3.3e-2.
7. **Sparse linear algebra is most of the gain.** scipy's BDF has an interpreted Python
   step loop and uses SuperLU on the sparse Jacobian. At matched accuracy it is already 3
   to 6 times faster than JAX dense at 726-1353 species, and 25 to 35 times faster at
   4488-5742. At full-MCM size it takes 23 s per 24 h at 900 s coupling and 228 s at 60 s.
   These numbers, not the dense ones, are the bar for a Julia solver and for the planned
   sparse JAX solver.
8. **Two traps for any comparison.**
   - The same nominal tolerances do not give the same accuracy. scipy BDF at rtol 1e-3,
     atol 1e4 has 3 to 200 times the error of Kvaerno3 at the same settings.
   - Restarting the solver every interval hurts multistep methods (BDF, FBDF, CVODE):
     each restart drops to order 1. One-step methods (Rosenbrock, SDIRK) restart cleanly.
     At 60 s coupling scipy BDF needed rtol 1e-4 to 1e-6 (and atol 1e2 in some cases) to
     match Kvaerno3 at 1e-3.

## 2. The problems

### 2.1 Mechanisms

| Mechanism | Species | Reactions | RO2 members | Jacobian nonzeros | What it is |
|---|---|---|---|---|---|
| `lim726` | 726 | 2135 | 146 | 6142 | the tutorial mechanism: the MCM v3.3.1 limonene subset (725 species) plus the NA_HET counter; reaction `<34>` (N2O5 = 2 NA_HET) has an externally set rate constant |
| `syn1353` | 1353 | 3989 | 280 | 11490 | 2 copies of lim726's chemistry of species with 4 or more C atoms |
| `syn4488` | 4488 | 13259 | 950 | 38230 | 7 copies |
| `syn5742` | 5742 | 16967 | 1218 | 48926 | 9 copies; this is full-MCM size (MCM v3.3.1 has 5832 species and 17224 reactions) |

A copy renames every species with 4 or more C atoms (`LIMONENE_1`, `LIMAL_1`, ...). It then
repeats every reaction that involves one of them, with the same rate expression. The
inorganic and C1-C3 chemistry appears once and is shared. RO2 members are copied too. The
Jacobian nonzero counts match the Pro mechanism plan (`docs/pro/mechanism/architecture.md`
section 8.2).

The full MCM was not downloaded, because its licence asks for permission. The synthetic sets
reach full-MCM size from the subset already in TOMAS-JAX.

### 2.2 Conditions

| Problem | Temperature, pressure, RH | Light | N2O5 uptake k_het | Held species | Initial mixing ratios (ppb) |
|---|---|---|---|---|---|
| `lim726_L2` | 298.15 K, 101325 Pa, 50 % (Level 2 preset) | constant: sun at 30 degrees, TUV-x spectral shape scaled to J_NO2 = 5e-3 s-1 | 0 (off in Level 2) | none | LIMONENE 10, O3 40, NO 0.5, NO2 2, SO2 1, CO 120, CH4 1900, H2 500 |
| `lim726_L3` | 293.15 K, 101325 Pa, 60 % (Level 3 preset) | TUV-x at Buffalo NY (42.89 N, 78.88 W) from 2026-07-01 16:00 UTC (noon local); dark in 35 of the 96 quarter hours | 1.545e-4 s-1 | NO2 at 1 ppb | LIMONENE 2, O3 40, NO 0.5, NO2 1, SO2 2, CO 120, CH4 1900, H2 500 |
| `syn1353_L3`, `syn4488_L3`, `syn5742_L3` | as lim726_L3 | as lim726_L3 | as lim726_L3 | NO2 at 1 ppb | as lim726_L3, with LIMONENE split equally over the N copies (2/N ppb each) |

The air number density is M = p / (k_B T), where p is pressure [Pa], k_B is the Boltzmann
constant and T is temperature [K]. It is 2.4615e19 cm-3 in Level 2 and 2.5035e19 cm-3 in
Level 3. Mixing ratios convert as y = ppb · 1e-9 · M.

Every problem runs 24 h. The Level 2 preset itself runs 6 h at 300 s coupling; here it runs
24 h at the benchmark's two couplings.

### 2.3 The synthetic problems are the limonene problem in disguise

In the limonene subset, no reaction has two reactants with 4 or more C atoms, and none makes
such a species from smaller ones. So if the precursor is split equally over N copies, the
exact solution has these properties:

- every shared species (O3, OH, NO, ...) stays equal to its value in `lim726_L3`;
- each copy of a species with 4 or more C atoms holds 1/N of its lim726_L3 value;
- the RO2 sum is unchanged.

This gives an exact and cheap check of any solver at full size. The limitation is that the
synthetic problems add no new chemistry and no new time scales. A real full-MCM mechanism
(143 VOCs with different rates) can need more steps per day. Step counts must be measured
again on a real multi-VOC export.

## 3. What is solved

Symbols:

- y_s: concentration of species s [molec/cm3]
- k_j: rate constant of reaction j [cm3 molec-1 s-1, or s-1]
- Δt: coupling interval [s]
- N_c: number of coupling intervals in 24 h

The equations:

- Rate of reaction j: r_j = k_j · ∏ y_a over its reactant entries a. There is one entry per
  occurrence, so NO2 + NO2 gives k · y_NO2². Every reaction has one or two entries.
- Right-hand side: dy_s/dt = Σ_j ν_sj r_j. Here ν_sj is the product coefficient of s in
  reaction j minus the number of reactant entries of s in j.
- Interval i, with i = 0, ..., N_c − 1 and start time t_i = i · Δt:
  1. RO2_i = Σ y_s over the RO2 members, at t_i.
  2. J_i = the photolysis frequencies at the midpoint t_i + Δt/2 (tabulated in the file).
  3. k_i = k_thermal ⊙ g_i + C_J · J_i, where ⊙ is element-wise multiplication. g_i is
     RO2_i for RO2-dependent reactions and 1 for all others. Then the external reaction
     (`<34>`) is set to k_het. k_thermal, the list of RO2-dependent reactions and the
     sparse matrix C_J are fixed per problem: they depend only on T, M, O2 and H2O.
  4. Solve dy/dt = f(y; k_i) from t_i to t_i + Δt with k_i held fixed. These are KPP
     semantics, as in TOMAS-JAX. Each interval is a new initial-value problem. A solver
     may carry its last step size into the next interval, but must report that it does.
  5. Clip negative values to zero. **This interval-end state is the scored output.**
  6. Set the counters SA and NA_HET to zero and, in Level 3, set NO2 back to its initial
     value. The result is the start state of interval i + 1.
- Δt is 900 s (N_c = 96) or 60 s (N_c = 1440). These are different problems: holding k
  fixed for 900 s or for 60 s changes the solution, so each has its own reference.
- The nominal solver tolerances are the tutorial's: rtol 1e-3, atol 1e4 molec/cm3.

This is the gas half of the TOMAS-JAX driver loop (`box/driver.py`). Three things are left
out: the aerosol step, the volatility-class gather and scatter, and the dependence of k_het
on the aerosol.

## 4. Files

Everything is in `_work/julia/bench/`. Indices are 0-based in every file (add 1 in Julia).

### 4.1 Problem files (`problems/`, 7.5 MB in total)

| File | Size | Content |
|---|---|---|
| `mech_lim726.json` | 0.09 MB | mechanism structure |
| `mech_syn1353.json` | 0.17 MB | mechanism structure |
| `mech_syn4488.json` | 0.60 MB | mechanism structure |
| `mech_syn5742.json` | 0.78 MB | mechanism structure |
| `problem_lim726_L2.json` | 0.97 MB | conditions, rate inputs, J tables, initial state, rules |
| `problem_lim726_L3.json` | 0.70 MB | same |
| `problem_syn1353_L3.json` | 0.73 MB | same |
| `problem_syn4488_L3.json` | 0.89 MB | same |
| `problem_syn5742_L3.json` | 0.95 MB | same |
| `checks_lim726_L2.json` ... `checks_syn5742_L3.json` | 0.09-0.72 MB | unit-test vectors |
| `MANIFEST.json` | small | size and sha256 of every file above |

`mech_<m>.json` (format `tomas-gas-bench/mechanism/1`) has these fields:

- `n_species`, `n_reactions`
- `species`: the species names
- `reactants`: for each reaction, a list of 1 or 2 species indices (repeated for A + A)
- `products`: for each reaction, a list of [species index, coefficient]
- `ro2`: indices of the RO2 members
- `labels`: MCM labels (copies get `_c`)
- `origin_reaction`: the index of the lim726 reaction that a reaction copies

`problem_<p>.json` (format `tomas-gas-bench/problem/1`) has these fields:

- `mechanism_file`
- `conditions`: T, p, RH, M, O2, N2, H2O, the photolysis settings and k_het
- `duration_s`, `couplings_s`, `tolerances`
- `rates`:
  - `k_thermal`: one value per reaction
  - `ro2_dependent`: reaction indices
  - `j_names`: the 28 photolysis names
  - `j_coef`: the sparse matrix C_J, as [reaction, J column, value]
  - `external`: [reaction, k]
- `photolysis`: key "900" (96 rows) and key "60" (1440 rows), each row 28 J values [s-1]
- `initial.y0_nonzero`: [species index, molec/cm3]
- `post_interval`: `reset_to_zero`, `reset_to_initial`
- `key_species` and their indices

Floats are written in Python's shortest round-trip form, so a correctly rounding JSON reader
recovers every bit.

`checks_<p>.json` holds:

- a test state y_test and a vector v
- k at y_test for interval 0 (900 s)
- RO2(y_test)
- f(y_test) and J(y_test) · v

A port should match f to about 1e-14 relative (summation order) before it is timed.

### 4.2 Reference files (`reference/`, 6.7 MB)

For each of the 10 cases there are two files:

- `ref_<p>_dt<Δt>.json`: key species at every interval end, RO2 at every interval start,
  and solver statistics.
- `ref_<p>_dt<Δt>_hourly.npz`: every species at t = 1, 2, ..., 24 h, as arrays `t_s`,
  `y` [24, n] and `species`.

The 20 files total 5.0 MB. The Radau cross-checks are in `chkradau_*` (1.7 MB) and the
summary of the checks is in `CHECKS.json`.

To score any solver's run, write the same two files (same keys) and call
`scripts/compare.py run.json reference.json`.

### 4.3 Results (`results/`, 35 MB)

- `jaxdense_*`: the JAX dense runs.
- `scipysparse_*`: the scipy sparse runs at each tolerance setting.

Each run has a run file and an hourly file. JAX runs also have a `_timing.json` with every
timing and load reading. The `logs/` folder holds the batch logs and `tables.md`, the
tables below as generated.

### 4.4 Scripts (`scripts/`)

| Script | What it does |
|---|---|
| `bench_common.py` | reads the files; builds the numpy right-hand side, the analytic sparse Jacobian (CSC) and the interval loop; computes the metric. It does not import JAX, so it serves as the executable specification |
| `export_problems.py` | writes `problems/` from TOMAS-JAX (`tutorial_mechanism`, `RateModel.at_env`, `JProvider`, `n2o5_uptake`) |
| `verify_export.py` | checks the files against TOMAS-JAX (right-hand side, rate constants, Jacobian) |
| `reference.py` | runs tight scipy BDF or Radau; with `--rtol 1e-3 --atol 1e4 --out-dir ../results` it is the sparse baseline |
| `check_references.py` | runs the Radau-against-BDF and synthetic-against-limonene checks; writes `reference/CHECKS.json` |
| `compare.py` | scores a run against a reference |
| `run_jax_dense.py` | times TOMAS-JAX `integrate_interval` with the protocol in section 7 |
| `make_tables.py` | prints the tables in this file from the result files |
| `run_references.sh`, `run_jax_dense_batch.sh`, `run_scipy_sparse_batch.sh`, `run_scipy_matched.sh`, `run_scipy_matched_atol.sh` | the batches that produced the numbers here; every run is bounded by a timeout |

## 5. Reference solutions

The solver is `scipy.integrate.BDF` (scipy 1.17.1, variable order 1 to 5):

- analytic sparse Jacobian, factorised by SuperLU
- rtol 1e-10, atol 1e-4 molec/cm3
- restarted every interval, with interval-end values clipped at zero as in the problem
  definition

These checks show the references are accurate to about 1e-7 in the metric of section 6:

| Check | Result: key metric (all-species hourly) |
|---|---|
| BDF at rtol 1e-8, atol 1e-2 against the reference (lim726_L3, 900 s) | 2.4e-7 |
| BDF at rtol 1e-9, atol 1e-3 against the reference (lim726_L3, 900 s) | 4.1e-8 |
| Radau at rtol 1e-9, atol 1e-3 against the reference: lim726_L2, 900 s | 6.8e-9 (6.5e-9) |
| same, lim726_L2, 60 s | 1.4e-7 (9.1e-8) |
| same, lim726_L3, 900 s | 6.9e-9 (6.9e-9) |
| same, lim726_L3, 60 s | 4.4e-8 (3.1e-9) |
| synthetic reference against the lim726_L3 reference (section 2.3), every species at every hour: syn1353 | 2.2e-9 at 900 s, 7.3e-10 at 60 s |
| same, syn4488 | 9.5e-9 at 900 s, 2.6e-9 at 60 s |
| same, syn5742 | 1.1e-8 at 900 s, 3.4e-9 at 60 s |

Cost of each reference (one process, shared machine):

| Case | 900 s coupling | 60 s coupling |
|---|---|---|
| lim726_L2 | 6 s | 38 s |
| lim726_L3 | 11 s | 80 s |
| syn1353_L3 | 31 s | 251 s |
| syn4488_L3 | 97 s | 748 s |
| syn5742_L3 | 116 s | 827 s |

That is 30,000-35,000 steps per day at 900 s and 200,000-230,000 at 60 s, with 130-360 MB
peak memory.

## 6. Accuracy metric

For key species s and interval end i, with y the run and ŷ the reference:

  e_si = |y_si − ŷ_si| / max(|ŷ_si|, F),   with F = 1e6 molec/cm3 (about 0.04 ppt)

- **Key metric** E_key = the maximum of e_si over s and i. Also reported: the median, the
  95th percentile and the worst species.
- **Key species:** O3, OH, HO2, NO, NO2, NO3, N2O5, HNO3, H2O2, HCHO, PAN, SA (the H2SO4
  formed in the interval), LIMONENE, LIMAL, LIMKET, LIMONONIC, LIMAOOH and LIMANO3. The
  synthetic problems add LIMONENE and LIMONONIC of the last copy.
- **All-species metric** E_all = the maximum of e over every species in the 24 hourly
  states.
- **Why the floor F:** it keeps species that pass through zero (NO at night, LIMONENE after
  it is used up) from dominating. Below F the error is absolute, in units of F. F is 100
  times the nominal atol.
- **Gate:** a run counts only if E_key ≤ G on that case, where
  G = min(5e-2, max(2 · E_key of JAX dense, 1e-2)). When there is no JAX dense run,
  G = 5e-2.
  - A candidate that fails G at the nominal tolerances is tightened (rtol first, then
    atol) until it passes, and that setting is the one timed.
  - Speed is compared only between runs that pass.
  - A work-precision curve (E_key against warm time for rtol 1e-2 to 1e-6) is the best
    summary when time allows.

Two observations made while defining the gate:

- The largest JAX dense errors are atol-limited transients. NO at sunset in the 60 s Level 3
  runs falls through about 1e6 molec/cm3, giving E_key 3.1e-2 with an absolute error of 3.6
  times atol. LIMONENE decays by 11 e-folds in the 900 s runs, giving 1.0e-2.
- For a species that decays over many e-folds, the relative error grows with the number of
  e-folds. LIMONENE is therefore the hardest key species. That is why scipy BDF needs much
  tighter settings in Level 2.

## 7. Timing protocol

Measure these quantities for each case and solver:

| Quantity | Definition |
|---|---|
| t_setup | from a fresh process start until the libraries are loaded, the problem is read and the arrays are ready. For Julia this includes package loading; report whether a system image or precompiled package image was used |
| t_first | the first complete 24 h run in that process, including all compilation |
| t_first_interval | compilation plus the first interval alone (the first-call latency) |
| t_warm | each further complete 24 h run in the same process. Do at least 3; report each one and the median |
| work | accepted steps; also rejected steps, right-hand-side evaluations, Jacobian evaluations and LU factorisations where the solver reports them |
| peak memory | peak resident set size of the whole process (getrusage ru_maxrss, or /usr/bin/time -l) |
| load | `uptime` before and after every timed run |
| accuracy | E_key and E_all of the first run (all runs are deterministic) |

Rules:

- **What is timed:** 24 h of simulated time, with the whole interval loop inside the timed
  region (rate constants, solves, clipping, resets). File reading is outside it.
- **Machine:** the same machine for both candidates. Here that is a 14-core Apple M4 Pro
  with 48 GB, running macOS 26.6.2.
- **Shared machine:** the machine is shared with other jobs. Record the load with every
  timing, interleave the two candidates (A, B, A, B), and repeat the key timings at the
  end of a session.
- **Threads:** report the threads used. For JAX that is the XLA defaults with no
  XLA_FLAGS; for Julia, the `-t` setting and the BLAS thread count. Run one process and one
  problem at a time.
- **What to report:** the numbers at nominal tolerances and the numbers at matched accuracy
  (section 6).
- **Wall-clock cap:** each run has a cap (20 min here). A run that hits the cap reports the
  intervals done and the time extrapolated to 24 h, and is labelled as capped.

## 8. Results: today's JAX dense solver

This is TOMAS-JAX `gaschem.solver.integrate_interval` exactly as the tutorial uses it:

- diffrax 0.7.2 Kvaerno3
- Jacobian by forward-mode autodiff, dense LU
- rtol 1e-3, atol 1e4, max_steps 2000
- JAX 0.9.1 on CPU, Python 3.11.15

The mechanism arrays come from the problem files. The machine was shared with another
workflow's builds; the 1-minute load is listed before each run (first run, then the warm
runs). The "(repeat)" rows were measured about 45 minutes later, at the end of the session.

| Case | n | t_setup [s] | t_first [s] | t_first_interval [s] | t_warm median [s] (runs) | steps | ms per step | peak RSS [MB] | E_key (worst species) | E_all | 1-min load |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lim726_L2 900 s | 726 | 1.2 | 6.5 | 0.99 | 5.7 (5.8, 5.7, 5.7) | 710 | 8.1 | 419 | 2.3e-03 (LIMONENE) | 1.7e-03 | 6.3, 6.2, 7.2, 7.2 |
| lim726_L3 900 s | 726 | 1.3 | 9.1 | 1.02 | 8.4 (8.2, 8.5, 8.4) | 1047 | 8.1 | 420 | 1.0e-02 (LIMONENE) | 8.2e-03 | 6.9, 6.4, 6.2, 6.4 |
| lim726_L3 900 s (repeat) | 726 | 1.1 | 8.6 | 0.89 | 7.8 (7.8, 7.8, 8.0) | 1047 | 7.5 | 423 | 1.0e-02 (LIMONENE) | 8.2e-03 | 9.2, 8.7, 12.3, 11.3 |
| syn1353_L3 900 s | 1353 | 1.1 | 30.9 | 1.41 | 29.8 (30.1, 29.8, 29.8) | 995 | 30.0 | 621 | 1.2e-02 (LIMONENE) | 8.4e-03 | 7.7, 7.1, 8.8, 7.2 |
| syn1353_L3 900 s (repeat) | 1353 | 0.9 | 29.8 | 1.30 | 28.2 (28.6, 28.2, 27.6) | 995 | 28.4 | 620 | 1.2e-02 (LIMONENE) | 8.4e-03 | 12.4, 22.8, 35.5, 31.9 |
| syn4488_L3 900 s | 4488 | 1.3 | 468.2 | 10.82 | not run (one run only) | 904 | 518 (first run) | 3449 | 1.0e-02 (LIMONENE_6) | 1.0e-02 | 7.1 |
| syn5742_L3 900 s | 5742 | 1.2 | 807.3 | 17.10 | not run (one run only) | 890 | 907 (first run) | 6180 | 8.6e-03 (LIMONENE_8) | 8.6e-03 | 10.0 |
| lim726_L2 60 s | 726 | 1.1 | 59.1 | 0.91 | 59.2 (58.6, 59.2, 61.6) | 7237 | 8.2 | 434 | 2.2e-04 (HO2) | 1.1e-04 | 8.0, 6.3, 6.3, 6.8 |
| lim726_L3 60 s | 726 | 1.2 | 70.9 | 0.91 | 69.4 (70.4, 69.4, 69.3) | 8666 | 8.0 | 434 | 3.1e-02 (NO) | 5.8e-04 | 6.2, 7.8, 7.0, 9.4 |
| lim726_L3 60 s (repeat) | 726 | 0.9 | 67.2 | 0.81 | 67.8 (67.8, 67.6, 68.1) | 8666 | 7.8 | 434 | 3.1e-02 (NO) | 5.8e-04 | 22.8, 13.1, 7.7, 5.7 |
| syn1353_L3 60 s | 1353 | 1.2 | 264.1 | 1.19 | 252.6 (255.0, 250.2) | 8189 | 30.8 | 645 | 3.3e-02 (NO) | 8.7e-04 | 7.2, 13.7, 10.6 |
| syn4488_L3 60 s | 4488 | - | - | - | skipped: about 1 h for 24 h (the 900 s time × 8.2, the step ratio of 60 s to 900 s at 726 and 1353 species) | - | - | - | - | - | - |
| syn5742_L3 60 s | 5742 | - | - | - | skipped: about 1.8 h for 24 h | - | - | - | - | - | - |

Notes:

- **Repeat runs** at the end of the session agree with the first ones to within 8 %, even
  though the machine load swung between 6 and 35.
- **The 4488 and 5742 runs at 900 s** finished under the 20 min cap. To bound machine time,
  they were run once, with no warm repeats. Their t_first includes compilation;
  t_first_interval (compilation plus the first interval) is 10.8 s and 17.1 s.
- **Against the plan's estimates.** The Pro mechanism plan (architecture.md section 8.3)
  estimated 12 s, 44 s, 14 min and 32 min for the four sizes at 900 s. The measured values
  are 8.4 s, 30 s, 7.8 min and 13.5 min. Its assumption that step counts do not depend on
  size holds for these synthetic problems (1047, 995, 904 and 890 steps).
- **Cost per step** grows from 8 ms to 30 ms, 518 ms and 907 ms. From 726 to 1353 species
  that is about n²; from 1353 to 4488 it is about n^2.4; from 4488 to 5742 it is about n^2.3.
  This is the dense Jacobian (n tangents through every gather) plus the dense LU.
- **Peak memory** is 0.42, 0.62, 3.4 and 6.2 GB for the four sizes.
- **First-call latency** (compile plus the first interval) is about 1 s up to 1353 species.

### 8.1 Extra baseline: scipy BDF with the sparse Jacobian

This baseline was not asked for. It shows what a sparse Jacobian alone buys, with no
compilation and an interpreted solver loop: `reference.py` at chosen tolerances (scipy BDF
with SuperLU, one process, single-threaded). At nominal tolerances each case was run twice,
and the two passes agree to within 4 %.

Matched accuracy is defined by the gate of section 6. The JAX column gives the warm median,
or the single run where marked.

| Case | JAX dense: 24 h [s], E_key | scipy sparse at nominal 1e-3/1e4: [s], E_key | scipy sparse, cheapest setting that passes the gate: rtol, atol, [s], E_key | JAX time / scipy time at matched accuracy |
|---|---|---|---|---|
| lim726_L2 900 s | 5.7, 2.3e-03 | 1.0, 3.6e-01 | 1e-5, 1e2, 1.7, 2.4e-03 | 3 |
| lim726_L3 900 s | 8.4, 1.0e-02 | 1.5, 1.6e-01 | 1e-5, 1e4, 1.8, 1.3e-02 | 5 |
| syn1353_L3 900 s | 29.8, 1.2e-02 | 4.5, 1.5e-01 | 1e-5, 1e4, 5.4, 1.4e-02 | 6 |
| syn4488_L3 900 s | 468.2 (one run), 1.0e-02 | 13.0, 1.8e-01 | 1e-4, 1e2, 18.7, 9.9e-03 | 25 |
| syn5742_L3 900 s | 807.3 (one run), 8.6e-03 | 15.5, 1.5e-01 | 1e-4, 1e2, 22.8, 1.2e-02 | 35 |
| lim726_L2 60 s | 59.2, 2.2e-04 | 10.6, 4.5e-02 | 1e-6, 1e2, 15.1, 5.3e-03 | 4 |
| lim726_L3 60 s | 69.4, 3.1e-02 | 13.7, 1.1e-01 | 1e-5, 1e4, 15.8, 2.7e-02 | 4 |
| syn1353_L3 60 s | 252.6, 3.3e-02 | 43.4, 1.2e-01 | 1e-5, 1e4, 50.3, 4.3e-02 | 5 |
| syn4488_L3 60 s | not run (over 20 min) | 136.1, 1.4e-01 | 1e-4, 1e2, 191.5, 3.5e-02 | - |
| syn5742_L3 60 s | not run (over 20 min) | 159.9, 2.1e-01 | 1e-4, 1e2, 228.0, 3.7e-02 | - |

The full sweep, covering every case at rtol and atol of (1e-3, 1e4), (1e-4, 1e4),
(1e-5, 1e4), (1e-3, 1e2) and (1e-4, 1e2), plus (1e-5, 1e2) and (1e-6, 1e2) for Level 2, is in
`logs/tables.md`. Peak memory of the scipy runs is 125-416 MB.

What this means for the plan:

- **The bar for a fast solver** is set by the sparse numbers. At full-MCM size, a scipy
  loop needs 23 s per 24 h at 900 s coupling and 228 s at 60 s.
- **Where the scipy time goes.** At 60 s coupling scipy BDF takes about 18,000 steps per
  day, and each restart starts again at order 1. Its time is mostly Python overhead and
  frequent refactorisations: about 9,700 LU factorisations per day.
- **What a candidate can win.** A compiled sparse solver can win on per-step overhead. A
  one-step method can win on restarts.
- **What the Julia candidate must show** is how far below these numbers it gets, at
  matched accuracy, against the planned sparse JAX solver (the Pro plan's Ros3 with the
  levelised LU). Beating the dense solver is not enough.

## 9. Assumptions and limits

1. **Gas chemistry only.** In the coupled model, the aerosol step also removes condensable
   organics each interval, and k_het changes as the aerosol grows. Both are left out so the
   problem is a pure ODE sequence that every solver can reproduce.
2. **k_het(N2O5) in Level 3 is held fixed** for 24 h. Its value is the one for the preset's
   initial seed: 3000 cm-3, 80 nm, σ 1.6, with water at RH 60 %.
3. **Level 2 is run for 24 h** instead of its preset 6 h, at 900 s and 60 s instead of
   300 s.
4. **The synthetic problems split the precursor equally** over the copies (section 2.3).
   This keeps them exactly checkable, but they add no stiffness of their own.
5. **The counters SA and NA_HET are reset** after every interval, as in the driver. They are
   never reactants, so this changes nothing else.
6. **The references come from scipy, not TOMAS-JAX.** A tight Kvaerno5 run in JAX was not
   needed: two independent scipy methods agree to 1.4e-7, and the synthetic references
   reproduce the limonene one.
7. **Real full-MCM mechanisms need a rerun** of this protocol on a real export. Step counts
   and Jacobian fill will be higher than for the synthetic copies.
8. **The machine was shared** with another workflow throughout (1-minute load 2 to 35).
   Timings are upper bounds. The repeat runs agree to within 8 %.

## 10. Reproduce

The Python environment is `_work/julia/venv`:

- made with uv, with `UV_CACHE_DIR=_work/julia/uv-cache`
- built with `uv sync --frozen --extra dev` in `_work/julia/tj`
- plus vbs-jax at its pinned commit, and netCDF4

TUV-x data come from the tuvx-jax checkout at `tomas-aaar/tuvx-jax`, through `PYTHONPATH`
and `TUVX_JAX_ROOT`. That checkout is at commit 655668c, whose `data/` and `examples/` are
identical to those of the pinned commit e5d9032.

From `scripts/`, run:

1. `python export_problems.py` (about 1 min), then `python verify_export.py`.
2. `./run_references.sh`, then `python check_references.py`.
3. `./run_jax_dense_batch.sh`, `./run_scipy_sparse_batch.sh`, `./run_scipy_matched.sh` and
   `./run_scipy_matched_atol.sh`.
4. `python make_tables.py`.
