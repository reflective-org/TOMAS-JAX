# JAX sparse solver spike on the shared gas-chemistry benchmark

Status: written 2026-10-17 in `_work/julia/jax-spike/`. It is for the Pro Simulator after
2026-10-27. The AAAR tutorial stays on the dense JAX solver, and nothing in TOMAS-JAX was
changed.

Why (Ali): "you should also deploy agents for Julia development since I think that has much
faster runtime if it is desiged properly to have it on laptops. That way we can tackle full
MCM. Do you agree? You should make a plan for it." The choice between Julia and JAX rests
on a fair benchmark (`_work/julia/bench/BENCHMARK.md`). The previous step measured today's
dense JAX solver and a scipy sparse baseline. This step builds the planned sparse JAX
solver, runs it on the same 10 cases with the same protocol, and compares.

## 1. Summary

1. **A sparse JAX solver now runs the whole benchmark.** It uses:
   - Ros3, a 3-stage Rosenbrock method of order 3 (the method of the Pro plan), with KPP's
     coefficients and step-size rules;
   - the analytic Jacobian, written straight into the sparse matrix;
   - a sparse LU with a fixed elimination order, grouped into levels that XLA runs as
     vectorised steps.

   It needs nothing new: plain JAX 0.9.1 on the CPU, about 900 lines of Python. The
   Jacobian, the right-hand side and the linear solves match the benchmark's unit-test
   vectors and scipy's SuperLU to rounding (residual 1e-16).
2. **At full-MCM size (5742 species) one simulated day takes 1.65 s at 900 s coupling and
   10.9 s at 60 s coupling,** at matched accuracy. Peak memory is 0.64 and 1.1 GB.

   | 5742 species, 24 h | Dense JAX (today) | scipy sparse BDF | Sparse JAX spike |
   |---|---|---|---|
   | 900 s coupling | 807 s (one run), 6.2 GB | 22.6 s | **1.65 s** (A); 2.7 s (B, 5 times more accurate) |
   | 60 s coupling | not run (about 1.8 h) | 226.5 s | **10.9 s** (B); 8.5 s as one compiled day |

   This is 490 times faster than the dense solver and 14 to 21 times faster than scipy.
   The Pro plan estimated 6.7 s at 900 s; the measured value is 4 times lower.
3. **Two configurations were timed.**
   - **A** keeps the dense harness's choices: the error is the root-mean-square over all
     species, and every interval starts with a fresh initial step (Hairer's rule, as diffrax
     does).
   - **B** uses the largest scaled error of any species, and starts each interval with the
     step size the controller proposed at the end of the previous one.

   A passes the accuracy gate at the nominal tolerances (rtol 1e-3, atol 1e4) in 6 of 10
   cases; the others need rtol 5e-4, or rtol 3e-4 with atol 1e2. B passes every gate at the
   nominal tolerances, with errors 2 to 100 times below the gate (E_key 1e-4 to 4.4e-3).
   A is cheaper in the Level 3 cases at 900 s coupling; B in every 60 s case and in
   Level 2.
4. **The elimination order mattered more than any code change.** A Markowitz order (pick the
   pivot with the fewest row times column entries, as KPP does) needs 5 times fewer update
   operations than minimum degree at 5742 species (161 thousand against 793 thousand). It
   also needs 7 times fewer triangular-solve levels (19 + 17 against 133 + 132). One step
   then costs 1.6 to 1.8 ms at 5742 species and 0.2 ms at 726.
5. **The root-mean-square error norm loosens as the mechanism grows.** The synthetic
   mechanisms add copies that are easy to integrate, so the few hard species (NO at sunset)
   count for less in the mean. At 60 s coupling, configuration A at the nominal tolerances
   has E_key 4.6e-2 at 726 species and 1.2e-1 at 5742. With the largest-error norm the step
   count and the error are the same at every size (in B: 3,940 steps and E_key 4.3e-3).
   Full-MCM runs should use the largest-error norm, or a mean over a fixed set of species.
6. **Restarts cost steps.** A fresh initial step every interval gives 7 to 9 steps per
   60 s interval, mostly ramping up from a tiny first step. With the largest-error norm,
   carrying the step size over cuts the steps per day at 60 s coupling from 12,700 to
   3,940, plus 1,753 rejected attempts (lim726_L3). Starting each interval at a fraction
   of the carried step (0.05 to 0.5) was slower.
7. **Startup is short.** Compilation plus the first interval takes 0.6 to 1.1 s at every
   size. The dense solver takes 0.9 s at 726 species and 17 s at 5742. Building the
   elimination order and the level schedule takes 0.1 to 0.4 s.
8. **Implication for Julia.** A Julia candidate (for example OrdinaryDiffEq Rodas4 or
   Rosenbrock23 with KLU, or Sundials CVODE with KLU) must beat these numbers at matched
   accuracy, not the dense ones. At 5742 species that means under 1.65 s per day at 900 s
   coupling and under 10.9 s at 60 s coupling on this machine. That is about 1.6 ms per
   Rosenbrock step (Jacobian, LU and three solves) at a similar step count.

## 2. What the solver does

Symbols: y the concentrations [molec/cm3], f(y) the right-hand side with the interval's
fixed rate constants, J = ∂f/∂y the Jacobian, h the step size [s].

**Rosenbrock step (Ros3).** One step from y_n:

- Form W = I/(h γ) − J(y_n), with γ = 0.43587. Factor W once.
- Solve three linear systems with that factorisation:
  - W K1 = f(y_n)
  - W K2 = f(y_n + K1) + (c21/h) K1
  - W K3 = f(y_n + K1) + (c31 K1 + c32 K2)/h
- The new state is y_{n+1} = y_n + m1 K1 + m2 K2 + m3 K3.
- The error estimate is e1 K1 + e2 K2 + e3 K3.

The coefficients are those of KPP's Ros3 (Sandu et al. 1997). On a test problem the method
converges with order 3 and its error estimate with order 2 (`logs/order_test.json`).

**Step-size control.** The scaled error is

  err = ‖ e_s / (atol + rtol · max(|y_n,s|, |y_{n+1},s|)) ‖

Here e_s is the error estimate of species s and ‖·‖ is the root-mean-square over species
(A) or the largest value (B). The step is accepted when err ≤ 1. The next step is
h · min(6, max(0.2, 0.9 · err^(−1/3))). It does not grow right after a rejection, and after
two rejections in a row h is divided by 10.

**Interval loop.** The loop is the benchmark's (section 3 of BENCHMARK.md), with every part
inside the timed region:

1. Rate constants from the RO2 sum and the tabulated photolysis.
2. The solve over the interval.
3. Clip negative values to zero.
4. Reset the counters and the held NO2.

**Jacobian.** Each reaction j with rate r_j = k_j y_a y_b adds ν_sj k_j y_b to ∂f_s/∂y_a, and
ν_sj k_j y_a to ∂f_s/∂y_b. These terms are summed straight into the positions of the sparse
LU.

**Sparse LU without pivoting.**

1. **Order.** The species are reordered once per mechanism by the Markowitz rule: eliminate
   next the species k whose remaining row and column counts r_k and c_k give the smallest
   (r_k − 1)(c_k − 1).
2. **Fill.** The fill-in for that order is computed exactly.
3. **Factorisation.** Eliminating pivot k updates a_ij ← a_ij − a_ik a_kj / a_kk for every
   i below and j right of k. Pivots whose rows and columns do not touch each other are
   grouped into a level, and one level is one vectorised gather, multiply and scatter-add.
   The level count is 48 at 726 species and 78 at 5742.
4. **Triangular solves.** The forward and backward solves are grouped the same way into
   16 + 17 levels (726) and 19 + 17 levels (5742). Each level's rows are stored next to
   each other, so a level is one fused gather-and-sum plus one in-place slice update.

## 3. Results

Machine: 14-core Apple M4 Pro, 48 GB, macOS 26.6.2, shared with another workflow's builds
(1-minute load 5 to 30 during the session). JAX 0.9.1 on the CPU, Python 3.11.15, XLA
defaults, no XLA_FLAGS, one process and one problem at a time. Times are for 24 h of
simulated time. "warm" is the median of 3 further runs in the same process. The gate is
the benchmark's: G = min(5e-2, max(2 × E_key of JAX dense, 1e-2)).

### 3.1 Comparison at matched accuracy

The scipy column is the cheapest scipy BDF setting that passes the gate, from BENCHMARK.md
section 8.1. The last column was measured in this session, interleaved with the spike runs.

| Case | Gate | JAX dense: 24 h [s], E_key | scipy sparse, matched: rtol, atol, [s], E_key | Spike A matched: rtol, atol, [s], E_key | Spike B: rtol, atol, [s], E_key | Dense / best spike | scipy / best spike | Same session: dense [s], scipy [s] |
|---|---|---|---|---|---|---|---|---|
| lim726_L2 900 s | 1.0e-2 | 5.7, 2.3e-3 | 1e-5, 1e2, 1.7, 2.4e-3 | 1e-3, 1e4, 0.19, 3.0e-3 | 1e-3, 1e4, **0.10**, 4.4e-3 | 56 | 17 | - |
| lim726_L3 900 s | 2.1e-2 | 8.4, 1.0e-2 | 1e-5, 1e4, 1.8, 1.3e-2 | 1e-3, 1e4, **0.22**, 1.6e-2 | 1e-3, 1e4, 0.36, 2.0e-3 | 38 | 8.2 | 8.1, 1.9 |
| syn1353_L3 900 s | 2.4e-2 | 29.8, 1.2e-2 | 1e-5, 1e4, 5.4, 1.4e-2 | 1e-3, 1e4, **0.54**, 1.6e-2 | 1e-3, 1e4, 0.77, 2.6e-3 | 55 | 10 | 30.2, - |
| syn4488_L3 900 s | 2.0e-2 | 468 (one run), 1.0e-2 | 1e-4, 1e2, 18.7, 9.9e-3 | 1e-3, 1e4, **1.42**, 1.4e-2 | 1e-3, 1e4, 2.50, 2.5e-3 | 330 | 13 | - |
| syn5742_L3 900 s | 1.7e-2 | 807 (one run), 8.6e-3 | 1e-4, 1e2, 22.8, 1.2e-2 | 5e-4, 1e4, **1.65**, 1.4e-2 | 1e-3, 1e4, 2.70, 2.6e-3 | 490 | 14 | -, 22.6 |
| lim726_L2 60 s | 1.0e-2 | 59.2, 2.2e-4 | 1e-6, 1e2, 15.1, 5.3e-3 | 1e-3, 1e4, 2.07, 6.2e-4 | 1e-3, 1e4, **0.63**, 9.8e-5 | 94 | 24 | - |
| lim726_L3 60 s | 5.0e-2 | 69.4, 3.1e-2 | 1e-5, 1e4, 15.8, 2.7e-2 | 1e-3, 1e4, 2.28, 4.6e-2 | 1e-3, 1e4, **1.23**, 4.3e-3 | 56 | 13 | 79.8, 18.3 |
| syn1353_L3 60 s | 5.0e-2 | 252.6, 3.3e-2 | 1e-5, 1e4, 50.3, 4.3e-2 | 5e-4, 1e4, 5.05, 4.5e-2 | 1e-3, 1e4, **2.58**, 4.3e-3 | 98 | 19 | - |
| syn4488_L3 60 s | 5.0e-2 | not run (about 1 h) | 1e-4, 1e2, 191.5, 3.5e-2 | 3e-4, 1e2, 20.7, 2.5e-2 | 1e-3, 1e4, **7.85**, 4.3e-3 | - | 24 | - |
| syn5742_L3 60 s | 5.0e-2 | not run (about 1.8 h) | 1e-4, 1e2, 228.0, 3.7e-2 | 3e-4, 1e2, 21.8, 2.7e-2 | 1e-3, 1e4, **10.9**, 4.3e-3 | - | 21 | -, 226.5 |

Notes:

- The dense 4488 and 5742 times are single first runs that include compilation (about
  11 s and 17 s); the ratios against them are about 2 % too high.
- The same-session runs agree with BENCHMARK.md within 4 % for scipy, except lim726_L3
  60 s (18.3 s against 15.8 s). For dense they agree within 4 % at 900 s and within 15 % at
  60 s (79.8 s against 69.4 s).
- The scipy reruns took 3,432 steps and 1,279 LU factorisations (lim726_L3 900 s) and
  24,014 steps and 12,773 factorisations (syn5742_L3 60 s).
- With the whole day compiled as one program (a scan over the intervals) instead of a
  Python loop over intervals, configuration B takes 1.12 s instead of 1.23 s
  (lim726_L3 60 s) and 8.45 s instead of 10.9 s (syn5742_L3 60 s).

### 3.2 Configuration A at matched accuracy (protocol table)

Ros3, root-mean-square error norm, Hairer's initial step every interval. t_setup includes
the symbolic analysis (in brackets). t_first_interval is compilation plus the first interval.

| Case | rtol, atol | t_setup [s] (symbolic) | t_first [s] | t_first_interval [s] | t_warm median [s] (runs) | Steps (rejected) | ms per step | Peak RSS [MB] | E_key (worst) | E_all | 1-min load (first, warm) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lim726_L2 900 s | 1e-3, 1e4 | 0.8 (0.2) | 1.74 | 0.78 | 0.19 (0.19, 0.19, 0.18) | 746 (0) | 0.25 | 471 | 3.0e-3 (LIMONENE) | 2.6e-3 | 14.1 |
| lim726_L3 900 s | 1e-3, 1e4 | 0.6 (0.2) | 1.45 | 0.63 | 0.22 (0.22, 0.22, 0.22) | 1124 (0) | 0.19 | 465 | 1.6e-2 (LIMONENE) | 1.2e-2 | 10.9 |
| syn1353_L3 900 s | 1e-3, 1e4 | 1.0 (0.2) | 2.52 | 0.99 | 0.54 (0.55, 0.54, 0.48) | 1055 (0) | 0.51 | 509 | 1.6e-2 (LIMONENE) | 1.2e-2 | 16.9, 16.1 |
| syn4488_L3 900 s | 1e-3, 1e4 | 1.3 (0.4) | 3.40 | 1.06 | 1.42 (1.42, 1.42, 1.38) | 1003 (0) | 1.41 | 604 | 1.4e-2 (LIMONENE) | 1.4e-2 | 14.9, 14.0 |
| syn5742_L3 900 s | 5e-4, 1e4 | 0.9 (0.3) | 3.44 | 0.94 | 1.65 (1.77, 1.65, 1.64) | 1022 (0) | 1.61 | 636 | 1.4e-2 (LIMONENE) | 1.4e-2 | 7.5, 7.1 |
| lim726_L2 60 s | 1e-3, 1e4 | 0.9 (0.2) | 3.70 | 0.81 | 2.07 (2.06, 2.07, 2.36) | 7671 (0) | 0.27 | 537 | 6.2e-4 (LIMAOOH) | 6.4e-5 | 11.9, 11.0 |
| lim726_L3 60 s | 1e-3, 1e4 | 0.7 (0.2) | 3.60 | 0.67 | 2.28 (2.34, 2.28, 2.22) | 10353 (0) | 0.22 | 539 | 4.6e-2 (NO) | 3.5e-4 | 5.5, 5.2 |
| syn1353_L3 60 s | 5e-4, 1e4 | 0.8 (0.2) | 5.99 | 0.82 | 5.05 (4.91, 5.05, 5.10) | 10271 (0) | 0.49 | 628 | 4.5e-2 (NO) | 3.7e-4 | 14.0, 22.6 |
| syn4488_L3 60 s | 3e-4, 1e2 | 1.2 (0.4) | 18.79 | 1.06 | 20.73 (21.61, 20.73, 19.61) | 11907 (0) | 1.74 | 972 | 2.5e-2 (NO) | 3.3e-4 | 26.2, 18.4 |
| syn5742_L3 60 s | 3e-4, 1e2 | 1.3 (0.4) | 23.36 | 1.12 | 21.81 (22.05, 21.81, 21.63) | 11852 (0) | 1.84 | 1033 | 2.7e-2 (NO) | 3.9e-4 | 8.0, 11.1 |

At the nominal tolerances (rtol 1e-3, atol 1e4), A fails the gate in four cases:

| Case | E_key | Gate |
|---|---|---|
| syn5742_L3 900 s | 1.7e-2 | 1.7e-2 |
| syn1353_L3 60 s | 5.4e-2 | 5.0e-2 |
| syn4488_L3 60 s | 1.0e-1 | 5.0e-2 |
| syn5742_L3 60 s | 1.2e-1 | 5.0e-2 |

The full rtol sweep is in `logs/report.md`.

### 3.3 Configuration B at the nominal tolerances (protocol table)

Ros3, largest-error norm, step size carried between intervals, rtol 1e-3, atol 1e4.

| Case | t_setup [s] (symbolic) | t_first [s] | t_first_interval [s] | t_warm median [s] (runs) | Steps (rejected) | ms per step | Peak RSS [MB] | E_key (worst) | E_all | 1-min load (first, warm) |
|---|---|---|---|---|---|---|---|---|---|---|
| lim726_L2 900 s | 0.9 (0.2) | 1.65 | 0.78 | 0.10 (0.10, 0.10, 0.10) | 309 (55) | 0.33 | 471 | 4.4e-3 (LIMONENE) | 3.2e-3 | 16.9 |
| lim726_L3 900 s | 0.6 (0.1) | 1.68 | 0.70 | 0.36 (0.36, 0.35, 0.36) | 1421 (370) | 0.25 | 464 | 2.0e-3 (LIMONENE) | 1.3e-3 | 10.2 |
| syn1353_L3 900 s | 1.0 (0.2) | 2.76 | 1.00 | 0.77 (0.74, 0.81, 0.77) | 1412 (371) | 0.55 | 515 | 2.6e-3 (LIMONENE) | 1.7e-3 | 16.1, 15.7 |
| syn4488_L3 900 s | 1.2 (0.4) | 4.61 | 1.13 | 2.50 (2.50, 2.51, 2.40) | 1410 (371) | 1.78 | 624 | 2.5e-3 (LIMONENE) | 2.5e-3 | 13.1, 11.9 |
| syn5742_L3 900 s | 0.9 (0.3) | 4.65 | 1.00 | 2.70 (2.70, 2.70, 2.82) | 1408 (371) | 1.92 | 634 | 2.6e-3 (LIMONENE) | 2.6e-3 | 7.1, 6.4 |
| lim726_L2 60 s | 0.9 (0.2) | 2.41 | 0.82 | 0.63 (0.63, 0.63, 0.63) | 1487 (3) | 0.42 | 536 | 9.8e-5 (NO) | 3.1e-4 | 14.8 |
| lim726_L3 60 s | 0.6 (0.1) | 2.84 | 0.73 | 1.23 (1.25, 1.23, 1.22) | 3940 (1753) | 0.31 | 533 | 4.3e-3 (NO) | 5.3e-4 | 5.2, 5.1 |
| syn1353_L3 60 s | 1.0 (0.2) | 4.56 | 1.02 | 2.58 (2.67, 2.56, 2.58) | 3940 (1754) | 0.65 | 632 | 4.3e-3 (NO) | 5.3e-4 | 21.5, 28.2 |
| syn4488_L3 60 s | 1.5 (0.4) | 10.25 | 1.14 | 7.85 (7.84, 7.85, 7.88) | 3941 (1755) | 1.99 | 957 | 4.3e-3 (NO) | 5.3e-4 | 21.5, 16.9 |
| syn5742_L3 60 s | 1.3 (0.4) | 12.54 | 1.09 | 10.87 (10.45, 10.87, 10.89) | 3941 (1755) | 2.76 | 1136 | 4.3e-3 (NO) | 5.3e-4 | 9.9, 9.2 |

ms per step counts accepted steps only; a rejected attempt costs a factorisation and three
solves but no Jacobian.

**Repeat at the end of the session (load 5.7 to 6.9):**

| Config | lim726_L3 900 s | syn5742_L3 900 s | lim726_L3 60 s | syn5742_L3 60 s |
|---|---|---|---|---|
| A | 0.24 s | 1.65 s | 2.37 s | 19.99 s |
| B | 0.35 s | 2.73 s | 1.33 s | 9.37 s |

The first runs measured 0.22, 1.65, 2.28 and 21.81 s (A) and 0.36, 2.70, 1.23 and 10.87 s
(B). They agree within 10 % (A) and within 14 % (B).

**Work per day** (from the "work" field of the `_timing.json` files):

| Run | Accepted steps | Rejected | Right-hand sides | Jacobians | LU factorisations | Linear solves |
|---|---|---|---|---|---|---|
| A, lim726_L3 900 s | 1,124 | 0 | 2,344 | 1,124 | 1,124 | 3,372 |
| A, syn5742_L3 900 s | 1,022 | 0 | 2,140 | 1,022 | 1,022 | 3,066 |
| A, syn5742_L3 60 s | 11,852 | 0 | 25,144 | 11,852 | 11,852 | 35,556 |
| B, syn5742_L3 900 s | 1,408 | 371 | 3,188 | 1,408 | 1,779 | 5,337 |
| B, syn5742_L3 60 s | 3,941 | 1,755 | 9,638 | 3,941 | 5,696 | 17,088 |

### 3.4 Cost of the pieces of one step

`logs/pieces.jsonl`. Each piece is jitted on its own, best of 9 runs, at a test state with
h = 10 s; load 6.7 to 7.8. The step estimate is 2 right-hand sides + Jacobian + LU +
3 solves (+ the reordering of the LU values for the solves). In the full program one step
measures 0.19-0.25 ms at 726 species and 1.6-1.8 ms at 5742.

| Species | Order | LU entries | Update operations | Levels (LU, forward, backward) | f [ms] | J [ms] | LU [ms] | Solve [ms] | Step estimate [ms] | SuperLU factor / solve [ms] |
|---|---|---|---|---|---|---|---|---|---|---|
| 726 | Markowitz | 7,814 | 19,511 | 48, 16, 17 | 0.042 | 0.043 | 0.079 | 0.061 | 0.40 | 1.25 / 0.028 |
| 726 | min degree | 8,971 | 41,757 | 60, 51, 49 | 0.037 | 0.041 | 0.096 | 0.076 | 0.45 | 1.20 / 0.027 |
| 1353 | Markowitz | 14,743 | 37,220 | 62, 19, 18 | 0.060 | 0.057 | 0.126 | 0.068 | 0.53 | 4.65 / 0.057 |
| 4488 | Markowitz | 49,303 | 126,191 | 72, 19, 17 | 0.159 | 0.162 | 0.465 | 0.155 | 1.45 | 12.0 / 0.156 |
| 4488 | min degree | 59,810 | 564,754 | 132, 119, 118 | 0.171 | 0.165 | 0.814 | 0.440 | 2.68 | 12.0 / 0.154 |
| 5742 | Markowitz | 63,062 | 161,447 | 78, 19, 17 | 0.219 | 0.187 | 0.553 | 0.182 | 1.77 | 21.6 / 0.296 |
| 5742 | min degree | 76,677 | 793,447 | 147, 133, 132 | 0.198 | 0.194 | 1.107 | 0.580 | 3.48 | 21.2 / 0.298 |

The SuperLU factor time includes its own column ordering on every call. With the
Markowitz order, compiling the LU takes 0.25 to 0.47 s and compiling the solve 0.21 to
0.24 s.

## 4. What was hard

1. **The Pro plan's sparse LU prototype was not available.** It was not in any branch,
   worktree or scratch folder of TOMAS-JAX. The symbolic analysis, the level schedule and
   the numeric kernels were rewritten from the description in architecture.md section 8.
2. **The elimination order sets the cost, not the flop count alone.** Minimum degree on the
   symmetrised pattern gave the plan's fill (8,971 entries at 726 species). It needed 41,757
   updates and 50 solve levels per triangle. Markowitz on the unsymmetric pattern gives
   7,814 entries, 19,511 updates and 16 to 17 levels. On the CPU each level costs a few
   microseconds of kernel launch no matter its size, so the level count dominates the
   solves.
3. **Per-level XLA overhead.** The first version (scattered rows, separate scaling of L,
   unique-target scatters) spent 0.73 ms per solve at 5742 species. Three changes brought
   the solve to 0.18 ms and the LU from 1.97 to 0.55 ms (together with the order):
   - storing each level's rows next to each other;
   - gathering the L and U values into solve order once per factorisation;
   - folding the division by the pivot into the update.
4. **Coefficients.** In the first Rodas3 transcription two coefficients of the stage
   matrix were swapped, and it converged with order 1. It was caught by the order test, which now
   checks every tableau (`logs/order_test.json`). Ros3, Rodas3, Ros2 and Rodas4 all show
   their nominal orders.
5. **The same tolerances do not give the same accuracy.**
   - Rodas3 at rtol 1e-3 failed the lim726_L3 900 s gate (3.2e-2) where Ros3 passed
     (1.6e-2).
   - The root-mean-square norm hides the few species that matter (NO at sunset, LIMONENE
     after 11 e-folds of decay), and hides them more as the mechanism grows (section 1,
     item 5).
   - Tightening rtol barely changes the step count at 60 s coupling, because the steps are
     spent ramping up from the initial step. The gate at 4488 and 5742 species was only
     reached by lowering atol to 1e2.
6. **Carrying the step size is not free.**
   - With the root-mean-square norm at rtol 1e-3, carrying gave one 60 s step per interval
     and E_key 0.38 (Rodas3, lim726_L3).
   - At rtol 1e-5, Ros3 passed the lim726_L3 60 s gate only just: E_key 4.6e-2 with 2,502
     steps and 957 rejections, 0.95 s with the earlier kernels.
   - With the largest-error norm it passes every case at the nominal tolerances with
     errors about 10 times smaller, but about 30 % of the attempts are rejected.
7. **Shared machine.** The 1-minute load swung between 5 and 30. All runs were interleaved
   with the dense and scipy reruns, and the key timings were repeated at the end (within
   14 %).

## 5. Limits

1. **Synthetic mechanisms.** The 1353, 4488 and 5742 problems are copies of the limonene
   chemistry (BENCHMARK.md section 2.3). A real full-MCM mechanism will have more fill, more
   levels and maybe more steps. The level and update counts above must be measured again on
   a real export.
2. **Rejected attempts in B.** About a third of B's attempts at 60 s coupling are rejected.
   A smarter first step per interval (not tried beyond fixed fractions) could save up to
   30 %.
3. **Not tuned further.** The right-hand side and the Jacobian are plain sorted segment
   sums; together they are a third of a step at 5742 species. No batching (vmap) or GPU was
   tried.
4. **One process, one thread.** XLA's defaults were used; the levelised kernels are small
   and run effectively on one core.
5. **Gas chemistry only,** as in the benchmark (no aerosol coupling).

## 6. Code layout

```
jax-spike/
  README.md                this file
  spike/
    paths.py               finds the benchmark (TOMAS_GAS_BENCH, ../bench, or ../julia_bench)
    chem.py                problem -> JAX right-hand side, Jacobian into the LU layout,
                           interval rate constants and post-interval rules
    ordering.py            Markowitz elimination order
    sparse_lu.py           minimum-degree order, symbolic fill, level schedules;
                           numeric LU (make_factor, make_factor_v2) and triangular solves
                           (make_solve, make_solve_v2); optional dense tail block
    rosenbrock.py          Ros2, Ros3, Rodas3, Rodas4 tableaux (KPP); one-interval integrator
                           (lax.while_loop, KPP step control, RMS or max norm, initial step
                           by Hairer's rule or carried in)
  scripts/
    run_jax_sparse.py      benchmark protocol runner (fresh process; t_setup, t_first,
                           t_first_interval, warm runs, work, peak RSS, uptime, accuracy);
                           writes results/<tag>_<problem>_dt<coupling>{.json,_hourly.npz,_timing.json}
    sweep.sh               runs a list of cases, one bounded process each, one summary line each
    sweep_tol.sh           rtol sweep of configuration A over the 10 cases
    final_batch.sh         final timings: A and B interleaved with dense and scipy reruns,
                           the nominal runs, the scan runs, and the end-of-session repeat
    make_report.py         markdown tables from the result files (logs/report.md)
    test_pieces.py         unit-test vectors, LU residual, SuperLU comparison, piece timings
    test_order.py          convergence order of each Rosenbrock tableau
    test_orderings.py      fill, updates and levels for minimum degree and Markowitz
    test_lu.py, test_solve_v2.py, test_factor_v2.py   checks of the earlier kernel versions
  results/                 run, hourly and timing files (32 MB; *_timing.json are small)
  logs/                    sweep_tol.log, explore*.log, final_batch.log, pieces.jsonl,
                           order_test.json, report.md (explore.log's first runs used the
                           minimum-degree order and the first kernels)
```

## 7. Reproduce

The Python environment is the benchmark's (`_work/julia/venv`; see BENCHMARK.md section 10).
The problem and reference files must exist in `_work/julia/bench`. From the committed copy
in TOMAS-JAX (`prototypes/jax_sparse_spike`), set `TOMAS_GAS_BENCH` to a folder that holds
them.

```
cd _work/julia/jax-spike/scripts
PY=../../venv/bin/python

# checks
$PY test_order.py                                   # Rosenbrock orders
$PY test_pieces.py syn5742_L3                       # unit-test vectors, LU, piece timings

# one case, configuration A (matched) and B (nominal)
$PY run_jax_sparse.py syn5742_L3 900 --rtol 5e-4 --tag final
$PY run_jax_sparse.py syn5742_L3 60 --h0 carry --norm max --tag best
$PY run_jax_sparse.py syn5742_L3 60 --h0 carry --norm max --mode scan --tag best_scan

# everything in this README (about 25 min each, every run bounded by gtimeout)
./sweep_tol.sh
./final_batch.sh
$PY make_report.py > ../logs/report.md
```

Runner options: `--method ros3|rodas3|rodas4|ros2`, `--rtol`, `--atol`,
`--norm rms|max`, `--h0 hairer|carry`, `--carry-scale`, `--mode loop|scan`,
`--order markowitz|mindeg`, `--factor v1|v2`, `--solve v1|v2`, `--tail N` (dense last block,
with factor v1), `--warm`, `--max-wall-s`, `--tag`.
