# MCM inorganic + CH4 + sulfur subset vs SANDBOX (planning prototype, 2026-10-16)

Question: can the MCM path give the pro simulator the SANDBOX species (O3, NOx, HOx,
HONO, HNO4, H2O2, N2O5, CH4, SO2 -> SO3 -> H2SO4) with comprehensive SO2 chemistry,
and how does it compare with SANDBOX's JPL 19-5 gas chemistry (halogens off)?

Short answer: yes. The subset parses, is atom-balanced (every imbalance is a species MCM
omits by convention), and runs in `tomas_jax.gaschem` unchanged in about 1 s per case.
Against SANDBOX with the same photolysis, the differences come from three things, each
traced to specific reactions and rate constants below:
(1) mechanism coverage (MCM lacks N2O5 and HNO4 photolysis, which matter in the
stratosphere; SANDBOX lacks HNO4 and N2O5 thermal decomposition, the O(3P) channel of O3
photolysis, and the CH4 -> CH3O2 -> HCHO -> CO chain, which matter in the troposphere);
(2) the OH + SO2 rate constant (MCM/IUPAC vs JPL 19-5, 38 % apart at 220 K, 50 hPa),
which sets the stratospheric SO2 lifetime; and (3) nothing in photolysis itself, since
both models use one shared TUV-x J table.

Nothing here is model code. No MCM files were downloaded; the mechanism is cut from
`tomas_jax/gaschem/mechanisms/limonene_mcm.eqn`. SANDBOX is only read.

## 1. The mechanism file

`../mcm_inorganic_sulfur.eqn`, built by `../build_mcm_inorganic_sulfur.py`.

* Seeds {O3, NO, NO2, CO, CH4, SO2, H2O2, HONO, HNO3, N2O5, H2, H2O}. H2O is not a state
  species in MCM (it sits inside rate expressions). The closure keeps every reaction
  whose reactants are all reachable from the seeds, repeated to a fixed point, so no
  limonene-derived species can enter (697 of the 725 species are excluded).
* Result: 28 species, 67 reactions, 12 photolysis reactions, RO2 = {CH3O2}. Labels, rate
  expressions and `J(...)` names are verbatim.
* Sulfur: `<30> O + SO2 = SO3` (4.0e-32 exp(-1000/T) M), `<31> OH + SO2 = HSO3` (KMT12),
  `<32> HSO3 = HO2 + SO3` (1.3e-12 exp(-330/T) O2), `<35> SO3 = SA` (1.2e-15 H2O). The ten
  Criegee + SO2 reactions of the limonene file (CH2OO, LIMBOO, ...) are not reachable:
  their Criegee intermediates come only from alkene ozonolysis. They come back with any
  alkene module (results/mcm_inorganic_sulfur_build.json lists them).
* Atom balance (results/mcm_inorganic_sulfur_build.json): 26 reactions balance exactly;
  the other 41 are each explained by an MCM convention: O2, H2O or CO2 omitted as
  products or folded into k; NA and SA declared IGNORE (aerosol sinks); or RO2
  pseudo-first-order reactions that do not write the partner peroxy radical. None is
  left unexplained. Sulfur closes to 1e-15 in every run.
* Compared with the sibling module `../modules/mcm_inorganic_c1_sulfur.eqn` (seeded from
  every carbon-free species), this file lacks only `<2676> CH4 + CL`, since Cl is not a
  seed.

## 2. Cases (`cases.json`, resolved in `cases_resolved.json`)

| | (a) troposphere | (b) lower stratosphere, SAI-like |
|---|---|---|
| T, p | 298.15 K, 1013.25 hPa | 220 K, 50 hPa (TUV-x box altitude 20.6 km) |
| H2O | RH 50 % = 15 635 ppm (Buck formula of `gaschem.environment`) | 4 ppm (SANDBOX `p55_210K_4ppm.yaml`) |
| place, date | 40 N, 0 E, 1 July 2026, start 00 UTC, 3 days | 30 N, 0 E, 21 June 2026 (SANDBOX `summer_30N_noon.yaml`), start 00 UTC, 5 days |
| start [ppb] | O3 40, NO 0.2, NO2 0.8, SO2 1, CH4 1900 | O3 1670, NO 0.5, NO2 0.5, HNO3 4.55, N2O5 0.02, OH 5e-4, HO2 3e-3 (SANDBOX 60 hPa preset in `driver.py`), CH4 1600 (SANDBOX), SO2 10 (injection) |
| NOx | held at 1 ppb (NO, NO2 rescaled after every 300 s, ratio kept) | free |
| analysis window | day 1 | day 5 |

Why NOx is held in (a): in a free run with no VOC, noon OH is about 2e7 cm-3, OH + NO2
removes 90 % of the NOx by local noon and the case is no longer a 1 ppb NOx case.
Holding NOx is the usual constrained-box treatment of a steady NOx source. Halogens,
C2H6 and aerosol surface area are zero in every run.

## 3. How the two models are run

* Photolysis (`make_jtables.py`): one J(t) table per case from the TUV-x JAX port
  (tuvx-jax e5d9032, `examples/tuv_5_4_no_aerosol.json`, `branching=True`) at the box
  altitude, which is the US Standard Atmosphere height of the box pressure (SANDBOX's
  adapter rule). SZA from `tuvx_photolysis.geometry`, J tabulated on a 0.25 deg SZA grid,
  output every 150 s. MCM reads its 12 `J_*` through the `TUVX_DIRECT` map of
  `gaschem.photolysis`; SANDBOX reads its halogen-free channels by equation through the
  `REACTION_MAP` of its `tuvx_photolysis_adapter.py`. Both therefore see identical J for
  every channel they share.
* MCM (`run_mcm.py`, mech-wt venv): `parse_eqn`, `RateModel`, `Mechanism`,
  `integrate_interval` (Kvaerno3, rtol 1e-3, atol 1e4 cm-3), k frozen per 300 s, J at
  the interval midpoint, as in `box/driver.py`. MCM's fixed aerosol sinks `<33> HNO3 = NA`
  (6e-6 s-1) and `<34> N2O5 = 2 NA` (4e-4 s-1) are set to zero; `<35> SO3 = SA` stays.
* SANDBOX (`run_sandbox.py`, SANDBOX venv, SANDBOX git 888e5ff): `reactions.MECHANISM.dCdt`
  with an `Env` built as `reactions.build_env` builds it, `sulfur_chain = True`, J from
  the shared table, `opt = 0` (explicit O(1D) chemistry), SciPy BDF rtol 1e-4. One change
  from `build_env`: the uptake gammas are set to 0 instead of computed, because
  `hetgammas_jpl00` divides by zero when HCl = 0; with SA = 0 every heterogeneous rate is
  zero anyway. `opt = 1` (the water-split O(1D) shortcut) is run as a sensitivity.
* Gap-fill diagnostics (added in memory, not proposed edits): "MCM + J(N2O5), J(HNO4)"
  adds N2O5 -> NO2 + NO3, HO2NO2 -> HO2 + NO2 and HO2NO2 -> OH + NO3 with the TUV-x J;
  "SANDBOX + fill" adds O3 -> O2 + O(3P) (TUV-x J), HNO4 -> HO2 + NO2 (MCM KMT10) and
  N2O5 -> NO2 + NO3 (MCM KMT04).

## 4. Results

Definitions (window W = the analysis day; noon = minimum SZA in W; [X] in cm-3):
SO2 lifetime tau = 1 / (k_OH+SO2 <[OH]>_W) (the decay of SO2 in the run gives the same
value to 1 %); H2SO4 production at noon = d[H2SO4]/dt (H2SO4 = SA in MCM); H2SO4 per day
= H2SO4 made over W; NOy = NO + NO2 + NO3 + 2 N2O5 + HNO3 + HONO + HNO4 (+ CH3NO3 +
CH3O2NO2 in MCM); P(O3) = integral over W of k(HO2+NO)[HO2][NO] (+ k(CH3O2+NO->NO2)
[CH3O2][NO] in MCM). Full table with every variant and the O3 budget terms:
`results/comparison.md`; machine-readable: `results/comparison.json`.

### (a) Troposphere, 298 K, 1 atm, NOx 1 ppb, SO2 1 ppb (day 1)

| Quantity | MCM, matched start | MCM + CO 100, H2 500 ppb | SANDBOX (opt 0) | SANDBOX (opt 1) | SANDBOX + fill |
|---|---|---|---|---|---|
| OH noon / 24 h mean [cm-3] | 3.6e7 / 1.1e7 | 2.8e7 / 8.6e6 | 2.2e7 / 6.2e6 | 2.5e7 / 7.0e6 | 3.3e7 / 9.5e6 |
| HO2 noon [cm-3] | 3.8e8 | 5.6e8 | 9.4e7 | 1.1e8 | 1.9e8 |
| O3 change [ppb/day] | +21.6 | +28.2 | -1.4 | -1.5 | +2.7 |
| P(O3) [ppb/day] | 35.6 (HO2 23.6, CH3O2 11.9) | 42.5 | 6.2 | 7.0 | 12.3 |
| NO/NO2 noon | 0.39 | 0.35 | 0.52 | 0.52 | 0.48 |
| NOx/NOy noon | 0.24 | 0.29 | 0.24 | 0.23 | 0.26 |
| HNO4, N2O5 noon [ppt] | 3, 0.1 | 5, 0.1 | 628, 276 | 671, 277 | 2, 0.0 |
| k(OH+SO2) [cm3 s-1] | 8.94e-13 | 8.94e-13 | 9.50e-13 | 9.50e-13 | 9.50e-13 |
| SO2 lifetime [days] | 1.2 | 1.5 | 2.0 | 1.8 | 1.3 |
| H2SO4 production noon [cm-3 s-1] | 5.3e5 | 4.4e5 | 3.9e5 | 4.3e5 | 5.2e5 |
| H2SO4 per day [ppt] | 576 | 489 | 395 | 434 | 538 |

### (b) Lower stratosphere, 220 K, 50 hPa, SO2 10 ppb (day 5)

| Quantity | MCM, matched start | MCM + CO 15, H2 500 ppb | MCM + J(N2O5), J(HNO4) | SANDBOX (opt 0) | SANDBOX (opt 1) | SANDBOX + fill |
|---|---|---|---|---|---|---|
| OH noon / 24 h mean [cm-3] | 2.29e6 / 7.6e5 | 2.23e6 / 7.5e5 | 2.32e6 / 7.3e5 | 2.32e6 / 7.2e5 | 2.21e6 / 6.9e5 | 2.32e6 / 7.2e5 |
| HO2 noon [cm-3] | 1.55e7 | 1.59e7 | 8.4e6 | 7.7e6 | 7.3e6 | 7.6e6 |
| O3 change [ppb/day] | -2.6 | -2.5 | -2.7 | -0.9 | -0.8 | -2.7 |
| NO/NO2 noon | 1.25 | 1.25 | 1.26 | 1.39 | 1.39 | 1.39 |
| NOx noon [ppt], NOx/NOy | 662, 0.12 | 661, 0.12 | 1535, 0.27 | 1736, 0.31 | 1742, 0.31 | 1738, 0.31 |
| N2O5, HNO4 noon [ppt] | 642, 73 | 641, 76 | 76, 44 | 81, 81 | 81, 79 | 81, 80 |
| k(OH+SO2) [cm3 s-1] | 3.53e-13 | 3.53e-13 | 3.53e-13 | 4.88e-13 | 4.88e-13 | 4.88e-13 |
| SO2 lifetime [days] | 43.0 | 44.0 | 44.7 | 32.9 | 34.6 | 32.9 |
| H2SO4 production noon [cm-3 s-1] | 1.20e4 | 1.17e4 | 1.22e4 | 1.61e4 | 1.55e4 | 1.61e4 |
| H2SO4 per day [ppt] | 210 | 206 | 203 | 264 | 253 | 264 |
| SO2 loss not via OH | 0.3 % (O + SO2) | 0.3 % | 0.3 % | 7e-4 % (HO2 + SO2) | same | same |

Plots: `fig_trop_timeseries.png`, `fig_strat_timeseries.png` (OH, HO2, O3, NOx or NOx
reservoirs, SO2, H2SO4 made; analysis day shaded) and `fig_rate_ratios.png`
(k(JPL 19-5) / k(MCM) for 12 shared reactions at both conditions).

## 5. Why they differ

Rate constants (`fig_rate_ratios.png`, values in `results/comparison.json`). Most shared
reactions agree within 10 %. The ones that matter:

* OH + SO2 (+M): JPL/MCM = 1.06 at 298 K, 1 atm and 1.38 at 220 K, 50 hPa. MCM's KMT12
  is the IUPAC Troe form k0 = 2.5e-31 (T/300)^-2.6, kinf = 2.0e-12, Fc = 0.53; SANDBOX uses
  JPL 19-5 k0 = 2.9e-31 (T/298)^-4.1, kinf = 1.7e-12 (T/298)^0.2, Fc = 0.6. The steeper JPL
  temperature dependence of k0 is what separates them at 220 K. With OH within 6 %, this
  one constant explains the 43 vs 33 day stratospheric SO2 lifetime and the 25 % gap in
  H2SO4 production. It is the single largest SAI-relevant number in the comparison, and
  which evaluation to use should be a registry option, not a silent default.
* HO2 + NO2 + M: JPL 1.7-1.9 times MCM (JPL Fc = 0.6 vs IUPAC 0.4 and a larger k0).
* HO2 + HO2: JPL 0.73-0.87 of MCM. OH + NO2 + M: JPL 0.77 of MCM at 220 K.
* SO3 + H2O: MCM is first order in H2O (1.2e-15 [H2O]), JPL 19-5 second order
  (8.5e-41 exp(6540/T) [H2O]^2); the SO3 lifetime differs by 92x at 298 K and 3.8x at
  220 K, but even the slower (MCM) SO3 lifetimes, 0.002 s and 200 s, are far shorter
  than the SO2 lifetime, so H2SO4 production equals the SO2 + OH rate in both models.

Missing reactions (each confirmed by a gap-fill run):

* Stratosphere, MCM side: MCM v3.3.1 has no N2O5 or HO2NO2 photolysis (it was designed
  for tropospheric chemistry). N2O5 piles up (642 vs 81 ppt), NOx is 2.6x lower and HO2 2x
  higher. Adding the three TUV-x channels brings MCM to NOx 1535 ppt, N2O5 76 ppt,
  HO2 8.4e6, within 12 % of SANDBOX.
* Stratosphere, SANDBOX side: SANDBOX drops O3 -> O2 + O(3P) (J = 5.2e-4 s-1 at noon,
  5.6x the O(1D) channel) as a "null cycle". It is null for O3 but not for [O]: noon O is
  6x lower than in MCM (3.8e5 vs 2.4e6 cm-3), so the O + NO2 and O + O3 odd-oxygen losses
  are 0.2 ppb/day in SANDBOX against 1.0 (MCM) and 2.0 (MCM with the photolysis gap-fill,
  which has SANDBOX's NOx). With the channel added, SANDBOX's O3 tendency is -2.7
  ppb/day, the same as MCM's. With both gap-fills OH, HO2, NOx and the O3 tendency agree
  within 12 %; what is left is the OH + SO2 constant above.
* Troposphere, SANDBOX side: SANDBOX has no thermal decomposition of HNO4 (MCM KMT10 =
  0.063 s-1 at 298 K) or N2O5 (KMT04 = 0.045 s-1). These are irrelevant at 220 K
  (3e-8 and 6e-8 s-1) but at 298 K HNO4 becomes a permanent HO2 sink (628 ppt at noon)
  and N2O5 a nighttime NOx trap. Adding them (plus the O(3P) channel) raises SANDBOX's
  OH to 3.3e7 (MCM 3.6e7) and the SO2 lifetime falls from 2.0 to 1.3 days (MCM 1.2).
* Troposphere, both: SANDBOX lumps CH4 + OH -> HO2 and has no CO, H2, HCHO, CH3OOH or
  CH3NO3. In MCM, CH3O2 + NO gives a third of the O3 production (11.9 of 35.6 ppb/day) and
  HCHO photolysis adds HO2. This is why SANDBOX + fill still has half of MCM's HO2 and a
  third of its P(O3). Adding 100 ppb CO (which SANDBOX cannot represent) lowers MCM's OH
  by 24 % and lengthens the SO2 lifetime from 1.2 to 1.5 days.
* Minor: O + SO2 (MCM only) is 0.3 % of stratospheric SO2 loss; HO2 + SO2 (SANDBOX, JPL
  upper limit with assumed products) is below 1e-5. O1D + CH4 (SANDBOX only) is
  negligible. `opt = 1` changes SANDBOX OH by 5-13 %.

Photolysis: shared by construction, so it causes no difference here. Two notes for the
real build: (1) the MCM clear-sky J parameterization ([Saunders et al., 2003](https://doi.org/10.5194/acp-3-161-2003))
is a surface fit and must not be used at 20 km; altitude-resolved TUV-x J (as here) is
required. (2) Neither model has O2 photolysis in these runs (SANDBOX disables it, MCM has
no O2 species). This omission is not small for O3 at 20 km (review, 2026-10-16): the TUV-x
port gives J(O2 -> 2 O) = 4.0e-12 s-1 at noon and 7.8e-13 s-1 as a 24 h mean at 20.6 km,
so P(Ox) = 2 J [O2] = 5.4e5 cm-3 s-1 = 28 ppb/day, ten times the net O3 tendencies in the
table. Re-running "MCM + J(N2O5), J(HNO4)" with O2 photolysis added turns the O3 change
from -2.65 to +25.1 ppb/day, but moves the SO2 lifetime only from 44.7 to 45.1 days and
OH by 1 % (`prototypes/mechanism/review/rerun_strat.py`). So the O3 tendency rows compare
mechanisms, not a stratospheric O3 budget; the SO2 rows stand.

## 6. What this means for the pro mechanism

* The MCM path works for the SANDBOX non-halogen species with no change to
  `tomas_jax.gaschem`: parse, rate translation, J mapping to TUV-x and the stiff solve all
  run as they do for limonene, in about 1 s per multi-day case for 28 species.
* MCM alone is not enough for the stratosphere: add N2O5, HNO4 (both channels) and O2
  photolysis (already the main Ox source at 20 km, see section 5), from JPL 19-5 with
  TUV-x J. Halogens (ClOx, BrOx, heterogeneous) come from SANDBOX/JPL as already planned.
* Read the absolute numbers with care: the tropospheric case has no CO or VOC, so its OH
  (2-4e7 cm-3 at noon) and 1.2-2.0 day SO2 lifetimes are far from ambient; the stratospheric
  case is not at NOy steady state on day 5 (HNO3 photolysis lifetime about 17 days from the
  same J table) and has no N2O5 hydrolysis on sulfate, which controls NOx under SAI. Both
  are valid mechanism-to-mechanism tests, not SAI predictions.
* The OH + SO2 rate (IUPAC via MCM vs JPL 19-5) changes the stratospheric SO2 lifetime by
  30 %. The registry should carry both with a named default and the source of each.
* MCM's `<33>`/`<34>` fixed aerosol sinks must be off or replaced by uptake on the TOMAS
  aerosol whenever aerosol is coupled (the tutorial already replaces `<34>`).

Sources: [MCM v3.3.1](https://mcm.york.ac.uk/MCM/) (rate sources: IUPAC evaluations, see
[MCM construction method](https://mcm.york.ac.uk/MCM/about/construction);
[Jenkin et al., 2015](https://doi.org/10.5194/acp-15-11433-2015));
[JPL 19-5](https://jpldataeval.jpl.nasa.gov/) (SANDBOX rates, as transcribed in
SANDBOX `stratchem-jax/reactions.py`); [IUPAC evaluation](https://iupac.aeris-data.fr/).
Rate expressions quoted above are copied from the two codes, not re-read from the
evaluations.

## 7. Reproduce (bounded; about 1 min in total)

```bash
WT=<mech-wt>; PY=$WT/.venv/bin/python; SB=/Users/ali/Documents/GitHub/gas-phase-chemistry/SANDBOX
export TUVX_JAX_ROOT=/Users/ali/Documents/GitHub/aerosol-microphysics/tomas-aaar/tuvx-jax PYTHONPATH=$WT
# venv: uv venv $WT/.venv --python 3.12; (cd $WT && uv pip install --python $PY -e '.[dev,photolysis]')
cd $WT/prototypes/mechanism && gtimeout 180 $PY build_mcm_inorganic_sulfur.py
cd sandbox_cmp && gtimeout 300 $PY make_jtables.py && gtimeout 500 $PY run_mcm.py \
  && gtimeout 590 $SB/.venv/bin/python run_sandbox.py $SB/stratchem-jax && gtimeout 120 $PY compare.py
```

`run_sandbox.py` reads `results/mcm_<case>_matched.json` (for KMT04/KMT10), so run it
after `run_mcm.py`. It also aborts any single run that exceeds 400 s of wall time.

## Assumptions to log for Shantanu

1. The subset comes from the limonene file Shantanu sent, not a fresh MCM export. Its
   inorganic + CH4 block should equal MCM v3.3.1's, but that is not verified against an
   export.
2. Troposphere case: NOx held at 1 ppb; RH 50 %; no VOC besides CH4 (so OH is high,
   2-4e7 cm-3 at noon).
3. Stratosphere case: SANDBOX 60 hPa preset composition used at 50 hPa and 220 K;
   lat/date from SANDBOX's `summer_30N_noon.yaml`, started at 00 UTC instead of noon.
4. Both models: no aerosol (MCM `<33>`, `<34>` off; SANDBOX SA = 0, gammas 0), no
   halogens, J from one TUV-x table at the box altitude.
5. SANDBOX `opt = 0` is the baseline (explicit O(1D)); its scenarios default to `opt = 1`.

## Regenerating the large result files

The per-case time series and photolysis tables in `results/` (`jtable_*.json`, `mcm_*.json`,
`sandbox_*.json`, 0.35-0.75 MB each) are not committed. Run `make_jtables.py`, then
`run_mcm.py` and `run_sandbox.py`, then `compare.py` to rebuild them and `comparison.md`.
