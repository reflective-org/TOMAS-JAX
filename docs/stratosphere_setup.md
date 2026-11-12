# Stratospheric model setup

Model settings for stratospheric runs that differ from the library defaults. The defaults stay as they are for Fortran parity; stratospheric runs set these explicitly. They are defined in `sensitivities/dunne_nuc/common.py` and used by every script of the nucleation sensitivity study ([nucleation_sensitivity.md](nucleation_sensitivity.md)).

## Settings

| Choice | Setting | Effect (217 K, SO₂ 1 ppbv, 6 h) |
|---|---|---|
| Nucleated cluster density (`NucleationParams.rho_nuc`) | 1770 kg/m³ (`common.NUC_PARAMS`), the density that sets the grid's 1.7 nm lower edge, instead of the library default 1350 | N(>3 nm) +0.8% (40 bins) and +7.8% (80), with the sparse-bin change below; see [Cluster density](#cluster-density-matched-to-the-grid) |
| Nucleated cluster composition (`NucleationParams.org_frac`) | 0, pure sulfate (`common.NUC_PARAMS`), instead of the library's 10% organic | N(>3 nm) −2%; see [Pure sulfate clusters](#pure-sulfate-clusters) |
| Sparse-bin density (`make_step(sparse_density=...)`) | 1770 kg/m³ (`common.SPARSE_DENSITY`, the grid density) instead of the Fortran 1000/1500 fallbacks, which per-cell thresholds trigger in sparse stratospheric bins. `None` keeps Fortran parity. Was 1700 (H₂SO₄ solution) until 2026-11-04. | at 1700: condensation sink −8%; N(>3 nm) +6.6% (40 bins) and −1.2% (80) |
| Water uptake | `WATER_SCHEME_H2SO4` (Tabazadeh 1997: 68.6 wt% H₂SO₄ at RH 2%) instead of the bisulfate fit (2.4% water) | N(>3 nm) +55% (40 bins), +83% (80); N_total −20%; H₂SO₄ −16% |
| SABRE sizing density | pure H₂SO₄, 1836 kg/m³ (`calc_density` gives 1902) | N(>3 nm) +7% |
| Fion, NH₃, organics | 10 ion pairs cm⁻³ s⁻¹, NH₃ = 0, org_conc = 0 | |
| OH | constant 1e6 cm⁻³ | an upper bound at high SO₂ |

`common.NUC_PARAMS_S2` makes the same two cluster changes to the Dunne Table S2 coefficients.

**Riccobono stays off.** At 217 K the Yu 2017 correction, extrapolated from 278 K, scales the organic rate by about 3×10⁸, so stratospheric runs must keep org_conc at 0.

## Cluster density matched to the grid

**The mismatch.** The nucleated cluster is d = 1.7 nm (`r_nuc_nm = 0.85`). The grid's lower edge `XK0` is also 1.7 nm, but computed at 1770 kg/m³ (`core/config.py`), while the library cluster density is 1350 (`nucleation.py`, from the Fortran `nucleation_driver.f`). The cluster mass is then 1350/1770 = 0.76 of xk[0].

**What MNFIX does with it.** Nucleation adds the clusters to bin 0. If bin 0's average mass falls below xk[0], MNFIX phase 2 keeps the mass and sets the number to M/(1.2·xk[0]). No mass is removed: the new particles are in effect born at 1.2·xk[0] = 1.58× the cluster mass (about 15% larger in diameter), and there are 0.76/1.2 = 0.64 as many of them for a bin of fresh clusters. With heavier particles already in bin 0 the reset may not fire. This is a grid bookkeeping clamp, not coagulation: no kernel or rate is involved, and the size the particles get is set by the grid, not by the cluster. The Fortran has the same mismatch, worse: its standard grid (`initbounds.f`, Mo = 1.5625e-23 kg, about 2.6 nm) puts the cluster at 0.22 of the edge.

**The setting.** `common.NUC_PARAMS = DEFAULT_NUCLEATION_PARAMS._replace(rho_nuc=1770, org_frac=0)` (and `NUC_PARAMS_S2` for Table S2), so the cluster mass equals xk[0] (to 2e-16, just above it). The library default stays 1350, for Fortran parity.

**Where the reset fired** (comparing runs at 1770 with runs at 1350): in a 1 h run at 1 ppbv (N +50%; elasticity of N to r_nuc and ρ_nuc +2.8 and +0.94 → about −0.01), in the background mid-latitude and plume scenarios, and at ≥ 100 ppbv; hardly in the TTL scenario, whose N changed by about 4%. It fires only when bin 0 holds mostly fresh clusters. (The sparse-bin density changed from 1700 to 1770 at the same time, so the comparisons include that too.)

**The cluster mass still matters where particles grow,** now not through the reset, plausibly because heavier clusters start closer to 3 nm (not tested). In the plume scenario the elasticity of N(>3 nm) to r_nuc is about +6 (40 bins) and +8 (80), larger than at 1350 (+4.4 and +5.9), and ρ_nuc has exactly a third of it.

**The cluster sits on the bin edge.** A downward perturbation of r_nuc or ρ_nuc could switch the reset back on. In a 1 h run it does not: condensation adds mass to bin 0 within each step, so FD from either side agrees with AD. A run with ρ_nuc = 1.05 × 1770 gives the same N and elasticities. Low-growth cases could still hit the edge.

**Densities in use** (none of these differences affects the particle count):

| Density [kg/m³] | Used for | A cluster of mass xk[0] appears as |
|---|---|---|
| 1770 | grid edges, cluster mass, sparse-bin diameter (kernel, condensation sink) | 1.70 nm |
| 1836 | pure H₂SO₄, dry diameter for SABRE binning (`sabre_bins.py`) | 1.68 nm |

## Pure sulfate clusters

The library's `org_frac = 0.1` comes from TRACER_SOM-TOMAS `nucleation.f`: 90% SO₄, 10% organic at `SRTORGLAST`, no source given. It adds organic mass to every new particle whether or not there is organic vapour. Stratospheric runs have no organic vapour and nucleate only through Dunne (H₂SO₄–H₂O), so the clusters are pure sulfate. Each cluster then takes all of its mass, 28 H₂SO₄ molecules rather than 25, from the gas.

Effect (study runs, against 10% organic): N after 1 h within 0.05%; N(>3 nm) after 6 h −2%; in a 10 ppmv plume after 48 h, N(>3 nm) +11.5% (40 bins) and +15% (80); in the TTL, N(>12 nm) −4 to −5%.

## History

| Date | Change | Earlier results |
|---|---|---|
| 2026-11-04 | cluster density 1350 → 1770, sparse-bin density 1700 → 1770 | `sensitivities/dunne_nuc/results/rho1350/` |
| 2026-11-05 | cluster `org_frac` 0.1 → 0 | `sensitivities/dunne_nuc/results/orgfrac0.1/` |
