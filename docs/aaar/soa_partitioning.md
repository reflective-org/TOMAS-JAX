# SOA gas–particle partitioning (AAAR tutorial)

`tomas_jax/physics/soa_partitioning.py::soa_partition_step` couples the
kinetic volatility-basis-set (VBS) solver of
[vbs-jax](https://github.com/reflective-org/vbs-jax) to the TOMAS sectional
state. It is the organic half of the aerosol step that all three tutorial
levels share (`box/aerosol_step.py`, see [aerosol_step.md](aerosol_step.md)).

## Physics

Each volatility class *j* has a saturation concentration C\*_j (µg m⁻³). It
is corrected for temperature with Clausius–Clapeyron:

    C*_j(T) = C*_j(298 K) · exp[ −ΔH_vap,j / R · (1/T − 1/298) ]

**Equilibrium (Pankow 1994; Donahue et al. 2006).** The particle fraction of
class *j* is F_j = 1 / (1 + C\*_j / C_OA). C_OA is the absorbing organic mass,
including non-volatile OA:

    C_OA = C_nv + Σ_j C_tot,j / (1 + C*_j / C_OA)

**Kinetics (Zaveri et al. 2014, soacond.f).** The solver never imposes
this equilibrium directly; it relaxes towards it. For each bin *k*:

    psat_jk  = C*_j · (m_jk / m_org,k) · Ke_jk               (Raoult by organic mass fraction + Kelvin)
    Ke_jk    = exp( 4 σ MW_j / (R T ρ_org Dp_k) )
    CS_j     = 2π Σ_k Dp_k n_k K_eff,jk                       (Fuchs–Sutugin gas side, two-film particle side)
    dm_jk/dt ∝ sinkfrac_jk · (p_amb,j − psat_jk)

The gas is depleted analytically towards the sink-weighted equilibrium in
adaptive sub-steps (Δt ≤ ln(100)/(2 max CS), ≤ 900 s, at most `n_substeps`).
Within each sub-step a fixed-point loop handles the Raoult coupling. vbs-jax
[docs/physics.md](https://github.com/reflective-org/vbs-jax/blob/main/docs/physics.md)
has the full algorithm.

Class properties are the vbs-jax defaults: σ = 0.025 N m⁻¹,
ρ_org = 1200 kg m⁻³, Fuller volume 120, D_b = 1e-10 m² s⁻¹, k_c = 0. The
per-class arrays c\*, ΔH_vap and MW are inputs.

## Slot map (`core/config.py`)

| Slot(s) | Name | Gc (vapour) | Mk (particle) | Role in vbs-jax |
|---|---|---|---|---|
| 0 | `SRTSO4` | H₂SO₄ | SO₄ | `m_inorg` |
| 1–15 | `SRTSOA1` … (`N_SOA_CLASSES` = 15) | class vapour | class SOA | `m_org` (first K), else `m_org_nonvol` |
| 16–39 | free organic slots | — | — | `m_org_nonvol` |
| 40 | `SRTNIT` | — | NO₃⁻ (as NO₃, MW 62) | `m_inorg` |
| 41 | `SRTORG_NUC` | — | nucleated organic | `m_org_nonvol` |
| 42 | `SRTNH4` | NH₃ | NH₄ | `m_inorg` |
| 43 | `SRTH2O` / `SRTSO2` | SO₂ | H₂O | `m_inorg` |

`K = len(cstar_ref)` classes use slots `SRTSOA1 … SRTSOA1+K−1`. **Level 1** uses
K = 1, with the user's surrogate vapour in slot 1. **Level 2** uses up to 15
classes, filled by the gas-chemistry lumping. Absorbing OA that never
evaporates is everything in the organic block (1–41) outside the K classes
and nitrate: the nucleated organic in slot 41, plus any mass in class slots
≥ K. Nitrate is inorganic, so it does not dissolve SOA.

## Units and hooks

TOMAS carries Nk [#/cell], Mk and Gc [kg/cell] and boxvol [cm³]. vbs-jax
works per m³. The adapter divides by `boxvol·1e-6` only for the vbs-jax
`ParticleState` and the class vapours. The full per-cell `(Nk, Mk)` goes
through vbs-jax as `aux`, so SO₄, NH₄ and the other columns never make a
unit round trip.

- `redistribute` hook: writes the new class masses into Mk. With
  `bin_transfer='ppm'` (default) that is all: the particles keep their bin
  during the vbs-jax sub-steps and are moved once afterwards (below). With
  `bin_transfer='mnfix'` (the previous scheme) the hook also calls
  `mnfix_jax` after every sub-step. vbs-jax takes the vapour change from
  the particle change after the hook. With `'ppm'` the remap conserves
  every species, and the SOA that the top-bin cap or the MNFIX floor
  afterwards takes out of the particles is added back to the vapour, so
  gas + particle mass is conserved per class with both settings (tested
  to 1e-12 per step, including particles at the top of the grid).
- `geometry_fn` hook: `calc_wet_diameters` (`condensation_sink.py`), the
  same wet diameter and density as the H₂SO₄ condensation sink, from each
  bin's actual mean mass. A bin uses its own mean mass if Nk >
  `NEPS_CONDSINK` = 1e10 #/cell (1 cm⁻³ at the tutorial boxvol of
  1e10 cm³), else a default size; it still takes part (n_tiny below).
- `n_tiny` = 1e-10 #/cell, the value of the tomas-jax solver vbs-jax was
  extracted from.

## Bin transfer

The SOA a bin gains or loses changes the mass of its particles in place.
`soa_partition_step(..., bin_transfer=...)` (static; `AerosolStepConfig.
growth_bin_transfer` in the box model) chooses how those particles then
move between the fixed bins:

- `'ppm'` (default): after vbs-jax returns, `physics/ppm_remap.py::
  remap_mass_change(Nk, Mk_before, Mk_after, xk)` moves number and every
  species to the bins they now belong to. Each bin's displacement is the
  log change of its mean dry mass (growth up, evaporation down). The bin's
  PPM sub-bin profile, the one H₂SO₄ condensation uses, is carried by that
  displacement, however many bins it is, and deposited onto the fixed bins
  by exact overlap integrals; each species' mass fraction is linear inside
  the bin (limited), so particles keep the composition that goes with their
  size. Particles whose mass did not change stay exactly in place, even
  while others grow or shrink past them: next to particles that change the
  same way their displacement is 0 at both edges, and a bin whose
  displacement is a local extremum (for example one unchanged bin between
  growing neighbours) gets no slope across it (minmod). Then `cap_top_bin` keeps SOA that
  would push the top bin past the grid in the gas, and MNFIX restores its
  floor in emptied bins (SOA it deletes there returns to the gas).
  Conservative to round-off, never negative, smooth (one extremum for a
  growing or shrinking mode at 40, 80 and 160 bins in the presets), and a
  grow-then-evaporate round trip comes back to the moving-section answer.
- `'mnfix'`: MNFIX after every vbs-jax sub-step. It combs a growing mode
  (3 / 13 / 30 interior extrema at 40 / 80 / 160 bins in Level 1 with SOA
  only) and makes the size distribution depend on resolution. Kept for
  comparison.

The partitioning physics (fluxes, Raoult, Kelvin, the absorbing floor,
class slots) is the same for both. Algorithm, measurements and
limitations: [aerosol_step.md](aerosol_step.md#bin-transfer-of-soa-and-nitrate-growth_bin_transfer).

## Absorbing floor (zero-OA bootstrap)

With no organic mass in a bin, vbs-jax sets psat = 0. Any vapour then
condenses, even at C\* = 1000 µg m⁻³: about 12% of a 2 µg m⁻³ C\* = 1000
vapour on pure sulfate seed in the first 60 s. It evaporates in the next
step and re-condenses every time the bin returns to exactly zero, giving
a sawtooth. The adapter therefore adds
`SOA_ABSORBING_FLOOR` (1e-4) × (bin dry mass outside the SOA classes) to
`m_org_nonvol`. This mass exists only in the Raoult denominator, never in
Mk. Effects:

- A subsaturated vapour (Σ C_tot/C\* < 1, no OA) stays in the gas phase,
  as Pankow predicts. The floor holds at most S·floor (S = saturation ratio).
- A supersaturated vapour starts slower. For C\* = 1, C = 2 µg m⁻³ on
  1000 cm⁻³ sulfate, the particle SOA is 6% below the no-floor run at
  20 min and 1.4% below at 30 min. Low C\* (S ≫ 1) is not delayed.
- Equilibrium C_OA is biased by at most 1e-4 × seed dry mass
  (ASSUMPTIONS.md AC1).

## Usage

```python
from tomas_jax.physics.soa_partitioning import soa_partition_step
Nk, Mk, Gc, diag = soa_partition_step(
    Nk, Mk, Gc, xk, temp, pres, boxvol, dt,
    cstar_ref, dhvap, mw_org, alpha_org,     # (K,), (K,), (K,), scalar
    bin_transfer='ppm')                      # static: 'ppm' or 'mnfix'
assert diag.time_integrated >= dt            # else raise n_substeps
```

Under `jax.jit`, pass `bin_transfer` (and `n_substeps`, `n_iter`) as static
arguments.

c\*, ΔH_vap (kJ mol⁻¹), MW and α are traced, so the step compiles once
per (nbins, K). The adapter applies no c\* or ΔH_vap rule. The tutorial
class arrays come from `tomas_jax/gaschem/volatility.py`: log10 c\*(298 K) =
−3 … 4 in half-decades and ΔH_vap = 131 − 11·log10 c\* (Epstein et al.
2010, SOM-TOMAS form). The species' c\* come from the Jathar per-species table
(Nannoolal via PyCHAM, 298.15 K) in Levels 2 and 3, with Li et al. (2016) for species
the table does not list (ASSUMPTIONS A8, `docs/aaar/gaschem.md`). The tests build their own arrays the same way, with
MW = 200 g mol⁻¹.

## Validation (`tests/test_aaar_aerosol.py`)

| Test | Result |
|---|---|
| Single class, C\* = 10, C_tot = 10, C_nv = 2 µg m⁻³, 1.5 µm seeds, 4 h | SOA within 0.35% (from gas) / 0.2% (from particles) of Pankow |
| 6 Donahue classes (0.01–1000) vs `vbs_jax.vbs_equilibrium` | per class within 0.5% |
| Zero-OA seed, C\* = 0.01 and 1000 | low C\* > 50% condensed; high C\* > 99% gas; no spikes |
| Per-class gas + particle, all processes, 600 s | conserved to 1e-10 (with sources) |
| The above with `bin_transfer` = `'ppm'` and `'mnfix'` | same results within the tolerances |
| SOA evaporation, c\* = 10, no vapour (`test_aaar_soa_ppm.py`) | mode returns to the seed size; class budget 1e-10 every step; one extremum with PPM |
| Particles in the top three bins, 2000 µg m⁻³ of c\* = 0.01 vapour, 12 × 600 s, both settings | class budget 1e-12 every step; with PPM the top-bin mean stays inside the grid and no sulfate is lost |
| +30 K at once after 6 h of growth (c\* = 1, 40 bins), both settings | class budget 1e-12 every step, mode moves down monotonically; with PPM number conserved and one extremum |

The equilibrium residuals come from the Kelvin term (Ke − 1 ≈ 0.45% at
1.5 µm; ≈ 1% at 640 nm). The analytic references have no Kelvin term.

## Limitations

- Raoult uses the organic **mass** fraction (soacond convention), not mole
  fraction. Inorganics, nitrate and water do not dissolve SOA.
- No particle-phase chemistry (k_c = 0), no oligomerisation, no
  phase-state change with RH; D_b is fixed.
- SOA takes the bisulfate water ratio through `equilibrate_water`, as all
  organic slots do (ASSUMPTIONS.md A9).
- vbs-jax sees each bin's particles at their in-place mean mass for the
  whole 60 s micro-step; the PPM remap moves them once at the end.
- Known vbs-jax issues: a degenerate mixed condensation/evaporation sum
  can stall one sub-step, and `diag.time_integrated` < dt flags
  under-integration (see vbs-jax `docs/porting_notes.md`).

## References

- Donahue, N. M., Robinson, A. L., Stanier, C. O., and Pandis, S. N.: Coupled partitioning, dilution, and chemical aging of semivolatile organics, *Environ. Sci. Technol.*, 40, 2635–2643, 2006.
- Epstein, S. A., Riipinen, I., and Donahue, N. M.: A semiempirical correlation between enthalpy of vaporization and saturation concentration for organic aerosol, *Environ. Sci. Technol.*, 44, 743–748, 2010.
- Pankow, J. F.: An absorption model of gas/particle partitioning of organic compounds in the atmosphere, *Atmos. Environ.*, 28, 185–188, 1994.
- Zaveri, R. A., Easter, R. C., Shilling, J. E., and Seinfeld, J. H.: Modeling kinetic partitioning of secondary organic aerosol and size distribution dynamics, *Atmos. Chem. Phys.*, 14, 5153–5181, 2014.
- Fuchs, N. A. and Sutugin, A. G.: High-dispersed aerosols, in *Topics in Current Aerosol Research*, Pergamon, 1971.
- soacond.f (J. Pierce, 2010), via vbs-jax.
