# N2O5 heterogeneous uptake (AAAR tutorial, Level 3)

`tomas_jax/physics/heterogeneous.py::n2o5_uptake` gives the first-order
loss rate of N₂O₅ to the particles and which bins it goes to. The Level 3
"Day & Night" driver uses it to turn nighttime N₂O₅ into particle nitrate.

## Physics

The uptake rate is the TOMAS condensation sink (getCondSink.f) with N₂O₅
gas properties, and with the reactive uptake coefficient γ in place of
the mass accommodation coefficient:

    k_het = 2π D_g Σ_k Dp_k N_k β_k / V                       [s⁻¹]
    β_k   = (1 + Kn_k) / (1 + 2 Kn_k (1 + Kn_k) / γ)          (Fuchs–Sutugin, Dahneke form)
    Kn_k  = 2 λ / Dp_k,   λ = 2 D_g / c̄                       (TOMAS mean free path)

For Kn ≫ 1, β → γ/(2 Kn) and the expression reduces exactly to the
free-molecular result

    k_het → γ c̄ A / 4,   A = Σ_k π Dp_k² N_k / V

(`tests/test_aaar_aerosol.py::test_n2o5_free_molecular_limit`: within 0.04%
for 3–5 nm particles at γ = 1, closer at smaller γ). For larger particles
gas-phase diffusion limits the rate. Dp_k is the TOMAS wet diameter
(`calc_wet_diameters`), so uptake grows with RH as the aerosol takes up water.

N₂O₅ properties (`core/config.py`): MW = 108.01 g mol⁻¹. The Fuller
diffusion volume is 2·5.69 (N) + 5·5.48 (O) = 38.78, from the Fuller,
Schettler & Giddings (1966) atomic volumes that `SV_H2SO4` also uses. At
298 K and 1 atm, D_g = 1.19e-5 m² s⁻¹, c̄ = 242 m s⁻¹ and λ = 98 nm.

**Bin weights.** `weights[k]` is the fraction of k_het from bin k (sum 1).
They are computed at γ_w = max(γ, 1e-8), so they stay well defined at
γ = 0. k_het = CS(γ_w)·γ/γ_w is exact there because β is linear in γ for
γ ≪ Kn. With γ = 0, k_het = 0 and the weights are the surface-area
fractions.

## Product: particle nitrate

    N₂O₅(g) + H₂O(p) → 2 HNO₃ → 2 NO₃⁻(p)

Nitrate is stored as NO₃ mass in `Mk[:, SRTNIT]` (slot 40, MW 62). The
driver computes the mass source, with N₂O₅ in kg/cell:

    nitrate_rate [kg/cell/s] = 2 · k_het · N2O5 · MW_NO3 / MW_N2O5

It passes this and `weights` to `aerosol_outer_step` as
`AerosolForcing.nitrate_rate` and `.nitrate_weights`. In each micro-step,
`rate·dt` is split over every bin with particles (Nk >
`aerosol_step.NITRATE_N_MIN` = 2e-5 #/cell, just above MNFIX's empty-bin
floor of 1e-5 #/cell) by the weights scaled by Nk/Nk(start of the outer
step), renormalised over those bins. The weights are computed once per
coupling interval, so this keeps each bin's weight per particle while
particles move in or out of it. The particles that gained
nitrate then move between bins with the PPM remap
(`AerosolStepConfig.growth_bin_transfer = 'ppm'`, default; see
[aerosol_step.md](aerosol_step.md#bin-transfer-of-soa-and-nitrate-growth_bin_transfer)),
and MNFIX follows. Weight on empty bins (N = 0 or the MNFIX floor) is
dropped, so no mass goes into a bin MNFIX would wipe. If no bin with
particles has weight, nothing is added. Until 2026-10-16 the gate was
`NEPS_CONDSINK` = 1e10 #/cell (1 cm⁻³ at the tutorial boxvol): thin tail
bins got no nitrate, stopped growing at night and stayed behind as a band
of a few cm⁻³ at 40–90 nm in Level 3 that grew with resolution (see
[aerosol_step.md](aerosol_step.md#micro-step)). The weights themselves
are not gated: `calc_condensation_sink` gives every bin a weight
∝ Nk·Dp·β, using a default size (1.4·xk, 1500 kg m⁻³) for bins with
Nk ≤ 1e10 #/cell, so a thin bin's share per particle is 0.5–0.8 times
the value for its own particle size at 40 bins (0.7–1.0 at 160 bins;
Level 3, measured 2026-10-16). This changes nitrate by under 0.01 % and
was the same before the gate fix; computing these weights with each
bin's own size is a possible follow-up. With the PPM remap, nitrate
that would push the top bin's mean past the top of the grid (17.5 µm) is
not added either (`cap_top_bin`), instead of MNFIX trimming that bin's
sulfate along with it. `AerosolDiag.nitrate_added` always reports the
mass actually added, for the driver's N budget; when it falls short of
the source the driver adds a warning.

## Context for γ

Laboratory γ(N₂O₅) on aqueous sulfate is of order 10⁻². Ambient values span
roughly 10⁻⁴–10⁻¹. Particulate nitrate (the "nitrate effect") and organic
coatings suppress uptake, and chloride produces ClNO₂ (Bertram & Thornton
2009; Chang et al. 2011). The JPL evaluation (Burkholder et al. 2019)
tabulates uptake coefficients by surface. Here γ is a user input (Level 3
slider), constant in time and the same for all bins.

## Limitations (ASSUMPTIONS.md A5)

- γ does not depend on RH, temperature, composition, nitrate or organic
  coatings.
- No ClNO₂ channel: the full yield goes to 2 NO₃⁻.
- Particle nitrate is irreversible and non-volatile. There is no NH₄NO₃
  thermodynamics, and NH₃ equilibrium (`eznh3eqm`) neutralises only sulfate.
- Existing TOMAS routines lump slot 40 with the organic block. Nitrate is
  counted as sulfate-like for density (`density.py`) and takes the
  ammonium-bisulfate water ratio (`water_equilibrium.py`).
- The weights are computed once per outer step by the driver, so particles
  that grow across bins within the outer step use stale weights.

## References

- Bertram, T. H. and Thornton, J. A.: Toward a general parameterization of N2O5 reactivity on aqueous particles: the competing effects of particle liquid water, nitrate and chloride, *Atmos. Chem. Phys.*, 9, 8351–8363, 2009.
- Burkholder, J. B., et al.: Chemical Kinetics and Photochemical Data for Use in Atmospheric Studies, Evaluation No. 19, JPL Publication 19-5, Jet Propulsion Laboratory, Pasadena, 2019.
- Chang, W. L., Bhave, P. V., Brown, S. S., Riemer, N., Stutz, J., and Dabdub, D.: Heterogeneous atmospheric chemistry, ambient measurements, and model calculations of N2O5: a review, *Aerosol Sci. Technol.*, 45, 665–695, 2011.
- Fuchs, N. A. and Sutugin, A. G.: High-dispersed aerosols, in *Topics in Current Aerosol Research*, Pergamon, 1971.
- Fuller, E. N., Schettler, P. D., and Giddings, J. C.: A new method for prediction of binary gas-phase diffusion coefficients, *Ind. Eng. Chem.*, 58(5), 18–27, 1966.
- getCondSink.f (J. Pierce, 2007).
