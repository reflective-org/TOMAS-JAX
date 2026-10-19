"""Coupled box-model driver: ``run(scenario) -> RunResult``.

Operator split per coupling interval dt_c (``Scenario.numerics.coupling_dt_s``):

1. photolysis J at the interval midpoint; k_het(N2O5) from the aerosol at the start;
   all rate constants (incl. the RO2 sum) frozen for the interval (KPP semantics)
2. gas chemistry over dt_c (gaschem, diffrax); SA and NA_HET are counters, read and
   reset each interval
3. gather condensable species into the c*-class gas slots Gc[SRTSOA1:SRTSOA1+K]
4. aerosol_outer_step over dt_c (sources as rates: H2SO4 from SA, nitrate from NA_HET)
5. scatter the class gas change back to species; reset "hold constant" species

Level 1 (no chemistry) skips 1-3 and 5: one surrogate class with a production rate.

With nucleation on, the coupling interval and the aerosol micro-step are shortened to
NUCLEATION_MAX_COUPLING_DT_S and NUCLEATION_MAX_MICRO_DT_S (``time_steps``).
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import math
import time
from importlib import metadata
from pathlib import Path
from typing import Callable, NamedTuple

import jax.numpy as jnp
import numpy as np

from .. import __version__
from ..core.config import (ICOMP, MW_H2SO4, MW_NO3, N_GAS_SPECIES, N_SOA_CLASSES, SRTH2O, SRTNH4,
                           SRTNIT, SRTORG_NUC, SRTSO4, SRTSOA1, SV_H2SO4)
from ..gaschem.config import (CSTAR_TABLE_SOURCES, DHVAP_A_KJ, DHVAP_B_KJ, MECHANISM_DIR,
                              TUVX_CONFIG)
from ..gaschem.environment import air_environment
from ..gaschem.mechanism import EnvRates, Mechanism, RateModel
from ..gaschem.photolysis import SOURCES as J_SOURCES
from ..gaschem.photolysis import JProvider, PhotolysisSettings, tuvx_root
from ..gaschem.solver import integrate_interval
from ..gaschem.volatility import assign_classes, load_cstar_table
from ..physics.condensation_sink import calc_condensation_sink
from ..physics.heterogeneous import n2o5_uptake
from ..physics.water_equilibrium import equilibrate_water
from .aerosol_step import AerosolForcing, AerosolState, SOAParams, aerosol_outer_step
from .config import AerosolStepConfig
from .coupling import (NA_HET, N2O5_HET_LABEL, RHO_REPORT, ClassBridge, bin_diameters_nm,
                       cm3_to_kg_cell,
                       kg_cell_to_cm3, kg_cell_to_ugm3, mass_grid, seed_distribution,
                       tutorial_mechanism, ugm3_to_kg_cell)
from .losses import gas_loss_step, lai_nazaroff_beta, particle_loss_step
from .properties import kappa_mixture, organic_o_to_c, surface_volume_distributions
from .result import (N_TOP_CONTRIBUTORS, REPORTED_GASES_PPB, REPORTED_J, REPORTED_RADICALS_CM3,
                     RunResult)
from .scenario import Scenario, Seed, load_preset, list_presets

ACCOMMODATION_ORGANIC = 1.0     # SOA mass accommodation coefficient [-] (vbs-jax default)
COUNTER_SPECIES = ("SA", NA_HET)   # read and reset by the driver; not user inputs

# Time-step caps with nucleation on (ASSUMPTIONS A12/A13; docs/aaar/coupling.md, "Time
# steps with nucleation"). Nucleation convergence study, 2026-10-16 (Levels 1-3 with
# nucleation bursts at 265-270 K; reference 30 s coupling with a 1 s micro-step): a 60 s
# coupling interval with a 5 s micro-step keeps peak N, final N and GMD within 5 % in
# every nucleating case (largest +3.5 %, Level 3 at 265 K and SO2 20 ppb). The 60 s
# default micro-step overshoots the burst (peak N +30 % in Level 2, +78 % in Level 3,
# because H2SO4 production is added at the start of each micro-step and binary J rises
# steeply with H2SO4); 10 s still gives +8 to +9 %. Coupling intervals longer than 60 s
# freeze OH, and so the H2SO4 production rate, for too long (Level 2 at 900 s: -26 %;
# at 120 s with a 2 s micro-step still -4 %, so 120 s / 5 s passes only because the two
# errors partly cancel).
# Runs without nucleation are not changed.
NUCLEATION_MAX_COUPLING_DT_S = 60.0   # [s]
NUCLEATION_MAX_MICRO_DT_S = 5.0       # [s]


class TimeSteps(NamedTuple):
    """Coupling interval and micro-step a run uses (see :func:`time_steps`)."""
    coupling_dt_s: float            # coupling interval used [s]
    micro_dt_s: float               # aerosol micro-step target [s]; the step takes
                                    # ceil(coupling_dt_s / micro_dt_s) equal micro-steps
    split: int                      # used intervals per requested coupling interval
    requested_coupling_dt_s: float  # Scenario.numerics.coupling_dt_s [s]
    requested_micro_dt_s: float     # AerosolStepConfig.micro_dt [s]


def time_steps(scenario: Scenario, aerosol_cfg: AerosolStepConfig | None = None,
               nucleation_caps: bool = True) -> TimeSteps:
    """The coupling interval and aerosol micro-step ``run`` uses for ``scenario``.

    Without nucleation they are the requested values. With nucleation on (and
    ``nucleation_caps``), the requested coupling interval is cut into the smallest whole
    number m of equal parts that are at most NUCLEATION_MAX_COUPLING_DT_S long (300, 600
    and 900 s all give 60 s; 90 s gives 45 s). Because each part is an exact fraction of
    the requested interval, the run still ends at the requested duration and writes output
    at the same times as without the cap. The micro-step is at most
    NUCLEATION_MAX_MICRO_DT_S; a shorter requested micro-step is kept.
    """
    cfg = AerosolStepConfig() if aerosol_cfg is None else aerosol_cfg
    dt_req, micro_req = float(scenario.numerics.coupling_dt_s), float(cfg.micro_dt)
    if not (nucleation_caps and scenario.processes.nucleation):
        return TimeSteps(dt_req, micro_req, 1, dt_req, micro_req)
    split = max(1, math.ceil(dt_req / NUCLEATION_MAX_COUPLING_DT_S - 1e-9))
    return TimeSteps(dt_req / split, min(micro_req, NUCLEATION_MAX_MICRO_DT_S), split,
                     dt_req, micro_req)


def _time_step_warning(ts: TimeSteps) -> list[str]:
    """One warning line when the nucleation caps shortened a time step."""
    parts = []
    if ts.coupling_dt_s != ts.requested_coupling_dt_s:
        parts.append(f"the coupling interval is {ts.coupling_dt_s:g} s instead of "
                     f"{ts.requested_coupling_dt_s:g} s")
    if ts.micro_dt_s != ts.requested_micro_dt_s:
        parts.append(f"the aerosol micro-step is {ts.micro_dt_s:g} s instead of "
                     f"{ts.requested_micro_dt_s:g} s")
    if not parts:
        return []
    # shown to students in the web app, so no reference to repository files here
    return [f"Nucleation is on, so {' and '.join(parts)}, to follow the nucleation burst. "
            f"Output times are unchanged."]


@functools.lru_cache(maxsize=1)
def _chemistry():
    parsed = tutorial_mechanism()
    return parsed, RateModel(parsed, external=(N2O5_HET_LABEL,)), Mechanism.build(parsed)


@functools.lru_cache(maxsize=4)
def _class_map(cstar_table: str | None):
    parsed = _chemistry()[0]
    if cstar_table and Path(cstar_table).name != cstar_table:
        raise ValueError(f"cstar_table must be a file name in {MECHANISM_DIR.name}/: {cstar_table!r}")
    if not cstar_table:
        return assign_classes(parsed)
    # MECHANISM_DIR is read at call time (the tutorial points it at its own table folder)
    return assign_classes(parsed, load_cstar_table(MECHANISM_DIR / cstar_table),
                          CSTAR_TABLE_SOURCES.get(cstar_table, f"table {cstar_table}"))


def _cstar_fallback_warnings(class_map, cstar_table: str | None) -> list[str]:
    """Warn when a c* table leaves condensable species on the Li et al. (2016) estimate.

    A partial table (e.g. a few user values written without the rest of the default
    table) silently mixes two c* sources; the shipped table leaves none (ASSUMPTIONS A8).
    """
    if not cstar_table:
        return []
    condensable = set(class_map.species)
    mixed = [n for n in class_map.fallback if n in condensable]
    if not mixed:
        return []
    shown = ", ".join(mixed[:5]) + (", ..." if len(mixed) > 5 else "")
    return [f"c* table {cstar_table!r} does not list {len(mixed)} of {len(condensable)} "
            f"condensable species ({shown}); they use the Li et al. (2016) estimate. "
            f"To change a few c* values, merge them into the full table."]


@functools.lru_cache(maxsize=1)
def code_version() -> str:
    """Hash of model sources, mechanism files, pinned dependencies and TUV-x inputs (cache key)."""
    def add_tree(h, root, suffixes=None):
        for path in sorted(p for p in root.rglob("*") if p.is_file()
                           and (suffixes is None or p.suffix in suffixes)):
            h.update(path.relative_to(root).as_posix().encode())
            h.update(path.read_bytes())

    h = hashlib.sha256(__version__.encode())
    add_tree(h, Path(__file__).resolve().parents[1], (".py", ".eqn", ".csv", ".toml"))
    for dist in ("vbs-jax", "tuvx-jax", "jax", "diffrax"):
        try:
            d = metadata.distribution(dist)
        except metadata.PackageNotFoundError:
            continue
        # git-pinned packages keep version 0.1.0; direct_url.json holds the commit
        h.update(f"{dist}={d.version}{d.read_text('direct_url.json') or ''}".encode())
    try:
        root = tuvx_root()
        h.update((root / TUVX_CONFIG).read_bytes())
        add_tree(h, root / "data")
        add_tree(h, root / "tuvx_photolysis", (".py",))
    except (ImportError, RuntimeError):
        pass
    return h.hexdigest()[:16]


class _Budget:
    """Cumulative amounts removed by losses and added by emissions (per cell / per cm3)."""

    def __init__(self, n_species: int, nbins: int):
        self.wall_vapour_kg = 0.0            # condensable organic vapour lost to walls
        self.vent_vapour_kg = 0.0            # condensable organic vapour ventilated out, net
        self.o3_surface_cm3 = 0.0            # O3 lost to surfaces [molec/cm3]
        self.vent_gas_cm3 = np.zeros(n_species)   # per species, net out [molec/cm3]
        self.emitted_cm3 = np.zeros(n_species)    # per species [molec/cm3]
        self.dep_m = np.zeros(ICOMP)         # deposited particle mass per component [kg/cell]
        self.vent_m = np.zeros(ICOMP)        # ventilated particle mass per component, net out
        self.dep_n = 0.0                     # deposited particles [#/cell]
        self.vent_n = 0.0                    # ventilated particles, net out [#/cell]

    def snapshot(self) -> dict:
        return {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in vars(self).items()}


class _LossSetup(NamedTuple):
    """Loss rates and outdoor states a run uses (see ``_loss_setup``)."""
    lam_s: float                 # air-exchange rate [1/s]
    beta_s: np.ndarray           # (nbins,) deposition rate [1/s]
    nk_out: np.ndarray           # (nbins,) outdoor particles [#/cell]
    mk_out: np.ndarray           # (nbins, ICOMP) outdoor particle mass [kg/cell]
    y_out: np.ndarray | None     # (n_species,) outdoor gases [molec/cm3]; None without chemistry
    y_extra: np.ndarray | None   # (n_species,) O3 surface + vapour wall rates [1/s]
    gc_lam: np.ndarray           # (N_GAS_SPECIES,) ventilation rate per Gc slot [1/s]
    gc_extra: np.ndarray         # (N_GAS_SPECIES,) wall rate per Gc slot [1/s]


def _loss_setup(s: Scenario, xk, boxvol, cfg, index: dict | None, bridge, k_cls: int) -> _LossSetup:
    """Rates and outdoor states for the loss steps (docs/aaar/coupling.md, "Emissions and
    losses"). Without chemistry the condensable vapour is the Level 1 surrogate in
    Gc[SRTSOA1]; with chemistry the condensable species live in y, so their Gc class
    slots are not ventilated again."""
    loss, env, nbins = s.losses, s.environment, xk.size - 1
    lam = loss.ventilation_per_h / 3600.0
    beta = np.zeros(nbins)
    if loss.particle_deposition == "lai_nazaroff":
        room = loss.room
        m_mid = np.sqrt(xk[:-1] * xk[1:])
        dp = (6.0 * m_mid / (np.pi * room.particle_density_g_cm3 * 1e3)) ** (1.0 / 3.0)
        beta = lai_nazaroff_beta(dp, env.temp_k, env.pres_pa, room.friction_velocity_cm_s,
                                 room.surface_to_volume_per_m, room.height_m,
                                 room.particle_density_g_cm3)
    nk_out, mk_out = np.zeros(nbins), np.zeros((nbins, ICOMP))
    if loss.outdoor_seed is not None and lam > 0:
        nk_out, mk_out = _seed(loss.outdoor_seed, xk, boxvol)
        mk_out = np.asarray(equilibrate_water(jnp.asarray(mk_out), env.rh, env.temp_k, cfg.water_scheme))
    gc_lam, gc_extra = np.zeros(N_GAS_SPECIES), np.zeros(N_GAS_SPECIES)
    k_w = loss.organic_vapour_wall_per_s
    if index is None:
        gc_lam[:] = lam
        gc_extra[SRTSOA1:SRTSOA1 + k_cls] = k_w
        return _LossSetup(lam, beta, nk_out, mk_out, None, None, gc_lam, gc_extra)
    air = air_environment(env.temp_k, env.pres_pa, env.rh)
    y_out, y_extra = np.zeros(len(index)), np.zeros(len(index))
    for sp, ppb in loss.outdoor_gases_ppb.items():
        y_out[index[sp]] = ppb * 1e-9 * air["M"]
    y_extra[index["O3"]] = loss.o3_surface_per_h / 3600.0
    y_extra[bridge.idx] += k_w
    gc_lam[SRTSO4] = lam                  # H2SO4 vapour; the class slots follow y
    return _LossSetup(lam, beta, nk_out, mk_out, y_out, y_extra, gc_lam, gc_extra)


def _seed(seed: Seed, xk, boxvol):
    return seed_distribution(seed.modes, seed.organic_mass_fraction, xk, boxvol,
                             ammonium_nitrate_mass_fraction=seed.ammonium_nitrate_mass_fraction,
                             ammonium=seed.ammonium)


def _processes(s: Scenario, has_organics: bool) -> tuple[str, ...]:
    p = s.processes
    out = []
    if p.nucleation:
        out.append("nucleation")
    if p.coagulation:
        out.append("coagulation")
    if p.h2so4_condensation:
        out.append("h2so4_condensation")
    if p.soa_partitioning and has_organics:
        out.append("soa_partitioning")
    return tuple(out)


def run(scenario: Scenario, progress_cb: Callable[[float], None] | None = None,
        cancel: Callable[[], bool] | None = None,
        observer: Callable[[float, dict], None] | None = None,
        aerosol_cfg: AerosolStepConfig | None = None,
        nucleation_caps: bool = True) -> RunResult:
    """Run one scenario.

    With nucleation on, the coupling interval and the aerosol micro-step are shortened
    to at most NUCLEATION_MAX_COUPLING_DT_S and NUCLEATION_MAX_MICRO_DT_S
    (:func:`time_steps`); a warning says so and ``diagnostics`` records the requested
    and the used values (``requested_coupling_dt_s``, ``coupling_dt_s``,
    ``requested_micro_dt_s``, ``micro_dt_s``). ``micro_dt_s`` is the target length: the
    aerosol step splits each coupling interval into ceil(coupling_dt_s / micro_dt_s)
    equal micro-steps, which can be slightly shorter. ``nucleation_caps=False`` turns
    the caps off; it exists for time-step convergence studies only.

    ``aerosol_cfg`` (optional) replaces the default ``AerosolStepConfig()``, e.g.
    ``AerosolStepConfig(growth_bin_transfer="mnfix")`` to compare with the previous (MNFIX)
    bin transfer of SOA and nitrate. Its non-default fields are listed in
    ``diagnostics["aerosol_cfg"]`` (empty for a default run).

    ``observer`` (optional, for benchmarks/diagnostics; it does not change the run) is
    called as ``observer(t_s, snapshot)`` at t = 0 and after every coupling interval.
    ``snapshot`` holds numpy copies: ``y`` (every gas species [molec/cm3], order of the
    tutorial mechanism; None without chemistry), ``Nk`` [#/cell], ``Mk`` [kg/cell],
    ``Gc`` [kg/cell], ``j`` (J [1/s] used for the interval, order of
    ``RateModel.j_names``; None at t = 0 or without chemistry), ``counters`` (SA and
    NA_HET formed in the interval [molec/cm3], before the driver resets them) and
    ``particle_tracker`` (per condensable species particle mass [kg/cell]; None
    without chemistry).
    """
    t_wall = time.perf_counter()
    s = scenario
    cfg = AerosolStepConfig() if aerosol_cfg is None else aerosol_cfg
    boxvol = cfg.boxvol_cm3
    env = s.environment
    ts = time_steps(s, cfg, nucleation_caps)
    dt_req = ts.requested_coupling_dt_s
    n_req = int(round(s.duration_h * 3600.0 / dt_req))
    if n_req < 1 or abs(n_req * dt_req - s.duration_h * 3600.0) > 1e-6:
        raise ValueError(f"duration ({s.duration_h} h) must be a positive multiple of the "
                         f"coupling interval ({dt_req:.0f} s)")
    # output times follow the requested interval; the caps only split each interval
    dt_c, n_int = ts.coupling_dt_s, n_req * ts.split
    out_every = max(1, int(round(s.numerics.output_dt_s / dt_req))) * ts.split
    warnings: list[str] = _time_step_warning(ts)
    if s.processes.chemistry and s.surrogate is not None:
        raise ValueError("a scenario uses either the surrogate organic or chemistry, not both")

    # --- aerosol initial state ---------------------------------------------------------
    xk = mass_grid(s.numerics.nbins)
    nk, mk = _seed(s.seed, xk, boxvol)
    mk = np.asarray(equilibrate_water(jnp.asarray(mk), env.rh, env.temp_k, cfg.water_scheme))
    gc = np.zeros(N_GAS_SPECIES)

    # --- volatility classes -----------------------------------------------------------------
    chem = s.processes.chemistry
    if chem:
        parsed, rate_model, mech = _chemistry()
        class_map = _class_map(s.cstar_table)
        cstar, dhvap, mw_cls = class_map.class_cstar, class_map.class_dhvap, class_map.class_mw
        cstar_source = class_map.source
        warnings.extend(_cstar_fallback_warnings(class_map, s.cstar_table))
    elif s.surrogate is not None:
        cstar = np.array([s.surrogate.cstar_ugm3])
        dhvap = DHVAP_A_KJ - DHVAP_B_KJ * np.log10(cstar)
        mw_cls = np.array([s.surrogate.mw_g_mol])
        gc[SRTSOA1] = ugm3_to_kg_cell(s.surrogate.initial_gas_ugm3, boxvol)
        cstar_source = "user-chosen surrogate"
    else:
        cstar, dhvap, mw_cls = np.array([1.0]), np.array([DHVAP_A_KJ]), np.array([200.0])
        cstar_source = "no organic vapor"
    k_cls = cstar.size
    assert k_cls <= N_SOA_CLASSES
    soa = SOAParams(jnp.asarray(cstar, dtype=jnp.float64), jnp.asarray(dhvap, dtype=jnp.float64),
                    jnp.asarray(mw_cls, dtype=jnp.float64), jnp.float64(ACCOMMODATION_ORGANIC))
    processes = _processes(s, chem or s.surrogate is not None)
    org_rate = np.zeros(k_cls)
    if s.surrogate is not None:
        org_rate[0] = ugm3_to_kg_cell(s.surrogate.production_ugm3_per_h, boxvol) / 3600.0

    state = AerosolState(jnp.asarray(nk), jnp.asarray(mk), jnp.asarray(gc), jnp.asarray(xk),
                         jnp.float64(env.temp_k), jnp.float64(env.pres_pa), jnp.float64(env.rh),
                         jnp.float64(boxvol))

    # --- gas chemistry setup ----------------------------------------------------------------
    air = air_environment(env.temp_k, env.pres_pa, env.rh)
    if chem:
        index = {sp: i for i, sp in enumerate(parsed.species)}
        named = (set(s.gases_ppb) | set(s.hold_constant) | set(s.emissions_ppb_per_h)
                 | set(s.losses.outdoor_gases_ppb))
        unknown = sorted(named - set(index))
        if unknown:
            raise ValueError(f"unknown mechanism species: {unknown}")
        counters = sorted(named & set(COUNTER_SPECIES))
        if counters:
            raise ValueError(f"{counters} are internal counter species and cannot be set or held")
        y = np.zeros(mech.n_species)
        for sp, ppb in s.gases_ppb.items():
            y[index[sp]] = ppb * 1e-9 * air["M"]
        emission = np.zeros(mech.n_species)          # E [molec/cm3/s], added each interval
        for sp, rate in s.emissions_ppb_per_h.items():
            emission[index[sp]] = rate * 1e-9 * air["M"] / 3600.0
        held = np.array([index[sp] for sp in s.hold_constant], dtype=int)
        held_values = y[held].copy()
        k_th, ro2_dep, j_coef = rate_model.at_env(air)
        rates = EnvRates(jnp.asarray(k_th), jnp.asarray(ro2_dep), jnp.asarray(j_coef))
        i_het = rate_model.labels.index(N2O5_HET_LABEL)
        i_sa, i_na_het, i_n2o5, i_o3 = index["SA"], index[NA_HET], index["N2O5"], index["O3"]
        bridge = ClassBridge(class_map, boxvol)
        j_names = rate_model.j_names
        jp = JProvider(j_names, PhotolysisSettings(
            s.photolysis.mode, s.photolysis.latitude_deg, s.photolysis.longitude_deg,
            s.start_utc, s.photolysis.jno2_per_s, s.photolysis.constant_sza_deg))
    else:
        j_names, jp = [], None
        if s.emissions_ppb_per_h or s.losses.outdoor_gases_ppb:
            raise ValueError("emissions and outdoor gases need chemistry on (processes.chemistry)")
    h2so4_rate_l1 = cm3_to_kg_cell(s.h2so4_production_cm3_s, MW_H2SO4, boxvol)

    # --- emissions, ventilation, surface and wall losses (all off by default) ---------------
    loss_on = s.losses.any
    ls = _loss_setup(s, xk, boxvol, cfg, index if chem else None, bridge if chem else None, k_cls)
    lam_bins = np.full(nk.size, ls.lam_s)
    budget = _Budget(mech.n_species if chem else 0, nk.size)
    j_factor = s.photolysis.j_factor          # cloud fraction: J = J_clear x (1 - 0.75 CF)

    def j_at(t_s):
        j = jp.at(t_s)
        return j if j_factor == 1.0 else j * j_factor

    # --- outputs -----------------------------------------------------------------------------
    rec = {key: [] for key in ("t", "nk", "mk", "gc", "y", "cs", "jnuc", "j", "sza", "tracker",
                               "budget")}
    gas_steps = 0
    soa_bad = 0
    nitrate_expected = nitrate_added = 0.0

    def record(t_s, st, diag_cs, diag_jnuc, y_now, j_now):
        rec["t"].append(t_s / 3600.0)
        rec["nk"].append(np.asarray(st.Nk))
        rec["mk"].append(np.asarray(st.Mk))
        rec["gc"].append(np.asarray(st.Gc))
        rec["y"].append(None if y_now is None else y_now.copy())
        rec["cs"].append(float(diag_cs))
        rec["jnuc"].append(float(diag_jnuc))
        rec["j"].append(j_now)
        rec["sza"].append(None if jp is None else jp.sza_deg(t_s))
        rec["tracker"].append(bridge.particle.copy() if chem else None)
        rec["budget"].append(budget.snapshot())

    j_now = None if jp is None else j_at(0.0)
    cs0, _ = calc_condensation_sink(state.Nk, state.Mk, state.temp, state.pres, state.boxvol,
                                    MW_H2SO4, SV_H2SO4, cfg.alpha_h2so4, xk=state.xk)
    record(0.0, state, cs0, 0.0, y if chem else None, j_now)
    if observer is not None:
        _observe(observer, 0.0, state, y if chem else None, None, (0.0, 0.0),
                 bridge if chem else None)
    for i in range(n_int):
        if cancel is not None and cancel():
            raise RuntimeError("run cancelled")
        t0 = i * dt_c
        h2so4_rate, nitrate_rate = h2so4_rate_l1, 0.0
        weights = jnp.zeros(nk.size)
        if chem:
            if s.emissions_ppb_per_h:
                y = y + emission * dt_c                     # constant sources, added up front
                budget.emitted_cm3 += emission * dt_c
            j_now = j_at(t0 + 0.5 * dt_c)
            k = rates.k(mech.ro2(jnp.asarray(y)), jnp.asarray(j_now))
            k_het = 0.0
            if s.processes.n2o5_uptake and s.gamma_n2o5 > 0:
                k_het, weights = n2o5_uptake(state.Nk, state.Mk, state.xk, state.temp, state.pres,
                                             state.boxvol, jnp.float64(s.gamma_n2o5))
            k = k.at[i_het].set(k_het)
            y_new, nsteps = integrate_interval(mech, jnp.asarray(y), k, dt_c)
            gas_steps += int(nsteps)
            y = np.asarray(y_new).copy()
            counts = (float(y[i_sa]), float(y[i_na_het]))
            h2so4_rate = cm3_to_kg_cell(y[i_sa], MW_H2SO4, boxvol) / dt_c
            nitrate_rate = cm3_to_kg_cell(y[i_na_het], MW_NO3, boxvol) / dt_c
            y[i_sa] = y[i_na_het] = 0.0
            if loss_on:
                # ventilation toward outdoor air, O3 surface loss, vapour wall loss
                y, gb = gas_loss_step(y, ls.y_out, ls.lam_s, ls.y_extra, dt_c)
                budget.vent_gas_cm3 += gb["out"] - gb["in"]
                budget.o3_surface_cm3 += gb["sink"][i_o3]
                sink_kg = cm3_to_kg_cell(gb["sink"][bridge.idx], bridge.mw, boxvol)
                budget.wall_vapour_kg += float(sink_kg.sum())
                budget.vent_vapour_kg += float(cm3_to_kg_cell(
                    (gb["out"] - gb["in"])[bridge.idx], bridge.mw, boxvol).sum())
            gas_before = bridge.gas_mass(y)
            gc_new = np.asarray(state.Gc).copy()
            gc_new[SRTSOA1:SRTSOA1 + k_cls] = bridge.class_sum(gas_before)
            state = state._replace(Gc=jnp.asarray(gc_new))
        forcing = AerosolForcing(jnp.float64(h2so4_rate), jnp.asarray(org_rate, dtype=jnp.float64),
                                 jnp.float64(nitrate_rate), jnp.asarray(weights, dtype=jnp.float64))
        state, diag = aerosol_outer_step(state, forcing, soa, dt_c, micro_dt=ts.micro_dt_s,
                                         processes=processes, cfg=cfg)
        if loss_on:
            # particle ventilation and deposition per bin, then the vapours in Gc
            nk_l, mk_l, pb = particle_loss_step(np.asarray(state.Nk), np.asarray(state.Mk),
                                                ls.nk_out, ls.mk_out, lam_bins, ls.beta_s, dt_c)
            budget.dep_m += pb["m_deposited"].sum(axis=0)
            budget.vent_m += (pb["m_out"] - pb["m_in"]).sum(axis=0)
            budget.dep_n += float(pb["n_deposited"].sum())
            budget.vent_n += float((pb["n_out"] - pb["n_in"]).sum())
            gc_l, gcb = gas_loss_step(np.asarray(state.Gc), np.zeros(N_GAS_SPECIES), ls.gc_lam,
                                      ls.gc_extra, dt_c)
            if not chem:
                budget.wall_vapour_kg += float(gcb["sink"][SRTSOA1:SRTSOA1 + k_cls].sum())
                budget.vent_vapour_kg += float((gcb["out"] - gcb["in"])[SRTSOA1:SRTSOA1 + k_cls].sum())
            mk_l = equilibrate_water(jnp.asarray(mk_l), state.rh, state.temp, cfg.water_scheme)
            state = state._replace(Nk=jnp.asarray(nk_l), Mk=mk_l, Gc=jnp.asarray(gc_l))
        soa_bad += int(not bool(diag.soa_time_integrated))
        nitrate_expected += nitrate_rate * dt_c
        nitrate_added += float(diag.nitrate_added)
        if chem:
            mk_now = np.asarray(state.Mk)
            class_particle = mk_now[:, SRTSOA1:SRTSOA1 + k_cls].sum(axis=0)
            class_gas = np.asarray(state.Gc)[SRTSOA1:SRTSOA1 + k_cls]
            y = bridge.scatter(y, gas_before, class_gas, class_particle)
            y[held] = held_values
        if observer is not None:
            _observe(observer, t0 + dt_c, state, y if chem else None, j_now if chem else None,
                     counts if chem else (0.0, 0.0), bridge if chem else None)
        if (i + 1) % out_every == 0 or i + 1 == n_int:
            record(t0 + dt_c, state, diag.cs_h2so4, diag.j_nuc, y if chem else None,
                   None if jp is None else j_at(t0 + dt_c))
        if progress_cb is not None:
            progress_cb((i + 1) / n_int)
    if soa_bad:
        warnings.append(f"SOA partitioning hit its sub-step budget in {soa_bad} of {n_int} intervals")
    if nitrate_added < nitrate_expected * (1.0 - 1e-6):
        lost = kg_cell_to_ugm3(nitrate_expected - nitrate_added, boxvol)
        warnings.append(f"{lost:.3g} ug/m3 of N2O5-uptake nitrate was not added (no populated bin, "
                        "or it would have pushed the top bin past the grid)")

    return _assemble(s, rec, xk, boxvol, air, chem, k_cls, cstar, cstar_source,
                     bridge if chem else None, j_names, warnings,
                     {"gas_solver_steps": gas_steps, "runtime_s": time.perf_counter() - t_wall,
                      "code_version": code_version(),
                      "requested_coupling_dt_s": dt_req, "coupling_dt_s": dt_c,
                      "requested_micro_dt_s": ts.requested_micro_dt_s,
                      "micro_dt_s": ts.micro_dt_s,
                      "processes": list(processes),
                      "aerosol_cfg": _cfg_changes(cfg)})


def _cfg_changes(cfg: AerosolStepConfig) -> dict:
    """Fields of ``cfg`` that differ from the defaults (empty for a default run)."""
    default = AerosolStepConfig()
    return {f.name: getattr(cfg, f.name) for f in dataclasses.fields(cfg)
            if getattr(cfg, f.name) != getattr(default, f.name)}


def _observe(observer, t_s, state, y, j, counts, bridge) -> None:
    """Hand a snapshot (numpy copies) to the optional ``run`` observer."""
    observer(t_s, {
        "y": None if y is None else y.copy(),
        "Nk": np.array(state.Nk), "Mk": np.array(state.Mk), "Gc": np.array(state.Gc),
        "j": None if j is None else np.array(j),
        "counters": {"SA": counts[0], NA_HET: counts[1]},
        "particle_tracker": None if bridge is None else bridge.particle.copy(),
    })


def _assemble(s, rec, xk, boxvol, air, chem, k_cls, cstar, cstar_source, bridge, j_names,
              warnings, extra) -> RunResult:
    d_mid, d_edges = bin_diameters_nm(xk)
    dlog10 = np.log10(xk[1:] / xk[:-1]) / 3.0
    nk = np.array(rec["nk"]) / boxvol                              # [t, bin] #/cm3
    mk = kg_cell_to_ugm3(np.array(rec["mk"]), boxvol)              # [t, bin, comp] ug/m3
    gc = np.array(rec["gc"])
    dry = mk[:, :, :SRTH2O].sum(axis=2)
    soa_slots = slice(SRTSOA1, SRTSOA1 + k_cls)
    poa_slots = [c for c in range(SRTSOA1 + k_cls, SRTORG_NUC + 1) if c != SRTNIT]
    totals = {
        "N_cm3": nk.sum(axis=1), "SO4_ugm3": mk[:, :, SRTSO4].sum(axis=1),
        "NIT_ugm3": mk[:, :, SRTNIT].sum(axis=1), "SOA_ugm3": mk[:, :, soa_slots].sum(axis=(1, 2)),
        "POA_ugm3": mk[:, :, poa_slots].sum(axis=(1, 2)), "NH4_ugm3": mk[:, :, SRTNH4].sum(axis=1),
        "H2O_ugm3": mk[:, :, SRTH2O].sum(axis=1), "dry_ugm3": dry.sum(axis=1),
    }
    # V from the dry mass (exact under coagulation), S at each bin's mean-mass diameter
    ds, dv = surface_volume_distributions(nk / dlog10, dry / dlog10, d_mid, RHO_REPORT / 1e3)
    totals["S_um2cm3"] = (ds * dlog10).sum(axis=1)
    totals["V_um3cm3"] = (dv * dlog10).sum(axis=1)
    # O:C of the particle organics and kappa (box/properties.py; ASSUMPTIONS A18)
    mk_cell = np.array(rec["mk"])
    seed_org = mk_cell[:, :, SRTORG_NUC].sum(axis=1)
    if chem and bridge is not None:
        parsed0 = _chemistry()[0]
        class_map0 = _class_map(s.cstar_table)
        atoms = [parsed0.atoms[name] for name in class_map0.species]
        per_kg = 1.0 / (bridge.mw * 1e-3)                                  # mol per kg
        o_per_kg = np.array([a.get("O", 0) for a in atoms]) * per_kg
        c_per_kg = np.array([a.get("C", 0) for a in atoms]) * per_kg
        o_to_c = organic_o_to_c(np.array(rec["tracker"]), o_per_kg, c_per_kg, seed_org,
                                s.seed.organic_o_to_c)
    else:
        o_to_c = organic_o_to_c(np.zeros((len(rec["t"]), 0)), np.zeros(0), np.zeros(0), seed_org,
                                s.seed.organic_o_to_c)
    properties = {"o_to_c": o_to_c, **kappa_mixture(mk[:, :, :SRTH2O], k_cls, o_to_c)}
    result = RunResult(
        scenario=_scenario_dict(s), time_h=rec["t"], diameter_nm=d_mid, diameter_edges_nm=d_edges,
        dndlog10dp_cm3=nk / dlog10, dmdlog10dp_ugm3=dry / dlog10, totals=totals,
        dsdlog10dp_um2cm3=ds, dvdlog10dp_um3cm3=dv, properties=properties,
        losses=_loss_budget(rec["budget"], s, air, boxvol, chem, k_cls),
        h2so4_cm3=kg_cell_to_cm3(gc[:, SRTSO4], MW_H2SO4, boxvol),
        volatility={"log10_cstar": np.log10(cstar),
                    "gas_ugm3": kg_cell_to_ugm3(gc[:, soa_slots], boxvol),
                    "particle_ugm3": mk[:, :, soa_slots].sum(axis=1)},
        diagnostics={"cs_per_s": rec["cs"], "nucleation_rate_cm3_s": rec["jnuc"],
                     "cstar_source": cstar_source, "warnings": warnings, **extra},
    )
    if chem:
        parsed = _chemistry()[0]
        index = {sp: i for i, sp in enumerate(parsed.species)}
        ys = np.array(rec["y"])
        result.gases_ppb = {g: ys[:, index[g]] / air["M"] * 1e9 for g in REPORTED_GASES_PPB}
        result.radicals_cm3 = {r: ys[:, index[r]] for r in REPORTED_RADICALS_CM3}
        class_map = _class_map(s.cstar_table)
        for o in np.argsort(bridge.particle)[::-1][:N_TOP_CONTRIBUTORS]:
            if bridge.particle[o] <= 0:
                break
            name = class_map.species[o]
            result.top_contributors.append({
                "species": name, "formula": _formula(parsed.atoms[name]),
                "particle_ugm3": float(kg_cell_to_ugm3(bridge.particle[o], boxvol)),
                "log10_cstar": float(class_map.log10_cstar[o])})
        # closed-shell organics missing from the c* table (Li et al. 2016 c* used)
        result.diagnostics["cstar_fallback"] = list(class_map.fallback)
        result.diagnostics["j_sources"] = {n: J_SOURCES[n] for n in j_names}
        js = np.array(rec["j"])
        result.photolysis = {n: js[:, j_names.index(n)] for n in REPORTED_J}
        result.photolysis["sza_deg"] = rec["sza"]
    return result


def _loss_budget(snapshots: list[dict], s: Scenario, air, boxvol, chem: bool, k_cls: int) -> dict:
    """Cumulative loss and emission amounts at the output times (RunResult.losses)."""
    soa = slice(SRTSOA1, SRTSOA1 + k_cls)
    poa = [c for c in range(SRTSOA1 + k_cls, SRTORG_NUC + 1) if c != SRTNIT]
    inorganic = [SRTSO4, SRTNH4, SRTNIT]

    def series(key):
        return np.array([snap[key] for snap in snapshots])

    def groups(prefix, m):                            # m: [t, ICOMP] kg/cell
        ug = kg_cell_to_ugm3(m, boxvol)
        return {f"{prefix}_dry_ugm3": ug[:, :SRTH2O].sum(axis=1),
                f"{prefix}_SOA_ugm3": ug[:, soa].sum(axis=1),
                f"{prefix}_POA_ugm3": ug[:, poa].sum(axis=1),
                f"{prefix}_inorganic_ugm3": ug[:, inorganic].sum(axis=1)}

    out = {
        "wall_vapour_ugm3": kg_cell_to_ugm3(series("wall_vapour_kg"), boxvol),
        "ventilated_vapour_ugm3": kg_cell_to_ugm3(series("vent_vapour_kg"), boxvol),
        "o3_surface_ppb": series("o3_surface_cm3") / air["M"] * 1e9,
        **groups("deposited", series("dep_m")), **groups("ventilated", series("vent_m")),
        "deposited_N_cm3": series("dep_n") / boxvol, "ventilated_N_cm3": series("vent_n") / boxvol,
        "ventilated_gases_ppb": {}, "emitted_ppb": {},
    }
    if chem:
        index = {sp: i for i, sp in enumerate(_chemistry()[0].species)}
        vent, emitted = series("vent_gas_cm3"), series("emitted_cm3")
        out["ventilated_gases_ppb"] = {g: vent[:, index[g]] / air["M"] * 1e9 for g in REPORTED_GASES_PPB}
        out["emitted_ppb"] = {g: emitted[:, index[g]] / air["M"] * 1e9 for g in s.emissions_ppb_per_h}
    return out


def _formula(atoms: dict[str, int]) -> str:
    order = ["C", "H", "N", "O", "S", "Cl", "Br"]
    return "".join(f"{a}{atoms[a] if atoms[a] > 1 else ''}" for a in order if atoms.get(a))


def _scenario_dict(s: Scenario) -> dict:
    d = dataclasses.asdict(s)
    d["start_utc"] = s.start_utc.isoformat()
    return d


def warmup(nucleation: bool = True) -> None:
    """Build the mechanism, the TUV-x J table and compile every preset's step shapes.

    With ``nucleation`` (default) each preset is also compiled with nucleation on, whose
    shorter time steps (NUCLEATION_MAX_COUPLING_DT_S, NUCLEATION_MAX_MICRO_DT_S) are
    separate compiles. 300, 600 and 900 s all split into 60 s intervals, so one 60 s
    variant per preset covers them. Every preset requests 300 s, so a run without
    nucleation at another interval (600 or 900 s) compiles once on its first call.
    """
    for name in list_presets():
        s = load_preset(name)
        variants = [s]
        if nucleation:
            variants.append(s.replace(processes={"nucleation": True},
                                      numerics={"coupling_dt_s": NUCLEATION_MAX_COUPLING_DT_S}))
        for v in variants:
            dt = v.numerics.coupling_dt_s
            run(v.replace(duration_h=2 * dt / 3600.0, numerics={"output_dt_s": dt}))
