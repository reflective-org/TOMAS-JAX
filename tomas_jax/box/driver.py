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
"""
from __future__ import annotations

import dataclasses
import functools
import hashlib
import time
from importlib import metadata
from pathlib import Path
from typing import Callable

import jax.numpy as jnp
import numpy as np

from .. import __version__
from ..core.config import (MW_H2SO4, MW_NO3, N_GAS_SPECIES, N_SOA_CLASSES, SRTH2O, SRTNH4,
                           SRTNIT, SRTORG_NUC, SRTSO4, SRTSOA1, SV_H2SO4)
from ..gaschem.config import DHVAP_A_KJ, DHVAP_B_KJ, MECHANISM_DIR, TUVX_CONFIG
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
from .coupling import (NA_HET, N2O5_HET_LABEL, ClassBridge, bin_diameters_nm, cm3_to_kg_cell,
                       kg_cell_to_cm3, kg_cell_to_ugm3, mass_grid, seed_distribution,
                       tutorial_mechanism, ugm3_to_kg_cell)
from .result import (N_TOP_CONTRIBUTORS, REPORTED_GASES_PPB, REPORTED_J, REPORTED_RADICALS_CM3,
                     RunResult)
from .scenario import Scenario, load_preset, list_presets

ACCOMMODATION_ORGANIC = 1.0     # SOA mass accommodation coefficient [-] (vbs-jax default)
COUNTER_SPECIES = ("SA", NA_HET)   # read and reset by the driver; not user inputs


@functools.lru_cache(maxsize=1)
def _chemistry():
    parsed = tutorial_mechanism()
    return parsed, RateModel(parsed, external=(N2O5_HET_LABEL,)), Mechanism.build(parsed)


@functools.lru_cache(maxsize=4)
def _class_map(cstar_table: str | None):
    parsed = _chemistry()[0]
    if cstar_table and Path(cstar_table).name != cstar_table:
        raise ValueError(f"cstar_table must be a file name in {MECHANISM_DIR.name}/: {cstar_table!r}")
    table = load_cstar_table(MECHANISM_DIR / cstar_table) if cstar_table else None
    return assign_classes(parsed, table)


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
        cancel: Callable[[], bool] | None = None) -> RunResult:
    t_wall = time.perf_counter()
    s = scenario
    cfg = AerosolStepConfig()
    boxvol = cfg.boxvol_cm3
    env = s.environment
    dt_c = float(s.numerics.coupling_dt_s)
    n_int = int(round(s.duration_h * 3600.0 / dt_c))
    if n_int < 1 or abs(n_int * dt_c - s.duration_h * 3600.0) > 1e-6:
        raise ValueError(f"duration ({s.duration_h} h) must be a positive multiple of the "
                         f"coupling interval ({dt_c:.0f} s)")
    out_every = max(1, int(round(s.numerics.output_dt_s / dt_c)))
    warnings: list[str] = []
    if s.processes.chemistry and s.surrogate is not None:
        raise ValueError("a scenario uses either the surrogate organic or chemistry, not both")

    # --- aerosol initial state ---------------------------------------------------------
    xk = mass_grid(s.numerics.nbins)
    nk, mk = seed_distribution(s.seed.modes, s.seed.organic_mass_fraction, xk, boxvol)
    mk = np.asarray(equilibrate_water(jnp.asarray(mk), env.rh, env.temp_k, cfg.water_scheme))
    gc = np.zeros(N_GAS_SPECIES)

    # --- volatility classes -----------------------------------------------------------------
    chem = s.processes.chemistry
    if chem:
        parsed, rate_model, mech = _chemistry()
        class_map = _class_map(s.cstar_table)
        cstar, dhvap, mw_cls = class_map.class_cstar, class_map.class_dhvap, class_map.class_mw
        cstar_source = class_map.source
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
        unknown = sorted((set(s.gases_ppb) | set(s.hold_constant)) - set(index))
        if unknown:
            raise ValueError(f"unknown mechanism species: {unknown}")
        counters = sorted((set(s.gases_ppb) | set(s.hold_constant)) & set(COUNTER_SPECIES))
        if counters:
            raise ValueError(f"{counters} are internal counter species and cannot be set or held")
        y = np.zeros(mech.n_species)
        for sp, ppb in s.gases_ppb.items():
            y[index[sp]] = ppb * 1e-9 * air["M"]
        held = np.array([index[sp] for sp in s.hold_constant], dtype=int)
        held_values = y[held].copy()
        k_th, ro2_dep, j_coef = rate_model.at_env(air)
        rates = EnvRates(jnp.asarray(k_th), jnp.asarray(ro2_dep), jnp.asarray(j_coef))
        i_het = rate_model.labels.index(N2O5_HET_LABEL)
        i_sa, i_na_het, i_n2o5 = index["SA"], index[NA_HET], index["N2O5"]
        bridge = ClassBridge(class_map, boxvol)
        j_names = rate_model.j_names
        jp = JProvider(j_names, PhotolysisSettings(
            s.photolysis.mode, s.photolysis.latitude_deg, s.photolysis.longitude_deg,
            s.start_utc, s.photolysis.jno2_per_s, s.photolysis.constant_sza_deg))
    else:
        j_names, jp = [], None
    h2so4_rate_l1 = cm3_to_kg_cell(s.h2so4_production_cm3_s, MW_H2SO4, boxvol)

    # --- outputs -----------------------------------------------------------------------------
    rec = {key: [] for key in ("t", "nk", "mk", "gc", "y", "cs", "jnuc", "j", "sza")}
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

    j_now = None if jp is None else jp.at(0.0)
    cs0, _ = calc_condensation_sink(state.Nk, state.Mk, state.temp, state.pres, state.boxvol,
                                    MW_H2SO4, SV_H2SO4, cfg.alpha_h2so4, xk=state.xk)
    record(0.0, state, cs0, 0.0, y if chem else None, j_now)
    for i in range(n_int):
        if cancel is not None and cancel():
            raise RuntimeError("run cancelled")
        t0 = i * dt_c
        h2so4_rate, nitrate_rate = h2so4_rate_l1, 0.0
        weights = jnp.zeros(nk.size)
        if chem:
            j_now = jp.at(t0 + 0.5 * dt_c)
            k = rates.k(mech.ro2(jnp.asarray(y)), jnp.asarray(j_now))
            k_het = 0.0
            if s.processes.n2o5_uptake and s.gamma_n2o5 > 0:
                k_het, weights = n2o5_uptake(state.Nk, state.Mk, state.xk, state.temp, state.pres,
                                             state.boxvol, jnp.float64(s.gamma_n2o5))
            k = k.at[i_het].set(k_het)
            y_new, nsteps = integrate_interval(mech, jnp.asarray(y), k, dt_c)
            gas_steps += int(nsteps)
            y = np.asarray(y_new).copy()
            h2so4_rate = cm3_to_kg_cell(y[i_sa], MW_H2SO4, boxvol) / dt_c
            nitrate_rate = cm3_to_kg_cell(y[i_na_het], MW_NO3, boxvol) / dt_c
            y[i_sa] = y[i_na_het] = 0.0
            gas_before = bridge.gas_mass(y)
            gc_new = np.asarray(state.Gc).copy()
            gc_new[SRTSOA1:SRTSOA1 + k_cls] = bridge.class_sum(gas_before)
            state = state._replace(Gc=jnp.asarray(gc_new))
        forcing = AerosolForcing(jnp.float64(h2so4_rate), jnp.asarray(org_rate, dtype=jnp.float64),
                                 jnp.float64(nitrate_rate), jnp.asarray(weights, dtype=jnp.float64))
        state, diag = aerosol_outer_step(state, forcing, soa, dt_c, processes=processes, cfg=cfg)
        soa_bad += int(not bool(diag.soa_time_integrated))
        nitrate_expected += nitrate_rate * dt_c
        nitrate_added += float(diag.nitrate_added)
        if chem:
            mk_now = np.asarray(state.Mk)
            class_particle = mk_now[:, SRTSOA1:SRTSOA1 + k_cls].sum(axis=0)
            class_gas = np.asarray(state.Gc)[SRTSOA1:SRTSOA1 + k_cls]
            y = bridge.scatter(y, gas_before, class_gas, class_particle)
            y[held] = held_values
        if (i + 1) % out_every == 0 or i + 1 == n_int:
            record(t0 + dt_c, state, diag.cs_h2so4, diag.j_nuc, y if chem else None,
                   None if jp is None else jp.at(t0 + dt_c))
        if progress_cb is not None:
            progress_cb((i + 1) / n_int)
    if soa_bad:
        warnings.append(f"SOA partitioning hit its sub-step budget in {soa_bad} of {n_int} intervals")
    if nitrate_added < nitrate_expected * (1.0 - 1e-6):
        lost = kg_cell_to_ugm3(nitrate_expected - nitrate_added, boxvol)
        warnings.append(f"{lost:.3g} ug/m3 of N2O5-uptake nitrate had no populated bin to go to")

    return _assemble(s, rec, xk, boxvol, air, chem, k_cls, cstar, cstar_source,
                     bridge if chem else None, j_names, warnings,
                     {"gas_solver_steps": gas_steps, "runtime_s": time.perf_counter() - t_wall,
                      "code_version": code_version(), "coupling_dt_s": dt_c,
                      "processes": list(processes)})


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
    result = RunResult(
        scenario=_scenario_dict(s), time_h=rec["t"], diameter_nm=d_mid, diameter_edges_nm=d_edges,
        dndlog10dp_cm3=nk / dlog10, dmdlog10dp_ugm3=dry / dlog10, totals=totals,
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
        result.diagnostics["j_sources"] = {n: J_SOURCES[n] for n in j_names}
        js = np.array(rec["j"])
        result.photolysis = {n: js[:, j_names.index(n)] for n in REPORTED_J}
        result.photolysis["sza_deg"] = rec["sza"]
    return result


def _formula(atoms: dict[str, int]) -> str:
    order = ["C", "H", "N", "O", "S", "Cl", "Br"]
    return "".join(f"{a}{atoms[a] if atoms[a] > 1 else ''}" for a in order if atoms.get(a))


def _scenario_dict(s: Scenario) -> dict:
    d = dataclasses.asdict(s)
    d["start_utc"] = s.start_utc.isoformat()
    return d


def warmup() -> None:
    """Build the mechanism, the TUV-x J table and compile every preset's step shapes."""
    for name in list_presets():
        s = load_preset(name)
        short = s.replace(duration_h=2 * s.numerics.coupling_dt_s / 3600.0,
                          numerics={"output_dt_s": s.numerics.coupling_dt_s})
        run(short)
