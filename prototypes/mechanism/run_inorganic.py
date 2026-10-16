"""24 h gas-only box run of the inorganic + CH4 + SO2 module, and the same run with the
full limonene mechanism (limonene = 0). With no VOC the two must agree; the difference
measures solver noise. Also the sulfur budget and timing.

Same operator split as box/driver.py (k frozen per 900 s interval, J at the midpoint,
RO2 from the interval start). MCM clear-sky J (mode "mcm"), Buffalo, 1 July 2026.

Run: PYTHONPATH=<mech-wt> TUVX_JAX_ROOT=<tuvx-jax> python run_inorganic.py
"""
from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import jax.numpy as jnp
import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE))
from tomas_jax.gaschem.config import LIMONENE_EQN  # noqa: E402
from tomas_jax.gaschem.environment import air_environment  # noqa: E402
from tomas_jax.gaschem.kpp_parser import parse_eqn  # noqa: E402
from tomas_jax.gaschem.mechanism import EnvRates, Mechanism, RateModel  # noqa: E402
from tomas_jax.gaschem.photolysis import JProvider, PhotolysisSettings  # noqa: E402
from tomas_jax.gaschem.solver import integrate_interval  # noqa: E402

PPB = {"O3": 40.0, "NO2": 1.0, "NO": 0.2, "CO": 100.0, "CH4": 1900.0, "SO2": 1.0,
       "H2O2": 1.0, "HCHO": 1.0, "H2": 500.0, "HNO3": 0.5}
DT, HOURS = 900.0, 24
REPORT = ("O3", "NO2", "OH", "HO2", "SO2", "HNO3", "SA", "H2O2")


def run(path):
    parsed = parse_eqn(path)
    air = air_environment(298.15, 101325.0, 0.5)
    rm = RateModel(parsed)
    mech = Mechanism.build(parsed)
    k_th, dep, jc = rm.at_env(air)
    rates = EnvRates(jnp.asarray(k_th), jnp.asarray(dep), jnp.asarray(jc))
    jp = JProvider(rm.j_names, PhotolysisSettings("mcm", 42.9, -78.9,
                                                  datetime(2026, 7, 1, 4, 0, tzinfo=timezone.utc)))
    idx = {s: i for i, s in enumerate(parsed.species)}
    y = np.zeros(mech.n_species)
    for s, v in PPB.items():
        y[idx[s]] = v * 1e-9 * air["M"]
    s_atoms = np.array([parsed.atoms[s].get("S", 0) if s != "SA" else 1 for s in parsed.species])
    s0 = float(s_atoms @ y)
    series = {s: [float(y[idx[s]])] for s in REPORT}
    steps, t_solve = 0, 0.0
    # first interval includes compilation; timed separately
    for i in range(int(HOURS * 3600 / DT)):
        k = rates.k(mech.ro2(jnp.asarray(y)), jnp.asarray(jp.at(i * DT + 0.5 * DT)))
        t0 = time.perf_counter()
        y_new, n = integrate_interval(mech, jnp.asarray(y), k, DT)
        y = np.asarray(y_new).copy()
        dt = time.perf_counter() - t0
        if i == 0:
            t_first = dt
        else:
            t_solve += dt
            steps += int(n)
        for s in REPORT:
            series[s].append(float(y[idx[s]]))
    return {"species": mech.n_species, "reactions": len(parsed.reactions),
            "first_interval_incl_compile_s": t_first, "warm_solve_s_95_intervals": t_solve,
            "accepted_steps_95_intervals": steps, "ms_per_step": 1e3 * t_solve / max(steps, 1),
            "sulfur_closure_rel": abs(float(s_atoms @ y) - s0) / s0,
            "end_ppb": {s: series[s][-1] / air["M"] * 1e9 for s in REPORT},
            "oh_max_cm3": max(series["OH"]), "series": series}


inorg = run(HERE / "modules" / "mcm_inorganic_c1_sulfur.eqn")
full = run(LIMONENE_EQN)
diff = {s: float(np.max(np.abs(np.array(inorg["series"][s]) - np.array(full["series"][s]))
                        / np.maximum(np.abs(np.array(full["series"][s])), 1e3)))
        for s in REPORT}
res = {"inorganic": {k: v for k, v in inorg.items() if k != "series"},
       "limonene_file_no_voc": {k: v for k, v in full.items() if k != "series"},
       "max_rel_diff_inorganic_vs_full": diff,
       "so2_lifetime_h_24h_mean": None}
so2 = np.array(inorg["series"]["SO2"])
res["so2_lifetime_h_24h_mean"] = float(HOURS / np.log(so2[0] / so2[-1])) if so2[-1] > 0 else None
(HERE / "results").mkdir(exist_ok=True)
(HERE / "results" / "run_inorganic.json").write_text(json.dumps(res, indent=1))
print(json.dumps(res, indent=1))
