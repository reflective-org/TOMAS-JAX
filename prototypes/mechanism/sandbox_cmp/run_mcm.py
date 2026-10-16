"""Gas-only box runs of mcm_inorganic_sulfur.eqn with tomas_jax.gaschem, both cases.

Same operator split as box/driver.py: k frozen over each 300 s interval, J (from the
shared TUV-x table) at the interval midpoint, RO2 from the interval start, Kvaerno3
with the package tolerances. MCM's fixed aerosol sinks <33> HNO3 = NA and
<34> N2O5 = NA + NA are set to zero (chemistry only, no aerosol); <35> SO3 = SA
(SO3 + H2O, gas phase) is kept. If the case sets hold_NOx_ppb, NO and NO2 are rescaled
after every interval so NO + NO2 = hold_NOx_ppb with the NO/NO2 ratio kept (a
constrained-NOx box, the usual way to represent a steady NOx source).

Variants per case:
  matched    - same initial species as SANDBOX (CO = H2 = HCHO = 0 at t = 0)
  background - matched + the case's mcm_background (CO, H2), which SANDBOX lacks
  jplphot    - matched + three photolysis channels MCM does not have but SANDBOX does
               (N2O5 -> NO2 + NO3, HO2NO2 -> HO2 + NO2, HO2NO2 -> OH + NO3), added in
               memory with J from the same TUV-x table (k supplied per interval via
               RateModel's `external` hook); tests how much of the stratospheric NOx
               difference is photolysis coverage

Run: TUVX_JAX_ROOT=<tuvx-jax> PYTHONPATH=<mech-wt> <mech-wt>/.venv/bin/python run_mcm.py
Writes results/mcm_<case>_<variant>.json.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import tomas_jax  # noqa: F401  (enables float64)
from tomas_jax.gaschem.kpp_parser import Reaction, parse_eqn
from tomas_jax.gaschem.mcm_rates import complex_rates
from tomas_jax.gaschem.mechanism import EnvRates, Mechanism, RateModel
from tomas_jax.gaschem.solver import integrate_interval

HERE = Path(__file__).parent
EQN = HERE.parent / "mcm_inorganic_sulfur.eqn"
DT = 300.0                                   # coupling interval [s]
NO_AEROSOL = ("33", "34")                    # MCM fixed aerosol sinks -> k = 0
SAVE = ["O3", "O", "O1D", "OH", "HO2", "NO", "NO2", "NO3", "N2O5", "HNO3", "HONO", "HO2NO2",
        "CH3NO3", "CH3O2NO2", "CH3O2", "CH3OOH", "HCHO", "CO", "H2", "H2O2", "SO2", "HSO3",
        "SO3", "SA", "CH4"]


# label -> (reactant, products, TUV-x key) for the jplphot variant
EXTRA_PHOTOLYSIS = {
    "JPL_N2O5": ("N2O5", (("NO2", 1.0), ("NO3", 1.0)), "N2O5+hv->NO2+NO3"),
    "JPL_HNO4a": ("HO2NO2", (("HO2", 1.0), ("NO2", 1.0)), "HNO4+hv->HO2+NO2"),
    "JPL_HNO4b": ("HO2NO2", (("OH", 1.0), ("NO3", 1.0)), "HNO4+hv->OH+NO3"),
}


def run_case(cname: str, case: dict, jt: dict, variant: str) -> dict:
    parsed = parse_eqn(EQN)
    extra = EXTRA_PHOTOLYSIS if variant == "jplphot" else {}
    for lab, (reac, prods, _) in extra.items():
        parsed.reactions.append(Reaction(lab, (reac,), prods, "0.0", True))
    # external k: NO_AEROSOL stay 0 (reaction off); extra photolysis set per interval
    rm = RateModel(parsed, external=NO_AEROSOL + tuple(extra))
    mech = Mechanism.build(parsed)
    env = {"TEMP": case["T"], "M": case["M"], "O2": case["O2"], "N2": case["N2"],
           "H2O": case["H2O"]}
    k_th, dep, jc = rm.at_env(env)
    rates = EnvRates(jnp.asarray(k_th), jnp.asarray(dep), jnp.asarray(jc))
    tab_t = np.asarray(jt["t_s"])
    tab = np.array([jt["J"][jt["mcm_map"][n]] for n in rm.j_names])   # [nJ, nt]
    extra_idx = np.array([rm.labels.index(lab) for lab in extra], dtype=int)
    extra_tab = [np.asarray(jt["J"][key]) for _, _, key in extra.values()]

    idx = {s: i for i, s in enumerate(parsed.species)}
    y = np.zeros(mech.n_species)
    ppb = dict(case["ppb"])
    if variant == "background":
        ppb.update(case["mcm_background"])
    for s, v in ppb.items():
        y[idx[s]] = v * 1e-9 * case["M"]
    n_int = int(round(case["days"] * 86400 / DT))
    out = {s: [float(y[idx[s]])] for s in SAVE}
    t_out = [0.0]
    steps, wall = 0, time.perf_counter()
    for i in range(n_int):
        tm = (i + 0.5) * DT
        j = np.array([np.interp(tm, tab_t, row) for row in tab])
        k = rates.k(mech.ro2(jnp.asarray(y)), jnp.asarray(j))
        if extra:
            k = k.at[extra_idx].set(jnp.asarray([np.interp(tm, tab_t, row) for row in extra_tab]))
        y_new, n = integrate_interval(mech, jnp.asarray(y), k, DT)
        y = np.asarray(y_new).copy()
        steps += int(n)
        if case.get("hold_NOx_ppb"):               # constrained NOx (NO/NO2 ratio kept)
            f = case["hold_NOx_ppb"] * 1e-9 * case["M"] / (y[idx["NO"]] + y[idx["NO2"]])
            y[idx["NO"]] *= f
            y[idx["NO2"]] *= f
        t_out.append((i + 1) * DT)
        for s in SAVE:
            out[s].append(float(y[idx[s]]))
    wall = time.perf_counter() - wall
    lab = rm.labels
    K = complex_rates(case["T"], case["M"], case["O2"], case["N2"], case["H2O"])
    kd = {
        "OH+SO2 (KMT12)": float(K["KMT12"]),
        "O+SO2 (<30>)": float(k_th[lab.index("30")]),
        "HSO3+O2 (<32>, 1/s)": float(k_th[lab.index("32")]),
        "SO3+H2O (<35>, 1/s)": float(k_th[lab.index("35")]),
        "OH+NO2 (KMT08)": float(K["KMT08"]),
        "HO2+NO (<24>)": float(k_th[lab.index("24")]),
        "NO+O3 (<7>)": float(k_th[lab.index("7")]),
        "OH+CH4 (<46>)": float(k_th[lab.index("46")]),
        "OH+CO (KMT05)": float(K["KMT05"]),
        "HO2+HO2 (<20>)": float(k_th[lab.index("20")]),
        "OH+HO2 (<19>)": float(k_th[lab.index("19")]),
        "OH+O3 (<14>)": float(k_th[lab.index("14")]),
        "HO2+O3 (<18>)": float(k_th[lab.index("18")]),
        "HO2+NO2+M (KMT09)": float(K["KMT09"]),
        "HO2NO2 -> HO2+NO2 (KMT10, 1/s)": float(K["KMT10"]),
        "N2O5 -> NO2+NO3 (KMT04, 1/s)": float(K["KMT04"]),
        "O1D -> 2OH (<13>, 1/s)": float(k_th[lab.index("13")]),
        "O1D -> O (<6>, 1/s)": float(k_th[lab.index("6")]),
        "CH3O2+NO -> NO2 (<47>)": float(k_th[lab.index("47")]),
        "O+NO2 -> NO (<4>)": float(k_th[lab.index("4")]),
        "O+O3 (<2>)": float(k_th[lab.index("2")]),
    }
    return {"model": "MCM v3.3.1 subset (tomas_jax.gaschem)", "case": cname, "variant": variant,
            "M": case["M"], "t_s": t_out, "conc": out, "k": kd,
            "species": len(parsed.species), "reactions": len(parsed.reactions),
            "accepted_steps": steps, "wall_s": wall,
            "s_total_rel_change": None}


def main() -> None:
    cases = json.loads((HERE / "cases_resolved.json").read_text())
    for cname, case in cases.items():
        jt = json.loads((HERE / "results" / f"jtable_{cname}.json").read_text())
        for variant in ("matched", "background", "jplphot"):
            res = run_case(cname, case, jt, variant)
            c = res["conc"]
            s0 = c["SO2"][0]
            s1 = c["SO2"][-1] + c["HSO3"][-1] + c["SO3"][-1] + c["SA"][-1]
            res["s_total_rel_change"] = abs(s1 - s0) / s0
            (HERE / "results" / f"mcm_{cname}_{variant}.json").write_text(json.dumps(res))
            print(f"{cname}/{variant}: {res['accepted_steps']} steps, {res['wall_s']:.1f} s, "
                  f"S closure {res['s_total_rel_change']:.1e}, OH max {max(c['OH']):.2e}, "
                  f"SA end {c['SA'][-1] / case['M'] * 1e12:.1f} ppt")


if __name__ == "__main__":
    main()
