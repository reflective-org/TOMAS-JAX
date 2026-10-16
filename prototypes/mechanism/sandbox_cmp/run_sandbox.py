"""SANDBOX stratchem-jax (JPL 19-5) gas-only runs of the same two cases, halogens zero.

Run with the SANDBOX venv. Reads SANDBOX only (sys.path), writes JSON here.

Setup, following SANDBOX driver.integrate_sza / rhs.concs_het:
  * dC/dt = reactions.MECHANISM.dCdt(env, x) with env = reactions.build_env(cfg, x, ...);
  * photolysis: absolute J from the shared TUV-x table (results/jtable_<case>.json, same
    numbers the MCM run uses), passed as env.j_values keyed by SANDBOX equation and linearly
    interpolated in time (as SANDBOX's adapter does); j_scale = 0, so no reference J leaks in;
  * sulfur_chain = True (SO2 -> SO3 -> H2SO4, the chain SANDBOX uses in its sza/tuvx modes);
  * opt = 0: explicit O(1D) chemistry (O1D + O2, O1D + N2 quenching, O1D + H2O, O1D + CH4),
    not the opt = 1 water-split shortcut (opt = 1 is run as a sensitivity);
  * SA = 0 (no aerosol: every heterogeneous rate is 0), all Cl/Br species 0, C2H6 0.
    The Env is built as reactions.build_env builds it, except that the uptake gammas are
    set to 0 instead of computed: hetgammas_jpl00 divides by zero when HCl = 0, and with
    SA = 0 every heterogeneous rate is 0 whatever gamma is;
  * M, O2 (0.2095 M) and H2O from cases_resolved.json, so both models see the same air;
  * variant opt0_fill: opt = 0 plus three reactions SANDBOX lacks and MCM has, added in
    memory: the O3 -> O2 + O(3P) photolysis channel (its adapter calls it a null cycle;
    J from the TUV-x O(3P) channel), and the thermal decompositions HNO4 -> HO2 + NO2 and
    N2O5 -> NO2 + NO3 with MCM's KMT10 and KMT04 at the case T and M (read from
    results/mcm_<case>_matched.json, so run_mcm.py runs first). A diagnostic of coverage,
    not a proposed SANDBOX change;
  * SciPy BDF, rtol 1e-4, atol 1e-3 molec/cm3, one-day chunks, output every 300 s;
    if the case sets hold_NOx_ppb, 300 s chunks and NO, NO2 rescaled after each chunk so
    NO + NO2 = hold_NOx_ppb (NO/NO2 ratio kept), exactly as run_mcm.py does.

Run: <SANDBOX>/.venv/bin/python run_sandbox.py <SANDBOX>/stratchem-jax
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

sys.path.insert(0, sys.argv[1])
from config import IDX, SPECIES, ModelConfig  # noqa: E402
from mechanism import Env, Mechanism, photo, react  # noqa: E402
from reactions import MECHANISM, REACTIONS  # noqa: E402

HERE = Path(__file__).parent
DT_OUT = 300.0
MAX_WALL_S = 400.0                             # per run; abort beyond this
SAVE = ["O3", "O", "O1D", "OH", "HO2", "NO", "NO2", "NO3", "N2O5", "HNO3", "HONO", "HNO4",
        "H2O2", "SO2", "SO3", "H2SO4", "CH4", "CH3", "O2", "H2O"]


ZERO_GAMMAS = {"Yhocl": 0.0, "Yclnh2o": 0.0, "Yclnhcl": 0.0, "Yn2o5": 0.0, "Ybrono2": 0.0}


def make_env(cfg, x, j_values) -> Env:
    """reactions.build_env without the gamma computation (SA = 0, see module docstring)."""
    return Env(T=cfg.T, M=cfg.M, P=cfg.P, SA=cfg.SA, WTR=cfg.WTR, opt=cfg.opt,
               gammas=ZERO_GAMMAS, j_scale=0.0, H2O=x[IDX["H2O"]], j_values=j_values,
               sulfur_chain=True)


def by_eq(eq: str):
    hits = [r for r in REACTIONS if r.equation == eq and r.active]
    if len(hits) != 1:
        raise KeyError(eq)
    return hits[0]


O3P_EQ = "O3 -> O2 + O"
O3P_KEY = "O3+hv->O2+O(3P)"


def fill_mechanism(cname: str) -> Mechanism:
    k = json.loads((HERE / "results" / f"mcm_{cname}_matched.json").read_text())["k"]
    k10, k04 = k["HO2NO2 -> HO2+NO2 (KMT10, 1/s)"], k["N2O5 -> NO2+NO3 (KMT04, 1/s)"]
    return Mechanism(REACTIONS + [
        photo(O3P_EQ, j45=0.0, note="TUV-x O(3P) channel"),
        react("HNO4 -> HO2 + NO2", lambda e: k10, "MCM KMT10 (diagnostic)"),
        react("N2O5 -> NO2 + NO3", lambda e: k04, "MCM KMT04 (diagnostic)", kind="gas"),
    ])


def run_case(cname: str, case: dict, jt: dict, opt: int, o3p: bool = False) -> dict:
    mech = fill_mechanism(cname) if o3p else MECHANISM
    cfg = ModelConfig(T=case["T"], P=case["P_hPa"], SA=0.0, WTR=case["H2O_ppm"], Yn2o5=0.0,
                      opt=opt, M=case["M"], photolysis="tuvx", latitude=case["lat"],
                      longitude=case["lon"])
    M = case["M"]
    x0 = np.zeros(len(SPECIES))
    for s, v in case["ppb"].items():
        x0[IDX[s]] = v * 1e-9 * M
    x0[IDX["O2"]] = case["O2"]
    x0[IDX["H2O"]] = case["H2O"]
    tab_t = np.asarray(jt["t_s"])
    jtab = {eq: np.asarray(jt["J"][key]) for eq, key in jt["sandbox_map"].items()}
    if o3p:
        jtab[O3P_EQ] = np.asarray(jt["J"][O3P_KEY])
    wall0 = time.perf_counter()
    nfev = [0]

    def rhs(t, x):
        if time.perf_counter() - wall0 > MAX_WALL_S:
            raise RuntimeError(f"{cname}: wall-clock budget {MAX_WALL_S} s exceeded")
        nfev[0] += 1
        jv = {eq: float(np.interp(t, tab_t, v)) for eq, v in jtab.items()}
        env = make_env(cfg, x, jv)
        return mech.dCdt(env, np.asarray(x, dtype=float))

    atol = 1e-3
    total = case["days"] * 86400.0
    hold = case.get("hold_NOx_ppb")
    chunk = DT_OUT if hold else 86400.0           # NOx reset needs one solve per 300 s
    t_all, x_all, cur, t0 = [0.0], [x0], x0, 0.0
    while t0 < total - 1e-6:
        t1 = min(t0 + chunk, total)
        grid = np.arange(t0, t1 + 0.5 * DT_OUT, DT_OUT)
        sol = solve_ivp(rhs, (t0, t1), cur, method="BDF", t_eval=grid, rtol=1e-4, atol=atol,
                        first_step=1e-6)
        if not sol.success:
            raise RuntimeError(f"{cname}: BDF failed on [{t0}, {t1}]: {sol.message}")
        cur = sol.y[:, -1].copy()
        if hold:                                   # same NOx constraint as run_mcm.py
            f = hold * 1e-9 * M / (cur[IDX["NO"]] + cur[IDX["NO2"]])
            cur[IDX["NO"]] *= f
            cur[IDX["NO2"]] *= f
        t_all += sol.t[1:].tolist()
        x_all += list(sol.y.T[1:-1]) + [cur]
        t0 = t1
    X = np.array(x_all)
    e = Env(T=cfg.T, M=M, P=cfg.P, WTR=cfg.WTR, opt=opt, H2O=case["H2O"], sulfur_chain=True)
    o2 = case["O2"]
    kd = {
        "OH+SO2 (JPL I4)": by_eq("SO2 + OH -> SO3 + HO2").coeff_fn(e),
        "HO2+SO2 (JPL I34 upper limit)": by_eq("SO2 + HO2 -> SO3 + OH").coeff_fn(e),
        "SO3+H2O (I79, 1/s)": by_eq("SO3 + H2O -> H2SO4").coeff_fn(e) * case["H2O"],
        "OH+NO2 (C4)": by_eq("OH + NO2 + M -> HNO3 + M").coeff_fn(e),
        "HO2+NO (C10)": by_eq("HO2 + NO -> NO2 + OH").coeff_fn(e),
        "NO+O3 (C19)": by_eq("NO + O3 -> NO2 + O2").coeff_fn(e),
        "OH+CH4 (D14)": by_eq("CH4 + OH -> HO2").coeff_fn(e),
        "HO2+HO2 (B13)": by_eq("HO2 + HO2 -> H2O2").coeff_fn(e),
        "OH+HO2": by_eq("OH + HO2 -> H2O + O2").coeff_fn(e),
        "OH+O3": by_eq("OH + O3 -> HO2 + O2").coeff_fn(e),
        "HO2+O3": by_eq("HO2 + O3 -> OH + 2 O2").coeff_fn(e),
        "HO2+NO2+M (C6)": by_eq("HO2 + NO2 + M -> HNO4 + M").coeff_fn(e),
        "O1D -> 2OH (1/s)": by_eq("O1D + H2O -> 2 OH").coeff_fn(e) * case["H2O"],
        "O1D -> O (1/s, opt=0)": (3.3e-11 * np.exp(55.0 / cfg.T) * o2
                                  + 2.15e-11 * np.exp(110.0 / cfg.T) * 0.79 * M),
        "O1D+CH4": by_eq("O1D + CH4 -> CH3 + OH").coeff_fn(e),
        "O+NO2 -> NO (K1 activation)": by_eq("O + NO2 -> NO + O2").coeff_fn(e),
        "O+O3": by_eq("O + O3 -> 2 O2").coeff_fn(e),
    }
    return {"model": "SANDBOX stratchem-jax (JPL 19-5)", "case": cname,
            "variant": f"opt{opt}" + ("_fill" if o3p else ""),
            "M": M, "t_s": t_all, "conc": {s: X[:, IDX[s]].tolist() for s in SAVE},
            "k": {k: float(v) for k, v in kd.items()},
            "rhs_calls": nfev[0], "wall_s": time.perf_counter() - wall0,
            "s_total_rel_change": float(abs(X[-1, IDX["SO2"]] + X[-1, IDX["SO3"]]
                                            + X[-1, IDX["H2SO4"]] - x0[IDX["SO2"]])
                                        / x0[IDX["SO2"]])}


def main() -> None:
    cases = json.loads((HERE / "cases_resolved.json").read_text())
    for cname, case in cases.items():
        jt = json.loads((HERE / "results" / f"jtable_{cname}.json").read_text())
        for opt, o3p in ((0, False), (1, False), (0, True)):
            res = run_case(cname, case, jt, opt, o3p)
            (HERE / "results" / f"sandbox_{cname}_{res['variant']}.json").write_text(
                json.dumps(res))
            c = res["conc"]
            print(f"{cname}/{res['variant']}: {res['rhs_calls']} RHS calls, {res['wall_s']:.1f} s, "
                  f"S closure {res['s_total_rel_change']:.1e}, OH max {max(c['OH']):.2e}, "
                  f"H2SO4 end {c['H2SO4'][-1] / case['M'] * 1e12:.1f} ppt")


if __name__ == "__main__":
    main()
