"""Check the exported problems against TOMAS-JAX itself.

1. RHS: bench_common.NumpyChem.rhs equals tomas_jax Mechanism.rhs (built from the
   parsed mechanism, not from the export) at the check state.
2. Jacobian: the analytic sparse Jacobian equals jax.jacfwd of Mechanism.rhs (726 and
   1353 species; dense autodiff is too big beyond that).
3. Rate constants: the k rule of the problem file equals the TOMAS-JAX driver's
   EnvRates.k(RO2, J) with the external reaction set, at the check state.
4. checks_<p>.json values are reproduced by the reader.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from tomas_jax.box.coupling import N2O5_HET_LABEL, tutorial_mechanism
from tomas_jax.box.scenario import load_preset
from tomas_jax.gaschem.environment import air_environment
from tomas_jax.gaschem.mechanism import EnvRates, Mechanism, RateModel

sys.path.insert(0, str(Path(__file__).parent))
from bench_common import PROBLEMS, NumpyChem, load_problem  # noqa: E402
from export_problems import COPIES, copies  # noqa: E402


def rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300 + np.abs(b).max() * 1e-14)))


def main():
    lim = tutorial_mechanism()
    rm = RateModel(lim, external=(N2O5_HET_LABEL,))
    i_het = rm.labels.index(N2O5_HET_LABEL)
    out = {}
    for pname in ("lim726_L2", "lim726_L3", "syn1353_L3", "syn4488_L3", "syn5742_L3"):
        P = load_problem(pname)
        mname = pname.split("_")[0]
        parsed, origin = (lim, np.arange(len(lim.reactions))) if mname == "lim726" else \
            copies(lim, COPIES[mname])
        mech = Mechanism.build(parsed)
        assert list(parsed.species) == P.mech.species
        chk = json.loads((PROBLEMS / f"checks_{pname}.json").read_text())
        y = np.asarray(chk["y_test"])
        k = np.asarray(chk["k_test"])
        chem = NumpyChem(P.mech)
        f_np = chem.rhs(y, k)
        f_jx = np.asarray(mech.rhs(jnp.asarray(y), jnp.asarray(k)))
        res = {"rhs_vs_tomas_jax": rel(f_np, f_jx),
               "rhs_vs_checks_file": rel(f_np, chk["f_test"]),
               "jac_v_vs_checks_file": rel(chem.jac(y, k) @ np.asarray(chk["v"]),
                                           chk["jac_v_test"])}
        # k rule vs the driver's EnvRates
        lev = "level2_limonene_soa" if pname.endswith("L2") else "level3_day_night"
        pr = load_preset(lev)
        air = air_environment(pr.environment.temp_k, pr.environment.pres_pa, pr.environment.rh)
        k_th, ro2_dep, j_coef = rm.at_env(air)
        rates = EnvRates(jnp.asarray(k_th[origin]), jnp.asarray(ro2_dep[origin]),
                         jnp.asarray(j_coef[origin]))
        k_drv = rates.k(mech.ro2(jnp.asarray(y)), jnp.asarray(P.j_table(900)[0]))
        k_het = P.meta["conditions"]["n2o5_uptake"]["k_het_per_s"]
        k_drv = np.asarray(k_drv.at[jnp.asarray(np.flatnonzero(origin == i_het))].set(k_het))
        res["k_rule_vs_driver"] = float(np.max(np.abs(k_drv - k) / np.maximum(np.abs(k_drv),
                                                                              1e-300)))
        if P.mech.n <= 1400:
            jd = np.asarray(jax.jacfwd(mech.rhs)(jnp.asarray(y), jnp.asarray(k)))
            js = chem.jac(y, k).toarray()
            res["jac_vs_jacfwd_max_abs_over_scale"] = float(np.max(np.abs(js - jd))
                                                           / np.max(np.abs(jd)))
            res["jac_entries_mismatch"] = int(np.sum(~np.isclose(js, jd, rtol=1e-12,
                                                                   atol=1e-30)))
        out[pname] = res
        print(pname, json.dumps(res))
    worst = max(v for r in out.values() for kk, v in r.items() if kk != "jac_entries_mismatch")
    assert worst < 1e-12, worst
    assert all(r.get("jac_entries_mismatch", 0) == 0 for r in out.values())
    print("all checks pass; worst relative difference", worst)


if __name__ == "__main__":
    main()
