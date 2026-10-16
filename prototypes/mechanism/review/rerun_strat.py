"""Independent re-run of the strat case (MCM matched; MCM + JPL photolysis + O2 photolysis)."""
import json, sys, time
from pathlib import Path
import numpy as np
CMP = Path(sys.argv[1]); OUT = Path(sys.argv[2])
sys.path.insert(0, str(CMP))
import run_mcm as R
from tomas_jax.gaschem.kpp_parser import Reaction, parse_eqn
cases = json.loads((CMP / "cases_resolved.json").read_text())
case = cases["strat"]
jt = json.loads((CMP / "results" / "jtable_strat.json").read_text())
jo2 = json.loads(Path(sys.argv[3]).read_text())
jt["J"]["O2+hv->O+O"] = jo2
def summarise(res):
    t = np.array(res["t_s"]); c = {k: np.array(v) for k, v in res["conc"].items()}
    M = case["M"]; d0 = (case["days"] - 1) * 86400
    w = (t >= d0) & (t <= d0 + 86400)
    sza = np.interp(t, jt["t_s"], jt["sza_deg"])
    noon = np.where(w)[0][np.argmin(sza[w])]
    k = res["k"]["OH+SO2 (KMT12)"]
    tau = 1 / (k * c["OH"][w].mean()) / 86400
    return {"OH_noon": c["OH"][noon], "OH_mean": c["OH"][w].mean(), "HO2_noon": c["HO2"][noon],
            "N2O5_noon_ppt": c["N2O5"][noon] / M * 1e12, "NOx_noon_ppt": (c["NO"][noon] + c["NO2"][noon]) / M * 1e12,
            "O3_change_ppb_day": (c["O3"][w][-1] - c["O3"][w][0]) / M * 1e9,
            "SO2_tau_days": tau, "SA_day_ppt": (c["SA"][w][-1] - c["SA"][w][0]) / M * 1e12}
out = {}
t0 = time.perf_counter()
out["matched"] = summarise(R.run_case("strat", case, jt, "matched"))
out["jplphot"] = summarise(R.run_case("strat", case, jt, "jplphot"))
# + O2 photolysis: dummy carrier O2X (held = [O2]) so O2X + hv -> O2X + 2 O is a source of 2 O at J_O2
orig_parse = R.parse_eqn
def parse_with_o2(p):
    pm = orig_parse(p)
    pm.species.append("O2X"); pm.atoms["O2X"] = {}
    pm.reactions.append(Reaction("JO2", ("O2X",), (("O2X", 1.0), ("O", 2.0)), "0.0", True))
    return pm
R.parse_eqn = parse_with_o2
R.EXTRA_PHOTOLYSIS = dict(R.EXTRA_PHOTOLYSIS, JO2=("O2X", (("O2X", 1.0), ("O", 2.0)), "O2+hv->O+O"))
case2 = dict(case, ppb=dict(case["ppb"], O2X=case["O2"] / case["M"] * 1e9))
# run_case appends EXTRA_PHOTOLYSIS reactions itself; avoid the double JO2 append
def parse_plain_species(p):
    pm = orig_parse(p); pm.species.append("O2X"); pm.atoms["O2X"] = {}; return pm
R.parse_eqn = parse_plain_species
res = R.run_case("strat", case2, jt, "jplphot")
out["jplphot_O2phot"] = summarise(res)
out["wall_s"] = time.perf_counter() - t0
OUT.write_text(json.dumps(out, indent=1, default=float))
for k, v in out.items():
    print(k, v if not isinstance(v, dict) else {a: f"{b:.3g}" for a, b in v.items()})
