import json, sys
from pathlib import Path
import numpy as np
CMP = Path(sys.argv[2]); sys.path.insert(0, str(CMP))
sys.argv = [sys.argv[0], sys.argv[1]]
import run_sandbox as S
case = json.loads((CMP / "cases_resolved.json").read_text())["strat"]
jt = json.loads((CMP / "results" / "jtable_strat.json").read_text())
res = S.run_case("strat", case, jt, 0, False)
t = np.array(res["t_s"]); c = {k: np.array(v) for k, v in res["conc"].items()}
M = case["M"]; d0 = 4 * 86400; w = (t >= d0) & (t <= d0 + 86400)
sza = np.interp(t, jt["t_s"], jt["sza_deg"]); noon = np.where(w)[0][np.argmin(sza[w])]
k = res["k"]["OH+SO2 (JPL I4)"]
print({"wall_s": round(res["wall_s"], 1), "OH_noon": f"{c['OH'][noon]:.3g}", "OH_mean": f"{c['OH'][w].mean():.3g}",
       "HO2_noon": f"{c['HO2'][noon]:.3g}", "N2O5_ppt": f"{c['N2O5'][noon]/M*1e12:.3g}",
       "SO2_tau_d": f"{1/(k*c['OH'][w].mean())/86400:.3g}", "k_OH_SO2": f"{k:.3g}",
       "O3_change": f"{(c['O3'][w][-1]-c['O3'][w][0])/M*1e9:.3g}"})
