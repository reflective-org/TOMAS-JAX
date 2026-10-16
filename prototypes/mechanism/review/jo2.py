import json, sys
import numpy as np
from tuvx_photolysis import PhotolysisCalculator
from tomas_jax.gaschem.config import TUVX_CONFIG
from tomas_jax.gaschem.photolysis import tuvx_root
root = tuvx_root()
calc = PhotolysisCalculator.from_tuvx_json(root / TUVX_CONFIG, data_root=root)
names = calc.reaction_names(branching=True)
print([n for n in names if n.startswith("O2")])
jt = json.load(open(sys.argv[1]))
z = jt["altitude_km"]; sza = np.array(jt["sza_deg"]); t = np.array(jt["t_s"])
grid = np.arange(0, 90, 2.0); jo2 = []
for s in grid:
    prof = calc.rate_constants_profile(float(s), 1.0162, branching=True)
    jo2.append(float(np.interp(z, calc.height_edges_km, prof["O2+hv->O+O"])))
jo2 = np.array(jo2)
day = t < 86400
jt_o2 = np.where(sza < 90, np.interp(np.minimum(sza, 89.9), grid, jo2), 0.0)
M = 1.6461296627363453e18; O2 = 0.2095 * M
mean = jt_o2[day].mean()
print(f"z={z:.2f} km J_O2 noon {jo2[0]:.3e} s-1, 24h mean {mean:.3e} s-1")
print(f"P(Ox)=2 J [O2]: 24h mean {2*mean*O2:.3e} cm-3 s-1 = {2*mean*O2*86400/M*1e9:.1f} ppb/day")
