import json, sys
import numpy as np
from tuvx_photolysis import PhotolysisCalculator
from tomas_jax.gaschem.config import TUVX_CONFIG
from tomas_jax.gaschem.photolysis import tuvx_root
root = tuvx_root(); calc = PhotolysisCalculator.from_tuvx_json(root / TUVX_CONFIG, data_root=root)
jt = json.load(open(sys.argv[1])); z = jt["altitude_km"]; sza = np.array(jt["sza_deg"])
grid = np.arange(0, 90, 1.0)
jo2 = np.array([float(np.interp(z, calc.height_edges_km, calc.rate_constants_profile(float(s), 1.0162, branching=True)["O2+hv->O+O"])) for s in grid])
j = np.where(sza < 90, np.interp(np.minimum(sza, 89.9), grid, jo2), 0.0)
json.dump(j.tolist(), open(sys.argv[2], "w"))
