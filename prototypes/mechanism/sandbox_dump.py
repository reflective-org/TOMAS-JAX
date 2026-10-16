"""Dump SANDBOX (stratchem-jax, JPL 19-5) gas-phase rate coefficients at fixed conditions.

Run with the SANDBOX venv; argv[1] = stratchem-jax directory, argv[2] = output JSON.
Only reads SANDBOX; writes the JSON consumed by rate_crosscheck.py.
"""
import json
import sys

sys.path.insert(0, sys.argv[1])
from mechanism import Env  # noqa: E402
from reactions import REACTIONS  # noqa: E402

KB = 1.380649e-23
CONDITIONS = {  # name: (T [K], p [Pa], H2O [mole fraction])
    "surface_298K": (298.15, 101325.0, 0.0155),     # about RH 49 % at 298 K
    "upper_trop_240K": (240.0, 30000.0, 1e-4),
    "strat_220K": (220.0, 5000.0, 5e-6),
}
out = []
for i, r in enumerate(REACTIONS):
    if r.kind != "gas" or not r.active:
        continue
    row = {"R": f"R{i + 1}", "klabel": r.klabel, "equation": r.equation,
           "reactants": r.reactants, "products": r.products, "note": r.note, "k": {}}
    for name, (t, p, x) in CONDITIONS.items():
        m = p / (KB * t) * 1e-6
        e = Env(T=t, M=m, P=p, H2O=x * m, WTR=x * 1e6, opt=0, sulfur_chain=True)  # full JPL set
        try:
            row["k"][name] = float(r.coeff_fn(e))
        except Exception as exc:  # noqa: BLE001
            row["k"][name] = f"error: {exc}"
    out.append(row)
json.dump({"conditions": CONDITIONS, "reactions": out}, open(sys.argv[2], "w"), indent=1)
print(len(out), "gas reactions")
