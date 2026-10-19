"""Molar masses PyCHAM will use for every species of an xml (run in the PyCHAM venv).

Usage: python -I pycham_mw.py species.xml out.json

PyCHAM (ac_by_cs = 0) takes molar masses from the SMILES in the xml with pybel
(``init_conc.py``: ``y_mw[i] = Pybel_objects[i].molwt``) and fills in a few species
without a SMILES (``xml_interr.py``). The benchmark needs these exact values to convert
c* [ug/m3] to the vapour pressures [Pa] PyCHAM is given, so that PyCHAM's equilibrium
gas concentration in mass units equals the intended c*.
"""
import json
import sys

import openbabel.pybel as pybel
import xmltodict

# xml_interr.py fallbacks for species without a SMILES
FALLBACK = {"O3": "[O-][O+]=O", "NO2": "[N+](=O)[O-]", "NO3": "[N+](=O)([O-])[O]",
            "PROD": "C", "NA": "[N+](=O)(O)[O-]", "SA": "OS(=O)(=O)O"}


def main(xml_path, out_path):
    with open(xml_path) as fh:
        doc = xmltodict.parse(fh.read())
    out = {}
    for sp in doc["mechanism"]["species_defs"]["species"]:
        name = sp["@species_name"]
        if not name:
            continue
        smiles = sp.get("smiles") or FALLBACK.get(name) or name
        out[name] = {"smiles": smiles, "mw": pybel.readstring("smi", smiles).molwt}
    with open(out_path, "w") as fh:
        json.dump(out, fh)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1], sys.argv[2]))
