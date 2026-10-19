"""Load PyCHAM (v5.8.0 csv-style) results into numpy arrays.

Usage:
    from load_pycham import load_pycham
    r = load_pycham("/path/to/output_folder")
    r["t_s"]            # (nt,) time since start [s]
    r["names"]          # list of component names (chemical-scheme names; also H2O, maybe 'core')
    r["gas_ppb"]        # (nt, ncomp) gas-phase mixing ratio [ppb]
    r["gas_molec_cm3"]  # (nt, ncomp) gas-phase concentration [molecules cm-3]
    r["part_molec_cm3"] # (nt, nbins, ncomp) particle phase per size bin [molecules cm-3 air]
    r["part_ug_m3"]     # (nt, nbins, ncomp) particle phase per size bin [ug m-3]
    r["wall_molec_cm3"] # (nt, nwall, ncomp) wall (gas-wall partitioning) [molecules cm-3 air]
    r["N_bin"]          # (nt, nbins) particle number of each bin [# cm-3]; the number that
                        #   goes with part_* and rad_um (file particle_number_concentration_wet)
    r["N_dry_rebinned"] # (nt, nbins) particle number re-binned by DRY volume against the
                        #   (wet) bin bounds [# cm-3] (file particle_number_concentration_dry,
                        #   rec.py). Does NOT line up with the per-bin composition: a
                        #   population whose dry volume falls below its bin's lower bound is
                        #   counted one bin lower. Only its sum over bins is meaningful.
    r["rad_um"]         # (nt, nbins) single-particle radius per bin [um]. At t = 0 with
                        #   Vwat_inc = 0 this is the radius of the size input (the DRY
                        #   radius, before seed water is added); later rows include water.
                        #   For a wet size at every time use the composition and MV.
    r["mv_cm3_mol"]     # (ncomp,) molar volume of every component [cm3 mol-1] (MV.npy);
                        #   wet volume per particle = sum(part_molec_cm3 / N_A * MV) / N_bin
    r["env"]            # (nt, 4) T [K], P [Pa], RH [0-1], light transmission factor
    r["mw_g_mol"]       # (ncomp,) molar masses [g mol-1]
    r["psat_298_Pa"]    # (ncomp,) pure-component saturation vapour pressure at 298.15 K [Pa]

Column layout of concentrations_all_components_all_times_gas_particle_wall:
[all comps gas (ppb)] + [all comps size bin 1 (molec/cm3)] + ... + [bin n] + [wall 1] + ...
Header: line 1 is a description, line 2 starts with '#' and lists names like
O3_g, O3_p1, ..., O3_w (np.loadtxt skips both via skiprows=1 + comments='#').

This is the benchmark's copy (tomas-jax/benchmarks/pycham); the copy in pycham-bench/
predates the N_bin / N_dry_rebinned / mv_cm3_mol keys.
"""
import os

import numpy as np

AVOGADRO = 6.02214076e23


def _constants(out):
    consts = {}
    with open(os.path.join(out, "model_and_component_constants")) as fh:
        for line in fh:
            key, _, val = line.partition(",")
            consts[key.strip()] = val.strip()
    return consts


def load_pycham(out):
    names = [str(n) for n in np.load(os.path.join(out, "comp_namelist.npy"), allow_pickle=True)]
    ncomp = len(names)
    consts = _constants(out)
    wall_on = int(float(consts.get("wall_on_flag_0forNO_>0forYES", "0")))

    t = np.atleast_1d(np.loadtxt(os.path.join(out, "time"), delimiter=",", skiprows=1))
    y = np.atleast_2d(np.loadtxt(
        os.path.join(out, "concentrations_all_components_all_times_gas_particle_wall"),
        delimiter=",", skiprows=1))
    # number_of_size_bins in the constants file = particle bins + wall bins (wall_on = number of walls)
    nwall = wall_on
    nbins = int(consts["number_of_size_bins"]) - nwall
    assert (1 + nbins + nwall) * ncomp == y.shape[1], "unexpected column count"

    gas_ppb = y[:, :ncomp]
    # ppb -> molec/cm3 factor per recorded time (depends on T and P)
    fac_txt = consts["factor_for_multiplying_ppb_to_get_molec/cm3_with_time"]
    cfac = np.array([float(v) for v in fac_txt.replace("[", " ").replace("]", " ").split(",") if v.strip()])
    cfac = cfac[: len(t)] if cfac.size >= len(t) else np.full(len(t), cfac[0])

    part = y[:, ncomp: ncomp * (1 + nbins)].reshape(len(t), nbins, ncomp)
    wall = y[:, ncomp * (1 + nbins):].reshape(len(t), nwall, ncomp)
    mw = np.load(os.path.join(out, "y_mw.npy"), allow_pickle=True).astype(float).ravel()

    res = dict(
        t_s=t, names=names, gas_ppb=gas_ppb, gas_molec_cm3=gas_ppb * cfac[:, None],
        part_molec_cm3=part, part_ug_m3=part / AVOGADRO * mw[None, None, :] * 1.0e12,
        wall_molec_cm3=wall, mw_g_mol=mw, ppb_to_molec_cm3=cfac,
        env=np.atleast_2d(np.loadtxt(os.path.join(out, "chamber_environmental_conditions"),
                                     delimiter=",", skiprows=1)),
        psat_298_Pa=np.load(os.path.join(out, "pure_component_saturation_vapour_pressures_at_298p15K_Pa.npy"),
                            allow_pickle=True).astype(float).ravel(),
    )
    if nbins > 0:
        def per_bin(name):
            arr = np.atleast_2d(np.loadtxt(os.path.join(out, name), delimiter=",", skiprows=1))
            if arr.shape != (len(t), nbins):
                raise ValueError(f"{name}: shape {arr.shape}, expected {(len(t), nbins)}")
            return arr
        res["N_bin"] = per_bin("particle_number_concentration_wet")
        res["N_dry_rebinned"] = per_bin("particle_number_concentration_dry")
        res["rad_um"] = per_bin("size_bin_radius")
        res["mv_cm3_mol"] = np.load(os.path.join(out, "MV.npy"),
                                    allow_pickle=True).astype(float).ravel()
    return res


if __name__ == "__main__":
    import sys

    r = load_pycham(sys.argv[1])
    print("times:", r["t_s"].shape, "components:", len(r["names"]),
          "bins:", r["part_molec_cm3"].shape[1], "walls:", r["wall_molec_cm3"].shape[1])
    for sp in sys.argv[2:]:
        i = r["names"].index(sp)
        print(f"{sp}: gas ppb t0={r['gas_ppb'][0, i]:.4g} tend={r['gas_ppb'][-1, i]:.4g}")
    if r["part_ug_m3"].shape[1]:
        # total particle mass excluding water and seed ('core')
        keep = [k for k, n in enumerate(r["names"]) if n not in ("H2O", "core")]
        soa = r["part_ug_m3"][:, :, keep].sum(axis=(1, 2))
        print("SOA (ug/m3, all bins, excl. H2O/core) at end:", soa[-1])
