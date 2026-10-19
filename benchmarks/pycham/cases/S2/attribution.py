"""Attribution runs (numerical experiments, not tuning): TOMAS-JAX with its organic
condensation sink scaled by a constant factor, optionally with the Kelvin term off.

    .venv/bin/python -m benchmarks.pycham.cases.S2.attribution FACTOR COUPLING_DT_S OUT.npz [--kelvin-off]

Wraps ``vbs_jax.partitioning.organic_condensation_sink`` so that the returned sink
``cs`` is multiplied by FACTOR (per-bin shares and the two-film Q unchanged). FACTOR is
the PyCHAM / TOMAS-JAX transfer-rate ratio of the condensing species (transfer.py), so
the run shows how much of the TOMAS-JAX vs PyCHAM SOA difference is mass-transfer rate.
``--kelvin-off`` also sets the vbs-jax surface tension to 0 (Kelvin factor 1), to pair
with the PyCHAM run in the surfT = 0 copy.

Must run in a fresh process: the aerosol step is jitted and would otherwise reuse a trace
made with the unpatched functions. The model code is untouched; run.py calls this module
as a subprocess with a timeout.
"""
from __future__ import annotations

import sys

import vbs_jax.partitioning as vbs_partitioning


def patch_condensation_sink(factor: float) -> None:
    orig = vbs_partitioning.organic_condensation_sink

    def scaled(*args, **kwargs):
        cs, sinkfrac, q = orig(*args, **kwargs)
        return cs * factor, sinkfrac, q

    vbs_partitioning.organic_condensation_sink = scaled


def patch_kelvin_off() -> None:
    from tomas_jax.physics import soa_partitioning
    soa_partitioning.DEFAULT_VBS_CONFIG = soa_partitioning.DEFAULT_VBS_CONFIG._replace(sigma=0.0)


def main(argv) -> int:
    args = [a for a in argv[1:] if not a.startswith("--")]
    kelvin_off = "--kelvin-off" in argv[1:]
    factor, dt, out = float(args[0]), float(args[1]), args[2]
    patch_condensation_sink(factor)
    if kelvin_off:
        patch_kelvin_off()
    from ... import ours                      # imported after the patch (no trace yet)
    from ...config import COMPARE_DT_S
    from . import case
    data = ours.run_ours(case.scenario(coupling_dt_s=dt))
    # keep the gas phase of the condensable species only (all the analysis uses; file size)
    from tomas_jax.box.driver import _class_map
    idx = ours.species_index(data)
    cols = [idx[n] for n in _class_map(None).species]
    data["species"], data["gas_ppb"] = data["species"][cols], data["gas_ppb"][:, cols]
    ours.save_npz(out, data, every_s=COMPARE_DT_S)
    print(f"cs factor {factor:.4f}, coupling {dt:.0f} s, kelvin_off {kelvin_off}, "
          f"runtime {data['runtime_s']:.1f} s")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
