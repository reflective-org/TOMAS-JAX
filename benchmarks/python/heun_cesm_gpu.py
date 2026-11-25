"""Wall-time benchmark of coag_solver='heun_adaptive' on a GPU, at CESM
stratospheric conditions over a range of SO2.

Conditions (T, P, RH, OH) come from one column of a CESM/CARMA hourly history
file (/data/training_sims/run, `h1` files, 192 x 288 x 70 levels, 48 hourly
steps per file), averaged over the file's time steps. RH is from Q with the
Murphy and Koop (2005) liquid saturation pressure. OH is held constant at that
mean. Everything else is the sensitivities/dunne_nuc study setting
(common.py): lognormal sulfate background, fion = 10 ion pairs cm-3 s-1,
boxvol 1e6 cm3, dt = 60 s, the library's Dunne nucleation with the study's
pure-sulfate clusters, Tabazadeh water, sparse density = grid density, and
make_step(gas_substep='adaptive') (the validated treatment at high SO2).

The script refuses to run unless JAX is on a GPU that is idle and has the
requested memory free, and it caps XLA's allocation (--mem-fraction, default
0.10 of the card, preallocation off). Run it through the GPU helper:

    source ~/gpu_env.sh && python -m benchmarks.python.heun_cesm_gpu --hours 1

Per (level, SO2) cell it compiles ahead of time (compile time reported
separately), runs once for the wall time, and records final number, sub-step
counts, cap warnings and peak GPU memory. A NaN or a cap warning is printed and
makes the exit code non-zero; nothing is skipped silently.
Output: benchmarks/results/heun_cesm/<tag>.json (one file per cell, rerun skips
cells that exist unless --force).
"""
import os
import sys
import json
import time
import argparse
import warnings

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUTDIR = os.path.join(ROOT, 'benchmarks', 'results', 'heun_cesm')
CESM_DIR = '/data/training_sims/run'
CESM_FILE = 'CARMA_training_1mon_19910601.cam.h1.1991-06-02-01800.nc'


def parse():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--hours', type=float, default=24.0)
    p.add_argument('--nbins', type=int, default=40, choices=(40, 80))
    p.add_argument('--so2', type=float, nargs='+',
                   default=[1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5],
                   help='SO2 [ppbv]')
    p.add_argument('--levels', type=int, nargs='+', default=[40, 44, 48],
                   help='CESM level indices (default ~25, 20, 16 km)')
    p.add_argument('--lat-index', type=int, default=95, help='95 = equator')
    p.add_argument('--lon-index', type=int, default=0)
    p.add_argument('--cesm-file', default=os.path.join(CESM_DIR, CESM_FILE))
    p.add_argument('--mem-fraction', type=float, default=0.10,
                   help='XLA_PYTHON_CLIENT_MEM_FRACTION: cap on the GPU memory this process may take')
    p.add_argument('--max-gpu-util', type=float, default=5.0,
                   help='refuse to run if the GPU is busier than this [%%]')
    p.add_argument('--tag', default='')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


ARGS = parse() if __name__ == '__main__' else None

# The memory limiter must be in the environment before JAX is imported.
if ARGS is not None:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f'{ARGS.mem_fraction:g}'

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402


def require_idle_gpu(mem_fraction, max_util):
    """Exit unless JAX runs on one idle GPU with enough free memory."""
    plat = jax.devices()[0].platform
    if plat not in ('gpu', 'cuda'):
        sys.exit(f"no GPU: JAX platform is '{plat}'. Run `source ~/gpu_env.sh` first "
                 f"(it sets LD_LIBRARY_PATH and picks an idle GPU).")
    cvd = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if not cvd.isdigit():
        sys.exit("CUDA_VISIBLE_DEVICES must name exactly one GPU; use `source ~/gpu_env.sh`.")
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(int(cvd))
    util = pynvml.nvmlDeviceGetUtilizationRates(h).gpu
    mem = pynvml.nvmlDeviceGetMemoryInfo(h)
    # JAX has initialised the context: its own few hundred MiB are in `used`
    need = mem_fraction * mem.total
    print(f"GPU {cvd}: util {util}%, {mem.used / 2**30:.1f}/{mem.total / 2**30:.0f} GiB used; "
          f"cap for this run {need / 2**30:.1f} GiB", flush=True)
    if util > max_util:
        sys.exit(f"GPU {cvd} is busy ({util}% > {max_util}%): not benchmarking on a shared GPU.")
    if mem.total - mem.used < need:
        sys.exit(f"GPU {cvd} has {(mem.total - mem.used) / 2**30:.1f} GiB free, "
                 f"the cap is {need / 2**30:.1f} GiB.")
    return int(cvd), h


# ---- CESM conditions ---------------------------------------------------------

def sat_liquid(T):
    """Saturation vapour pressure over supercooled liquid water [Pa] (Murphy and Koop 2005)."""
    return np.exp(54.842763 - 6763.22 / T - 4.210 * np.log(T) + 0.000367 * T
                  + np.tanh(0.0415 * (T - 218.8))
                  * (53.878 - 1331.22 / T - 9.44523 * np.log(T) + 0.014025 * T))


def cesm_conditions(path, lat_i, lon_i, levels):
    """Time-mean conditions of one CESM column at the given level indices."""
    import netCDF4 as nc
    d = nc.Dataset(path)
    get = lambda v, *ix: np.ma.filled(d[v][(slice(None), *ix)], np.nan).astype(float)
    T, Q = get('T', slice(None), lat_i, lon_i), get('Q', slice(None), lat_i, lon_i)
    OH, SO2 = get('OH', slice(None), lat_i, lon_i), get('SO2', slice(None), lat_i, lon_i)
    PS = get('PS', lat_i, lon_i)
    P = d['hyam'][:][None, :] * d['P0'][:] + d['hybm'][:][None, :] * PS[:, None]
    eps = 0.622
    out = []
    for k in levels:
        x = Q[:, k] / (eps + (1.0 - eps) * Q[:, k])
        rh = np.clip(x * P[:, k] / sat_liquid(T[:, k]), 1e-4, 0.99)
        nair = P[:, k] / (1.380649e-23 * T[:, k]) * 1e-6          # molec/cm3
        out.append(dict(level=int(k), lev_hpa=float(d['lev'][k]),
                        temp=float(T[:, k].mean()), pres=float(P[:, k].mean()),
                        rh=float(rh.mean()), alpha=1.0,
                        oh_conc=float((OH[:, k] * nair).mean()),
                        so2_cesm_ppbv=float(SO2[:, k].mean() * 1e9),
                        lat=float(d['lat'][lat_i]), lon=float(d['lon'][lon_i]),
                        n_time=int(T.shape[0]), file=os.path.basename(path)))
    if not all(np.isfinite(v) for c in out for v in
               (c['temp'], c['pres'], c['rh'], c['oh_conc'])):
        raise ValueError(f"non-finite CESM conditions in {path}: {out}")
    return out


# ---- benchmark --------------------------------------------------------------

def run_cell(cond, so2_ppbv, hours, nbins, dev, nvml_handle):
    from tomas_jax.core.config import make_grid, make_grid_80bin
    from sensitivities.dunne_nuc import common as c
    import pynvml

    xk = jnp.asarray(make_grid_80bin() if nbins == 80 else make_grid())
    nsteps = int(round(hours * 3600.0 / c.DT))
    state = c.lognormal_state(xk, so2=c.ppbv_to_molec(so2_ppbv, cond['temp'], cond['pres']))
    runner = c.make_runner(
        xk, nsteps,
        cond=dict(temp=cond['temp'], pres=cond['pres'], rh=cond['rh'], alpha=cond['alpha']),
        oh_conc=cond['oh_conc'], coag_solver='heun_adaptive', gas_substep='adaptive')
    f = jax.jit(runner)
    args = (c.NUC_PARAMS, *state)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        t0 = time.perf_counter()
        compiled = f.lower(*args).compile()
        t_compile = time.perf_counter() - t0
        t0 = time.perf_counter()
        Nk, Mk, Gc, diag = jax.block_until_ready(compiled(*args))
        t_run = time.perf_counter() - t0
        jax.effects_barrier()          # flush the cap-warning debug callbacks
    stats = dev.memory_stats() or {}
    n_total = np.asarray(diag['N_total'])
    n_sub = np.asarray(diag['n_sub'])
    res = dict(cond, so2_ppbv=so2_ppbv, hours=hours, nbins=nbins, nsteps=nsteps,
               dt=c.DT, coag_solver='heun_adaptive', gas_substep='adaptive',
               t_compile_s=t_compile, t_run_s=t_run,
               t_per_outer_step_ms=1e3 * t_run / nsteps,
               N_final_cm3=float(n_total[-1]), N_max_cm3=float(n_total.max()),
               nan=bool(not np.isfinite(np.asarray(Nk)).all()
                        or not np.isfinite(n_total).all()),
               gas_substeps_mean=float(n_sub.mean()), gas_substeps_max=int(n_sub.max()),
               gas_clamp_steps=int(np.asarray(diag['clamp']).sum()),
               warnings=sorted({f"{w.category.__name__}: {str(w.message)[:160]}" for w in caught}),
               gpu_peak_bytes_in_use=int(stats.get('peak_bytes_in_use', -1)),
               gpu_bytes_limit=int(stats.get('bytes_limit', -1)),
               gpu=str(dev), jax=jax.__version__)
    res['gpu_name'] = pynvml.nvmlDeviceGetName(nvml_handle)
    return res


def main():
    a = ARGS
    gpu_idx, h = require_idle_gpu(a.mem_fraction, a.max_gpu_util)
    dev = jax.devices()[0]
    os.makedirs(OUTDIR, exist_ok=True)
    conds = cesm_conditions(a.cesm_file, a.lat_index, a.lon_index, a.levels)
    for cnd in conds:
        print(f"CESM level {cnd['level']}: {cnd['lev_hpa']:.1f} hPa, T={cnd['temp']:.1f} K, "
              f"P={cnd['pres']:.0f} Pa, RH={cnd['rh']:.3g}, OH={cnd['oh_conc']:.3g} /cm3, "
              f"CESM SO2={cnd['so2_cesm_ppbv']:.3g} ppbv", flush=True)
    failed = []
    for cnd in conds:
        for so2 in a.so2:
            tag = f"{a.tag + '_' if a.tag else ''}L{cnd['level']}_so2_{so2:g}_{a.hours:g}h_{a.nbins}b"
            fn = os.path.join(OUTDIR, tag + '.json')
            if os.path.exists(fn) and not a.force:
                print(f"{tag}: exists, skipped (--force to rerun)", flush=True)
                continue
            res = run_cell(cnd, so2, a.hours, a.nbins, dev, h)
            with open(fn, 'w') as fh:
                json.dump(res, fh, indent=1)
            bad = res['nan'] or res['warnings']
            print(f"{tag}: run {res['t_run_s']:.1f} s (compile {res['t_compile_s']:.1f} s), "
                  f"N_final={res['N_final_cm3']:.4g} /cm3, gas sub-steps mean "
                  f"{res['gas_substeps_mean']:.1f} max {res['gas_substeps_max']}, "
                  f"peak {res['gpu_peak_bytes_in_use'] / 2**30:.2f} GiB"
                  f"{'  ** NaN' if res['nan'] else ''}"
                  f"{'  ** ' + '; '.join(res['warnings']) if res['warnings'] else ''}", flush=True)
            if bad:
                failed.append(tag)
    if failed:
        sys.exit(f"FAILED (NaN or cap warnings): {failed}")


if __name__ == '__main__':
    main()
