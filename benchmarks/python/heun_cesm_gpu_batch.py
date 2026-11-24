"""Batched GPU timing of coag_solver='heun_adaptive' over every CESM cell near 50 hPa.

Target: a global stratospheric run of ~4M cells (192 x 288 x 70) in parallel on
one GPU. Not that many are run: all cells of the CESM levels around 50 hPa
(default levels 43-45, ~43/52/62 hPa, 3 x 55,296 = 165,888 cells) are run as
a jax.vmap batch, in chunks that fit the memory cap, for each SO2 in a sweep,
and the time is extrapolated to 3.87M cells.

Per cell: T, P (hyam*P0 + hybm*PS), RH (CESM RELHUM) and OH (mol/mol x n_air)
from one snapshot hour of /data/training_sims/run (h1 hourly files). SO2 is the
same in every cell for each sweep value. The rest is the sensitivities/dunne_nuc
study: lognormal sulfate background (10 /cm3, 100 nm), fion = 10 ion pairs cm-3
s-1 (the CESM files carry no ionization rate), boxvol 1e6 cm3, the study's
pure-sulfate clusters, Tabazadeh water, sparse density = grid density, and
make_step(gas_substep='adaptive', coag_solver='heun_adaptive') on the
SO2 chemistry -> nucleation -> coagulation -> condensation split. Default
dt = 360 s, 6 h, 40 bins.

Batching caveats this script handles (docs/gas_substep.md, 2026-11-23 probe):
 * make_step(cap_warnings=False): the always-on cap-warning debug callbacks are
   unrolled once per batch member by JAX (8192 cells: compile 217 s against 9 s,
   step 5.3 s against 0.06 s). Without them the script reads info['hit_cap'] and
   info['coag_dt_left'] per cell and treats any binding cap as a failure.
 * XLA aborts compiling a vmap of batch size exactly 1: chunks are >= 2 cells.
 * Batched while_loops run until the slowest cell finishes: cells are sorted by
   temperature (a cost proxy) so slow cells share chunks (--order lonlat: grid order).

The script refuses to run unless JAX is on an idle GPU with the memory cap free
(XLA_PYTHON_CLIENT_PREALLOCATE=false, XLA_PYTHON_CLIENT_MEM_FRACTION=--mem-fraction):

    source ~/gpu_env.sh && python -m benchmarks.python.heun_cesm_gpu_batch --verify
    source ~/gpu_env.sh && python -m benchmarks.python.heun_cesm_gpu_batch

(source in the same shell command as the run, not through a pipe.)
Output: benchmarks/results/heun_cesm_batch/<tag>so2_<value>.json (one per SO2;
rerun skips existing files unless --force) and calibration.json. Any NaN, any
binding cap or an incomplete sweep (--max-wall-minutes) gives a non-zero exit.
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUTDIR = os.path.join(ROOT, 'benchmarks', 'results', 'heun_cesm_batch')
CESM_FILE = '/data/training_sims/run/CARMA_training_1mon_19910601.cam.h1.1991-06-02-01800.nc'
N_CELLS_TARGET = 192 * 288 * 70      # the 4M-cell production grid


def parse():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--hours', type=float, default=6.0)
    p.add_argument('--dt', type=float, default=360.0, help='outer step [s]')
    p.add_argument('--nbins', type=int, default=40, choices=(40, 80))
    p.add_argument('--so2', type=float, nargs='+',
                   default=[1e-2, 1e-1, 1.0, 1e1, 1e2, 1e3, 1e4, 1e5], help='SO2 [ppbv]')
    p.add_argument('--levels', type=int, nargs='+', default=[43, 44, 45])
    p.add_argument('--time-index', type=int, default=24, help='snapshot hour in the file')
    p.add_argument('--cesm-file', default=CESM_FILE)
    p.add_argument('--max-cells', type=int, default=0, help='first N cells after ordering (0: all)')
    p.add_argument('--order', choices=('sorted', 'lonlat'), default='sorted')
    p.add_argument('--chunk', type=int, default=0,
                   help='cells per chunk (0: min(calibration memory fit, 4096); the time is linear '
                        'in cells above ~1k, so a bigger chunk only adds lockstep waste)')
    p.add_argument('--chunk-stride', type=int, default=1,
                   help='run every k-th chunk of the ordered cells (evenly spaced, so the sample '
                        'keeps the real chunk composition); the rate is then extrapolated')
    p.add_argument('--mnfix-trim', choices=('auto', 'true', 'false'), default='auto')
    p.add_argument('--gas-max-substeps', type=int, default=65536)
    p.add_argument('--mem-fraction', type=float, default=0.25,
                   help='XLA_PYTHON_CLIENT_MEM_FRACTION: cap on GPU memory for this process')
    p.add_argument('--max-gpu-util', type=float, default=5.0)
    p.add_argument('--max-wall-minutes', type=float, default=0.0,
                   help='stop between chunks once one SO2 value has run this long (0: no limit)')
    p.add_argument('--verify', action='store_true',
                   help='only check batched vs single-cell results (1 h, 3 SO2) and exit')
    p.add_argument('--calibrate-only', action='store_true')
    p.add_argument('--save-cells', action='store_true',
                   help='also save per-cell n_sub / N for every SO2 (<name>_cells.npz, indices '
                        'into the cell table saved once as cells.npz) for offline chunking replays')
    p.add_argument('--tag', default='')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


ARGS = parse() if __name__ == '__main__' else None
if ARGS is not None:
    # The memory limiter must be in the environment before JAX is imported.
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f'{ARGS.mem_fraction:g}'

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from benchmarks.python.heun_cesm_gpu import require_idle_gpu, sat_liquid  # noqa: E402

PROCS = ['so2_chemistry', 'nucleation', 'coagulation', 'condensation']


# ---- CESM fields -------------------------------------------------------------

def cesm_cells(path, levels, time_index):
    """Per-cell conditions at one snapshot hour, flattened (level, lat, lon)."""
    import netCDF4 as nc
    d = nc.Dataset(path)
    f = lambda a: np.ma.filled(a, np.nan).astype(np.float64)
    ps = f(d['PS'][time_index])                                   # (lat, lon)
    hyam, hybm, p0 = d['hyam'][:], d['hybm'][:], float(d['P0'][:])
    out = {k: [] for k in ('temp', 'pres', 'rh', 'rh_q', 'oh_conc', 'level', 'lat', 'lon')}
    lat, lon = d['lat'][:], d['lon'][:]
    for k in levels:
        T = f(d['T'][time_index, k])
        q = f(d['Q'][time_index, k])
        oh = f(d['OH'][time_index, k])
        rel = f(d['RELHUM'][time_index, k]) / 100.0
        P = hyam[k] * p0 + hybm[k] * ps
        x = q / (0.622 + (1.0 - 0.622) * q)
        rh_q = x * P / sat_liquid(T)
        nair = P / (1.380649e-23 * T) * 1e-6
        for name, a in (('temp', T), ('pres', P), ('rh', np.clip(rel, 1e-4, 0.99)),
                        ('rh_q', np.clip(rh_q, 1e-4, 0.99)), ('oh_conc', oh * nair)):
            out[name].append(a.ravel())
        out['level'].append(np.full(T.size, k))
        out['lat'].append(np.repeat(lat, lon.size))
        out['lon'].append(np.tile(lon, lat.size))
    out = {k: np.concatenate(v) for k, v in out.items()}
    for k in ('temp', 'pres', 'rh', 'oh_conc'):
        bad = ~np.isfinite(out[k])
        if bad.any():
            raise ValueError(f"{bad.sum()} non-finite {k} values in {path} levels {levels}")
    out['lev_hpa'] = {int(k): float(d['lev'][k]) for k in levels}
    return out


# ---- batched step ------------------------------------------------------------

def make_cell_fn(xk, nsteps, dt, gas_max_substeps, mnfix_trim):
    """cell(so2_kg, temp, pres, rh, oh) -> per-cell scalars after nsteps steps."""
    from tomas_jax.core.config import SRTSO2
    from tomas_jax.solvers.condensation import make_step
    from sensitivities.dunne_nuc import common as c

    step = make_step(PROCS, coag_solver='heun_adaptive', gas_substep='adaptive',
                     gas_max_substeps=gas_max_substeps, mnfix_trim=mnfix_trim,
                     return_substep_info=True, cap_warnings=False,
                     sparse_density=c.SPARSE_DENSITY, water_scheme=c.WATER_SCHEME, jit=False)
    Nk0, Mk0, Gc0 = c.lognormal_state(xk)

    def cell(so2_kg, temp, pres, rh, oh):
        def body(carry, _):
            Nk, Mk, Gc, nmax, nsum, hit, left = carry
            Nk, Mk, Gc, info = step(Nk, Mk, Gc, xk, temp, pres, c.BOXVOL, rh, 1.0, dt,
                                    oh_conc=oh, org_conc=0.0, nh3_conc=0.0, fion=c.FION,
                                    nucl_params=c.NUC_PARAMS)
            return (Nk, Mk, Gc, jnp.maximum(nmax, info['n_sub']), nsum + info['n_sub'],
                    hit | info['hit_cap'], jnp.maximum(left, info['coag_dt_left'])), None
        init = (Nk0, Mk0, Gc0.at[SRTSO2].set(so2_kg), jnp.int32(0), jnp.int32(0),
                jnp.bool_(False), jnp.float64(0.0))
        (Nk, Mk, Gc, nmax, nsum, hit, left), _ = jax.lax.scan(body, init, None, length=nsteps)
        ok = jnp.isfinite(Nk).all() & jnp.isfinite(Mk).all() & jnp.isfinite(Gc).all()
        return dict(N=jnp.sum(Nk) / c.BOXVOL, n_sub_max=nmax, n_sub_sum=nsum,
                    hit_cap=hit, coag_left=left, finite=ok)
    return cell


def compile_batch(cell, chunk):
    f = jax.jit(jax.vmap(cell))
    z = jnp.zeros((chunk,))
    t0 = time.perf_counter()
    exe = f.lower(z, z + 200.0, z + 5000.0, z + 0.02, z + 5e5).compile()
    return exe, time.perf_counter() - t0


def so2_kg_cells(so2_ppbv, temp, pres):
    from sensitivities.dunne_nuc import common as c
    from tomas_jax.core.config import MW_SO2, AVOGADRO
    return c.ppbv_to_molec(so2_ppbv, temp, pres) * c.BOXVOL * (MW_SO2 / 1000.0) / AVOGADRO


def pad_chunk(cells, idx, chunk):
    """Cell arrays for idx, padded to `chunk` with copies of the last cell."""
    sel = np.concatenate([idx, np.full(chunk - idx.size, idx[-1])])
    return {k: v[sel] for k, v in cells.items() if isinstance(v, np.ndarray)}


def run_chunk(exe, so2_ppbv, sub, n_real):
    args = (so2_kg_cells(so2_ppbv, sub['temp'], sub['pres']), sub['temp'], sub['pres'],
            sub['rh'], sub['oh_conc'])
    args = [jnp.asarray(a) for a in args]
    jax.block_until_ready(args)
    t0 = time.perf_counter()
    out = jax.block_until_ready(exe(*args))
    t = time.perf_counter() - t0
    return t, {k: np.asarray(v)[:n_real] for k, v in out.items()}


def peak_bytes(dev):
    return int((dev.memory_stats() or {}).get('peak_bytes_in_use', -1))


# ---- calibration, sweep, verification -----------------------------------------

def calibrate(a, cells, xk, dev):
    """Peak memory per cell at three batch sizes, and the MNFIX trim A/B timing."""
    nsteps = max(1, int(round(3600.0 / a.dt)))
    sizes = [s for s in (1024, 4096, 16384) if s <= cells['temp'].size] or [cells['temp'].size]
    base = peak_bytes(dev)
    res = dict(nsteps=nsteps, dt=a.dt, base_peak_bytes=base, sizes={}, mnfix={})
    cell = make_cell_fn(xk, nsteps, a.dt, a.gas_max_substeps, False)
    for B in sizes:
        exe, tc = compile_batch(cell, B)
        idx = np.linspace(0, cells['temp'].size - 1, B).astype(int)   # spans the cost range
        t, out = run_chunk(exe, 1e2, pad_chunk(cells, idx, B), B)
        pk = peak_bytes(dev)
        res['sizes'][B] = dict(compile_s=tc, run_s=t, peak_bytes=pk,
                               bytes_per_cell=(pk - base) / B)
        print(f"calibration B={B}: compile {tc:.1f} s, 1 h run {t:.2f} s, peak "
              f"{pk / 2**20:.0f} MiB ({(pk - base) / B / 1024:.1f} KiB/cell)", flush=True)
    per_cell = res['sizes'][sizes[-1]]['bytes_per_cell']
    limit = (dev.memory_stats() or {}).get('bytes_limit', 0)
    fit = int(0.7 * limit / max(per_cell, 1.0))
    res['bytes_limit'] = limit
    res['chunk_fit'] = 1 << max(1, fit.bit_length() - 1)       # power of two <= fit
    # MNFIX trim A/B on a 4096-cell chunk, in the order that is used for the sweep
    B = min(4096, cells['temp'].size)
    idx = np.linspace(0, cells['temp'].size - 1, B).astype(int)       # mixed cells: lockstep worst case
    times = {False: 0.0, True: 0.0}
    for trim in (False, True):
        cell = make_cell_fn(xk, nsteps, a.dt, a.gas_max_substeps, trim)
        exe, tc = compile_batch(cell, B)
        for so2 in (1e2, 1e4):
            t, out = run_chunk(exe, so2, pad_chunk(cells, idx, B), B)
            times[trim] += t
            res['mnfix'].setdefault(str(trim), {})[f'so2_{so2:g}'] = dict(run_s=t, compile_s=tc)
            print(f"calibration MNFIX trim={trim} SO2={so2:g}: compile {tc:.1f} s, run {t:.2f} s", flush=True)
    res['mnfix_best_trim'] = bool(times[True] < times[False])
    print(f"calibration: chunk fits {res['chunk_fit']} cells; MNFIX trim "
          f"{'True' if res['mnfix_best_trim'] else 'False'} is faster", flush=True)
    return res


def verify(a, cells, xk):
    """Batched final N against the same cells run alone (1 h, three SO2 values)."""
    nsteps = max(1, int(round(3600.0 / a.dt)))
    cell = make_cell_fn(xk, nsteps, a.dt, a.gas_max_substeps, False)
    rng = np.random.default_rng(0)
    n = cells['temp'].size
    idx = rng.choice(n, size=8, replace=False)
    sub = pad_chunk(cells, idx, 8)
    single = jax.jit(cell)
    exe, _ = compile_batch(cell, 8)
    worst = 0.0
    for so2 in (1e-1, 1e2, 1e4):
        _, bat = run_chunk(exe, so2, sub, 8)
        for j in range(8):
            one = single(so2_kg_cells(so2, sub['temp'][j], sub['pres'][j]), sub['temp'][j],
                         sub['pres'][j], sub['rh'][j], sub['oh_conc'][j])
            rel = abs(float(one['N']) - bat['N'][j]) / max(abs(float(one['N'])), 1e-300)
            worst = max(worst, rel)
            if int(one['n_sub_sum']) != int(bat['n_sub_sum'][j]):
                # the sub-step ladder is discrete: an ulp can move a threshold
                print(f"  note: SO2={so2:g} cell {j}: sub-steps {int(one['n_sub_sum'])} alone, "
                      f"{int(bat['n_sub_sum'][j])} batched", flush=True)
        print(f"verify SO2={so2:g}: max rel diff in N so far {worst:.2e}", flush=True)
    if worst > 1e-8:
        sys.exit(f"VERIFY FAILED: batched vs single-cell N differs by {worst:.2e}")
    print(f"verify OK: worst relative difference {worst:.2e}")


def sweep_so2(a, cells, order, exe, chunk, so2, n_total):
    n = order.size
    chunks = [order[i:i + chunk] for i in range(0, n, chunk)][::a.chunk_stride]
    t_chunks, n_sub_max, n_sub_mean, N_final, fails = [], [], [], [], dict(
        nan=0, hit_cap=0, coag_cap=0)
    idx_done, n_sub_sum_all = [], []
    t_wall0 = time.perf_counter()
    done = 0
    complete = True
    for ch in chunks:
        if a.max_wall_minutes and time.perf_counter() - t_wall0 > 60 * a.max_wall_minutes:
            complete = False
            break
        t, out = run_chunk(exe, so2, pad_chunk(cells, ch, chunk), ch.size)
        t_chunks.append(t)
        done += ch.size
        n_sub_max.append(out['n_sub_max'])
        n_sub_mean.append(out['n_sub_sum'] / (a.hours * 3600.0 / a.dt))
        N_final.append(out['N'])
        idx_done.append(ch)
        n_sub_sum_all.append(out['n_sub_sum'])
        fails['nan'] += int((~out['finite']).sum())
        fails['hit_cap'] += int(out['hit_cap'].sum())
        fails['coag_cap'] += int((out['coag_left'] > 0.0).sum())
    nmax, nmean, Nf = (np.concatenate(x) for x in (n_sub_max, n_sub_mean, N_final))
    t_run = float(np.sum(t_chunks))
    res = dict(so2_ppbv=so2, hours=a.hours, dt=a.dt, nbins=a.nbins, order=a.order,
               levels=a.levels, time_index=a.time_index, chunk=chunk,
               chunk_stride=a.chunk_stride, n_cells=int(done),
               n_cells_total=int(n), complete=complete, n_chunks_run=len(t_chunks),
               t_run_s=t_run, t_chunk_s=[float(x) for x in t_chunks],
               cells_per_s=done / t_run, cell_hours_per_s=done * a.hours / t_run,
               est_s_for_target_cells=t_run * N_CELLS_TARGET / done,
               n_cells_target=N_CELLS_TARGET,
               n_sub_max_over_cells=int(nmax.max()),
               n_sub_per_step_max_p50_p99=[float(np.percentile(nmax, 50)),
                                           float(np.percentile(nmax, 99))],
               n_sub_per_step_mean_p50_p99=[float(np.percentile(nmean, 50)),
                                            float(np.percentile(nmean, 99))],
               N_final_cm3_p01_p50_p99=[float(np.percentile(Nf, p)) for p in (1, 50, 99)],
               failures=fails, gas_max_substeps=a.gas_max_substeps)
    percell = dict(index=np.concatenate(idx_done), n_sub_max=nmax, n_sub_sum=np.concatenate(n_sub_sum_all),
                   N_final=Nf)
    return res, percell


def main():
    a = ARGS
    from tomas_jax.core.config import make_grid, make_grid_80bin
    gpu_idx, handle = require_idle_gpu(a.mem_fraction, a.max_gpu_util)
    dev = jax.devices()[0]
    os.makedirs(OUTDIR, exist_ok=True)
    xk = jnp.asarray(make_grid_80bin() if a.nbins == 80 else make_grid())

    cells = cesm_cells(a.cesm_file, a.levels, a.time_index)
    n_all = cells['temp'].size
    d_rh = np.abs(cells['rh'] - cells['rh_q'])
    print(f"CESM {os.path.basename(a.cesm_file)} hour {a.time_index}: {n_all} cells, levels "
          f"{ {k: round(v, 1) for k, v in cells['lev_hpa'].items()} } hPa; T "
          f"{cells['temp'].min():.1f}-{cells['temp'].max():.1f} K, RH(RELHUM) "
          f"{cells['rh'].min():.3g}-{cells['rh'].max():.3g}, |RH(RELHUM) - RH(Q)| median "
          f"{np.median(d_rh):.3g} max {d_rh.max():.3g}, OH {cells['oh_conc'].min():.3g}-"
          f"{cells['oh_conc'].max():.3g} /cm3", flush=True)
    order = np.argsort(cells['temp'], kind='stable') if a.order == 'sorted' else np.arange(n_all)
    if a.max_cells:
        order = order[:a.max_cells]
    # A subset for calibration / verification that spans the whole cost range
    spread = np.linspace(0, n_all - 1, min(n_all, 16384)).astype(int)
    cal_cells = {k: v[np.argsort(cells['temp'], kind='stable')[spread]]
                 for k, v in cells.items() if isinstance(v, np.ndarray)}

    if a.verify:
        return verify(a, cal_cells, xk)

    cal_fn = os.path.join(OUTDIR, f"{a.tag}calibration.json")
    if os.path.exists(cal_fn) and not a.force and not a.calibrate_only:
        cal = json.load(open(cal_fn))
        print(f"calibration from {cal_fn}", flush=True)
    else:
        cal = calibrate(a, cal_cells, xk, dev)
        json.dump(cal, open(cal_fn, 'w'), indent=1)
        if a.calibrate_only:
            return
    chunk = a.chunk or min(cal['chunk_fit'], 4096, 1 << int(np.floor(np.log2(max(2, order.size)))))
    chunk = max(2, chunk)
    trim = {'auto': cal['mnfix_best_trim'], 'true': True, 'false': False}[a.mnfix_trim]
    nsteps = int(round(a.hours * 3600.0 / a.dt))
    print(f"sweep: {order.size} cells in chunks of {chunk}, MNFIX trim={trim}, {nsteps} steps of "
          f"{a.dt:g} s, gas_max_substeps={a.gas_max_substeps}", flush=True)
    cell = make_cell_fn(xk, nsteps, a.dt, a.gas_max_substeps, trim)
    exe, t_compile = compile_batch(cell, chunk)
    print(f"compiled in {t_compile:.1f} s", flush=True)
    run_chunk(exe, a.so2[0], pad_chunk(cells, order[:chunk], chunk), min(chunk, order.size))  # warm-up

    if a.save_cells:
        np.savez(os.path.join(OUTDIR, f"{a.tag}cells.npz"),
                 **{k: v for k, v in cells.items() if isinstance(v, np.ndarray)})
    bad, incomplete = [], []
    for so2 in a.so2:
        fn = os.path.join(OUTDIR, f"{a.tag}so2_{so2:g}_{a.hours:g}h_dt{a.dt:g}_{a.order}"
                          f"_stride{a.chunk_stride}.json")
        if os.path.exists(fn) and not a.force:
            print(f"SO2={so2:g}: exists, skipped (--force to rerun)", flush=True)
            continue
        res, percell = sweep_so2(a, cells, order, exe, chunk, so2, order.size)
        if a.save_cells:
            np.savez(fn.replace('.json', '_cells.npz'), **percell)
        res.update(t_compile_s=t_compile, mnfix_trim=trim, gpu=str(dev), gpu_index=gpu_idx,
                   gpu_peak_bytes=peak_bytes(dev), jax=jax.__version__)
        json.dump(res, open(fn, 'w'), indent=1)
        f = res['failures']
        print(f"SO2={so2:g}: {res['n_cells']} cells in {res['t_run_s']:.1f} s "
              f"({res['cells_per_s']:.0f} cells/s; est {res['est_s_for_target_cells'] / 60:.1f} min "
              f"for {N_CELLS_TARGET} cells), n_sub/step max over cells {res['n_sub_max_over_cells']}, "
              f"median-cell mean {res['n_sub_per_step_mean_p50_p99'][0]:.1f}, peak "
              f"{res['gpu_peak_bytes'] / 2**30:.2f} GiB"
              f"{'' if res['complete'] else '  ** INCOMPLETE (wall limit)'}"
              f"{'  ** FAILURES ' + str(f) if any(f.values()) else ''}", flush=True)
        if any(f.values()):
            bad.append(so2)
        if not res['complete']:
            incomplete.append(so2)
    if bad:
        sys.exit(f"FAILED (NaN or binding caps) at SO2 = {bad}")
    if incomplete:
        sys.exit(f"INCOMPLETE at SO2 = {incomplete}: stopped by --max-wall-minutes")


if __name__ == '__main__':
    main()
