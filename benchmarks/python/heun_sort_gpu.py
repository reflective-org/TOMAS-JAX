"""Chunk ordering for a vmapped heun_adaptive step: sort by H2SO4 production? (sulfate layout)

A vmap chunk runs as long as its slowest cell (lockstep while_loops). This times several
orderings of the same cells into chunks, for a population where SO2 is elevated only in a
band of levels:

 * cells: all levels PR 20 simulates (CESM levels 27-50, 1.24-143 hPa, 24 x 192 x 288 cells),
   per-cell T, P, RH, OH from the CESM file (as heun_cesm_gpu_batch.py);
 * SO2: the file's own SO2 (background, ppbv) everywhere, replaced by one uniform value
   (--so2) in the elevated levels (--elevated-levels, default 43 44 45, ~43/52/62 hPa);
 * step: make_step(coag_solver='heun_adaptive', gas_substep='adaptive'), dt = 360 s, 6 h,
   40 bins, TOMAS_SPECIES=sulfate and mnfix_trim=True (both asserted).

Orderings (cost key: H2SO4 production k1(T,P,RH) * OH * SO2, known before the step):
 lonlat    cells in level-major file order (levels are contiguous, so this already groups SO2)
 columns   (lat, lon, level) order: a column-wise memory layout, levels mixed in every chunk
 T         sorted by temperature (heun_cesm_gpu_batch.py default)
 SO2       sorted by SO2, descending
 prod      sorted by k1*OH*SO2, descending
 tier+T    half-decade tiers of k1*OH*SO2 (descending), temperature within a tier
 bucketed  prod order, --hot-chunk cells per chunk for cells with prod >= hot_factor x the
           median prod, --chunk for the rest
 oracle    sorted by the measured per-cell sub-step total (lower bound; needs a first pass)
 T_again   T repeated at the end, to show timing drift on the shared GPU

    source ~/gpu_env.sh && TOMAS_SPECIES=sulfate python -m benchmarks.python.heun_sort_gpu --so2 100

(source in the same shell command as the run.) --cpu --max-cells N is a small dry run.
Output: benchmarks/results/heun_sort/<tag>so2_<value>.json. Non-zero exit on NaN, a binding
cap, a batched/single-cell mismatch, or orderings that disagree per cell.
"""
import os
import sys
import json
import time
import argparse

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
OUTDIR = os.path.join(ROOT, 'benchmarks', 'results', 'heun_sort')
CESM_FILE = '/data/training_sims/run/CARMA_training_1mon_19910601.cam.h1.1991-06-02-01800.nc'
ALL_ORDERINGS = ['lonlat', 'columns', 'T', 'SO2', 'prod', 'tier+T', 'bucketed', 'oracle', 'T_again']


def parse():
    p = argparse.ArgumentParser(description=__doc__.split('\n\n')[0])
    p.add_argument('--so2', type=float, nargs='+', default=[1.0, 1e2, 1e3],
                   help='SO2 [ppbv] in the elevated levels (one pass set per value)')
    p.add_argument('--levels', type=int, nargs='+', default=list(range(27, 51)))
    p.add_argument('--elevated-levels', type=int, nargs='+', default=[43, 44, 45])
    p.add_argument('--orderings', nargs='+', default=ALL_ORDERINGS, choices=ALL_ORDERINGS)
    p.add_argument('--hours', type=float, default=6.0)
    p.add_argument('--dt', type=float, default=360.0, help='outer step [s]')
    p.add_argument('--chunk', type=int, default=4096)
    p.add_argument('--hot-chunk', type=int, default=512)
    p.add_argument('--hot-factor', type=float, default=10.0)
    p.add_argument('--gas-max-substeps', type=int, default=65536)
    p.add_argument('--time-index', type=int, default=24)
    p.add_argument('--cesm-file', default=CESM_FILE)
    p.add_argument('--max-cells', type=int, default=0,
                   help='random subset of N cells (dry runs; 0: all)')
    p.add_argument('--mem-fraction', type=float, default=0.25)
    p.add_argument('--max-gpu-util', type=float, default=5.0)
    p.add_argument('--max-wall-minutes', type=float, default=0.0,
                   help='per ordering pass; 0: no limit (an incomplete pass exits non-zero)')
    p.add_argument('--cpu', action='store_true', help='dry run on CPU, no GPU guard')
    p.add_argument('--tag', default='')
    p.add_argument('--force', action='store_true')
    return p.parse_args()


ARGS = parse() if __name__ == '__main__' else None
if ARGS is not None:
    os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'
    os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = f'{ARGS.mem_fraction:g}'
    if ARGS.cpu:
        os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np  # noqa: E402
import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

from benchmarks.python import heun_cesm_gpu_batch as B  # noqa: E402
from benchmarks.python.heun_cesm_gpu import require_idle_gpu  # noqa: E402

NLAT, NLON = 192, 288


def so2_background_ppbv(path, levels, time_index):
    """The file's SO2 [mol/mol -> ppbv], flattened (level, lat, lon) like cesm_cells."""
    import netCDF4 as nc
    d = nc.Dataset(path)
    out = [np.ma.filled(d['SO2'][time_index, k], np.nan).astype(np.float64).ravel() * 1e9
           for k in levels]
    so2 = np.concatenate(out)
    if not np.isfinite(so2).all() or (so2 < 0).any():
        raise ValueError("SO2 background has non-finite or negative values")
    return so2


def make_orderings(cells, so2, prod, a, n_sub_sum=None):
    """name -> list of index arrays (chunks), the cost key being prod."""
    n = so2.size
    T = cells['temp']
    nl = n // (NLAT * NLON)
    chunk, hot_chunk = a.chunk, a.hot_chunk

    def split(order, size):
        return [order[i:i + size] for i in range(0, order.size, size)]

    out = {}
    out['lonlat'] = split(np.arange(n), chunk)
    if n == nl * NLAT * NLON:
        out['columns'] = split(np.arange(n).reshape(nl, NLAT, NLON).transpose(1, 2, 0).ravel(), chunk)
    out['T'] = split(np.argsort(T, kind='stable'), chunk)
    out['SO2'] = split(np.argsort(-so2, kind='stable'), chunk)
    o_prod = np.argsort(-prod, kind='stable')
    out['prod'] = split(o_prod, chunk)
    tier = np.floor(np.log10(prod + 1e-30) * 2)
    out['tier+T'] = split(np.lexsort((T, -tier)), chunk)
    hot = prod[o_prod] >= a.hot_factor * np.median(prod)
    n_hot = int(hot.sum())
    out['bucketed'] = split(o_prod[:n_hot], hot_chunk) + split(o_prod[n_hot:], chunk)
    out['T_again'] = out['T']
    if n_sub_sum is not None:
        out['oracle'] = split(np.argsort(-n_sub_sum, kind='stable'), chunk)
    return out, n_hot


class Runner:
    """Compiled batch executables by chunk size, and one timed pass over chunks."""

    def __init__(self, a, xk, trim):
        self.a = a
        nsteps = int(round(a.hours * 3600.0 / a.dt))
        self.nsteps = nsteps
        self.cell = B.make_cell_fn(xk, nsteps, a.dt, a.gas_max_substeps, trim)
        self.exe = {}
        self.t_compile = {}

    def get(self, size, cells):
        if size not in self.exe:
            self.exe[size], self.t_compile[size] = B.compile_batch(self.cell, size)
            # warm-up on real cells: first call carries one-off costs
            idx = np.arange(size) % cells['temp'].size
            self.run(size, cells, idx, size)
        return self.exe[size]

    def run(self, size, cells, idx, n_real):
        sub = B.pad_chunk(cells, idx, size)
        return B.run_chunk(self.exe[size], sub['so2'], sub, n_real)

    def pass_(self, name, chunks, cells):
        a = self.a
        n = sum(c.size for c in chunks)
        sizes = sorted({a.hot_chunk if (name == 'bucketed' and c.size <= a.hot_chunk) else a.chunk
                        for c in chunks})
        for s in sizes:
            self.get(s, cells)
        nsub = np.zeros(cells['temp'].size)
        Nf = np.zeros(cells['temp'].size)
        t_chunks, waste_num, fails, done = [], 0.0, dict(nan=0, hit_cap=0, coag_cap=0), 0
        mx, mean_ratio = [], []
        w0 = time.perf_counter()
        complete = True
        for c in chunks:
            if a.max_wall_minutes and time.perf_counter() - w0 > 60 * a.max_wall_minutes:
                complete = False
                break
            size = a.hot_chunk if (name == 'bucketed' and c.size <= a.hot_chunk) else a.chunk
            t, out = self.run(size, cells, c, c.size)
            t_chunks.append(t)
            nsub[c] = out['n_sub_sum']
            Nf[c] = out['N']
            fails['nan'] += int((~out['finite']).sum())
            fails['hit_cap'] += int(out['hit_cap'].sum())
            fails['coag_cap'] += int((out['coag_left'] > 0.0).sum())
            mx.append(float(out['n_sub_sum'].max()))
            mean_ratio.append(float(out['n_sub_sum'].max() / max(out['n_sub_sum'].mean(), 1e-30)))
            waste_num += size * out['n_sub_sum'].max()
            done += c.size
        t_run = float(np.sum(t_chunks))
        res = dict(ordering=name, n_cells=int(done), n_cells_total=int(n), complete=complete,
                   n_chunks=len(t_chunks), t_run_s=t_run, cells_per_s=done / max(t_run, 1e-30),
                   lockstep_waste=float(waste_num / max(nsub.sum(), 1e-30)),
                   chunk_max_nsub_p50_p99=[float(np.percentile(mx, 50)), float(np.percentile(mx, 99))],
                   chunk_max_over_mean_mean=float(np.mean(mean_ratio)),
                   t_chunk_max_s=float(np.max(t_chunks)), failures=fails,
                   t_compile_s={str(k): v for k, v in self.t_compile.items()})
        return res, nsub, Nf


def check_batched_vs_single(runner, cells, so2_val, a):
    """A few hot and cold cells: batched final N against the same cell run alone."""
    hot = np.where(cells['so2'] >= so2_val)[0]
    cold = np.where(cells['so2'] < so2_val)[0]
    rng = np.random.default_rng(0)
    idx = np.concatenate([rng.choice(hot, 4, replace=False) if hot.size >= 4 else hot,
                          rng.choice(cold, 4, replace=False)])
    size = a.chunk
    runner.get(size, cells)
    _, bat = runner.run(size, cells, idx, idx.size)
    single = jax.jit(runner.cell)
    worst = 0.0
    for j, i in enumerate(idx):
        sk = B.so2_kg_cells(cells['so2'][i], cells['temp'][i], cells['pres'][i])
        one = single(sk, cells['temp'][i], cells['pres'][i], cells['rh'][i], cells['oh_conc'][i])
        worst = max(worst, abs(float(one['N']) - bat['N'][j]) / max(abs(float(one['N'])), 1e-300))
    return worst


def main():
    a = ARGS
    from tomas_jax.core.config import make_grid, SPECIES_LAYOUT
    from tomas_jax.physics.so2_chemistry import calc_k1_so2_oh
    if SPECIES_LAYOUT != 'sulfate':
        sys.exit("heun_sort_gpu.py runs the sulfate layout: set TOMAS_SPECIES=sulfate")
    if a.cpu:
        gpu_idx = None
    else:
        gpu_idx, _ = require_idle_gpu(a.mem_fraction, a.max_gpu_util)
    dev = jax.devices()[0]
    os.makedirs(OUTDIR, exist_ok=True)
    xk = jnp.asarray(make_grid())

    cells = B.cesm_cells(a.cesm_file, a.levels, a.time_index)
    bg = so2_background_ppbv(a.cesm_file, a.levels, a.time_index)
    n_all = cells['temp'].size
    elev = np.isin(cells['level'], a.elevated_levels)
    print(f"{n_all} cells, levels {a.levels[0]}-{a.levels[-1]} "
          f"({min(cells['lev_hpa'].values()):.2f}-{max(cells['lev_hpa'].values()):.1f} hPa), "
          f"elevated levels {a.elevated_levels}: {int(elev.sum())} cells", flush=True)
    if not set(a.elevated_levels) <= set(a.levels):
        sys.exit("--elevated-levels must be inside --levels")
    for k in a.levels:
        m = cells['level'] == k
        print(f"  level {k} ({cells['lev_hpa'][k]:.1f} hPa): background SO2 p50/p99/max ppbv "
              f"{np.percentile(bg[m], 50):.3g}/{np.percentile(bg[m], 99):.3g}/{bg[m].max():.3g}"
              f"{'  [elevated]' if k in a.elevated_levels else ''}", flush=True)

    sel = None
    if a.max_cells and a.max_cells < n_all:
        sel = np.sort(np.random.default_rng(1).choice(n_all, a.max_cells, replace=False))
        cells = {k: (v[sel] if isinstance(v, np.ndarray) else v) for k, v in cells.items()}
        bg, elev = bg[sel], elev[sel]
        print(f"dry run: random subset of {a.max_cells} cells (columns ordering skipped)", flush=True)

    trim = True
    failed = []
    for so2_val in a.so2:
        fn = os.path.join(OUTDIR, f"{a.tag}so2_{so2_val:g}.json")
        if os.path.exists(fn) and not a.force:
            print(f"SO2={so2_val:g}: {fn} exists, skipped (--force)", flush=True)
            continue
        so2 = np.where(elev, so2_val, bg)
        cells['so2'] = so2
        k1 = np.asarray(calc_k1_so2_oh(jnp.asarray(cells['temp']), jnp.asarray(cells['pres']),
                                       jnp.asarray(cells['rh'])))
        prod = k1 * cells['oh_conc'] * so2
        print(f"\nSO2 = {so2_val:g} ppbv in levels {a.elevated_levels}; production k1*OH*SO2 "
              f"p50/p99/max {np.percentile(prod, 50):.3g}/{np.percentile(prod, 99):.3g}/{prod.max():.3g}; "
              f"cells above 1/10/100 ppbv: " + '/'.join(str(int((so2 > x).sum())) for x in (1, 10, 100)),
              flush=True)

        runner = Runner(a, xk, trim)
        worst_bs = check_batched_vs_single(runner, cells, so2_val, a)
        print(f"batched vs single-cell: worst relative difference in N {worst_bs:.2e}", flush=True)
        if worst_bs > 1e-10:
            failed.append((so2_val, f'batched/single {worst_bs:.2e}'))
            continue

        results, ref_N, nsub_ref = {}, None, None
        pending = [o for o in a.orderings if o != 'oracle'] + (['oracle'] if 'oracle' in a.orderings else [])
        pending = [o for o in pending if o != 'T_again'] + (['T_again'] if 'T_again' in a.orderings else [])
        for name in pending:
            ords, n_hot = make_orderings(cells, so2, prod, a, nsub_ref)
            if name not in ords:
                print(f"  {name}: skipped (unavailable here)", flush=True)
                continue
            res, nsub, Nf = runner.pass_(name, ords[name], cells)
            if nsub_ref is None and res['complete']:
                nsub_ref, ref_N = nsub.copy(), Nf.copy()
            if ref_N is not None and res['complete']:
                res['max_rel_N_vs_first'] = float(np.max(np.abs(Nf - ref_N) / np.maximum(np.abs(ref_N), 1e-300)))
            if name == 'bucketed':
                res['n_hot_cells'] = n_hot
            results[name] = res
            f = res['failures']
            print(f"  {name:9s} {res['t_run_s']:8.1f} s  {res['cells_per_s']:9.0f} cells/s  "
                  f"lockstep waste {res['lockstep_waste']:.2f}x  chunk max-nsub p50/p99 "
                  f"{res['chunk_max_nsub_p50_p99'][0]:.0f}/{res['chunk_max_nsub_p50_p99'][1]:.0f}  "
                  f"max rel dN vs first {res.get('max_rel_N_vs_first', float('nan')):.1e}"
                  f"{'' if res['complete'] else '  ** INCOMPLETE'}"
                  f"{'  ** FAILURES ' + str(f) if any(f.values()) else ''}", flush=True)
            if any(f.values()) or not res['complete']:
                failed.append((so2_val, name))
            elif res.get('max_rel_N_vs_first', 0.0) > 1e-8:
                failed.append((so2_val, f"{name} disagrees per cell"))
        base = results.get('T')
        if base:
            for r in results.values():
                r['speedup_vs_T'] = base['t_run_s'] / r['t_run_s']
        json.dump(dict(so2_ppbv=so2_val, elevated_levels=a.elevated_levels, levels=a.levels,
                       n_cells=int(so2.size), hours=a.hours, dt=a.dt, chunk=a.chunk,
                       hot_chunk=a.hot_chunk, hot_factor=a.hot_factor, mnfix_trim=trim,
                       species_layout=SPECIES_LAYOUT, gpu=str(dev), gpu_index=gpu_idx,
                       jax=jax.__version__, batched_vs_single_worst=worst_bs, results=results),
                  open(fn, 'w'), indent=1)
        print(f"wrote {fn}", flush=True)
        if base:
            print("  speed-up vs T: " + ', '.join(f"{k} {v['speedup_vs_T']:.2f}x" for k, v in results.items()),
                  flush=True)
    if failed:
        sys.exit(f"FAILED: {failed}")


if __name__ == '__main__':
    main()
