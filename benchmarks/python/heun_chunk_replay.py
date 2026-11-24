"""Offline replay of chunk orderings from the saved per-cell sub-step counts.

Cost model: a chunk's lockstep time is proportional to chunk_size x (max over its cells of the
cell's total gas sub-steps over the run, n_sub_sum). That is a lower bound on the true lockstep
cost (the real one takes the max per outer step, then sums), exact when each cell's worst steps
coincide. Seconds are calibrated on the measured sorted-by-T 1e3 ppbv run.
Usage: python -m benchmarks.python.heun_chunk_replay [--bg 0.1] [--hot 1000] [--seeds 20]
"""
import argparse, json, os
import numpy as np

D = 'benchmarks/results/heun_cesm_batch'


def fn(so2):
    s = f'{so2:g}'
    return f'{D}/full_so2_{s}_6h_dt360_sorted_stride1_cells.npz'


def load(so2):
    d = np.load(fn(so2))
    o = np.argsort(d['index'])
    return d['n_sub_sum'][o].astype(float)


def chunk_cost(order, cost, chunk):
    c = cost[order]
    n = len(c)
    pad = (-n) % chunk
    if pad:
        c = np.concatenate([c, np.full(pad, c[-1])])
    return chunk * c.reshape(-1, chunk).max(1).sum()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bg', type=float, default=0.1)
    ap.add_argument('--hot', type=float, default=1000.0)
    ap.add_argument('--seeds', type=int, default=20)
    ap.add_argument('--chunk', type=int, default=4096)
    a = ap.parse_args()
    os.environ.setdefault('JAX_PLATFORMS', 'cpu')
    from tomas_jax.core import config  # noqa: F401 (x64)
    from tomas_jax.physics.so2_chemistry import calc_k1_so2_oh
    t = np.load(f'{D}/full_cells.npz')
    T, P, RH, OH = t['temp'], t['pres'], t['rh'], t['oh_conc']
    n = len(T)
    k1 = np.asarray(calc_k1_so2_oh(T, P, RH))
    cb, ch = load(a.bg), load(a.hot)
    # scale seconds from the measured hot sweep (sorted by T)
    meta = json.load(open(fn(a.hot).replace('_cells.npz', '.json')))
    tot_s = meta['t_run_s']
    ref = chunk_cost(np.argsort(T), ch, a.chunk)
    scale = tot_s / ref if tot_s else None
    print(f'{n} cells, chunk {a.chunk}; bg {a.bg:g} ppbv, hot {a.hot:g} ppbv; measured 1e3 sweep {tot_s} s')
    print(f'proxy check: spearman(cost_hot, k*OH*SO2) and (cost_hot, T):')
    from scipy.stats import spearmanr
    print('  kOH', round(spearmanr(ch, k1 * OH).statistic, 3), ' T', round(spearmanr(ch, -T).statistic, 3),
          ' lowT+kOH rank', round(spearmanr(ch, np.log(k1 * OH + 1e-30) - 0.05 * T).statistic, 3))
    rng = np.random.default_rng(0)
    names = ['lonlat', 'T', 'SO2', 'kOH*SO2', 'kOH*SO2 tier+T', 'oracle']
    for nh in (2, 50, 5000, n):
        res = {k: [] for k in names}
        ideal = []
        for s in range(a.seeds if nh < n else 1):
            hot = np.zeros(n, bool)
            hot[rng.choice(n, nh, replace=False)] = True
            cost = np.where(hot, ch, cb)
            so2 = np.where(hot, a.hot, a.bg)
            kos = k1 * OH * so2
            tier = np.floor(np.log10(kos + 1e-30) * 2)  # half-decade tiers
            orders = {
                'lonlat': np.arange(n),
                'T': np.argsort(T, kind='stable'),
                'SO2': np.argsort(so2, kind='stable'),
                'kOH*SO2': np.argsort(kos, kind='stable'),
                'kOH*SO2 tier+T': np.lexsort((T, tier)),
                'oracle': np.argsort(cost, kind='stable'),
            }
            for k, o in orders.items():
                res[k].append(chunk_cost(o, cost, a.chunk))
            ideal.append(cost.sum())
        idl = np.mean(ideal)
        print(f'\n{nh} hot cells of {n}:  (cost / perfect packing; time for 165,888 cells)')
        for k in names:
            m = np.mean(res[k])
            tt = f' {m * scale:8.1f} s' if scale else ''
            print(f'  {k:16s} {m / idl:7.2f}x{tt}')
    print('\nlower-bound model: real lockstep takes the max per outer step, so absolute times are optimistic;')
    print('prev-step n_sub as key equals the oracle row when cell cost persists between steps.')


if __name__ == '__main__':
    main()
