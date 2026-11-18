"""40-bin vs 80-bin difference from the stored gas_substep results.

Reads results/gas_substep/grid.npz (40 bins, snapshots at 6 and 24 h) and
results/gas_substep/psd80/T*_*_<run>.npz (80 bins), both on the T x SO2 grid
at 5500 Pa. 80-bin pairs are merged onto the 40-bin grid. Compares the same
kind of run (ref vs ref, f=0.1 vs f=0.1, ...) at 6 and 24 h snapshots.
Reference dt differs between the grids (40 bins coarser), so adaptive runs
are the cleaner like-for-like comparison.

    python -m benchmarks.python.compare_40_80_bins
"""
from pathlib import Path
import numpy as np

R = Path(__file__).resolve().parents[1] / 'results' / 'gas_substep'
RUNS = {'ref': 'ref', 'f=0.3': 'f0.3', 'f=0.1': 'f0.1', 'f=0.03': 'f0.03', 'dt_for': 'dt_for'}


def l1(a, b, norm):
    return np.abs(a - b).sum() / norm.sum()


def main():
    g = np.load(R / 'grid.npz', allow_pickle=True)
    if not (R / 'psd80').is_dir():
        raise FileNotFoundError(f'Missing {R / "psd80"}')
    rows = []
    for T in sorted(set(g['T'])):
        for so2 in sorted(set(g['so2'])):
            for run, tag in RUNS.items():
                f = R / 'psd80' / f'T{T:g}_{so2:g}_{tag}.npz'
                if not f.exists():
                    raise FileNotFoundError(f)
                p = np.load(f, allow_pickle=True)
                for i, h in enumerate(p['hours']):
                    m = (g['T'] == T) & (g['so2'] == so2) & (g['run'] == run) & (g['hour'] == h)
                    if m.sum() != 1:
                        raise ValueError(f'{T} {so2} {run} {h}: {m.sum()} 40-bin rows')
                    N40, M40 = g['Nk'][m][0], g['Mk'][m][0]
                    N80 = p['Nk'][i].reshape(40, 2).sum(1)
                    M80 = p['Mk'][i].reshape(40, 2).sum(1)
                    rows.append(dict(T=T, so2=so2, run=run, hour=int(h),
                                     N40=N40.sum(), N80=N80.sum(),
                                     N3_40=g['N3'][m][0], N3_80=p['nmass'][i][0],
                                     M40=M40.sum(), M80=M80.sum(),
                                     eN=l1(N40, N80, N80), eM=l1(M40, M80, M80)))
    np.save(R / 'compare_40_80.npy', np.array(rows, dtype=object), allow_pickle=True)
    for run in RUNS:
        for h in (6, 24):
            s = [r for r in rows if r['run'] == run and r['hour'] == h]
            eN = np.array([r['eN'] for r in s]); eM = np.array([r['eM'] for r in s])
            rN = np.array([r['N40'] / r['N80'] for r in s])
            r3 = np.array([r['N3_40'] / r['N3_80'] for r in s])
            rM = np.array([r['M40'] / r['M80'] for r in s])
            print(f'{run:7s} {h:2d} h  L1 N: med {np.median(eN):.3f} max {eN.max():.3f} | '
                  f'L1 M: med {np.median(eM):.3f} max {eM.max():.3f} | '
                  f'N40/N80 {rN.min():.2f}-{rN.max():.2f} | N3 {r3.min():.2f}-{r3.max():.2f} | '
                  f'Mdry {rM.min():.3f}-{rM.max():.3f}')
    print('\nper cell, f=0.1, 24 h: T so2 L1N L1M N40/N80 N3_40/N3_80')
    for r in rows:
        if r['run'] == 'f=0.1' and r['hour'] == 24:
            print(f"{r['T']:5.0f} {r['so2']:8g} {r['eN']:.3f} {r['eM']:.3f} "
                  f"{r['N40']/r['N80']:.3f} {r['N3_40']/r['N3_80']:.3f}")


if __name__ == '__main__':
    main()
