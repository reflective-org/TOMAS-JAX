"""Fill, update count and level counts of the sparse LU for several elimination orders."""
import sys, json, time, argparse
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from spike.sparse_lu import symbolic, min_degree_order
from spike.ordering import markowitz_order
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "bench" / "scripts"))
from bench_common import load_problem, NumpyChem
ap=argparse.ArgumentParser(); ap.add_argument("problem"); a=ap.parse_args()
P=load_problem(a.problem); nc=NumpyChem(P.mech); n=P.mech.n
key=np.unique(nc.e_col*n+nc.e_row); rows, cols = key % n, key // n
for name, fn in [("mindeg", lambda: min_degree_order(rows, cols, n)), ("markowitz", lambda: markowitz_order(rows, cols, n)),
                 ]:
    t0=time.perf_counter(); perm=fn(); t_ord=time.perf_counter()-t0
    pl=symbolic(rows, cols, n, perm=perm)
    upd=sum(L.u_l.size for L in pl.levels)
    print(json.dumps({"order":name,"t_order_s":round(t_ord,2),"lu_nnz":pl.nnz,"updates":int(upd),"factor_levels":len(pl.levels),
                      "fwd_levels":len(pl.fwd),"bwd_levels":len(pl.bwd)}))
