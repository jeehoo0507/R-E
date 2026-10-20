"""Exp 2: sparse LU vs dense full-Z cost.  usage: python exp2_factor_cost.py [dense]"""
import sys, time, resource, numpy as np, torch
from scipy.sparse.linalg import splu
from common import *

F = CFG.n_freq
fs = [1e8, 1e9, 1e10]
print("N,n,perm,f_GHz,t_factor_s,nnzLU,LU_MB,t_solve_K1_ms,t_solve_K16_ms,t_solve_K256_ms,"
      "t_diagZ_est_s(n cols),fullZ_complex64_GB(32f)")
for N in (50, 100, 200, 300):
    n = N * N; mask = rect(N); lap = lap_grid(mask)
    for perm in ("COLAMD", "MMD_AT_PLUS_A"):
        for f in fs:
            Y = sparse_Y(mask, f, lap=lap).tocsc()
            t = time.time(); lu = splu(Y, permc_spec=perm); tf = time.time() - t
            nnz = lu.L.nnz + lu.U.nnz
            ts = {}
            for K in (1, 16, 256):
                Bm = np.zeros((n, K), complex); Bm[np.arange(K) * (n // K), np.arange(K)] = 1
                t = time.time(); X = lu.solve(Bm); ts[K] = (time.time() - t) * 1e3
            tdiag = ts[256] / 1e3 * n / 256
            print(f"{N},{n},{perm},{f/1e9:g},{tf:.3f},{nnz},{nnz*20/1e6:.1f},{ts[1]:.2f},{ts[16]:.2f},{ts[256]:.1f},"
                  f"{tdiag:.1f},{n*n*F*8/1e9:.2f}", flush=True)
            if perm == "COLAMD" and f != 1e9:
                continue
    del lu
print("peak RSS GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)

if len(sys.argv) > 1 and sys.argv[1] == "dense":
    from decap_loop.pdn import build_Z
    print("torch threads", torch.get_num_threads())
    for N in (20, 30, 40, 50):
        t = time.time(); Z = build_Z(torch.ones(1, N, N, dtype=torch.bool), CFG); dt = time.time() - t
        print(f"dense build_Z N={N}: {dt:.2f}s, Z bytes={Z.numel()*8/1e9:.3f} GB", flush=True)
        del Z
    for N in (70, 100):
        n = N * N
        Y = torch.from_numpy(sparse_Y(rect(N), 1e9).toarray())
        t = time.time(); Zi = torch.linalg.inv(Y); dt = time.time() - t
        print(f"dense inv one freq N={N} (n={n}): {dt:.2f}s -> x32 freqs = {32*dt:.0f}s; Z(32f,c64)={n*n*32*8/1e9:.1f} GB", flush=True)
        del Y, Zi
    print("peak RSS GB", resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9)
