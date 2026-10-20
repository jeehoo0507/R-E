"""Validate sparse Y against pdn.build_Y, print lumped constants, find plane resonances from |Z| peaks."""
import time, numpy as np, torch
from scipy.sparse.linalg import splu
from scipy.signal import find_peaks
from common import *

cfg = CFG
print(f"c_cell={cfg.c_cell:.4e} F  l_edge={cfg.l_edge:.4e} H  r_edge={cfg.r_edge:.4e} ohm")
print(f"R/L corner f = {cfg.r_edge/(2*np.pi*cfg.l_edge):.3e} Hz; cells/s = {1/np.sqrt(cfg.l_edge*cfg.c_cell):.3e}")
rng = np.random.default_rng(0)
m = sample_mask(7, 9, "L", rng)
Yd = build_Y(torch.from_numpy(m), cfg).numpy()
fs = cfg.freqs().numpy()
err = max(np.abs(sparse_Y(m, f).toarray() - Yd[i]).max() / np.abs(Yd[i]).max() for i, f in enumerate(fs))
print("max rel err sparse vs build_Y:", err)

fgrid = np.logspace(7, np.log10(2e10), 600)
for N in (10, 20, 50, 100):
    mask = rect(N); lap = lap_grid(mask); n = N * N
    corner, center = 0, (N // 2) * N + N // 2
    zc = []
    t = time.time()
    for f in fgrid:
        lu = splu(sparse_Y(mask, f, lap=lap).tocsc())
        b = np.zeros(n, complex); b[corner] = 1
        zc.append(lu.solve(b)[corner])
    zc = np.abs(np.array(zc))
    pk, _ = find_peaks(np.log(zc), prominence=0.3)
    v = cfg.cell_size / np.sqrt(cfg.l_edge * cfg.c_cell)
    f10 = v / (2 * N * cfg.cell_size)
    print(f"N={N}: analytic f10 = {f10/1e9:.3f} GHz; |Z| peaks (GHz): {np.round(fgrid[pk][:6]/1e9,3)}  "
          f"|Z|@10MHz={zc[0]:.3g}  min|Z|={zc.min():.3g}  ({time.time()-t:.1f}s)")
