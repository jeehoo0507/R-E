"""Exp 1c: locality. Z_pp computed on a (2r+1)^2 window around the probe vs the full 100x100 board.
A purely local T-step message-passing network can only 'see' radius r=T."""
import numpy as np
from scipy.sparse.linalg import splu
from common import *
N = 100; mask = rect(N); c = N // 2
rng = np.random.default_rng(1)
f1 = CFG.cell_size / np.sqrt(CFG.l_edge * CFG.c_cell) / (2 * N * CFG.cell_size)
fs = [1e7, 1e8, 3e8, 1e9, 1.5e9, 3e9, 1e10]
def zpp(m, p, f, dec=()):
    n = m.size; b = np.zeros(n, complex); b[p] = 1
    return splu(sparse_Y(m, f, dec).tocsc()).solve(b)[p]
for label, dens in (("bare", 0.0), ("2% random decaps", 0.02)):
    dmask = rng.random((N, N)) < dens; dmask[c, c] = False
    dtype = rng.integers(3, size=(N, N))
    print(f"\n{label}: |dB error| of Z_pp from window radius r (full board N=100, f1={f1/1e9:.2f} GHz)")
    print("f_GHz  " + "  ".join(f"r={r:>3d}" for r in (5, 10, 20, 35, 49)))
    for f in fs:
        decf = [(i * N + j, int(dtype[i, j])) for i, j in zip(*np.nonzero(dmask))]
        zf = zpp(mask, c * N + c, f, decf)
        errs = []
        for r in (5, 10, 20, 35, 49):
            sl = slice(c - r, c + r + 1); w = 2 * r + 1
            sub = np.zeros((N, N), bool); sub[sl, sl] = True
            m2 = np.ones((w, w), bool)
            dec2 = [((i - (c - r)) * w + (j - (c - r)), int(dtype[i, j])) for i, j in zip(*np.nonzero(dmask & sub))]
            zw = zpp(m2, r * w + r, f, dec2)
            errs.append(abs(20 * np.log10(abs(zw) / abs(zf))))
        print(f"{f/1e9:6.2f} " + "  ".join(f"{e:6.2f}" for e in errs), flush=True)
