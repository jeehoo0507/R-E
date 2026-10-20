"""Shared helpers: sparse version of decap_loop.pdn.build_Y (same formulas), decap admittance,
geometric multigrid. Project files are only imported, never modified."""
import sys
import numpy as np
import scipy.sparse as sp

PROJ = "/Users/choejihu/Documents/claude/R&E"
sys.path.insert(0, PROJ)
from decap_loop.pdn import PDNConfig, build_Y, sample_mask  # noqa: E402

CFG = PDNConfig()


def lap_grid(mask):
    """Graph Laplacian of valid cells (4-neighbour), as in pdn.build_Y."""
    H, W = mask.shape
    n = H * W
    idx = np.arange(n).reshape(H, W)
    rows, cols = [], []
    for a, b, ok in ((idx[:, :-1], idx[:, 1:], mask[:, :-1] & mask[:, 1:]),
                     (idx[:-1, :], idx[1:, :], mask[:-1, :] & mask[1:, :])):
        a, b = a[ok], b[ok]
        rows += [a, b]
        cols += [b, a]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return sp.diags(np.asarray(A.sum(1)).ravel()) - A


def decap_y(f, types, cfg=CFG):
    w = 2 * np.pi * f
    out = []
    for t in types:
        c, esl, esr = cfg.decaps[t]
        out.append(1.0 / (esr + 1j * w * esl + 1.0 / (1j * w * c)))
    return np.array(out)


def ye_ys(f, cfg=CFG):
    w = 2 * np.pi * f
    ye = 1.0 / (cfg.r_edge + 1j * w * cfg.l_edge)
    ys = w * cfg.c_cell * cfg.tan_d + 1j * w * cfg.c_cell
    return ye, ys


def sparse_Y(mask, f, decaps=(), cfg=CFG, lap=None, shift=1.0):
    """Y(f) (csr complex128). decaps: list of (node, type).  shift multiplies the shunt term
    (shift=1-0.5j gives the complex-shifted-Laplacian preconditioner)."""
    if lap is None:
        lap = lap_grid(mask)
    v = mask.reshape(-1).astype(float)
    ye, ys = ye_ys(f, cfg)
    d = shift * ys * v + (1 - v)
    if len(decaps):
        nodes = np.array([p for p, _ in decaps])
        yd = decap_y(f, [t for _, t in decaps], cfg)
        np.add.at(d, nodes, yd)
    return (ye * lap + sp.diags(d)).tocsr().astype(np.complex128)


def rect(N):
    return np.ones((N, N), dtype=bool)


def random_decaps(mask, n_dec, rng, exclude=()):
    cells = np.setdiff1d(np.flatnonzero(mask.reshape(-1)), np.array(exclude, dtype=int))
    nodes = rng.choice(cells, n_dec, replace=False)
    return [(int(p), int(rng.integers(CFG.n_types))) for p in nodes]


# ------------------------------------------------------------------ multigrid
def _p1(n):
    c = list(range(0, n, 2))
    if c[-1] != n - 1:
        c.append(n - 1)
    pos = {ci: j for j, ci in enumerate(c)}
    r, cc, v = [], [], []
    for i in range(n):
        if i in pos:
            r.append(i); cc.append(pos[i]); v.append(1.0)
        else:
            r += [i, i]; cc += [pos[i - 1], pos[i + 1]]; v += [0.5, 0.5]
    return sp.csr_matrix((v, (r, cc)), shape=(n, len(c)))


class MG:
    """Geometric V-cycle, bilinear interpolation, Galerkin coarse ops (P^T A P, complex symmetric),
    damped-Jacobi smoother.  Idealised stand-in for 'Jacobi message passing + coarse-grid attention'."""

    def __init__(self, A, shape, nu=2, omega=0.7, coarse_n=64):
        from scipy.sparse.linalg import splu
        self.A, self.P, self.Dinv = [A.tocsr()], [], []
        H, W = shape
        while H * W > coarse_n and min(H, W) >= 3:
            P = sp.kron(_p1(H), _p1(W)).tocsr()
            self.P.append(P)
            Ac = (P.T @ self.A[-1] @ P).tocsr()
            self.A.append(Ac)
            H, W = _p1(H).shape[1], _p1(W).shape[1]
        self.Dinv = [1.0 / A_.diagonal() for A_ in self.A]
        self.coarse = splu(self.A[-1].tocsc())
        self.nu, self.om = nu, omega
        self.levels = len(self.A)

    def vcycle(self, b, lvl=0):
        A, Di = self.A[lvl], self.Dinv[lvl]
        if lvl == self.levels - 1:
            return self.coarse.solve(b)
        x = np.zeros_like(b)
        for _ in range(self.nu):
            x = x + self.om * Di * (b - A @ x)
        r = b - A @ x
        P = self.P[lvl]
        x = x + P @ self.vcycle(P.T @ r, lvl + 1)
        for _ in range(self.nu):
            x = x + self.om * Di * (b - A @ x)
        return x
