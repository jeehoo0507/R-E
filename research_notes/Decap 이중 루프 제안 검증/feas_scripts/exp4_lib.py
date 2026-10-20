import numpy as np
from common import *

def instance(N, rng, shape=None):
    shape = shape or ["rect", "L", "holes"][rng.integers(3)]
    m = sample_mask(N, N, shape, rng)
    cells = np.flatnonzero(m.reshape(-1))
    probe = int(rng.choice(cells))
    rest = cells[cells != probe]
    ko = rng.choice(rest, int(rng.uniform(0, 0.3) * len(rest)), replace=False)
    cand = np.setdiff1d(rest, ko)
    return m, probe, cand, shape

def dense_Z(m, fs):
    lap = lap_grid(m)
    v = m.reshape(-1)
    Z = np.stack([np.linalg.inv(sparse_Y(m, f, lap=lap).toarray()) for f in fs])
    return Z * (v[None, :, None] & v[None, None, :])

def zd_table(fs):
    return np.array([[1 / decap_y(f, [t])[0] for t in range(CFG.n_types)] for f in fs])  # [F,T]

def state(Z, probe, placed, types, zd):
    """Z'_ip [F,n], Z'_ii [F,n] after placements (Schur)."""
    zip_, zii = Z[:, :, probe].copy(), np.einsum('fii->fi', Z).copy()
    if placed:
        P = np.array(placed)
        C = Z[:, :, P]                                     # [F,n,k]
        M = Z[:, P][:, :, P] + np.stack([np.diag(zd[f, types]) for f in range(len(Z))])
        Mi = np.linalg.inv(M)
        zip_ -= np.einsum('fnk,fkl,fl->fn', C, Mi, Z[:, P, probe])
        zii -= np.einsum('fnk,fkl,fnl->fn', C, Mi, C)
    return zip_, zii

def zpp_after(Z, probe, placed, types, zd):
    return state(Z, probe, placed, types, zd)[0][:, probe]

def greedy_mincount(Z, probe, cand, zd, Zt, kmax=200, prune=True):
    placed, types = [], []
    avail = np.zeros(Z.shape[1], bool); avail[cand] = True
    while True:
        zip_, zii = state(Z, probe, placed, types, zd)
        zpp = zip_[:, probe]
        if np.all(np.abs(zpp) <= Zt):
            break
        if len(placed) >= kmax or not avail.any():
            return None, None, np.abs(zpp)
        new = zpp[:, None, None] - zip_[:, :, None] ** 2 / (zii[:, :, None] + zd[:, None, :])
        viol = np.clip(np.log10(np.abs(new) / Zt[:, None, None]), 0, None).sum(0)   # [n,T]
        viol[~avail] = np.inf
        i, t = np.unravel_index(np.argmin(viol), viol.shape)
        placed.append(int(i)); types.append(int(t)); avail[i] = False
    if prune:
        changed = True
        while changed:
            changed = False
            for j in range(len(placed)):
                p2, t2 = placed[:j] + placed[j + 1:], types[:j] + types[j + 1:]
                if np.all(np.abs(zpp_after(Z, probe, p2, t2, zd)) <= Zt):
                    placed, types, changed = p2, t2, True
                    break
    return placed, types, np.abs(zpp_after(Z, probe, placed, types, zd))
