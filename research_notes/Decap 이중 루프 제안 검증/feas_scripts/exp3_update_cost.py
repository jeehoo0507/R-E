"""Exp 3: cost of evaluating / committing one decap: refactorize vs Sherman-Morrison-Woodbury on base LU.
usage: python exp3_update_cost.py N"""
import sys, time, numpy as np
from scipy.sparse.linalg import splu
from common import *

N = int(sys.argv[1]); n = N * N; mask = rect(N); lap = lap_grid(mask)
fs = CFG.freqs().numpy(); F = len(fs)
probe = (N // 2) * N + N // 2
rng = np.random.default_rng(0)
K = 16
seq = random_decaps(mask, K, rng, exclude=[probe])
zd = np.array([[1 / decap_y(f, [t])[0] for t in range(CFG.n_types)] for f in fs])  # [F,T]

# one-time setup
t = time.time(); lus = [splu(sparse_Y(mask, f, lap=lap).tocsc(), permc_spec="MMD_AT_PLUS_A") for f in fs]; t_fac = time.time() - t
e = np.zeros(n, complex); e[probe] = 1
t = time.time(); colp = np.stack([lu.solve(e) for lu in lus]); t_probe = time.time() - t  # [F,n]
nd = F if N <= 50 else 2
t = time.time()
diag = np.zeros((F, n), complex)
for fi in range(nd):
    for s in range(0, n, 512):
        Bm = np.zeros((n, min(512, n - s)), complex); Bm[np.arange(s, s + Bm.shape[1]), np.arange(Bm.shape[1])] = 1
        diag[fi, s:s + Bm.shape[1]] = lus[fi].solve(Bm)[np.arange(s, s + Bm.shape[1]), np.arange(Bm.shape[1])]
t_diag = (time.time() - t) * F / nd
if nd < F:  # fill remaining freqs (timing extrapolated) -- compute cheaply via exactness not needed for timing
    for fi in range(nd, F):
        diag[fi] = diag[1]
print(f"N={N} n={n} F={F}: setup factor(32f)={t_fac:.2f}s probe col={t_probe:.3f}s diag(Z) all nodes={t_diag:.1f}s"
      f"{' (extrapolated from %d freqs)' % nd if nd < F else ''}")

cols = []   # columns Z e_P for placed decaps, [F,n] each
placed, types = [], []
t_ref, t_commit, t_eval, errs = [], [], [], []
for step, (node, typ) in enumerate(seq):
    # (b1) evaluate ALL candidates x types with current placement (needs diag, probe col, placed cols)
    t = time.time()
    if placed:
        C = np.stack(cols, 1)                                       # [F,k,n]
        P = np.array(placed)
        M = C[:, :, P] + np.stack([np.diag(zd[fi, types]) for fi in range(F)])  # [F,k,k]
        Minv = np.linalg.inv(M)
        zpP = colp[:, P]                                            # [F,k]
        zip_ = colp - np.einsum('fkn,fkl,fl->fn', C, Minv, zpP)     # Z'_ip
        zii = diag - np.einsum('fkn,fkl,fln->fn', C, Minv, C)       # Z'_ii
    else:
        zip_, zii = colp, diag
    zpp = zip_[:, probe]
    new = zpp[:, None, None] - zip_[:, :, None] ** 2 / (zii[:, :, None] + zd[:, None, :])   # [F,n,T]
    score = np.log10(np.abs(new)).mean(0)
    t_eval.append(time.time() - t)
    # (b2) commit chosen decap: one extra solve per freq on the base LU
    t = time.time()
    ei = np.zeros(n, complex); ei[node] = 1
    cols.append(np.stack([lu.solve(ei) for lu in lus])); placed.append(node); types.append(typ)
    t_commit.append(time.time() - t)
    P = np.array(placed); C = np.stack(cols, 1)
    M = C[:, :, P] + np.stack([np.diag(zd[fi, types]) for fi in range(F)])
    zpp_smw = colp[:, probe] - np.einsum('fk,fkl,fl->f', colp[:, P], np.linalg.inv(M), colp[:, P])
    # (a) refactorize with decaps stamped into Y, solve probe column
    t = time.time()
    zpp_ref = np.array([splu(sparse_Y(mask, f, list(zip(placed, types)), lap=lap).tocsc(), permc_spec="MMD_AT_PLUS_A").solve(e)[probe] for f in fs])
    t_ref.append(time.time() - t)
    errs.append(np.abs(zpp_ref - zpp_smw).max() / np.abs(zpp_ref).max())
print("step k: refactor+solve(32f) [s] | SMW commit(32f) [s] | SMW eval all n*3 candidates(32f) [s] | rel err SMW vs refactor")
for k in (0, 3, 7, 15):
    print(f"k={k+1:2d}: {t_ref[k]:.3f} | {t_commit[k]:.4f} | {t_eval[k]:.3f} | {errs[k]:.1e}")
print(f"per-candidate eval via refactorization = {np.mean(t_ref):.3f}s -> all {n*3} candidates would be {np.mean(t_ref)*n*3/3600:.1f} h per greedy step")
print(f"per-candidate eval via SMW (given diag) = {np.mean(t_eval)/(n*3)*1e6:.2f} us")
