"""T5: pole-residue (in-band modes + d + jw e residual) model sufficiency for decap action ranking."""
import sys, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, __import__("os").path.dirname(__file__))
from modal_check import F, sweep, modes, PDNConfig
w = 2*np.pi*F; s = 1j*w

def basis(poles, extra_lowpole=True):
    cols = []
    for p in poles:
        cols += [1/(s-p) + 1/(s-np.conj(p)), 1j/(s-p) - 1j/(s-np.conj(p))]
    cols += [np.ones_like(s), s]
    return np.stack(cols, 1)            # [F, 2K+2] real-coefficient basis

def fit(Zcols, A):
    # weighted LS in relative sense: rows scaled by 1/|Z|
    out = np.empty_like(Zcols)
    for j in range(Zcols.shape[1]):
        z = Zcols[:, j]; wt = 1/np.maximum(np.abs(z), 1e-6)
        M = np.vstack([(A*wt[:, None]).real, (A*wt[:, None]).imag])
        sc = np.linalg.norm(M, axis=0); M = M/sc
        b = np.r_[(z*wt).real, (z*wt).imag]
        c, *_ = np.linalg.lstsq(M, b, rcond=None); out[:, j] = A @ (c/sc)
    return out

cfg = PDNConfig(); rng = np.random.default_rng(2)
for N in (12, 20):
  for trial in range(3):
    mask = np.ones((N, N), bool); idx = np.arange(N*N)
    entries = rng.choice(idx, 2, replace=False); probe = int(rng.choice(np.setdiff1d(idx, entries)))
    colp, dg = sweep(mask, entries, probe, cfg, need_diag=True)
    ms = modes(colp, probe); poles = [-2*np.pi*f/(2*Q) + 2j*np.pi*f for _, f, Q, _ in ms]
    A = basis(poles)
    cand = np.setdiff1d(idx, np.r_[entries, probe])
    zpp_f = fit(colp[:, [probe]], A)[:, 0]; zpi_f = fit(colp[:, cand], A); zii_f = fit(dg[:, cand], A)
    err = lambda a, b: np.median(np.abs(a-b)/np.abs(b))
    exact, mod = [], []
    for c, esl, esr in cfg.decaps:
        zd = (esr + s*esl + 1/(s*c))[:, None]
        exact.append(np.abs(colp[:, [probe]] - colp[:, cand]**2/(dg[:, cand]+zd)).max(0))
        mod.append(np.abs(zpp_f[:, None] - zpi_f**2/(zii_f+zd)).max(0))
    exact = np.concatenate(exact); mod = np.concatenate(mod)
    te, tm = np.argsort(exact)[:10], np.argsort(mod)[:10]
    print(f"N={N} K={len(poles)} fit err Zpp={err(zpp_f,colp[:,probe]):.3f} Zpi={err(zpi_f,colp[:,cand]):.3f} Zii={err(zii_f,dg[:,cand]):.3f} "
          f"spearman={spearmanr(exact,mod).correlation:.3f} top10={len(set(te)&set(tm))}/10 regret={(exact[tm[0]]-exact[te[0]])/exact[te[0]]*100:.2f}%")
