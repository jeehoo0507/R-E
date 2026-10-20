"""(1) Does a 32-point log grid miss target violations seen on a dense grid?
(2) Poles from dense probe sweep + residues fitted from 32 coarse samples only: action-rank quality."""
import sys, os, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from modal_check import F, sweep, modes, PDNConfig
s = 2j*np.pi*F
coarse = np.searchsorted(F, np.logspace(7, np.log10(2e10), 32)).clip(0, len(F)-1)

def basis(poles, ss):
    cols = []
    for p in poles: cols += [1/(ss-p) + 1/(ss-np.conj(p)), 1j/(ss-p) - 1j/(ss-np.conj(p))]
    return np.stack(cols + [np.ones_like(ss), ss], 1)

def fit_eval(Z, A_fit, A_eval):
    out = np.empty((A_eval.shape[0], Z.shape[1]), complex)
    for j in range(Z.shape[1]):
        z = Z[:, j]; wt = 1/np.maximum(np.abs(z), 1e-9)
        M = np.vstack([(A_fit*wt[:, None]).real, (A_fit*wt[:, None]).imag]); sc = np.linalg.norm(M, axis=0)
        c, *_ = np.linalg.lstsq(M/sc, np.r_[(z*wt).real, (z*wt).imag], rcond=None); out[:, j] = A_eval @ (c/sc)
    return out

cfgs = {"pcb": PDNConfig(), "ip": PDNConfig(cell_size=200e-6, diel_thickness=1e-6, metal_thickness=1e-6, vrm_l=50e-12,
        decaps=((100e-12,5e-12,0.05),(20e-12,2e-12,0.1),(5e-12,1e-12,0.2)))}
rng = np.random.default_rng(5)
for name, cfg in cfgs.items():
  for N in (12, 20):
    mask = np.ones((N, N), bool); idx = np.arange(N*N)
    entries = rng.choice(idx, 2, replace=False); probe = int(rng.choice(np.setdiff1d(idx, entries)))
    colp, dg = sweep(mask, entries, probe, cfg, need_diag=True)
    cand = np.setdiff1d(idx, np.r_[entries, probe])
    # (1) random 3-8 decap configs: worst |Z| dense vs coarse  (multi-decap via sequential Woodbury on probe only needs full Z; use 1-decap configs over all cand x type)
    miss = []
    for c, esl, esr in cfg.decaps:
        zd = (esr + s*esl + 1/(s*c))[:, None]
        z = np.abs(colp[:, [probe]] - colp[:, cand]**2/(dg[:, cand]+zd))
        miss.append(z.max(0)/z[coarse].max(0))
    miss = np.concatenate(miss)
    # (2) coarse-residue modal model
    ms = modes(colp, probe); poles = [-2*np.pi*f/(2*Q) + 2j*np.pi*f for _, f, Q, _ in ms]
    Af, Ae = basis(poles, s[coarse]), basis(poles, s)
    zpp = fit_eval(colp[coarse][:, [probe]], Af, Ae)[:, 0]; zpi = fit_eval(colp[coarse][:, cand], Af, Ae); zii = fit_eval(dg[coarse][:, cand], Af, Ae)
    ex, mo = [], []
    for c, esl, esr in cfg.decaps:
        zd = (esr + s*esl + 1/(s*c))[:, None]
        ex.append(np.abs(colp[:, [probe]] - colp[:, cand]**2/(dg[:, cand]+zd)).max(0))
        mo.append(np.abs(zpp[:, None] - zpi**2/(zii+zd)).max(0))
    ex, mo = np.concatenate(ex), np.concatenate(mo); te, tm = np.argsort(ex), np.argsort(mo)
    print(f"{name} N={N} K={len(poles)} | dense/coarse worst|Z|: median {np.median(miss):.2f}, max {miss.max():.2f}, "
          f">1.2x in {np.mean(miss>1.2)*100:.0f}% of actions | coarse-residue modal: spearman {spearmanr(ex,mo).correlation:.3f}, "
          f"regret {(ex[tm[0]]-ex[te[0]])/ex[te[0]]*100:.1f}%")
