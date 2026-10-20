"""Rival: fixed log-spaced frequency samples (M) instead of pole-residue. Same Woodbury, max over samples."""
import sys, os, numpy as np
from scipy.stats import spearmanr
sys.path.insert(0, os.path.dirname(__file__))
from modal_check import F, sweep, modes, PDNConfig
s = 2j*np.pi*F
def score(colp, dg, probe, cand, cfg, fi):
    out = []
    for c, esl, esr in cfg.decaps:
        zd = (esr + s[fi]*esl + 1/(s[fi]*c))[:, None]
        out.append(np.abs(colp[fi][:, [probe]] - colp[fi][:, cand]**2/(dg[fi][:, cand]+zd)).max(0))
    return np.concatenate(out)
cfgs = {"pcb": PDNConfig(), "ip": PDNConfig(cell_size=200e-6, diel_thickness=1e-6, metal_thickness=1e-6, vrm_l=50e-12,
        decaps=((100e-12,5e-12,0.05),(20e-12,2e-12,0.1),(5e-12,1e-12,0.2)))}
rng = np.random.default_rng(2)
for name, cfg in cfgs.items():
  for N in (12, 20):
    for trial in range(2):
        mask = np.ones((N, N), bool); idx = np.arange(N*N)
        entries = rng.choice(idx, 2, replace=False); probe = int(rng.choice(np.setdiff1d(idx, entries)))
        colp, dg = sweep(mask, entries, probe, cfg, need_diag=True)
        cand = np.setdiff1d(idx, np.r_[entries, probe]); K = len(modes(colp, probe))
        ex = score(colp, dg, probe, cand, cfg, np.arange(len(F))); te = np.argsort(ex)[:10]
        row = []
        for M in (4, 8, 16, 32):
            fi = np.linspace(0, len(F)-1, M).round().astype(int); sc = score(colp, dg, probe, cand, cfg, fi)
            tm = np.argsort(sc)
            row.append(f"M={M}:rho={spearmanr(ex,sc).correlation:.2f},regret={(ex[tm[0]]-ex[te[0]])/ex[te[0]]*100:.0f}%")
        print(name, f"N={N} K={K}", " ".join(row))
