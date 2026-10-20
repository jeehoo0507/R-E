"""Fair comparison: same data budget for both arms.
B1: only M log-spaced samples of Z_pp, Z_pi, Z_ii.  modal poles from AAA on M samples of Z_pp.
B2: B1 + dense probe column Z_pp (cheap: one solve per freq). modal poles from dense peaks.
Fixed-freq arm: exact Woodbury at the M samples, worst over samples.
Metric: Spearman of predicted worst |Z| (over full band) vs exact dense; top-1 regret."""
import sys, os, numpy as np
from scipy.stats import spearmanr
from scipy.interpolate import AAA
sys.path.insert(0, os.path.dirname(__file__))
from modal_check import F, sweep, modes, PDNConfig
s = 2j*np.pi*F

def basis(poles, ss):
    cols = []
    for p in poles: cols += [1/(ss-p) + 1/(ss-np.conj(p)), 1j/(ss-p) - 1j/(ss-np.conj(p))]
    return np.stack(cols + [np.ones_like(ss), ss], 1)

def fit_eval(Z, Af, Ae):
    out = np.empty((Ae.shape[0], Z.shape[1]), complex)
    for j in range(Z.shape[1]):
        z = Z[:, j]; wt = 1/np.maximum(np.abs(z), 1e-12)
        M = np.vstack([(Af*wt[:, None]).real, (Af*wt[:, None]).imag]); sc = np.linalg.norm(M, axis=0) + 1e-300
        c, *_ = np.linalg.lstsq(M/sc, np.r_[(z*wt).real, (z*wt).imag], rcond=None); out[:, j] = Ae @ (c/sc)
    return out

def aaa_poles(zs, ss, fmin, fmax):
    # rational fit in variable x = s/(2pi*1e9) for conditioning
    x = ss/(2j*np.pi*1e9) * 1j  # = s/(2pi 1e9)
    r = AAA(x, zs, rtol=1e-6, max_terms=min(len(zs)//2, 20))
    p = r.poles() * 2*np.pi*1e9
    keep = (p.imag > 2*np.pi*fmin) & (p.imag < 2*np.pi*fmax*1.2) & (p.real < 0)
    return list(p[keep])

def score(zpp, zpi, zii, cfg, ss):
    out = []
    for c, esl, esr in cfg.decaps:
        zd = (esr + ss*esl + 1/(ss*c))[:, None]
        out.append(np.abs(zpp[:, None] - zpi**2/(zii+zd)).max(0))
    return np.concatenate(out)

cfgs = {"pcb": PDNConfig(), "ip": PDNConfig(cell_size=200e-6, diel_thickness=1e-6, metal_thickness=1e-6, vrm_l=50e-12,
        decaps=((100e-12,5e-12,0.05),(20e-12,2e-12,0.1),(5e-12,1e-12,0.2)))}
rng = np.random.default_rng(11)
rows = []
for name, cfg in cfgs.items():
  for N in (12, 20):
    for trial in range(2):
        mask = np.ones((N, N), bool); idx = np.arange(N*N)
        entries = rng.choice(idx, 2, replace=False); probe = int(rng.choice(np.setdiff1d(idx, entries)))
        colp, dg = sweep(mask, entries, probe, cfg, need_diag=True)
        cand = np.setdiff1d(idx, np.r_[entries, probe])
        ex = score(colp[:, probe], colp[:, cand], dg[:, cand], cfg, s); best = ex.min()
        dense_poles = [-2*np.pi*f/(2*Q) + 2j*np.pi*f for _, f, Q, _ in modes(colp, probe)]
        line = f"{name} N={N} t{trial} K={len(dense_poles)}"
        for M in (8, 16, 32):
            fi = np.unique(np.searchsorted(F, np.logspace(7, np.log10(2e10), M)).clip(0, len(F)-1))
            res = {}
            sc = score(colp[fi, probe], colp[fi][:, cand], dg[fi][:, cand], cfg, s[fi]); res["fix"] = sc
            for tag, poles in (("B1", aaa_poles(colp[fi, probe], s[fi], 1e7, 2e10)), ("B2", dense_poles)):
                Af, Ae = basis(poles, s[fi]), basis(poles, s)
                zpp = fit_eval(colp[fi][:, [probe]], Af, Ae)[:, 0]
                if tag == "B2":  # dense probe column is available in B2
                    zpp = colp[:, probe]
                res[tag] = score(zpp, fit_eval(colp[fi][:, cand], Af, Ae), fit_eval(dg[fi][:, cand], Af, Ae), cfg, s)
                res[tag + "K"] = len(poles)
            for tag in ("fix", "B1", "B2"):
                r = spearmanr(ex, res[tag]).correlation; reg = (ex[np.argmin(res[tag])] - best)/best*100
                rows.append((name, N, M, tag, r, reg))
            line += f" | M={M} fix {spearmanr(ex,res['fix']).correlation:.2f} B1(K={res['B1K']}) {spearmanr(ex,res['B1']).correlation:.2f} B2 {spearmanr(ex,res['B2']).correlation:.2f}"
        print(line, flush=True)
import collections
print("\nSUMMARY (min..max Spearman, median top-1 regret %) over 8 instances")
g = collections.defaultdict(list)
for name, N, M, tag, r, reg in rows: g[(M, tag)].append((r, reg))
for (M, tag), v in sorted(g.items()):
    r = np.array([a for a, _ in v]); reg = np.array([b for _, b in v])
    print(f" M={M:2d} {tag:3s} rho {r.min():.2f}..{r.max():.2f} (median {np.median(r):.2f})  regret median {np.median(reg):.1f}% max {reg.max():.0f}%")
