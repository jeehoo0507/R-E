"""Numerical checks for the Modal-Token idea on the R&E PDN simulator.
T1 mesh invariance: same physical board at cell 0.5mm vs 0.25mm -> mode tokens match?
T2 participation ratio Z_pi(f_k)/Z_pp(f_k) vs lossless eigenvector ratio phi_k(i)/phi_k(p)
T3 token count K vs board size (fixed band)
T4 surrogate: rank of decap actions from modal tokens only vs exact worst-case improvement
"""
import sys, time, dataclasses
import numpy as np, scipy.sparse as sp, scipy.linalg as sla
from scipy.signal import find_peaks
sys.path.insert(0, "/Users/choejihu/Documents/claude/R&E")
from decap_loop.pdn import PDNConfig, lap_grid, sample_mask

F = np.logspace(7, np.log10(2e10), 700)

def Y_of(mask, entries, cfg, f):
    lap = lap_grid(mask); v = mask.reshape(-1).astype(float); w = 2*np.pi*f
    ye = 1/(cfg.r_edge + 1j*w*cfg.l_edge); ys = w*cfg.c_cell*cfg.tan_d + 1j*w*cfg.c_cell
    d = ys*v + (1-v); d[entries] += 1/(cfg.vrm_r + 1j*w*cfg.vrm_l)
    return (ye*lap + sp.diags(d)).toarray()

def sweep(mask, entries, probe, cfg, need_diag=False):
    n = mask.size; colp = np.zeros((len(F), n), complex); dg = np.zeros((len(F), n), complex) if need_diag else None
    e = np.zeros(n); e[probe] = 1
    for k, f in enumerate(F):
        Y = Y_of(mask, entries, cfg, f)
        if need_diag:
            Z = np.linalg.inv(Y); colp[k] = Z[:, probe]; dg[k] = np.diag(Z)
        else:
            colp[k] = np.linalg.solve(Y, e)
    return colp, dg

def modes(colp, probe, rel=0.0):
    zpp = np.abs(colp[:, probe]); lz = np.log(zpp)
    pk, pr = find_peaks(lz, prominence=0.3)
    out = []
    for k in pk:
        # half-power width in log-f -> Q
        half = zpp[k]/np.sqrt(2); l = k; r = k
        while l > 0 and zpp[l] > half: l -= 1
        while r < len(F)-1 and zpp[r] > half: r += 1
        Q = F[k]/max(F[r]-F[l], 1e-9)
        out.append((k, F[k], Q, zpp[k]))
    return out

def t1_mesh():
    print("== T1 mesh invariance (8mm x 8mm board, VRM at corner, probe at (5.25mm,3.25mm))")
    res = {}
    for cs, N in ((0.5e-3, 16), (0.25e-3, 32)):
        cfg = PDNConfig(cell_size=cs)
        mask = np.ones((N, N), bool)
        def node(xm, ym): return int(ym/cs)*N + int(xm/cs)
        entries = np.array([node(0.1e-3, 0.1e-3)]); probe = node(5.3e-3, 3.3e-3)
        t = time.time(); colp, _ = sweep(mask, entries, probe, cfg)
        ms = modes(colp, probe)
        # participation at 6 fixed physical points
        pts = [(1.1e-3, 6.9e-3), (7.6e-3, 7.6e-3), (4.1e-3, 4.1e-3), (2.6e-3, 0.6e-3), (6.6e-3, 1.6e-3), (0.6e-3, 3.6e-3)]
        rho = np.array([[colp[k, node(*p)]/colp[k, probe] for p in pts] for k, *_ in ms])
        res[N] = (ms, rho); print(f" N={N} cells={N*N} sweep {time.time()-t:.1f}s  modes K={len(ms)}")
        for (k, f, Q, z), r in zip(ms, rho):
            print(f"   f={f/1e9:7.3f}GHz Q={Q:6.1f} |Zpk|={z*1e3:8.2f}mOhm  rho={np.round(np.real(r),2)}")
    return res

def t2_eig():
    print("== T2 rho vs lossless eigenvector ratio (12x12 rect and L)")
    cfg = PDNConfig(); rng = np.random.default_rng(0)
    for shape in ("rect", "L"):
        N = 12; mask = sample_mask(N, N, shape, rng) if shape != "rect" else np.ones((N, N), bool)
        idx = np.flatnonzero(mask.reshape(-1)); entries = np.array([idx[0]]); probe = int(idx[len(idx)//2 + 3])
        colp, _ = sweep(mask, entries, probe, cfg); ms = modes(colp, probe)
        lap = lap_grid(mask).toarray()[np.ix_(idx, idx)]
        lam, V = np.linalg.eigh(lap); wk = np.sqrt(np.maximum(lam, 0)/(cfg.l_edge*cfg.c_cell))/(2*np.pi)
        pi = np.searchsorted(idx, probe)
        for k, f, Q, z in ms:
            j = np.argmin(np.abs(wk - f)); near = np.abs(wk - wk[j]) < 0.02*wk[j]  # degenerate cluster
            rho = (colp[k, idx]/colp[k, probe]).real
            # project probe onto degenerate subspace: phi = V_S V_S^T e_p (probe-visible mode)
            Vs = V[:, near]; phi = Vs @ Vs[pi]; ratio = phi/phi[pi]
            c = np.corrcoef(rho, ratio)[0, 1]
            print(f"  {shape} f={f/1e9:6.2f}GHz eig={wk[j]/1e9:6.2f}GHz degen={near.sum()} corr(rho,phi_i/phi_p)={c:.3f}")

def t3_count():
    print("== T3 number of resolvable peaks K vs size (band 10MHz-20GHz)")
    cfg = PDNConfig(); rng = np.random.default_rng(1)
    for N in (8, 12, 16, 24, 30):
        Ks = []
        for _ in range(3):
            mask = np.ones((N, N), bool); idx = np.arange(N*N)
            entries = rng.choice(idx, 2, replace=False); probe = int(rng.choice(np.setdiff1d(idx, entries)))
            colp, _ = sweep(mask, entries, probe, cfg); Ks.append(len(modes(colp, probe)))
        print(f"  N={N:3d} cells={N*N:4d} f1={cfg.f1(N,N)/1e9:6.2f}GHz K={Ks}")

def t4_rank():
    print("== T4 modal-only action score vs exact score (one step, 12x12, type sweep)")
    cfg = PDNConfig(); rng = np.random.default_rng(2); N = 12
    zd_all = np.stack([esr + 1j*2*np.pi*F*esl + 1/(1j*2*np.pi*F*c) for c, esl, esr in cfg.decaps])
    from scipy.stats import spearmanr
    for trial in range(4):
        mask = np.ones((N, N), bool); idx = np.arange(N*N)
        entries = rng.choice(idx, 2, replace=False); probe = int(rng.choice(np.setdiff1d(idx, entries)))
        colp, dg = sweep(mask, entries, probe, cfg, need_diag=True)
        zpp = colp[:, probe]; ms = modes(colp, probe); kk = np.array([m[0] for m in ms])
        cand = np.setdiff1d(idx, np.r_[entries, probe])
        exact, modal = [], []
        for t in range(len(cfg.decaps)):
            zd = zd_all[t][:, None]
            znew = zpp[:, None] - colp[:, cand]**2/(dg[:, cand] + zd)       # exact on full grid
            exact.append(np.abs(znew).max(0))                                 # worst |Z| over band
            zm = zpp[kk, None] - colp[kk][:, cand]**2/(dg[kk][:, cand] + zd[kk])  # only at mode freqs
            modal.append(np.abs(zm).max(0))
        exact = np.concatenate(exact); modal = np.concatenate(modal)
        rs = spearmanr(exact, modal).correlation
        top_e = np.argsort(exact)[:10]; top_m = np.argsort(modal)[:10]
        print(f"  trial{trial} K={len(kk)} spearman={rs:.3f} top10 overlap={len(set(top_e)&set(top_m))}/10 "
              f"best-modal exact regret={(exact[top_m[0]]-exact[top_e[0]])/exact[top_e[0]]*100:.2f}%")

if __name__ == "__main__":
    for t in sys.argv[1:] or ["t1", "t2", "t3", "t4"]:
        globals()[{"t1": "t1_mesh", "t2": "t2_eig", "t3": "t3_count", "t4": "t4_rank"}[t]]()
