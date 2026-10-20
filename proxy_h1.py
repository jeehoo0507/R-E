"""Day-2 proxy for H1 (no RL): does a learned action scorer transfer better across board size and
technology when it sees the frequency axis as resonance modes rather than fixed samples?

States: random boards (rect / L / holes, 1 probe, 1-3 power entries) with 0-6 random decaps already
placed. Label of action (cell i, type t) = exact worst log10 |Z_pp'(f)| / L over 700 log points after
adding that decap (L = current worst |Z_pp|, so 0 = no improvement).

Both arms get the SAME data: the loaded network's Z_pp, Z_pi, Z_ii at the 32 base log points
(the expensive data; dense data is only used for labels). Shared per-action features: decap
impedance and Z_pi, Z_ii at 8 feature points (log, / L). The arms differ only in the frequency axis:
  fixed : post-action |Z_pp'| at the 32 sample points (exact Woodbury at the samples)
  modal : AAA poles from the 32 samples of Z_pp (K <= 15), residues of Z_pi, Z_ii fitted at the
          samples, post-action curve evaluated from that rational model; features = model worst,
          and (log f_k / f_c, log Q_k, lv at f_k) for the 4 most severe poles of the post-action curve
Model: MLP 2x64 on per-state centred labels. Train: PCB 10x10. Test: PCB 10x10 (held out),
PCB 16x16, interposer-like 10x10 and 16x16. Also the untrained score of each arm (sample max /
model max) as a no-learning reference.

Run: python proxy_h1.py [--train 60 --test 20] [--out reports/proxy_h1.md]
"""
import argparse
import time

import numpy as np
import torch
from scipy.interpolate import AAA

from decap_loop.pdn import PDNConfig, lap_grid, sample_mask

F_D = np.logspace(7, np.log10(2e10), 700)
F_S = np.logspace(7, np.log10(2e10), 32)
FEAT8 = np.linspace(0, 31, 8).round().astype(int)
F_C = np.sqrt(F_D[0] * F_D[-1])
PCB = PDNConfig()
IP = PDNConfig(cell_size=200e-6, diel_thickness=1e-6, metal_thickness=1e-6, vrm_l=50e-12,
               decaps=((100e-12, 5e-12, 0.05), (20e-12, 2e-12, 0.1), (5e-12, 1e-12, 0.2)))


def zfull(mask, ent, occ, cfg, f):
    """Loaded Z at frequencies f, dense inverse: returns Z[F, n, n] for valid boards up to ~400 cells."""
    n = mask.size
    v = mask.reshape(-1).astype(float)
    L = torch.from_numpy(lap_grid(mask).toarray()).to(torch.complex128)
    w = 2 * np.pi * f
    d = (w * cfg.c_cell * cfg.tan_d + 1j * w * cfg.c_cell)[:, None] * v[None] + (1 - v)[None]
    d[:, ent] += (1.0 / (cfg.vrm_r + 1j * w * cfg.vrm_l))[:, None]
    on = np.flatnonzero(occ >= 0)
    if len(on):
        d[:, on] += (1.0 / cfg.decap_z_np(f))[occ[on]].T
    ye = torch.from_numpy(1.0 / (cfg.r_edge + 1j * w * cfg.l_edge))
    out = []
    for i in range(0, len(f), 64):
        Y = ye[i:i + 64, None, None] * L + torch.diag_embed(torch.from_numpy(d[i:i + 64]))
        out.append(torch.linalg.inv(Y))
    return torch.cat(out).numpy()


def basis(poles, s):
    cols = []
    for p in poles:
        cols += [1 / (s - p) + 1 / (s - np.conj(p)), 1j / (s - p) - 1j / (s - np.conj(p))]
    return np.stack(cols + [np.ones_like(s), s], 1)


def fit_eval(Z, Af, Ae):
    """Weighted LS of every column of Z [Fs, k] on basis Af, evaluated on Ae."""
    wt = 1 / np.maximum(np.abs(Z), 1e-12)
    out = np.empty((Ae.shape[0], Z.shape[1]), complex)
    for j in range(Z.shape[1]):
        M = np.vstack([(Af * wt[:, j:j + 1]).real, (Af * wt[:, j:j + 1]).imag])
        sc = np.linalg.norm(M, axis=0) + 1e-300
        c, *_ = np.linalg.lstsq(M / sc, np.r_[(Z[:, j] * wt[:, j]).real, (Z[:, j] * wt[:, j]).imag], rcond=None)
        out[:, j] = Ae @ (c / sc)
    return out


def aaa_poles(z, f):
    x = f / 1e9
    r = AAA(x, z, rtol=1e-6, max_terms=16)
    p = r.poles() * 2j * np.pi * 1e9
    p = p[(p.imag > 2 * np.pi * f[0]) & (p.imag < 2 * np.pi * f[-1] * 1.2) & (p.real < 0)]
    return list(p)


def sample_state(cfg, size, rng):
    mask = sample_mask(size, size, ["rect", "L", "holes"][rng.integers(3)], rng)
    cells = np.flatnonzero(mask)
    ent = rng.choice(cells, int(rng.integers(1, 4)), replace=False)
    probe = int(rng.choice(np.setdiff1d(cells, ent)))
    free = np.setdiff1d(cells, np.r_[ent, probe])
    occ = np.full(mask.size, -1)
    k = int(rng.integers(0, 7))
    occ[rng.choice(free, k, replace=False)] = rng.integers(cfg.n_types, size=k)
    return mask, ent, probe, occ


def build(cfg, size, n_states, rng):
    """Per state: features of every legal action for both arms + exact labels."""
    rows = []
    T = cfg.n_types
    zd_D, zd_S = cfg.decap_z_np(F_D), cfg.decap_z_np(F_S)
    s_D, s_S = 2j * np.pi * F_D, 2j * np.pi * F_S
    for _ in range(n_states):
        mask, ent, p, occ = sample_state(cfg, size, rng)
        cand = np.flatnonzero(mask.reshape(-1) & (occ < 0))
        cand = cand[(cand != p) & ~np.isin(cand, ent)]
        ZD, ZS = zfull(mask, ent, occ, cfg, F_D), zfull(mask, ent, occ, cfg, F_S)
        zpp_D, zpi_D, zii_D = ZD[:, p, p], ZD[:, cand, p], np.einsum("fii->fi", ZD)[:, cand]
        zpp_S, zpi_S, zii_S = ZS[:, p, p], ZS[:, cand, p], np.einsum("fii->fi", ZS)[:, cand]
        L = np.abs(zpp_S).max()                      # normaliser known from the samples only
        # exact labels on the dense grid
        y = np.stack([np.log10(np.abs(zpp_D[:, None] - zpi_D ** 2 / (zii_D + zd_D[t][:, None])).max(0) / L)
                      for t in range(T)], 1)                                   # [n_cand, T]
        # fixed-frequency arm: exact post-action values at the 32 samples
        fx = np.stack([np.log10(np.abs(zpp_S[:, None] - zpi_S ** 2 / (zii_S + zd_S[t][:, None])) / L).T
                       for t in range(T)], 1)                                  # [n_cand, T, 32]
        # modal arm: rational model from the same 32 samples
        poles = aaa_poles(zpp_S, F_S)
        Af, Ae = basis(poles, s_S), basis(poles, s_D)
        mpp = fit_eval(zpp_S[:, None], Af, Ae)[:, 0]
        mpi, mii = fit_eval(zpi_S, Af, Ae), fit_eval(zii_S, Af, Ae)
        md = []
        for t in range(T):
            curve = np.log10(np.abs(mpp[:, None] - mpi ** 2 / (mii + zd_D[t][:, None])) / L)   # [700, n_cand]
            # 4 most severe local maxima of the post-action model curve (pole tokens of the new state)
            pk = np.zeros((curve.shape[1], 4, 3))
            for c in range(curve.shape[1]):
                y_c = curve[:, c]
                i = np.flatnonzero((y_c[1:-1] > y_c[:-2]) & (y_c[1:-1] >= y_c[2:])) + 1
                i = i[np.argsort(-y_c[i])[:4]]
                for r, j in enumerate(i):
                    half = y_c[j] - 0.15                                   # ~ -3 dB in log10 |Z|
                    lft = j
                    while lft > 0 and y_c[lft] > half:
                        lft -= 1
                    rgt = j
                    while rgt < len(y_c) - 1 and y_c[rgt] > half:
                        rgt += 1
                    q = F_D[j] / max(F_D[rgt] - F_D[lft], 1e-9)
                    pk[c, r] = (np.log10(F_D[j] / F_C), np.log10(q), y_c[j])
            md.append(np.concatenate([curve.max(0)[:, None], pk.reshape(len(cand), -1)], 1))
        md = np.stack(md, 1)                                                   # [n_cand, T, 13]
        # shared features: decap impedance and local Z at 8 points
        sh_zd = np.log10(np.abs(zd_S[:, FEAT8]) / L)                           # [T, 8]
        sh_loc = np.concatenate([np.log10(np.abs(zpi_S[FEAT8]) / L).T, np.log10(np.abs(zii_S[FEAT8]) / L).T], 1)
        shared = np.concatenate([np.broadcast_to(sh_loc[:, None], (len(cand), T, 16)),
                                 np.broadcast_to(sh_zd[None], (len(cand), T, 8))], 2)
        rows.append(dict(y=y.reshape(-1), fixed=np.concatenate([shared, fx], 2).reshape(len(cand) * T, -1),
                         modal=np.concatenate([shared, md], 2).reshape(len(cand) * T, -1),
                         raw_fixed=fx.max(2).reshape(-1), raw_modal=md[..., 0].reshape(-1), K=len(poles)))
    return rows


def fit_mlp(rows, arm, seed, epochs=300):
    torch.manual_seed(seed)
    X = torch.tensor(np.concatenate([r[arm] for r in rows]), dtype=torch.float32)
    Y = torch.tensor(np.concatenate([r["y"] - r["y"].mean() for r in rows]), dtype=torch.float32)
    mu, sd = X.mean(0), X.std(0) + 1e-6
    net = torch.nn.Sequential(torch.nn.Linear(X.shape[1], 64), torch.nn.ReLU(), torch.nn.Linear(64, 64),
                              torch.nn.ReLU(), torch.nn.Linear(64, 1))
    opt = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=1e-4)
    for _ in range(epochs):
        perm = torch.randperm(len(X))
        for i in range(0, len(X), 1024):
            b = perm[i:i + 1024]
            loss = ((net((X[b] - mu) / sd)[:, 0] - Y[b]) ** 2).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return lambda Z: net((torch.tensor(Z, dtype=torch.float32) - mu) / sd)[:, 0].detach().numpy()


def spearman(a, b):
    ra, rb = np.argsort(np.argsort(a)), np.argsort(np.argsort(b))
    return np.corrcoef(ra, rb)[0, 1]


def score(rows, pred):
    rho, reg, hit = [], [], []
    for r in rows:
        p = pred(r)
        rho.append(spearman(p, r["y"]))
        best = r["y"].min()
        reg.append((10 ** (r["y"][np.argmin(p)] - best) - 1) * 100)        # % worse worst-|Z| than best action
        hit.append(np.argmin(p) in np.argsort(r["y"])[:5])
    return np.median(rho), np.mean(reg), np.median(reg), np.mean(hit)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=int, default=60)
    ap.add_argument("--test", type=int, default=20)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--out", default="reports/proxy_h1.md")
    a = ap.parse_args()
    t0 = time.time()
    train = build(PCB, 10, a.train, np.random.default_rng(1))
    tests = {"PCB 10x10 (같은 분포)": build(PCB, 10, a.test, np.random.default_rng(2)),
             "PCB 16x16 (크기)": build(PCB, 16, a.test, np.random.default_rng(3)),
             "인터포저형 10x10 (기술)": build(IP, 10, a.test, np.random.default_rng(4)),
             "인터포저형 16x16 (크기+기술)": build(IP, 16, a.test, np.random.default_rng(5))}
    print(f"data built in {time.time() - t0:.0f}s", flush=True)
    res = {}
    for name, rows in tests.items():
        res[name] = {"fixed (학습 없음: 샘플 최대)": [score(rows, lambda r: r["raw_fixed"])],
                     "modal (학습 없음: 모델 최대)": [score(rows, lambda r: r["raw_modal"])]}
    for arm in ("fixed", "modal"):
        for s in range(a.seeds):
            f = fit_mlp(train, arm, s)
            for name, rows in tests.items():
                res[name].setdefault(f"{arm} + MLP", []).append(score(rows, lambda r, f=f: f(r[arm])))
    lines = ["# 2일차 대리 실험: 고정 주파수 vs 모달 (학습된 행동 점수기의 전이)", "",
             f"학습: PCB 10x10 상태 {a.train}개. 시험 세트마다 상태 {a.test}개. MLP 시드 {a.seeds}개 평균. "
             "두 입력 모두 같은 32점 데이터만 사용. 정답 = 700점 정확 계산.", "",
             "| 시험 세트 | 입력 | 순위 상관 (중앙값) | top-1 regret 평균 % | top-1 regret 중앙값 % | top-5 적중 |",
             "|---|---|---|---|---|---|"]
    for name in tests:
        for arm, vals in res[name].items():
            v = np.mean(np.array(vals), 0)
            lines.append(f"| {name} | {arm} | {v[0]:.3f} | {v[1]:.1f} | {v[2]:.1f} | {v[3]:.2f} |")
    Ks = {name: np.mean([r["K"] for r in rows]) for name, rows in tests.items()}
    lines += ["", "평균 pole 수 K: " + ", ".join(f"{k} {v:.1f}" for k, v in Ks.items()),
              f"", f"총 시간 {time.time() - t0:.0f}s"]
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
