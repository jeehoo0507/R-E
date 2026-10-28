"""Do multi-token feature combinations (resonance + others) carry more transferable information?

For each port of each board: feature groups at the bare board, label = port is in the LS teacher
solution (same boards/seeds as token_analysis2.py). A small L2 logistic regression is fitted on one
technology and scored (per-board AUC) on the other. Standardisation statistics come from the
training technology only. This is a proxy: it measures what the features tell about single ports,
not the multi-step interactions a transformer would learn.

Usage: python token_combo.py [n_boards_per_tech] [N] [workers]   (results cached in combo_cache.pkl)
"""
import os, sys, pickle, time
import numpy as np
from multiprocessing import Pool

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import token_analysis2 as T   # noqa: E402


def feats_job(args):
    tech, N, seed = args
    t0 = time.time()
    b = T.Board(tech, N, seed); rng = np.random.default_rng(seed + 1)
    zrow, zii = b.parts([], [])
    P = b.cand; W = b.mask.shape[1]; py, px = divmod(b.p, W); y, x = np.divmod(P, W)
    fi = T.fixed_idx(b)
    one = b.onestep(zrow, zii, P)                                   # [P, T]
    m = T.modal(b, zrow, zii, P)
    sol, gp = T.ls_teacher(b, rng)
    G = {
        "coord": np.c_[np.hypot(y - py, x - px) / N],
        "loopL": np.c_[2 * np.pi * b.fc * T.loop_L(b, zrow, zii, P) / b.Zt],
        "fixed8": np.log(np.abs(zrow[fi][:, P]).T / b.Zt + 1e-12),
        "onestep": np.c_[np.log(one.min(1)), np.log(one)],
    }
    if m is not None:
        G["modal"] = np.c_[m["part_ii"], m["part_pi_rel"], m["damp"], m["eii"], m["g"]]
    return dict(tech=tech, seed=seed, G=G, label=np.isin(P, [i for i, _ in sol]), ls=len(sol), gp=gp,
                t=time.time() - t0)


SETS = {
    "좌표만 (Park식)": ["coord"],
    "loop L만 (Manoharan 포트 특징)": ["loopL"],
    "고정 8점 Z (Manoharan ΔZ / R4식)": ["fixed8"],
    "loop L + 고정 8점 (Manoharan 전체)": ["loopL", "fixed8"],
    "정확한 1수 효과": ["onestep"],
    "공진(모달)만": ["modal"],
    "공진 + 1수 효과": ["modal", "onestep"],
    "공진 + 1수 + 고정 8점": ["modal", "onestep", "fixed8"],
    "전부": ["coord", "loopL", "fixed8", "onestep", "modal"],
}


def stack(rows, keys):
    X, y, bid = [], [], []
    for j, r in enumerate(rows):
        if not all(k in r["G"] for k in keys):
            continue
        X.append(np.hstack([r["G"][k] for k in keys])); y.append(r["label"]); bid.append(np.full(len(r["label"]), j))
    return np.vstack(X), np.concatenate(y), np.concatenate(bid)


def logreg(X, y, lam=1e-2, iters=500, lr=0.5):
    w = np.zeros(X.shape[1]); b0 = 0.0
    pos = y.mean(); cw = np.where(y, 0.5 / max(pos, 1e-6), 0.5 / max(1 - pos, 1e-6))   # class balance
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w + b0)))
        g = (p - y) * cw
        w -= lr * (X.T @ g / len(y) + lam * w); b0 -= lr * g.mean()
    return w, b0


def auc(s, l):
    pos, neg = s[l], s[~l]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def evaluate(train, test, keys):
    Xa, ya, _ = stack(train, keys); Xb, yb, bb = stack(test, keys)
    mu, sd = Xa.mean(0), Xa.std(0) + 1e-9
    Xa, Xb = np.clip((Xa - mu) / sd, -8, 8), np.clip((Xb - mu) / sd, -8, 8)
    w, b0 = logreg(Xa, ya); s = Xb @ w + b0
    return np.nanmean([auc(s[bb == j], yb[bb == j]) for j in np.unique(bb)])


def split_eval(rows, keys):
    """In-domain: fit on even boards, test on odd boards."""
    return evaluate(rows[0::2], rows[1::2], keys)


def main():
    NB = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    WK = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    cache = os.path.join(HERE, f"combo_cache_{NB}x{N}.pkl")
    if os.path.exists(cache):
        res = pickle.load(open(cache, "rb"))
    else:
        jobs = [(t, N, 1000 * (i + 1) + (0 if t == "pcb" else 7)) for t in T.TECH for i in range(NB)]
        res = []; t0 = time.time()
        with Pool(WK) as pool:
            for r in pool.imap_unordered(feats_job, jobs):
                res.append(r); print(f"[{r['tech']} {r['seed']}] {r['t']:.0f}s (total {time.time()-t0:.0f}s)", flush=True)
        pickle.dump(res, open(cache, "wb"))
    pcb = sorted([r for r in res if r["tech"] == "pcb"], key=lambda r: r["seed"])
    ip = sorted([r for r in res if r["tech"] == "ip"], key=lambda r: r["seed"])
    print("\n=== 특징 묶음별 'LS 선생님 해 포트' 예측 AUC (로지스틱 회귀, 판별 평균) ===")
    print("  PCB→PCB  IP→IP  | PCB→IP  IP→PCB | 특징 묶음")
    for name, keys in SETS.items():
        try:
            a = split_eval(pcb, keys); b = split_eval(ip, keys)
            c = evaluate(pcb, ip, keys); d = evaluate(ip, pcb, keys)
            print(f"   {a:.2f}    {b:.2f}  |  {c:.2f}    {d:.2f}  | {name}")
        except ValueError:
            print(f"   (특징 없음) | {name}")


if __name__ == "__main__":
    main()
