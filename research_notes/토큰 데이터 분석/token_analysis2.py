"""Training-free token analysis v2: bigger boards, LS teacher, modal normalisation variants.

Changes from token_analysis.py
- loaded state kept as (bare Z, placed set) and only the probe row + diagonal are rebuilt by
  Woodbury, so 20x20 boards fit in memory
- teacher = greedy+prune followed by a destroy-and-repair local search (remove 2, greedy repair,
  prune; accept if not worse)
- port features evaluated at the bare board; modal port features come in several normalisations
- mode-level features: several ways to normalise the mode frequency and size
- boards are built in parallel worker processes

A2: KS distance PCB vs interposer   A3: AUC for "port in teacher solution", direction fixed on PCB
A4: one-feature greedy rules, counts relative to the LS teacher

Usage: python token_analysis2.py [n_boards_per_tech] [N] [workers]
"""
import os, sys, time
import numpy as np
from multiprocessing import Pool
from scipy.signal import find_peaks
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "모달 토큰 검증"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from modal_check import F as F_FULL, Y_of, PDNConfig   # noqa: E402
from decap_loop.pdn import sample_mask                # noqa: E402

F = F_FULL[::2]                       # 350 log points, 10 MHz - 20 GHz
S = 2j * np.pi * F
M_FIX = 8
LS_TRIALS = 40
TECH = {
    "pcb": dict(cfg=PDNConfig(), band=(10e6, 5e9)),
    "ip": dict(cfg=PDNConfig(cell_size=200e-6, diel_thickness=1e-6, metal_thickness=1e-6, vrm_l=50e-12,
                             decaps=((100e-12, 5e-12, 0.05), (20e-12, 2e-12, 0.1), (5e-12, 1e-12, 0.2))),
               band=(200e6, 20e9)),
}


# ------------------------------------------------------------------ exact physics on (bare Z, placed set)
class Board:
    def __init__(self, tech, N, seed):
        rng = np.random.default_rng(seed)
        spec = TECH[tech]; self.tech = tech; self.cfg = spec["cfg"]
        self.shape = str(rng.choice(["rect", "L", "holes"]))
        self.mask = sample_mask(N, N, self.shape, rng)
        valid = np.flatnonzero(self.mask.reshape(-1))
        self.entries = rng.choice(valid, rng.integers(1, 3), replace=False)
        self.p = int(rng.choice(np.setdiff1d(valid, self.entries)))
        self.cand = np.setdiff1d(valid, np.r_[self.entries, self.p])
        self.Z = np.stack([np.linalg.inv(Y_of(self.mask, self.entries, self.cfg, f)) for f in F])
        self.diag0 = np.einsum("fii->fi", self.Z).copy()
        self.zd = np.stack([esr + S * esl + 1 / (S * c) for c, esl, esr in self.cfg.decaps], 1)   # [F, T]
        self.srf = np.array([1 / (2 * np.pi * np.sqrt(esl * c)) for c, esl, _ in self.cfg.decaps])
        lo, hi = spec["band"]; self.lo, self.hi = lo, hi
        self.inb = (F >= lo) & (F <= hi); self.fc = np.sqrt(lo * hi)
        self.Zt = 1.0
        best = min(self.worst_raw(list(self.cand), [t] * len(self.cand)) for t in range(self.zd.shape[1]))
        self.Zt = best * rng.uniform(1.5, 3.0)

    def parts(self, ports, types):
        """Loaded probe row zpi [F, n] and diagonal zii [F, n]."""
        zrow, zii = self.Z[:, self.p, :], self.diag0
        if len(ports) == 0:
            return zrow, zii
        P = np.asarray(ports); T = np.asarray(types)
        A = self.Z[:, P][:, :, P] + np.einsum("fk,kl->fkl", self.zd[:, T], np.eye(len(P)))
        ZnS = self.Z[:, :, P]                                     # [F, n, s]
        X = np.linalg.solve(A, np.swapaxes(ZnS, 1, 2))           # [F, s, n]
        zrow = zrow - np.einsum("fs,fsn->fn", self.Z[:, self.p, P], X)
        zii = zii - np.einsum("fns,fsn->fn", ZnS, X)
        return zrow, zii

    def zpp(self, ports, types):
        if len(ports) == 0:
            return self.Z[:, self.p, self.p]
        P = np.asarray(ports); T = np.asarray(types)
        A = self.Z[:, P][:, :, P] + np.einsum("fk,kl->fkl", self.zd[:, T], np.eye(len(P)))
        zp = self.Z[:, self.p, P]
        return self.Z[:, self.p, self.p] - np.einsum("fs,fs->f", zp, np.linalg.solve(A, zp[:, :, None])[..., 0])

    def worst_raw(self, ports, types):
        return np.abs(self.zpp(ports, types)[self.inb]).max()

    def worst(self, sol):
        return self.worst_raw([i for i, _ in sol], [t for _, t in sol]) / self.Zt

    def onestep(self, zrow, zii, ports):
        zp = zrow[:, self.p][:, None, None]; zpi = zrow[:, ports][:, :, None]; zi = zii[:, ports][:, :, None]
        return (np.abs(zp - zpi ** 2 / (zi + self.zd[:, None, :]))[self.inb] / self.Zt).max(0)   # [P, T]


def prune(b, sol):
    sol = list(sol); changed = True
    while changed:
        changed = False
        for k in range(len(sol)):
            trial = sol[:k] + sol[k + 1:]
            if b.worst(trial) <= 1:
                sol = trial; changed = True; break
    return sol


def greedy_fill(b, sol, cap=80):
    sol = list(sol)
    for _ in range(cap):
        if b.worst(sol) <= 1:
            return sol, True
        zrow, zii = b.parts([i for i, _ in sol], [t for _, t in sol])
        free = np.setdiff1d(b.cand, [i for i, _ in sol])
        sc = b.onestep(zrow, zii, free); i, t = np.unravel_index(np.argmin(sc), sc.shape)
        sol.append((int(free[i]), int(t)))
    return sol, b.worst(sol) <= 1


def ls_teacher(b, rng):
    sol, ok = greedy_fill(b, [])
    sol = prune(b, sol); best = list(sol); gp = len(sol)
    for _ in range(LS_TRIALS):
        if len(best) < 2:
            break
        drop = set(rng.choice(len(best), 2, replace=False).tolist())
        cand, ok = greedy_fill(b, [s for k, s in enumerate(best) if k not in drop])
        if ok:
            cand = prune(b, cand)
            if len(cand) <= len(best):
                best = cand
    return best, gp


# ------------------------------------------------------------------ modal extraction
def modes(b, zpp):
    z = np.abs(zpp); pk, _ = find_peaks(np.log(z), prominence=0.3)
    out = []
    for k in pk:
        if not b.inb[k]:
            continue
        half = z[k] / np.sqrt(2); l = r = k
        while l > 0 and z[l] > half: l -= 1
        while r < len(F) - 1 and z[r] > half: r += 1
        out.append((k, F[k], F[k] / max(F[r] - F[l], 1e-9), z[k]))
    return out


def basis(poles):
    cols = []
    for p in poles:
        cols += [1 / (S - p) + 1 / (S - np.conj(p)), 1j / (S - p) - 1j / (S - np.conj(p))]
    return np.stack(cols + [np.ones_like(S), S], 1)


def fit(A, Zc):
    K = (A.shape[1] - 2) // 2; n = Zc.shape[1]
    R = np.zeros((K, n), complex); d = np.zeros(n); e = np.zeros(n)
    for j in range(n):
        z = Zc[:, j]; wt = 1 / np.maximum(np.abs(z), 1e-12)
        Mx = np.vstack([(A * wt[:, None]).real, (A * wt[:, None]).imag]); sc = np.linalg.norm(Mx, axis=0) + 1e-30
        c, *_ = np.linalg.lstsq(Mx / sc, np.r_[(z * wt).real, (z * wt).imag], rcond=None); c = c / sc
        R[:, j] = c[0:2 * K:2] + 1j * c[1:2 * K:2]; d[j] = c[-2]; e[j] = c[-1]
    return R, d, e


def modal(b, zrow, zii, ports):
    ms = modes(b, zrow[:, b.p])
    if not ms:
        return None
    ks = int(np.argmax([z for *_, z in ms]))
    poles = [-np.pi * f / Q + 2j * np.pi * f for _, f, Q, _ in ms]
    A = basis(poles)
    Rp, _, _ = fit(A, zrow[:, [b.p]]); Rpi, _, _ = fit(A, zrow[:, ports]); Rii, dii, eii = fit(A, zii[:, ports])
    rpp, rpi, rii = np.abs(Rp[ks, 0]), np.abs(Rpi[ks]), np.abs(Rii[ks])
    k0 = ms[ks][0]
    zii_k = np.abs(zii[k0, ports]); zd_k = np.abs(b.zd[k0]).min()
    return dict(
        g=rpi / np.sqrt(rpp * rii + 1e-30),                     # v1 (rank-1 => ~1 everywhere)
        part_ii=np.log(rii / (rpp + 1e-30) + 1e-30),            # participation ratio phi_i^2/phi_p^2
        part_pi_rel=rpi / (rpi.max() + 1e-30),                   # transfer residue relative to board max
        damp=np.log(zii_k / zd_k),                               # local resonance amplitude vs best decap
        eii=2 * np.pi * b.fc * eii / b.Zt,
        f_worst=ms[ks][1], modes=ms)


def loop_L(b, zrow, zii, ports):
    k0 = int(np.argmin(np.abs(np.log(F / b.fc))))
    zs = zrow[k0, b.p] - zrow[k0, ports] ** 2 / zii[k0, ports]
    return zs.imag / (2 * np.pi * F[k0])


def fixed_idx(b):
    idx = np.flatnonzero(b.inb)
    return idx[np.linspace(0, len(idx) - 1, M_FIX).round().astype(int)]


def type_rule(b, f_target):
    return int(np.argmin(np.abs(np.log(b.srf / f_target))))


def dense_worst_f(b, zpp):
    return F[int(np.argmax(np.abs(zpp) * b.inb))]


# ------------------------------------------------------------------ rules
def rule(b, name, zrow, zii, free):
    p = b.p; W = b.mask.shape[1]
    if name == "greedy":
        sc = b.onestep(zrow, zii, free); i, t = np.unravel_index(np.argmin(sc), sc.shape); return free[i], t
    if name == "coord":
        py, px = divmod(p, W); y, x = np.divmod(free, W)
        return free[np.argmin(np.hypot(y - py, x - px))], type_rule(b, dense_worst_f(b, zrow[:, p]))
    if name == "loopL":
        return free[np.argmin(loop_L(b, zrow, zii, free))], type_rule(b, dense_worst_f(b, zrow[:, p]))
    if name == "fixed":
        fi = fixed_idx(b); k = fi[np.argmax(np.abs(zrow[fi, p]))]
        return free[np.argmax(np.abs(zrow[k, free]))], type_rule(b, F[k])
    if name in ("modal_g", "modal_damp"):
        m = modal(b, zrow, zii, free)
        if m is None:
            return rule(b, "fixed", zrow, zii, free)
        sc = m["g"] * m["part_pi_rel"] if name == "modal_g" else m["damp"]
        return free[np.argmax(sc)], type_rule(b, m["f_worst"])
    raise ValueError(name)


RULES = ["greedy", "coord", "loopL", "fixed", "modal_g", "modal_damp"]


def run_rule(b, name, cap=80):
    sol = []
    for _ in range(cap):
        if b.worst(sol) <= 1:
            break
        zrow, zii = b.parts([i for i, _ in sol], [t for _, t in sol])
        free = np.setdiff1d(b.cand, [i for i, _ in sol])
        i, t = rule(b, name, zrow, zii, free); sol.append((int(i), int(t)))
    ok = b.worst(sol) <= 1
    return len(prune(b, sol)) if ok else np.nan, ok


# ------------------------------------------------------------------ per-board job
def job(args):
    tech, N, seed = args
    t0 = time.time()
    b = Board(tech, N, seed); rng = np.random.default_rng(seed + 1)
    zrow, zii = b.parts([], [])
    P = b.cand; W = b.mask.shape[1]; py, px = divmod(b.p, W); y, x = np.divmod(P, W)
    fi = fixed_idx(b); ws = fi[np.argmax(np.abs(zrow[fi, b.p]))]
    L = loop_L(b, zrow, zii, P); m = modal(b, zrow, zii, P)
    feats = {
        "좌표: probe와 거리 / 판 크기": np.hypot(y - py, x - px) / N,
        "loopL 정규화 (Manoharan식)": 2 * np.pi * b.fc * L / b.Zt,
        "고정 8점 |Z_pi|/Zt (최악 샘플)": np.abs(zrow[ws, P]) / b.Zt,
        "정확한 1수 효과": b.onestep(zrow, zii, P).min(1),
    }
    mode_rows = []
    if m is not None:
        feats["모달 v1: 결합도 g"] = m["g"]
        feats["모달 v2: 참여비 log(r_ii/r_pp)"] = m["part_ii"]
        feats["모달 v3: r_pi / 판 최대"] = m["part_pi_rel"]
        feats["모달 v4: log(|Z_ii(f_k)|/|z_d|) 감쇠력"] = m["damp"]
        feats["모달: 잔여 L 정규화"] = m["eii"]
        srf_mid = np.sqrt(b.srf.min() * b.srf.max())
        for _, f, Q, z in m["modes"]:
            mode_rows.append(dict(f_raw=f, f_fc=f / b.fc, f_band=(np.log(f) - np.log(b.lo)) / (np.log(b.hi) - np.log(b.lo)),
                                  f_srf=f / srf_mid, f_srf_rank=np.log(f / b.srf.min()) / np.log(b.srf.max() / b.srf.min()),
                                  Q=Q, z_raw=z, z_zt=z / b.Zt))
    sol, gp = ls_teacher(b, rng)
    label = np.isin(P, [i for i, _ in sol])
    rules = {r: run_rule(b, r) for r in RULES}
    return dict(tech=tech, seed=seed, shape=b.shape, feats=feats, label=label, mode_rows=mode_rows,
                ls=len(sol), gp=gp, rules=rules, K=0 if m is None else len(m["modes"]), t=time.time() - t0)


def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def main():
    NB = int(sys.argv[1]) if len(sys.argv) > 1 else 10
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    WK = int(sys.argv[3]) if len(sys.argv) > 3 else 6
    jobs = [(t, N, 1000 * (i + 1) + (0 if t == "pcb" else 7)) for t in TECH for i in range(NB)]
    t0 = time.time(); res = []
    with Pool(WK) as pool:
        for r in pool.imap_unordered(job, jobs):
            res.append(r)
            print(f"[{r['tech']} seed={r['seed']}] {r['shape']:5s} K={r['K']} LS={r['ls']} g+p={r['gp']} "
                  f"{r['t']:.0f}s (total {time.time()-t0:.0f}s)", flush=True)
    R = {t: [r for r in res if r["tech"] == t] for t in TECH}

    print(f"\n=== 선생님: LS 개수 / greedy+prune 개수 ===")
    for t in TECH:
        ratio = np.array([r["ls"] / r["gp"] for r in R[t]])
        print(f"  {t}: 중앙값 {np.median(ratio):.2f}, LS가 더 적은 판 {np.mean(ratio < 1):.0%}, "
              f"모드 수/판 {np.mean([r['K'] for r in R[t]]):.1f}")

    keys = [k for k in R["pcb"][0]["feats"]]
    keys += [k for r in R["pcb"] + R["ip"] for k in r["feats"] if k not in keys]
    keys = list(dict.fromkeys(keys))
    print("\n=== A2 포트 특징 종류 불변성 (KS, 0=같음) / A3 LS 선생님 해 예측 AUC (방향은 PCB 기준) ===")
    print("   KS    PCB→IP   특징")
    for k in keys:
        a = [r["feats"][k] for r in R["pcb"] if k in r["feats"]]; c = [r["feats"][k] for r in R["ip"] if k in r["feats"]]
        if not a or not c:
            continue
        ks = ks_2samp(np.concatenate(a), np.concatenate(c)).statistic
        ap = np.nanmean([auc(r["feats"][k], r["label"]) for r in R["pcb"] if k in r["feats"]])
        sg = 1 if ap >= 0.5 else -1
        ai = np.nanmean([auc(sg * r["feats"][k], r["label"]) for r in R["ip"] if k in r["feats"]])
        print(f"  {ks:4.2f}  {max(ap, 1-ap):.2f}→{ai:.2f}  {k}")

    print("\n=== 모드 정규화 점검: PCB vs IP 분포 차이 (KS) ===")
    mp = [m for r in R["pcb"] for m in r["mode_rows"]]; mi = [m for r in R["ip"] for m in r["mode_rows"]]
    names = dict(f_raw="주파수 원값", f_fc="f / 대역 중심", f_band="대역 안 상대 위치 (0~1)", f_srf="f / decap SRF 중간값",
                 f_srf_rank="decap SRF 범위 안 상대 위치", Q="Q", z_raw="|Z| 원값", z_zt="|Z| / Zt")
    if mp and mi:
        for k, nm in names.items():
            a = np.array([m[k] for m in mp]); c = np.array([m[k] for m in mi])
            print(f"  {ks_2samp(a, c).statistic:4.2f}  {nm:28s} PCB 중앙 {np.median(a):.3g} / IP 중앙 {np.median(c):.3g}")
        print(f"  모드 수: PCB {len(mp)}, IP {len(mi)}")

    print("\n=== A4 특징 하나로 만든 규칙 (prune 후 개수 / LS 선생님 개수, 중앙값) ===")
    for t in TECH:
        print(f"  [{t}]")
        for rn in RULES:
            n = np.array([r["rules"][rn][0] for r in R[t]], float); ok = np.array([r["rules"][rn][1] for r in R[t]])
            ls = np.array([r["ls"] for r in R[t]], float)
            print(f"    성공 {ok.mean():4.0%}  {np.nanmedian(n[ok] / ls[ok]) if ok.any() else np.nan:4.2f}  {rn}")
    print(f"\n총 {time.time()-t0:.0f}s, 판 {NB}개 x 2종류, {N}x{N}칸, F={len(F)}")


if __name__ == "__main__":
    main()
