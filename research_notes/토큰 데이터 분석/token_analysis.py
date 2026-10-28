"""Training-free analysis of candidate token features (PCB vs interposer-type boards).

A2 invariance : KS distance of each feature between PCB and interposer boards (raw vs normalised)
A3 relevance  : AUC of each single feature for "port is in the teacher (greedy+prune) solution",
                direction fixed on PCB and re-used on interposer (transfer consistency)
A4 rules      : greedy placement driven by one feature (coord / loop-L / fixed-freq / modal) vs the
                exact greedy, on both technologies. Same type rule for all (SRF nearest to the worst
                violation frequency), then exact prune. Success is checked on the dense 700-pt grid.

Simulator: research_notes/모달 토큰 검증/modal_check.py (dense Z = Y^-1 per frequency, small boards).
Usage: python token_analysis.py [n_boards_per_tech] [N]
"""
import os, sys, time
import numpy as np
from scipy.signal import find_peaks
from scipy.stats import ks_2samp

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "모달 토큰 검증"))
sys.path.insert(0, os.path.join(HERE, "..", ".."))
from modal_check import F, Y_of, PDNConfig          # noqa: E402
from decap_loop.pdn import sample_mask              # noqa: E402

NB = int(sys.argv[1]) if len(sys.argv) > 1 else 10
N = int(sys.argv[2]) if len(sys.argv) > 2 else 10
S = 2j * np.pi * F
TECH = {
    # band = frequencies where the target is checked; decaps follow each technology
    "pcb": dict(cfg=PDNConfig(), band=(10e6, 5e9)),
    "ip": dict(cfg=PDNConfig(cell_size=200e-6, diel_thickness=1e-6, metal_thickness=1e-6, vrm_l=50e-12,
                             decaps=((100e-12, 5e-12, 0.05), (20e-12, 2e-12, 0.1), (5e-12, 1e-12, 0.2))),
               band=(200e6, 20e9)),
}
M_FIX = 8          # fixed log-spaced frequency samples (R4 / Manoharan-style dZ)


# ------------------------------------------------------------------ board + exact physics
def make_board(cfg, band, rng):
    shape = rng.choice(["rect", "L", "holes"])
    mask = sample_mask(N, N, shape, rng)
    valid = np.flatnonzero(mask.reshape(-1))
    entries = rng.choice(valid, rng.integers(1, 3), replace=False)
    probe = int(rng.choice(np.setdiff1d(valid, entries)))
    cand = np.setdiff1d(valid, np.r_[entries, probe])
    Z = np.stack([np.linalg.inv(Y_of(mask, entries, cfg, f)) for f in F])          # [F, n, n]
    zd = np.stack([esr + S * esl + 1 / (S * c) for c, esl, esr in cfg.decaps], 1)  # [F, T]
    inb = (F >= band[0]) & (F <= band[1])
    fc = np.sqrt(band[0] * band[1])
    # target: feasible by construction (alpha x best "all ports filled with one type" worst |Z|)
    best = np.inf
    for t in range(zd.shape[1]):
        Zl = load(Z, cand, np.full(len(cand), t), zd)
        best = min(best, np.abs(Zl[inb, probe, probe]).max())
    Zt = best * rng.uniform(1.5, 3.0)
    return dict(mask=mask, entries=entries, probe=probe, cand=cand, Z=Z, zd=zd, inb=inb, fc=fc, Zt=Zt,
                shape=shape)


def load(Z, ports, types, zd):
    """Exact Z with shunt decaps at ports (Schur complement)."""
    if len(ports) == 0:
        return Z
    P = np.asarray(ports)
    A = Z[:, P][:, :, P] + np.einsum("fk,kl->fkl", zd[:, types], np.eye(len(P)))
    return Z - Z[:, :, P] @ np.linalg.solve(A, Z[:, P, :])


def rank1(Z, i, z):
    """Woodbury: add one shunt decap of impedance z[F] at port i."""
    col = Z[:, :, i]
    return Z - col[:, :, None] * col[:, None, :] / (Z[:, i, i] + z)[:, None, None]


def worst(Zl, b):
    return (np.abs(Zl[b["inb"], b["probe"], b["probe"]]) / b["Zt"]).max()


def onestep(Zl, b, ports):
    """Exact worst ratio after adding (port, type) for all ports x types: [P, T]."""
    p = b["probe"]; zp = Zl[:, p, p][:, None, None]; zpi = Zl[:, p, ports][:, :, None]
    zii = Zl[:, ports, ports][:, :, None]
    new = np.abs(zp - zpi ** 2 / (zii + b["zd"][:, None, :]))[b["inb"]] / b["Zt"]
    return new.max(0)


def prune(b, sol):
    sol = list(sol); changed = True
    while changed:
        changed = False
        for k in range(len(sol)):
            trial = sol[:k] + sol[k + 1:]
            Zl = load(b["Z"], [i for i, _ in trial], np.array([t for _, t in trial], int), b["zd"])
            if worst(Zl, b) <= 1:
                sol = trial; changed = True; break
    return sol


def run_policy(b, choose, cap=60):
    Zl, sol = b["Z"], []
    for _ in range(cap):
        if worst(Zl, b) <= 1:
            break
        free = np.setdiff1d(b["cand"], [i for i, _ in sol])
        i, t = choose(Zl, b, free)
        sol.append((int(i), int(t)))
        Zl = rank1(Zl, i, b["zd"][:, t])
    ok = worst(Zl, b) <= 1
    return (prune(b, sol) if ok else sol), ok


# ------------------------------------------------------------------ features
def modes(Zl, b):
    z = np.abs(Zl[:, b["probe"], b["probe"]]); lz = np.log(z)
    pk, _ = find_peaks(lz, prominence=0.3)
    out = []
    for k in pk:
        if not b["inb"][k]:
            continue
        half = z[k] / np.sqrt(2); l = r = k
        while l > 0 and z[l] > half: l -= 1
        while r < len(F) - 1 and z[r] > half: r += 1
        Q = F[k] / max(F[r] - F[l], 1e-9)
        out.append((k, F[k], Q, z[k]))
    return out


def basis(poles):
    cols = []
    for p in poles:
        cols += [1 / (S - p) + 1 / (S - np.conj(p)), 1j / (S - p) - 1j / (S - np.conj(p))]
    cols += [np.ones_like(S), S]
    return np.stack(cols, 1)


def fit(A, Zcols):
    """Relative-weighted LS per column; returns complex residues [K, n], d [n], e [n]."""
    K = (A.shape[1] - 2) // 2
    R = np.zeros((K, Zcols.shape[1]), complex); d = np.zeros(Zcols.shape[1]); e = np.zeros(Zcols.shape[1])
    for j in range(Zcols.shape[1]):
        z = Zcols[:, j]; wt = 1 / np.maximum(np.abs(z), 1e-12)
        Mx = np.vstack([(A * wt[:, None]).real, (A * wt[:, None]).imag]); sc = np.linalg.norm(Mx, axis=0) + 1e-30
        c, *_ = np.linalg.lstsq(Mx / sc, np.r_[(z * wt).real, (z * wt).imag], rcond=None); c = c / sc
        R[:, j] = c[0:2 * K:2] + 1j * c[1:2 * K:2]; d[j] = c[-2]; e[j] = c[-1]
    return R, d, e


def modal_features(Zl, b, ports):
    """Coupling of each port to the worst in-band mode (+ residual L, R). None if no mode."""
    ms = modes(Zl, b)
    if not ms:
        return None
    k_star = int(np.argmax([z / b["Zt"] for *_, z in ms]))
    poles = [-np.pi * f / Q + 2j * np.pi * f for _, f, Q, _ in ms]
    A = basis(poles); p = b["probe"]
    Rp, _, _ = fit(A, Zl[:, p, [p]])
    Rpi, dpi, epi = fit(A, Zl[:, p, ports])
    Rii, dii, eii = fit(A, Zl[:, ports, ports])
    rpp = np.abs(Rp[k_star, 0]); rpi = np.abs(Rpi[k_star]); rii = np.abs(Rii[k_star])
    g = rpi / np.sqrt(rpp * rii + 1e-30)
    return dict(g=g, rpi_n=rpi / b["Zt"] / (2 * np.pi * ms[k_star][1]),
                eii_n=2 * np.pi * b["fc"] * eii / b["Zt"], dii_n=dii / b["Zt"],
                f_worst=ms[k_star][1], modes=ms)


def loop_L(Zl, b, ports):
    """Manoharan-style loop inductance at one mid-band f0 (port shorted, seen from probe)."""
    k0 = int(np.argmin(np.abs(np.log(F / b["fc"]))))
    p = b["probe"]
    zs = Zl[k0, p, p] - Zl[k0, p, ports] ** 2 / Zl[k0, ports, ports]
    return zs.imag / (2 * np.pi * F[k0])


def fixed_idx(b):
    idx = np.flatnonzero(b["inb"])
    return idx[np.linspace(0, len(idx) - 1, M_FIX).round().astype(int)]


def port_features(b):
    Z, p, P = b["Z"], b["probe"], b["cand"]
    H, W = b["mask"].shape
    py, px = divmod(p, W); y, x = np.divmod(P, W)
    fi = fixed_idx(b)
    worst_s = fi[np.argmax(np.abs(Z[fi, p, p]))]
    L = loop_L(Z, b, P)
    mf = modal_features(Z, b, P)
    gain = onestep(Z, b, P).min(1)
    feats = {
        "coord: probe와 거리 (칸)": np.hypot(y - py, x - px),
        "coord: probe와 거리 (판 크기 비율)": np.hypot(y - py, x - px) / max(H, W),
        "loopL (H, 원값)": L,
        "loopL 정규화 2πf_c·L/Zt": 2 * np.pi * b["fc"] * L / b["Zt"],
        "fixed: |Z_pi| 최악 샘플 (Ω, 원값)": np.abs(Z[worst_s, p, P]),
        "fixed: |Z_pi|/Zt 최악 샘플": np.abs(Z[worst_s, p, P]) / b["Zt"],
        "oracle: 정확한 1수 후 최악 비율": gain,
    }
    if mf is not None:
        feats["modal: 결합도 g (최악 모드)"] = mf["g"]
        feats["modal: 잔여 L 정규화"] = mf["eii_n"]
    return feats, mf


# ------------------------------------------------------------------ policies (port rule + common type rule)
def type_rule(Zl, b, f_target):
    srf = np.array([1 / (2 * np.pi * np.sqrt(esl * c)) for c, esl, _ in b["cfg"].decaps])
    return int(np.argmin(np.abs(np.log(srf / f_target))))


def dense_worst_f(Zl, b):
    z = np.abs(Zl[:, b["probe"], b["probe"]]) * b["inb"]
    return F[int(np.argmax(z))]


def pol_greedy(Zl, b, free):
    sc = onestep(Zl, b, free); i, t = np.unravel_index(np.argmin(sc), sc.shape)
    return free[i], t


def pol_coord(Zl, b, free):
    W = b["mask"].shape[1]; py, px = divmod(b["probe"], W); y, x = np.divmod(free, W)
    return free[np.argmin(np.hypot(y - py, x - px))], type_rule(Zl, b, dense_worst_f(Zl, b))


def pol_loopL(Zl, b, free):
    return free[np.argmin(loop_L(Zl, b, free))], type_rule(Zl, b, dense_worst_f(Zl, b))


def pol_fixed(Zl, b, free):
    fi = fixed_idx(b); p = b["probe"]
    k = fi[np.argmax(np.abs(Zl[fi, p, p]))]
    return free[np.argmax(np.abs(Zl[k, p, free]))], type_rule(Zl, b, F[k])


def pol_modal(Zl, b, free):
    mf = modal_features(Zl, b, free)
    if mf is None:
        return pol_fixed(Zl, b, free)
    score = mf["g"] * np.abs(mf["rpi_n"])
    return free[np.argmax(score)], type_rule(Zl, b, mf["f_worst"])


POLICIES = {"greedy (정확한 1수)": pol_greedy, "coord (probe에 가까운 순)": pol_coord,
            "loopL (인덕턴스 작은 순)": pol_loopL, "fixed (고정 8점 최악 주파수)": pol_fixed,
            "modal (최악 모드 결합도)": pol_modal}


# ------------------------------------------------------------------ main
def auc(score, label):
    pos, neg = score[label], score[~label]
    if len(pos) == 0 or len(neg) == 0:
        return np.nan
    return (pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean()


def main():
    rng = np.random.default_rng(0)
    rows = {k: [] for k in TECH}; feat_pool = {k: {} for k in TECH}; mode_pool = {k: [] for k in TECH}
    pol_res = {k: {p: [] for p in POLICIES} for k in TECH}
    t0 = time.time()
    for tech, spec in TECH.items():
        for nb in range(NB):
            b = make_board(spec["cfg"], spec["band"], rng); b["cfg"] = spec["cfg"]
            feats, mf = port_features(b)
            res = {}
            for name, pol in POLICIES.items():
                sol, ok = run_policy(b, pol)
                res[name] = (len(sol), ok)
            pol_res_t = pol_res[tech]
            for name in POLICIES:
                pol_res_t[name].append(res[name])
            teacher = {i for i, _ in run_policy(b, pol_greedy)[0]}
            label = np.isin(b["cand"], list(teacher))
            rows[tech].append((feats, label))
            for k, v in feats.items():
                feat_pool[tech].setdefault(k, []).append(v)
            if mf is not None:
                for _, f, Q, z in mf["modes"]:
                    mode_pool[tech].append((f, f / b["fc"], Q, z / b["Zt"], z))
            print(f"[{tech} {nb+1}/{NB}] {b['shape']:5s} teacher={len(teacher):2d} "
                  f"K={0 if mf is None else len(mf['modes'])} t={time.time()-t0:.0f}s", flush=True)

    print("\n=== A2 종류 불변성: PCB vs 인터포저형 분포 차이 (KS 거리, 0=같음, 1=완전히 다름) ===")
    for k in feat_pool["pcb"]:
        if k in feat_pool["ip"]:
            a = np.concatenate(feat_pool["pcb"][k]); c = np.concatenate(feat_pool["ip"][k])
            print(f"  {ks_2samp(a, c).statistic:5.2f}  {k}")
    mp, mi = np.array(mode_pool["pcb"]), np.array(mode_pool["ip"])
    if len(mp) and len(mi):
        for j, name in enumerate(["모드 주파수 f (Hz, 원값)", "모드 주파수 f/f_c (정규화)", "모드 Q",
                                  "모드 크기 |Z|/Zt (정규화)", "모드 크기 |Z| (Ω, 원값)"]):
            print(f"  {ks_2samp(mp[:, j], mi[:, j]).statistic:5.2f}  {name}")
        print(f"  모드 수/판: PCB {len(mp)/NB:.1f}, 인터포저형 {len(mi)/NB:.1f}")

    print("\n=== A3 선생님 해 관련성: 특징 하나로 '선생님 해에 들어가는 포트' 맞히기 (AUC) ===")
    print("    방향은 PCB에서 정하고 인터포저형에 그대로 적용. 0.5 = 무작위")
    for k in feat_pool["pcb"]:
        if k not in feat_pool["ip"]:
            continue
        ap = np.nanmean([auc(f[k], l) for f, l in rows["pcb"] if k in f])
        sign = 1 if ap >= 0.5 else -1
        ai = np.nanmean([auc(sign * f[k], l) for f, l in rows["ip"] if k in f])
        ap2 = ap if sign == 1 else 1 - ap
        print(f"  PCB {ap2:.2f} → IP {ai:.2f}  {k}")

    print("\n=== A4 특징 하나로 만든 규칙 배치 (prune 후, 성공 = 700점 검사) ===")
    for tech in TECH:
        g = np.array([n for n, ok in pol_res[tech]["greedy (정확한 1수)"]], float)
        print(f"  [{'PCB' if tech == 'pcb' else '인터포저형'}]")
        for name in POLICIES:
            r = pol_res[tech][name]; n = np.array([x for x, _ in r], float); ok = np.array([o for _, o in r])
            ratio = np.median(n[ok] / g[ok]) if ok.any() else np.nan
            print(f"    성공 {ok.mean():4.0%}  개수/greedy 중앙값 {ratio:4.2f}  {name}")
    print(f"\n총 {time.time()-t0:.0f}s, 판 {NB}개 x 2종류, {N}x{N}칸")


if __name__ == "__main__":
    main()
