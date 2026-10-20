"""H0: can the environment's frequency grid be trusted to decide "target met"?

Same instances under three working grids:
  A  base     : cfg.n_freq log points only (the old environment)
  B  seeded   : A + peak frequencies of the bare and filled boards
  C  adaptive : B + cutting plane (certify, append violated frequencies, re-run)
Every final solution is certified on the dense exact check (verify.exact_worst: cfg.n_check log
points + golden-section refinement of the top peaks, decaps placed directly in Y).

H0 passes if, on the adaptive grid, (i) no claimed-feasible solution violates the target by more
than 1% and (ii) for >= 99% of claimed solutions the certified worst |Z| is within 5% of the
grid's worst |Z|.

Run: python h0_check.py [--sizes 10,16,24] [--n 16] [--out reports/h0_result.md]
"""
import argparse
import dataclasses
import time

import numpy as np
import torch

from decap_loop.baselines import greedy, local_search
from decap_loop.pdn import PDNConfig, sample_batch
from decap_loop.sim import FEAS_TOL
from decap_loop.verify import certify, cutting_plane, grid_worst

TOL1 = np.log10(1.01)   # "violates by more than 1%"
TOL5 = np.log10(1.05)


def methods(cfg):
    def gp(batch):
        return greedy(batch, cfg)

    def ls(batch):
        r = greedy(batch, cfg)
        occ = local_search(batch, cfg, r["occ"], r["feasible"])
        inst = torch.arange(batch.B, device=occ.device)
        feas = torch.from_numpy(grid_worst(batch, inst, occ, cfg) <= FEAS_TOL)
        return dict(occ=occ, feasible=feas, count=(occ >= 0).sum(1))
    return {"greedy+prune": gp, "local_search": ls}


def run(cond, cfg_base, size, n, seed, mname, mfun_of):
    cfg = dataclasses.replace(cfg_base, seed_peaks=(cond != "A"))
    t0 = time.time()
    batch = sample_batch(n, size, size, cfg, np.random.default_rng(seed))
    t_setup = time.time() - t0
    mfun = mfun_of(cfg)[mname]
    inst = torch.arange(batch.B)
    t0 = time.time()
    if cond == "C":
        res, info = cutting_plane(mfun, batch, cfg)
        worst, grid, rounds = info["worst"], info["grid"], info["rounds"]
    else:
        res = mfun(batch)
        worst, _, _ = certify(batch, inst, res["occ"], cfg)
        grid, rounds = grid_worst(batch, inst, res["occ"], cfg), 0
    t_run = time.time() - t0
    claimed = res["feasible"].cpu().numpy().astype(bool)
    cnt = (res["occ"] >= 0).sum(1).cpu().numpy()
    return dict(cond=cond, size=size, method=mname, n=n, claimed=claimed, worst=worst, grid=grid,
                count=cnt, F=batch.freqs.shape[1], rounds=rounds, t_setup=t_setup, t_run=t_run)


def summarize(r):
    c = r["claimed"]
    w, g = r["worst"][c], r["grid"][c]
    ff1 = float(np.mean(w > TOL1)) if c.any() else float("nan")
    gap = w - g                                   # certified minus grid worst (log10), >= 0
    within5 = float(np.mean(gap <= TOL5)) if c.any() else float("nan")
    true_ok = (r["worst"] <= TOL1)
    return dict(claimed=c.mean(), ff1=ff1, within5=within5, ratio_med=10 ** np.median(gap) if c.any() else np.nan,
                ratio_max=10 ** gap.max() if c.any() else np.nan, true_success=true_ok.mean(),
                count=r["count"][c & true_ok].mean() if (c & true_ok).any() else np.nan,
                F=r["F"], rounds=r["rounds"], t=r["t_setup"] + r["t_run"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10,16,24")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--methods", default="greedy+prune,local_search")
    ap.add_argument("--out", default="reports/h0_result.md")
    a = ap.parse_args()
    # the dense bank stores the full Z0 per frequency: with a growing grid that is B*F*n^2 complex
    # numbers, so boards above 144 cells use the sparse bank (same exact numbers, columns on demand)
    cfg = PDNConfig(dense_max_nodes=144)
    torch.manual_seed(a.seed)
    rows = []
    for size in map(int, a.sizes.split(",")):
        for mname in a.methods.split(","):
            for cond in ("A", "B", "C"):
                r = run(cond, cfg, size, a.n, a.seed + size, mname, methods)
                s = summarize(r)
                rows.append((size, mname, cond, s, r))
                print(f"{size:3d} {mname:13s} {cond}: F={s['F']:3d} claimed={s['claimed']:.2f} "
                      f"false>1%={s['ff1']:.2f} within5%={s['within5']:.2f} cert/grid med={s['ratio_med']:.2f} "
                      f"max={s['ratio_max']:.2f} true_success={s['true_success']:.2f} count={s['count']:.1f} "
                      f"rounds={s['rounds']} t={s['t']:.1f}s", flush=True)
    lines = ["# H0 결과: 검사 격자를 믿을 수 있는가", "",
             f"설정: PDNConfig 기본값, 크기 {a.sizes}, 크기마다 {a.n}문제, seed {a.seed}. "
             "인증 = 700점 log 격자 + 상위 피크 golden-section 정밀화, decap을 Y에 직접 넣어 푼 값.", "",
             "| 크기 | 방법 | 격자 | F | 주장 성공 | 거짓 성공(>1%) | 차이 5% 이내 | 인증/격자 최악 (중앙, 최대) | 실제 성공 | 개수 | 라운드 | 시간(s) |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for size, mname, cond, s, _ in rows:
        lines.append(f"| {size} | {mname} | {cond} | {s['F']} | {s['claimed']:.2f} | {s['ff1']:.2f} | {s['within5']:.2f} | "
                     f"{s['ratio_med']:.2f}, {s['ratio_max']:.2f} | {s['true_success']:.2f} | {s['count']:.1f} | {s['rounds']} | {s['t']:.1f} |")
    allC = [r for size, mname, cond, s, r in rows if cond == "C"]
    w = np.concatenate([r["worst"][r["claimed"]] for r in allC])
    g = np.concatenate([r["grid"][r["claimed"]] for r in allC])
    ff = float(np.mean(w > TOL1)); w5 = float(np.mean(w - g <= TOL5))
    verdict = "통과" if ff == 0 and w5 >= 0.99 else "실패"
    lines += ["", f"**H0 (격자 C, 주장 성공 해 {len(w)}개): 거짓 성공 {ff:.3f}, 차이 5% 이내 {w5:.3f} → {verdict}**"]
    open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines[-1:]))


if __name__ == "__main__":
    main()
