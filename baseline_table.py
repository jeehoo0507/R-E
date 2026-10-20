"""Baseline table with exact success (stage 1 of the 3-day plan).

For each board size, the same instances are solved by every non-learning method. Each method runs
inside verify.cutting_plane, so a solution only counts as a success when the dense exact check
(700 log points + peak refinement) confirms it. The grid refined along the way is saved as a
problem bank (runs/bank_<size>.pkl) for the next experiments.

Question this table answers: under the exact check, is there a real gap between the fast rule
(greedy+prune) and the slow searches (local search, GA)? Without a gap a learned policy has
nothing to win on these instances.

Run: python baseline_table.py [--sizes 10,16] [--n 16] [--out reports/baseline_table.md]
"""
import argparse
import os
import time

import numpy as np
import torch

from decap_loop.bank import save_bank
from decap_loop.baselines import genetic, greedy, local_search, modal_heuristic, random_prune
from decap_loop.pdn import PDNConfig, sample_batch
from decap_loop.sim import FEAS_TOL
from decap_loop.verify import cutting_plane, grid_worst

TOL1 = np.log10(1.01)


def make_methods(cfg, state):
    def ls(b):
        g = greedy(b, cfg)
        occ = local_search(b, cfg, g["occ"], g["feasible"])
        inst = torch.arange(b.B, device=occ.device)
        return dict(occ=occ, feasible=torch.from_numpy(grid_worst(b, inst, occ, cfg) <= FEAS_TOL))
    return [  # greedy+prune first: it does most of the grid refinement
        ("greedy+prune", lambda b: greedy(b, cfg)),
        ("greedy", lambda b: greedy(b, cfg, prune=False)),
        ("modal_rule", lambda b: modal_heuristic(b, cfg)),
        ("random+prune(8)", lambda b: random_prune(b, cfg, 8)),
        ("local_search", ls),
        ("GA(32x40)", lambda b: genetic(b, cfg, pop=32, gens=40)),
    ]


def timed(fn, box):
    def run(b):
        t = time.time()
        r = fn(b)
        box["last"] = time.time() - t
        return r
    return run


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", default="10,16")
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--seed", type=int, default=100)
    ap.add_argument("--out", default="reports/baseline_table.md")
    a = ap.parse_args()
    cfg = PDNConfig(dense_max_nodes=144)   # growing grids: sparse bank above 12x12 (see h0_check.py)
    torch.manual_seed(a.seed)
    os.makedirs("runs", exist_ok=True)
    lines = ["# 기준선 표 (정확한 검사로 성공 판정)", "",
             f"설정: PDNConfig 기본값, 크기 {a.sizes}, 크기마다 {a.n}문제, seed {a.seed}, difficulty 0.02~0.2. "
             "성공 = 700점 + 피크 정밀화 검사에서 최악 |Z| ≤ 1.01·Zt. 모든 방법은 cutting-plane 라운드 안에서 실행.", ""]
    for size in map(int, a.sizes.split(",")):
        batch = sample_batch(a.n, size, size, cfg, np.random.default_rng(a.seed + size))
        res = {}
        for name, fn in make_methods(cfg, None):
            box = {}
            t0 = time.time()
            r, info = cutting_plane(timed(fn, box), batch, cfg, max_rounds=10)
            ok = info["worst"] <= TOL1
            cnt = (r["occ"] >= 0).sum(1).cpu().numpy()
            res[name] = dict(occ=r["occ"].cpu().numpy(), worst=info["worst"], ok=ok, count=cnt,
                             t_total=time.time() - t0, t_last=box.get("last", np.nan), rounds=info["rounds"])
            print(f"{size:3d} {name:16s} success {ok.mean():.2f} count {cnt[ok].mean() if ok.any() else np.nan:.2f} "
                  f"rounds {info['rounds']} last-run {box.get('last', 0):.1f}s total {time.time()-t0:.0f}s "
                  f"F={batch.freqs.shape[1]}", flush=True)
        save_bank(f"runs/bank_{size}.pkl", batch, res)
        # per-instance best-known among certified solutions
        names = list(res)
        C = np.stack([np.where(res[m]["ok"], res[m]["count"], np.inf) for m in names])   # [M,B]
        best = C.min(0)
        gp = C[names.index("greedy+prune")]
        lines += [f"## {size}×{size} (문제 {a.n}개, 최종 격자 F={batch.freqs.shape[1]})", "",
                  "| 방법 | 실제 성공 | 개수 (성공한 문제 평균) | greedy+prune 대비 | best-known 대비 초과 | best 달성 비율 | 마지막 실행 시간 (문제당 s) | 라운드 |",
                  "|---|---|---|---|---|---|---|---|"]
        for i, m in enumerate(names):
            ok = np.isfinite(C[i])
            both = ok & np.isfinite(gp)
            rel = np.mean(C[i][both] / gp[both]) if both.any() else np.nan
            okb = ok & np.isfinite(best)
            gap = np.mean((C[i][okb] - best[okb]) / best[okb]) * 100 if okb.any() else np.nan
            hit = np.mean(C[i][okb] == best[okb]) if okb.any() else np.nan
            lines.append(f"| {m} | {ok.mean():.2f} | {C[i][ok].mean() if ok.any() else np.nan:.2f} | {rel:.3f} | "
                         f"{gap:.1f}% | {hit:.2f} | {res[m]['t_last'] / a.n:.3f} | {res[m]['rounds']} |")
        solved = np.isfinite(best)
        lines += ["", f"best-known이 있는 문제 {solved.sum()}/{a.n}개. "
                  f"greedy+prune이 best를 낸 비율 {np.mean(gp[solved] == best[solved]):.2f}, "
                  f"greedy+prune과 best의 평균 차이 {np.mean((gp[solved & np.isfinite(gp)] - best[solved & np.isfinite(gp)])):.2f}개.", ""]
        open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
