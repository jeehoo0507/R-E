"""Certified evaluation of trained policies vs classical baselines on the test pools.

Every method runs inside verify.cutting_plane (same rule for all: re-run on a grown grid until its
claimed solutions are certified or the round cap is hit), and success is the exact dense check.

Run: python eval_mott.py --runs runs/mott,runs/grid_prior [--pool runs/pool] [--out reports/mott_result.md]
"""
import argparse
import os
import time

import numpy as np
import torch

from decap_loop.bank import load_bank
from decap_loop.baselines import greedy, local_search, modal_heuristic
from decap_loop.rollout import rollout
from decap_loop.sim import FEAS_TOL, Env
from decap_loop.verify import cutting_plane, grid_worst
from train import make_policy

TOL1 = np.log10(1.01)


def load_policy(run, which):
    ck = torch.load(os.path.join(run, which), map_location="cpu", weights_only=False)
    a = ck["args"]
    from decap_loop.pdn import PDNConfig
    pol = make_policy(PDNConfig(), a)
    pol.load_state_dict(ck["model"])
    pol.eval()
    return pol, a, ck.get("iter")


def policy_method(pol, a, samples):
    @torch.no_grad()
    def run(b):
        cfg = b.cfg
        if samples <= 1:
            env = Env(b, torch.arange(b.B), cfg, feat_mode=a["feat"])
            out = rollout(pol, env, greedy=True, amp=False)
            return dict(occ=out["occ"], feasible=out["feasible"])
        inst = torch.arange(b.B).repeat_interleave(samples)
        env = Env(b, inst, cfg, feat_mode=a["feat"])
        out = rollout(pol, env, amp=False, gen=torch.Generator().manual_seed(0))
        key = (out["count"] + (~out["feasible"]) * 10 ** 6).view(b.B, samples)
        j = key.argmin(1) + torch.arange(b.B) * samples
        return dict(occ=out["occ"][j], feasible=out["feasible"][j])
    return run


def ls_method(b):
    g = greedy(b, b.cfg)
    occ = local_search(b, b.cfg, g["occ"], g["feasible"], passes=2, trials=16)
    inst = torch.arange(b.B)
    return dict(occ=occ, feasible=torch.from_numpy(grid_worst(b, inst, occ, b.cfg) <= FEAS_TOL))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/mott,runs/grid_prior")
    ap.add_argument("--ckpt", default="best.pt")
    ap.add_argument("--pool", default="runs/pool")
    ap.add_argument("--tests", default="test_pcb10_0,test_pcb16_0,test_ip10_0,test_ip16_0")
    ap.add_argument("--samples", type=int, default=16)
    ap.add_argument("--rounds", type=int, default=4)
    ap.add_argument("--no_ls", action="store_true")
    ap.add_argument("--out", default="reports/mott_result.md")
    a = ap.parse_args()
    torch.manual_seed(0)
    methods = [("greedy+prune", lambda b: greedy(b, b.cfg)), ("modal_rule", lambda b: modal_heuristic(b, b.cfg))]
    if not a.no_ls:
        methods.append(("local_search", ls_method))
    for run in a.runs.split(","):
        pol, pa, it = load_policy(run, a.ckpt)
        name = os.path.basename(run.rstrip("/"))
        methods.append((f"{name} (greedy, it {it})", policy_method(pol, pa, 1)))
        methods.append((f"{name} (best of {a.samples})", policy_method(pol, pa, a.samples)))
    lines = ["# MoTT 평가 (정밀 검사 기준)", "",
             f"체크포인트 {a.ckpt}. 모든 방법은 cutting-plane(최대 {a.rounds}라운드) 안에서 실행. "
             "성공 = 700점 + 피크 정밀화 검사에서 최악 |Z| ≤ 1.01·Zt.", ""]
    for test in a.tests.split(","):
        path = os.path.join(a.pool, test + ".pkl")
        if not os.path.exists(path):
            print("skip (missing)", test)
            continue
        res = {}
        for mname, fn in methods:
            b, _ = load_bank(path)           # fresh copy of the grid for each method
            t = time.time()
            r, info = cutting_plane(fn, b, b.cfg, max_rounds=a.rounds)
            ok = info["worst"] <= TOL1
            res[mname] = dict(ok=ok, count=(r["occ"] >= 0).sum(1).cpu().numpy(), t=(time.time() - t) / b.B)
            print(f"{test:14s} {mname:32s} success {ok.mean():.2f} count {res[mname]['count'][ok].mean() if ok.any() else np.nan:.2f} "
                  f"{res[mname]['t']:.2f}s/inst", flush=True)
        names = list(res)
        C = np.stack([np.where(res[m]["ok"], res[m]["count"], np.inf) for m in names])
        gp = C[0]
        best = C.min(0)
        lines += [f"## {test} ({b.cfg.cell_size * 1e6:.0f} µm 셀, {b.H}×{b.W}, 문제 {b.B}개)", "",
                  "| 방법 | 실제 성공 | 개수 (성공 평균) | greedy+prune 대비 (둘 다 성공) | best 달성 | 문제당 시간 (s) |",
                  "|---|---|---|---|---|---|"]
        for i, m in enumerate(names):
            ok = np.isfinite(C[i])
            both = ok & np.isfinite(gp)
            rel = np.mean(C[i][both] / gp[both]) if both.any() else np.nan
            hit = np.mean(C[i][np.isfinite(best)] == best[np.isfinite(best)]) if np.isfinite(best).any() else np.nan
            lines.append(f"| {m} | {ok.mean():.2f} | {C[i][ok].mean() if ok.any() else np.nan:.2f} | {rel:.3f} | {hit:.2f} | {res[m]['t']:.2f} |")
        lines.append("")
        open(a.out, "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
