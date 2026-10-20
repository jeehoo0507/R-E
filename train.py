"""Train the add / remove / change-type / stop policy (min decap count s.t. target impedance).

Loss = REINFORCE with a shared multi-sample baseline (POMO style, M samples per instance)
     + imitation of the best-known solution per instance (best sample, or the classical teacher
       greedy+prune / local search, whichever has fewer decaps).
Reward = -count / N_ref - lambda * (failed + violation), N_ref = greedy+prune count of the same
instance. lambda is a Lagrange multiplier pushed up while the failure rate is above --fail_target.

Example (A5000):
  python train.py --out runs/gnn_phys                      # main model
  python train.py --out runs/gnn_coord --feat coord        # coordinate-input baseline (H1')
  python train.py --out runs/gnn_k3 --layers 3             # message-passing depth (H2')
  python train.py --out runs/gnn_noprune --no_prune        # add + stop only (H3')
"""
import argparse
import csv
import json
import os
import time
from dataclasses import asdict

import numpy as np
import torch

from decap_loop.baselines import greedy, local_search, trajectory_to
from decap_loop.bank import load_bank
from decap_loop.model import MoTT, GridPolicy
from decap_loop.pdn import PDNConfig, sample_batch
from decap_loop.rollout import backward_logp, rollout
from decap_loop.sim import K_MODES, Env, feat_dim, mott_dims
from decap_loop.verify import certify


def rng_range(s, typ=int):
    a, b = s.split("-") if "-" in s else (s, s)
    return typ(a), typ(b)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--out", default="runs/gnn_phys")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=5000)
    p.add_argument("--B", type=int, default=16, help="instances per batch")
    p.add_argument("--M", type=int, default=8, help="samples per instance (shared baseline)")
    p.add_argument("--sizes", default="8-12", help="train board side range (H, W sampled independently)")
    p.add_argument("--shapes", default="rect,L,holes")
    p.add_argument("--targets", default="flat,step")
    p.add_argument("--n_probe", default="1-3")
    p.add_argument("--difficulty", default="0.02-0.2", help="target level u (0 = hardest)")
    p.add_argument("--max_steps", type=int, default=64)
    p.add_argument("--feat", choices=["phys", "coord", "mott"], default="phys")
    p.add_argument("--model", choices=["grid", "mott"], default="grid", help="mott implies --feat mott")
    p.add_argument("--pool", default="", help="directory of certified pools (build_pool.py); train_*.pkl / test_*.pkl")
    p.add_argument("--val_pools", default="test_pcb10_0,test_ip10_0")
    p.add_argument("--pool_glob", default="train_*.pkl", help="comma-separated globs of training pool files (memory: "
                   "each 12x12 batch holds ~0.6 GB of Z0 in complex64)")
    p.add_argument("--hours", type=float, default=0, help="stop after this wall time (0 = off)")
    p.add_argument("--certify", type=int, default=1, help="pools: exact-check the best sample of each instance every "
                   "iteration; a violation counts as failure and its frequencies are added to the pool grid")
    p.add_argument("--prior", action="store_true", help="grid model: add the physics prior (as in MoTT)")
    p.add_argument("--no_prune", action="store_true", help="ablation: no remove / change-type actions")
    p.add_argument("--layers", type=int, default=6)
    p.add_argument("--glob", action="store_true", help="add global self-attention to every block")
    p.add_argument("--d", type=int, default=128)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--imit", type=float, default=1.0, help="weight of imitating the best-known solution (0=off)")
    p.add_argument("--teacher", choices=["gp", "ls"], default="gp", help="greedy+prune, or + local search (slower)")
    p.add_argument("--lam", type=float, default=1.0, help="initial Lagrange multiplier")
    p.add_argument("--lam_lr", type=float, default=0.05)
    p.add_argument("--fail_target", type=float, default=0.02)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--no_amp", action="store_true")
    p.add_argument("--grad_ckpt", action="store_true")
    p.add_argument("--val_every", type=int, default=250)
    p.add_argument("--val_sizes", default="10,20")
    p.add_argument("--val_n", type=int, default=16)
    return p.parse_args()


def make_policy(cfg, a):
    if a.get("model") == "mott":
        return MoTT(cfg.n_types, K_MODES, mott_dims(cfg.n_types), d=a["d"], layers=a["layers"], heads=a["heads"])
    return GridPolicy(feat_dim(cfg, a["feat"]), cfg.n_types, d=a["d"], layers=a["layers"],
                      heads=a["heads"], glob=a["glob"], prior=a.get("prior", False))


def batch_kw(args):
    return dict(shapes=tuple(args.shapes.split(",")), targets=tuple(args.targets.split(",")),
                n_probe=rng_range(args.n_probe), difficulty=rng_range(args.difficulty, float))


@torch.no_grad()
def validate(policy, val_sets, args, cfg, dev):
    policy.eval()
    res = {}
    for size, (vb, ref) in val_sets.items():
        env = Env(vb, torch.arange(vb.B, device=dev), vb.cfg or cfg, allow_prune=not args.no_prune, feat_mode=args.feat)
        out = rollout(policy, env, greedy=True, amp=not args.no_amp)
        feas = out["feasible"].clone()
        if args.pool:   # certified validation: a solution that violates the exact check is a failure
            wv, _, _ = certify(vb, torch.arange(vb.B, device=dev), out["occ"], vb.cfg)
            feas &= torch.from_numpy(wv <= np.log10(1.01)).to(dev)
            out["feasible"] = feas
        ok = feas & ref["feasible"]
        ratio = (out["count"][ok].float() / ref["count"][ok].clamp(min=1).float()).mean().item() if ok.any() else float("nan")
        res[size] = (ratio, 1 - out["feasible"].float().mean().item())
    policy.train()
    return res


def main():
    args = parse()
    if args.model == "mott":
        args.feat = "mott"
    os.makedirs(args.out, exist_ok=True)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device(args.device)
    cfg = PDNConfig()
    smin, smax = rng_range(args.sizes)
    kw = batch_kw(args)
    amp = not args.no_amp

    policy = make_policy(cfg, vars(args)).to(dev)
    policy.grad_ckpt = args.grad_ckpt
    opt = torch.optim.AdamW(policy.parameters(), lr=args.lr, weight_decay=1e-5)
    print(f"params: {sum(p.numel() for p in policy.parameters())/1e6:.2f}M  device: {dev}")
    meta = dict(args=vars(args), cfg=asdict(cfg))
    json.dump(meta, open(os.path.join(args.out, "config.json"), "w"), indent=2)

    vrng = np.random.default_rng(12345)  # same val sets for every run
    val_sets, pool = {}, []
    if args.pool:
        import glob
        files = sorted({f for g in args.pool_glob.split(",") for f in glob.glob(os.path.join(args.pool, g))})
        for f in files:
            pb, _ = load_bank(f, dev, dense=True, z_dtype=torch.complex64)
            pool.append(pb)
        for name in args.val_pools.split(","):
            vb, _ = load_bank(os.path.join(args.pool, name + ".pkl"), dev)
            val_sets[name] = (vb, greedy(vb, vb.cfg))
        print(f"pool: {len(pool)} train batches, {sum(p.B for p in pool)} instances", flush=True)
    else:
        for s in map(int, args.val_sizes.split(",")):
            vb = sample_batch(args.val_n, s, s, cfg, vrng, device=dev, **kw)
            val_sets[s] = (vb, greedy(vb, cfg))

    log = open(os.path.join(args.out, "log.csv"), "w", newline="")
    w = csv.writer(log)
    w.writerow(["iter", "ratio", "best_ratio", "fail", "lam", "loss", "sec"] +
               [f"val{s}_{m}" for s in val_sets for m in ("ratio", "fail")])
    best_val, lam, t0 = 1e9, args.lam, time.time()
    for it in range(1, args.iters + 1):
        if args.hours and time.time() - t0 > args.hours * 3600:
            print(f"time limit reached at iter {it}", flush=True)
            break
        if pool:
            batch = pool[int(rng.integers(len(pool)))]
            H, W = batch.H, batch.W
            cfg = batch.cfg
            ib = torch.from_numpy(rng.permutation(batch.B)[:args.B]).to(dev)
        else:
            H, W = (int(x) for x in rng.integers(smin, smax + 1, size=2))
            batch = sample_batch(args.B, H, W, cfg, rng, device=dev, **kw)
            ib = torch.arange(args.B, device=dev)
        teach = greedy(batch, cfg, inst=ib, max_steps=args.max_steps)
        n_ref = teach["count"].clamp(min=1).float()
        if args.no_prune:  # the teacher may only use the actions the policy has
            teach = greedy(batch, cfg, prune=False, inst=ib, max_steps=args.max_steps)
        elif args.teacher == "ls":
            assert not pool, "--teacher ls is not wired for pools"
            occ = local_search(batch, cfg, teach["occ"], teach["feasible"], passes=2, trials=16, seed=it)
            teach = trajectory_to(batch, cfg, ib, occ, max_steps=args.max_steps)

        inst = ib.repeat_interleave(args.M)
        env = Env(batch, inst, cfg, max_steps=args.max_steps, allow_prune=not args.no_prune, feat_mode=args.feat)
        out = rollout(policy, env, store=True, amp=amp)
        cnt, fail = out["count"].float().view(args.B, args.M), (~out["feasible"]).view(args.B, args.M).clone()
        n_bad = n_add = 0
        if pool and args.certify:
            # the reward is computed on the working grid; without this the policy learns to prune into
            # resonance peaks that fall between grid points (certified success of untrained policies ~50%)
            j = (cnt + fail * 1e6).argmin(1)
            sel = torch.nonzero(~fail[torch.arange(args.B, device=dev), j]).squeeze(-1)
            if sel.numel():
                e_idx = sel * args.M + j[sel]
                wv, _, add = certify(batch, ib[sel], out["occ"][e_idx], cfg)
                bad = torch.from_numpy(wv > 1e-3).to(dev)
                fail[sel[bad], j[sel[bad]]] = True
                n_bad = int(bad.sum())
                if n_bad:
                    n_add = batch.add_freqs(add)
        R = -cnt / n_ref[:, None] - lam * (fail.float() + out["V"].float().view(args.B, args.M))
        adv = ((R - R.mean(1, keepdim=True)) / (R.std(1, keepdim=True) + 1e-6)).view(-1)

        opt.zero_grad(set_to_none=True)
        E = inst.shape[0]
        loss = backward_logp(policy, out["traj"], adv, H, W, amp, scale=1.0 / E)
        if args.imit > 0:  # imitate the best-known solution of each instance
            key = cnt + fail * 1e6
            best = key.min(1)
            t_key = teach["count"].float() + (~teach["feasible"]) * 1e6
            use_sample = best.values <= t_key
            ep = torch.arange(args.B, device=dev) * args.M + best.indices
            steps = (out["actions"] >= 0).sum(1).clamp(min=1).float()
            wt = torch.zeros(E, device=dev)
            wt[ep[use_sample]] = args.imit / steps[ep[use_sample]]
            loss += backward_logp(policy, out["traj"], wt, H, W, amp, scale=1.0 / args.B)
            tb = torch.nonzero(~use_sample & teach["feasible"]).squeeze(-1)
            if tb.numel():
                tenv = Env(batch, ib[tb], cfg, max_steps=args.max_steps, allow_prune=not args.no_prune, feat_mode=args.feat)
                rep = rollout(policy, tenv, forced=teach["actions"][tb], store=True, amp=amp)
                tsteps = (rep["actions"] >= 0).sum(1).clamp(min=1).float()
                loss += backward_logp(policy, rep["traj"], args.imit / tsteps, H, W, amp, scale=1.0 / args.B)
        torch.nn.utils.clip_grad_norm_(policy.parameters(), args.clip)
        opt.step()
        fail_rate = fail.float().mean().item()
        lam = max(0.0, lam + args.lam_lr * (fail_rate - args.fail_target))

        ok = ~fail
        ratio = (cnt / n_ref[:, None])[ok].mean().item() if ok.any() else float("nan")
        solved = ok.any(1)  # instances with at least one feasible sample
        best_ratio = ((cnt + fail * 1e6).min(1).values / n_ref)[solved].mean().item() if solved.any() else float("nan")
        if it % 25 == 0:
            print(f"it {it:6d} | {H}x{W} P={batch.P} F={batch.freqs.shape[1]} cert-bad {n_bad} | count/ref {ratio:.3f} best-of-M {best_ratio:.3f} "
                  f"fail {fail_rate:.2f} lam {lam:.2f} | loss {loss:.3f} | {time.time()-t0:.0f}s", flush=True)
        if it % args.val_every == 0 or it == args.iters:
            v = validate(policy, val_sets, args, cfg, dev)
            print("  val " + "  ".join(f"{s}: count/greedy+prune={r:.3f} fail={f:.2f}" for s, (r, f) in v.items()), flush=True)
            w.writerow([it, ratio, best_ratio, fail_rate, lam, loss, time.time() - t0] +
                       [x for r, f in v.values() for x in (r, f)])
            log.flush()
            ck = dict(model=policy.state_dict(), **meta, iter=it)
            torch.save(ck, os.path.join(args.out, "last.pt"))
            # select checkpoints on in-distribution sizes only (larger sizes = test, no leakage)
            ind = [(r, f) for s, (r, f) in v.items() if (isinstance(s, str) and "pcb10" in s) or
                   (not isinstance(s, str) and smin <= s <= smax)] or list(v.values())
            score = np.mean([(r if r == r else 9.0) + 2 * f for r, f in ind])
            if score < best_val:
                best_val = score
                torch.save(ck, os.path.join(args.out, "best.pt"))


if __name__ == "__main__":
    main()
