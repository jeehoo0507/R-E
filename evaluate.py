"""Evaluate trained policies vs classical baselines on unseen sizes / shapes (min decap count).

Example:
  python evaluate.py --ckpt gnn=runs/gnn_phys/best.pt coord=runs/gnn_coord/best.pt \
                     --sizes 10,15,20,30,50 --out results
Outputs:
  results/per_instance.csv  every (size, shape, target, method, instance): count, feasible, seconds
  results/summary.csv       per (size, method): success rate, mean count, count / greedy+prune,
                            gap to best-known, seconds per instance (setup included)
Times include the board setup (sparse LU + diag(Z0)) for every method, since every method needs it.
"""
import argparse
import csv
import os
import time
from collections import defaultdict

import numpy as np
import torch

from decap_loop.baselines import genetic, greedy, local_search, random_prune
from decap_loop.pdn import PDNConfig, sample_batch
from decap_loop.rollout import rollout
from decap_loop.sim import Env
from train import make_policy, rng_range

METHODS = ("greedy", "greedy+prune", "LS", "GA")  # "random+prune" too (slow on big boards: random adds)


def parse():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", nargs="*", default=[], help="name=path ...")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--sizes", default="10,15,20,30,50")
    p.add_argument("--shapes", default="rect,L,holes")
    p.add_argument("--targets", default="flat,step")
    p.add_argument("--n_probe", default="1-3")
    p.add_argument("--difficulty", default="0.02-0.2")
    p.add_argument("--n_inst", type=int, default=16, help="instances per (size, shape)")
    p.add_argument("--methods", default=",".join(METHODS))
    p.add_argument("--samples", type=int, default=16, help="also report best-of-S sampling for models (0=off)")
    p.add_argument("--random_n", type=int, default=8)
    p.add_argument("--ls", default="4x32", help="local search passes x trials")
    p.add_argument("--ga", default="64x150", help="GA pop x gens")
    p.add_argument("--chunk_mem_gb", type=float, default=2.0, help="max dense Z0 memory per chunk")
    p.add_argument("--seed", type=int, default=777)
    p.add_argument("--out", default="results")
    return p.parse_args()


def load(path, dev):
    ck = torch.load(path, map_location=dev, weights_only=False)
    a = ck["args"]
    cfg = PDNConfig(**{k: tuple(map(tuple, v)) if k == "decaps" else v for k, v in ck["cfg"].items()})
    pol = make_policy(cfg, a).to(dev)
    pol.load_state_dict(ck["model"])
    pol.eval()
    return pol, a, cfg


def sync(dev):
    if dev.type == "cuda":
        torch.cuda.synchronize()


def main():
    args = parse()
    os.makedirs(args.out, exist_ok=True)
    dev = torch.device(args.device)
    models = {}
    for spec in args.ckpt:
        name, path = spec.split("=", 1)
        models[name] = load(path, dev)
    cfg = next(iter(models.values()))[2] if models else PDNConfig()
    methods = args.methods.split(",")
    ls_p, ls_t = map(int, args.ls.split("x"))
    ga_pop, ga_gens = map(int, args.ga.split("x"))
    kw = dict(targets=tuple(args.targets.split(",")), n_probe=rng_range(args.n_probe),
              difficulty=rng_range(args.difficulty, float))

    rows = []
    for size in map(int, args.sizes.split(",")):
        n = size * size
        dense = n <= cfg.dense_max_nodes
        chunk = max(1, min(args.n_inst, int(args.chunk_mem_gb * 1e9 / (cfg.n_freq * n * n * 16)))) if dense else 1
        for shape in args.shapes.split(","):
            rng = np.random.default_rng([args.seed, size, sum(map(ord, shape))])  # same instances across runs
            done = 0
            while done < args.n_inst:
                nb = min(chunk, args.n_inst - done)
                batch = sample_batch(nb, size, size, cfg, rng, shapes=(shape,), device=dev, **kw)
                ib = torch.arange(nb, device=dev)
                res = {}

                def timed(name, fn):
                    sync(dev)
                    t = time.time()
                    r = fn()
                    sync(dev)
                    res[name] = (r["count"].cpu().numpy(), r["feasible"].cpu().numpy(),
                                 (time.time() - t) / nb + batch.setup_sec)
                    return r

                gp = timed("greedy+prune", lambda: greedy(batch, cfg))
                if "greedy" in methods:
                    timed("greedy", lambda: greedy(batch, cfg, prune=False))
                if "random+prune" in methods:
                    timed("random+prune", lambda: random_prune(batch, cfg, args.random_n))
                if "LS" in methods:
                    timed("LS", lambda: (lambda o: dict(count=(o >= 0).sum(1), feasible=gp["feasible"]))(
                        local_search(batch, cfg, gp["occ"], gp["feasible"], ls_p, ls_t)))
                if "GA" in methods:
                    timed("GA", lambda: genetic(batch, cfg, ga_pop, ga_gens, init_occ=gp["occ"]))
                with torch.no_grad():
                    for name, (pol, a, _) in models.items():
                        def run(inst, greedy_dec):
                            env = Env(batch, inst, cfg, allow_prune=not a["no_prune"], feat_mode=a["feat"])
                            return rollout(pol, env, greedy=greedy_dec)
                        timed(name, lambda: run(ib, True))
                        if args.samples > 1:
                            def best_of():
                                o = run(ib.repeat_interleave(args.samples), False)
                                key = (o["count"] + (~o["feasible"]) * 10 ** 6).view(nb, -1)
                                j = key.argmin(1) + ib * args.samples
                                return dict(count=o["count"][j], feasible=o["feasible"][j])
                            timed(f"{name}_best{args.samples}", best_of)
                for m, (c, f, t) in res.items():
                    rows += [dict(size=size, shape=shape, target=batch.targets[i], P=batch.P, method=m,
                                  inst=done + i, count=int(c[i]), feasible=bool(f[i]), sec=t) for i in range(nb)]
                done += nb
                print(f"{size}x{size} {shape}: {done}/{args.n_inst} | " + " ".join(
                    f"{m}={c[f].mean() if f.any() else float('nan'):.1f}({f.mean():.2f})" for m, (c, f, _) in res.items()),
                    flush=True)

    # best-known = fewest decaps among feasible results of all methods on that instance
    best, ref = {}, {}
    for r in rows:
        k = (r["size"], r["shape"], r["inst"])
        if r["feasible"]:
            best[k] = min(best.get(k, 10 ** 9), r["count"])
        if r["method"] == "greedy+prune" and r["feasible"]:
            ref[k] = r["count"]
    for r in rows:
        k = (r["size"], r["shape"], r["inst"])
        r["ratio_gp"] = r["count"] / max(ref[k], 1) if r["feasible"] and k in ref else ""
        r["gap_best"] = r["count"] / max(best[k], 1) - 1 if r["feasible"] and k in best else ""
    with open(os.path.join(args.out, "per_instance.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    agg = defaultdict(list)
    for r in rows:
        agg[(r["size"], r["method"])].append(r)
    with open(os.path.join(args.out, "summary.csv"), "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["size", "method", "success", "count", "count_over_gp", "gap_best_pct", "sec_per_inst"])
        print(f"\n{'size':>4} {'method':>18} {'success':>7} {'count':>6} {'/g+p':>6} {'gap%':>6} {'sec':>8}")
        for (s, m), rs in sorted(agg.items()):
            ok = [r for r in rs if r["feasible"]]
            succ = len(ok) / len(rs)
            cnt = np.mean([r["count"] for r in ok]) if ok else float("nan")
            rat = np.mean([r["ratio_gp"] for r in ok if r["ratio_gp"] != ""] or [float("nan")])
            gap = 100 * np.mean([r["gap_best"] for r in ok] or [float("nan")])
            sec = np.mean([r["sec"] for r in rs])
            w.writerow([s, m, succ, cnt, rat, gap, sec])
            print(f"{s:>4} {m:>18} {succ:7.2f} {cnt:6.1f} {rat:6.3f} {gap:6.1f} {sec:8.3f}")


if __name__ == "__main__":
    main()
