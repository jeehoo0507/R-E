"""Certified problem pools for training and evaluation.

Each pool file holds one batch of instances (same board size) whose working grids were refined by
verify.cutting_plane with greedy+prune, plus the greedy+prune result (certified). Training samples
these files instead of generating and certifying new instances every iteration.

Run: python build_pool.py [--out runs/pool]
"""
import argparse
import dataclasses
import os
import time

import numpy as np
import torch

from decap_loop.bank import save_bank
from decap_loop.baselines import greedy
from decap_loop.pdn import PDNConfig, sample_batch
from decap_loop.verify import cutting_plane

PCB = PDNConfig(dense_max_nodes=144)
IP = dataclasses.replace(PCB, cell_size=200e-6, diel_thickness=1e-6, metal_thickness=1e-6, vrm_l=50e-12,
                         decaps=((100e-12, 5e-12, 0.05), (20e-12, 2e-12, 0.1), (5e-12, 1e-12, 0.2)))
CFGS = {"pcb": PCB, "ip": IP}

# (name, cfg, size, batches, instances per batch, seed base, difficulty)
# Interposer test targets use an easier difficulty: at 0.02-0.2 the library's small capacitors meet the
# target only with almost every cell holding the one right type, and greedy (no type changes) fails.
HARD, EASY = (0.02, 0.2), (0.2, 0.5)
JOBS = [("test_pcb10", "pcb", 10, 1, 16, 9000, HARD), ("test_ip10", "ip", 10, 1, 16, 9100, EASY),
        ("train_pcb8", "pcb", 8, 6, 16, 100, HARD), ("train_pcb10", "pcb", 10, 6, 16, 200, HARD),
        ("train_pcb12", "pcb", 12, 6, 16, 300, HARD),
        ("test_pcb16", "pcb", 16, 1, 8, 9200, HARD), ("test_ip16", "ip", 16, 1, 8, 9300, EASY)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/pool")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    torch.manual_seed(0)
    for name, cname, size, nb, n, seed, diff in JOBS:
        cfg = CFGS[cname]
        for k in range(nb):
            path = os.path.join(a.out, f"{name}_{k}.pkl")
            if os.path.exists(path):
                continue
            t = time.time()
            batch = sample_batch(n, size, size, cfg, np.random.default_rng(seed + k), difficulty=diff)
            r, info = cutting_plane(lambda b: greedy(b, cfg), batch, cfg, max_rounds=10)
            ok = info["worst"] <= np.log10(1.01)
            res = {"greedy+prune": dict(occ=r["occ"].cpu().numpy(), worst=info["worst"], ok=ok,
                                        count=(r["occ"] >= 0).sum(1).cpu().numpy())}
            save_bank(path, batch, res)
            print(f"{name}_{k}: {cname} {size}x{size} n={n} F={batch.freqs.shape[1]} g+p ok {ok.mean():.2f} "
                  f"count {res['greedy+prune']['count'][ok].mean() if ok.any() else float('nan'):.1f} "
                  f"rounds {info['rounds']} {time.time() - t:.0f}s", flush=True)


if __name__ == "__main__":
    main()
