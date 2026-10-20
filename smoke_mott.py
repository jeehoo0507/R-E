"""End-to-end smoke test of the MoTT pipeline on tiny boards (a few minutes, CPU or GPU).

Checks, in order (stops at the first failure, exit code 1):
  1. tiny certified pools (6x6, PCB and interposer-like) with cutting_plane
  2. bank reload: Woodbury on the grown grid == direct solve with decaps inside Y
  3. MoTT and GridPolicy(+prior): rollout + backward on the pool, finite outputs
  4. train.py for both models, 2 iterations each on the pool, checkpoints written
  5. eval_mott.py on the tiny test pools, report written

Run: python smoke_mott.py [--device cuda] [--keep]
"""
import argparse
import os
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

from build_pool import CFGS
from decap_loop.bank import load_bank, save_bank
from decap_loop.baselines import greedy
from decap_loop.model import MoTT, GridPolicy
from decap_loop.pdn import sample_batch
from decap_loop.rollout import backward_logp, rollout
from decap_loop.sim import K_MODES, Env, feat_dim, mott_dims, probe_impedance
from decap_loop.verify import _inst_data, _probe_z_exact, cutting_plane

POOL, RUNS = "runs/smoke_pool", "runs/smoke"


def step(name, fn):
    t = time.time()
    try:
        msg = fn()
    except Exception as e:  # report and stop
        print(f"[FAIL] {name}: {type(e).__name__}: {e}", flush=True)
        sys.exit(1)
    print(f"[ OK ] {name} ({time.time() - t:.1f}s){' — ' + msg if msg else ''}", flush=True)


def build_pools():
    os.makedirs(POOL, exist_ok=True)
    out = []
    for name, cname, diff, seed in (("train_pcb6_0", "pcb", (0.02, 0.2), 1), ("test_pcb6_0", "pcb", (0.02, 0.2), 2),
                                    ("test_ip6_0", "ip", (0.2, 0.5), 3)):
        cfg = CFGS[cname]
        b = sample_batch(4, 6, 6, cfg, np.random.default_rng(seed), difficulty=diff)
        r, info = cutting_plane(lambda bb: greedy(bb, cfg), b, cfg, max_rounds=3)
        save_bank(os.path.join(POOL, name + ".pkl"), b, {})
        out.append(f"{name} F={b.freqs.shape[1]}")
    return ", ".join(out)


def check_reload():
    b, _ = load_bank(os.path.join(POOL, "train_pcb6_0.pkl"))
    occ = torch.full((1, b.n), -1, dtype=torch.long)
    cells = torch.nonzero(b.placeable()[0]).squeeze(-1)[:4]
    occ[0, cells] = torch.arange(len(cells)) % b.cfg.n_types
    inst = torch.tensor([0])
    b.bank.ensure(inst, torch.where(occ >= 0, torch.arange(b.n), -1))
    zw = probe_impedance(b, inst, occ, b.zd()[inst])[0].abs().numpy()
    mask, ent, P = _inst_data(b, 0)
    ze = _probe_z_exact(mask, ent, P, occ[0].numpy(), b.cfg, b.freqs[0])
    err = float(np.abs(zw - ze).max() / ze.max())
    assert err < 1e-8, f"Woodbury vs direct rel err {err:.1e}"
    return f"rel err {err:.1e} on F={b.freqs.shape[1]}"


def check_models(dev):
    b, _ = load_bank(os.path.join(POOL, "train_pcb6_0.pkl"), dev, dense=True, z_dtype=torch.complex64)
    T = b.cfg.n_types
    msgs = []
    for name, pol, fm in (("mott", MoTT(T, K_MODES, mott_dims(T), d=32, layers=1, heads=2), "mott"),
                          ("grid+prior", GridPolicy(feat_dim(b.cfg, "phys"), T, d=32, layers=1, heads=2, prior=True), "phys")):
        pol = pol.to(dev)
        env = Env(b, torch.arange(b.B, device=dev).repeat_interleave(2), b.cfg, feat_mode=fm)
        obs = env.observe()
        assert torch.isfinite(obs["x"]).all(), f"{name}: non-finite features"
        out = rollout(pol, env, store=True, amp=False)
        loss = backward_logp(pol, out["traj"], torch.randn(env.E, device=dev), b.H, b.W, amp=False, scale=1 / env.E)
        assert np.isfinite(loss), f"{name}: loss {loss}"
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in pol.parameters()), f"{name}: bad grads"
        msgs.append(f"{name} {len(out['traj'])} steps")
    return ", ".join(msgs)


def run(cmd):
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(" ".join(cmd) + "\n" + (r.stdout + r.stderr)[-2000:])
    return r.stdout


def check_train(dev):
    common = [sys.executable, "train.py", "--pool", POOL, "--pool_glob", "train_*.pkl", "--val_pools", "test_pcb6_0",
              "--iters", "2", "--val_every", "2", "--B", "4", "--M", "2", "--d", "32", "--layers", "1", "--heads", "2",
              "--device", dev]
    for name, extra in (("mott", ["--model", "mott"]), ("grid_prior", ["--model", "grid", "--feat", "phys", "--prior"])):
        run(common + extra + ["--out", os.path.join(RUNS, name)])
        assert os.path.exists(os.path.join(RUNS, name, "last.pt")), f"{name}: no checkpoint"
    return "2 iterations each, checkpoints written"


def check_eval():
    out = os.path.join(RUNS, "eval.md")
    run([sys.executable, "eval_mott.py", "--runs", f"{RUNS}/mott,{RUNS}/grid_prior", "--ckpt", "last.pt",
         "--pool", POOL, "--tests", "test_pcb6_0,test_ip6_0", "--samples", "2", "--rounds", "1", "--no_ls",
         "--out", out])
    txt = open(out).read()
    assert "test_pcb6_0" in txt and "test_ip6_0" in txt, "report misses a test set"
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--keep", action="store_true", help="keep runs/smoke_pool and runs/smoke")
    a = ap.parse_args()
    torch.manual_seed(0)
    for d in (POOL, RUNS):
        shutil.rmtree(d, ignore_errors=True)
    print(f"smoke test on {a.device}", flush=True)
    step("1 tiny certified pools", build_pools)
    step("2 bank reload + Woodbury vs direct", check_reload)
    step("3 models: rollout + backward", lambda: check_models(torch.device(a.device)))
    step("4 train.py (mott, grid_prior)", lambda: check_train(a.device))
    step("5 eval_mott.py", check_eval)
    if not a.keep:
        for d in (POOL, RUNS):
            shutil.rmtree(d, ignore_errors=True)
    print("SMOKE PASSED", flush=True)


if __name__ == "__main__":
    main()
