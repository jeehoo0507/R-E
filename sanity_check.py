"""Correctness + physics sanity checks. Run: python sanity_check.py [--big]"""
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from decap_loop.baselines import genetic, greedy, local_search, random_prune
from decap_loop.model import GridPolicy
from decap_loop.pdn import PDNConfig, build_Y0, sample_batch
from decap_loop.rollout import backward_logp, rollout
from decap_loop.sim import Env, action_effects, feat_dim, loaded_state, probe_impedance

torch.manual_seed(0)
cfg = PDNConfig()
rng = np.random.default_rng(0)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
zd = cfg.decap_z(dev)
T = cfg.n_types

print(f"cell C={cfg.c_cell*1e12:.3f} pF, edge L={cfg.l_edge*1e12:.2f} pH, edge R={cfg.r_edge*1e3:.2f} mohm, "
      f"f1(10x10)={cfg.f1(10, 10)/1e9:.1f} GHz")


def direct_zpp(batch, b, occ_row):
    """Reference: put decap admittances into Y0 and solve the sparse system directly."""
    mask = batch.valid[b].view(batch.H, batch.W).cpu().numpy()
    ent = torch.nonzero(batch.entry[b]).squeeze(-1).cpu().numpy()
    Ys = build_Y0(mask, ent, cfg, batch.freqs[b])
    P = batch.probes[b].cpu().numpy()
    occ = occ_row.cpu().numpy()
    yd = 1.0 / cfg.decap_z_np(batch.freqs[b])
    rhs = np.zeros((batch.n, len(P)), dtype=np.complex128)
    rhs[P, np.arange(len(P))] = 1
    out = []
    for f, Y in enumerate(Ys):
        d = np.zeros(batch.n, dtype=np.complex128)
        d[occ >= 0] = yd[occ[occ >= 0], f]
        out.append(spla.splu((Y + sp.diags(d)).tocsc()).solve(rhs)[P, np.arange(len(P))])
    return torch.from_numpy(np.stack(out)).to(dev)


def rand_occ(batch, b, k):
    occ = torch.full((1, batch.n), -1, dtype=torch.long, device=dev)
    cand = torch.nonzero(batch.placeable()[b]).squeeze(-1)
    occ[0, cand[torch.randperm(len(cand))[:k]]] = torch.randint(T, (min(k, len(cand)),), device=dev)
    return occ


# 1) Woodbury state (dense + sparse bank) == direct sparse solve with decaps inside Y
for dense in (True, False):
    bb = sample_batch(2, 7, 9, cfg, np.random.default_rng(1), shapes=("holes",), n_probe=(2, 2), device=dev, dense=dense)
    occ = rand_occ(bb, 0, 8)
    inst = torch.tensor([0], device=dev)
    bb.bank.ensure(inst, torch.where(occ >= 0, torch.arange(bb.n, device=dev), -1))
    ref = direct_zpp(bb, 0, occ[0])
    _, _, zpp = loaded_state(bb, inst, occ, bb.zd()[inst])
    zpp2 = probe_impedance(bb, inst, occ, bb.zd()[inst])
    e1 = ((zpp[0] - ref).abs() / ref.abs()).max().item()
    e2 = ((zpp2[0] - ref).abs() / ref.abs()).max().item()
    print(f"[1] {'dense ' if dense else 'sparse'} bank: Woodbury vs direct solve: loaded_state {e1:.1e}, probe_impedance {e2:.1e}")

# 2) closed-form action effects (add / change type / remove) == recomputed state
bb = sample_batch(1, 8, 8, cfg, np.random.default_rng(2), n_probe=(3, 3), device=dev)
inst = torch.tensor([0], device=dev)
occ = rand_occ(bb, 0, 6)
z_ip, z_ii, z_pp = loaded_state(bb, inst, occ, bb.zd()[inst])
logZt = torch.log10(bb.Zt[inst])
Vn, mn = action_effects(z_ip, z_ii, z_pp, occ, bb.zd()[inst], logZt)
from decap_loop.sim import violation  # noqa: E402
errs = []
empty = torch.nonzero((occ[0] < 0) & bb.placeable()[0]).squeeze(-1)[:3]
placed = torch.nonzero(occ[0] >= 0).squeeze(-1)[:3]
for node in torch.cat([empty, placed]).tolist():
    for c in range(T + 1):
        if occ[0, node] < 0 and c == T:
            continue
        o2 = occ.clone()
        o2[0, node] = c if c < T else -1
        bb.bank.ensure(inst, o2.clamp(max=0) * 0 + torch.where(o2 >= 0, torch.arange(bb.n, device=dev), -1))
        V2, m2 = violation(probe_impedance(bb, inst, o2, bb.zd()[inst]), logZt)
        errs.append(abs(m2.item() - mn[0, node, c].item()) + abs(V2.item() - Vn[0, node, c].item()))
print(f"[2] rank-1 action effects vs recomputed (add/swap/remove, {len(errs)} cases): max err {max(errs):.1e}")

# 3) baselines on 10x10 boards
batch = sample_batch(8, 10, 10, cfg, rng, device=dev)
print(f"[3] 10x10, {batch.B} inst, {batch.P} probes, targets {batch.targets}")
res = {}
for name, fn in (("greedy", lambda: greedy(batch, cfg, prune=False)),
                 ("greedy+prune", lambda: greedy(batch, cfg)),
                 ("random+prune(8)", lambda: random_prune(batch, cfg, 8))):
    t = time.time()
    res[name] = fn()
    print(f"    {name:16s} count {res[name]['count'].tolist()}  feasible {res[name]['feasible'].float().mean():.2f}"
          f"  ({time.time()-t:.2f}s)")
gp = res["greedy+prune"]
# incremental rank-1 state after a long random episode == from-scratch state
env = Env(batch, torch.arange(batch.B, device=dev), cfg, resync=10 ** 9)
from decap_loop.baselines import heuristic_rollout  # noqa: E402
heuristic_rollout(env, True, mode="random", gen=torch.Generator(device=dev).manual_seed(0))
z_ip, z_ii, _ = loaded_state(batch, env.inst, env.occ, batch.zd()[env.inst])
drift = max(((env.z_ip - z_ip).abs() / z_ip.abs().clamp(min=1e-12)).max().item(),
            ((env.z_ii - z_ii).abs() / z_ii.abs()).max().item())
print(f"    incremental vs from-scratch state after {int(env.steps.max())} steps (no resync): max rel err {drift:.1e}")
t = time.time()
ls = local_search(batch, cfg, gp["occ"], gp["feasible"])
print(f"    {'local search':16s} count {(ls >= 0).sum(1).tolist()}  ({time.time()-t:.2f}s)")
t = time.time()
ga = genetic(batch, cfg, pop=32, gens=30, init_occ=gp["occ"])
print(f"    {'GA 32x30':16s} count {ga['count'].tolist()}  feasible {ga['feasible'].float().mean():.2f}  ({time.time()-t:.2f}s)")
# final solutions really meet the target (direct solve)
ok = []
for b in range(batch.B):
    if bool(gp["feasible"][b]):
        z = direct_zpp(batch, b, gp["occ"][b]).abs()
        ok.append(bool((z <= batch.Zt[b][:, None] * (1 + 1e-5)).all()))
print(f"    greedy+prune solutions verified by direct solve: {sum(ok)}/{len(ok)}")

# 4) policy: rollout, replay of the greedy trajectory, backward
for mode in ("phys", "coord"):
    pol = GridPolicy(feat_dim(cfg, mode), T, d=32, layers=2, heads=4).to(dev)
    opt = torch.optim.Adam(pol.parameters(), 1e-3)
    inst = torch.arange(batch.B, device=dev).repeat_interleave(2)
    out = rollout(pol, Env(batch, inst, cfg, feat_mode=mode, max_steps=40), store=True)
    w = torch.randn(inst.shape[0], device=dev)
    loss = backward_logp(pol, out["traj"], w, batch.H, batch.W)
    opt.step()
    rep = rollout(pol, Env(batch, torch.arange(batch.B, device=dev), cfg, feat_mode=mode),
                  forced=gp["actions"], store=True)
    same = bool((rep["count"] == gp["count"]).all() and (rep["feasible"] == gp["feasible"]).all())
    print(f"[4] {mode}: sampled count {out['count'].float().mean():.1f} feasible {out['feasible'].float().mean():.2f}"
          f", backward OK (loss {loss:.2f}), greedy replay reproduces greedy+prune: {same}")

# 5) (--big) sparse setup + one greedy+prune at 100x100
if "--big" in sys.argv:
    for s in (50, 100):
        t = time.time()
        bb = sample_batch(1, s, s, cfg, np.random.default_rng(5), shapes=("rect",), device=dev)
        t_gen = time.time() - t
        t = time.time()
        r = greedy(bb, cfg)
        print(f"[5] {s}x{s}: instance+target {t_gen:.1f}s (bank setup {bb.setup_sec:.1f}s), greedy+prune "
              f"{time.time()-t:.1f}s -> {int(r['count'][0])} decaps, feasible {bool(r['feasible'][0])}")
print("all checks done")
