"""Non-learning baselines for min decap count s.t. |Z_probe| <= Zt, all on the same environment.

greedy        : add the (cell, type) that leaves the least violation until the target is met
greedy+prune  : then remove, one at a time, the decap whose removal keeps the most margin
random+prune  : random adds until the target is met, then prune
local_search  : destroy-and-repair from a solution: remove 1-2 decaps, greedy repair, prune;
                accept if the count drops (large-neighbourhood search)
genetic       : GA over a candidate pool (all placeable cells, or the PAGA-like top cells by
                transfer impedance on large boards); feasible-first fitness, then repair + prune
"""
import numpy as np
import torch

from .sim import FEAS_TOL, Env, _probe_diag, probe_impedance, violation

INF = float("inf")


@torch.no_grad()
def heuristic_rollout(env: Env, prune=True, mode="greedy", gen=None, only=None):
    """Drive an Env with a rule. Returns the actions taken [E, steps] (-1 after done).
    only[E,n] (optional): adds are restricted to these (cell -> type) choices (-1 = not allowed)."""
    E, n, T = env.E, env.n, env.T
    acts = []
    while True:
        obs = env.observe(feats=False)
        if bool(env.done.all()):
            break
        allowed = obs["mask"][:, :-1].view(E, n, T + 1).clone()
        allowed[..., :T] &= (env.occ < 0)[..., None]          # no type changes
        if only is not None:
            allowed[..., :T] &= torch.nn.functional.one_hot(only.clamp(min=0), T).bool() & (only >= 0)[..., None]
        if not prune:
            allowed[..., T] = False
        if mode == "greedy":
            Vn, mn = obs["Vn"], obs["mn"]
            score = torch.where(Vn <= 0, mn - 100.0, Vn)       # meeting the target beats any violation
            score[..., T] = mn[..., T]                          # remove: keep the most margin
        else:
            score = torch.rand(E, n, T + 1, device=env.occ.device, generator=gen, dtype=torch.float64)
            score[..., T] = obs["mn"][..., T]
        score = score.masked_fill(~allowed, INF).view(E, -1)
        best, a = score.min(1)
        a = torch.where(torch.isfinite(best), a, torch.full_like(a, env.stop_idx))
        acts.append(torch.where(env.done, torch.full_like(a, -1), a))
        env.step(a)
    if not acts:
        return torch.full((E, 0), -1, dtype=torch.long, device=env.occ.device)
    return torch.stack(acts, 1)


def _result(env, actions=None):
    return dict(count=env.count(), feasible=env.feas.clone(), occ=env.occ.clone(), actions=actions)


@torch.no_grad()
def greedy(batch, cfg, prune=True, inst=None, occ0=None, max_steps=None):
    inst = torch.arange(batch.B, device=batch.valid.device) if inst is None else inst
    env = Env(batch, inst, cfg, occ0=occ0, max_steps=max_steps)
    return _result(env, heuristic_rollout(env, prune))


@torch.no_grad()
def trajectory_to(batch, cfg, inst, occ_target, max_steps=None):
    """An action sequence that ends in (a pruned subset of) a given solution: add its decaps in
    greedy order until the target is met, then prune. Used to imitate solutions found by
    local search / GA, which have no trajectory of their own."""
    env = Env(batch, inst, cfg, max_steps=max_steps)
    return _result(env, heuristic_rollout(env, True, only=occ_target))


@torch.no_grad()
def random_prune(batch, cfg, n_samples=8, seed=0):
    """Best (feasible, fewest) of n_samples random-add + greedy-prune runs per instance."""
    gen = torch.Generator(device=batch.valid.device).manual_seed(seed)
    inst = torch.arange(batch.B, device=batch.valid.device).repeat_interleave(n_samples)
    env = Env(batch, inst, cfg)
    heuristic_rollout(env, True, mode="random", gen=gen)
    key = (env.count() + (~env.feas) * 10 ** 6).view(batch.B, n_samples)
    j = key.argmin(1) + torch.arange(batch.B, device=inst.device) * n_samples
    return dict(count=env.count()[j], feasible=env.feas[j], occ=env.occ[j])


@torch.no_grad()
def local_search(batch, cfg, occ, feasible, passes=4, trials=32, destroy=2, seed=0):
    """Destroy-and-repair around a feasible solution, per instance. occ[B,n] -> improved occ."""
    rng = np.random.default_rng(seed)
    dev = occ.device
    occ = occ.clone()
    for b in range(batch.B):
        if not bool(feasible[b]):
            continue
        for _ in range(passes):
            placed = torch.nonzero(occ[b] >= 0).squeeze(-1).cpu().numpy()
            if len(placed) == 0:
                break
            subsets = [[p] for p in placed]
            while len(subsets) < trials and destroy > 1 and len(placed) > 1:
                subsets.append(list(rng.choice(placed, min(destroy, len(placed)), replace=False)))
            subsets = subsets[:max(trials, len(placed))]
            start = occ[b].repeat(len(subsets), 1)
            for i, s in enumerate(subsets):
                start[i, s] = -1
            r = greedy(batch, cfg, inst=torch.full((len(subsets),), b, device=dev), occ0=start)
            key = r["count"] + (~r["feasible"]) * 10 ** 6
            i = int(key.argmin())
            if bool(r["feasible"][i]) and int(r["count"][i]) < int((occ[b] >= 0).sum()):
                occ[b] = r["occ"][i]
            else:
                break
    return occ


@torch.no_grad()
def genetic(batch, cfg, pop=64, gens=100, p_mut=None, elite=0.1, pool_max=400, init_occ=None, seed=0):
    """Min-count GA per instance. Genome = decap type (or empty) on every pool cell.
    Fitness (lower is better) = count, + 1000 + 100 * violation if the target is not met.
    The best genome is repaired (greedy) if needed and pruned. Returns dict(count, feasible, occ)."""
    rng = np.random.default_rng(seed)
    dev, T, n = batch.valid.device, cfg.n_types, batch.n
    zd = batch.zd(dev)
    out_occ = torch.full((batch.B, n), -1, dtype=torch.long, device=dev)
    n_el = max(1, int(elite * pop))
    for b in range(batch.B):
        cand = torch.nonzero(batch.placeable()[b]).squeeze(-1)
        if len(cand) > pool_max:  # PAGA-like priority: strongest coupling to the probes
            score = batch.bank.cols(torch.tensor([b], device=dev), batch.probes[b:b + 1])[0].abs().mean(0).amax(-1)
            if init_occ is not None:  # the seed solution must stay representable
                score = score + (init_occ[b] >= 0) * 1e30
            cand = cand[score[cand].argsort(descending=True)[:pool_max]]
        batch.bank.ensure(torch.tensor([b], device=dev), cand[None])
        pool = cand.cpu().numpy()
        m = len(pool)
        k0 = int((init_occ[b] >= 0).sum()) if init_occ is not None else max(1, m // 8)
        pm = p_mut if p_mut is not None else 1.0 / m
        G = np.full((pop, m), -1)
        for i in range(pop):
            kk = int(np.clip(round(k0 * rng.uniform(0.5, 1.5)), 1, m))
            G[i, rng.choice(m, kk, replace=False)] = rng.integers(T, size=kk)
        if init_occ is not None:
            G[0] = init_occ[b][torch.from_numpy(pool).to(dev)].cpu().numpy()
        inst = torch.full((pop,), b, device=dev)
        logZt = torch.log10(batch.Zt[inst])

        def fitness(G):
            occ = torch.full((pop, n), -1, dtype=torch.long, device=dev)
            occ[:, torch.from_numpy(pool).to(dev)] = torch.from_numpy(G).to(dev)
            V, mm = violation(probe_impedance(batch, inst, occ, zd[inst]), logZt)
            cnt = (G >= 0).sum(1)
            return cnt + np.where(mm.cpu().numpy() <= FEAS_TOL, 0, 1000 + 100 * V.cpu().numpy())

        fit = fitness(G)
        for _ in range(gens):
            order = np.argsort(fit)
            G, fit = G[order], fit[order]
            kids = []
            for _ in range(pop - n_el):
                i, j = [min(rng.integers(pop, size=2)) for _ in range(2)]   # tournament (lower = fitter)
                child = np.where(rng.random(m) < 0.5, G[i], G[j])
                mut = rng.random(m) < pm
                child[mut] = rng.integers(-1, T, size=int(mut.sum()))
                if rng.random() < 0.3 and (child >= 0).any():                   # push the count down
                    child[rng.choice(np.flatnonzero(child >= 0))] = -1
                kids.append(child)
            G = np.concatenate([G[:n_el], np.stack(kids)])
            fit = np.concatenate([fit[:n_el], fitness(G)[n_el:]])
        out_occ[b, torch.from_numpy(pool).to(dev)] = torch.from_numpy(G[np.argmin(fit)]).to(dev)
    r = greedy(batch, cfg, occ0=out_occ)  # repair if needed, then prune
    return dict(count=r["count"], feasible=r["feasible"], occ=r["occ"])


@torch.no_grad()
def modal_heuristic(batch, cfg, inst=None, max_steps=None):
    """Physics rule, no learning and no look-ahead over all actions (the classical resonance rule,
    cf. US 6789241): while the target is not met, take the worst violation (f*, probe p*), add the
    decap whose SRF is closest to f* at the cell with the strongest coupling to p* at f*,
        g_i = |Z_ip(f*)| / sqrt(|Z_pp(f*)| |Z_ii(f*)|)
    (single-mode residue coherence). Then prune like greedy+prune."""
    dev = batch.valid.device
    inst = torch.arange(batch.B, device=dev) if inst is None else inst
    env = Env(batch, inst, cfg, max_steps=max_steps)
    E, n, T = env.E, env.n, env.T
    srf = torch.tensor([1.0 / (2 * np.pi * np.sqrt(esl * c)) for c, esl, _ in cfg.decaps],
                       dtype=torch.float64, device=dev)
    freqs = torch.from_numpy(batch.freqs).to(dev)[inst]                              # [E,F]
    ar = torch.arange(E, device=dev)
    while True:
        obs = env.observe(feats=False)
        if bool(env.done.all()):
            break
        allowed = obs["mask"][:, :-1].view(E, n, T + 1).clone()
        allowed[..., :T] &= (env.occ < 0)[..., None]
        z_pp = _probe_diag(env.z_ip, batch.probes[inst])                              # [E,F,P]
        lv = torch.log10(z_pp.abs() + 1e-30) - env.logZt[..., None]                     # [E,F,P]
        flat = lv.reshape(E, -1).argmax(1)
        fi, pi = flat // lv.shape[2], flat % lv.shape[2]
        fstar = freqs[ar, fi]
        zpi = env.z_ip[ar, fi, :, pi]                                                  # [E,n]
        zii = env.z_ii[ar, fi, :]
        zpp = z_pp[ar, fi, pi]
        g = zpi.abs() / torch.sqrt(zpp.abs()[:, None] * zii.abs() + 1e-30)
        trank = (torch.log(srf)[None, :] - torch.log(fstar)[:, None]).abs().argsort(1)  # [E,T] best type first
        a = torch.full((E,), env.stop_idx, dtype=torch.long, device=dev)
        add_ok = allowed[..., :T]                                                      # [E,n,T]
        chosen = torch.zeros(E, dtype=torch.bool, device=dev)
        for r in range(T):                                                             # closest SRF that still has a free cell
            t = trank[:, r]
            ok = add_ok[ar, :, t]                                                      # [E,n]
            sc = torch.where(ok, g, torch.full_like(g, -1.0))
            best, cell = sc.max(1)
            take = (~chosen) & (best >= 0)
            a = torch.where(take, cell * (T + 1) + t, a)
            chosen |= take
        rem = allowed[..., T]
        sc = torch.where(rem, obs["mn"][..., T], torch.full_like(obs["mn"][..., T], INF))
        best, cell = sc.min(1)
        prune = (~chosen) & torch.isfinite(best)
        a = torch.where(prune, cell * (T + 1) + T, a)
        env.step(a)
    return _result(env)
