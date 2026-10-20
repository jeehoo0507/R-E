"""Run a policy on an Env. Rollouts run without gradients and store (features, mask, action);
log-probabilities are recomputed with gradients afterwards in chunks (episodes can be long,
so keeping the autograd graph of every step would not fit in memory)."""
import torch

from .sim import Env


def amp_ctx(device, enabled):
    if enabled and device.type == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return torch.autocast("cpu", enabled=False)


def _logits(policy, x, valid, H, W, mask, amp):
    with amp_ctx(x.device, amp):
        node, stop = policy(x, valid, H, W)
    logits = torch.cat([node.reshape(x.shape[0], -1), stop[:, None]], 1).float()
    return logits.masked_fill(~mask, float("-inf"))


@torch.no_grad()
def rollout(policy, env: Env, greedy=False, forced=None, store=False, amp=True, gen=None):
    """Sample (or argmax, or replay `forced` [E,steps]) actions until every episode is done.
    Returns dict(count, feasible, V, actions, traj); traj = list of per-step dicts if store."""
    E, dev = env.E, env.occ.device
    acts, traj, t = [], [], 0
    while True:
        obs = env.observe(feats=True)
        if forced is not None:
            env.done |= forced[:, t] < 0 if t < forced.shape[1] else torch.ones_like(env.done)
        if bool(env.done.all()):
            break
        act = ~env.done
        mask = obs["mask"].clone()
        mask[~act] = False
        mask[~act, env.stop_idx] = True
        lg = _logits(policy, obs["x"], env.valid, env.b.H, env.b.W, mask, amp)
        if forced is not None:
            a = torch.where(act, forced[:, t].clamp(min=0), env.stop_idx)
            ok = mask.gather(1, a[:, None]).squeeze(1)
            env.done |= act & ~ok          # replay diverged from the env rules (numerical edge case)
            act = act & ok
        elif greedy:
            a = lg.argmax(-1)
        else:
            a = torch.multinomial(torch.softmax(lg, -1), 1, generator=gen).squeeze(-1)
        if store and bool(act.any()):
            e = torch.nonzero(act).squeeze(-1)
            traj.append(dict(x=obs["x"][e], valid=env.valid[e], mask=mask[e], a=a[e], ep=e))
        acts.append(torch.where(act, a, -1))
        env.step(torch.where(act, a, env.stop_idx))
        t += 1
    actions = torch.stack(acts, 1) if acts else torch.full((E, 0), -1, dtype=torch.long, device=dev)
    return dict(count=env.count(), feasible=env.feas.clone(), V=env.V.clone(), occ=env.occ.clone(),
                actions=actions, traj=traj)


def backward_logp(policy, traj, weight, H, W, amp=True, chunk_cells=1 << 16, scale=1.0):
    """Accumulate gradients of -(scale * sum_t weight[ep_t] * log pi(a_t | s_t)).
    weight[E] per episode (advantage, or imitation weight). Returns the loss value."""
    total = 0.0
    if not traj:
        return total
    x = torch.cat([s["x"] for s in traj])
    valid = torch.cat([s["valid"] for s in traj])
    mask = torch.cat([s["mask"] for s in traj])
    a = torch.cat([s["a"] for s in traj])
    w = weight[torch.cat([s["ep"] for s in traj])].float()
    keep = w != 0
    x, valid, mask, a, w = x[keep], valid[keep], mask[keep], a[keep], w[keep]
    step = max(1, chunk_cells // max(1, x.shape[1]))
    for i in range(0, x.shape[0], step):
        lg = _logits(policy, x[i:i + step], valid[i:i + step], H, W, mask[i:i + step], amp)
        lp = torch.log_softmax(lg, -1).gather(1, a[i:i + step, None]).squeeze(1)
        loss = -(w[i:i + step] * lp).sum() * scale
        loss.backward()
        total += loss.item()
    return total
