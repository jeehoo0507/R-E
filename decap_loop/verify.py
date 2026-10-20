"""Exact check of the target on a dense frequency grid, independent of the Woodbury bank.

The working grid of the environment (32 log points + appended points) can miss a resonance peak
that falls between grid points. Here the decaps are put directly into Y(f) and the system is
solved on cfg.n_check log points; the highest local maxima are then refined by golden-section
search in log f with more exact solves. This is the ground truth used for the final success
metric and for H0.

cutting_plane() turns it into an adaptive working grid: run a method, certify its solutions,
append the frequencies where certified solutions violate the target, re-run, until every
claimed solution is certified (lazy constraint generation).
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch

from .pdn import PDNConfig, lap_grid, target_at

GOLD = (np.sqrt(5) - 1) / 2


def _probe_z_exact(mask, entries, probes, occ, cfg: PDNConfig, f, chunk_bytes=2 ** 28):
    """|Z_probe| [F, P] with decaps occ (cell -> type, -1 empty; None = bare) placed inside Y."""
    f = np.atleast_1d(np.asarray(f, dtype=float))
    n = mask.size
    v = mask.reshape(-1).astype(float)
    w = 2 * np.pi * f
    d = (w * cfg.c_cell * cfg.tan_d + 1j * w * cfg.c_cell)[:, None] * v[None] + (1 - v)[None]
    d[:, entries] += (1.0 / (cfg.vrm_r + 1j * w * cfg.vrm_l))[:, None]
    if occ is not None and (np.asarray(occ) >= 0).any():
        occ = np.asarray(occ)
        on = np.flatnonzero(occ >= 0)
        d[:, on] += (1.0 / cfg.decap_z_np(f))[occ[on]].T
    ye = 1.0 / (cfg.r_edge + 1j * w * cfg.l_edge)
    lap = lap_grid(mask)
    P = len(probes)
    out = np.empty((len(f), P))
    if n <= 900:                                         # batched dense solves, chunked
        L = torch.from_numpy(lap.toarray()).to(torch.complex128)
        rhs = torch.zeros(n, P, dtype=torch.complex128)
        rhs[torch.as_tensor(probes), torch.arange(P)] = 1
        fc = max(1, chunk_bytes // (16 * n * n))
        for i in range(0, len(f), fc):
            ye_c = torch.from_numpy(ye[i:i + fc])
            Y = ye_c[:, None, None] * L + torch.diag_embed(torch.from_numpy(d[i:i + fc]))
            x = torch.linalg.solve(Y, rhs.expand(Y.shape[0], n, P))
            out[i:i + fc] = x[:, probes, torch.arange(P)].abs().numpy()
    else:
        rhs = np.zeros((n, P), dtype=np.complex128)
        rhs[probes, np.arange(P)] = 1
        for i in range(len(f)):
            Y = (ye[i] * lap + sp.diags(d[i])).tocsc()
            out[i] = np.abs(spla.splu(Y).solve(rhs)[probes, np.arange(P)])
    return out


def exact_worst(mask, entries, probes, occ, cfg: PDNConfig, par, n_refine=3, iters=14):
    """Worst ratio max_{f, probe} |Z|/Zt on the dense check grid + refinement of the top peaks.
    par [1, 3] target parameters. Returns (worst ratio, f at worst, (grid f, ratio curve))."""
    f = cfg.check_freqs()
    zt = target_at(np.asarray(par).reshape(1, 3), f)[0]
    r = (_probe_z_exact(mask, entries, probes, occ, cfg, f) / zt[:, None]).max(1)
    best, fbest = r.max(), f[r.argmax()]
    ratio = lambda x: (_probe_z_exact(mask, entries, probes, occ, cfg, [np.exp(x)]) /
                       target_at(np.asarray(par).reshape(1, 3), [np.exp(x)])[0][:, None]).max()
    i = np.flatnonzero((r[1:-1] > r[:-2]) & (r[1:-1] >= r[2:])) + 1
    for j in i[np.argsort(-r[i])[:n_refine]]:
        a, b = np.log(f[j - 1]), np.log(f[j + 1])
        knee = par[0][1] if np.ndim(par) == 2 else par[1]
        if np.isfinite(knee) and a < np.log(knee) < b:     # the target jumps here: stay on the grid
            continue
        c, e = b - GOLD * (b - a), a + GOLD * (b - a)
        rc, re = ratio(c), ratio(e)
        for _ in range(iters):
            if rc > re:
                b, e, re = e, c, rc
                c = b - GOLD * (b - a)
                rc = ratio(c)
            else:
                a, c, rc = c, e, re
                e = a + GOLD * (b - a)
                re = ratio(e)
        x, rx = (c, rc) if rc > re else (e, re)
        if rx > best:
            best, fbest = rx, float(np.exp(x))
    return float(best), float(fbest), (f, r)


def _inst_data(batch, b):
    mask = batch.valid[b].view(batch.H, batch.W).cpu().numpy()
    ent = torch.nonzero(batch.entry[b]).squeeze(-1).cpu().numpy()
    return mask, ent, batch.probes[b].cpu().numpy()


@torch.no_grad()
def certify(batch, inst, occ, cfg: PDNConfig, tol=1e-3, k_peaks=3, rel=3e-3, span=3):
    """Exact worst log10(|Z|/Zt) per episode, plus the frequencies to add to the working grid:
    around the worst point and the k highest violating peaks of each violated episode, a cluster
    f * exp(rel * j), |j| <= span. A resonance with Q ~ 25 moves by a fraction of its 4% width when
    decaps are pruned, so a single point keeps missing the shifted peak."""
    inst_np, occ_np = inst.cpu().numpy(), occ.cpu().numpy()
    worst, fworst = np.empty(len(inst_np)), np.empty(len(inst_np))
    add = [[] for _ in range(batch.B)]                  # per instance
    for e, b in enumerate(inst_np):
        mask, ent, P = _inst_data(batch, b)
        w, fw, (f, r) = exact_worst(mask, ent, P, occ_np[e], cfg, batch.zt_par[b:b + 1])
        worst[e], fworst[e] = np.log10(w), fw
        if worst[e] > tol:
            i = np.flatnonzero((r[1:-1] > r[:-2]) & (r[1:-1] >= r[2:]) & (r[1:-1] > 10 ** tol)) + 1
            for fp in [fw] + list(f[i[np.argsort(-r[i])[:k_peaks]]]):
                add[b] += list(fp * np.exp(rel * np.arange(-span, span + 1)))
    return worst, fworst, add


def grid_worst(batch, inst, occ, cfg):
    """Worst log10(|Z|/Zt) on the current working grid (what the environment believes)."""
    from .sim import probe_impedance
    batch.bank.ensure(inst, torch.where(occ >= 0, torch.arange(batch.n, device=occ.device), -1))
    z = probe_impedance(batch, inst, occ, batch.zd()[inst])
    lv = torch.log10(z.abs() + 1e-30) - torch.log10(batch.Zt[inst])[..., None]
    return lv.amax((1, 2)).cpu().numpy()


def cutting_plane(method, batch, cfg, max_rounds=12, tol=1e-3, gap_tol=np.log10(1.05), rel=3e-3, span=3, log=None):
    """method(batch) -> dict(occ, feasible, ...). Re-run on a growing grid until every solution the
    method calls feasible is (a) certified on the dense check and (b) its grid worst is within
    gap_tol of the certified worst, so the margins the method sees are also right.
    Missing points are added around the certified worst of (a) violated and (b) under-resolved
    solutions. Returns (result, info)."""
    info = dict(rounds=0, added=0)
    inst = torch.arange(batch.B, device=batch.valid.device)
    for rnd in range(max_rounds + 1):
        res = method(batch)
        worst, fw, add = certify(batch, inst, res["occ"], cfg, tol, rel=rel, span=span)
        grid = grid_worst(batch, inst, res["occ"], cfg)
        claimed = res["feasible"].cpu().numpy()
        bad = claimed & (worst > tol)
        gap = claimed & ~bad & (worst - grid > gap_tol)
        for e in np.flatnonzero(gap):
            add[int(inst[e])] += list(fw[e] * np.exp(rel * np.arange(-span, span + 1)))
        info.update(rounds=rnd, worst=worst, grid=grid, claimed=claimed, F=batch.freqs.shape[1])
        if log:
            log(f"  round {rnd}: grid F={batch.freqs.shape[1]} claimed {claimed.sum()}/{len(claimed)} "
                f"false-feasible {bad.sum()} under-resolved {gap.sum()}")
        if not (bad.any() or gap.any()) or rnd == max_rounds:
            break
        added = batch.add_freqs(add)
        info["added"] += added
        if added == 0:
            break
    return res, info
