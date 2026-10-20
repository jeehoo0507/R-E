"""Decap placement environment (min decap count s.t. |Z_probe(f)| <= Zt(f)).

State = the decap type on every cell (occ = -1: empty). With S the placed cells and
D = diag(decap impedances), the loaded network is (Woodbury)
    Z_S = Z0 - Z0[:,S] (D + Z0[S,S])^-1 Z0[S,:]
so only the columns Z0[:, S + probes] and diag(Z0) are needed, never the full Z0.

Each single action (add / remove / change type at cell j) changes the admittance at one cell by dy:
    Z' = Z - Z[:,j] Z[j,:] dy / (1 + dy Z_jj)
so the effect of every candidate action on the probe impedances is evaluated in closed form from
Z_S[:, probes] and diag(Z_S). All solves run in complex128.

Episode rules (outer loop):
  target not met -> only "add (cell, type)" is allowed
  target met     -> "remove" or "change type" of a placed decap, only if the target stays met
                    (a changed-away type is tabu on that cell), or "stop"
  the episode ends on stop, when nothing is allowed, or after max_steps.
"""
import numpy as np
import torch

from .pdn import PDNBatch, PDNConfig

EPS = 1e-30
FEAS_TOL = 1e-6   # target met if max over (f, probe) of log10(|Z| / Zt) <= FEAS_TOL
LOG_CLIP = 6.0


def placed_list(occ):
    """occ[E,n] -> S[E,k] cell ids (-1 pad), types[E,k]; (None, None) if nothing is placed."""
    k = int((occ >= 0).sum(1).max()) if occ.numel() else 0
    if k == 0:
        return None, None
    order = torch.sort((occ < 0).to(torch.int8), dim=1, stable=True).indices[:, :k]
    t = occ.gather(1, order)
    return torch.where(t >= 0, order, -1), t.clamp(min=0)


def per_episode(zd, E):
    """Decap impedances as [E, T, F]: zd is shared [T, F] or already per episode [E, T, F]."""
    return zd.expand(E, *zd.shape) if zd.dim() == 2 else zd


def _core(bank, inst, S, types, zd):
    """M = D + Z0[S,S], identity on padding rows/cols. [E,F,k,k]"""
    zd = per_episode(zd, S.shape[0])
    assert zd.shape[0] == S.shape[0], "zd must be per episode [E, T, F] for these episodes"
    E, k, F = S.shape[0], S.shape[1], zd.shape[-1]
    dz = zd.gather(1, types[:, :, None].expand(E, k, F)).permute(0, 2, 1)   # [E,F,k]
    dz = torch.where((S < 0)[:, None, :], torch.ones_like(dz), dz)
    return bank.sub(inst, S, S) + torch.diag_embed(dz)


def _probe_diag(z, probes):
    """z[E,F,n,P] -> z[e,f,probes[e,q],q] as [E,F,P]."""
    E, F, _, P = z.shape
    idx = probes[:, None, :, None].expand(E, F, P, P)
    return torch.diagonal(z.gather(2, idx), dim1=-2, dim2=-1)


def loaded_state(batch: PDNBatch, inst, occ, zd):
    """Loaded network with the current decaps, from scratch (O(k^2 n) per frequency).
    Returns z_ip [E,F,n,P] (every cell to every probe), z_ii [E,F,n] (self), z_pp [E,F,P]."""
    bank, P = batch.bank, batch.probes[inst]
    z_ip, z_ii = bank.cols(inst, P), bank.diag0[inst]
    S, t = placed_list(occ)
    if S is not None:
        A = bank.cols(inst, S)                                           # Z0[:,S]   [E,F,n,k]
        X = torch.linalg.solve(_core(bank, inst, S, t, zd), A.transpose(-1, -2))  # [E,F,k,n]
        Ap = bank.sub(inst, P, S)                                        # Z0[p,S]   [E,F,P,k]
        z_ip = z_ip - (Ap @ X).transpose(-1, -2)
        z_ii = z_ii - (A * X.transpose(-1, -2)).sum(-1)
    return z_ip, z_ii, _probe_diag(z_ip, P)


def probe_impedance(batch: PDNBatch, inst, occ, zd):
    """Only z_pp [E,F,P] (cheap: never touches all n cells). Used by GA and checks."""
    bank, P = batch.bank, batch.probes[inst]
    z = torch.diagonal(bank.sub(inst, P, P), dim1=-2, dim2=-1)
    S, t = placed_list(occ)
    if S is None:
        return z
    Bm = bank.sub(inst, S, P)                                            # Z0[S,p]   [E,F,k,P]
    X = torch.linalg.solve(_core(bank, inst, S, t, zd), Bm)
    return z - (Bm * X).sum(-2)


def violation(z_pp, logZt):
    """lv = log10(|Z|/Zt) [E,F,P]. V = mean positive part (violation area), m = worst margin."""
    lv = torch.log10(z_pp.abs() + EPS) - logZt[..., None]
    return lv.clamp(min=0).mean((1, 2)), lv.amax((1, 2))


def action_effects(z_ip, z_ii, z_pp, occ, zd, logZt):
    """(V, m) after each single action, [E,n,T+1]. Channel t<T: cell holds type t afterwards
    (add on an empty cell, change type on a placed one); channel T: cell emptied (remove)."""
    E, n = occ.shape
    zd = per_episode(zd, E)
    assert zd.shape[0] == E, "zd must be per episode [E, T, F]"
    T, F = zd.shape[1], zd.shape[2]
    zo = zd.gather(1, occ.clamp(min=0)[:, :, None].expand(E, n, F)).permute(0, 2, 1)
    y_old = torch.where((occ >= 0)[:, None, :], 1.0 / zo,
                        torch.zeros((), dtype=zd.dtype, device=zd.device))   # [E,F,n]
    z2 = z_ip ** 2
    lzt = logZt[:, :, None, None]
    Vn, mn = [], []
    for c in range(T + 1):
        y_new = (1.0 / zd[:, c])[:, :, None] if c < T else 0.0
        dy = y_new - y_old
        coef = dy / (1 + dy * z_ii)
        lv = torch.log10((z_pp[:, :, None, :] - z2 * coef[..., None]).abs() + EPS) - lzt   # [E,F,n,P]
        Vn.append(lv.clamp(min=0).mean((1, 3)))
        mn.append(lv.amax((1, 3)))
    return torch.stack(Vn, -1), torch.stack(mn, -1)


K_MODES = 8   # mode tokens per state (MoTT)


def mott_dims(T):
    """(node, mode, edge, decap) feature sizes of the packed MoTT input."""
    return 4 + T + 3 + 2 * (T + 1), 4, 2 + T, 4


def feat_dim(cfg: PDNConfig, mode: str):
    T, nf = cfg.n_types, cfg.n_feat_freq
    if mode == "mott":
        dn, dm, de, dc = mott_dims(T)
        return dn + K_MODES * (de + dm) + T * dc
    glob = 2 * nf + 3
    return 4 + T + glob + (2 * nf + 2 * (T + 1) if mode == "phys" else 4)


class Env:
    """Batched episodes. inst[E] maps each episode to an instance of `batch`.

    The loaded state (z_ip, z_ii) is updated by one rank-1 step per action, O(n k + k^3) per
    frequency instead of O(n k^2) for a full recompute. It is recomputed from scratch every
    `resync` steps to stop round-off drift."""

    def __init__(self, batch: PDNBatch, inst, cfg: PDNConfig, occ0=None, max_steps=None,
                 allow_prune=True, feat_mode="phys", resync=64):
        self.b, self.inst, self.cfg = batch, inst, cfg
        dev = inst.device
        self.E, self.n, self.T = inst.shape[0], batch.n, cfg.n_types
        self.zd = batch.zd(dev)[inst]                # [E, T, F] on each instance's working grid
        self.logZt = torch.log10(batch.Zt[inst])
        self.placeable = batch.placeable()[inst]
        self.valid = batch.valid[inst]
        self.occ = torch.full((self.E, self.n), -1, dtype=torch.long, device=dev) if occ0 is None else occ0.clone()
        self.tabu = torch.zeros(self.E, self.n, self.T, dtype=torch.bool, device=dev)
        self.done = torch.zeros(self.E, dtype=torch.bool, device=dev)
        self.steps = torch.zeros(self.E, dtype=torch.long, device=dev)
        self.max_steps = max_steps or 2 * int(self.placeable.sum(1).max()) + 8
        self.allow_prune, self.feat_mode = allow_prune, feat_mode
        self.stop_idx = self.n * (self.T + 1)
        self.fidx = torch.from_numpy(cfg.feat_freq_idx()).to(dev)
        f = cfg.freqs()[cfg.feat_freq_idx()]
        self.logf_f1 = torch.from_numpy(np.log10(f / cfg.f1(batch.H, batch.W))).float().to(dev)
        self.feas = torch.zeros(self.E, dtype=torch.bool, device=dev)
        self.V = torch.zeros(self.E, dtype=torch.float64, device=dev)
        self.resync = resync
        batch.bank.ensure(inst, torch.where(self.occ >= 0, torch.arange(self.n, device=dev), -1))
        self._grid()
        self._sync()

    def _grid(self):
        """Per-episode working-grid frequencies and their sorted order (for peak finding)."""
        self.freqs_t = torch.from_numpy(self.b.freqs).to(self.inst.device)[self.inst]       # [E,F]
        self.forder = self.freqs_t.argsort(1)

    def count(self):
        return (self.occ >= 0).sum(1)

    def refresh(self):
        """Call after batch.add_freqs(): rebuild everything that depends on the frequency grid.
        Appended points go after the base grid, so feature frequencies (self.fidx) do not move."""
        self.zd = self.b.zd(self.inst.device)[self.inst]
        self.logZt = torch.log10(self.b.Zt[self.inst])
        self._grid()
        self.b.bank.ensure(self.inst, torch.where(self.occ >= 0, torch.arange(self.n, device=self.inst.device), -1))
        self._sync()

    def _sync(self):
        self.z_ip, self.z_ii, _ = loaded_state(self.b, self.inst, self.occ, self.zd)
        self.since_sync = 0

    def _rank1(self, e, node, y_new):
        """Change the admittance at cell node[i] of episode e[i] to y_new[i] ([Ea,F]; 0 = empty).
        The loaded column Z_S[:, j] is rebuilt from exact Z0 columns (k x k solve) every time:
        updating it recursively instead amplifies round-off at every step."""
        b, bank = self.b, self.b.bank
        inst = self.inst[e]
        Ea, F = e.shape[0], self.z_ii.shape[1]
        cur = self.occ[e, node]
        zero = torch.zeros((), dtype=torch.complex128, device=e.device)
        y_old = torch.where((cur >= 0)[:, None], 1.0 / self.zd[e, cur.clamp(min=0)], zero)
        dy = y_new - y_old
        bank.ensure(inst, node[:, None])
        col = bank.cols(inst, node[:, None])[..., 0]                          # Z0[:, j]  [Ea,F,n]
        S, t = placed_list(self.occ[e])
        if S is not None:
            r = bank.sub(inst, S, node[:, None])                               # Z0[S, j]  [Ea,F,k,1]
            x = torch.linalg.solve(_core(bank, inst, S, t, self.zd[e]), r)
            col = col - (bank.cols(inst, S) @ x)[..., 0]                       # Z_S[:, j]
        zjj = col.gather(2, node[:, None, None].expand(Ea, F, 1))[..., 0]
        c = dy / (1 + dy * zjj)
        P = b.probes[inst]
        col_p = col.gather(2, P[:, None, :].expand(Ea, F, P.shape[1]))
        self.z_ip[e] = self.z_ip[e] - col[..., None] * (col_p * c[..., None])[:, :, None, :]
        self.z_ii[e] = self.z_ii[e] - col ** 2 * c[..., None]

    def observe(self, feats=True):
        z_ip, z_ii = self.z_ip, self.z_ii
        z_pp = _probe_diag(z_ip, self.b.probes[self.inst])
        V, m = violation(z_pp, self.logZt)
        feas = m <= FEAS_TOL
        Vn, mn = action_effects(z_ip, z_ii, z_pp, self.occ, self.zd, self.logZt)
        self.feas, self.V = feas, V
        T = self.T
        empty = (self.occ < 0) & self.placeable
        placed = self.occ >= 0
        legal = torch.zeros(self.E, self.n, T + 1, dtype=torch.bool, device=self.occ.device)
        legal[..., :T] = (~feas)[:, None, None] & empty[..., None]
        if self.allow_prune:
            keep = mn <= 0
            other = torch.arange(T, device=self.occ.device) != self.occ[..., None]
            legal[..., :T] |= feas[:, None, None] & placed[..., None] & keep[..., :T] & ~self.tabu & other
            legal[..., T] = feas[:, None] & placed & keep[..., T]
        mask = torch.cat([legal.reshape(self.E, -1), feas[:, None]], 1)
        self.done |= (self.steps >= self.max_steps) | ~mask.any(1)
        obs = dict(mask=mask, V=V, m=m, feas=feas, Vn=Vn, mn=mn)
        if feats:
            obs["x"] = self.features(z_ip, z_ii, z_pp, V, Vn, mn, feas)
        return obs

    def step(self, a):
        """a[E] flat action index (cell * (T+1) + channel, or stop_idx). Ignored for done episodes."""
        act = ~self.done
        stop = act & (a == self.stop_idx)
        e = torch.nonzero(act & ~stop).squeeze(-1)
        node, ch = a[e] // (self.T + 1), a[e] % (self.T + 1)
        if e.numel():
            y_new = torch.where((ch < self.T)[:, None], 1.0 / self.zd[e, ch.clamp(max=self.T - 1)],
                                torch.zeros((), dtype=torch.complex128, device=a.device))
            self._rank1(e, node, y_new)
        cur = self.occ[e, node]
        sw = (cur >= 0) & (ch < self.T)
        self.tabu[e[sw], node[sw], cur[sw]] = True
        self.occ[e, node] = torch.where(ch < self.T, ch, -1)
        self.steps[act] += 1
        self.done |= stop
        self.since_sync += 1
        if self.since_sync >= self.resync:
            self._sync()

    def features(self, z_ip, z_ii, z_pp, V, Vn, mn, feas):
        """[E,n,D] float32. Physics mode has no coordinates: everything is a log ratio to the target,
        so nothing in the input is tied to the board size."""
        if self.feat_mode == "mott":
            return self.features_mott(z_ip, z_ii, z_pp, V, Vn, mn, feas)
        b, inst, E, n, T = self.b, self.inst, self.E, self.n, self.T
        fi = self.fidx
        lzt = self.logZt[:, fi]                                               # [E,nf]
        vf = self.valid.float()
        flags = torch.stack([vf, b.keepout[inst].float(), b.probe_mask()[inst].float(),
                             b.entry[inst].float()], -1)
        occ_oh = torch.zeros(E, n, T, device=vf.device)
        pl = self.occ >= 0
        occ_oh[pl] = torch.nn.functional.one_hot(self.occ[pl], T).float()
        lv = (torch.log10(z_pp[:, fi].abs() + EPS) - lzt[..., None]).amax(-1)  # [E,nf]
        glob = torch.cat([lv.clamp(-LOG_CLIP, LOG_CLIP).float(), self.logf_f1.expand(E, -1),
                          feas.float()[:, None], (1 - self.steps.float() / self.max_steps)[:, None],
                          (pl.sum(1).float() / vf.sum(1))[:, None]], 1)
        parts = [flags, occ_oh, glob[:, None, :].expand(E, n, glob.shape[1])]
        if self.feat_mode == "phys":
            zii = torch.log10(z_ii[:, fi].abs() + EPS) - lzt[..., None]
            zip_ = torch.log10(z_ip[:, fi].abs() + EPS).amax(-1) - lzt[..., None]
            dV = (V[:, None, None] - Vn).float()
            parts += [zii.permute(0, 2, 1).clamp(-LOG_CLIP, LOG_CLIP).float(),
                      zip_.permute(0, 2, 1).clamp(-LOG_CLIP, LOG_CLIP).float(),
                      dV.clamp(-LOG_CLIP, LOG_CLIP), mn.clamp(-3, 3).float()]
        else:  # coordinate baseline (Park et al. / DevFormer style input)
            H, W = b.H, b.W
            ar = torch.arange(n, device=vf.device)
            y, x = (ar // W).float() / max(H - 1, 1), (ar % W).float() / max(W - 1, 1)
            p = b.probes[inst][:, 0]
            py, px = (p // W).float() / max(H - 1, 1), (p % W).float() / max(W - 1, 1)
            parts += [torch.stack([x.expand(E, n), y.expand(E, n),
                                   x[None] - px[:, None], y[None] - py[:, None]], -1)]
        return torch.cat(parts, -1) * vf[..., None]

    def features_mott(self, z_ip, z_ii, z_pp, V, Vn, mn, feas):
        """Packed MoTT input [E, n, D] (see mott_dims): no coordinates, no fixed frequencies.

        node  : flags, placed type, (feasible, steps left, fraction placed), exact one-step effect of
                every action on the working grid (V - V', worst margin after) = the physics prior
        modes : the K highest peaks of the loaded max_probe |Z_pp| / Zt on the sorted working grid:
                log10(f_k / f_c), peak level, violating, valid
        edges : per (cell, mode): coupling g = |Z_ip| / sqrt(|Z_pp| |Z_ii|) at the peak (single-mode
                residue coherence), |Z_ii| / Zt, and |Z_pp| / Zt after adding each type, at that peak
        decaps: log SRF / f_c and ESL, ESR, 1/(wC) normalised by Zt at f_c
        Modes and decaps are per episode; they are repeated on every cell so the tensor stays [E,n,D]."""
        b, inst, E, n, T = self.b, self.inst, self.E, self.n, self.T
        dev = z_ip.device
        K = min(K_MODES, z_pp.shape[1])
        lvp = torch.log10(z_pp.abs() + EPS) - self.logZt[..., None]                    # [E,F,P]
        lv, pk = lvp.max(-1)
        lvs = lv.gather(1, self.forder)
        ninf = torch.full_like(lvs[:, :1], -float("inf"))
        peak = (lvs >= torch.cat([ninf, lvs[:, :-1]], 1)) & (lvs >= torch.cat([lvs[:, 1:], ninf], 1))
        top, j = torch.where(peak, lvs, -float("inf")).topk(K, 1)
        mvalid = torch.isfinite(top)
        fk = self.forder.gather(1, j)                                                   # grid index [E,K]
        pr = pk.gather(1, fk)                                                           # worst probe
        P = z_pp.shape[2]
        zpp = z_pp.gather(1, fk[..., None].expand(E, K, P)).gather(2, pr[..., None])[..., 0]          # [E,K]
        zip_ = z_ip.gather(1, fk[:, :, None, None].expand(E, K, n, P))
        zip_ = zip_.gather(3, pr[:, :, None, None].expand(E, K, n, 1))[..., 0]                         # [E,K,n]
        zii = z_ii.gather(1, fk[..., None].expand(E, K, n))                                             # [E,K,n]
        lzt = self.logZt.gather(1, fk)                                                                  # [E,K]
        zdk = self.zd.gather(2, fk[:, None, :].expand(E, T, K))                                         # [E,T,K]
        g = zip_.abs() / torch.sqrt(zpp.abs()[..., None] * zii.abs() + EPS)
        ed = [torch.log10(g + 1e-6), torch.log10(zii.abs() + EPS) - lzt[..., None]]
        for t in range(T):
            after = zpp[..., None] - zip_ ** 2 / (zii + zdk[:, t, :, None])
            ed.append(torch.log10(after.abs() + EPS) - lzt[..., None])
        edge = torch.stack(ed, -1).clamp(-LOG_CLIP, LOG_CLIP).float() * mvalid[..., None, None]       # [E,K,n,De]
        edge = edge.permute(0, 2, 1, 3).reshape(E, n, -1)
        cfg = self.b.cfg                               # the batch's own physics (decap library, band)
        fc = float(np.sqrt(cfg.f_min * cfg.f_max))
        fkv = self.freqs_t.gather(1, fk)
        mode = torch.stack([torch.log10(fkv / fc), top.clamp(-LOG_CLIP, LOG_CLIP), (top > FEAS_TOL).double(),
                            mvalid.double()], -1).float()
        mode = torch.where(mvalid[..., None], mode, torch.zeros_like(mode))
        if K < K_MODES:
            mode = torch.cat([mode, mode.new_zeros(E, K_MODES - K, mode.shape[-1])], 1)
            edge = torch.cat([edge, edge.new_zeros(E, n, (K_MODES - K) * (2 + T))], 2)
        ztc = 10 ** self.logZt.median(1).values                                                       # [E]
        wc = 2 * np.pi * fc
        dec = []
        for c, esl, esr in cfg.decaps:
            srf = 1.0 / (2 * np.pi * np.sqrt(esl * c))
            dec.append(torch.stack([torch.full_like(ztc, np.log10(srf / fc)), torch.log10(wc * esl / ztc),
                                    torch.log10(esr / ztc), torch.log10(1.0 / (wc * c) / ztc)], -1))
        dec = torch.stack(dec, 1).float()                                                               # [E,T,4]
        vf = self.valid.float()
        flags = torch.stack([vf, b.keepout[inst].float(), b.probe_mask()[inst].float(), b.entry[inst].float()], -1)
        occ_oh = torch.zeros(E, n, T, device=dev)
        pl = self.occ >= 0
        occ_oh[pl] = torch.nn.functional.one_hot(self.occ[pl], T).float()
        glob = torch.stack([feas.float(), 1 - self.steps.float() / self.max_steps, pl.sum(1).float() / vf.sum(1)], -1)
        dV = (V[:, None, None] - Vn).float().clamp(-LOG_CLIP, LOG_CLIP)
        node = torch.cat([flags, occ_oh, glob[:, None, :].expand(E, n, 3), dV, mn.clamp(-3, 3).float()], -1) * vf[..., None]
        return torch.cat([node, edge, mode.reshape(E, 1, -1).expand(E, n, -1), dec.reshape(E, 1, -1).expand(E, n, -1)], -1)
