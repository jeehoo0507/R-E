"""Size-agnostic grid GNN policy for add / remove / change-type / stop.

One block = local message passing on the PDN grid (masked 3x3 neighbour aggregation: neighbour
            weights depend only on relative position, so the same weights work on any board size)
          + global mean-pool context (cheap, size-free)
          + optional global self-attention (no positional encoding)
          + MLP.
Long-range physics does not have to travel through the layers: the node features already hold
exact loaded-network quantities from the classical solver (see sim.Env.features).
Outputs: per-cell logits over T+1 channels (type t / remove) and one stop logit.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint


def masked_mean(h, vm):
    return (h * vm).sum(1, keepdim=True) / vm.sum(1, keepdim=True).clamp(min=1)


class GridMessage(nn.Module):
    """Depthwise 3x3 neighbour aggregation: out_i = b + sum_o w_o * h_{i+o} over the 9 offsets o.
    Same as a depthwise Conv2d, written with shifts because CPU depthwise conv is very slow."""

    def __init__(self, d):
        super().__init__()
        self.w = nn.Parameter(torch.randn(9, d) / 3)
        self.b = nn.Parameter(torch.zeros(d))

    def forward(self, x, H, W):
        E, N, d = x.shape
        xp = F.pad(x.view(E, H, W, d), (0, 0, 1, 1, 1, 1))
        out = self.b.expand(E, H, W, d)
        for o, (dy, dx) in enumerate((a, b) for a in range(3) for b in range(3)):
            out = out + xp[:, dy:dy + H, dx:dx + W] * self.w[o]
        return out.reshape(E, N, d)


class Block(nn.Module):
    def __init__(self, d, heads, glob=False):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.dw = GridMessage(d)
        self.pw = nn.Linear(d, d)
        self.pool = nn.Linear(d, d)
        self.attn = nn.MultiheadAttention(d, heads, batch_first=True) if glob else None
        if glob:
            self.n2 = nn.LayerNorm(d)
        self.n3 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, h, H, W, valid):
        E, N, d = h.shape
        vm = valid[..., None].to(h.dtype)
        x = self.n1(h) * vm
        loc = self.dw(x, H, W)
        h = h + (self.pw(F.gelu(loc)) + self.pool(masked_mean(x, vm))) * vm
        if self.attn is not None:
            x = self.n2(h)
            x, _ = self.attn(x, x, x, key_padding_mask=~valid, need_weights=False)
            h = h + x * vm
        return h + self.mlp(self.n3(h)) * vm


class GridPolicy(nn.Module):
    def __init__(self, in_dim, n_types, d=128, layers=6, heads=8, glob=False, prior=False):
        super().__init__()
        # prior: add the same physics prior as MoTT (learned weights on the exact one-step effects,
        # the last 2(T+1) phys features), so a comparison isolates the representation
        self.prior = prior
        if prior:
            self.w_dv = nn.Parameter(torch.full((n_types + 1,), 20.0))
            self.w_mn = nn.Parameter(torch.full((n_types + 1,), 3.0))
        self.embed = nn.Sequential(nn.Linear(in_dim, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([Block(d, heads, glob) for _ in range(layers)])
        self.node_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, n_types + 1))
        self.stop_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.grad_ckpt = False

    def forward(self, x, valid, H, W):
        """x[E,n,D] -> node logits [E,n,T+1], stop logit [E] (unmasked)."""
        h = self.embed(x)
        for blk in self.blocks:
            if self.grad_ckpt and self.training and torch.is_grad_enabled():
                h = checkpoint(blk, h, H, W, valid, use_reentrant=False)
            else:
                h = blk(h, H, W, valid)
        vm = valid[..., None].to(h.dtype)
        g = masked_mean(h, vm)
        mx = h.masked_fill(~valid[..., None], -1e4).amax(1, keepdim=True)
        node = self.node_head(torch.cat([h, g.expand_as(h)], -1))
        if self.prior:
            T1 = node.shape[-1]
            node = node + self.w_dv * x[..., -2 * T1:-T1] - self.w_mn * x[..., -T1:]
        stop = self.stop_head(torch.cat([g, mx], -1))[:, 0, 0]
        return node, stop


# ---------------------------------------------------------------- MoTT

class BiasedAttention(nn.Module):
    """Multi-head attention with an additive per-(query, key, head) bias and a key mask."""

    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.q, self.k, self.v, self.o = (nn.Linear(d, d) for _ in range(4))

    def forward(self, xq, xk, bias=None, kmask=None, vadd=None):
        """xq[E,Lq,d], xk[E,Lk,d], bias[E,h,Lq,Lk], kmask[E,Lk] (True = keep),
        vadd[E,Lq,Lk,d]: extra per-pair value (edge message) added to the keys' values."""
        E, Lq, d = xq.shape
        Lk = xk.shape[1]
        q = self.q(xq).view(E, Lq, self.h, self.dh).transpose(1, 2)
        k = self.k(xk).view(E, Lk, self.h, self.dh).transpose(1, 2)
        v = self.v(xk).view(E, Lk, self.h, self.dh).transpose(1, 2)
        s = q @ k.transpose(-1, -2) / self.dh ** 0.5
        if bias is not None:
            s = s + bias
        if kmask is not None:
            s = s.masked_fill(~kmask[:, None, None, :], -1e4)
        a = torch.softmax(s, -1)                                                  # [E,h,Lq,Lk]
        out = (a @ v).transpose(1, 2).reshape(E, Lq, d)
        if vadd is not None:
            am = a.mean(1)                                                        # [E,Lq,Lk]
            out = out + (am[..., None] * vadd).sum(2)
        return self.o(out)


class MoTTBlock(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.nm1, self.nm2, self.nm3, self.np1, self.nc = (nn.LayerNorm(d) for _ in range(5))
        self.mode_self = BiasedAttention(d, heads)
        self.mode_from_port = BiasedAttention(d, heads)
        self.port_from_mode = BiasedAttention(d, heads)
        self.dec_from_mode = BiasedAttention(d, heads)
        self.mlp_m = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))
        self.mlp_p = nn.Sequential(nn.Linear(d, 2 * d), nn.GELU(), nn.Linear(2 * d, d))

    def forward(self, m, u, c, bias, ev, mmask, pmask):
        """m modes [E,K,d], u ports [E,n,d], c decaps [E,T,d]; bias [E,h,n,K]; ev edge values [E,n,K,d]."""
        x = self.nm1(m)
        m = m + self.mode_self(x, x, kmask=mmask)
        m = m + self.mode_from_port(self.nm2(m), self.np1(u), bias=bias.transpose(-1, -2), kmask=pmask)
        m = m + self.mlp_m(self.nm3(m))
        u = u + self.port_from_mode(self.np1(u), self.nm1(m), bias=bias, kmask=mmask, vadd=ev)
        u = u + self.mlp_p(self.np1(u))
        c = c + self.dec_from_mode(self.nc(c), self.nm1(m), kmask=mmask)
        return m, u, c


class MoTT(nn.Module):
    """Modal-Token Transformer policy. Tokens: resonance modes (K), cells (n), decap types (T).
    Cells talk only through the modes (no cell-cell attention): O(n K) per layer.
    Output = physics prior (learned per-channel weights on the exact one-step effects) + learned
    residual, so the untrained policy already behaves like a one-step greedy and learning only
    has to correct it (the proxy experiment showed a free MLP loses the transfer of the physics score)."""

    def __init__(self, n_types, k_modes, dims, d=64, layers=3, heads=4):
        super().__init__()
        self.T, self.K = n_types, k_modes
        self.dn, self.dm, self.de, self.dc = dims
        self.embed_u = nn.Sequential(nn.Linear(self.dn, d), nn.GELU(), nn.Linear(d, d))
        self.embed_m = nn.Sequential(nn.Linear(self.dm, d), nn.GELU(), nn.Linear(d, d))
        self.embed_c = nn.Sequential(nn.Linear(self.dc, d), nn.GELU(), nn.Linear(d, d))
        self.edge_bias = nn.Sequential(nn.Linear(self.de, d), nn.GELU(), nn.Linear(d, heads))
        self.edge_val = nn.Sequential(nn.Linear(self.de, d), nn.GELU(), nn.Linear(d, d))
        self.blocks = nn.ModuleList([MoTTBlock(d, heads) for _ in range(layers)])
        self.add_head = nn.Sequential(nn.Linear(3 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.rm_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.stop_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        T = n_types
        self.w_dv = nn.Parameter(torch.full((T + 1,), 20.0))       # prior: violation reduction
        self.w_mn = nn.Parameter(torch.full((T + 1,), 3.0))        # prior: worst margin after
        self.grad_ckpt = False

    def forward(self, x, valid, H, W):
        E, n, _ = x.shape
        T, K = self.T, self.K
        o = 0
        node = x[..., o:o + self.dn]; o += self.dn
        edge = x[..., o:o + K * self.de].view(E, n, K, self.de); o += K * self.de
        mode = x[:, 0, o:o + K * self.dm].view(E, K, self.dm); o += K * self.dm
        dec = x[:, 0, o:o + T * self.dc].view(E, T, self.dc)
        mmask = mode[..., 3] > 0.5
        mmask = mmask | ~mmask.any(1, keepdim=True)              # keep softmax defined when no peak
        u, m, c = self.embed_u(node), self.embed_m(mode), self.embed_c(dec)
        bias = self.edge_bias(edge).permute(0, 3, 1, 2)          # [E,h,n,K]
        ev = self.edge_val(edge)                                  # [E,n,K,d]
        for blk in self.blocks:
            m, u, c = blk(m, u, c, bias, ev, mmask, valid)
        uu = u[:, :, None, :].expand(E, n, T, u.shape[-1])
        cc = c[:, None, :, :].expand(E, n, T, c.shape[-1])
        add = self.add_head(torch.cat([uu, cc, uu * cc], -1))[..., 0]          # [E,n,T]
        rm = self.rm_head(u)                                                    # [E,n,1]
        nb = 4 + T + 3
        dV, mn = node[..., nb:nb + T + 1], node[..., nb + T + 1:nb + 2 * (T + 1)]
        prior = self.w_dv * dV - self.w_mn * mn
        logits = torch.cat([add, rm], -1) + prior
        vm = valid[..., None].to(u.dtype)
        g = (u * vm).sum(1) / vm.sum(1).clamp(min=1)
        mg = (m * mmask[..., None]).sum(1) / mmask.sum(1, keepdim=True).clamp(min=1)
        stop = self.stop_head(torch.cat([g, mg], -1))[:, 0]
        return logits, stop
