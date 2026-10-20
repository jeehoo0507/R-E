"""PDN instance generator and impedance bank.

A power/ground plane pair is a grid of unit cells (lumped RLGC):
  - each valid cell is a node with a shunt C (plane capacitance) and G (dielectric loss)
  - neighbouring valid cells are connected by a series R + L branch
  - power-entry cells (bumps / balls) connect to the ideal VRM source through R_vrm + L_vrm
Y0(f) is sparse. We factorise it once per (instance, frequency) with sparse LU and never build
the full Z(f) = Y0(f)^-1: only the columns that are actually used (probes, placed decaps) and
diag(Z0) are kept in a ZBank. Decaps are diagonal admittances added on top (see sim.py).

Every instance gets a target impedance Zt(f) (flat or step). Its level is set between "all
placeable cells filled with one decap type" (surely feasible) and "no decap" (nothing needed),
so every generated target is feasible. Both levels are measured on the dense check grid
(cfg.n_check points + peak refinement, see verify.py), not on the coarse working grid, because
the coarse grid misses resonance peaks.

Frequency grids: every instance has its own working grid (batch.freqs[b]). It starts as the
cfg.n_freq log points and grows (PDNBatch.add_freqs) where verify.py finds a violation between
grid points. New points are appended, so the first cfg.n_freq entries (and the feature
frequencies) never move. Instances that got fewer new points than others in the same call are
padded with copies of their own last point: a duplicate point changes nothing in the max over
frequencies (feasibility) and only slightly reweights the mean violation.

Physical constants are placeholders: replace them with the values of the board you model
(Park et al. 2203.15722 uses a 375 um HBM unit cell and a 1.055 nF NMOS decap).
"""
from dataclasses import dataclass, field

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from scipy import ndimage

MU0 = 4e-7 * np.pi
EPS0 = 8.854e-12


@dataclass
class PDNConfig:
    cell_size: float = 0.5e-3        # unit cell edge [m]
    diel_thickness: float = 10e-6    # plane-to-plane distance [m]
    eps_r: float = 4.0
    tan_d: float = 0.02
    metal_thickness: float = 2e-6    # [m]
    sigma: float = 5.8e7             # copper [S/m]
    vrm_r: float = 5e-3              # per power-entry cell: package + VRM path [ohm]
    vrm_l: float = 200e-12           # [H]
    f_min: float = 1e7               # target band = whole frequency grid
    f_max: float = 2e10
    n_freq: int = 32
    n_feat_freq: int = 8             # frequencies used as model input features
    # decap library: (C [F], ESL [H], ESR [ohm]) -- placeholder values
    decaps: tuple = ((10e-9, 100e-12, 0.02),
                     (1e-9, 30e-12, 0.05),
                     (100e-12, 10e-12, 0.10))
    dense_max_nodes: int = 900       # boards up to this many cells keep the full Z0 (fast); larger -> sparse LU
    n_check: int = 700               # dense log grid of the exact check (verify.py)
    target_on_check: bool = True     # set target levels from the dense check grid (exact, slower setup)
    seed_peaks: bool = True          # add the bare-board and filled-board peak frequencies to the working grid

    # derived per-cell / per-edge lumped values
    @property
    def c_cell(self):
        return EPS0 * self.eps_r * self.cell_size ** 2 / self.diel_thickness

    @property
    def l_edge(self):
        return MU0 * self.diel_thickness  # one square of plane pair

    @property
    def r_edge(self):
        return 2.0 / (self.sigma * self.metal_thickness)  # two planes, one square

    @property
    def n_types(self):
        return len(self.decaps)

    def freqs(self):
        return np.logspace(np.log10(self.f_min), np.log10(self.f_max), self.n_freq)

    def feat_freq_idx(self):
        return np.linspace(0, self.n_freq - 1, self.n_feat_freq).round().astype(int)

    def f1(self, H, W):
        """First cavity resonance of an open H x W board [Hz] (cell = LC transmission-line segment)."""
        return 1.0 / (2 * max(H, W) * np.sqrt(self.l_edge * self.c_cell))

    def check_freqs(self):
        return np.logspace(np.log10(self.f_min), np.log10(self.f_max), self.n_check)

    def decap_z_np(self, f=None):
        """[T, F] complex128 impedance of each decap type (default: the base grid)."""
        w = 2 * np.pi * (self.freqs() if f is None else np.asarray(f))
        return np.stack([esr + 1j * w * esl + 1.0 / (1j * w * c) for c, esl, esr in self.decaps])

    def decap_z(self, device="cpu", f=None):
        return torch.from_numpy(self.decap_z_np(f)).to(device)


# ---------------------------------------------------------------- shapes

SHAPES = ("rect", "L", "holes")
TARGETS = ("flat", "step")


def _largest_component(mask):
    lab, n = ndimage.label(mask)
    if n <= 1:
        return mask
    sizes = ndimage.sum(mask, lab, range(1, n + 1))
    return lab == (1 + int(np.argmax(sizes)))


def sample_mask(H, W, shape, rng):
    m = np.ones((H, W), dtype=bool)
    if shape == "L":
        h = rng.integers(max(1, H // 3), max(2, H // 2 + 1))
        w = rng.integers(max(1, W // 3), max(2, W // 2 + 1))
        corner = rng.integers(4)
        rs = slice(0, h) if corner < 2 else slice(H - h, H)
        cs = slice(0, w) if corner % 2 == 0 else slice(W - w, W)
        m[rs, cs] = False
    elif shape == "holes":
        for _ in range(rng.integers(1, 4)):
            h = rng.integers(1, max(2, H // 4 + 1))
            w = rng.integers(1, max(2, W // 4 + 1))
            r = rng.integers(1, max(2, H - h))
            c = rng.integers(1, max(2, W - w))
            m[r:r + h, c:c + w] = False
    elif shape != "rect":
        raise ValueError(shape)
    return _largest_component(m)


# ---------------------------------------------------------------- admittance

def lap_grid(mask):
    """4-neighbour graph Laplacian of the valid cells (csr)."""
    H, W = mask.shape
    n = H * W
    idx = np.arange(n).reshape(H, W)
    rows, cols = [], []
    for a, b, ok in ((idx[:, :-1], idx[:, 1:], mask[:, :-1] & mask[:, 1:]),
                     (idx[:-1, :], idx[1:, :], mask[:-1, :] & mask[1:, :])):
        rows += [a[ok], b[ok]]
        cols += [b[ok], a[ok]]
    rows, cols = np.concatenate(rows), np.concatenate(cols)
    A = sp.csr_matrix((np.ones(len(rows)), (rows, cols)), shape=(n, n))
    return (sp.diags(np.asarray(A.sum(1)).ravel()) - A).tocsr()


def build_Y0(mask, entries, cfg: PDNConfig, freqs=None):
    """Bare nodal admittance per frequency: list of F csc complex128 matrices (default: base grid).
    Invalid cells become isolated dummy nodes (Y_ii = 1) so Y stays invertible."""
    lap = lap_grid(mask)
    v = mask.reshape(-1).astype(float)
    Ys = []
    for f in (cfg.freqs() if freqs is None else freqs):
        w = 2 * np.pi * f
        ye = 1.0 / (cfg.r_edge + 1j * w * cfg.l_edge)
        ys = w * cfg.c_cell * cfg.tan_d + 1j * w * cfg.c_cell
        d = ys * v + (1 - v)
        d[entries] += 1.0 / (cfg.vrm_r + 1j * w * cfg.vrm_l)
        Ys.append((ye * lap + sp.diags(d)).tocsc().astype(np.complex128))
    return Ys


# ---------------------------------------------------------------- impedance bank

class ZBank:
    """Columns of Z0(f) = Y0(f)^-1 for every instance of a batch.

    dense  (small boards): the whole Z0 is computed once; every column is available.
    sparse (large boards): one sparse LU per (instance, frequency); a column is solved the first
                           time a node is used (probe, placed decap, GA pool) and cached.
    diag(Z0) is always computed exactly (n solves per frequency in sparse mode) because greedy
    and the policy features need every node's self impedance. It is the expensive part of setup.
    Z0 is complex symmetric, so a column is also a row.
    """

    def __init__(self, Ys, probes, device, dense, z_dtype=torch.complex128):
        self.B, self.F, self.n = len(Ys), len(Ys[0]), Ys[0][0].shape[0]
        self.dense, self.device, self.z_dtype = dense, device, z_dtype
        self.slot = torch.full((self.B, self.n), -1, dtype=torch.long, device=device)
        if dense:
            Yd = torch.from_numpy(np.stack([[Y.toarray() for Y in Yb] for Yb in Ys])).to(device)
            Z = torch.linalg.inv(Yd)
            self.diag0 = torch.diagonal(Z, dim1=-2, dim2=-1).clone()
            self.Zc = Z.to(z_dtype)
            self.slot[:] = torch.arange(self.n, device=device)
            self.lus = None
        else:
            self.lus = [[spla.splu(Y) for Y in Yb] for Yb in Ys]
            self.Zc = torch.zeros(self.B, self.F, self.n, 0, dtype=z_dtype, device=device)
            self.n_cached = [0] * self.B
            self.diag0 = torch.from_numpy(np.stack([self._diag(b) for b in range(self.B)])).to(device)
        self.ensure(torch.arange(self.B, device=device), probes)

    def add_freqs(self, Ys_new):
        """Append frequencies. Ys_new[b] = list of new csc Y0 matrices of instance b (same count for all).
        Every cached column and diag(Z0) is extended, so the bank stays exact on the grown grid."""
        Fn = len(Ys_new[0])
        if Fn == 0:
            return
        if self.dense:
            Yd = torch.from_numpy(np.stack([[Y.toarray() for Y in Yb] for Yb in Ys_new])).to(self.device)
            Z = torch.linalg.inv(Yd)
            self.diag0 = torch.cat([self.diag0, torch.diagonal(Z, dim1=-2, dim2=-1)], 1)
            self.Zc = torch.cat([self.Zc, Z.to(self.z_dtype)], 1)
        else:
            new_lus = [[spla.splu(Y) for Y in Yb] for Yb in Ys_new]
            Zc_new = torch.zeros(self.B, Fn, self.n, self.Zc.shape[-1], dtype=self.z_dtype, device=self.device)
            dg_new = []
            for b in range(self.B):
                dg_new.append(self._diag(b, lus=new_lus[b]))
                c = self.n_cached[b]
                if c:
                    js = torch.nonzero(self.slot[b] >= 0).squeeze(-1)
                    sl = self.slot[b, js]
                    rhs = np.zeros((self.n, len(js)), dtype=np.complex128)
                    rhs[js.cpu().numpy(), np.arange(len(js))] = 1
                    X = torch.from_numpy(np.stack([lu.solve(rhs) for lu in new_lus[b]])).to(self.device)
                    Zc_new[b][..., sl] = X.to(self.z_dtype)
                self.lus[b] += new_lus[b]
            self.Zc = torch.cat([self.Zc, Zc_new], 1)
            self.diag0 = torch.cat([self.diag0, torch.from_numpy(np.stack(dg_new)).to(self.device)], 1)
        self.F += Fn

    def _diag(self, b, chunk=512, lus=None):
        lus = self.lus[b] if lus is None else lus
        out = np.empty((len(lus), self.n), dtype=np.complex128)
        for f, lu in enumerate(lus):
            for j0 in range(0, self.n, chunk):
                j1 = min(self.n, j0 + chunk)
                rhs = np.zeros((self.n, j1 - j0), dtype=np.complex128)
                rhs[np.arange(j0, j1), np.arange(j1 - j0)] = 1
                out[f, j0:j1] = lu.solve(rhs)[np.arange(j0, j1), np.arange(j1 - j0)]
        return out

    def ensure(self, inst, nodes):
        """Make sure columns Z0[:, nodes] are cached. inst[E], nodes[E,k] (-1 = padding)."""
        if self.dense:
            return
        need = {}
        miss = (nodes >= 0) & (self.slot[inst[:, None].expand_as(nodes), nodes.clamp(min=0)] < 0)
        for e, j in torch.nonzero(miss).tolist():
            need.setdefault(int(inst[e]), set()).add(int(nodes[e, j]))
        for b, js in need.items():
            js = sorted(js)
            rhs = np.zeros((self.n, len(js)), dtype=np.complex128)
            rhs[js, np.arange(len(js))] = 1
            X = torch.from_numpy(np.stack([lu.solve(rhs) for lu in self.lus[b]])).to(self.device)
            c0, c1 = self.n_cached[b], self.n_cached[b] + len(js)
            if c1 > self.Zc.shape[-1]:
                grow = max(c1, 2 * self.Zc.shape[-1], 8) - self.Zc.shape[-1]
                pad = torch.zeros(self.B, self.F, self.n, grow, dtype=self.z_dtype, device=self.device)
                self.Zc = torch.cat([self.Zc, pad], -1)
            self.Zc[b, :, :, c0:c1] = X.to(self.z_dtype)
            self.slot[b, js] = torch.arange(c0, c1, device=self.device)
            self.n_cached[b] = c1

    def _slots(self, inst, cols):
        sl = self.slot[inst[:, None].expand_as(cols), cols.clamp(min=0)]
        if bool(((sl < 0) & (cols >= 0)).any()):
            raise RuntimeError("ZBank column used before ensure()")
        return sl.clamp(min=0)

    def cols(self, inst, cols):
        """Z0[:, cols] -> [E, F, n, k] complex128 (zero for padding cols = -1)."""
        sl = self._slots(inst, cols)
        out = self.Zc[inst[:, None], :, :, sl].permute(0, 2, 3, 1).to(torch.complex128)
        return out * (cols >= 0)[:, None, None, :]

    def sub(self, inst, rows, cols):
        """Z0[rows, cols] -> [E, F, r, k] complex128 (zero for padding = -1)."""
        sl = self._slots(inst, cols)
        out = self.Zc[inst[:, None, None], :, rows.clamp(min=0)[:, :, None], sl[:, None, :]]
        out = out.permute(0, 3, 1, 2).to(torch.complex128)
        return out * ((rows >= 0)[:, None, :, None] & (cols >= 0)[:, None, None, :])


# ---------------------------------------------------------------- batch

@dataclass
class PDNBatch:
    H: int
    W: int
    valid: torch.Tensor      # [B, n] bool  cell exists
    keepout: torch.Tensor    # [B, n] bool  cell exists but decap not allowed
    probes: torch.Tensor     # [B, P] long  observation ports (target must hold at each)
    entry: torch.Tensor      # [B, n] bool  power-entry cells (VRM path)
    Zt: torch.Tensor         # [B, F] float64 target impedance [ohm]
    bank: ZBank
    shapes: list = field(default_factory=list)
    targets: list = field(default_factory=list)
    setup_sec: float = 0.0   # time to factorise + build the bank (counted in method timings)
    freqs: np.ndarray = None   # [B, F] working grid per instance (base cfg.n_freq points first, then appended)
    zt_par: np.ndarray = None  # [B, 3] target parameters (L, knee, mult): Zt(f) = L * (mult if f >= knee else 1)
    cfg: PDNConfig = None

    def zd(self, device=None):
        """[B, T, F] decap impedances on each instance's working grid."""
        z = np.stack([self.cfg.decap_z_np(f) for f in self.freqs])
        return torch.from_numpy(z).to(device or self.valid.device)

    def add_freqs(self, f_new, min_rel=1e-3):
        """f_new[b] = frequencies to append to instance b (points within min_rel in log of an
        existing or already chosen point are skipped). Returns the number of distinct points added.
        Envs built on this batch must call refresh()."""
        keep = []
        for b in range(self.B):
            have = list(np.log(self.freqs[b]))
            kb = []
            for f in np.unique(np.asarray(f_new[b], dtype=float)):
                if np.min(np.abs(np.array(have) - np.log(f))) > min_rel:
                    kb.append(f)
                    have.append(np.log(f))
            keep.append(kb)
        Fn = max(len(k) for k in keep)
        if Fn == 0:
            return 0
        pad = [np.array(k + [k[-1] if k else self.freqs[b][-1]] * (Fn - len(k))) for b, k in enumerate(keep)]
        Ys_new = []
        for b in range(self.B):
            mask = self.valid[b].view(self.H, self.W).cpu().numpy()
            ent = torch.nonzero(self.entry[b]).squeeze(-1).cpu().numpy()
            Ys_new.append(build_Y0(mask, ent, self.cfg, pad[b]))
        self.bank.add_freqs(Ys_new)
        pad = np.stack(pad)
        self.freqs = np.concatenate([self.freqs, pad], 1)
        zt = np.stack([target_at(self.zt_par[b:b + 1], pad[b])[0] for b in range(self.B)])
        self.Zt = torch.cat([self.Zt, torch.from_numpy(zt).to(self.Zt.device)], 1)
        return sum(len(k) for k in keep)

    @property
    def B(self):
        return self.valid.shape[0]

    @property
    def n(self):
        return self.H * self.W

    @property
    def P(self):
        return self.probes.shape[1]

    def probe_mask(self):
        return torch.zeros_like(self.valid).scatter_(1, self.probes, True)

    def placeable(self):
        return self.valid & ~self.keepout & ~self.probe_mask()


def _probe_z(Ys, probes, extra_diag=None):
    """Self impedance at the probes for every frequency -> [F, P]."""
    n, P = Ys[0].shape[0], len(probes)
    rhs = np.zeros((n, P), dtype=np.complex128)
    rhs[probes, np.arange(P)] = 1
    out = []
    for f, Y in enumerate(Ys):
        if extra_diag is not None:
            Y = (Y + sp.diags(extra_diag[f])).tocsc()
        out.append(spla.splu(Y).solve(rhs)[probes, np.arange(P)])
    return np.stack(out)


def target_at(par, f):
    """par [B, 3] = (L, knee, mult) -> Zt [B, F] at frequencies f."""
    f = np.asarray(f, dtype=float)
    return par[:, :1] * np.where(f[None, :] >= par[:, 1:2], par[:, 2:3], 1.0)


def _peaks(f, y, k=8):
    """Frequencies of the k highest local maxima of y(f)."""
    i = np.flatnonzero((y[1:-1] > y[:-2]) & (y[1:-1] >= y[2:])) + 1
    return f[i[np.argsort(-y[i])[:k]]]


def make_target(Ys, probes, placeable, cfg: PDNConfig, rng, kind, difficulty, mask=None, entries=None):
    """Target parameters (L, knee, mult) and peak frequencies to seed the working grid.
    Zt(f) = L * s(f). s = 1 (flat) or a step up by 2-10x at a random knee frequency.
    L sits between Lref (every placeable cell holds one decap type: feasible by construction)
    and L0 (no decap needed): log L = log Lref + u (log L0 - log Lref), u ~ U(difficulty).
    Smaller u = harder target = more decaps. With cfg.target_on_check both levels come from the
    exact dense check (verify.exact_worst), otherwise from the base grid Ys."""
    f = cfg.freqs()
    knee, mult = np.inf, 1.0
    if kind == "step":
        knee = float(np.exp(rng.uniform(np.log(f[2]), np.log(f[-3]))))
        mult = float(np.exp(rng.uniform(np.log(2), np.log(10))))
    elif kind != "flat":
        raise ValueError(kind)
    unit = np.array([[1.0, knee, mult]])
    seeds = []
    if cfg.target_on_check:
        from .verify import exact_worst
        L0, fw, curve = exact_worst(mask, entries, probes, None, cfg, unit)
        seeds += list(_peaks(*curve, k=3))
        yd_all = []
        for t in range(cfg.n_types):
            occ = np.where(placeable > 0, t, -1)
            Lt, _, curve = exact_worst(mask, entries, probes, occ, cfg, unit)
            yd_all.append(Lt)
            seeds += list(_peaks(*curve, k=1))
        Lref = min(yd_all)
    else:
        s = target_at(unit, f)[0]
        z0 = np.abs(_probe_z(Ys, probes))
        L0 = (z0 / s[:, None]).max()
        yd = 1.0 / cfg.decap_z_np()                                       # [T, F]
        Lref = min((np.abs(_probe_z(Ys, probes, yd[t][:, None] * placeable[None])) / s[:, None]).max()
                   for t in range(cfg.n_types))
    if Lref >= L0:
        L = L0
    else:
        u = rng.uniform(*difficulty)
        L = float(np.exp(np.log(Lref) + u * (np.log(L0) - np.log(Lref))))
    return np.array([L, knee, mult]), seeds


def sample_batch(B, H, W, cfg: PDNConfig, rng, shapes=SHAPES, targets=TARGETS, keepout_max=0.3,
                 n_probe=(1, 3), n_entry=(1, 4), difficulty=(0.02, 0.2), device="cpu",
                 dense=None, z_dtype=torch.complex128) -> PDNBatch:
    """B random instances of one board size. The number of probes is shared within a batch."""
    import time
    n = H * W
    P = int(rng.integers(n_probe[0], n_probe[1] + 1))
    valid = np.zeros((B, n), dtype=bool)
    keepout = np.zeros((B, n), dtype=bool)
    entry = np.zeros((B, n), dtype=bool)
    probes = np.zeros((B, P), dtype=np.int64)
    par, seeds, Ys_all, names, kinds = [], [], [], [], []
    for b in range(B):
        s = shapes[rng.integers(len(shapes))]
        mask = sample_mask(H, W, s, rng)
        cells = np.flatnonzero(mask)
        valid[b] = mask.reshape(-1)
        probes[b] = rng.choice(cells, P, replace=False)
        ent = rng.choice(cells, int(rng.integers(n_entry[0], n_entry[1] + 1)), replace=False)
        entry[b, ent] = True
        rest = np.setdiff1d(cells, probes[b])
        keepout[b, rng.choice(rest, int(rng.uniform(0, keepout_max) * len(rest)), replace=False)] = True
        Ys = build_Y0(mask, ent, cfg)
        placeable = valid[b] & ~keepout[b]
        placeable[probes[b]] = False
        kind = targets[rng.integers(len(targets))]
        p_b, s_b = make_target(Ys, probes[b], placeable.astype(float), cfg, rng, kind, difficulty, mask, ent)
        par.append(p_b)
        seeds.append(s_b)
        Ys_all.append(Ys)
        names.append(s)
        kinds.append(kind)
    t = time.time()
    if dense is None:
        dense = n <= cfg.dense_max_nodes
    probes_t = torch.from_numpy(probes).to(device)
    bank = ZBank(Ys_all, probes_t, device, dense, z_dtype)
    setup = time.time() - t
    par = np.stack(par)
    f = cfg.freqs()
    batch = PDNBatch(H, W, torch.from_numpy(valid).to(device), torch.from_numpy(keepout).to(device),
                     probes_t, torch.from_numpy(entry).to(device),
                     torch.from_numpy(target_at(par, f)).to(device), bank, names, kinds, setup / B,
                     freqs=np.tile(f, (B, 1)), zt_par=par, cfg=cfg)
    if cfg.seed_peaks and any(seeds):
        t = time.time()
        batch.add_freqs(seeds)
        batch.setup_sec += (time.time() - t) / B
    return batch
