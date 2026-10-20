"""Problem bank: instances whose working grids were refined by verify.cutting_plane, saved so
that later experiments reuse the same instances and grids instead of re-certifying them.

A bank file stores only the instance definition (masks, probes, power entries, target parameters,
grids) plus any method results; the impedance bank is rebuilt on load (exact, deterministic)."""
import pickle
from dataclasses import asdict

import numpy as np
import torch

from .pdn import PDNBatch, PDNConfig, ZBank, build_Y0, target_at


def save_bank(path, batch: PDNBatch, results=None):
    d = dict(H=batch.H, W=batch.W, valid=batch.valid.cpu().numpy(), keepout=batch.keepout.cpu().numpy(),
             probes=batch.probes.cpu().numpy(), entry=batch.entry.cpu().numpy(), zt_par=batch.zt_par,
             freqs=batch.freqs, shapes=batch.shapes, targets=batch.targets, cfg=asdict(batch.cfg),
             results=results or {})
    with open(path, "wb") as f:
        pickle.dump(d, f)


def load_bank(path, device="cpu", cfg=None, dense=None, z_dtype=torch.complex128):
    with open(path, "rb") as f:
        d = pickle.load(f)
    cfg = cfg or PDNConfig(**{k: (tuple(map(tuple, v)) if k == "decaps" else v) for k, v in d["cfg"].items()})
    B, H, W = d["valid"].shape[0], d["H"], d["W"]
    Ys = []
    for b in range(B):
        mask = d["valid"][b].reshape(H, W)
        Ys.append(build_Y0(mask, np.flatnonzero(d["entry"][b]), cfg, d["freqs"][b]))
    if dense is None:
        dense = H * W <= cfg.dense_max_nodes
    probes = torch.from_numpy(d["probes"]).to(device)
    bank = ZBank(Ys, probes, device, dense, z_dtype)
    Zt = np.stack([target_at(d["zt_par"][b:b + 1], d["freqs"][b])[0] for b in range(B)])
    batch = PDNBatch(H, W, torch.from_numpy(d["valid"]).to(device), torch.from_numpy(d["keepout"]).to(device),
                     probes, torch.from_numpy(d["entry"]).to(device), torch.from_numpy(Zt).to(device), bank,
                     d["shapes"], d["targets"], 0.0, freqs=d["freqs"], zt_par=d["zt_par"], cfg=cfg)
    return batch, d["results"]
