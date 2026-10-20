import numpy as np, time
from exp4_lib import *
fs = np.logspace(7, np.log10(2e10), 40)
zd = zd_table(fs)
rng = np.random.default_rng(3)
for N in (10, 20):
    m, probe, cand, sh = instance(N, rng, "rect")
    Z = dense_Z(m, fs)
    z0 = np.abs(Z[:, probe, probe])
    print(f"N={N} probe={probe} bare |Zpp|:", " ".join(f"{f/1e9:.2g}G:{z:.3g}" for f, z in zip(fs[::3], z0[::3])))
    # floor: greedy with big K minimising sum log|Z| (no target) -> how low can we go?
    for fhi in (1e9, 3e9, 1e10, 2e10):
        for zt in (0.5, 0.2, 0.1):
            band = fs <= fhi
            Zt = np.where(band, zt, np.inf)
            t = time.time()
            p, ty, zf = greedy_mincount(Z, probe, cand, zd, Zt, kmax=80, prune=False)
            print(f"  band 10M-{fhi/1e9:g}G Zt={zt}: greedy count={'infeasible(>80)' if p is None else len(p)}"
                  f"  worst |Z|/Zt={np.max(zf[band]/zt):.2f}  ({time.time()-t:.1f}s)", flush=True)
