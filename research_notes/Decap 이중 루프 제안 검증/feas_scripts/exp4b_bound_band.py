"""Exp 4: physics lower bound on decap count vs greedy(+prune) min-count, plus a local-search check
whether greedy count can be beaten by one."""
import time, warnings, numpy as np
from scipy.optimize import milp, LinearConstraint, Bounds
from exp4_lib import *
warnings.filterwarnings("ignore")

fs = np.logspace(7, np.log10(2e10), 40)
zd = zd_table(fs); yd = np.abs(1 / zd)          # [F,T]
Cs = np.array([c for c, _, _ in CFG.decaps]); ESL = np.array([l for _, l, _ in CFG.decaps])
settings = [(1e8, 3e9, 0.2), (1e8, 3e9, 0.15), (3e8, 5e9, 0.25), (3e8, 1e10, 0.5)]


def bounds(m, z0, flo, fhi, zt):
    band = (fs >= flo) & (fs <= fhi)
    flo = fs[band][0]
    cpl = m.sum() * CFG.c_cell
    c_need = 1 / (2 * np.pi * flo * zt)
    lb_c = int(np.ceil(max(0.0, c_need - cpl) / Cs.max()))
    lb_l = int(np.ceil(2 * np.pi * fhi * ESL.min() / zt))           # n parallel ESLs: w*ESL/n <= Zt
    # per-frequency admittance-sum bound (exact if all decaps sat on the probe node), integer program
    need = np.clip(1 / zt - 1 / z0[band], 0, None)
    A = yd[band]                                                     # [Fb,T]
    res = milp(c=np.ones(3), constraints=LinearConstraint(A, lb=need, ub=np.inf),
               integrality=np.ones(3), bounds=Bounds(0, np.inf))
    lb_y = int(round(res.fun)) if res.success else -1
    lb_y_single = int(max(np.ceil(need / A.max(1)).max(), 0))     # per-f, any type, max over f
    return lb_c, lb_l, lb_y_single, lb_y


def local_search_fewer(Z, probe, cand, Zt, k, iters, rng):
    """Try to find a feasible placement with k decaps by random swap hill-climbing on total violation."""
    if k <= 0:
        return np.all(np.abs(Z[:, probe, probe]) <= Zt)
    def cost(p, t):
        z = np.abs(zpp_after(Z, probe, list(p), list(t), zd))
        return np.clip(np.log10(z / Zt), 0, None).sum()
    p = list(rng.choice(cand, k, replace=False)); t = list(rng.integers(3, size=k)); c = cost(p, t)
    for _ in range(iters):
        q, u = p.copy(), t.copy(); j = rng.integers(k)
        if rng.random() < 0.5:
            free = np.setdiff1d(cand, q)
            q[j] = int(rng.choice(free))
        else:
            u[j] = int(rng.integers(3))
        cq = cost(q, u)
        if cq <= c:
            p, t, c = q, u, cq
        if c == 0:
            return True
    return False


rng = np.random.default_rng(7)
print("N,inst,shape,band_GHz,Zt,greedy_raw,greedy_pruned,LB_C,LB_ESL,LB_Y1,LB_Ymilp,ratio_vs_max(LB_C,LB_ESL),ratio_vs_LBYmilp,LS_found_k-1")
rows = []
for N in (10, 20):
    for inst in range(6):
        m, probe, cand, sh = instance(N, rng)
        Z = dense_Z(m, fs)
        z0 = np.abs(Z[:, probe, probe])
        for flo, fhi, zt in settings:
            band = (fs >= flo) & (fs <= fhi)
            Zt = np.where(band, zt, np.inf)
            p_raw, _, _ = greedy_mincount(Z, probe, cand, zd, Zt, kmax=40, prune=False)
            p, ty, zf = greedy_mincount(Z, probe, cand, zd, Zt, kmax=40, prune=True)
            lb_c, lb_l, lb_y1, lb_y = bounds(m, z0, flo, fhi, zt)
            if p is None:
                print(f"{N},{inst},{sh},{flo/1e9:g}-{fhi/1e9:g},{zt},infeasible,infeasible,{lb_c},{lb_l},{lb_y1},{lb_y},-,-,-", flush=True)
                continue
            k = len(p)
            ls = local_search_fewer(Z, probe, cand, Zt, k - 1, 600, rng) if k >= 1 else "-"
            r1 = k / max(lb_c, lb_l, 1); r2 = k / max(lb_y, 1)
            rows.append((N, fhi, zt, k, lb_c, lb_l, lb_y, r1, r2, ls))
            print(f"{N},{inst},{sh},{flo/1e9:g}-{fhi/1e9:g},{zt},{len(p_raw)},{k},{lb_c},{lb_l},{lb_y1},{lb_y},{r1:.2f},{r2:.2f},{ls}", flush=True)
np.save("exp4b_rows.npy", np.array(rows, dtype=object), allow_pickle=True)
r = np.array([(x[7], x[8], x[3] < max(x[4], x[5]), x[3] < x[6]) for x in rows], dtype=float)
print(f"\nratio greedy/max(LB_C,LB_ESL): median {np.median(r[:,0]):.2f}, min {r[:,0].min():.2f}, max {r[:,0].max():.2f}")
print(f"ratio greedy/LB_Ymilp: median {np.median(r[:,1]):.2f}, min {r[:,1].min():.2f}, max {r[:,1].max():.2f}")
print(f"bound violations (greedy < bound): simple {int(r[:,2].sum())}, LB_Y {int(r[:,3].sum())} of {len(r)}")
