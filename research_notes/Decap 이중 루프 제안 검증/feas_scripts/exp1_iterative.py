"""Exp 1: convergence of local / Krylov / multigrid iterations on Y(f) v = e_probe.
usage: python exp1_iterative.py N  -> writes exp1_N{N}.csv"""
import sys, time, warnings, numpy as np, scipy.sparse as sp
from scipy.sparse.linalg import splu, spilu, gmres, bicgstab, LinearOperator, eigs, ArpackNoConvergence
from common import *
warnings.filterwarnings("ignore")

N = int(sys.argv[1]); n = N * N
mask = rect(N); lap = lap_grid(mask)
probe = (N // 2) * N + N // 2
v = CFG.cell_size / np.sqrt(CFG.l_edge * CFG.c_cell); f1 = v / (2 * N * CFG.cell_size)
freqs = sorted(set([1e7, 1e8, 3e8, 1e9, 2e9, 5e9, 1e10, 2e10, 0.5 * f1, 1.5 * f1]))
freqs = [f for f in freqs if f <= 2e10]
rng = np.random.default_rng(1)
dec = random_decaps(mask, max(4, round(0.02 * n)), rng, exclude=[probe])
TOL, CAP, KCAP = 1e-3, 20000, 3000
b = np.zeros(n, complex); b[probe] = 1.0
red = ((np.arange(n) // N + np.arange(n) % N) % 2 == 0)
R, B = np.flatnonzero(red), np.flatnonzero(~red)


def stationary(step, Y, cap):
    x = np.zeros(n, complex); hist = []
    for k in range(1, cap + 1):
        x = step(x)
        r = np.linalg.norm(b - Y @ x)
        hist.append(r)
        if not np.isfinite(r) or r > 1e8:
            return "div", k, hist
        if r < TOL:
            return "ok", k, hist
    return "cap", cap, hist


def rate(hist):
    h = np.array(hist[-200:])
    if len(h) < 20 or not np.all(np.isfinite(h)) or h[0] == 0:
        return np.nan
    return float((h[-1] / h[0]) ** (1 / (len(h) - 1)))


def fmt(st, k, hist):
    return f"{k}" if st == "ok" else (f"div@{k}" if st == "div" else f">{k}(r={hist[-1]:.1e})")


def krylov(kind, Y, M=None):
    cnt = [0]
    def cb(*a): cnt[0] += 1
    t = time.time()
    if kind == "gmres":
        x, info = gmres(Y, b, rtol=TOL, restart=50, maxiter=KCAP // 50, M=M, callback=cb, callback_type="pr_norm")
    else:
        x, info = bicgstab(Y, b, rtol=TOL, maxiter=KCAP // 2, M=M, callback=cb)
    rr = np.linalg.norm(b - Y @ x) / np.linalg.norm(b)
    s = f"{cnt[0]}" if rr < 2 * TOL else f">{cnt[0]}(r={rr:.1e})"
    return s, time.time() - t


def spec_radius(Y):
    Di = 1.0 / Y.diagonal()
    if n <= 500:
        G = np.eye(n) - Di[:, None] * Y.toarray()
        ev = np.linalg.eigvals(G)
        return float(np.abs(ev).max()), float((np.abs(ev) > 1).sum())
    op = LinearOperator((n, n), matvec=lambda x: x - Di * (Y @ x), dtype=complex)
    try:
        ev = eigs(op, k=6, which="LM", tol=1e-8, maxiter=20000, ncv=60, return_eigenvectors=False)
    except ArpackNoConvergence as e:
        ev = e.eigenvalues
    return (float(np.abs(ev).max()) if len(ev) else np.nan), np.nan


rows = []
hdr = ("N,decaps,f_GHz,f_over_f1,rhoJ,n_unstable,asym_rate_J,Jacobi,dJacobi(2/3),RB-GS,MG-V(Y),"
       "GMRES50,GMRES50+Jac,GMRES50+ILU(1e-2),GMRES50+ILU(1e-4),GMRES50+CSLexact,GMRES50+MG(Y),GMRES50+MG(CSL),"
       "BiCGStab,BiCGStab+ILU(1e-4),BiCGStab+CSLexact,t_splu_s")
print(hdr, flush=True)
out = open(f"exp1_N{N}.csv", "w"); out.write(hdr + "\n")
for dname, dl in (("none", []), (f"{len(dec)}rand", dec)):
    for f in freqs:
        Y = sparse_Y(mask, f, dl, lap=lap)
        d = Y.diagonal(); Di = 1 / d
        rhoJ, nun = spec_radius(Y)
        res = {}
        st = stationary(lambda x: x + Di * (b - Y @ x), Y, CAP); res["J"] = fmt(*st); aJ = rate(st[2])
        res["dJ"] = fmt(*stationary(lambda x: x + (2 / 3) * Di * (b - Y @ x), Y, CAP))
        YRB, YBR = Y[R][:, B], Y[B][:, R]
        def rbgs(x):
            x = x.copy(); x[R] = (b[R] - YRB @ x[B]) / d[R]; x[B] = (b[B] - YBR @ x[R]) / d[B]; return x
        res["GS"] = fmt(*stationary(rbgs, Y, CAP))
        mgY = MG(Y, (N, N))
        res["MG"] = fmt(*stationary(lambda x: x + mgY.vcycle(b - Y @ x), Y, 300))
        res["G"], _ = krylov("gmres", Y)
        res["GJ"], _ = krylov("gmres", Y, LinearOperator((n, n), matvec=lambda x: Di * x, dtype=complex))
        for tag, dt, ff in (("GI2", 1e-2, 2), ("GI4", 1e-4, 10)):
            try:
                il = spilu(Y.tocsc(), drop_tol=dt, fill_factor=ff)
                res[tag], _ = krylov("gmres", Y, LinearOperator((n, n), matvec=il.solve, dtype=complex))
                if tag == "GI4":
                    res["BI"], _ = krylov("bicgstab", Y, LinearOperator((n, n), matvec=il.solve, dtype=complex))
            except Exception as e:
                res[tag] = "fail"; res["BI"] = "fail"
        Ms = sparse_Y(mask, f, dl, lap=lap, shift=1 - 0.5j)
        t = time.time(); csl = splu(Ms.tocsc()); tlu = time.time() - t
        Mcsl = LinearOperator((n, n), matvec=csl.solve, dtype=complex)
        res["GC"], _ = krylov("gmres", Y, Mcsl)
        res["GMY"], _ = krylov("gmres", Y, LinearOperator((n, n), matvec=mgY.vcycle, dtype=complex))
        mgC = MG(Ms, (N, N))
        res["GMC"], _ = krylov("gmres", Y, LinearOperator((n, n), matvec=mgC.vcycle, dtype=complex))
        res["B"], _ = krylov("bicgstab", Y)
        res["BC"], _ = krylov("bicgstab", Y, Mcsl)
        row = [N, dname, f"{f/1e9:.3g}", f"{f/f1:.3g}", f"{rhoJ:.6f}", nun, f"{aJ:.6f}", res["J"], res["dJ"], res["GS"], res["MG"],
               res["G"], res["GJ"], res["GI2"], res["GI4"], res["GC"], res["GMY"], res["GMC"], res["B"], res["BI"], res["BC"], f"{tlu:.3f}"]
        line = ",".join(map(str, row)); print(line, flush=True); out.write(line + "\n"); out.flush()
