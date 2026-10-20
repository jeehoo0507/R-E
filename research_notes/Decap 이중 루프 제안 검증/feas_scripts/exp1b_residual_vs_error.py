"""Exp 1b: does 'relative residual 1e-3' mean 1e-3 accuracy in Z_probe?  GMRES + CSL-exact preconditioner
stopped at true residual <= tol; compare with direct solve."""
import warnings, numpy as np
from scipy.sparse.linalg import splu, gmres, LinearOperator
from common import *
warnings.filterwarnings("ignore")
print("N,f_GHz,f/f1,tol,iters,rel_err_x,rel_err_Zpp,dB_err_Zpp,||Y^-1||est_ratio")
for N in (50, 100):
    n = N * N; mask = rect(N); lap = lap_grid(mask); probe = (N // 2) * N + N // 2
    f1 = CFG.cell_size / np.sqrt(CFG.l_edge * CFG.c_cell) / (2 * N * CFG.cell_size)
    b = np.zeros(n, complex); b[probe] = 1
    for f in (1e8, 0.5 * f1, 0.98 * f1, 2 * f1, 1e10):
        Y = sparse_Y(mask, f, lap=lap)
        xs = splu(Y.tocsc()).solve(b)
        csl = splu(sparse_Y(mask, f, lap=lap, shift=1 - 0.5j).tocsc())
        M = LinearOperator((n, n), matvec=csl.solve, dtype=complex)
        for tol in (1e-2, 1e-3, 1e-5):
            cnt = [0]
            x, _ = gmres(Y, b, rtol=tol, restart=200, maxiter=10, M=M, callback=lambda *a: cnt.__setitem__(0, cnt[0] + 1), callback_type="pr_norm")
            r = np.linalg.norm(b - Y @ x)
            ex = np.linalg.norm(x - xs) / np.linalg.norm(xs)
            ez = abs(x[probe] - xs[probe]) / abs(xs[probe])
            print(f"{N},{f/1e9:.3g},{f/f1:.2f},{tol:g},{cnt[0]},{ex:.2e},{ez:.2e},{20*np.log10(abs(x[probe])/abs(xs[probe])):.3f},"
                  f"true_res={r:.1e}", flush=True)
