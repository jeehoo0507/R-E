# Numerical feasibility of the "solver-mimicking inner loop" decap proposal (experiments on the project's own PDN simulator)

All evidence below comes from small experiments run on 2026-10-15 on an Apple M4 Pro (14 cores, 48 GB RAM, CPU only) with the project venv (numpy 2.5.4, scipy 1.18.1, torch 2.14.1). No project file was modified; the project's `decap_loop/pdn.py` was only imported. Scripts, logs and CSVs are in the session scratchpad (temporary, copy them if they should be kept):
`/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/` (abbreviated `FEAS/` below). "Source" links point to the script/log that produced each number.

## Q0. Setup: what does the simulator's Y(f) look like?

### Takeaway
`pdn.py`'s Y(f) = y_e(f)·Lap + y_s(f)·I (+ decap admittances on the diagonal). Above ~220 MHz it is, up to a factor, the discrete Helmholtz operator (Lap − k²I): complex-symmetric, indefinite, and with the board's bulk capacitive mode always on the "wrong" side. This is the textbook hard case for local iterative solvers.

### Cited Findings
- Sparse re-implementation of `build_Y` (`FEAS/common.py::sparse_Y`) matches `pdn.build_Y` to 6.9e-17 relative error on an L-shaped 7×9 board, all 32 frequencies — [exp0.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp0.log), [exp0_validate_resonance.py](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp0_validate_resonance.py)
- Default `PDNConfig` lumped values: C_cell = 0.885 pF, L_edge = 12.6 pH, R_edge = 17.2 mΩ; R/L corner at 218 MHz; wave speed 3.0e11 cells/s (0.5 mm cells, εr = 4 → c/2). Decap library: 10 nF/100 pH/20 mΩ, 1 nF/30 pH/50 mΩ, 100 pF/10 pH/100 mΩ — [exp0.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp0.log), `decap_loop/pdn.py`
- First plane resonance (|Z| peak at corner port, 600 log-spaced points 10 MHz–20 GHz) vs analytic f10 = v/(2·N·cell): N=10 14.94 GHz (analytic 14.99); N=20 7.43 (7.50); N=50 2.94 (3.00); N=100 1.45 (1.50). Bare |Z| at 10 MHz: 180 / 44.9 / 7.19 / 1.8 Ω — [exp0.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp0.log)
- Consequence: the 32-point band 10 MHz–20 GHz covers f/f1 ≤ 1.33 for N=10 but up to f/f1 = 13 (≈13 half-wavelengths across the board) for N=100.

### Inferences
- With y_s/y_e ≈ (jωC)(R + jωL) → −ω²LC = −k² above the R/L corner, Y ∝ Lap − k²I. The Neumann Laplacian of the board has eigenvalue 0 (uniform mode), so even far below the first resonance there is one eigenvalue −k² < 0 (the plane's capacitance) — Y is indefinite at every frequency in the band except where losses dominate (< ~200 MHz, where Y ≈ (Lap + jωRC·I)/R is a nearly singular diffusion operator).

### Gaps
- Physical constants are the README's placeholders (10 µm dielectric etc.); with a thicker dielectric / larger cells the resonances and R/L corner move, but the operator class (Helmholtz) does not change.

## Q1. Do Jacobi-like local iterations converge on Y(f) v = i, above and below resonance, and how does the iteration count scale with N? (proposal H2 and Risk 1)

### Takeaway
No. Plain/damped Jacobi and Gauss–Seidel never reach 1e-3 relative residual in 20,000 sweeps on a bare board at any N or frequency, and their residual grows instead of shrinking for f ≳ 2 GHz at every N (explicit blow-up past 1e8 from 3.75–5 GHz): the Jacobi spectral radius is ≥ 1 everywhere (1.000000 at 10 MHz → 1.047 at 20 GHz). A geometric multigrid V-cycle (Jacobi smoother + Galerkin coarse grids, i.e. the idealised version of "Jacobi message passing + coarse-grid attention") converges in 3–4 cycles independent of N up to ~1.5–2·f1, but diverges above ~3·f1 and near decap-loaded resonances. Robust convergence across the whole band needs a Krylov outer loop with a shifted-Laplacian/multigrid preconditioner, and its iteration count grows with the number of wavelengths across the board (≈ with N at fixed frequency): 2–5 iterations below f1, 12 at 3.3·f1, 34 at 6.7·f1, 45 (exact CSL) to 175 (MG-CSL) at 13·f1 for N=100.

### Cited Findings
Experiment: rectangular N×N boards, probe = centre node, b = unit current at probe, x0 = 0, stop at ‖b − Yx‖/‖b‖ ≤ 1e-3. Decaps: none, or random 2% of cells (≥4) with random types. ρJ = spectral radius of I − D⁻¹Y (dense eig for n ≤ 500, ARPACK otherwise; agrees with the asymptotic residual ratio of the Jacobi run to 4–6 digits). Caps: 20,000 sweeps (Jacobi family), 300 V-cycles, 3,000 Krylov iterations. "div@k" = residual > 1e8 at sweep k; ">cap (r=…)" = not converged, final residual shown. GMRES50 = restarted GMRES(50). CSLexact = exact LU of the complex-shifted operator (shunt term × (1 − 0.5j)), i.e. the best case of a shifted-Laplacian preconditioner; MG(·) = one V-cycle (bilinear P, Galerkin PᵀAP, 2+2 damped-Jacobi ω=0.7, direct solve at ≤64 nodes). BiCGStab counts iterations (2 matvecs each). Source: [exp1_iterative.py](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp1_iterative.py), CSVs [exp1_N10.csv](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp1_N10.csv) [exp1_N20.csv](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp1_N20.csv) [exp1_N50.csv](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp1_N50.csv) [exp1_N100.csv](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp1_N100.csv) (also contain GMRES+Jacobi-precond., ILU(1e-2), BiCGStab+ILU/CSL, number of |λ|>1 eigenvalues).

| N | decaps | f_GHz | f_over_f1 | rhoJ | Jacobi | dJacobi(2/3) | RB-GS | MG-V(Y) | GMRES50 | GMRES50+ILU(1e-4) | BiCGStab | GMRES50+CSLexact | GMRES50+MG(Y) | GMRES50+MG(CSL) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 10 | none | 0.01 | 0.000667 | 1.000000 | >20000 (r=1.4e-01) | >20000 (r=1.0e-01) | >20000 (r=1.4e-01) | 4 | 42 | 1 | 34 | 2 | 3 | 5 |
| 10 | none | 0.1 | 0.00667 | 1.000001 | >20000 (r=1.5e-01) | >20000 (r=1.0e-01) | >20000 (r=1.5e-01) | 4 | 39 | 1 | 32 | 2 | 3 | 5 |
| 10 | none | 0.3 | 0.02 | 1.000011 | >20000 (r=1.8e-01) | >20000 (r=1.2e-01) | >20000 (r=2.2e-01) | 4 | 38 | 1 | 31 | 2 | 3 | 5 |
| 10 | none | 1 | 0.0667 | 1.000121 | >20000 (r=1.6e+00) | >20000 (r=5.1e-01) | >20000 (r=1.8e+01) | 4 | 36 | 1 | 30 | 2 | 3 | 4 |
| 10 | none | 2 | 0.133 | 1.000487 | >20000 (r=2.4e+03) | >20000 (r=6.7e+01) | >20000 (r=4.1e+07) | 4 | 34 | 1 | 29 | 3 | 3 | 4 |
| 10 | none | 5 | 0.334 | 1.003058 | div@6671 | div@10172 | div@3336 | 4 | 32 | 1 | 26 | 3 | 3 | 4 |
| 10 | none | 7.49 | 0.5 | 1.006904 | div@2962 | div@4513 | div@1481 | 4 | 32 | 1 | 25 | 4 | 3 | 4 |
| 10 | none | 10 | 0.667 | 1.012367 | div@1659 | div@2525 | div@830 | 4 | 32 | 1 | 26 | 4 | 3 | 5 |
| 10 | none | 20 | 1.33 | 1.051685 | div@406 | div@614 | div@203 | 4 | 36 | 1 | 38 | 7 | 3 | 7 |
| 20 | none | 0.01 | 0.00133 | 1.000000 | >20000 (r=7.1e-02) | >20000 (r=5.0e-02) | >20000 (r=7.1e-02) | 3 | 396 | 2 | 69 | 2 | 3 | 4 |
| 20 | none | 0.1 | 0.0133 | 1.000001 | >20000 (r=7.3e-02) | >20000 (r=5.1e-02) | >20000 (r=7.4e-02) | 3 | 304 | 2 | 69 | 2 | 3 | 4 |
| 20 | none | 0.3 | 0.04 | 1.000010 | >20000 (r=8.7e-02) | >20000 (r=5.8e-02) | >20000 (r=1.1e-01) | 3 | 304 | 2 | 58 | 2 | 3 | 5 |
| 20 | none | 1 | 0.133 | 1.000115 | >20000 (r=7.1e-01) | >20000 (r=2.3e-01) | >20000 (r=7.1e+00) | 3 | 196 | 2 | 54 | 3 | 3 | 4 |
| 20 | none | 2 | 0.267 | 1.000462 | >20000 (r=7.2e+02) | >20000 (r=2.4e+01) | >20000 (r=7.4e+06) | 3 | 100 | 2 | 51 | 3 | 3 | 4 |
| 20 | none | 3.75 | 0.5 | 1.001624 | div@12981 | div@19786 | div@6491 | 3 | 66 | 2 | 44 | 4 | 3 | 4 |
| 20 | none | 5 | 0.667 | 1.002897 | div@7285 | div@11102 | div@3643 | 3 | 69 | 1 | 43 | 4 | 3 | 4 |
| 20 | none | 10 | 1.33 | 1.011710 | div@1813 | div@2759 | div@907 | 4 | 71 | 1 | 57 | 7 | 3 | 7 |
| 20 | none | 11.2 | 1.5 | 1.014855 | div@1432 | div@2178 | div@716 | 4 | 69 | 1 | 58 | 8 | 4 | 8 |
| 20 | none | 20 | 2.67 | 1.048881 | div@446 | div@674 | div@223 | 8 | 141 | 1 | 133 | 12 | 7 | 12 |
| 50 | none | 0.01 | 0.00334 | 1.000000 | >20000 (r=2.8e-02) | >20000 (r=2.0e-02) | >20000 (r=2.8e-02) | 4 | >3000 (r=1.9e-02) | 28 | 145 | 2 | 3 | 5 |
| 50 | none | 0.1 | 0.0334 | 1.000001 | >20000 (r=2.9e-02) | >20000 (r=2.0e-02) | >20000 (r=3.0e-02) | 4 | >3000 (r=1.6e-02) | 26 | 130 | 2 | 3 | 5 |
| 50 | none | 0.3 | 0.1 | 1.000010 | >20000 (r=3.5e-02) | >20000 (r=2.3e-02) | >20000 (r=4.2e-02) | 4 | >3000 (r=9.5e-03) | 24 | 124 | 2 | 3 | 4 |
| 50 | none | 1 | 0.334 | 1.000112 | >20000 (r=2.6e-01) | >20000 (r=8.9e-02) | >20000 (r=2.5e+00) | 4 | 1389 | 21 | 112 | 3 | 3 | 4 |
| 50 | none | 1.5 | 0.5 | 1.000251 | >20000 (r=4.3e+00) | >20000 (r=5.7e-01) | >20000 (r=6.5e+02) | 4 | 259 | 21 | 104 | 4 | 3 | 4 |
| 50 | none | 2 | 0.667 | 1.000447 | >20000 (r=2.2e+02) | >20000 (r=7.8e+00) | >20000 (r=1.7e+06) | 4 | 143 | 22 | 99 | 4 | 3 | 5 |
| 50 | none | 4.5 | 1.5 | 1.002270 | div@9705 | div@14782 | div@4853 | 4 | 195 | 19 | 164 | 8 | 3 | 8 |
| 50 | none | 5 | 1.67 | 1.002808 | div@7849 | div@11953 | div@3925 | 4 | 196 | 20 | 183 | 9 | 3 | 12 |
| 50 | none | 10 | 3.34 | 1.011349 | div@1958 | div@2977 | div@979 | div@28 | 412 | 24 | 599 | 11 | 7 | 15 |
| 50 | none | 20 | 6.67 | 1.047415 | div@486 | div@734 | div@243 | div@12 | 703 | 33 | 872 | 29 | 30 | 33 |
| 50 | 50rand | 0.01 | 0.00334 | 1.000000 | >20000 (r=2.9e-02) | >20000 (r=2.0e-02) | >20000 (r=2.9e-02) | 4 | >3000 (r=2.8e-03) | 24 | 120 | 2 | 3 | 3 |
| 50 | 50rand | 0.1 | 0.0334 | 1.000033 | >20000 (r=5.5e-02) | >20000 (r=3.1e-02) | >20000 (r=1.1e-01) | 4 | 191 | 20 | 98 | 2 | 4 | 4 |
| 50 | 50rand | 0.3 | 0.1 | 0.999851 | >20000 (r=1.4e-03) | >20000 (r=2.7e-03) | 11207 | 4 | 214 | 21 | 103 | 2 | 4 | 4 |
| 50 | 50rand | 1 | 0.334 | 0.999342 | 5018 | 6737 | 2509 | 5 | 124 | 16 | 83 | 2 | 5 | 4 |
| 50 | 50rand | 1.5 | 0.5 | 0.999921 | >20000 (r=5.5e-03) | >20000 (r=6.6e-03) | >20000 (r=1.1e-03) | 8 | 367 | 20 | 116 | 4 | 5 | 7 |
| 50 | 50rand | 2 | 0.667 | 1.000527 | >20000 (r=9.6e+02) | >20000 (r=2.0e+01) | >20000 (r=3.6e+07) | 5 | 291 | 22 | 116 | 4 | 5 | 5 |
| 50 | 50rand | 4.5 | 1.5 | 0.998890 | 3207 | 4336 | 1604 | div@105 | 127 | 17 | 93 | 5 | 13 | 13 |
| 50 | 50rand | 5 | 1.67 | 0.999811 | 18946 | >20000 (r=2.0e-03) | 9473 | 15 | 190 | 19 | 165 | 7 | 11 | 12 |
| 50 | 50rand | 10 | 3.34 | 1.009182 | div@2404 | div@3657 | div@1202 | div@57 | 935 | 27 | 1500 | 28 | 14 | 27 |
| 50 | 50rand | 20 | 6.67 | 1.045233 | div@506 | div@765 | div@253 | div@12 | 860 | 32 | 1406 | 44 | 55 | 41 |
| 100 | none | 0.01 | 0.00667 | 1.000000 | >20000 (r=1.4e-02) | >20000 (r=1.0e-02) | >20000 (r=1.4e-02) | 3 | >3000 (r=9.8e-03) | >3000 (r=4.5e-03) | 243 | 2 | 3 | 5 |
| 100 | none | 0.1 | 0.0667 | 1.000001 | >20000 (r=1.4e-02) | >20000 (r=1.0e-02) | >20000 (r=1.5e-02) | 3 | >3000 (r=8.0e-03) | 239 | 227 | 2 | 3 | 4 |
| 100 | none | 0.3 | 0.2 | 1.000010 | >20000 (r=1.7e-02) | >20000 (r=1.1e-02) | >20000 (r=2.1e-02) | 3 | >3000 (r=6.5e-03) | 154 | 204 | 3 | 3 | 4 |
| 100 | none | 0.749 | 0.5 | 1.000062 | >20000 (r=4.9e-02) | >20000 (r=2.3e-02) | >20000 (r=1.7e-01) | 3 | 3000 | 96 | 142 | 3 | 3 | 4 |
| 100 | none | 1 | 0.667 | 1.000110 | >20000 (r=1.3e-01) | >20000 (r=4.4e-02) | >20000 (r=1.2e+00) | 3 | 1048 | 97 | 136 | 4 | 3 | 5 |
| 100 | none | 2 | 1.33 | 1.000443 | >20000 (r=9.9e+01) | >20000 (r=3.6e+00) | >20000 (r=6.9e+05) | 3 | 1741 | 150 | 146 | 5 | 3 | 7 |
| 100 | none | 2.25 | 1.5 | 1.000560 | >20000 (r=1.0e+03) | >20000 (r=1.7e+01) | >20000 (r=7.5e+07) | 3 | 1676 | 120 | 160 | 7 | 3 | 8 |
| 100 | none | 5 | 3.34 | 1.002780 | div@8188 | div@12463 | div@4094 | div@39 | 1063 | 279 | 655 | 12 | 8 | 13 |
| 100 | none | 10 | 6.67 | 1.011236 | div@2046 | div@3110 | div@1023 | div@12 | 1310 | 379 | >1500 (r=1.1e-02) | 34 | 31 | 24 |
| 100 | none | 20 | 13.3 | 1.047079 | div@492 | div@745 | div@246 | div@3 | 967 | 545 | 1286 | 45 | 1592 | 175 |
| 100 | 200rand | 0.01 | 0.00667 | 1.000000 | >20000 (r=1.4e-02) | >20000 (r=1.0e-02) | >20000 (r=1.4e-02) | 3 | 3000 | 133 | 195 | 2 | 3 | 3 |
| 100 | 200rand | 0.1 | 0.0667 | 0.999999 | >20000 (r=1.8e-02) | >20000 (r=1.3e-02) | >20000 (r=1.7e-02) | 3 | 305 | 82 | 184 | 2 | 3 | 3 |
| 100 | 200rand | 0.3 | 0.2 | 0.999831 | 14990 | 19420 | 7495 | 4 | 327 | 107 | 171 | 2 | 4 | 4 |
| 100 | 200rand | 0.749 | 0.5 | 1.000019 | >20000 (r=4.1e-02) | >20000 (r=2.6e-02) | >20000 (r=6.0e-02) | 7 | 155 | 50 | 118 | 2 | 5 | 5 |
| 100 | 200rand | 1 | 0.667 | 0.998954 | 2052 | 2633 | 1026 | 5 | 96 | 35 | 74 | 2 | 5 | 5 |
| 100 | 200rand | 2 | 1.33 | 1.000056 | >20000 (r=3.6e-02) | >20000 (r=1.7e-02) | >20000 (r=1.1e-01) | 65 | 412 | 86 | 197 | 7 | 8 | 7 |
| 100 | 200rand | 2.25 | 1.5 | 1.000396 | >20000 (r=2.8e+01) | >20000 (r=1.4e+00) | >20000 (r=7.5e+04) | div@223 | 522 | 131 | 279 | 7 | 9 | 8 |
| 100 | 200rand | 5 | 3.34 | 1.000742 | >20000 (r=4.0e+03) | >20000 (r=2.0e+01) | div@16815 | 18 | 304 | 58 | 246 | 10 | 14 | 16 |
| 100 | 200rand | 10 | 6.67 | 1.009665 | div@2488 | div@3781 | div@1244 | div@13 | 1040 | 300 | >1500 (r=1.6e-02) | 31 | 73 | 31 |
| 100 | 200rand | 20 | 13.3 | 1.045784 | div@518 | div@784 | div@259 | div@3 | 924 | 544 | 1312 | 47 | >3000 (r=3.6e-02) | 116 |

(N=10/20 rows with decaps are in the CSVs; same pattern, e.g. N=20 + 8 decaps: MG-V(Y) div@74 at 3.75 GHz and div@67 at 10 GHz.)

- Jacobi family: bare boards never converge (any N, any f); ρJ ≥ 1.000000 at all 40 bare (N, f) points; divergence starts at 5 GHz (N=10, f/f1 = 0.33), 3.75 GHz (N=20), 4.5 GHz (N=50), 5 GHz (N=100). Number of eigenvalues with |λ| > 1: 2 (one ± pair; grid is bipartite) below f1, 6–16 above f1 (N=10/20) — [exp1_N10.csv](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp1_N10.csv), [exp1_N20.csv](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp1_N20.csv)
- Decaps occasionally make ρJ < 1 (decap ESR damps the bulk mode; ρJ = 0.9932–0.9999), giving convergence after 330–19,332 sweeps, but non-monotonically in f and N (e.g. N=100 + 200 decaps: 1,026 RB-GS sweeps at 1 GHz, no convergence at 0.749 or 2 GHz) — table above.
- Red-black Gauss–Seidel diverges about 2× faster than Jacobi (consistent with ρGS = ρJ² for consistently ordered matrices); damping (ω = 2/3) only delays divergence — table above.
- MG V-cycle as a stationary solver: 3–4 cycles for all N at f ≤ 1.5·f1 (bare), 3 cycles even at N=100 up to 2.25 GHz; diverges at f ≥ 3.3·f1 (N=50 at 10/20 GHz, N=100 at 5/10/20 GHz) and near decap-loaded resonances (N=50 4.5 GHz div@105, N=100 2.25 GHz div@223; N=100 2 GHz needs 65 cycles) — table above.
- Unpreconditioned GMRES(50): 32–42 (N=10), 66–396 (N=20), 124–>3000 (N=50), 96–>3000 (N=100) iterations; worst at low frequency (near-singular Laplacian), i.e. iteration count grows ~N² there — table above.
- GMRES + exact CSL: N=100 bare 2, 2, 3, 3, 4, 5, 7, 12, 34, 45 iterations for f/f1 = 0.007 … 13.3; at fixed 10 GHz: 4 (N=10), 7 (N=20), 11 (N=50), 34 (N=100). GMRES + MG(CSL) at N=100: 3–8 up to 1.5·f1, 13–16 at 3.3·f1, 24–31 at 6.7·f1, 116–175 at 13.3·f1 — table above.
- Residual-based stopping is reliable in this lossy model: with GMRES+CSL stopped at rtol 1e-3, relative error of Z_pp was ≤ 1.2e-3 (≤ 0.004 dB) in all 10 tested (N, f) cases incl. 0.98·f1; at rtol 1e-2 the error was ≤ 5.6e-3 (≤ 0.025 dB) — [exp1b.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp1b.log), [exp1b_residual_vs_error.py](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp1b_residual_vs_error.py)
- Locality (what a purely local T-step network can see): Z_pp computed on a (2r+1)² window around the centre of a 100×100 board vs. the full board, |error| in dB — [exp1c.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp1c.log), [exp1c_locality.py](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp1c_locality.py):

| f (GHz) | bare r=5 | r=10 | r=20 | r=35 | r=49 | 2% decaps r=5 | r=10 | r=20 | r=35 | r=49 |
|---|---|---|---|---|---|---|---|---|---|---|
| 0.01 | 38.4 | 27.1 | 15.5 | 5.95 | 0.17 | 55.0 | 28.9 | 11.7 | 4.24 | 0.01 |
| 0.1 | 38.6 | 27.4 | 15.7 | 6.08 | 0.18 | 39.8 | 10.1 | 0.51 | 0.36 | 0.00 |
| 0.3 | 41.0 | 29.7 | 17.8 | 7.38 | 0.23 | 24.2 | 16.0 | 2.45 | 0.73 | 0.00 |
| 1.0 | 29.7 | 17.7 | 1.71 | 5.14 | 0.10 | 6.92 | 3.92 | 0.29 | 0.01 | 0.00 |
| 1.5 | 20.6 | 7.56 | 16.1 | 2.41 | 0.06 | 12.0 | 8.80 | 1.13 | 0.19 | 0.00 |
| 3.0 | 1.92 | 26.8 | 8.12 | 4.27 | 0.74 | 20.7 | 4.90 | 1.55 | 0.05 | 0.01 |
| 10 | 4.37 | 2.83 | 7.73 | 4.31 | 0.87 | 6.81 | 0.23 | 0.39 | 1.80 | 0.02 |

### Inferences
- Risk 1 is confirmed and is worse than stated: local Jacobi-type message passing fails not only above resonance but at every frequency of this simulator, because the plane's bulk capacitance makes Y indefinite (ρJ > 1) and, below ~200 MHz, near-singular (ρJ → 1). Damping cannot fix eigenvalues of D⁻¹Y with negative real part. A network that "mimics Jacobi" inherits this unless it learns something qualitatively different from Jacobi.
- H2 ("iterations ∝ board size") holds only in a weak sense. Information-theoretically a local scheme needs ≥ the board radius in steps (exp1c: the bare-board Z_pp depends on the whole 100×100 board at all frequencies; even with 2% decaps the 10 MHz value needs r ≥ 35). But the classical scheme that actually converges (multigrid) needs O(1) cycles below ~2·f1, and ~O(#wavelengths) ≈ O(N·f) Krylov iterations above. So the right scaling knob is "number of wavelengths across the board", not N alone; a learned inner loop should be conditioned on f/f1 (or k·L), not just on board size.
- The coarse-grid ("multigrid attention") part is what makes convergence possible; but a plain V-cycle on Y is unstable once coarse grids under-resolve the wave (k·h_coarse large) — here f ≳ 3·f1 — and near decap-induced resonances. To cover 10 MHz–20 GHz on 100×100 boards the inner loop needs an outer Krylov/residual-minimising wrapper or a shifted (damped) coarse operator; plain "stop on residual" fixed-point iteration will run away.
- Stop-on-residual itself is fine (residual ≈ error here), so the criterion is not the problem — convergence is.

### Gaps
- Only rectangular boards in exp1 (L/holes not tested); MG was a basic implementation (no Krylov smoothers, no deflation, no wave-ray); a learned smoother/coarse operator could in principle do better — not tested (no network was trained).
- GPU timing / low-precision (bf16) behaviour of iterative solves not measured; ARPACK was used for n ≥ 2,500 and was cross-checked only against the Jacobi asymptotic rate.

## Q2. What does the classical alternative cost (sparse LU per frequency, probe-only columns) vs building the full all-port Z? Are the proposal's 29 s / 1.6 GB (50×50) and 26 GB (100×100) plausible?

### Takeaway
The proposal's numbers reproduce (50×50 dense `build_Z`: 35.8 s, 1.6 GB complex64), but they are an artefact of the all-port dense design. A sparse LU of Y(f) is tiny and fast on a 2-D grid: 0.004 s / 1.4 MB (50×50), 0.019 s / 7.4 MB (100×100), 0.11 s / 39 MB (200×200), 0.30–0.35 s / 100 MB (300×300) per frequency, and one probe column costs 0.2–12 ms more. The probe-only state for 32 frequencies therefore costs ~0.15 s (50×50) to ~11 s (300×300). What remains expensive is anything needing all-port information — in particular diag(Z) (self-impedance at every candidate node, used by greedy and by the current `node_features`), which costs n solves per frequency.

### Cited Findings
Source: [exp2_factor_cost.py](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp2_factor_cost.py), [exp2.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp2.log); f = 1 GHz shown (0.1 and 10 GHz within ±10%). LU memory = nnz(L+U) × 20 B (complex128 value + int32 index). scipy SuperLU, single thread; MMD_AT_PLUS_A ordering (COLAMD 1.7–1.8× slower and larger).

| board | nodes n | splu time / freq | LU memory | solve 1 col | solve 16 cols | solve 256 cols | diag(Z) via n cols / freq (extrap.) | full Z, 32 f, complex64 |
|---|---|---|---|---|---|---|---|---|
| 50×50 | 2,500 | 0.004 s | 1.4 MB | 0.19 ms | 1.6 ms | 27 ms | 0.3 s | 1.6 GB |
| 100×100 | 10,000 | 0.019 s | 7.4 MB | 0.70 ms | 6.6 ms | 146 ms | 5.7 s | 25.6 GB |
| 200×200 | 40,000 | 0.113 s | 39 MB | 3.9 ms | 37 ms | 690 ms | 108 s | 410 GB |
| 300×300 | 90,000 | 0.30–0.35 s | 100 MB | 10–12 ms | 91–107 ms | 1.65–2.07 s | 580–730 s | 2.07 TB |

- Dense path (project code, torch, 10 threads, complex128 solve): `build_Z` on a full rect board took 0.50 s (20×20), 3.67 s (30×30), 13.9 s (40×40), 35.8 s (50×50; Z = 1.60 GB complex64). One dense inverse at a single frequency: 6.2 s (70×70), 43.1 s (100×100) → ×32 frequencies ≈ 23 min for 100×100 — [exp2.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp2.log). (Dense timings were taken while two exp1 processes were running in the background, so they may be ~10–20% pessimistic.)
- Peak RSS of the run (50×50 `build_Z` + one 100×100 dense inverse) was 18.1 GB — [exp2.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp2.log)
- Measured end-to-end probe-only setup for all 32 frequencies: factorize 0.15 s + probe column 0.006 s (50×50); 0.67 s + 0.024 s (100×100) — [exp3.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp3.log)
- Measured diag(Z) for all nodes, 32 frequencies, via chunked (512-RHS) LU solves: 9.4 s (50×50); 196 s (100×100, extrapolated from 2 frequencies) — [exp3.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp3.log)

### Inferences
- "26 GB at 100×100" is the final complex64 storage only. As written, `pdn.build_Y` materialises the dense [32, n, n] complex128 Y (51 GB at n = 10⁴, plus a same-size `diag_embed` temporary and the inverse), so `build_Z` cannot actually run at 100×100 on a 48 GB machine; the true peak is > 100 GB (inference from the code, not run).
- The scaling problem is solved classically for the probe-only quantities: a 300×300 board's Z_probe,all for 32 frequencies costs ~11 s on one CPU core, with ~3 GB of LU factors in total if all 32 are kept. A learned inner-loop solver must beat ~20 ms/frequency (factor + solve) at 100×100 to claim a speed advantage, and must match its exactness.
- The real wall is all-port information: diag(Z) via n solves grows ~n^1.5–n² (0.3 s → 5.7 s → 108 s → ~650 s per frequency). Options (not tested here): selected inversion (Takahashi-type recurrences give diag(Y⁻¹) at roughly factorization cost), computing z_ii only at the 8 feature frequencies, restricting/subsampling candidate nodes, or approximating z_ii locally (exp1c suggests windows of r ≈ 20 cells are accurate to ~1–2.5 dB above 100 MHz once some decaps are present, but not at 10 MHz).

### Gaps
- No METIS/nested-dissection ordering, multithreaded or GPU sparse solver (cuDSS, PARDISO, MUMPS) was tried; these would be faster than SuperLU. Selected inversion was not implemented. Memory was computed from nnz, not measured by a profiler.

## Q3. When one decap is added, how does refactorizing compare with a Sherman–Morrison–Woodbury update?

### Takeaway
Never refactorize. Keeping the base LU and the columns Z·e_p (probe) and Z·e_P (placed decaps), committing a decap costs one extra solve per frequency (5 ms for 32 f at 50×50, 24 ms at 100×100), and evaluating a candidate costs ~4.7 µs for all 32 frequencies, versus 0.14 s (50×50) / 0.67 s (100×100) for refactorize-and-solve — a 30,000–140,000× gap. SMW and refactorization agree to ≤ 5.6e-13.

### Cited Findings
Source: [exp3_update_cost.py](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp3_update_cost.py), [exp3.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp3.log). 32 frequencies, 16 random decaps added one at a time, centre probe.

| board | one-time: factor 32 f | one-time: probe col | one-time: diag(Z) 32 f | refactor + probe solve per decap (32 f) | SMW commit per decap (32 f) | SMW evaluate all n×3 candidates (32 f), k = 1 / 16 placed | per-candidate (32 f) | max rel. err SMW vs refactor |
|---|---|---|---|---|---|---|---|---|
| 50×50 | 0.15 s | 0.006 s | 9.4 s | 0.133–0.140 s | 0.0052 s | 0.004 s / 0.090 s | 4.65 µs | 5.6e-13 |
| 100×100 | 0.67 s | 0.024 s | 196 s (extrap.) | 0.655–0.688 s | 0.024 s | 0.016 s / 0.366 s | 4.65 µs | 2.7e-13 |

- Exhaustive greedy step by refactorization would take 0.3 h (50×50, 7,500 candidates) or 5.6 h (100×100, 30,000 candidates); with SMW it is 0.004–0.37 s — [exp3.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp3.log)

### Inferences
- For each decap placement step, the inner problem is a rank-1 update of a fixed factorization, not a new linear solve. An inner loop that re-solves Y v = i iteratively at every step (as the proposal implies) would be strictly more work than the classical route; its only possible advantage is avoiding the one-time diag(Z) cost.
- The bottleneck for a 100×100 greedy/RL environment is the one-time diag(Z) (196 s, 32 f) — not per-step updates.

### Gaps
- Timings are CPU/numpy; the einsum for k = 16 is not optimised (O(F·n·k²)), so the k-dependence is an upper bound.

## Q4. Is the "physics lower bound" (C-based + ESL-based decap count) tight enough to normalise rewards?

### Takeaway
Only in the trivial regime. When the target band starts at 10 MHz and stops at ≤ 1 GHz, the bulk-capacitance bound ceil(C_needed/C_decap) is exact (ratio 1.00 in 36/36 cases) — but that regime is location-insensitive. For the location-sensitive bands this project actually rewards (≥ 100 MHz), the simple bound is 1–2 decaps while greedy needs 2–13 (ratio 1.0–9.0, median ≈ 3.5); a stronger per-frequency admittance-sum MILP bound only improves this to 1.0–5.5. The bounds were never violated (0 of 108 feasible cases), but they cannot detect infeasibility (12 of 120 setting-instances had no solution with ≤ 40–60 decaps while the bound said 2–16).

### Cited Findings
Setup: random 10×10 and 20×20 boards (rect/L/holes, random probe, 0–30% keep-out, as in `pdn.sample_batch`), 40 log-spaced frequencies 10 MHz–20 GHz, flat target Zt inside the band. Greedy min-count = add the (node, type) that most reduces Σ_f max(0, log10|Z|/Zt) until |Z| ≤ Zt in band (cap 60 resp. 40 decaps), then repeatedly drop any decap whose removal keeps feasibility. Bounds: LB_C = ceil(max(0, 1/(2π f_lo Zt) − C_plane)/10 nF); LB_ESL = ceil(2π f_hi · 10 pH / Zt); LB_Y = integer min Σ n_t s.t. |1/Z0_pp(f)| + Σ_t n_t |1/z_t(f)| ≥ 1/Zt ∀ f in band (exact if all decaps sat on the probe node). "Local search found k−1" = random swap hill-climbing (600 moves) found a feasible placement with one decap fewer than greedy. Sources: [exp4_lib.py](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp4_lib.py), [exp4_bound.py](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp4_bound.py), [exp4.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp4.log), [exp4b_bound_band.py](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp4b_bound_band.py), [exp4b.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp4b.log), target exploration [exp4_explore.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp4_explore.log).

Band starting at 10 MHz ([exp4.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp4.log)):

| N | band (GHz) | Zt (ohm) | inst | infeasible | greedy+prune count min/med/max | max(LB_C, LB_ESL) | LB_Y (MILP) | greedy / simple LB | greedy / LB_Y | local search found k-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.01-1 | 0.5 | 6 | 0 | 4/4/4 | 4-4 | 4-4 | 1.00-1.00 | 1.00-1.00 | 0/6 |
| 10 | 0.01-1 | 0.2 | 6 | 0 | 8/8/8 | 8-8 | 8-8 | 1.00-1.00 | 1.00-1.00 | 0/6 |
| 10 | 0.01-1 | 0.1 | 6 | 0 | 16/16/16 | 16-16 | 16-16 | 1.00-1.00 | 1.00-1.00 | 0/6 |
| 10 | 0.01-3 | 0.2 | 6 | 0 | 11/11/12 | 8-8 | 8-8 | 1.38-1.50 | 1.38-1.50 | 0/6 |
| 10 | 0.01-3 | 0.1 | 6 | 6 | - | 16-16 | 16-16 | - | - | - |
| 10 | 0.01-10 | 0.5 | 6 | 0 | 6/7/11 | 4-4 | 4-4 | 1.50-2.75 | 1.50-2.75 | 2/6 |
| 20 | 0.01-1 | 0.5 | 6 | 0 | 4/4/4 | 4-4 | 4-4 | 1.00-1.00 | 1.00-1.00 | 0/- |
| 20 | 0.01-1 | 0.2 | 6 | 0 | 8/8/8 | 8-8 | 8-8 | 1.00-1.00 | 1.00-1.00 | 0/- |
| 20 | 0.01-1 | 0.1 | 6 | 0 | 16/16/16 | 16-16 | 16-16 | 1.00-1.00 | 1.00-1.00 | 0/- |
| 20 | 0.01-3 | 0.2 | 6 | 0 | 11/11/12 | 8-8 | 8-8 | 1.38-1.50 | 1.38-1.50 | 0/- |
| 20 | 0.01-3 | 0.1 | 6 | 4 | 27/30/34 | 16-16 | 16-16 | 1.69-2.12 | 1.69-2.12 | 0/- |
| 20 | 0.01-10 | 0.5 | 6 | 0 | 6/6/7 | 4-4 | 5-5 | 1.50-1.75 | 1.20-1.40 | 0/- |

Band starting at ≥ 100 MHz (the project's reward band starts at 100 MHz) ([exp4b.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp4b.log)):

| N | band (GHz) | Zt (ohm) | inst | infeasible | greedy+prune count min/med/max | max(LB_C, LB_ESL) | LB_Y (MILP) | greedy / simple LB | greedy / LB_Y | local search found k-1 |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.1-3 | 0.2 | 6 | 0 | 7/8/9 | 1-1 | 2-3 | 7.00-9.00 | 2.33-3.50 | 1/6 |
| 10 | 0.1-3 | 0.15 | 6 | 1 | 10/12/13 | 2-2 | 3-4 | 5.00-6.50 | 2.75-4.00 | 0/5 |
| 10 | 0.3-5 | 0.25 | 6 | 0 | 5/5/6 | 2-2 | 2-2 | 2.50-3.00 | 2.50-3.00 | 0/6 |
| 10 | 0.3-10 | 0.5 | 6 | 1 | 3/4/4 | 2-2 | 1-2 | 1.50-2.00 | 2.00-4.00 | 0/5 |
| 20 | 0.1-3 | 0.2 | 6 | 0 | 7/8/9 | 1-1 | 2-2 | 7.00-9.00 | 3.50-4.50 | 1/6 |
| 20 | 0.1-3 | 0.15 | 6 | 0 | 8/10/11 | 2-2 | 2-2 | 4.00-5.50 | 4.00-5.50 | 1/6 |
| 20 | 0.3-5 | 0.25 | 6 | 0 | 3/3/4 | 2-2 | 1-2 | 1.50-2.00 | 1.50-3.00 | 0/6 |
| 20 | 0.3-10 | 0.5 | 6 | 0 | 2/2/4 | 2-2 | 2-2 | 1.00-2.00 | 1.00-2.00 | 0/6 |

- Summary over all feasible cases: band from 10 MHz — greedy/simple-LB 1.00–2.75 (median 1.00); band from ≥ 100 MHz — greedy/simple-LB 1.00–9.00 (median 3.50), greedy/LB_Y 1.00–5.50 (median 3.00); bound violations 0/62 and 0/46 — [exp4.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp4.log), [exp4b.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp4b.log)
- Stricter targets (0.1–3 GHz @ 0.1/0.05 Ω, 0.3–10 GHz @ 0.2/0.1 Ω, 1–20 GHz @ 0.3 Ω) were infeasible for greedy with 60 decaps on 10×10 boards in 11 of 13 completed cases (LB_Y said 3–10); the 2 feasible ones needed 19 and 21 decaps vs LB_Y = 5 — [exp4b_strict_targets_partial.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp4b_strict_targets_partial.log)
- Greedy is not always optimal: local search beat greedy+prune by one decap in 2/6 cases (10×10, 10 MHz–10 GHz @ 0.5 Ω) and 3/46 cases in the ≥ 100 MHz bands — tables above.

### Inferences
- The C-based bound is the right physics only for the lumped low-frequency regime, where placement does not matter; the ESL-based bound ignores the plane's own capacitance and the spreading inductance between probe and decap, so in the band where placement matters it is 2–9× below what is achievable. Because the true optimum lies between the bound and the greedy count, the ratio greedy/LB mixes "bound slack" and "greedy slack"; but since local search rarely improved greedy (5 of 76 feasible cases where it was run), most of the gap is bound slack.
- As a reward normaliser it would not equalise difficulty: across instances and targets the ratio spans 1–9. A cheap, much tighter reference is the greedy+prune count itself (on 20×20 boards it runs in well under a second with SMW), or best-known counts.
- A bound that cannot certify infeasibility is a problem for a min-count task: the reward must handle "unreachable target" explicitly (e.g., penalise residual violation), which the bound does not help with.

### Gaps
- Only 10×10 and 20×20 boards, 6 instances per setting, single probe, placeholder decap library; no exact optimum (MILP/exhaustive) was computed, so "greedy slack" is only bounded from below by the local-search check.

## Q5. Implications for the proposal: what is feasible, what is not, what to change

### Takeaway
The outer RL loop on a classical environment is feasible at 100×100–300×300 today (sparse LU + SMW); the "solver-mimicking inner loop" as specified (Jacobi-like message passing + coarse attention, stop on residual) is not justified by numerics: Jacobi-type steps diverge or stall at every frequency of this simulator, a plain multigrid cycle diverges above ~3·f1, and the classical direct route is already ~20 ms/frequency at 100×100. The physics lower bound is too loose (1–9×) to normalise rewards except in the trivial low-frequency regime.

### Cited Findings
- Jacobi/GS: never converge on bare boards (0/40 bare (N, f) points), ρJ ≥ 1 — Q1, [exp1 CSVs](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp1_N100.csv)
- MG V-cycle: 3–4 cycles independent of N up to ~1.5–2·f1; diverges at ≥ 3.3·f1 — Q1
- GMRES+CSL: 2–47 iterations, growing with f/f1 — Q1
- Sparse LU 100×100: 19 ms/f; 300×300: ~0.3 s/f, 100 MB — Q2, [exp2.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp2.log)
- SMW candidate evaluation 4.65 µs vs refactorization 0.14–0.67 s — Q3, [exp3.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp3.log)
- Lower bound ratio 1.0–9.0 in location-sensitive bands — Q4, [exp4b.log](/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/exp4b.log)

### Inferences
What is feasible:
- Scaling the *environment* to 100×100+ without the full Z: factor Y(f) once per frequency (sparse LU), keep Z·e_probe and Z·e_P for placed decaps, use rank-1 (SMW) updates per placement. Per-step cost is milliseconds; the dense 26 GB / 23 min path disappears.
- A learned *multigrid-structured* inner network (V-cycle/U-Net-like with Galerkin-style coarse levels) is numerically plausible for the band f ≲ 1.5–2·f1, where the classical V-cycle converges in 3–4 cycles independent of N — this is the regime in which "fixed small number of inner iterations, size-independent" is true.

What is not feasible as stated:
- Jacobi-like local message passing with a residual stop: its residual grows (above ~2 GHz, i.e. 0.13–1.3·f1 depending on N) or stalls (below), independent of the network's cleverness unless it learns a non-Jacobi update; and purely local propagation needs ≥ board radius steps (exp1c).
- "Iterations ∝ board size" as the scaling law: the convergent methods scale with the number of wavelengths across the board (f/f1 ∝ f·N), not with N alone, and are O(1) below resonance.
- Claiming speed over classical solvers at 100×100: the baseline is ~20 ms/frequency (one CPU core, SuperLU); GPU direct solvers would be faster still.

What to change:
1. Use the classical sparse solver + SMW as the environment/oracle; drop "build full Z" from the pipeline and from the scaling argument. Compute diag(Z) by selected inversion or at only the 8 feature frequencies, or replace z_ii features with cheaper ones.
2. If a learned inner solver stays in scope, frame it as a learned preconditioner/multigrid inside a Krylov (GMRES/FGMRES) loop, with damping/complex shift on coarse levels, and condition the number of cycles on f/f1 (k·L), not on N. Report iteration counts against GMRES+MG(CSL) (table in Q1) as the classical reference.
3. Restrict H2 claims to f ≲ 2·f1 (or show explicit handling above), and test irregular shapes.
4. Replace the C/ESL lower bound as reward normaliser by the greedy+prune count (or best-known), and add an explicit infeasibility/violation term; keep the bound only as a sanity floor (it was never violated).
5. Because greedy with exact SMW is strong and cheap here (README already notes this), the learning contribution should target what greedy cannot do cheaply: many probes/transfer impedances, cost/type constraints, or boards where even diag(Z) is unaffordable (≥ 300×300, many frequencies).

### Gaps
- No neural network was trained; conclusions are about the numerical operators a network would have to approximate. A trained network could learn non-Jacobi updates, but would then need to be compared against GMRES+MG(CSL), not against Jacobi.
- Placeholder physical constants, rectangular boards in exp1, single-threaded CPU timings.
