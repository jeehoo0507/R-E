# Numerical feasibility of the "solver-mimicking inner loop" decap proposal (experiments on the project's own PDN simulator)

All evidence below comes from small experiments run on 2026-10-15 on an Apple M4 Pro (14 cores, 48 GB RAM, CPU only) with the project venv (numpy 2.5.4, scipy 1.18.1, torch 2.14.1). No project file was modified; the project's `decap_loop/pdn.py` was only imported. Scripts, logs and CSVs are in the session scratchpad (temporary, copy them if they should be kept):
`/private/tmp/claude-501/-Users-choejihu-Documents-claude-R-E/7f6a048a-578c-40f5-88fe-41e0a04e63ed/scratchpad/feas/` (abbreviated `FEAS/` below). "Source" links point to the script/log that produced each number.

## Q0. Setup: what does the simulator's Y(f) look like?

### Takeaway
`pdn.py`'s Y(f) = y_e(f)·Lap + y_s(f)·I (+ decap admittances on the diagonal). Above ~220 MHz it is, up to a factor, the discrete Helmholtz operator (Lap − k²I): complex-symmetric, indefinite, and with the board's bulk capacitive mode always on the "wrong" side. This is the textbook hard case for local iterative solvers.

### Cited Findings
- Sparse re-implementation of `build_Y` (`FEAS/common.py::sparse_Y`) matches `pdn.build_Y` to 6.9e-17 relative error on an L-shaped 7×9 board, all 32 frequencies — [exp0.log](FEAS/exp0.log), [exp0_validate_resonance.py](FEAS/exp0_validate_resonance.py)
- Default `PDNConfig` lumped values: C_cell = 0.885 pF, L_edge = 12.6 pH, R_edge = 17.2 mΩ; R/L corner at 218 MHz; wave speed 3.0e11 cells/s (0.5 mm cells, εr = 4 → c/2). Decap library: 10 nF/100 pH/20 mΩ, 1 nF/30 pH/50 mΩ, 100 pF/10 pH/100 mΩ — [exp0.log](FEAS/exp0.log), `decap_loop/pdn.py`
- First plane resonance (|Z| peak at corner port, 600 log-spaced points 10 MHz–20 GHz) vs analytic f10 = v/(2·N·cell): N=10 14.94 GHz (analytic 14.99); N=20 7.43 (7.50); N=50 2.94 (3.00); N=100 1.45 (1.50). Bare |Z| at 10 MHz: 180 / 44.9 / 7.19 / 1.8 Ω — [exp0.log](FEAS/exp0.log)
- Consequence: the 32-point band 10 MHz–20 GHz covers f/f1 ≤ 1.33 for N=10 but up to f/f1 = 13 (≈13 half-wavelengths across the board) for N=100.

### Inferences
- With y_s/y_e ≈ (jωC)(R + jωL) → −ω²LC = −k² above the R/L corner, Y ∝ Lap − k²I. The Neumann Laplacian of the board has eigenvalue 0 (uniform mode), so even far below the first resonance there is one eigenvalue −k² < 0 (the plane's capacitance) — Y is indefinite at every frequency in the band except where losses dominate (< ~200 MHz, where Y ≈ (Lap + jωRC·I)/R is a nearly singular diffusion operator).

### Gaps
- Physical constants are the README's placeholders (10 µm dielectric etc.); with a thicker dielectric / larger cells the resonances and R/L corner move, but the operator class (Helmholtz) does not change.

## Q1. Do Jacobi-like local iterations converge on Y(f) v = i, above and below resonance, and how does the iteration count scale with N? (proposal H2 and Risk 1)

### Takeaway
No. Plain/damped Jacobi and Gauss–Seidel never reach 1e-3 relative residual in 20,000 sweeps on a bare board at any N or frequency, and they diverge for f ≳ 0.3·f1 (N=10) or f ≳ 1.3·f1 (N=100): the Jacobi spectral radius is ≥ 1 everywhere (1.000000 at 10 MHz → 1.047 at 20 GHz). A geometric multigrid V-cycle (Jacobi smoother + Galerkin coarse grids, i.e. the idealised version of "Jacobi message passing + coarse-grid attention") converges in 3–4 cycles independent of N up to ~1.5–2·f1, but diverges above ~3·f1 and near decap-loaded resonances. Robust convergence across the whole band needs a Krylov outer loop with a shifted-Laplacian/multigrid preconditioner, and its iteration count grows with the number of wavelengths across the board (≈ with N at fixed frequency): 2–5 iterations below f1, 12 at 3.3·f1, 34 at 6.7·f1, 45 (exact CSL) to 175 (MG-CSL) at 13·f1 for N=100.

### Cited Findings
Experiment: rectangular N×N boards, probe = centre node, b = unit current at probe, x0 = 0, stop at ‖b − Yx‖/‖b‖ ≤ 1e-3. Decaps: none, or random 2% of cells (≥4) with random types. ρJ = spectral radius of I − D⁻¹Y (dense eig for n ≤ 500, ARPACK otherwise; agrees with the asymptotic residual ratio of the Jacobi run to 4–6 digits). Caps: 20,000 sweeps (Jacobi family), 300 V-cycles, 3,000 Krylov iterations. "div@k" = residual > 1e8 at sweep k; ">cap (r=…)" = not converged, final residual shown. GMRES50 = restarted GMRES(50). CSLexact = exact LU of the complex-shifted operator (shunt term × (1 − 0.5j)), i.e. the best case of a shifted-Laplacian preconditioner; MG(·) = one V-cycle (bilinear P, Galerkin PᵀAP, 2+2 damped-Jacobi ω=0.7, direct solve at ≤64 nodes). BiCGStab counts iterations (2 matvecs each). Source: [exp1_iterative.py](FEAS/exp1_iterative.py), CSVs [exp1_N10.csv](FEAS/exp1_N10.csv) [exp1_N20.csv](FEAS/exp1_N20.csv) [exp1_N50.csv](FEAS/exp1_N50.csv) [exp1_N100.csv](FEAS/exp1_N100.csv) (also contain GMRES+Jacobi-precond., ILU(1e-2), BiCGStab+ILU/CSL, number of |λ|>1 eigenvalues).

