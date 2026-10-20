
(N=10/20 rows with decaps are in the CSVs; same pattern, e.g. N=20 + 8 decaps: MG-V(Y) div@74 at 3.75 GHz and div@67 at 10 GHz.)

- Jacobi family: bare boards never converge (any N, any f); ρJ ≥ 1.000000 at all 40 bare (N, f) points; divergence starts at 5 GHz (N=10, f/f1 = 0.33), 3.75 GHz (N=20), 4.5 GHz (N=50), 5 GHz (N=100). Number of eigenvalues with |λ| > 1: 2 (one ± pair; grid is bipartite) below f1, 6–16 above f1 (N=10/20) — [exp1_N10.csv](FEAS/exp1_N10.csv), [exp1_N20.csv](FEAS/exp1_N20.csv)
- Decaps occasionally make ρJ < 1 (decap ESR damps the bulk mode; ρJ = 0.9932–0.9999), giving convergence after 330–19,332 sweeps, but non-monotonically in f and N (e.g. N=100 + 200 decaps: 1,026 RB-GS sweeps at 1 GHz, no convergence at 0.749 or 2 GHz) — table above.
- Red-black Gauss–Seidel diverges about 2× faster than Jacobi (consistent with ρGS = ρJ² for consistently ordered matrices); damping (ω = 2/3) only delays divergence — table above.
- MG V-cycle as a stationary solver: 3–4 cycles for all N at f ≤ 1.5·f1 (bare), 3 cycles even at N=100 up to 2.25 GHz; diverges at f ≥ 3.3·f1 (N=50 at 10/20 GHz, N=100 at 5/10/20 GHz) and near decap-loaded resonances (N=50 4.5 GHz div@105, N=100 2.25 GHz div@223; N=100 2 GHz needs 65 cycles) — table above.
- Unpreconditioned GMRES(50): 32–42 (N=10), 66–396 (N=20), 124–>3000 (N=50), 96–>3000 (N=100) iterations; worst at low frequency (near-singular Laplacian), i.e. iteration count grows ~N² there — table above.
- GMRES + exact CSL: N=100 bare 2, 2, 3, 3, 4, 5, 7, 12, 34, 45 iterations for f/f1 = 0.007 … 13.3; at fixed 10 GHz: 4 (N=10), 7 (N=20), 11 (N=50), 34 (N=100). GMRES + MG(CSL) at N=100: 3–8 up to 1.5·f1, 13–16 at 3.3·f1, 24–31 at 6.7·f1, 116–175 at 13.3·f1 — table above.
- Residual-based stopping is reliable in this lossy model: with GMRES+CSL stopped at rtol 1e-3, relative error of Z_pp was ≤ 1.2e-3 (≤ 0.004 dB) in all 10 tested (N, f) cases incl. 0.98·f1; at rtol 1e-2 the error was ≤ 5.6e-3 (≤ 0.025 dB) — [exp1b.log](FEAS/exp1b.log), [exp1b_residual_vs_error.py](FEAS/exp1b_residual_vs_error.py)
- Locality (what a purely local T-step network can see): Z_pp computed on a (2r+1)² window around the centre of a 100×100 board vs. the full board, |error| in dB — [exp1c.log](FEAS/exp1c.log), [exp1c_locality.py](FEAS/exp1c_locality.py):

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
Source: [exp2_factor_cost.py](FEAS/exp2_factor_cost.py), [exp2.log](FEAS/exp2.log); f = 1 GHz shown (0.1 and 10 GHz within ±10%). LU memory = nnz(L+U) × 20 B (complex128 value + int32 index). scipy SuperLU, single thread; MMD_AT_PLUS_A ordering (COLAMD 1.7–1.8× slower and larger).

| board | nodes n | splu time / freq | LU memory | solve 1 col | solve 16 cols | solve 256 cols | diag(Z) via n cols / freq (extrap.) | full Z, 32 f, complex64 |
|---|---|---|---|---|---|---|---|---|
| 50×50 | 2,500 | 0.004 s | 1.4 MB | 0.19 ms | 1.6 ms | 27 ms | 0.3 s | 1.6 GB |
| 100×100 | 10,000 | 0.019 s | 7.4 MB | 0.70 ms | 6.6 ms | 146 ms | 5.7 s | 25.6 GB |
| 200×200 | 40,000 | 0.113 s | 39 MB | 3.9 ms | 37 ms | 690 ms | 108 s | 410 GB |
| 300×300 | 90,000 | 0.30–0.35 s | 100 MB | 10–12 ms | 91–107 ms | 1.65–2.07 s | 580–730 s | 2.07 TB |

- Dense path (project code, torch, 10 threads, complex128 solve): `build_Z` on a full rect board took 0.50 s (20×20), 3.67 s (30×30), 13.9 s (40×40), 35.8 s (50×50; Z = 1.60 GB complex64). One dense inverse at a single frequency: 6.2 s (70×70), 43.1 s (100×100) → ×32 frequencies ≈ 23 min for 100×100 — [exp2.log](FEAS/exp2.log). (Dense timings were taken while two exp1 processes were running in the background, so they may be ~10–20% pessimistic.)
- Peak RSS of the run (50×50 `build_Z` + one 100×100 dense inverse) was 18.1 GB — [exp2.log](FEAS/exp2.log)
- Measured end-to-end probe-only setup for all 32 frequencies: factorize 0.15 s + probe column 0.006 s (50×50); 0.67 s + 0.024 s (100×100) — [exp3.log](FEAS/exp3.log)
- Measured diag(Z) for all nodes, 32 frequencies, via chunked (512-RHS) LU solves: 9.4 s (50×50); 196 s (100×100, extrapolated from 2 frequencies) — [exp3.log](FEAS/exp3.log)

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
Source: [exp3_update_cost.py](FEAS/exp3_update_cost.py), [exp3.log](FEAS/exp3.log). 32 frequencies, 16 random decaps added one at a time, centre probe.

| board | one-time: factor 32 f | one-time: probe col | one-time: diag(Z) 32 f | refactor + probe solve per decap (32 f) | SMW commit per decap (32 f) | SMW evaluate all n×3 candidates (32 f), k = 1 / 16 placed | per-candidate (32 f) | max rel. err SMW vs refactor |
|---|---|---|---|---|---|---|---|---|
| 50×50 | 0.15 s | 0.006 s | 9.4 s | 0.133–0.140 s | 0.0052 s | 0.004 s / 0.090 s | 4.65 µs | 5.6e-13 |
| 100×100 | 0.67 s | 0.024 s | 196 s (extrap.) | 0.655–0.688 s | 0.024 s | 0.016 s / 0.366 s | 4.65 µs | 2.7e-13 |

- Exhaustive greedy step by refactorization would take 0.3 h (50×50, 7,500 candidates) or 5.6 h (100×100, 30,000 candidates); with SMW it is 0.004–0.37 s — [exp3.log](FEAS/exp3.log)

### Inferences
- For each decap placement step, the inner problem is a rank-1 update of a fixed factorization, not a new linear solve. An inner loop that re-solves Y v = i iteratively at every step (as the proposal implies) would be strictly more work than the classical route; its only possible advantage is avoiding the one-time diag(Z) cost.
- The bottleneck for a 100×100 greedy/RL environment is the one-time diag(Z) (196 s, 32 f) — not per-step updates.

### Gaps
- Timings are CPU/numpy; the einsum for k = 16 is not optimised (O(F·n·k²)), so the k-dependence is an upper bound.

## Q4. Is the "physics lower bound" (C-based + ESL-based decap count) tight enough to normalise rewards?

### Takeaway
Only in the trivial regime. When the target band starts at 10 MHz and stops at ≤ 1 GHz, the bulk-capacitance bound ceil(C_needed/C_decap) is exact (ratio 1.00 in 36/36 cases) — but that regime is location-insensitive. For the location-sensitive bands this project actually rewards (≥ 100 MHz), the simple bound is 1–2 decaps while greedy needs 2–13 (ratio 1.0–9.0, median ≈ 3.5); a stronger per-frequency admittance-sum MILP bound only improves this to 1.0–5.5. The bounds were never violated (0 of 108 feasible cases), but they cannot detect infeasibility (12 of 120 setting-instances had no solution with ≤ 40–60 decaps while the bound said 2–16).

### Cited Findings
Setup: random 10×10 and 20×20 boards (rect/L/holes, random probe, 0–30% keep-out, as in `pdn.sample_batch`), 40 log-spaced frequencies 10 MHz–20 GHz, flat target Zt inside the band. Greedy min-count = add the (node, type) that most reduces Σ_f max(0, log10|Z|/Zt) until |Z| ≤ Zt in band (cap 60 resp. 40 decaps), then repeatedly drop any decap whose removal keeps feasibility. Bounds: LB_C = ceil(max(0, 1/(2π f_lo Zt) − C_plane)/10 nF); LB_ESL = ceil(2π f_hi · 10 pH / Zt); LB_Y = integer min Σ n_t s.t. |1/Z0_pp(f)| + Σ_t n_t |1/z_t(f)| ≥ 1/Zt ∀ f in band (exact if all decaps sat on the probe node). "Local search found k−1" = random swap hill-climbing (600 moves) found a feasible placement with one decap fewer than greedy. Sources: [exp4_lib.py](FEAS/exp4_lib.py), [exp4_bound.py](FEAS/exp4_bound.py), [exp4.log](FEAS/exp4.log), [exp4b_bound_band.py](FEAS/exp4b_bound_band.py), [exp4b.log](FEAS/exp4b.log), target exploration [exp4_explore.log](FEAS/exp4_explore.log).

Band starting at 10 MHz ([exp4.log](FEAS/exp4.log)):

