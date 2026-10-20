
- Summary over all feasible cases: band from 10 MHz — greedy/simple-LB 1.00–2.75 (median 1.00); band from ≥ 100 MHz — greedy/simple-LB 1.00–9.00 (median 3.50), greedy/LB_Y 1.00–5.50 (median 3.00); bound violations 0/62 and 0/46 — [exp4.log](FEAS/exp4.log), [exp4b.log](FEAS/exp4b.log)
- Stricter targets (0.1–3 GHz @ 0.1/0.05 Ω, 0.3–10 GHz @ 0.2/0.1 Ω, 1–20 GHz @ 0.3 Ω) were infeasible for greedy with 60 decaps on 10×10 boards in 13 of 15 completed cases, while LB_Y said 3–10 — [exp4b_strict_targets_partial.log](FEAS/exp4b_strict_targets_partial.log)
- Greedy is not always optimal: local search beat greedy+prune by one decap in 2/6 cases (10×10, 10 MHz–10 GHz @ 0.5 Ω) and 3/46 cases in the ≥ 100 MHz bands — tables above.

### Inferences
- The C-based bound is the right physics only for the lumped low-frequency regime, where placement does not matter; the ESL-based bound ignores the plane's own capacitance and the spreading inductance between probe and decap, so in the band where placement matters it is 2–9× below what is achievable. Because the true optimum lies between the bound and the greedy count, the ratio greedy/LB mixes "bound slack" and "greedy slack"; but since local search rarely improved greedy (5/88 cases where it was run), most of the gap is bound slack.
- As a reward normaliser it would not equalise difficulty: across instances and targets the ratio spans 1–9. A cheap, much tighter reference is the greedy+prune count itself (on 20×20 boards it runs in well under a second with SMW), or best-known counts.
- A bound that cannot certify infeasibility is a problem for a min-count task: the reward must handle "unreachable target" explicitly (e.g., penalise residual violation), which the bound does not help with.

### Gaps
- Only 10×10 and 20×20 boards, 6 instances per setting, single probe, placeholder decap library; no exact optimum (MILP/exhaustive) was computed, so "greedy slack" is only bounded from below by the local-search check.

## Q5. Implications for the proposal: what is feasible, what is not, what to change

### Takeaway
The outer RL loop on a classical environment is feasible at 100×100–300×300 today (sparse LU + SMW); the "solver-mimicking inner loop" as specified (Jacobi-like message passing + coarse attention, stop on residual) is not justified by numerics: Jacobi-type steps diverge or stall at every frequency of this simulator, a plain multigrid cycle diverges above ~3·f1, and the classical direct route is already ~20 ms/frequency at 100×100. The physics lower bound is too loose (1–9×) to normalise rewards except in the trivial low-frequency regime.

### Cited Findings
- Jacobi/GS: never converge on bare boards (0/40 bare (N, f) points), ρJ ≥ 1 — Q1, [exp1 CSVs](FEAS/exp1_N100.csv)
- MG V-cycle: 3–4 cycles independent of N up to ~1.5–2·f1; diverges at ≥ 3.3·f1 — Q1
- GMRES+CSL: 2–47 iterations, growing with f/f1 — Q1
- Sparse LU 100×100: 19 ms/f; 300×300: ~0.3 s/f, 100 MB — Q2, [exp2.log](FEAS/exp2.log)
- SMW candidate evaluation 4.65 µs vs refactorization 0.14–0.67 s — Q3, [exp3.log](FEAS/exp3.log)
- Lower bound ratio 1.0–9.0 in location-sensitive bands — Q4, [exp4b.log](FEAS/exp4b.log)

### Inferences
What is feasible:
- Scaling the *environment* to 100×100+ without the full Z: factor Y(f) once per frequency (sparse LU), keep Z·e_probe and Z·e_P for placed decaps, use rank-1 (SMW) updates per placement. Per-step cost is milliseconds; the dense 26 GB / 23 min path disappears.
- A learned *multigrid-structured* inner network (V-cycle/U-Net-like with Galerkin-style coarse levels) is numerically plausible for the band f ≲ 1.5–2·f1, where the classical V-cycle converges in 3–4 cycles independent of N — this is the regime in which "fixed small number of inner iterations, size-independent" is true.

What is not feasible as stated:
- Jacobi-like local message passing with a residual stop: it diverges (above ~0.3–1.3·f1) or stalls (below), independent of the network's cleverness unless it learns a non-Jacobi update; and purely local propagation needs ≥ board radius steps (exp1c).
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
