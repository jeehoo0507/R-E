# Learned iterative solvers and looped networks: can they deliver H1 and H2 for decap placement?

Scope: whether a weight-tied, physics-structured message-passing loop that iterates on Y(f)·v = i of a 2D RLGC plane-pair grid (trained on 8–12 cell boards, tested on 15–100+) can (H1) generalize to larger boards better than coordinate-based transformer RL and (H2) hold its performance as boards grow by stopping on the physics residual.

Note on evidence types: items marked **[own numerical check]** come from a small numpy experiment run for these notes (script: normalized 2D Neumann-boundary plane-pair grid, per-cell L = C = 1, a = ω²LC, loss tanδ and R/ωL, N random decaps with ESR/ESL/C). It is not a published source. Treat it as a sanity check that the report-writer may quote with that caveat.

---

## Q1. Size extrapolation of looped / recurrent models: evidence, sizes, failure modes

### Takeaway
Weight-tied recurrent models can extrapolate far beyond training size (9×9 to 201×201 and 801×801 mazes; 32-bit to 512-bit prefix sums) on discrete, local, algorithmic tasks. They need specific fixes: input re-injection ("recall"), progressive loss, and many more test-time iterations. Without these fixes they "overthink" and drop to 0% on larger inputs. Adaptive-depth looped transformers also extrapolate (about 2× training length), but they needed ground-truth step-count supervision. General GNNs on CLRS lose a lot of accuracy going from n = 16 to n = 64.

### Cited Findings
- **Bansal et al. 2022 (DeepThinking with recall + progressive loss).**
  - Training: 9×9 mazes with at most 30 recurrent iterations.
  - 59×59 mazes: DT-Recall with progressive loss (α = 0.01) reaches 97.30% ± 0.68 at about 984 iterations. Recall alone (α = 0) reaches 82.72% ± 15.14. Plain DT (no recall) and feed-forward nets reach **0%**.
  - 201×201 mazes: about 74% solved with 2,000 iterations. Many failures have only one or a few wrong pixels.
  - 801×801 mazes: solved, using 10,000–20,000 iterations. The paper is internally inconsistent on the exact count. — [Bansal et al. 2022, arXiv 2202.05826](https://arxiv.org/html/2202.05826v3)
- **Bansal et al. 2022, prefix sums.** Trained on 32-bit strings. At 512 bits, DT-Recall with progressive loss reaches 97.12% ± 1.88 (iteration 237). Plain DT reaches 0.00% (α = 0) or 11.26% (α = 1). Feed-forward reaches 0%. — [Bansal et al. 2022](https://arxiv.org/html/2202.05826v3)
- **Overthinking and instability (Bansal et al.).**
  - Plain DT chess accuracy decays after about 70 iterations.
  - Adding noise to features after one iteration drops plain DT to 0% on 13×13 mazes, while DT-Recall stays at 100%.
  - Swapping in a different maze's features after 50 iterations sends plain DT to 0%, while DT-Recall recovers to 97%.
  - The abstract describes the degeneration under many iterations as "overthinking". Recall (keeping a copy of the input) and progressive training are the stated cures. — [Bansal et al. 2022 abstract](https://arxiv.org/abs/2202.05826); [HTML](https://arxiv.org/html/2202.05826v3)
- **Looped transformers (Fan et al., ICLR 2025).**
  - Training lengths: 1–20 for parity, 1–19 for copy, addition, binary sum and unique set, 1–11 for multiplication.
  - Test lengths: parity 40, addition 30, copy 35, multiplication 16, binary sum 24, unique set 35. Accuracy is about 1 at the selected step.
  - Vanilla next-token prediction fails at training max + 10. A fixed-depth (20-step) looped model "falls behind" the adaptive one.
  - Training **requires the ground-truth step count T(n) per sample**.
  - The stopping rule is maximum confidence (lowest cross-entropy over the step range). Per-sample stopping works almost as well on converging tasks and is worse on parity and binary sum.
  - Input injection helps. — [Fan et al. 2024/2025, arXiv 2409.15647](https://arxiv.org/html/2409.15647v4)
- **CLRS-30 neural algorithmic reasoning.** The OOD protocol trains on 16-node inputs and tests on 64-node inputs — [CLRS leaderboard summary](https://alphaxiv.org/benchmarks/google-deepmind/clrs). Average single-task OOD micro-F1 at that size jump:

  | Model | Avg OOD micro-F1 (n = 16 → 64) |
  |---|---|
  | Triplet-GMPNN | 74.14% |
  | PGN | 50.84% |
  | MPNN | 44.99% |
  | Memnet | 38.88% |

  — [Ibarz et al. 2022, arXiv 2209.11142](https://arxiv.org/pdf/2209.11142)
- **Size generalization of GNNs in general.** Yehudai et al. prove that when local structure differs between small and large graphs, there exist "bad" global minima that fit small graphs but fail on large ones. Empirically, GNNs tend to converge to these non-generalizing solutions. — [Yehudai et al., ICML 2021](https://proceedings.mlr.press/v139/yehudai21a.html); [arXiv 2010.08853](https://arxiv.org/abs/2010.08853v2)

### Inferences
- The positive results (Bansal, Fan) share three properties:
  1. The correct per-step rule is purely local and identical at every scale. Maze dead-end filling and carry propagation are examples.
  2. The task has a discrete, self-correcting fixed point.
  3. Iteration count scales roughly linearly with size: about 30 iterations for 9×9, about 1,000 for 59×59, and 2,000–20,000 for 201–801.

  The PDN inner loop meets (1) only partly. The Jacobi-like update is local, but the solution of Y·v = i near resonance is a global standing wave. It does not meet (2): the fixed point is continuous, and the classical local iteration does not contract (see Q3).
- Iteration budgets scale steeply. If board size grows 8× (12 → 100 cells per side), a local-propagation loop needs at least about 8× more iterations just for information to cross the board. If the loop behaves like an unpreconditioned elliptic iteration, it needs about 64× more. The coarse-grid attention in the proposal targets exactly this problem. Its own generalization is then the risk, as with Lerer et al. in Q3.
- Board edges, corners and decap sites are the "local structures" whose proportions change with board size. Edge/interior ratio is about 4/N, so 33% at N = 12 versus 4% at N = 100. Yehudai's result therefore applies directly: degree and boundary statistics shift with N.
- Bansal's 0% for plain recurrent nets on 59×59 is direct evidence for the H2 intuition that fixed-depth models fail. It is equally evidence that a looped model fails without recall and progressive loss. Both must be in the design. Input re-injection is already in the proposal. Progressive loss, or an equivalent random-depth training scheme, is not.

### Gaps
- I did not fetch primary numbers this pass for: Schwarzschild et al. 2021 (the earlier "Can you learn an algorithm?" paper), Universal Transformers, PonderNet/ACT, DEQ stability results, Neural Cellular Automata, or Veličković et al. 2020 "Neural execution of graph algorithms". The qualitative picture above does not depend on them, but no numbers from them are reported here.
- No looped-network study I found works on continuous, oscillatory (wave/resonant) fixed points as its algorithmic task. The closest are the learned Helmholtz solvers in Q3.

---

## Q2. Learned iterative PDE / linear solvers: do they generalize to larger grids than trained?

### Takeaway
Yes, moderately, on positive-definite (Poisson/Laplace) problems, especially when the network corrects a classical convergent iteration rather than replacing it. Reported jumps:

| Method | Train size | Test size | Problem class |
|---|---|---|---|
| Hsieh et al. 2019 (Conv) | 16×16 | 64×64 | Laplace |
| Hsieh et al. 2019 (U-Net) | 64×64 | 256×256 (timed up to 1024) | Laplace |
| Deep Statistical Solvers | ~500 nodes | 750–1000 nodes | Poisson (FEM) |

Accuracy in these is either "correct by construction" (Hsieh) or ~1e-3 NRMSE (DSS). Fully learned solvers trained on a physics-residual loss reach engineering accuracy but not solver accuracy. DSS power-flow residuals are about 1e-5 to 1e-6, versus about 1e-12 for Newton-Raphson.

### Cited Findings
- **Hsieh et al. 2019 (ICLR), learned correction to Jacobi.**
  - Update rule: Φ_H(u) = Ψ(u) + G·H(Ψ(u) − u). Ψ is the Jacobi step and H is a learned convolution. The correction term is zero at Ψ's fixed point, so any fixed point is the true PDE solution.
  - Validity condition: spectral radius of (T + GHT − GH) < 1. This condition does not depend on f or the boundary values. If the iterator fails, "it cannot produce a wrong answer; it simply fails to converge."
  - Train/test sizes: Conv models trained on 16×16, tested on 64×64. U-Net trained on 64×64, tested on 256×256, with timings for 256/512/1024.
  - Cost: about 2–3× fewer operations than Jacobi or multigrid.
  - Scope: only 2D Poisson/Laplace with Dirichlet boundaries. Generalization across sizes is empirical, not guaranteed. — [Hsieh et al. 2019, full text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2019/learning_neural_pde_solvers_with_convergence_guarantees__ceb96d3a.txt); [arXiv 1906.01200](https://arxiv.org/pdf/1906.01200)
- **Deep Statistical Solvers (Donon et al., NeurIPS 2020).** A GNN trained by minimizing the physics residual directly (Kirchhoff violation), with no labels — [DSS full text](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2020/deep_statistical_solvers__73bd83b5.txt).
  - **Poisson:** k = 30 message-passing steps, about equal to the mesh diameter. Training graphs have about 500 nodes (max 599). NRMSE vs LU is 1.6e-3, with correlation above 99.99%. Median loss is 1.2e-3 for DSS versus 6e-26 for LU.
  - **"Super-generalization":** on unseen 100–250 and 750–1000-node meshes, correlation stays about 99.99%. The authors say larger sizes are needed to find the limit.
  - **AC power flow (IEEE 14 and 118):**

    | Metric | case14 | case118 |
    |---|---|---|
    | Correlation with Newton-Raphson, \|V\| | 99.93% | 99.79% |
    | Correlation with Newton-Raphson, θ | 99.86% | **81.31%** |
    | Median residual, DSS | 1.0e-5 | 1.7e-6 |
    | Median residual, Newton-Raphson | 2.1e-12 | 4.2e-14 |
    | Speed-up vs Newton-Raphson | ~2000× | ~100× |

  - Power-flow generalization to *different/larger* topologies was **not tested**. Only random line outages were used. The theory covers only graphs in the support of the training distribution.
- **DSS review caveat.** A NeurIPS reviewer noted that the high correlations are mostly for P/Q, and that losses are large compared with Newton-Raphson. — [NeurIPS 2020 review](https://proceedings.neurips.cc/paper/2020/file/5a16bce575f3ddce9c819de125ba0029-Review.html)
- **Greenfeld et al. 2019.** Learns multigrid prolongation operators for a family of 2D diffusion PDEs with an unsupervised loss, and reports improved convergence rates over Black-Box multigrid. Exact grid-size numbers were not retrieved (abstract only). — [arXiv 1902.10248](https://arxiv.org/abs/1902.10248)

### Inferences
- What generalizes reliably in this literature is a *learned modification of a convergent classical iteration*, as in Hsieh, Greenfeld/Luz and learned preconditioners. The fixed point stays exact, and the network only speeds things up. Fully learned residual-minimizers (DSS) generalize somewhat (about 2× node count) but plateau at modest accuracy.
- The Poisson/diffusion problems in Q2 are symmetric positive (semi)definite, so a Jacobi base iteration converges. The proposal's "one Jacobi step with admittance weights" inherits that convergence **only if Y is in that class**. Q3 shows the PDN Y is not.
- DSS needed k ≈ graph diameter in steps. A 100×100 board then needs at least about 100–200 steps without coarse-grid help, versus about 12–24 at training size. That is the same scaling problem H2 wants to address.

### Gaps
- Not retrieved this pass, so no numbers are reported for: Luz et al. 2020 (learned AMG), UGrid (Han et al. 2024), MgNO, NeuralMG, Li et al. 2023 and Häusner et al. 2023 (learned preconditioners), Brandstetter et al. 2022 (MP-PDE solvers, which are time-steppers rather than linear solvers), and DeepOPF.
- I found no learned solver that was trained on the *complex-symmetric, non-Hermitian* admittance matrix of a PDN or circuit and tested on larger grids.

---

## Q3. Critical physics risk: indefinite (Helmholtz-like) Y(f)

### Takeaway
The risk is real and somewhat worse than the proposal assumes. Lossless or realistically lossy plane-pair Y(f) is indefinite at **every** nonzero frequency, not only above the first resonance. Plain Jacobi on it diverges in my check at all tested frequencies, and realistic dielectric/conductor loss (about 2%) does not rescue it. The literature agrees that classical relaxation and multigrid fail on indefinite Helmholtz.

The best learned Helmholtz solvers do work on 8–32× larger grids than trained. However:
- they embed the network as a *preconditioner* inside FGMRES with shifted-Laplacian multigrid;
- they use an artificially large training damping;
- their iteration counts still grow with size, from about 19 at 128² to about 143 at 4096²;
- some instances fail to converge.

A free-standing weight-tied loop with "stop when residual small" has no such safety net.

### Cited Findings
- **Ernst & Gander 2012** ("Why it is difficult to solve Helmholtz problems with classical iterative methods", LNCSE 83, pp. 325–363). Unpreconditioned Krylov methods stagnate, ILU can make things worse, and classical domain decomposition and multigrid fail to converge. Indefiniteness is the root cause. — [Ernst & Gander, BCAM seminar abstract](https://bcamath.org/bcam-seminar-why-it-so-difficult-solve-helmholtz-problems-classical-iterative-methods); [book chapter record](https://archive-ouverte.unige.ch/unige:171313)
- **Lerer, Ben-Yair & Treister (2023/2024), restating the classical facts.** Relaxation methods "do not converge due to the indefiniteness" of the Helmholtz matrix, though one or two sweeps still smooth the error. Standard multigrid "does not perform well" for Helmholtz. Smooth error modes stay oscillatory at high wavenumber, so performance deteriorates as levels are added. — [Lerer et al., arXiv 2306.17486](https://arxiv.org/html/2306.17486v3)
- **How much damping is needed (Cocquet & Gander, SISC 2017).** The complex shift must be at least O(k²) for multigrid to invert the shifted operator. But a shift of at most O(k) is needed for the shifted operator to be a good, wavenumber-independent preconditioner (Gander, Graham & Spence). The gap between the two means preconditioned GMRES iterations "inevitably grow like O(k²)" with one V-cycle. — [Cocquet & Gander, unige archive](https://archive-ouverte.unige.ch/unige:168606); [HAL copy](https://hal.univ-reunion.fr/hal-01622022/document); [Gander, Graham & Spence](https://people.bath.ac.uk/eas25/GaGrSp14.pdf)
- **Learned Helmholtz preconditioner (Lerer et al. 2023/24, extending Azulay & Treister, SISC 2022).**
  - Setup: a U-Net, with an implicit FFT-inverted kernel at the coarsest level, used as a preconditioner inside FGMRES(10) together with a shifted-Laplacian V-cycle. Training is multiscale on 128²/256²/512².
  - Tolerance: residual reduced by 1e-7, capped at 2,000 iterations.
  - Iterations, implicit U-Net trained up to 512² (OpenFWI): 18.82, 27.34, 40.62, 63.39, 94.13 and 143.03 at 128², 256², 512², 1K², 2K² and 4K².
  - Iterations, shifted-Laplacian V-cycle alone: 112.9 at 128², 1983.8 at 1K², and more than 2,000 at 2K² and 4K².
  - Networks trained only on 128² or 256² "may perform poorly or even fail to converge" at 4K². Up to 8% of instances did not converge.
  - Training used a high damping γ = 0.05 because it gives more consistent training. Out-of-distribution media degrade iteration counts.
  - Azulay & Treister's original version reports about 25, 52 and 101 iterations at 128², 256² and 512² (CIFAR-10 media), with no results at 1K² and above. — [Lerer et al., arXiv 2306.17486](https://arxiv.org/html/2306.17486v3); [Azulay & Treister, arXiv 2203.11025](https://ar5iv.labs.arxiv.org/html/2203.11025)
- **Fully learned Helmholtz iterative solver (Stanziola et al., J. Comput. Phys. 2021).** — [Stanziola et al., arXiv 2010.15761](https://ar5iv.arxiv.org/html/2010.15761)
  - Model: a UNet with a recurrent hidden state. Inputs are the field, the residual and the PML channels, with update u ← u + Δu. It is trained on the physics residual, on 96×96 grids at 2π points per wavelength, with lossless Helmholtz and a PML.
  - Test set (96×96, 1,000 iterations): mean ℓ∞ error 0.36% versus k-Wave. Outliers reach a few percent; these have whispering-gallery / mode-like fields with "oscillating residuals".
  - Versus GMRES at 200 iterations: learned solver has residual 5.6e-5 and 1.0% error; GMRES has residual 5.7e-3 and 20.4% error.
  - Larger domains:
    - 480×480 (5× per side): **1% error after about 600 iterations**.
    - 512×512 CT skull: final 0.8% error, but the error reached its minimum only after about 3,000 iterations.
  - At intermediate residuals (~1e-4) the error "varied widely".
  - Stated limitations: no convergence guarantee, and validation loss oscillates during training. The model is fixed to one points-per-wavelength value and needs retraining for other frequencies/resolutions.
- **[own numerical check] Jacobi on a plane-pair Y.**
  - Setup: normalized 2D grid with Neumann (open-edge) boundary. Y = jωC(1 − j·tanδ)·I + Lap/(R + jωL), with N random decaps.
  - The Jacobi spectral radius ρ_J was **above 1 at every tested frequency and size** (N = 10, 20, 30):
    - Below the first plane resonance (a = 0.3·λ₁): ρ_J ≈ 1.001–1.008, i.e. slow divergence.
    - Several resonances up (a = 0.5): ρ_J ≈ 1.16.
    - Near the grid cutoff: ρ_J up to 5, or a vanishing diagonal at edge nodes.
  - 2% loss (tanδ = R/ωL = 0.02) changed ρ_J by less than 0.001. Adding N decaps did not bring it below 1.
  - Analytic reason (interior nodes): the Jacobi eigenvalue is (λ − 4)/(a − 4), with λ ∈ [0, 8] the grid-Laplacian eigenvalue and a = ω²LC per cell. The uniform mode (λ = 0) gives 4/(4 − a) > 1 for any 0 < a < 4.
  - The number of eigenmodes "on the other side" of zero was 1, 4, 22/43 and 84/180 as frequency rises. It grows with board area at fixed physical cell size.
  - Even the **best complex-damped Jacobi** (coarse grid search over the complex step) needed about 1e4–6e4 iterations to reach 1e-6 at N = 10–30 for low frequency. It **diverged** (ρ ≈ 1.0001–1.0002) at a = 0.5 with 0–2% loss. Only 10% loss made it converge, and then needed about 8,000–17,000 iterations.
  - The 2-norm condition number of the lossy Y was about 1e2–2.4e3, larger for bigger boards at low frequency.

### Inferences
- **Lossiness helps only marginally.**
  - Dielectric loss tanδ (~0.01–0.02 for FR-4-like dielectrics; typical-value assumption, not sourced here) acts as a complex shift of relative size ~tanδ·k². Cocquet & Gander show multigrid needs a shift of order k² with an O(1) constant. The learned-preconditioner papers use β = 0.5 in the shifted Laplacian and γ = 0.05 during training.
  - Physical PDN damping is therefore far too small to make a Jacobi/multigrid-like loop converge.
  - Decap ESR damps only near the decap sites, and only in the band where the decap is effective.
- **Indefinite even below the first resonance.**
  - For a plane pair with open edges, the zero-wavenumber "capacitive" mode is always on the opposite side of the spectrum from the inductive modes. So Y is indefinite at every frequency, mildly below the first resonance and increasingly so above it.
  - The proposal's framing ("indefinite above first resonance") understates this. Below resonance the problem is mild, but plain Jacobi still diverges (ρ slightly above 1).
- **Residual-based stopping near resonance is unreliable for two reasons.**
  1. The residual may not decrease monotonically (Stanziola: "oscillating residuals" in mode-like cases), or may never fall below threshold (divergent base iteration).
  2. Residual and error are decoupled by the condition number: ‖e‖ ≤ ‖Y⁻¹‖·‖r‖, with cond(Y) up to about 1e3 in my small test. Near a high-Q resonance, ‖Y⁻¹‖ is exactly the impedance peak the optimizer cares about.

  Stanziola found errors varying widely at residual ~1e-4. A threshold calibrated on 12-cell boards will not mean the same error on 100-cell boards.
- **Iteration count vs size.**
  - Even the best learned+MG+Krylov hybrid shows iteration growth of about 7.6× from 128² to 4096² (Lerer). Stanziola's fully learned solver needs 600–3,000 iterations at 5× size.
  - At inference the proposal would need hundreds to thousands of loop iterations per frequency point, times tens to hundreds of frequency points, times every RL step. Training through such loops with BPTT is also expensive; Stanziola needed truncated BPTT plus a replay buffer.

### Gaps
- I could not access the Ernst & Gander chapter full text; claims come from the abstract/seminar description and from later papers that restate them.
- No learned solver for the specific *2D RLGC plane-pair / PDN* admittance problem was found.
- Typical tanδ and ESR-based damping values for HBM interposer/package PDNs were not sourced here. The 2% figure is an assumption.
- My numerical check uses normalized units, a uniform grid, Neumann boundaries and a coarse search over the damping parameter. A real PDN with VRM, package and port terminations adds boundary damping and could shift the numbers, though the analysis suggests it will not change the sign of the conclusion.

---

## Q4. Is solving Y·v = i with a learned loop necessary? Hybrid alternatives

### Takeaway
Probably not. The learned-solver literature that generalizes best keeps a classical solver in charge of correctness:
- Hsieh: fixed point preserved by construction.
- Lerer/Azulay: network inside FGMRES with multigrid.
- Even DSS: correct only to about 1e-3.

For 2D PDN grids of up to about 100×100 cells (~1e4 nodes), a classical sparse direct or Krylov solve is cheap and exact. So the simplest generalizable design is: **the classical solver computes physics features, and the network does only the decision part**.

### Cited Findings
- Learned *corrections* to classical solvers keep exactness and still generalize across grid size (16 → 64, 64 → 256) and geometry. — [Hsieh et al. 2019](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/ICLR/2019/learning_neural_pde_solvers_with_convergence_guarantees__ceb96d3a.txt)
- Learned preconditioners inside a Krylov method reach 4K² grids. A non-converging preconditioner shows up as a high iteration count or failure, not a silently wrong answer. — [Lerer et al. 2023/24](https://arxiv.org/html/2306.17486v3)
- Fully learned physics-residual solvers are fast (100–2000× vs Newton-Raphson on IEEE 14/118). But their residuals are about 6–8 orders of magnitude larger, and angle correlation drops to 81% on the 118-bus case. — [Donon et al. 2020](https://object.cloud.sdsc.edu/v1/AUTH_da4962d3368042ac8337e2dfdd3e7bf3/ml-papers-txt/NeurIPS/2020/deep_statistical_solvers__73bd83b5.txt)
- Algorithmic-reasoning OOD accuracy (n = 16 → 64) is well below in-distribution even for the best processor: 74% average micro-F1. Learning the *computation* itself is the brittle part. — [Ibarz et al. 2022](https://arxiv.org/pdf/2209.11142)

### Inferences
These points are standard circuit and numerical-linear-algebra facts, not sourced in this pass.
- **Cost of a direct solve.**
  - A 2D grid with n nodes factorizes by nested dissection in about O(n^1.5) flops with O(n log n) fill. For n = 1e4 (a 100×100 board) this is milliseconds per frequency.
  - Because Y is complex symmetric, one factorization gives both forward and adjoint solves (Yᵀ = Y). Sensitivities ∂Z/∂y_k at all candidate sites therefore cost one extra triangular solve, not a new factorization.
- **Decaps as low-rank updates.**
  - Adding a decap changes a single diagonal entry, which is a rank-1 update.
  - The standard PDN workflow:
    1. Compute the board's port impedance matrix Z_ports(f) once over the candidate decap sites plus the observation port(s).
    2. Treat decaps as terminations by Sherman–Morrison/Woodbury or network-parameter reduction.
  - After step 1, evaluating any decap configuration requires **no** grid solve at all. This removes the inner loop's raison d'être for the reward computation.
- **Recommended hybrid.** A sparse direct (or AMG/shifted-Laplacian-preconditioned GMRES) solver produces per-node or per-site physics features: |Z_self(f)|, |Z_transfer to the IC port(f)|, adjoint sensitivity, and dominant mode shapes at the resonances inside the target band. A GNN/transformer policy acts on the board graph with these features.
  - Generalization to larger boards then rests on features that are already *exact* at any size.
  - The policy only needs to learn local decision rules ("place where transfer impedance × sensitivity is largest, accounting for the rank-1 interaction"). The proposal's intended inductive bias is kept, but the brittle part is removed.
- **If a learned loop is retained for research novelty**, the evidence-backed form is:
  - a Hsieh-style correction or a Lerer-style preconditioner around a convergent base method, such as GMRES/FGMRES with a shifted-Laplacian multigrid;
  - with the residual check performed by the Krylov method, not by the network;
  - trained with recall plus progressive/random-depth loss (Bansal);
  - trained multiscale, including at least some boards larger than 12 cells (Lerer showed 128/256-only training fails at 4K²).

### Gaps
- I did not find a published head-to-head comparison of "physics features from a direct solver + NN policy" versus "fully learned solver + policy" on decap or PDN placement specifically. The inference rests on the general-solver literature.
- Exact timing of scipy `splu` on 1e4-node complex grids was not benchmarked here; scipy was unavailable in this environment.

---

## Q5. Bottom line: feasibility of H1 and H2, and the most realistic modification

### Takeaway
- **H1 (physics-structured loop generalizes better than coordinate-transformer RL): MEDIUM if "physics-structured" means physics features or graph locality; LOW if it relies on the learned Y·v = i loop being accurate.**
- **H2 (residual-based iteration count keeps performance as the board grows): LOW as stated.**
- The most realistic modification is a hybrid: an exact classical solve or port-Z precomputation provides physics features, and a size-agnostic GNN/attention policy makes the decisions. H2 is reframed as "solver cost scales, policy accuracy holds".

### Cited Findings
- Support for the *direction* of H1: local, weight-tied rules with input injection extrapolate. Bansal: 9×9 → 201×201 at about 74% and 801×801 solved. Fan: about 2× length. — [Bansal et al. 2022](https://arxiv.org/html/2202.05826v3); [Fan et al. 2024](https://arxiv.org/html/2409.15647v4)
- Risk to H1: GNN size-generalization fails when local-structure statistics change with size. — [Yehudai et al. 2021](https://proceedings.mlr.press/v139/yehudai21a.html). Even the best OOD processor averages 74% on n = 16 → 64. — [Ibarz et al. 2022](https://arxiv.org/pdf/2209.11142)
- Risk to H2:
  - Plain recurrent models collapse to 0% at larger size without recall and progressive loss. — [Bansal et al. 2022](https://arxiv.org/html/2202.05826v3)
  - Indefinite Helmholtz breaks relaxation and multigrid. — [Ernst & Gander](https://bcamath.org/bcam-seminar-why-it-so-difficult-solve-helmholtz-problems-classical-iterative-methods); [Lerer et al.](https://arxiv.org/html/2306.17486v3)
  - Required damping is of order k². — [Cocquet & Gander](https://archive-ouverte.unige.ch/unige:168606)
  - Iterations grow 7.6× from 128² to 4K² even with MG + Krylov + learned preconditioner, and some instances fail. — [Lerer et al.](https://arxiv.org/html/2306.17486v3)
  - The fully learned solver needs 600–3,000 iterations at 5× size, and residual ~1e-4 does not pin down error. — [Stanziola et al.](https://ar5iv.arxiv.org/html/2010.15761)
- Adaptive iteration counts can work, but the successful demonstration used ground-truth step counts for supervision and a confidence-based stopping rule, not a physics residual. — [Fan et al. 2024](https://arxiv.org/html/2409.15647v4)

### Inferences
- **H1, medium.** A coordinate-based transformer (absolute (x, y) inputs) has a well-known reason to extrapolate poorly: unseen coordinate ranges. A local, translation-equivariant GNN plausibly beats it. But the expected gain comes from locality and physics *inputs*. A learned inner solver is not needed for it. Test H1 with an ablation:
  - (a) coordinate transformer;
  - (b) GNN policy with exact solver features;
  - (c) GNN policy with learned-loop features.

  The evidence above predicts (b) > (c) on 50–100+ cell boards.
- **H2, low.** Reasons:
  1. The natural base iteration (Jacobi with admittance weights) diverges for plane-pair Y at all frequencies (own check), and realistic loss does not fix this.
  2. Where learned Helmholtz solvers succeed, they rely on Krylov + multigrid + large artificial damping, and iteration counts still grow with size.
  3. The residual is a poor error proxy exactly at resonances, which drive the decap objective.
  4. A board 8× larger implies at least 8× and plausibly 50×+ iterations per frequency per RL step.

  H2's *intuition* (fixed-depth propagation limits) is supported. Its *proposed fix* (learned loop with residual stopping) is not.
- **Most realistic modification (ranked).**
  1. **Precompute per board** the port/candidate-site impedance matrix Z(f) with a sparse direct solver. Model decaps as rank-1 terminations, giving exact, cheap reward evaluation. Feed |Z_self|, |Z_to-IC|, sensitivities and resonance mode shapes as node features to a GNN or local-attention policy with relative (not absolute) positional encodings. Train on 8–12 cells with **some multiscale exposure** (e.g. 16–20), and test at 30–100+.
  2. If a learned solver component is desired, use it as a **preconditioner inside FGMRES** (Lerer-style) or a **fixed-point-preserving correction** (Hsieh-style). The Krylov residual governs stopping, which gives the honest version of H2: "solver iterations adapt to size; policy quality does not degrade".
  3. Reformulate H2 as a measurable claim: compare fixed-k message-passing GNN policies (k = 3, 6, 12) against policies fed global solver features, on 15/20/30/50/100-cell boards. The literature predicts fixed-k fails once board diameter exceeds about k cells, while solver-feature policies hold.
- **Training details if any loop remains:** input re-injection (already proposed), progressive/random-depth loss, multiscale training data, and frequency normalization (points per wavelength per cell). Stanziola's solver needed retraining for new points-per-wavelength. Without normalization, a board with more cells at the same physical size, or the same cell size and a larger area, changes the effective wavenumber per cell.

### Gaps
- No direct published evidence on coordinate-based transformer RL versus GNN for *decap placement* size generalization was gathered in this pass. That comparison belongs to the Park et al. / DevFormer-line literature, outside this note's scope.
- Universal Transformer, PonderNet, ACT and DEQ numbers were not retrieved. Their typical halting-instability and fixed-point-divergence issues are not quantified here.
