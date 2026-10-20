# H3 / H4 / Training-Reward Design: Evidence for the Decap Dual-Loop Proposal

Scope: decap-count minimization methods and numbers (H3), Z-matrix cost and alternatives (H4), training/reward design, practical resources. Researched 2026-10-15. Primary PDFs were downloaded and text-extracted (PAGA TMTT 2024, Zhang TMTT 2023, Park arXiv 2203.15722, DevFormer ICML 2023, Ling Zhang IBIS Nov-2025 slides). Items marked "[not re-fetched]" are well-known references cited from prior knowledge with their canonical URLs. They were not re-opened in this session.

---

## Q1. Decap minimization methods and results: is "add then prune" standard, and how big is the gap to GA?

### Takeaway
The state of the art for "fewest decaps meeting Z_target" is PAGA. It builds a greedy, physics-prioritized add-one-at-a-time initial solution and then runs a GA that progressively deletes low-priority ports and mutates toward fewer decaps. So "add, then reduce" is already the core recipe, but the reduction step is stochastic (GA), not a learned or deterministic add/remove local search. Pure sequential-add greedy is typically 10–40% worse than GA/DRL/PAGA in decap count. That margin is the headroom H3 targets.

### Cited Findings
**PAGA (Jiang, Zhang, Tan, Li, Hwang, Fan, Li, IEEE TMTT 72(8):4498–4507, Aug 2024, DOI 10.1109/TMTT.2024.3356575).**
- Method:
  1. Rank ports by physical loop inductance seen from the IC port. Short ports progressively; this is done once, at one representative frequency.
  2. Build an initial solution by adding decaps one by one at the highest-priority location, picking the best type, until Z_target is met. This takes at most M·N impedance evaluations.
  3. Run a GA with a "port removal" scheme: unused low-priority ports are removed from the search and their rows and columns are deleted from Z. A modified mutation never zeroes "elite" ports (elite ratio 0.5).

  Source: [PAGA, NSF PAR full text](https://par.nsf.gov/servlets/purl/10566602)
- PDN impedance after decaps comes from the segmentation formula Z'_aa = Z_aa − Z_ap (Z_pp + Z_dd)^-1 Z_pa. The authors call this "the most time-consuming part" and say it must be performed repeatedly. Port removal speeds it up because it shrinks the matrix — [PAGA](https://par.nsf.gov/servlets/purl/10566602)
- Z is exported beforehand from SIwave, Cadence OptimizePI, or a boundary-integration solver. Optimization time does not include that extraction — [PAGA](https://par.nsf.gov/servlets/purl/10566602)
- Case 1 setup: random 4-layer board, 120 ports, flat Z_target 0.5 mΩ over 200 kHz–1 MHz, VRM modeled as 3 mΩ + 2.2 nH, population 50, 300 GA iterations, Intel i9 CPU.
- Case 1 results: final decap counts were 79 (Xu GA [7]), 43 (Juang modified GA [8]), 24 (Zhang physics-based GA 2023 [9]) and 24 (PAGA). PAGA's 24 decaps sit on the 24 highest-priority ports. Sequential search [6] and Newton–Hessian [15] "cannot find the optimal solution", and sorting search [16] found no feasible solution — [PAGA](https://par.nsf.gov/servlets/purl/10566602)
- Case 2, real boards:
  - 16-layer board, 123 ports, two-step target (5 mΩ / 11 mΩ). PAGA beat Ansys SIwave running on 24 cores. When the second step was lowered to 10 mΩ, SIwave found no feasible solution in 100 iterations while PAGA found 14 decaps.
  - 12-layer board, 68 ports, R–L target. PAGA used fewer decaps and less time than OptimizePI.

  Source: [PAGA](https://par.nsf.gov/servlets/purl/10566602)
- Case 3: 120 ports with package-size constraints. OptimizePI took about 600 s for 58 decaps over 500 iterations. PAGA reached 48 decaps in under 100 s — [PAGA](https://par.nsf.gov/servlets/purl/10566602)
- IBIS summit slides (Ling Zhang, Nov 7 2025):
  - Board #1 (100 ports): the modified GA found only 37 decaps after 12,000 s. With PAGA's initial solution it found 31 in 3,000 s; with port removal, 31 in 1,000 s.
  - Board #3 (123 ports, real product): EDA Tool A took about 1 min and used 34 decaps; PAGA took about 18 s and used 10.
  - Board #4 (68 ports): EDA Tool B took about 1 min and used 34 decaps; PAGA took about 31 s and used 29.

  Source: [IBIS slides](https://ibis.org/summits/nov25a/zhang.pdf)
- PAGA cost function: if Z_target is satisfied, minimize decap count; otherwise minimize the maximum impedance violation — [IBIS slides](https://ibis.org/summits/nov25a/zhang.pdf)
- PAGA's own stated limitation (directly relevant to H4): it "still needs the simulated S-parameters including the IC ports and decap ports". With extremely many candidate locations in pre-layout design, "simulating and exporting the S-parameters may be very challenging". Future work is to avoid that simulation — [IBIS slides, slide 28](https://ibis.org/summits/nov25a/zhang.pdf)

**Juang et al. modified GA (EMC 2021).**
- Steers the GA toward fewer decaps by progressively increasing the number of forced zeros (empty slots) after each feasible solution is found. This is effectively a stochastic removal schedule — [IBIS slides](https://ibis.org/summits/nov25a/zhang.pdf)

**Sequential full search (de Paulis et al., EMC 2020).**
- Adds decaps one by one, choosing the type and location that minimize the area above Z_target. It is efficient but "cannot guarantee the optimal solution" — [IBIS slides](https://ibis.org/summits/nov25a/zhang.pdf)

**Zhang et al. DRL (TMTT 71(9):3773–3783, Sept 2023).**
- PPO agent (modified) picking location and type, from 10 decap types. Z comes from a boundary-integration method — [Zhang 2023 full text](https://par.nsf.gov/servlets/purl/10465812)
- Results:

  | Case | Setup | DRL | GA | Sequential "full search" | Newton–Hessian |
  |---|---|---|---|---|---|
  | 1 | 20 locations, R–L target | 13–14 decaps in minutes | 13–14 | 15 | — |
  | 2 | 60 locations, R–L target | 26–28 decaps after 20–70 min of training | 33 | 33–38 after 30–60 min | — |
  | 3 | flat 0.012 Ω target | 7–8 decaps | — | — | 10 |

  Training ran on an NVIDIA Tesla K80 — [Zhang 2023](https://par.nsf.gov/servlets/purl/10465812)
- Note: this is a per-instance "train-to-solve" DRL, not a generalizing policy. The best episode found during training is reported as the solution.

**Koo et al. (IEEE TEMC 60(3):725–732, June 2018), "Fast algorithm for minimizing the number of decap".**
- Assignment algorithms that meet a semi-arbitrary target with the fewest decaps, taking "a few seconds to an hour". They identify "shadowed" decap ports from the extracted inductance matrix; removing those ports gives more effective layouts — [Missouri S&T ScholarsMine](https://scholarsmine.mst.edu/ele_comeng_facwork/2751)
- A 2017 conference version needs only "a few seconds" even with large capacitor libraries, for R–L-type targets — [ScholarsMine](https://scholarsmine.mst.edu/ele_comeng_facwork/2752)

**Integer SDP (Cai, Mao, Chang, Lu, Bai, Tseng — ICCAD 2025, DOI 10.1109/ICCAD66269.2025.11240747).**
- Reformulates nonlinear decap placement as an equivalent integer SDP, with integer relaxation, port reduction and an adaptive scheme to cut decap usage. Reported to beat non-deterministic state-of-the-art methods and "significantly" outperform a commercial tool in quality and runtime on industrial benchmarks — [NTU repository](https://scholars.lib.ntu.edu.tw/handle/123456789/737210)
- It won a 2025 William J. McCalla ICCAD Best Paper Award (back-end) — [IEEE CEDA announcement](https://ieee-ceda.org/post/announcement/2025-william-j-mccalla-iccad-best-paper-award-winners)
- No numeric results were accessible (paywalled).

**DevFormer (ICML 2023) on the HBM case.**
- With a fixed-K objective, it reached the score that GA needed 40 decaps for with 26 decaps (and AM-CIL needed 34), in zero-shot. That is a ">30% reduction" vs the best baseline — [DevFormer arXiv 2205.13225](https://arxiv.org/pdf/2205.13225)

### Inferences
- Measured greedy-to-GA gap:
  - Sequential add vs GA/DRL: 15 vs 13–14 (Zhang Case 1), 33–38 vs 26–28 (Zhang Case 2), 10 vs 7–8 (Newton–Hessian, Zhang Case 3).
  - Weak GA vs PAGA: 43 → 24 (PAGA Case 1); 37 → 31 (IBIS slides).
  - So greedy-add leaves roughly 10–40% excess decaps. A deterministic add-then-prune or add/remove/swap local search plausibly recovers a large part of this, because PAGA's own strength comes from a greedy initial solution plus a reduction phase.
- "Add then prune" is not novel as a concept. PAGA (greedy init → port removal), Juang (progressive zeros) and Koo (shadowed-port removal) all embody it. The novelty of H3 must be a **learned** add/remove/swap/stop policy that generalizes across boards without per-instance search. Hand-coded alternation is not enough.
- H3's baselines must include a cheap classical "greedy add + greedy backward elimination + 1-swap local search" heuristic. If that heuristic already matches PAGA, the learned policy's contribution is only speed and generalization.

### Gaps
- PAGA Table II (time and decap count for sequential search and Newton–Hessian in Case 1) is an image; those numbers were not extractable. The decap count of PAGA's greedy initial solution alone is also not stated in the text.
- ICCAD 2025 SDP numeric results were not found (paywalled; no arXiv version found).
- No paper was found that explicitly benchmarks "greedy forward + greedy backward elimination (prune)" against GA for decaps. Whether simple pruning closes the GA gap is untested in the literature, which is an opportunity for the proposal.
- No simulated-annealing decap-minimization paper with numbers was found in this pass.

---

## Q2. A lower bound on the minimum decap count

### Takeaway
Only crude per-frequency counting rules exist in the practitioner and patent literature, such as N ≥ ESR/Z_T(f) and capacitance or inductance counts. No paper was found that uses a lower bound to normalize the decap count in an RL reward. Simple capacitance/ESL bounds are not rigorous for spatially distributed, hierarchical PDNs. A rigorous and tighter bound could come from the continuous relaxation of the integer-SDP formulation (ICCAD 2025).

### Cited Findings
- A patent claim gives the per-frequency capacitor count N = ESR/Z_T(f), rounded up — [US 6,385,565 / related patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/6385565)
- A lecture note gives the count as n ≈ 2L/(Z_t·t_r), from per-capacitor inductance and rise time — [UiO IN5230 decoupling notes](https://www.uio.no/studier/emner/matnat/ifi/IN5230/h19/nedlastninger/decoupling_2018.pdf)
- The board inductance limit is a physical floor: beyond it, adding decaps stops helping — [US 8,166,447 PDN calculator patent](https://image-ppubs.uspto.gov/dirsearch-public/print/downloadPdf/8166447)
- Park et al. on-chip unit decap (verified): a 375×375 µm² NMOS decap with 1.055 nF capacitance and 0.7 mΩ ESR. No ESL is given; the decap is modeled as part of the cell — [Park et al. arXiv 2203.15722](https://arxiv.org/pdf/2203.15722)
- The ICCAD 2025 integer SDP uses "integer relaxation" — [NTU repository](https://scholars.lib.ntu.edu.tw/handle/123456789/737210). An integer program's relaxation yields a valid lower bound on the integer optimum, which is standard optimization theory.

### Inferences
- Example capacitance bound: if the decaps alone must hold |Z| ≤ Z_t at f_min, then N ≥ 1/(2π f_min Z_t C_unit). With Z_t = 0.5 Ω, f_min = 10 MHz and C = 1.055 nF, N ≥ 1/(6.28e7·0.5·1.055e-9) ≈ 30.
- That bound is invalid in practice because plane, package and VRM paths also supply low-frequency impedance. It can also be very loose at the high end, where spreading and via inductance dominate. Use it only as a heuristic scale, not as a "certificate".
- An ESL/ESR bound (N ≥ max_f ESR/Z_t, or N ≥ ω·ESL_mount/Z_t) ignores plane spreading inductance. On-chip NMOS decaps have negligible mounting ESL, so this bound is near-useless for Park's grid.
- Recommended reward normalization:
  - Use count/N_ref, where N_ref is a strong cheap baseline: greedy-add + prune on the same instance. This is computable per instance and well correlated with difficulty. It gives a relative "% vs greedy" metric. POMO and Sym-NCO similarly use instance-wise shared baselines, not absolute bounds.
  - Optionally report a relaxation lower bound (continuous-relaxed SDP or LP) to quote optimality gaps.

### Gaps
- No source was found that reports how tight analytic count bounds are against true optima, or that uses such a bound for RL normalization.
- A rigorous Smith/Novak attribution for the counting rules was not found.

---

## Q3. Z-matrix cost: are 29 s / 1.6 GB (50×50) and 26 GB (100×100) plausible, and does a cheap classical alternative exist?

### Takeaway
The proposal's numbers match exactly a **dense, all-port Z stored at about 32 frequencies in complex64**. This is a storage artifact of materializing every candidate port, not an intrinsic cost of decap evaluation. A sparse or block solve of the nodal admittance Y(f) for only the probe and placed-decap columns makes evaluation cheap without ever forming full Z. Reaching 100×100 or 200×200 needs at most a few hundred MB and well under 1 s per frequency, even in naive Python. "Works where full Z is infeasible" is therefore not a unique advantage of a learned policy: greedy, prune and local search can use the same sparse-solve trick.

### Cited Findings
- Practice in the field: every reviewed method first exports a full port-Z from an EM tool and then cascades decaps with the segmentation/Schur formula. This holds for PAGA (Z'_aa = Z_aa − Z_ap(Z_pp+Z_dd)^-1 Z_pa), Zhang 2023, and Park et al. (whose full-port model Z_fp has 816 ports: 48 probe candidates and 768 decap candidates) — [PAGA](https://par.nsf.gov/servlets/purl/10566602), [Park](https://arxiv.org/pdf/2203.15722)
- RL4CO's DPP environment does the same: it loads `raw_pdn` of shape [num_freq, size², size²] and computes `zout = z1aa - z1ap @ inverse(z1pp + z2qq) @ z1pa` — [RL4CO DPP env source](https://raw.githubusercontent.com/ai4co/rl4co/main/rl4co/envs/eda/dpp/env.py)
- DevFormer: a 10×10 grid with 201 frequencies (200 MHz–20 GHz) means "100×100×201 ≈ 2M impedances" per task, which the authors call "simulation intensive" — [DevFormer](https://arxiv.org/pdf/2205.13225)
- Park et al. training data cost was 30.8 GB for the 100/20 setting and up to 270.6 GB for the 300/60 setting. Training took 2,200 s for 100/20 and up to 104,800 s for 300/150. Storage of Z matrices is the dominant data cost — [Park arXiv 2203.15722, Table VII](https://arxiv.org/pdf/2203.15722)
- PAGA authors acknowledge that S-parameter or Z export becomes "very challenging" when candidate locations become extremely large — [IBIS slides](https://ibis.org/summits/nov25a/zhang.pdf)
- A Woodbury-identity reuse for evaluating connected multi-port networks exists in recent literature (general networks, not decaps) — [arXiv 2412.17884](https://arxiv.org/abs/2412.17884v1)
- CNN surrogates predict PDN impedance in about 0.1 s vs 10–30 s for BEM — [Zhang et al., arXiv 2106.10693](https://arxiv.org/pdf/2106.10693)

**Own benchmark.** Run in this session on an Apple M4 Pro with numpy only, on a synthetic n×n RLGC grid with nodal Y = block-tridiagonal. Script: scratchpad `zbench2.py`.

| Board | Nodes | Full Z per freq (c64) | Dense inv of Y per freq | Probe/decap columns via block-tridiagonal solve (naive Python) |
|---|---|---|---|---|
| 50×50 | 2,500 | 50 MB | 1.06 s | 16 ms (K=4), 24 ms (K=64) |
| 100×100 | 10,000 | 800 MB | (not run) | about 0.2 s, for K=4 or K=64 |
| 200×200 | 40,000 | 12.8 GB | (not run) | about 1 s, for K=4 or K=64 |

- At 50×50, the block-solve columns match the dense-inverse columns to relative error 6.8e-13.
- 32 frequencies × full Z (c64): 1.6 GB at 50×50 and 25.6 GB at 100×100. This reproduces the proposal's 1.6 GB and 26 GB exactly. 32 × 1.06 s ≈ 34 s, consistent with the measured 29 s.
- Memory for K columns at 100×100 is 0.3–5 MB per frequency, vs 800 MB for full Z.

### Inferences
- **Cost is dense-Z materialization.** Memory scales as N² per frequency. Dense inversion scales as N³: 100×100 would be about 64× the 50×50 time, roughly 30+ min. The proposal's cost model is therefore correct for "build full Z", but that step is optional.
- **Classical alternatives that avoid full Z.** Adding a decap at node i adds y_d(f) to Y_ii, which is a rank-1 diagonal update of a sparse matrix.
  - (a) Direct check: factor Y'(f) = Y + Σ y_d e_i e_iᵀ once per frequency with sparse LU and solve for the K probe columns. Nested dissection gives O(N^1.5) for 2-D grids.
  - (b) Incremental greedy/prune via Sherman–Morrison/Woodbury: keep the factor of the base Y and the probe columns. Each add or remove needs one extra solve (column i) plus an O(N) update.
  - (c) Candidate scoring for all locations at once: by reciprocity, Z(probe, ·) for all candidates comes from K solves, so a greedy step needs no full Z.
  - A greedy add + prune baseline on a 100×100 or 200×200 grid is then seconds to minutes, not GBs.
- **Consequence for H4.** "Z not needed" is a real advantage over PAGA, SDP and RL4CO-style pipelines as implemented (they consume exported full-port Z). It is not an advantage over a competent classical solver on an RLGC grid, which the researcher controls end-to-end.
  - The honest H4 claim is: the policy needs only the geometry/RLGC context plus K-column feedback, and amortizes search across instances.
  - The speed comparison must then be "policy + verification solves" vs "greedy/prune/PAGA + the same sparse solver". Comparing against "PAGA + dense Z" would be a strawman.
- **When full Z really is infeasible:** EM-extracted boards (SIwave/OPI/BIM) where Y is not available and each port costs an EM simulation. There, a policy that proposes few candidate ports before extraction (port pre-selection) would be the genuine win. This matches PAGA's stated future work.
- **Remaining 100× vs PAGA:** PAGA's reported times exclude Z extraction and run on 68–123 ports in 18–100 s. A 100× speedup means about 0.2–1 s per instance. That is achievable for neural inference plus a few sparse solves. It should be stated at matched port counts and with all I/O included.

### Gaps
- No primary reference was found that explicitly uses Sherman–Morrison or sparse-LU incremental updates for decap placement. The DevFormer appendix (simulator details) was truncated in the fetched PDF, so it could not be checked for this. This is a plausible gap for the proposal to fill or cite as its own baseline.
- The benchmark uses a synthetic homogeneous grid. Park's hierarchical HBM model (chip + interposer + TSV + package lumped) has more coupling but remains sparse. Real nested-dissection sparse LU (scipy/PARDISO) timings were not measured because scipy was not installed.

---

## Q4. Training-design check: constrained RL, self-imitation, POMO/Sym-NCO, expert iteration, progressive loss, reward shaping, and evidence that improvement policies work

### Takeaway
Every ingredient in the proposal has direct precedent:
- Zhang 2023 already uses a best-experience memory, which is a form of self-imitation.
- DevFormer/RL4CO use symmetric/shared-baseline training.
- Learned improvement policies (2-opt and rewrite operators) are established in neural combinatorial optimization.

The main risks are sparse terminal reward, credit assignment for "remove" actions, and the feasibility constraint. A Lagrangian/feasibility-first reward plus greedy baselines plus self-imitation addresses them.

### Cited Findings
- **Zhang 2023 decap DRL reward:**
  - r_t = (N_{t+1} − N_t)/N + T, where N_t counts frequency points meeting Z_target.
  - T = 1 + N − N_decap when the target is met, and T = −p (the maximum % violation) when locations are full and the target is still unmet.
  - "Memory B" stores the best experiences with the fewest decaps and is replayed together with on-policy memory A. This is reported to converge faster than vanilla PPO and GA, averaged over 18 runs.

  Source: [Zhang TMTT 2023](https://par.nsf.gov/servlets/purl/10465812)
- **Park et al.:** REINFORCE with a rollout greedy baseline (paired t-test update, as in Kool et al.). Reward is the mean reduction of self/transfer impedance at 4 probes with a fixed decap budget m, not count minimization. Even sampling width 1000 for GA/RS does not reach the policy's single-shot performance, a claimed >1000× computing advantage — [Park arXiv 2203.15722](https://arxiv.org/pdf/2203.15722)
- **DevFormer:** offline expert data from GA (100 sims per instance), plus "collaborative symmetricity exploitation" using action-permutation symmetry (self-exploitation loss). Uses K = 20 fixed; scores are impedance suppression — [DevFormer](https://arxiv.org/pdf/2205.13225)
- **RL4CO DPP reward:** computed once at episode end (zero per-step) because "decap simulation is time-consuming". The multi-probe mDPP offers `minmax` or `meansum` reward — [RL4CO EDA env docs](https://rl4co.readthedocs.io/en/v0.3.1/_content/api/envs/eda.html), [env source](https://raw.githubusercontent.com/ai4co/rl4co/main/rl4co/envs/eda/dpp/env.py)
- **Symmetric Replay Training** (Kim et al. 2023) is a sample-efficiency method for DRL in combinatorial optimization from the same group. It appears in decap-RL search results, and its DPP use is likely but not verified — [arXiv 2306.01276](https://arxiv.org/pdf/2306.01276)
- **Learned improvement heuristics:**
  - Wu, Song, Cao, Zhang, Lim learn a self-attention policy that picks local moves on TSP/CVRP and beat hand-crafted improvement rules — [arXiv 1912.05784](https://arxiv.org/pdf/1912.05784)
  - Learning 2-opt local search via DRL — [SN Computer Science 2021](https://link.springer.com/article/10.1007/s42979-021-00779-2)
- **Progressive loss / iteration-count robustness:** Bansal, Schwarzschild et al. (NeurIPS 2022) use a recall architecture and progressive training that discourages iteration-count-specific behavior. They extrapolate to much larger and harder instances without "overthinking" — [arXiv 2202.05826](https://arxiv.org/abs/2202.05826v3)
- **[not re-fetched]** RCPO, reward-constrained policy optimization (Tessler, Mankowitz, Mannor, ICLR 2019) — [arXiv 1805.11074](https://arxiv.org/abs/1805.11074)
- **[not re-fetched]** PPO-Lagrangian baseline (Ray, Achiam, Amodei 2019, Safety Gym) — [OpenAI](https://cdn.openai.com/safexp-short.pdf)
- **[not re-fetched]** Self-Imitation Learning (Oh, Guo, Singh, Lee, ICML 2018) — [arXiv 1806.05635](https://arxiv.org/abs/1806.05635)
- **[not re-fetched]** POMO (Kwon et al., NeurIPS 2020) — [arXiv 2010.16011](https://arxiv.org/abs/2010.16011)
- **[not re-fetched]** Sym-NCO (Kim, Park, Park, NeurIPS 2022) — [arXiv 2205.13209](https://arxiv.org/abs/2205.13209)
- **[not re-fetched]** Expert Iteration (Anthony, Tian, Barber, NeurIPS 2017) — [arXiv 1705.08439](https://arxiv.org/abs/1705.08439)
- **[not re-fetched]** NeuRewriter, learning local rewriting (Chen & Tian, NeurIPS 2019) — [arXiv 1810.00337](https://arxiv.org/abs/1810.00337)
- **[not re-fetched]** DACT improvement policy (Ma et al., NeurIPS 2021) — [arXiv 2110.02544](https://arxiv.org/abs/2110.02544)

### Inferences
- **Reward design.** Use the terminal objective R = −N_decap/N_ref − λ·max_f[violation(f)]₊.
  - Either learn λ by dual ascent (RCPO/PPO-Lagrangian), or use a lexicographic "feasible beats infeasible" rule as in PAGA's cost function.
  - Add potential-based shaping from the violation area (Zhang's ΔN_t-style term) to densify the signal without changing the optimum.
  - N_ref is the per-instance greedy+prune count (see Q2).
- **Credit assignment for remove/swap.** In an improvement MDP, the natural per-step reward is the change in objective (Δcount, Δviolation), as in Wu et al. This makes remove actions immediately rewarded when they keep feasibility and penalized when they break it.
  - Mask removals that would make the solution infeasible (one Sherman–Morrison check per candidate), or let the policy see per-decap "removability" features.
- **Baselines and variance.** A POMO-style shared baseline over multiple rollouts per instance (multiple start decaps or probe-symmetric augmentations) fits well. Action-permutation symmetry (DevFormer) applies to add-only sequences. For add/remove trajectories the symmetry is weaker, because the order of removals matters.
- **Self-imitation / expert iteration.** Zhang's "memory B" is already SIL-like. A clean variant is to periodically replace the imitation target by the best of {policy rollouts, greedy+prune, policy+local-search} per instance (ExIt). This guards against the policy being worse than the classical baseline.
- **Random iteration budget.** Training with a random number of improvement steps plus a stop action, and a progressive-loss-style consistency term, is the right way to make the outer loop anytime and stable beyond the trained horizon (Bansal 2022).
- **Known pitfalls:**
  - Sparse terminal reward. RL4CO gives zero step reward, which makes long add/remove episodes hard.
  - The policy learning to "stop early" when the infeasibility penalty is too weak; a too-strong penalty makes it over-place decaps.
  - Oscillation (add i / remove i cycles). Use a tabu mask or penalize repeated states.
  - Distribution shift between greedy-initialized starts at train vs test.

### Gaps
- No decap-specific paper uses Lagrangian constrained RL, an add/remove improvement MDP, or progressive loss. No direct empirical evidence exists in this domain, which is a novelty opportunity but also a risk.
- The "[not re-fetched]" references were not re-opened this session. Their identities and URLs are standard but should be spot-checked before citation.

---

## Q5. Practical availability: code, data, validation

### Takeaway
RL4CO ships a ready DPP/mDPP environment with DevFormer's 10×10 data: 1 probe (2–5 for mDPP), 1–50 keep-outs, up to 20 decaps, 201 frequencies, a single 0.1 nF decap type. It is a fixed-K impedance-suppression task, not count minimization, has no multi-type decaps, and stores dense size²×size² Z per frequency. It is useful as a sanity benchmark but not as a direct H3/H4 testbed. No public PAGA code was found.

### Cited Findings
- **RL4CO DPP:**
  - 10×10 grid of 100 locations; one probing port; keep-outs from `num_keepout_min=1` to `num_keepout_max=50`; `max_decaps=20`.
  - Data files `10x10_pkg_chip.npy`, `01nF_decap.npy` and `freq_201.npy`, from a Google Drive link in the docs.
  - mDPP has 2–5 probes and minmax/meansum rewards.

  Source: [RL4CO EDA docs](https://rl4co.readthedocs.io/en/v0.3.1/_content/api/envs/eda.html)
- RL4CO DPP evaluation uses a dense Schur-complement with `torch.inverse(z1pp + z2qq)`. The episode ends after `max_decaps` steps — [RL4CO env source](https://raw.githubusercontent.com/ai4co/rl4co/main/rl4co/envs/eda/dpp/env.py)
- **DevFormer:**
  - Official code at github.com/ai4co/devformer (about 19 stars) — [ecosyste.ms listing](https://science.ecosyste.ms/projects/28314)
  - Chip-package hierarchical PDN, 10×10 grid, 201 frequencies over 200 MHz–20 GHz, test set of 100 PDN cases, also 15×15. Encoder L = 3, hidden size 128 — [DevFormer](https://arxiv.org/pdf/2205.13225)
  - Fixed-K scores at 10×10:

    | K | GA | AM-CIL | DevFormer |
    |---|---|---|---|
    | 12 | 11.77 | 10.22 | 12.23 |
    | 20 | 12.53 | 11.71 | 12.81 |
    | 30 | 13.02 | 12.62 | 13.11 |

    At 15×15 and K = 20: GA 7.61, DevFormer 8.47.

    Source: [DevFormer Table 3](https://arxiv.org/pdf/2205.13225)
- **Park et al. HBM model (the proposal's reference):**
  - 375×375 µm² unit decap-sized cells; NMOS decap 1.055 nF and 0.7 mΩ ESR (verified).
  - 816-port full Z: 48 probe candidates and 768 decap candidates.
  - The on-chip cell is 75×75 sub-unit-cells and the interposer cell 25×25; W-element RLGC modeling.
  - Data split 1000/100/100.

  Source: [Park arXiv 2203.15722](https://arxiv.org/pdf/2203.15722)
- PAGA's decap library (Table I) is from a commercial library. Zhang 2023's 10 decap types are defined in their ref [18] — [PAGA](https://par.nsf.gov/servlets/purl/10566602), [Zhang 2023](https://par.nsf.gov/servlets/purl/10465812)

### Inferences
- For multi-type decaps with different C/ESL on Park's grid, the proposal must build its own type library. Park's on-chip NMOS decap has only C and ESR; board MLCC libraries (PAGA/Zhang) give C/ESR/ESL.
- **ngspice validation.** Export the RLGC grid plus chosen decaps as a netlist and run an AC sweep with a 1 A AC source at the probe, so |V| = |Z|. This independently checks the sparse/Schur evaluator on a few instances. This is standard practice; no specific citation was fetched.
- RL4CO DPP can serve as a reproducibility check by re-implementing the evaluator with sparse solves and confirming identical Z on its 10×10 data.

### Gaps
- No public PAGA source code URL was found. Searches surfaced only the paper and slides, and a US patent on GA-based decap selection (US 11,501,044).
- The DevFormer appendix (decap model, simulator time) was not extractable. The "0.1 nF" decap type is inferred from the filename `01nF_decap.npy` only.

---

## Q6. Verdict on H3 and H4: feasibility and recommended modifications

### Takeaway
- **H3** is plausible (moderate–high feasibility) but under-differentiated: greedy+prune is the obvious strong baseline, and PAGA already combines greedy initialization with progressive removal.
- **H4** as phrased is partially a strawman. The 1.6 GB / 26 GB cost is dense full-Z storage, which classical methods can also avoid with sparse solves. The defensible version is "amortized inference with K-column feedback, at matched solver."

### Cited Findings
- Greedy/sequential methods leave 10–40% excess decaps vs GA/DRL/PAGA: 15 vs 13–14, 33–38 vs 26–28, 10 vs 7–8, and 43 → 24 — [Zhang 2023](https://par.nsf.gov/servlets/purl/10465812), [PAGA](https://par.nsf.gov/servlets/purl/10566602)
- PAGA solves 68–123-port real boards in about 18–100 s, excluding Z export, and acknowledges that Z/S-parameter export for very many candidates is its bottleneck — [PAGA](https://par.nsf.gov/servlets/purl/10566602), [IBIS slides](https://ibis.org/summits/nov25a/zhang.pdf)
- The proposal's measured 1.6 GB and 26 GB equal 32 frequencies of complex64 full Z (own computation above). K-column solves need well under 1 s and a few MB per frequency at 100×100 (own benchmark above).

### Inferences
**H3 feasibility: moderate–high.**
- Required modifications:
  1. Add classical baselines: greedy-add, greedy-add + backward elimination, greedy + 1-swap local search, PAGA re-implementation, and optionally the SDP relaxation as a lower bound.
  2. Claim "reaches PAGA level at X× less time and generalizes zero-shot", not merely "beats greedy".
  3. Report count relative to greedy+prune and to the best of many PAGA runs.
- Expected outcome: a learned add/remove/swap/stop policy should match greedy+prune easily. Matching PAGA within about 1–2 decaps on 100–120-port problems is plausible given the Wu et al. evidence, but not guaranteed.

**H4 feasibility: high for "faster" and low for "uniquely avoids Z".**
- Recommended rewording: "the policy never needs the all-port Z; it consumes the RLGC grid and per-step K-probe sparse-solve feedback. Time-to-solution includes all solves, and is compared with greedy/PAGA using the same sparse evaluator."
- The 100× target vs PAGA is achievable only if PAGA is run with the same fast evaluator, and the policy uses ≤ about 100 evaluations per instance.
- The real "infeasible full Z" niche is EM-extracted boards, where the policy could pre-select candidate ports before extraction.

**Training.**
- Feasibility-first/Lagrangian terminal reward, normalized by N_greedy+prune.
- Δ-objective step rewards for improvement moves, with a tabu mask against add/remove cycles.
- POMO-style multi-rollout baseline, plus SIL/ExIt replay of the best-known solution (including classical local-search outputs).
- Random step budgets with a stop action and a progressive-loss consistency term.

### Gaps
- No public numbers were found for PAGA on more than about 200 ports or on regular RLGC grids like 50×50. The 100× claim cannot be pre-checked against literature at the proposal's board sizes; PAGA's Fig. 12 scalability numbers were not extractable.
- The ICCAD 2025 SDP runtimes and decap counts are unknown. If it is both fast and near-optimal, it raises the H3/H4 bar considerably.
