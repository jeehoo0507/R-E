# Prior art and fact-check: "dual-loop" learned decap placement proposal

Research date: 2026-10-15. Verdict legend: CONFIRMED / MOSTLY CORRECT (needs a fix) / WRONG / UNVERIFIABLE.
Full-text access is stated for every source. "Abstract only" means the claim rests on an abstract or search-engine summary, not on the paper itself.

## Q1. Park et al. 2022 (arXiv 2203.15722): coordinate-only input, single HBM PDN, +0.2~0.7% vs GA

### Takeaway
MOSTLY CORRECT. The input really is coordinates plus a probe flag, and every experiment runs on one HBM VDDQ PDN. The gain over GA in Table IV is +0.09% to +0.64%, not +0.2~0.7%. The objective also differs from the proposal: the paper places a fixed number of decaps m to maximize impedance reduction. It does not minimize the decap count.

### Cited Findings (full text read: arXiv v2 PDF)
- Title: "Transformer Network-based Reinforcement Learning Method for Power Distribution Network (PDN) Optimization of High Bandwidth Memory (HBM)", H. Park, M. Kim, ... C. Hwang, J. Kim. arXiv v1 on 29 Mar 2022, v2 on 23 Aug 2022. Published in IEEE TMTT, DOI 10.1109/TMTT.2022.3202221 — [arXiv](https://arxiv.org/abs/2203.15722)
- Input: each node x_i = (x-coord, y-coord, z-coord, isProbingPort). The state is s = {P, X}: 4 probing ports plus n decap candidates. No impedance or Z data goes into the network. **CONFIRMED** — [arXiv PDF, Eq. 3](https://arxiv.org/pdf/2203.15722v2)
- PDN: a single "HBM VDDQ PDN" based on HBM gen2, modeled as on-chip plus interposer unit cells of 375×375 µm². Z_fp has 816 ports in total. Each instance draws 4 random probing ports and n random candidates from this one PDN. **CONFIRMED (single PDN, random port subsets)** — [arXiv PDF §III](https://arxiv.org/pdf/2203.15722v2)
- Reward: the reduction of 10 self- and transfer impedances (Z11..Z44, Z12..Z34) relative to the initial PDN, with m fixed. Problems are n/m = 100/20 ... 300/150 — [arXiv PDF §II](https://arxiv.org/pdf/2203.15722v2)
- Table IV, average reward over 100 test sets. Proposed (sampling width 1000) vs GA (pop 100 × gen 10 = 1000): 100/20 16.797 vs 16.751; 100/50 19.934 vs 19.917; 200/40 19.448 vs 19.354; 300/20 16.928 vs 16.836; 300/60 20.604 vs 20.473; 300/150 22.057 vs 21.979. Across the 12 problems the gain over GA works out to **+0.085% (100/50) to +0.64% (300/60)**. With greedy sampling width 1 the proposed model still beats GA@1000, by +0.06% to +0.57% (my arithmetic from Table IV) — [arXiv PDF Table IV](https://arxiv.org/pdf/2203.15722v2)
- The paper's main claimed advantage is time, not optimality. On decap 100/20 the proposed method takes 11 s, against "3h>" for GA/RS to reach the same performance. On 300/20 it takes 2m16s against "1d 14h >" (Table V) — [arXiv PDF Table V](https://arxiv.org/pdf/2203.15722v2)
- Scalability: a model trained on 100/20 was applied to 200/20 (reward 16.849) and to 300/20 (16.862 vs 16.874 when trained in-distribution) — [arXiv PDF §IV-C](https://arxiv.org/pdf/2203.15722v2)

### Inferences
- The proposal should change "+0.2~0.7%" to "+0.1~0.6% (Table IV, equal sampling budget)". It should also say the gap is small because the objective is a fixed-m impedance reduction.
- "Coordinates only" is a fair criticism. The network never sees Z, so generalizing to a different PDN would require retraining. That supports the proposal's motivation.

### Gaps
- None for this claim.

## Q2. DevFormer (Kim et al., ICML 2023): 10x10-centric; "25x25 score drops −10.5 → −7.0" per PhysEDA

### Takeaway
CONFIRMED, with a nuance. DevFormer's own experiments use a 10×10 grid, plus a 15×15 zero-shot test and an HBM case. It contains no 25×25 experiment. The numbers −10.50 → −7.04 come from PhysEDA's re-implementation, trained with RL rather than DevFormer's offline imitation learning: −10.50 is the in-distribution 10×10 score and −7.04 is the zero-shot 10×10→25×25 score.

### Cited Findings
- Citation: H. Kim, M. Kim, F. Berto, J. Kim, J. Park, "DevFormer: A Symmetric Transformer for Context-Aware Device Placement," ICML 2023, PMLR 202:16541–16566. Code at github.com/ai4co/devformer — [PMLR](https://proceedings.mlr.press/v202/kim23h.html), [arXiv](https://arxiv.org/abs/2205.13225)
- Setup (full HTML read via fetch tool): 10×10 PDN grid, 201 frequency points from 200 MHz to 20 GHz, K = 20 decaps. Expert labels come from a GA with M = 100 simulations per problem, using N = 2000 expert samples (ablation over 100/500/1000/2000). The symmetry method is "Collaborative Symmetricity Exploitation": P = 4 permutation augmentations plus a permutation-consistency loss with λ = 8 — [arXiv HTML v3](https://arxiv.org/html/2205.13225v3)
- Generalization in DevFormer itself: pretrained on 10×10 with K = 20, then zero-shot to K ∈ {12, 16, 20, 24, 30} on 10×10 and K ∈ {20, 40} on a 15×15 PDN. The HBM case is a 40×40 package, 40×60 interposer and 15×20 on-chip PDN, tested on 100 unseen cases. To reach a score of 26.68, DevFormer needed 26 decaps, AM-CIL 34 and GA 40 — [arXiv HTML v3](https://arxiv.org/html/2205.13225v3)
- PhysEDA's numbers for DevFormer (score is negative; more negative is better):
  - vanilla RL on 10×10: **−10.50**
  - vanilla RL trained on 25×25: −10.53
  - zero-shot, trained on 10×10 and tested on 25×25: **−7.04**
  - RL+PBRS: 10×10 −13.01, 25×25 −13.34
  - IL on 25×25: −15.88
  
  PhysEDA calls this a ~33% degradation for DevFormer under transfer — [PhysEDA arXiv HTML](https://arxiv.org/html/2605.10547)

### Inferences
- In the proposal, phrase it as: "In PhysEDA's re-implementation, DevFormer trained on 10×10 drops from −10.50 to −7.04 on 25×25 zero-shot (~33%)." Scores on different board sizes are not strictly comparable (25×25 places 101 decaps; PhysEDA's 10×10 places 25), so the 33% figure is PhysEDA's own framing.

### Gaps
- I did not verify whether the original DevFormer authors ever reported a 25×25 result elsewhere, for example in RL4CO. None appears in the ICML paper.

## Q3. PhysEDA (arXiv 2605.10547): Manhattan-decay attention, up to 25×25, learned baselines only, one synthetic dataset

### Takeaway
CONFIRMED. PhysEDA uses a separable exp(−α·d_Manhattan) bias inside linear attention. Its task scores cover only 10×10 and 25×25 decap placement; larger grids are used for latency and memory only. The baselines are all learned models (DevFormer plus 4 linear-attention variants), and the decap data is synthetic, generated from DevFormer's formula.

### Cited Findings (full HTML read in two chunks)
- Single author, Zetao Yang. Submitted 11 May 2026, cs.LG. Code and data are "released upon publication" — [arXiv abs](https://arxiv.org/abs/2605.10547)
- Mechanism: "Physics-Structured Linear Attention" (PSLA) with kernel exp(−α·d_M), d_M = |x_i−x_j| + |y_i−y_j|. α is learnable per head, in the range [1.2, 1.8], initialized at 1.5. The rank-1 form gives directional decay; a symmetric version is in Appendix B. The paper also uses PBRS reward shaping for sparse terminal reward — [arXiv HTML](https://arxiv.org/html/2605.10547)
- Grids: task scores only at 10×10 (L = 100, 25 capacitors) and 25×25 (L = 625, 101 capacitors). The 50×50 to 150×150 runs measure latency and memory only. The abstract cites a 14× speedup and 98.5% memory saving at 100×100 — [arXiv HTML](https://arxiv.org/html/2605.10547), [abs](https://arxiv.org/abs/2605.10547)
- Data: the ×10 set is DevFormer's public 2,000 instances (2,000 train / 100 test). The ×25 set has 2,300 instances "generated by DevFormer's physical formula" (2,000/100/200). Both are synthetic; no measured or real board data. Test scores average 4 test-time augmentations — [arXiv HTML appendix](https://arxiv.org/html/2605.10547)
- Baselines (Appendix I.1): DevFormer, Plain GLA, FAVOR+, Katharopoulos linear attention, CosFormer. **No GA, random search or greedy.** — [arXiv HTML appendix](https://arxiv.org/html/2605.10547)
- PSLA headline numbers: zero-shot 10→25 is −11.04 vs DevFormer −7.04 (+56.8%). Vanilla RL on 25×25 is −15.70 vs −10.53. RL+PBRS on 25×25 is −17.40 vs −13.34 — [arXiv HTML](https://arxiv.org/html/2605.10547)
- The paper also has other tasks (macro placement; IR-drop on CircuitNet). "One dataset" is accurate for the decap task only — [arXiv abs](https://arxiv.org/abs/2605.10547)

### Inferences
- PhysEDA's distance-decay prior is a hand-set physical bias. The proposal's admittance-weighted message passing is a different, circuit-derived bias. PhysEDA is the natural learned baseline to beat on 25×25, together with a GA, which PhysEDA itself omits.

### Gaps
- PhysEDA is a single-author preprint and is not peer reviewed (as of Oct 2026, no venue found).

## Q4. Manoharan, Wang, Hwang, IEEE TSIPI 2026 (doi 10.1109/TSIPI.2026.3691276)

### Takeaway
The paper exists, but its full text and abstract were not accessible, so the specific claims are UNVERIFIABLE from the paper itself. The authors' own IEEE EMC+SIPI 2025 IBIS Summit slides on the same post-layout transformer-RL work support most of the claims: Z-matrix input, loop-inductance port ordering, and testing up to 150 ports. They report 0–5 fewer decaps than GA, not 0–3.

### Cited Findings
- Crossref metadata: title "Reinforcement Learning for Decoupling Capacitor Optimization in Postlayout PDNs Using Transformer Architecture", authors Manoharan, Wang, Hwang, *IEEE Transactions on Signal and Power Integrity*, 2026. The DOI resolves to IEEE Xplore document 11511858, whose page content was not retrievable — [Crossref API](https://api.crossref.org/works/10.1109/TSIPI.2026.3691276), [IEEE Xplore](https://ieeexplore.ieee.org/document/11511858/)
- Slides (full text read): H. Manoharan, C. Hwang, "Reinforcement Learning-Driven PCB PDN Design Automation," IBIS Summit at IEEE EMC+SIPI 2025, 22 Aug 2025 — [IBIS PDF](https://www.ibis.org/summits/aug25/manoharan.pdf)
  - Input state: "Z matrix for M ports, Target impedance, Decap library". Decaps are featured by ESL, ESR, C and package size. Training is REINFORCE. → **Z-based input CONFIRMED (slides)**
  - Port priority: "Ports are prioritized based on their loop inductance with respect to the IC port… At each step, the port with the minimum loop inductance is shorted, and the impedance matrix is updated." → **location order by rule CONFIRMED (slides)**
  - Training data: 10,000 board cases with 25–75 decap ports, targets from 1 mΩ to 1 Ω, about 1,000 Murata decaps, ~1 day of training.
  - Generalization: decaps from a different vendor on a 50-port board. Decaps (proposed vs GA v3): 22 vs 23, 27 vs 28, 29 vs 31, 34 vs 34, 39 vs 40, at 300–610 ms vs 459–544 s.
  - Scalability: a 150-port board: 87 vs 87, 99 vs 100, 112 vs 114, 120 vs 125, at 0.76–1.4 s vs 59–90 min. → **"≤150 ports" CONFIRMED; "0~3 decaps vs GA" is slightly WRONG: the slides show 0–2 fewer at 50 ports and 0–5 fewer at 150 ports.**
  - "Zero-shot to unseen PCB": the slides test unseen vendors and board sizes, presumably untrained boards, but never use the term "zero-shot". → PARTIAL.
- Manoharan's dissertation, "Machine Learning Based Automation of PCB PDN Design and Optimization" (Missouri S&T), covers the same framework, including "a sequential Transformer-decoder-based RL model for post-layout decap assignment" (summary only) — [Scholars' Mine](https://scholarsmine.mst.edu/doctoral_dissertations/3459/)

### Inferences
- In the proposal, cite the TSIPI paper by title and back the numbers with the IBIS slides until the paper's full text is read. The best approach so far reads the Z matrix after an inductance-ranked port ordering. That is exactly the "needs Z / rule-based order" limitation the proposal targets.

### Gaps
- The exact TSIPI numbers (port counts, GA comparison, zero-shot wording) need IEEE Xplore access.

## Q5. PAGA (IEEE TMTT 2024): greedy 61 vs PAGA 24 on a 120-port case; public code; a greedy "no solution" case

### Takeaway
"61 vs 24 on 120 ports" is CONFIRMED from the authors' table image in the official repo. The "61" baseline is a "Sequential search method", ref [6] = de Paulis et al., EMC+SI 2020. The "no solution" entry is WRONG as attributed: it belongs to the "Sorting search method" of Koo et al., IEEE TEMC 2018, on the same 120-port case, not Juang et al. 2021 on ~20 ports. Juang et al.'s modified GA (EMC 2021) needed 43 decaps on that case.

### Cited Findings
- Citation: L. Jiang, L. Zhang, S. Tan, D. Li, C. Hwang, J. Fan, E.-P. Li, "A Novel Physics-Assisted Genetic Algorithm for Decoupling Capacitor Optimization," IEEE TMTT vol. 72, no. 8, pp. 4498–4507, Aug 2024, DOI 10.1109/TMTT.2024.3356575 — [NSF PAR](https://par.nsf.gov/biblio/10566602-novel-physics-assisted-genetic-algorithm-decoupling-capacitor-optimization)
- Code: **https://github.com/SoFoolish250/Physics-Assisted-Genetic-Algorithm-PAGA-for-Decap-Optimization**. It has folders PAGA, ModifiedGA, NewtonMethod, OriginalGA and SequentialSearch, and takes a Z-parameter .snp/.npy input. Maintainers are Li Jiang and Ling Zhang (Zhejiang Univ.); the GA framework comes from rmsolgi/geneticalgorithm. The GitHub UI lists a GPL-3.0 LICENSE, while the README says "copyright 2023 Li Jiang". Case 1 and case 3 are 120-port cases per the README captions — [GitHub](https://github.com/SoFoolish250/Physics-Assisted-Genetic-Algorithm-PAGA-for-Decap-Optimization)
- Table II, Case 1 (image `table-comparison-case1.png` in the repo, read directly):

  | Method | Time | Decaps |
  |---|---|---|
  | GA [7] | 1266 s | 79 |
  | GA [8] | 887 s | 43 |
  | GA [9] | 187 s | 24 |
  | Sequential search [6] | — | **61** |
  | Sorting search [16] | — | **No Sol.** |
  | Newton–Hessian [15] | — | 101 |
  | **PAGA** | **61 s** | **24** |

  — [GitHub repo image](https://raw.githubusercontent.com/SoFoolish250/Physics-Assisted-Genetic-Algorithm-PAGA-for-Decap-Optimization/main/table-comparison-case1.png)
- Table III, Case 2 (port count not stated): GA[7] 595 s/78; GA[8] 441 s/32; GA[9] 140 s/10; SIwave tool 57 s/34; PAGA 18 s/10 — [GitHub repo image](https://raw.githubusercontent.com/SoFoolish250/Physics-Assisted-Genetic-Algorithm-PAGA-for-Decap-Optimization/main/table-comparison-case2.png)
- Robustness figure: from 50 to 200 ports, PAGA's time stays at ~20–30 s and its decap count stays at ~30, matching GA[9]. GA[7] and GA[8] rise to ~100+ decaps and ~1000+ s — [GitHub fig-robustness.png](https://github.com/SoFoolish250/Physics-Assisted-Genetic-Algorithm-PAGA-for-Decap-Optimization)
- Reference identities, resolved through Crossref's ordered reference list for the TMTT DOI (I assume the order matches the paper's numbering; consistent for [7],[8],[9],[15],[16]):

  | Ref | Paper |
  |---|---|
  | [6] | de Paulis, Cecchetti, Olivieri, Buecker, "Genetic Algorithm PDN Optimization based on Minimum Number of Decoupling Capacitors Applied to Arbitrary Target Impedance," EMCSI 2020 |
  | [7] | Xu et al., "Jitter-Aware Economic PDN Optimization With a GA," TMTT 2021 |
  | [8] | Juang, Zhang, Kiguradze, Pu, Jin, Hwang, "A Modified Genetic Algorithm for the Selection of Decoupling Capacitors in PDN Design," EMC/SI/PI+EMC Europe 2021 |
  | [9] | Zhang, Jiang et al., "A Physics-Based GA for Decap Optimization in PDNs," EMC+SIPI 2023 (PAGA's own conference precursor) |
  | [15] | Wang et al., "Multiport PDN Optimization With the Newton–Hessian Minimization Method," TMTT 2021 |
  | [16] | Koo et al., "Fast Algorithm for Minimizing the Number of decap in PDNs," TEMC 2018 |

  — [Crossref](https://api.crossref.org/works/10.1109/TMTT.2024.3356575)
- Method: PAGA ranks ports by physical loop inductance, seeds an initial solution by placing decaps in priority order, then runs a GA. An aggregator summary says it also includes a port-removal step (abstract-level, not verified in full text) — [search summary of PAGA PDF](https://www.researchgate.net/publication/376201929_A_Novel_Physics-Assisted_Genetic_AlgorithmPAGA_for_Decoupling_Capacitor_Optimization)

### Inferences
- In the proposal, write: "On a 120-port case, sequential search (de Paulis 2020) needs 61 decaps and PAGA needs 24; the sorting-search method (Koo 2018) found no solution." Drop the "~20 ports / Juang" attribution. Calling sequential search "greedy" is reasonable but should be labeled as the paper's "sequential search".
- PAGA (and GA[9]) reaching 24 on the 120-port case is a strong classical baseline, and the repo makes it runnable. Any learned method should report against it.

### Gaps
- PAGA's full text was blocked (ResearchGate 403). The case-2 port count and the details of the removal step are not verified.
- I found no source showing a greedy method failing on a ~20-port board. That claim should be removed or re-sourced.

## Q6. Bansal/Schwarzschild et al. 2022: "recurrent models extrapolate 9×9 → 201×201 mazes"

### Takeaway
CONFIRMED, with a qualification. Training uses only 9×9 mazes. The headline benchmark is 59×59 (97.30% ± 0.68). 201×201 reaches 74% after 2,000 iterations, and one 801×801 maze is shown solved. The proposal's "input re-injection every iteration" and "random iteration counts during training" are this paper's "recall" and "progressive training". They must be cited as borrowed, not presented as new.

### Cited Findings (full text via ar5iv)
- A. Bansal, A. Schwarzschild, E. Borgnia, Z. Emam, F. Huang, M. Goldblum, T. Goldstein, "End-to-end Algorithm Synthesis with Recurrent Networks: Logical Extrapolation Without Overthinking," arXiv 2202.05826 (NeurIPS 2022) — [arXiv](https://arxiv.org/abs/2202.05826)
- "All training in our work is done on 9×9 mazes." Table 4: on 13×13, DT-Recall reaches 99.94 ± 0.03%; on 59×59, DT-Recall with progressive loss (α = 0.01) reaches 97.30 ± 0.68%, while DT-Recall with α = 0 reaches 82.72 ± 15.14% and plain DT and feed-forward models reach 0.00% — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.05826)
- "our models can solve 74% of the 201×201 mazes". Footnote: of 100 mazes, 20 have one pixel wrong and no maze has more than seven mistakes. The text says 201×201 used 2,000 iterations and 801×801 used 20,000, while Appendix A.10 says 10,000 for the 801×801 example, an internal inconsistency — [ar5iv](https://ar5iv.labs.arxiv.org/html/2202.05826)
- The two fixes are a recall architecture (concatenating the input to the recurrent state every iteration) and progressive training (training from a random intermediate iteration to discourage iteration-specific behavior) — [arXiv abs](https://arxiv.org/abs/2202.05826)
- The precursor is Schwarzschild et al., "Can You Learn an Algorithm?" (NeurIPS 2021, arXiv 2106.04537) — [arXiv](https://arxiv.org/abs/2106.04537)

### Inferences
- Rewrite the proposal sentence as: "trained on 9×9, 97% on 59×59 and 74% on 201×201 (more iterations at test time)."

### Gaps
- None.

## Q7 (KEY). Novelty: learned iterative solver or physics-structured message passing inside a decap/PDN policy; GNN surrogates for PDN impedance

### Takeaway
I found no published work that puts a weight-tied, admittance-weighted, Jacobi-like message-passing solver with residual-based stopping inside a decap-placement policy. Nor did I find a GNN surrogate that predicts PDN Z(f) from the RLGC grid. The closest pieces exist separately:
- (a) Decap RL/GA work uses graph or transformer policies over ports or coordinates (Manoharan GCN-assisted GA 2024; Park 2022; DevFormer; PhysEDA).
- (b) Chip IR-drop GNN surrogates (PDNNet; IRGNN) handle DC or transient droop, not Z(f).
- (c) Physics-structured "neural solvers" exist outside the PDN domain: the Graph Neural Solver for power flow minimizes Kirchhoff violation; Hsieh 2019 is a learned Jacobi correction with convergence guarantee; learned AMG and preconditioners provide coarse grids; DEQ surrogates exist for power-system dynamics; Bansal 2022 provides recall and progressive training.

The novelty claim should therefore be "first combination of these in decap placement", not "first neural circuit solver".

### Cited Findings: decap/PDN-specific (closest)
- **Manoharan, Juang, Zhang, Wang, Pan, Qiu, Gao, Hwang, "Graph Convolutional Neural Network Assisted Genetic Algorithm for PDN Decap Optimization," IEEE EMC+SIPI 2024, pp. 146–150, DOI 10.1109/EMCSIPI49824.2024.10705608.** An RL agent with a GCN policy predicts a decap solution from the PDN impedance and target, and that solution seeds the GA population. It handles variable numbers of ports, transfers across decap libraries, and is about 2× faster than GA alone. Closeness: PARTIAL. It is a GNN-in-the-loop decap policy, but there is no evidence that edges carry circuit admittances, that it iterates like a solver, or that it removes the GA. *Abstract and intro only (full PDF 403).* — [NSF PAR](https://par.nsf.gov/biblio/10566625-graph-convolutional-neural-network-assisted-genetic-algorithm-pdn-decap-optimization), [Scholars' Mine](https://scholarsmine.mst.edu/ele_comeng_facwork/6794)
- **Manoharan et al., "Graph-Based RL Approach for Multi-Power-Domain PCB PDN Shape and Stackup Synthesis," DesignCon 2025**, plus multi-agent RL decap layout synthesis for multi-power-domain PCBs (DesignCon 2025). These are graph-based RL for pre-layout synthesis. Closeness: DIFFERENT (synthesis, not a solver-structured surrogate). *Metadata and slides only.* — [NSF PAR author list](https://par.nsf.gov/search/author:%22Hwang,%20Chulsoon%22), [IBIS slides](https://www.ibis.org/summits/aug25/manoharan.pdf)
- **"Augmented Genetic Algorithm v2 with Reinforcement Learning for PDN Decap Optimization"** (Missouri S&T / Hwang group). An RL agent speeds up GA convergence. A search summary also describes a "brute force check function" that removes decaps one at a time while the target is still met, in inductance-priority order. Closeness: PARTIAL for the add-then-prune idea. *Search summary only; authors not confirmed.* — [NSF PAR PDF](https://par.nsf.gov/servlets/purl/10465877)
- **Zhang et al., PPO for decap optimization of complex/irregular PDNs**: minimizes decap count under the target impedance, converging to 26–28 decaps vs 33–38 from full search. Closeness: PARTIAL (count-minimizing RL, conventional networks). *Summary only.* — [Scholars' Mine PDF](https://scholarsmine.mst.edu/cgi/viewcontent.cgi?article=5766&context=ele_comeng_facwork)
- **Enhanced DRL for decap selection (Missouri S&T)**: double Q-learning, prioritized replay, impedance-matrix reduction for fast impedance calculation, and the minimum-count objective — [ResearchGate](https://www.researchgate.net/publication/344819822_An_Enhanced_Deep_Reinforcement_Learning_Algorithm_for_Decoupling_Capacitor_Selection_in_Power_Distribution_Network_Design)
- **Suh, An, H. Kim, Choi, T. Kim, Son, K. Kim, J. Yoon, B. Kim, Y. Yoon, Bae, ... (KAIST), "Multi-Chiplet Power Delivery Network (PDN) Co-optimization Considering Current Spectrum by Deep Reinforcement Learning-based Decoupling Capacitor Placement," ECTC 2025, p. 181.** The title is confirmed from the official proceedings TOC. Abstract and keep-out details are not accessible, so overlap on keep-out constraints is UNVERIFIABLE. The RL4CO/DevFormer DPP benchmark (same KAIST lineage) already includes keep-out regions, where decaps cannot go on probe ports or keep-out cells — [ECTC 2025 TOC](https://www.proceedings.com/content/080/080803webtoc.pdf), [RL4CO](https://arxiv.org/abs/2306.17100)
- **Duan et al., "Hierarchical Decap Optimization for PDN of 2.5D ICs with Co-Analysis of Frequency and Time Domains Based on DRL," arXiv 2407.04737.** Uses DRL on chiplet and interposer 1 mm × 1 mm unit cells: frequency-domain target first, then time-domain voltage violation integral. Closeness: DIFFERENT (standard DRL; v2 shows a withdrawal note, v3 is current) — [arXiv](https://arxiv.org/abs/2407.04737v3)
- **GNN on PCB schematics (arXiv 2506.10577).** Predicts whether and how many decaps an engineer adds between nets, as a schematic-level classifier. The authors find node features matter more than graph structure for decap insertion. Closeness: DIFFERENT — [arXiv](https://arxiv.org/pdf/2506.10577)
- **Non-GNN PDN impedance surrogates.** A CNN trained on BEM gives a ~500× speedup and is paired with a GA (ResearchGate, "Fast impedance prediction for PDN using deep learning"). An ANN-surrogate GA iteratively increases the decap count (Adv. Radio Sci. 2024). MLP vs GPR for PDN Z with varying decaps finds GPR more accurate (IJAI). Closeness: DIFFERENT (black-box surrogates) — [ResearchGate](https://www.researchgate.net/publication/355125017_Fast_impedance_prediction_for_power_distribution_network_using_deep_learning), [ARS 2024](https://ars.copernicus.org/articles/21/123/2024/), [IJAI](https://ijai.iaescore.com/index.php/IJAI/article/view/21358)
- **PINN droop surrogate for 2.5-D chiplet PDN (IEEE, 2026)** embeds circuit current and voltage constraints in training. Closeness: PARTIAL (physics-informed PDN surrogate, but a PINN, not a message-passing solver or placement policy). *Abstract only.* — [IEEE CS DL](https://www.computer.org/csdl/journal/si/2026/04/11372611/2dZJiB5wzbq)
- **TechRxiv review (Oct 2025) of AI for SI/PI** notes AI surrogates for PDN impedance. I could not confirm whether it discusses GNNs — [TechRxiv](https://www.techrxiv.org/doi/pdf/10.36227/techrxiv.176045677.72121155/v1)

### Cited Findings: chip power-grid GNN surrogates (DC/transient, not Z(f))
- **PDNNet (arXiv 2403.18569)**: a GNN-CNN heterogeneous network for dynamic IR drop. "PDNGraph" encodes PDN structure and cell-to-PDN relations, motivated by the nodal system GV = J. Closeness: PARTIAL (a graph over the PDN, but supervised, single-pass, time-domain droop, no placement) — [arXiv](https://arxiv.org/pdf/2403.18569)
- **IRGNN**: a graph framework for static IR drop that integrates a numerical solution with point clouds and exploits grid topology. Closeness: PARTIAL (hybrid with a numerical solver). *Abstract/snippet only; the snippet mixed in unrelated text, so verify.* — [ResearchGate](https://www.researchgate.net/publication/395628847_IRGNN_A_Graph-based_Framework_Integrating_Numerical_Solution_and_Point_Cloud_for_Static_IR_Drop_Prediction)
- Most static IR-drop predictors are U-Net/CNN (MAUnet, WACA-UNet, Attention U-Net) and supervised — [WACA-UNet](https://arxiv.org/pdf/2507.19197), [MAUnet](https://www.ssslab.cn/assets/papers/2024-wang-MAUnet.pdf)

### Cited Findings: physics-structured / learned iterative solvers (outside PDN; methodological precedents for the inner loop)
- **Donon et al., "Neural Networks for Power Flow: Graph Neural Solver," PSCC 2020** (also "Graph Neural Solver for Power Systems," IJCNN 2019). A GNN trained without labels by minimizing Kirchhoff-law violation at every bus, with iterative message passing over line parameters. Robust to topology and line changes; IEEE 9/14/30/118 cases. Closeness: the CLOSEST conceptual precedent for "residual-driven message passing weighted by branch admittances", but for transmission-grid AC power flow with no placement policy — [HAL](https://hal.archives-ouvertes.fr/hal-02372741), [PSCC PDF](https://pscc-central.epfl.ch/repo/papers/2020/715.pdf)
- Follow-ups: Böttcher et al., unsupervised GNN AC power flow on distribution grids, which beats Donon in accuracy; Lopez-Garcia & Domínguez-Navarro — [PwC](https://ml.paperswithcode.com/paper/solving-ac-power-flow-with-graph-neural), [Zaguan](https://zaguan.unizar.es/record/131215/files/texto_completo.pdf). A 2025 grid study found purely unsupervised physics-loss training fragile (initial losses near 1e8) and preferred supervised pre-training plus physics fine-tuning — [arXiv 2509.25158](https://arxiv.org/pdf/2509.25158). This supports the proposal's auxiliary supervised loss on small boards.
- **Hsieh, Zhao, Eismann, Mirabella, Ermon, "Learning Neural PDE Solvers with Convergence Guarantees," ICLR 2019.** Adds a learned linear correction H (conv net or U-Net) to a standard iterative solver such as Jacobi, preserving the fixed point and convergence. Reports a 2–3× reduction in multiply-adds and generalization to unseen geometries. Closeness: direct precedent for "one Jacobi step plus learned correction, weight-tied, iterate to residual". *Abstract and summaries.* — [arXiv](https://arxiv.org/abs/1906.01200)
- **Learned multigrid and preconditioners (coarse-grid precedent):**
  - Luz et al., "Learning algebraic multigrid using GNNs," ICML 2020 (cited from reference lists only).
  - Chen, "Graph Neural Preconditioners for Iterative Solutions of Sparse Linear Systems," arXiv 2406.00809, evaluated on 800+ matrices.
  - RAPNet (arXiv 2605.26854): GNN sparse corrections to AMG, weight sharing from small to large graphs.
  - "Graph Neural Multilevel Preconditioners" (arXiv 2607.28456): MPNN smoother plus coarse solve.
  - Trifonov et al. (arXiv 2502.01397): message-passing GNNs cannot approximate sparse triangular factorizations for broad problem classes. This is a relevant expressivity caveat for a purely local inner loop and supports keeping the coarse-grid attention.

  Closeness: PARTIAL (solvers only, no circuits or placement) — [arXiv 2406.00809](https://arxiv.org/abs/2406.00809), [RAPNet](https://arxiv.org/html/2605.26854), [GNMP](https://arxiv.org/html/2607.28456), [Trifonov](https://arxiv.org/pdf/2502.01397)
- **Admittance-as-attention in circuit GNNs: "RF-Informed GNNs for … Circuit Performance Prediction" (arXiv 2508.16403).** The attention weight is interpreted as normalized admittance g/Σg, and message mismatch is related to the Newton–Raphson current mismatch. Closeness: PARTIAL (the analogy is built into the architecture but does not enforce KCL; RF performance prediction, not PDN or placement) — [arXiv](https://arxiv.org/html/2508.16403)
- **SuperCond-GNN (arXiv 2606.23548)**: a MeshGraphNet-style encode → message-pass → decode model to nodal voltages for superconducting circuits. Closeness: DIFFERENT domain — [arXiv](https://arxiv.org/pdf/2606.23548)
- **DEQ for power-system dynamics (arXiv 2405.06827; IEEE TPWRS 2025)**: deep-equilibrium layers plus a neural-ODE reduced-order surrogate. I found no DEQ applied to chip or package PDN impedance or decap placement — [arXiv](https://arxiv.org/pdf/2405.06827)

### Cited Findings: add/remove (prune) moves in the decap literature
- Pruning by sequential removal while the target still holds already exists as GA post-processing (AGA v2 "brute force check function"), with removal order set by inductance priority — [NSF PAR](https://par.nsf.gov/servlets/purl/10465877)
- PAGA reportedly includes a port-removal step (aggregator summary, unverified) — [NSF PAR record](https://par.nsf.gov/biblio/10566602-novel-physics-assisted-genetic-algorithm-decoupling-capacitor-optimization)
- An ANN-surrogate GA iterates over increasing decap counts to find the minimum — [ARS 2024](https://ars.copernicus.org/articles/21/123/2024/)
- Koo et al., TEMC 2018, "Fast Algorithm for Minimizing the Number of decap in PDNs" is a classical count-minimizing heuristic. It is the "sorting search" baseline in PAGA — [Crossref](https://api.crossref.org/works/10.1109/TMTT.2024.3356575)
- I found no RL policy whose action space includes an explicit *remove* action learned jointly with *add* and *stop* for decap placement.

### Inferences
- The defensible novelty points, given what I found:
  1. A circuit-solver-structured inner loop (admittance-weighted, weight-tied, residual-stopped) used as the state encoder of a decap policy. No decap paper found does this; Donon (power flow) and Hsieh (PDE Jacobi) are the precedents to cite.
  2. Incremental state update: adding a decap edits only a diagonal admittance, so there is no Z-matrix input. This contrasts with Manoharan/PAGA (Z-matrix input) and Park/DevFormer/PhysEDA (coordinates only).
  3. Learned add/remove/stop actions. Removal exists only as a heuristic post-processing step.
- Pieces that must be cited, not claimed as new:
  - recall/input re-injection and random iteration counts (Bansal 2022)
  - coarse-grid correction (learned AMG literature)
  - physics-residual training (Donon; PINN line)
  - self-imitation / best-of-K (standard RL; DevFormer and Park use sampling widths)
  - a reward normalized by a lower bound (not verified as prior art in decap work; I did not search for it specifically)
- Main risk: the Hwang group (Manoharan) is already moving toward graph-based and Transformer RL with Z input. A GCN-based policy exists (EMC+SIPI 2024), so the proposal must state precisely how its message passing differs: admittance-weighted, solver-structured, and iterated to convergence, versus a generic GCN over ports.

### Gaps
- No full-text access to the GCN-assisted GA paper (EMC+SIPI 2024), the TSIPI 2026 paper, the ECTC 2025 Suh et al. paper, or PAGA's text. Whether any of them weights graph edges by circuit admittance or uses multi-frequency node channels is UNVERIFIED. This is the single most important check left: read the GCN-assisted GA paper's architecture section.
- IEEE Xplore pages for EPEPS, EDAPS, DesignCon 2025–2026 and ICCAD/DAC 2025–2026 could not be searched directly. Recent (≤3 months old) arXiv or IEEE early-access papers may not be indexed by the search tool.
- I did not find, but cannot rule out, Korean-venue (KAIST TERA lab) theses or EDAPS 2025 talks with GNN decap surrogates. The EDAPS 2025 program lists a Switch-Transformer RL for HBM PSIJ and a GAIL chiplet placement talk, neither a GNN solver — [EDAPS 2025 program](https://easychair.org/smart-program/EDAPS2025/2025-12-17.html)
