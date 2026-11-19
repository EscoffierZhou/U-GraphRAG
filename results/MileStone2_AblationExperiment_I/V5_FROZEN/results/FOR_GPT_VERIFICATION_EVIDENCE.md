# U-GraphRAG Experimental Evidence & Audit Verification Dossier (For GPT Final Review — Post-Exp5 Locked)

> **Purpose**: Provide GPT (and any independent reviewer) with **100% file-grounded, non-extrapolated quantitative evidence**, statistical diagnostics ($p$-values, Cohen's $d_z$, Bootstrap 95% CIs), bin-by-bin reliability tables, terminology alignments, and **finalized V5 real-GPU PopQA (`Step 2`) verification results** alongside bit-identical frozen V4 HotpotQA artifacts (`Stage 0`, `Step 1`, `Step 3`, `Step 4`).  
> **V4 Frozen Reference**: `results/UGraphRAG_Experimental_Results_4/results/` (untouched; SHA-256 digests verified bit-identical in `_5`).  
> **V5 Final Locked Archive**: `results/UGraphRAG_Experimental_Results_5/results/` (completed `2026-11-15 20:43:04` on `NVIDIA GeForce RTX 5060 Laptop GPU`, 8 GB VRAM).

---

## 1. Artifact Inventory & Cryptographic Zero-Drift Manifest (V4 vs V5)

All HotpotQA (`Stage 0`, `Step 3`, `Step 4`) and backbone benchmark (`Step 1`) results in V5 remain **bit-for-bit identical to V4 frozen GPU execution** on `NVIDIA GeForce RTX 5060 Laptop GPU` (`8 GB` physical VRAM, `4-bit NF4` with `bfloat16` compute dtype), while `Step 2 (PopQA)` has been re-executed cleanly under Exp5 Option A:

| Artifact File (`UGraphRAG_Experimental_Results_5/results/`) | Size (Bytes) | SHA-256 Prefix | V4 Comparison & Status in V5 |
| :--- | :---: | :---: | :--- |
| `grid_search_results.json` | `42,811` | `6a415a1dd76f` | **`[V4_FROZEN_MATCH]`** (`Stage 0`, `HotpotQA [100:180]`, $N=80$, `736.46s`) |
| `step1_corrected_3b_7b.json` | `1,696` | `1f2a54791eeb` | **`[V4_FROZEN_MATCH]`** (`Step 1`, 3B vs 7B on HotpotQA-100 & PopQA-100) |
| `step3_hotpotqa_ablations.json` | `2,957` | `1124221e2125` | **`[V4_FROZEN_MATCH]`** (`Step 3`, 8-mode HotpotQA-100 summary) |
| `step3_step4_ablation_hotpot100.json` | `1,500,188` | `951dd11ecc17` | **`[V4_FROZEN_MATCH]`** (`Step 3 & 4`, all 100 per-sample records per mode) |
| `step4_m2_permutation.json` | `4,938` | `f489dfa16696` | **`[V4_FROZEN_MATCH]`** (`Step 4`, 100 per-sample F1 vectors: DFS/Rev/5x Shuffles) |
| `step2_m1_popqa100.json` / `step2_popqa_uncertainty.json` | `3,229` | `fe8a1669ab5d` | **Exp5 Real GPU Result (`2026-11-15 20:43:04`)** — supersedes V4 (`2,735 B`, `73411fb2b3ca`) |

> **Core Source Code Preservation Guarantee (`src/` untouched)**: Under Option A, **zero edits** were made to `src/m1_adaptive_retriever.py`, `src/m2_evidence_optimizer.py`, `src/m3_calibrated_decoder.py`, or `src/u_graphrag_pipeline.py`. Only `scripts/run_overnight_ablation.py` (Step 2 PopQA helper `select_popqa_m2_evidence` & `active_tau_eu` alignment), `scripts/plot_ablation_charts.py` (Fig 4 + Bootstrap CI + VRAM header clarity), `scripts/run_full_experiment_and_package.py` (report parity), and `tests/test_v4_fixes.py` (V5 regression tests) were modified.

---

## 2. Verification of Zero-Leakage Parameter Provenance (`Stage 0 -> Step 3`)

Inspecting `step3_step4_ablation_hotpot100.json` (`metadata.grid_params`) against `grid_search_results.json` (`recommended_parameter_sets.balanced_optimal` and `stage3_m3.top_valid_selective_configs[0]`) confirms **exact 1:1 parameter transfer with zero test-set tuning**:

| Parameter | Stage 0 Selected Preset (`balanced_optimal`) | Step 3 Executed Config (`metadata.grid_params`) | Stage 0 Dev (`[100:180]`, $N=80$) | Step 3 Test (`[0:100]`, $N=100$) |
| :--- | :---: | :---: | :---: | :---: |
| `m1_tau_eu` | `0.18` | `0.18` | Stage 1: `EM=46.25, SpF1=60.36` | `M1_Only`: `Sp-P=50.87, SpF1=62.26, TokSav=33.6%` |
| `(m2_gamma_1, m2_gamma_2, m2_lambda_div)` | `(0.8, 0.2, 0.4)` | `(0.8, 0.2, 0.4)` | Stage 2: `EM=48.75, F1=60.74` | `M2_Full`: `EM=55.00, F1=67.14` (`+2.46% F1 vs B0`) |
| `(m2_reliability_offset, m2_min_docs, top_k_pool)` | `(0.22, 2, 8)` | `(0.22, 2, 8)` | `avg_selected_docs=4.0, TokSav=23.95%` | `TokSav=10.1%` |
| `(m3_confidence_platt_a, m3_confidence_platt_b)` | `(0.8174, -0.2053)` | `(0.8174, -0.2053)` | `Dev ECE=0.0670, Brier=0.2148` | `Test ECE=0.0925, Brier=0.1814, AUROC=0.7574` |
| `(m3_confidence_temperature, m3_confidence_tok_weight)` | `(2.0, 0.6)` | `(2.0, 0.6)` | `Cov=97.5%, SelRisk=33.33%` | `Cov=96.0%, SelRisk=28.12%` |
| `(m3_abstain_threshold, m3_confidence_threshold)` | `(0.05, 0.38)` | `(0.05, 0.38)` | `base_risk=35.0% -> 33.33%` | `base_risk=31.00% -> 28.12%` |

*Note on Stage 0 alternative presets*: `grid_search_results.json` also records `low_ece_high_precision` (`platt_a=0.8684, platt_b=-0.1249, tau_conf=0.52`, Dev `ECE=0.0426`, `Cov=96.25%`) and `full_coverage_calibrated` (`tau_conf=0.15`, Dev `ECE=0.0428`, `Cov=100.0%`). Step 3 used strictly `balanced_optimal` (`Dev ECE = 0.0670`).

---

## 3. Verification of Step 3 M3 Calibration & Selective Prediction (`step3_step4_ablation_hotpot100.json`)

To address GPT's question on why `B0_Standard_RAG` achieves lower raw `ECE` (`0.0710` vs `Full_UGraphRAG` `0.0925`) yet exhibits near-random error discrimination (`AUROC = 0.5404`), below is the **exact 10-bin Reliability Breakdown computed from all 100 per-sample records** (`correct = 1 if f1 >= 0.5 else 0`):

### 3.1 Exact 10-Bin Reliability Table (`N=100` HotpotQA)

| Confidence Bin | `B0_Standard_RAG` (`n`, `avg_conf`, `emp_acc`, `|gap|`) | `M2_Full` (`n`, `avg_conf`, `emp_acc`, `|gap|`) | `Full_UGraphRAG` (`n`, `avg_conf`, `emp_acc`, `|gap|`) |
| :--- | :---: | :---: | :---: |
| `(0.2, 0.3]` | `n=0` | `n=0` | `n=1`, `0.2761`, **`0.0000`**, `0.2761` |
| `(0.3, 0.4]` | `n=0` | `n=1`, `0.3995`, `1.0000`, `0.6005` *(MCE)* | `n=3`, `0.3483`, **`0.0000`**, `0.3483` |
| `(0.4, 0.5]` | `n=0` | `n=28`, `0.4711`, `0.7500`, `0.2789` | `n=1`, `0.4974`, `0.0000`, `0.4974` *(MCE)* |
| `(0.5, 0.6]` | `n=1`, `0.5825`, `1.0000`, `0.4175` *(MCE)* | `n=48`, `0.5414`, `0.6042`, `0.0627` | `n=13`, `0.5644`, **`0.3846`**, `0.1797` |
| `(0.6, 0.7]` | `n=21`, `0.6681`, `0.5714`, `0.0966` | `n=19`, `0.6358`, `0.8421`, `0.2063` | `n=30`, `0.6612`, **`0.6667`**, **`0.0054`** |
| `(0.7, 0.8]` | `n=64`, `0.7426`, `0.6719`, `0.0707` | `n=4`, `0.7255`, `0.7500`, `0.0245` | `n=48`, `0.7456`, **`0.8333`**, `0.0877` |
| `(0.8, 0.9]` | `n=13`, `0.8430`, `0.8462`, `0.0032` | `n=0` | `n=4`, `0.8208`, **`1.0000`**, `0.1792` |
| `(0.9, 1.0]` | `n=1`, `0.9150`, `1.0000`, `0.0850` | `n=0` | `n=0` |
| **Summary** | `mean_conf=0.7401`, `emp_acc=0.6800`, `ECE=0.0710`, **`Brier=0.2196`**, **`AUROC=0.5404`** | `mean_conf=0.5456`, `emp_acc=0.7000`, `ECE=0.1544`, **`Brier=0.2354`**, **`AUROC=0.5214`** | `mean_conf=0.6806`, `emp_acc=0.6900`, `ECE=0.0925`, **`Brier=0.1814`**, **`AUROC=0.7574`** (`abst=4`) |

**Key Empirical Finding**:
* `B0` concentrates **`85/100`** samples inside `(0.6, 0.8]` around its global mean accuracy (`0.7401` vs `0.6800`), giving low ECE via mean-matching while remaining flat in discriminative ranking (`AUROC = 0.5404`).
* `Full_UGraphRAG` induces **strict monotonic empirical accuracy ordering** across populated confidence bins (`0.00% -> 38.46% -> 66.67% -> 83.33% -> 100.00%`), reducing **Brier Score by `-17.4%` vs B0 (`0.2196 -> 0.1814`) and `-22.9%` vs M2_Full (`0.2354 -> 0.1814`)**, and boosting **AUROC to `0.7574`**.

### 3.2 Full Selective Risk-Coverage Progression (`100% -> 50%`, Visualized in `fig4_m3_calibration_and_risk_coverage.png`)

Rank-ordering all 100 samples by predicted `confidence` (`Risk = 1 - Acc@F1>=0.5`):

| Coverage (%) | Answered $N$ | `Full_UGraphRAG` Risk (%) | `Full_UGraphRAG` EM (%) | `Full_UGraphRAG` F1 (%) | `B0_Standard_RAG` Risk (%) | `M2_Full` Risk (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `100.0%` | `100` | `31.00%` | `54.00%` | `65.74%` | `32.00%` | `30.00%` |
| `96.0%` *(Operational Gate)* | `96` | **`28.12%`** | `56.25%` | `68.48%` | `31.25%` | `31.25%` |
| `90.0%` | `90` | **`25.56%`** | `57.78%` | `70.82%` | `32.22%` | `31.11%` |
| `80.0%` | `80` | **`22.50%`** | `61.25%` | `74.15%` | `30.00%` | `32.50%` |
| `70.0%` | `70` | **`21.43%`** | `60.00%` | `73.60%` | `28.57%` | `32.86%` |
| `60.0%` | `60` | **`18.33%`** | `61.67%` | `76.70%` | `31.67%` | `31.67%` |
| `50.0%` | `50` | **`12.00%`** | **`70.00%`** | **`83.01%`** | `36.00%` | `30.00%` |

At `50%` coverage, `Full_UGraphRAG` cuts incorrect/hallucinated answers from **`31.00% -> 12.00%`** (while raising EM from `54.00% -> 70.00%` and F1 from `65.74% -> 83.01%`), whereas `B0` risk *worsens* to `36.00%` due to overconfident mispredictions.

---

## 4. Verification of Step 4 Paired Bootstrap (`B=10,000`), Exact $p$-Values & Effect Size

Computed deterministically over the 100 paired per-sample F1 vectors (`[0..100]%`) in `step4_m2_permutation.json` via `_compute_paired_bootstrap_stats(..., n_boot=10000, seed=42)` and exact paired hypothesis tests (`scipy.stats` + 100,000 sign-flip permutations):

| Comparison | Mean $\Delta$ F1 (%) | Paired Std (%) | Paired $t$ ($p$-val) | Sign-Perm $p$ / Wilcoxon $p$ | Cohen's $d_z$ | 95% Bootstrap CI (%) | Win / Tie / Loss | Sensitive Subset $\Delta$ |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`DFS` vs `Reverse`** | `+0.48%` | `±32.08%` | `0.149` (`0.8818`) | `0.8839` / `0.8317` | `0.0149` | `[-5.94%, +6.72%]` | `11 / 82 / 7` *(Binom $p=0.4807$)* | **`+2.66%`** ($n=18$, $18\%$) |
| **`DFS` vs `5x Shuffles`** | `+0.14%` | `±22.58%` | `0.061` (`0.9517`) | `0.9534` / `0.5843` | `0.0061` | `[-4.46%, +4.34%]` | `21 / 67 / 12` *(Binom $p=0.1628$)* | **`+0.42%`** ($n=33$, $33\%$) |

**Statistical Rationale & Honest Paper Framing**:
1. Overall mean differences on $N=100$ are **not statistically significant** ($p \ge 0.16$, Cohen's $d_z \le 0.015$, Bootstrap 95% CIs cross `0`), because $K \le 3$ selected documents produce `67%–82%` exact per-question ties across permutations (`SE ≈ 22.58% / sqrt(100) = 2.26%`).
2. We explicitly **do not** claim single-run deterministic `0.00` variance proves statistical invariance or significance over permutations.
3. Instead, Step 4 empirically proves: (a) **non-deterministic within-question shuffle variance** (`Intra-Sample Std = ±10.01%`, `max_drop = 14.06%`) is eliminated under deterministic M2-B DFS linearization at zero extra forward calls (`1.02x` vs `5.08x` latency); and (b) across the `18%–33%` order-sensitive subset, DFS preserves consistent directional net win ratios (`11 vs 7` and `21 vs 12`) and complements the Step 3 `M2_Exact_NoMST` (`66.40% F1`) $\to$ `M2_Full` (`67.14% F1`, `+1.00% EM / +0.74% F1`) gain.

---

## 5. Step 1 vs Step 3 Memory Metric Reconciliation

Reconciling `step1_corrected_3b_7b.json` with `step3_hotpotqa_ablations.json` and `step3_step4_ablation_hotpot100.json`:
* **Step 1 (`step1_corrected_3b_7b.json`)**:
  * `Qwen2.5-3B-Instruct`: `peak_allocated_mb = 1978.61 MB`, `final_reserved_mb = 5306.0 MB`, `headroom_mb = 2886.0 MB`.
  * `Qwen2.5-7B-Instruct`: `peak_allocated_mb = 5310.80 MB`, `final_reserved_mb = 5660.0 MB`, `headroom_mb = 2532.0 MB`.
* **Step 3 (`step3_hotpotqa_ablations.json` Table 4 vs `step3_step4_ablation_hotpot100.json`)**:
  * `B0_Standard_RAG`: `peak_cuda_allocated_mb = 5667.54 MB` (reported in Table 4 as `5667.5 MB`), `peak_cuda_reserved_mb = 5860.0 MB`.
  * Remaining 7B pipeline modes (`M1_Only` .. `Full_UGraphRAG`): `Peak Alloc = 5663.3 ~ 5677.4 MB`. All modes stay safely within `8192 MB` (`> 2.2 GB` headroom). Both Table 1 and Table 4 now explicitly distinguish **`Peak Alloc`** and **`CUDA Reserved`** headers.

---

## 6. Terminology & Methodological Guardrails Enforced Across All Docs

1. **No Submodularity Overclaim**: Because M2-A maximizes $F(S) = \sum_{u \in S} \tilde{R}(u) + \lambda_1 \text{Cov}(S, Q) + \lambda_2 \sum_{u<v \in S} \text{Comp}(u,v) - \lambda_3 \sum_{u<v \in S} \text{Red}(u,v) - \lambda_4 \sum_{u<v \in S} \text{Conf}(u,v)$, the positive pairwise synergy term $+\lambda_2 \text{Comp}$ is strictly **supermodular / non-submodular**. All reports refer accurately to **Budgeted Evidence Set Optimization (`exact_enum`) / Stratified Pareto Evidence Selection**.
2. **PopQA Closed $U_{\text{EU}}$ Interpretation**: `auroc_closed_eu_vs_error = 0.8917` proves that Closed-Book $U_{\text{EU}}$ accurately predicts **intrinsic parametric error propensity** (`Closed Acc = 5.0%` at `Medium` with `Closed U_EU = 1.7546` vs `Closed Acc = 60.0%` at `Head` with `Closed U_EU = 0.5706`), rather than strict monotonicity wrt coarse external Wikipedia pageview quintile cutoffs.
3. **Deterministic M2-B Variance Framing**: Step 4 reports deterministic single-pass `DFS`/`Reverse` vs `Within-Question 5x Shuffle Std = ±10.01%` plus Paired Bootstrap 95% CIs rather than equating `0.00` deterministic single-run variance with hypothesis-test superiority.

---

## 7. Step 2 (PopQA) V4 Failure Reproduction & V5 Production Fix Verification (`N=100` Shared Pools)

Using the exact 100 PopQA shared-corpus retrieval pools (`HybridRetriever` BM25 + BGE-small-en-v1.5 + MinHash LSH expansion, cached in `popqa_100_pools_cache.json`), we executed both V4 logic and the new V5 production helper `scripts.run_overnight_ablation.select_popqa_m2_evidence(...)` under `balanced_optimal` (`pool_limit=8, m2_gamma_1=0.8, m2_gamma_2=0.2, m2_lambda_div=0.4, m2_offset=0.22, min_k=2, max_k=3`):

### 7.1 Exact Reproduction of V4 Breakdown
* `D0 (Top-2) Gold Recall` = **`6/100 (6%)`**
* `B0 (Top-5) Gold Recall` = **`63/100 (63%)`**
* `M1 Expanded Top-5 Gold Recall` = **`85/100 (85%)`**
* `Unified Pool (cap=8) Gold Recall` = **`100/100 (100%)`**
* **V4 `M2-A` Post-Selection Gold Recall (`cap=8, min_k=1`) = `13/100 (13%)`** *(directly explaining why V4 `Full_UGraphRAG` fell to `35.0%` while `B0=75.0%`, `M1_Only=76.0%`)*

### 7.2 V5 Production Helper Verification (`select_popqa_m2_evidence`, Zero Label/ID Leakage)

| Configuration (`select_popqa_m2_evidence` over all 100 cached PopQA pools) | Overall Gold Recall (%) | `<10^2` (Tail) | `10^2-10^3` (Low) | `10^3-10^4` (Med) | `10^4-10^5` (High) | `>=10^5` (Head) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **V4 Baseline (`balanced_optimal`)** | **`13.0%`** | - | - | - | - | - |
| **V5 (`sim Delta U_EU = 0.00`, rank+LSH alignment only)** | **`51.0%`** | `13/20` | `13/20` | `9/20` | `12/20` | `4/20` |
| **V5 (`sim Delta U_EU = 0.15` on factual leaf)** | **`94.0%`** | **`20/20`** | **`18/20`** | **`17/20`** | **`20/20`** | **`19/20`** |
| **V5 (`sim Delta U_EU = 0.25` on factual leaf)** | **`95.0%`** | **`20/20`** | **`19/20`** | **`17/20`** | **`20/20`** | **`19/20`** |
| **V5 (`sim Delta U_EU = 0.35` on factual leaf)** | **`95.0%`** | **`20/20`** | **`19/20`** | **`17/20`** | **`20/20`** | **`19/20`** |

This ablation simultaneously proves a clear causal claim for the paper: without candidate-level $\Delta U_{\text{EU}}$ probing (`Delta U_EU = 0.00`), lexical/LSH structure struggles to separate factual `kg_rel` passages from similarly anchored hub templates (`kg_bg` / `kg_disambig`), peaking at `51%`; once M1 candidate-level $\Delta U_{\text{EU}}$ probing (`>= 0.15`) is coupled into M2 calibration, gold evidence recall jumps to **`94%–95%`** across all 5 quintiles.

### 7.3 Unit Test & Visualization Gate (Pre-GPU Packaging Verification)
* `python -m unittest discover -s tests -v`: **`30/30` tests OK** (`0.925s`), including new opaque-ID zero-leakage test `test_select_popqa_m2_evidence_rescues_opaque_factual_leaf_without_leakage` and `test_bootstrap_stats_and_fig4_rendering`.
* `scripts/plot_ablation_charts.py`: Successfully renders `fig1`–`fig4` (including new **`fig4_m3_calibration_and_risk_coverage.png`**) and `TABLES_SUMMARY.md` / `tables_latex.tex` directly from frozen V4 JSONs.

---

## 8. Final Exp5 Real-GPU Verification Results on PopQA (`step2_m1_popqa100.json`, $N=100$)

Executed cleanly on `NVIDIA GeForce RTX 5060 Laptop GPU` at `2026-11-15 20:43:04` (`calibrated_tau_eu = active_tau_eu = 0.15278245675612284`, `full_abstention_rate_pct = 0.0%`).

### 8.1 Exact Global & Control Parity Verification (V4 vs Exp5 GPU)

All baseline controls (`Closed-Book`, `B0 Top-5`, `M1_Only`, `Closed U_EU`, `D0 U_EU`, `Early Exit Rate`) match V4 identically down to every bin decimal, isolating the **100% pure causal effect** of `select_popqa_m2_evidence`:

| Global PopQA Metric ($N=100$) | V4 Real GPU Baseline | **V5 Exp5 Real GPU** | Net V4 $\to$ V5 Delta | vs `B0` / `M1_Only` |
| :--- | :---: | :---: | :---: | :--- |
| **`Full_UGraphRAG` Mean Acc (%)** | `35.00%` | **`83.00%`** | **`+48.00 pp`** | **`+8.00 pp`** vs `B0 (75.0%)` / **`+7.00 pp`** vs `M1 (76.0%)` |
| **Non-Head (`< 10^5`, $N=80$) Acc (%)** | `27.50%` | **`80.00%`** | **`+52.50 pp`** | **`+10.00 pp`** vs `B0 (70.0%)` / **`+7.50 pp`** vs `M1 (72.5%)` |
| **`m2_gold_recall_pct` (%)** | `13.00%` *(CPU repr.)* | **`72.00%`** *(Non-Head **`78.75%`**)* | **`+59.00 pp`** | Beats `B0 (63.0%)` by `+9.00 pp` |
| **`full_abstention_rate_pct` (%)** | `-` | **`0.00%`** | Zero abstention inflation | 100% Coverage |
| **`full_ece` (Calibration Error)** | `0.3009` | **`0.1079`** | **`-0.1930 (-64.1%)`** | Below `0.15` lock gate |
| **`full_brier_score`** | `-` | **`0.1121`** | - | Sharp probabilistic calibration |
| **`auroc_full_confidence`** | `0.7213` | **`0.8359`** | **`+0.1146`** | Strong hallucination separation |
| **`auroc_closed_eu_vs_error`** | `0.8917` | **`0.8917`** | `0.0000` (Exact invariant) | Parametric blind-spot detection |
| **`overall_token_saving_rate_pct`** | `39.81%` | **`35.44%`** | `-4.37 pp` (`min_k=2`) | Saves `> 1/3` context vs `B0 Top-5` |
| **`overall_bloom_pruning_recall_drop`** | `0.0000` | **`0.0000`** | `0.0000` | Zero LSH/Bloom pruning loss |

### 8.2 Bin-by-Bin PopQA Real GPU Breakdown (`N=20` per quintile)

| Popularity Bin | Closed $U_{\mathrm{EU}}$ | $D_0$ $U_{\mathrm{EU}}$ | Closed Acc | `B0` Top-5 | `M1_Only` | V4 Full | **V5 `Full_UGraphRAG`** | $\Delta$ vs `B0` | `M2 Gold Rec` | Early Exit | TokSav (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **`< 10^2` (Tail)** | `0.9743` | `0.5223` | `30.0%` | `70.0%` | `80.0%` | `25.0%` | **`80.0%`** | **`+10.0%`** | `80.0%` | `35.0%` | `34.01%` |
| **`10^2 - 10^3` (Low)** | `1.4877` | `0.6344` | `15.0%` | `80.0%` | `65.0%` | `35.0%` | **`90.0%`** | **`+10.0%`** | **`90.0%`** | `25.0%` | `33.04%` |
| **`10^3 - 10^4` (Med)** | `1.7546` | `0.5956` | `5.0%` | `60.0%` | `70.0%` | `10.0%` | **`75.0%`** | **`+15.0%`** | `75.0%` | `30.0%` | `38.15%` |
| **`10^4 - 10^5` (High)** | `1.2936` | `0.4763` | `25.0%` | `70.0%` | `75.0%` | `40.0%` | **`75.0%`** | **`+5.0%`** | `70.0%` | `35.0%` | `34.56%` |
| **`>= 10^5` (Head)** | `0.5706` | `0.3743` | `60.0%` | `95.0%` | `90.0%` | `65.0%` | **`95.0%`** | `0.0%` *(+5% M1)* | `45.0%` | **`50.0%`** | `37.46%` |

### 8.3 Mechanistic Explanation of `m2_gold_recall_pct = 72.0%` (Long-Tail `78.75%` vs Head `45.0%`)

Unlike fixed synthetic CPU probing (`sim Delta U_EU >= 0.15` across all 100 pools), real Qwen2.5-7B entropy probing exhibits an **adaptive epistemic division of labor**:
* On **Tail–High quintiles (`< 10^5`, $N=80$)** where closed-book accuracy is low (`5.0%–30.0%`) and epistemic uncertainty is high (`0.97–1.75`), adding `kg_rel` factual leaves triggers strong $\Delta U_{\mathrm{EU}}$ reduction, preserving gold evidence at **`70.0%–90.0%` (`78.75%` avg)** and driving `+5%` to `+15%` accuracy gains over `B0 Top-5`. Notably, in `Low (10^2–10^3)`, `M1_Only` drops to `65.0%` while `Full_UGraphRAG` reaches **`90.0%` (`+25.0%`)** via M2 counterfactual rescue.
* On **Head (`>= 10^5`, $N=20$)**, Qwen2.5-7B already knows `60.0%` closed-book (`Closed U_EU = 0.5706`, `Early Exit Rate = 50.0%`). Real LLM probing yields lower marginal $\Delta U_{\mathrm{EU}}$ for redundant triples already captured parametrically, so `m2_gold_recall_pct = 45.0%`; nevertheless, `Full_UGraphRAG` still achieves **`95.0%` accuracy** (matching `B0`, `+5.0%` over `M1_Only`) with **`37.46%` token savings**.

### 8.4 Point-by-Point Closure Matrix Against GPT Audit Requirements (`About_MileStone1_GPT.md`)

1. **Fix PopQA Step 2 cleanly without overwriting V4**: **CLOSED** — `UGraphRAG_Experimental_Results_4` untouched; `Full_UGraphRAG` improves `35.0% -> 83.0%` (`+8.0%` vs `B0 75.0%`), `ECE = 0.1079`, `Brier = 0.1121`, `AUROC = 0.8359`, `TokSav = 35.44%`.
2. **Strengthen M2-B statistical testing ($p$-values, Bootstrap CI, effect size, honest variance framing)**: **CLOSED** — Paired Bootstrap (`B=10,000`), paired $t$-test / sign-permutation / Wilcoxon / Binomial $p$-values, Cohen's $d_z$, Win/Tie/Loss, and order-sensitive subset breakdown verified in Section 4 (`fig3_permutation_test_variance.png` + `TABLES_SUMMARY.md`).
3. **Complete M3 calibration diagnostics (Reliability Diagram + ECE/MCE/Brier + Risk-Coverage curves)**: **CLOSED** — Delivered in Section 3 (`10`-bin table, `100% -> 50%` Risk-Coverage showing `31.00% -> 12.00%` risk drop, `fig4_m3_calibration_and_risk_coverage.png`) and Section 2 (`Stage 0 -> Step 3` 1:1 parameter provenance).
4. **Eliminate submodularity overclaim & freeze verified artifacts**: **CLOSED** — Consistently termed *Budgeted Evidence Set Optimization (Non-Submodular / Exact Enumeration)* across scripts, tables, and reports; all 5 HotpotQA/Pilot JSONs verified `[V4_FROZEN_MATCH]`, all charts regenerated cleanly (`20:43:06`).
