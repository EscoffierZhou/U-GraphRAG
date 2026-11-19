# V4 Audit Critique & V5 Surgical Option-A Fix Log

> **Target Revision**: U-GraphRAG V4 (`UGraphRAG_Experimental_Results_4`) $\to$ **V5 (`实验包5.zip`)**  
> **Strategy (`Option A`)**: Preserve `src/` 100% frozen to guarantee zero drift on already-validated **HotpotQA `Stage 0`, `Step 1`, `Step 3`, and `Step 4`** GPU results, while executing a surgical fix inside `scripts/run_overnight_ablation.py` for **Step 2 (`PopQA`, $N=100$)** alongside zero-GPU statistical & plotting upgrades in `scripts/plot_ablation_charts.py` and `scripts/run_full_experiment_and_package.py`.

---

## 1. Summary of V4 Audit Findings

### 1.1 Confirmed Strong Results (Frozen in V4 & Pre-Seeded in V5)
- **Stage 0 (`HotpotQA [100:180]`, $N=80$, `736.46s`)**: Non-leaking dev grid search escaped V3 `offset=0.34` collapse and locked **`balanced_optimal`** (`m1_tau_eu=0.18, m2_gamma_1=0.8, m2_gamma_2=0.2, m2_lambda_div=0.4, offset=0.22, m2_min_docs=2, top_k_pool=8, platt_a=0.8174, platt_b=-0.2053, T_conf=2.0, w_tok=0.6, tau_conf=0.38, tau_abs=0.05`, Dev `EM=48.75%`, `F1=61.36%`, `ECE=0.0670`).
- **Step 1 (`RTX 5060 8GB`, $N=100$)**: Zero-error NF4+`bfloat16` execution (`7B` HotpotQA `EM=54.00% / F1=64.68%` vs `3B` `41.00% / 53.75%`; `7B` PopQA Closed `27.00% / 33.89%`; Peak Allocated `5310.8 MB`, Final Reserved `5660.0 MB`, Headroom `+2532.0 MB`).
- **Step 3 (`HotpotQA-100`, 8 Modes)**:
  - `M2_Full` reaches **`EM=55.00% / F1=67.14%`** (`+12.00% EM / +14.26% F1` over `B1_Naive_MST`, `+1.00% EM / +2.46% F1` over `B0`, `10.1%` token saving).
  - Removing M2-B MST-DFS (`M2_Exact_NoMST`) drops to `EM=54.00% / F1=66.40%`.
  - `M1_Only` lifts Supporting-Fact Precision from `36.00% -> 50.87%` (`Sp-F1 51.00% -> 62.26%`) with `33.6%` token saving.
  - `Full_UGraphRAG` improves **`AUROC` (`0.5214 -> 0.7574`)**, **`Brier` (`0.2354 -> 0.1814`)**, **`ECE` (`0.1544 -> 0.0925`)**, and cuts Selective Risk monotonically from **`31.00%` (`100%` cov) $\to$ `28.12%` (`96%` cov) $\to$ `12.00%` (`50%` cov, `EM=70.00%`, `F1=83.01%`)**.
- **Step 4 (`HotpotQA-100` Ordering Permutation)**:
  - `DFS (67.14%) > Random Shuffles (67.00%) > Reverse (66.66%)`;
  - Eliminates `±10.01%` within-question shuffle volatility (`max_drop = 14.06%`) and delivers consistent net wins on order-sensitive subsets (`11/82/7` vs Reverse, `+2.66%` on $n=18$; `21/67/12` vs Shuffles, `+0.42%` on $n=33$), with 10,000-resample Paired Bootstrap 95% CIs (`[-5.94%, +6.72%]` and `[-4.46%, +4.34%]`). Step 4 GPU is therefore frozen.

### 1.2 Step 2 (`PopQA-100`) Failure Root-Cause Diagnosis
While `Closed U_EU AUROC = 0.8917`, `bloom_pruning_recall_drop = 0.0`, `B0_Acc = 75.0%`, and `M1_Only_Acc = 76.0%` (`39.81%` token saving) succeeded, `Full_UGraphRAG` collapsed to `35.0%` (`full_ece = 0.3009`). Replaying V4 code over all 100 exact cached PopQA pools proved:
1. Retrieval/LSH expansion succeeded (`D0=6%`, `B0=63%`, `M1_Exp_Top5=85%`, `pool_cap8=100%`), yet V4 `M2-A` dropped gold `kg_rel` retention to **`13%`**:
   - **Bug A (Score Clobbering)**: `exp_res.expanded_candidates` came first with dummy corpus `"score": 0.65`, masking true dense+BM25 rank scores and burying LSH structural bridges;
   - **Bug B (Zero $\Delta U_{\text{EU}}$ Signal)**: `run_experiment_2` left `epistemic_reduction = 0.0` and `aleatoric_uncertainty = 0.0` for all candidates (unlike HotpotQA's `u_graphrag_pipeline.py`), reducing reliability calibration to raw score minmax where redundant hubs tied with factual leaves;
   - **Bug C (`min_k=1` + Anchor Redundancy)**: `kg_rel` shared its canonical anchor prefix with `kg_bg`/`kg_disambig`, causing `M2-A` (`min_k=1, offset=0.22`) to prune `kg_rel`;
   - **Bug D (`tau_eu` Mismatch)**: Using HotpotQA's `0.18` instead of PopQA dev `calib_tau_eu = 0.1528`.

---

## 2. Implemented V5 Surgical Fixes

1. **`scripts/run_overnight_ablation.py`**:
   - Added zero-leakage top-level helper **`select_popqa_m2_evidence(...)`**:
     - Preserves `ranked_pool` rank scores and promotes genuine non-$D_0$ LSH structural bridges (`graph_similarity >= 0.18`) via `0.74 + 0.16 * clip(gsim, 0, 1)`;
     - Computes candidate-level marginal epistemic reduction $\Delta U_{\text{EU}}$ (leave-one-out on $D_0$, counterfactual add-one on non-$D_0$) and $U_{\text{AU}}$ using `decoupler.decompose(..., model_fn=model_fn_probe)`;
     - Adds gentle utility boost (`score + 0.35 * er_val`) prior to `minmax` calibration, keeps `explicit_edges=None`, enforces `min_k=min(m2_min_docs, len(cands))` (`2`), and restores pre-offset `calibrated_reliability` before M3 decoding;
     - Aligns `active_tau_eu = min(gp["m1_tau_eu"], calib_output["tau_eu"])` when `calib_output["tau_eu"] > 0.05` (`0.1528`);
     - Logs post-hoc non-leaking telemetry in `step2_m1_popqa100.json`: `m2_gold_recall_pct`, `full_abstention_rate_pct`, `full_brier_score`.
   - Verified over all 100 cached PopQA pools on CPU (`sim_er=0.00 -> 51%` vs V4 `13%`; `sim_er=0.15..0.35 -> 94%~95%`).
2. **`scripts/plot_ablation_charts.py`**:
   - Added **`plot_fig4_m3_calibration_and_risk_coverage`** generating **`fig4_m3_calibration_and_risk_coverage.png`** directly from per-sample records in `step3_step4_ablation_hotpot100.json`;
   - Added **`_compute_paired_bootstrap_stats`** (`10,000` resamples, `seed=42`, Win/Tie/Loss, sensitive subset mean gain) cleanly skipping summary-only mocks;
   - Added Table 1 `Peak Alloc / CUDA Res. (MB)`, Table 4 `Peak Alloc (MB)` clarification, Table 3b Bootstrap, and Table 5 Risk-Coverage progression in `TABLES_SUMMARY.md` & `tables_latex.tex`.
3. **`scripts/run_full_experiment_and_package.py` & `tests/test_v4_fixes.py`**:
   - Synced Stage 0 $\to$ Step 3 provenance, Risk-Coverage table, Bootstrap CIs, accurate non-submodular terminology (`预算约束集合优化`), VRAM reconciliation, and `FOR_GPT_VERIFICATION_EVIDENCE.md` / `fig4` asset links;
   - Added `TestV5PopQAM2FixAndStatisticalReporting` (**`30/30` tests passing**);
   - Updated `package_results_to_zip` to skip nested `UGraphRAG_Experimental_Results*` folders cleanly.
4. **Documentation**: Updated `results/MileStone_1.md` and authored `results/FOR_GPT_VERIFICATION_EVIDENCE.md`.

---

## 3. RTX 5060 V5 Execution Command

Run Step 2 (`PopQA-100`) and automatically regenerate all publication figures (`fig1`–`fig4`) and tables using the pre-seeded frozen V4 JSONs:

```powershell
python -u scripts/run_overnight_ablation.py --params-file results/grid_search_results.json --params-preset balanced_optimal --skip-exp1 --skip-exp3
```
