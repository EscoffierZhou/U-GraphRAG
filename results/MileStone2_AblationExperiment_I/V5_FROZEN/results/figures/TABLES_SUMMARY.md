# U-GraphRAG Experimental Benchmark Summary Tables

## Table 1: Baseline Benchmark (HotpotQA-100 on NVIDIA GeForce RTX 5060 Laptop GPU)
| Model Backbone | Precision | EM (%) | F1 (%) | Avg Latency (s) | Peak Alloc / CUDA Res. (MB) | Safety Headroom (MB) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Qwen2.5-3B-Instruct | 4-bit NF4 | 41.00% | 53.75% | 0.86s | 1978.6 / 5306.0 MB | +2886.0 MB |
| Qwen2.5-7B-Instruct | 4-bit NF4 | 54.00% | 64.68% | 0.99s | 5310.8 / 5660.0 MB | +2532.0 MB |

## Table 2: HotpotQA 8-Stage Ablation Progression
| Pipeline Mode | EM (%) | F1 (%) | Sp-P (%) | Sp-R (%) | Sp-F1 (%) | Token Saving (%) | ECE | AUROC | Cov. (%) | Sel. Risk (%) | Latency (s) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| B0: Standard RAG | 54.0% | 64.7% | 36.0% | 88.5% | 51.0% | 0.0% | 0.071 | 0.540 | 100.0% | 32.00% | 1.00s |
| B1: Naive MST (KG2RAG Control) | 43.0% | 52.9% | 26.8% | 65.5% | 37.9% | 7.6% | 0.100 | 0.707 | 100.0% | 45.00% | 1.01s |
| M1 Only (Uncertainty Adapt) | 54.0% | 64.9% | 50.9% | 82.0% | 62.3% | 33.6% | 0.142 | 0.676 | 100.0% | 32.00% | 1.29s |
| M2 Greedy (M1 + Greedy + MST) | 55.0% | 67.1% | 39.5% | 86.0% | 53.4% | 10.1% | 0.154 | 0.521 | 100.0% | 30.00% | 1.01s |
| M2 Exact w/o MST (Exact Selection) | 54.0% | 66.4% | 39.5% | 86.0% | 53.4% | 10.1% | 0.164 | 0.552 | 100.0% | 29.00% | 1.01s |
| M2 Full (Exact + MST DFS) | 55.0% | 67.1% | 39.5% | 86.0% | 53.4% | 10.1% | 0.154 | 0.521 | 100.0% | 30.00% | 1.02s |
| M2 Permutation (Mean over Orders) | 55.0% | 67.1% | 39.5% | 86.0% | 53.4% | 10.1% | 0.150 | 0.500 | 100.0% | 30.00% | 5.08s |
| Full U-GraphRAG (M1+M2+M3 Dec) | 54.0% | 65.7% | 39.5% | 86.0% | 53.4% | 10.1% | 0.092 | 0.757 | 96.0% | 28.12% | 1.05s |

## Table 3: M2 Context Ordering & Permutation Sensitivity (Lost-in-the-Middle)
| Ordering Strategy | Mean F1 (%) | Intra-Seed Std | Cross-Sample Std | Delta vs DFS (%) |
| :--- | :---: | :---: | :---: | :---: |
| DFS Topological (Ours) | 67.14% | 0.00 | 42.90 | 0.00% |
| Reverse Order | 66.66% | 0.00 | 41.93 | -0.48% |
| 5x Random Shuffles | 67.00% | 10.01 | 37.96 | -0.14% |

### Table 3b: Paired Bootstrap 95% CI & Order-Sensitive Subset Breakdown (N=100, B=10000)
| Comparison | Mean Delta F1 (%) | Paired Std (%) | 95% Bootstrap CI (%) | Win / Tie / Loss | Sensitive Subset Mean Gain |
| :--- | :---: | :---: | :---: | :---: | :---: |
| DFS vs Reverse | +0.48% | ±32.08% | [-5.94%, +6.72%] | 11 / 82 / 7 | +2.66% (n=18) |
| DFS vs 5x Shuffles | +0.14% | ±22.58% | [-4.46%, +4.34%] | 21 / 67 / 12 | +0.42% (n=33) |

## Table 4: Hardware Memory Footprint (Peak CUDA Allocated VRAM)
| Pipeline Mode | Peak Alloc (MB) | Forward Calls | Avg Time (s) | Token Saving (%) |
| :--- | :---: | :---: | :---: | :---: |
| B0: Standard RAG | 5667.5 MB | 1.00 | 1.00s | 0.0% |
| B1: Naive MST (KG2RAG Control) | 5677.4 MB | 1.00 | 1.01s | 7.6% |
| M1 Only (Uncertainty Adapt) | 5663.3 MB | 1.00 | 1.29s | 33.6% |
| M2 Greedy (M1 + Greedy + MST) | 5667.0 MB | 1.00 | 1.01s | 10.1% |
| M2 Exact w/o MST (Exact Selection) | 5667.0 MB | 1.00 | 1.01s | 10.1% |
| M2 Full (Exact + MST DFS) | 5667.0 MB | 1.00 | 1.02s | 10.1% |
| M2 Permutation (Mean over Orders) | 5671.0 MB | 7.00 | 5.08s | 10.1% |
| Full U-GraphRAG (M1+M2+M3 Dec) | 5667.0 MB | 1.00 | 1.05s | 10.1% |

## Table 5: M3 Selective Prediction Risk-Coverage Progression (Full_UGraphRAG vs B0 vs M2_Full)
| Coverage (%) | N | Full Risk (%) | Full EM (%) | Full F1 (%) | B0 Risk (%) | M2_Full Risk (%) |
| :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| 100.0% | 100 | **31.00%** | 54.00% | 65.74% | 32.00% | 30.00% |
| 96.0% | 96 | **28.12%** | 56.25% | 68.48% | 31.25% | 31.25% |
| 90.0% | 90 | **25.56%** | 57.78% | 70.82% | 32.22% | 31.11% |
| 80.0% | 80 | **22.50%** | 61.25% | 74.15% | 30.00% | 32.50% |
| 70.0% | 70 | **21.43%** | 60.00% | 73.60% | 28.57% | 32.86% |
| 60.0% | 60 | **18.33%** | 61.67% | 76.70% | 31.67% | 31.67% |
| 50.0% | 50 | **12.00%** | 70.00% | 83.01% | 36.00% | 30.00% |

## Table 6: Step 2 PopQA Long-Tail Uncertainty & End-to-End Evaluation (N=100, Exp5 Real GPU)
`auroc_closed_eu_vs_error = 0.8917` | `auroc_full_confidence = 0.8359` | `full_ece = 0.1079` | `full_brier_score = 0.1121` | `full_abstention = 0.0%` | `tok_sav = 35.44%`

| Popularity Bin (N=20) | Closed EU | D0 EU | Closed Acc (%) | B0 Top-5 (%) | M1 Only (%) | V4 Full (%) | V5 Full U-GraphRAG (%) | M2 Gold Recall (%) | Early Exit (%) | Token Saving (%) |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `< 10^2 (Tail)` | 0.9743 | 0.5223 | 30.00% | 70.00% | 80.00% | 25.00% | **80.00%** (+10.0%) | 80.00% | 35.00% | 34.01% |
| `10^2 - 10^3 (Low)` | 1.4877 | 0.6344 | 15.00% | 80.00% | 65.00% | 35.00% | **90.00%** (+10.0%) | 90.00% | 25.00% | 33.04% |
| `10^3 - 10^4 (Medium)` | 1.7546 | 0.5956 | 5.00% | 60.00% | 70.00% | 10.00% | **75.00%** (+15.0%) | 75.00% | 30.00% | 38.15% |
| `10^4 - 10^5 (High)` | 1.2936 | 0.4763 | 25.00% | 70.00% | 75.00% | 40.00% | **75.00%** (+5.0%) | 70.00% | 35.00% | 34.56% |
| `>= 10^5 (Head)` | 0.5706 | 0.3743 | 60.00% | 95.00% | 90.00% | 65.00% | **95.00%** (0.0%) | 45.00% | 50.00% | 37.46% |
| **Overall Mean (N=100)** | **1.2162** | **0.5206** | **27.00%** | **75.00%** | **76.00%** | **35.00%** | **83.00% (+8.0%)** | **72.00%** | **35.00%** | **35.44%** |
