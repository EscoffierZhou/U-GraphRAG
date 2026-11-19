# MileStone_After: SOTA Horse-Race & Mechanism Extension Tables

> Model: `qwen7b` | Scale: `medium` | Quantization: `4bit` | M1 tau_eu: `0.0749` | Conformal tau (alpha=0.15): `0.3995`

## S4 All-Irrelevant M1 Relevance Gate Audit
- Un-gated `V5 Proxy` Early-Exit Rate on S4: `3.33%`
- With `RelevanceLowerBoundGate`: `0.0%` (blocked **`3.33 pp`** of blind off-topic early exits)

## Cross-Method Horse-Race Comparison

| Benchmark | Method | Raw EM (%) | Raw F1 (%) | Eff F1 (%) | ConfRet (%) | AUROC | ECE | AURC | SelAcc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ramdocs | B0_StandardRAG | 57.50 | 65.59 | 65.59 | 38.00 | 0.7274 | 0.2734 | 0.2176 | 65.59 |
| ramdocs | BGE_Reranker_Top5 | 55.00 | 63.44 | 63.44 | 50.00 | 0.7462 | 0.2858 | 0.2215 | 63.44 |
| ramdocs | BGE_Reranker_MMR | 56.00 | 63.94 | 63.94 | 49.50 | 0.7173 | 0.2816 | 0.2439 | 63.94 |
| ramdocs | CK_PLUG | 53.00 | 63.50 | 63.50 | 50.50 | 0.7559 | 0.3128 | 0.2114 | 63.50 |
| ramdocs | InstructRAG_ICL | 2.00 | 9.91 | 9.91 | 50.50 | 0.6714 | 0.7198 | 0.8757 | 9.91 |
| ramdocs | CAD_a05 | 39.00 | 49.87 | 49.87 | 50.50 | 0.8341 | 0.4281 | 0.2588 | 49.87 |
| ramdocs | U_GraphRAG_Plus | 63.00 | 69.40 | 58.19 | 37.50 | 0.7084 | 0.1198 | 0.2049 | 75.58 |
| nqswap_dual | B0_StandardRAG | 6.00 | 9.33 | 9.33 | 100.00 | 0.4399 | 0.8013 | 0.9247 | 9.33 |
| nqswap_dual | BGE_Reranker_Top5 | 5.00 | 7.44 | 7.44 | 100.00 | 0.4408 | 0.8149 | 0.9532 | 7.44 |
| nqswap_dual | BGE_Reranker_MMR | 5.50 | 8.18 | 8.18 | 100.00 | 0.4608 | 0.7989 | 0.9429 | 8.18 |
| nqswap_dual | CK_PLUG | 7.50 | 11.46 | 11.46 | 100.00 | 0.4121 | 0.8333 | 0.9048 | 11.46 |
| nqswap_dual | InstructRAG_ICL | 1.50 | 4.90 | 4.90 | 100.00 | 0.7919 | 0.7590 | 0.9319 | 4.90 |
| nqswap_dual | CAD_a05 | 4.50 | 13.53 | 13.53 | 100.00 | 0.5761 | 0.7775 | 0.8259 | 13.53 |
| nqswap_dual | U_GraphRAG_Plus | 39.00 | 42.09 | 35.78 | 48.50 | 0.5913 | 0.1704 | 0.5026 | 48.35 |
| hotpotqa_conflict | B0_StandardRAG | 39.17 | 50.31 | 50.31 | 100.00 | 0.7598 | 0.3938 | 0.3399 | 50.31 |
| hotpotqa_conflict | BGE_Reranker_Top5 | 44.17 | 59.53 | 59.53 | 95.83 | 0.7534 | 0.3488 | 0.2535 | 59.53 |
| hotpotqa_conflict | BGE_Reranker_MMR | 40.00 | 57.23 | 57.23 | 72.50 | 0.7669 | 0.3607 | 0.2519 | 57.23 |
| hotpotqa_conflict | CK_PLUG | 33.33 | 51.06 | 51.06 | 100.00 | 0.7771 | 0.4257 | 0.3030 | 51.06 |
| hotpotqa_conflict | InstructRAG_ICL | 5.83 | 13.42 | 13.42 | 100.00 | 0.6068 | 0.6900 | 0.7935 | 13.42 |
| hotpotqa_conflict | CAD_a05 | 24.17 | 41.84 | 41.84 | 100.00 | 0.7332 | 0.5092 | 0.4205 | 41.84 |
| hotpotqa_conflict | U_GraphRAG_Plus | 44.17 | 61.54 | 49.79 | 9.17 | 0.6869 | 0.1094 | 0.2690 | 66.39 |
| hotpotqa_s4_irrelevant | B0_StandardRAG | 3.33 | 3.40 | 3.40 | 0.00 | 0.4784 | 0.6007 | 0.9723 | 3.40 |
| hotpotqa_s4_irrelevant | BGE_Reranker_Top5 | 3.33 | 3.89 | 3.89 | 0.00 | 0.5965 | 0.5946 | 0.9392 | 3.89 |
| hotpotqa_s4_irrelevant | BGE_Reranker_MMR | 3.33 | 3.89 | 3.89 | 0.00 | 0.5948 | 0.5975 | 0.9335 | 3.89 |
| hotpotqa_s4_irrelevant | CK_PLUG | 7.50 | 12.28 | 12.28 | 0.00 | 0.7521 | 0.6691 | 0.7870 | 12.28 |
| hotpotqa_s4_irrelevant | InstructRAG_ICL | 0.00 | 0.83 | 0.83 | 0.00 | 0.5000 | 0.7749 | 0.9894 | 0.83 |
| hotpotqa_s4_irrelevant | CAD_a05 | 0.83 | 1.78 | 1.78 | 0.00 | 0.9748 | 0.8167 | 0.9599 | 1.78 |
| hotpotqa_s4_irrelevant | U_GraphRAG_Plus | 2.50 | 3.98 | 0.00 | 0.00 | 0.5878 | 0.1866 | 0.9507 | 0.00 |
| hotpotqa_clean | B0_StandardRAG | 36.67 | 51.47 | 51.47 | 0.00 | 0.6765 | 0.3962 | 0.3310 | 51.47 |
| hotpotqa_clean | BGE_Reranker_Top5 | 40.00 | 55.69 | 55.69 | 0.00 | 0.7663 | 0.3785 | 0.2610 | 55.69 |
| hotpotqa_clean | BGE_Reranker_MMR | 43.33 | 59.08 | 59.08 | 0.00 | 0.7661 | 0.3526 | 0.2396 | 59.08 |
| hotpotqa_clean | CK_PLUG | 30.83 | 49.68 | 49.68 | 0.00 | 0.7371 | 0.4475 | 0.3418 | 49.68 |
| hotpotqa_clean | InstructRAG_ICL | 7.50 | 17.41 | 17.41 | 0.00 | 0.6631 | 0.6723 | 0.7708 | 17.41 |
| hotpotqa_clean | CAD_a05 | 22.50 | 37.58 | 37.58 | 0.00 | 0.7683 | 0.5367 | 0.4202 | 37.58 |
| hotpotqa_clean | U_GraphRAG_Plus | 42.50 | 58.65 | 49.79 | 0.00 | 0.6687 | 0.0846 | 0.2948 | 62.24 |
