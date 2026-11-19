# MileStone_After: SOTA Horse-Race & Mechanism Extension Tables

> Model: `qwen3b` | Scale: `medium` | Quantization: `4bit` | M1 tau_eu: `0.2015` | Conformal tau (alpha=0.15): `0.3959`

## S4 All-Irrelevant M1 Relevance Gate Audit
- Un-gated `V5 Proxy` Early-Exit Rate on S4: `85.83%`
- With `RelevanceLowerBoundGate`: `0.0%` (blocked **`85.83 pp`** of blind off-topic early exits)

## Cross-Method Horse-Race Comparison

| Benchmark | Method | Raw EM (%) | Raw F1 (%) | Eff F1 (%) | ConfRet (%) | AUROC | ECE | AURC | SelAcc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ramdocs | B0_StandardRAG | 54.50 | 60.75 | 60.75 | 38.00 | 0.7271 | 0.2918 | 0.2400 | 60.75 |
| ramdocs | BGE_Reranker_Top5 | 57.00 | 64.91 | 64.91 | 50.00 | 0.7304 | 0.2552 | 0.1880 | 64.91 |
| ramdocs | BGE_Reranker_MMR | 57.50 | 65.14 | 65.14 | 49.50 | 0.7185 | 0.2555 | 0.2010 | 65.14 |
| ramdocs | CK_PLUG | 52.00 | 61.83 | 61.83 | 50.50 | 0.7126 | 0.3104 | 0.2146 | 61.83 |
| ramdocs | InstructRAG_ICL | 0.50 | 7.32 | 7.32 | 50.50 | 0.5973 | 0.7046 | 0.9244 | 7.32 |
| ramdocs | CAD_a05 | 28.00 | 44.58 | 44.58 | 50.50 | 0.7657 | 0.4474 | 0.3477 | 44.58 |
| ramdocs | U_GraphRAG_Plus | 53.00 | 60.07 | 54.24 | 37.00 | 0.6451 | 0.1093 | 0.2900 | 63.81 |
| nqswap_dual | B0_StandardRAG | 12.50 | 14.18 | 14.18 | 100.00 | 0.4383 | 0.7847 | 0.8722 | 14.18 |
| nqswap_dual | BGE_Reranker_Top5 | 17.00 | 18.48 | 18.48 | 100.00 | 0.4100 | 0.7233 | 0.8612 | 18.48 |
| nqswap_dual | BGE_Reranker_MMR | 18.00 | 20.33 | 20.33 | 100.00 | 0.4002 | 0.7014 | 0.8529 | 20.33 |
| nqswap_dual | CK_PLUG | 17.00 | 18.86 | 18.86 | 100.00 | 0.4034 | 0.7661 | 0.8625 | 18.86 |
| nqswap_dual | InstructRAG_ICL | 1.00 | 5.18 | 5.18 | 100.00 | 0.8720 | 0.7440 | 0.9120 | 5.18 |
| nqswap_dual | CAD_a05 | 3.00 | 13.85 | 13.85 | 100.00 | 0.4901 | 0.7307 | 0.8511 | 13.85 |
| nqswap_dual | U_GraphRAG_Plus | 34.00 | 37.09 | 31.71 | 53.00 | 0.6312 | 0.2260 | 0.5591 | 42.00 |
| hotpotqa_conflict | B0_StandardRAG | 30.83 | 42.31 | 42.31 | 100.00 | 0.7201 | 0.4218 | 0.4275 | 42.31 |
| hotpotqa_conflict | BGE_Reranker_Top5 | 38.33 | 50.79 | 50.79 | 95.83 | 0.7305 | 0.3814 | 0.3318 | 50.79 |
| hotpotqa_conflict | BGE_Reranker_MMR | 38.33 | 52.78 | 52.78 | 72.50 | 0.7176 | 0.3466 | 0.3242 | 52.78 |
| hotpotqa_conflict | CK_PLUG | 35.00 | 48.20 | 48.20 | 100.00 | 0.7371 | 0.4604 | 0.3583 | 48.20 |
| hotpotqa_conflict | InstructRAG_ICL | 4.17 | 11.16 | 11.16 | 100.00 | 0.8048 | 0.6751 | 0.7941 | 11.16 |
| hotpotqa_conflict | CAD_a05 | 10.00 | 27.74 | 27.74 | 100.00 | 0.6895 | 0.5675 | 0.5780 | 27.74 |
| hotpotqa_conflict | U_GraphRAG_Plus | 38.33 | 50.31 | 48.36 | 9.17 | 0.6833 | 0.0914 | 0.3813 | 59.22 |
| hotpotqa_s4_irrelevant | B0_StandardRAG | 1.67 | 2.33 | 2.33 | 0.00 | 0.3020 | 0.8736 | 0.9854 | 2.33 |
| hotpotqa_s4_irrelevant | BGE_Reranker_Top5 | 2.50 | 4.32 | 4.32 | 0.00 | 0.2339 | 0.8003 | 0.9794 | 4.32 |
| hotpotqa_s4_irrelevant | BGE_Reranker_MMR | 2.50 | 4.32 | 4.32 | 0.00 | 0.2456 | 0.7968 | 0.9783 | 4.32 |
| hotpotqa_s4_irrelevant | CK_PLUG | 5.00 | 10.01 | 10.01 | 0.00 | 0.5830 | 0.7673 | 0.8369 | 10.01 |
| hotpotqa_s4_irrelevant | InstructRAG_ICL | 0.00 | 2.62 | 2.62 | 0.00 | 0.5546 | 0.8175 | 0.9624 | 2.62 |
| hotpotqa_s4_irrelevant | CAD_a05 | 0.83 | 1.87 | 1.87 | 0.00 | 0.8403 | 0.9077 | 0.9765 | 1.87 |
| hotpotqa_s4_irrelevant | U_GraphRAG_Plus | 1.67 | 2.33 | 0.00 | 0.00 | 0.5641 | 0.3465 | 0.9694 | 0.00 |
| hotpotqa_clean | B0_StandardRAG | 35.00 | 46.61 | 46.61 | 0.00 | 0.7110 | 0.4025 | 0.3879 | 46.61 |
| hotpotqa_clean | BGE_Reranker_Top5 | 43.33 | 57.38 | 57.38 | 0.00 | 0.7128 | 0.3199 | 0.2841 | 57.38 |
| hotpotqa_clean | BGE_Reranker_MMR | 40.83 | 54.98 | 54.98 | 0.00 | 0.7611 | 0.3306 | 0.2869 | 54.98 |
| hotpotqa_clean | CK_PLUG | 39.17 | 53.66 | 53.66 | 0.00 | 0.7893 | 0.3970 | 0.3008 | 53.66 |
| hotpotqa_clean | InstructRAG_ICL | 0.83 | 8.03 | 8.03 | 0.00 | 0.7078 | 0.7221 | 0.8685 | 8.03 |
| hotpotqa_clean | CAD_a05 | 11.67 | 32.98 | 32.98 | 0.00 | 0.6892 | 0.5299 | 0.5402 | 32.98 |
| hotpotqa_clean | U_GraphRAG_Plus | 36.67 | 49.58 | 47.08 | 0.00 | 0.6657 | 0.1261 | 0.3980 | 55.39 |
