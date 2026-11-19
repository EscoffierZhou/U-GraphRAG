# MileStone_After: SOTA Horse-Race & Mechanism Extension Tables

> Model: `llama3_8b` | Scale: `medium` | Quantization: `4bit` | M1 tau_eu: `0.6281` | Conformal tau (alpha=0.15): `0.3934`

## S4 All-Irrelevant M1 Relevance Gate Audit
- Un-gated `V5 Proxy` Early-Exit Rate on S4: `9.17%`
- With `RelevanceLowerBoundGate`: `0.0%` (blocked **`9.17 pp`** of blind off-topic early exits)

## Cross-Method Horse-Race Comparison

| Benchmark | Method | Raw EM (%) | Raw F1 (%) | Eff F1 (%) | ConfRet (%) | AUROC | ECE | AURC | SelAcc (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| ramdocs | B0_StandardRAG | 47.50 | 59.33 | 59.33 | 38.00 | 0.6424 | 0.1439 | 0.2977 | 59.33 |
| ramdocs | BGE_Reranker_Top5 | 45.00 | 59.26 | 59.26 | 50.00 | 0.6524 | 0.1280 | 0.2987 | 59.26 |
| ramdocs | BGE_Reranker_MMR | 45.00 | 59.82 | 59.82 | 49.50 | 0.6931 | 0.1151 | 0.2775 | 59.82 |
| ramdocs | CK_PLUG | 46.00 | 58.61 | 58.61 | 50.50 | 0.6922 | 0.2295 | 0.2684 | 58.61 |
| ramdocs | InstructRAG_ICL | 14.50 | 24.96 | 24.96 | 50.50 | 0.3787 | 0.5433 | 0.7775 | 24.96 |
| ramdocs | CAD_a05 | 33.50 | 52.32 | 52.32 | 50.50 | 0.6389 | 0.2976 | 0.3498 | 52.32 |
| ramdocs | U_GraphRAG_Plus | 47.00 | 60.07 | 51.21 | 36.50 | 0.6324 | 0.0679 | 0.2843 | 60.97 |
| nqswap_dual | B0_StandardRAG | 9.50 | 15.88 | 15.88 | 100.00 | 0.5551 | 0.4961 | 0.8262 | 15.88 |
| nqswap_dual | BGE_Reranker_Top5 | 6.50 | 12.26 | 12.26 | 100.00 | 0.5125 | 0.5403 | 0.8781 | 12.26 |
| nqswap_dual | BGE_Reranker_MMR | 7.50 | 12.88 | 12.88 | 100.00 | 0.5267 | 0.5347 | 0.8621 | 12.88 |
| nqswap_dual | CK_PLUG | 14.00 | 21.31 | 21.31 | 100.00 | 0.5598 | 0.6202 | 0.7474 | 21.31 |
| nqswap_dual | InstructRAG_ICL | 1.50 | 2.91 | 2.91 | 100.00 | 0.2143 | 0.7612 | 0.9841 | 2.91 |
| nqswap_dual | CAD_a05 | 18.50 | 29.79 | 29.79 | 100.00 | 0.7846 | 0.4602 | 0.4949 | 29.79 |
| nqswap_dual | U_GraphRAG_Plus | 31.50 | 37.12 | 30.84 | 51.00 | 0.5969 | 0.1973 | 0.5483 | 39.79 |
| hotpotqa_conflict | B0_StandardRAG | 25.83 | 37.01 | 37.01 | 100.00 | 0.7153 | 0.1829 | 0.4394 | 37.01 |
| hotpotqa_conflict | BGE_Reranker_Top5 | 31.67 | 46.22 | 46.22 | 95.83 | 0.6874 | 0.1364 | 0.3603 | 46.22 |
| hotpotqa_conflict | BGE_Reranker_MMR | 38.33 | 51.77 | 51.77 | 72.50 | 0.6491 | 0.1190 | 0.3301 | 51.77 |
| hotpotqa_conflict | CK_PLUG | 30.00 | 45.51 | 45.51 | 100.00 | 0.6997 | 0.3264 | 0.4002 | 45.51 |
| hotpotqa_conflict | InstructRAG_ICL | 7.50 | 17.51 | 17.51 | 100.00 | 0.6599 | 0.6034 | 0.7235 | 17.51 |
| hotpotqa_conflict | CAD_a05 | 19.17 | 38.36 | 38.36 | 100.00 | 0.7476 | 0.3649 | 0.4301 | 38.36 |
| hotpotqa_conflict | U_GraphRAG_Plus | 32.50 | 49.42 | 35.96 | 10.00 | 0.6916 | 0.1193 | 0.3532 | 52.62 |
| hotpotqa_s4_irrelevant | B0_StandardRAG | 7.50 | 11.38 | 11.38 | 0.00 | 0.6105 | 0.2335 | 0.8175 | 11.38 |
| hotpotqa_s4_irrelevant | BGE_Reranker_Top5 | 5.83 | 9.97 | 9.97 | 0.00 | 0.5895 | 0.2682 | 0.8783 | 9.97 |
| hotpotqa_s4_irrelevant | BGE_Reranker_MMR | 5.00 | 8.52 | 8.52 | 0.00 | 0.5713 | 0.2881 | 0.8977 | 8.52 |
| hotpotqa_s4_irrelevant | CK_PLUG | 7.50 | 14.88 | 14.88 | 0.00 | 0.7203 | 0.4506 | 0.7396 | 14.88 |
| hotpotqa_s4_irrelevant | InstructRAG_ICL | 0.00 | 1.92 | 1.92 | 0.00 | 0.5000 | 0.6897 | 0.9700 | 1.92 |
| hotpotqa_s4_irrelevant | CAD_a05 | 1.67 | 2.98 | 2.98 | 0.00 | 0.5551 | 0.5200 | 0.9612 | 2.98 |
| hotpotqa_s4_irrelevant | U_GraphRAG_Plus | 2.50 | 7.69 | 0.00 | 0.00 | 0.6027 | 0.2415 | 0.9077 | 0.00 |
| hotpotqa_clean | B0_StandardRAG | 28.33 | 40.35 | 40.35 | 0.00 | 0.7032 | 0.2111 | 0.4224 | 40.35 |
| hotpotqa_clean | BGE_Reranker_Top5 | 30.83 | 47.54 | 47.54 | 0.00 | 0.6521 | 0.1486 | 0.3749 | 47.54 |
| hotpotqa_clean | BGE_Reranker_MMR | 31.67 | 47.45 | 47.45 | 0.00 | 0.6564 | 0.1649 | 0.3767 | 47.45 |
| hotpotqa_clean | CK_PLUG | 33.33 | 50.24 | 50.24 | 0.00 | 0.6382 | 0.3287 | 0.3804 | 50.24 |
| hotpotqa_clean | InstructRAG_ICL | 8.33 | 20.63 | 20.63 | 0.00 | 0.4505 | 0.5822 | 0.7470 | 20.63 |
| hotpotqa_clean | CAD_a05 | 25.00 | 42.83 | 42.83 | 0.00 | 0.7264 | 0.3327 | 0.3876 | 42.83 |
| hotpotqa_clean | U_GraphRAG_Plus | 30.00 | 46.06 | 35.85 | 0.00 | 0.7054 | 0.1210 | 0.3691 | 49.44 |
