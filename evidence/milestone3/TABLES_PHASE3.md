# Milestone 3 Camera-Ready Tables Summary

## Table 1: WP6 Cross-Family Generalization (Mean F1% / ECE / AURC)

| Backbone | Dataset | B0 Hybrid | B1 BM25 | B3 MMR | Full U-GraphRAG (F1 / ECE / AURC) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| qwen3b | hotpotqa | 42.53 | 44.94 | 47.83 | 47.56 / 0.1094 / 0.4336 |
| qwen3b | popqa | 96.13 | 94.44 | 98.33 | 98.89 / 0.0380 / 0.0006 |
| qwen3b | 2wikimultihop | 24.43 | 20.03 | 26.87 | 26.45 / 0.0801 / 0.5942 |
| qwen3b | musique | 17.11 | 18.91 | 24.59 | 21.87 / 0.1063 / 0.6588 |
| qwen7b | hotpotqa | 52.86 | 51.84 | 58.12 | 60.68 / 0.0815 / 0.2768 |
| qwen7b | popqa | 93.75 | 87.96 | 91.40 | 95.29 / 0.0653 / 0.0060 |
| qwen7b | 2wikimultihop | 40.08 | 38.35 | 42.90 | 45.01 / 0.0853 / 0.3961 |
| qwen7b | musique | 19.22 | 18.23 | 29.40 | 26.40 / 0.0963 / 0.5456 |
| llama3_8b | hotpotqa | 47.21 | 43.83 | 50.62 | 52.53 / 0.0534 / 0.3246 |
| llama3_8b | popqa | 76.08 | 73.86 | 78.16 | 89.08 / 0.0516 / 0.0845 |
| llama3_8b | 2wikimultihop | 26.48 | 24.67 | 28.75 | 31.09 / 0.0540 / 0.6059 |
| llama3_8b | musique | 18.75 | 15.48 | 23.03 | 29.73 / 0.0430 / 0.6235 |

## Table 2: WP7 In-Domain vs Out-of-Domain Zero-Tuning Transfer

| Target Dataset | Source Tuning | F1 (%) | ECE | Brier | Coverage (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| popqa | from_hotpotqa | 87.80 | 0.3572 | 0.1878 | 74.17 |
| popqa | in_domain | 87.80 | 0.0492 | 0.0556 | 79.17 |
| hotpotqa | in_domain | 50.60 | 0.0508 | 0.1468 | 72.50 |
| hotpotqa | from_popqa | 50.49 | 0.3336 | 0.2677 | 76.25 |
| 2wikimultihop | from_hotpotqa | 27.61 | 0.2201 | 0.1741 | 70.91 |
| 2wikimultihop | from_popqa | 27.52 | 0.5593 | 0.4445 | 74.55 |
| musique | from_hotpotqa | 26.11 | 0.2100 | 0.1750 | 64.09 |
| musique | from_popqa | 26.11 | 0.5427 | 0.4414 | 67.73 |

## Table 3: WP10 Hardware Efficiency & Exact vs Greedy M2 Comparison

| Backbone | Variant | E2E (ms) | M2 (ms) | QPS | Peak Reserved (MB) | F1 (%) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| llama3_8b | Standard_RAG | 8418.6 | 0.008 | 0.12 | 5844.0 | 40.46 |
| llama3_8b | MMR_RAG | 8258.8 | 1.187 | 0.12 | 5844.0 | 48.79 |
| llama3_8b | M1_Only | 8567.8 | 0.007 | 0.12 | 5824.0 | 42.09 |
| llama3_8b | M1_M2 | 9645.0 | 3.284 | 0.10 | 5824.0 | 46.86 |
| llama3_8b | M2_Exact_Enum | 8748.8 | 5.406 | 0.11 | 5844.0 | 45.27 |
| llama3_8b | M2_Greedy | 8876.8 | 4.364 | 0.11 | 5844.0 | 45.27 |
| llama3_8b | Full_UGraphRAG | 8234.2 | 3.323 | 0.12 | 6016.0 | 44.20 |
| qwen3b | Standard_RAG | 8799.2 | 0.008 | 0.11 | 2188.0 | 45.83 |
| qwen3b | MMR_RAG | 8621.1 | 1.094 | 0.12 | 2188.0 | 51.45 |
| qwen3b | M1_Only | 8849.2 | 0.007 | 0.11 | 2186.0 | 44.64 |
| qwen3b | M1_M2 | 9631.0 | 2.010 | 0.10 | 2184.0 | 46.46 |
| qwen3b | M2_Exact_Enum | 8948.4 | 5.377 | 0.11 | 2188.0 | 44.83 |
| qwen3b | M2_Greedy | 9168.9 | 4.362 | 0.11 | 2188.0 | 44.83 |
| qwen3b | Full_UGraphRAG | 8022.7 | 1.959 | 0.12 | 2394.0 | 44.63 |
| qwen7b | Standard_RAG | 6540.6 | 0.007 | 0.15 | 5698.0 | 47.67 |
| qwen7b | MMR_RAG | 6560.3 | 1.132 | 0.15 | 5696.0 | 48.06 |
| qwen7b | M1_Only | 6699.7 | 0.008 | 0.15 | 5694.0 | 52.58 |
| qwen7b | M1_M2 | 7028.7 | 2.391 | 0.14 | 5696.0 | 56.27 |
| qwen7b | M2_Exact_Enum | 7363.8 | 5.512 | 0.14 | 5696.0 | 58.77 |
| qwen7b | M2_Greedy | 7377.5 | 4.525 | 0.14 | 5696.0 | 58.77 |
| qwen7b | Full_UGraphRAG | 6409.6 | 2.379 | 0.16 | 5924.0 | 59.66 |
