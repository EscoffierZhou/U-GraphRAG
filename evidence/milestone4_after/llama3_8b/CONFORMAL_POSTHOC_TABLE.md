# Post-Hoc Split Conformal Risk Control Audit (Zero Extra GPU Calls)

| Source | Dataset | Policy | Tau Conf | Coverage (%) | Sel Acc (%) | HarmAb (%) | AURC |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| WP8 | hotpotqa | Dev Adaptive Baseline | 0.3226 | 78.75 | 55.13 | 17.65 | 0.4018 |
| WP8 | hotpotqa | Split Conformal (alpha_10) | 0.3228 | 78.75 | 55.13 | 17.65 | 0.4018 |
| WP8 | hotpotqa | Split Conformal (alpha_15) | 0.3959 | 73.12 | 55.09 | 27.91 | 0.4018 |
| WP8 | hotpotqa | Split Conformal (alpha_20) | 0.4287 | 67.50 | 57.52 | 28.85 | 0.4018 |
| WP8 | hotpotqa | Split Conformal (alpha_30) | 0.5043 | 58.13 | 58.56 | 34.33 | 0.4018 |
| WP8 | musique | Dev Adaptive Baseline | 0.2422 | 72.50 | 26.41 | 2.27 | 0.6566 |
| WP8 | musique | Split Conformal (alpha_10) | 0.1906 | 86.88 | 22.33 | 4.76 | 0.6566 |
| WP8 | musique | Split Conformal (alpha_15) | 0.2393 | 73.12 | 26.18 | 2.33 | 0.6566 |
| WP8 | musique | Split Conformal (alpha_20) | 0.2495 | 71.88 | 26.64 | 2.22 | 0.6566 |
| WP8 | musique | Split Conformal (alpha_30) | 0.2744 | 68.75 | 27.85 | 2.00 | 0.6566 |
| WP6-qwen3b | hotpotqa | Dev Adaptive | 0.3226 | 81.87 | 54.45 | 13.79 | 0.4167 |
| WP6-qwen3b | hotpotqa | Split Conformal (alpha=0.15) | 0.3959 | 76.56 | 53.61 | 26.67 | 0.4167 |
| WP6-qwen3b | popqa | Dev Adaptive | 0.6075 | 98.33 | 98.31 | 50.00 | 0.0019 |
| WP6-qwen3b | popqa | Split Conformal (alpha=0.15) | 0.9016 | 87.50 | 99.05 | 86.67 | 0.0019 |
| WP6-qwen3b | 2wikimultihop | Dev Adaptive | 0.1515 | 75.45 | 35.46 | 9.26 | 0.5477 |
| WP6-qwen3b | 2wikimultihop | Split Conformal (alpha=0.15) | 0.1612 | 71.82 | 37.05 | 8.06 | 0.5477 |
| WP6-qwen3b | musique | Dev Adaptive | 0.2422 | 77.73 | 25.82 | 8.16 | 0.6854 |
| WP6-qwen3b | musique | Split Conformal (alpha=0.15) | 0.2393 | 77.73 | 25.82 | 8.16 | 0.6854 |
| WP6-qwen7b | hotpotqa | Dev Adaptive | 0.3800 | 78.12 | 67.31 | 38.57 | 0.2773 |
| WP6-qwen7b | hotpotqa | Split Conformal (alpha=0.15) | 0.3995 | 75.94 | 68.83 | 36.36 | 0.2773 |
| WP6-qwen7b | popqa | Dev Adaptive | 0.7312 | 89.17 | 97.84 | 84.62 | 0.0049 |
| WP6-qwen7b | popqa | Split Conformal (alpha=0.15) | 0.7546 | 88.33 | 97.82 | 85.71 | 0.0049 |
| WP6-qwen7b | 2wikimultihop | Dev Adaptive | 0.3332 | 74.55 | 56.62 | 10.71 | 0.3994 |
| WP6-qwen7b | 2wikimultihop | Split Conformal (alpha=0.15) | 0.3402 | 74.09 | 56.97 | 10.53 | 0.3994 |
| WP6-qwen7b | musique | Dev Adaptive | 0.2531 | 71.82 | 32.31 | 11.29 | 0.5643 |
| WP6-qwen7b | musique | Split Conformal (alpha=0.15) | 0.2616 | 69.55 | 33.28 | 10.45 | 0.5643 |
| WP6-llama3_8b | hotpotqa | Dev Adaptive | 0.4047 | 72.50 | 57.69 | 38.64 | 0.3383 |
| WP6-llama3_8b | hotpotqa | Split Conformal (alpha=0.15) | 0.3934 | 74.69 | 57.32 | 37.04 | 0.3383 |
| WP6-llama3_8b | popqa | Dev Adaptive | 0.7820 | 80.83 | 87.55 | 86.96 | 0.1016 |
| WP6-llama3_8b | popqa | Split Conformal (alpha=0.15) | 0.7376 | 88.33 | 88.11 | 78.57 | 0.1016 |
| WP6-llama3_8b | 2wikimultihop | Dev Adaptive | 0.1956 | 78.64 | 31.47 | 23.40 | 0.6431 |
| WP6-llama3_8b | 2wikimultihop | Split Conformal (alpha=0.15) | 0.1966 | 76.82 | 31.03 | 25.49 | 0.6431 |
| WP6-llama3_8b | musique | Dev Adaptive | 0.2535 | 93.64 | 31.34 | 0.00 | 0.6132 |
| WP6-llama3_8b | musique | Split Conformal (alpha=0.15) | 0.2468 | 95.00 | 30.99 | 0.00 | 0.6132 |
