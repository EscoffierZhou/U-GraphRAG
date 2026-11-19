# 实验证据索引总览（交付物 4）

本目录（`evidence/`）汇集了 **Agentic Research Challenge 2026 Research Package 规范第 4 项：实验证据** 的全部机器生成汇总 JSON 检查点、Markdown 指标总表、LaTeX 表格源文件及出版级高清图表（`fig2`–`fig8`）。

---

## 1. Milestone 3 跨模型家族与消融外部验证（`evidence/milestone3/`）

> 硬件环境记录（见 [reproducibility_manifest.json](milestone3/reproducibility_manifest.json)）：NVIDIA GeForce RTX 4090（24 GB 显存）、Python 3.12.4、PyTorch 2.15.0+cu132、Transformers 5.23.0、bitsandbytes 0.50.2；`V5_FROZEN` 完整性检查 `61/61` 通过（`status: ok`）。

| 工件文件 | 核心内容与可验证关键结论 |
| :--- | :--- |
| [MILESTONE3_EXPERIMENT_REPORT.md](milestone3/MILESTONE3_EXPERIMENT_REPORT.md) | `WP6`–`WP11` 外部验证与方法学闭环总结报告，含真实失败案例分布统计（`selection_loss`: 3、`generation_or_reasoning_error`: 3、`harmful_abstention`: 3）。 |
| [TABLES_PHASE3.md](milestone3/TABLES_PHASE3.md) | `WP6`（3 家族模型 × 4 数据集）、`WP7` 跨库零微调迁移、以及 `WP10` 硬件效率对比总表。 |
| [wp6_cross_family_matrix.json](milestone3/wp6_cross_family_matrix.json) | 完整 `3 × 4` 跨家族实验矩阵：例如在 `HotpotQA` 上，`Qwen2.5-7B` 的 F1 从 `52.86%`（`B0 Hybrid`）/ `58.12%`（`B3 MMR`）提升至 **`60.68%`**（`ECE = 0.0815`）；在 `PopQA` 上，`Qwen2.5-3B` 达到 **`98.89%`**，`Llama-3.1-8B` 从 `76.08%` 提升至 **`89.08%`**；在 `MuSiQue` 上，`Llama-3.1-8B` 从 `18.75%` 提升至 **`29.73%`**。 |
| [wp7_cross_dataset_transfer.json](milestone3/wp7_cross_dataset_transfer.json) | 零微调跨数据集阈值迁移评估，清晰揭示领域内标定（`ECE ~ 0.05`）与直接跨库迁移（`ECE = 0.21–0.56`）之间的协变量偏移边界。 |
| [wp8_method_validity_results.json](milestone3/wp8_method_validity_results.json) | `WP8.1` 四象限不确定性检验（`AUROC = 0.5939` / `0.6206`，严谨界定 $U_{\mathrm{EU}}^{v5}$ 为“运维路由信号”而非理论纯贝叶斯互信息）与 `WP8.2` `overlap_gated_js` 多跳桥接保护消融证明。 |
| [wp9_stress_curves.json](milestone3/wp9_stress_curves.json) | `Llama-3.1-8B`（`n=120`）在噪声注入、数值实体冲突注入及上下文截断单轴压力下的稳健性曲线。 |
| [wp10_efficiency_pareto.json](milestone3/wp10_efficiency_pareto.json) | 端到端时延、M2 图选择器耗时（Full U-GraphRAG 仅 **`1.96–3.32 ms`**；15 选 5 精确枚举与贪心耗时约 `4.4–5.5 ms` 且 F1 一致）、吞吐（QPS）与 4-bit 峰值保留显存（`Qwen2.5-3B` 仅 **`2.39 GB`**，`Qwen2.5-7B` / `Llama-3.1-8B` 约 **`5.92–6.02 GB`**）。 |
| [conformal_posthoc_summary.json](milestone3/conformal_posthoc_summary.json) | 零额外 GPU 开销的事后 Split Conformal 校准曲线。 |
| [qualitative_failure_cases.json](milestone3/qualitative_failure_cases.json) | 从 Full U-GraphRAG 输出中自动挖掘的真实失败样例轨迹。 |
| [reproducibility_manifest.json](milestone3/reproducibility_manifest.json) | 实验时间戳、软硬件依赖版本号及 `V5_FROZEN` 校验状态清单。 |

---

## 2. `MileStone_After`：前沿竞品对标与 S4 安全门控（`evidence/milestone4_after/`）

各模型子目录（[llama3_8b/](milestone4_after/llama3_8b/)、[qwen3b/](milestone4_after/qwen3b/)、[qwen7b/](milestone4_after/qwen7b/)）分别包含 `milestone_after_horserace_results.json`、`TABLES_MILESTONE_AFTER.md`、`tables_milestone_after.tex` 及 Split Conformal 审计输出。在 `RAMDocs`（200 题）、`NQ-Swap`（`dual_conflict`，200 题）与 `HotpotQA`（`Clean` / `Conflict` / `S4 All-Irrelevant`，各 120 题）上的核心实证结果如下：

1. **`S4` 全无关池盲目早退拦截效果（`RelevanceLowerBoundGate`）**：
   - `Qwen2.5-3B`：盲早退率从 **`85.83%`** 降至 **`0.0%`**（拦截 `-85.83 pp`）。
   - `Llama-3.1-8B`：盲早退率从 **`9.17%`** 降至 **`0.0%`**（拦截 `-9.17 pp`）。
   - `Qwen2.5-7B`：盲早退率从 **`3.33%`** 降至 **`0.0%`**（拦截 `-3.33 pp`）。
2. **`NQ-Swap` 反事实知识冲突表现（`U_GraphRAG_Plus`）**：
   - **`Qwen2.5-7B`**：F1 达 **`42.09%`**（`ECE = 0.1704`），显著领先 `B0_StandardRAG`（`9.33%`）、`BGE_Reranker_MMR`（`8.18%`）、`CK_PLUG`（`11.46%`）与 `CAD_a05`（`13.53%`）。
   - **`Llama-3.1-8B`**：F1 达 **`37.12%`**（`ECE = 0.1973`），对比 `B0`（`15.88%`）、`CK_PLUG`（`21.31%`）、`CAD_a05`（`29.79%`）。
   - **`Qwen2.5-3B`**：F1 达 **`37.09%`**（`ECE = 0.2260`），对比 `B0`（`14.18%`）、`BGE_Reranker_MMR`（`20.33%`）、`CK_PLUG`（`18.86%`）。
3. **`RAMDocs` 混合冲突与多义性证据优选中表现（`Qwen2.5-7B`）**：
   - `U_GraphRAG_Plus` 取得 **`69.40%` Raw F1** / **`63.00%` EM** / **`0.1198` ECE** / **`75.58%` SelAcc**，全面优于 `B0`（`65.59%`）、`BGE_Reranker_MMR`（`63.94%`）与 `CK_PLUG`（`63.50%`）。

---

## 3. 论文与证据高清图表（`evidence/figures/`）

全部图表均基于上述 JSON 检查点直接重绘，不含内部工程标签，并与论文正文中插图完全一致：

- `fig2_wp8_m1_uncertainty.png`：M1 四象限不确定性分离与路由行为对比图。
- `fig3_wp8_m2_optimization.png`：M2 重叠门控图选择 Pareto 前沿曲线。
- `fig4_wp6_cross_family_main.png`：三模型家族 × 四基准 F1 / ECE / AURC 综合对比图。
- `fig5_wp8_calibration_rc.png`：事后 Platt + Split Conformal 校准与 Risk–Coverage 曲线。
- `fig6_wp9_stress_curves.png`：噪声、冲突注入与上下文截断压力曲线。
- `fig7_wp10_quality_latency_vram.png`：质量–时延–显存三维度 Pareto 散点气泡图。
- `fig8_milestone_after_horserace.png`：对比 `BGE-Reranker-MMR`、`CK-PLUG`、`InstructRAG-ICL`、`CAD` 的赛马对比及 `S4` 早退门控效果图。
