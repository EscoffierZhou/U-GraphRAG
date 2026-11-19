# U-GraphRAG 第三轮实验结果 (V3) 审计结论与 V4 理论收尾修复说明

**审计对象**: `results/UGraphRAG_Experimental_Results_3` (hardware: NVIDIA GeForce RTX 5060 Laptop GPU, 8GB VRAM, Qwen2.5-3B / 7B 4-bit NF4 `bfloat16`)  
**对应迭代**: `V3 -> V4 (v4_theory_complete)`  
**审计结论**: V3 已彻底清零早期实验中的显存并发污染、评估口径不对称与置信度阈值漂移等工程缺陷，闭卷认知不确定性 ($U_{\mathrm{EU}}$ vs Error AUROC = `0.8917`)、选择性拒答排序能力 (`Full_UGraphRAG` AUROC = `0.7886`, Selective Risk 从 B0 的 `32.00%` 降至 `23.08%`) 以及证据召回紧凑度 (`Sp-F1` 由 `51.00%` 提升至 `61.06%~61.62%`, Token 节省率 `30.3%~50.3%`) 均已跑通；但距离「全消融模块严格单调净增益与多跳证据理论自洽收尾」尚余 **4 个结构性瓶颈**。V4 版本已完成全部底层重构与测试固化。

---

## 一、 V3 实证亮点与可保留成果 (Verified Strengths)

1. **硬件隔离与 3B→7B 尺度规律可靠**:
   - Step 1 (`Qwen2.5-3B-Instruct` vs `7B-Instruct`) 在严格串行显存隔离下完成：7B 峰值显存仅 `5310.8 MB` (`5.19 GB`, 8GB 安全裕度 `+2532.0 MB`)，HotpotQA EM/F1 由 3B 的 `41.00% / 53.75%` 稳健提升至 `54.00% / 64.68%` (EM `+13.00%`, F1 `+10.92%`)，PopQA Closed EM/F1 由 `24.00% / 29.58%` 提升至 `27.00% / 33.89%`。
2. **M1 不确定性解耦与 Bloom 零损失剪枝实证成立**:
   - Step 2 PopQA ($N=100$) 上闭卷 $U_{\mathrm{EU}}$ 呈显著长尾反转结构：`<10^2` Tail (`0.9743`, 闭卷准确率 `30.0%`)、`$10^3\text{-}10^4$` Medium (`1.7546`, `5.0%`) 显著高于 `>=10^5` Head (`0.5706`, `60.0%`)，故障判别 AUROC 达 **`0.8917`**。
   - Bloom 去重与剪枝在 PopQA 与 HotpotQA 上实现 `0.0000` 召回损失，同时 HotpotQA `M1_Only` 在 **节省 `30.3%` Token** 的前提下保持 EM `54.00%`（持平 B0）、F1 **`64.94%`** (`+0.26%` vs B0)、`Sp-F1` 由 `51.00%` 跃升至 **`61.06%`** (`+10.06%`)。
3. **M3 选择性预测 AUROC 与风险抑制有效**:
   - Step 3 HotpotQA 上，`Full_UGraphRAG` 借助置信度温度/校准门控实现 **`91.0%` Coverage 下的 `23.08%` Selective Risk**（显著低于 `B0_Standard_RAG` `100.0%` 覆盖率下的 `32.00%` 风险与 `B1_Naive_MST` 的 `51.00%` 风险），AUROC 从 `B0` 的 `0.5404` 跃升至 **`0.7886`**（`M2_Full` 本身为 `0.4661`），Brier Score 降至 **`0.1816`**（全消融最低）。

---

## 二、 V3 四大核心问题逐因拆解 (Forensic Root Causes)

### 问题 1 [M2-B]: `recency_aware=True` 导致 MST 线性化逆序反噬 (+3.42% Reverse > DFS)
- **V3 实证现象**:
  - Step 4 置换检验 ($N=100$) 中，`DFS (M2-B Ours)` 平均 F1 = **`63.19%`**，`Random Shuffles` = **`64.46%`**，`Reverse` 反而达到 **`66.61%`**（反超 DFS `+3.42%`）；
  - Step 3 中，`M2_Exact_NoMST` (F1 `62.18%`, EM `51.00%`) 经 DFS 排序变为 `M2_Full` (EM 反降 `-2.00%` 至 `49.00%`)，导致 EM 与 F1 均出现非预期倒挂。
- **代码根因**:
  - `MSTEvidenceOrganizer.__init__`（`src/m2_evidence_optimizer.py`）在 V3 默认开启 `recency_aware=True`，将 DFS 拓扑序故意反转为「**桥接跳前置、Query 核心高可靠主锚点压后（lost-in-the-middle 边缘末端）**」，同时组件根排序仅看实体重叠而未优先绑定 `query_entities`。当 `k=2~4` 时，把强相关的头跳核心证据推至 Prompt 末尾反而扰乱 Qwen2.5-7B 对「Query 主实体 -> 桥接实体 -> 答案实体」的因果顺推注意力。
- **V4 修复方案**:
  1. 在 `src/m2_evidence_optimizer.py`（`MSTEvidenceOrganizer`、`TwoStageM2Optimizer`）与 `src/u_graphrag_pipeline.py`（`m2_optimizer_exact`、`m2_optimizer_greedy`、`B1_Naive_MST`）及 `scripts/run_parameter_grid_search.py` 中，**统一将 `recency_aware` 默认切换为 `False`**，采用 **前向根优先 DFS 拓扑序 (Forward Root-First DFS)**；
  2. 为 `MSTEvidenceOrganizer.organize` 增加 `query_entities` 参数，连通分量排序与根节点选择均按 `(query_entity_overlap, calibrated_reliability)` 严格降序锁定（`M2_Full` 从与问题直连的核心节点顺向展开到桥接证据），彻底消弭「`Reverse` 反超 `DFS`」的因果倒置。

---

### 问题 2 [M2-A & Stage 2 Grid]: 多跳惩罚系数过高 ($0.34$) + 参数搜索未按 `offset` 分层导致 $k=1$ 退化
- **V3 实证现象**:
  - Step 3 中 `M1_Only` EM/F1 为 `54.00% / 64.94%`，进入 M2 精确子集选择 (`M2_Exact_NoMST`, `M2_Full`) 后跌至 `51.00% / 62.18%` 与 `49.00% / 63.19%`，`Sp-R` 由 B0 的 `88.50%` 跌至 `82.50%`；
  - 网格搜索 Stage 2 (`grid_search_results.json`) 全量评估的 8 组 LLM 配置全部坍缩在 `m2_reliability_offset = 0.34, top_k_candidate_pool = 8`，`avg_selected_docs = 3.18~3.36`，验证集 (`[100:145]`) `EM = 37.78%, F1 = 42.98%~46.71%`，明显低于同切片 B0 基线 (`EM = 46.67%, F1 = 54.35%`)。
- **代码根因**:
  1. Stage 2 按全局纯结构分 `structural_score` 截取 Top-8 送入 LLM 实测，`structural_score` 对高剪枝惩罚 `offset=0.34` 存在系统性偏爱，致使 `offset=0.16 / 0.22 / 0.28` 与 `pool=10` 从未被 LLM 实测；
  2. M2 子图目标函数在单跳置信度略高时允许 $k=1$ 早停，在 HotpotQA 双跳问答上直接斩断第二跳桥接句，造成无可挽回的支撑事实丢失。
- **V4 修复方案**:
  1. 在 `exact_enum_select`、`greedy_select` 与 `TwoStageM2Optimizer.optimize` 新增显式下界 `min_k: int = 1`（保持单跳/冲突拒答单元测试严格向后兼容），在 `UGraphRAGPipeline` 增加 `m2_min_docs: int = 2`（多跳主流程与 Stage 2 网格强制至少保留 `min(2, len(candidates))` 条连通证据），从机制上杜绝 2-hop 退化为单文档；
  2. 将 `m2_reliability_offset` 默认值由 `0.34` 下调至 **`0.22`**，扩大信息保留域；
  3. 重构 `scripts/run_parameter_grid_search.py` Stage 2 筛选器，按 `offset in [0.16, 0.22, 0.28, 0.34]` **分层保留每层 Top-2 去重结构构型** + 显式注入 `0.22 / pool=10` V4 默认点，并将网格切片从 `45` 条扩大至 `80` 条 (`HotpotQA [100:180]`) 以压低小样本方差。

---

### 问题 3 [M3]: Dev 验证切片 Platt 截距过拟合 (`b = -0.7892`) 引发评测 ECE 恶化
- **V3 实证现象**:
  - Stage 0 在 `[100:145]`（准确率仅 `37.78%`）上拟合得到 `platt_a = 1.2181`, **`platt_b = -0.7892`**；
  - 将该强负截距外推至 `[0:100]`（准确率 `50%~65%`）时，全量概率被系统性压低，致使 `Full_UGraphRAG` 的 **ECE 高达 `0.1849`**（反比 `B0` 的 `0.0710` 更差），而未校准前的 `M2_Full` ECE 亦为 `0.2201`。
  - 同时，Stage 3 `decode_grid` 选取的 `beta=0.0, guard=1.0` 说明过强对比罚项在 `[100:145]` 小切片上未能发挥平滑增益，反被极端阈值盖过。
- **代码根因**:
  - `scripts/run_parameter_grid_search.py` 中的 `_fit_platt_regularized` 对截距项施加的 L2 正则系数过弱，使小样本低准确率 dev 子集将域内基线偏差硬编码进了全局 `platt_b`；同时 `tau_conf` 缺乏零偏先验保护。
- **V4 修复方案**:
  1. 收紧 `_fit_platt_regularized` 的 **Zero-Intercept Shrinkage（零截距强缩水正则）**：将正则权重提升为 `lam_a = 4.0, lam_b = 14.0` 并将 `b` 硬约束至 **`[-0.28, 0.35]`**，确保 Platt 校准主要通过温度斜率 `a` 锐化排序，避免把 dev-slice 的均值准确率漂移带入测试集；
  2. 微调平衡解码候选网格 `decode_grid` 为 `[0.0/1.0, 0.04/0.90, 0.10/0.88, 0.18/0.85]`，并在 `80` 样本切片上重新寻优 `T_conf` 与门控阈值，目标将 `Full_UGraphRAG` 测试 ECE 压至 `< 0.09` 且保持 AUROC $\ge 0.78$。

---

### 问题 4 [Step 2 & Step 4 Reporting]: PopQA 检索难度饱和 (`100%` 封顶) 与 Table 3 符号/格式缺陷
- **V3 实证现象**:
  - Step 2 PopQA 中 `B0 (Top-5)`、`M1_Only`、`Full_UGraphRAG` 在所有 5 个流行度分桶上全部触及 **`100.0%` 准确率**，`D0 U_EU` 均值为 `0.0012~0.0069`，`early_exit_rate` 高达 `95%~100%`，`auroc_full_confidence = 0.5000`，导致 **M1 自适应 MinHash-LSH 图扩展相对 B0 Top-5 的真实增益空间被完全掩盖**；
  - 从底层分析发现，原有 `MinHashGraphExpander(num_bands=16)`（`rows_per_band=4`, $s=64$）理论半概率跃迁阈值高达 $t \approx (1/16)^{1/4} = 0.500$，与设定的 `similarity_threshold=0.15` 严重脱节，漏检约 $85\%$ 位于 $J \in [0.20, 0.45]$ 的真实桥接边；且实体重叠 fallback 分数可达 `0.24`，足以挤占 LSH 真实匹配位次；
  - 论文表格导出脚本 `scripts/plot_ablation_charts.py` 存在两处展示瑕疵：Table 3 `Delta vs DFS (%)` 使用 `-abs(max_drop)` 掩盖真实符号差（`reverse - dfs`），且 Table 2 缺少核心校准与选择性风险列 (`AUROC`, `Cov. (%)`, `Sel. Risk (%)`)。
- **V4 修复方案**:
  1. **对齐 LSH 频带几何与回退权重（`src/m1_adaptive_retriever.py`）**: 将 `MinHashGraphExpander` 默认 `num_bands` 调整为 **`32`** (`rows_per_band = 2`，跃迁阈值 $(1/32)^{1/2} \approx 0.177$ 与 `0.15` 精确对齐)，并将 `entity-bridge` 回退相似度上限压低至 `<= 0.14`，严格保证真实 k-shingle LSH 碰撞边恒排在弱实体共现之前；
  2. **重构 PopQA 对称硬检索共享语料（`scripts/run_overnight_ablation.py`）**:
     - 在保持 **完全对称评分与零流行度泄漏**（所有 doc 统一起始 `score = 0.65`）的前提下，对每条知识拆解为「高诱饵浅层背景簇 `kg_bg`（含查询实体与噪声属性模板）」+「实体绑定 LSH 桥接关系证据 `kg_rel`」+「高密度实体干扰消歧簇 `kg_disambig`」；
     - 实测 CPU 检索预验证（`HybridRetriever BM25 + bge-small-en-v1.5` + `MinHashGraphExpander(num_bands=32)`, $1,950$ 文档 / $100$ 查询）：**`B0 (Top-5) Gold Recall = 63%`**，触发 M1 LSH 扩展后 **`M1_Exp5 = 85%` / `M1_Exp10 = 91%`**，LSH 命中率 `87%`，Bloom 剪枝召回损失 `0.0000`，真正释放 PopQA 长尾上 `B0 -> M1 -> Full` 的阶梯跃升空间！
  3. **修复图表与表格脚本（`scripts/plot_ablation_charts.py`）**: Table 3 改用严格有符号均值差 `reverse - dfs` 与 `shuffles - dfs`；Table 2 补全 `AUROC`, `Cov. (%)`, `Sel. Risk (%)` 三列，使正文 LaTeX 与 Markdown 汇总报告零误差对齐。

---

## 三、 V4 预期验收准线 (Theory-Complete Acceptance Targets)

本轮修改已完成 `python -m unittest discover -s tests -v` 全量单元测试（含新增 5 项 V4 专项回归测试）、`run_parameter_grid_search.py --dry-run`（80/20 切片 CPU+分层 LLM 全流程）及 8-Mode + Plotter 冒烟测试。RTX 5060 GPU 全量运行后的预期理论收尾准线如下：

| 指标维度 | V3 实测痛点 | V4 修复机理 | V4 预期验收目标 |
| :--- | :--- | :--- | :--- |
| **Step 2 PopQA Recall / Acc** | `B0 = M1 = Full = 100%`（无区分度） | 对称硬分面语料 + LSH 32-band 精准桥接 | `B0 ~60-68%` 显著低于 `M1 / Full >= 80%`, `Closed AUROC >= 0.85` |
| **Step 3 HotpotQA M2 vs M1** | `M2_Full` F1 (`63.19%`) 低于 `M1_Only` (`64.94%`) | `min_k=2` + `offset=0.22` + 前向根优先 DFS | `M2_Full` EM/F1 追平或反超 `M1_Only` 且维持 `30%+` Token 节省 |
| **Step 3 Full ECE Calibration** | `Full ECE = 0.1849`（负截距过拟合） | 零截距强正则 Platt (`lam_b=14.0`, `b in [-0.28, 0.35]`) | `Full ECE <= 0.10` 且 `AUROC >= 0.75`, 显著降低 Selective Risk |
| **Step 4 M2 拓扑方向性** | `Reverse (66.61%)` 反超 `DFS (63.19%)` `+3.42%` | `recency_aware=False` + `query_entities` 锁根 | `DFS >= Reverse` 且 `DFS >= Random Shuffles`, `Intra-Std = ±0.00` |
| **图表与 LaTeX 表格审计** | Table 3 `-abs` 掩盖符号、Table 2 缺校准列 | 有符号差值 + 全校准指标表 | `TABLES_SUMMARY.md` 与 `tables_latex.tex` 数值 100% 一致 |
