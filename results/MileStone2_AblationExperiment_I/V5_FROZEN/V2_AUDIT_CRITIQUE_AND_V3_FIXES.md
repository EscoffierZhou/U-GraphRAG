# U-GraphRAG V2 审查评判报告与 V3 修复升级对照（实验包 3 基线档案）

> [!CAUTION]
> **审查员对 V2（`UGraphRAG_Experimental_Results_2`）的最终裁决：Reject / 不可直接投稿（存在 4 项阻断级系统 bug，导致 Full 系统表现大幅低于模块子集）**。  
> 本文档基于对 [UGraphRAG_Experimental_Results_2](file:///F:/desktop/MyRouter/Nekoya/nekolab/%5BSDUFE%5DGeminiOutOfToken_Team/results/UGraphRAG_Experimental_Results_2/results/FULL_EXPERIMENT_REPORT.md) 全套日志、逐样本指标与 [GRID_SEARCH_REPORT.md](file:///F:/desktop/MyRouter/Nekoya/nekolab/%5BSDUFE%5DGeminiOutOfToken_Team/results/UGraphRAG_Experimental_Results_2/results/GRID_SEARCH_REPORT.md) 的代码级取证审计，如实记录 V2 已攻克的亮点、仍存在的 4 大阻断问题及精确根因，以及 `实验包 3（V3）` 所落地的代码与调参引擎修复方案。

---

## 一、 V2 已验证攻克的真实亮点（实证保留）

1. **实验真实性与工程一致性通过**：确系 `NVIDIA GeForce RTX 5060 Laptop GPU`（8 GB VRAM, 4-bit NF4）真机完整运行，正文与 JSON 数据零自相矛盾，三张图表布局清晰零文字遮挡。
2. **3B 显存计数清零修复完成**：`scaling_3b` 组 `B0 / Full` 的峰值已分配显存已从 V1 污染值 `5719.67 MB` 回落至真实的 **`2187.47 / 2239.07 MB`**。
3. **M1 尺度断裂修复见效**：接入与 `m1_calibration.json` 一致的 LLM Logits 探针后，HotpotQA-100 实现 **`73 / 100` 自适应早退**（彻底结束 V1 `100/100` 盲扩图）；`M1_Only` 在 **节省 `30.3%` Token** 的前提下取得 **`EM = 54.00%`、`F1 = 64.94%`、`Sp-F1 = 61.06%`**（全面打平或超越 `B0` 的 `EM = 54.00% / F1 = 64.68% / Sp-F1 = 51.00%`）。
4. **不确定度排序区分度（AUROC）极具潜力**：
   - PopQA 闭卷 $U_{\mathrm{EU}}$ 预测模型知识盲区的 **`AUROC = 0.8917`**（跨 5 个流行度分桶呈清晰单调分化）；
   - HotpotQA 主实验中 `Full_UGraphRAG` 的 **`AUROC = 0.8685`**（相对 `B0 = 0.5404`、`M2_Full = 0.5310` 提升 `+0.33` 以上），证明 token-level + evidence reliability 信号在排序维度上已经高度有效。

---

## 二、 V2 四大阻断级硬伤与精确代码根因

尽管 M1 探针与 AUROC 信号已跑通，V2 主表仍呈现 **`Full_UGraphRAG (EM = 33.00%, F1 = 39.18%)` 远低于 `M2_Full (51.00%, 61.38%)` 与 `B0 (54.00%, 64.68%)`**、PopQA 上 `Full` 被误拒答拉低至 `50%~85%`（低于 `B0 = 100%`），经逐行排查锁定 4 个精确代码原因：

### 硬伤 1：`Full_UGraphRAG` 独占早退截断 Bug（`src/u_graphrag_pipeline.py:920-921`）
- **问题代码**：
  ```python
  if mode == "Full_UGraphRAG" and exp_res.early_exit:
      m1_expanded_docs = [dict(d) for d in initial_docs]
  ```
- **实证后果**：仅对 `Full_UGraphRAG` 在 M1 早退（`73 / 100` 题）时强行把传入 M2 的候选池从 `8~10` 篇截断为区区 `3` 篇，而对比模式 `M2_Full` 反而能看到完整候选池。结果导致 `Full_UGraphRAG` 平均入选文档数由 `2.94` 篇骤降至 `1.74` 篇，支持句召回 `Sp-Recall` 由 `78.00%` 暴跌至 `62.50%`；即使在未被拒答的 `62` 题上，原本与 `M2_Full` 可比的准确率也被直接打伤（`11` 处非拒答差异中有 `10` 处改为错答，EM 由 `58.06%` 跌至 `53.23%`）。

### 硬伤 2：`152K` 词表 $T=3.0$ 概率坍缩 + `0 / 2016` 有效配置 + PopQA `-m2_offset` 漏还原
- **Stage 3 搜索坍缩**：V2 Stage 3 扫描 `temp_grid = [0.8..3.0]` 且 `conf_thresh_grid` 最低仅到 `0.35`。对于 Qwen2.5 `152,064` 词表，温度除数 $T=3.0$ 使归一化分母 $\sum_v \exp(z_v / 3.0)$ 急剧膨胀，直接把 `mean_confidence` 压低至 `0.4876`，导致全部 `2,016` 组参数出现 **`0` 组有效校准配置（`valid_selective_count = 0, valid_full_cov_count = 0`，最低 `ECE = 0.1580 > 0.095`）**。
- **过度硬拒答破坏主表 EM**：兜底选出的 `T=3.0, tau_conf=0.45` 在 HotpotQA 测试集上仅保留 **`62.0%` 覆盖率（强行拒答 `38 / 100` 题）**，而被拒答的 38 题里 `M2_Full` 原本做对了 `39.47%`，导致 `Full_UGraphRAG` 整表 `EM` 直接腰斩至 `33.00%`；同时 `full_coverage_calibrated` 因同样落入零有效兜底，继承了 `57.78%` 覆盖率。
- **PopQA 额外 Bug（`scripts/run_overnight_ablation.py:725`）**：`run_experiment_2` 对候选文档减去 `m2_offset` 后**漏掉像 `u_graphrag_pipeline.py` 一样恢复预偏移 `orig_rel`**，将扣减后的低 `calibrated_reliability` 直接喂给 `M3CalibratedDecoder`，在 `B0 = 100%` 全对的题目上误拒答 `15%~50%`，拉垮 PopQA `Full` 至 `70/85/75/50/60%`（`ECE = 0.2588`）。

### 硬伤 3：Stage 2 调参目标过分奖励极端截断，伤害基准检索完整性
- **实证数据**：在 V2 [GRID_SEARCH_REPORT.md](file:///F:/desktop/MyRouter/Nekoya/nekolab/%5BSDUFE%5DGeminiOutOfToken_Team/results/UGraphRAG_Experimental_Results_2/results/GRID_SEARCH_REPORT.md) Stage 2 中：
  - `(pool_k=10, offset=0.32)` 取得 dev **`EM = 42.22% / F1 = 51.47%`**（Token Saving `16.8%`）；
  - `(pool_k=8, offset=0.42)` 仅得 dev **`EM = 40.00% / F1 = 48.22%`**（Token Saving `49.3%`）。
- **目标失衡**：原公式 `+ 0.15 * token_saving_pct` 给后者额外加了 `+4.88` 分，使以损失 `-3.25% F1 / -2.22% EM` 为代价的激进剪枝方案险胜，直接造成主测试集上 `M2_Full` 从 V1 的 `56.00% EM / 68.07% F1` 滑坡到 V2 的 `51.00% EM / 61.38% F1`。

### 硬伤 4：M2-B 线性化逆因果注意力偏好 & PopQA 100% 天花板失真
- **因果大模型的近因偏好（Recency Bias）**：V2 Step 4 扩充至 `N=100` 后暴露出客观规律：**`Reverse Order (66.74% F1) > Random Shuffles (65.00% F1) > DFS Ordered (61.38% F1)`**（逐题对比 `Reverse > DFS` 为 `9 : 3`，`88` 题持平）。根因在于旧版 MST DFS 将最高可靠度核心锚点（`root_node`）排在索引 `0`（远离末尾 `Question:`），反倒把弱关联叶子节点贴在问题前面；而 `Reverse` 恰好把最关键的推理锚点放到紧邻 `Question:` 处。
- **PopQA 语料过纯**：V2 的 `popqa_shared_corpus` 仅包含 `300` 句超干净单句模板，零干扰文档，导致 `B0 (Top-5)` 五个桶全数 `100.0%`，`(q, D_0)` 的检索不确定性被压至 `0.0013~0.0092`，无法检验多跳/混淆语境下 M1 自适应扩图与 M2 去噪价值。

---

## 三、 V3（实验包 3）代码与调参引擎修复清单

| 对应 V2 硬伤 | V3 修改文件与函数 | 精确修复方案 |
| :--- | :--- | :--- |
| **硬伤 1（Full 截断 Bug）** | `src/u_graphrag_pipeline.py` (`run_pipeline_single`) | 彻底删除 `if mode == "Full_UGraphRAG" and exp_res.early_exit:` 特判分支，使 `Full_UGraphRAG` 与 `M2_Full` 共享完全一致的候选证据池：M1 早退仅跳过图拓展步（零扩图开销），不再腰斩 M2 候选集 |
| **硬伤 2A（152K 概率坍缩 + Platt 校准）** | `src/m3_calibrated_decoder.py`<br>`src/u_graphrag_pipeline.py` | 1. 新增单调保序 **Platt Logistic 校准参数** `(confidence_platt_a, confidence_platt_b)`：在保留高 `AUROC` 排序完全不变前提下（$a \ge 0.55$ 严格单调），对齐词表温度缩放偏移以压低 `ECE`；<br>2. 在 Pipeline `__init__`、`configure_parameters` 及 `Exp 2 / Exp 3`（7B + 3B）全链路打通持久化读取 |
| **硬伤 2B（Stage 3 搜索与高覆盖门控）** | `scripts/run_parameter_grid_search.py` (`run_m3_grid_search`) | 1. 引入向恒等映射 $(a=1, b=0)$ 收缩的 **5-Fold 折外正则化 Platt IRLS 拟合法**（零过拟合），搜索范围扩展 `tau_conf in [0.15..0.52]`、`T_conf in [0.8..2.0]`；<br>2. `balanced_score` 改用 **`effective_f1 / effective_em`**，强制 `is_valid_selective` 满足高覆盖约束 **`coverage >= 84%`**、`ECE <= 0.095`、`Selective Risk < Base Risk`；`full_coverage_calibrated` 硬性兜底保证 **`coverage >= 97%`** |
| **硬伤 2C（PopQA 偏移恢复）** | `scripts/run_overnight_ablation.py` (`run_experiment_2`) | 对齐 HotpotQA 逻辑：在扣除 `m2_offset` 前保存 `orig_rel`，在 M2 选择之后立即还原真实可靠度再输入 `M3CalibratedDecoder`，彻底消除 PopQA 虚假低可靠度触发误拒答的问题 |
| **硬伤 3（Stage 2 过剪枝）** | `scripts/run_parameter_grid_search.py` (`run_m2_grid_search`) | 收敛 `offset_grid = [0.22, 0.28, 0.34]`，并在 `joint_score` 引入高精度底线保护（`F1 >= max_F1 - 1.5%` & `EM >= max_EM - 2.0%` 加权优先），严防为追求极限压缩率牺牲检索召回与 EM/F1 |
| **硬伤 4A（近因偏好感知 MST-DFS）** | `src/m2_evidence_optimizer.py` (`MSTEvidenceOrganizer`) | 新增 **`recency_aware: bool = True`** 自底向上拓扑线性化：先排次要连通分量、分量内按从外围叶节点 $\to$ 桥接节点 $\to$ 核心高权重锚点 `root_node` 的连贯路径顺序排列，使主推理链条紧贴 `Question:`，适配因果 Transformer 近因注意力机制 |
| **硬伤 4B（PopQA 对称混淆语料）** | `scripts/run_overnight_ablation.py` (`run_experiment_2`) | 在同一共享语料 `popqa_shared_corpus` 中对称混入同实体背景/同名歧义混淆块 + `900` 条真实 HotpotQA Wikipedia 段落（所有文档统一初始化 `score=0.65`，杜绝先验泄露），打破 `B0` `100%` 虚高天花板，真实检验 `(q, D_0)` 歧义触发与 M1/M2/M3 联合增益 |

---

## 四、 V3（RTX 5060 8GB）重跑验收指标红线

1. **HotpotQA 主表（P0 必须达标）**：
   - `Full_UGraphRAG` 的 `Sp-Recall / Sp-F1` 与 `M2_Full` 对齐提升，不再出现 `78% -> 62.5%` 断崖下跌；
   - `Full_UGraphRAG` 的 `EM / F1` 高于 `B0_Standard_RAG` 且 $\ge$（或稳定匹配并领先于）`M2_Full`，同时保留高覆盖率（`Coverage >= 85%~98%`）而非以砍掉 `38%` 题目换分数；
   - `Full_UGraphRAG` 维持高 `AUROC` 的同时显著压低 **`ECE`（目标 `<= 0.08`）** 与 **`Selective Risk`（显著低于 `M2_Full` / `B0`）**。
2. **Stage 0 网格有效性**：`valid_selective_count > 0` 且 `valid_full_cov_count > 0`，`full_coverage_calibrated` 覆盖率必须稳定在 **`>= 97%`**。
3. **Step 4 置换检验（`N=100`）**：升级近因感知 `M2-B DFS` 后，检验 `DFS >= Reverse` 且 `DFS > Random Shuffles`。
4. **Step 2（PopQA 对称混淆库）**：`B0 (Top-5)` 退出全桶 `100%` 失真状态，呈现长尾分层，`Full_UGraphRAG` 与 `M1_Only` 展现合理检索/校准提升且无误拒答崩盘。
