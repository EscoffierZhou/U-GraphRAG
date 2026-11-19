# AI Agent 使用与人机协同日志（交付物 6）

本文档对应 **Agentic Research Challenge 2026（智能体科研创新挑战赛）Research Package 规范第 6 项：Agent 使用日志**，按研发阶段客观记录人工研究团队与 AI Agent 协同完成 **U-GraphRAG** 的全过程。所有日志均已完成隐私与凭据脱敏处理，不含任何 Token、密钥或云主机登录口令。

---

## 1. 全流程多智能体协同演进记录

| 阶段 | 调用的主要 Agent / 工具链 | 核心任务目标与关键操作 | 产出与可验证审计线索 |
| :--- | :--- | :--- | :--- |
| **阶段一：选题研判与文献创新性审计** | 文献检索 Agent + 学术批判对抗 Agent | 围绕 RAG 幻觉治理、自适应检索（`Adaptive-RAG`、`DRAGIN`、`SeaKR`）、冲突感知解码（`CAD`、`CK-PLUG`、`InstructRAG`）与共形风险控制开展布尔检索与机制缺口分析，收敛至 U-GraphRAG 三阶段框架。 | `RESEARCH_PROPOSAL.md`、`Debate/FINAL_METHODOLOGY_PROTOTYPE.md`、`Debate/NOVELTY_BOOLEAN_SEARCH_AUDIT.md` |
| **阶段二：核心算法实现与 V1–V5 多轮自审** | 编码实现 Agent + 数学不变量审查 | 完成 `M1AdaptiveRetriever`（`UncertaintyDecoupler`、`MinHashGraphExpander`、`BloomFilter`）、`TwoStageM2Optimizer`（`MSTEvidenceOrganizer`、`BudgetedEvidenceSelector`）与 `M3CalibratedDecoder` 模块编写；针对 PopQA 模板泄漏、划分隔离及朴素 MST 毒瘤节点反例完成 V1–V4 修复，最终冻结 V5 基线（`SHA256SUMS.txt` 共 `61` 份哈希校验文件）。 | `src/`、`V1_AUDIT_CRITIQUE_AND_V2_FIXES.md` 至 `V4_AUDIT_CRITIQUE_AND_POPQA_FIX.md`、`tests/test_m1_adaptive_retriever.py`、`tests/test_m2_optimizer.py`、`tests/test_v4_fixes.py` |
| **阶段三：跨模型家族外部验证（`MileStone 3`, WP6–WP11）** | 实验编排 Agent + RTX 4090 GPU 算力 | 构建并执行 `scripts/phase3/`（`wp6`–`wp11`），覆盖 `Qwen2.5-3B-Instruct`、`Qwen2.5-7B-Instruct`、`Meta-Llama-3.1-8B-Instruct` 三家族模型与四大 QA 基准。通过 WP8 自检揭示了 `V5 Proxy` 在缺失证据象限 `Q4` 的边界并诚实收敛声明范围，同时将 M2 图边构建修正为 `overlap_gated_js` 以保护多跳桥接文档（WP8.2）。 | `evidence/milestone3/`（`wp6`–`wp10` JSON 检查点、`conformal_posthoc_summary.json`、`MILESTONE3_EXPERIMENT_REPORT.md`） |
| **阶段四：免训练前沿基线对标与安全门控闭环（`MileStone_After`）** | 竞品算子隔离 Agent + 统计审计 Agent | 从 2024–2026 顶会开源仓库中抽取出纯 PyTorch 单文件兼容算子（`BGE-Reranker-v2-m3`、`CK-PLUG`、`InstructRAG-ICL`、`CAD`），避免污染底层 `transformers`；新增 `RelevanceLowerBoundGate` 将 `S4` 全无关低熵盲早退从 `85.83%`（Qwen-3B）/ `9.17%`（Llama-8B）/ `3.33%`（Qwen-7B）全部降至 **`0.0%`**，并在 `RAMDocs`、`NQ-Swap`、`HotpotQA` 完成受控赛马及 Split Conformal 事后风险审计。 | `results/MileStone4_Competitor/reuse/`、`results/MileStone4_Competitor/dist/`、`evidence/milestone4_after/`（三模型完整输出） |
| **阶段五：论文精修、引用核验、图表重绘与开源发布** | LaTeX 排版 Agent + 联网学术核验 Agent | 从原始 JSON 检查点直接重绘清除内部标签并消除气泡遮挡的高清 `fig2`–`fig8`；逐条核验全部 `44` 条参考文献；补充 Kirsch-Mitzenmacher 双哈希、布隆过滤器、MinHash 与 Kruskal MST 的算法原理前置讲解；完成 arXiv 独立投稿压缩包编译验证（EN `25pp` / ZH `27pp`，零报错）与 GitHub 多贡献者同步发布。 | `paper/`、`docs/`、`evidence/`、`README.md` |

---

## 2. 智能体协作过程中的工程防线

1. **零数据泄漏切分防线**：验证集（`dev`）与最终报告集（`test`）在 `scripts/phase2/cache_builder.py` 中硬隔离，所有校准阈值均仅在 `dev` 上拟合；`is_synthetic_conflict` 仅用于离线指标统计。
2. **冻结快照只读保护**：`scripts/phase2/common.py` 内置 `assert_safe_output_path()` 保护守卫，防止任何测试脚本覆盖已冻结的 `V5_FROZEN` 工件。
3. **人工最终审查把关**：论文全部实验表格数字、图表视觉质量、正负面边界表述以及合作者署名前后一致性均经人类研发团队审阅确认后定稿。
