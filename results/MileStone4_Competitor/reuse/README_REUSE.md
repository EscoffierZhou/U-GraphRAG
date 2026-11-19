# `MileStone4_Competitor/reuse` — 精简可复用顶会资产清单

> 本目录从 `results/MileStone4_Competitor/repos/`（11 个顶会克隆仓库）中**去除了所有臃肿的第三方库复刻（如 CK-PLUG 4,992 个文件的 `transformers-4.49` fork、AdaRAGUE 上千个历史预测文件）**，仅沉淀可直接导入 `U-GraphRAG` 与 `MileStone_After` 的纯净数据集、独立算子和原始对照参考脚本。

---

## 一、目录结构与来源对照

### 1. `datasets/`（公开真实冲突基准 + 统一转换器）
- [RAMDocs_test.jsonl](file:///F:/desktop/Nekoya/nekolab/[SDUFE]GeminiOutOfToken_Team/results/MileStone4_Competitor/reuse/datasets/RAMDocs_test.jsonl)
  - **来源**：`repos/RAMDocs/RAMDocs_test.jsonl`（arXiv:2504.13079, MADAM-RAG）。
  - **规模与特性**：**500 条**真实多文档冲突 QA（含 `correct` / `misinfo` / `noise` 三类标注、`disambig_entity` 多义答案 `gold_answers` 与 `wrong_answers`）。
- [nqswap_dev.parquet](file:///F:/desktop/Nekoya/nekolab/[SDUFE]GeminiOutOfToken_Team/results/MileStone4_Competitor/reuse/datasets/nqswap_dev.parquet)
  - **来源**：`repos/DeCoRe/data/NQ-Swap/data/dev-00000-of-00001.parquet`。
  - **规模与特性**：**4,746 条**反事实实体替换问答（含成对 `org_context` / `org_answer` 与 `sub_context` / `sub_answer`），是检验参数知识 vs 上下文忠诚度（CK-PLUG / CAD / DeCoRe）的标准基准。
- [dataset_adapters.py](file:///F:/desktop/Nekoya/nekolab/[SDUFE]GeminiOutOfToken_Team/results/MileStone4_Competitor/reuse/datasets/dataset_adapters.py)
  - **统一适配**：提供 `load_ramdocs_dataset()`、`load_nqswap_dataset(mode='dual_conflict'|'sub_only')` 及 `build_all_irrelevant_s4_slice()`（构造 S4 全无关低熵陷阱切片），全部对齐 U-GraphRAG 标准字段。

### 2. `operators/`（纯 Python/PyTorch 独立即插即用算子）
- [ck_plug_operator.py](file:///F:/desktop/Nekoya/nekolab/[SDUFE]GeminiOutOfToken_Team/results/MileStone4_Competitor/reuse/operators/ck_plug_operator.py)
  - **提取自**：`CK-PLUG/transformers-4.49/.../utils.py#L2585-L2773`（ICLR 2026）。
  - **功能**：纯 PyTorch 复刻 `relative_top_filter_pair` + `compute_ck_plug_step_logits`（Confidence Gain `IG = H_student - H_ctx` 冲突检测与自适应权重调制），**无需覆盖环境里的 `transformers`**，且同步输出调制前/后概率分布以便同口径核查 ECE/AUROC。
- [cad_operator.py](file:///F:/desktop/Nekoya/nekolab/[SDUFE]GeminiOutOfToken_Team/results/MileStone4_Competitor/reuse/operators/cad_operator.py)
  - **提取自**：`DeCoRe/src/models/context_aware_decoding.py`（NAACL 2024 CAD）。
  - **功能**：标准 `(1+alpha)*logits_ctx - alpha*logits_prior` 对比解码（内置相同 EOS/换行/尾巴早停边界与原始分布遥测）。
- [conformal_risk_operator.py](file:///F:/desktop/Nekoya/nekolab/[SDUFE]GeminiOutOfToken_Team/results/MileStone4_Competitor/reuse/operators/conformal_risk_operator.py)
  - **借鉴自**：`conformal-rag-guardrails/src/models/core_algorithm.py#L455-L478`（CRG）。
  - **功能**：`fit_split_conformal_abstention()` 与 `fit_conformal_risk_control()`，零 GPU 开销直接将 Dev Platt 概率升级为带有限样本风险/覆盖界的共形拒答策略。
- [relevance_gate_operator.py](file:///F:/desktop/Nekoya/nekolab/[SDUFE]GeminiOutOfToken_Team/results/MileStone4_Competitor/reuse/operators/relevance_gate_operator.py)
  - **功能**：轻量 M1 `check_query_evidence_relevance_gate()`（词面重叠下界 + 相似度门控），专门阻断 S4 全无关高同质错误池的低熵误早退。
- [bge_reranker_operator.py](file:///F:/desktop/Nekoya/nekolab/[SDUFE]GeminiOutOfToken_Team/results/MileStone4_Competitor/reuse/operators/bge_reranker_operator.py)
  - **功能**：`IsolatedBGEReranker` 封装 `BAAI/bge-reranker-v2-m3`（568M Cross-Encoder，支持 ModelScope 国内源、单阶段打分后立即 `teardown()` 释放显存，兼容无 GPU `dry_run`）。
- [instruct_rag_icl_operator.py](file:///F:/desktop/Nekoya/nekolab/[SDUFE]GeminiOutOfToken_Team/results/MileStone4_Competitor/reuse/operators/instruct_rag_icl_operator.py)
  - **提取自**：`InstructRAG/src/rag.json` + `src/data_utils.py`（ICLR 2025 免微调 ICL 去噪推理）。

### 3. `scripts/`（原始精简参考脚本）
保留了 8 个原仓库关键单文件参考（`ref_ck_plug_ck.py`, `ref_ck_plug_eval_rag.py`, `ref_cad_context_aware.py`, `ref_crg_core_algorithm.py`, `ref_ql_internal_confidence.py`, `ref_instructrag_inference.py`, `ref_instructrag_prompts.json`, `ref_madam_rag.py`），供写论文 Related Work 或核对超参公式时随时翻阅。
