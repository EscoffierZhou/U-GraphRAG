# 数据来源、开源许可与隐私脱敏说明（交付物 3）

本文档对应 **Agentic Research Challenge 2026（智能体科研创新挑战赛）Research Package 规范第 3 项：数据来源说明**，完整记录 **U-GraphRAG** 所使用的全部公开学术基准数据集、获取方式、切分协议、开源许可及脱敏合规声明。

---

## 1. 公开基准数据集与获取溯源

本研究全部实验均严格基于公开发布的英文学术问答（QA）与知识冲突基准数据集，不使用任何私有数据、商业敏感数据或包含个人身份信息（PII）的数据集。

| 数据集 | 原始文献 | 获取来源与仓库内路径 | 样本规模与零泄漏切分协议 | 开源许可 |
| :--- | :--- | :--- | :--- | :--- |
| **HotpotQA** | Yang et al. (EMNLP 2018) | 通过 `scripts/download_datasets.py` 与 `scripts/phase2/cache_builder.py` 从 HuggingFace `hotpotqa/hotpot_qa`（`distractor` / `validation`）获取 → `data/hotpotqa_dev_300.json`、`data/phase2/hotpotqa_dev_500.json`、`data/phase2/caches/hotpotqa_pool_k16.json` | 抽取 Dev-500 子集并由 `_assign_split()` 严格划分：`[0:100]` 为 `v5_eval`、`[100:180]` 为 `dev`、`[180:500]` 为 `test`（主评测 `120–160` 题） | CC BY-SA 4.0 / Apache-2.0 |
| **PopQA** | Mallen et al. (ACL 2023) | 通过 `scripts/download_datasets.py` 从 HuggingFace `akariasai/PopQA`（`test.tsv`）获取 → `data/popqa_test_300.json`、`data/phase2/caches/popqa_pool_k16.json` | 按主实体流行度 `s_pop` 划分 5 层（`[0,100)` 至 `[10^5, 10^15)`），使用 `random_state=42` 分层均衡抽样 300 题；通过 `build_popqa_neutral_corpus(mode="natural")` 构建无模板锚点泄漏的自然语言维基背景库 | MIT License |
| **2WikiMultiHopQA** | Ho et al. (COLING 2020) | 通过 `scripts/phase2/cache_builder.py` 从 HuggingFace `framolfese/2WikiMultihopQA` 获取 → `data/phase2/2wikimultihop_300.json`、`data/phase2/caches/2wikimultihop_pool_k16.json` | 300 题严格划分为 `[0:80]` `dev` 与 `[80:300]` `test`（220 条测试集） | Apache-2.0 |
| **MuSiQue** | Trivedi et al. (TACL 2022) | 通过 `scripts/phase2/cache_builder.py` 从 HuggingFace `dgslibisey/MuSiQue` 获取 → `data/phase2/musique_300.json`、`data/phase2/caches/musique_pool_k16.json` | 300 题严格划分为 `[0:80]` `dev` 与 `[80:300]` `test`（220 条测试集） | CC BY 4.0 |
| **RAMDocs** | Wang et al. (arXiv:2504.13079, MADAM-RAG) | `results/MileStone4_Competitor/reuse/datasets/RAMDocs_test.jsonl`，由 `dataset_adapters.py` 中的 `load_ramdocs_dataset()` 统一适配 | 共 500 条真实多文档“正确 / 误导冲突 / 无关噪声”混合 QA（单卡 `medium` 赛马评测抽取 200 题，`seed=42`） | 学术开源发布 |
| **NQ-Swap** | Longpre et al. (EMNLP 2021)（源自 DeCoRe 发布包） | `results/MileStone4_Competitor/reuse/datasets/nqswap_dev.parquet`，由 `load_nqswap_dataset(mode="dual_conflict")` 统一适配 | 共 4,746 条反事实实体替换问答对（含 `sub_context` 与冲突的 `org_context` 及噪声干扰文档；`medium` 评测抽取 200 题） | CC BY-SA / 学术研究 |

---

## 2. 受控检索池与压力测试切片构建原则

1. **离线冻结共享 Top-$K$ 检索池（$K=16$）**：使用 `BM25Retriever` 与 `BAAI/bge-small-en-v1.5`（384 维稠密向量，模型元数据见 `data/DATASET_MANIFEST.json`）通过倒数排序融合（RRF）预先计算，确保所有基线方法与 U-GraphRAG 变体在完全一致的候选证据池上公平对比。
2. **受控冲突注入与标签隔离**：`scripts/phase2/cache_builder.py` 中的 `inject_numeric_or_entity_conflict()` 仅用于生成 `hotpotqa_conflict` 压力测试冲突切片；标记字段 `is_synthetic_conflict` **仅用于离线统计 Conflict Retention 指标，严禁暴露给 M1/M2/M3 推理管线**。
3. **S4 全无关饥饿切片**：通过 `build_all_irrelevant_s4_slice()` 将金标文档替换为无关干扰文档，用于审计模型在纯无关低熵池中的盲目早退率与选择性拒答安全性。
4. **四象限诊断探针集**：`data/phase2/caches/wp1_four_quadrant_probe.json` 将查询-证据对划分为 `Q1_known_consistent`、`Q2_unknown_sufficient`、`Q3_known_conflicting`、`Q4_unknown_missing` 四个受控象限以检验 M1 不确定性行为。

---

## 3. 隐私脱敏、数据合规与参数冻结校验

- **零个人隐私信息（Zero PII）**：所有语料均源自公开 Wikipedia / Wikidata / Natural Questions 学术基准，不含任何个人隐私、学籍账务、医疗病历或未经授权的网络爬取隐私数据。
- **凭据清理**：代码库与数据集中不包含任何 API 密钥、GitHub Token（PAT）、HuggingFace Token 或服务器连接密码。
- **V5 冻结参数哈希校验**：`results/MileStone2_AblationExperiment_I/V5_FROZEN/SHA256SUMS.txt` 记录了 `61` 份核心脚本、数据与消融结果的 SHA-256 校验和，保证从 Milestone 2 到 Milestone 3/4 的零数据泄漏与参数不可篡改性。
