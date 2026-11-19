# 可复现说明与运行指南（交付物 7）

本文档对应 **Agentic Research Challenge 2026（智能体科研创新挑战赛）Research Package 规范第 7 项：可复现说明**，提供完整的环境依赖、随机种子、`V5_FROZEN` 完整性校验、离线 CPU 测试命令以及 GPU 全量实验 / LaTeX 论文复现流程。

---

## 1. 已验证软硬件环境

1. **本地开发与 CPU 单元测试环境**（[requirements.txt](../requirements.txt)）：
   - 操作系统与硬件：Windows 11，NVIDIA RTX 4070 Laptop（8 GB VRAM）。
   - Python `3.12.10` + CUDA `12.6`。
   - 核心库版本：`torch>=2.5.0`（实测 `2.9.0+cu126`）、`transformers>=4.50.0`（实测 `4.57.3`）、`bitsandbytes>=0.45.0`（实测 `0.48.2`）、`accelerate>=1.0.0`、`sentence-transformers>=3.0.0`、`scikit-learn>=1.3.0`、`numpy>=1.26.4`、`scipy>=1.13.0`、`networkx>=3.0`、`matplotlib>=3.8.4`、`pandas>=2.2.0`、`datasets>=3.0.0`、`rank-bm25>=0.2.2`。
2. **云端外部验证 GPU 环境**（[reproducibility_manifest.json](../evidence/milestone3/reproducibility_manifest.json)）：
   - 操作系统与硬件：`Linux-5.4.0-196-generic-x86_64-with-glibc2.35`，NVIDIA GeForce RTX 4090（`24080.9 MB` / `24564 MiB` 显存）。
   - 软件栈：Python `3.12.4`、`torch 2.15.0+cu132`、`transformers 5.23.0`、`bitsandbytes 0.50.2`、`sentence-transformers 6.3.0`、`scipy 1.18.1`、`datasets 5.2.0`。

```bash
# 建议先安装匹配本机驱动的 PyTorch CUDA 版，再安装其余依赖
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install -r requirements.txt
```

---

## 2. 随机种子与冻结基线完整性校验

- **全局默认随机种子**：`42`（多轮压力曲线与稳定性验证使用种子集合 `42 1024 2026`）。
- **冻结 V5 参数校验**：所有 Phase 2 / Phase 3 / `MileStone_After` 脚本均会自动调用 `verify_v5_frozen_checksums()` 核对 `results/MileStone2_AblationExperiment_I/V5_FROZEN/SHA256SUMS.txt` 中的 **61 份核心文件 SHA-256 校验和**（`status: ok, verified_count: 61`），确保超参数预设 `balanced_optimal` 未发生篡改。

---

## 3. 零 GPU 离线单元测试与 `--dry-run` 全流程验证（57 / 57 全部通过）

在仓库根目录下直接执行以下命令即可在纯 CPU 环境下完成全部 7 组单元测试与端到端冒烟验证（**共 57 项测试，实测 57/57 通过，无跳过、无失败**）：

```bash
python tests/test_m1_adaptive_retriever.py                        # 11 tests：熵分解非负性、MinHash、Kirsch-Mitzenmacher 布隆过滤器、门控早退
python tests/test_m2_optimizer.py                                 # 12 tests：可靠性校准、朴素 MST 毒瘤节点反例、精确枚举 vs 贪心对比
python tests/test_v4_fixes.py                                     # 7 tests：PopQA 语料去偏中和化与绘图检查
python tests/test_phase2_p0.py                                    # 5 tests：P0 闭环修复与 V5_FROZEN 61 份文件 SHA-256 校验
python tests/test_phase2_wp.py                                    # 5 tests：WP1-WP5 校准拟合（仅用 dev 集）与指标核查
python tests/test_phase3.py                                       # 13 tests：WP6-WP11 AutoDL 管线与论文数据包 --dry-run 端到端验证
python results/MileStone4_Competitor/dist/test_milestone_after.py # 4 tests：RAMDocs/NQ-Swap 适配、CK-PLUG/CAD/InstructRAG/BGE 算子、共形拒答与 S4 门控
```

---

## 4. 完整 GPU 实验复现步骤

### 第一步：准备公开基准数据与稠密检索向量模型

```bash
# 下载 HotpotQA dev-300、PopQA 分层抽样 test-300、BAAI/bge-small-en-v1.5 并生成 DATASET_MANIFEST.json
python scripts/download_datasets.py

# 构建 Phase 2/3 共享冻结 K=16 混合检索池（支持 HuggingFace 与 hf-mirror 自动回退）
python -m scripts.phase2.cache_builder --datasets hotpotqa popqa 2wikimultihop musique --download
```

### 第二步：复现 `MileStone 3`（WP6–WP11，单卡 RTX 4090 / ≥8GB 显存 4-bit 模式）

```bash
# 环境冒烟校验（支持 ModelScope / HF Mirror 下载并检查单条生成与 EOS 终止）
bash scripts/phase3/run_autodl_phase3.sh smoke

# 执行完整 WP8 -> WP6 -> WP7 -> WP9/10 -> WP11 外部验证矩阵
bash scripts/phase3/run_autodl_phase3.sh all medium core3 "42 1024 2026"
```

### 第三步：复现 `MileStone_After`（2024–2026 顶会竞品对标与 Split Conformal 风险控制）

```bash
# 自动依次运行 CPU 事后共形风险审计 + BGE-Reranker-v2-m3 / CK-PLUG / InstructRAG-ICL / CAD / U-GraphRAG+ 五基准对比
bash results/MileStone4_Competitor/dist/run_milestone_after.sh llama3_8b medium 4bit
bash results/MileStone4_Competitor/dist/run_milestone_after.sh qwen3b    medium 4bit
bash results/MileStone4_Competitor/dist/run_milestone_after.sh qwen7b    medium 4bit
```

---

## 5. 中英文论文 PDF 编译指南（`paper/`）

```bash
cd paper
# 编译英文论文（25 页）
pdflatex -interaction=nonstopmode ugraphrag_full_paper_en.tex
bibtex ugraphrag_full_paper_en
pdflatex -interaction=nonstopmode ugraphrag_full_paper_en.tex
pdflatex -interaction=nonstopmode ugraphrag_full_paper_en.tex

# 编译中文论文（27 页，需安装 TeX Live xelatex + ctex 宏包与中文字体）
xelatex -interaction=nonstopmode ugraphrag_full_paper_zh.tex
bibtex ugraphrag_full_paper_zh
xelatex -interaction=nonstopmode ugraphrag_full_paper_zh.tex
xelatex -interaction=nonstopmode ugraphrag_full_paper_zh.tex
```
