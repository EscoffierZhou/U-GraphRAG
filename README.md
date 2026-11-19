# U-GraphRAG：基于不确定性路由、冲突感知证据选择与校准解码的可靠检索增强生成

**山东财经大学 `GeminiOutOfToken_Team` — 智能体科研创新挑战赛（Agentic Research Challenge 2026）官方 Research Package**  
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)](https://www.python.org/)
[![Unit Tests](https://img.shields.io/badge/CPU%20Tests-57%2F57%20Passing-brightgreen.svg)](docs/07_REPRODUCIBILITY.md)
[![License: Academic](https://img.shields.io/badge/License-Apache%202.0%20%2F%20MIT-lightgrey.svg)](docs/03_DATA_PROVENANCE.md)

- **团队成员与贡献者**：

- 周兴（`EscoffierZhou`，`zhouxing@sdufe.edu.cn`）、宋丽（`LindaSong0315`）、Nekoya(Gemini猫屋)（`EscoffierWithNeko`）
- **所属单位**：

- 山东财经大学 计算机与人工智能学院（School of Computing and Artificial Intelligence, Shandong University of Finance and Economics）
- **代码仓库地址**：[https://github.com/EscoffierZhou/U-GraphRAG](https://github.com/EscoffierZhou/U-GraphRAG)

---

## 1. Research Package 八大交付物索引（Agentic Research Challenge 2026 规范）

本仓库与本地同步压缩包（`U-GraphRAG_Research_Package_2026.zip`）严格对应大赛 **Research Package 规范全部 8 项要求**：

| 序号 | 竞赛交付规范项 | 仓库对应路径与内容说明 |
| :-: | :--- | :--- |
| **1** | **论文原型 PDF** | [paper/ugraphrag_full_paper_en.pdf](paper/ugraphrag_full_paper_en.pdf)（英文版 25 页）与 [paper/ugraphrag_full_paper_zh.pdf](paper/ugraphrag_full_paper_zh.pdf)（中文版 27 页），附完整 LaTeX 源码、`ugraphrag_references.bib`、样式文件与高清插图 `paper/figures/` |
| **2** | **代码仓库** | [src/](src/) 核心模块、[scripts/](scripts/) 自动化实验流水线、[tests/](tests/) 测试套件、[results/MileStone4_Competitor/dist/](results/MileStone4_Competitor/dist/) 与 [reuse/](results/MileStone4_Competitor/reuse/) 竞品对比算子、[requirements.txt](requirements.txt) |
| **3** | **数据来源说明** | [docs/03_DATA_PROVENANCE.md](docs/03_DATA_PROVENANCE.md)（HotpotQA、PopQA、2WikiMultiHopQA、MuSiQue、RAMDocs、NQ-Swap 获取方式、切分协议、开源许可与零 PII 脱敏声明） |
| **4** | **实验证据** | [evidence/EVIDENCE_INDEX.md](evidence/EVIDENCE_INDEX.md)、[evidence/milestone3/](evidence/milestone3/)、`WP6–WP10` 原始检查点、[evidence/milestone4_after/](evidence/milestone4_after/) 竞品赛马汇总与 [evidence/figures/](evidence/figures/)（`fig2`–`fig8`） |
| **5** | **文献引用核验表** | [docs/05_CITATION_VERIFICATION.md](docs/05_CITATION_VERIFICATION.md)（对 `ugraphrag_references.bib` 全部 **44/44** 篇文献逐一核验标题、作者、会议/arXiv 编号） |
| **6** | **Agent 使用日志** | [docs/06_AGENT_USAGE_LOG.md](docs/06_AGENT_USAGE_LOG.md)（按 5 大阶段客观记录 AI Agent 工具使用与安全护栏，已清除全部敏感凭据） |
| **7** | **可复现说明** | [docs/07_REPRODUCIBILITY.md](docs/07_REPRODUCIBILITY.md)（依赖环境、随机种子、`V5_FROZEN` 61 份 SHA-256 校验、57/57 CPU 单元测试命令与 RTX 4090 / LaTeX 复现指南） |
| **8** | **人机分工声明** | [docs/08_HUMAN_AI_CONTRIBUTION.md](docs/08_HUMAN_AI_CONTRIBUTION.md)（人类科研团队与 AI 智能体各自完成工作的权责清单与诚信声明） |

---

## 2. 核心架构概述（`M1` + `M2` + `M3`）

**U-GraphRAG** 提出了一套无需微调、可在单张消费级 GPU（4-bit NF4 峰值保留显存仅 `2.39 GB`–`6.02 GB`）上运行的轻量级三阶段高可靠 RAG 框架：

1. **M1：不确定性路由与关联下界安全门控检索器**（`src/m1_adaptive_retriever.py`、`RelevanceLowerBoundGate`）
   - 基于操作性认知路由信号 $U_{\mathrm{EU}}^{v5}$ 动态决定早退或扩展；引入 MinHash 局部敏感哈希碰撞图与 **Kirsch-Mitzenmacher 双哈希布隆过滤器** 进行高效去重扩展；并通过轻量级 `check_query_evidence_relevance_gate()` 阻断纯无关证据池（`S4`）下的低熵盲目早退。
2. **M2：重叠门控冲突消解与预算受限证据精选**（`src/m2_evidence_optimizer.py`）
   - 采用 **`overlap_gated_js`** 构造交互图，在相同实体/主题局部邻域内严厉惩罚矛盾片段，同时对实体不交的多跳推理“桥接文档”豁免冲突惩罚（`WP8.2`），再经 Kruskal 最大生成树（MST）与预算子集优选提取高信噪比证据组合（15 选 5 仅需约 `1.96–5.5 ms` CPU 时间）。
3. **M3：忠实度感知校准与 Split Conformal 选择性拒答**（`src/m3_calibrated_decoder.py`、`conformal_risk_operator.py`）
   - 联合词元长度归一化对数概率（`w_tok`）、无偏基准锚定与 Platt 缩放，并进一步结合无分布假设的 Split Conformal 分位数标定阈值 $\tau_{\mathrm{conf}}$，实现可控的风险–覆盖率权衡。

---

## 3. 核心实证结果速览

| 实验模块 | 模型与基准数据集 | 核心实证指标对比 |
| :--- | :--- | :--- |
| **WP6 跨模型家族泛化** | `Qwen2.5-7B` @ `HotpotQA` | F1 达 **`60.68%`**（`ECE = 0.0815`），高于 `B0 Hybrid`（`52.86%`）与 `B3 MMR`（`58.12%`） |
| **WP6 跨模型家族泛化** | `Llama-3.1-8B` @ `PopQA` / `MuSiQue` | `PopQA` 达 **`89.08%`**（vs `B0 76.08%`）；`MuSiQue` 达 **`29.73%`**（vs `B0 18.75%`） |
| **`MileStone_After` S4 安全门控** | `Qwen2.5-3B` / `Llama-3.1-8B` / `Qwen2.5-7B` | 盲目早退率分别由 **`85.83%`**、**`9.17%`**、**`3.33%`** 全部降至 **`0.0%`** |
| **`MileStone_After` 反事实冲突** | `NQ-Swap` (`dual_conflict`, `Qwen2.5-7B`) | F1 达 **`42.09%`**（`ECE = 0.1704`），远超 `B0`（`9.33%`）、`BGE-MMR`（`8.18%`）、`CK-PLUG`（`11.46%`）、`CAD`（`13.53%`） |
| **`MileStone_After` 多文档歧义** | `RAMDocs` (`Qwen2.5-7B`) | **`69.40%` Raw F1** / **`63.00%` EM** / **`0.1198` ECE** / **`75.58%` SelAcc**（优于 `B0 65.59%`、`BGE-MMR 63.94%`） |

---

## 4. 快速开始：CPU 离线验证（57 / 57 全部通过）

```bash
pip install -r requirements.txt

python tests/test_m1_adaptive_retriever.py
python tests/test_m2_optimizer.py
python tests/test_v4_fixes.py
python tests/test_phase2_p0.py
python tests/test_phase2_wp.py
python tests/test_phase3.py
python results/MileStone4_Competitor/dist/test_milestone_after.py
```

完整数据集配置、RTX 4090 GPU 实验脚本与中英文论文 LaTeX 编译方法请参阅 [docs/07_REPRODUCIBILITY.md](docs/07_REPRODUCIBILITY.md)。
