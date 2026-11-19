# U-GraphRAG V1 审查评判报告与 V2 修复升级对照（实验包 2 基线档案）

> [!CAUTION]
> **审查员对 V1（`第一次实验结果.txt` 与旧归档 `UGraphRAG_Experimental_Results`）的最终裁决：Reject / 不可直接投稿**。
> 本文档如实记录对 V1 实验结果与工程代码的逐项学术审查意见、已被证明真正有效的研究亮点、6 大阻断级硬伤与精确根因溯源，以及 `实验包 2（V2）` 所落地的代码修复与单轮多参数网格寻优升级。

---

## 一、 V1 总体审查评判（纠偏原乐观测评）

### 1. 实验真实性审计：真机实测结果，非伪造数据
- 核查 TensorBoard 事件文件与 [step1_corrected_3b_7b.json](file:///F:/desktop/MyRouter/Nekoya/nekolab/%5BSDUFE%5DGeminiOutOfToken_Team/results/UGraphRAG_Experimental_Results/results/step1_corrected_3b_7b.json) 等原始日志，V1 确实在 `NVIDIA GeForce RTX 5060 Laptop GPU`（8 GB 显存）上真实运行完成了 4-bit NF4 量化的 `Qwen2.5-3B-Instruct` 与 `Qwen2.5-7B-Instruct` 推理全流程（含 HotpotQA 100 题逐样本预测记录）。
- 对侧实验机新增的评估与工程容错工具（`evaluation_toolkit.py` 扩展指标、断点恢复、TensorBoard、自动打包脚本）工程实用性强，已全部同步回本主工程。

### 2. 驳回“可直接投稿”结论的原因
`第一次实验结果.txt` 将 V1 定级为 `[Promising]` 并建议直接冲刺 `KBS / IP&M / ESWA / COLING / CIKM / ECIR / ACL Findings`。站在严肃会议与一区/二区期刊审稿人视角，**V1 版本存在多处“完整系统劣于单模块 / 正文与表格自相矛盾 / 对照组检索不公平 / 校准崩溃”的硬伤，直接投稿几乎必然被拒**：
1. 主实验 HotpotQA-100（7B NF4）上，`Full_UGraphRAG` 的 `EM = 53.00%` **低于纯检索基线** `B0_Standard_RAG`（`54.00%`），且比中间子系统 `M2_Full`（`56.00%`）倒退了 `3.00%`；
2. `M3` 校准解码非但未降低误判风险，反而使 `ECE` 校准误差由 `B0 = 0.0710` / `M2_Full = 0.1335` **暴涨至 `0.2028`**（3B 模型上更从 `0.1702` 恶化至 `0.3010`），且 100 题全部未能触发可控拒答（`abstained = 0 / 100`）；
3. 报告正文硬编码模板多处写出与同页表格相反的“报喜”结论，并混杂 30 题 Pilot 数据与 `RTX 4070` 错配硬件名称；
4. Step 2（PopQA）存在“给 B0 传空桩、只给 M1 查由真值生成的单句语料”的不对称评测缺陷；
5. Step 4 置换检验仅评测 $N=30$，测出随机打乱均值反而高于 DFS 拓扑序，且跨题离散度与同题随机种子标准差口径混用；
6. 三张主结果图存在文字遮挡、图例重叠，且 3B 消融峰值显存被前序 7B 残留水位污染。

---

## 二、 V1 真正站住脚的亮点资产（必须保留与放大）

**问题并非研究假设全盘失败，而是核心有效模块 M2 被失效的 M1 阈值尺度与 M3 对比过锐化反向拖垮。** 在 V1 HotpotQA-100（7B NF4）中，M2 已经产出极强的正向因果实证：

| 对比模式（V1 HotpotQA-100, 7B NF4） | EM (%) | F1 (%) | Sp-F1 (%) | Token 节省率 (%) | 科学归因 |
| :--- | :---: | :---: | :---: | :---: | :--- |
| `B0_Standard_RAG` | 54.00 | 64.68 | 51.00 | 0.0 | 标准 Top-5 混合检索基线 |
| `B1_Naive_MST` | 49.00 | 57.50 | 35.86 | 7.7 | 未做筛选直接建 MST，劣于 `B0`（EM `-5.00%` / F1 `-7.18%`） |
| `M2_Exact_NoMST`（同选集，按分数降序） | 50.00 | 61.97 | 59.45 | 27.7 | 仅筛选不按图拓扑重排，多跳整合受限 |
| **`M2_Full`（两阶段净化 + MST DFS 排序）** | **56.00** | **68.07** | **59.45** | **27.7** | **V1 全场真实最高点** |

* **纯净拓扑因果增益（+6.00% EM / +6.10% F1）**：`M2_Full` 与 `M2_Exact_NoMST` 选出的证据集合完全一致（支持句 `Sp-F1` 同为 `59.45%`、`Token Saving` 同为 `27.7%`），仅因将排序从分数降序改为 MST + DFS 拓扑线性化，EM 由 `50.00%` 跃升至 `56.00%`，F1 由 `61.97%` 跃升至 `68.07%`。
* **相对 Naive MST 碾压优势**：`M2_Full` 相比 `B1_Naive_MST` 提升 **+7.00% EM / +10.57% F1 / +23.59% Sp-F1**，同时压缩上下文约 `27.7%`。

---

## 三、 V1 六大核心硬伤逐项拆解与根因溯源

### 硬伤 1：报告模板硬编码“报喜不报忧”，与同页表格矛盾
- **事实核查**：V1 的 `FULL_EXPERIMENT_REPORT.md` 声称 *“M1-Only 在证据召回上带来显著增长”*，实际同表数据中 `M1_Only` 的 `Sp-Recall` 由 `88.50%` 跌至 `86.00%`、`Sp-F1` 由 `51.00%` 跌至 `49.57%`、`EM` 由 `54.00%` 跌至 `52.00%`；声称 *“Full U-GraphRAG 使 ECE 显著收敛”*，实际 7B `ECE` 由 `0.0710 -> 0.1335 -> 0.2028`（恶化近 3 倍），3B 由 `0.1702 -> 0.3010`（`MCE = 0.5854`）。
- **版本混写与硬件错配**：Step 1 汇总表为 `N=100` 数据（3B/7B 峰值显存 `1978.45 / 5309.91 MB`，EM `41% / 54%`），下方文字却硬编码了旧 30 题 Pilot 数据（`5.56 GB / +10.00% EM / +14.81% F1`）；表注与图注更错写为 `RTX 4070`（实机为 `NVIDIA GeForce RTX 5060 Laptop GPU`）。

### 硬伤 2：M3 校准解码造成“负增益 + 置信度虚高崩盘”
- **量化实证**：在传入完全相同证据图与排序的前提下，`Full_UGraphRAG` 相比 `M2_Full` 导致 `EM` 从 `56.00% -> 53.00%`（`-3.00%`）、`F1` 从 `68.07% -> 65.71%`（`-2.36%`），前向调用翻倍（`1.0 -> 2.0`）、平均时延由 `1.13s` 增至 `1.43s`。100 题中有 `14` 题答案改变，**`EM` 仅改对 1 题却把 4 题原本满分答案改错**，典型翻车：
  - 极性与数值翻转：标准答案 `"yes"` 被改成 `"no"`（置信度 `0.8864`），`"250 million"` 被改成 `"no"`（置信度 `0.7270`）；
  - 短实体被强拉后缀破坏 EM：`"2009 Big 12 Conference"` 被改成 `"2009 Big 12 Conference football season"`（置信度 `0.9160`），`"Keith Bostic"` 被改成 `"William Keith Bostic"`（置信度 `0.8947`）。
- **三重代码根因**：
  1. `beta = 0.35` 对对比对数几率 $L_{\mathrm{cal}} = (1 + \beta)L_{\mathrm{ctx}} - \beta L_{\mathrm{prior}}$ 过度锐化，又直接把锐化后的 softmax 概率混入置信度计算，导致全组平均置信度高达 `0.8828`（远超 `53%` 准确率），`ECE` 飙升至 `0.2028`；而 `abstain_threshold = 0.20`、`confidence_threshold = 0.25` 远低于真实下界，造成 `0 / 100` 拒答；
  2. 先验分布扣减在 `L_prior` 较高时误压 `"yes"`、停用界限与 `EOS` Token，诱发极性反转与冗余拖尾；
  3. `B0..M2_Full` 以文档 `calibrated_reliability` 均值充当置信度，而 `Full_UGraphRAG` 换用未做温度缩放的 token 级概率，跨模式 `ECE` 尺度本身不在同一基准面。

### 硬伤 3：M1“校准-推理尺度断裂”，沦为 `100/100` 全盲扩图
- **量化实证**：V1 的 `models/m1_calibration.json` 经 LLM 探针测得 $\overline{H}_{\mathrm{total}}=0.4262$、$\overline{U}_{\mathrm{EU}}=0.1303$、$\tau_{\mathrm{EU}}=0.1528$；但在主消融流 `u_graphrag_pipeline.py` 运行时未传入 `model_fn`，触发了 32 词对数词频启发式兜底分支，信息熵骤升至 $\approx 3\text{ nats}$，导致实测 HotpotQA-100 的 $U_{\mathrm{EU}} \in [0.7768, 1.8716]$（均值 `1.5529`，达阈值 **10 倍**），最终 **`100 / 100` 题全部强制扩图**，反把无关字面近似块挤进 Top-5，拉低支持句召回（`88.50% -> 86.00%`）。

### 硬伤 4：Step 2（PopQA）语料不对称且核心单调性假设反转
- **公平性漏洞**：V1 中 `B0_Standard_RAG` 仅见到不含答案的实体-属性空白桩，而 `M1 / Full` 则检索由测试题 `ground_truth` 拼接的真值句 `kg_rel_{iid}`，造成 `M1_Only` 与 `Full_UGraphRAG` 在全部 5 个分桶准确率逐位完全相等（`80% / 75% / 55% / 90% / 95%`）、`Token Saving = 89.07%`、`Bloom 召回损失 = 0.0`。
- **单调性反转**：5 桶实测 $U_{\mathrm{EU}}$ 依次为 `0.4773 -> 0.5014 -> 0.4528 -> 0.7296 -> 0.4178`，`10^4 - 10^5` 高流行度桶反而出现全场最高峰 `0.7296`，未提供无泄漏的样本级不确定度判别力指标（如闭卷 $U_{\mathrm{EU}}$ 预测错误率 AUROC）。

### 硬伤 5：M2 贪心与精确求解零区分、Step 4 样本少且方差口径混杂
- `M2_Greedy` 与 `M2_Full` 在 V1 的 100 题中选中文档数与输出答案 `100/100` 完全重合（同为 `56.00% / 68.07%`），缺乏对二者差异率的透明刻画。
- Step 4 仅评测 `N = 30` 样本，`5× Random Shuffles`（`64.51%`）反而略高于 `DFS`（`63.88%`），`Reverse = 62.41%`；且 `tables_latex.tex` 将跨样本标准差（`±44.63 / ±43.99 / ±37.97`）与单题随机置换标准差（`random_shuffles.f1_std = 12.01`）混为一谈，报告正文还把前两项误印为 `±0.00`。

### 硬伤 6：3B 显存计数器未清零与图表遮挡
- `step3_step4_ablation_hotpot100.json` 中 `scaling_3b` 的 `B0` 与 `Full_UGraphRAG` 均误报 `peak_cuda_allocated_mb = 5719.67 MB`（`reserved = 6554.0 MB`），与 7B 组相同——根因是卸载 7B 后漏调 `torch.cuda.reset_peak_memory_stats()`，延续了前序 7B 的历史峰值（Step 1 中 3B 实际峰值仅 `1978.45 MB`）。
- 三张输出图片存在明显遮挡：Fig 1 标题与图例压盖；Fig 2 横轴长名称互相撞车且顶部 `Total Gain` 标签掩盖了 `Full` 低于 `M2_Full` 的事实；Fig 3 的红色 `Max Performance Drop: -15.4%` 方框盖住柱顶数值。

---

## 四、 V2（实验包 2）逐项修复与网格调参升级清单

| 对应 V1 硬伤 | V2 修复模块与文件 | 落地改动内容 |
| :--- | :--- | :--- |
| **硬伤 1（报告失真）** | `scripts/run_full_experiment_and_package.py` | 移除所有硬编码 `RTX 4070` / 30 题 Pilot 数字及固定乐观文案；完全由实测 JSON 动态解析硬件型号、Stage 0 网格参数汇总及 Step 1–4 实际增减归因（正负号如实呈现） |
| **硬伤 2（M3 崩盘）** | `src/m3_calibrated_decoder.py`<br>`src/u_graphrag_pipeline.py` | 1. 修复 `compute_abstention_metrics` 导入与共享温度校准函数 `compute_calibrated_confidence(...)`（温度网格 `0.8~3.0`）；<br>2. 新增高置信度上下文保护 `contrastive_conf_guard` 与 `EOS` Token 免疫机制，防止极性词/短答案被过度对比误伤；<br>3. 改为基于**未锐化原始 `cur_ctx_logits`** 记录多温度轨迹用于置信度与 ECE 校准，并将 SynCheck 前移至拒答门控之前 |
| **硬伤 3（M1 尺度断裂）** | `src/m1_adaptive_retriever.py`<br>`src/u_graphrag_pipeline.py` | 新增 `default_model_fn` 与 `_build_m1_model_fn()`，使 `decompose()`（含 Stage 1 与主实验）严格使用与 `m1_calibration.json` 一致的 LLM Logits 探针尺度计算 $U_{\mathrm{EU}}$，同时设 `use_model_probe_for_candidates = False` 保持扩展阶段 $O(1)$ 低延迟 |
| **硬伤 4（PopQA 不公平）** | `scripts/run_overnight_ablation.py` (`run_experiment_2`) | 重构为**对称共享知识库 `popqa_shared_corpus`** 与共享 CPU BGE `HybridRetriever`：B0（Top-5）与 M1 / Full（轻量初始 $D_0 =$ Top-2）在同一语料池公平检索；新增闭卷推理与闭卷 $U_{\mathrm{EU}}$ 判别盲区 **AUROC** 及 Full 校准 ECE/AUROC 输出 |
| **硬伤 5（Step 4 混乱）** | `scripts/run_overnight_ablation.py` (`run_experiment_3`)<br>`scripts/plot_ablation_charts.py` | `M2_Permutation_Test` 支持 `--perm-limit` 扩展至全量 `100` 题；显式区分并分别记录 **单样本 5 次置换波动 `intra_sample_std`** 与 **跨样本分布标准差 `cross_sample_std`**；Table 3 双列展示并修复双负号缺陷；支持断点版本标记 `v2_grid_calibrated` 与 `--force` |
| **硬伤 6（显存与图表）** | `scripts/run_overnight_ablation.py`<br>`scripts/plot_ablation_charts.py` | `clean_gpu_memory()` 及 7B/3B 每次模式评测前调用 `torch.cuda.reset_peak_memory_stats()`；重构 Fig 1–3 图例外置、防撞角度旋转与画布边距，视觉复核零遮挡；动态解析硬件名 |
| **新增核心（单轮多参网格寻优）** | `scripts/run_parameter_grid_search.py` | 在严格隔离 `[0:100]` 评测集的独立切片（默认 `hotpotqa_dev[100:145]`）上执行「一次模型加载 + 特征/Prompt 缓存」，一次实验高效产出一套帕累托多目标参数集（`balanced_optimal`、`low_ece_high_precision`、`full_coverage_calibrated`），覆盖 $\tau_{\mathrm{EU}}$ 网格、108 组 M2 结构网格及 `2016+` 组 M3 温度/拒答组合 |

---

## 五、 目标 GPU 机（RTX 5060 8GB）重跑验收红线

在目标机重跑完成后，打开新生成的 `results/FULL_EXPERIMENT_REPORT.md` 必须按以下红线核验是否达到可撰写投稿水平：

1. **P0 门禁（必须满足）**：
   - `M1_Only` 不再出现 `100%` 全扩图退化，应在保持有效 `Token Saving` 的同时稳定匹配或优于 `B0`；
   - `Full_UGraphRAG` 的 `EM / F1` 应不再被 M3 拉到低于 `B0`（理想情形 $\ge$ `M2_Full` 或在持平条件下大幅改善可靠性）；
   - `Full_UGraphRAG` 的 `ECE` 需显著下降（理想接近或优于基线 `B0 / M2_Full`），选择性拒答在设定覆盖率下真实压低 `Selective Risk`；
   - `scaling_3b` 的峰值已分配显存应回落至 ~`2.0 GB` 真实区间，不再残留 7B 的 `~5.7 GB` 数值；
   - 图 1–3 与正文中硬件名、样本数、正负号与 JSON 完全一致。
2. **P1 门禁（PopQA 公平线与置换检验）**：
   - PopQA 在共享混合检索库下，`M1_Only` / `Full_UGraphRAG` 相对于 `B0 (Top-5)` 需呈现真实、合理的差距（不再逐桶与 Full 零差异），并核对闭卷 $U_{\mathrm{EU}}$ 预测盲区 `AUROC > 0.5`；
   - Step 4 在 `N=100` 下，`DFS` 拓扑序表现与 `Intra-Sample Seed Std` / `Cross-Sample Std` 数据口径清晰分列。
3. **P2 冲刺顶会建议（可选增强）**：
   - 若冲刺 CCF-A 长文或中科院一区 Top 期刊，在 V2 主线指标全部跑绿后，可进一步将 HotpotQA 扩大至全量 $N=300$（数据文件已内置），补充 2WikiMultihopQA / MuSiQue 跨库检验与外部图 RAG 基线（如 HippoRAG / LightRAG）同台对比并附 Paired Bootstrap 显著性检验。
