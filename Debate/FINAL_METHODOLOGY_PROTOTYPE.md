# U-GraphRAG V1.1 — Research Prototype 方法论原型与实验设计规范
## 面向受限计算资源的不确定性解耦自适应检索、两阶段证据集合优化与校准生成系统
### Uncertainty-Decoupled Adaptive Retrieval, Two-Stage Evidence Set Optimization, and Calibrated Generation under Constrained Resources

> **文档定位与状态**：`[SDUFE] GeminiOutOfToken_Team` 选题方法论原型与实验设计规范（V1.1 Research Prototype）。  
> **版本标识**：`U-GraphRAG V1.1 (Research Prototype)`  
> **冻结与未冻结边界声明**：本文档基于本地 `mempalace_acl/index.sqlite` 真实文献题录与布尔查重事实，吸收学术审阅意见进行重构。**当前阶段仅冻结核心研究问题、三层可靠性中心架构方向、数据集靶场矩阵与渐进式消融研发路线；不冻结尚未通过真实代码与端到端实验闭环验证的具体算子细节与超参配置**（所有未实测性能指标均为待验证目标）。  
> **硬件与预算约束**：单卡 8GB VRAM (`NVIDIA RTX 3070 / RTX 4060`，`Qwen2.5-7B-Instruct` NF4 4-bit 量化 / CPU 混合推断) · 零商业 LLM API Token 预算。  
> **学术规范与科研诚信底线**：严禁伪造学术概念、臆测实验结论或编造未发表引文标识；严禁将未经验证的数学假说标注为定理；所有源算子均与顶会原版双栏论文、公式和本地题录严格对应。

---

## 一、 核心定位与方法论总纲

### 1.1 中英文规范题名
- **中文题名**：`U-GraphRAG`：面向受限计算资源的不确定性解耦自适应检索、两阶段证据集合优化与校准生成系统
- **英文题名**：*U-GraphRAG: Uncertainty-Decoupled Adaptive Retrieval, Two-Stage Evidence Set Optimization, and Calibrated Generation under Constrained Resources*

### 1.2 核心主张与创新收窄（Core Thesis）
在单卡 8GB 显存与零 LLM-API Token 约束的受限资源场景下，传统 RAG 面临“盲目扩检消耗算力、候选证据语义冗余与事实对抗冲突、自回归解码先验常识幻觉”的三重失真。  
**U-GraphRAG 绝不主张“绝对首创了不确定性或生成树算法”**，而是**严格将核心创新聚焦于：在受限计算资源 RAG 中，将任务校准的表征不确定性解耦信号、候选证据集合级可靠性组合优化（M2-A）与图结构引导编排（M2-B）有机连接，并在串行生成端注入全局可靠性先验，系统验证其在遏制幻觉上的端到端增益**。

系统由三层闭环架构组成：
1. **M1（检索层 · 自适应扩展）**：采用免大模型参数微调（No LLM Parameter Fine-tuning）的因果注意力头干预，在线测定全熵与偶然预测歧义度（AU），闭式解耦出认知不确定度（EU）。仅在面临知识盲区（$U_{\text{EU}} > \tau_{\text{EU}}$）时长尾触发基于离线 MinHash 碰撞图的自适应扩展，常识明确时早停，节约算力与显存；
2. **M2（组织层 · 核心主创新 · 两阶段重构）**：针对候选证据集合，解耦单块非负可靠性得分；
   - **M2-A 阶段（预算证据集合优化，Budgeted Evidence Set Selection）**：针对标准生成树“只能删边、无法过滤低可靠节点”的本质缺陷，在候选池（$|\mathcal{D}_{\text{cand}}| \le 15$）中显式定义集合目标函数（融合可靠性、覆盖度、互补性，重罚语义冗余与 JS 散度预测冲突），在 CPU 端进行毫秒级组合子集优化（$K \le 5$），筛选出高质证据子集 $\mathcal{D}^*$；
   - **M2-B 阶段（生成树拓扑编排，MST-Guided Organization）**：仅在 M2-A 提炼的黄金子集 $\mathcal{D}^*$ 上构图，求解加权最大生成树（MST）提取骨干拓扑依赖，并以最大权边为根执行 DFS 遍历线性化，提供抗“Lost-in-the-Middle”干扰的紧凑上下文；
3. **M3（生成层 · 可靠性校准与渐进门控）**：严格对齐 CLeHe 原文规范，将 M2 的全局可靠性先验与动态高熵参考层注入几何平均对比解码公式；在生成前设置证据集拒答门控，在句级渐进接入轻量 SynCheck 忠实度监控与选择性回溯。

---

## 二、 真实文献核查与创新边界认证（Novelty & Related Work Boundary）

基于本地解析的 **10,020 篇 ACL/EMNLP/NAACL 顶会全文 XML 数据库**（`mempalace_acl/index.sqlite`）与跨平台学术索引，团队对系统核心概念与两两组合执行了查重审计，纠正前期文献误标并确立清晰创新边界。

### 2.1 查重审计实测矩阵（纠正与真实核验）

| 查询 ID | 检索层级与描述 | 核心布尔检索表达式 | 本地 ACL | 跨平台主流索引 | 审计核验结论与对应文献 |
| :---: | :--- | :--- | :---: | :---: | :--- |
| **Q1** | **全链路核心交集** | `("Retrieval-Augmented Generation" OR "RAG") AND ("aleatoric uncertainty" OR "epistemic uncertainty") AND ("maximum spanning tree" OR "MinHash")` | **0** | **0** | **Novelty Validated**：无全链路直接重合系统 |
| **Q2** | **M1 核心交集** | `("Retrieval-Augmented Generation" OR "RAG") AND ("epistemic uncertainty") AND ("MinHash" OR "locality-sensitive hashing" OR "bloom filter")` | **0** | **0** | **Novelty Validated**：无不确定性直接触发哈希图工作 |
| **Q3** | **M2 核心交集** | `("Retrieval-Augmented Generation" OR "RAG") AND ("aleatoric uncertainty" OR "epistemic uncertainty") AND ("maximum spanning tree" OR "MST") AND ("conflict" OR "redundancy")` | **0** | **0** | **Novelty Validated**：无不确定性+生成树结合抗冲突工作 |
| **Q4** | **两两模块：AU/EU in RAG** | `("Retrieval-Augmented Generation" OR "RAG") AND ("aleatoric uncertainty") AND ("epistemic uncertainty")` | 2 | 3 | **相关文献存在**：包含 `2026.acl-long.95` (ICL 不确定性量化)、`2025.naacl-long.453` (DUR 动态不确定性排序) |
| **Q5** | **两两模块：MST in RAG** | `("Retrieval-Augmented Generation" OR "RAG" OR "GraphRAG") AND ("maximum spanning tree")` | 1 | 1 | **图算法先例**：真实命中 `2025.naacl-long.449` (KG2RAG，修正前期 Q5 漏报错误) |
| **Q6** | **两两模块：Hash Graph RAG** | `("Retrieval-Augmented Generation" OR "RAG") AND ("MinHash" OR "hash collision") AND ("graph")` | 1 | 1 | **基线来源**：真实命中 `2026.acl-long.1156` (MeshRAG) |
| **Q7** | **全链路强约束检索** | `("Retrieval-Augmented Generation" OR "RAG") AND ("aleatoric" OR "epistemic") AND ("maximum spanning tree") AND ("contrastive decoding" OR "abstention")` | **0** | **0** | **Novelty Validated**：无整体架构重合先验 |
| **Q8** | **M3 句级忠实度监测** | `("Retrieval-Augmented Generation" OR "RAG") AND ("contrastive decoding" OR "entropy") AND ("selective abstention" OR "synchronous faithfulness")` | 1 | 1 | **基线来源**：真实命中 `2024.emnlp-main.527` (Synchronous Faithfulness Monitoring, SynCheck/FOD) |
| **Q9** | **M2-M3 接口检索** | `("Retrieval-Augmented Generation" OR "RAG") AND ("evidence reliability" OR "uncertainty") AND ("contrastive decoding")` | 1 | 1 | **基线来源**：真实命中 `2025.naacl-long.236` (Entropy-Based Decoding, CLeHe) |
| **Q10** | **对比解码界定** | `("Retrieval-Augmented Generation" OR "RAG") AND ("contrastive decoding") AND ("hallucination" OR "faithfulness")` | 2 | 5 | **成熟基线**：已纳入对比基线体系 |
| **Q11** | **拒答门控界定** | `("Retrieval-Augmented Generation" OR "RAG") AND ("uncertainty") AND ("abstention" OR "unanswerable")` | 2 | 16 | **成熟机制**：基于风险-覆盖（Risk-Coverage）曲线校准 |
| **Q12** | **注意力头干预解耦** | `("function vector" OR "self-function vector" OR "attention head intervention") AND ("Retrieval-Augmented Generation" OR "RAG") AND ("uncertainty")` | **0** | **0** | **假说迁移验证**：借鉴表征干预思路，属于待实验验证的机制假说 |

### 2.2 核心创新边界与高相关文献机制辨析（Related Work Boundary）

严禁宣称“AU/EU/MST/JSD 无人研究”。必须将 U-GraphRAG 与近期最相关工作划清界限并明确继承与超越关系：

1. **与不确定性感知 RAG（`ERA 2026`、`UA-RAG 2026`、`DUR 2025.naacl-long.453`）的边界**：
   - *先验工作机制*：`ERA` 在 RAG 中显式区分 AU 和 EU，量化模型知识与检索知识的冲突；`UA-RAG`（2026）利用内部表征不稳定性与输出分布歧义指导检索路由；`DUR`（`2025.naacl-long.453`）采用有监督强化学习动态分配检索预算。
   - *我方收窄主张*：上述工作主要停留在检索触发或粗粒度路由器决策层，未将解耦后的 EU/AU 信号显式传递至下游证据子集组合优化中。我方将首 Token 浅中层注意力头干预测出的 EU 作为极简自适应门控，并将单块 AU 作为下游集合优化的负权惩罚项，实现免微调、轻量化的跨层信号传递。
2. **与生成树及知识图谱 RAG（`KG2RAG 2025.naacl-long.449`、`GRAFT-RAG`）的边界**：
   - *先验工作机制*：`KG2RAG`（`2025.naacl-long.449`）在显式实体关系图的连通分量上使用最大生成树（MST）剪除冗余边，边权基于相似度打分。
   - *我方收窄主张*：第一，我方彻底摒弃重型实体关系抽取，基于免大模型调用的纯文本 MinHash 碰撞图；第二，深刻指出**“标准生成树只能剪边、无法过滤低可靠文档节点”**的图论固有局限，提出**“M2-A 预算集合筛选 + M2-B 生成树拓扑编排”两阶段方案**；第三，在边权中显式惩罚两两证据的条件预测冲突（JS 散度），而非仅依据字面相关度。
3. **与分布分歧分析及知识冲突诊断（`ARC-JSD 2026`、`CDD 2026`）的边界**：
   - *先验工作机制*：`ARC-JSD`（ICLR 2026）使用 Jensen-Shannon 散度度量上下文片段对回答的重要性；`CDD`（2026）通过分布分歧诊断参数化知识与检索证据的冲突。
   - *我方收窄主张*：JSD 作为经典的对称分布散度，本身并非我方首创。我方明确将 JS 散度定位于**“度量不同候选证据导致模型首 Token 预测分布分歧的代理指标”**，并明确其局限性（分布分歧不完全等价于 NLI 事实矛盾），在 M2-A 目标函数中作为事实冲突惩罚项，并设立专门的受控消融实验进行可信度核验。
4. **与零 Token 哈希图基线（`MeshRAG 2026.acl-long.1156`）的边界**：
   - *先验工作机制*：`MeshRAG` 利用 MinHash 与双层哈希社区实现零 API Token 的图索引构建，但其检索属于无差别全量哈希扩展，容易引入高度同质化的冗余社区或噪音块。
   - *我方收窄主张*：以 $U_{\text{EU}}$ 作为门控条件按需触发，避免无盲区时的无效扩检；扩展后的证据池交由 M2 两阶段优化器进行显式冗余剔除与事实冲突隔离，并在主实验中严格汇报布隆过滤器剪枝造成的召回率损失（Recall Drop under Bloom Pruning）。
5. **与对比解码与忠实度监控（`CLeHe 2025.naacl-long.236`、`SYNCHECK 2024.emnlp-main.527`）的边界**：
   - *先验工作机制*：`CLeHe`（`2025.naacl-long.236`）基于各证据在当前步的局部熵进行归一化加权几何平均解码；`SYNCHECK`（`2024.emnlp-main.527`）在句边界训练轻量分类器进行同步忠实度监控。
   - *我方收窄主张*：`CLeHe` 假设所有证据具有无差别的先验可靠性。我方将 M2 产出的全局拓扑可靠性先验 $\tilde{R}_i$ 注入 CLeHe 对数概率公式，形成“全局先验 + 局部动态熵”的几何平均加权；同时在单卡 8GB 显存下严格采用串行实现，将 SynCheck/FOD 明确作为渐进式可选增强模块，拒绝伪集成。

---

## 三、 源算子溯源与真实文献元数据（Source Operator Traceability）

严格基于本地 `mempalace_acl/index.sqlite` 数据库与顶会原版双栏 PDF，杜绝虚构篇名与臆测 ID：

| 算子名称 | 知识库短码 | 真实顶会来源与文献篇名 | 原文精确出处 | 原文核心方法与支撑点 | 对应痛点咬合 |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Self-FV 假说**<br>*(因果头干预解耦)* | `W3-R06` | ACL 2026<br>*"Quantifying Aleatoric Uncertainty of In-Context Learning for Robust Measure of LLM Prediction Confidence"* (`2026.acl-long.95`) | PDF p. 3–5<br>Section 3.1–3.3<br>Eq. (4)–(7) | 通过表征干预与输出分布解耦衡量 AU，注意力头因果效应分析 | **咬合痛点 D3**：模型无法感知自身盲区，缺乏自适应检索触发；通过浅中层干预快速解耦 AU/EU。 |
| **MeshRAG**<br>*(零 Token 哈希图)* | `W3-R07` | ACL 2026<br>*"Collision to Cognition: Hash-Driven Graph Construction for Efficient RAG"* (`2026.acl-long.1156`) | PDF p. 3–5<br>Section 3.2–3.3<br>Algorithm 1 | MinHash 签名 + Bloom Filter 社区索引，零 LLM API Token 离线建图 | **咬合痛点 D1**：传统图 RAG 严重依赖昂贵 API；本地哈希碰撞实现 0 预算图扩展。 |
| **DUR**<br>*(动态不确定性基线)* | `W3-R05` | NAACL 2025<br>*"Dynamic Uncertainty Ranking: Enhancing Retrieval-Augmented In-Context Learning for Long-Tail Knowledge in LLMs"* (`2025.naacl-long.453`) | PDF p. 4–5<br>Section 3.2–3.3<br>Eq. (4)–(10) | 基于模型不确定度排序的长尾实体检索增强，有监督强化学习动态预算 | **对比基线**：作为 M1 动态自适应扩展思想的对比与灵感来源。 |
| **KG2RAG**<br>*(生成树拓扑骨干)* | `W3-R07` | NAACL 2025<br>*"Knowledge Graph-Guided Retrieval Augmented Generation"* (`2025.naacl-long.449`) | PDF p. 4–5<br>Section 3.3<br>Eq. (6)–(8) | 在实体关系图连通分量上使用 MST 剪除冗余边，以主干边为根排序展平 | **咬合痛点 D2**：传统检索上下文冗余度高；借鉴 MST 骨干提炼并扩展为抗事实冲突。 |
| **CLeHe**<br>*(多证据对比解码)* | `W3-R05` | NAACL 2025<br>*"Entropy-Based Decoding for Retrieval-Augmented Large Language Models"* (`2025.naacl-long.236`) | PDF p. 4<br>Section 3.2<br>Eq. (7) | 基于多证据前向概率与局部熵几何平均，结合动态高熵参考层对比解码 | **咬合痛点 D2**：解码端常识偏见与多证据干扰；利用可靠性加权几何平均对比消除幻觉。 |
| **SYNCHECK & FOD**<br>*(同步监测与回溯)* | `W1-R01` | EMNLP 2024<br>*"Synchronous Faithfulness Monitoring for Trustworthy Retrieval-Augmented Generation"* (`2024.emnlp-main.527`) | PDF p. 3–5<br>Section 2.2–2.3<br>Eq. (3)–(6) | 句级轻量分类器实时监控忠实度，FOD 局部 Beam Search 回溯重采 | **咬合痛点 D2/D3**：后置验证耗时长；作为 M3 可选顶层增强，不达标则截断拒答。 |

---

## 四、 U-GraphRAG 三大核心模块闭式数学原型与重构规范

系统整体架构流转如下图所示：

```text
Query q ──▶ 初始轻量混合检索 (BM25 + Dense Top-K0, K0=3) ──▶ 初始锚点集 D_0
                 │
                 ▼
 [M1: Epistemic Uncertainty Decoupling & Adaptive Hash Expansion]
   ├─ 单次注意力头干预 Prefill (N=1) ──▶ 测定 H(q, D_0) 与 U_AU(q, D_0) ──▶ 解耦 U_EU(q, D_0)
   ├─ 门控判定:
   │    ├─ 若 U_EU <= tau_EU: 认知明确 ──▶ 早停，D_cand = D_0 (节省后续 100% 图检索开销)
   │    └─ 若 U_EU >  tau_EU: 存在盲区 ──▶ 激活 MeshRAG 零 Token 哈希图扩展 ──▶ 候选池 D_cand (|D_cand| <= 15)
                 │
                 ▼
 [M2: Two-Stage Evidence Set Optimization & Topological Organization] (核心主创新)
   ├─ [M2-A 阶段: 预算证据集合优化 (Budgeted Evidence Set Selection)]
   │    ├─ 逐块计算归一化可靠性 R~_i、覆盖度、实体互补度 w_bridge、JS 散度事实冲突
   │    └─ CPU 端精确枚举 (C(15, <=5) = 4,944 状态) 求解 J(S) 最优 ──▶ 黄金证据子集 D* (|D*| <= 5)
   └─ [M2-B 阶段: 生成树拓扑编排 (MST-Guided Organization)]
        ├─ 仅在 D* 诱导子图上构建边权 W(e_ij)，处理 |D*| <= 1 退化边界
        ├─ 连通分量求解 Kruskal MST 提取拓扑骨干 ──▶ 以最高权边为根执行 DFS 线性化
        └─ 形成抗 Lost-in-the-Middle 紧凑上下文 (规范执行 Permutation Test 打乱顺序消融)
                 │
                 ▼
 [M3: Reliability-Calibrated Faithful Generation & Progressive Gating]
   ├─ 解码前全局拒答门控: 若 max_{d_i in D*} R~_i < tau_abstain ──▶ 触发可靠性拒答 (Selective Abstention)
   ├─ 原文严格对齐的 CLeHe 可靠性先验几何平均对比解码 (单卡 8GB 串行计算, 动态高熵层 l_t*):
   │    Logits_cal(y_t) = (1 + beta) * sum_i w_{i,t} * log p(y_t | q, d_i, y_{<t}) - beta * log p^(l_t*)(y_t | q, y_{<t})
   └─ 渐进可选增强: 句边界触发轻量 SynCheck 分类器 ──▶ 忠实度不足时执行 FOD 单句回溯或安全截断
```

### 4.1 Method 1：认知不确定度 $U_{\text{EU}}$ 解耦与零 Token 哈希图自适应扩展

#### 4.1.1 机制定位与因果头标定假说（No LLM Parameter Fine-tuning）
- **机制定位**：本模块属于 **免大模型参数微调（No LLM Parameter Fine-tuning）**。不改动模型底层权重，但需要利用离线少样本验证集进行因果注意力头筛选与阈值校准，非完全无监督。
- **离线标定流程**：在离线验证集上构造两类提示：黄金证据提示 $P_{\text{gold}} = (q, \mathcal{D}_{\text{gold}})$ 与随机干扰证据提示 $P_{\text{distractor}} = (q, \mathcal{D}_{\text{distractor}})$。通过计算各注意力头激活值对黄金答案预测的因果效应（Causal Effect），在前 $1/3$ 浅中层注意力层中筛选出前 $M$ 个对检索证据显著敏感的注意力头集合 $\mathcal{S}_T^{\text{RAG}}$。
- **在线干预构造**：在在线推理阶段，对于当前输入的 $(q, \mathcal{D})$，根据浅中层注意力头的实际激活值动态构造表征干预向量 $\hat{\mathbf{v}}_{q, \mathcal{D}}$（抵消模型对检索上下文的过拟合依恋），而非全局使用同一个固定静态向量。

#### 4.1.2 闭式不确定性解耦计算（严格非负截断保护）
给定初始轻量混合检索召回的初始锚点集 $\mathcal{D}_0$（默认 $K_0=3$），模型在首 Token 生成前执行 Prefill：
1. **当前上下文总预测全熵**：
   $$H(q, \mathcal{D}_0) = \mathcal{H}\big(p_\theta(y_1 \mid q, \mathcal{D}_0)\big) = -\sum_{v \in \mathcal{V}} p_\theta(v \mid q, \mathcal{D}_0) \log p_\theta(v \mid q, \mathcal{D}_0)$$
2. **因果干预评估偶然预测歧义度 $U_{\text{AU}}$**：
   在注意力头集合 $\mathcal{S}_T^{\text{RAG}}$ 处注入动态干预 $\hat{\mathbf{v}}_{q, \mathcal{D}_0}$，测定模型输出分布的熵，表征查询与证据文本表述本身带来的不可消除模糊性：
   $$U_{\text{AU}}(q, \mathcal{D}_0) = \mathcal{H}\big(p_\theta(y_1 \mid q, \mathcal{D}_0, \hat{\mathbf{v}}_{q, \mathcal{D}_0})\big)$$
3. **闭式解耦认知不确定度 $U_{\text{EU}}$**：
   $$U_{\text{EU}}(q, \mathcal{D}_0) = \Big[H(q, \mathcal{D}_0) - U_{\text{AU}}(q, \mathcal{D}_0)\Big]_+ = \max\Big(0, \, H(q, \mathcal{D}_0) - U_{\text{AU}}(q, \mathcal{D}_0)\Big)$$

#### 4.1.3 自适应门控与 MeshRAG 扩展
- 若 $U_{\text{EU}}(q, \mathcal{D}_0) \le \tau_{\text{EU}}$：模型内部认知明确，直接锁定 $\mathcal{D}_{\text{cand}} = \mathcal{D}_0$，跳过图扩展（节约 100% 后续图检索耗时与显存）；
- 若 $U_{\text{EU}}(q, \mathcal{D}_0) > \tau_{\text{EU}}$：模型面临知识盲区，触发 `MeshRAG` 扩展。利用离线 MinHash 签名与 Bloom Filter 剪枝快速获取 $k$-hop 候选证据块，形成候选证据池 $\mathcal{D}_{\text{cand}}$（默认 $|\mathcal{D}_{\text{cand}}| \le 15$）。实验中严格记录 Bloom Filter 剪枝带来的召回率变动。
---

### 4.2 Method 2：两阶段证据集合优化与生成树拓扑编排（核心主创新）

针对候选证据池 $\mathcal{D}_{\text{cand}}$（$|\mathcal{D}_{\text{cand}}| \le 15$），传统检索直接 Top-K 拼接会导致严重的“Lost-in-the-Middle”与事实冲突。  
**关键图论理论澄清**：标准最大生成树（MST）的数学定义是包含连通图中的**全部节点**（优化的是边集而非点集）。若直接在候选图上求 MST，即使边权被赋予负冲突惩罚，为了保持全连通，低可靠性节点与冲突边依然可能被强行保留，且当连通分量仅含 1 个孤立文档时生成树无边可取。  
因此，本方案将 M2 **彻底重构为解耦的两阶段创新**：**M2-A 阶段显式优化证据点集（解决“选哪些文档”）；M2-B 阶段基于 MST 优化拓扑图结构（解决“如何组织与排序文档”）**。

#### 4.2.1 步骤 2.1：单块非负可靠性得分 $\tilde{R}_i$ 与边际消盲增益
对候选池中的任意文本块 $d_i \in \mathcal{D}_{\text{cand}}$：
1. **免微调混合相关度 $S_{\text{rel}}(q, d_i)$ 与归一化**：
   $$S_{\text{rel}}(q, d_i) = \alpha \cdot \text{Cosine}(\mathbf{e}_q, \mathbf{e}_{d_i}) + (1 - \alpha) \cdot \text{NormBM25}(q, d_i), \quad \alpha \in [0, 1]$$
   为避免余弦负值破坏后续乘法与指数加权，将其通过 Min-Max 映射至非负区间 $[0, 1]$ 得到 $\bar{S}_{\text{rel}}(q, d_i)$。
2. **严格重新计算的边际认知消盲增益 $\Delta U_{\text{EU}}(q; d_i)$**：
   将候选块 $d_i$ 加入初始锚点集后，形成新集合 $\mathcal{D}_0 \cup \{d_i\}$。**必须在新集合上重新计算干预全熵与 $U_{\text{AU}}$，严禁直接套用旧值 $U_{\text{AU}, 0}$**：
   $$U_{\text{AU}}(q, \mathcal{D}_0 \cup \{d_i\}) = \mathcal{H}\big(p_\theta(y_1 \mid q, \mathcal{D}_0 \cup \{d_i\}, \hat{\mathbf{v}}_{q, \mathcal{D}_0 \cup \{d_i\}})\big)$$
   $$U_{\text{EU}}(q, \mathcal{D}_0 \cup \{d_i\}) = \Big[\mathcal{H}\big(p_\theta(y_1 \mid q, \mathcal{D}_0 \cup \{d_i\})\big) - U_{\text{AU}}(q, \mathcal{D}_0 \cup \{d_i\})\Big]_+$$
   $$\Delta U_{\text{EU}}(q; d_i) = \Big[U_{\text{EU}}(q, \mathcal{D}_0) - U_{\text{EU}}(q, \mathcal{D}_0 \cup \{d_i\})\Big]_+$$
3. **单块预测歧义度惩罚 $U_{\text{AU}}(q, d_i)$**：
   $$U_{\text{AU}}(q, d_i) = \mathcal{H}\big(p_\theta(y_1 \mid q, \{d_i\}, \hat{\mathbf{v}}_{q, \{d_i\}})\big)$$
4. **单块非负综合可靠性得分 $\tilde{R}_i$**：
   $$\tilde{R}_i = R(d_i \mid q) = \bar{S}_{\text{rel}}(q, d_i) \cdot \exp\Big(\gamma_1 \cdot \Delta U_{\text{EU}}(q; d_i) - \gamma_2 \cdot U_{\text{AU}}(q, d_i)\Big)$$
   保证 $\tilde{R}_i \ge 0$。

#### 4.2.2 步骤 2.2：M2-A 阶段 —— 预算证据集合组合优化（Budgeted Evidence Set Selection）
- **子集优化目标函数**：
  在上下文预算 $K \le 5$ 约束下，寻找最优证据子集 $\mathcal{D}^* \subseteq \mathcal{D}_{\text{cand}}$，最大化综合目标函数：
  $$J(\mathcal{S}) = \sum_{d_i \in \mathcal{S}} \tilde{R}_i + \lambda_1 \text{Coverage}(\mathcal{S}, q) + \lambda_2 \text{Complementarity}(\mathcal{S}) - \lambda_3 \text{Redundancy}(\mathcal{S}) - \lambda_4 \text{Conflict}(\mathcal{S})$$
  其中各项严格界定为：
  - **$\text{Coverage}(\mathcal{S}, q)$**：子集对 Query 实体与关键词的覆盖率；
  - **$\text{Complementarity}(\mathcal{S})$**：子集两两间的跨块桥接互补度之和 $\sum_{i < j} w_{\text{bridge}}(d_i, d_j)$。**重要辨析**：`w_bridge` 严格区分“实体共现关联”与“关系/语义互补”，不能仅依赖纯字面 Jaccard 相似度，避免将高重叠同义段落误判为互补；
  - **$\text{Redundancy}(\mathcal{S})$**：子集文档间的 MinHash 相似度惩罚 $\sum_{i < j} \text{Sim}_{\text{hash}}(d_i, d_j)$；
  - **$\text{Conflict}(\mathcal{S})$**：两两单文档首 Token 预测分布的 Jensen-Shannon (JS) 散度之和 $\sum_{i < j} D_{\text{JS}}\big(p_\theta(\cdot \mid q, d_i) \parallel p_\theta(\cdot \mid q, d_j)\big)$。**学术客观性界定**：JS 散度度量的是输出分布的统计分歧，不直接等价于自然语言推理（NLI）中的事实矛盾；实验中设立专门消融对比 JS 散度与受控 NLI 矛盾分类器的效能。
- **CPU 端组合求解算法**：
  由于受限场景下 $|\mathcal{D}_{\text{cand}}| \le 15, K \le 5$，全部可行解空间为：
  $$\sum_{k=0}^5 \binom{15}{k} = 1 + 15 + 105 + 455 + 1365 + 3003 = 4,944 \text{ 种组合}$$
  在现代 CPU 上，遍历 4,944 种组合的目标函数计算仅需 **5 ~ 15 ms**，属于轻量常数级开销。系统支持：
  1. **精确枚举（Exact Enumeration / Branch-and-Bound）**：保证在 4,944 状态空间内求得全局最优解 $\mathcal{D}^*$；
  2. **贪心前向选择（Greedy Forward Selection）**：作为消融对比基线；
  3. **标准 Top-K 检索基线**：直接按相关度截断。

#### 4.2.3 步骤 2.3：M2-B 阶段 —— 生成树拓扑编排与保序线性化（MST-Guided Organization）
获得高质量黄金子集 $\mathcal{D}^*$（$|\mathcal{D}^*| \le 5$）后，系统基于图论结构决定输入大模型的上下文排列：
1. **子集诱导图构建与边权定义**：
   在 $\mathcal{D}^*$ 上构建无向完全加权图 $\mathcal{G}^* = (\mathcal{D}^*, \mathcal{E}^*)$，边权定义为：
   $$W(e_{ij}) = \frac{\tilde{R}_i + \tilde{R}_j}{2} + \lambda_{\text{edge}} \cdot w_{\text{bridge}}(d_i, d_j) - \mu_{\text{edge}} \cdot D_{\text{JS}}(p_i \parallel p_j)$$
2. **退化边界情况处理**：
   - 若 $|\mathcal{D}^*| \le 1$（如仅选出 1 篇文档或集合为空）：直接线性化返回该文档，跳过生成树构建，防止无边时算法崩溃；
   - 若图分裂为多个无边连通分量（孤立点）：单节点连通分量作为无边退化树，根即为该节点本身。
3. **Kruskal 最大生成树求解**：
   对含有 2 个及以上节点的连通分量，求解最大生成树 $T^* = \arg\max_{T} \sum_{e \in T} W(e)$，提取权重最大的拓扑骨干，切断环路与低权边。
4. **最高权边为根的 DFS 拓扑排序**：
   在树 $T^*$ 中选取最大权边 $e^* = (u^*, v^*) = \arg\max_{e \in T^*} W(e)$，以 $u^*$（或较高可靠性节点）为根节点，按子边权重降序执行 DFS 拓扑遍历，展平为线性证据上下文序列。
5. **打乱顺序对照实验规范（Permutation Test）**：
   为验证拓扑序是否能有效缓解“Lost-in-the-Middle”注意力偏置，主实验必须设立**顺序消融对比**：DFS 拓扑序 vs 完全逆序（Reverse Order） vs 5 次随机打乱（Random Shuffling），记录准确率均值与方差。

---

### 4.3 Method 3：全局证据可靠性校准的 `CLeHe` 对数概率对比解码与渐进式忠实度门控

在单卡 8GB 显存环境下，M3 采取**渐进式落地与消融路线**，对比解码公式严格对齐顶会原版规范：

#### 4.3.1 阶段 3.1：解码前全局拒答门控（Selective Abstention）
在自回归生成前，评估黄金证据集 $\mathcal{D}^*$ 的可靠性上界：
$$\max_{d_i \in \mathcal{D}^*} \tilde{R}_i < \tau_{\text{abstain}}$$
- 若黄金集中无任何一篇文档达到可靠性门槛，说明检索信息严重匮乏或完全不可信。系统直接输出拒答模板，终止后续前向计算，节约 100% 解码显存与延迟；
- **学术规范与指标界定**：拒答阈值 $\tau_{\text{abstain}}$ 需在验证集上通过**风险-覆盖曲线（Risk-Coverage Curve）**进行精确校准，严禁声称“100% 阻断幻觉”，如实汇报 Abstention Precision、Recall 与 F1。

#### 4.3.2 阶段 3.2：原文严格对齐的 `CLeHe` 加权对数概率对比解码
对齐 `2025.naacl-long.236` PDF p. 4 Eq. (7) 原文规范，将 M2 产出的全局可靠性先验 $\tilde{R}_i$ 与动态局部熵统一建模：
1. **可靠性加权的动态局部熵权重 $w_{i,t}$（统一步调）**：
   在生成步 $t$，对于证据 $d_i \in \mathcal{D}^*$，其局部预测熵为 $H_{i,t} = \mathcal{H}\big(p_\theta(\cdot \mid q, d_i, y_{<t})\big)$。归一化权重统一表达为：
   $$w_{i,t} = \operatorname{softmax}_i\left( \log(\tilde{R}_i + \epsilon) - \frac{H_{i,t}}{\tau} \right) = \frac{(\tilde{R}_i + \epsilon) \cdot \exp(-H_{i,t} / \tau)}{\sum_{d_j \in \mathcal{D}^*} (\tilde{R}_j + \epsilon) \cdot \exp(-H_{j,t} / \tau)}$$
   其中 $\tau > 0$ 为温度超参，$\epsilon > 0$ 为数值稳定性微量。该式消除了此前形式与计算的不一致。
2. **动态高熵参考层对比解码**：
   严格遵循原文规范，负向参考层并非离线固定的单一中间层，而是在每步动态从候选浅中层集合 $\mathcal{L}$ 中挑出熵最大的层：
   $$l_t^* = \arg\max_{l \in \mathcal{L}} \mathcal{H}\big(p_\theta^{(l)}(\cdot \mid q, y_{<t})\big)$$
   校准对数几率严格计算为：
   $$\text{Logits}_{\text{cal}}(y_t) = (1 + \beta) \sum_{d_i \in \mathcal{D}^*} w_{i,t} \log p_\theta(y_t \mid q, d_i, y_{<t}) - \beta \log p_\theta^{(l_t^*)}(y_t \mid q, y_{<t})$$
   其中 $\beta \in [0, 1]$ 为对比强度因子。
3. **单卡 8GB 串行前向开销与延迟代价标注**：
   在单卡 8GB 限制下，为杜绝多文档并发导致 OOM，采用 **Batch Size = 1 文档串行前向（Document-Serial Execution）**。每步需对 $|\mathcal{D}^*| \le 5$ 篇文档执行 5 次前向计算，加上 1 次无证据先验计算，单步前向次数为 $K+1$ 次。此设计以**时间换显存**，理论延迟约为标准自回归解码的 $5 \sim 6 \times$。实验中必须如实呈现该 Trade-off 并统计单题端到端耗时。

#### 4.3.3 阶段 3.3：渐进可选增强 —— 句级 SynCheck 监控与 FOD 回溯
- **轻量监控分类器**：`SYNCHECK`（`2024.emnlp-main.527`）从解码过程的注意力权重和输出概率中提取动态统计特征（如平均熵、注意力偏置方差），输入在少样本上预训练的 CPU 逻辑回归探针；
- **回溯重采与防御**：仅当检测到句级忠实度 $F(s_m \mid \mathcal{D}^*) < \tau_{\text{faith}}$ 时，触发 `FOD` 局部 Beam Search（最大重试 $B_{\text{retry}}=2$）；若依然不达标，则就地优雅截断并附加低置信警告。该模块作为独立顶层增强，主实验中实施严格剥离消融。

---

### 4.4 完整端到端算法伪代码：Algorithm 1

```python
====================================================================================================
Algorithm 1: U-GraphRAG End-to-End Inference Pipeline (V1.1 Research Prototype)
====================================================================================================
Input:  User Query q, Offline Document Corpus D_all, Pre-indexed MeshRAG Hash Communities G_hash,
        Frozen LLM M_theta (NF4 4-bit), Offline Calibrated RAG Attention Heads S_T^{RAG},
        Hyperparameters: tau_EU, tau_abstain, tau_faith, K0=3, K_star=5, alpha=0.5,
        lambda_1, lambda_2, lambda_3, lambda_4, lambda_edge, mu_edge, beta, tau, eps=1e-6
Output: Faithfully Generated Answer y_hat (or Abstention Signal)

1:  // Step 1: Initial Hybrid Retrieval (CPU/GPU)
2:  D_0 = TopK_Hybrid_Retrieve(q, D_all, K=K0)  // Normalized BM25 + bge-small Cosine
3:
4:  // Step 2: Method 1 - Epistemic Uncertainty Decoupling & Adaptive Expansion
5:  H_0 = Entropy(M_theta.forward_first_token(q, D_0))
6:  v_hat_0 = Dynamic_Intervention_Vector(M_theta, q, D_0, S_T^{RAG})
7:  U_AU_0 = Entropy(M_theta.forward_first_token(q, D_0, intervention=v_hat_0))
8:  U_EU_0 = max(0.0, H_0 - U_AU_0)
9:
10: if U_EU_0 <= tau_EU then
11:     D_cand = D_0  // Epistemic uncertainty low; early exit expansion
12: else
13:     Q_fp = Extract_Entity_Fingerprint(q)
14:     Active_Comm = Bloom_Filter_Prune(G_hash.communities, Q_fp)  // Track Recall drop
15:     D_cand = D_0 ∪ MinHash_Hamming_Expand(Active_Comm, D_0, max_cand=15)
16: end if
17:
18: // Step 3: Method 2-A - Budgeted Evidence Set Selection (CPU Combinatorial Optimization)
19: for each chunk d_i in D_cand do
20:     S_rel_i = alpha * Cosine(e_q, e_di) + (1 - alpha) * NormBM25(q, d_i)
21:     S_rel_norm_i = MinMax_Normalize(S_rel_i)
22:     // Re-evaluate AU and EU for D_0 ∪ {d_i} strictly
23:     v_hat_cand = Dynamic_Intervention_Vector(M_theta, q, D_0 ∪ {d_i}, S_T^{RAG})
24:     U_AU_cand = Entropy(M_theta.forward(q, D_0 ∪ {d_i}, intervention=v_hat_cand))
25:     U_EU_cand = max(0.0, Entropy(M_theta.forward(q, D_0 ∪ {d_i})) - U_AU_cand)
26:     Delta_U_EU_i = max(0.0, U_EU_0 - U_EU_cand)
27:     
28:     v_hat_single = Dynamic_Intervention_Vector(M_theta, q, {d_i}, S_T^{RAG})
29:     U_AU_i = Entropy(M_theta.forward(q, {d_i}, intervention=v_hat_single))
30:     R_tilde[i] = S_rel_norm_i * exp(gamma_1 * Delta_U_EU_i - gamma_2 * U_AU_i)
31: end for
32:
33: // Exact enumeration over 4,944 subsets (or branch-and-bound) on CPU
34: D_star = argmax_{S ⊆ D_cand, |S| <= K_star} J(S, R_tilde, lambda_1..4)
35:
36: // Step 4: Method 2-B - MST-Guided Topological Organization
37: if |D_star| <= 1 then
38:     ordered_context = list(D_star)  // Degenerate handling for <=1 document
39: else
40:     Construct weighted graph G_star = (D_star, E_star):
41:     for each pair (d_i, d_j) in E_star do
42:         W(e_ij) = (R_tilde[i] + R_tilde[j]) / 2.0 + lambda_edge * w_bridge - mu_edge * JS_Div(p_i, p_j)
43:     end for
44:     T_star = Kruskal_MST(G_star, edge_weights=W, maximize=True)
45:     e_root = argmax_{e in T_star} W(e)
46:     ordered_context = DFS_Traverse(T_star, root=e_root)
47: end if
48: // Permutation Test baseline shuffles ordered_context to evaluate Lost-in-the-Middle robustness
49:
50: // Step 5: Method 3 - Selective Abstention & Calibrated Decoding
51: if max_{d_i in D_star} R_tilde[i] < tau_abstain then
52:     return "According to verified evidence, the question cannot be reliably answered."
53: end if
54:
55: y = [], t = 1
56: while t <= Max_Tokens and not Is_EOS(y_{t-1}) do
57:     // Document-serial forward passes on 8GB GPU (batch_size=1)
58:     for each d_i in ordered_context do
59:         p_theta_i = M_theta.forward_token_dist(q, d_i, y_{<t})
60:         H_it = Entropy(p_theta_i)
61:         logits_w[i] = log(R_tilde[i] + eps) - H_it / tau
62:     end for
63:     w_norm = Softmax(logits_w)
64:     
65:     // Dynamic layer selection for early distribution
66:     l_star = argmax_{l in L_cand} Entropy(M_theta.forward_layer_dist(q, y_{<t}, layer=l))
67:     p_early = M_theta.forward_layer_dist(q, y_{<t}, layer=l_star)
68:     
69:     Logits_cal = (1 + beta) * sum_i (w_norm[i] * log(p_theta_i)) - beta * log(p_early)
70:     y_t = Sample_or_Greedy(Logits_cal)
71:     y.append(y_t)
72:     
73:     // Optional Progressive SynCheck Monitoring at Sentence Boundary
74:     if y_t in {'.', '。', '\n'} then
75:         F_sent = SynCheck_Probe.eval(current_sentence, ordered_context)
76:         if F_sent < tau_faith then
77:             y_repaired = FOD_Local_Beam_Search(current_sentence, ordered_context, max_retry=2)
78:             if y_repaired is None then break and Truncate_With_Warning(y)
79:             Replace_Sentence(y, y_repaired)
80:         end if
81:     end if
82:     t = t + 1
83: end while
84: return Detokenize(y)
====================================================================================================
```

---

## 五、 靶向特色数据集矩阵（3+1 黄金实验靶场）与 5 步研发路线

### 5.1 通用主实验 3 大核心黄金靶场 + 1 对抗基准矩阵

通用主实验（Step 1 ~ Step 4）必须在严谨的学术公开数据集上确立方法论增益，垂直领域数据划定为 Step 5 泛化压力测试：

| 评测靶场 | 选定公开数据集 | 数据集官方真实构造与特性 | 靶向验证核心模块与实证指标 |
| :--- | :--- | :--- | :--- |
| **黄金靶场 1：多跳关联与干扰项过滤**<br>*(主实验核心基石)* | **`HotpotQA` (Distractor 评测集)**<br>*(ACL 官方原生)* | 官方自带 **2 篇黄金支持段落 + 8 篇高相关 TF-IDF 干扰段落**（真实检索干扰分布）。 | **验证 M1 + M2 (M2-A 筛选 + M2-B 拓扑)**：<br>1. 干扰段落过滤精确率 (Precision@K)<br>2. 黄金支持段落召回率 (Recall@K)<br>3. 终端问答准确率 (Exact Match / F1)<br>4. **打乱顺序实验 (Permutation Test)**：DFS 拓扑序 vs 逆序 vs 5 次随机打乱的指标均值与方差。 |
| **黄金靶场 2：无连接断链多跳推理**<br>*(深度推理检验)* | **`MuSiQue` (Multi-Hop)**<br>*(ACL 官方原生)* | 包含 2~4 跳断链问题，单篇文档无法提供完整推理链。 | **验证 M1 (MeshRAG 扩展) + M2 桥接互补**：<br>1. 多跳推理链逐跳召回率 (Hop-wise Recall)<br>2. 复合多步问答准确率 (EM / F1) |
| **黄金靶场 3：长尾实体与动态预算**<br>*(长尾自适应检验)* | **`PopQA`**<br>*(实体月访问量分层)* | 包含官方 Wikipedia 实体月访问量流行度分层标签。 | **验证 M1 (EU 自适应门控与消盲假说)**：<br>1. 绘制 EU 随实体流行度下降的真实散点图，验证盲区度量有效性<br>2. 热门 vs 长尾分桶 Accuracy<br>3. Token 与推理开销节约率 (Token Saving Rate) |
| **对抗强化靶场：幻觉片段与反事实冲突**<br>*(M3 生成校准检验)* | **`RAGTruth`**<br>&<br>**`F-100-anti`** (反事实子集) | `RAGTruth` 提供词级/句级细粒度幻觉标注；`F-100-anti` 包含实体替换构造的反事实冲突。 | **验证 M3 (CLeHe 对比解码 + SynCheck/FOD)**：<br>1. 句级幻觉捕获准确率 (AUROC / AUPR)<br>2. 期望校准误差 (Expected Calibration Error, ECE)<br>3. 拒答准确率与 F1 (Abstention Quality) |

### 5.2 专项后置：Step 5 垂直领域泛化压力测试（Domain Stress Test）

在 Step 1 ~ Step 4 完成通用方法论定型后，引入特色垂直领域数据集进行零微调泛化评估：
- **跨期演化数据**：`TimeQuestions` 子集（验证在新旧知识事实冲突时，边权 JS 散度与 M2-A 能否有效剔除过时冲突段落）；
- **金融专业问答数据**：`FiQA` / `FinQA` 抽取式问答子集（验证在低容错数值与复杂财报审计上下文下的忠实度迁移表现）。
- **学术规范底线**：垂直领域测试指标单独呈现在专辟章节中，绝不与通用主基准混合计算，保证方法论归因纯粹。

### 5.3 单卡 8GB 物理可行性与强制实测的 5 大系统级指标

在单卡 8GB VRAM (`RTX 3070 / RTX 4060`) 硬件约束下，显存预算设计目标如下：
- **模型主体 (`Qwen2.5-7B-Instruct` 4-bit NF4)**：约 `4.5 ~ 4.8 GB`
- **动态 KV Cache & 上下文 (Max Length $\le 2048$, Batch=1)**：约 `0.8 ~ 1.2 GB`
- **因果头干预 Prefill (单层单向量干预)**：约 `0.3 ~ 0.5 GB`
- **CPU 驻留组件 (bge-small、MinHash、Bloom Filter、M2-A 枚举优化、MST 求解)**：消耗主机内存，GPU 占用 `0 GB`
- **理论峰值显存预估（待实测目标）**：约 `6.5 GB`（安全余量约 `1.5 GB`）。

**主实验强制实测并记录的 5 大系统级指标**：
1. **PyTorch 峰值显存分配（`max_memory_allocated`, MB）**：真实张量分配峰值；
2. **PyTorch 缓存显存峰值（`max_memory_reserved`, MB）**：CUDA 内存池预留峰值；
3. **NVML 物理显存峰值（Peak VRAM Allocation, MB）**：由 `pynvml` 捕获的整卡实际物理显存占用（含驱动与系统开销）；
4. **端到端单题耗时（End-to-End Latency per Query, ms）**：串行解码与图搜索各阶段真实时延；
5. **单题前向推理次数（Forward Passes per Query）**：量化 M1 干预与 M3 串行解码的计算代价。

### 5.4 严格渐进的研发落地 5 步路线图

```text
Step 1: Baseline 跑通与基线固化
  ├─ 搭建 BM25 + Dense (bge-small) 检索器与 Qwen2.5-7B-Instruct NF4 4-bit 端到端管线；
  ├─ 验证单卡 8GB 环境探测脚手架与物理显存测量闭环；
  └─ 在 HotpotQA 轻量切片 (100 题) 上打出第一组可复现基线指标 (EM, F1, Peak VRAM, Latency)。
        │
        ▼
Step 2: M1 模块独立验证与假说检验
  ├─ 接入 MeshRAG 哈希图，实测 Bloom Filter 剪枝前后的真实段落召回率变化；
  └─ 离线标定显著因果头 S_T^{RAG}，验证 EU/AU 能否有效解耦，并在 PopQA 上绘制 EU-流行度散点图。
        │
        ▼
Step 3: M2 核心创新研发与消融攻坚 (主创新核心战场)
  ├─ 实现 M2-A 预算证据集合优化 (CPU 端 4,944 组合精确枚举 vs 贪心 vs Top-K)；
  ├─ 实现 M2-B 基于最大生成树 (MST) 的拓扑骨干提取与 DFS 保序线性化；
  └─ 必须执行 Permutation Test (打乱顺序实验)，量化验证生成树拓扑序对 Lost-in-the-Middle 方差的抑制效果。
        │
        ▼
Step 4: M3 渐进消融验证与对比解码
  ├─ 接入原文对齐的 CLeHe 加权对数概率对比解码 (单卡 8GB 串行实现，动态高熵层)，实测忠实度与时延；
  ├─ 独立接入 CPU 端轻量 SynCheck 监控与 FOD 选择性回溯/拒答，完成全流程渐进消融实验表；
  └─ 锁定最终最佳超参配置组合。
        │
        ▼
Step 5: 垂直领域泛化压力测试 (Domain Stress Test)
  └─ 将定型系统直接零微调应用于 TimeQuestions (跨期事实冲突) 与 FiQA/FinQA (金融审计) 进行泛化压力评估。
```

---

## 六、 原型冻结范围与阶段性结论

本文档正式确立了 **`U-GraphRAG V1.1 (Research Prototype)`** 的研发框架。  
明确以下管理规则：
1. **冻结范围**：核心科学问题、三层可靠性中心架构骨架、靶向特色数据集矩阵与 5 步渐进式消融研发路线；
2. **开放与实证范围**：具体算子实现细节（如因果头选取层数、组合搜索目标函数超参、对比解码温度与高熵层集合）、代码工程落地与硬件性能指标，以真实代码运行与实验实测数据为唯一检验标准；
3. **团队执行纪律**：严禁伪造指标或篡改文献，所有开发按 Step 1 至 Step 5 稳步攻关。
