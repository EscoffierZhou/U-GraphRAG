# U-GraphRAG V0.1 方法论原型敲定书：基于不确定性解耦与零 Token 哈希生成树的自适应证据可靠性检索增强生成

> **文档定位**：`[SDUFE] GeminiOutOfToken_Team` 选题方法论正式拍板与原型冻结稿（存放于 `Debate/`）  
> **所属翼展**：`W3` (`RAG_Hallucination_Mitigation`) 深度重构版（吸收 `W1-R01` 同步忠实度监测跨翼算子）  
> **版本标识**：`U-GraphRAG V0.1 (Research Prototype Frozen)`  
> **硬件与预算约束**：单卡 8GB VRAM (`Qwen2.5-7B-Instruct` 4-bit / 本地 CPU 沙箱) · 极度受限 Token 预算  
> **审阅吸收声明**：本文档完全吸收并落实 `Debate/GPT_opinion.md` 的专家审阅意见与原版双栏 PDF 核验结论。不盲目拼接顶会组件，不编造未经证明的数学性质，不预设未经验证的实验结论，严格落实 7 处核心修正，以严谨的研究假说与渐进式验证路线驱动探索。

---

## 一、 核心痛点与组件定位重审（逐模块审阅定性）

在以 **GPT 的三层证据可靠性方法论（M1 不确定性驱动检索 $\to$ M2 证据集合优化 $\to$ M3 生成忠实度校准）** 为主轴重新审视候选顶会组件后，我们对组件的角色边界、数学形式与算子性质进行了彻底的审敛与降级界定，拒绝“组件堆砌”：

### 1.1 组件逐一拍板与定性矩阵

| 候选组件 | 柜子短码 (`cabinet_code`) | 论文出处与年份 | 原型中的定位 | 定性理由与边界说明（吸收审阅意见） |
| :--- | :--- | :--- | :--- | :--- |
| **Self-FV** | `W3-R06#2026.acl-long.95` | ACL 2026 | **保留为待验证的核心研究假说** | 原文针对 ICL 标签扰动识别显著注意力头。将其迁移至 RAG 评估检索知识的 AU/EU 属于本课题**核心迁移假说**，需重新实验标定。 |
| **MeshRAG** | `W3-R07#2026.acl-long.1156` | ACL 2026 | **保留为主候选证据扩展引擎** | 免 LLM API Token 构建社区-文本块图。严谨界定“零 Token”是指不消耗大模型 API，实验必须报告 **Bloom Filter 剪枝前后的候选召回率变化**。 |
| **DUR** | `W3-R05#2025.naacl-long.453` | NAACL 2025 | **降级为思想参考与核心对比基线** | 原文是 BERT + 线性层并通过 REINFORCE 策略梯度训练的检索器，非现成免训练核密度函数 $\mathcal{K}_{\text{sim}}$。主方法 M2 改用免训练混合检索度量 $S_{\text{rel}}$。 |
| **KG2RAG** | `W3-R07#2025.naacl-long.449` | NAACL 2025 | **保留图组织思想，重做整合** | 纠正“最小生成树”笔误，严格还原原文 **Maximum Spanning Tree (最大生成树, MST)**；明确是在哈希碰撞/实体图上优化证据集合；**彻底删除未证明的 Submodular 术语**。 |
| **CLeHe** | `W3-R05#2025.naacl-long.236` | NAACL 2025 | **保留为解码对比基线与扩展机制** | 严格还原原文 Eq. (7) **对数概率加权和（几何平均）** 形式；诚实声明多文档串行前向计算对延迟与模型调用的影响，进行开销评测。 |
| **SYNCHECK & FOD** | `W1-R01#2024.emnlp-main.527` | EMNLP 2024 | **保留为 M3 可选渐进验证模块** | 明确其属于有监督轻量分类器，FOD 包含回溯与 Beam Search 搜索开销；作为 M3 渐进消融的最后一环独立检验。 |

### 1.2 六项关键边界与修正界定

1. **`DUR` 的规范降级**：
   - 原文（`2025.naacl-long.453` PDF p. 4–5 Eq. 4–10）通过 LLM 对不同样本的预测反馈执行 REINFORCE 策略梯度更新线性打分层 $h(\cdot) = \mathbf{W}(\text{BERT}(\cdot)) + \mathbf{b}$。因此，原案中将“现成核密度 $\mathcal{K}_{\text{sim}}$”归因于 DUR 缺乏原文支持。
   - **修正方案**：将 `DUR` 明确列为**长尾动态检索预算的思想来源与核心对比基线（Baseline）**。在 M2 的公式中，基础相关性项直接定义为无监督免训练的混合检索相关度得分 $S_{\text{rel}}(q, d_i)$（归一化 BM25 与 Dense 向量余弦/汉明相似度的融合）。
2. **`Self-FV` 的迁移边界与 $\Delta U_{\text{EU}}$ 参照系澄清**：
   - 原文（`2026.acl-long.95` PDF p. 4 Section 3.1 Eq. 6）针对 In-Context Learning 任务，需在特定任务验证集上通过标签置换（Corrupted Prompt）计算因果效应 $CE(P)$ 筛选注意力头集合 $S_T$。将该机制迁移至 RAG（通过在验证集上置换相关段落与干扰段落来离线标定 RAG 显著因果头 $S_T^{\text{RAG}}$，并在推理时评估检索上下文的 AU/EU）属于**待实验检验的核心迁移假说（Research Hypothesis）**。
   - **消盲收益参照系采纳 GPT 修正**：单块证据引入的认知消盲增益严格定义为相对于**初始锚点检索集 $\mathcal{D}_0$** 的边际下降量：
     $$\Delta U_{\text{EU}}(q; d_i) = \Big[U_{\text{EU}}(q, \mathcal{D}_0) - U_{\text{EU}}(q, \mathcal{D}_0 \cup \{d_i\})\Big]_+$$
     确保正值严格代表加入候选证据 $d_i$ 之后模型认知不确定性的净下降。
   - **AU 语义严格区分**：$U_{\text{AU}}$ 度量的是模型在当前证据上下文下的预测歧义度（数据噪声），不能直接等同于“证据来源不权威”。证据事实冲突必须由独立的预测分布分歧项 $\text{Conflict}(d_i, d_j)$（如 JS 散度）显式度量。
3. **`MeshRAG` 的“零 Token”严谨界定与剪枝召回代价测试**：
   - “零 Token”（`2026.acl-long.1156`）严格限定为“**无需调用商业 LLM API 执行实体抽取与图谱构建**”。这并不意味着计算开销为零，其底层的 CPU 实体提取、MinHash 计算与 Bloom Filter 维护仍消耗 CPU 周期与时间。
   - 实验必须报告 **Bloom Filter 剪枝前后的候选召回率变化（Recall Drop under Bloom Pruning）**，不能仅报告图构建速度与剪枝比例。
4. **`KG2RAG` 的图组织还原与“Submodular”术语彻底剥离**：
   - 严格落实 **Maximum Spanning Tree (最大生成树, MST)** 表述（`2025.naacl-long.449` PDF p. 4–5 Eq. 6–8），用于在加权图上提取骨干通路、剔除冗余多边与冲突边。
   - 明确指出：`KG2RAG` 原文在知识图谱实体三元组上建图，而本研究 M2 是在 `MeshRAG` 的文本块哈希碰撞/实体共现图上自建优化目标。
   - **彻底删除“Submodular（子模优化）”术语**，避免在未给出严格增益单调递减数学证明的情况下借用理论标签。
   - 明确声明：“生成树 + DFS 拓扑排序能否降低文档拼接顺序敏感性”属于待实验验证的假设，必须通过显式的**随机打乱顺序（Permutation Test）** 消融实验来证实。
5. **`CLeHe` 原文公式严格对齐与串行推理开销**：
   - 纠正原案中先对条件概率求和再取对数的算术平均形式，严格还原 `2025.naacl-long.236` PDF p. 4 Eq. (7) 的**加权对数概率和（几何平均）** 标准形式。
   - 诚实声明：`CLeHe` 需对 $|\mathcal{D}^*|$ 篇证据分别计算前向条件分布，在 8GB 显存下必须串行执行，实验需实测单题端到端耗时与前向计算次数。
6. **`SYNCHECK & FOD` 的有监督属性与渐进消融策略**：
   - 明确 `SYNCHECK` 是利用标注数据训练的轻量级有监督分类器，并非免训练启发式探针；`FOD` 涉及子句回溯与 Beam Search 搜索开销。
   - 采纳渐进验证策略：M3 绝不同时无脑堆叠全部模块，而是按 **普通 RAG $\to$ + 可靠性加权 CLeHe $\to$ + SynCheck 监测与 FOD 回溯** 的顺序逐一消融精度与时延。

---

## 二、 U-GraphRAG V0.1 核心方法论猜想原型与严密数学表述

**题目定稿**：  
- **中文题目**：`U-GraphRAG`: 面向受限计算资源的不确定性解耦自适应检索、生成树证据集合优化与校准生成系统  
- **英文题目**：*U-GraphRAG: Uncertainty-Decoupled Adaptive Retrieval, Spanning-Tree Evidence Set Optimization, and Calibrated Generation under Constrained Resources*

```text
Query q ──▶ 初始检索 (BM25+Dense Top-K0, K0=3) ──▶ 锚点证据集 D_0
                 │
                 ▼
 [Method 1: Uncertainty-Guided Adaptive Retrieval] (Self-FV 假说 + MeshRAG 扩展)
   ├─ 单次因果头干预 Prefill (N=1) ──▶ 计算预测熵 H_0 与 AU(q, D_0)，解耦得到 EU(q, D_0)
   ├─ 门控判定:
   │    ├─ 若 U_EU(q, D_0) <= tau_EU : 模型认知明确 ──▶ 终止扩展，D_cand = D_0 (极速节约 Token)
   │    └─ 若 U_EU(q, D_0) >  tau_EU : 模型面临认知盲区 ──▶ 触发 MeshRAG 零 Token 哈希图扩展
   │                                       │
   │                                       ├─ Bloom Filter 社区剪枝 (实测 Recall 损失)
   │                                       └─ MinHash 碰撞与汉明相似度扩展 ──▶ 候选集 D_cand
                 │
                 ▼
 [Method 2: Reliability-Calibrated Evidence Set Optimization] (核心创新主战场)
   ├─ 逐块评估基础相关度 S_rel 与消盲收益 ΔU_EU(q; d_i) = [U_EU(q, D_0) - U_EU(q, D_0 ∪ {d_i})]_+
   ├─ 解耦度量单块可靠性: R(d_i | q) = S_rel(q, d_i) * exp(gamma_1 * ΔU_EU - gamma_2 * U_AU)
   ├─ 构造证据图 G_cand: 边权融合两端可靠性均值、实体桥接互补度 w_bridge 与预测分歧惩罚 Conflict(d_i, d_j)
   ├─ 求解连通分量最大生成树 (Maximum Spanning Tree, MST): 剔除环路冗余与事实冲突
   └─ 最高权边为根 DFS 排序 + 预算截断 ──▶ 最终高可靠、低冗余证据子集 D* (设计 Permutation Test 消融)
                 │
                 ▼
 [Method 3: Evidence-Calibrated Faithful Generation] (CLeHe 对齐 + 渐进 SynCheck/FOD)
   ├─ 解码前全局拒答门控: 若 max_{d_i} R(d_i | q) < tau_abstain ──▶ 触发可靠性拒答 (Selective Abstention)
   ├─ 证据可靠性加权对比解码 (CLeHe 原文对齐公式):
   │    Logits_cal(y_t) = (1 + beta) * \sum_{d_i} w_{i,t}^{rel} * log p(y_t | q, d_i, y_{<t}) - beta * log p^(l*)(y_t | q, y_{<t})
   │    (在 8GB 显存下单模型串行前向计算，记录延迟代价)
   └─ 渐进可选增强: 遇句号触发有监督 SynCheck 探针 ──▶ 忠实度 < tau_faith 触发单句 FOD 回溯修正
```

---

### 2.1 Method 1：Uncertainty-Guided Adaptive Retrieval（不确定性引导的自适应检索）

#### 科学假说与迁移设定 ($H_1$)
- **迁移假说**：借鉴 `Self-FV` (`2026.acl-long.95`) 在 In-Context Learning 任务中的机制，假设大模型在前 $1/3$ 深度层存在一组与任务上下文相关的显著因果注意力头 $S_T^{\text{RAG}}$。通过在验证集上离线执行段落置换标定该集合后，推理期只需**单次前向（$N=1$）**提取自功能向量并施加干预，即可在生成首个 Token 前将输出分布的总预测不确定性解耦为：
  - **偶然不确定度 $U_{\text{AU}}$**：当前上下文引入的数据歧义度与固有噪声；
  - **认知不确定度 $U_{\text{EU}}$**：模型面对当前查询及证据时的内部参数化知识盲区。
- **闭式计算公式（含非负截断保护）**：
  给定初始混合检索召回的锚点集 $\mathcal{D}_0$（默认 $K_0=3$），模型计算首 Token 预测熵：
  $$H(q, \mathcal{D}_0) = \mathcal{H}\big(p_\theta(y_1 \mid q, \mathcal{D}_0)\big) = -\sum_{v \in \mathcal{V}} p_\theta(v \mid q, \mathcal{D}_0) \log p_\theta(v \mid q, \mathcal{D}_0)$$
  通过在 $S_T^{\text{RAG}}$ 注意力头施加自功能向量干预 $\hat{\mathbf{v}}_T$，估计偶然歧义度：
  $$U_{\text{AU}}(q, \mathcal{D}_0) = \mathcal{H}\big(p_\theta(y_1 \mid q, \mathcal{D}_0, \hat{\mathbf{v}}_T)\big)$$
  进而解耦出认知盲区程度：
  $$U_{\text{EU}}(q, \mathcal{D}_0) = \Big[H(q, \mathcal{D}_0) - U_{\text{AU}}(q, \mathcal{D}_0)\Big]_+ = \max\Big(0, \, H(q, \mathcal{D}_0) - U_{\text{AU}}(q, \mathcal{D}_0)\Big)$$

#### 动态触发与 MeshRAG 图扩展
- **自适应阈值门控**：
  - 若 $U_{\text{EU}}(q, \mathcal{D}_0) \le \tau_{\text{EU}}$：说明模型在初始证据支撑下具有充分的认知确信度，无需耗费额外算力，直接锁定 $\mathcal{D}_{\text{cand}} = \mathcal{D}_0$，跳过后续检索扩展；
  - 若 $U_{\text{EU}}(q, \mathcal{D}_0) > \tau_{\text{EU}}$：说明初始证据无法消除模型的认知盲区（常见于长尾实体或复杂多跳关联），激活 `MeshRAG` 哈希图扩展。
- **MeshRAG 零 API Token 图扩展与剪枝验证**：
  - 利用离线在 CPU 上预建的文本块-社区双层哈希图（由 MinHash 哈希碰撞与实体集合构建）；
  - 先通过布隆过滤器（Bloom Filter）以 $O(\eta)$ 复杂度剪除无关社区，再在剩余连通子图中依据汉明相似度扩展 $k$-hop 关联邻域，获取扩展证据候选池 $\mathcal{D}_{\text{cand}}$；
  - **实证规约**：必须严格记录布隆过滤器剪枝前后黄金答案段落的召回率变化（$\text{Recall}_{\text{pruned}}$ vs $\text{Recall}_{\text{raw}}$），验证剪枝是否造成证据断链。

---

### 2.2 Method 2：Reliability-Calibrated Evidence Set Optimization（可靠性校准证据集合优化 —— 核心主创新）

本阶段是 U-GraphRAG 最具学术独立性与创新价值的环节。针对检索出的候选池 $\mathcal{D}_{\text{cand}}$，不仅考虑相关性，更从**可靠性（Reliability）**、**互补覆盖性（Complementarity）**与**冲突惩罚（Conflict Penalization）**多维定义证据质量，并用图论算法进行结构化编排。

#### 步骤 2.1：单块解耦可靠性评分定义
对候选池中的任一文档文本块 $d_i \in \mathcal{D}_{\text{cand}}$，其最终可靠性得分 $R(d_i \mid q)$ 由三部分共同决定：
1. **基础免训练混合相关度 $S_{\text{rel}}(q, d_i)$**：  
   摒弃原案中对 `DUR` 神经网络打分器的不严谨依赖，直接采用稠密向量余弦相似度与稀疏 BM25 得分的归一化加权和：
   $$S_{\text{rel}}(q, d_i) = \alpha \cdot \text{Cosine}(\mathbf{e}_q, \mathbf{e}_{d_i}) + (1 - \alpha) \cdot \text{NormBM25}(q, d_i), \quad \alpha \in [0, 1]$$
2. **相对于初始集合 $\mathcal{D}_0$ 的边际认知消盲增益 $\Delta U_{\text{EU}}(q; d_i)$**：  
   严格采纳 GPT 意见，以 $\mathcal{D}_0$ 为基线，衡量将候选块 $d_i$ 加入上下文后，模型认知盲区的边际下降幅度（非负保护）：
   $$\Delta U_{\text{EU}}(q; d_i) = \Big[U_{\text{EU}}(q, \mathcal{D}_0) - U_{\text{EU}}(q, \mathcal{D}_0 \cup \{d_i\})\Big]_+$$
3. **单块预测歧义度惩罚 $U_{\text{AU}}(q, d_i)$**：  
   在单文档提示下，由注意力头干预测得的数据预测歧义度：
   $$U_{\text{AU}}(q, d_i) = \mathcal{H}\big(p_\theta(y_1 \mid q, d_i, \hat{\mathbf{v}}_T)\big)$$
4. **单块综合可靠性评分函数**：
   $$R(d_i \mid q) = S_{\text{rel}}(q, d_i) \cdot \exp\Big(\gamma_1 \cdot \Delta U_{\text{EU}}(q; d_i) - \gamma_2 \cdot U_{\text{AU}}(q, d_i)\Big)$$
   - 当 $d_i$ 能显著降低模型盲区（$\Delta U_{\text{EU}} > 0$）且自身提示下的模型预测歧义极低（$U_{\text{AU}}$ 小）时，$R(d_i \mid q)$ 获得显著增强；
   - 若 $d_i$ 虽然与 Query 字面高相关，但引入了巨大的预测歧义或无法消减盲区，则受到指数惩罚。

#### 步骤 2.2：冲突感知图构建与最大生成树（MST）骨干提取
1. **构建候选证据无向加权图 $\mathcal{G}_{\text{cand}} = (\mathcal{D}_{\text{cand}}, \mathcal{E})$**：  
   节点为文本块，任意两块 $d_i, d_j$ 之间的边权 $W(e_{ij})$ 综合两端可靠性、跨块实体覆盖互补性与事实冲突：
   $$W(e_{ij}) = \frac{R(d_i \mid q) + R(d_j \mid q)}{2} + \lambda \cdot w_{\text{bridge}}(d_i, d_j) - \mu \cdot \text{Conflict}(d_i, d_j)$$
   - **跨块实体覆盖互补性 $w_{\text{bridge}}(d_i, d_j)$**：由 `MeshRAG` 的共享实体 Jaccard 相似度或多跳关系边权重给出，衡量两篇文档能否在推理链中起到“桥接”作用；
   - **事实冲突显式惩罚项 $\text{Conflict}(d_i, d_j)$**：采纳 GPT 意见，AU 度量的是上下文预测歧义，不能直接代表跨文档事实冲突。因此采用两个独立文档各自条件预测分布之间的对称 Jensen-Shannon 散度度量事实冲突：
     $$\text{Conflict}(d_i, d_j) = D_{\text{JS}}\big(p_\theta(\cdot \mid q, d_i) \parallel p_\theta(\cdot \mid q, d_j)\big)$$
     当两篇文档诱导模型输出相互矛盾的答案分布时，JS 散度趋近于 1，边权受到强力惩罚，阻断两者同时被选入核心骨干。
2. **连通分量上的最大生成树（Maximum Spanning Tree, MST）过滤**：  
   借鉴 `KG2RAG`（`2025.naacl-long.449` Eq. 7）在三元组图上的组织哲学，我们在文本块图 $\mathcal{G}_{\text{cand}}$ 的每个连通分量 $\mathcal{C}_m$ 上求解**最大生成树**：
   $$T_m = \text{MST}(\mathcal{C}_m) = \arg\max_{T \in \mathcal{T}(\mathcal{C}_m)} \sum_{e_{ij} \in T} W(e_{ij})$$
   - **去冗余与抗冲突机理**：最大生成树在保留最高权边（高可靠、高互补）的同时，精确破坏了图中的所有环路（剔除了语义重合度极高的冗余多边），并优先切断具有严重预测冲突（高 $\text{Conflict}$ 导致权值为负或极低）的边；
   - **学术严谨性声明**：彻底删除“Submodular（子模优化）”术语，本方法基于图论 Kruskal 贪心算法在精确多项式时间 $O(|E|\log|E|)$ 内完成 MST 求解。

#### 步骤 2.3：DFS 拓扑序编排与顺序敏感性消融假说
- **以最大权边为根的 DFS 拓扑遍历**：  
  在每棵生成树 $T_m$ 中，选取权值最高的边 $e^* = \arg\max_{e \in T_m} W(e)$，以其关联节点为根，执行**深度优先搜索（DFS）**，将树上相关的文本块串联为连贯的阅读链路（对应 `KG2RAG` Eq. 8）；
- **最终证据集合截断**：  
  按各连通树的综合可靠性得分降序截取前 $K^*$ 个文本块（默认 $K^*=5$），构成提供给下游的黄金证据集 $\mathcal{D}^*$；
- **排列顺序敏感性实验规约**：  
  生成树拓扑序是否能够降低 LLM 对“Lost-in-the-Middle”文档拼接顺序的敏感性，属于待检验的实证假设。主实验必须设置**随机打乱顺序（Random Permutation Test）**对照组，对比 DFS 拓扑序 vs 逆序 vs 随机打乱下的 EM / F1 方差。

---

### 2.3 Method 3：Evidence-Calibrated Faithful Generation（证据校准忠实度生成与渐进验证）

在单卡 8GB 显存环境下，为避免多组件并发导致 OOM 或极端推理延迟，M3 采取**渐进式消融验证路线**，核心生成公式严格对齐顶会原版规范：

#### 阶段 3.1：解码前可靠性全局拒答门控（Selective Abstention）
若经过 M2 优化后，黄金证据集中的最大单块可靠性依然极低：
$$\max_{d_i \in \mathcal{D}^*} R(d_i \mid q) < \tau_{\text{abstain}}$$
说明检索池中全部证据均缺乏对 Query 的可靠支撑或存在无法消解的认知盲区。系统直接触发拒答机制，输出预设模板 *“根据现有可靠证据无法回答该问题”*，终止生成，将幻觉风险扼杀在解码之前，节约 100% 解码 Token。

#### 阶段 3.2：原文公式对齐的 `CLeHe` 加权对数对比解码
在进入生成时，严格采用 `2025.naacl-long.236` Eq. (7) 原文的**加权对数概率和（几何平均）**，将 M2 得到的全局证据可靠性得分 $R(d_i \mid q)$ 融合进局部 Token 熵权重中：
1. **动态可靠性局部熵权重**：
   $$w_{i,t}^{\text{rel}} = \frac{R(d_i \mid q) \cdot \exp\big(-H(p_\theta(y_t \mid q, d_i, y_{<t})) / \tau\big)}{\sum_{d_j \in \mathcal{D}^*} R(d_j \mid q) \cdot \exp\big(-H(p_\theta(y_t \mid q, d_j, y_{<t})) / \tau\big)}$$
   - 局部熵越低且全局可靠性越高的文档，在当前 Token 解码步获得越高的投票权重。
2. **对比解码校准公式（原文 Eq. 7 严格对齐版）**：
   $$\text{Logits}_{\text{cal}}(y_t) = (1 + \beta) \sum_{d_i \in \mathcal{D}^*} w_{i,t}^{\text{rel}} \log p_\theta(y_t \mid q, d_i, y_{<t}) - \beta \log p_\theta^{(l^*)}(y_t \mid q, y_{<t})$$
   - 其中第二项 $\log p_\theta^{(l^*)}$ 取自高熵中间层 $l^*$ 的早退参数化分布，用于抵消模型的先验常识幻觉偏见；
   - **8GB 显存串行实现声明**：对 $|\mathcal{D}^*|$ 篇文档的条件分布计算，必须在 Batch Size = 1 条件下**按文档串行执行**前向推理（缓存 KV Cache），严禁直接复制模型实例或开辟大 Batch 造成显存溢出；必须详细统计单题额外增加的前向次数与毫秒级延迟。

#### 阶段 3.3：渐进可选增强 —— 句级有监督 SynCheck 监测与 FOD 回溯
- **模块定性**：`SYNCHECK`（`2024.emnlp-main.527`）是基于标注特征在 CPU 上训练的轻量逻辑回归分类器；`FOD` 包含句级回溯与局部 Beam Search 开销；
- **消融执行顺序**：首先检验仅含阶段 3.2（CLeHe）的性能；在此基础上，作为扩展消融实验接入 `SYNCHECK & FOD`：
  - 解码过程中，每当生成句号（Sentence End）时，提取该句的注意力分布与置信度特征，输入 `SYNCHECK` 计算忠实度得分 $F(s_m \mid \mathcal{D}^*)$；
  - 若 $F(s_m \mid \mathcal{D}^*) < \tau_{\text{faith}}$，触发 `FOD` 撤销当前句子并执行局部 Beam Search 重新采样；若多次重采仍不达标，则在该句处优雅截断并拒答。

---

## 三、 靶向特色数据集矩阵与评测规范

完全采纳 GPT 审阅意见，严格厘清数据集的评测定位与统计特性，**将金融/跨期领域明确后置为通用方法验证有效后的第 5 阶段压力测试**，不在前 4 个阶段混入垂直领域噪声：

### 3.1 主实验阶段：五大通用基准靶场矩阵

| 评测靶场 | 选定公开数据集 | 数据集官方真实构造与统计特性（吸收 GPT 意见核验） | 靶向验证的核心模块与可证伪指标 |
| :--- | :--- | :--- | :--- |
| **靶场 1：多跳关联与干扰项过滤**<br>*(主实验核心基石)* | **`HotpotQA` (Distractor)**<br>*(ACL 原生)* | 每题自带 **2 篇黄金支持段落 + 8 篇 TF-IDF 高相关干扰段落**（并非专门构造的反事实对抗集，而是真实检索干扰分布）。 | **验证 M1 + M2**：<br>1. 干扰项剔除率 (Precision@K)<br>2. 黄金段落保留率 (Recall@K)<br>3. 问答准确率 (EM / F1)<br>4. 随机打乱顺序消融 (Permutation Invariance) |
| **靶场 2：多跳推理链追踪**<br>*(深度多跳检验)* | **`MuSiQue` (Multi-Hop)**<br>*(ACL 原生)* | 包含 2~4 跳复杂的无连接断链问题，需要跨文档连续推理。 | **验证 M1 (`MeshRAG`) + M2**：<br>1. 推理链中**每一跳**的逐跳召回率 (Hop-wise Recall)<br>2. 最终跨跳多步答案连贯性与 EM |
| **靶场 3：长尾实体与动态预算**<br>*(长尾自适应检验)* | **`PopQA`**<br>*(实体流行度分层)* | 带有 Wikipedia 实体月访问量的严格**流行度（Popularity）分层标签**。 | **验证 M1 (自适应门控)**：<br>1. **实测检验**：EU 是否严格随实体流行度下降而上升（不做预设结论，诚实绘制相关散点图）<br>2. 热门/长尾分桶 Accuracy 与 Token 节约率 |
| **靶场 4：忠实度与幻觉片段评测**<br>*(M3 生成校准检验)* | **`RAGTruth`**<br>*(QA & Summarization)* | 原始数据以**词级（Word-Level）幻觉片段标注**为主，按照 SynCheck 论文标准映射为句级幻觉标签。 | **验证 M3 (CLeHe & SynCheck)**：<br>1. 句级幻觉分类 AUROC / AUPR<br>2. 期望校准误差 (ECE)<br>3. 生成忠实度 (Faithfulness EM / Rouge-L) |
| **靶场 5：反事实冲突压力测试**<br>*(对抗性生成检验)* | **`F-100-anti`**<br>*(SynCheck 构造子集)* | 基于名人传记通过命名实体替换人工构造的**反事实冲突集**，迫使模型在预训练强记忆与上下文检索证据间选择。 | **验证 M3 (FOD & 拒答)**：<br>1. 事实遵从率 (Context Reliance vs Parametric Bias)<br>2. 拒答准确率与 F1 (Abstention Quality) |

### 3.2 专项后置：第 5 阶段垂直领域泛化压力测试（Domain Stress Test）

在完成上述通用基准的扎实检验后，方案进入第 5 阶段，引入学校与战队特色的垂直领域数据进行**泛化压力测试**：
- **跨期演化数据**：`TimeQuestions` 子集（验证时间跨度下新旧证据冲突）；
- **金融专业问答数据**：`FiQA` / `FinQA` 抽取式子集（验证在低容错数值与财报审计上下文中的忠实度迁移表现）。
- **原则红线**：垂直领域测试结果绝不与主实验结果混合报告，确保方法创新性的归因纯粹、逻辑严密。

---

## 四、 单卡 8GB 硬件物理可行性与预算约束实测规范

### 4.1 理论显存预算分解（严格纠正：6.5 GB 为估计值，非已证实结论）

在单卡 8GB VRAM (`NVIDIA RTX 3070 / RTX 4060`) 硬件约束下，系统的理论静态显存开销分解如下：

| 系统组件 | 承载硬件 | 显存/内存理论占用 | 优化技术与约束机制 |
| :--- | :--- | :--- | :--- |
| **大模型主体 (`Qwen2.5-7B-Instruct`)** | 单卡 GPU | 约 `4.5 ~ 4.8 GB` | BitsAndBytes NF4 4-bit 量化，冻结权重 |
| **模型推理上下文与 KV Cache** | 单卡 GPU | 约 `0.8 ~ 1.2 GB` | 控制 Max Context Length $\le 2048$，Batch Size = 1 |
| **Self-FV 因果头干预 Prefill** | 单卡 GPU | 约 `0.3 ~ 0.5 GB` | 单层单向量干预（$N=1$），不保留冗余激活梯度 |
| **混合检索向量模型 (`bge-small-zh-v1.5`)** | CPU 内存 | **`0 GB` (GPU)** | ONNX Runtime 在主机 CPU 内存中运行（约 200MB RAM） |
| **MeshRAG MinHash + Bloom Filter** | CPU 内存 | **`0 GB` (GPU)** | 纯主机 RAM 数组运算，位图占用约 50MB RAM |
| **KG2RAG 加权最大生成树 (Kruskal)** | CPU 内存 | **`0 GB` (GPU)** | NetworkX / Scipy 纯 CPU 计算，毫秒级图过滤 |
| **理论峰值显存总计** | 单卡 GPU | **`~ 6.5 GB` (理论估计)** | **安全余量约 `1.5 GB`**，但必须经受实测核验 |

### 4.2 必须严格实测记录的 4 大系统级物理指标

不能仅仅宣称“模型能装入 8GB”，在 Step 1 ~ Step 4 的每一步运行中，必须通过 `pynvml` 或 `torch.cuda.max_memory_allocated()` 脚本严格记录并报告：
1. **真实硬件峰值显存（Peak VRAM Allocation, MB）**：监控在执行 CLeHe 多文档串行前向时的瞬时显存尖峰，核验是否突破 8GB 物理红线；
2. **端到端单题推理延迟（End-to-End Latency per Query, ms）**：包含检索、图扩展、MST 求解与解码的完整时延；
3. **模型单题前向传播次数（Forward Passes per Query）**：诚实量化 Self-FV 干预与 CLeHe 多文档串行计算带来的额外推理负担；
4. **API Token / 上下文消耗量（Tokens per Query）**：验证自适应早停通道是否相比传统暴力 Top-K 检索实现了显著的 Token 压缩。

---

## 五、 原型落地路线图与执行顺序（GPT 拍板 5 步法）

为杜绝“无限扩充顶会组件、迟迟无法交付代码”的工程陷阱，战队正式确立并冻结以下 5 步递进式研发路线图：

```text
Step 1: Baseline 跑通
  ├─ 建立 BM25 + Dense (bge-small) 混合检索器
  ├─ 搭建 Qwen2.5-7B-Instruct 4-bit 基础端到端 RAG 流水线
  └─ 在 HotpotQA 轻量切片 (100 题) 上打出第一组可复现基线指标 (EM, F1, VRAM, Latency)
        │
        ▼
Step 2: M1 模块独立验证
  ├─ 单独接入 MeshRAG 哈希碰撞图，实测 Bloom Filter 剪枝前后的候选召回率变化
  └─ 单独实现 Self-FV 在 RAG 上下文中的因果头标定，实测 AU 与 EU 能否有效解耦 (验证假说 H1)
        │
        ▼
Step 3: M2 核心创新研发 (整篇研究的最关键攻坚点)
  ├─ 在文本块图上实现消盲收益 ΔU_EU、歧义惩罚 U_AU 与预测冲突 Conflict 的严密数学计算
  ├─ 实现加权最大生成树 (MST) 过滤与 DFS 拓扑排序编排算子
  ├─ 对比 Baseline: 普通 Top-K 拼接 vs KG2RAG 三元组图 vs M2 MST 编排
  └─ 执行打乱顺序实验 (Permutation Test)，验证生成树 DFS 对排列顺序敏感性的抑制效果
        │
        ▼
Step 4: M3 渐进消融验证
  ├─ 保持 M1+M2 稳定，接入原文对齐的 CLeHe 加权对数对比解码，实测忠实度增益与串行延迟
  └─ 独立接入 CPU 端轻量 SynCheck 监控与 FOD 选择性回溯/拒答，完成全流程渐进消融表
        │
        ▼
Step 5: 垂直领域泛化压力测试 (Domain Stress Test)
  └─ 锁定最优超参，进入 TimeQuestions 跨期演化与 FiQA/FinQA 金融数据，评测零样本领域迁移表现
```

---

## 六、 结论与原型冻结签署

本敲定书正式确立了 **`U-GraphRAG V0.1`** 作为 `[SDUFE] GeminiOutOfToken_Team` 的核心选题原型。  
战队全体成员停止组件层面的无序发散，严格围绕本方案第 二、三、五 节所定义的数学定义、数据集矩阵与 5 步递进路线推进后续代码开发与实验论证。
