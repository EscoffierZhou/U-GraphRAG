# 科研课题立项论证与技术方案设计报告 (Research Proposal)
**项目战队**：[SDUFE] GeminiOutOfToken_Team  
**赛事课题**：Agentic Research：从科学问题到高水平论文原型  
**拟投目标**：中科院 1/2 区 / TOP 期刊 (如 KBS / ESWA / IEEE TKDE) 或 CCF 高水平学术会议  
**研究周期**：7天科研创造冲刺（自动化实验与证据闭环）

---

## 1. 核心科学问题定义 (Research Question)

### 1.1 研究背景与痛点
大语言模型（LLMs）与自主智能体（LLM-based Autonomous Agents）在金融研报分析、舆情监控与量化投资决策中展现出前所未有的文本理解与推理潜力。然而，真实金融市场环境具有**极高信噪比、多源虚假信息（谣言/夸大吹捧/蓄意做空恐慌）以及对抗性噪声**。

现有前沿研究暴露出单智能体（Single Agent）与传统时序/情感模型的两大致命缺陷：
1. **认知易感性与过度自信（Overconfidence & Cognitive Fragility）**：面对虚假研报或对抗性谣言，单大模型极易产生幻觉（Hallucination），其置信度严重失真（校准误差 ECE 偏高）；
2. **缺乏多重视角交叉验证与审计制衡**：现有方法大多依赖单一提示词（Direct Prompting）或链式思考（CoT），无法模拟真实金融机构内部“分析师（Analyst）- 审慎风控官（Risk Controller）- 独立审计师（Arbiter/Auditor）”的对抗博弈与动态信念更新机制。

### 1.2 核心研究问题 (Research Question, RQ)
> **RQ**: 如何设计一种轻量级、Token 高效的多智能体对抗辩论与贝叶斯信念校准架构（FinTrust-Agent），使得智能体决策系统在面对不同强度的高噪声金融信息时，依然能够保持决策鲁棒性、降低校准误差并实现稳定的风险收益？

### 1.3 科学假设 (Scientific Hypotheses, $H_1, H_2, H_3$)
* **假设 $H_1$ (抗噪鲁棒性)**：引入红蓝对抗辩论机制（Adversarial Debate）能显著过滤虚假夸大信息，在 20%~50% 噪声扰动下，决策 F1 分数与夏普比率衰减幅度显著低于单模型基线（$p < 0.01$）。
* **假设 $H_2$ (可信校准度)**：通过审慎审计师的动态信念校准（Dynamic Belief Calibration），能有效压缩预测概率与真实准确率之间的预期校准误差（ECE），消除大模型的“盲目自信”。
* **假设 $H_3$ (Token 高效性 - 契合 GeminiOutOfToken 队名)**：基于信息熵的动态记忆剪枝算法（Token-Efficient Memory Pruning）可在减少 60% 上下文 Token 消耗的前提下，保留 95% 以上的核心决策证据。

---

## 2. 文献与研究缺口 (Literature Gap & Benchmark)

| 研究门类 | 代表性工作 | 核心局限性 | 本研究的突破点 (Our Delta) |
| :--- | :--- | :--- | :--- |
| **传统金融情感分析** | FinBERT (Araci, 2019), RoBERTa-Fin | 仅做表层情感分类，缺乏长程因果逻辑推断，对结构化虚假研报无防御力 | 从表层词频分类跃迁至**因果链提取与审慎交叉验证** |
| **单智能体直接推理** | GPT-4 / DeepSeek-V3 (CoT / ToT) | 在复杂对抗性扰动下容易顺从噪声逻辑，校准误差偏大 | 引入**对立红方批判 Agent**，形成博弈平衡 |
| **通用多智能体系统** | MetaGPT (Hong et al., 2023), ChatDev | 面向软件开发/代码生成，通信轮次冗长、易发生上下文溢出（Out of Token） | **专为金融量化设计的轻量三方博弈 + 记忆剪枝机制** |

---

## 3. FinTrust-Agent 架构设计与理论机制

系统由三大核心 Agent 角色构成闭环：
1. **市场信息感知体 (Information Extraction & Analyst Agent)**：
   提取原始文本中的事实主张（Factual Claims）、因果关系链与初始观点，输出初步预测分布 $P_0(y|X)$。
2. **怀疑论对抗体 (Skeptic & Risk Control Agent)**：
   专门承担“空方/挑刺者”角色，对初步事实主张进行反向质询（Counter-factual Probing），寻找事实断裂点、过度承诺或夸大词汇，形成批判报告 $R_{critic}$。
3. **独立仲裁与置信校准体 (Arbiter & Calibration Agent)**：
   结合多方证据向量与质疑报告，执行贝叶斯式置信度重加权，计算最终决策：
   $$\hat{y} = \arg\max_y \sum_{k} w_k \cdot \sigma\left(\frac{z_{k}}{\tau}\right)$$
   其中 $\tau$ 为温度校准系数，$w_k$ 为基于对抗证据密度的自适应置信权重。

---

## 4. 实验设计与评估指标体系

### 4.1 核心评价指标
1. **决策准确度**：Macro-F1, Precision, Recall, Accuracy
2. **可信校准度**：Expected Calibration Error (ECE), Brier Score
3. **量化金融表现**：Cumulative Return (累计收益), Sharpe Ratio (夏普比率), Max Drawdown (最大回撤)
4. **系统效率**：Average Tokens per Decision, Latency (ms)

### 4.2 对照组 (Baselines)
* Baseline 1: 传统 ML / NLP 基准 (FinBERT / TF-IDF + XGBoost)
* Baseline 2: 单大模型直接预测 (Direct LLM)
* Baseline 3: 单大模型链式思考 (CoT Prompting)
* Ours (Full): FinTrust-Agent
* Ours (Ablation): w/o Adversarial Debate, w/o Belief Calibration, w/o Memory Pruning

---

## 5. 7天冲刺时间线与里程碑
* **Day 1**: 确定科学问题、数据集审计与环境构建（已完成）
* **Day 2**: 运行 Baseline 代码并获取第一批对照指标
* **Day 3**: 跑通 FinTrust-Agent 核心多智能体博弈与抗噪实验
* **Day 4**: 运行 5-Fold 交叉检验、消融实验与统计显著性检验 ($p$-value)
* **Day 5**: 绘制顶会级高清学术图表 (Figure 1~4)
* **Day 6**: 生成标准 LaTeX 双栏论文原型并运行 AI 审稿人对抗审查
* **Day 7**: 整理代码仓库、一键复现包与答辩汇报 PPT
