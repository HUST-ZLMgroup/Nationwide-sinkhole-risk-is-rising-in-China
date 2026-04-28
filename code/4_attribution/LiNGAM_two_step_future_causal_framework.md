# 第二点结论归因分析研究框架

## 1. 研究目标

本部分归因分析的目标不是再做一次预测模型重要性排序，而是构建一套**以 LiNGAM 为核心**的因果归因框架，用于回答两个问题：

1. 在现有 12 个驱动因素与地面塌陷灾害之间，哪些因果方向在经验知识上是合理的、哪些是不合理的。
2. 在未来情景下，这些因果关系是否稳定出现，以及哪些因果链在不同 `SSP × 年份` 组合中具有较高重复性和可解释性。

换句话说，这部分工作的核心不是 `RF + LiNGAM`，而是：

**先验知识约束 + LiNGAM 重复计算 + 稳定因果链筛选**

---

## 2. 总体思路

拟采用两步法。

### 第一步：基于经验知识构建先验因果约束

对 12 个因素与灾害节点之间的潜在因果方向进行人为约束，形成一套**先验知识矩阵**。  
该步骤的目标是排除明显不合理的边，例如：

- `LAI -> Depth_to_Bedrock`
- `UrbanFrac -> Distance_to_Fault`
- `Precip -> Distance_to_karst`
- `Disaster -> 任意驱动因素`

也就是说，先验知识的作用不是“替代” LiNGAM，而是为 LiNGAM 提供一个合理的搜索空间，避免出现物理意义或地学意义上明显错误的因果方向。

### 第二步：在未来情景下重复运行 LiNGAM

针对每个未来情景组合，重复抽样并运行 LiNGAM，例如：

- `ssp1-2040`
- `ssp1-2060`
- `ssp1-2080`
- `ssp1-2100`
- …
- `ssp5-2100`

每个 `SSP × 年份` 组合重复运行 `100` 次。  
每次运行都记录因果边、方向、符号和效应大小。  
最终只保留：

- 出现率 `>= 80%`
- 不违反先验知识
- 不存在明显物理或地学逻辑问题
- 方向与符号较稳定

的因果链，用于后续展示。

---

## 3. 节点体系

本研究拟使用 `12` 个驱动因素，加上 `1` 个灾害节点，共 `13` 个节点。

### 3.1 三大类因素

当前建议沿用项目中已经较成熟的三类划分：

| 大类 | 缩写 | 变量 |
|---|---|---|
| Anthropogenic activities | UF | UrbanFrac |
| Anthropogenic activities | IP | ImperviousIndex |
| Anthropogenic activities | PT | PopTotal |
| Anthropogenic activities | LAI | LAI |
| Anthropogenic activities | WTD | WTD |
| Climate change | PR | Precip |
| Climate change | TAS | Tas |
| Climate change | HUSS | Huss |
| Hydrogeology | DK | Distance_to_karst |
| Hydrogeology | DB | Depth_to_Bedrock |
| Hydrogeology | DF | Distance_to_Fault_m |
| Hydrogeology | HDS | HDS |

### 3.2 灾害节点

灾害节点建议记为：

- 历史阶段：`Disaster`
- 未来阶段：`Predicted disaster risk` 或 `Sinkhole risk`

注意：

- 历史阶段可以直接使用观测到的 `Disaster`。
- 未来阶段通常没有真实灾害观测，因此更合理的做法是使用模型输出的**灾害发生概率 / 易发性概率**作为最终响应节点。

这意味着，未来阶段虽然仍然讨论“地面塌陷灾害”，但统计意义上的终点节点实际上是：

**预测灾害风险，而不是未来真实观测灾害事件。**

### 3.3 未来阶段终点节点的具体定义

未来阶段的终点节点建议明确记为：

- `Predicted sinkhole probability`

而且这个概率不是任意构造出来的概率，而是严格沿用当前 GWR 分类工作流中的连续概率口径。

当前项目里的实际过程是：

1. GWR 模型先输出原始**回归值 / raw score**。
2. 再将这个 raw score 用训练阶段拟合好的稳健 sigmoid 变换映射到 `0-1`。
3. 这个 `0-1` 连续值作为未来归因阶段的最终响应节点输入 LiNGAM。

因此，未来阶段不建议使用：

- 离散风险等级
- Jenks 自然断点分级结果
- 人为阈值二分类标签

而应使用：

- **连续型风险概率**

### 3.4 GWR 概率映射的实现口径

这一点必须与现有模型训练代码保持一致。

参考代码位置：

- [Train_National_GWR_classification.ipynb](/path/to/sinkhole-risk-china/code/3_gwr_model_train/national/GWR/Train_National_GWR_classification.ipynb)
- [gwr_sigmoid_utils.py](/path/to/sinkhole-risk-china/code/mgtwr/gwr_sigmoid_utils.py)

当前实际口径不是普通直接 sigmoid，而是：

- 先在训练集 raw GWR scores 上拟合 `transform_metadata`
- 其中包括：
  - `center`
  - `scale`
  - `clip_z`
- 再对未来情景 raw GWR scores 使用**同一组训练集 metadata**做稳健 sigmoid 映射

对应函数是：

- `mgtwr.gwr_sigmoid_utils.gwr_scores_to_probabilities`

其核心逻辑是：

1. `z = (raw_score - center) / scale`
2. `z` 被裁剪到 `[-clip_z, clip_z]`
3. 再通过 sigmoid 映射到 `0-1`

这里最重要的约束是：

**未来情景不能重新单独拟合自己的 sigmoid 参数，而必须复用训练阶段得到的 transform metadata。**

否则不同情景之间的概率尺度将不再可比，后续 LiNGAM 的终点节点定义也会失去一致性。

---

## 4. 第一步：先验知识构建框架

## 4.1 先验知识的作用

先验知识矩阵的目标是将 LiNGAM 的搜索限制在“有物理意义、地学意义和时间逻辑意义”的范围内。

这一步建议构建两类规则：

### A. 硬约束（hard constraints）

这些边一律禁止出现。

建议包括：

- `Disaster -> 任意驱动因素` 禁止
- 任意动态因子 `->` 静态地质背景因子 禁止
- 人类活动因子 `-> DK / DB / DF / HDS` 禁止
- 气候变化因子 `-> DK / DB / DF / HDS` 禁止
- 生态状态变量 `LAI -> DB / DF / DK / HDS` 禁止
- 任意自环禁止

这里“静态地质背景因子”建议包括：

- `DK`
- `DB`
- `DF`
- `HDS`

### 4.1.1 节点级先验约束建议

为了让先验矩阵可以直接编码，建议把 13 个节点拆成 5 组角色：

#### A. 地质背景根节点

- `DK`
- `DB`
- `DF`
- `HDS`

建议作为**根节点 / exogenous nodes** 处理。

含义是：

- 不接受任何来自其他节点的入边
- 尤其禁止：
  - `UF/IP/PT/LAI/WTD -> DK/DB/DF/HDS`
  - `PR/TAS/HUSS -> DK/DB/DF/HDS`
  - `Disaster probability -> DK/DB/DF/HDS`

是否允许这 4 个地质背景变量之间相互连边，有两种口径：

- 更保守口径：全部禁止，统一视为并列背景条件
- 稍宽松口径：允许 `DK/DB/DF/HDS` 之间存在方向，但不作为论文重点解释

从可解释性出发，建议优先采用：

- **地质背景变量之间也不设方向边**

#### B. 气候强迫根节点

- `PR`
- `TAS`
- `HUSS`

建议作为第二组 exogenous nodes。

理由是：

- 这些变量在当前研究设计里代表同一时间窗口内的外部气候状态
- 它们不应被地质背景、人类活动或灾害节点反向决定

建议硬约束：

- 禁止 `UF/IP/PT/LAI/WTD -> PR/TAS/HUSS`
- 禁止 `DK/DB/DF/HDS -> PR/TAS/HUSS`
- 禁止 `Disaster probability -> PR/TAS/HUSS`

对 `PR/TAS/HUSS` 三者之间的方向边，建议也先禁止，原因是：

- 它们来源于同一时间窗统计量
- 同时放开容易形成难解释的“同层内部因果”

#### C. 人类活动根-中介节点

- `PT`
- `UF`
- `IP`

建议做更细的方向约束：

- `PT -> UF` 允许
- `PT -> IP` 允许
- `UF -> IP` 允许

相反方向建议禁止：

- `UF -> PT`
- `IP -> PT`
- `IP -> UF`

原因是当前尺度下更合理的解释是：

- 人口压力推动城市扩张
- 城市扩张进一步提高不透水率

#### D. 环境状态中介节点

- `WTD`
- `LAI`

这两个节点建议作为主要中介层。

##### `WTD` 的允许入边

- `DK/DB/DF/HDS -> WTD`
- `PR/TAS/HUSS -> WTD`
- `PT/UF/IP -> WTD`

##### `WTD` 的禁止入边

- `Disaster probability -> WTD`
- `LAI -> WTD` 建议先禁止

这里建议优先采用：

- `WTD -> LAI`
- 而不是 `LAI -> WTD`

因为在当前研究尺度下，更容易解释为：

- 地下水埋深或地下水环境制约植被状态

而非：

- 植被反向决定地下水埋深

##### `LAI` 的允许入边

- `PR/TAS/HUSS -> LAI`
- `PT/UF/IP -> LAI`
- `WTD -> LAI`
- 如需更宽松，也可允许 `DK/DB/DF/HDS -> LAI`

##### `LAI` 的禁止入边/出边

- 禁止 `Disaster probability -> LAI`
- 禁止 `LAI -> DK/DB/DF/HDS`
- 禁止 `LAI -> PR/TAS/HUSS`
- 禁止 `LAI -> PT/UF/IP`

#### E. 灾害风险终点节点

- `Predicted sinkhole probability`

它必须作为**纯 sink node** 处理。

允许的入边：

- 所有 12 个驱动因素都可以指向该节点

禁止的出边：

- `Predicted sinkhole probability -> 任意其他节点`

### 4.1.2 建议的先验约束矩阵语义

在真正落地到 LiNGAM 前，建议把矩阵语义明确成三种状态：

- `forbidden`
- `allowed`
- `unspecified`

具体上：

- **forbidden**：根据地学 / 物理 / 时间逻辑，明确不允许
- **allowed**：方向合理，允许 LiNGAM 判断是否存在
- **unspecified**：理论上不完全排斥，但当前不作为重点解释对象

论文分析时，主文图只解释：

- `allowed` 且稳定出现的边

### 4.1.3 一组更具体的推荐 allowed edges

为了减少后续编码时的模糊性，建议优先允许以下边：

- `PT -> UF`
- `PT -> IP`
- `UF -> IP`
- `PR -> WTD`
- `TAS -> WTD`
- `HUSS -> WTD`
- `PR -> LAI`
- `TAS -> LAI`
- `HUSS -> LAI`
- `DK -> WTD`
- `DB -> WTD`
- `DF -> WTD`
- `HDS -> WTD`
- `PT -> WTD`
- `UF -> WTD`
- `IP -> WTD`
- `PT -> LAI`
- `UF -> LAI`
- `IP -> LAI`
- `WTD -> LAI`
- `DK -> Disaster probability`
- `DB -> Disaster probability`
- `DF -> Disaster probability`
- `HDS -> Disaster probability`
- `PR -> Disaster probability`
- `TAS -> Disaster probability`
- `HUSS -> Disaster probability`
- `PT -> Disaster probability`
- `UF -> Disaster probability`
- `IP -> Disaster probability`
- `LAI -> Disaster probability`
- `WTD -> Disaster probability`

### 4.1.4 一组更具体的推荐 forbidden edges

建议明确写入 prior matrix 的禁止边包括：

- `LAI -> DK`
- `LAI -> DB`
- `LAI -> DF`
- `LAI -> HDS`
- `WTD -> DK`
- `WTD -> DB`
- `WTD -> DF`
- `WTD -> HDS`
- `UF -> DK/DB/DF/HDS`
- `IP -> DK/DB/DF/HDS`
- `PT -> DK/DB/DF/HDS`
- `PR -> DK/DB/DF/HDS`
- `TAS -> DK/DB/DF/HDS`
- `HUSS -> DK/DB/DF/HDS`
- `Disaster probability -> 任意驱动因素`
- `LAI -> WTD`
- `UF -> PR/TAS/HUSS`
- `IP -> PR/TAS/HUSS`
- `PT -> PR/TAS/HUSS`
- `PR/TAS/HUSS -> DK/DB/DF/HDS`

### B. 软约束（soft expectations）

这些边不是强制要求一定存在，但若出现则更容易被接受。

建议包括：

- `PR / TAS / HUSS -> LAI`
- `PR / TAS / HUSS -> WTD`
- `UF / IP / PT -> LAI`
- `UF / IP / PT -> WTD`
- `DK / DB / DF / HDS -> WTD`
- `DK / DB / DF / HDS -> Disaster`
- `PR / TAS / HUSS -> Disaster`
- `UF / IP / PT / LAI / WTD -> Disaster`

软约束主要用于解释阶段，而不是用于机械地裁剪结果。

---

## 4.2 推荐的因果层级结构

为了避免未来网络过于混乱，建议采用分层理解：

### 第一层：背景约束层

- `DK`
- `DB`
- `DF`
- `HDS`

这类变量主要代表相对稳定的地质 / 地貌背景，不应被人类活动或短期气候状态反向决定。

### 第二层：环境调节层

- `PR`
- `TAS`
- `HUSS`
- `UF`
- `IP`
- `PT`
- `LAI`
- `WTD`

这些变量允许存在相互作用，但建议优先允许“气候 / 人类活动 -> 地表与地下环境状态”的方向。

### 第三层：灾害响应层

- `Disaster` 或 `Predicted disaster risk`

灾害节点应当作为终点节点（sink），不再向上游变量发出因果边。

---

## 5. 第二步：未来情景下重复 LiNGAM

## 5.1 计算对象

对每个未来情景组合分别运行：

- `ssp1, ssp2, ssp3, ssp4, ssp5`
- `2040, 2060, 2080, 2100`

如果需要和历史进行对照，也可以加入：

- `hist-2020`

---

## 5.2 每个情景的重复计算方案

对每个 `SSP × 年份` 组合，重复计算 `100` 次。

每次重复建议流程如下：

1. 先基于当前 GWR 预测结果得到该情景下每个样本的 **continuous probability**。
2. 抽取该情景样本的固定比例子样本。
3. 抽样方式固定为：**无放回抽样**。
4. 抽样比例固定，例如统一使用 `80%`。
5. 对 12 个驱动因素和终点节点做统一标准化。
6. 使用带先验知识矩阵的 `DirectLiNGAM` 拟合。
7. 记录所有边的方向、符号和总效应。

这里建议进一步明确：

- 每个情景都采用同样的抽样比例
- 每个情景都重复 `100` 次
- 每次重复仅改变随机种子，不改变流程

### 5.2.1 抽样方式的最终建议

这一步建议正式固定为：

- **固定比例无放回抽样**

原因有三点：

1. 各情景之间结果更容易横向比较
2. 避免 bootstrap 有放回抽样带来的重复样本堆积
3. 更适合后续统计“因果链出现率”

推荐实现方式：

- 设该情景有效样本量为 `N`
- 每次抽取 `floor(r × N)` 个样本
- 其中 `r` 为固定比例，例如 `0.8`
- 每次重复使用不同 `random_state`

如果后续发现区域样本组成波动过大，再考虑：

- 在 `DIV_EN` 内部分层后做等比例无放回抽样

但第一版建议先用：

- **整体固定比例无放回抽样**

如果未来阶段使用的是模型输出风险，则每次输入的最后一个节点应为：

- `Predicted disaster probability`
- 或 `Predicted sinkhole susceptibility`

而不是历史观测 `Disaster`。

### 5.2.2 未来风险概率的具体输入口径

每次 LiNGAM 重复计算前，未来阶段的终点节点都建议按以下固定流程得到：

1. 使用 GWR 预测得到 raw regression score
2. 调用训练阶段拟合好的 sigmoid metadata 做映射
3. 得到 `Predicted sinkhole probability`
4. 将该概率作为 LiNGAM 中唯一的灾害终点节点

建议写成一条明确规则：

**所有未来情景的灾害节点统一定义为：由 GWR raw score 经训练集拟合的 robust sigmoid 变换后得到的连续型风险概率。**

---

## 5.3 每次运行需记录的结果

每次 LiNGAM 运行至少记录以下内容：

- 邻接矩阵 `B`
- 总效应矩阵
- 每条边是否出现
- 每条边方向
- 每条边正负号
- 每条边效应值大小

建议统一输出为长表格式，例如：

| scenario | repeat_id | source | target | direct_effect | total_effect | sign |
|---|---:|---|---|---:|---:|---|
| ssp1_2040 | 1 | PR | WTD | 0.21 | 0.29 | + |
| ssp1_2040 | 1 | WTD | Disaster | 0.18 | 0.18 | + |

---

## 6. 稳定因果链筛选规则

未来最终展示的不是“所有跑出来的边”，而是**稳定且合理的因果链**。

建议采用以下筛选标准。

### 6.1 边稳定性标准

一条边 `A -> B` 被保留，需要同时满足：

- 出现率 `>= 80%`
- 方向一致
- 符号一致率 `>= 80%`
- 中位总效应不接近 `0`
- 不违反先验知识

### 6.2 链稳定性标准

一条链，例如：

- `PR -> WTD -> Disaster`
- `UF -> LAI -> Disaster`
- `DB -> WTD -> Disaster`

被保留，需要满足：

- 链中每条边都满足边稳定性标准
- 链的整体方向符合经验逻辑
- 链不存在明显“反常方向”

例如以下链应被排除：

- `Disaster -> PR`
- `LAI -> DB -> Disaster`
- `UrbanFrac -> Distance_to_Fault -> Disaster`

### 6.3 建议的最小报告标准

最终图文中建议只展示：

- 出现率 `>= 80%` 的单边
- 出现率 `>= 80%` 的两跳链
- 如需更严格，可将核心主文图提升到 `>= 90%`

---

## 7. 建议的结果展示形式

## 7.1 先验知识部分

建议输出一张“先验约束框架图”，展示：

- 三大类因素
- 12 个变量
- 灾害节点
- 明确禁止的方向
- 允许存在的主要方向

这张图的作用是告诉读者：

**LiNGAM 不是在无约束乱找边，而是在经验知识支持下识别稳定因果结构。**

## 7.2 未来情景部分

建议输出三类结果。

### A. 单情景稳定因果网络图

每个 `SSP × 年份` 一张图，仅显示：

- 出现率 `>= 80%`
- 不违反先验知识
- 符号稳定

的边。

### B. 跨情景稳定性热图

横轴为 `SSP × 年份`，纵轴为边或链，颜色表示出现率。

这张图适合展示：

- 哪些因果链具有跨情景稳定性
- 哪些链只在高排放情景后期出现

### C. 主文图只保留高频主链

主文可只保留最稳定、最有解释力的链，例如：

- `Climate -> WTD -> Disaster`
- `Anthropogenic activities -> LAI / WTD -> Disaster`
- `Hydrogeology -> Disaster`

---

## 8. 推荐的技术实现路径

建议在现有代码基础上拆成三个模块。

### 模块 A：先验知识生成

输出：

- `prior_knowledge_matrix.csv`
- `prior_knowledge_edges_allowed.csv`
- `prior_knowledge_edges_forbidden.csv`

### 模块 B：情景重复 LiNGAM

输入：

- 某一 `SSP × 年份` 的样本数据
- 先验知识矩阵

输出：

- `edge_records_long.csv`
- `adjacency_repeat_*.csv`
- `total_effect_repeat_*.csv`

### 模块 C：稳定链汇总与作图

输出：

- `stable_edges_freq_ge_0.8.csv`
- `stable_chains_freq_ge_0.8.csv`
- `scenario_stable_network.svg`
- `cross_scenario_chain_heatmap.svg`

---

## 9. 建议的论文表述口径

为了避免方法口径混乱，建议论文中明确写成：

> We adopted a two-step LiNGAM-based causal attribution framework.  
> First, prior knowledge was used to constrain physically implausible directions among the 12 drivers and the sinkhole disaster node.  
> Second, for each future scenario, LiNGAM was repeatedly fitted on resampled datasets, and only causal links with occurrence frequencies of at least 80% and without obvious physical inconsistencies were retained.

中文可表述为：

> 本研究采用“两步法 LiNGAM 因果归因框架”。首先，基于经验知识约束 12 个驱动因素与地面塌陷灾害之间不合理的因果方向；其次，针对每个未来情景进行重复抽样和 LiNGAM 拟合，仅保留出现率不低于 80%、且不存在明显物理逻辑问题的稳定因果链。

---

## 10. 当前最需要确认的两个技术口径

在正式编码前，建议优先确认以下两点。

### A. 未来阶段的终点节点

这里现在建议直接固定，不再作为待确认项：

- 使用 **GWR 输出经 sigmoid 映射后的连续型风险概率**

不使用：

- 二元标签替代值
- Jenks 等分级后的风险等级

并且必须复用训练阶段已有的 `transform_metadata`。

### B. 抽样方式

这里也建议直接固定，不再作为待确认项：

- 采用 **固定比例无放回抽样**

推荐默认值：

- 抽样比例 `r = 0.8`
- 每个情景重复 `100` 次

---

## 11. 本框架的简化版一句话总结

**经验知识限定可接受因果方向，LiNGAM 在各未来情景下重复识别因果结构，最终仅报告出现率不低于 80% 且不存在明显逻辑问题的稳定因果链。**
