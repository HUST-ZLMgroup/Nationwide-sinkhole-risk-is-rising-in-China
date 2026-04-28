# 第二点结论归因分析出图方案

## 1. 出图总原则

这部分图不建议再沿用“单张图各说一点”的组织方式，而应直接围绕两点子结论组织成 **两个组图**。  
每个组图内部至少包含 `2` 张子图，最好控制在 `2-3` 张，形成一条完整证据链。

建议把第二点结论拆成以下两个子结论：

### 子结论 1

**地面塌陷风险并不是由单一变量决定，而是由水文地质背景、气候强迫和人类活动共同驱动，其中一些路径具有明确的中介结构。**

这一点需要回答两个问题：

- 哪些因果方向在机理上是合理且被模型支持的
- 三大类因素中，谁是直接作用，谁更多通过中介变量间接作用

### 子结论 2

**在未来情景下，稳定的因果链并非随机波动，而是在不同 `SSP × 年份` 组合中重复出现；同时，不同情景下主导路径强度会发生系统变化。**

这一点需要回答两个问题：

- 哪些边或链条在未来情景中反复出现
- 哪些路径是“稳定存在”，哪些路径是“情景敏感”

---

## 2. 组图一：当前机制归因图组

这一组图用于支撑 **子结论 1**。

建议至少包含 `3` 张子图。

### 图 1a：先验知识约束框架图

#### 图意

展示 12 个驱动因素与终点节点之间的理论结构，不是数据结果图，而是方法框架图。

#### 推荐图型

- 分层有向图
- 左到右三层或四层结构

建议分层：

1. `Hydrogeology`
   - `DK`
   - `DB`
   - `DF`
   - `HDS`
2. `Climate change`
   - `PR`
   - `TAS`
   - `HUSS`
3. `Anthropogenic activities / mediators`
   - `PT`
   - `UF`
   - `IP`
   - `WTD`
   - `LAI`
4. `Predicted sinkhole probability` 或历史阶段 `Disaster`

#### 想表达的点

- 你不是让 LiNGAM 无约束地乱找边
- 你是先用经验知识排除了明显错误方向
- 真正的结果是在受约束空间内识别到的

#### 这张图的价值

这张图是整组图的入口。  
它本身不证明结果，但能提前解决“因果方向是否乱指”的质疑。

---

### 图 1b：全国尺度 LiNGAM 总效应排序图

#### 图意

展示 12 个变量对终点节点的总效应强度，并按三大类着色。

#### 推荐图型

- 横向条形图
- 按 `|total effect|` 从大到小排序
- 条形颜色对应三大类：
  - 水文地质
  - 人类活动
  - 气候变化

#### 建议横轴

- `|Total effect on sinkhole probability|`

#### 建议纵轴

- 12 个变量缩写或全称

#### 想表达的点

- 三大类因素都进入了有效驱动集合
- 不是只有气候，或者只有人类活动在起作用
- 可以直观看出哪些变量更偏“直接控制”，哪些只是次级变量

#### 和现有内容的衔接

这个图可以沿用：

- [lingam_division.ipynb](/path/to/sinkhole-risk-china/code/4_attribution/national_fig_lingam/lingam_division.ipynb)

里的总效应思路，但要从原先的 `Disaster` 口径，转向你新定义的：

- 历史阶段：`Disaster`
- 未来阶段：`Predicted sinkhole probability`

---

### 图 1c：三大类因素到终点节点的路径分解图

#### 图意

这张图不再停留在“单变量有多重要”，而是展示：

- 哪些变量是直接指向风险概率
- 哪些变量主要通过 `WTD` / `LAI` 等中介变量间接起作用

#### 推荐图型

优先推荐二选一：

- 三列桑基图
- 简化 DAG 路径图

如果追求论文展示力，优先用：

- `三列桑基图`

三列分别为：

1. 12 个变量
2. 三大类或关键中介节点
3. `Sinkhole probability`

#### 想表达的点

- 水文地质背景更多体现为“背景约束 + 直接作用”
- 气候变化更多通过 `WTD` 等中介路径传递
- 人类活动既有直接路径，也有通过生态 / 水文状态的间接路径

#### 和现有内容的衔接

这张图最接近现有：

- [LiNGAM_sankey_updated_12vars_equalheight_v2.ipynb](/path/to/sinkhole-risk-china/code/4_attribution/national_fig_lingam/LiNGAM_sankey_updated_12vars_equalheight_v2.ipynb)

但需要把终点节点口径统一成：

- `Predicted sinkhole probability`

#### 这张图的必要性

如果没有这张图，图 1b 只能说明“谁重要”；  
有了这张图，才能说明“为什么重要，以及通过什么链条重要”。

---

## 3. 组图二：未来情景稳定因果链图组

这一组图用于支撑 **子结论 2**。

建议至少包含 `3` 张子图。

### 图 2a：稳定因果边出现频率热图

#### 图意

对每个 `SSP × 年份` 组合重复抽样、重复运行 LiNGAM 后，统计每一条候选边的出现频率。

#### 推荐图型

- 热图

#### 建议坐标

- 行：候选边，例如
  - `PR -> WTD`
  - `WTD -> Risk`
  - `UF -> LAI`
  - `DB -> Risk`
- 列：未来情景组合，例如
  - `ssp1-2040`
  - `ssp1-2060`
  - ...
  - `ssp5-2100`

#### 色值含义

- 单元格颜色表示该边在 `100` 次重复中的出现频率
- 可以在 `80%` 位置加显著阈值标记

#### 想表达的点

- 哪些边是跨情景稳定存在的
- 哪些边只在高排放或远未来阶段增强
- 哪些边本来以为重要，但实际上出现频率并不稳定

#### 这是这组图里最关键的一张

因为它直接把“稳定”这个概念量化了。  
这张图是未来归因部分的主证据图。

---

### 图 2b：高频稳定因果链汇总图

#### 图意

把图 2a 中出现频率 `>= 80%` 的边筛出来，组合成最终保留的稳定因果链网络。

#### 推荐图型

- 精简 DAG 图
- 或情景汇总桑基图

优先建议：

- `精简 DAG 图`

原因是未来阶段重点不是“流量感”，而是“链条结构稳定性”。

#### 建议画法

- 仅展示通过先验知识筛选且出现频率 `>= 80%` 的边
- 边粗表示平均效应绝对值
- 边颜色表示正负号
- 节点颜色表示所属大类

#### 想表达的点

- 真正稳定保留下来的链条并不多
- 这些链条有明确的组织方式，例如：
  - `Climate -> WTD -> Risk`
  - `Anthropogenic -> LAI / WTD -> Risk`
  - `Hydrogeology -> Risk`

#### 与图 2a 的关系

- 图 2a 是“边级别统计”
- 图 2b 是“论文主结果网络”

这两张图配套后，逻辑会比较完整。

---

### 图 2c：情景间主导路径强度变化图

#### 图意

这张图不是看“边有没有”，而是看：

- 三大类路径在不同情景下谁更强
- 主导机制是否随着 `SSP` 和时间推进发生偏移

#### 推荐图型

二选一：

- 分组折线图
- 堆叠条形图

更推荐：

- `分组折线图`

#### 推荐指标

把边先聚合成三类路径贡献：

- `Hydrogeology -> Risk`
- `Climate / Climate-mediated -> Risk`
- `Anthropogenic / Anthropogenic-mediated -> Risk`

然后针对每个 `SSP × 年份` 计算：

- 平均总效应绝对值
- 或稳定边的平均加权强度

#### 想表达的点

- 未来变化不是只有“更多风险”，而是“驱动结构在变”
- 例如：
  - 近未来仍以背景约束为主
  - 远未来气候-水文链条增强
  - 某些 SSP 下人类活动链条更强

#### 这张图的作用

如果没有图 2c，未来部分容易停留在“重复出现了哪些边”；  
有了图 2c，才真正能说“机制结构随情景发生了怎样的系统变化”。

---

## 4. 两个组图分别对应的论文表述

为了避免出图和结论脱节，建议把论文里的两点子结论直接定成下面这种写法。

### 子结论 1 对应表述

> Sinkhole risk is jointly shaped by hydrogeological background, climate forcing, and anthropogenic disturbance, with part of the influence transmitted through mediating environmental states such as groundwater conditions and vegetation status.

对应证据：

- 图 1a：先验因果结构是合理约束的
- 图 1b：三大类变量都有可观总效应
- 图 1c：存在清晰的中介路径分解

### 子结论 2 对应表述

> Under future scenarios, only a limited subset of causal links remains stable across repeated LiNGAM runs, and the strength of dominant pathways shifts systematically with scenario intensity and time horizon.

对应证据：

- 图 2a：稳定性是按出现率量化的
- 图 2b：最终只保留高频且合理的因果链
- 图 2c：不同情景下主导路径强度发生系统偏移

---

## 5. 推荐的最终出图组合

如果你只准备做两组图，每组图都不少于 `2` 张子图，我建议直接定成下面这套。

### 方案 A：最稳妥版本

#### 组图一：当前机制

- `a` 先验知识约束框架图
- `b` 全国尺度 LiNGAM 总效应排序图
- `c` 三大类因素到风险概率的路径分解图

#### 组图二：未来稳定机制

- `a` 稳定因果边出现频率热图
- `b` 高频稳定因果链 DAG 图
- `c` 三大类主导路径强度变化折线图

这是最推荐的版本。  
逻辑完整，而且每张图承担的职责清楚，不重复。

---

## 6. 如果你想压缩工作量

如果你希望先做一个更轻量但仍然成立的版本，可以压缩成每组 `2` 张图。

### 轻量版组图一

- `a` 全国尺度 LiNGAM 总效应排序图
- `b` 三列桑基路径图

### 轻量版组图二

- `a` 稳定因果边出现频率热图
- `b` 高频稳定因果链 DAG 图

这个版本能成立，但缺点是：

- 少了方法约束图，审稿人更容易追问先验是怎么定的
- 少了未来主导路径变化图，未来部分会更像“边筛选结果”，而不是“机制变化分析”

---

## 7. 现阶段最建议优先落地的图

如果按实现顺序排，我建议先做下面四张：

1. 图 1b：全国尺度 LiNGAM 总效应排序图
2. 图 1c：三列桑基路径图
3. 图 2a：稳定因果边出现频率热图
4. 图 2b：高频稳定因果链 DAG 图

原因很直接：

- 这四张图最接近你现有代码基础
- 它们已经足够组成两组各 `2` 张子图的论文主图
- 后续若时间允许，再补图 1a 和图 2c，提高论文解释力

---

## 8. 一句话结论

最适合你的两组图，不是“一个做相关性、一个做重要性”，而是：

- **组图一回答：当前风险是怎样被三大类因素共同驱动的**
- **组图二回答：未来哪些因果链是稳定存在的，以及它们如何随情景发生变化**

这样组织之后，图和第二点结论会直接一一对应，不会出现“图做了很多，但无法支撑核心结论”的问题。

---

## 9. 目录与产物组织建议

如果后面要把这套流程真正跑起来，建议在 `code/4_attribution` 下按下面的逻辑组织。

### 9.1 建议脚本目录

- `pre_attribution/`
  - 负责配置、变量映射、先验知识矩阵、样本准备
- `post_attribution/`
  - 负责汇总、稳定边筛选、路径聚合、绘图

### 9.2 建议输出目录

建议统一输出到：

- `outputs/attribution/`

其下再拆为：

- `outputs/attribution/common/`
- `outputs/attribution/current/`
- `outputs/attribution/future/`
- `outputs/attribution/figures/`

进一步建议结构：

- `outputs/attribution/common/`
  - 放变量字典、先验知识矩阵、全局配置副本
- `outputs/attribution/current/`
  - 放当前阶段 LiNGAM 结果
- `outputs/attribution/future/{ssp}/{year}/`
  - 放单情景重复 LiNGAM 的全部结果
- `outputs/attribution/figures/group1_current/`
  - 放组图一
- `outputs/attribution/figures/group2_future/`
  - 放组图二

### 9.3 统一绘图规范

这部分图后面不应由各个脚本各自决定风格，而应统一遵循同一套出图规范。

#### 输出格式

统一要求：

- 只保存 **组图**
- 不保存单独子图
- 输出格式统一为：
  - `SVG`

不建议同时导出：

- `PNG`
- `PDF`

除非投稿或排版阶段另有明确需要。

建议组图文件名简洁固定，例如：

- `group1_current_attribution.svg`
- `group2_future_stable_paths.svg`

#### 背景与画布

统一要求：

- 纯白背景
- 不使用透明底
- 不使用灰底、米色底或渐变底

建议导出时显式设置：

- `figure.facecolor = "#FFFFFF"`
- `axes.facecolor = "#FFFFFF"`

#### 字体规范

统一要求：

- 全部文字使用：
  - `Times New Roman`
- 标准字号统一为：
  - `9 pt`

适用范围包括：

- 坐标轴标题
- 坐标轴刻度
- 图例
- 节点标签
- 热图标签
- 面板标记 `a / b / c`

除非个别图型出现严重重叠，否则不要随意改变字号层级。

#### Nature 风格的基本要求

这里的 “Nature 风格” 应理解为：

- 简洁
- 克制
- 可读
- 信息密度高但不拥挤

因此统一要求：

- 不使用 3D 效果
- 不使用阴影
- 不使用装饰性背景
- 不使用粗重黑边
- 不使用高饱和荧光色
- 不使用花哨标题
- 不使用过密网格线

建议图形特征：

- 线宽中等偏细
- 坐标轴简洁
- 网格线极淡或仅保留必要参考线
- 图例紧凑，避免大块留白
- 每张图的颜色数量受控

#### 莫兰迪配色方案

所有图统一使用低饱和、偏灰度的莫兰迪色系，整体风格应接近高水平期刊常见的克制配色。

建议固定一套主色，不要每张图重新选色。

分组主色建议：

- `Hydrogeology`
  - `#7E8FA3`
- `Climate`
  - `#9AAF9A`
- `Anthropogenic`
  - `#C69074`
- `Target / Risk`
  - `#8D6E63`

辅助中性色建议：

- 白色背景
  - `#FFFFFF`
- 浅灰网格
  - `#E6E2DD`
- 中灰文字辅色
  - `#6F6A64`
- 轴线灰
  - `#8A857F`

如果某些图必须显示正负效应，可采用低饱和对比色：

- 正效应
  - `#C69074`
- 负效应
  - `#8CA0B3`

注意：

- 正负色也必须维持低饱和
- 不要使用纯红、纯蓝

#### 热图与连续色带规范

像图 2a 这类频率热图，不建议使用彩虹色。

建议：

- 使用从白色到单一莫兰迪深色的连续色带
- 例如：
  - `#FFFFFF -> #D8E0E5 -> #AEBCCA -> #7E8FA3`

如果需要强调阈值 `0.8`，建议：

- 在单元格上加数值
- 或在色标上标出 `0.8`

不要用突兀亮色硬强调。

#### 线条、边框与图例规范

建议统一：

- 主线宽：
  - `0.8 - 1.2 pt`
- 参考网格线：
  - `0.4 - 0.6 pt`
- 节点边框：
  - 不超过 `0.8 pt`

图例统一要求：

- 图例只保留必要项
- 图例文字为 `9 pt`
- 图例背景保持白色
- 图例不加重边框，或仅使用极淡边框
- 图例应靠近主图，不产生大块空白

#### SVG 导出要求

统一要求：

- 文本尽量保持为可编辑文字
- 不把所有文字转成路径
- 避免透明蒙版和复杂滤镜

原因：

- 后期论文排版更容易微调
- Illustrator / Inkscape / Affinity Designer 更容易直接编辑

#### 组图输出原则

由于你已经明确要求只保存组图，因此后续绘图脚本应直接生成组合版面，而不是先导出每张子图再拼接。

建议：

- 每个组图由一个总脚本负责排版
- 子图函数只返回 `matplotlib axes` 或 `svg` 片段
- 最终只由组图脚本执行保存

---

## 10. 固定变量命名与标准字段

如果列名不先统一，后面每个脚本都会重复做映射，最后一定会乱。

### 10.1 节点标准缩写

建议整套归因流程内部统一使用以下缩写：

| 标准缩写 | 原始变量名 | 分组 |
|---|---|---|
| `DK` | `Distance_to_karst` | `Hydrogeology` |
| `DB` | `Depth_to_Bedrock` | `Hydrogeology` |
| `DF` | `Distance_to_Fault_m` | `Hydrogeology` |
| `HDS` | `HDS` | `Hydrogeology` |
| `PR` | `Precip` | `Climate` |
| `TAS` | `Tas` | `Climate` |
| `HUSS` | `Huss` | `Climate` |
| `PT` | `PopTotal` | `Anthropogenic` |
| `UF` | `UrbanFrac` | `Anthropogenic` |
| `IP` | `ImperviousIndex` | `Anthropogenic` |
| `LAI` | `LAI` | `Anthropogenic` |
| `WTD` | `WTD` | `Anthropogenic` |
| `RISK` | `Predicted sinkhole probability` | `Target` |

### 10.2 分组标准字段

所有中间表都建议至少保留这几个分组字段：

- `node`
- `node_full_name`
- `group_lv1`
- `group_lv2`

其中：

- `group_lv1`
  - `Hydrogeology`
  - `Climate`
  - `Anthropogenic`
  - `Target`
- `group_lv2`
  - 可以更细分，例如：
    - `Static geology`
    - `Climate forcing`
    - `Human pressure`
    - `Environmental mediator`
    - `Target`

### 10.3 情景标准字段

未来阶段所有表统一保留：

- `scenario_id`
- `ssp`
- `year`
- `repeat_id`

建议：

- `scenario_id`
  - 例如 `ssp1_2040`
  - 例如 `ssp5_2100`

### 10.4 样本标准字段

所有样本级表统一保留：

- `sample_id`
- `Longitude`
- `Latitude`
- `ADCODE99`
- `NAME_EN_JX`

如果某些阶段没有行政区信息，也建议至少保留：

- `sample_id`

### 10.5 目标变量标准字段

历史阶段与未来阶段不要混名。

建议固定：

- 历史阶段真实标签：
  - `disaster_observed`
- GWR 原始输出：
  - `gwr_raw_score`
- sigmoid 映射后的连续概率：
  - `risk_probability`

LiNGAM 在未来阶段的终点节点统一使用：

- `risk_probability`

---

## 11. 中间表设计

这里把整套流程需要的中间表直接定下来。

建议按“公共表、当前阶段表、未来阶段表、绘图汇总表”四类组织。

### 11.1 公共表 1：变量字典表

文件名建议：

- `variable_dictionary.csv`

作用：

- 全流程唯一变量映射表
- 所有脚本都从这里读取节点名称和分组，而不是各自手写

建议列名：

- `node`
- `node_full_name`
- `source_column`
- `group_lv1`
- `group_lv2`
- `is_target`
- `is_static`
- `is_dynamic`
- `plot_order`
- `node_order`
- `color_hex`

说明：

- `source_column`
  - 对应原始 CSV 中真实列名
- `plot_order`
  - 控制条形图、热图、DAG 图中的显示顺序

---

### 11.2 公共表 2：先验知识边约束表

文件名建议：

- `prior_knowledge_edges.csv`

作用：

- 直接记录每一对节点之间的允许/禁止关系
- 比单纯矩阵更容易人工检查

建议列名：

- `source`
- `target`
- `constraint_type`
- `constraint_value`
- `reason`
- `source_group`
- `target_group`

建议取值：

- `constraint_type`
  - `forbidden`
  - `allowed`
  - `unknown`
- `constraint_value`
  - `-1`
  - `1`
  - `0`

说明：

- 这里建议沿用你在框架文档里已经定下的三态语义：
  - `forbidden = -1`
  - `allowed = 1`
  - `unknown = 0`

---

### 11.3 公共表 3：先验知识矩阵表

文件名建议：

- `prior_knowledge_matrix.csv`

作用：

- 给 `DirectLiNGAM` 直接喂矩阵

建议格式：

- 行列都是节点顺序
- 单元格为 `-1 / 0 / 1`

额外建议生成一个元数据文件：

- `prior_knowledge_matrix_nodes.csv`

列名：

- `node`
- `matrix_order`

原因：

- 后续读取矩阵时必须知道行列顺序

---

### 11.4 当前阶段样本表

文件名建议：

- `current_samples_for_lingam.csv`

作用：

- 组图一的全国 LiNGAM 统一输入表

建议列名：

- `sample_id`
- `ADCODE99`
- `NAME_EN_JX`
- `Longitude`
- `Latitude`
- `DK`
- `DB`
- `DF`
- `HDS`
- `PR`
- `TAS`
- `HUSS`
- `PT`
- `UF`
- `IP`
- `LAI`
- `WTD`
- `disaster_observed`

说明：

- 当前阶段如果是历史观测归因，就保留 `disaster_observed`
- 如果当前阶段也统一走概率口径，则再补一列：
  - `risk_probability`

---

### 11.5 当前阶段 LiNGAM 边结果长表

文件名建议：

- `current_lingam_edges_long.csv`

作用：

- 图 1b、图 1c 的核心来源之一

建议列名：

- `source`
- `target`
- `source_group`
- `target_group`
- `edge_sign`
- `edge_weight`
- `edge_weight_abs`
- `is_direct_to_target`
- `is_mediator_edge`
- `is_selected_for_main_figure`

说明：

- `edge_sign`
  - `positive`
  - `negative`
- `edge_weight`
  - 原始系数
- `edge_weight_abs`
  - 绝对值

---

### 11.6 当前阶段总效应表

文件名建议：

- `current_total_effects_to_target.csv`

作用：

- 图 1b 全国 LiNGAM 总效应排序图的直接输入

建议列名：

- `node`
- `node_full_name`
- `group_lv1`
- `group_lv2`
- `total_effect`
- `total_effect_abs`
- `effect_sign`
- `rank_abs`
- `selected_topk`

说明：

- `effect_sign`
  - `positive`
  - `negative`

---

### 11.7 当前阶段路径汇总表

文件名建议：

- `current_path_summary.csv`

作用：

- 图 1c 的路径分解、桑基图或简化 DAG 图输入

建议列名：

- `path_id`
- `path_str`
- `source_root`
- `mediator`
- `target`
- `path_length`
- `path_sign`
- `path_effect`
- `path_effect_abs`
- `source_group`
- `mediator_group`
- `target_group`
- `is_main_path`

样例：

- `PR->WTD->RISK`
- `UF->LAI->RISK`
- `DB->RISK`

---

### 11.8 未来阶段样本表

文件名建议：

- `future_samples_for_lingam.csv`

建议按情景存放在：

- `outputs/attribution/future/{ssp}/{year}/future_samples_for_lingam.csv`

作用：

- 某一情景下重复 LiNGAM 的统一输入表

建议列名：

- `sample_id`
- `ADCODE99`
- `NAME_EN_JX`
- `Longitude`
- `Latitude`
- `scenario_id`
- `ssp`
- `year`
- `DK`
- `DB`
- `DF`
- `HDS`
- `PR`
- `TAS`
- `HUSS`
- `PT`
- `UF`
- `IP`
- `LAI`
- `WTD`
- `gwr_raw_score`
- `risk_probability`

说明：

- `risk_probability` 必须是用训练期 `transform_metadata` 映射出来的连续概率
- 不要使用分级结果作为 LiNGAM 目标

---

### 11.9 单次重复 LiNGAM 边记录长表

文件名建议：

- `edge_records_long.csv`

作用：

- 未来阶段最核心的原始结果表
- 后面所有频率统计、热图、稳定链提取都从这里来

建议列名：

- `scenario_id`
- `ssp`
- `year`
- `repeat_id`
- `source`
- `target`
- `source_group`
- `target_group`
- `edge_present`
- `edge_sign`
- `edge_weight`
- `edge_weight_abs`
- `target_is_risk`
- `passes_prior_check`

说明：

- `edge_present`
  - `1`
- 这张表只记录“本次出现的边”
- 不出现的边不要硬填 0，后面再补全集统计

---

### 11.10 单次重复总效应表

文件名建议：

- `total_effect_records_long.csv`

作用：

- 用于后续做“路径强度如何随情景变化”

建议列名：

- `scenario_id`
- `ssp`
- `year`
- `repeat_id`
- `node`
- `group_lv1`
- `group_lv2`
- `total_effect`
- `total_effect_abs`
- `effect_sign`

---

### 11.11 边频率汇总表

文件名建议：

- `edge_frequency_summary.csv`

作用：

- 图 2a 的直接输入
- 图 2b 的筛选基础

建议列名：

- `scenario_id`
- `ssp`
- `year`
- `source`
- `target`
- `source_group`
- `target_group`
- `n_repeats`
- `n_present`
- `freq_present`
- `freq_positive`
- `freq_negative`
- `mean_weight`
- `mean_weight_abs`
- `median_weight`
- `sd_weight`
- `passes_freq_threshold`
- `passes_sign_stability`
- `is_stable_edge`

建议规则：

- `passes_freq_threshold`
  - `freq_present >= 0.8`
- `passes_sign_stability`
  - 正负号有一边占绝对多数，例如 `>= 0.8`
- `is_stable_edge`
  - 同时满足前两者

---

### 11.12 稳定边矩阵宽表

文件名建议：

- `edge_frequency_heatmap_matrix.csv`

作用：

- 图 2a 热图直接读取

建议格式：

- 行：`edge_label`
- 列：各 `scenario_id`

建议列名示例：

- `edge_label`
- `ssp1_2040`
- `ssp1_2060`
- `ssp1_2080`
- `ssp1_2100`
- ...
- `ssp5_2100`

`edge_label` 建议统一为：

- `PR -> WTD`
- `WTD -> RISK`
- `UF -> LAI`

---

### 11.13 稳定路径汇总表

文件名建议：

- `stable_path_summary.csv`

作用：

- 图 2b、图 2c 的核心来源

建议列名：

- `scenario_id`
- `ssp`
- `year`
- `path_id`
- `path_str`
- `path_length`
- `source_root`
- `mediator`
- `target`
- `source_group`
- `mediator_group`
- `target_group`
- `freq_path`
- `mean_path_effect`
- `mean_path_effect_abs`
- `path_sign`
- `is_stable_path`
- `is_main_path`

样例：

- `PR->WTD->RISK`
- `UF->LAI->RISK`
- `DB->RISK`

---

### 11.14 三大类贡献汇总表

文件名建议：

- `group_path_contribution_summary.csv`

作用：

- 图 2c 的直接输入

建议列名：

- `scenario_id`
- `ssp`
- `year`
- `path_group`
- `contribution_mean`
- `contribution_mean_abs`
- `contribution_share`
- `n_stable_edges`
- `n_stable_paths`

建议 `path_group` 固定为：

- `Hydrogeology_to_RISK`
- `Climate_to_RISK`
- `Anthropogenic_to_RISK`
- `Climate_mediated_to_RISK`
- `Anthropogenic_mediated_to_RISK`

如果想避免过细，主文图也可以聚合成三类：

- `Hydrogeology`
- `Climate`
- `Anthropogenic`

---

## 12. 图和中间表的一一对应关系

为了避免后面写图脚本时重复摸索，建议直接固定成下面的映射。

### 图 1a：先验知识约束框架图

主要输入：

- `variable_dictionary.csv`
- `prior_knowledge_edges.csv`

### 图 1b：全国尺度 LiNGAM 总效应排序图

主要输入：

- `current_total_effects_to_target.csv`
- `variable_dictionary.csv`

### 图 1c：当前阶段路径分解图

主要输入：

- `current_lingam_edges_long.csv`
- `current_path_summary.csv`
- `variable_dictionary.csv`

### 图 2a：稳定因果边出现频率热图

主要输入：

- `edge_frequency_summary.csv`
- `edge_frequency_heatmap_matrix.csv`

### 图 2b：高频稳定因果链 DAG 图

主要输入：

- `stable_path_summary.csv`
- `edge_frequency_summary.csv`
- `variable_dictionary.csv`

### 图 2c：主导路径强度变化图

主要输入：

- `group_path_contribution_summary.csv`

---

## 13. 脚本拆分建议

这里直接按“最小可维护单元”拆。

### 13.1 配置与公共模块

#### `pre_attribution/attribution_config.py`

职责：

- 统一配置路径
- 定义 `SSP`、年份列表
- 定义重复次数、抽样比例、阈值

建议集中管理的参数：

- `SSPS = ["ssp1", "ssp2", "ssp3", "ssp4", "ssp5"]`
- `YEARS = ["2040", "2060", "2080", "2100"]`
- `N_REPEATS = 100`
- `SAMPLE_RATIO = 0.8`
- `FREQ_THRESHOLD = 0.8`
- `SIGN_THRESHOLD = 0.8`

#### `pre_attribution/variable_schema.py`

职责：

- 生成 `variable_dictionary.csv`
- 提供节点顺序和分组映射函数

#### `pre_attribution/prior_knowledge_builder.py`

职责：

- 根据先验规则生成：
  - `prior_knowledge_edges.csv`
  - `prior_knowledge_matrix.csv`
  - `prior_knowledge_matrix_nodes.csv`

这个脚本应当是最先实现的。

#### `post_attribution/nature_figure_style.py`

职责：

- 统一管理所有绘图风格常量
- 输出 Nature 风格的共享样式接口

建议包括：

- `FONT_FAMILY = "Times New Roman"`
- `BASE_FONT_SIZE = 9`
- `FIGURE_FACE = "#FFFFFF"`
- `GROUP_COLORS = {...}`
- `SIGN_COLORS = {...}`
- `GRID_COLOR = "#E6E2DD"`
- `AXIS_COLOR = "#8A857F"`

这个模块一旦建立，后续所有绘图脚本都应导入它，而不是重复写颜色、字体和导出参数。

---

### 13.2 当前阶段归因脚本

#### `pre_attribution/build_current_samples_for_lingam.py`

职责：

- 读取历史样本
- 只保留 12 个变量和目标列
- 输出 `current_samples_for_lingam.csv`

#### `post_attribution/run_current_lingam.py`

职责：

- 读取：
  - `current_samples_for_lingam.csv`
  - `prior_knowledge_matrix.csv`
- 拟合一次当前阶段 LiNGAM
- 输出：
  - `current_lingam_edges_long.csv`
  - `current_total_effects_to_target.csv`

#### `post_attribution/summarize_current_paths.py`

职责：

- 从当前阶段边结果中提取主要路径
- 输出 `current_path_summary.csv`

---

### 13.3 未来阶段准备脚本

#### `pre_attribution/build_future_samples_for_lingam.py`

职责：

- 读取每个 `SSP × 年份` 的未来驱动因子表
- 调用 GWR 预测
- 使用训练期 metadata 做 sigmoid 映射
- 生成每个情景的：
  - `future_samples_for_lingam.csv`

这个脚本必须显式依赖：

- [Train_National_GWR_classification.ipynb](/path/to/sinkhole-risk-china/code/3_gwr_model_train/national/GWR/Train_National_GWR_classification.ipynb)
- [gwr_sigmoid_utils.py](/path/to/sinkhole-risk-china/code/mgtwr/gwr_sigmoid_utils.py)

---

### 13.4 未来阶段重复 LiNGAM 主脚本

#### `post_attribution/run_future_lingam_repeats.py`

职责：

- 对单个 `scenario_id`：
  - 固定比例无放回抽样
  - 重复运行 LiNGAM
  - 记录边和总效应

输入：

- `future_samples_for_lingam.csv`
- `prior_knowledge_matrix.csv`

输出：

- `edge_records_long.csv`
- `total_effect_records_long.csv`

建议参数：

- `--ssp`
- `--year`
- `--n_repeats`
- `--sample_ratio`
- `--random_state`

---

### 13.5 未来阶段汇总脚本

#### `post_attribution/summarize_edge_frequency.py`

职责：

- 从 `edge_records_long.csv` 计算：
  - `edge_frequency_summary.csv`
  - `edge_frequency_heatmap_matrix.csv`

#### `post_attribution/summarize_stable_paths.py`

职责：

- 从稳定边中提取稳定路径
- 输出：
  - `stable_path_summary.csv`
  - `group_path_contribution_summary.csv`

---

### 13.6 绘图脚本

#### `post_attribution/plot_prior_knowledge_framework.py`

对应：

- 图 1a

#### `post_attribution/plot_current_total_effects.py`

对应：

- 图 1b

#### `post_attribution/plot_current_path_sankey.py`

对应：

- 图 1c

#### `post_attribution/plot_edge_frequency_heatmap.py`

对应：

- 图 2a

#### `post_attribution/plot_stable_future_dag.py`

对应：

- 图 2b

#### `post_attribution/plot_group_path_shift.py`

对应：

- 图 2c

---

## 14. 推荐执行顺序

不要一开始就写所有图脚本。  
正确顺序应该是先把“表”跑通，再做图。

建议顺序：

1. `variable_schema.py`
2. `prior_knowledge_builder.py`
3. `build_current_samples_for_lingam.py`
4. `run_current_lingam.py`
5. `summarize_current_paths.py`
6. `plot_current_total_effects.py`
7. `plot_current_path_sankey.py`
8. `build_future_samples_for_lingam.py`
9. `run_future_lingam_repeats.py`
10. `summarize_edge_frequency.py`
11. `summarize_stable_paths.py`
12. `plot_edge_frequency_heatmap.py`
13. `plot_stable_future_dag.py`
14. `plot_group_path_shift.py`

这样做的原因很简单：

- 当前阶段先跑通，能先产出组图一
- 未来阶段的重复 LiNGAM 最耗时，也最容易出问题
- 等边频率与稳定路径表稳定后，再做组图二，返工最少

---

## 15. 最小可实施版本

如果你准备立刻开工，我建议先只实现下面 `6` 个脚本。

### 第一批必须实现

- `pre_attribution/variable_schema.py`
- `pre_attribution/prior_knowledge_builder.py`
- `post_attribution/run_current_lingam.py`
- `post_attribution/run_future_lingam_repeats.py`
- `post_attribution/summarize_edge_frequency.py`
- `post_attribution/summarize_stable_paths.py`

### 这样做的原因

这 `6` 个脚本一旦完成，你就已经拥有：

- 当前总效应排序图所需数据
- 当前路径图所需数据
- 未来稳定边热图所需数据
- 未来稳定链 DAG 图所需数据

也就是说，两组主图的骨架就已经齐了。

---

## 16. 一句话落地建议

后面真正开发时，不要再以“每张图”为单位思考，而要以：

- **先跑出标准中间表**
- **再让每张图只读取标准表**

为原则。

这样做之后，图和统计口径才会稳定，后续改图样式时也不会反复改分析逻辑。
