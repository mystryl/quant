# Findings & Decisions

<!-- 
  WHAT: Your knowledge base for the task. Stores everything you discover and decide.
  WHY: Context windows are limited. This file is your "external memory" - persistent and unlimited.
  WHEN: Update after ANY discovery, especially after 2 view/browser/search operations (2-Action Rule).
-->

## Requirements
<!-- 用户需求 -->
- 设计多因子量化分析系统
- 给定因子，根据回测数据判断因子的 IC IR 等指标
- 告诉用户因子是否可靠
- 考虑因子的周期对齐（是否需要提前周期）
- 判断因子可靠性用什么指标（未来收益率？）
- 编写因子和回测时避免未来函数，仅在判断因子性能时引入未来收益率
- 是否需要考虑对数计算
- 评判因子指标要考虑看涨策略、看跌策略、波动率升高策略
- 是否需要补充的

## Research Findings
### Qlib 框架研究发现

#### 1. 未来收益计算
- **Label 表达式**: `Ref($close, -2) / Ref($close, -1) - 1`
- **关键设计理念**:
  - T+1 到 T+2：计算从第 2 个交易日到第 3 个交易日的收益率
  - 避免未来数据泄露：文档明确说明原因："when getting T day close price of a china stock, stock can be bought on T+1 day and sold on T+2 day"
  - 这个设计考虑了实际交易的可执行性，避免使用 T 日的收盘价（因为 T 日收盘时已无法买入）

#### 2. IC（信息系数）计算
- **位置**: `/Users/mystryl/Documents/Quant/frameworks/qlib/qlib/contrib/eva/alpha.py`
- **函数**: `calc_ic(pred, label, date_col="datetime", dropna=False)`
- **计算逻辑**:
  - 普通 IC：使用 Pearson 相关系数
    ```python
    ic = df.groupby(date_col).apply(lambda df: df["pred"].corr(df["label"]))
    ```
  - Rank IC：使用 Spearman 秩相关系数
    ```python
    ric = df.groupby(date_col).apply(lambda df: df["pred"].corr(df["label"], method="spearman"))
    ```
- **ICIR（IC Information Ratio）**:
  - IC均值 / IC标准差
  - 衡量 IC 的稳定性

#### 3. IR（信息比率）计算
- **函数**: `calc_long_short_return(pred, label, date_col="datetime", quantile=0.2, dropna=False)`
- **计算方式**:
  - 分子：平均超额收益（策略收益 vs 基准）
  - 分母：超额收益的标准差
  - 公式：IR = 平均超额收益 / 超额收益标准差
- **多空收益计算**:
  ```python
  r_long = group.apply(lambda x: x.nlargest(N(x), columns="pred").label.mean())
  r_short = group.apply(lambda x: x.nsmallest(N(x), columns="pred").label.mean())
  r_avg = group.label.mean()
  return (r_long - r_short) / 2, r_avg
  ```

#### 4. Qlib 现有项目
- **Alpha158**: 使用 close 计算的 Alpha 因子数据集
- **Alpha360**: 使用 close 计算的 Alpha 因子数据集
- **Alpha158vwap**: 使用 vwap 计算的 Alpha 因子数据集
- **位置**: `/Users/mystryl/Documents/Quant/frameworks/qlib/qlib/contrib/data/handler.py`

### jqfactor_analyzer 研究发现
- **来源**: 聚宽开源的单因子分析工具
- **主要功能**:
  - 配合 jqdatasdk 进行归因分析
  - 因子数据缓存
  - 单因子分析
- **未来收益计算**:
  - forward_return = (第二天的收盘价 - 今天的收盘价) / 今天的收盘价
  - 调仓周期为 1 天时使用此公式
  - 与 Qlib 不同，假设 T 日可买入
- **归因分析**:
  - 基于多因子风险模型
  - 分解收益来源：国家因子、风格因子、行业因子、特异收益
  - 支持风格归因和行业归因

### Panda_factor 研究发现
- **来源**: PandaAI 推出的高性能量化因子库
- **主要功能**:
  - 提供量化算子用于金融数据分析
  - 技术指标计算和因子构建
  - 可视化图表
- **编写方式**:
  - Python 方式：继承 Factor 类，实现 calculate 方法
  - 公式方式：类似 Excel 公式的语法
- **因子计算示例**:
  ```python
  returns = (close / DELAY(close, 20)) - 1
  volatility = STDDEV((close / DELAY(close, 1)) - 1, 20)
  momentum = RANK(returns)
  ```
- **特点**: 专注于因子计算和构建，不提供 IC/IR 分析功能

### Qlib vs Panda_factor 主要区别
| 对比项      | Qlib                                                  | Panda_factor (jqfactor_analyzer)                       |
| -------- | ----------------------------------------------------- | ------------------------------------------------------ |
| Label 计算 | Ref($close, -2) / Ref($close, -1) - 1 <br>（T+1 到 T+2） | forward_return = price_t+N / price_t - 1 <br>（T 到 T+N） |
| 设计原因     | 避免 T 日买入无法实现（中国市场）                                    | 假设 T 日可买入                                              |
| IC 计算    | 按 groupby(date) 计算每日 IC                               | 计算整段时间的 IC 序列                                          |
| IR 含义    | 实际回测的超额收益/风险                                          | 基于策略组合的信息比率                                          |

### 现有项目分析
- **akquant**: 量化交易框架
- **rdagent**: LLM 驱动的量化研究代理
- **qlib_backtest**: Qlib 回测相关项目
- **Multi-Frame_Entry**: 多时间框架入场策略

### 未来收益计算方式分析
#### 两种主流方式对比

**方式 1: Qlib 方式（推荐）**
- **公式**: `Ref($close, -2) / Ref($close, -1) - 1`
- **含义**: 计算 T+1 到 T+2 的收益率
- **优点**:
  - 严格避免未来数据泄露
  - 符合中国 T+1 交易规则
  - T 日收盘后计算因子，T+1 日买入，T+2 日卖出
- **适用场景**: 中国 A 股市场

**方式 2: jqfactor_analyzer 方式**
- **公式**: `(第二天的收盘价 - 今天的收盘价) / 今天的收盘价`
- **含义**: 计算 T 到 T+1 的收益率
- **优点**:
  - 计算简单直观
  - 假设 T 日可买入
- **缺点**:
  - 可能在实际交易中无法执行
- **适用场景**: 假设可 T 日买入的市场或模拟环境

#### 对数收益率 vs 简单收益率

**简单收益率** (Simple Return):
- 公式: `R = (P_t - P_0) / P_0 = P_t / P_0 - 1`
- 优点: 直观易懂，便于解释
- 缺点: 多期收益计算需要使用复利公式 `(1+R1)*(1+R2)-1`

**对数收益率** (Log Return):
- 公式: `r = ln(P_t / P_0) = ln(P_t) - ln(P_0)`
- 优点:
  - 多期收益率可相加: `r_total = r1 + r2 + ... + rn`
  - 更接近正态分布，适合统计建模
  - 对称性好（涨跌幅幅度一致）
- 缺点: 解释性不如简单收益率直观

**选择建议**:
- 使用简单收益率：因子分析、回测（更符合实际交易）
- 使用对数收益率：统计建模、机器学习（更符合假设）

### 周期对齐问题分析

#### 为什么需要周期对齐？

不同类型的因子具有不同的周期特性：
- **高频因子**（如分钟级动量）：周期短，更新快
- **中频因子**（如日线动量）：周期中等，按日更新
- **低频因子**（如基本面因子）：周期长，按季/年更新

#### 周期对齐策略

1. **因子周期与持仓周期匹配**
   - 如果因子是 20 日均线，持仓周期建议 ≥ 20 日
   - 避免高频因子配低频持仓（换手率过高）
   - 避免低频因子配高频持仓（信号滞后）

2. **未来收益率计算周期**
   - 因子预测周期应与未来收益计算周期匹配
   - 例如：20 日动量因子 → 计算 20 日未来收益率
   - Qlib 的 T+1 到 T+2 是固定 1 天，适用于日频因子

3. **周期自动检测**
   - 分析因子的自相关性确定有效周期
   - 通过因子衰减曲线确定最佳持仓周期
   - 考虑市场摩擦成本（手续费、滑点）

## Technical Decisions
| Decision | Rationale |
|----------|-----------|
| 采用 Qlib 的 Label 设计 | 符合中国 T+1 交易规则，避免未来数据泄露 |
| 混合使用 IC/IR 和多空收益 | IC/IR 衡量预测能力，多空收益衡量实际交易效果 |
| 支持多种周期对齐方式 | 不同因子可能有不同的周期特性 |
| 引入策略场景分析 | 同一因子在不同市场环境下表现不同 |

## Issues Encountered
| Issue | Resolution |
|-------|------------|
| 未找到本地 Panda_factor 库 | 需要通过网络搜索或查看文档了解其实现 |

## Resources
- Qlib 源码: `/Users/mystryl/Documents/Quant/frameworks/qlib/`
- Qlib IC/IR 计算: `/Users/mystryl/Documents/Quant/frameworks/qlib/qlib/contrib/eva/alpha.py`
- Qlib Label 配置: `/Users/mystryl/Documents/Quant/frameworks/qlib/qlib/contrib/data/handler.py`

## Visual/Browser Findings
<!-- 目前没有查看浏览器或图像 -->

---
*Update this file after every 2 view/browser/search operations*
*This prevents visual information from being lost*
