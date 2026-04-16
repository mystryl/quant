# 周期对齐与策略分析模块使用指南

本文档详细介绍如何使用周期对齐模块（CycleAligner）和策略场景分析器（StrategyAnalyzer）。

## 目录

1. [周期对齐模块 (CycleAligner)](#周期对齐模块-cyclealigner)
2. [策略场景分析器 (StrategyAnalyzer)](#策略场景分析器-strategyanalyzer)
3. [完整示例](#完整示例)
4. [最佳实践](#最佳实践)

---

## 周期对齐模块 (CycleAligner)

### 概述

周期对齐模块用于对齐因子数据和未来收益率，支持多种对齐策略：

1. **默认对齐（Qlib 方式）**: T+1 到 T+2
2. **灵活对齐**: 自定义偏移量
3. **自动检测**: 基于 IC 最优化

### 基本使用

#### 1. 默认对齐方式

```python
from src.core.cycle_aligner import CycleAligner

# 创建对齐器
aligner = CycleAligner()

# 使用默认对齐（T+1 to T+2）
factor_aligned, returns_aligned = aligner.align(
    factor_df,      # 因子数据 (MultiIndex: datetime, instrument)
    price_df,       # 价格数据 (MultiIndex: datetime, instrument)
    method='default'
)
```

**说明**：
- T 日因子 -> 预测 T+2 的收益率
- 符合中国 T+1 交易规则
- T 日收盘时无法买入，只能 T+1 买入，T+2 卖出

#### 2. 灵活对齐方式

```python
# 自定义偏移量
factor_aligned, returns_aligned = aligner.align(
    factor_df,
    price_df,
    method='flexible',
    shift=1  # T to T+1 (jqfactor_analyzer 方式)
)

# shift=2: T+1 to T+2 (Qlib 默认)
# shift=5: T+4 to T+5 (5 日持有期)
# shift=N: T+(N-1) to T+N
```

#### 3. 自动检测最优对齐

```python
# 自动检测最优偏移量
factor_aligned, returns_aligned, best_shift = aligner.align(
    factor_df,
    price_df,
    method='auto',
    auto_search_range=(1, 10),  # 搜索范围
    ic_calc_method='pearson'    # IC 计算方法
)

print(f"最优偏移量: {best_shift}")
```

### 高级功能

#### 对齐验证

```python
# 验证对齐结果质量
validation = aligner.validate_alignment(
    factor_aligned,
    returns_aligned,
    max_nan_ratio=0.1  # 最大允许 NaN 比例
)

# 检查结果
if validation['is_valid']:
    print("对齐验证通过")
else:
    print("对齐验证失败")
    print(f"错误: {validation['errors']}")
    print(f"警告: {validation['warnings']}")
```

#### 计算未来收益率

```python
# 直接计算未来收益率
returns = aligner.calculate_forward_returns(
    price_df,
    shift=2,
    method='simple'  # 或 'log' 使用对数收益率
)
```

#### 获取对齐摘要

```python
# 获取对齐方案摘要
summary = aligner.get_alignment_summary(
    factor_df,
    price_df,
    shift=2
)

print(f"偏移量: {summary['shift']}")
print(f"描述: {summary['description']}")
print(f"原始日期数: {summary['original_dates']}")
print(f"对齐后日期数: {summary['aligned_dates']}")
print(f"数据损失: {summary['data_loss']:.2%}")
```

### 使用对数收益率

```python
# 创建支持对数收益率的对齐器
aligner = CycleAligner(use_log_return=True)

factor_aligned, returns_aligned = aligner.align(
    factor_df,
    price_df,
    method='default'
)
```

**对数收益率使用场景**：
- 长期投资策略（多日/多周持有）
- 高波动性资产（如加密货币、期货）
- 需要计算复合收益率时

### 便捷函数

```python
from src.core.cycle_aligner import align_factor_returns

# 快速对齐
factor_aligned, returns_aligned = align_factor_returns(
    factor_df,
    price_df,
    method='default'
)
```

---

## 策略场景分析器 (StrategyAnalyzer)

### 概述

策略场景分析器用于分析因子在不同策略场景下的表现：

1. **看涨策略**: 做多因子值高的股票
2. **看跌策略**: 做多因子值低的股票（反向策略）
3. **多空策略**: 多头对冲空头
4. **波动率策略**: 根据市场波动率调整仓位
5. **牛熊市分析**: 不同市场环境下的表现
6. **行业轮动**: 按行业分组分析
7. **市值分组**: 大中小盘分组分析

### 基本使用

#### 1. 看涨策略

```python
from src.core.strategy_analyzer import StrategyAnalyzer

# 创建分析器
analyzer = StrategyAnalyzer(annualization_factor=252)  # 年化因子

# 分析看涨策略
bull_result = analyzer.analyze_bull_strategy(
    factor_aligned,    # 对齐后的因子数据
    returns_aligned,   # 对齐后的收益率数据
    top_pct=0.2        # 选取前 20% 的股票
)

# 查看结果
print(f"总收益率: {bull_result['total_return']:.2%}")
print(f"年化收益率: {bull_result['annual_return']:.2%}")
print(f"夏普比率: {bull_result['sharpe_ratio']:.2f}")
print(f"最大回撤: {bull_result['max_drawdown']:.2%}")
print(f"胜率: {bull_result['win_rate']:.2%}")
print(f"卡玛比率: {bull_result['calmar_ratio']:.2f}")

# 访问每日收益序列
daily_returns = bull_result['strategy_returns']
```

#### 2. 看跌策略

```python
# 分析看跌策略（反向策略）
bear_result = analyzer.analyze_bear_strategy(
    factor_aligned,
    returns_aligned,
    top_pct=0.2  # 选取后 20% 的股票
)

print(f"年化收益率: {bear_result['annual_return']:.2%}")
print(f"夏普比率: {bear_result['sharpe_ratio']:.2f}")
```

#### 3. 多空策略

```python
# 分析多空策略
ls_result = analyzer.analyze_long_short_strategy(
    factor_aligned,
    returns_aligned,
    top_pct=0.2,      # 多头比例
    bottom_pct=0.2    # 空头比例
)

print(f"多空年化收益: {ls_result['annual_return']:.2%}")
print(f"多头年化收益: {ls_result['long_returns'].mean() * 252:.2%}")
print(f"空头年化收益: {ls_result['short_returns'].mean() * 252:.2%}")
```

### 高级策略

#### 4. 波动率策略

```python
# 根据市场波动率调整仓位
vol_result = analyzer.analyze_volatility_strategy(
    factor_aligned,
    returns_aligned,
    price_df,         # 价格数据（用于计算波动率）
    top_pct=0.2,
    vol_window=20,    # 波动率计算窗口
    position_method='inverse'  # 'inverse' 或 'threshold'
)

print(f"波动率调整后年化收益: {vol_result['annual_return']:.2%}")
print(f"基础策略年化收益: {vol_result['base_returns'].mean() * 252:.2%}")
```

**仓位调整方法**：
- `inverse`: 波动率越高，仓位越低（反向关系）
- `threshold`: 高于阈值降低仓位（阈值方法）

#### 5. 牛熊市场景分析

```python
# 分析牛熊市表现
regime_result = analyzer.analyze_market_regime(
    factor_aligned,
    returns_aligned,
    market_index,     # 市场指数序列
    bull_threshold=0.0,  # 牛市阈值
    window=20         # 市场趋势计算窗口
)

# 查看结果
for regime_name, metrics in regime_result.items():
    print(f"\n{regime_name}:")
    print(f"  交易日数: {metrics['days']}")
    print(f"  总收益率: {metrics['total_return']:.2%}")
    print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"  胜率: {metrics['win_rate']:.2%}")
```

#### 6. 行业轮动分析

```python
# 按行业分组分析
industry_result = analyzer.analyze_industry_rotation(
    factor_aligned,
    returns_aligned,
    industry_df,      # 行业分类数据
    top_pct=0.2
)

# 查看各行业表现
for industry, metrics in industry_result.items():
    if 'error' not in metrics:
        print(f"\n{industry}:")
        print(f"  年化收益: {metrics['total_return']:.2%}")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
```

#### 7. 市值分组分析

```python
# 按市值分组分析
cap_result = analyzer.analyze_market_cap_groups(
    factor_aligned,
    returns_aligned,
    market_cap_df,    # 市值数据
    top_pct=0.2,
    n_groups=3        # 分组数量（大中小盘）
)

# 查看各组表现
for group_name, metrics in cap_result.items():
    if 'error' not in metrics:
        print(f"\n{group_name}:")
        print(f"  年化收益: {metrics['total_return']:.2%}")
        print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
```

### 分析所有场景

```python
# 一键分析所有场景
all_results = analyzer.analyze_all_scenarios(
    factor_aligned,
    returns_aligned,
    price_df,              # 可选：用于波动率策略
    market_index,          # 可选：用于牛熊市分析
    industry_df,           # 可选：用于行业轮动
    market_cap_df,         # 可选：用于市值分组
    top_pct=0.2
)

# 访问各种策略结果
print("看涨策略:", all_results['bull']['annual_return'])
print("多空策略:", all_results['long_short']['annual_return'])
print("波动率策略:", all_results['volatility']['annual_return'])
```

### 便捷函数

```python
from src.core.strategy_analyzer import analyze_factor_strategies

# 快速分析所有场景
results = analyze_factor_strategies(
    factor_aligned,
    returns_aligned,
    price_df,
    top_pct=0.2
)
```

---

## 完整示例

### 示例 1: 基本因子分析流程

```python
import pandas as pd
import numpy as np
from src.core.cycle_aligner import CycleAligner
from src.core.strategy_analyzer import StrategyAnalyzer

# 1. 准备数据（假设已有 factor_df 和 price_df）
# factor_df: MultiIndex (datetime, instrument)
# price_df: MultiIndex (datetime, instrument)

# 2. 周期对齐
aligner = CycleAligner()
factor_aligned, returns_aligned = aligner.align(
    factor_df,
    price_df,
    method='default'
)

# 3. 验证对齐结果
validation = aligner.validate_alignment(factor_aligned, returns_aligned)
if not validation['is_valid']:
    print("警告: 对齐验证未通过")
    print(validation['warnings'])

# 4. 策略分析
analyzer = StrategyAnalyzer()

# 看涨策略
bull_result = analyzer.analyze_bull_strategy(
    factor_aligned,
    returns_aligned,
    top_pct=0.2
)

print(f"看涨策略年化收益: {bull_result['annual_return']:.2%}")
print(f"夏普比率: {bull_result['sharpe_ratio']:.2f}")

# 多空策略
ls_result = analyzer.analyze_long_short_strategy(
    factor_aligned,
    returns_aligned,
    top_pct=0.2
)

print(f"多空策略年化收益: {ls_result['annual_return']:.2%}")
print(f"夏普比率: {ls_result['sharpe_ratio']:.2f}")
```

### 示例 2: 自动检测最优周期

```python
# 自动检测最优对齐方式
aligner = CycleAligner()

factor_aligned, returns_aligned, best_shift = aligner.align(
    factor_df,
    price_df,
    method='auto',
    auto_search_range=(1, 10)
)

print(f"检测到最优偏移量: {best_shift}")

# 使用最优对齐进行策略分析
analyzer = StrategyAnalyzer()
result = analyzer.analyze_bull_strategy(
    factor_aligned,
    returns_aligned,
    top_pct=0.2
)

print(f"使用最优对齐的年化收益: {result['annual_return']:.2%}")
```

### 示例 3: 对比不同市场环境

```python
# 准备市场指数
market_index = price_df['close'].groupby(level='datetime').mean()

# 分析牛熊市表现
analyzer = StrategyAnalyzer()

regime_result = analyzer.analyze_market_regime(
    factor_aligned,
    returns_aligned,
    market_index,
    bull_threshold=0.0,
    window=20
)

# 对比表现
print("\n牛市表现:")
print(f"  年化收益: {regime_result['牛市']['total_return']:.2%}")
print(f"  夏普比率: {regime_result['牛市']['sharpe_ratio']:.2f}")

print("\n熊市表现:")
print(f"  年化收益: {regime_result['熊市']['total_return']:.2%}")
print(f"  夏普比率: {regime_result['熊市']['sharpe_ratio']:.2f}")
```

---

## 最佳实践

### 1. 数据准备

确保输入数据格式正确：

```python
# 必须是 MultiIndex 格式
index = pd.MultiIndex.from_product(
    [dates, stocks],
    names=['datetime', 'instrument']
)

# 因子数据
factor_df = pd.DataFrame(factor_values, index=index, columns=['factor_name'])

# 价格数据
price_df = pd.DataFrame(price_values, index=index, columns=['close'])
```

### 2. 选择合适的对齐方式

- **默认对齐（T+1 to T+2）**: 符合中国交易规则，推荐使用
- **灵活对齐（T to T+1）**: 适合短期策略
- **自动检测**: 探索因子周期特性

### 3. 策略分析建议

- 从简单的看涨策略开始
- 对比看跌策略，检查因子方向性
- 使用多空策略降低市场风险
- 在不同市场环境下验证因子稳定性

### 4. 结果解读

- **IC > 0.03**: 因子有较好的预测能力
- **ICIR > 0.5**: IC 稳定性较好
- **夏普比率 > 1**: 风险调整后收益良好
- **最大回撤 < 20%**: 风险可控
- **胜率 > 55%**: 交易成功率较高

### 5. 性能优化

- 使用向量化操作提高计算效率
- 对于大规模数据，考虑分批处理
- 缓存中间结果避免重复计算

---

## 常见问题

### Q1: 如何选择 top_pct？

**A**: 根据策略需求选择：
- 保守策略：top_pct=0.1（选前 10%）
- 平衡策略：top_pct=0.2（选前 20%）
- 激进策略：top_pct=0.3（选前 30%）

### Q2: 对数收益率和简单收益率有什么区别？

**A**:
- **简单收益率**: (P_t - P_0) / P_0，直观易懂
- **对数收益率**: ln(P_t / P_0)，可加性好，适合长期持有

### Q3: 如何处理 NaN 值？

**A**: 模块会自动处理：
- 对齐时会自动去除尾部 NaN
- 计算策略收益时会跳过 NaN
- 可通过 `validate_alignment` 检查 NaN 比例

### Q4: 波动率策略适合什么场景？

**A**:
- 市场波动较大的时期
- 需要控制风险的策略
- 长期持有的投资组合

### Q5: 如何解读牛熊市分析结果？

**A**:
- **牛市表现好**: 因子适合趋势市场
- **熊市表现好**: 因子有防御属性
- **两者都好**: 因子稳定性强，推荐使用
- **两者都差**: 因子不可靠，需要优化

---

## 参考文档

- [系统设计文档](../SYSTEM_DESIGN.md)
- [API 文档](../docs/API.md)
- [用户指南](../docs/USER_GUIDE.md)

---

## 更新日志

- **2024-03-19**: 初始版本，支持周期对齐和策略场景分析
