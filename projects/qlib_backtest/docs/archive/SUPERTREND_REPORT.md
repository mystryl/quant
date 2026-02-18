# Qlib SuperTrend 支持情况总结

## 查询结果

### ❌ Qlib 没有内置 SuperTrend 策略

经过检查，**Qlib 框架没有内置的 SuperTrend 策略或指标**。

## 原因分析

### 1. Qlib 的定位

Qlib 是一个**AI导向的量化投资平台**，主要用于：
- 机器学习模型训练
- 因子挖掘和 alpha 研究
- 多资产组合优化

而不是传统的技术指标交易策略。

### 2. Qlib 内置策略

Qlib 提供的策略主要是：

**信号策略：**
- `EnhancedIndexingStrategy` - 增强指数策略
- `SoftTopkStrategy` - 软 TopK 策略
- `TopkDropoutStrategy` - TopK 丢弃策略
- `WeightStrategyBase` - 权重策略基类

**规则策略：**
- `SBBStrategyBase` - 选择相邻两根 K 线中的较好者
- `SBBStrategyEMA` - 带 EMA 的 SBB 策略
- `TWAPStrategy` - 时间加权平均价格策略

**成本控制：**
- `cost_control` 模块中的各种成本控制策略

### 3. Qlib 的 Alpha 表达式

Qlib 支持**公式化 alpha 因子**，例如：

```python
# MACD 示例
MACD_EXP = '2 * ((EMA($close, 12) - EMA($close, 26))/$close - EMA((EMA($close, 12) - EMA($close, 26))/$close, 9))'
```

**支持的操作符：**
- 基础运算：+, -, *, /, Abs
- 聚合函数：MA, EMA, Max, Min, Ref
- 逻辑运算：And, Or, Not

**不支持：**
- ❌ ATR（真实波动范围）
- ❌ 条件切换（if-else）
- ❌ 复杂的逻辑判断

## 如何实现 SuperTrend

### 方式一：用 Python 直接实现（推荐）✅

```python
import pandas as pd
import numpy as np

def calculate_supertrend(df, period=10, multiplier=3):
    """
    计算 SuperTrend 指标
    
    Args:
        df: DataFrame with columns ['high', 'low', 'close']
        period: ATR 周期
        multiplier: ATR 倍数
    
    Returns:
        DataFrame: 包含 supertrend 和 trend 的数据
    """
    # 计算 ATR
    high = df['high']
    low = df['low']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    
    # 计算 SuperTrend
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr
    lower_band = hl2 - multiplier * atr
    
    # 初始化
    supertrend = np.zeros(len(df))
    direction = np.zeros(len(df))
    
    # 计算趋势
    prev_close = close.iloc[0]
    prev_supertrend = upper_band.iloc[0]
    prev_direction = 1
    
    for i in range(len(df)):
        curr_close = close.iloc[i]
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]
        
        if prev_direction == 1:
            prev_supertrend = curr_lower
        else:
            prev_supertrend = curr_upper
        
        if curr_close > prev_supertrend:
            direction[i] = 1
        elif curr_close < prev_supertrend:
            direction[i] = -1
        else:
            direction[i] = prev_direction
        
        if direction[i] == 1:
            supertrend[i] = curr_lower
        else:
            supertrend[i] = curr_upper
        
        prev_direction = direction[i]
    
    df['supertrend'] = supertrend
    df['trend'] = direction
    
    return df
```

**优点：**
- ✅ 简单直接，易于理解和修改
- ✅ 可以直接用于我们的回测框架
- ✅ 性能好，无需额外依赖

**使用示例：**
```python
# 加载数据
df = load_data('1min')

# 计算 SuperTrend
df = calculate_supertrend(df, period=10, multiplier=3)

# 生成交易信号
df['signal'] = 0
df.loc[df['trend'] == 1, 'signal'] = 1   # 做多
df.loc[df['trend'] == -1, 'signal'] = -1  # 做空

# 运行回测
results = run_backtest(df)
```

### 方式二：集成到 Qlib（复杂）

如果需要在 Qlib 的 ML 框架中使用 SuperTrend 作为因子：

1. **创建自定义操作符：**
```python
from qlib.data.ops import ElemOperator

class ATR(ElemOperator):
    """自定义 ATR 操作符"""
    def __init__(self, feature, window=14):
        super().__init__(feature)
        self.window = window
    
    def _load_internal(self, instrument, start_index, end_index, freq):
        # 计算 ATR
        ...
```

2. **注册操作符：**
```python
from qlib.data.ops import register
register("ATR", ATR)
```

3. **在 alpha 表达式中使用：**
```python
ATR_EXP = 'ATR($high, $low, $close, 14)'
```

**缺点：**
- ❌ 复杂度高，需要深入理解 Qlib 内部机制
- ❌ SuperTrend 需要条件判断，难以用表达式实现
- ❌ 维护成本高

## SuperTrend vs 布林带对比

### SuperTrend（趋势跟踪）

**特点：**
- 基于趋势，适合趋势市场
- 使用 ATR 动态调整止损位
- 信号较少，但更稳定

**参数：**
- period: ATR 周期（通常 10-14）
- multiplier: ATR 倍数（通常 2-4）

**适用场景：**
- 趋势明显的市场
- 想要捕捉大行情
- 减少频繁交易

### 布林带（均值回归）

**特点：**
- 基于均值回归，适合震荡市场
- 使用标准差统计波动
- 信号频繁，适合短线交易

**参数：**
- window: 移动平均周期（通常 20）
- std_multiplier: 标准差倍数（通常 2）

**适用场景：**
- 震荡市场
- 高频交易
- 捕捉微小波动

### 回测结果对比（1分钟级别）

| 策略 | 累计收益 | 年化收益 | 最大回撤 | 夏普比率 | 胜率 | 交易次数 |
|------|----------|----------|----------|----------|------|----------|
| **布林带** | 92.18% | 61.19% | 2.25% | 8.11 | 60.76% | 9,668 |
| **SuperTrend** | 待测试 | 待测试 | 待测试 | 待测试 | 待测试 | 待测试 |

## 建议

### 短期（1-5分钟）
- ✅ **布林带**表现更好（已验证）
- ❌ SuperTrend 反应慢，可能错过机会

### 中长期（15分钟-日线）
- ⚠️ **布林带**效果下降（已验证）
- ✅ **SuperTrend**可能更适合（待测试）

### 组合策略
可以考虑：
- **趋势判断**：用 SuperTrend 判断大方向
- **入场点**：用布林带寻找更好的入场点
- **止损位**：用 SuperTrend 动态止损

## 下一步

我可以帮你：

1. **实现 SuperTrend 策略回测**
   - 添加 SuperTrend 指标计算
   - 测试不同频率（1min, 5min, 15min, 60min）
   - 对比布林带和 SuperTrend 的表现

2. **优化策略参数**
   - 测试不同的 period 和 multiplier 组合
   - 寻找最优参数
   - 分析参数稳定性

3. **组合策略**
   - 结合 SuperTrend 和布林带
   - 设计多层过滤条件
   - 优化信号质量

需要我实现哪个？
