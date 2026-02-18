# SuperTrend 指标在 Qlib 中的使用指南

## 概述

已成功将 ssquant 中的 SuperTrend 指标复刻到 qlib 回测环境中。

## 核心实现

### 1. 指标计算

```python
from qlib_supertrend import supertrend, supertrend_signals

# 计算 SuperTrend
st_line, trend = supertrend(
    high=df['high'],
    low=df['low'],
    close=df['close'],
    period=10,        # ATR 周期
    multiplier=3.0    # ATR 倍数
)

# 生成买卖信号
buy_signals, sell_signals = supertrend_signals(st_line, trend)
```

### 2. 策略使用

```python
from qlib_supertrend import SupertrendStrategy

# 创建策略
strategy = SupertrendStrategy(period=10, multiplier=3.0)

# 生成交易信号
df_with_signals = strategy.generate_signal(df)

# df_with_signals 包含以下列:
# - supertrend: SuperTrend 线
# - trend: 趋势方向 (1=上涨, -1=下跌)
# - buy_signal: 买入信号 (True/False)
# - sell_signal: 卖出信号 (True/False)
# - position: 持仓方向 (1=做多, -1=做空)
```

## 回测结果示例

基于 RB9999.XSGE (螺纹钢连续合约) 2023 年 1 分钟数据：

| 参数组合 | 交易次数 | 累计收益 | 年化收益 | 最大回撤 | 夏普比率 | 胜率 |
|---------|---------|---------|---------|---------|---------|------|
| (10, 3.0) | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| (10, 2.0) | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| (7, 3.0) | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |
| (14, 2.5) | 待测 | 待测 | 待测 | 待测 | 待测 | 待测 |

## 测试验证

已成功测试 1000 行 1 分钟数据，结果如下：

```
时间范围: 2023-01-03 09:01:00 ~ 2023-01-05 22:25:00

趋势统计:
  上涨趋势(1): 482 次
  下跌趋势(-1): 518 次

信号统计:
  买入信号: 11 次
  卖出信号: 12 次
```

## 参数说明

### SuperTrend 指标参数

- **period**: ATR 计算周期，默认 10
  - 较小的值：更敏感，信号更多
  - 较大的值：更平滑，信号更少

- **multiplier**: ATR 倍数，默认 3.0
  - 较小的值：更窄的带宽，趋势切换更频繁
  - 较大的值：更宽的带宽，趋势更稳定

### 常用参数组合

1. **保守型** (period=10, multiplier=3.0)
   - 趋势稳定，适合大趋势
   - 信号较少，减少假信号

2. **平衡型** (period=10, multiplier=2.0)
   - 适中敏感度
   - 平衡信号数量和准确性

3. **激进型** (period=7, multiplier=3.0)
   - 更快捕捉趋势变化
   - 信号较多，可能增加交易成本

4. **长期型** (period=14, multiplier=2.5)
   - 过滤短期波动
   - 适合长期趋势跟踪

## 数据要求

需要的 OHLCV 数据：
- **high**: 最高价
- **low**: 最低价
- **close**: 收盘价
- (可选) **volume**: 成交量
- (可选) **amount**: 成交额

## 文件结构

```
/mnt/d/quant/qlib_backtest/
├── qlib_supertrend.py          # SuperTrend 指标实现和策略类
├── test_supertrend_simple.py   # 简化测试脚本
└── SUPERTREND_USAGE.md         # 使用文档 (本文件)
```

## 与 ssquant 的差异

### 相同点

1. 完全复刻了 ssquant 中的 SuperTrend 算法
2. 使用相同的 ATR 计算方法
3. 使用相同的基础带计算逻辑
4. 趋势切换逻辑完全一致

### 不同点

1. **数据格式**: 适配 qlib 数据格式（CSV 文件存储）
2. **返回值**: 返回 pandas Series，便于后续处理
3. **策略接口**: 提供统一的策略接口，方便集成到回测框架

## 下一步

1. 运行完整回测，获取各参数组合的详细结果
2. 可视化 SuperTrend 线和买卖信号
3. 优化参数选择
4. 与其他策略进行对比

## 运行方式

### 快速测试

```bash
cd /mnt/d/quant/qlib_backtest
python3 test_supertrend_simple.py
```

### 完整回测

```bash
cd /mnt/d/quant/qlib_backtest
python3 qlib_supertrend.py
```

### 导入使用

```python
from qlib_supertrend import SupertrendStrategy
import pandas as pd

# 加载数据
df = pd.read_csv('your_data.csv')

# 创建策略
strategy = SupertrendStrategy(period=10, multiplier=3.0)

# 生成信号
df_with_signals = strategy.generate_signal(df)

# 查看信号
print(df_with_signals[['close', 'supertrend', 'trend', 'buy_signal', 'sell_signal']])
```
