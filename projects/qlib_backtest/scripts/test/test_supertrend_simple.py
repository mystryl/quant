#!/usr/bin/env python3
"""
简化版 SuperTrend 指标测试
"""
import pandas as pd
import numpy as np

# 只读取部分数据进行快速测试
data_file = "/mnt/d/quant/qlib_data/instruments/RB9999.XSGE/close.csv"
df = pd.read_csv(data_file, index_col=0, parse_dates=True, nrows=1000)

print("数据加载:")
print(f"  行数: {len(df)}")
print(f"  时间范围: {df.index.min()} ~ {df.index.max()}")
print(f"  前5行:\n{df.head()}")
print(f"  后5行:\n{df.tail()}")

# 读取其他数据
high = pd.read_csv("/mnt/d/quant/qlib_data/instruments/RB9999.XSGE/high.csv", index_col=0, parse_dates=True, nrows=1000).iloc[:, 0]
low = pd.read_csv("/mnt/d/quant/qlib_data/instruments/RB9999.XSGE/low.csv", index_col=0, parse_dates=True, nrows=1000).iloc[:, 0]
close = df.iloc[:, 0]

print(f"\nHigh: {high.shape}")
print(f"Low: {low.shape}")
print(f"Close: {close.shape}")

# 计算 ATR
def atr(high, low, close, period=10):
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_value = tr.rolling(period).mean()
    return atr_value

# 计算 SuperTrend
def supertrend(high, low, close, period=10, multiplier=3.0):
    atr_value = atr(high, low, close, period)
    hl2 = (high + low) / 2
    upper_band = hl2 + multiplier * atr_value
    lower_band = hl2 - multiplier * atr_value

    supertrend_line = pd.Series(index=close.index, dtype=float)
    trend = pd.Series(index=close.index, dtype=int)

    if len(close) > 0:
        supertrend_line.iloc[0] = lower_band.iloc[0]
        trend.iloc[0] = 1

    for i in range(1, len(close)):
        prev_trend = trend.iloc[i-1]
        prev_st = supertrend_line.iloc[i-1]
        curr_upper = upper_band.iloc[i]
        curr_lower = lower_band.iloc[i]
        curr_close = close.iloc[i]

        if prev_trend == 1:
            if curr_close < prev_st:
                trend.iloc[i] = -1
                supertrend_line.iloc[i] = curr_upper
            else:
                trend.iloc[i] = 1
                supertrend_line.iloc[i] = max(curr_lower, prev_st)
        else:
            if curr_close > prev_st:
                trend.iloc[i] = 1
                supertrend_line.iloc[i] = curr_lower
            else:
                trend.iloc[i] = -1
                supertrend_line.iloc[i] = min(curr_upper, prev_st)

    return supertrend_line, trend

print("\n计算 SuperTrend...")
st_line, trend = supertrend(high, low, close, period=10, multiplier=3.0)

print(f"SuperTrend 线: {st_line.shape}")
print(f"趋势: {trend.shape}")
print(f"\n最后10行:")
print(f"收盘价: {close.tail(10).values}")
print(f"SuperTrend: {st_line.tail(10).values}")
print(f"趋势: {trend.tail(10).values}")

# 统计
trend_counts = trend.value_counts()
print(f"\n趋势统计:")
print(f"  上涨趋势(1): {trend_counts.get(1, 0)} 次")
print(f"  下跌趋势(-1): {trend_counts.get(-1, 0)} 次")

# 买卖信号
trend_change = trend.diff()
buy_signals = (trend_change == 2).sum()
sell_signals = (trend_change == -2).sum()
print(f"\n信号统计:")
print(f"  买入信号: {buy_signals} 次")
print(f"  卖出信号: {sell_signals} 次")

print("\n✅ SuperTrend 指标计算成功！")
