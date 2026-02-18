#!/usr/bin/env python3
"""
简单的回测策略示例：基于布林带的均值回归策略
"""
import qlib
from qlib.config import REG_CN
import pandas as pd
import numpy as np

# 初始化 qlib
qlib.init(provider_uri="/mnt/d/quant/qlib_backtest/data/qlib_data", region=REG_CN)

# 获取数据
from qlib.data import D

instruments = ["RB9999.XSGE"]

# 定义特征
fields = [
    "$close",
    "$high",
    "$low",
    "$volume",
    "$vwap",
    "$open_interest"
]

# 获取数据（指定频率为 1min）
df = D.features(instruments, fields, start_time="2023-01-03", end_time="2023-12-31", freq="1min")
print("数据形状:", df.shape)
print("\n前5行数据:")
print(df.head())

# 计算技术指标
df['$ma20'] = df['$close'].rolling(window=20).mean()
df['$std20'] = df['$close'].rolling(window=20).std()
df['$upper_band'] = df['$ma20'] + 2 * df['$std20']
df['$lower_band'] = df['$ma20'] - 2 * df['$std20']

# 简单策略：价格触及下轨买入，触及上轨卖出
df['$signal'] = 0
df.loc[df['$close'] < df['$lower_band'], '$signal'] = 1  # 买入信号
df.loc[df['$close'] > df['$upper_band'], '$signal'] = -1  # 卖出信号

# 移除前20行（用于计算MA的数据不足）
df = df.iloc[20:]

# 计算收益
df['$returns'] = df['$close'].pct_change()
df['$strategy_returns'] = df['$signal'].shift(1) * df['$returns']

# 统计结果
total_trades = (df['$signal'].diff() != 0).sum()
total_return = df['$strategy_returns'].sum()
cumulative_return = (1 + df['$strategy_returns']).prod() - 1
sharpe_ratio = df['$strategy_returns'].mean() / df['$strategy_returns'].std() * np.sqrt(252 * 240)  # 假设252天，每天240个1分钟K线

print(f"\n=== 回测结果 ===")
print(f"总交易次数: {total_trades}")
print(f"累计收益: {cumulative_return:.2%}")
print(f"年化夏普比率: {sharpe_ratio:.2f}")

# 保存结果
df.to_csv("/mnt/d/quant/qlib_backtest/backtest_results.csv")
print(f"\n结果已保存到: /mnt/d/quant/qlib_backtest/backtest_results.csv")
