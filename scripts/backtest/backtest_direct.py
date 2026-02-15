#!/usr/bin/env python3
"""
直接使用 CSV 数据进行回测（不依赖 qlib 数据加载器）
"""
import pandas as pd
import numpy as np
from pathlib import Path

# 配置路径
DATA_DIR = Path("/mnt/d/quant/qlib_data")

# 读取 RB9999.XSGE 的数据
instrument = "RB9999.XSGE"
instrument_dir = DATA_DIR / "instruments" / instrument

print(f"读取合约数据: {instrument}")

# 读取各个字段
data = {}
for field in ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'open_interest']:
    field_file = instrument_dir / f"{field}.csv"
    if field_file.exists():
        df = pd.read_csv(field_file, index_col=0, parse_dates=True)
        data[field] = df.iloc[:, 0]  # 获取第一列数据
        print(f"  {field}: {len(df)} 行")

# 合并为单个 DataFrame
df = pd.DataFrame(data)
print(f"\n数据形状: {df.shape}")
print(f"时间范围: {df.index.min()} ~ {df.index.max()}")
print(f"\n前5行数据:")
print(df.head())

# 限制回测时间范围（2023年）
start_date = "2023-01-01"
end_date = "2023-12-31"
df = df[(df.index >= start_date) & (df.index <= end_date)]
print(f"\n回测时间范围: {start_date} ~ {end_date}")
print(f"回测数据量: {len(df)} 行")

# ====== 策略：布林带均值回归 ======

# 计算技术指标
window = 20
df['ma'] = df['close'].rolling(window=window).mean()
df['std'] = df['close'].rolling(window=window).std()
df['upper_band'] = df['ma'] + 2 * df['std']
df['lower_band'] = df['ma'] - 2 * df['std']

# 生成交易信号
# 1: 做多, 0: 空仓, -1: 做空
df['position'] = 0

# 简单的均值回归策略
df.loc[df['close'] < df['lower_band'], 'position'] = 1  # 价格低于下轨，做多
df.loc[df['close'] > df['upper_band'], 'position'] = -1  # 价格高于上轨，做空

# 移除前 window 行（用于计算 MA 的数据不足）
df_strategy = df.iloc[window:].copy()

# 计算收益
df_strategy['returns'] = df_strategy['close'].pct_change()
df_strategy['strategy_returns'] = df_strategy['position'].shift(1) * df_strategy['returns']

# 基准收益（买入持有）
df_strategy['buy_hold_returns'] = df_strategy['returns']

# 统计结果
total_trades = (df_strategy['position'].diff() != 0).sum()
total_return = df_strategy['strategy_returns'].sum()
cumulative_return = (1 + df_strategy['strategy_returns']).prod() - 1
max_drawdown = (1 + df_strategy['strategy_returns'].cumsum()).expanding().max() - (1 + df_strategy['strategy_returns'].cumsum())
max_drawdown = max_drawdown.max()

win_rate = (df_strategy['strategy_returns'] > 0).sum() / (df_strategy['strategy_returns'] != 0).sum() * 100

# 年化指标（假设每天240个1分钟K线，252个交易日）
annual_trading_minutes = 240 * 252
annual_return = (1 + cumulative_return) ** (annual_trading_minutes / len(df_strategy)) - 1

# 夏普比率（假设无风险利率为0）
sharpe_ratio = df_strategy['strategy_returns'].mean() / df_strategy['strategy_returns'].std() * np.sqrt(annual_trading_minutes)

# 买入持有基准
buy_hold_return = df_strategy['buy_hold_returns'].sum()
buy_hold_cumulative = (1 + df_strategy['buy_hold_returns']).prod() - 1

print(f"\n{'='*50}")
print(f"布林带均值回归策略回测结果")
print(f"{'='*50}")
print(f"回测期间: {start_date} ~ {end_date}")
print(f"总交易次数: {total_trades}")
print(f"\n策略表现:")
print(f"  累计收益: {cumulative_return:.2%}")
print(f"  年化收益: {annual_return:.2%}")
print(f"  最大回撤: {max_drawdown:.2%}")
print(f"  夏普比率: {sharpe_ratio:.2f}")
print(f"  胜率: {win_rate:.2f}%")
print(f"\n买入持有基准:")
print(f"  累计收益: {buy_hold_cumulative:.2%}")

# 保存结果
output_file = Path("/mnt/d/quant/qlib_backtest/backtest_results_direct.csv")
df_strategy.to_csv(output_file)
print(f"\n结果已保存到: {output_file}")

# 绘制累计收益曲线（可选）
try:
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 6))
    plt.plot((1 + df_strategy['strategy_returns']).cumprod() - 1, label='策略累计收益', linewidth=2)
    plt.plot((1 + df_strategy['buy_hold_returns']).cumprod() - 1, label='买入持有', linewidth=2)
    plt.xlabel('时间')
    plt.ylabel('累计收益')
    plt.title('布林带均值回归策略 vs 买入持有')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plot_file = Path("/mnt/d/quant/qlib_backtest/backtest_curve.png")
    plt.savefig(plot_file, dpi=150, bbox_inches='tight')
    print(f"收益曲线已保存到: {plot_file}")
except ImportError:
    print("\n提示: 安装 matplotlib 可以生成收益曲线图")
    print("  pip install matplotlib")
