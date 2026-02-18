#!/usr/bin/env python3
"""
简化版标准版测试 - 只测试一个参数组合，用完整2023年数据
"""
from qlib_supertrend import SupertrendStrategy, load_data, run_backtest
import time

print("="*60)
print("标准版 SuperTrend 回测 - 简化测试")
print("="*60)

# 加载数据
print("\n1. 加载数据...")
df = load_data(freq="1min", start_date="2023-01-01", end_date="2023-12-31")
print(f"   数据加载完成: {len(df)} 行")
print(f"   时间范围: {df.index.min()} ~ {df.index.max()}")

# 测试标准参数
print("\n2. 运行标准参数回测...")
start_time = time.time()

strategy = SupertrendStrategy(
    period=10,
    multiplier=3.0
)

print(f"   策略: {strategy.name}")

df_strategy = strategy.generate_signal(df)
df_strategy = df_strategy.dropna(subset=['position'])

results = run_backtest(df_strategy, strategy.name)

print(f"\n   回测完成，耗时: {time.time() - start_time:.2f}秒")
print("\n3. 回测结果:")
print(f"   策略名称: {results['strategy_name']}")
print(f"   总交易次数: {results['total_trades']}")
print(f"   累计收益: {results['cumulative_return']:.2%}")
print(f"   年化收益: {results['annual_return']:.2%}")
print(f"   最大回撤: {results['max_drawdown']:.2%}")
print(f"   夏普比率: {results['sharpe_ratio']:.2f}")
print(f"   胜率: {results['win_rate']:.2f}%")
print(f"   买入持有收益: {results['buy_hold_return']:.2%}")

# 额外统计
print("\n4. 详细统计:")
print(f"   趋势变化次数: {(df_strategy['trend'].diff() != 0).sum()}")
print(f"   持仓行数: {(df_strategy['position'] != 0).sum()}")

# 统计多空次数
long_positions = (df_strategy['position'] == 1).sum()
short_positions = (df_strategy['position'] == -1).sum()
print(f"   做多行数: {long_positions}")
print(f"   做空行数: {short_positions}")

# 计算平均持仓时长
if long_positions + short_positions > 0:
    avg_holding_ratio = (long_positions + short_positions) / len(df_strategy)
    print(f"   持仓比例: {avg_holding_ratio:.2%}")

print("\n" + "="*60)
print("测试完成!")
print("="*60)
