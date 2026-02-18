#!/usr/bin/env python3
"""
V03混合策略 - 优化结果总结脚本
"""
import sys
import json
import pandas as pd
from pathlib import Path

current_dir = Path(__file__).parent
results_dir = current_dir / "results"

# 查找最新的优化历史
histories = list(results_dir.glob("*/optimization_history.csv"))
if not histories:
    print("错误: 未找到优化历史文件")
    sys.exit(1)

latest_history = max(histories, key=lambda p: p.stat().st_mtime)
print(f"加载优化历史: {latest_history}")

# 读取优化历史
df = pd.read_csv(latest_history)

# 找到最优参数
best_row = df.loc[df['score'].idxmax()]

print("\n" + "="*80)
print("V03混合策略 - 参数优化结果总结")
print("="*80)

print("\n【训练集最优参数】")
print(f"  pivot_len:      {int(best_row['pivot_len'])} (原值: 7)")
print(f"  msb_zscore:     {best_row['msb_zscore']:.3f} (原值: 0.3)")
print(f"  atr_period:     {int(best_row['atr_period'])} (原值: 14)")
print(f"  atr_multiplier: {best_row['atr_multiplier']:.3f} (原值: 1.0)")
print(f"  tp1_multiplier: {best_row['tp1_multiplier']:.3f} (原值: 0.5)")
print(f"  tp2_multiplier: {best_row['tp2_multiplier']:.3f} (原值: 1.0)")
print(f"  tp3_multiplier: {best_row['tp3_multiplier']:.3f} (原值: 1.5)")

print(f"\n【训练集性能指标】")
print(f"  得分:           {best_row['score']:.4f}")
print(f"  盈亏比:         {best_row['profit_loss_ratio']:.2f} (原值: 0.46)")
print(f"  夏普比率:       {best_row['sharpe_ratio']:.2f} (原值: 1.17)")
print(f"  累计收益:       {best_row['cumulative_return']:.2%} (原值: 16.31%)")
print(f"  胜率:           {best_row['win_rate']:.2f}% (原值: 83.12%)")
print(f"  最大回撤:       {best_row['max_drawdown']:.2%} (原值: 3.40%)")
print(f"  交易次数:       {int(best_row['total_trades'])}")

print(f"\n【参数变化分析】")
print(f"  关键变化:")
print(f"    - MSB阈值: 0.3 → {best_row['msb_zscore']:.3f} ({'降低' if best_row['msb_zscore'] < 0.3 else '提高'} {(best_row['msb_zscore']/0.3-1)*100:+.1f}%)")
print(f"    - ATR倍数: 1.0 → {best_row['atr_multiplier']:.3f} (提高 {(best_row['atr_multiplier']/1.0-1)*100:+.1f}%)")
print(f"    - TP1倍数: 0.5 → {best_row['tp1_multiplier']:.3f} (提高 {(best_row['tp1_multiplier']/0.5-1)*100:+.1f}%)")
print(f"    - TP3倍数: 1.5 → {best_row['tp3_multiplier']:.3f} (提高 {(best_row['tp3_multiplier']/1.5-1)*100:+.1f}%)")

print(f"\n【优化效果评估】")
train_pl_ratio = best_row['profit_loss_ratio']
train_sharpe = best_row['sharpe_ratio']
train_return = best_row['cumulative_return']

pl_ratio_improve = (train_pl_ratio - 0.46) / 0.46 * 100
sharpe_improve = (train_sharpe - 1.17) / 1.17 * 100
return_improve = (train_return - 0.1631) / 0.1631 * 100

print(f"  训练集指标提升:")
print(f"    盈亏比:   0.46 → {train_pl_ratio:.2f} ({pl_ratio_improve:+.1f}%)")
print(f"    夏普比率: 1.17 → {train_sharpe:.2f} ({sharpe_improve:+.1f}%)")
print(f"    累计收益: 16.31% → {train_return*100:.2f}% ({return_improve:+.1f}%)")

# 检查达标情况
print(f"\n【目标达成情况】")
success_count = 0
total_criteria = 3

if train_pl_ratio >= 1.2:
    print(f"  ✓ 盈亏比目标: {train_pl_ratio:.2f} >= 1.2")
    success_count += 1
else:
    print(f"  ✗ 盈亏比目标: {train_pl_ratio:.2f} < 1.2 (差距: {(1.2-train_pl_ratio)/1.2*100:.1f}%)")

if train_sharpe >= 2.0:
    print(f"  ✓ 夏普比率目标: {train_sharpe:.2f} >= 2.0")
    success_count += 1
else:
    print(f"  ✗ 夏普比率目标: {train_sharpe:.2f} < 2.0 (差距: {(2.0-train_sharpe)/2.0*100:.1f}%)")

if train_return >= 0.25:
    print(f"  ✓ 累计收益目标: {train_return:.2%} >= 25%")
    success_count += 1
else:
    print(f"  ✗ 累计收益目标: {train_return:.2%} < 25% (差距: {(0.25-train_return)/0.25*100:.1f}%)")

print(f"\n  达成率: {success_count}/{total_criteria} ({success_count/total_criteria*100:.1f}%)")

# 验证集警告
print(f"\n【验证集性能警告】")
print(f"  ⚠️ 验证集(2025年)结果显示:")
print(f"    - 盈亏比: 0.47 (训练集: {train_pl_ratio:.2f})")
print(f"    - 累计收益: 4.04% (训练集: {train_return*100:.2f}%)")
print(f"    - 夏普比率: 0.98 (训练集: {train_sharpe:.2f})")
print(f"  ⚠️ 存在过拟合风险！训练集性能显著优于验证集")

print(f"\n【建议】")
print(f"  1. 训练集盈亏比提升至{train_pl_ratio:.2f}，但验证集仅0.47，存在明显过拟合")
print(f"  2. 建议:")
print(f"     - 增加训练数据量（包含更多年份）")
print(f"     - 使用更严格的约束条件（如提高最低交易次数要求）")
print(f"     - 考虑使用Walk-Forward验证方法")
print(f"     - 尝试不同的优化目标（降低盈亏比权重，增加稳定性权重）")
print(f"  3. 或者接受当前策略在60分钟周期上的限制，考虑其他时间周期")

print("\n" + "="*80)

# 保存最优参数到JSON
best_params = {
    'pivot_len': int(best_row['pivot_len']),
    'msb_zscore': float(best_row['msb_zscore']),
    'atr_period': int(best_row['atr_period']),
    'atr_multiplier': float(best_row['atr_multiplier']),
    'tp1_multiplier': float(best_row['tp1_multiplier']),
    'tp2_multiplier': float(best_row['tp2_multiplier']),
    'tp3_multiplier': float(best_row['tp3_multiplier']),
}

output_file = latest_history.parent / "best_params_summary.json"
with open(output_file, 'w', encoding='utf-8') as f:
    json.dump(best_params, f, indent=2, ensure_ascii=False)

print(f"\n最优参数已保存: {output_file}")
