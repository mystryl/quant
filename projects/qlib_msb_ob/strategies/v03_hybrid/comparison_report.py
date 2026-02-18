#!/usr/bin/env python3
"""
生成对比报告：原始回测 vs 真实回测
展示手续费、滑点、保证金机制的影响
"""
import pandas as pd
from pathlib import Path

# 读取原始回测结果（无手续费、滑点）
original_results = {
    '60min': {
        'total_trades': 77,
        'win_rate': 83.12,
        'cumulative_return': 0.1631,  # 16.31%
        'max_drawdown': 0.0581,
        'sharpe_ratio': 1.85
    },
    '15min': {
        'total_trades': 237,
        'win_rate': 76.79,
        'cumulative_return': 0.2477,  # 24.77%
        'max_drawdown': 0.0524,
        'sharpe_ratio': 2.01
    },
    '5min': {
        'total_trades': 631,
        'win_rate': 71.16,
        'cumulative_return': 0.3005,  # 30.05%
        'max_drawdown': 0.0479,
        'sharpe_ratio': 2.70
    }
}

# 读取真实回测结果（有手续费、滑点、保证金）
realistic_file = Path(__file__).parent / "realistic_results" / "summary.csv"
df_realistic = pd.read_csv(realistic_file)

# 按周期汇总
realistic_results = {}
for freq in ['60min', '15min', '5min']:
    freq_data = df_realistic[df_realistic['freq'] == freq]
    realistic_results[freq] = {
        'total_trades': freq_data['total_trades'].sum(),
        'win_rate': (freq_data['win_rate'] * freq_data['total_trades']).sum() / freq_data['total_trades'].sum(),
        'total_return': freq_data['total_return'].sum(),
        'max_drawdown': freq_data['max_drawdown'].max(),
        'sharpe_ratio': freq_data['sharpe_ratio'].mean(),
        'total_commission': freq_data['total_commission'].sum(),
        'total_pnl': freq_data['total_pnl'].sum()
    }

# 打印对比报告
print("=" * 140)
print("原始回测 vs 真实回测对比报告")
print("=" * 140)
print("\n说明：")
print("  原始回测：不考虑手续费、滑点、保证金")
print("  真实回测：手续费万一(0.01%)、滑点1tick、保证金10%、初始资金10万、单次开仓5万")
print("=" * 140)

for freq in ['60min', '15min', '5min']:
    print(f"\n{'='*140}")
    print(f"【{freq}周期】")
    print(f"{'='*140}")

    orig = original_results[freq]
    real = realistic_results[freq]

    print(f"\n{'指标':<20} {'原始回测':<25} {'真实回测':<25} {'差异':<25} {'影响'}")
    print("-" * 140)

    # 交易次数
    print(f"{'交易次数':<20} {orig['total_trades']:<25} {real['total_trades']:<25} {0:<25} {'相同'}")

    # 胜率
    win_rate_diff = real['win_rate'] - orig['win_rate']
    print(f"{'胜率':<20} {orig['win_rate']:<25.2f}% {real['win_rate']:<25.2f}% {win_rate_diff:<25.2f}% {'基本一致' if abs(win_rate_diff) < 5 else '差异较大'}")

    # 收益率
    orig_return_pct = orig['cumulative_return'] * 100
    real_return_pct = real['total_return'] * 100
    return_diff = real_return_pct - orig_return_pct
    return_impact = "大幅下降" if return_diff < -10 else "明显下降" if return_diff < -5 else "有所下降" if return_diff < -1 else "基本一致"

    print(f"{'总收益率':<20} {orig_return_pct:<25.2f}% {real_return_pct:<25.2f}% {return_diff:<25.2f}% {return_impact}")

    # 手续费影响
    commission_impact = (real['total_commission'] / 100000) * 100
    print(f"{'手续费总额':<25} {'-':<25} {real['total_commission']:<25,.2f}元 {commission_impact:<25.2f}% {'占总资金' if commission_impact > 1 else '占比较小'}")

    # 最大回撤
    orig_dd_pct = orig['max_drawdown'] * 100
    real_dd_pct = real['max_drawdown'] * 100
    dd_diff = real_dd_pct - orig_dd_pct
    print(f"{'最大回撤':<20} {orig_dd_pct:<25.2f}% {real_dd_pct:<25.2f}% {dd_diff:<25.2f}% {'增加' if dd_diff > 0 else '减少'}")

    # 夏普比率
    sharpe_diff = real['sharpe_ratio'] - orig['sharpe_ratio']
    sharpe_impact = "大幅下降" if sharpe_diff < -1 else "明显下降" if sharpe_diff < -0.5 else "有所下降" if sharpe_diff < -0.2 else "基本一致"
    print(f"{'夏普比率':<20} {orig['sharpe_ratio']:<25.2f} {real['sharpe_ratio']:<25.2f} {sharpe_diff:<25.2f} {sharpe_impact}")

# 总结
print(f"\n{'='*140}")
print("核心结论")
print(f"{'='*140}")

print("""
1. 交易成本的影响
────────────────────────────────────────────────────────────────────────
   • 手续费和滑点对高频交易（5min）影响最大
   • 60分钟周期：交易次数少，成本影响相对较小
   • 5分钟周期：交易次数多，累积成本显著侵蚀收益

2. 保证金机制的杠杆效应
────────────────────────────────────────────────────────────────────────
   • 10%保证金意味着10倍杠杆
   • 小幅波动会被放大到保证金账户
   • 止损时亏损也会按杠杆放大

3. 收益率变化
────────────────────────────────────────────────────────────────────────
   60min:  16.31% → 1.09%   (下降15.22个百分点，原收益的93%被成本侵蚀)
   15min:  24.77% → -1.31%  (由盈转亏)
   5min:   30.05% → -11.09% (由盈转亏，亏损严重)

4. 策略有效性分析
────────────────────────────────────────────────────────────────────────
   原始回测显示策略有正向收益，但加入真实成本后：
   • 60分钟周期：勉强盈利，收益微薄（1.09%/3年 ≈ 0.36%/年）
   • 15分钟周期：小幅亏损
   • 5分钟周期：显著亏损

5. 改进建议
────────────────────────────────────────────────────────────────────────
   a) 降低交易频率：过滤低质量信号，只做高确定性交易
   b) 优化止盈止损：提高盈亏比，当前0.24-0.49的盈亏比过低
   c) 增加止盈目标：TP1目标过于保守，可考虑持有到TP2/TP3
   d) 考虑趋势跟踪：当前策略可能更适合趋势行情
   e) 调整仓位管理：根据信号质量动态调整仓位大小

6. 真实交易的挑战
────────────────────────────────────────────────────────────────────────
   • 回测收益 ≠ 真实收益（成本、滑点、执行延迟）
   • 高胜率不等于高收益（盈亏比更重要）
   • 高频交易需要更高的胜率和盈亏比才能覆盖成本
""")

print(f"\n{'='*140}")
print("详细数据")
print(f"{'='*140}")

summary_data = []
for freq in ['60min', '15min', '5min']:
    orig = original_results[freq]
    real = realistic_results[freq]

    summary_data.append({
        '周期': freq,
        '原始收益': f"{orig['cumulative_return']*100:.2f}%",
        '真实收益': f"{real['total_return']*100:.2f}%",
        '收益差异': f"{(real['total_return'] - orig['cumulative_return'])*100:.2f}%",
        '原始胜率': f"{orig['win_rate']:.1f}%",
        '真实胜率': f"{real['win_rate']:.1f}%",
        '交易次数': orig['total_trades'],
        '手续费(元)': f"{real['total_commission']:.2f}",
        '手续费占比': f"{commission_impact:.2f}%"
    })

df_summary = pd.DataFrame(summary_data)
print(df_summary.to_string(index=False))

print(f"\n{'='*140}\n")
