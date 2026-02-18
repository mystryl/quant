#!/usr/bin/env python3
"""
V03策略详细回测
输出每个年份和周期的完整回测指标
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.v03_hybrid.strat_v03_hybrid import HybridMSBOBStrategy, calculate_metrics, load_data


def run_detailed_backtest(year, freq):
    """运行详细回测"""
    # 加载数据
    df = load_data(year, freq)
    print(f"  数据行数: {len(df)}")
    print(f"  时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")

    # 创建策略
    strategy = HybridMSBOBStrategy(
        pivot_len=7,
        msb_zscore=0.3,
        atr_period=14,
        atr_multiplier=1.0,
        tp1_multiplier=0.5,
        tp2_multiplier=1.0,
        tp3_multiplier=1.5
    )

    # 运行策略
    df_sig = strategy.run_strategy(df)
    metrics = calculate_metrics(df_sig, freq=freq)

    # 统计MSB和OB
    msb_bull = df_sig['msb_bullish'].sum()
    msb_bear = df_sig['msb_bearish'].sum()
    msb_total = msb_bull + msb_bear
    ob_created = df_sig['ob_created'].sum()

    return {
        **metrics,
        'msb_bull': msb_bull,
        'msb_bear': msb_bear,
        'msb_total': msb_total,
        'ob_created': ob_created
    }


def main():
    """主函数"""
    print("="*100)
    print("V03混合策略 - 详细回测报告")
    print("="*100)

    years = [2023, 2024, 2025]
    freqs = ['60min', '15min', '5min']

    for freq in freqs:
        print(f"\n{'='*100}")
        print(f"【{freq}周期】")
        print(f"{'='*100}")

        freq_results = []

        for year in years:
            print(f"\n{year}年:")
            print("-" * 100)

            try:
                result = run_detailed_backtest(year, freq)
                result['year'] = year
                result['freq'] = freq
                freq_results.append(result)

                # 打印详细指标
                print(f"  MSB统计:")
                print(f"    看涨MSB: {int(result['msb_bull'])}")
                print(f"    看跌MSB: {int(result['msb_bear'])}")
                print(f"    MSB总计: {int(result['msb_total'])}")
                print(f"    OB创建数: {int(result['ob_created'])}")
                print()
                print(f"  交易统计:")
                print(f"    总交易次数: {result['total_trades']}")
                print(f"    胜率: {result['win_rate']:.2f}%")
                print(f"    平均盈利: {result['avg_win']:.4f}")
                print(f"    平均亏损: {result['avg_loss']:.4f}")
                print(f"    盈亏比: {result['profit_loss_ratio']:.2f}")
                print()
                print(f"  风险收益:")
                print(f"    累计收益: {result['cumulative_return']:.2%}")
                print(f"    最大回撤: {result['max_drawdown']:.2%}")
                print(f"    夏普比率: {result['sharpe_ratio']:.2f}")

            except Exception as e:
                print(f"  错误: {e}")
                import traceback
                traceback.print_exc()

        # 打印周期汇总
        if freq_results:
            print(f"\n{'-'*100}")
            print(f"{freq}周期汇总:")
            print("-" * 100)

            total_trades = sum(r['total_trades'] for r in freq_results)
            total_msb = sum(r['msb_total'] for r in freq_results)
            total_ob = sum(r['ob_created'] for r in freq_results)

            # 加权平均胜率（按交易次数）
            weighted_win_rate = sum(r['win_rate'] * r['total_trades'] for r in freq_results if r['total_trades'] > 0)
            weighted_win_rate = weighted_win_rate / total_trades if total_trades > 0 else 0

            total_return = sum(r['cumulative_return'] for r in freq_results)
            max_dd = max(r['max_drawdown'] for r in freq_results)

            # 加权平均夏普
            weighted_sharpe = sum(r['sharpe_ratio'] for r in freq_results) / len(freq_results)

            print(f"  总交易次数: {total_trades}")
            print(f"  总MSB信号: {int(total_msb)}")
            print(f"  总OB创建: {int(total_ob)}")
            print(f"  加权胜率: {weighted_win_rate:.2f}%")
            print(f"  总累计收益: {total_return:.2%}")
            print(f"  最大回撤: {max_dd:.2%}")
            print(f"  平均夏普: {weighted_sharpe:.2f}")

    # 生成最终对比表
    print(f"\n{'='*100}")
    print("最终对比表")
    print(f"{'='*100}")

    all_results = []
    for freq in freqs:
        for year in years:
            try:
                result = run_detailed_backtest(year, freq)
                result['year'] = year
                result['freq'] = freq
                all_results.append(result)
            except:
                pass

    if all_results:
        print(f"\n{'年份':<8} {'周期':<10} {'交易':<6} {'胜率':<8} {'平均盈利':<12} {'平均亏损':<12} "
              f"{'盈亏比':<8} {'累计收益':<10} {'回撤':<10} {'夏普':<8}")
        print("-" * 110)

        for r in all_results:
            print(f"{r['year']:<8} {r['freq']:<10} {r['total_trades']:<6} "
                  f"{r['win_rate']:>6.2f}% "
                  f"{r['avg_win']:>10.4f}   "
                  f"{r['avg_loss']:>10.4f}   "
                  f"{r['profit_loss_ratio']:>6.2f}   "
                  f"{r['cumulative_return']:>8.2%}   "
                  f"{r['max_drawdown']:>8.2%}   "
                  f"{r['sharpe_ratio']:>6.2f}")

    print(f"\n{'='*100}")


if __name__ == "__main__":
    main()
