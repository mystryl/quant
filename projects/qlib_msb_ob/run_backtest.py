#!/usr/bin/env python3
"""
MSB+OB 策略快速回测脚本

2023-2025年 60分钟线回测
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加路径
sys.path.insert(0, str(Path(__file__).parent))

from strategy.msb_ob_strategy import MSBOBStrategy, calculate_metrics, calculate_ob_stats


def load_data(year):
    """加载指定年份的数据"""
    data_file = Path(__file__).parent / "data" / f"RB9999_{year}_60min.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    df = pd.read_csv(data_file, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def run_backtest_for_year(year):
    """对单年数据运行回测"""
    print(f"\n{'='*80}")
    print(f"回测年份: {year}")
    print(f"{'='*80}")

    # 加载数据
    df = load_data(year)
    print(f"数据行数: {len(df)}")
    print(f"时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")

    # 创建策略
    strategy = MSBOBStrategy(
        pivot_len=7,
        msb_zscore=0.5,
        ob_max_count=10,
        hpz_threshold=80,
        atr_period=14,
        atr_multiplier=1.0,
        tp1_multiplier=0.5,
        tp2_multiplier=1.0,
        tp3_multiplier=1.5
    )

    print(f"\n策略参数: {strategy.name}")

    # 生成信号
    print("\n生成交易信号...")
    df_signals = strategy.generate_signal(df)

    # 计算指标
    print("计算回测指标...")
    metrics = calculate_metrics(df_signals, freq='60min')
    ob_stats = calculate_ob_stats(df_signals)

    # 打印结果
    print(f"\n{'='*80}")
    print(f"回测结果 - {year}年")
    print(f"{'='*80}")
    print(f"总交易次数: {metrics['total_trades']}")
    print(f"胜率: {metrics['win_rate']:.2f}%")
    print(f"平均盈利: {metrics['avg_win']:.2%}")
    print(f"平均亏损: {metrics['avg_loss']:.2%}")
    print(f"盈亏比: {metrics['profit_loss_ratio']:.2f}")
    print(f"\n累计收益: {metrics['cumulative_return']:.2%}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"\nOB统计:")
    print(f"  OB创建数: {ob_stats['total_ob_created']}")
    print(f"  OB失效数: {ob_stats['total_ob_mitigated']}")
    print(f"  OB可靠性: {ob_stats['ob_reliability']:.2f}%")
    print(f"  HP-OB数: {ob_stats['hp_ob_count']}")

    # 统计信号
    if 'bullish_msb' in df_signals.columns:
        msb_bull_count = df_signals['bullish_msb'].sum()
        msb_bear_count = df_signals['bearish_msb'].sum()
        print(f"\nMSB信号统计:")
        print(f"  看涨MSB: {int(msb_bull_count)}")
        print(f"  看跌MSB: {int(msb_bear_count)}")

    return {**metrics, **ob_stats, 'year': year}


def main():
    """主程序"""
    print("="*80)
    print("MSB+OB 策略回测 (2023-2025)")
    print("="*80)

    years = [2023, 2024, 2025]
    all_results = []

    for year in years:
        try:
            result = run_backtest_for_year(year)
            all_results.append(result)
        except Exception as e:
            print(f"\n⚠️ {year}年回测失败: {e}")
            import traceback
            traceback.print_exc()

    # 汇总结果
    if all_results:
        print(f"\n{'='*80}")
        print("汇总结果")
        print(f"{'='*80}")

        print(f"\n{'年份':<8} {'交易':<6} {'胜率':<8} {'累计':<10} {'回撤':<10} {'夏普':<8}")
        print("-"*60)

        for r in all_results:
            print(f"{r['year']:<8} {r['total_trades']:<6} "
                  f"{r['win_rate']:>6.2f}% "
                  f"{r['cumulative_return']:>8.2%} "
                  f"{r['max_drawdown']:>8.2%} "
                  f"{r['sharpe_ratio']:>6.2f}")

        # 保存结果
        output_file = Path(__file__).parent / "reports" / "backtest_results.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)

        df_results = pd.DataFrame(all_results)
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存到: {output_file}")

        # 总体统计
        total_trades = sum(r['total_trades'] for r in all_results)
        avg_win_rate = np.mean([r['win_rate'] for r in all_results])
        total_return = sum(r['cumulative_return'] for r in all_results)
        max_drawdown = max(r['max_drawdown'] for r in all_results)
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results])

        print(f"\n{'='*80}")
        print("总体统计")
        print(f"{'='*80}")
        print(f"总交易次数: {total_trades}")
        print(f"平均胜率: {avg_win_rate:.2f}%")
        print(f"总累计收益: {total_return:.2%}")
        print(f"最大回撤: {max_drawdown:.2%}")
        print(f"平均夏普: {avg_sharpe:.2f}")

        return all_results
    else:
        print("\n⚠️ 没有生成任何回测结果")
        return []


if __name__ == "__main__":
    results = main()
