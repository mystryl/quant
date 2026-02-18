#!/usr/bin/env python3
"""
SuperTrend 增强版参数优化工具

帮助找到适合不同频率的最佳参数组合
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
from qlib_supertrend_enhanced import SupertrendEnhancedStrategy, run_backtest
from qlib_supertrend import SupertrendStrategy as StandardSupertrendStrategy


def load_data_multi_freq(freq="1min", start_date="2023-01-01", end_date="2023-12-31"):
    """加载 qlib 数据（支持多频率）"""
    DATA_DIR = Path("/mnt/d/quant/qlib_data")
    RESAMPLED_DIR = Path("/mnt/d/quant/qlib_backtest/qlib_data_multi_freq")

    FIELD_MAPPING = {
        'open': '$open',
        'high': '$high',
        'low': '$low',
        'close': '$close',
        'volume': '$volume',
        'amount': '$amount',
        'vwap': '$vwap',
        'open_interest': '$open_interest'
    }

    instrument = "RB9999.XSGE"

    if freq == "1min":
        data_dir = DATA_DIR / "instruments" / instrument
    else:
        data_dir = RESAMPLED_DIR / "instruments" / freq

    # 读取各个字段
    data = {}
    for field in ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'open_interest']:
        if freq == "1min":
            field_file = data_dir / f"{field}.csv"
        else:
            feature_name = FIELD_MAPPING[field]
            field_file = data_dir / feature_name / f"{instrument}.csv"

        if field_file.exists():
            df = pd.read_csv(field_file, index_col=0, parse_dates=True)
            data[field] = df.iloc[:, 0]

    if len(data) == 0:
        print(f"   ⚠️  警告: {freq} 数据文件不存在")
        return None

    df = pd.DataFrame(data)
    df = df.sort_index()
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    return df


def optimize_for_freq(freq="15min", test_years=[2023, 2024, 2025]):
    """
    针对特定频率优化参数

    Args:
        freq: 测试频率
        test_years: 测试年份列表
    """
    print(f"\n{'='*80}")
    print(f"频率: {freq} - 参数优化")
    print(f"{'='*80}")

    # 参数组合
    param_grids = {
        '1min': [
            (50, 20, 3, 80),  # SF14Re 标准参数
            (30, 15, 2, 70),
            (40, 18, 3, 75),
        ],
        '15min': [
            (20, 10, 2, 65),
            (25, 12, 2, 70),
            (30, 15, 2, 65),
            (20, 8, 2, 60),
            (25, 10, 2, 65),
        ],
        '60min': [
            (12, 6, 2, 55),
            (10, 5, 2, 50),
            (15, 8, 2, 60),
            (12, 5, 2, 50),
            (10, 6, 2, 55),
        ]
    }

    # 选择合适的参数网格
    if freq in param_grids:
        test_params = param_grids[freq]
    else:
        print(f"   ⚠️  频率 {freq} 没有预定义参数网格，使用默认参数")
        test_params = [(50, 20, 3, 80)]

    print(f"\n测试参数组合: {len(test_params)} 个")
    all_results = []

    # 测试每个参数组合
    for idx, (period, multiplier, n, ts_rate) in enumerate(test_params, 1):
        print(f"\n{'-'*80}")
        print(f"参数组合 {idx}/{len(test_params)}: period={period}, multiplier={multiplier}, n={n}, ts={ts_rate}")
        print(f"{'-'*80}")

        yearly_results = []

        for year in test_years:
            print(f"\n  年份: {year}")

            # 加载数据
            df = load_data_multi_freq(freq=freq, start_date=f"{year}-01-01", end_date=f"{year}-12-31")

            if df is None or len(df) == 0:
                print(f"     ⚠️  数据不存在，跳过")
                continue

            print(f"     数据行数: {len(df)}")

            # 创建策略
            strategy = SupertrendEnhancedStrategy(
                period=period,
                multiplier=multiplier,
                n=n,
                trailing_stop_rate=ts_rate,
                max_holding_period=100,
                min_liqka=0.5,
                max_liqka=1.0
            )

            # 生成信号
            df_strategy = strategy.generate_signal(df.copy())
            df_strategy = df_strategy.dropna(subset=['position'])

            # 运行回测
            results = run_backtest(df_strategy, strategy.name)
            yearly_results.append(results)

            # 输出结果
            print(f"     累计收益: {results['cumulative_return']:.2%}")
            print(f"     夏普比率: {results['sharpe_ratio']:.2f}")
            print(f"     交易次数: {results['total_trades']}")

        # 计算平均表现
        if len(yearly_results) > 0:
            avg_cum_return = np.mean([r['cumulative_return'] for r in yearly_results])
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in yearly_results])
            avg_max_drawdown = np.mean([r['max_drawdown'] for r in yearly_results])
            avg_trades = np.mean([r['total_trades'] for r in yearly_results])

            summary_result = {
                'period': period,
                'multiplier': multiplier,
                'n': n,
                'trailing_stop_rate': ts_rate,
                'avg_cumulative_return': avg_cum_return,
                'avg_sharpe_ratio': avg_sharpe,
                'avg_max_drawdown': avg_max_drawdown,
                'avg_total_trades': avg_trades,
                'yearly_results': yearly_results
            }
            all_results.append(summary_result)

    # 排序结果
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        results_df = results_df.sort_values('avg_sharpe_ratio', ascending=False)

        print(f"\n{'='*80}")
        print(f"频率: {freq} - 参数优化结果排名（按夏普比率排序）")
        print(f"{'='*80}")

        print(f"\n{'排名':<4} {'period':<8} {'multiplier':<10} {'n':<4} {'ts':<6} "
              f"{'平均累计':<10} {'平均夏普':<10} {'平均回撤':<10} {'平均交易':<10}")
        print("-"*80)

        for idx, row in results_df.head(5).iterrows():
            print(f"{int(row.name)+1:<4} {row['period']:<8} {row['multiplier']:<10} "
                  f"{row['n']:<4} {row['trailing_stop_rate']:<6} "
                  f"{row['avg_cumulative_return']:>8.2%} {row['avg_sharpe_ratio']:>8.2f} "
                  f"{row['avg_max_drawdown']:>8.2%} {row['avg_total_trades']:>8.1f}")

        print(f"\n{'='*80}")
        print("最佳参数组合")
        print(f"{'='*80}")

        best = results_df.iloc[0]
        print(f"  period: {best['period']}")
        print(f"  multiplier: {best['multiplier']}")
        print(f"  n: {best['n']}")
        print(f"  trailing_stop_rate: {best['trailing_stop_rate']}")
        print(f"  平均夏普比率: {best['avg_sharpe_ratio']:.2f}")
        print(f"  平均累计收益: {best['avg_cumulative_return']:.2%}")

        return results_df
    else:
        print(f"\n   ⚠️  没有有效的回测结果")
        return None


def main():
    """主程序"""
    print("="*80)
    print("SuperTrend 增强版参数优化工具")
    print("="*80)

    # 测试配置
    test_freqs = ["15min", "60min"]
    test_years = [2023, 2024, 2025]

    all_freq_results = {}

    for freq in test_freqs:
        results_df = optimize_for_freq(freq, test_years)
        if results_df is not None:
            all_freq_results[freq] = results_df

    # 保存结果
    if len(all_freq_results) > 0:
        print(f"\n{'='*80}")
        print("保存优化结果...")
        print(f"{'='*80}")

        for freq, results_df in all_freq_results.items():
            output_file = f"/mnt/d/quant/qlib_backtest/optimized_params_{freq}.csv"
            results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
            print(f"   {freq} 结果已保存到: {output_file}")

    print(f"\n{'='*80}")
    print("参数优化完成!")
    print(f"{'='*80}")

    return all_freq_results


if __name__ == "__main__":
    results = main()
