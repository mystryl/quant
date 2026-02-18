#!/usr/bin/env python3
"""
使用 Optuna 优化得到的最优参数进行回测

最优参数：
  period: 19
  multiplier: 1.55
  n: 1
  trailing_stop_rate: 80
  max_holding_period: 100
  min_liqka: 0.5
  max_liqka: 1.0
"""
import pandas as pd
import numpy as np
from pathlib import Path
from qlib_supertrend_enhanced import SupertrendEnhancedStrategy, run_backtest


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


def backtest_with_params(df, period, multiplier, n, trailing_stop_rate,
                       max_holding_period, min_liqka, max_liqka):
    """使用指定参数运行回测"""
    # 创建策略
    strategy = SupertrendEnhancedStrategy(
        period=period,
        multiplier=multiplier,
        n=n,
        trailing_stop_rate=trailing_stop_rate,
        max_holding_period=max_holding_period,
        min_liqka=min_liqka,
        max_liqka=max_liqka
    )

    # 生成信号
    df_strategy = strategy.generate_signal(df.copy())
    df_strategy = df_strategy.dropna(subset=['position'])

    if len(df_strategy) == 0:
        return None

    # 运行回测
    results = run_backtest(df_strategy, strategy.name)

    # 添加数据长度
    results['data_length'] = len(df_strategy)

    return results


def main():
    """主程序"""
    print("="*80)
    print("最优参数回测 - Optuna 优化结果")
    print("="*80)

    # 最优参数
    params = {
        'period': 19,
        'multiplier': 1.55,
        'n': 1,
        'trailing_stop_rate': 80,
        'max_holding_period': 100,
        'min_liqka': 0.5,
        'max_liqka': 1.0
    }

    print(f"\n最优参数（Optuna 试验 #37）：")
    print(f"   period: {params['period']}")
    print(f"   multiplier: {params['multiplier']}")
    print(f"   n: {params['n']}")
    print(f"   trailing_stop_rate: {params['trailing_stop_rate']}")
    print(f"   max_holding_period: {params['max_holding_period']}")
    print(f"   min_liqka: {params['min_liqka']}")
    print(f"   max_liqka: {params['max_liqka']}")

    # 测试配置
    years = [2023, 2024, 2025]
    freqs = ["15min", "60min"]

    all_results = []

    for year in years:
        for freq in freqs:
            print(f"\n{'='*80}")
            print(f"年份: {year}, 频率: {freq}")
            print(f"{'='*80}")

            # 加载数据
            print(f"加载数据...")
            df = load_data_multi_freq(freq=freq,
                                      start_date=f"{year}-01-01",
                                      end_date=f"{year}-12-31")

            if df is None or len(df) == 0:
                print(f"   ⚠️  数据不存在，跳过")
                continue

            print(f"   数据加载完成: {len(df)} 行")
            print(f"   时间范围: {df.index.min()} ~ {df.index.max()}")

            # 运行回测
            print(f"\n运行回测...")
            results = backtest_with_params(
                df,
                params['period'],
                params['multiplier'],
                params['n'],
                params['trailing_stop_rate'],
                params['max_holding_period'],
                params['min_liqka'],
                params['max_liqka']
            )

            if results is None:
                print(f"   ⚠️  回测失败，跳过")
                continue

            # 添加年份和频率信息
            results['year'] = year
            results['freq'] = freq

            all_results.append(results)

            # 输出结果
            print(f"\n回测结果：")
            print(f"   策略名称: {results['strategy_name']}")
            print(f"   累计收益: {results['cumulative_return']:.2%}")
            print(f"   年化收益: {results['annual_return']:.2%}")
            print(f"   最大回撤: {results['max_drawdown']:.2%}")
            print(f"   夏普比率: {results['sharpe_ratio']:.2f}")
            print(f"   胜率: {results['win_rate']:.2f}%")
            print(f"   总交易次数: {results['total_trades']}")
            print(f"   买入持有收益: {results['buy_hold_return']:.2%}")
            print(f"   止损平仓次数: {results['stopped_out_count']}")

    # 保存结果
    if len(all_results) > 0:
        print(f"\n{'='*80}")
        print("保存结果...")
        print(f"{'='*80}")

        # 转换为 DataFrame
        results_df = pd.DataFrame(all_results)

        # 调整列顺序
        cols_order = [
            'year', 'freq', 'strategy_name',
            'total_trades', 'cumulative_return', 'annual_return',
            'max_drawdown', 'sharpe_ratio', 'win_rate',
            'buy_hold_return', 'stopped_out_count', 'data_length'
        ]
        existing_cols = [col for col in cols_order if col in results_df.columns]
        results_df = results_df[existing_cols]

        # 保存到 CSV
        output_file = Path("/mnt/d/quant/qlib_backtest/supertrend_optuna_optimized_results.csv")
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"   结果已保存到: {output_file}")

        # 显示汇总表格
        print(f"\n{'='*80}")
        print("汇总结果")
        print(f"{'='*80}\n")

        # 格式化表格
        display_df = results_df.copy()
        display_df['cumulative_return'] = display_df['cumulative_return'].apply(lambda x: f"{x:.2%}")
        display_df['annual_return'] = display_df['annual_return'].apply(lambda x: f"{x:.2%}")
        display_df['max_drawdown'] = display_df['max_drawdown'].apply(lambda x: f"{x:.2%}")
        display_df['buy_hold_return'] = display_df['buy_hold_return'].apply(lambda x: f"{x:.2%}")
        display_df['win_rate'] = display_df['win_rate'].apply(lambda x: f"{x:.2f}%")

        # 只显示关键列
        display_cols = ['year', 'freq', 'cumulative_return', 'annual_return',
                       'max_drawdown', 'sharpe_ratio', 'total_trades']

        print(display_df[display_cols].to_string(index=False))

        # 计算总体统计
        print(f"\n{'='*80}")
        print("总体统计")
        print(f"{'='*80}")

        # 按频率汇总
        freq_summary = results_df.groupby('freq').agg({
            'cumulative_return': 'mean',
            'annual_return': 'mean',
            'max_drawdown': 'mean',
            'sharpe_ratio': 'mean',
            'total_trades': 'sum'
        }).reset_index()

        print("\n按频率汇总：")
        for idx, row in freq_summary.iterrows():
            print(f"\n{row['freq']}:")
            print(f"   平均累计收益: {row['cumulative_return']:.2%}")
            print(f"   平均年化收益: {row['annual_return']:.2%}")
            print(f"   平均最大回撤: {row['max_drawdown']:.2%}")
            print(f"   平均夏普比率: {row['sharpe_ratio']:.2f}")
            print(f"   总交易次数: {int(row['total_trades'])}")

    else:
        print(f"\n   ⚠️  没有有效的回测结果")

    print(f"\n{'='*80}")
    print("回测完成!")
    print(f"{'='*80}")

    return all_results


if __name__ == "__main__":
    results = main()
