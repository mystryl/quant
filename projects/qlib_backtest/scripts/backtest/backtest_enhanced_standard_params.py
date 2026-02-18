#!/usr/bin/env python3
"""
增强版 SuperTrend 多年份多频率回测（使用标准版参数）

ATR 周期和倍数与标准版一致，但保留：
- 双重突破确认（n=3）
- 渐进式跟踪止损（ts=80）

测试参数：period=10, multiplier=3.0（与标准版最常用的参数一致）
"""
import pandas as pd
import numpy as np
from pathlib import Path
import time
from qlib_supertrend_enhanced import SupertrendEnhancedStrategy, run_backtest


def load_data_multi_freq(freq="1min", start_date="2023-01-01", end_date="2023-12-31"):
    """
    加载 qlib 数据（支持多频率）

    Args:
        freq: 频率 (1min, 5min, 15min, 60min)
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame 包含 OHLCV 数据
    """
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


def run_enhanced_backtest_for_params(df, strategy_name, results_list, freq, year):
    """运行增强版回测"""
    print(f"   测试策略: {strategy_name}")

    # 创建策略（使用标准版参数：period=10, multiplier=3.0）
    strategy = SupertrendEnhancedStrategy(
        period=10,          # 与标准版一致
        multiplier=3.0,      # 与标准版一致
        n=3,                # 保留突破确认
        trailing_stop_rate=80,
        max_holding_period=100,
        min_liqka=0.5,
        max_liqka=1.0
    )

    # 生成信号
    df_strategy = strategy.generate_signal(df.copy())
    df_strategy = df_strategy.dropna(subset=['position'])

    # 运行回测
    results = run_backtest(df_strategy, strategy_name)
    
    # 添加额外信息
    results['freq'] = freq
    results['year'] = year
    results['data_length'] = len(df_strategy)
    
    results_list.append(results)

    # 输出结果
    print(f"     累计收益: {results['cumulative_return']:.2%}")
    print(f"     年化收益: {results['annual_return']:.2%}")
    print(f"     最大回撤: {results['max_drawdown']:.2%}")
    print(f"     夏普比率: {results['sharpe_ratio']:.2f}")
    print(f"     胜率: {results['win_rate']:.2f}%")
    print(f"     总交易次数: {results['total_trades']}")
    print(f"     买入持有收益: {results['buy_hold_return']:.2%}")
    print(f"     止损平仓次数: {results['stopped_out_count']}")


def main():
    """主程序"""
    print("="*80)
    print("增强版 SuperTrend 回测（使用标准版参数）")
    print("="*80)

    # 测试配置
    years = [2023, 2024, 2025]
    freqs = ["15min", "60min"]

    all_results = []

    for year in years:
        for freq in freqs:
            print(f"\n{'='*80}")
            print(f"回测年份: {year}, 频率: {freq}")
            print(f"{'='*80}")

            # 加载数据
            print(f"\n加载数据...")
            df = load_data_multi_freq(freq=freq, start_date=f"{year}-01-01", end_date=f"{year}-12-31")

            if df is None or len(df) == 0:
                print(f"   ⚠️  {year} 年 {freq} 数据不存在，跳过")
                continue

            print(f"   数据加载完成: {len(df)} 行")
            print(f"   时间范围: {df.index.min()} ~ {df.index.max()}")

            # 运行增强版回测
            print(f"\n运行增强版 SuperTrend 回测...")
            start_time = time.time()

            strategy_name = f"SuperTrend_Enhanced(10,3,n=3,ts=80)"
            run_enhanced_backtest_for_params(df, strategy_name, all_results, freq, year)

            elapsed = time.time() - start_time
            print(f"   回测完成，耗时: {elapsed:.2f}秒")

    # 保存结果到 CSV
    if len(all_results) > 0:
        print(f"\n{'='*80}")
        print("保存结果...")
        print(f"{'='*80}")

        results_df = pd.DataFrame(all_results)
        
        # 调整列顺序
        cols_order = ['strategy_name', 'total_trades', 'cumulative_return', 'annual_return',
                     'max_drawdown', 'sharpe_ratio', 'win_rate', 'buy_hold_return',
                     'stopped_out_count', 'data_length', 'freq', 'year']
        
        # 只保留存在的列
        existing_cols = [col for col in cols_order if col in results_df.columns]
        results_df = results_df[existing_cols]

        output_file = Path("/mnt/d/quant/qlib_backtest/supertrend_enhanced_standard_params.csv")
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"   结果已保存到: {output_file}")

    # 打印汇总表
    print(f"\n{'='*80}")
    print("增强版 SuperTrend 回测结果汇总（使用标准版参数）")
    print(f"{'='*80}")

    print(f"\n{'年份':<6} {'频率':<8} {'累计收益':<10} {'年化收益':<10} {'最大回撤':<10} "
          f"{'夏普':<8} {'胜率':<8} {'交易次数':<8} {'止损次数':<8}")
    print("-"*90)

    for results in all_results:
        print(f"{results['year']:<6} {results['freq']:<8} "
              f"{results['cumulative_return']:>8.2%} {results['annual_return']:>8.2%} "
              f"{results['max_drawdown']:>8.2%} {results['sharpe_ratio']:>6.2f} "
              f"{results['win_rate']:>6.2f}% {results['total_trades']:>8} "
              f"{results['stopped_out_count']:>8}")

    print(f"\n{'='*80}")
    print("回测完成!")
    print(f"{'='*80}")

    return all_results


if __name__ == "__main__":
    results = main()
