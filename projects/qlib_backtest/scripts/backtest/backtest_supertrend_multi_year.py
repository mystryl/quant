#!/usr/bin/env python3
"""
SuperTrend 多年份多周期回测

对 2023、2024、2025 年的螺纹钢数据，分别在 15 分钟和 60 分钟频率上回测
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.append('/mnt/d/quant/qlib_backtest')

from qlib_supertrend import SupertrendStrategy, atr, supertrend, supertrend_signals


# ============================================
# 数据加载
# ============================================

def load_data(freq="15min", start_date="2023-01-01", end_date="2023-12-31"):
    """
    加载 qlib 数据

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

    # 检查数据目录是否存在
    if not data_dir.exists():
        print(f"⚠️  数据目录不存在: {data_dir}")
        return None

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

    if not data:
        print(f"⚠️  没有找到任何数据文件")
        return None

    df = pd.DataFrame(data)
    df = df.sort_index()

    # 过滤日期范围
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    if len(df) == 0:
        print(f"⚠️  指定日期范围内没有数据: {start_date} ~ {end_date}")
        return None

    return df


# ============================================
# 回测函数
# ============================================

def run_backtest(df, strategy_name):
    """
    运行回测

    Args:
        df: 包含 position 列的 DataFrame
        strategy_name: 策略名称

    Returns:
        回测结果字典
    """
    df = df.copy()

    # 计算收益
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['buy_hold_returns'] = df['returns']

    # 移除NaN
    df = df.dropna(subset=['position', 'strategy_returns'])

    if len(df) == 0:
        return None

    # 统计结果
    total_trades = (df['position'].diff() != 0).sum()
    cumulative_return = (1 + df['strategy_returns']).prod() - 1
    buy_hold_cumulative = (1 + df['buy_hold_returns']).prod() - 1

    # 计算最大回撤
    cum_returns = (1 + df['strategy_returns']).cumprod()
    running_max = cum_returns.expanding().max()
    max_drawdown = (running_max - cum_returns) / running_max
    max_drawdown = max_drawdown.max()

    # 胜率
    strategy_returns_nonzero = df['strategy_returns'][df['strategy_returns'] != 0]
    win_rate = (strategy_returns_nonzero > 0).sum() / len(strategy_returns_nonzero) * 100 if len(strategy_returns_nonzero) > 0 else 0

    # 年化指标
    trading_periods_per_year = {
        '1min': 240 * 252,
        '5min': 48 * 252,
        '15min': 16 * 252,
        '60min': 4 * 252
    }

    annual_trading_periods = trading_periods_per_year.get('15min', 16 * 252)
    annual_return = (1 + cumulative_return) ** (annual_trading_periods / len(df)) - 1
    sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(annual_trading_periods) if df['strategy_returns'].std() > 0 else 0

    results = {
        'strategy_name': strategy_name,
        'total_trades': int(total_trades),
        'cumulative_return': cumulative_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'buy_hold_return': buy_hold_cumulative,
        'data_length': len(df)
    }

    return results


# ============================================
# 主程序
# ============================================

def main():
    """主程序"""
    print("="*80)
    print("SuperTrend 多年份多周期回测")
    print("="*80)

    # 定义年份和周期
    years = [2023, 2024, 2025]
    freqs = ['15min', '60min']

    # 测试参数组合
    params_list = [
        (10, 3.0),
        (10, 2.0),
        (7, 3.0),
        (14, 2.5),
    ]

    # 存储所有结果
    all_results = []

    # 遍历年份
    for year in years:
        print(f"\n{'='*80}")
        print(f"年份: {year}")
        print(f"{'='*80}")

        # 遍历周期
        for freq in freqs:
            print(f"\n{'-'*80}")
            print(f"周期: {freq}")
            print(f"{'-'*80}")

            # 加载数据
            start_date = f"{year}-01-01"
            end_date = f"{year}-12-31"

            df = load_data(freq=freq, start_date=start_date, end_date=end_date)

            if df is None or len(df) == 0:
                print(f"⚠️  {year}年 {freq} 数据不可用，跳过")
                continue

            print(f"数据加载: {len(df)} 行")
            print(f"时间范围: {df.index.min()} ~ {df.index.max()}")

            # 遍历参数组合
            for period, multiplier in params_list:
                print(f"\n  参数组合: period={period}, multiplier={multiplier}")

                # 创建策略
                strategy = SupertrendStrategy(period=period, multiplier=multiplier)

                # 生成信号
                df_strategy = strategy.generate_signal(df.copy())

                # 移除前几行（用于计算指标）
                df_strategy = df_strategy.dropna(subset=['position'])

                if len(df_strategy) == 0:
                    print(f"    ⚠️  无有效数据")
                    continue

                # 运行回测
                results = run_backtest(df_strategy, strategy.name)

                if results is None:
                    print(f"    ⚠️  回测失败")
                    continue

                # 添加年份和周期信息
                results['year'] = year
                results['freq'] = freq
                results['period'] = period
                results['multiplier'] = multiplier

                all_results.append(results)

                # 统计买卖信号
                buy_count = df_strategy['buy_signal'].sum()
                sell_count = df_strategy['sell_signal'].sum()

                print(f"    交易次数: {results['total_trades']}")
                print(f"    累计收益: {results['cumulative_return']:.2%}")
                print(f"    年化收益: {results['annual_return']:.2%}")
                print(f"    最大回撤: {results['max_drawdown']:.2%}")
                print(f"    夏普比率: {results['sharpe_ratio']:.2f}")
                print(f"    胜率: {results['win_rate']:.2f}%")
                print(f"    买入信号: {int(buy_count)}, 卖出信号: {int(sell_count)}")

    # 汇总结果
    if all_results:
        print(f"\n{'='*80}")
        print("汇总结果")
        print(f"{'='*80}")

        # 按年份和周期分组
        for year in years:
            for freq in freqs:
                year_freq_results = [r for r in all_results if r['year'] == year and r['freq'] == freq]

                if not year_freq_results:
                    continue

                print(f"\n{'='*80}")
                print(f"{year}年 - {freq}")
                print(f"{'='*80}")

                print(f"\n{'参数':<20} {'交易':<6} {'累计':<10} {'年化':<10} {'回撤':<10} {'夏普':<8} {'胜率':<8}")
                print("-"*70)

                for results in year_freq_results:
                    param_str = f"({results['period']}, {results['multiplier']})"
                    print(f"{param_str:<20} {results['total_trades']:<6} "
                          f"{results['cumulative_return']:>8.2%} {results['annual_return']:>8.2%} "
                          f"{results['max_drawdown']:>8.2%} {results['sharpe_ratio']:>6.2f} {results['win_rate']:>6.2f}%")

        # 保存结果到 CSV
        df_results = pd.DataFrame(all_results)
        output_file = Path("/mnt/d/quant/qlib_backtest/supertrend_backtest_results.csv")
        df_results.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n✅ 结果已保存到: {output_file}")

        return all_results
    else:
        print("\n⚠️  没有生成任何回测结果")
        return []


if __name__ == "__main__":
    results = main()
