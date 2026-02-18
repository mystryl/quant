#!/usr/bin/env python3
"""
多频率回测脚本：支持 1min, 5min, 15min, 60min 等不同频率的回测
"""
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

# 配置路径
DATA_DIR = Path("/mnt/d/quant/qlib_data")
RESAMPLED_DIR = Path("/mnt/d/quant/qlib_backtest/qlib_data_multi_freq")

# 字段映射
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

def load_data(freq, start_date="2023-01-01", end_date="2023-12-31"):
    """
    加载指定频率的数据

    Args:
        freq: 频率（1min, 5min, 15min, 60min）
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        DataFrame: 合并后的数据
    """
    print(f"\n{'='*60}")
    print(f"加载 {freq} 数据")
    print(f"{'='*60}")

    instrument = "RB9999.XSGE"

    if freq == "1min":
        # 1分钟数据从原始目录读取
        data_dir = DATA_DIR / "instruments" / instrument
    else:
        # 其他频率从重采样目录读取
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
            data[field] = df.iloc[:, 0]  # 获取第一列数据
            print(f"  {field}: {len(df)} 行")

    # 合并为单个 DataFrame
    df = pd.DataFrame(data)
    df = df.sort_index()

    # 限制时间范围
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    print(f"\n数据形状: {df.shape}")
    print(f"时间范围: {df.index.min()} ~ {df.index.max()}")

    return df


def run_backtest(df, freq, window=20, std_multiplier=2):
    """
    运行布林带均值回归策略

    Args:
        df: 数据 DataFrame
        freq: 数据频率（1min, 5min, 15min, 60min）
        window: 布林带窗口
        std_multiplier: 标准差倍数

    Returns:
        DataFrame: 包含策略结果的数据
    """
    print(f"\n{'='*60}")
    print(f"运行布林带策略（窗口={window}, 标准差倍数={std_multiplier}）")
    print(f"{'='*60}")

    # 计算技术指标
    df['ma'] = df['close'].rolling(window=window).mean()
    df['std'] = df['close'].rolling(window=window).std()
    df['upper_band'] = df['ma'] + std_multiplier * df['std']
    df['lower_band'] = df['ma'] - std_multiplier * df['std']

    # 生成交易信号
    df['position'] = 0
    df.loc[df['close'] < df['lower_band'], 'position'] = 1  # 做多
    df.loc[df['close'] > df['upper_band'], 'position'] = -1  # 做空

    # 移除前 window 行
    df_strategy = df.iloc[window:].copy()

    # 计算收益
    df_strategy['returns'] = df_strategy['close'].pct_change()
    df_strategy['strategy_returns'] = df_strategy['position'].shift(1) * df_strategy['returns']
    df_strategy['buy_hold_returns'] = df_strategy['returns']

    # 统计结果
    total_trades = (df_strategy['position'].diff() != 0).sum()
    total_return = df_strategy['strategy_returns'].sum()
    cumulative_return = (1 + df_strategy['strategy_returns']).prod() - 1

    # 计算最大回撤
    cum_returns = (1 + df_strategy['strategy_returns']).cumprod()
    running_max = cum_returns.expanding().max()
    max_drawdown = (running_max - cum_returns) / running_max
    max_drawdown = max_drawdown.max()

    # 胜率
    win_rate = (df_strategy['strategy_returns'] > 0).sum() / (df_strategy['strategy_returns'] != 0).sum() * 100

    # 年化指标
    if freq == "1min":
        annual_trading_periods = 240 * 252  # 每天240个1分钟，252天
    elif freq == "5min":
        annual_trading_periods = 48 * 252   # 每天48个5分钟
    elif freq == "15min":
        annual_trading_periods = 16 * 252   # 每天16个15分钟
    elif freq == "60min":
        annual_trading_periods = 4 * 252    # 每天4个小时
    else:
        annual_trading_periods = 252        # 每天1个日线

    annual_return = (1 + cumulative_return) ** (annual_trading_periods / len(df_strategy)) - 1
    sharpe_ratio = df_strategy['strategy_returns'].mean() / df_strategy['strategy_returns'].std() * np.sqrt(annual_trading_periods)

    # 买入持有基准
    buy_hold_cumulative = (1 + df_strategy['buy_hold_returns']).prod() - 1

    results = {
        'total_trades': total_trades,
        'cumulative_return': cumulative_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'buy_hold_return': buy_hold_cumulative,
        'df_strategy': df_strategy
    }

    print(f"\n{'='*60}")
    print(f"回测结果")
    print(f"{'='*60}")
    print(f"总交易次数: {total_trades}")
    print(f"累计收益: {cumulative_return:.2%}")
    print(f"年化收益: {annual_return:.2%}")
    print(f"最大回撤: {max_drawdown:.2%}")
    print(f"夏普比率: {sharpe_ratio:.2f}")
    print(f"胜率: {win_rate:.2f}%")
    print(f"买入持有收益: {buy_hold_cumulative:.2%}")

    return results


def compare_frequencies():
    """
    比较不同频率的回测结果
    """
    print("\n" + "="*60)
    print("多频率回测对比")
    print("="*60)

    frequencies = ["1min", "5min", "15min", "60min"]
    all_results = {}

    for freq in frequencies:
        try:
            # 加载数据
            df = load_data(freq)

            # 运行回测
            results = run_backtest(df, freq)
            all_results[freq] = results

        except Exception as e:
            print(f"\n{freq} 回测失败: {e}")
            import traceback
            traceback.print_exc()
            continue

    # 汇总结果
    print(f"\n{'='*60}")
    print(f"汇总结果对比")
    print(f"{'='*60}")

    print(f"\n{'频率':<10} {'交易次数':<10} {'累计收益':<12} {'年化收益':<12} {'最大回撤':<12} {'夏普比率':<10} {'胜率':<10}")
    print("-"*80)

    for freq, results in all_results.items():
        print(f"{freq:<10} {results['total_trades']:<10} {results['cumulative_return']:>10.2%} "
              f"{results['annual_return']:>10.2%} {results['max_drawdown']:>10.2%} "
              f"{results['sharpe_ratio']:>9.2f} {results['win_rate']:>9.2f}%")

    # 绘制对比图
    try:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('不同频率回测结果对比', fontsize=16, fontweight='bold')

        # 累计收益对比
        ax1 = axes[0, 0]
        freqs = list(all_results.keys())
        cumulative_returns = [all_results[f]['cumulative_return'] for f in freqs]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
        bars = ax1.bar(freqs, cumulative_returns, color=colors)
        ax1.set_ylabel('累计收益')
        ax1.set_title('累计收益对比')
        ax1.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax1.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, cumulative_returns):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.01 if val >= 0 else -0.05),
                    f'{val:.1%}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

        # 年化收益对比
        ax2 = axes[0, 1]
        annual_returns = [all_results[f]['annual_return'] for f in freqs]
        bars = ax2.bar(freqs, annual_returns, color=colors)
        ax2.set_ylabel('年化收益')
        ax2.set_title('年化收益对比')
        ax2.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax2.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, annual_returns):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + (0.05 if val >= 0 else -0.05),
                    f'{val:.1%}', ha='center', va='bottom' if val >= 0 else 'top', fontsize=9)

        # 最大回撤对比
        ax3 = axes[1, 0]
        max_drawdowns = [all_results[f]['max_drawdown'] for f in freqs]
        bars = ax3.bar(freqs, max_drawdowns, color=colors)
        ax3.set_ylabel('最大回撤')
        ax3.set_title('最大回撤对比（越小越好）')
        ax3.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, max_drawdowns):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                    f'{val:.2%}', ha='center', va='bottom', fontsize=9)

        # 夏普比率对比
        ax4 = axes[1, 1]
        sharpe_ratios = [all_results[f]['sharpe_ratio'] for f in freqs]
        bars = ax4.bar(freqs, sharpe_ratios, color=colors)
        ax4.set_ylabel('夏普比率')
        ax4.set_title('夏普比率对比')
        ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.5)
        ax4.grid(axis='y', alpha=0.3)
        for bar, val in zip(bars, sharpe_ratios):
            ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.3,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plot_file = Path("/mnt/d/quant/qlib_backtest/multi_freq_comparison.png")
        plt.savefig(plot_file, dpi=150, bbox_inches='tight')
        print(f"\n对比图已保存到: {plot_file}")

    except Exception as e:
        print(f"\n生成对比图失败: {e}")
        print("提示: 安装 matplotlib 可以生成对比图")
        print("  pip install matplotlib")

    return all_results


if __name__ == "__main__":
    # 运行多频率回测对比
    results = compare_frequencies()

    # 保存详细结果
    for freq, result in results.items():
        output_file = Path(f"/mnt/d/quant/qlib_backtest/backtest_{freq}.csv")
        result['df_strategy'].to_csv(output_file)
        print(f"{freq} 详细数据已保存到: {output_file}")

    print(f"\n{'='*60}")
    print("回测完成！")
    print(f"{'='*60}")
