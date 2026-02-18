#!/usr/bin/env python3
"""
V01: 原版Pine策略
完全按照Pine源码逻辑实现的MSB+OB策略

2023-2025年 60分钟线回测
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategy.msb_ob_strategy import MSBOBStrategy


def _identify_trades(df):
    """识别每笔完整交易"""
    trades = []
    in_position = False
    entry_idx = None
    entry_price = None
    pos_type = None

    for i in range(len(df)):
        if not in_position and df['position'].iloc[i] != 0:
            # 开仓
            in_position = True
            entry_idx = i
            entry_price = df['close'].iloc[i]
            pos_type = 'long' if df['position'].iloc[i] > 0 else 'short'
        elif in_position and df['position'].iloc[i] == 0:
            # 平仓
            exit_price = df['close'].iloc[i]
            if pos_type == 'long':
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price
            trades.append({'entry': entry_idx, 'exit': i, 'pnl': pnl, 'type': pos_type})
            in_position = False

    return trades


def calculate_metrics(df, freq='60min'):
    """计算回测指标"""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df = df.dropna(subset=['position', 'strategy_returns'])

    if len(df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_loss_ratio': 0,
            'cumulative_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }

    # 计算累计收益
    cumulative_return = (1 + df['strategy_returns']).prod() - 1

    # 计算最大回撤
    cum_returns = (1 + df['strategy_returns']).cumprod()
    max_drawdown = ((cum_returns.expanding().max() - cum_returns) / cum_returns.expanding().max()).max()

    # 计算夏普比率
    trading_periods = {'5min': 48*252, '15min': 16*252, '60min': 4*252}
    periods = trading_periods.get(freq, 4*252)
    sharpe = (df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(periods)
              if df['strategy_returns'].std() > 0 else 0)

    # 修正：基于完整交易计算胜率
    trades = _identify_trades(df)
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    losing_trades = sum(1 for t in trades if t['pnl'] < 0)
    total_trades = len(trades)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    # 计算平均盈利和亏损（基于完整交易）
    winning_pnls = [t['pnl'] for t in trades if t['pnl'] > 0]
    losing_pnls = [t['pnl'] for t in trades if t['pnl'] < 0]
    avg_win = np.mean(winning_pnls) if winning_pnls else 0
    avg_loss = np.mean(losing_pnls) if losing_pnls else 0

    # 盈亏比
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'cumulative_return': cumulative_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe
    }


def calculate_ob_stats(df):
    """计算OB统计"""
    stats = {
        'total_ob_created': df['total_ob'].max() if 'total_ob' in df.columns else 0,
        'total_ob_mitigated': 0,
        'ob_reliability': 0,
        'hp_ob_count': 0
    }

    if 'mitigated_ob_count' in df.columns:
        stats['total_ob_mitigated'] = df['mitigated_ob_count'].max()

    if stats['total_ob_created'] > 0:
        stats['ob_reliability'] = (stats['total_ob_mitigated'] / stats['total_ob_created']) * 100

    if 'hpz_count' in df.columns:
        stats['hp_ob_count'] = df['hpz_count'].max()

    return stats


def load_data(year, freq='60min'):
    """加载指定年份的数据"""
    data_file = project_root / "data" / f"RB9999_{year}_{freq}.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    df = pd.read_csv(data_file, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def run_backtest_for_year(year, freq='60min'):
    """对单年数据运行回测"""
    print(f"\n{'='*80}")
    print(f"回测年份: {year} ({freq})")
    print(f"{'='*80}")

    # 加载数据
    df = load_data(year, freq)
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

    print(f"\n策略参数: MSB+OB (原版Pine)")

    # 生成信号
    print("\n生成交易信号...")
    df_signals = strategy.generate_signal(df)

    # 计算指标
    print("计算回测指标...")
    metrics = calculate_metrics(df_signals, freq=freq)
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

    return {**metrics, **ob_stats, 'year': year, 'freq': freq}


def main():
    """主程序"""
    print("="*80)
    print("V01: 原版Pine策略 - MSB+OB (2023-2025)")
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
        output_file = Path(__file__).parent / "report_v01_original_pine.csv"
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
