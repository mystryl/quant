#!/usr/bin/env python3
"""
方案对比测试：原始方案 vs ATR分层止盈 vs 移动止盈
对比15分钟周期2023-2025年的表现
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

# 导入三个方案
from strategies.v03_hybrid.strat_v03_hybrid import HybridMSBOBStrategy
from strategies.v03_hybrid.strat_v03_atr_partial_exit import HybridATRPartialStrategy
from strategies.v03_hybrid.strat_v03_trailing_stop import HybridTrailingStopStrategy


def load_data(year, freq='15min'):
    """加载数据"""
    data_file = project_root / "data" / f"RB9999_{year}_{freq}.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")
    df = pd.read_csv(data_file, parse_dates=['datetime'])
    df = df.sort_values('datetime').set_index('datetime')
    return df


def run_backtest(strategy, df, year, strategy_name):
    """运行单个回测"""
    print(f"\n{year}年 {strategy_name}:", end=' ')

    df_sig = strategy.run_strategy(df)

    # 计算基本指标
    df_returns = df_sig[['position', 'close']].copy()
    df_returns['returns'] = df_returns['close'].pct_change()
    df_returns['strategy_returns'] = df_returns['position'].shift(1) * df_returns['returns']
    df_returns = df_returns.dropna()

    if len(df_returns) == 0:
        print("无交易")
        return None

    cumulative_return = (1 + df_returns['strategy_returns']).prod() - 1
    cum_returns = (1 + df_returns['strategy_returns']).cumprod()
    max_drawdown = ((cum_returns.expanding().max() - cum_returns) / cum_returns.expanding().max()).max()

    periods = 16 * 252  # 15分钟周期
    sharpe = (df_returns['strategy_returns'].mean() / df_returns['strategy_returns'].std() * np.sqrt(periods)
              if df_returns['strategy_returns'].std() > 0 else 0)

    # 统计交易次数
    position_changes = (df_sig['position'].diff() != 0).sum()

    print(f"交易{position_changes:3}笔  "
          f"收益{cumulative_return:6.2%}  "
          f"回撤{max_drawdown:6.2%}  "
          f"夏普{sharpe:6.2f}")

    return {
        'year': year,
        'strategy': strategy_name,
        'total_trades': position_changes,
        'cumulative_return': cumulative_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe
    }


def main():
    """主函数"""
    print("="*100)
    print("方案对比测试：原始方案 vs ATR分层止盈 vs 移动止盈")
    print("="*100)
    print("\n测试配置:")
    print("  周期: 15分钟")
    print("  年份: 2023-2025")
    print("  参数: 原始参数（pivot_len=7, msb_zscore=0.3, atr_period=14, atr_multiplier=1.0）")
    print("="*100)

    years = [2023, 2024, 2025]
    freq = '15min'

    all_results = []

    # 原始参数
    original_params = {
        'pivot_len': 7,
        'msb_zscore': 0.3,
        'atr_period': 14,
        'atr_multiplier': 1.0,
        'tp1_multiplier': 0.5,
        'tp2_multiplier': 1.0,
        'tp3_multiplier': 1.5
    }

    print(f"\n{'='*100}")
    print("方案1: 原始方案（OB宽度止盈，TP1全部平仓）")
    print(f"{'='*100}")

    for year in years:
        df = load_data(year, freq)
        strategy = HybridMSBOBStrategy(**original_params)
        result = run_backtest(strategy, df, year, "原始方案")
        if result:
            all_results.append(result)

    print(f"\n{'='*100}")
    print("方案2: ATR分层止盈（ATR倍数止盈，TP1/TP2/TP3各平1/3）")
    print(f"{'='*100}")

    for year in years:
        df = load_data(year, freq)
        strategy = HybridATRPartialStrategy(**original_params)
        result = run_backtest(strategy, df, year, "ATR分层止盈")
        if result:
            all_results.append(result)

    print(f"\n{'='*100}")
    print("方案3: 移动止盈（Trailing Stop，盈利1倍ATR后激活移动止损）")
    print(f"{'='*100}")

    for year in years:
        df = load_data(year, freq)
        strategy = HybridTrailingStopStrategy(
            pivot_len=7,
            msb_zscore=0.3,
            atr_period=14,
            atr_multiplier=1.0,
            tp1_multiplier=0.5,
            trailing_activation=1.0,
            trailing_distance=0.5
        )
        result = run_backtest(strategy, df, year, "移动止盈")
        if result:
            all_results.append(result)

    # 汇总对比
    if all_results:
        df_res = pd.DataFrame(all_results)

        print(f"\n{'='*100}")
        print("三年汇总对比 (2023-2025)")
        print(f"{'='*100}\n")

        # 按方案分组
        for strategy_name in ["原始方案", "ATR分层止盈", "移动止盈"]:
            strategy_data = df_res[df_res['strategy'] == strategy_name]
            if len(strategy_data) == 0:
                continue

            total_trades = strategy_data['total_trades'].sum()
            total_return = strategy_data['cumulative_return'].sum()
            max_dd = strategy_data['max_drawdown'].max()
            avg_sharpe = strategy_data['sharpe_ratio'].mean()

            print(f"\n{strategy_name}:")
            print(f"  总交易: {total_trades}笔")
            print(f"  三年收益: {total_return:.2%}")
            print(f"  最大回撤: {max_dd:.2%}")
            print(f"  平均夏普: {avg_sharpe:.2f}")

        # 详细对比表
        print(f"\n{'='*100}")
        print("详细对比表")
        print(f"{'='*100}\n")

        pivot_df = df_res.pivot(index='year', columns='strategy', values='cumulative_return')
        print("年度收益率对比:")
        print(pivot_df.to_string())

        print(f"\n{'='*100}")
        print("关键指标对比")
        print(f"{'='*100}\n")

        comparison_data = []
        for strategy_name in ["原始方案", "ATR分层止盈", "移动止盈"]:
            strategy_data = df_res[df_res['strategy'] == strategy_name]
            if len(strategy_data) > 0:
                comparison_data.append({
                    '方案': strategy_name,
                    '三年总收益': strategy_data['cumulative_return'].sum(),
                    '最大回撤': strategy_data['max_drawdown'].max(),
                    '平均夏普': strategy_data['sharpe_ratio'].mean(),
                    '总交易数': strategy_data['total_trades'].sum()
                })

        df_comp = pd.DataFrame(comparison_data)
        print(df_comp.to_string(index=False))

        # 保存结果
        output_file = Path(__file__).parent / "comparison_results.csv"
        df_res.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存: {output_file}")

        return df_res


if __name__ == "__main__":
    main()
