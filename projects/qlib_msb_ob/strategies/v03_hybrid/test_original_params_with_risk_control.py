#!/usr/bin/env python3
"""
对比测试：原始参数 + 风险控制 vs 原始参数 + 固定仓位
"""
import sys
from pathlib import Path
import pandas as pd

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.v03_hybrid.strat_v03_hybrid import HybridMSBOBStrategy
from strategies.v03_hybrid.realistic_backtest import (
    RealisticBacktestEngine,
    TradingConfig,
    calculate_realistic_metrics,
)

# 原始参数
ORIGINAL_PARAMS = {
    'pivot_len': 7,
    'msb_zscore': 0.3,
    'atr_period': 14,
    'atr_multiplier': 1.0,
    'tp1_multiplier': 0.5,
    'tp2_multiplier': 1.0,
    'tp3_multiplier': 1.5
}


def run_backtest_with_risk_control(year, freq):
    """使用风险控制的回测"""
    data_file = project_root / "data" / f"RB9999_{year}_{freq}.csv"
    df = pd.read_csv(data_file, parse_dates=['datetime']).set_index('datetime')

    strategy = HybridMSBOBStrategy(**ORIGINAL_PARAMS)
    config = TradingConfig()
    engine = RealisticBacktestEngine(strategy, config)

    df_result = engine.run_backtest(df)
    metrics = calculate_realistic_metrics(engine, df_result, freq=freq)

    return engine, metrics


def run_backtest_fixed_position(year, freq):
    """使用固定仓位的回测（模拟原始行为）"""
    data_file = project_root / "data" / f"RB9999_{year}_{freq}.csv"
    df = pd.read_csv(data_file, parse_dates=['datetime']).set_index('datetime')

    strategy = HybridMSBOBStrategy(**ORIGINAL_PARAMS)
    config = TradingConfig()

    engine = RealisticBacktestEngine(strategy, config)

    # 修改run_backtest，强制使用固定仓位（不传入quality_score）
    df_sig = strategy.run_strategy(df)

    # 初始化
    engine.total_capital = config.INITIAL_CAPITAL
    engine.available_cash = config.INITIAL_CAPITAL
    engine.margin_used = 0.0
    engine.position_value = 0.0
    engine.trades = []
    engine.snapshots = []
    engine.current_position = None

    # 逐K线处理，但不传递quality_score
    for i in range(len(df_sig)):
        row = df_sig.iloc[i]
        timestamp = df_sig.index[i]

        # 平仓检查
        if engine.current_position is not None:
            pos = engine.current_position
            should_close = False
            close_reason = None

            if pos['direction'] == 'long':
                if row['low'] <= pos['stop_loss']:
                    should_close = True
                    close_reason = 'stop_loss'
                elif row['high'] >= pos['take_profits']['tp1']:
                    should_close = True
                    close_reason = 'tp1'
            else:
                if row['high'] >= pos['stop_loss']:
                    should_close = True
                    close_reason = 'stop_loss'
                elif row['low'] <= pos['take_profits']['tp1']:
                    should_close = True
                    close_reason = 'tp1'

            if should_close:
                close_price = pos['stop_loss'] if close_reason == 'stop_loss' else pos['take_profits']['tp1']
                engine.close_position(timestamp, close_price, close_reason)

        # 开仓检查（不传递quality_score，使用固定仓位）
        if engine.current_position is None and row['position'] != 0:
            direction = 'long' if row['position'] > 0 else 'short'
            stop_loss = row['stop_loss']
            tp1 = row['take_profit']
            take_profits = {'tp1': tp1, 'tp2': tp1, 'tp3': tp1}

            # 不传递quality_score参数，使用原始的calculate_position_size
            engine.open_position(
                timestamp=timestamp,
                price=row['close'],
                direction=direction,
                stop_loss=stop_loss,
                take_profits=take_profits
                # 注意：不传递quality_score
            )

        engine.take_snapshot(timestamp, row['close'])

    # 强制平仓
    if engine.current_position is not None:
        last_row = df_sig.iloc[-1]
        engine.close_position(df_sig.index[-1], last_row['close'], 'force_close')

    # 计算指标
    metrics = calculate_realistic_metrics(engine, df_sig, freq=freq)

    return engine, metrics


def main():
    print("=" * 100)
    print("原始参数对比测试：风险控制 vs 固定仓位")
    print("=" * 100)
    print("\n参数配置:")
    for key, value in ORIGINAL_PARAMS.items():
        print(f"  {key}: {value}")

    years = [2023, 2024, 2025]
    freq = '15min'

    all_results = []

    for year in years:
        print(f"\n{'='*100}")
        print(f"{year}年 {freq}周期")
        print(f"{'='*100}")

        # 1. 固定仓位（基线）
        print(f"\n[固定仓位]")
        engine_fixed, metrics_fixed = run_backtest_fixed_position(year, freq)
        print(f"  交易: {metrics_fixed['total_trades']}笔  "
              f"胜率: {metrics_fixed['win_rate']:.1f}%  "
              f"收益: {metrics_fixed['total_return']:.2%}  "
              f"回撤: {metrics_fixed['max_drawdown']:.2%}  "
              f"夏普: {metrics_fixed['sharpe_ratio']:.2f}")

        # 2. 风险控制
        print(f"\n[风险控制]")
        engine_rc, metrics_rc = run_backtest_with_risk_control(year, freq)
        print(f"  交易: {metrics_rc['total_trades']}笔  "
              f"胜率: {metrics_rc['win_rate']:.1f}%  "
              f"收益: {metrics_rc['total_return']:.2%}  "
              f"回撤: {metrics_rc['max_drawdown']:.2%}  "
              f"夏普: {metrics_rc['sharpe_ratio']:.2f}")

        # 分析质量评分
        if engine_rc.trades:
            quality_scores = [t.quality_score for t in engine_rc.trades if t.quality_score is not None]
            if quality_scores:
                print(f"\n[质量评分分析]")
                print(f"  有评分的交易: {len(quality_scores)}笔")
                print(f"  评分范围: {min(quality_scores):.1f} - {max(quality_scores):.1f}")
                print(f"  平均评分: {sum(quality_scores)/len(quality_scores):.1f}")
                print(f"  高质量(80+): {sum(1 for s in quality_scores if s >= 80)}笔")
                print(f"  中质量(65-79): {sum(1 for s in quality_scores if 65 <= s < 80)}笔")
                print(f"  低质量(50-64): {sum(1 for s in quality_scores if 50 <= s < 65)}笔")
                print(f"  被过滤(<50): {sum(1 for s in quality_scores if s < 50)}笔")

        # 对比
        print(f"\n[对比]")
        trade_reduction = ((metrics_fixed['total_trades'] - metrics_rc['total_trades']) /
                          metrics_fixed['total_trades'] * 100) if metrics_fixed['total_trades'] > 0 else 0
        return_improvement = (metrics_rc['total_return'] - metrics_fixed['total_return'])

        print(f"  交易次数变化: {metrics_fixed['total_trades']} -> {metrics_rc['total_trades']} "
              f"({trade_reduction:+.1f}%)")
        print(f"  收益率变化: {metrics_fixed['total_return']:.2%} -> {metrics_rc['total_return']:.2%} "
              f"({return_improvement:+.2%})")

        metrics_fixed['method'] = 'fixed'
        metrics_rc['method'] = 'risk_control'
        all_results.append(metrics_fixed)
        all_results.append(metrics_rc)

    # 三年汇总
    print(f"\n{'='*100}")
    print("三年汇总对比 (2023-2025)")
    print(f"{'='*100}\n")

    fixed_results = [r for r in all_results if r['method'] == 'fixed']
    rc_results = [r for r in all_results if r['method'] == 'risk_control']

    def aggregate(results):
        return {
            'total_trades': sum(r['total_trades'] for r in results),
            'win_rate': sum(r['win_rate'] * r['total_trades'] for r in results) /
                       sum(r['total_trades'] for r in results),
            'total_return': sum(r['total_return'] for r in results),
            'max_drawdown': max(r['max_drawdown'] for r in results),
            'sharpe_ratio': sum(r['sharpe_ratio'] for r in results) / len(results),
        }

    fixed_agg = aggregate(fixed_results)
    rc_agg = aggregate(rc_results)

    print(f"{'指标':<20} {'固定仓位':<20} {'风险控制':<20} {'改进幅度'}")
    print("-" * 100)

    metrics_list = [
        ('总交易次数', 'total_trades', '笔', False),
        ('加权胜率', 'win_rate', '%', True),
        ('三年总收益', 'total_return', '%', True),
        ('最大回撤', 'max_drawdown', '%', False),
        ('平均夏普', 'sharpe_ratio', '', True),
    ]

    for name, key, unit, higher_better in metrics_list:
        fixed_val = fixed_agg[key]
        rc_val = rc_agg[key]

        if unit == '%':
            fixed_str = f"{fixed_val:.2f}%"
            rc_str = f"{rc_val:.2f}%"
        elif unit == '笔':
            fixed_str = f"{int(fixed_val)}笔"
            rc_str = f"{int(rc_val)}笔"
        else:
            fixed_str = f"{fixed_val:.2f}{unit}" if unit != '' else f"{fixed_val:.2f}"
            rc_str = f"{rc_val:.2f}{unit}" if unit != '' else f"{rc_val:.2f}"

        if higher_better:
            improvement = ((rc_val - fixed_val) / abs(fixed_val) * 100) if fixed_val != 0 else 0
        else:
            improvement = ((fixed_val - rc_val) / fixed_val * 100) if fixed_val != 0 else 0

        imp_str = f"({improvement:+.1f}%)"
        print(f"{name:<20} {fixed_str:<20} {rc_str:<20} {imp_str}")

    print(f"\n{'='*100}")
    print("结论：")
    if rc_agg['total_return'] > fixed_agg['total_return']:
        print("✓ 风险控制系统改善了收益率")
    else:
        print("✗ 风险控制系统未能改善收益率")

    if rc_agg['total_trades'] < fixed_agg['total_trades']:
        print("✓ 风险控制系统减少了交易次数（过滤低质量信号）")
    else:
        print("✗ 风险控制系统未减少交易次数")

    if rc_agg['max_drawdown'] < fixed_agg['max_drawdown']:
        print("✓ 风险控制系统降低了最大回撤")
    else:
        print("✗ 风险控制系统未能降低回撤")

    print(f"{'='*100}\n")


if __name__ == "__main__":
    main()
