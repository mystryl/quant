"""
多时间框架趋势策略 - 主脚本
三时间框架：
- 大周期（日线）：EMA60方向判断
- 中周期（1小时）：VWAP5和VWAP60金叉死叉
- 小周期（15分钟）：精确入场
"""

import sys
from pathlib import Path
import yaml
import pandas as pd
import numpy as np
from datetime import datetime

# 添加项目路径
project_root = Path(__file__).parent
sys.path.append(str(project_root))

from data.provider import DataProvider
from indicators.trend import TrendIndicator
from indicators.supertrend import SuperTrendIndicator
from indicators.signal import SignalGenerator
from backtest.engine import BacktestEngine


def load_config(config_path: str = None) -> dict:
    """
    加载配置文件

    Args:
        config_path: 配置文件路径

    Returns:
        dict: 配置字典
    """
    if config_path is None:
        config_path = project_root / 'config' / 'default.yaml'

    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)

    return config


def run_backtest(config: dict) -> dict:
    """
    运行回测

    Args:
        config: 配置字典

    Returns:
        dict: 回测结果
    """
    print("=" * 80)
    print("多时间框架趋势策略回测（三时间框架）")
    print("=" * 80)
    print(f"策略名称: 多时间框架趋势策略")
    print(f"版本: 1.1")
    print(f"执行时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    # ============== 步骤1：加载大周期数据（日线） ==============
    print("\n[步骤1] 加载大周期数据（日线）...")
    data_config = config['data']

    daily_provider = DataProvider(
        instrument=data_config['instrument'],
        start_date=data_config['start_date'],
        end_date=data_config['end_date'],
        frequency="1D"
    )

    daily_df = daily_provider.load_data()
    print(f"日线数据加载完成: {len(daily_df)} 条记录\n")

    # ============== 步骤2：计算大周期趋势（EMA60） ==============
    print("[步骤2] 计算大周期趋势（EMA60）...")
    trend_config = config['trend_daily']

    trend_indicator = TrendIndicator(
        ema_period=trend_config['ema_period'],
        direction_threshold=trend_config['direction_threshold']
    )

    daily_df = trend_indicator.calculate(daily_df)
    print(f"大周期趋势计算完成\n")

    # 显示当前趋势
    current_trend = trend_indicator.get_current_trend()
    print(f"当前大周期趋势:")
    print(f"  方向: {current_trend.direction}")
    print(f"  EMA60: {current_trend.ema_60:.2f}")
    print(f"  强度: {current_trend.strength}")
    print()

    # ============== 步骤3：加载中周期数据（1小时） ==============
    print("[步骤3] 加载中周期数据（1小时）...")
    hourly_provider = DataProvider(
        instrument=data_config['instrument'],
        start_date=data_config['start_date'],
        end_date=data_config['end_date'],
        frequency="1H"
    )

    hourly_df = hourly_provider.load_data()
    print(f"小时线数据加载完成: {len(hourly_df)} 条记录\n")

    # ============== 步骤4：计算Supertrend（用于止损） ==============
    print("[步骤4] 计算Supertrend...")
    supertrend_config = config['supertrend']

    supertrend_indicator = SuperTrendIndicator(
        atr_period=supertrend_config['atr_period'],
        atr_factor=supertrend_config['atr_factor']
    )

    hourly_df = supertrend_indicator.calculate(hourly_df)
    print(f"Supertrend计算完成\n")

    # ============== 步骤5：生成入场信号（VWAP金叉死叉） ==============
    print("[步骤5] 生成入场信号（VWAP金叉死叉）...")
    entry_config = config['entry_hourly']

    signal_generator = SignalGenerator(
        vwap_short_period=entry_config['vwap_short_period'],
        vwap_long_period=entry_config['vwap_long_period'],
        confirmation_bars=entry_config['confirmation_bars'],
        use_rsi=config['filters']['use_rsi'],
        rsi_period=config['filters']['rsi_period'],
        rsi_overbought=config['filters']['rsi_overbought'],
        rsi_oversold=config['filters']['rsi_oversold']
    )

    # 将日线趋势对齐到小时线
    hourly_df = signal_generator._align_daily_trend(hourly_df, daily_df['trend_direction'])

    # 生成信号
    signals = signal_generator.generate_signals(hourly_df, daily_df['trend_direction'])
    print(f"入场信号生成完成: {len(signals)} 个信号\n")

    # 显示信号摘要
    if signals:
        golden_crosses = [s for s in signals if s.entry_type == 'golden_cross']
        death_crosses = [s for s in signals if s.entry_type == 'death_cross']

        print("信号摘要:")
        print(f"  金叉（看涨）: {len(golden_crosses)}")
        print(f"  死叉（看跌）: {len(death_crosses)}")
        print(f"  平均质量: {np.mean([s.quality_score for s in signals]):.1f}")
        print()

    # ============== 步骤6：运行回测 ==============
    print("[步骤6] 运行回测...")
    backtest_engine = BacktestEngine(config=config)

    result = backtest_engine.run(hourly_df, signals)
    print(f"回测完成\n")

    # ============== 步骤7：显示结果 ==============
    print("=" * 80)
    print("回测结果摘要")
    print("=" * 80)
    print(f"测试合约: {data_config['instrument']}")
    print(f"测试期间: {data_config['start_date']} 到 {data_config['end_date']}")
    print(f"数据频率: 1H")
    print()

    print("【交易指标】")
    print(f"  总交易数: {result.total_trades}")
    print(f"  盈利交易: {result.winning_trades}")
    print(f"  亏损交易: {result.losing_trades}")
    print(f"  胜率: {result.win_rate:.1%}")
    print(f"  平均盈利: {result.avg_win:,.2f}")
    print(f"  平均亏损: {result.avg_loss:,.2f}")
    print(f"  盈亏比: {result.avg_win / abs(result.avg_loss) if result.avg_loss != 0 else float('inf'):.2f}")
    print(f"  盈利因子: {result.profit_factor:.2f}")
    print(f"  期望收益: {result.expectancy:,.2f}")
    print()

    print("【收益指标】")
    print(f"  初始资金: 100,000.00")
    print(f"  最终资金: {100000 + result.total_pnl:,.2f}")
    print(f"  总收益: {result.total_pnl:,.2f} ({result.total_pnl_pct:+.1%})")
    print(f"  最大回撤: {result.max_drawdown:,.2f} ({result.max_drawdown_pct:.1%})")
    print(f"  总手续费: {result.total_commission:,.2f}")
    print(f"  总滑点: {result.total_slippage:,.2f}")
    print(f"  净收益: {result.total_pnl - result.total_commission - result.total_slippage:,.2f}")
    print()

    print("【效率指标】")
    print(f"  平均持仓K线数: {np.mean([t.duration_bars for t in result.trades]) if result.trades else 0:.1f}")
    print(f"  平均最大盈利: {np.mean([t.max_profit for t in result.trades]) if result.trades else 0:,.2f}")
    print(f"  平均最大回撤: {np.mean([t.max_drawdown for t in result.trades]) if result.trades else 0:,.2f}")
    print()

    # ============== 步骤8：保存结果 ==============
    print("[步骤7] 保存结果...")
    results_dir = project_root / 'results'
    results_dir.mkdir(exist_ok=True)

    # 保存交易记录
    trades_df = pd.DataFrame([{
        'id': t.id,
        'instrument': t.instrument,
        'entry_type': t.entry_type.value,
        'entry_price': t.entry_price,
        'entry_timestamp': t.entry_timestamp,
        'entry_bar_index': t.entry_bar_index,
        'stop_loss': t.stop_loss,
        'tp1_price': t.tp1_price,
        'tp2_price': t.tp2_price,
        'tp3_price': t.tp3_price,
        'exit_price': t.exit_price,
        'exit_timestamp': t.exit_timestamp,
        'exit_bar_index': t.exit_bar_index,
        'exit_reason': t.exit_reason.value if t.exit_reason else None,
        'pnl': t.pnl,
        'pnl_pct': t.pnl_pct,
        'commission': t.commission,
        'slippage': t.slippage,
        'position_size': t.position_size,
        'duration_bars': t.duration_bars,
        'max_profit': t.max_profit,
        'max_drawdown': t.max_drawdown,
        'quality_score': t.entry_signal.quality_score
    } for t in result.trades])

    trades_file = results_dir / f'trades_{data_config["instrument"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    trades_df.to_csv(trades_file, index=False, encoding='utf-8-sig')
    print(f"交易记录已保存: {trades_file}")

    # 保存权益曲线
    equity_curve = pd.DataFrame({
        'equity': backtest_engine.equity_curve,
        'drawdown': backtest_engine.drawdown_curve
    })

    equity_file = results_dir / f'equity_{data_config["instrument"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
    equity_curve.to_csv(equity_file, index=False, encoding='utf-8-sig')
    print(f"权益曲线已保存: {equity_file}")

    # 保存回测摘要
    summary = {
        'strategy': 'multi_timeframe_trend_v2',
        'version': '1.1',
        'instrument': data_config['instrument'],
        'start_date': data_config['start_date'],
        'end_date': data_config['end_date'],
        'frequency': '1H',
        'total_trades': result.total_trades,
        'winning_trades': result.winning_trades,
        'losing_trades': result.losing_trades,
        'win_rate': result.win_rate,
        'avg_win': result.avg_win,
        'avg_loss': result.avg_loss,
        'profit_factor': result.profit_factor,
        'expectancy': result.expectancy,
        'total_pnl': result.total_pnl,
        'total_pnl_pct': result.total_pnl_pct,
        'max_drawdown': result.max_drawdown,
        'max_drawdown_pct': result.max_drawdown_pct,
        'total_commission': result.total_commission,
        'total_slippage': result.total_slippage,
        'net_pnl': result.total_pnl - result.total_commission - result.total_slippage,
        'avg_duration_bars': np.mean([t.duration_bars for t in result.trades]) if result.trades else 0
    }

    summary_file = results_dir / f'summary_{data_config["instrument"]}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.yaml'
    with open(summary_file, 'w', encoding='utf-8') as f:
        yaml.dump(summary, f, allow_unicode=True, default_flow_style=False)
    print(f"回测摘要已保存: {summary_file}")

    print()
    print("=" * 80)
    print("回测完成！")
    print("=" * 80)

    return summary


if __name__ == "__main__":
    # 加载配置
    config = load_config()

    # 运行回测
    result = run_backtest(config)
