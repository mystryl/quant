#!/usr/bin/env python3
"""
简单测试脚本 - 验证风险控制系统是否工作
使用修改后的RealisticBacktestEngine
"""
import sys
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.v03_hybrid.strat_v03_hybrid import HybridMSBOBStrategy
from strategies.v03_hybrid.realistic_backtest import (
    RealisticBacktestEngine,
    TradingConfig,
    calculate_realistic_metrics,
    save_trades_to_csv
)
import pandas as pd


def test_risk_control_backtest():
    """测试风险控制系统"""
    print("=" * 80)
    print("测试风险控制系统 - 2025年15分钟周期")
    print("=" * 80)

    # 加载数据
    data_file = project_root / "data" / "RB9999_2025_15min.csv"
    df = pd.read_csv(data_file, parse_dates=['datetime']).set_index('datetime')

    print(f"\n数据加载完成:")
    print(f"  行数: {len(df)}")
    print(f"  时间范围: {df.index.min()} ~ {df.index.max()}")

    # 使用优化参数创建策略
    strategy = HybridMSBOBStrategy(
        pivot_len=7,
        msb_zscore=0.3018,
        atr_period=12,
        atr_multiplier=1.143,
        tp1_multiplier=0.300,
        tp2_multiplier=0.888,
        tp3_multiplier=1.456
    )

    # 创建回测引擎
    config = TradingConfig()
    engine = RealisticBacktestEngine(strategy, config)

    # 运行回测
    print("\n运行回测...")
    df_result = engine.run_backtest(df)

    # 计算指标
    metrics = calculate_realistic_metrics(engine, df_result, freq='15min')

    # 打印结果
    print("\n" + "=" * 80)
    print("回测结果")
    print("=" * 80)
    print(f"总交易次数: {metrics['total_trades']}")
    print(f"胜率: {metrics['win_rate']:.2f}%")
    print(f"盈亏比: {metrics['profit_loss_ratio']:.2f}")
    print(f"总收益率: {metrics['total_return']:.2%}")
    print(f"最大回撤: {metrics['max_drawdown']:.2%}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"净利润: {metrics['total_pnl']:.2f}元")
    print(f"手续费占比: {metrics['commission_ratio']:.2%}")

    # 分析质量评分
    print("\n" + "=" * 80)
    print("质量评分分析")
    print("=" * 80)

    if engine.trades:
        quality_scores = [t.quality_score for t in engine.trades if t.quality_score is not None]

        if quality_scores:
            print(f"[OK] 质量评分系统已生效！")
            print(f"  总交易数: {len(engine.trades)}")
            print(f"  有质量评分的交易: {len(quality_scores)}")
            print(f"  质量评分范围: {min(quality_scores):.1f} - {max(quality_scores):.1f}")
            print(f"  平均质量评分: {sum(quality_scores)/len(quality_scores):.1f}")
            print()
            print("质量评分区间分布:")
            print(f"  高质量(80+): {sum(1 for s in quality_scores if s >= 80)}笔")
            print(f"  中质量(65-79): {sum(1 for s in quality_scores if 65 <= s < 80)}笔")
            print(f"  低质量(50-64): {sum(1 for s in quality_scores if 50 <= s < 65)}笔")
            print(f"  极低(<50): {sum(1 for s in quality_scores if s < 50)}笔")
            print()

            # 分析不同质量分数的收益率
            quality_groups = {
                '高质量(80+)': [t for t in engine.trades if t.quality_score and t.quality_score >= 80],
                '中质量(65-79)': [t for t in engine.trades if t.quality_score and 65 <= t.quality_score < 80],
                '低质量(50-64)': [t for t in engine.trades if t.quality_score and 50 <= t.quality_score < 65],
            }

            print("不同质量级别的表现:")
            for level, trades in quality_groups.items():
                if trades:
                    avg_pnl = sum(t.pnl for t in trades) / len(trades)
                    win_rate = sum(1 for t in trades if t.pnl > 0) / len(trades) * 100
                    print(f"  {level}: {len(trades)}笔, 平均盈亏: {avg_pnl:.2f}元, 胜率: {win_rate:.1f}%")
        else:
            print("[X] 质量评分系统未生效！所有交易都没有质量评分。")

            # 检查为什么质量评分没有生效
            print("\n诊断信息:")
            if 'signal_ob_id' in df_result.columns:
                has_ob_id = df_result['signal_ob_id'].notna().sum()
                print(f"  signal_ob_id列存在，有{has_ob_id}行有值")
            else:
                print("  signal_ob_id列不存在 - 策略未添加OB ID到信号中")

            if 'signal_ema20' in df_result.columns:
                has_ema = df_result['signal_ema20'].notna().sum()
                print(f"  signal_ema20列存在，有{has_ema}行有值")
            else:
                print("  signal_ema20列不存在 - 策略未添加EMA到信号中")
    else:
        print("没有交易记录")

    # 保存交易记录
    output_dir = Path(__file__).parent / "test_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    trades_file = output_dir / "test_trades_2025_15min.csv"
    save_trades_to_csv(engine, trades_file)
    print(f"\n交易记录已保存: {trades_file}")

    print("\n" + "=" * 80)

    return metrics, engine


if __name__ == "__main__":
    metrics, engine = test_risk_control_backtest()
