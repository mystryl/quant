"""
使用 AKQuant 回测 SuperTrend 策略。

回测 RB9999 2024 年数据（15分钟和60分钟级别），并生成可视化报告。
"""

import pandas as pd
from pathlib import Path
import sys
import os

# 添加项目路径到 sys.path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

from akquant import run_backtest, BacktestConfig
from supertrend_strategy import SuperTrendStrategy


def run_backtest_for_frequency(
    data_file: str,
    freq_name: str,
    period: int = 10,
    multiplier: float = 3.0,
    initial_cash: float = 100_000.0
):
    """
    对指定频率的数据运行回测。

    Args:
        data_file: 数据文件路径
        freq_name: 频率名称（如 '15min'）
        period: SuperTrend ATR 周期
        multiplier: SuperTrend ATR 倍数
        initial_cash: 初始资金
    """
    print(f"\n{'='*60}")
    print(f"回测: RB9999 2024 - {freq_name}")
    print(f"参数: period={period}, multiplier={multiplier}")
    print(f"{'='*60}\n")

    # 读取数据
    df = pd.read_csv(data_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    print(f"数据加载完成: {len(df)} 行")

    # 准备 AKQuant 数据格式
    data_dict = {df['symbol'].iloc[0]: df}

    # 创建策略实例
    strategy = SuperTrendStrategy(period=period, multiplier=multiplier)

    # 运行回测
    result = run_backtest(
        data=data_dict,
        strategy=strategy,
        symbol='RB9999.XSGE',
        initial_cash=initial_cash,
        commission_rate=0.0001,  # 万一佣金
        min_commission=5.0,
        lot_size=1,
        show_progress=True
    )

    # 打印核心指标
    print("\n" + "="*60)
    print("回测结果摘要")
    print("="*60)
    metrics = result.metrics
    print(f"总收益率:    {metrics.total_return_pct:.2f}%")
    print(f"年化收益率:  {metrics.annualized_return:.2f}%")
    print(f"夏普比率:    {metrics.sharpe_ratio:.2f}")
    print(f"最大回撤:    {metrics.max_drawdown_pct:.2f}%")
    print(f"胜率:        {metrics.win_rate:.2f}%")
    print(f"交易次数:    {len(result.trades_df)}")
    print(f"最终权益:    {result.metrics.end_market_value:,.2f}")

    # 生成报告
    report_file = project_dir / "reports" / f"RB9999_supertrend_{freq_name}_p{period}_m{multiplier}.html"
    report_file.parent.mkdir(parents=True, exist_ok=True)

    result.report(
        title=f"SuperTrend 策略回测报告 - RB9999 2024 ({freq_name})",
        filename=str(report_file),
        show=False
    )

    print(f"\n报告已保存: {report_file}")

    return result


def main():
    """主函数：运行不同频率和参数组合的回测。"""

    # 配置参数
    data_dir = project_dir / "data"

    # 频率配置
    freq_configs = [
        {'file': data_dir / 'RB9999_2024_15min.csv', 'name': '15min'},
        {'file': data_dir / 'RB9999_2024_60min.csv', 'name': '60min'}
    ]

    # 测试参数组合（基于之前的优化结果）
    param_configs = [
        {'period': 7, 'multiplier': 3.0, 'name': '(7, 3.0)'},
        {'period': 10, 'multiplier': 3.0, 'name': '(10, 3.0)'},
        {'period': 14, 'multiplier': 2.5, 'name': '(14, 2.5)'}
    ]

    # 运行回测
    results = []
    for freq_config in freq_configs:
        if not freq_config['file'].exists():
            print(f"警告: 数据文件不存在: {freq_config['file']}")
            continue

        for param_config in param_configs:
            result = run_backtest_for_frequency(
                data_file=str(freq_config['file']),
                freq_name=freq_config['name'],
                period=param_config['period'],
                multiplier=param_config['multiplier']
            )
            results.append({
                'freq': freq_config['name'],
                'params': param_config['name'],
                'total_return': result.metrics.total_return_pct,
                'sharpe': result.metrics.sharpe_ratio,
                'max_dd': result.metrics.max_drawdown_pct,
                'win_rate': result.metrics.win_rate,
                'trades': len(result.trades_df)
            })

    # 生成对比表
    print("\n" + "="*60)
    print("回测结果对比")
    print("="*60)
    df_results = pd.DataFrame(results)
    df_results = df_results.sort_values('total_return', ascending=False)
    print(df_results.to_string(index=False))

    # 保存对比表
    summary_file = project_dir / "reports" / "backtest_summary.csv"
    df_results.to_csv(summary_file, index=False)
    print(f"\n对比表已保存: {summary_file}")

    print("\n" + "="*60)
    print("所有回测完成！")
    print("="*60)


if __name__ == '__main__':
    main()
