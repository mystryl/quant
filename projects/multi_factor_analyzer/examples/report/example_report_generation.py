"""
报告生成和可视化示例

本示例展示如何使用 ReportGenerator 和 Visualizer 生成因子分析报告和图表。
"""

import sys
from pathlib import Path

# 添加项目根目录到 Python 路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / 'src'))

import numpy as np
import pandas as pd

from report.generator import ReportGenerator
from report.visualizer import Visualizer


def generate_sample_data():
    """生成示例数据"""
    print("生成示例数据...")

    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')

    # 模拟 IC 序列（带趋势）
    ic_trend = np.linspace(0.02, 0.05, 366)
    ic_noise = np.random.randn(366) * 0.03
    ic_series = pd.Series(ic_trend + ic_noise, index=dates)

    # 模拟收益率序列
    returns = pd.Series(
        np.random.randn(366) * 0.015 + 0.001,
        index=dates
    )

    # 性能指标
    metrics = {
        'ic': ic_series,
        'rank_ic': pd.Series(np.random.randn(366) * 0.04 + 0.03, index=dates),
        'ic_mean': ic_series.mean(),
        'ic_std': ic_series.std(),
        'icir': ic_series.mean() / ic_series.std(),
        'rank_ic_mean': 0.035,
        'rank_ic_std': 0.045,
        'rank_icir': 0.78,
        'long_short_return': returns,
        'annual_return': 0.12,
        'sharpe_ratio': 1.95,
        'max_drawdown': -0.11,
        'win_rate': 0.58
    }

    # 策略场景结果
    scenario_results = {
        'bull': {
            'strategy_returns': pd.Series(np.random.randn(366) * 0.015 + 0.002, index=dates),
            'annual_return': 0.15,
            'sharpe_ratio': 2.2,
            'max_drawdown': -0.09,
            'win_rate': 0.60
        },
        'bear': {
            'strategy_returns': pd.Series(np.random.randn(366) * 0.015 - 0.001, index=dates),
            'annual_return': -0.03,
            'sharpe_ratio': -0.4,
            'max_drawdown': -0.18,
            'win_rate': 0.47
        },
        'long_short': {
            'strategy_returns': returns,
            'annual_return': 0.12,
            'sharpe_ratio': 1.95,
            'max_drawdown': -0.11,
            'win_rate': 0.58
        }
    }

    # 可靠性评估结果
    reliability_result = {
        'scores': {
            'ic_stability': 0.40,
            'ic_absolute': 0.19,
            'ir': 0.19,
            'long_short_return': 0.09,
            'win_rate': 0.07
        },
        'total_score': 0.94,
        'reliability': 'A+',
        'recommendation': '该因子表现优秀，建议重点使用。'
    }

    # 周期分析结果
    cycle_analysis = {
        'best_shift': 2,
        'ic_by_shift': {
            1: {'ic_mean': 0.028, 'icir': 0.52},
            2: {'ic_mean': 0.036, 'icir': 0.68},
            3: {'ic_mean': 0.032, 'icir': 0.61},
            5: {'ic_mean': 0.025, 'icir': 0.48}
        },
        'recommendation': '建议使用 T+2 对齐周期，ICIR 最高。'
    }

    return metrics, scenario_results, reliability_result, cycle_analysis


def main():
    """主函数"""
    print("=" * 60)
    print("报告生成和可视化示例")
    print("=" * 60)

    # 生成示例数据
    metrics, scenario_results, reliability_result, cycle_analysis = generate_sample_data()

    # 创建输出目录
    output_dir = Path("./output/report_example")
    report_dir = output_dir / "reports"
    chart_dir = output_dir / "charts"

    # ========== 1. 生成报告 ==========
    print("\n1. 生成报告...")
    print("-" * 60)

    generator = ReportGenerator(output_dir=str(report_dir))

    reports = generator.generate_full_report(
        factor_name="MA20 移动平均线因子",
        metrics=metrics,
        scenario_results=scenario_results,
        reliability_result=reliability_result,
        cycle_analysis=cycle_analysis,
        factor_description="基于 20 日移动平均线的趋势跟踪因子，"
                        "通过比较当前价格与 MA20 的关系来判断趋势。",
        backtest_period=('2020-01-01', '2020-12-31')
    )

    # 保存不同格式的报告
    print("保存报告...")

    md_path = generator.save_markdown(
        reports['markdown'],
        "MA20_analysis_report.md"
    )
    print(f"  ✓ Markdown: {md_path}")

    html_path = generator.save_html(
        reports['html'],
        "MA20_analysis_report.html"
    )
    print(f"  ✓ HTML: {html_path}")

    txt_path = generator.save_text(
        reports['text'],
        "MA20_analysis_report.txt"
    )
    print(f"  ✓ Text: {txt_path}")

    json_path = generator.save_json(
        reports['json'],
        "MA20_analysis_report.json"
    )
    print(f"  ✓ JSON: {json_path}")

    # ========== 2. 生成图表 ==========
    print("\n2. 生成图表...")
    print("-" * 60)

    viz = Visualizer(output_dir=str(chart_dir))

    chart_paths = viz.save_all_charts(
        metrics=metrics,
        scenario_results=scenario_results,
        factor_name="MA20"
    )

    print(f"共生成 {len(chart_paths)} 个图表:")
    for name, path in chart_paths.items():
        print(f"  ✓ {name}: {path}")

    # ========== 3. 生成单独图表示例 ==========
    print("\n3. 生成单独图表示例...")
    print("-" * 60)

    # IC 时间序列图（高分辨率）
    fig = viz.plot_ic_timeseries(
        metrics['ic'],
        factor_name="MA20",
        window=20
    )
    path = viz.save_figure(
        fig,
        "MA20_ic_timeseries_hd.png",
        dpi=300
    )
    print(f"  ✓ IC 时间序列图（高分辨率）: {path}")

    # 月度 IC 热力图
    fig = viz.plot_monthly_ic_heatmap(
        metrics['ic'],
        factor_name="MA20"
    )
    path = viz.save_figure(
        fig,
        "MA20_monthly_ic_heatmap.png"
    )
    print(f"  ✓ 月度 IC 热力图: {path}")

    # 周期对比图
    fig = viz.plot_cycle_comparison(
        cycle_analysis['ic_by_shift'],
        factor_name="MA20"
    )
    path = viz.save_figure(
        fig,
        "MA20_cycle_comparison.png"
    )
    print(f"  ✓ 周期对比图: {path}")

    # ========== 4. 打印报告摘要 ==========
    print("\n4. 报告摘要")
    print("-" * 60)

    print(f"""
因子名称: MA20 移动平均线因子
可靠性等级: {reliability_result['reliability']}
综合评分: {reliability_result['total_score']:.2f}

关键指标:
  - IC 均值:     {metrics['ic_mean']:.4f}
  - ICIR:        {metrics['icir']:.4f}
  - 年化收益:    {metrics['annual_return']:.2%}
  - 夏普比率:    {metrics['sharpe_ratio']:.2f}
  - 最大回撤:    {metrics['max_drawdown']:.2%}
  - 胜率:       {metrics['win_rate']:.2%}

策略表现:
  - 看涨策略:    {scenario_results['bull']['annual_return']:.2%} (年化)
  - 看跌策略:    {scenario_results['bear']['annual_return']:.2%} (年化)
  - 多空策略:    {scenario_results['long_short']['annual_return']:.2%} (年化)

周期分析:
  - 最佳周期:    T+{cycle_analysis['best_shift']}
  - 建议使用 T+2 对齐周期，ICIR 最高

建议: {reliability_result['recommendation']}
    """)

    # ========== 完成 ==========
    print("\n" + "=" * 60)
    print("报告生成完成!")
    print("=" * 60)
    print(f"\n所有文件已保存到: {output_dir.absolute()}")
    print(f"  - 报告: {report_dir.absolute()}")
    print(f"  - 图表: {chart_dir.absolute()}")
    print("\n建议下一步:")
    print("  1. 在浏览器中打开 HTML 报告查看完整分析")
    print("  2. 查看生成的图表，直观了解因子表现")
    print("  3. 根据建议优化因子或调整策略参数")


if __name__ == "__main__":
    main()
