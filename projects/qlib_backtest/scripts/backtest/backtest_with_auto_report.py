#!/usr/bin/env python3
"""
带自动报告生成的回测脚本模板

使用方法：
1. 在此脚本中实现你的回测逻辑
2. 调用 generate_report() 函数自动生成报告
3. 报告会自动保存到 backtest_reports/ 目录

示例：
```python
# 你的回测逻辑
results = run_backtest(...)
params = {'period': 50, 'multiplier': 20}

# 自动生成报告
generate_report("MyStrategy", params, results, results_df=df)
```
"""

import sys
from pathlib import Path

# 添加脚本目录到路径
script_dir = Path(__file__).parent
sys.path.insert(0, str(script_dir.parent))

# 导入报告生成器
from other.report_generator import create_report


def generate_report(strategy_name, params, results, results_df=None,
                    data_config=None, backtest_config=None,
                    benchmark_results=None, source_file=None,
                    charts=None, base_dir=None):
    """
    生成回测报告（便捷函数）

    Args:
        strategy_name: 策略名称
        params: 参数字典
        results: 回测结果字典，必须包含：
            - strategy_name: 策略名称
            - total_trades: 总交易次数
            - cumulative_return: 累计收益
            - annual_return: 年化收益
            - max_drawdown: 最大回撤
            - sharpe_ratio: 夏普比率
            - win_rate: 胜率
            - buy_hold_return: 买入持有收益
            - stopped_out_count: 止损次数（可选）
        results_df: 回测结果DataFrame（可选）
        data_config: 数据配置字典，例如：
            {
                '数据来源': 'Qlib数据',
                '频率': '15min',
                '年份': '2023',
                '合约': 'RB9999.XSGE'
            }
        backtest_config: 回测配置字典，例如：
            {
                '初始资金': '1,000,000 CNY',
                '交易手续费': '0',
                '滑点': '0'
            }
        benchmark_results: 基准结果字典（可选），用于对比
        source_file: 源代码文件路径（可选），会自动复制到报告目录
        charts: 图表字典，例如：
            {
                'equity_curve.png': fig1,
                'drawdown_chart.png': fig2
            }
        base_dir: 报告根目录（可选），默认为 backtest_reports/

    Returns:
        report_dir: 报告目录路径

    示例：
    ```python
    # 基本用法
    report_dir = generate_report(
        strategy_name="SuperTrend_SF14Re",
        params={'period': 50, 'multiplier': 20, 'n': 3},
        results=results
    )

    # 完整用法
    report_dir = generate_report(
        strategy_name="SuperTrend_SF14Re",
        params={'period': 50, 'multiplier': 20, 'n': 3},
        results=results,
        results_df=df_with_signals,
        data_config={
            '数据来源': 'Qlib数据',
            '频率': '15min',
            '年份': '2023, 2024, 2025',
            '合约': 'RB9999.XSGE',
            '数据长度': '18,592 根K线'
        },
        backtest_config={
            '初始资金': '1,000,000 CNY',
            '交易手续费': '万分之一',
            '滑点': '1跳'
        },
        source_file=__file__,
        charts={
            'equity_curve.png': equity_fig,
            'drawdown_chart.png': drawdown_fig
        }
    )
    ```
    """
    report_dir = create_report(
        strategy_name=strategy_name,
        params=params,
        results=results,
        results_df=results_df,
        data_config=data_config,
        backtest_config=backtest_config,
        benchmark_results=benchmark_results,
        source_file=source_file,
        charts=charts,
        base_dir=base_dir
    )

    print(f"\n{'='*60}")
    print("✅ 回测报告已生成")
    print(f"{'='*60}")
    print(f"报告目录: {report_dir}")
    print(f"\n目录结构:")
    print(f"  {report_dir}/")
    print(f"    README.md      - 参数和配置说明")
    print(f"    SUMMARY.md     - 结果摘要和结论")
    print(f"    results/")
    print(f"      *.csv       - 回测结果CSV")
    print(f"      metrics.json - 性能指标JSON")
    print(f"    code/")
    print(f"      *.py        - 源代码备份")
    print(f"    charts/")
    print(f"      *.png       - 图表文件")

    return report_dir


# 示例：修改现有的回测脚本以使用自动报告生成

def example_usage():
    """示例：如何在回测脚本中使用报告生成器"""

    # 1. 运行你的回测逻辑
    # results = run_backtest(...)
    # params = {'period': 50, 'multiplier': 20, 'n': 3}

    # 2. 准备配置信息
    data_config = {
        '数据来源': 'Qlib数据',
        '频率': '15min',
        '年份': '2023',
        '合约': 'RB9999.XSGE',
        '数据长度': '6,480 根K线'
    }

    backtest_config = {
        '初始资金': '1,000,000 CNY',
        '交易手续费': '万分之一',
        '滑点': '1跳'
    }

    # 3. 生成报告
    # report_dir = generate_report(
    #     strategy_name="SuperTrend_SF14Re",
    #     params=params,
    #     results=results,
    #     results_df=df_with_signals,
    #     data_config=data_config,
    #     backtest_config=backtest_config,
    #     source_file=__file__
    # )

    print("示例代码，请根据你的回测逻辑修改此文件。")


if __name__ == "__main__":
    example_usage()
