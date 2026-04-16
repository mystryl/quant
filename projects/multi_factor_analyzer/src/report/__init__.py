"""
报告生成和可视化模块

本模块提供因子分析报告的生成和可视化功能。

主要类：
- ReportGenerator: 生成文本报告（Markdown、HTML、纯文本、JSON）
- Visualizer: 生成可视化图表（IC 时间序列、累计收益、回撤等）

使用示例：
    >>> from report import ReportGenerator, Visualizer
    >>>
    >>> # 生成报告
    >>> generator = ReportGenerator(output_dir="./reports")
    >>> reports = generator.generate_full_report(
    ...     factor_name="MA20",
    ...     metrics=metrics,
    ...     scenario_results=scenario_results
    ... )
    >>>
    >>> # 生成图表
    >>> viz = Visualizer(output_dir="./charts")
    >>> charts = viz.save_all_charts(
    ...     metrics=metrics,
    ...     scenario_results=scenario_results,
    ...     factor_name="MA20"
    ... )
"""

from .generator import ReportGenerator, generate_factor_report
from .visualizer import Visualizer, create_factor_charts

__all__ = [
    "ReportGenerator",
    "generate_factor_report",
    "Visualizer",
    "create_factor_charts",
]
