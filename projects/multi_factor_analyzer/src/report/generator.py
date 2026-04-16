"""
报告生成器模块 (Report Generator)

该模块提供因子分析报告的生成功能，包括：
1. 文本报告（Markdown、HTML）
2. 执行摘要
3. IC 分析
4. IR 分析
5. 策略场景分析
6. 周期分析
7. 可靠性评估

设计理念：
- 生成专业、可读的分析报告
- 支持多种输出格式
- 灵活的报告模板
- 完整的性能指标展示
"""

from typing import Dict, Optional, Any
from datetime import datetime
from pathlib import Path
import json

import numpy as np
import pandas as pd

try:
    from ..core.performance_eval import PerformanceEvaluator  # noqa: F401
    from ..core.strategy_analyzer import StrategyAnalyzer  # noqa: F401
except ImportError:
    # 当直接运行时，使用绝对导入
    pass


class ReportGenerator:
    """
    报告生成器

    生成因子分析报告，包括性能指标、策略分析、可靠性评估等内容。

    支持的报告格式：
    - Markdown (.md)
    - HTML (.html)
    - 纯文本 (.txt)
    - JSON (.json)

    报告内容：
    1. 执行摘要：可靠性等级、关键指标
    2. IC 分析：IC 时间序列、IC 分布、ICIR
    3. IR 分析：多空收益、夏普比率、最大回撤
    4. 策略场景：看涨/看跌/波动率环境表现
    5. 周期分析：最佳周期、周期对齐效果
    6. 稳定性分析：时间稳定性、市场环境稳定性
    7. 建议：是否使用、如何优化

    Attributes:
        output_dir: 报告输出目录
        include_charts: 是否包含图表（HTML 格式）
        date_format: 日期格式

    Examples:
        >>> from report.generator import ReportGenerator
        >>>
        >>> # 创建报告生成器
        >>> generator = ReportGenerator(output_dir="./reports")
        >>>
        >>> # 生成完整报告
        >>> report = generator.generate_full_report(
        ...     factor_name="MA20",
        ...     metrics=metrics,
        ...     scenario_results=scenario_results,
        ...     reliability_result=reliability_result
        ... )
        >>>
        >>> # 保存为不同格式
        >>> generator.save_markdown(report, "MA20_report.md")
        >>> generator.save_html(report, "MA20_report.html")
    """

    def __init__(self, output_dir: str = "./reports", include_charts: bool = True, date_format: str = "%Y-%m-%d"):
        """
        初始化报告生成器

        Args:
            output_dir: 报告输出目录
            include_charts: 是否在 HTML 报告中包含图表
            date_format: 日期格式
        """
        self.output_dir = Path(output_dir)
        self.include_charts = include_charts
        self.date_format = date_format

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_full_report(
        self,
        factor_name: str,
        metrics: Dict[str, Any],
        scenario_results: Dict[str, Any],
        reliability_result: Optional[Dict[str, Any]] = None,
        cycle_analysis: Optional[Dict[str, Any]] = None,
        factor_description: Optional[str] = None,
        backtest_period: Optional[tuple] = None,
    ) -> Dict[str, str]:
        """
        生成完整的因子分析报告

        Args:
            factor_name: 因子名称
            metrics: 性能指标（来自 PerformanceEvaluator.calculate_all）
            scenario_results: 策略场景分析结果
            reliability_result: 可靠性评估结果（可选）
            cycle_analysis: 周期分析结果（可选）
            factor_description: 因子描述（可选）
            backtest_period: 回测时间段 (start_date, end_date)（可选）

        Returns:
            Dict: 包含不同格式报告的字典
                - markdown: Markdown 格式报告
                - html: HTML 格式报告
                - text: 纯文本格式报告
                - json: JSON 格式数据
        """
        # 生成各个部分
        summary = self._generate_summary(factor_name, metrics, reliability_result, backtest_period)

        ic_analysis = self._generate_ic_analysis(metrics)
        ir_analysis = self._generate_ir_analysis(metrics)
        scenario_analysis = self._generate_scenario_analysis(scenario_results)

        stability_analysis = self._generate_stability_analysis(metrics)

        additional_sections = ""
        if cycle_analysis:
            additional_sections += self._generate_cycle_analysis(cycle_analysis)

        if reliability_result:
            additional_sections += self._generate_reliability_section(reliability_result)

        recommendations = self._generate_recommendations(metrics, scenario_results, reliability_result)

        # 组合完整报告
        report_data = {
            "factor_name": factor_name,
            "factor_description": factor_description,
            "generation_time": datetime.now().strftime(self.date_format),
            "summary": summary,
            "ic_analysis": ic_analysis,
            "ir_analysis": ir_analysis,
            "scenario_analysis": scenario_analysis,
            "stability_analysis": stability_analysis,
            "additional_sections": additional_sections,
            "recommendations": recommendations,
        }

        # 生成不同格式
        markdown_report = self._format_markdown(report_data)
        html_report = self._format_html(report_data)
        text_report = self._format_text(report_data)
        json_report = self._format_json(report_data)

        return {
            "markdown": markdown_report,
            "html": html_report,
            "text": text_report,
            "json": json_report,
            "data": report_data,
        }

    def _generate_summary(
        self,
        factor_name: str,
        metrics: Dict[str, Any],
        reliability_result: Optional[Dict[str, Any]],
        backtest_period: Optional[tuple],
    ) -> str:
        """生成执行摘要"""
        lines = []
        lines.append("## 执行摘要")
        lines.append("")
        lines.append(f"**因子名称**: {factor_name}")
        lines.append("")

        if backtest_period:
            start_date, end_date = backtest_period
            lines.append(f"**回测期间**: {start_date} 至 {end_date}")
            lines.append("")

        # 可靠性等级
        if reliability_result:
            reliability = reliability_result.get("reliability", "N/A")
            total_score = reliability_result.get("total_score", 0)
            lines.append(f"**可靠性等级**: {reliability}")
            lines.append(f"**综合评分**: {total_score:.2f} / 1.00")
            lines.append("")

        # 关键指标
        lines.append("### 关键指标")
        lines.append("")
        lines.append("| 指标 | 数值 | 评价 |")
        lines.append("|------|------|------|")

        # IC 均值
        ic_mean = metrics.get("ic_mean", 0)
        ic_eval = self._evaluate_ic(ic_mean)
        lines.append(f"| IC 均值 | {ic_mean:.4f} | {ic_eval} |")

        # ICIR
        icir = metrics.get("icir", 0)
        icir_eval = self._evaluate_icir(icir)
        lines.append(f"| ICIR | {icir:.4f} | {icir_eval} |")

        # 年化收益
        annual_return = metrics.get("annual_return", 0)
        return_eval = self._evaluate_return(annual_return)
        lines.append(f"| 年化收益 | {annual_return:.2%} | {return_eval} |")

        # 夏普比率
        sharpe = metrics.get("sharpe_ratio", 0)
        sharpe_eval = self._evaluate_sharpe(sharpe)
        lines.append(f"| 夏普比率 | {sharpe:.2f} | {sharpe_eval} |")

        # 最大回撤
        max_dd = metrics.get("max_drawdown", 0)
        dd_eval = self._evaluate_drawdown(max_dd)
        lines.append(f"| 最大回撤 | {max_dd:.2%} | {dd_eval} |")

        lines.append("")

        # 总体评价
        if reliability_result:
            recommendation = reliability_result.get("recommendation", "")
            lines.append("### 总体评价")
            lines.append("")
            lines.append(recommendation)
            lines.append("")

        return "\n".join(lines)

    def _generate_ic_analysis(self, metrics: Dict[str, Any]) -> str:
        """生成 IC 分析部分"""
        lines = []
        lines.append("## IC 分析")
        lines.append("")
        lines.append("IC (Information Coefficient) 衡量因子值与未来收益率的相关性，")
        lines.append("是评估因子预测能力的核心指标。")
        lines.append("")

        # IC 统计
        ic_mean = metrics.get("ic_mean", 0)
        ic_std = metrics.get("ic_std", 0)
        icir = metrics.get("icir", 0)

        lines.append("### IC 统计指标")
        lines.append("")
        lines.append(f"- **IC 均值**: {ic_mean:.4f}")
        lines.append(f"  - 衡量因子的平均预测能力")
        lines.append(f"  - 评价: {self._evaluate_ic(ic_mean)}")
        lines.append("")
        lines.append(f"- **IC 标准差**: {ic_std:.4f}")
        lines.append(f"  - 衡量 IC 的波动性")
        lines.append(f"  - 越小表示 IC 越稳定")
        lines.append("")
        lines.append(f"- **ICIR**: {icir:.4f}")
        lines.append(f"  - IC 信息比率 = IC 均值 / IC 标准差")
        lines.append(f"  - 评价: {self._evaluate_icir(icir)}")
        lines.append("")

        # Rank IC
        rank_ic_mean = metrics.get("rank_ic_mean", 0)
        rank_ic_std = metrics.get("rank_ic_std", 0)
        rank_icir = metrics.get("rank_icir", 0)

        lines.append("### Rank IC 统计指标")
        lines.append("")
        lines.append(f"- **Rank IC 均值**: {rank_ic_mean:.4f}")
        lines.append(f"  - 衡量因子排名与收益率排名的相关性")
        lines.append(f"  - 对异常值不敏感，更稳健")
        lines.append("")
        lines.append(f"- **Rank IC 标准差**: {rank_ic_std:.4f}")
        lines.append("")
        lines.append(f"- **Rank ICIR**: {rank_icir:.4f}")
        lines.append("")

        # IC 时间序列分析（如果有 IC 序列）
        ic_series = metrics.get("ic")
        if ic_series is not None and len(ic_series) > 0:
            lines.append("### IC 时间序列特征")
            lines.append("")

            # 计算 IC 的正负比例
            positive_ic_ratio = (ic_series > 0).mean()
            lines.append(f"- **IC 为正的比例**: {positive_ic_ratio:.2%}")
            lines.append("  - 衡量因子的方向稳定性")
            lines.append("")

            # IC 的滚动统计
            if len(ic_series) >= 20:
                rolling_mean = ic_series.rolling(window=20).mean()
                rolling_std = ic_series.rolling(window=20).std()

                lines.append(f"- **最近 20 期 IC 均值**: {rolling_mean.iloc[-1]:.4f}")
                lines.append(f"- **最近 20 期 IC 标准差**: {rolling_std.iloc[-1]:.4f}")
                lines.append("")

        return "\n".join(lines)

    def _generate_ir_analysis(self, metrics: Dict[str, Any]) -> str:
        """生成 IR 分析部分"""
        lines = []
        lines.append("## IR 分析")
        lines.append("")
        lines.append("IR (Information Ratio) 综合考虑收益和风险，衡量因子的实际交易效果。")
        lines.append("")

        # 收益指标
        annual_return = metrics.get("annual_return", 0)
        total_return = (1 + metrics.get("long_short_return", pd.Series([0]))).prod() - 1
        if isinstance(total_return, pd.Series):
            total_return = total_return.iloc[0] if len(total_return) > 0 else 0

        lines.append("### 收益指标")
        lines.append("")
        lines.append(f"- **年化收益率**: {annual_return:.2%}")
        lines.append(f"  - 评价: {self._evaluate_return(annual_return)}")
        lines.append("")
        lines.append(f"- **累计收益率**: {total_return:.2%}")
        lines.append("")

        # 风险调整收益
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown", 0)
        win_rate = metrics.get("win_rate", 0)

        lines.append("### 风险调整收益指标")
        lines.append("")
        lines.append(f"- **夏普比率**: {sharpe:.2f}")
        lines.append(f"  - 评价: {self._evaluate_sharpe(sharpe)}")
        lines.append(f"  - 衡量单位风险的超额收益")
        lines.append("")
        lines.append(f"- **最大回撤**: {max_dd:.2%}")
        lines.append(f"  - 评价: {self._evaluate_drawdown(max_dd)}")
        lines.append(f"  - 衡量最大损失幅度")
        lines.append("")
        lines.append(f"- **卡玛比率**: {annual_return / abs(max_dd) if max_dd != 0 else 0:.2f}")
        lines.append(f"  - 年化收益 / 最大回撤")
        lines.append("")

        # 交易指标
        lines.append("### 交易指标")
        lines.append("")
        lines.append(f"- **胜率**: {win_rate:.2%}")
        lines.append(f"  - 评价: {self._evaluate_win_rate(win_rate)}")
        lines.append(f"  - 盈利天数占总交易天数的比例")
        lines.append("")

        return "\n".join(lines)

    def _generate_scenario_analysis(self, scenario_results: Dict[str, Any]) -> str:
        """生成策略场景分析部分"""
        lines = []
        lines.append("## 策略场景分析")
        lines.append("")
        lines.append("不同市场环境下的因子表现分析。")
        lines.append("")

        # 看涨策略
        if "bull" in scenario_results:
            bull = scenario_results["bull"]
            lines.append("### 看涨策略")
            lines.append("")
            lines.append("选取因子值最高的股票做多。")
            lines.append("")

            lines.append(f"- **年化收益率**: {bull.get('annual_return', 0):.2%}")
            lines.append(f"- **夏普比率**: {bull.get('sharpe_ratio', 0):.2f}")
            lines.append(f"- **最大回撤**: {bull.get('max_drawdown', 0):.2%}")
            lines.append(f"- **胜率**: {bull.get('win_rate', 0):.2%}")
            lines.append("")

        # 看跌策略
        if "bear" in scenario_results:
            bear = scenario_results["bear"]
            lines.append("### 看跌策略")
            lines.append("")
            lines.append("选取因子值最低的股票做多（反向策略）。")
            lines.append("")

            lines.append(f"- **年化收益率**: {bear.get('annual_return', 0):.2%}")
            lines.append(f"- **夏普比率**: {bear.get('sharpe_ratio', 0):.2f}")
            lines.append(f"- **最大回撤**: {bear.get('max_drawdown', 0):.2%}")
            lines.append(f"- **胜率**: {bear.get('win_rate', 0):.2%}")
            lines.append("")

        # 多空策略
        if "long_short" in scenario_results:
            ls = scenario_results["long_short"]
            lines.append("### 多空策略")
            lines.append("")
            lines.append("买入因子值最高的股票，卖出因子值最低的股票。")
            lines.append("")

            lines.append(f"- **年化收益率**: {ls.get('annual_return', 0):.2%}")
            lines.append(f"- **夏普比率**: {ls.get('sharpe_ratio', 0):.2f}")
            lines.append(f"- **最大回撤**: {ls.get('max_drawdown', 0):.2%}")
            lines.append(f"- **胜率**: {ls.get('win_rate', 0):.2%}")
            lines.append("")

        # 波动率策略
        if "volatility" in scenario_results:
            vol = scenario_results["volatility"]
            lines.append("### 波动率策略")
            lines.append("")
            lines.append("根据市场波动率动态调整仓位。")
            lines.append("")

            lines.append(f"- **年化收益率**: {vol.get('annual_return', 0):.2%}")
            lines.append(f"- **夏普比率**: {vol.get('sharpe_ratio', 0):.2f}")
            lines.append(f"- **最大回撤**: {vol.get('max_drawdown', 0):.2%}")
            lines.append(f"- **胜率**: {vol.get('win_rate', 0):.2%}")
            lines.append("")

        return "\n".join(lines)

    def _generate_stability_analysis(self, metrics: Dict[str, Any]) -> str:
        """生成稳定性分析部分"""
        lines = []
        lines.append("## 稳定性分析")
        lines.append("")
        lines.append("因子性能的时间稳定性评估。")
        lines.append("")

        ic_series = metrics.get("ic")
        if ic_series is not None and len(ic_series) > 0:
            # IC 标准差
            ic_std = metrics.get("ic_std", 0)
            lines.append(f"- **IC 标准差**: {ic_std:.4f}")
            lines.append("  - 衡量 IC 的波动性")
            lines.append(f"  - 评价: {self._evaluate_ic_stability(ic_std)}")
            lines.append("")

            # ICIR
            icir = metrics.get("icir", 0)
            lines.append(f"- **ICIR**: {icir:.4f}")
            lines.append(f"  - IC 信息比率，综合衡量预测能力和稳定性")
            lines.append(f"  - 评价: {self._evaluate_icir(icir)}")
            lines.append("")

            # IC 趋势分析
            if len(ic_series) >= 40:
                # 将 IC 序列分为前后两半，比较均值
                mid_point = len(ic_series) // 2
                ic_first_half = ic_series.iloc[:mid_point].mean()
                ic_second_half = ic_series.iloc[mid_point:].mean()
                ic_change = ic_second_half - ic_first_half

                lines.append("- **IC 时间趋势**")
                lines.append(f"  - 前半段 IC 均值: {ic_first_half:.4f}")
                lines.append(f"  - 后半段 IC 均值: {ic_second_half:.4f}")
                lines.append(f"  - 变化: {ic_change:+.4f}")

                if abs(ic_change) > 0.01:
                    trend = "下降" if ic_change < 0 else "上升"
                    lines.append(f"  - 评价: IC 呈现{trend}趋势，稳定性需要关注")
                else:
                    lines.append(f"  - 评价: IC 保持稳定")
                lines.append("")

        return "\n".join(lines)

    def _generate_cycle_analysis(self, cycle_analysis: Dict[str, Any]) -> str:
        """生成周期分析部分"""
        lines = []
        lines.append("## 周期分析")
        lines.append("")
        lines.append("因子在不同周期对齐方式下的表现。")
        lines.append("")

        if "best_shift" in cycle_analysis:
            best_shift = cycle_analysis["best_shift"]
            lines.append(f"- **最佳对齐周期**: T+{best_shift}")
            lines.append("")

        if "ic_by_shift" in cycle_analysis:
            ic_by_shift = cycle_analysis["ic_by_shift"]
            lines.append("### 不同周期的 IC 值")
            lines.append("")
            lines.append("| 周期 | IC 均值 | ICIR |")
            lines.append("|------|---------|------|")

            for shift, ic_data in ic_by_shift.items():
                ic_mean = ic_data.get("ic_mean", 0)
                icir = ic_data.get("icir", 0)
                lines.append(f"| T+{shift} | {ic_mean:.4f} | {icir:.4f} |")

            lines.append("")

        if "recommendation" in cycle_analysis:
            lines.append("### 周期建议")
            lines.append("")
            lines.append(cycle_analysis["recommendation"])
            lines.append("")

        return "\n".join(lines)

    def _generate_reliability_section(self, reliability_result: Dict[str, Any]) -> str:
        """生成可靠性评估部分"""
        lines = []
        lines.append("## 可靠性评估")
        lines.append("")

        reliability = reliability_result.get("reliability", "N/A")
        total_score = reliability_result.get("total_score", 0)

        lines.append(f"**可靠性等级**: {reliability}")
        lines.append(f"**综合评分**: {total_score:.2f} / 1.00")
        lines.append("")

        # 各项得分
        scores = reliability_result.get("scores", {})
        if scores:
            lines.append("### 各项得分")
            lines.append("")
            lines.append("| 评估维度 | 得分 | 权重 | 评价 |")
            lines.append("|----------|------|------|------|")

            for dimension, score in scores.items():
                # 将维度名转换为中文
                dimension_cn = self._translate_dimension(dimension)
                evaluation = "优秀" if score >= 0.8 else "良好" if score >= 0.6 else "一般"
                lines.append(f"| {dimension_cn} | {score:.2%} | - | {evaluation} |")

            lines.append("")

        # 建议
        recommendation = reliability_result.get("recommendation", "")
        if recommendation:
            lines.append("### 建议")
            lines.append("")
            lines.append(recommendation)
            lines.append("")

        return "\n".join(lines)

    def _generate_recommendations(
        self, metrics: Dict[str, Any], scenario_results: Dict[str, Any], reliability_result: Optional[Dict[str, Any]]
    ) -> str:
        """生成建议部分"""
        lines = []
        lines.append("## 建议")
        lines.append("")

        # 总体建议
        ic_mean = abs(metrics.get("ic_mean", 0))
        icir = metrics.get("icir", 0)
        annual_return = metrics.get("annual_return", 0)
        sharpe = metrics.get("sharpe_ratio", 0)

        if ic_mean > 0.05 and icir > 0.5 and annual_return > 0.05 and sharpe > 1.5:
            lines.append("### 总体建议")
            lines.append("")
            lines.append("该因子表现优秀，建议重点使用。")
            lines.append("")
            lines.append("### 使用建议")
            lines.append("")
            lines.append("- 可作为核心因子用于多因子模型")
            lines.append("- 建议与其他低相关性因子组合")
            lines.append("- 可适当提高该因子的权重")
            lines.append("")

        elif ic_mean > 0.03 and icir > 0.3 and annual_return > 0.02:
            lines.append("### 总体建议")
            lines.append("")
            lines.append("该因子表现良好，建议使用。")
            lines.append("")
            lines.append("### 使用建议")
            lines.append("")
            lines.append("- 可作为辅助因子用于多因子模型")
            lines.append("- 建议与其他因子组合使用")
            lines.append("- 注意监控因子性能变化")
            lines.append("")

        elif ic_mean > 0.02 and icir > 0.2:
            lines.append("### 总体建议")
            lines.append("")
            lines.append("该因子表现一般，建议谨慎使用或考虑优化。")
            lines.append("")
            lines.append("### 优化建议")
            lines.append("")
            lines.append("- 考虑因子标准化或中性化处理")
            lines.append("- 尝试不同的周期对齐方式")
            lines.append("- 结合其他指标进行筛选")
            lines.append("")

        else:
            lines.append("### 总体建议")
            lines.append("")
            lines.append("该因子表现较差，不建议使用，建议重新设计。")
            lines.append("")
            lines.append("### 改进建议")
            lines.append("")
            lines.append("- 重新审视因子逻辑和计算方法")
            lines.append("- 检查是否存在未来函数")
            lines.append("- 考虑因子组合或变换")
            lines.append("")

        # 风险提示
        max_dd = metrics.get("max_drawdown", 0)
        if max_dd > 0.2:
            lines.append("### 风险提示")
            lines.append("")
            lines.append(f"- 最大回撤较大 ({max_dd:.2%})，建议加强风险控制")
            lines.append("- 建议设置止损位")
            lines.append("- 考虑动态调整仓位")
            lines.append("")

        return "\n".join(lines)

    def _format_markdown(self, report_data: Dict[str, str]) -> str:
        """格式化为 Markdown"""
        sections = []

        # 标题
        sections.append(f"# {report_data['factor_name']} 因子分析报告")
        sections.append("")
        sections.append(f"**生成时间**: {report_data['generation_time']}")
        sections.append("")

        if report_data.get("factor_description"):
            sections.append(f"**因子描述**: {report_data['factor_description']}")
            sections.append("")

        # 各个部分
        sections.append(report_data["summary"])
        sections.append(report_data["ic_analysis"])
        sections.append(report_data["ir_analysis"])
        sections.append(report_data["scenario_analysis"])
        sections.append(report_data["stability_analysis"])
        sections.append(report_data["additional_sections"])
        sections.append(report_data["recommendations"])

        return "\n".join(sections)

    def _format_html(self, report_data: Dict[str, str]) -> str:
        """格式化为 HTML"""
        html_template = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{factor_name} 因子分析报告</title>
    <style>
        body {{
            font-family: "Microsoft YaHei", Arial, sans-serif;
            line-height: 1.6;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-bottom: 2px solid #ecf0f1;
            padding-bottom: 8px;
            margin-top: 30px;
        }}
        h3 {{
            color: #7f8c8d;
            margin-top: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 20px 0;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 12px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            min-width: 200px;
        }}
        .metric-label {{
            font-weight: bold;
            color: #7f8c8d;
        }}
        .metric-value {{
            font-size: 24px;
            color: #2c3e50;
            margin: 5px 0;
        }}
        .grade-excellent {{ color: #27ae60; }}
        .grade-good {{ color: #2980b9; }}
        .grade-fair {{ color: #f39c12; }}
        .grade-poor {{ color: #c0392b; }}
        .summary {{
            background-color: #ecf0f1;
            padding: 20px;
            border-radius: 5px;
            margin: 20px 0;
        }}
        .recommendation {{
            background-color: #d5f4e6;
            border-left: 4px solid #27ae60;
            padding: 15px;
            margin: 20px 0;
        }}
        .warning {{
            background-color: #ffeaa7;
            border-left: 4px solid #fdcb6e;
            padding: 15px;
            margin: 20px 0;
        }}
        .footer {{
            margin-top: 40px;
            padding-top: 20px;
            border-top: 1px solid #ecf0f1;
            text-align: center;
            color: #7f8c8d;
            font-size: 0.9em;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{factor_name} 因子分析报告</h1>
        <p><strong>生成时间</strong>: {generation_time}</p>
        {factor_description}

        {content}

        <div class="footer">
            <p>本报告由多因子量化分析系统自动生成</p>
        </div>
    </div>
</body>
</html>"""

        # 将 Markdown 转换为 HTML（简化版本）
        # 这里可以使用 markdown 库，但为了保持依赖简单，我们做简单转换
        content = self._markdown_to_html(self._format_markdown(report_data))

        factor_desc = ""
        if report_data.get("factor_description"):
            factor_desc = f'<p><strong>因子描述</strong>: {report_data["factor_description"]}</p>'

        return html_template.format(
            factor_name=report_data["factor_name"],
            generation_time=report_data["generation_time"],
            factor_description=factor_desc,
            content=content,
        )

    def _format_text(self, report_data: Dict[str, str]) -> str:
        """格式化为纯文本"""
        markdown_text = self._format_markdown(report_data)

        # 移除 Markdown 格式
        text = markdown_text
        text = text.replace("#", "")
        text = text.replace("*", "")
        text = text.replace("|", " ")
        text = text.replace("-", " ")

        # 清理多余空行
        lines = text.split("\n")
        cleaned_lines = []
        prev_empty = False

        for line in lines:
            line = line.strip()
            if line:
                cleaned_lines.append(line)
                prev_empty = False
            elif not prev_empty:
                cleaned_lines.append("")
                prev_empty = True

        return "\n".join(cleaned_lines)

    def _format_json(self, report_data: Dict[str, str]) -> str:
        """格式化为 JSON"""
        # 只返回数据部分，不包含格式化的文本
        data = {
            "factor_name": report_data["factor_name"],
            "generation_time": report_data["generation_time"],
            "factor_description": report_data.get("factor_description"),
        }
        return json.dumps(data, ensure_ascii=False, indent=2)

    def _markdown_to_html(self, markdown_text: str) -> str:
        """简单的 Markdown 到 HTML 转换"""
        html = markdown_text

        # 标题
        html = html.replace("### ", "<h3>").replace("\n", "</h3>\n", 1)
        html = html.replace("## ", "<h2>").replace("\n", "</h2>\n", 1)
        html = html.replace("# ", "<h1>").replace("\n", "</h1>\n", 1)

        # 粗体
        html = html.replace("**", "<strong>").replace("**", "</strong>")

        # 表格（简化处理）
        # 这里需要更复杂的处理，暂时跳过

        # 列表
        html = html.replace("- ", "<li>")

        # 段落
        lines = html.split("\n")
        html_lines = []
        in_list = False

        for line in lines:
            line = line.strip()
            if not line:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append("<p></p>")
            elif line.startswith("<li>"):
                if not in_list:
                    html_lines.append("<ul>")
                    in_list = True
                html_lines.append(line)
            else:
                if in_list:
                    html_lines.append("</ul>")
                    in_list = False
                html_lines.append(f"<p>{line}</p>")

        if in_list:
            html_lines.append("</ul>")

        return "\n".join(html_lines)

    def save_markdown(self, content: str, filename: str) -> Path:
        """保存 Markdown 报告"""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

    def save_html(self, content: str, filename: str) -> Path:
        """保存 HTML 报告"""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

    def save_text(self, content: str, filename: str) -> Path:
        """保存纯文本报告"""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

    def save_json(self, content: str, filename: str) -> Path:
        """保存 JSON 报告"""
        filepath = self.output_dir / filename
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        return filepath

    # ========== 评估辅助方法 ==========

    def _evaluate_ic(self, ic_mean: float) -> str:
        """评估 IC 均值"""
        abs_ic = abs(ic_mean)
        if abs_ic > 0.05:
            return "优秀"
        elif abs_ic > 0.03:
            return "良好"
        elif abs_ic > 0.02:
            return "一般"
        else:
            return "较差"

    def _evaluate_icir(self, icir: float) -> str:
        """评估 ICIR"""
        abs_icir = abs(icir)
        if abs_icir > 0.5:
            return "优秀"
        elif abs_icir > 0.3:
            return "良好"
        elif abs_icir > 0.2:
            return "一般"
        else:
            return "较差"

    def _evaluate_return(self, annual_return: float) -> str:
        """评估年化收益"""
        if annual_return > 0.10:
            return "优秀"
        elif annual_return > 0.05:
            return "良好"
        elif annual_return > 0.02:
            return "一般"
        else:
            return "较差"

    def _evaluate_sharpe(self, sharpe: float) -> str:
        """评估夏普比率"""
        if sharpe > 2.0:
            return "优秀"
        elif sharpe > 1.5:
            return "良好"
        elif sharpe > 1.0:
            return "一般"
        else:
            return "较差"

    def _evaluate_drawdown(self, max_dd: float) -> str:
        """评估最大回撤"""
        abs_dd = abs(max_dd)
        if abs_dd < 0.10:
            return "优秀"
        elif abs_dd < 0.15:
            return "良好"
        elif abs_dd < 0.25:
            return "一般"
        else:
            return "较差"

    def _evaluate_win_rate(self, win_rate: float) -> str:
        """评估胜率"""
        if win_rate > 0.55:
            return "优秀"
        elif win_rate > 0.52:
            return "良好"
        elif win_rate > 0.50:
            return "一般"
        else:
            return "较差"

    def _evaluate_ic_stability(self, ic_std: float) -> str:
        """评估 IC 稳定性"""
        if ic_std < 0.05:
            return "优秀"
        elif ic_std < 0.08:
            return "良好"
        elif ic_std < 0.10:
            return "一般"
        else:
            return "较差"

    def _translate_dimension(self, dimension: str) -> str:
        """将评估维度翻译为中文"""
        translations = {
            "ic_stability": "IC 稳定性",
            "ic_absolute": "IC 绝对值",
            "ir": "信息比率",
            "long_short_return": "多空收益",
            "win_rate": "胜率",
        }
        return translations.get(dimension, dimension)


# 便捷函数
def generate_factor_report(
    factor_name: str,
    metrics: Dict[str, Any],
    scenario_results: Dict[str, Any],
    reliability_result: Optional[Dict[str, Any]] = None,
    output_dir: str = "./reports",
    **kwargs,
) -> Dict[str, str]:
    """
    生成因子分析报告的便捷函数

    Args:
        factor_name: 因子名称
        metrics: 性能指标
        scenario_results: 策略场景分析结果
        reliability_result: 可靠性评估结果（可选）
        output_dir: 输出目录
        **kwargs: 其他参数传递给 ReportGenerator

    Returns:
        Dict: 包含不同格式报告的字典

    Examples:
        >>> from report.generator import generate_factor_report
        >>>
        >>> # 快速生成报告
        >>> reports = generate_factor_report(
        ...     factor_name="MA20",
        ...     metrics=metrics,
        ...     scenario_results=scenario_results
        ... )
        >>>
        >>> # 保存报告
        >>> generator = ReportGenerator()
        >>> generator.save_markdown(reports['markdown'], "MA20_report.md")
    """
    generator = ReportGenerator(output_dir=output_dir, **kwargs)
    return generator.generate_full_report(
        factor_name=factor_name,
        metrics=metrics,
        scenario_results=scenario_results,
        reliability_result=reliability_result,
    )


if __name__ == "__main__":
    # 示例：使用报告生成器
    print("=" * 60)
    print("报告生成器示例")
    print("=" * 60)

    # 创建模拟数据
    print("\n生成模拟数据...")
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")

    # 模拟性能指标
    metrics = {
        "ic": pd.Series(np.random.randn(100) * 0.05 + 0.03, index=dates[:100]),
        "rank_ic": pd.Series(np.random.randn(100) * 0.06 + 0.04, index=dates[:100]),
        "ic_mean": 0.035,
        "ic_std": 0.052,
        "icir": 0.673,
        "rank_ic_mean": 0.042,
        "rank_ic_std": 0.061,
        "rank_icir": 0.689,
        "long_short_return": pd.Series(np.random.randn(100) * 0.01 + 0.001, index=dates[:100]),
        "annual_return": 0.085,
        "sharpe_ratio": 1.85,
        "max_drawdown": -0.12,
        "win_rate": 0.56,
    }

    # 模拟策略场景结果
    scenario_results = {
        "bull": {"annual_return": 0.12, "sharpe_ratio": 2.1, "max_drawdown": -0.10, "win_rate": 0.58},
        "bear": {"annual_return": -0.02, "sharpe_ratio": -0.5, "max_drawdown": -0.15, "win_rate": 0.48},
        "long_short": {"annual_return": 0.085, "sharpe_ratio": 1.85, "max_drawdown": -0.12, "win_rate": 0.56},
    }

    # 模拟可靠性评估结果
    reliability_result = {
        "scores": {"ic_stability": 0.40, "ic_absolute": 0.18, "ir": 0.18, "long_short_return": 0.08, "win_rate": 0.06},
        "total_score": 0.90,
        "reliability": "A+",
        "recommendation": "该因子表现优秀，建议重点使用。",
    }

    # 创建报告生成器
    generator = ReportGenerator(output_dir="./reports")

    # 生成报告
    print("\n生成报告...")
    reports = generator.generate_full_report(
        factor_name="MA20 移动平均线因子",
        metrics=metrics,
        scenario_results=scenario_results,
        reliability_result=reliability_result,
        factor_description="基于 20 日移动平均线的趋势因子",
        backtest_period=("2020-01-01", "2020-12-31"),
    )

    # 保存报告
    print("\n保存报告...")
    md_path = generator.save_markdown(reports["markdown"], "MA20_report.md")
    html_path = generator.save_html(reports["html"], "MA20_report.html")
    txt_path = generator.save_text(reports["text"], "MA20_report.txt")
    json_path = generator.save_json(reports["json"], "MA20_report.json")

    print(f"\n报告已保存到:")
    print(f"  - Markdown: {md_path}")
    print(f"  - HTML: {html_path}")
    print(f"  - Text: {txt_path}")
    print(f"  - JSON: {json_path}")

    # 打印 Markdown 报告预览
    print("\n" + "=" * 60)
    print("Markdown 报告预览")
    print("=" * 60)
    print(reports["markdown"][:1000] + "...")
