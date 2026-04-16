"""
可靠性评估器 (Reliability Evaluator)

该模块实现了因子可靠性评估功能，综合多个维度评估因子的质量和可靠性。

核心功能：
1. 可配置的权重系统（默认/保守/激进）
2. 综合评估因子可靠性（IC、IR、多空收益、胜率等）
3. 提供详细的评分和建议
4. 支持多种评估场景

设计理念：
- 因子可靠性评估是多维度的，需要综合考虑多个指标
- 不同策略类型可能需要不同的评估重点
- 权重配置应该基于理论依据和实践经验
- 评估结果应该易于理解和应用

参考文献：
1. Grinold & Kahn, "Active Portfolio Management" - IC 和 IR 的理论基础
2. Sharpe, "Information Ratio and Performance" - 信息比率的应用
3. Jacobs & Levy, "Long-Short Equity Strategies" - 多空收益的重要性
4. Kaufman, "Trading Systems and Methods" - 胜率对交易心理的影响
"""

from typing import Dict, Optional, Tuple, Union

import pandas as pd

from .config import (
    RELIABILITY_GRADES,
    get_weights,
    get_thresholds,
    get_reliability_grade,
    validate_weights,
)


class ReliabilityEvaluator:
    """
    可靠性评估器

    综合评估因子的可靠性，考虑多个维度的性能指标。

    评估维度：
    1. IC 稳定性 (ICIR) - 因子预测能力的稳定性
    2. IC 绝对值 - 因子的预测能力强度
    3. IR (信息比率) - 风险调整后的收益
    4. 多空收益 - 实际交易效果
    5. 胜率 - 交易成功率

    Attributes:
        weights: 权重配置字典
        thresholds: 评分阈值字典
        custom_weights: 是否使用自定义权重
        custom_thresholds: 是否使用自定义阈值

    Examples:
        >>> from core.reliability import ReliabilityEvaluator
        >>>
        >>> # 使用默认配置
        >>> evaluator = ReliabilityEvaluator()
        >>>
        >>> # 使用保守型配置
        >>> evaluator = ReliabilityEvaluator(strategy_type='conservative')
        >>>
        >>> # 使用自定义权重
        >>> custom_weights = {'ic_stability': 0.5, 'ic_absolute': 0.2, ...}
        >>> evaluator = ReliabilityEvaluator(weights=custom_weights)
        >>>
        >>> # 评估因子
        >>> result = evaluator.evaluate(metrics, scenario_results)
        >>> print(f"可靠性等级: {result['reliability']}")
        >>> print(f"综合评分: {result['total_score']:.2f}")
        >>> print(f"建议: {result['recommendation']}")
    """

    def __init__(
        self,
        weights: Optional[Dict[str, float]] = None,
        thresholds: Optional[Dict[str, Dict[str, float]]] = None,
        strategy_type: str = "default",
        strictness: str = "default",
    ):
        """
        初始化可靠性评估器

        Args:
            weights: 自定义权重配置，如果为 None，则使用 strategy_type 指定的预定义配置
            thresholds: 自定义评分阈值，如果为 None，则使用 strictness 指定的预定义配置
            strategy_type: 预定义策略类型
                - 'default': 默认权重
                - 'conservative': 保守型权重（更注重稳定性）
                - 'aggressive': 激进型权重（更注重收益）
            strictness: 评分严格程度
                - 'default': 默认阈值
                - 'strict': 严格阈值
                - 'relaxed': 宽松阈值

        Raises:
            ValueError: 如果权重配置无效

        Examples:
            >>> # 使用默认配置
            >>> evaluator = ReliabilityEvaluator()
            >>>
            >>> # 使用保守型配置和严格阈值
            >>> evaluator = ReliabilityEvaluator(
            ...     strategy_type='conservative',
            ...     strictness='strict'
            ... )
            >>>
            >>> # 使用完全自定义的配置
            >>> evaluator = ReliabilityEvaluator(
            ...     weights={'ic_stability': 0.5, 'ic_absolute': 0.2, ...},
            ...     thresholds={'ic_mean': {'excellent': 0.06, ...}}
            ... )
        """
        # 设置权重
        if weights is not None:
            # 使用自定义权重
            validate_weights(weights)
            self.weights = weights.copy()
            self.custom_weights = True
        else:
            # 使用预定义权重
            self.weights = get_weights(strategy_type)
            self.custom_weights = False

        # 设置阈值
        if thresholds is not None:
            # 使用自定义阈值
            self.thresholds = thresholds.copy()
            self.custom_thresholds = True
        else:
            # 使用预定义阈值
            self.thresholds = get_thresholds(strictness)
            self.custom_thresholds = False

        # 记录配置信息
        self.strategy_type = strategy_type if not self.custom_weights else "custom"
        self.strictness = strictness if not self.custom_thresholds else "custom"

    def evaluate(
        self,
        metrics: Dict[str, Union[float, pd.Series]],
        scenario_results: Optional[Dict[str, any]] = None,
        factor_name: str = "Factor",
    ) -> Dict[str, any]:
        """
        综合评估因子可靠性

        这是最常用的方法，根据性能指标和策略场景结果评估因子可靠性。

        Args:
            metrics: 性能指标字典，应包含：
                - ic_mean: IC 均值
                - icir: IC 信息比率
                - rank_ic_mean: Rank IC 均值
                - rank_icir: Rank IC 信息比率
                - long_short_return: 多空收益序列
                - annual_return: 年化收益率
                - sharpe_ratio: 夏普比率
                - max_drawdown: 最大回撤
                - win_rate: 胜率
                - calmar_ratio: 卡玛比率（可选）
            scenario_results: 策略场景分析结果（可选），如果提供，会使用实际的策略收益
            factor_name: 因子名称

        Returns:
            包含评估结果的字典：
                - scores: 各维度得分
                - total_score: 综合评分 (0-1)
                - reliability: 可靠性等级 ('A+', 'A', 'B', 'C', 'D', 'F')
                - recommendation: 建议文本
                - details: 详细评分信息
                - config: 使用的配置信息

        Examples:
            >>> evaluator = ReliabilityEvaluator()
            >>>
            >>> # 准备性能指标
            >>> metrics = {
            ...     'ic_mean': 0.05,
            ...     'icir': 0.8,
            ...     'rank_ic_mean': 0.06,
            ...     'rank_icir': 0.9,
            ...     'long_short_return': long_short_series,
            ...     'annual_return': 0.12,
            ...     'sharpe_ratio': 2.5,
            ...     'max_drawdown': -0.08,
            ...     'win_rate': 0.65,
            ... }
            >>>
            >>> # 评估因子
            >>> result = evaluator.evaluate(metrics, factor_name="MA20")
            >>>
            >>> print(f"可靠性等级: {result['reliability']}")
            >>> print(f"综合评分: {result['total_score']:.2f}")
            >>> print(result['recommendation'])
        """
        # 1. 评估各个维度
        scores = {}
        details = {}

        # 1.1 IC 稳定性评估
        icir_score, icir_detail = self._evaluate_ic_stability(metrics)
        scores["ic_stability"] = icir_score * self.weights["ic_stability"]
        details["ic_stability"] = icir_detail

        # 1.2 IC 绝对值评估
        ic_abs_score, ic_abs_detail = self._evaluate_ic_absolute(metrics)
        scores["ic_absolute"] = ic_abs_score * self.weights["ic_absolute"]
        details["ic_absolute"] = ic_abs_detail

        # 1.3 IR 评估
        ir_score, ir_detail = self._evaluate_ir(metrics)
        scores["ir"] = ir_score * self.weights["ir"]
        details["ir"] = ir_detail

        # 1.4 多空收益评估
        ls_score, ls_detail = self._evaluate_long_short_return(metrics, scenario_results)
        scores["long_short_return"] = ls_score * self.weights["long_short_return"]
        details["long_short_return"] = ls_detail

        # 1.5 胜率评估
        win_score, win_detail = self._evaluate_win_rate(metrics)
        scores["win_rate"] = win_score * self.weights["win_rate"]
        details["win_rate"] = win_detail

        # 2. 计算综合评分
        total_score = sum(scores.values())

        # 3. 确定可靠性等级
        reliability = get_reliability_grade(total_score)

        # 4. 生成建议
        recommendation = self._generate_recommendation(reliability, scores, details, metrics)

        return {
            "scores": scores,
            "total_score": total_score,
            "reliability": reliability,
            "recommendation": recommendation,
            "details": details,
            "config": {
                "strategy_type": self.strategy_type,
                "strictness": self.strictness,
                "weights": self.weights.copy(),
                "custom_weights": self.custom_weights,
                "custom_thresholds": self.custom_thresholds,
            },
        }

    def _evaluate_ic_stability(self, metrics: Dict[str, Union[float, pd.Series]]) -> Tuple[float, Dict[str, any]]:
        """
        评估 IC 稳定性

        使用 ICIR 作为主要指标，综合考虑 IC 的均值和标准差。

        Args:
            metrics: 性能指标字典

        Returns:
            (得分, 详细信息)
        """
        icir = metrics.get("icir", 0.0)
        rank_icir = metrics.get("rank_icir", 0.0)

        # 使用平均 ICIR
        avg_icir = (icir + rank_icir) / 2

        # 评估等级
        threshold = self.thresholds["icir"]

        if avg_icir >= threshold["excellent"]:
            score = 1.0
            grade = "excellent"
        elif avg_icir >= threshold["good"]:
            score = 0.7
            grade = "good"
        elif avg_icir >= threshold["warning"]:
            score = 0.4
            grade = "warning"
        else:
            score = 0.1
            grade = "poor"

        detail = {
            "icir": icir,
            "rank_icir": rank_icir,
            "avg_icir": avg_icir,
            "grade": grade,
            "weight": self.weights["ic_stability"],
            "description": self._get_icir_description(avg_icir, grade),
        }

        return score, detail

    def _evaluate_ic_absolute(self, metrics: Dict[str, Union[float, pd.Series]]) -> Tuple[float, Dict[str, any]]:
        """
        评估 IC 绝对值

        使用 IC 均值的绝对值作为主要指标。

        Args:
            metrics: 性能指标字典

        Returns:
            (得分, 详细信息)
        """
        ic_mean = abs(metrics.get("ic_mean", 0.0))
        rank_ic_mean = abs(metrics.get("rank_ic_mean", 0.0))

        # 使用平均 IC 绝对值
        avg_ic_abs = (ic_mean + rank_ic_mean) / 2

        # 评估等级
        threshold = self.thresholds["ic_mean"]

        if avg_ic_abs >= threshold["excellent"]:
            score = 1.0
            grade = "excellent"
        elif avg_ic_abs >= threshold["good"]:
            score = 0.7
            grade = "good"
        elif avg_ic_abs >= threshold["warning"]:
            score = 0.4
            grade = "warning"
        else:
            score = 0.1
            grade = "poor"

        detail = {
            "ic_mean_abs": ic_mean,
            "rank_ic_mean_abs": rank_ic_mean,
            "avg_ic_abs": avg_ic_abs,
            "grade": grade,
            "weight": self.weights["ic_absolute"],
            "description": self._get_ic_description(avg_ic_abs, grade),
        }

        return score, detail

    def _evaluate_ir(self, metrics: Dict[str, Union[float, pd.Series]]) -> Tuple[float, Dict[str, any]]:
        """
        评估信息比率 (IR)

        使用夏普比率作为 IR 的代理指标。

        Args:
            metrics: 性能指标字典

        Returns:
            (得分, 详细信息)
        """
        sharpe_ratio = metrics.get("sharpe_ratio", 0.0)
        calmar_ratio = metrics.get("calmar_ratio", None)

        # 评估等级
        threshold = self.thresholds["sharpe_ratio"]

        if sharpe_ratio >= threshold["excellent"]:
            score = 1.0
            grade = "excellent"
        elif sharpe_ratio >= threshold["good"]:
            score = 0.7
            grade = "good"
        elif sharpe_ratio >= threshold["warning"]:
            score = 0.4
            grade = "warning"
        else:
            score = 0.1
            grade = "poor"

        detail = {
            "sharpe_ratio": sharpe_ratio,
            "calmar_ratio": calmar_ratio,
            "grade": grade,
            "weight": self.weights["ir"],
            "description": self._get_ir_description(sharpe_ratio, grade),
        }

        return score, detail

    def _evaluate_long_short_return(
        self, metrics: Dict[str, Union[float, pd.Series]], scenario_results: Optional[Dict[str, any]] = None
    ) -> Tuple[float, Dict[str, any]]:
        """
        评估多空收益

        使用年化收益率作为主要指标。

        Args:
            metrics: 性能指标字典
            scenario_results: 策略场景分析结果（可选）

        Returns:
            (得分, 详细信息)
        """
        annual_return = metrics.get("annual_return", 0.0)

        # 如果有策略场景结果，可以使用看涨策略的收益
        if scenario_results and "bull" in scenario_results:
            bull_return = scenario_results["bull"].get("annual_return", annual_return)
            # 使用两者中较大的值
            annual_return = max(annual_return, bull_return)

        # 评估等级
        threshold = self.thresholds["annual_return"]

        if annual_return >= threshold["excellent"]:
            score = 1.0
            grade = "excellent"
        elif annual_return >= threshold["good"]:
            score = 0.7
            grade = "good"
        elif annual_return >= threshold["warning"]:
            score = 0.4
            grade = "warning"
        else:
            score = 0.1
            grade = "poor"

        detail = {
            "annual_return": annual_return,
            "grade": grade,
            "weight": self.weights["long_short_return"],
            "description": self._get_return_description(annual_return, grade),
        }

        return score, detail

    def _evaluate_win_rate(self, metrics: Dict[str, Union[float, pd.Series]]) -> Tuple[float, Dict[str, any]]:
        """
        评估胜率

        Args:
            metrics: 性能指标字典

        Returns:
            (得分, 详细信息)
        """
        win_rate = metrics.get("win_rate", 0.0)

        # 评估等级
        threshold = self.thresholds["win_rate"]

        if win_rate >= threshold["excellent"]:
            score = 1.0
            grade = "excellent"
        elif win_rate >= threshold["good"]:
            score = 0.7
            grade = "good"
        elif win_rate >= threshold["warning"]:
            score = 0.4
            grade = "warning"
        else:
            score = 0.1
            grade = "poor"

        detail = {
            "win_rate": win_rate,
            "grade": grade,
            "weight": self.weights["win_rate"],
            "description": self._get_win_rate_description(win_rate, grade),
        }

        return score, detail

    def _generate_recommendation(
        self,
        reliability: str,
        scores: Dict[str, float],
        details: Dict[str, Dict[str, any]],
        metrics: Dict[str, Union[float, pd.Series]],
    ) -> str:
        """
        生成可靠性评估建议

        Args:
            reliability: 可靠性等级
            scores: 各维度得分
            details: 详细评分信息
            metrics: 性能指标

        Returns:
            建议文本
        """
        grade_info = RELIABILITY_GRADES[reliability]

        recommendation_lines = [f"可靠性等级: {reliability} ({grade_info['description']})\n"]

        # 总体建议
        recommendation_lines.append(f"总体评价: {grade_info['recommendation']}")

        # 优势分析
        strong_points = [key for key, score in scores.items() if score >= 0.7 * self.weights[key]]

        if strong_points:
            recommendation_lines.append("\n优势:")
            for point in strong_points:
                detail = details[point]
                recommendation_lines.append(f"  ✅ {self._get_dimension_name(point)}: " f"{detail['description']}")

        # 劣势分析
        weak_points = [key for key, score in scores.items() if score < 0.4 * self.weights[key]]

        if weak_points:
            recommendation_lines.append("\n待改进:")
            for point in weak_points:
                detail = details[point]
                recommendation_lines.append(f"  ⚠️  {self._get_dimension_name(point)}: " f"{detail['description']}")

        # 具体建议
        recommendation_lines.append("\n改进建议:")

        if reliability in ["A+", "A"]:
            recommendation_lines.append(
                "  - 该因子表现优秀，建议直接使用\n"
                "  - 可以作为核心因子构建多因子模型\n"
                "  - 定期监控因子表现，确保稳定性"
            )
        elif reliability == "B":
            recommendation_lines.append(
                "  - 该因子表现中等，可以与其他因子组合使用\n"
                "  - 建议优化因子计算方法或参数\n"
                "  - 可以考虑在不同市场环境下分别使用"
            )
        elif reliability == "C":
            recommendation_lines.append(
                "  - 该因子表现一般，需要进一步优化\n"
                "  - 建议重新设计因子逻辑\n"
                "  - 或者作为辅助因子，不建议单独使用"
            )
        else:  # D 或 F
            recommendation_lines.append(
                "  - 该因子表现较差，不建议使用\n" "  - 建议重新设计因子或放弃使用\n" "  - 如果要使用，需要大幅改进"
            )

        return "\n".join(recommendation_lines)

    def _get_dimension_name(self, key: str) -> str:
        """获取评估维度的中文名称"""
        name_map = {
            "ic_stability": "IC 稳定性",
            "ic_absolute": "IC 预测能力",
            "ir": "信息比率",
            "long_short_return": "多空收益",
            "win_rate": "胜率",
        }
        return name_map.get(key, key)

    def _get_icir_description(self, value: float, grade: str) -> str:
        """获取 ICIR 评估描述"""
        if grade == "excellent":
            return f"ICIR 为 {value:.2f}，稳定性非常好"
        elif grade == "good":
            return f"ICIR 为 {value:.2f}，稳定性良好"
        elif grade == "warning":
            return f"ICIR 为 {value:.2f}，稳定性一般，需要改进"
        else:
            return f"ICIR 为 {value:.2f}，稳定性较差"

    def _get_ic_description(self, value: float, grade: str) -> str:
        """获取 IC 评估描述"""
        if grade == "excellent":
            return f"IC 均值为 {value:.3f}，预测能力很强"
        elif grade == "good":
            return f"IC 均值为 {value:.3f}，预测能力良好"
        elif grade == "warning":
            return f"IC 均值为 {value:.3f}，预测能力一般"
        else:
            return f"IC 均值为 {value:.3f}，预测能力较弱"

    def _get_ir_description(self, value: float, grade: str) -> str:
        """获取 IR 评估描述"""
        if grade == "excellent":
            return f"夏普比率为 {value:.2f}，风险调整后收益极佳"
        elif grade == "good":
            return f"夏普比率为 {value:.2f}，风险调整后收益良好"
        elif grade == "warning":
            return f"夏普比率为 {value:.2f}，风险调整后收益一般"
        else:
            return f"夏普比率为 {value:.2f}，风险调整后收益较差"

    def _get_return_description(self, value: float, grade: str) -> str:
        """获取收益评估描述"""
        if grade == "excellent":
            return f"年化收益率为 {value:.2%}，收益表现优异"
        elif grade == "good":
            return f"年化收益率为 {value:.2%}，收益表现良好"
        elif grade == "warning":
            return f"年化收益率为 {value:.2%}，收益表现一般"
        else:
            return f"年化收益率为 {value:.2%}，收益表现较差"

    def _get_win_rate_description(self, value: float, grade: str) -> str:
        """获取胜率评估描述"""
        if grade == "excellent":
            return f"胜率为 {value:.2%}，交易成功率很高"
        elif grade == "good":
            return f"胜率为 {value:.2%}，交易成功率良好"
        elif grade == "warning":
            return f"胜率为 {value:.2%}，交易成功率一般"
        else:
            return f"胜率为 {value:.2%}，交易成功率较低"

    def generate_report(self, evaluation_result: Dict[str, any], factor_name: str = "Factor") -> str:
        """
        生成可靠性评估报告

        Args:
            evaluation_result: evaluate 方法返回的评估结果
            factor_name: 因子名称

        Returns:
            格式化的报告文本

        Examples:
            >>> report = evaluator.generate_report(result, "MA20")
            >>> print(report)
        """
        lines = [
            "=" * 70,
            f"因子可靠性评估报告: {factor_name}",
            "=" * 70,
            "",
        ]

        # 基本信息
        lines.append("1. 评估配置")
        lines.append("-" * 70)
        config = evaluation_result["config"]
        lines.append(f"  策略类型: {config['strategy_type']}")
        lines.append(f"  严格程度: {config['strictness']}")
        lines.append("  权重配置:")
        for key, value in config["weights"].items():
            lines.append(f"    {self._get_dimension_name(key)}: {value:.2%}")
        lines.append("")

        # 综合评分
        lines.append("2. 综合评分")
        lines.append("-" * 70)
        total_score = evaluation_result["total_score"]
        reliability = evaluation_result["reliability"]
        lines.append(f"  综合评分: {total_score:.2f} / 1.00")
        lines.append(f"  可靠性等级: {reliability}")
        lines.append("")

        # 各维度得分
        lines.append("3. 各维度得分")
        lines.append("-" * 70)
        scores = evaluation_result["scores"]
        details = evaluation_result["details"]

        for key in ["ic_stability", "ic_absolute", "ir", "long_short_return", "win_rate"]:
            score = scores[key]
            detail = details[key]
            weight = self.weights[key]
            lines.append(f"  {self._get_dimension_name(key)} (权重 {weight:.2%}): " f"{score:.3f} / {weight:.3f}")
            lines.append(f"    {detail['description']}")
        lines.append("")

        # 建议
        lines.append("4. 评估建议")
        lines.append("-" * 70)
        lines.append(evaluation_result["recommendation"])
        lines.append("")

        lines.append("=" * 70)

        return "\n".join(lines)


# 便捷函数
def evaluate_factor_reliability(
    metrics: Dict[str, Union[float, pd.Series]],
    scenario_results: Optional[Dict[str, any]] = None,
    strategy_type: str = "default",
    strictness: str = "default",
    factor_name: str = "Factor",
) -> Dict[str, any]:
    """
    评估因子可靠性的便捷函数

    Args:
        metrics: 性能指标字典
        scenario_results: 策略场景分析结果（可选）
        strategy_type: 策略类型
        strictness: 严格程度
        factor_name: 因子名称

    Returns:
        评估结果字典

    Examples:
        >>> from core.reliability import evaluate_factor_reliability
        >>>
        >>> # 快速评估
        >>> result = evaluate_factor_reliability(
        ...     metrics,
        ...     scenario_results,
        ...     strategy_type='conservative'
        ... )
        >>>
        >>> print(f"可靠性等级: {result['reliability']}")
        >>> print(f"综合评分: {result['total_score']:.2f}")
    """
    evaluator = ReliabilityEvaluator(strategy_type=strategy_type, strictness=strictness)
    return evaluator.evaluate(metrics, scenario_results, factor_name)


if __name__ == "__main__":
    # 示例：使用可靠性评估器
    print("=" * 70)
    print("可靠性评估器示例")
    print("=" * 70)

    # 创建模拟性能指标
    print("\n生成模拟性能指标...")
    metrics = {
        "ic_mean": 0.05,
        "icir": 0.8,
        "rank_ic_mean": 0.06,
        "rank_icir": 0.9,
        "long_short_return": pd.Series([0.01, 0.02, -0.01, 0.03, 0.01]),
        "annual_return": 0.12,
        "sharpe_ratio": 2.5,
        "max_drawdown": -0.08,
        "win_rate": 0.65,
        "calmar_ratio": 1.5,
    }

    print(f"IC 均值: {metrics['ic_mean']:.3f}")
    print(f"ICIR: {metrics['icir']:.2f}")
    print(f"年化收益: {metrics['annual_return']:.2%}")
    print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"胜率: {metrics['win_rate']:.2%}")

    # 使用默认配置评估
    print("\n" + "=" * 70)
    print("使用默认配置评估")
    print("=" * 70)

    evaluator = ReliabilityEvaluator()
    result = evaluator.evaluate(metrics, factor_name="模拟因子")

    print(f"\n综合评分: {result['total_score']:.2f}")
    print(f"可靠性等级: {result['reliability']}")
    print("\n各维度得分:")
    for key, score in result["scores"].items():
        print(f"  {evaluator._get_dimension_name(key)}: {score:.3f}")

    print("\n" + evaluator.generate_report(result, "模拟因子"))

    # 使用保守型配置评估
    print("\n" + "=" * 70)
    print("使用保守型配置评估")
    print("=" * 70)

    conservative_evaluator = ReliabilityEvaluator(strategy_type="conservative")
    conservative_result = conservative_evaluator.evaluate(metrics, factor_name="模拟因子")

    print(f"\n综合评分: {conservative_result['total_score']:.2f}")
    print(f"可靠性等级: {conservative_result['reliability']}")

    # 使用激进型配置评估
    print("\n" + "=" * 70)
    print("使用激进型配置评估")
    print("=" * 70)

    aggressive_evaluator = ReliabilityEvaluator(strategy_type="aggressive")
    aggressive_result = aggressive_evaluator.evaluate(metrics, factor_name="模拟因子")

    print(f"\n综合评分: {aggressive_result['total_score']:.2f}")
    print(f"可靠性等级: {aggressive_result['reliability']}")
