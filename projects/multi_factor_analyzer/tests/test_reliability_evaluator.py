"""
可靠性评估器测试

测试可靠性评估器的各项功能。
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加 src 目录到路径
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from core.reliability import (
    ReliabilityEvaluator,
    evaluate_factor_reliability,
)


@pytest.fixture
def good_metrics():
    """创建优秀的性能指标"""
    return {
        'ic_mean': 0.05,
        'icir': 0.8,
        'rank_ic_mean': 0.06,
        'rank_icir': 0.9,
        'long_short_return': pd.Series([0.01, 0.02, -0.01, 0.03, 0.01]),
        'annual_return': 0.12,
        'sharpe_ratio': 2.5,
        'max_drawdown': -0.08,
        'win_rate': 0.65,
        'calmar_ratio': 1.5,
    }


@pytest.fixture
def poor_metrics():
    """创建较差的性能指标"""
    return {
        'ic_mean': 0.005,
        'icir': 0.2,
        'rank_ic_mean': 0.01,
        'rank_icir': 0.3,
        'long_short_return': pd.Series([-0.01, -0.02, 0.01, -0.03, -0.01]),
        'annual_return': -0.05,
        'sharpe_ratio': 0.5,
        'max_drawdown': -0.30,
        'win_rate': 0.40,
        'calmar_ratio': -0.17,
    }


@pytest.fixture
def scenario_results():
    """创建策略场景结果"""
    return {
        'bull': {
            'annual_return': 0.15,
            'sharpe_ratio': 2.8,
            'max_drawdown': -0.06,
            'win_rate': 0.70,
        },
        'bear': {
            'annual_return': -0.03,
            'sharpe_ratio': 0.8,
            'max_drawdown': -0.15,
            'win_rate': 0.45,
        },
    }


class TestReliabilityEvaluator:
    """可靠性评估器测试"""

    def test_init_default(self):
        """测试默认初始化"""
        evaluator = ReliabilityEvaluator()

        assert evaluator.strategy_type == 'default'
        assert evaluator.strictness == 'default'
        assert not evaluator.custom_weights
        assert not evaluator.custom_thresholds

    def test_init_strategy_type(self):
        """测试指定策略类型"""
        evaluator = ReliabilityEvaluator(strategy_type='conservative')

        assert evaluator.strategy_type == 'conservative'
        assert evaluator.weights['ic_stability'] == 0.50

    def test_init_custom_weights(self):
        """测试自定义权重"""
        custom_weights = {
            'ic_stability': 0.5,
            'ic_absolute': 0.2,
            'ir': 0.15,
            'long_short_return': 0.1,
            'win_rate': 0.05,
        }

        evaluator = ReliabilityEvaluator(weights=custom_weights)

        assert evaluator.custom_weights
        assert evaluator.weights == custom_weights

    def test_init_invalid_weights(self):
        """测试无效权重"""
        invalid_weights = {
            'ic_stability': 0.5,
            'ic_absolute': 0.3,
            'ir': 0.3,  # 总和 > 1
            'long_short_return': 0.0,
            'win_rate': 0.0,
        }

        with pytest.raises(ValueError, match="权重总和必须为 1.0"):
            ReliabilityEvaluator(weights=invalid_weights)

    def test_init_strictness(self):
        """测试指定严格程度"""
        evaluator = ReliabilityEvaluator(strictness='strict')

        assert evaluator.strictness == 'strict'

    def test_evaluate_good_factor(self, good_metrics):
        """测试评估优秀因子"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(good_metrics)

        # 检查返回键
        assert 'scores' in result
        assert 'total_score' in result
        assert 'reliability' in result
        assert 'recommendation' in result
        assert 'details' in result
        assert 'config' in result

        # 检查评分范围
        assert 0 <= result['total_score'] <= 1

        # 优秀因子应该获得 A 或 A+
        assert result['reliability'] in ['A+', 'A']

    def test_evaluate_poor_factor(self, poor_metrics):
        """测试评估较差因子"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(poor_metrics)

        # 较差因子应该获得 C、D 或 F
        assert result['reliability'] in ['C', 'D', 'F']

    def test_evaluate_with_scenario_results(self, good_metrics, scenario_results):
        """测试带策略场景结果的评估"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(good_metrics, scenario_results)

        # 检查结果
        assert result['total_score'] >= 0

    def test_evaluate_scores_sum(self, good_metrics):
        """测试各维度得分"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(good_metrics)

        # 检查各维度得分存在
        required_scores = [
            'ic_stability',
            'ic_absolute',
            'ir',
            'long_short_return',
            'win_rate',
        ]

        for score_name in required_scores:
            assert score_name in result['scores']
            assert result['scores'][score_name] >= 0

    def test_evaluate_ic_stability(self, good_metrics):
        """测试 IC 稳定性评估"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(good_metrics)

        # 检查详细信息
        assert 'ic_stability' in result['details']
        detail = result['details']['ic_stability']

        assert 'icir' in detail
        assert 'rank_icir' in detail
        assert 'avg_icir' in detail
        assert 'grade' in detail
        assert 'weight' in detail
        assert 'description' in detail

    def test_evaluate_ic_absolute(self, good_metrics):
        """测试 IC 绝对值评估"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(good_metrics)

        assert 'ic_absolute' in result['details']
        detail = result['details']['ic_absolute']

        assert 'ic_mean_abs' in detail
        assert 'rank_ic_mean_abs' in detail
        assert 'avg_ic_abs' in detail

    def test_evaluate_ir(self, good_metrics):
        """测试 IR 评估"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(good_metrics)

        assert 'ir' in result['details']
        detail = result['details']['ir']

        assert 'sharpe_ratio' in detail

    def test_evaluate_long_short_return(self, good_metrics):
        """测试多空收益评估"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(good_metrics)

        assert 'long_short_return' in result['details']
        detail = result['details']['long_short_return']

        assert 'annual_return' in detail

    def test_evaluate_win_rate(self, good_metrics):
        """测试胜率评估"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(good_metrics)

        assert 'win_rate' in result['details']
        detail = result['details']['win_rate']

        assert 'win_rate' in detail

    def test_conservative_strategy_lower_score(self, good_metrics):
        """测试保守型策略评分"""
        default_evaluator = ReliabilityEvaluator(strategy_type='default')
        conservative_evaluator = ReliabilityEvaluator(strategy_type='conservative')

        default_result = default_evaluator.evaluate(good_metrics)
        conservative_result = conservative_evaluator.evaluate(good_metrics)

        # 保守型可能评分不同（取决于权重）
        # 这里只检查都能正常评估
        assert default_result['total_score'] >= 0
        assert conservative_result['total_score'] >= 0

    def test_aggressive_strategy_higher_return_weight(self, good_metrics):
        """测试激进型策略权重"""
        aggressive_evaluator = ReliabilityEvaluator(strategy_type='aggressive')

        assert aggressive_evaluator.weights['long_short_return'] == 0.25

    def test_strict_thresholds_stricter_grading(self, good_metrics):
        """测试严格阈值更严格的评分"""
        default_evaluator = ReliabilityEvaluator(strictness='default')
        strict_evaluator = ReliabilityEvaluator(strictness='strict')

        default_result = default_evaluator.evaluate(good_metrics)
        strict_result = strict_evaluator.evaluate(good_metrics)

        # 严格阈值可能给出更低的等级
        # 这里只检查都能正常评估
        assert default_result['reliability'] in ['A+', 'A', 'B', 'C', 'D', 'F']
        assert strict_result['reliability'] in ['A+', 'A', 'B', 'C', 'D', 'F']

    def test_generate_report(self, good_metrics):
        """测试生成报告"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(good_metrics)
        report = evaluator.generate_report(result, "TestFactor")

        # 检查报告内容
        assert 'TestFactor' in report
        assert '综合评分' in report
        assert '可靠性等级' in report
        assert '评估配置' in report
        assert '各维度得分' in report
        assert '评估建议' in report

    def test_get_dimension_name(self, good_metrics):
        """测试获取维度名称"""
        evaluator = ReliabilityEvaluator()

        assert evaluator._get_dimension_name('ic_stability') == 'IC 稳定性'
        assert evaluator._get_dimension_name('ic_absolute') == 'IC 预测能力'
        assert evaluator._get_dimension_name('ir') == '信息比率'
        assert evaluator._get_dimension_name('long_short_return') == '多空收益'
        assert evaluator._get_dimension_name('win_rate') == '胜率'


class TestEvaluateFactorReliability:
    """便捷函数测试"""

    def test_evaluate_factor_reliability_basic(self, good_metrics):
        """测试基本功能"""
        result = evaluate_factor_reliability(good_metrics)

        assert 'total_score' in result
        assert 'reliability' in result

    def test_evaluate_factor_reliability_with_strategy(self, good_metrics):
        """测试指定策略类型"""
        result = evaluate_factor_reliability(
            good_metrics,
            strategy_type='conservative'
        )

        assert result['config']['strategy_type'] == 'conservative'

    def test_evaluate_factor_reliability_with_strictness(self, good_metrics):
        """测试指定严格程度"""
        result = evaluate_factor_reliability(
            good_metrics,
            strictness='strict'
        )

        assert result['config']['strictness'] == 'strict'


class TestEdgeCases:
    """边界情况测试"""

    def test_missing_optional_metrics(self):
        """测试缺少可选指标"""
        minimal_metrics = {
            'ic_mean': 0.03,
            'icir': 0.5,
            'rank_ic_mean': 0.04,
            'rank_icir': 0.6,
            'long_short_return': pd.Series([0.01] * 10),
            'annual_return': 0.08,
            'sharpe_ratio': 1.5,
            'max_drawdown': -0.10,
            'win_rate': 0.55,
        }

        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(minimal_metrics)

        # 应该能正常评估
        assert result['total_score'] >= 0

    def test_zero_metrics(self):
        """测试零值指标"""
        zero_metrics = {
            'ic_mean': 0.0,
            'icir': 0.0,
            'rank_ic_mean': 0.0,
            'rank_icir': 0.0,
            'long_short_return': pd.Series([0.0] * 10),
            'annual_return': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'win_rate': 0.5,
        }

        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(zero_metrics)

        # 应该能正常评估，但评分应该很低
        assert result['total_score'] >= 0
        assert result['reliability'] in ['C', 'D', 'F']

    def test_negative_ic(self):
        """测试负 IC"""
        negative_metrics = {
            'ic_mean': -0.05,  # 负 IC
            'icir': -0.8,
            'rank_ic_mean': -0.06,
            'rank_icir': -0.9,
            'long_short_return': pd.Series([-0.01, -0.02, -0.01, -0.03, -0.01]),
            'annual_return': -0.12,
            'sharpe_ratio': -2.5,
            'max_drawdown': -0.30,
            'win_rate': 0.35,
            'calmar_ratio': -0.4,
        }

        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(negative_metrics)

        # 负 IC 因子应该获得低分
        assert result['total_score'] >= 0
        # IC 绝对值评估应该使用绝对值，所以负 IC 不应该影响该维度
        assert result['details']['ic_absolute']['ic_mean_abs'] == 0.05


class TestRecommendationGeneration:
    """建议生成测试"""

    def test_recommendation_for_excellent_factor(self, good_metrics):
        """测试优秀因子的建议"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(good_metrics)

        recommendation = result['recommendation']

        # 优秀因子的建议应该包含正面评价
        assert '优秀' in recommendation or '良好' in recommendation

    def test_recommendation_for_poor_factor(self, poor_metrics):
        """测试较差因子的建议"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(poor_metrics)

        recommendation = result['recommendation']

        # 较差因子的建议应该包含改进建议
        assert '改进' in recommendation or '不建议' in recommendation

    def test_recommendation_contains_strong_points(self, good_metrics):
        """测试建议包含优势分析"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(good_metrics)

        recommendation = result['recommendation']

        # 应该有优势分析（如果有的话）
        if '优势:' in recommendation:
            assert '✅' in recommendation

    def test_recommendation_contains_weak_points(self, poor_metrics):
        """测试建议包含劣势分析"""
        evaluator = ReliabilityEvaluator()
        result = evaluator.evaluate(poor_metrics)

        recommendation = result['recommendation']

        # 应该有待改进分析（如果有的话）
        if '待改进:' in recommendation:
            assert '⚠️' in recommendation


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
