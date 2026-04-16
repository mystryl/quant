"""
可靠性评估配置文件测试

测试配置文件中的权重、阈值和等级定义。
"""

import pytest
import sys
from pathlib import Path

# 添加 src 目录到路径
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from core.config import (
    DEFAULT_WEIGHTS,
    CONSERVATIVE_WEIGHTS,
    AGGRESSIVE_WEIGHTS,
    HIGH_FREQUENCY_WEIGHTS,
    DEFAULT_THRESHOLDS,
    STRICT_THRESHOLDS,
    RELAXED_THRESHOLDS,
    RELIABILITY_GRADES,
    CORRELATION_THRESHOLDS,
    get_weights,
    get_thresholds,
    get_reliability_grade,
    validate_weights,
)


class TestWeights:
    """权重配置测试"""

    def test_default_weights_sum_to_one(self):
        """测试默认权重总和为 1"""
        total = sum(DEFAULT_WEIGHTS.values())
        assert total == pytest.approx(1.0, rel=0.01)

    def test_conservative_weights_sum_to_one(self):
        """测试保守型权重总和为 1"""
        total = sum(CONSERVATIVE_WEIGHTS.values())
        assert total == pytest.approx(1.0, rel=0.01)

    def test_aggressive_weights_sum_to_one(self):
        """测试激进型权重总和为 1"""
        total = sum(AGGRESSIVE_WEIGHTS.values())
        assert total == pytest.approx(1.0, rel=0.01)

    def test_high_frequency_weights_sum_to_one(self):
        """测试高频交易权重总和为 1"""
        total = sum(HIGH_FREQUENCY_WEIGHTS.values())
        assert total == pytest.approx(1.0, rel=0.01)

    def test_conservative_higher_stability_weight(self):
        """测试保守型权重的稳定性权重更高"""
        assert CONSERVATIVE_WEIGHTS['ic_stability'] > DEFAULT_WEIGHTS['ic_stability']

    def test_aggressive_higher_return_weight(self):
        """测试激进型权重的收益权重更高"""
        assert AGGRESSIVE_WEIGHTS['long_short_return'] > DEFAULT_WEIGHTS['long_short_return']

    def test_all_weights_between_zero_and_one(self):
        """测试所有权重在 [0, 1] 范围内"""
        for weights in [DEFAULT_WEIGHTS, CONSERVATIVE_WEIGHTS, AGGRESSIVE_WEIGHTS]:
            for key, value in weights.items():
                assert 0 <= value <= 1

    def test_get_weights_default(self):
        """测试获取默认权重"""
        weights = get_weights('default')
        assert weights == DEFAULT_WEIGHTS

    def test_get_weights_conservative(self):
        """测试获取保守型权重"""
        weights = get_weights('conservative')
        assert weights == CONSERVATIVE_WEIGHTS

    def test_get_weights_aggressive(self):
        """测试获取激进型权重"""
        weights = get_weights('aggressive')
        assert weights == AGGRESSIVE_WEIGHTS

    def test_get_weights_invalid_strategy(self):
        """测试获取无效策略类型的权重"""
        with pytest.raises(ValueError, match="不支持的策略类型"):
            get_weights('invalid_strategy')

    def test_get_weights_returns_copy(self):
        """测试获取权重返回副本"""
        weights1 = get_weights('default')
        weights2 = get_weights('default')
        weights1['ic_stability'] = 0.99
        assert weights2['ic_stability'] != 0.99


class TestThresholds:
    """阈值配置测试"""

    def test_default_thresholds_exist(self):
        """测试默认阈值存在"""
        required_keys = [
            'ic_mean', 'icir', 'rank_ic_mean', 'rank_icir',
            'annual_return', 'sharpe_ratio', 'max_drawdown', 'win_rate', 'calmar_ratio'
        ]
        for key in required_keys:
            assert key in DEFAULT_THRESHOLDS

    def test_threshold_levels_exist(self):
        """测试阈值等级存在"""
        for metric_thresholds in DEFAULT_THRESHOLDS.values():
            assert 'excellent' in metric_thresholds
            assert 'good' in metric_thresholds
            assert 'warning' in metric_thresholds

    def test_thresholds_excellent_greater_than_good(self):
        """测试优秀阈值大于良好阈值（对于越大越好的指标）"""
        # 对于 max_drawdown，越小越好
        assert DEFAULT_THRESHOLDS['max_drawdown']['excellent'] < DEFAULT_THRESHOLDS['max_drawdown']['good']

        # 对于其他指标，越大越好
        for key in ['ic_mean', 'icir', 'rank_ic_mean', 'annual_return', 'sharpe_ratio', 'win_rate']:
            assert DEFAULT_THRESHOLDS[key]['excellent'] > DEFAULT_THRESHOLDS[key]['good']
            assert DEFAULT_THRESHOLDS[key]['good'] > DEFAULT_THRESHOLDS[key]['warning']

    def test_strict_thresholds_higher_than_default(self):
        """测试严格阈值高于默认阈值"""
        # 对于 max_drawdown，越小越好
        assert STRICT_THRESHOLDS['max_drawdown']['excellent'] < DEFAULT_THRESHOLDS['max_drawdown']['excellent']

        # 对于其他指标，越大越好
        for key in ['ic_mean', 'icir', 'annual_return']:
            assert STRICT_THRESHOLDS[key]['excellent'] > DEFAULT_THRESHOLDS[key]['excellent']

    def test_relaxed_thresholds_lower_than_default(self):
        """测试宽松阈值低于默认阈值"""
        # 对于 max_drawdown，越小越好
        assert RELAXED_THRESHOLDS['max_drawdown']['excellent'] > DEFAULT_THRESHOLDS['max_drawdown']['excellent']

        # 对于其他指标，越大越好
        for key in ['ic_mean', 'icir', 'annual_return']:
            assert RELAXED_THRESHOLDS[key]['excellent'] < DEFAULT_THRESHOLDS[key]['excellent']

    def test_get_thresholds_default(self):
        """测试获取默认阈值"""
        thresholds = get_thresholds('default')
        assert thresholds == DEFAULT_THRESHOLDS

    def test_get_thresholds_strict(self):
        """测试获取严格阈值"""
        thresholds = get_thresholds('strict')
        assert thresholds == STRICT_THRESHOLDS

    def test_get_thresholds_relaxed(self):
        """测试获取宽松阈值"""
        thresholds = get_thresholds('relaxed')
        assert thresholds == RELAXED_THRESHOLDS

    def test_get_thresholds_invalid_strictness(self):
        """测试获取无效严格程度的阈值"""
        with pytest.raises(ValueError, match="不支持的严格程度"):
            get_thresholds('invalid_strictness')


class TestReliabilityGrades:
    """可靠性等级测试"""

    def test_all_grades_exist(self):
        """测试所有等级存在"""
        required_grades = ['A+', 'A', 'B', 'C', 'D', 'F']
        for grade in required_grades:
            assert grade in RELIABILITY_GRADES

    def test_grade_ranges(self):
        """测试等级分数范围"""
        assert RELIABILITY_GRADES['A+']['score_range'] == (0.90, 1.0)
        assert RELIABILITY_GRADES['A']['score_range'] == (0.80, 0.90)
        assert RELIABILITY_GRADES['B']['score_range'] == (0.70, 0.80)
        assert RELIABILITY_GRADES['C']['score_range'] == (0.60, 0.70)
        assert RELIABILITY_GRADES['D']['score_range'] == (0.50, 0.60)
        assert RELIABILITY_GRADES['F']['score_range'] == (0.0, 0.50)

    def test_grade_attributes(self):
        """测试等级属性"""
        for grade, info in RELIABILITY_GRADES.items():
            assert 'score_range' in info
            assert 'description' in info
            assert 'color' in info
            assert 'recommendation' in info

    def test_get_reliability_grade_a_plus(self):
        """测试获取 A+ 等级"""
        assert get_reliability_grade(0.95) == 'A+'
        assert get_reliability_grade(0.90) == 'A+'

    def test_get_reliability_grade_a(self):
        """测试获取 A 等级"""
        assert get_reliability_grade(0.85) == 'A'

    def test_get_reliability_grade_b(self):
        """测试获取 B 等级"""
        assert get_reliability_grade(0.75) == 'B'

    def test_get_reliability_grade_c(self):
        """测试获取 C 等级"""
        assert get_reliability_grade(0.65) == 'C'

    def test_get_reliability_grade_d(self):
        """测试获取 D 等级"""
        assert get_reliability_grade(0.55) == 'D'

    def test_get_reliability_grade_f(self):
        """测试获取 F 等级"""
        assert get_reliability_grade(0.45) == 'F'
        assert get_reliability_grade(0.0) == 'F'

    def test_get_reliability_grade_boundary(self):
        """测试边界值"""
        # 测试各个区间的边界
        assert get_reliability_grade(0.90) == 'A+'
        assert get_reliability_grade(0.80) == 'A'
        assert get_reliability_grade(0.70) == 'B'
        assert get_reliability_grade(0.60) == 'C'
        assert get_reliability_grade(0.50) == 'D'


class TestCorrelationThresholds:
    """相关性阈值测试"""

    def test_correlation_thresholds_exist(self):
        """测试相关性阈值存在"""
        assert 'high' in CORRELATION_THRESHOLDS
        assert 'medium' in CORRELATION_THRESHOLDS
        assert 'low' in CORRELATION_THRESHOLDS

    def test_correlation_thresholds_values(self):
        """测试相关性阈值值"""
        assert CORRELATION_THRESHOLDS['high'] == 0.70
        assert CORRELATION_THRESHOLDS['medium'] == 0.50
        assert CORRELATION_THRESHOLDS['low'] == 0.30

    def test_correlation_thresholds_ordering(self):
        """测试相关性阈值顺序"""
        assert CORRELATION_THRESHOLDS['high'] > CORRELATION_THRESHOLDS['medium']
        assert CORRELATION_THRESHOLDS['medium'] > CORRELATION_THRESHOLDS['low']


class TestValidateWeights:
    """权重验证测试"""

    def test_validate_valid_weights(self):
        """测试验证有效权重"""
        assert validate_weights(DEFAULT_WEIGHTS) is True

    def test_validate_weights_sum_not_one(self):
        """测试权重总和不为 1"""
        invalid_weights = {'ic_stability': 0.5, 'ic_absolute': 0.3, 'ir': 0.3}
        with pytest.raises(ValueError, match="权重总和必须为 1.0"):
            validate_weights(invalid_weights)

    def test_validate_weights_out_of_range(self):
        """测试权重超出范围"""
        invalid_weights = {
            'ic_stability': 1.5,  # > 1
            'ic_absolute': -0.5,  # < 0
            'ir': 0.0,
            'long_short_return': 0.0,
            'win_rate': 0.0,
        }
        with pytest.raises(ValueError, match="权重.*的值必须在.*范围内"):
            validate_weights(invalid_weights)

    def test_validate_weights_with_floating_point_error(self):
        """测试浮点误差范围内的权重"""
        weights = {
            'ic_stability': 0.40000001,
            'ic_absolute': 0.19999999,
            'ir': 0.2,
            'long_short_return': 0.1,
            'win_rate': 0.1,
        }
        assert validate_weights(weights) is True


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
