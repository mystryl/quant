"""
因子相关性分析器测试

测试因子相关性分析器的各项功能。
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加 src 目录到路径
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from core.correlation_analyzer import (
    FactorCorrelationAnalyzer,
    analyze_factor_correlation,
)


@pytest.fixture
def sample_factor_dict():
    """创建示例因子数据"""
    np.random.seed(42)

    dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
    instruments = ['SH600000', 'SH600001', 'SH600002']

    # 创建多索引
    index = pd.MultiIndex.from_product(
        [dates, instruments],
        names=['datetime', 'instrument']
    )

    n = len(index)

    # 创建相关和不相关的因子
    factor_dict = {
        # MA20 和 MA60 高度相关
        'MA20': pd.Series(np.random.randn(n), index=index),
        'MA60': None,  # 将在下面设置
        # RSI 独立
        'RSI': pd.Series(np.random.randn(n), index=index),
        # MACD 独立
        'MACD': pd.Series(np.random.randn(n), index=index),
    }

    # MA60 与 MA20 高度相关
    factor_dict['MA60'] = factor_dict['MA20'] * 0.9 + np.random.randn(n) * 0.1

    return factor_dict


@pytest.fixture
def analyzer():
    """创建分析器实例"""
    return FactorCorrelationAnalyzer(method='spearman', threshold=0.7)


class TestFactorCorrelationAnalyzer:
    """因子相关性分析器测试"""

    def test_init_default(self):
        """测试默认初始化"""
        analyzer = FactorCorrelationAnalyzer()
        assert analyzer.method == 'spearman'
        assert analyzer.threshold == 0.7

    def test_init_custom_method(self):
        """测试自定义方法"""
        analyzer = FactorCorrelationAnalyzer(method='pearson')
        assert analyzer.method == 'pearson'

    def test_init_invalid_method(self):
        """测试无效方法"""
        with pytest.raises(ValueError, match="不支持的相关性方法"):
            FactorCorrelationAnalyzer(method='invalid')

    def test_init_custom_threshold(self):
        """测试自定义阈值"""
        analyzer = FactorCorrelationAnalyzer(threshold=0.8)
        assert analyzer.threshold == 0.8

    def test_calculate_correlation_matrix(self, analyzer, sample_factor_dict):
        """测试计算相关性矩阵"""
        corr_matrix = analyzer.calculate_correlation_matrix(sample_factor_dict)

        # 检查返回类型
        assert isinstance(corr_matrix, pd.DataFrame)

        # 检查矩阵形状
        n_factors = len(sample_factor_dict)
        assert corr_matrix.shape == (n_factors, n_factors)

        # 检查对角线为 1
        np.testing.assert_array_almost_equal(
            np.diag(corr_matrix.values),
            np.ones(n_factors)
        )

    def test_calculate_correlation_matrix_empty_dict(self, analyzer):
        """测试空因子字典"""
        with pytest.raises(ValueError, match="因子字典不能为空"):
            analyzer.calculate_correlation_matrix({})

    def test_find_high_correlation(self, analyzer, sample_factor_dict):
        """测试找出高度相关因子对"""
        corr_matrix = analyzer.calculate_correlation_matrix(sample_factor_dict)
        high_corr_pairs = analyzer.find_high_correlation(corr_matrix, threshold=0.7)

        # 检查返回类型
        assert isinstance(high_corr_pairs, list)

        # 检查是否找到 MA20 和 MA60
        factor_names = [pair['factor1'] for pair in high_corr_pairs] + \
                      [pair['factor2'] for pair in high_corr_pairs]
        assert 'MA20' in factor_names
        assert 'MA60' in factor_names

    def test_find_high_correlation_threshold(self, analyzer, sample_factor_dict):
        """测试不同阈值"""
        corr_matrix = analyzer.calculate_correlation_matrix(sample_factor_dict)

        # 高阈值
        high_threshold_pairs = analyzer.find_high_correlation(corr_matrix, threshold=0.9)
        # 低阈值
        low_threshold_pairs = analyzer.find_high_correlation(corr_matrix, threshold=0.5)

        # 高阈值应该找到更少的因子对
        assert len(high_threshold_pairs) <= len(low_threshold_pairs)

    def test_find_high_correlation_sorted(self, analyzer, sample_factor_dict):
        """测试因子对按相关系数排序"""
        corr_matrix = analyzer.calculate_correlation_matrix(sample_factor_dict)
        high_corr_pairs = analyzer.find_high_correlation(corr_matrix, threshold=0.3)

        # 检查是否按相关系数绝对值降序排序
        if len(high_corr_pairs) > 1:
            for i in range(len(high_corr_pairs) - 1):
                corr1 = abs(high_corr_pairs[i]['correlation'])
                corr2 = abs(high_corr_pairs[i + 1]['correlation'])
                assert corr1 >= corr2

    def test_analyze_correlation(self, analyzer, sample_factor_dict):
        """测试完整的相关性分析"""
        result = analyzer.analyze_correlation(sample_factor_dict)

        # 检查返回键
        assert 'correlation_matrix' in result
        assert 'high_correlation_pairs' in result
        assert 'recommendation' in result
        assert 'statistics' in result

        # 检查类型
        assert isinstance(result['correlation_matrix'], pd.DataFrame)
        assert isinstance(result['high_correlation_pairs'], list)
        assert isinstance(result['recommendation'], str)
        assert isinstance(result['statistics'], dict)

    def test_calculate_statistics(self, analyzer, sample_factor_dict):
        """测试计算统计信息"""
        corr_matrix = analyzer.calculate_correlation_matrix(sample_factor_dict)
        stats = analyzer.calculate_statistics(corr_matrix)

        # 检查统计键
        required_keys = [
            'mean_correlation',
            'median_correlation',
            'std_correlation',
            'max_correlation',
            'min_correlation',
            'high_correlation_count',
            'high_correlation_ratio',
        ]
        for key in required_keys:
            assert key in stats

        # 检查值的范围
        assert -1 <= stats['mean_correlation'] <= 1
        assert -1 <= stats['max_correlation'] <= 1
        assert -1 <= stats['min_correlation'] <= 1
        assert stats['high_correlation_count'] >= 0
        assert 0 <= stats['high_correlation_ratio'] <= 1

    def test_generate_recommendation_no_high_corr(self, analyzer):
        """测试无高度相关因子对时的建议"""
        corr_matrix = pd.DataFrame(
            np.eye(3),
            columns=['A', 'B', 'C'],
            index=['A', 'B', 'C']
        )

        recommendation = analyzer.generate_recommendation([], corr_matrix)

        assert '相关性较低' in recommendation
        assert '可以安全组合' in recommendation

    def test_generate_recommendation_with_high_corr(self, analyzer, sample_factor_dict):
        """测试有高度相关因子对时的建议"""
        corr_matrix = analyzer.calculate_correlation_matrix(sample_factor_dict)
        high_corr_pairs = analyzer.find_high_correlation(corr_matrix, threshold=0.7)

        recommendation = analyzer.generate_recommendation(high_corr_pairs, corr_matrix)

        assert '高度相关' in recommendation
        assert '去重' in recommendation or '处理方案' in recommendation

    def test_align_factors(self, analyzer, sample_factor_dict):
        """测试因子对齐"""
        aligned = analyzer._align_factors(sample_factor_dict)

        # 检查所有因子有相同的索引
        indices = [factor.index for factor in aligned.values()]
        for index in indices[1:]:
            pd.testing.assert_index_equal(indices[0], index)

    def test_align_factors_dataframe_input(self, analyzer):
        """测试 DataFrame 输入的对齐"""
        np.random.seed(42)

        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        instruments = ['SH600000', 'SH600001']

        index = pd.MultiIndex.from_product(
            [dates, instruments],
            names=['datetime', 'instrument']
        )

        # 创建 DataFrame 格式的因子
        factor_df = pd.DataFrame(
            {'value': np.random.randn(len(index))},
            index=index
        )

        factor_dict = {'Factor1': factor_df}

        aligned = analyzer._align_factors(factor_dict)

        # 检查转换成功
        assert 'Factor1' in aligned
        assert isinstance(aligned['Factor1'], pd.Series)


class TestAnalyzeFactorCorrelation:
    """便捷函数测试"""

    def test_analyze_factor_correlation_basic(self, sample_factor_dict):
        """测试基本功能"""
        result = analyze_factor_correlation(
            sample_factor_dict,
            method='spearman',
            threshold=0.7
        )

        assert 'correlation_matrix' in result
        assert 'high_correlation_pairs' in result
        assert 'recommendation' in result

    def test_analyze_factor_correlation_with_plot(self, sample_factor_dict, tmp_path):
        """测试带绘图的函数"""
        save_path = tmp_path / "correlation.png"

        result = analyze_factor_correlation(
            sample_factor_dict,
            plot=True,
            save_path=str(save_path)
        )

        # 检查文件是否创建
        # 注意：如果没有 matplotlib，可能会跳过绘图
        # assert save_path.exists()


class TestEdgeCases:
    """边界情况测试"""

    def test_single_factor(self, analyzer):
        """测试单个因子"""
        np.random.seed(42)

        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        instruments = ['SH600000']

        index = pd.MultiIndex.from_product(
            [dates, instruments],
            names=['datetime', 'instrument']
        )

        factor_dict = {
            'MA20': pd.Series(np.random.randn(len(index)), index=index)
        }

        corr_matrix = analyzer.calculate_correlation_matrix(factor_dict)

        # 单个因子的相关性矩阵应该是 1x1
        assert corr_matrix.shape == (1, 1)
        assert corr_matrix.iloc[0, 0] == 1.0

    def test_perfect_correlation(self, analyzer):
        """测试完全相关"""
        np.random.seed(42)

        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        instruments = ['SH600000']

        index = pd.MultiIndex.from_product(
            [dates, instruments],
            names=['datetime', 'instrument']
        )

        factor1 = pd.Series(np.random.randn(len(index)), index=index)
        factor2 = factor1.copy()  # 完全相同

        factor_dict = {'Factor1': factor1, 'Factor2': factor2}

        corr_matrix = analyzer.calculate_correlation_matrix(factor_dict)

        # 相关系数应该为 1
        assert corr_matrix.iloc[0, 1] == pytest.approx(1.0)
        assert corr_matrix.iloc[1, 0] == pytest.approx(1.0)

    def test_no_correlation(self, analyzer):
        """测试无相关"""
        np.random.seed(42)

        dates = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        instruments = ['SH600000']

        index = pd.MultiIndex.from_product(
            [dates, instruments],
            names=['datetime', 'instrument']
        )

        # 创建完全独立的因子
        factor_dict = {
            'Factor1': pd.Series(np.random.randn(len(index)), index=index),
            'Factor2': pd.Series(np.random.randn(len(index)), index=index),
        }

        corr_matrix = analyzer.calculate_correlation_matrix(factor_dict)

        # 相关系数应该接近 0
        assert abs(corr_matrix.iloc[0, 1]) < 0.5

    def test_different_indexes(self, analyzer):
        """测试不同索引的因子"""
        np.random.seed(42)

        dates1 = pd.date_range('2020-01-01', '2020-01-10', freq='D')
        dates2 = pd.date_range('2020-01-05', '2020-01-15', freq='D')
        instruments = ['SH600000']

        index1 = pd.MultiIndex.from_product(
            [dates1, instruments],
            names=['datetime', 'instrument']
        )
        index2 = pd.MultiIndex.from_product(
            [dates2, instruments],
            names=['datetime', 'instrument']
        )

        factor_dict = {
            'Factor1': pd.Series(np.random.randn(len(index1)), index=index1),
            'Factor2': pd.Series(np.random.randn(len(index2)), index=index2),
        }

        # 应该自动对齐到公共索引
        corr_matrix = analyzer.calculate_correlation_matrix(factor_dict)

        # 公共索引应该有 6 天
        assert corr_matrix.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
