"""数据验证器测试

测试 DataValidator 的质量检查、标准化、中性化和去极值功能。
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

# 添加 src 目录到路径
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from data.validator import DataValidator


@pytest.fixture
def sample_data():
    """创建示例数据"""
    return pd.DataFrame({
        'close': [100, 101, 102, 103, 104, 105, 106, 107, 108, 109],
        'volume': [1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900],
        'high': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
        'low': [99, 100, 101, 102, 103, 104, 105, 106, 107, 108],
    })


@pytest.fixture
def validator():
    """创建 DataValidator 实例"""
    return DataValidator()


class TestDataValidatorInit:

    def test_init(self):
        """测试默认初始化"""
        v = DataValidator()
        assert v.nan_threshold == 0.1
        assert v.outlier_method == 'zscore'
        assert v.outlier_threshold == 3.0

    def test_init_custom(self):
        """测试自定义参数初始化"""
        v = DataValidator(nan_threshold=0.2, outlier_method='iqr', outlier_threshold=1.5)
        assert v.nan_threshold == 0.2
        assert v.outlier_method == 'iqr'
        assert v.outlier_threshold == 1.5


class TestDataValidatorQualityCheck:

    def test_check_data_quality_basic(self, validator, sample_data):
        """测试基本数据质量检查"""
        report = validator.check_data_quality(sample_data, return_report=True)

        assert report['total_rows'] == 10
        assert report['total_columns'] == 4
        assert 'missing_values' in report
        assert 'outliers' in report
        assert 'statistics' in report
        assert 'warnings' in report
        assert 'errors' in report
        assert len(report['errors']) == 0

    def test_check_data_quality_with_missing(self, validator):
        """测试含缺失值的数据质量检查"""
        data = pd.DataFrame({
            'close': [100, 101, np.nan, 103, np.nan, 105],
            'volume': [1000, 1100, 1200, 1300, 1400, 1500],
        })

        report = validator.check_data_quality(data)

        assert 'close' in report['missing_values']
        assert report['missing_values']['close']['count'] == 2
        assert report['missing_values']['close']['ratio'] == pytest.approx(2 / 6)

    def test_check_data_quality_high_low_error(self, validator):
        """测试 high < low 时产生错误"""
        data = pd.DataFrame({
            'high': [100, 101, 90, 103, 104],
            'low': [99, 100, 95, 102, 103],
        })

        report = validator.check_data_quality(data)

        # 应有错误报告
        assert len(report['errors']) > 0
        assert any('high < low' in e for e in report['errors'])

    def test_check_data_quality_no_return(self, validator, sample_data):
        """测试不返回报告"""
        result = validator.check_data_quality(sample_data, return_report=False)
        assert result is None


class TestDataValidatorStandardize:

    def test_standardize_zscore(self, validator):
        """测试 z-score 标准化"""
        factor = pd.Series([1, 2, 3, 4, 5, 100], dtype=float)
        standardized = validator.standardize_factor(factor, method='zscore')

        # 标准化后均值应接近 0
        assert standardized.mean() == pytest.approx(0.0, abs=1e-10)
        # 标准化后标准差应接近 1
        assert standardized.std() == pytest.approx(1.0, abs=0.1)

    def test_standardize_minmax(self, validator):
        """测试 min-max 标准化"""
        factor = pd.Series([1, 2, 3, 4, 5, 100], dtype=float)
        normalized = validator.standardize_factor(factor, method='minmax')

        # min-max 后最小值应为 0，最大值应为 1
        assert normalized.min() == pytest.approx(0.0)
        assert normalized.max() == pytest.approx(1.0)

    def test_standardize_rank(self, validator):
        """测试秩标准化"""
        factor = pd.Series([10, 30, 20, 50, 40], dtype=float)
        ranked = validator.standardize_factor(factor, method='rank')

        # 秩标准化后最小值应为 0，最大值应为 1
        assert ranked.min() == pytest.approx(0.0)
        assert ranked.max() == pytest.approx(1.0)
        # 秩应为单调递增（原始值从小到大排序后映射）
        assert ranked.iloc[0] < ranked.iloc[1]

    def test_standardize_robust(self, validator):
        """测试鲁棒标准化"""
        factor = pd.Series([1, 2, 3, 4, 5, 100], dtype=float)
        robust = validator.standardize_factor(factor, method='robust')

        # 鲁棒标准化应基于中位数和 IQR
        # 中位数附近的值应接近 0
        median_factor = factor.median()
        iqr_factor = factor.quantile(0.75) - factor.quantile(0.25)
        expected_median = (median_factor - median_factor) / iqr_factor
        assert robust.median() == pytest.approx(expected_median, abs=0.01)

    def test_standardize_invalid_method(self, validator):
        """测试无效标准化方法抛出 ValueError"""
        factor = pd.Series([1, 2, 3, 4, 5])
        with pytest.raises(ValueError, match='无效的 method'):
            validator.standardize_factor(factor, method='invalid')

    def test_standardize_with_nan(self, validator):
        """测试含 NaN 的标准化"""
        factor = pd.Series([1, np.nan, 3, 4, 5], dtype=float)
        standardized = validator.standardize_factor(factor, method='zscore')

        # NaN 位置应保持为 NaN
        assert pd.isna(standardized.iloc[1])
        # 有效值应被标准化
        valid = standardized.dropna()
        assert valid.mean() == pytest.approx(0.0, abs=1e-10)

    def test_standardize_clip(self, validator):
        """测试标准化后裁剪"""
        factor = pd.Series([1, 2, 3, 4, 5, 100], dtype=float)
        clipped = validator.standardize_factor(
            factor, method='zscore', clip_range=(-3, 3)
        )

        assert clipped.min() >= -3.0
        assert clipped.max() <= 3.0


class TestDataValidatorNeutralize:

    def test_neutralize_factor_market_cap(self, validator):
        """测试市值中性化"""
        np.random.seed(42)
        n = 100
        factor = pd.Series(np.random.randn(n))
        market_cap = pd.Series(np.random.uniform(100, 10000, n))

        neutralized = validator.neutralize_factor(
            factor=factor, market_cap=market_cap
        )

        # 中性化后因子与对数市值的相关性应降低
        log_mcap = np.log(market_cap)
        orig_corr = np.abs(np.corrcoef(factor, log_mcap)[0, 1])
        neu_corr = np.abs(np.corrcoef(neutralized, log_mcap)[0, 1])

        assert neu_corr <= orig_corr + 1e-10

    def test_neutralize_factor_industry(self, validator):
        """测试行业中性化"""
        factor = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        industry = pd.Series(['A', 'A', 'B', 'B', 'C', 'C'])

        neutralized = validator.neutralize_factor(
            factor=factor, industry=industry
        )

        # 行业中性化后，同行业因子的和应为 0
        for ind in ['A', 'B', 'C']:
            mask = industry == ind
            group_sum = neutralized[mask].sum()
            assert group_sum == pytest.approx(0.0, abs=1e-10)


class TestDataValidatorWinsorize:

    def test_winsorize_factor(self, validator):
        """测试去极值 - 使用足够的比例确保有效"""
        factor = pd.Series([1, 2, 3, 4, 5, 100, -50], dtype=float)
        # 使用 20% 的比例确保至少裁剪 1 个值（7 * 0.2 = 1.4）
        winsorized = validator.winsorize_factor(factor, limits=(0.2, 0.2))

        # 去极值后范围应比原始更小
        assert winsorized.max() < factor.max()
        assert winsorized.min() > factor.min()

    def test_winsorize_factor_preserves_length(self, validator):
        """测试去极值保持长度"""
        factor = pd.Series([1, 2, 3, 4, 5, 100], dtype=float)
        winsorized = validator.winsorize_factor(factor, limits=(0.1, 0.1))

        assert len(winsorized) == len(factor)

    def test_winsorize_factor_with_nan(self, validator):
        """测试含 NaN 的去极值"""
        factor = pd.Series([1, np.nan, 3, 4, 5, 100], dtype=float)
        # 使用 20% 的比例确保至少裁剪 1 个值（5 valid * 0.2 = 1）
        winsorized = validator.winsorize_factor(factor, limits=(0.2, 0.2))

        assert pd.isna(winsorized.iloc[1])
        assert winsorized.max() < 100


class TestDataValidatorOutliers:

    def test_detect_outliers_zscore(self):
        """测试 z-score 异常值检测"""
        # 使用 20 个正常值 + 1 个极端异常值，确保 z-score > 3
        np.random.seed(42)
        normal = np.random.randn(20) * 0.5  # std=0.5, mean~0
        factor = pd.Series(np.append(normal, 50.0))

        validator = DataValidator(outlier_method='zscore', outlier_threshold=3.0)
        outliers = validator._detect_outliers(factor)

        assert isinstance(outliers, pd.Series)
        # 50.0 应远超 3 倍标准差
        assert outliers.sum() >= 1

    def test_detect_outliers_iqr(self):
        """测试 IQR 异常值检测"""
        validator = DataValidator(outlier_method='iqr', outlier_threshold=1.5)
        factor = pd.Series([1, 2, 3, 4, 5, 100], dtype=float)

        outliers = validator._detect_outliers(factor)

        assert isinstance(outliers, pd.Series)
        assert outliers.sum() >= 1  # 100 应被检测为异常值

    def test_detect_outliers_none(self):
        """测试不检测异常值"""
        validator = DataValidator(outlier_method='none')
        factor = pd.Series([1, 2, 3, 4, 5, 100], dtype=float)

        outliers = validator._detect_outliers(factor)

        assert outliers.sum() == 0
        assert all(outliers == False)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
