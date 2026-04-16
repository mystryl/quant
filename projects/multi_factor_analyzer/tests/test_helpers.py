"""
辅助函数模块单元测试

测试数据处理、时间序列、验证等辅助函数。
"""

import pytest
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.utils.helpers import (
    remove_outliers,
    normalize,
    neutralize,
    calculate_returns,
    calculate_forward_returns,
    validate_data_format,
    check_missing_values,
    check_infinite_values,
    calculate_ic,
    calculate_long_short_return,
    align_data,
    split_data,
)


class TestDataProcessing:
    """测试数据处理工具"""

    def test_remove_outliers_quantile(self):
        """测试基于分位数的异常值移除"""
        data = pd.Series([1, 2, 3, 100, 5])
        result = remove_outliers(data, method='quantile', lower_quantile=0.01, upper_quantile=0.99)

        # 100 应该被移除
        assert result[3] != 100
        assert pd.isna(result[3]) or result[3] < 100

    def test_remove_outliers_std(self):
        """测试基于标准差的异常值移除"""
        # 使用更容易识别为异常值的数据
        data = pd.Series([10, 11, 10, 12, 11, 100])
        result = remove_outliers(data, method='std', n_std=2.0)

        # 100 应该被移除（2倍标准差）
        assert result[5] != 100
        assert pd.isna(result[5]) or result[5] < 100

    def test_remove_outliers_iqr(self):
        """测试基于 IQR 的异常值移除"""
        data = pd.Series([1, 2, 3, 100, 5])
        result = remove_outliers(data, method='iqr')

        # 100 应该被移除
        assert result[3] != 100
        assert pd.isna(result[3]) or result[3] < 100

    def test_normalize_zscore(self):
        """测试 Z-score 标准化"""
        data = pd.Series([1, 2, 3, 4, 5])
        result = normalize(data, method='zscore')

        # 检查均值接近 0，标准差接近 1
        assert abs(result.mean()) < 1e-10
        assert abs(result.std() - 1.0) < 1e-10

    def test_normalize_minmax(self):
        """测试 Min-Max 标准化"""
        data = pd.Series([1, 2, 3, 4, 5])
        result = normalize(data, method='minmax')

        # 检查范围在 [0, 1]
        assert result.min() >= 0
        assert result.max() <= 1
        assert abs(result.min() - 0.0) < 1e-10
        assert abs(result.max() - 1.0) < 1e-10

    def test_normalize_rank(self):
        """测试排名标准化"""
        data = pd.Series([1, 2, 3, 4, 5])
        result = normalize(data, method='rank')

        # 检查范围在 [0, 1]
        assert result.min() >= 0
        assert result.max() <= 1

    def test_normalize_rolling_zscore(self):
        """测试滚动 Z-score 标准化"""
        data = pd.Series([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        result = normalize(data, method='rolling_zscore', window=3)

        # 检查长度相同
        assert len(result) == len(data)

        # 检查没有 NaN（除了可能的初始点）
        assert result.notna().sum() > 0


class TestTimeSeries:
    """测试时间序列工具"""

    def test_calculate_returns_simple(self):
        """测试简单收益率计算"""
        prices = pd.DataFrame({
            'A': [100, 102, 101, 103, 105],
            'B': [50, 51, 52, 51, 53]
        })

        returns = calculate_returns(prices, method='simple', periods=1)

        # 检查第一个值为 NaN
        assert pd.isna(returns.iloc[0, 0])

        # 检查收益率计算正确
        expected_return_a = (102 - 100) / 100
        assert abs(returns.iloc[1, 0] - expected_return_a) < 1e-10

    def test_calculate_returns_log(self):
        """测试对数收益率计算"""
        prices = pd.DataFrame({
            'A': [100, 102, 101, 103, 105],
            'B': [50, 51, 52, 51, 53]
        })

        returns = calculate_returns(prices, method='log', periods=1)

        # 检查对数收益率计算正确
        expected_return = np.log(102 / 100)
        assert abs(returns.iloc[1, 0] - expected_return) < 1e-10

    def test_calculate_forward_returns(self):
        """测试未来收益率计算"""
        dates = pd.date_range('2020-01-01', periods=5)
        prices = pd.DataFrame({
            'A': [100, 102, 101, 103, 105],
            'B': [50, 51, 52, 51, 53]
        }, index=dates)

        forward_returns = calculate_forward_returns(prices, forward_period=2)

        # 检查最后两个值为 NaN
        assert pd.isna(forward_returns.iloc[-1, 0])
        assert pd.isna(forward_returns.iloc[-2, 0])

        # 检查未来收益率计算正确
        # T+2 收益率 = (price[t+2] / price[t+1]) - 1
        expected_return = (101 / 102) - 1
        assert abs(forward_returns.iloc[0, 0] - expected_return) < 1e-10


class TestValidation:
    """测试验证工具"""

    def test_validate_data_format_correct(self):
        """测试正确的数据格式"""
        dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-01', '2020-01-02'])
        instruments = ['A', 'A', 'B', 'B']
        index = pd.MultiIndex.from_arrays([dates, instruments], names=['datetime', 'instrument'])

        data = pd.DataFrame({'value': [1, 2, 3, 4]}, index=index)

        # 应该不抛出异常
        validate_data_format(data, "测试数据")

    def test_validate_data_format_wrong_index(self):
        """测试错误的索引格式"""
        data = pd.DataFrame({'value': [1, 2, 3, 4]})

        with pytest.raises(ValueError):
            validate_data_format(data, "测试数据")

    def test_validate_data_format_empty(self):
        """测试空数据"""
        dates = pd.to_datetime([])
        instruments = []
        index = pd.MultiIndex.from_arrays([dates, instruments], names=['datetime', 'instrument'])

        data = pd.DataFrame({'value': []}, index=index)

        with pytest.raises(ValueError):
            validate_data_format(data, "测试数据")

    def test_check_missing_values(self):
        """检查缺失值检测"""
        data = pd.DataFrame({
            'A': [1, 2, np.nan, 4],
            'B': [1, np.nan, np.nan, 4]
        })

        stats = check_missing_values(data)

        # 检查统计信息
        assert stats['missing_count']['A'] == 1
        assert stats['missing_count']['B'] == 2
        assert stats['missing_rate']['A'] == 0.25
        assert stats['missing_rate']['B'] == 0.5

    def test_check_infinite_values(self):
        """检查无穷值检测"""
        data = pd.DataFrame({
            'A': [1, 2, np.inf, 4],
            'B': [1, -np.inf, 3, 4]
        })

        stats = check_infinite_values(data)

        # 检查统计信息
        assert stats['positive_infinite']['A'] == 1
        assert stats['negative_infinite']['B'] == 1
        assert stats['total_infinite']['A'] == 1
        assert stats['total_infinite']['B'] == 1


class TestPerformanceCalculation:
    """测试性能计算工具"""

    def setup_method(self):
        """设置测试数据"""
        # 创建测试数据
        dates = pd.to_datetime(['2020-01-01', '2020-01-02', '2020-01-03'] * 3)
        instruments = ['A', 'B', 'C'] * 3

        # 创建因子数据
        factor_values = [1.0, 2.0, 3.0, 2.0, 3.0, 1.0, 3.0, 1.0, 2.0]
        self.factor = pd.DataFrame({
            'factor': factor_values
        })
        self.factor.index = pd.MultiIndex.from_arrays(
            [dates, instruments],
            names=['datetime', 'instrument']
        )

        # 创建收益率数据
        return_values = [0.01, 0.02, 0.03, 0.02, 0.03, 0.01, 0.03, 0.01, 0.02]
        self.returns = pd.DataFrame({
            'return': return_values
        })
        self.returns.index = pd.MultiIndex.from_arrays(
            [dates, instruments],
            names=['datetime', 'instrument']
        )

    def test_calculate_ic(self):
        """测试 IC 计算"""
        ic, rank_ic = calculate_ic(self.factor, self.returns)

        # 检查返回 Series
        assert isinstance(ic, pd.Series)
        assert isinstance(rank_ic, pd.Series)

        # 检查长度（3个日期）
        assert len(ic) == 3
        assert len(rank_ic) == 3

    def test_calculate_long_short_return(self):
        """测试多空收益计算"""
        ls_return, avg_return = calculate_long_short_return(
            self.factor,
            self.returns,
            quantile=0.33
        )

        # 检查返回 Series
        assert isinstance(ls_return, pd.Series)
        assert isinstance(avg_return, pd.Series)

        # 检查长度
        assert len(ls_return) == 3
        assert len(avg_return) == 3

    def test_align_data(self):
        """测试数据对齐"""
        # 创建部分重叠的数据
        factor1 = self.factor.copy()
        factor2 = self.returns.copy()

        aligned1, aligned2 = align_data(factor1, factor2)

        # 检查对齐后的索引相同
        assert aligned1.index.equals(aligned2.index)

        # 检查长度相同
        assert len(aligned1) == len(aligned2)

    def test_split_data(self):
        """测试数据分割"""
        # 创建更多的数据点以支持分割
        dates = pd.to_datetime(pd.date_range('2020-01-01', periods=100).repeat(3))
        instruments = ['A', 'B', 'C'] * 100

        data = pd.DataFrame({
            'value': np.random.randn(300)
        })
        data.index = pd.MultiIndex.from_arrays(
            [dates, instruments],
            names=['datetime', 'instrument']
        )

        train, val, test = split_data(data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)

        # 检查分割比例
        total = len(data)
        assert abs(len(train) / total - 0.7) < 0.05
        assert abs(len(val) / total - 0.15) < 0.05
        assert abs(len(test) / total - 0.15) < 0.05


class TestNeutralize:
    """测试因子中性化"""

    def test_neutralize_industry(self):
        """测试行业中性化"""
        # 创建测试数据 - 每个日期每个工具只出现一次
        dates = pd.to_datetime(['2020-01-01'] * 4 + ['2020-01-02'] * 4)
        instruments = ['A', 'B', 'C', 'D', 'A', 'B', 'C', 'D']

        # 因子数据
        factor_values = [1.0, 2.0, 3.0, 4.0, 2.0, 3.0, 4.0, 5.0]
        factor = pd.DataFrame({'factor': factor_values})
        factor.index = pd.MultiIndex.from_arrays(
            [dates, instruments],
            names=['datetime', 'instrument']
        )

        # 行业数据
        industry_values = ['Tech', 'Tech', 'Finance', 'Finance',
                          'Tech', 'Tech', 'Finance', 'Finance']
        industry = pd.DataFrame({'industry': industry_values})
        industry.index = pd.MultiIndex.from_arrays(
            [dates, instruments],
            names=['datetime', 'instrument']
        )

        # 执行中性化 - 使用 orthogonal 方法（更简单）
        try:
            from sklearn.linear_model import LinearRegression
            result = neutralize(factor, industry=industry, method='orthogonal')

            # 检查返回 DataFrame
            assert isinstance(result, pd.DataFrame)
            assert result.shape == factor.shape
        except ImportError:
            # 如果 sklearn 不可用，跳过此测试
            pytest.skip("sklearn not available")


class TestEdgeCases:
    """测试边界情况"""

    def test_empty_data(self):
        """测试空数据"""
        data = pd.DataFrame()

        with pytest.raises(ValueError):
            validate_data_format(data, "空数据")

    def test_single_value(self):
        """测试单个值"""
        data = pd.Series([1.0])

        # 标准化应该返回 NaN（标准差为0）
        result = normalize(data, method='zscore')
        assert pd.isna(result.iloc[0])

    def test_all_same_values(self):
        """测试所有值相同"""
        data = pd.Series([1.0, 1.0, 1.0, 1.0])

        # Min-Max 标准化应该返回 NaN（范围为0）
        result = normalize(data, method='minmax')
        assert result.isna().all()


def run_tests():
    """运行所有测试"""
    print("=" * 80)
    print("运行辅助函数模块单元测试")
    print("=" * 80)

    # 运行 pytest
    pytest.main([__file__, "-v", "--tb=short"])


if __name__ == "__main__":
    run_tests()
