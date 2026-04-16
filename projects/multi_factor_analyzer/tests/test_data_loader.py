"""数据加载器测试

测试 DataLoader 的加载、预处理和质量检查功能。
"""

import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

# 添加 src 目录到路径
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

from data.loader import DataLoader


@pytest.fixture
def sample_df():
    """创建示例数据"""
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
    return pd.DataFrame({
        'open': [100 + i * 0.5 for i in range(len(dates))],
        'high': [101 + i * 0.5 for i in range(len(dates))],
        'low': [99 + i * 0.5 for i in range(len(dates))],
        'close': [100.5 + i * 0.5 for i in range(len(dates))],
        'volume': [10000 + i * 100 for i in range(len(dates))],
    }, index=dates)


@pytest.fixture
def sample_df_with_missing():
    """创建含缺失值的数据"""
    dates = pd.date_range('2024-01-01', '2024-01-10', freq='D')
    df = pd.DataFrame({
        'close': [100, 101, np.nan, 103, np.nan, 105, 106, 107, 108, 109],
        'volume': [1000, np.nan, 1200, 1300, 1400, 1500, np.nan, 1700, 1800, 1900],
    }, index=dates)
    return df


@pytest.fixture
def mock_provider(sample_df):
    """创建模拟数据提供者"""
    provider = MagicMock()
    provider.get_factor_data.return_value = sample_df.copy()
    return provider


@pytest.fixture
def loader(mock_provider):
    """创建 DataLoader 实例"""
    loader = DataLoader(data_provider=mock_provider, fill_method='none')
    return loader


class TestDataLoaderInit:

    def test_init_default(self):
        """测试默认参数初始化"""
        with patch('data.loader.FactorDataProvider') as MockProvider:
            loader = DataLoader(fill_method='none')
            MockProvider.assert_called_once()

        assert loader.fill_method == 'none'
        assert loader.missing_threshold == 0.1
        assert loader.zero_threshold == 0.5

    def test_init_invalid_fill_method(self):
        """测试无效的 fill_method 抛出 ValueError"""
        with pytest.raises(ValueError, match='无效的 fill_method'):
            DataLoader(fill_method='invalid_method')


class TestDataLoaderLoad:

    def test_load_data_single(self, loader, sample_df):
        """测试加载单个合约数据"""
        data = loader.load_data(
            instruments='HC8888.XSGE',
            start_date='2024-01-01',
            end_date='2024-01-10',
            check_quality=False
        )

        assert isinstance(data, pd.DataFrame)
        assert len(data) == len(sample_df)
        loader.data_provider.get_factor_data.assert_called_once()

    def test_load_data_empty_raises(self, mock_provider):
        """测试空数据抛出 ValueError"""
        mock_provider.get_factor_data.return_value = pd.DataFrame()
        loader = DataLoader(data_provider=mock_provider, fill_method='none')

        with pytest.raises(ValueError, match='数据为空'):
            loader.load_data(
                instruments='HC8888.XSGE',
                start_date='2024-01-01',
                end_date='2024-01-10',
                check_quality=False
            )

    def test_load_data_with_fields(self, loader):
        """测试指定字段加载数据"""
        loader.load_data(
            instruments='HC8888.XSGE',
            start_date='2024-01-01',
            end_date='2024-01-10',
            fields=['close', 'volume'],
            check_quality=False
        )

        call_kwargs = loader.data_provider.get_factor_data.call_args
        assert call_kwargs[1]['fields'] == ['close', 'volume']


class TestDataLoaderFill:

    def test_fill_missing_ffill(self, sample_df_with_missing):
        """测试前向填充缺失值"""
        loader = DataLoader(fill_method='none')
        filled = loader._fill_missing_values(sample_df_with_missing.copy(), 'ffill')

        assert filled['close'].isnull().sum() == 0
        assert filled['volume'].isnull().sum() == 0

        # 验证前向填充逻辑: NaN at index 2 should be filled with 101
        assert filled['close'].iloc[2] == 101.0

    def test_fill_missing_interpolate(self, sample_df_with_missing):
        """测试线性插值填充缺失值"""
        loader = DataLoader(fill_method='none')
        filled = loader._fill_missing_values(
            sample_df_with_missing.copy(), 'interpolate'
        )

        assert filled['close'].isnull().sum() == 0
        assert filled['volume'].isnull().sum() == 0

        # 验证插值逻辑: NaN at index 2 should be (101 + 103) / 2 = 102
        assert filled['close'].iloc[2] == pytest.approx(102.0)

    def test_fill_missing_drop(self, sample_df_with_missing):
        """测试删除缺失值行"""
        loader = DataLoader(fill_method='none')
        filled = loader._fill_missing_values(
            sample_df_with_missing.copy(), 'drop'
        )

        assert filled['close'].isnull().sum() == 0
        assert len(filled) < len(sample_df_with_missing)


class TestDataLoaderQuality:

    def test_check_data_quality(self, sample_df):
        """测试数据质量检查通过正常数据"""
        loader = DataLoader(fill_method='none')

        # 正常数据应不抛异常
        loader._check_data_quality(sample_df)

    def test_check_data_quality_high_low_error(self):
        """测试 high < low 时抛出 ValueError"""
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        bad_df = pd.DataFrame({
            'high': [100, 101, 90, 103, 104],
            'low': [99, 100, 95, 102, 103],
        }, index=dates)

        loader = DataLoader(fill_method='none')

        with pytest.raises(ValueError, match='high < low'):
            loader._check_data_quality(bad_df)

    def test_check_data_quality_no_error_good_data(self):
        """测试正常 OHLC 数据不抛异常"""
        dates = pd.date_range('2024-01-01', '2024-01-05', freq='D')
        good_df = pd.DataFrame({
            'open': [100, 101, 102, 103, 104],
            'high': [102, 103, 104, 105, 106],
            'low': [99, 100, 101, 102, 103],
            'close': [101, 102, 103, 104, 105],
        }, index=dates)

        loader = DataLoader(fill_method='none')
        loader._check_data_quality(good_df)  # should not raise


class TestDataLoaderResample:

    def test_resample_data(self, sample_df):
        """测试重采样到日线"""
        # 创建更高频率的数据用于测试
        dates = pd.date_range('2024-01-01', '2024-01-07', freq='h')
        hourly_df = pd.DataFrame({
            'open': [100 + i * 0.1 for i in range(len(dates))],
            'high': [100.5 + i * 0.1 for i in range(len(dates))],
            'low': [99.5 + i * 0.1 for i in range(len(dates))],
            'close': [100.2 + i * 0.1 for i in range(len(dates))],
            'volume': [1000 for _ in range(len(dates))],
        }, index=dates)

        loader = DataLoader(fill_method='none')
        daily = loader.resample_data(hourly_df, freq='1D')

        assert isinstance(daily, pd.DataFrame)
        assert len(daily) <= len(hourly_df)
        assert 'close' in daily.columns


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
