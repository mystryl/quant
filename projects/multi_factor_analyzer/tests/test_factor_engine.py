"""因子管理器测试

测试 FactorManager 的注册、计算、缓存和管理功能。
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

from core.factor_engine import FactorManager


@pytest.fixture
def mock_provider():
    """创建模拟数据提供者"""
    provider = MagicMock()
    return provider


@pytest.fixture
def manager(mock_provider):
    """创建 FactorManager 实例（禁用缓存）"""
    with patch('core.factor_engine.FactorExpressionParser'):
        mgr = FactorManager(mock_provider, cache_enabled=False)
    return mgr


def mock_factor_func(provider, instruments, start_date, end_date, **kwargs):
    """模拟因子计算函数"""
    dates = pd.date_range(start_date, end_date, freq='B')
    idx = pd.MultiIndex.from_product(
        [instruments, dates], names=['instrument', 'datetime']
    )
    return pd.DataFrame({'factor': np.random.randn(len(idx))}, index=idx)


class TestFactorManagerInit:

    def test_init(self, mock_provider):
        """测试基本初始化"""
        with patch('core.factor_engine.FactorExpressionParser') as MockParser:
            mgr = FactorManager(mock_provider, cache_enabled=False)
            assert mgr.data_provider is mock_provider
            assert mgr.cache_enabled is False
            assert mgr.factors == {}
            assert mgr.factor_cache == {}
            MockParser.assert_called_once()

    def test_init_with_cache_dir(self, mock_provider):
        """测试自定义缓存目录"""
        with patch('core.factor_engine.FactorExpressionParser'):
            mgr = FactorManager(
                mock_provider, cache_enabled=False, cache_dir='/tmp/test_cache_fe'
            )
            assert str(mgr.cache_dir) == '/tmp/test_cache_fe'


class TestFactorManagerRegistration:

    def test_register_expression_factor(self, manager):
        """测试注册表达式因子"""
        with patch.object(
            manager.expr_parser, 'validate_no_future_functions', return_value=True
        ):
            manager.register_factor(
                'MA20', 'Ref($close, 20) / $close - 1'
            )

        assert 'MA20' in manager.factors
        assert manager.factors['MA20']['type'] == 'expression'
        assert manager.factors['MA20']['definition'] == 'Ref($close, 20) / $close - 1'

    def test_register_function_factor(self, manager):
        """测试注册函数因子"""
        manager.register_factor('custom', mock_factor_func)

        assert 'custom' in manager.factors
        assert manager.factors['custom']['type'] == 'function'
        assert callable(manager.factors['custom']['definition'])

    def test_register_duplicate_raises(self, manager):
        """测试重复注册抛出 ValueError"""
        with patch.object(
            manager.expr_parser, 'validate_no_future_functions', return_value=True
        ):
            manager.register_factor('MA20', 'Ref($close, 20)')

        with patch.object(
            manager.expr_parser, 'validate_no_future_functions', return_value=True
        ):
            with pytest.raises(ValueError, match='已存在'):
                manager.register_factor('MA20', 'Ref($close, 10)')

    def test_register_invalid_type_raises(self, manager):
        """测试不支持的类型抛出 TypeError"""
        with pytest.raises(TypeError, match='不支持'):
            manager.register_factor('bad', 12345)

    def test_register_with_metadata(self, manager):
        """测试带元数据注册"""
        meta = {'description': '20日均线偏离度', 'author': 'test'}
        with patch.object(
            manager.expr_parser, 'validate_no_future_functions', return_value=True
        ):
            manager.register_factor('MA20_meta', '$close', metadata=meta)

        assert manager.factors['MA20_meta']['metadata'] == meta


class TestFactorManagerUnregister:

    def test_unregister_factor(self, manager):
        """测试删除已注册的因子"""
        manager.register_factor('temp', mock_factor_func)
        assert 'temp' in manager.factors

        manager.unregister_factor('temp')
        assert 'temp' not in manager.factors

    def test_unregister_nonexistent_raises(self, manager):
        """测试删除不存在的因子抛出 KeyError"""
        with pytest.raises(KeyError, match='不存在'):
            manager.unregister_factor('nonexistent')


class TestFactorManagerCalculation:

    def test_calculate_function_factor(self, manager):
        """测试计算函数因子"""
        manager.register_factor('my_factor', mock_factor_func)

        result = manager.calculate_factor(
            'my_factor',
            instruments=['SH600000'],
            start_date='2024-01-01',
            end_date='2024-01-10'
        )

        assert isinstance(result, pd.DataFrame)
        assert 'factor' in result.columns
        assert isinstance(result.index, pd.MultiIndex)

    def test_calculate_nonexistent_raises(self, manager):
        """测试计算未注册的因子抛出 KeyError"""
        with pytest.raises(KeyError, match='未注册'):
            manager.calculate_factor(
                'nonexistent',
                instruments=['SH600000'],
                start_date='2024-01-01',
                end_date='2024-01-10'
            )

    def test_calculate_function_factor_multiple_instruments(self, manager):
        """测试多合约函数因子计算"""
        manager.register_factor('my_factor', mock_factor_func)

        result = manager.calculate_factor(
            'my_factor',
            instruments=['SH600000', 'SH600001', 'SH600002'],
            start_date='2024-01-01',
            end_date='2024-01-05'
        )

        instruments_in_result = result.index.get_level_values('instrument').unique()
        assert len(instruments_in_result) == 3


class TestFactorManagerListing:

    def test_list_factors(self, manager):
        """测试列出所有因子"""
        with patch.object(
            manager.expr_parser, 'validate_no_future_functions', return_value=True
        ):
            manager.register_factor('MA20', '$close', metadata={'desc': 'ma'})
        manager.register_factor('custom', mock_factor_func)

        factors = manager.list_factors()

        assert len(factors) == 2
        names = [f['name'] for f in factors]
        assert 'MA20' in names
        assert 'custom' in names

        ma20_info = next(f for f in factors if f['name'] == 'MA20')
        assert ma20_info['type'] == 'expression'
        custom_info = next(f for f in factors if f['name'] == 'custom')
        assert custom_info['type'] == 'function'

    def test_list_factors_empty(self, manager):
        """测试无因子时列出"""
        factors = manager.list_factors()
        assert factors == []


class TestFactorManagerInfo:

    def test_get_factor_info(self, manager):
        """测试获取因子详细信息"""
        meta = {'description': 'test factor'}
        manager.register_factor('info_test', mock_factor_func, metadata=meta)

        info = manager.get_factor_info('info_test')

        assert info['type'] == 'function'
        assert info['metadata'] == meta
        assert callable(info['definition'])

    def test_get_factor_info_nonexistent_raises(self, manager):
        """测试获取不存在因子信息抛出 KeyError"""
        with pytest.raises(KeyError, match='不存在'):
            manager.get_factor_info('nonexistent')


class TestFactorManagerCache:

    def test_cache_key_generation(self, manager):
        """测试缓存键生成的确定性"""
        key1 = manager._generate_cache_key(
            'MA20', ['SH600001', 'SH600000'], '2024-01-01', '2024-01-10'
        )
        key2 = manager._generate_cache_key(
            'MA20', ['SH600000', 'SH600001'], '2024-01-01', '2024-01-10'
        )

        # 相同的合约集合（不同顺序）应生成相同的键
        assert key1 == key2
        assert key1 == 'MA20_SH600000_SH600001_2024-01-01_2024-01-10'

    def test_cache_key_different_params(self, manager):
        """测试不同参数生成不同缓存键"""
        key1 = manager._generate_cache_key(
            'MA20', ['SH600000'], '2024-01-01', '2024-01-10'
        )
        key2 = manager._generate_cache_key(
            'MA20', ['SH600001'], '2024-01-01', '2024-01-10'
        )
        key3 = manager._generate_cache_key(
            'MA60', ['SH600000'], '2024-01-01', '2024-01-10'
        )

        assert key1 != key2
        assert key1 != key3
        assert key2 != key3

    def test_calculate_with_cache(self, mock_provider):
        """测试缓存机制"""
        with patch('core.factor_engine.FactorExpressionParser'):
            mgr = FactorManager(mock_provider, cache_enabled=True)

        mgr.register_factor('cached', mock_factor_func)

        # 第一次计算
        result1 = mgr.calculate_factor(
            'cached',
            instruments=['SH600000'],
            start_date='2024-01-01',
            end_date='2024-01-05',
            use_cache=True
        )

        # 缓存应该存在
        cache_key = mgr._generate_cache_key(
            'cached', ['SH600000'], '2024-01-01', '2024-01-05'
        )
        assert cache_key in mgr.factor_cache

        # 第二次计算应从缓存获取
        result2 = mgr.calculate_factor(
            'cached',
            instruments=['SH600000'],
            start_date='2024-01-01',
            end_date='2024-01-05',
            use_cache=True
        )

        pd.testing.assert_frame_equal(result1, result2)


class TestFactorManagerBatch:

    def test_calculate_batch_factors(self, manager):
        """测试批量计算因子"""
        manager.register_factor('f1', mock_factor_func)
        manager.register_factor('f2', mock_factor_func)

        results = manager.calculate_batch_factors(
            ['f1', 'f2'],
            instruments=['SH600000'],
            start_date='2024-01-01',
            end_date='2024-01-05'
        )

        assert 'f1' in results
        assert 'f2' in results
        assert isinstance(results['f1'], pd.DataFrame)
        assert isinstance(results['f2'], pd.DataFrame)


class TestFactorManagerClearCache:

    def test_clear_all_cache(self, mock_provider):
        """测试清除所有缓存"""
        with patch('core.factor_engine.FactorExpressionParser'):
            mgr = FactorManager(mock_provider, cache_enabled=True)

        mgr.register_factor('f1', mock_factor_func)
        mgr.calculate_factor('f1', ['SH600000'], '2024-01-01', '2024-01-05')

        assert len(mgr.factor_cache) > 0
        mgr.clear_cache()
        assert len(mgr.factor_cache) == 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
