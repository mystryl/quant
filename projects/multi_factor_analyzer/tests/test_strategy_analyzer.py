"""
策略场景分析器测试
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.strategy_analyzer import StrategyAnalyzer, analyze_factor_strategies
from src.core.cycle_aligner import CycleAligner


class TestStrategyAnalyzer(unittest.TestCase):
    """策略场景分析器测试"""

    def setUp(self):
        """设置测试数据"""
        np.random.seed(42)

        # 生成测试数据
        n_dates = 100
        n_stocks = 20

        dates = pd.date_range('2024-01-01', periods=n_dates, freq='B')
        stocks = [f'STOCK_{i:04d}' for i in range(n_stocks)]

        index = pd.MultiIndex.from_product(
            [dates, stocks],
            names=['datetime', 'instrument']
        )

        # 生成价格数据
        returns = pd.DataFrame(
            np.random.normal(0.0005, 0.02, size=(n_dates, n_stocks)),
            index=dates,
            columns=stocks
        )

        prices = 100 * (1 + returns).cumprod()
        self.price_df = prices.stack().reindex(index).to_frame('close')

        # 生成因子数据
        factor = returns.rolling(window=10).mean().stack().reindex(index)
        self.factor_df = factor.to_frame('test_factor')

        # 生成市场指数
        market_returns = returns.mean(axis=1)
        self.market_index = (1 + market_returns).cumprod()

        # 对齐数据
        aligner = CycleAligner()
        self.factor_aligned, self.returns_aligned = aligner.align(
            self.factor_df,
            self.price_df,
            method='default'
        )

        self.analyzer = StrategyAnalyzer()

    def test_bull_strategy(self):
        """测试看涨策略分析"""
        result = self.analyzer.analyze_bull_strategy(
            self.factor_aligned,
            self.returns_aligned,
            top_pct=0.2
        )

        # 检查返回的指标
        self.assertIn('total_return', result)
        self.assertIn('annual_return', result)
        self.assertIn('sharpe_ratio', result)
        self.assertIn('max_drawdown', result)
        self.assertIn('win_rate', result)
        self.assertIn('calmar_ratio', result)
        self.assertIn('strategy_returns', result)

        # 检查类型
        self.assertIsInstance(result['strategy_returns'], pd.Series)

    def test_bear_strategy(self):
        """测试看跌策略分析"""
        result = self.analyzer.analyze_bear_strategy(
            self.factor_aligned,
            self.returns_aligned,
            top_pct=0.2
        )

        # 检查返回的指标
        self.assertIn('total_return', result)
        self.assertIn('annual_return', result)
        self.assertIn('sharpe_ratio', result)

    def test_long_short_strategy(self):
        """测试多空策略分析"""
        result = self.analyzer.analyze_long_short_strategy(
            self.factor_aligned,
            self.returns_aligned,
            top_pct=0.2
        )

        # 检查返回的指标
        self.assertIn('total_return', result)
        self.assertIn('long_returns', result)
        self.assertIn('short_returns', result)
        self.assertIn('strategy_returns', result)

        # 检查多空收益 = 多头 - 空头
        expected = result['long_returns'] - result['short_returns']
        pd.testing.assert_series_equal(
            result['strategy_returns'],
            expected,
            check_names=False
        )

    def test_volatility_strategy(self):
        """测试波动率策略分析"""
        result = self.analyzer.analyze_volatility_strategy(
            self.factor_aligned,
            self.returns_aligned,
            self.price_df,
            top_pct=0.2,
            position_method='inverse'
        )

        # 检查返回的指标
        self.assertIn('total_return', result)
        self.assertIn('strategy_returns', result)
        self.assertIn('base_returns', result)
        self.assertIn('position', result)
        self.assertIn('market_volatility', result)

    def test_market_regime(self):
        """测试牛熊市场景分析"""
        result = self.analyzer.analyze_market_regime(
            self.factor_aligned,
            self.returns_aligned,
            self.market_index,
            bull_threshold=0.0,
            window=20
        )

        # 检查返回的指标
        self.assertIn('牛市', result)
        self.assertIn('熊市', result)

        # 检查每个场景的指标
        for regime_name in ['牛市', '熊市']:
            self.assertIn('total_return', result[regime_name])
            self.assertIn('sharpe_ratio', result[regime_name])
            self.assertIn('win_rate', result[regime_name])
            self.assertIn('days', result[regime_name])

    def test_all_scenarios(self):
        """测试所有场景分析"""
        result = self.analyzer.analyze_all_scenarios(
            self.factor_aligned,
            self.returns_aligned,
            self.price_df,
            market_index=self.market_index,
            top_pct=0.2
        )

        # 检查返回的场景
        self.assertIn('bull', result)
        self.assertIn('bear', result)
        self.assertIn('long_short', result)
        self.assertIn('volatility', result)
        self.assertIn('market_regime', result)

    def test_strategy_metrics_calculation(self):
        """测试策略指标计算"""
        # 创建一个简单的收益序列
        returns = pd.Series([0.01, -0.005, 0.02, 0.015, -0.01])

        metrics = self.analyzer._calculate_strategy_metrics(returns)

        # 检查返回的指标
        self.assertIn('total_return', metrics)
        self.assertIn('annual_return', metrics)
        self.assertIn('sharpe_ratio', metrics)
        self.assertIn('max_drawdown', metrics)
        self.assertIn('win_rate', metrics)
        self.assertIn('calmar_ratio', metrics)

    def test_select_top_factor(self):
        """测试选股功能"""
        selected = self.analyzer._select_top_factor(
            self.factor_aligned,
            top_pct=0.2,
            ascending=False
        )

        # 检查返回类型
        self.assertIsInstance(selected, pd.DataFrame)

        # 检查列名
        self.assertIn('selected', selected.columns)

        # 检查值
        self.assertTrue(selected['selected'].dtype == bool)

    def test_calculate_strategy_returns(self):
        """测试策略收益计算"""
        # 创建一个简单的选股掩码
        selected = pd.DataFrame(
            False,
            index=self.factor_aligned.index,
            columns=['selected']
        )

        # 随机选择一些股票
        n_dates = len(self.factor_aligned.index.get_level_values('datetime').unique())
        for date in self.factor_aligned.index.get_level_values('datetime').unique()[:10]:
            daily_stocks = self.factor_aligned.loc[date].index
            for stock in daily_stocks[:5]:  # 选择前5个股票
                selected.loc[(date, stock), 'selected'] = True

        # 计算策略收益
        strategy_returns = self.analyzer._calculate_strategy_returns(
            selected,
            self.returns_aligned
        )

        # 检查返回类型
        self.assertIsInstance(strategy_returns, pd.Series)

    def test_convenient_function(self):
        """测试便捷函数"""
        result = analyze_factor_strategies(
            self.factor_aligned,
            self.returns_aligned,
            self.price_df,
            top_pct=0.2
        )

        # 检查返回的场景
        self.assertIn('bull', result)
        self.assertIn('bear', result)
        self.assertIn('long_short', result)


if __name__ == '__main__':
    unittest.main()
