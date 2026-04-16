"""
周期对齐模块测试
"""

import unittest
import numpy as np
import pandas as pd
import sys
import os

# 添加项目路径
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.core.cycle_aligner import CycleAligner, align_factor_returns


class TestCycleAligner(unittest.TestCase):
    """周期对齐器测试"""

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

        self.aligner = CycleAligner()

    def test_default_alignment(self):
        """测试默认对齐方式"""
        factor_aligned, returns_aligned = self.aligner.align(
            self.factor_df,
            self.price_df,
            method='default'
        )

        # 检查形状
        self.assertEqual(factor_aligned.shape[0], len(self.factor_df) - 2)
        self.assertEqual(returns_aligned.shape[0], len(self.factor_df) - 2)

        # 检查索引对齐
        self.assertTrue(factor_aligned.index.equals(returns_aligned.index))

    def test_flexible_alignment(self):
        """测试灵活对齐方式"""
        for shift in [1, 2, 5]:
            factor_aligned, returns_aligned = self.aligner.align(
                self.factor_df,
                self.price_df,
                method='flexible',
                shift=shift
            )

            # 检查形状
            expected_length = len(self.factor_df) - shift
            self.assertEqual(factor_aligned.shape[0], expected_length)
            self.assertEqual(returns_aligned.shape[0], expected_length)

    def test_auto_alignment(self):
        """测试自动检测对齐方式"""
        factor_aligned, returns_aligned, best_shift = self.aligner.align(
            self.factor_df,
            self.price_df,
            method='auto',
            auto_search_range=(1, 5)
        )

        # 检查返回了最佳偏移量
        self.assertIsInstance(best_shift, int)
        self.assertGreaterEqual(best_shift, 1)
        self.assertLessEqual(best_shift, 5)

        # 检查数据对齐
        self.assertTrue(factor_aligned.index.equals(returns_aligned.index))

    def test_validation(self):
        """测试对齐验证功能"""
        factor_aligned, returns_aligned = self.aligner.align(
            self.factor_df,
            self.price_df,
            method='default'
        )

        validation = self.aligner.validate_alignment(
            factor_aligned,
            returns_aligned
        )

        # 检查验证结果
        self.assertIn('is_valid', validation)
        self.assertIn('statistics', validation)
        self.assertIn('warnings', validation)
        self.assertIn('errors', validation)

    def test_ic_calculation(self):
        """测试 IC 计算"""
        factor_aligned, returns_aligned = self.aligner.align(
            self.factor_df,
            self.price_df,
            method='default'
        )

        # 计算 Pearson IC
        ic_pearson = self.aligner._calculate_ic(
            factor_aligned,
            returns_aligned,
            method='pearson'
        )

        # 计算 Spearman IC
        ic_spearman = self.aligner._calculate_ic(
            factor_aligned,
            returns_aligned,
            method='spearman'
        )

        # 检查返回类型
        self.assertIsInstance(ic_pearson, pd.Series)
        self.assertIsInstance(ic_spearman, pd.Series)

        # 检查长度
        n_dates = len(factor_aligned.index.get_level_values('datetime').unique())
        self.assertEqual(len(ic_pearson), n_dates)
        self.assertEqual(len(ic_spearman), n_dates)

    def test_alignment_summary(self):
        """测试对齐摘要"""
        summary = self.aligner.get_alignment_summary(
            self.factor_df,
            self.price_df,
            shift=2
        )

        # 检查摘要内容
        self.assertIn('shift', summary)
        self.assertIn('description', summary)
        self.assertIn('original_dates', summary)
        self.assertIn('aligned_dates', summary)
        self.assertIn('data_loss', summary)

        # 检查值
        self.assertEqual(summary['shift'], 2)
        self.assertEqual(summary['description'], 'T+1 to T+2')

    def test_invalid_method(self):
        """测试不合法的对齐方法"""
        with self.assertRaises(ValueError):
            self.aligner.align(
                self.factor_df,
                self.price_df,
                method='invalid_method'
            )

    def test_invalid_shift(self):
        """测试不合法的偏移量"""
        with self.assertRaises(ValueError):
            self.aligner.align(
                self.factor_df,
                self.price_df,
                method='flexible',
                shift=0
            )

    def test_convenient_function(self):
        """测试便捷函数"""
        factor_aligned, returns_aligned = align_factor_returns(
            self.factor_df,
            self.price_df,
            method='default'
        )

        # 检查返回类型
        self.assertIsInstance(factor_aligned, pd.DataFrame)
        self.assertIsInstance(returns_aligned, pd.DataFrame)

        # 检查索引对齐
        self.assertTrue(factor_aligned.index.equals(returns_aligned.index))


if __name__ == '__main__':
    unittest.main()
