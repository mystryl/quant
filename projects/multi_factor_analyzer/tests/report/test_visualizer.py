"""
测试可视化工具模块
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from src.report.visualizer import Visualizer, create_factor_charts


@pytest.fixture
def sample_metrics():
    """创建示例性能指标"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')

    return {
        'ic': pd.Series(np.random.randn(366) * 0.05 + 0.03, index=dates),
        'rank_ic': pd.Series(np.random.randn(366) * 0.06 + 0.04, index=dates),
        'ic_mean': 0.035,
        'ic_std': 0.052,
        'icir': 0.673,
        'rank_ic_mean': 0.042,
        'rank_ic_std': 0.061,
        'rank_icir': 0.689,
        'long_short_return': pd.Series(np.random.randn(366) * 0.01 + 0.001, index=dates),
        'annual_return': 0.085,
        'sharpe_ratio': 1.85,
        'max_drawdown': -0.12,
        'win_rate': 0.56
    }


@pytest.fixture
def sample_scenario_results():
    """创建示例策略场景结果"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')

    return {
        'bull': {
            'strategy_returns': pd.Series(np.random.randn(366) * 0.01 + 0.002, index=dates),
            'annual_return': 0.12,
            'sharpe_ratio': 2.1,
            'max_drawdown': -0.10,
            'win_rate': 0.58
        },
        'bear': {
            'strategy_returns': pd.Series(np.random.randn(366) * 0.01 - 0.001, index=dates),
            'annual_return': -0.02,
            'sharpe_ratio': -0.5,
            'max_drawdown': -0.15,
            'win_rate': 0.48
        },
        'long_short': {
            'strategy_returns': pd.Series(np.random.randn(366) * 0.01 + 0.001, index=dates),
            'annual_return': 0.085,
            'sharpe_ratio': 1.85,
            'max_drawdown': -0.12,
            'win_rate': 0.56
        }
    }


@pytest.fixture
def sample_cycle_analysis():
    """创建示例周期分析结果"""
    return {
        1: {'ic_mean': 0.025, 'icir': 0.45},
        2: {'ic_mean': 0.035, 'icir': 0.67},
        3: {'ic_mean': 0.028, 'icir': 0.52}
    }


class TestVisualizer:
    """测试 Visualizer 类"""

    def test_initialization(self, tmp_path):
        """测试初始化"""
        viz = Visualizer(output_dir=str(tmp_path))
        assert viz.output_dir == tmp_path
        assert viz.dpi == 100
        assert tmp_path.exists()

    def test_plot_ic_timeseries(self, sample_metrics):
        """测试绘制 IC 时间序列图"""
        viz = Visualizer()

        fig = viz.plot_ic_timeseries(
            sample_metrics['ic'],
            factor_name="TestFactor"
        )

        # 检查图形对象
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # 检查图形有轴
        axes = fig.get_axes()
        assert len(axes) > 0

        plt.close(fig)

    def test_plot_ic_distribution(self, sample_metrics):
        """测试绘制 IC 分布图"""
        viz = Visualizer()

        fig = viz.plot_ic_distribution(
            sample_metrics['ic'],
            factor_name="TestFactor"
        )

        # 检查图形对象
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_plot_cumulative_return(self, sample_scenario_results):
        """测试绘制累计收益曲线"""
        viz = Visualizer()

        fig = viz.plot_cumulative_return(
            sample_scenario_results['bull']['strategy_returns'],
            factor_name="TestFactor"
        )

        # 检查图形对象
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_plot_cumulative_return_with_benchmark(self, sample_scenario_results):
        """测试绘制带基准的累计收益曲线"""
        viz = Visualizer()

        benchmark = sample_scenario_results['bear']['strategy_returns']

        fig = viz.plot_cumulative_return(
            sample_scenario_results['bull']['strategy_returns'],
            factor_name="TestFactor",
            benchmark=benchmark
        )

        # 检查图形对象
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_plot_drawdown(self, sample_scenario_results):
        """测试绘制回撤图"""
        viz = Visualizer()

        fig = viz.plot_drawdown(
            sample_scenario_results['bull']['strategy_returns'],
            factor_name="TestFactor"
        )

        # 检查图形对象
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_plot_strategy_comparison(self, sample_scenario_results):
        """测试绘制策略对比图"""
        viz = Visualizer()

        fig = viz.plot_strategy_comparison(
            sample_scenario_results,
            factor_name="TestFactor"
        )

        # 检查图形对象
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_plot_cycle_comparison(self, sample_cycle_analysis):
        """测试绘制周期对比图"""
        viz = Visualizer()

        fig = viz.plot_cycle_comparison(
            sample_cycle_analysis,
            factor_name="TestFactor"
        )

        # 检查图形对象
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # 应该有两个子图
        axes = fig.get_axes()
        assert len(axes) == 2

        plt.close(fig)

    def test_plot_rolling_ic(self, sample_metrics):
        """测试绘制滚动 IC 图"""
        viz = Visualizer()

        fig = viz.plot_rolling_ic(
            sample_metrics['ic'],
            factor_name="TestFactor",
            windows=[20, 40, 60]
        )

        # 检查图形对象
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_plot_monthly_ic_heatmap(self, sample_metrics):
        """测试绘制月度 IC 热力图"""
        viz = Visualizer()

        fig = viz.plot_monthly_ic_heatmap(
            sample_metrics['ic'],
            factor_name="TestFactor"
        )

        # 检查图形对象
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_plot_ic_qq(self, sample_metrics):
        """测试绘制 IC Q-Q 图"""
        viz = Visualizer()

        fig = viz.plot_ic_qq(
            sample_metrics['ic'],
            factor_name="TestFactor"
        )

        # 检查图形对象
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        plt.close(fig)

    def test_create_full_report_chart(self, sample_metrics, sample_scenario_results):
        """测试创建完整报告图表"""
        viz = Visualizer()

        fig = viz.create_full_report_chart(
            metrics=sample_metrics,
            scenario_results=sample_scenario_results,
            factor_name="TestFactor"
        )

        # 检查图形对象
        assert fig is not None
        assert isinstance(fig, plt.Figure)

        # 检查有多个子图
        axes = fig.get_axes()
        assert len(axes) >= 5

        plt.close(fig)

    def test_save_figure_png(self, sample_metrics, tmp_path):
        """测试保存 PNG 图表"""
        viz = Visualizer(output_dir=str(tmp_path))

        fig = viz.plot_ic_timeseries(sample_metrics['ic'], factor_name="TestFactor")
        filepath = viz.save_figure(fig, "test_ic.png")

        # 检查文件存在
        assert filepath.exists()
        assert filepath.suffix == ".png"

    def test_save_figure_svg(self, sample_metrics, tmp_path):
        """测试保存 SVG 图表"""
        viz = Visualizer(output_dir=str(tmp_path))

        fig = viz.plot_ic_timeseries(sample_metrics['ic'], factor_name="TestFactor")
        filepath = viz.save_figure(fig, "test_ic.svg")

        # 检查文件存在
        assert filepath.exists()
        assert filepath.suffix == ".svg"

    def test_save_figure_pdf(self, sample_metrics, tmp_path):
        """测试保存 PDF 图表"""
        viz = Visualizer(output_dir=str(tmp_path))

        fig = viz.plot_ic_timeseries(sample_metrics['ic'], factor_name="TestFactor")
        filepath = viz.save_figure(fig, "test_ic.pdf")

        # 检查文件存在
        assert filepath.exists()
        assert filepath.suffix == ".pdf"

    def test_save_all_charts(self, sample_metrics, sample_scenario_results, tmp_path):
        """测试保存所有图表"""
        viz = Visualizer(output_dir=str(tmp_path))

        paths = viz.save_all_charts(
            metrics=sample_metrics,
            scenario_results=sample_scenario_results,
            factor_name="TestFactor"
        )

        # 检查返回的路径
        assert isinstance(paths, dict)
        assert len(paths) > 0

        # 检查文件存在
        for name, path in paths.items():
            assert path.exists(), f"{name} 文件不存在: {path}"


class TestCreateFactorCharts:
    """测试便捷函数"""

    def test_create_factor_charts(self, sample_metrics, sample_scenario_results, tmp_path):
        """测试创建因子图表便捷函数"""
        paths = create_factor_charts(
            metrics=sample_metrics,
            scenario_results=sample_scenario_results,
            factor_name="TestFactor",
            output_dir=str(tmp_path)
        )

        # 检查返回的路径
        assert isinstance(paths, dict)
        assert len(paths) > 0

        # 检查文件存在
        for name, path in paths.items():
            assert path.exists()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
