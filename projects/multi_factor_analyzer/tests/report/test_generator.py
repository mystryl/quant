"""
测试报告生成器模块
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime

import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'src'))

from src.report.generator import ReportGenerator, generate_factor_report


@pytest.fixture
def sample_metrics():
    """创建示例性能指标"""
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')

    return {
        'ic': pd.Series(np.random.randn(100) * 0.05 + 0.03, index=dates[:100]),
        'rank_ic': pd.Series(np.random.randn(100) * 0.06 + 0.04, index=dates[:100]),
        'ic_mean': 0.035,
        'ic_std': 0.052,
        'icir': 0.673,
        'rank_ic_mean': 0.042,
        'rank_ic_std': 0.061,
        'rank_icir': 0.689,
        'long_short_return': pd.Series(np.random.randn(100) * 0.01 + 0.001, index=dates[:100]),
        'annual_return': 0.085,
        'sharpe_ratio': 1.85,
        'max_drawdown': -0.12,
        'win_rate': 0.56
    }


@pytest.fixture
def sample_scenario_results():
    """创建示例策略场景结果"""
    return {
        'bull': {
            'strategy_returns': pd.Series(np.random.randn(100) * 0.01 + 0.002, index=pd.date_range('2020-01-01', periods=100)),
            'annual_return': 0.12,
            'sharpe_ratio': 2.1,
            'max_drawdown': -0.10,
            'win_rate': 0.58
        },
        'bear': {
            'strategy_returns': pd.Series(np.random.randn(100) * 0.01 - 0.001, index=pd.date_range('2020-01-01', periods=100)),
            'annual_return': -0.02,
            'sharpe_ratio': -0.5,
            'max_drawdown': -0.15,
            'win_rate': 0.48
        },
        'long_short': {
            'strategy_returns': pd.Series(np.random.randn(100) * 0.01 + 0.001, index=pd.date_range('2020-01-01', periods=100)),
            'annual_return': 0.085,
            'sharpe_ratio': 1.85,
            'max_drawdown': -0.12,
            'win_rate': 0.56
        }
    }


@pytest.fixture
def sample_reliability_result():
    """创建示例可靠性评估结果"""
    return {
        'scores': {
            'ic_stability': 0.40,
            'ic_absolute': 0.18,
            'ir': 0.18,
            'long_short_return': 0.08,
            'win_rate': 0.06
        },
        'total_score': 0.90,
        'reliability': 'A+',
        'recommendation': '该因子表现优秀，建议重点使用。'
    }


@pytest.fixture
def sample_cycle_analysis():
    """创建示例周期分析结果"""
    return {
        'best_shift': 2,
        'ic_by_shift': {
            1: {'ic_mean': 0.025, 'icir': 0.45},
            2: {'ic_mean': 0.035, 'icir': 0.67},
            3: {'ic_mean': 0.028, 'icir': 0.52}
        },
        'recommendation': '建议使用 T+2 对齐周期。'
    }


class TestReportGenerator:
    """测试 ReportGenerator 类"""

    def test_initialization(self, tmp_path):
        """测试初始化"""
        generator = ReportGenerator(output_dir=str(tmp_path))
        assert generator.output_dir == tmp_path
        assert generator.include_charts == True
        assert tmp_path.exists()

    def test_generate_full_report(
        self,
        sample_metrics,
        sample_scenario_results,
        sample_reliability_result,
        sample_cycle_analysis
    ):
        """测试生成完整报告"""
        generator = ReportGenerator()

        reports = generator.generate_full_report(
            factor_name="TestFactor",
            metrics=sample_metrics,
            scenario_results=sample_scenario_results,
            reliability_result=sample_reliability_result,
            cycle_analysis=sample_cycle_analysis
        )

        # 检查返回的格式
        assert 'markdown' in reports
        assert 'html' in reports
        assert 'text' in reports
        assert 'json' in reports
        assert 'data' in reports

        # 检查内容不为空
        assert len(reports['markdown']) > 0
        assert len(reports['html']) > 0
        assert len(reports['text']) > 0
        assert len(reports['json']) > 0

    def test_generate_summary(self, sample_metrics, sample_reliability_result):
        """测试生成执行摘要"""
        generator = ReportGenerator()

        summary = generator._generate_summary(
            factor_name="TestFactor",
            metrics=sample_metrics,
            reliability_result=sample_reliability_result,
            backtest_period=('2020-01-01', '2020-12-31')
        )

        # 检查关键内容
        assert 'TestFactor' in summary
        assert '执行摘要' in summary
        assert 'IC 均值' in summary
        assert 'ICIR' in summary
        assert '可靠性等级' in summary

    def test_generate_ic_analysis(self, sample_metrics):
        """测试生成 IC 分析"""
        generator = ReportGenerator()

        ic_analysis = generator._generate_ic_analysis(sample_metrics)

        # 检查关键内容
        assert 'IC 分析' in ic_analysis
        assert 'IC 均值' in ic_analysis
        assert 'ICIR' in ic_analysis
        assert 'Rank IC' in ic_analysis

    def test_generate_ir_analysis(self, sample_metrics):
        """测试生成 IR 分析"""
        generator = ReportGenerator()

        ir_analysis = generator._generate_ir_analysis(sample_metrics)

        # 检查关键内容
        assert 'IR 分析' in ir_analysis
        assert '年化收益率' in ir_analysis
        assert '夏普比率' in ir_analysis
        assert '最大回撤' in ir_analysis
        assert '胜率' in ir_analysis

    def test_generate_scenario_analysis(self, sample_scenario_results):
        """测试生成策略场景分析"""
        generator = ReportGenerator()

        scenario_analysis = generator._generate_scenario_analysis(sample_scenario_results)

        # 检查关键内容
        assert '策略场景分析' in scenario_analysis
        assert '看涨策略' in scenario_analysis
        assert '看跌策略' in scenario_analysis
        assert '多空策略' in scenario_analysis

    def test_generate_cycle_analysis(self, sample_cycle_analysis):
        """测试生成周期分析"""
        generator = ReportGenerator()

        cycle_analysis = generator._generate_cycle_analysis(sample_cycle_analysis)

        # 检查关键内容
        assert '周期分析' in cycle_analysis
        assert 'T+2' in cycle_analysis
        assert 'IC 均值' in cycle_analysis

    def test_generate_recommendations(
        self,
        sample_metrics,
        sample_scenario_results,
        sample_reliability_result
    ):
        """测试生成建议"""
        generator = ReportGenerator()

        recommendations = generator._generate_recommendations(
            metrics=sample_metrics,
            scenario_results=sample_scenario_results,
            reliability_result=sample_reliability_result
        )

        # 检查关键内容
        assert '建议' in recommendations
        assert len(recommendations) > 0

    def test_evaluate_ic(self):
        """测试 IC 评估"""
        generator = ReportGenerator()

        assert generator._evaluate_ic(0.06) == "优秀"
        assert generator._evaluate_ic(0.04) == "良好"
        assert generator._evaluate_ic(0.025) == "一般"
        assert generator._evaluate_ic(0.01) == "较差"

    def test_evaluate_icir(self):
        """测试 ICIR 评估"""
        generator = ReportGenerator()

        assert generator._evaluate_icir(0.6) == "优秀"
        assert generator._evaluate_icir(0.4) == "良好"
        assert generator._evaluate_icir(0.25) == "一般"
        assert generator._evaluate_icir(0.1) == "较差"

    def test_evaluate_return(self):
        """测试收益率评估"""
        generator = ReportGenerator()

        assert generator._evaluate_return(0.15) == "优秀"
        assert generator._evaluate_return(0.08) == "良好"
        assert generator._evaluate_return(0.03) == "一般"
        assert generator._evaluate_return(0.01) == "较差"

    def test_save_markdown(self, sample_metrics, sample_scenario_results, tmp_path):
        """测试保存 Markdown 报告"""
        generator = ReportGenerator(output_dir=str(tmp_path))

        reports = generator.generate_full_report(
            factor_name="TestFactor",
            metrics=sample_metrics,
            scenario_results=sample_scenario_results
        )

        filepath = generator.save_markdown(reports['markdown'], "test_report.md")

        # 检查文件存在
        assert filepath.exists()
        assert filepath.name == "test_report.md"

        # 检查内容
        content = filepath.read_text(encoding='utf-8')
        assert 'TestFactor' in content
        assert len(content) > 0

    def test_save_html(self, sample_metrics, sample_scenario_results, tmp_path):
        """测试保存 HTML 报告"""
        generator = ReportGenerator(output_dir=str(tmp_path))

        reports = generator.generate_full_report(
            factor_name="TestFactor",
            metrics=sample_metrics,
            scenario_results=sample_scenario_results
        )

        filepath = generator.save_html(reports['html'], "test_report.html")

        # 检查文件存在
        assert filepath.exists()
        assert filepath.name == "test_report.html"

        # 检查内容
        content = filepath.read_text(encoding='utf-8')
        assert '<!DOCTYPE html>' in content
        assert 'TestFactor' in content

    def test_save_text(self, sample_metrics, sample_scenario_results, tmp_path):
        """测试保存纯文本报告"""
        generator = ReportGenerator(output_dir=str(tmp_path))

        reports = generator.generate_full_report(
            factor_name="TestFactor",
            metrics=sample_metrics,
            scenario_results=sample_scenario_results
        )

        filepath = generator.save_text(reports['text'], "test_report.txt")

        # 检查文件存在
        assert filepath.exists()
        assert filepath.name == "test_report.txt"

        # 检查内容
        content = filepath.read_text(encoding='utf-8')
        assert 'TestFactor' in content

    def test_save_json(self, sample_metrics, sample_scenario_results, tmp_path):
        """测试保存 JSON 报告"""
        generator = ReportGenerator(output_dir=str(tmp_path))

        reports = generator.generate_full_report(
            factor_name="TestFactor",
            metrics=sample_metrics,
            scenario_results=sample_scenario_results
        )

        filepath = generator.save_json(reports['json'], "test_report.json")

        # 检查文件存在
        assert filepath.exists()
        assert filepath.name == "test_report.json"

        # 检查内容
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        assert 'factor_name' in data
        assert data['factor_name'] == 'TestFactor'


class TestGenerateFactorReport:
    """测试便捷函数"""

    def test_generate_factor_report(
        self,
        sample_metrics,
        sample_scenario_results,
        tmp_path
    ):
        """测试生成因子报告便捷函数"""
        reports = generate_factor_report(
            factor_name="TestFactor",
            metrics=sample_metrics,
            scenario_results=sample_scenario_results,
            output_dir=str(tmp_path)
        )

        # 检查返回的格式
        assert 'markdown' in reports
        assert 'html' in reports
        assert 'text' in reports
        assert 'json' in reports


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
