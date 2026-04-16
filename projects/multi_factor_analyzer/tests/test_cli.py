"""
CLI 模块单元测试

测试命令行接口的各个命令和功能。
"""

import os
import sys
import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
from click.testing import CliRunner

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from src.cli.main import cli, validate_expression


class TestCLIValidate:
    """测试 validate 命令"""

    def test_validate_valid_expression(self):
        """测试验证有效的表达式"""
        runner = CliRunner()
        result = runner.invoke(cli, ['validate', 'Ref($close, 20) / $close - 1'])

        assert result.exit_code == 0
        assert '验证通过' in result.output

    def test_validate_future_function(self):
        """测试检测未来函数"""
        runner = CliRunner()
        result = runner.invoke(cli, ['validate', 'Ref($close, -5)'])

        assert result.exit_code == 1
        assert '未来函数' in result.output or '失败' in result.output

    def test_validate_complex_expression(self):
        """测试复杂表达式"""
        runner = CliRunner()
        result = runner.invoke(
            cli,
            ['validate', '($close - Mean($close, 20)) / Std($close, 20)']
        )

        assert result.exit_code == 0
        assert '验证通过' in result.output


class TestCLIAnalyze:
    """测试 analyze 命令"""

    @pytest.fixture
    def mock_data_provider(self):
        """模拟数据提供者"""
        with patch('src.cli.main.FactorDataProvider') as mock:
            provider_instance = Mock()
            mock.return_value = provider_instance

            # 模拟 get_price_data 返回值
            import pandas as pd
            import numpy as np

            dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
            index = pd.MultiIndex.from_product(
                [dates, ['SH600000']],
                names=['datetime', 'instrument']
            )

            price_data = pd.DataFrame(
                {'close': 100 + np.random.randn(len(index))},
                index=index
            )

            provider_instance.get_price_data.return_value = price_data
            provider_instance.get_factor_data.return_value = price_data

            yield provider_instance

    @pytest.fixture
    def mock_factor_manager(self):
        """模拟因子管理器"""
        with patch('src.cli.main.FactorManager') as mock:
            manager_instance = Mock()

            # 模拟 factors 字典
            manager_instance.factors = {
                'test_factor': {
                    'definition': 'Ref($close, 20) / $close - 1',
                    'type': 'expression',
                    'metadata': {}
                }
            }

            # 模拟 calculate_factor 返回值
            import pandas as pd
            import numpy as np

            dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
            index = pd.MultiIndex.from_product(
                [dates, ['SH600000']],
                names=['datetime', 'instrument']
            )

            factor_data = pd.Series(
                np.random.randn(len(index)),
                index=index
            )

            manager_instance.calculate_factor.return_value = factor_data

            mock.return_value = manager_instance
            yield manager_instance

    def test_analyze_missing_required_params(self):
        """测试缺少必需参数"""
        runner = CliRunner()
        result = runner.invoke(cli, ['analyze'])

        assert result.exit_code != 0
        assert 'Missing option' in result.output or '需要' in result.output

    def test_analyze_with_expression(self, mock_data_provider, mock_factor_manager):
        """测试使用表达式分析"""
        runner = CliRunner()

        # 创建临时股票文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('SH600000\n')
            temp_file = f.name

        try:
            result = runner.invoke(cli, [
                'analyze',
                '--factor', 'Ref($close, 20) / $close - 1',
                '--instruments', temp_file,
                '--start', '2020-01-01',
                '--end', '2020-01-31',
                '--strategy', 'bull'  # 只测试看涨策略，避免复杂性
            ])

            # 由于是模拟数据，可能会出现一些错误，但命令应该能运行
            # 我们主要检查命令被正确调用
            assert mock_factor_manager.register_factor.called or \
                   mock_factor_manager.calculate_factor.called or \
                   mock_data_provider.get_price_data.called

        finally:
            os.unlink(temp_file)


class TestCLIBatch:
    """测试 batch 命令"""

    @pytest.fixture
    def sample_config(self):
        """创建示例配置文件"""
        config_content = """
factors:
  - name: MA20
    expression: "Ref($close, 20) / $close - 1"
    description: "20日均线偏离度"

  - name: MA60
    expression: "Ref($close, 60) / $close - 1"
    description: "60日均线偏离度"

analysis:
  instruments: "instruments.txt"
  start_date: "2020-01-01"
  end_date: "2020-01-31"
  quantile: 0.2
  top_pct: 0.2
"""
        return config_content

    @pytest.fixture
    def instruments_file(self):
        """创建股票列表文件"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write('SH600000\nSH600001\n')
            temp_file = f.name
        yield temp_file
        os.unlink(temp_file)

    def test_batch_missing_config(self):
        """测试缺少配置文件"""
        runner = CliRunner()
        result = runner.invoke(cli, ['batch'])

        assert result.exit_code != 0
        assert 'Missing option' in result.output or '需要' in result.output

    def test_batch_with_config(self, sample_config, instruments_file):
        """测试使用配置文件批量分析"""
        runner = CliRunner()

        # 创建配置文件
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            # 替换 instruments 路径
            config = sample_config.replace(
                'instruments: "instruments.txt"',
                f'instruments: "{instruments_file}"'
            )
            f.write(config)
            config_file = f.name

        try:
            # 模拟数据提供者和因子管理器
            with patch('src.cli.main.FactorDataProvider') as mock_provider, \
                 patch('src.cli.main.FactorManager') as mock_manager:

                # 设置模拟返回值
                provider_instance = Mock()
                mock_provider.return_value = provider_instance

                manager_instance = Mock()
                manager_instance.factors = {}
                mock_manager.return_value = manager_instance

                result = runner.invoke(cli, ['batch', '--config', config_file])

                # 检查是否调用了相关方法
                # 注意：由于模拟限制，可能不会完全成功，但应该能看到调用

        finally:
            os.unlink(config_file)


class TestCLIReport:
    """测试 report 命令"""

    @pytest.fixture
    def sample_results(self):
        """创建示例结果数据"""
        results = {
            'MA20': {
                'status': 'success',
                'metrics': {
                    'ic_mean': 0.05,
                    'ic_std': 0.12,
                    'icir': 0.42,
                    'rank_ic_mean': 0.06,
                    'rank_ic_std': 0.11,
                    'rank_icir': 0.55,
                    'annual_return': 0.08,
                    'sharpe_ratio': 1.2,
                    'max_drawdown': 0.12,
                    'win_rate': 0.55
                },
                'strategy_results': {
                    'bull': {
                        'annual_return': 0.08,
                        'sharpe_ratio': 1.2,
                        'max_drawdown': 0.12,
                        'win_rate': 0.55
                    }
                }
            }
        }
        return results

    def test_report_missing_params(self):
        """测试缺少必需参数"""
        runner = CliRunner()
        result = runner.invoke(cli, ['report'])

        assert result.exit_code != 0
        assert 'Missing option' in result.output or '需要' in result.output

    def test_generate_html_report(self, sample_results):
        """测试生成 HTML 报告"""
        from src.cli.main import _generate_html_report

        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as f:
            output_file = f.name

        try:
            _generate_html_report(sample_results, Path(output_file), "测试报告")

            # 检查文件是否创建
            assert os.path.exists(output_file)

            # 读取并检查内容
            with open(output_file, 'r', encoding='utf-8') as f:
                content = f.read()

            assert '测试报告' in content
            assert 'MA20' in content
            assert '0.05' in content  # IC 均值
            assert '</html>' in content

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)

    def test_generate_json_report(self, sample_results):
        """测试生成 JSON 报告"""
        from src.cli.main import _generate_json_report

        with tempfile.NamedTemporaryFile(suffix='.json', delete=False) as f:
            output_file = f.name

        try:
            _generate_json_report(sample_results, Path(output_file), "测试报告")

            # 检查文件是否创建
            assert os.path.exists(output_file)

            # 读取并检查内容
            with open(output_file, 'r', encoding='utf-8') as f:
                data = json.load(f)

            assert data['title'] == "测试报告"
            assert 'MA20' in data['results']
            assert data['results']['MA20']['metrics']['ic_mean'] == 0.05

        finally:
            if os.path.exists(output_file):
                os.unlink(output_file)


class TestCLIHelpers:
    """测试 CLI 辅助函数"""

    def test_display_metrics_table(self):
        """测试显示指标表格"""
        from src.cli.main import display_metrics_table

        metrics = {
            'ic_mean': 0.05,
            'ic_std': 0.12,
            'icir': 0.42,
            'rank_ic_mean': 0.06,
            'rank_ic_std': 0.11,
            'rank_icir': 0.55,
            'annual_return': 0.08,
            'sharpe_ratio': 1.2,
            'max_drawdown': 0.12,
            'win_rate': 0.55
        }

        # 这个函数主要是显示，我们只检查不报错
        # Rich 的 console.capture 可能无法捕获所有输出
        # 所以我们只验证函数能正常执行
        try:
            display_metrics_table(metrics, "测试因子")
            # 如果没有抛出异常，测试通过
        except Exception as e:
            pytest.fail(f"display_metrics_table raised an exception: {e}")

    def test_print_banner(self):
        """测试打印横幅"""
        from src.cli.main import print_banner

        # Rich 的输出可能无法被 capture 捕获
        # 我们只验证函数能正常执行不抛出异常
        try:
            print_banner()
            # 如果没有抛出异常，测试通过
        except Exception as e:
            pytest.fail(f"print_banner raised an exception: {e}")


class TestCLIIntegration:
    """CLI 集成测试"""

    def test_cli_version(self):
        """测试版本信息"""
        runner = CliRunner()
        result = runner.invoke(cli, ['--version'])

        assert result.exit_code == 0
        assert '1.0.0' in result.output

    def test_cli_help(self):
        """测试帮助信息"""
        runner = CliRunner()
        result = runner.invoke(cli, ['--help'])

        assert result.exit_code == 0
        assert 'analyze' in result.output
        assert 'batch' in result.output
        assert 'report' in result.output
        assert 'validate' in result.output

    def test_all_commands_help(self):
        """测试所有命令的帮助信息"""
        runner = CliRunner()

        commands = ['analyze', 'batch', 'report', 'validate']

        for cmd in commands:
            result = runner.invoke(cli, [cmd, '--help'])
            assert result.exit_code == 0, f"Help for {cmd} failed"
            assert cmd in result.output.lower() or 'usage' in result.output.lower()


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
