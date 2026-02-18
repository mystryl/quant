#!/usr/bin/env python3
"""
回测报告自动生成器

功能：
1. 自动生成文件夹名（时间戳+策略+参数）
2. 创建标准目录结构
3. 生成README.md和SUMMARY.md
4. 保存CSV结果、图表、指标JSON
"""
import pandas as pd
import json
from pathlib import Path
from datetime import datetime
import shutil


class BacktestReportGenerator:
    """回测报告生成器"""

    def __init__(self, base_dir=None):
        """
        初始化报告生成器

        Args:
            base_dir: 报告根目录，默认为 backtest_reports/
        """
        if base_dir is None:
            base_dir = Path(__file__).parent.parent.parent / "backtest_reports"
        self.base_dir = Path(base_dir)

    def generate_folder_name(self, strategy_name, params, timestamp=None):
        """
        生成报告文件夹名称

        Args:
            strategy_name: 策略名称
            params: 参数字典
            timestamp: 时间戳（datetime对象），默认为当前时间

        Returns:
            文件夹名称字符串

        格式：{YYYYMMDD}_{HHMM}_{strategy}_{key_params}
        """
        if timestamp is None:
            timestamp = datetime.now()

        # 生成时间戳
        date_str = timestamp.strftime("%Y%m%d")
        time_str = timestamp.strftime("%H%M")

        # 简化策略名称（移除空格和特殊字符）
        strategy_simple = strategy_name.replace(" ", "_").replace("(", "").replace(")", "").replace(",", "_")

        # 提取关键参数（最多3个）
        key_params = self._extract_key_params(params)
        params_str = "_".join([f"{k}{v}" for k, v in key_params.items()])

        # 组合文件夹名称
        folder_name = f"{date_str}_{time_str}_{strategy_simple}_{params_str}"

        return folder_name

    def _extract_key_params(self, params):
        """
        从参数字典中提取关键参数

        Args:
            params: 参数字典

        Returns:
            关键参数字典（最多3个）
        """
        # 关键参数优先级
        priority_keys = [
            'period',           # ATR周期
            'multiplier',        # ATR倍数
            'n',                 # 突破确认系数
            'freq',              # 频率
            'year',              # 年份
            'trials',            # 试验次数（用于优化）
            'trailing_stop_rate',  # 跟踪止损率
        ]

        key_params = {}
        count = 0

        for key in priority_keys:
            if key in params and count < 3:
                value = params[key]
                # 格式化参数值
                if isinstance(value, float):
                    # 小数点后2位
                    value_str = f"{value:.2f}"
                    # 移除末尾的0
                    if value_str.endswith('.00'):
                        value_str = value_str[:-3]
                elif isinstance(value, int):
                    value_str = str(value)
                else:
                    value_str = str(value)

                key_params[key] = value_str
                count += 1

        return key_params

    def create_report_folder(self, folder_name):
        """
        创建报告文件夹和标准目录结构

        Args:
            folder_name: 文件夹名称

        Returns:
            report_dir: 报告目录路径
        """
        report_dir = self.base_dir / folder_name

        # 创建目录结构
        (report_dir / "results").mkdir(parents=True, exist_ok=True)
        (report_dir / "code").mkdir(parents=True, exist_ok=True)
        (report_dir / "charts").mkdir(parents=True, exist_ok=True)

        return report_dir

    def generate_readme(self, report_dir, strategy_name, params, results, data_config=None, backtest_config=None):
        """
        生成README.md文件

        Args:
            report_dir: 报告目录
            strategy_name: 策略名称
            params: 参数字典
            results: 回测结果字典
            data_config: 数据配置字典（可选）
            backtest_config: 回测配置字典（可选）
        """
        readme_content = self._create_readme_content(
            strategy_name, params, results, data_config, backtest_config
        )

        readme_path = report_dir / "README.md"
        with open(readme_path, 'w', encoding='utf-8') as f:
            f.write(readme_content)

        return readme_path

    def _create_readme_content(self, strategy_name, params, results, data_config, backtest_config):
        """创建README内容"""
        content = f"# 回测报告 - {strategy_name}\n\n"
        content += f"**回测时间**: {datetime.now().strftime('%Y-%m-%d %H:%M')}\n\n"
        content += "---\n\n"

        # 策略信息
        content += "## 策略信息\n\n"
        content += f"**策略名称**: {strategy_name}\n\n"
        content += "**参数设置**:\n\n"

        for key, value in params.items():
            content += f"- `{key}`: {value}\n"

        content += "\n"

        # 策略描述
        content += "**策略描述**:\n\n"
        if 'n' in params:
            content += "增强版SuperTrend指标，包含双重突破确认机制。\n"
        else:
            content += "标准SuperTrend指标。\n"
        content += "\n"

        # 数据配置
        content += "## 数据配置\n\n"

        if data_config:
            for key, value in data_config.items():
                content += f"**{key}**: {value}\n"
            content += "\n"
        else:
            content += "**数据来源**: Qlib数据\n"
            content += "**频率**: 根据数据自动识别\n"
            content += "**合约**: RB9999.XSGE\n\n"

        # 回测配置
        content += "## 回测配置\n\n"

        if backtest_config:
            for key, value in backtest_config.items():
                content += f"**{key}**: {value}\n"
            content += "\n"
        else:
            content += "**初始资金**: 1,000,000 CNY\n"
            content += "**交易手续费**: 0（假设）\n"
            content += "**滑点**: 0（假设）\n\n"

        # 文件说明
        content += "## 文件说明\n\n"
        content += "- `results/` - 详细回测结果目录\n"
        content += "  - `*.csv` - 回测结果CSV\n"
        content += "  - `metrics.json` - 性能指标JSON\n"
        content += "- `charts/` - 图表目录\n"
        content += "  - `equity_curve.png` - 资金曲线图\n"
        content += "  - `drawdown_chart.png` - 回撤图\n"
        content += "- `code/` - 使用的代码（可选）\n\n"

        return content

    def generate_summary(self, report_dir, results, benchmark_results=None):
        """
        生成SUMMARY.md文件

        Args:
            report_dir: 报告目录
            results: 回测结果字典
            benchmark_results: 基准结果字典（可选）
        """
        summary_content = self._create_summary_content(results, benchmark_results)

        summary_path = report_dir / "SUMMARY.md"
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write(summary_content)

        return summary_path

    def _create_summary_content(self, results, benchmark_results):
        """创建SUMMARY内容"""
        content = "# 回测结果摘要\n\n"
        content += f"**策略**: {results.get('strategy_name', 'N/A')}\n\n"
        content += "---\n\n"

        # 关键性能指标
        content += "## 关键性能指标\n\n"
        content += "| 指标 | 数值 |\n"
        content += "|------|------|\n"

        content += f"| 总交易次数 | {results.get('total_trades', 'N/A')} |\n"
        content += f"| 累计收益 | {results.get('cumulative_return', 0):.2%} |\n"
        content += f"| 年化收益 | {results.get('annual_return', 0):.2%} |\n"
        content += f"| 最大回撤 | {results.get('max_drawdown', 0):.2%} |\n"
        content += f"| 夏普比率 | {results.get('sharpe_ratio', 0):.2f} |\n"
        content += f"| 胜率 | {results.get('win_rate', 0):.2f}% |\n"
        content += f"| 买入持有收益 | {results.get('buy_hold_return', 0):.2%} |\n"

        if 'stopped_out_count' in results:
            content += f"| 止损平仓次数 | {results.get('stopped_out_count', 0)} |\n"

        content += "\n"

        # 基准对比
        if benchmark_results:
            content += "## 基准对比\n\n"
            content += "| 指标 | 策略 | 基准 |\n"
            content += "|------|------|------|\n"

            content += f"| 累计收益 | {results.get('cumulative_return', 0):.2%} | {benchmark_results.get('cumulative_return', 0):.2%} |\n"
            content += f"| 年化收益 | {results.get('annual_return', 0):.2%} | {benchmark_results.get('annual_return', 0):.2%} |\n"
            content += f"| 最大回撤 | {results.get('max_drawdown', 0):.2%} | {benchmark_results.get('max_drawdown', 0):.2%} |\n"
            content += f"| 夏普比率 | {results.get('sharpe_ratio', 0):.2f} | {benchmark_results.get('sharpe_ratio', 0):.2f} |\n"
            content += f"| 胜率 | {results.get('win_rate', 0):.2f}% | {benchmark_results.get('win_rate', 0):.2f}% |\n"
            content += "\n"

            # 对比结论
            content += "## 对比结论\n\n"

            if results.get('sharpe_ratio', 0) > benchmark_results.get('sharpe_ratio', 0):
                content += "✅ **策略优于基准**：夏普比率更高\n\n"
            else:
                content += "⚠️ **策略低于基准**：夏普比率较低\n\n"

        # 结论和建议
        content += "## 结论和建议\n\n"

        sharpe = results.get('sharpe_ratio', 0)
        max_dd = results.get('max_drawdown', 0)

        if sharpe > 1.5 and max_dd < 0.15:
            content += "✅ **表现优秀**：策略具有良好的风险调整收益，回撤可控。\n\n"
        elif sharpe > 0.8 and max_dd < 0.2:
            content += "🟡 **表现良好**：策略表现尚可，但仍有优化空间。\n\n"
        elif sharpe > 0:
            content += "🟠 **表现一般**：策略有一定收益，但风险较高。\n\n"
        else:
            content += "🔴 **表现较差**：策略表现不佳，建议重新调整参数。\n\n"

        return content

    def save_results_csv(self, report_dir, results_df, filename="backtest_results.csv"):
        """
        保存回测结果CSV

        Args:
            report_dir: 报告目录
            results_df: 回测结果DataFrame
            filename: 文件名
        """
        results_path = report_dir / "results" / filename
        results_df.to_csv(results_path, index=False, encoding='utf-8-sig')
        return results_path

    def save_metrics_json(self, report_dir, results, filename="metrics.json"):
        """
        保存性能指标JSON

        Args:
            report_dir: 报告目录
            results: 回测结果字典
            filename: 文件名
        """
        metrics_path = report_dir / "results" / filename

        # 转换numpy类型为Python类型
        results_json = {}
        for key, value in results.items():
            if hasattr(value, 'item'):  # numpy类型
                results_json[key] = value.item()
            else:
                results_json[key] = value

        with open(metrics_path, 'w', encoding='utf-8') as f:
            json.dump(results_json, f, indent=2, ensure_ascii=False)

        return metrics_path

    def save_chart(self, report_dir, fig, filename="equity_curve.png"):
        """
        保存图表

        Args:
            report_dir: 报告目录
            fig: matplotlib图表对象
            filename: 文件名
        """
        chart_path = report_dir / "charts" / filename
        fig.savefig(str(chart_path), dpi=150, bbox_inches='tight')
        return chart_path

    def save_code(self, report_dir, source_file):
        """
        保存使用的代码

        Args:
            report_dir: 报告目录
            source_file: 源代码文件路径
        """
        if not Path(source_file).exists():
            return None

        dest_file = report_dir / "code" / Path(source_file).name
        shutil.copy2(source_file, dest_file)
        return dest_file

    def generate_full_report(self, strategy_name, params, results, results_df=None,
                             data_config=None, backtest_config=None,
                             benchmark_results=None, source_file=None,
                             charts=None):
        """
        生成完整回测报告

        Args:
            strategy_name: 策略名称
            params: 参数字典
            results: 回测结果字典
            results_df: 回测结果DataFrame（可选）
            data_config: 数据配置字典（可选）
            backtest_config: 回测配置字典（可选）
            benchmark_results: 基准结果字典（可选）
            source_file: 源代码文件路径（可选）
            charts: 图表字典 {filename: fig}（可选）

        Returns:
            report_dir: 报告目录路径
        """
        # 生成文件夹名称
        folder_name = self.generate_folder_name(strategy_name, params)

        # 创建文件夹
        report_dir = self.create_report_folder(folder_name)

        # 生成README
        self.generate_readme(report_dir, strategy_name, params, results, data_config, backtest_config)

        # 生成SUMMARY
        self.generate_summary(report_dir, results, benchmark_results)

        # 保存结果CSV
        if results_df is not None:
            self.save_results_csv(report_dir, results_df)

        # 保存指标JSON
        self.save_metrics_json(report_dir, results)

        # 保存代码
        if source_file is not None:
            self.save_code(report_dir, source_file)

        # 保存图表
        if charts is not None:
            for filename, fig in charts.items():
                self.save_chart(report_dir, fig, filename)

        return report_dir


# 便捷函数
def create_report(strategy_name, params, results, results_df=None,
                  data_config=None, backtest_config=None,
                  benchmark_results=None, source_file=None,
                  charts=None, base_dir=None):
    """
    便捷函数：创建完整回测报告

    Args:
        strategy_name: 策略名称
        params: 参数字典
        results: 回测结果字典
        results_df: 回测结果DataFrame（可选）
        data_config: 数据配置字典（可选）
        backtest_config: 回测配置字典（可选）
        benchmark_results: 基准结果字典（可选）
        source_file: 源代码文件路径（可选）
        charts: 图表字典 {filename: fig}（可选）
        base_dir: 报告根目录（可选）

    Returns:
        report_dir: 报告目录路径
    """
    generator = BacktestReportGenerator(base_dir)
    return generator.generate_full_report(
        strategy_name, params, results, results_df,
        data_config, backtest_config, benchmark_results,
        source_file, charts
    )


if __name__ == "__main__":
    # 测试代码
    print("回测报告生成器测试\n")

    # 模拟参数和结果
    strategy_name = "SuperTrend_SF14Re"
    params = {
        'period': 50,
        'multiplier': 20,
        'n': 3,
        'trailing_stop_rate': 80,
        'freq': '15min',
        'year': 2023
    }

    results = {
        'strategy_name': strategy_name,
        'total_trades': 5,
        'cumulative_return': -0.16842189499590143,
        'annual_return': -0.8212224232819632,
        'max_drawdown': 0.17135913111054077,
        'sharpe_ratio': -5.626454551359018,
        'win_rate': 48.570209133589415,
        'buy_hold_return': -0.006947890818869595,
        'stopped_out_count': 6
    }

    # 生成报告
    report_dir = create_report(strategy_name, params, results)

    print(f"✅ 测试报告已生成到: {report_dir}")
    print("\n报告结构:")
    print(f"  {report_dir}/")
    print(f"    README.md")
    print(f"    SUMMARY.md")
    print(f"    results/")
    print(f"      metrics.json")
    print(f"    code/")
    print(f"    charts/")
