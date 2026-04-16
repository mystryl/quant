"""
多因子量化分析系统 - 命令行接口 (CLI)

本模块提供命令行接口，支持以下主要功能：
1. 单因子分析 (analyze) - 分析单个因子的性能
2. 批量分析 (batch) - 批量分析多个因子
3. 报告生成 (report) - 生成分析报告
4. 表达式验证 (validate) - 验证因子表达式

主要特性：
- 友好的用户界面（使用 Rich 库）
- 进度显示
- 彩色输出
- 表格显示
- 错误处理和友好提示
- 支持配置文件
"""

import sys
from pathlib import Path

import click
import yaml
from rich.console import Console
from rich.table import Table
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeRemainingColumn
from rich.panel import Panel

# 添加项目根目录到 Python 路径
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root))

from src.core.factor_engine import FactorManager
from src.core.performance_eval import PerformanceEvaluator
from src.core.strategy_analyzer import StrategyAnalyzer
from src.data.provider import FactorDataProvider

# 创建 Rich Console 实例
console = Console()


# =============================================================================
# 全局选项和辅助函数
# =============================================================================


def print_banner():
    """打印程序欢迎横幅"""
    banner = """
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║        [bold cyan]Multi-Factor Quantitative Analyzer[/bold cyan]                   ║
║                  [bold yellow]多因子量化分析系统[/bold yellow]                              ║
║                                                                   ║
║                   Version: [green]1.0.0[/green]                                   ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
"""
    console.print(Panel(banner, border_style="cyan", padding=(1, 2)))


def print_error(message: str, exit_code: int = 1):
    """打印错误信息并退出

    Args:
        message: 错误信息
        exit_code: 退出码
    """
    console.print(f"[bold red]错误:[/bold red] {message}", style="red")
    sys.exit(exit_code)


def print_warning(message: str):
    """打印警告信息

    Args:
        message: 警告信息
    """
    console.print(f"[bold yellow]警告:[/bold yellow] {message}", style="yellow")


def print_success(message: str):
    """打印成功信息

    Args:
        message: 成功信息
    """
    console.print(f"[bold green]成功:[/bold green] {message}", style="green")


def print_info(message: str):
    """打印信息

    Args:
        message: 信息内容
    """
    console.print(f"[bold blue]信息:[/bold blue] {message}", style="blue")


def create_progress_bar(description: str = "处理中..."):
    """创建进度条

    Args:
        description: 进度条描述

    Returns:
        Progress 对象
    """
    return Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeRemainingColumn(),
        console=console,
    )


def display_metrics_table(metrics: dict, factor_name: str):
    """以表格形式显示指标

    Args:
        metrics: 指标字典
        factor_name: 因子名称
    """
    table = Table(
        title=f"[bold cyan]因子性能指标: {factor_name}[/bold cyan]", show_header=True, header_style="bold magenta"
    )

    table.add_column("指标", style="cyan", width=30)
    table.add_column("数值", justify="right", style="green")
    table.add_column("说明", style="dim")

    # 预测能力指标
    table.add_row("IC 均值", f"{metrics.get('ic_mean', 0):.4f}", "因子预测能力")
    table.add_row("IC 标准差", f"{metrics.get('ic_std', 0):.4f}", "IC 波动性")
    table.add_row("ICIR", f"{metrics.get('icir', 0):.4f}", "IC 稳定性")
    table.add_row("Rank IC 均值", f"{metrics.get('rank_ic_mean', 0):.4f}", "排名预测能力")
    table.add_row("Rank ICIR", f"{metrics.get('rank_icir', 0):.4f}", "排名 IC 稳定性")

    table.add_section()

    # 交易效果指标
    table.add_row("年化收益率", f"{metrics.get('annual_return', 0):.2%}", "策略年化收益")
    table.add_row("夏普比率", f"{metrics.get('sharpe_ratio', 0):.2f}", "风险调整后收益")
    table.add_row("最大回撤", f"{metrics.get('max_drawdown', 0):.2%}", "最大损失幅度")
    table.add_row("胜率", f"{metrics.get('win_rate', 0):.2%}", "盈利天数占比")

    console.print(table)


def display_reliability_assessment(score: float, reliability: str, recommendation: str):
    """显示可靠性评估结果

    Args:
        score: 综合评分
        reliability: 可靠性等级
        recommendation: 建议
    """
    # 根据等级选择颜色
    color_map = {"A+": "bright_green", "A": "green", "B": "yellow", "C": "orange3", "D": "red"}
    color = color_map.get(reliability.split()[0], "white")

    panel_content = f"""
[bold]可靠性等级:[/bold] [{color}]{reliability}[/{color}]
[bold]综合评分:[/bold] {score:.2f}/1.00

[bold cyan]评估建议:[/bold cyan]
{recommendation}
"""

    console.print(Panel(panel_content, title="[bold]可靠性评估[/bold]", border_style=color))


# =============================================================================
# CLI 主命令组
# =============================================================================


@click.group(invoke_without_command=True)
@click.option("--version", is_flag=True, help="显示版本信息")
@click.option("--verbose", "-v", is_flag=True, help="显示详细输出")
@click.pass_context
def cli(ctx, version, verbose):
    """
    [bold cyan]多因子量化分析系统 - 命令行工具[/bold cyan]

    用于分析、评估和报告因子性能的命令行工具集。

    [bold yellow]主要命令:[/bold yellow]

    \b
      mfa analyze    分析单个因子的性能
      mfa batch      批量分析多个因子
      mfa report     生成分析报告
      mfa validate   验证因子表达式

    [bold yellow]示例:[/bold yellow]

    \b
      mfa analyze MA20 --instruments SH600000 --start 2020-01-01 --end 2020-12-31
      mfa batch --config factors.yaml
      mfa report --input results/ --output report.html

    [bold yellow]获取帮助:[/bold yellow]

    \b
      mfa COMMAND --help    查看具体命令的帮助
    """
    if version:
        console.print("[bold cyan]Multi-Factor Analyzer[/bold cyan] version [green]1.0.0[/green]")
        sys.exit(0)

    # 如果没有提供子命令，显示帮助信息
    if ctx.invoked_subcommand is None:
        print_banner()
        console.print(ctx.get_help())


# =============================================================================
# analyze 命令 - 单因子分析
# =============================================================================


@cli.command("analyze")
@click.option("--factor", "-f", required=True, help="因子名称或表达式")
@click.option("--instruments", "-i", required=True, help="股票代码或代码文件（每行一个代码）")
@click.option("--start", "-s", required=True, help="开始日期 (YYYY-MM-DD)")
@click.option("--end", "-e", required=True, help="结束日期 (YYYY-MM-DD)")
@click.option("--output", "-o", help="输出文件路径（JSON 或 CSV 格式）")
@click.option("--quantile", "-q", default=0.2, type=float, help="多空分组分位数（默认：0.2）")
@click.option("--top-pct", default=0.2, type=float, help="选股比例（默认：0.2）")
@click.option(
    "--strategy",
    type=click.Choice(["all", "bull", "bear", "long_short", "volatility"]),
    default="all",
    help="策略类型（默认：all）",
)
@click.option("--no-cache", is_flag=True, help="禁用缓存")
@click.option("--verbose", "-v", is_flag=True, help="显示详细输出")
def analyze_factor(factor, instruments, start, end, output, quantile, top_pct, strategy, no_cache, verbose):
    """
    分析单个因子的性能

    [bold cyan]示例:[/bold cyan]

    \b
      # 使用表达式分析
      mfa analyze -f "Ref($close, 20) / $close - 1" -i SH600000 -s 2020-01-01 -e 2020-12-31

    \b
      # 使用已注册的因子
      mfa analyze -f MA20 -i instruments.txt -s 2020-01-01 -e 2020-12-31

    \b
      # 只分析看涨策略
      mfa analyze -f MA20 -i SH600000 -s 2020-01-01 -e 2020-12-31 --strategy bull

    \b
      # 保存结果到文件
      mfa analyze -f MA20 -i SH600000 -s 2020-01-01 -e 2020-12-31 -o results.json
    """
    print_banner()

    try:
        # 1. 加载股票列表
        print_info("加载股票列表...")
        instruments_file = Path(instruments)
        if instruments_file.exists():
            # 从文件读取
            with open(instruments_file, "r", encoding="utf-8") as f:
                instruments_list = [line.strip() for line in f if line.strip()]
            console.print(f"从文件加载 [green]{len(instruments_list)}[/green] 个股票")
        else:
            # 使用逗号分隔的字符串
            instruments_list = [i.strip() for i in instruments.split(",")]
            console.print(f"加载 [green]{len(instruments_list)}[/green] 个股票")

        # 2. 初始化数据提供者
        print_info("初始化数据提供者...")
        provider = FactorDataProvider()

        # 3. 初始化因子管理器
        print_info("初始化因子管理器...")
        factor_manager = FactorManager(provider, cache_enabled=not no_cache)

        # 4. 检查因子是否为表达式
        if factor.startswith("$") or "(" in factor or "Ref" in factor:
            # 表达式因子
            print_info("检测到表达式因子，注册中...")
            factor_manager.register_factor(
                factor, factor, metadata={"type": "expression", "description": "用户自定义表达式"}
            )
            factor_name = "custom_factor"
        else:
            # 已注册的因子
            factor_name = factor
            if factor_name not in factor_manager.factors:
                print_error(f"因子 '{factor_name}' 未注册")
            print_info(f"使用已注册的因子: {factor_name}")

        # 5. 计算因子值
        with create_progress_bar("计算因子值") as progress:
            task = progress.add_task("计算因子值...", total=100)

            factor_data = factor_manager.calculate_factor(
                factor_name, instruments=instruments_list, start_date=start, end_date=end, use_cache=not no_cache
            )

            progress.update(task, completed=100)

        console.print(f"因子数据计算完成: [green]{len(factor_data)}[/green] 条记录")

        # 6. 计算未来收益率
        print_info("计算未来收益率...")
        price_data = provider.get_price_data(
            instruments=instruments_list, start_date=start, end_date=end, price_field="close"
        )

        # 计算未来收益率（T+1 到 T+2）
        returns = price_data.groupby(level="instrument").apply(lambda x: x.shift(-2) / x.shift(-1) - 1).droplevel(0)

        # 7. 性能评估
        print_info("评估因子性能...")
        evaluator = PerformanceEvaluator()
        metrics = evaluator.calculate_all(factor_data, returns, quantile=quantile)

        # 8. 策略分析
        if strategy != "none":
            print_info("分析策略场景...")
            strategy_analyzer = StrategyAnalyzer()

            if strategy == "all":
                # 分析所有策略
                with create_progress_bar("策略分析") as progress:
                    task = progress.add_task("分析策略...", total=100)
                    scenario_results = strategy_analyzer.analyze_all_scenarios(
                        factor_data, returns, price_data, top_pct=top_pct
                    )
                    progress.update(task, completed=100)
            elif strategy == "bull":
                scenario_results = {
                    "bull": strategy_analyzer.analyze_bull_strategy(factor_data, returns, top_pct=top_pct)
                }
            elif strategy == "bear":
                scenario_results = {
                    "bear": strategy_analyzer.analyze_bear_strategy(factor_data, returns, top_pct=top_pct)
                }
            elif strategy == "long_short":
                scenario_results = {
                    "long_short": strategy_analyzer.analyze_long_short_strategy(factor_data, returns, top_pct=top_pct)
                }
            elif strategy == "volatility":
                scenario_results = {
                    "volatility": strategy_analyzer.analyze_volatility_strategy(
                        factor_data, returns, price_data, top_pct=top_pct
                    )
                }
        else:
            scenario_results = {}

        # 9. 显示结果
        console.print()
        display_metrics_table(metrics, factor_name)

        if scenario_results and "bull" in scenario_results:
            console.print()
            console.print(
                Panel(
                    "[bold cyan]看涨策略:[/bold cyan]\n"
                    f"  年化收益: [green]{scenario_results['bull']['annual_return']:.2%}[/green]\n"
                    f"  夏普比率: [green]{scenario_results['bull']['sharpe_ratio']:.2f}[/green]\n"
                    f"  最大回撤: [red]{scenario_results['bull']['max_drawdown']:.2%}[/red]\n"
                    f"  胜率: [green]{scenario_results['bull']['win_rate']:.2%}[/green]",
                    title="[bold]策略表现[/bold]",
                    border_style="cyan",
                )
            )

        # 10. 保存结果
        if output:
            print_info(f"保存结果到 {output}...")
            import json

            output_path = Path(output)

            # 准备输出数据
            output_data = {
                "factor": factor_name,
                "instruments": instruments_list,
                "start_date": start,
                "end_date": end,
                "metrics": {
                    "ic_mean": float(metrics["ic_mean"]),
                    "ic_std": float(metrics["ic_std"]),
                    "icir": float(metrics["icir"]),
                    "rank_ic_mean": float(metrics["rank_ic_mean"]),
                    "rank_ic_std": float(metrics["rank_ic_std"]),
                    "rank_icir": float(metrics["rank_icir"]),
                    "annual_return": float(metrics["annual_return"]),
                    "sharpe_ratio": float(metrics["sharpe_ratio"]),
                    "max_drawdown": float(metrics["max_drawdown"]),
                    "win_rate": float(metrics["win_rate"]),
                },
                "strategy_results": {},
            }

            # 添加策略结果
            for strategy_name, result in scenario_results.items():
                output_data["strategy_results"][strategy_name] = {
                    "annual_return": float(result.get("annual_return", 0)),
                    "sharpe_ratio": float(result.get("sharpe_ratio", 0)),
                    "max_drawdown": float(result.get("max_drawdown", 0)),
                    "win_rate": float(result.get("win_rate", 0)),
                }

            # 保存到文件
            if output_path.suffix == ".json":
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(output_data, f, indent=2, ensure_ascii=False)
            elif output_path.suffix == ".csv":
                import pandas as pd

                # 展开为扁平的 CSV
                flat_data = {"factor": factor_name, "start_date": start, "end_date": end, **output_data["metrics"]}
                df = pd.DataFrame([flat_data])
                df.to_csv(output_path, index=False)
            else:
                print_error(f"不支持的输出格式: {output_path.suffix}")

            print_success(f"结果已保存到 {output}")

        console.print()
        print_success("因子分析完成！")

    except Exception as e:
        import traceback

        console.print(traceback.format_exc())
        print_error(f"分析失败: {str(e)}")


# =============================================================================
# batch 命令 - 批量因子分析
# =============================================================================


@cli.command("batch")
@click.option("--config", "-c", required=True, help="配置文件路径（YAML 格式）")
@click.option("--output", "-o", help="输出目录路径")
@click.option("--parallel", "-p", default=1, type=int, help="并行任务数（默认：1）")
@click.option("--verbose", "-v", is_flag=True, help="显示详细输出")
def batch_analyze(config, output, parallel, verbose):
    """
    批量分析多个因子

    [bold cyan]配置文件格式 (YAML):[/bold cyan]

    \b
      factors:
        - name: MA20
          expression: "Ref($close, 20) / $close - 1"
          description: "20日均线偏离度"

        - name: MA60
          expression: "Ref($close, 60) / $close - 1"
          description: "60日均线偏离度"

      analysis:
        instruments: instruments.txt
        start_date: "2020-01-01"
        end_date: "2020-12-31"
        quantile: 0.2
        top_pct: 0.2

    [bold cyan]示例:[/bold cyan]

    \b
      mfa batch -c factors.yaml
      mfa batch -c factors.yaml -o results/ -p 4
    """
    print_banner()

    try:
        # 1. 加载配置文件
        print_info(f"加载配置文件: {config}")
        config_path = Path(config)

        if not config_path.exists():
            print_error(f"配置文件不存在: {config}")

        with open(config_path, "r", encoding="utf-8") as f:
            config_data = yaml.safe_load(f)

        # 2. 解析配置
        factors = config_data.get("factors", [])
        analysis = config_data.get("analysis", {})

        if not factors:
            print_error("配置文件中没有定义因子")

        console.print(f"找到 [green]{len(factors)}[/green] 个因子")

        # 3. 加载股票列表
        instruments_file = analysis.get("instruments")
        if not instruments_file:
            print_error("配置文件中缺少 instruments 字段")

        instruments_path = Path(instruments_file)
        if instruments_path.exists():
            with open(instruments_path, "r", encoding="utf-8") as f:
                instruments_list = [line.strip() for line in f if line.strip()]
        else:
            instruments_list = [i.strip() for i in instruments_file.split(",")]

        console.print(f"加载 [green]{len(instruments_list)}[/green] 个股票")

        start_date = analysis.get("start_date")
        end_date = analysis.get("end_date")
        quantile = analysis.get("quantile", 0.2)
        top_pct = analysis.get("top_pct", 0.2)

        # 4. 初始化
        print_info("初始化数据提供者和因子管理器...")
        provider = FactorDataProvider()
        factor_manager = FactorManager(provider)

        # 5. 注册所有因子
        print_info("注册因子...")
        for factor_def in factors:
            factor_name = factor_def["name"]
            expression = factor_def["expression"]
            metadata = {"description": factor_def.get("description", ""), "type": "expression"}
            factor_manager.register_factor(factor_name, expression, metadata=metadata)
            console.print(f"  - 注册因子: [cyan]{factor_name}[/cyan]")

        # 6. 批量分析
        results = {}

        with create_progress_bar("批量分析") as progress:
            task = progress.add_task("分析因子...", total=len(factors))

            for factor_def in factors:
                factor_name = factor_def["name"]

                try:
                    # 计算因子
                    factor_data = factor_manager.calculate_factor(
                        factor_name, instruments=instruments_list, start_date=start_date, end_date=end_date
                    )

                    # 计算收益率
                    price_data = provider.get_price_data(
                        instruments=instruments_list, start_date=start_date, end_date=end_date, price_field="close"
                    )
                    returns = (
                        price_data.groupby(level="instrument")
                        .apply(lambda x: x.shift(-2) / x.shift(-1) - 1)
                        .droplevel(0)
                    )

                    # 性能评估
                    evaluator = PerformanceEvaluator()
                    metrics = evaluator.calculate_all(factor_data, returns, quantile=quantile)

                    # 策略分析
                    strategy_analyzer = StrategyAnalyzer()
                    scenario_results = strategy_analyzer.analyze_all_scenarios(
                        factor_data, returns, price_data, top_pct=top_pct
                    )

                    results[factor_name] = {
                        "metrics": metrics,
                        "strategy_results": scenario_results,
                        "status": "success",
                    }

                    console.print(
                        f"  ✓ [green]{factor_name}[/green]: "
                        f"IC={metrics['ic_mean']:.4f}, "
                        f"ICIR={metrics['icir']:.4f}, "
                        f"年化收益={metrics['annual_return']:.2%}"
                    )

                except Exception as e:
                    console.print(f"  ✗ [red]{factor_name}[/red]: {str(e)}")
                    results[factor_name] = {"status": "failed", "error": str(e)}

                progress.update(task, advance=1)

        # 7. 生成汇总报告
        console.print()
        console.print(Panel("[bold cyan]批量分析汇总[/bold cyan]", border_style="cyan"))

        # 创建结果表格
        table = Table(show_header=True, header_style="bold magenta")
        table.add_column("因子", style="cyan", width=20)
        table.add_column("IC 均值", justify="right")
        table.add_column("ICIR", justify="right")
        table.add_column("年化收益", justify="right")
        table.add_column("夏普比率", justify="right")
        table.add_column("状态", justify="center")

        for factor_name, result in results.items():
            if result["status"] == "success":
                metrics = result["metrics"]
                status = "[green]成功[/green]"
                table.add_row(
                    factor_name,
                    f"{metrics['ic_mean']:.4f}",
                    f"{metrics['icir']:.4f}",
                    f"{metrics['annual_return']:.2%}",
                    f"{metrics['sharpe_ratio']:.2f}",
                    status,
                )
            else:
                table.add_row(factor_name, "-", "-", "-", "-", "[red]失败[/red]")

        console.print(table)

        # 8. 保存结果
        if output:
            print_info(f"保存结果到 {output}...")
            import json

            output_dir = Path(output)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 保存汇总结果
            summary_file = output_dir / "summary.json"
            summary_data = {
                "config": config,
                "start_date": start_date,
                "end_date": end_date,
                "total_factors": len(factors),
                "successful": sum(1 for r in results.values() if r["status"] == "success"),
                "failed": sum(1 for r in results.values() if r["status"] == "failed"),
                "results": {},
            }

            for factor_name, result in results.items():
                if result["status"] == "success":
                    summary_data["results"][factor_name] = {
                        "metrics": {
                            "ic_mean": float(result["metrics"]["ic_mean"]),
                            "icir": float(result["metrics"]["icir"]),
                            "annual_return": float(result["metrics"]["annual_return"]),
                            "sharpe_ratio": float(result["metrics"]["sharpe_ratio"]),
                            "max_drawdown": float(result["metrics"]["max_drawdown"]),
                            "win_rate": float(result["metrics"]["win_rate"]),
                        }
                    }
                else:
                    summary_data["results"][factor_name] = {"error": result["error"]}

            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)

            # 保存详细结果（每个因子一个文件）
            for factor_name, result in results.items():
                if result["status"] == "success":
                    factor_file = output_dir / f"{factor_name}.json"
                    with open(factor_file, "w", encoding="utf-8") as f:
                        json.dump(result, f, indent=2, ensure_ascii=False, default=str)

            print_success(f"结果已保存到 {output_dir}")

        console.print()
        print_success("批量分析完成！")

    except Exception as e:
        import traceback

        console.print(traceback.format_exc())
        print_error(f"批量分析失败: {str(e)}")


# =============================================================================
# validate 命令 - 表达式验证
# =============================================================================


@cli.command("validate")
@click.argument("expression")
@click.option("--verbose", "-v", is_flag=True, help="显示详细检查信息")
def validate_expression(expression, verbose):
    """
    验证因子表达式

    检查表达式是否包含未来函数或其他语法错误。

    [bold cyan]示例:[/bold cyan]

    \b
      mfa validate "Ref($close, 20) / $close - 1"
      mfa validate "$close / Ref($close, -5)"  # 会报错，包含未来函数
    """
    print_banner()

    try:
        print_info("验证表达式...")
        console.print(f"表达式: [cyan]{expression}[/cyan]")

        # 导入表达式解析器
        from src.core.factor_expression_parser import FactorExpressionParser

        parser = FactorExpressionParser()

        # 验证表达式
        parser.validate_no_future_functions(expression)

        # 提取字段
        if verbose:
            fields = parser.extract_fields(expression)
            console.print()
            console.print(
                Panel(
                    "[bold cyan]提取的字段:[/bold cyan]\n" + "\n".join([f"  - ${field}" for field in fields]),
                    title="[bold]详细信息[/bold]",
                    border_style="green",
                )
            )

        console.print()
        print_success("表达式验证通过！")
        console.print("  - 未检测到未来函数")
        console.print("  - 语法检查通过")

    except ValueError as e:
        console.print()
        print_error(f"表达式验证失败: {str(e)}")
    except Exception as e:
        import traceback

        console.print(traceback.format_exc())
        print_error(f"验证失败: {str(e)}")


# =============================================================================
# report 命令 - 生成报告
# =============================================================================


@cli.command("report")
@click.option("--input", "-i", required=True, help="输入目录或文件路径")
@click.option("--output", "-o", required=True, help="输出报告文件路径")
@click.option(
    "--format", "-f", type=click.Choice(["html", "pdf", "json"]), default="html", help="报告格式（默认：html）"
)
@click.option("--title", "-t", default="因子分析报告", help="报告标题")
def generate_report(input, output, format, title):
    """
    生成分析报告

    [bold cyan]示例:[/bold cyan]

    \b
      mfa report -i results/ -o report.html
      mfa report -i summary.json -o report.pdf --title "我的因子分析"
    """
    print_banner()

    try:
        print_info(f"生成{format.upper()}报告...")

        input_path = Path(input)
        output_path = Path(output)

        # 读取输入数据
        if input_path.is_file():
            # 单个文件
            import json

            with open(input_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if input_path.name == "summary.json":
                # 汇总报告
                results = data.get("results", {})
            else:
                # 单个因子报告
                results = {input_path.stem: data}
        elif input_path.is_dir():
            # 目录，读取所有 JSON 文件
            import json

            results = {}

            for json_file in input_path.glob("*.json"):
                if json_file.name == "summary.json":
                    continue

                with open(json_file, "r", encoding="utf-8") as f:
                    factor_data = json.load(f)
                    results[json_file.stem] = factor_data

            if not results:
                print_error(f"目录中没有找到结果文件: {input}")
        else:
            print_error(f"输入路径不存在: {input}")

        console.print(f"加载 [green]{len(results)}[/green] 个因子的结果")

        # 生成报告
        if format == "html":
            _generate_html_report(results, output_path, title)
        elif format == "json":
            _generate_json_report(results, output_path, title)
        elif format == "pdf":
            print_warning("PDF 报告生成功能尚未实现")
            print_info("请先生成 HTML 报告，然后使用浏览器打印为 PDF")
            return

        print_success(f"报告已生成: {output}")

    except Exception as e:
        import traceback

        console.print(traceback.format_exc())
        print_error(f"报告生成失败: {str(e)}")


def _generate_html_report(results, output_path, title):
    """生成 HTML 报告"""
    # 计算统计信息
    total = len(results)
    success_count = sum(1 for r in results.values() if r.get("status") != "failed")
    failed_count = sum(1 for r in results.values() if r.get("status") == "failed")

    # 构建因子对比表格
    comparison_rows = []
    for factor_name, result in results.items():
        if result.get("status") == "failed":
            continue

        metrics = result.get("metrics", {})
        annual_return = metrics.get("annual_return", 0)
        annual_return_class = "positive" if annual_return > 0 else "negative"

        row = f"""
            <tr>
                <td>{factor_name}</td>
                <td>{metrics.get('ic_mean', 0):.4f}</td>
                <td>{metrics.get('icir', 0):.4f}</td>
                <td class="{annual_return_class}">{annual_return:.2%}</td>
                <td>{metrics.get('sharpe_ratio', 0):.2f}</td>
                <td class="negative">{metrics.get('max_drawdown', 0):.2%}</td>
                <td>{metrics.get('win_rate', 0):.2%}</td>
            </tr>"""
        comparison_rows.append(row)

    comparison_table = "\n".join(comparison_rows)

    # 构建每个因子的详细信息
    factor_details = []
    for factor_name, result in results.items():
        if result.get("status") == "failed":
            continue

        metrics = result.get("metrics", {})
        annual_return = metrics.get("annual_return", 0)
        annual_return_class = "positive" if annual_return > 0 else "negative"

        detail = f"""
        <div class="factor-section">
            <h2>{factor_name}</h2>

            <h3>性能指标</h3>
            <div class="metric">
                <div class="metric-label">IC 均值</div>
                <div class="metric-value">{metrics.get('ic_mean', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">ICIR</div>
                <div class="metric-value">{metrics.get('icir', 0):.4f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">年化收益率</div>
                <div class="metric-value {annual_return_class}">{annual_return:.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">夏普比率</div>
                <div class="metric-value">{metrics.get('sharpe_ratio', 0):.2f}</div>
            </div>
            <div class="metric">
                <div class="metric-label">最大回撤</div>
                <div class="metric-value negative">{metrics.get('max_drawdown', 0):.2%}</div>
            </div>
            <div class="metric">
                <div class="metric-label">胜率</div>
                <div class="metric-value">{metrics.get('win_rate', 0):.2%}</div>
            </div>
        </div>"""
        factor_details.append(detail)

    factors_detail_html = "\n".join(factor_details)

    # 组装完整的 HTML
    html_template = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{title}</title>
    <style>
        body {{
            font-family: 'Microsoft YaHei', Arial, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }}
        .container {{
            max-width: 1200px;
            margin: 0 auto;
            background-color: white;
            padding: 30px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        h1 {{
            color: #2c3e50;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #34495e;
            border-left: 5px solid #3498db;
            padding-left: 10px;
            margin-top: 30px;
        }}
        h3 {{
            color: #555;
            margin-top: 20px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #ddd;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:hover {{
            background-color: #f5f5f5;
        }}
        .metric {{
            display: inline-block;
            margin: 10px;
            padding: 15px;
            background-color: #ecf0f1;
            border-radius: 5px;
            min-width: 150px;
        }}
        .metric-label {{
            font-size: 12px;
            color: #7f8c8d;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #2c3e50;
        }}
        .positive {{
            color: #27ae60;
        }}
        .negative {{
            color: #e74c3c;
        }}
        .factor-section {{
            margin: 30px 0;
            padding: 20px;
            background-color: #f9f9f9;
            border-radius: 5px;
        }}
    </style>
</head>
<body>
    <div class="container">
        <h1>{title}</h1>

        <h2>汇总统计</h2>
        <table>
            <tr>
                <th>因子数量</th>
                <th>成功</th>
                <th>失败</th>
            </tr>
            <tr>
                <td>{total}</td>
                <td>{success_count}</td>
                <td>{failed_count}</td>
            </tr>
        </table>

        <h2>因子对比</h2>
        <table>
            <tr>
                <th>因子</th>
                <th>IC 均值</th>
                <th>ICIR</th>
                <th>年化收益</th>
                <th>夏普比率</th>
                <th>最大回撤</th>
                <th>胜率</th>
            </tr>
{comparison_table}
        </table>
{factors_detail_html}
    </div>
</body>
</html>
"""

    # 保存 HTML 文件
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_template)


def _generate_json_report(results, output_path, title):
    """生成 JSON 报告"""
    import json

    report_data = {"title": title, "total_factors": len(results), "results": results}

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2, ensure_ascii=False, default=str)


# =============================================================================
# 主程序入口
# =============================================================================

if __name__ == "__main__":
    cli()
