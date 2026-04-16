"""
因子管理和性能评估综合示例

本示例展示如何使用因子管理器和性能评估器来完成以下任务：
1. 定义和注册因子
2. 计算因子值
3. 评估因子性能
4. 生成性能报告
"""

import numpy as np
import pandas as pd
from pathlib import Path
import sys

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.core.factor_expression_parser import FactorExpressionParser
from src.core.performance_eval import PerformanceEvaluator


def create_mock_data():
    """
    创建模拟数据用于演示

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (因子值, 未来收益率)
    """
    print("=" * 60)
    print("创建模拟数据")
    print("=" * 60)

    # 设置随机种子以保证可重复性
    np.random.seed(42)

    # 创建日期和股票代码
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    instruments = [f'SH{i:06d}' for i in range(600000, 600010)]

    # 创建多索引
    index = pd.MultiIndex.from_product(
        [dates, instruments],
        names=['datetime', 'instrument']
    )

    # 生成模拟因子值（动量因子）
    # 使用随机游走生成具有一定持续性的因子
    n_samples = len(index)
    factor_values = np.zeros(n_samples)

    # 为每个股票生成因子序列
    for i, instrument in enumerate(instruments):
        idx_start = i * len(dates)
        idx_end = (i + 1) * len(dates)

        # 生成具有自相关性的序列（模拟真实因子）
        factor_values[idx_start:idx_end] = np.random.randn(len(dates)).cumsum()

    # 标准化因子值
    factor_values = (factor_values - factor_values.mean()) / factor_values.std()

    # 创建因子 DataFrame
    factor_df = pd.Series(factor_values, index=index, name='factor').to_frame()

    # 生成未来收益率（与因子有一定相关性）
    # 真实的收益率应该由数据提供者计算，这里仅用于演示
    ic_true = 0.05  # 设置真实的 IC
    noise_std = 0.1

    returns = (
        factor_values * ic_true +  # 因子贡献
        np.random.randn(n_samples) * noise_std  # 随机噪声
    )

    # 创建收益率 DataFrame
    return_df = pd.Series(returns, index=index, name='return').to_frame()

    print(f"✓ 数据创建完成")
    print(f"  时间范围: {dates[0].date()} 到 {dates[-1].date()}")
    print(f"  股票数量: {len(instruments)}")
    print(f"  观测值数量: {len(index):,}")
    print(f"  因子值范围: [{factor_values.min():.2f}, {factor_values.max():.2f}]")
    print(f"  收益率范围: [{returns.min():.2%}, {returns.max():.2%}]")

    return factor_df, return_df


def example_1_validate_expression():
    """
    示例 1: 验证因子表达式
    """
    print("\n" + "=" * 60)
    print("示例 1: 验证因子表达式")
    print("=" * 60)

    parser = FactorExpressionParser()

    # 安全的表达式
    safe_expressions = [
        "Ref($close, 20) / $close - 1",  # 20日收益率
        "Mean($close, 5) / Mean($close, 20) - 1",  # 短期/长期均线比
        "Std($close, 20) / Mean($close, 20)",  # 波动率
        "$close / Ref($close, 1) - 1",  # 日收益率
    ]

    print("\n安全的表达式:")
    for expr in safe_expressions:
        try:
            is_valid = parser.validate_no_future_functions(expr)
            if is_valid:
                print(f"  ✓ {expr}")
        except ValueError as e:
            print(f"  ✗ {expr}")
            print(f"    错误: {e}")

    # 危险的表达式
    dangerous_expressions = [
        "Ref($close, -1)",  # 引用未来数据
        "$close[-1]",  # 负索引
        "Ref($close, -5) / Ref($close, -1) - 1",  # 未来收益率
    ]

    print("\n危险的表达式（包含未来函数）:")
    for expr in dangerous_expressions:
        try:
            is_valid = parser.validate_no_future_functions(expr)
            print(f"  ✗ {expr} (应该被拒绝但通过了)")
        except ValueError as e:
            print(f"  ✓ {expr}")
            print(f"    正确拒绝: {str(e)[:80]}...")

    # 提取表达式字段
    print("\n提取表达式字段:")
    expr = "($close - Ref($low, 5)) / Ref($volume, 10)"
    fields = parser.extract_fields(expr)
    print(f"  表达式: {expr}")
    print(f"  提取的字段: {fields}")

    # 复杂度分析
    print("\n复杂度分析:")
    simple_expr = "$close"
    complex_expr = "Ref($close, 5) / Ref($close, 20) - 1 + Mean($volume, 10)"

    for expr in [simple_expr, complex_expr]:
        score = parser.get_complexity_score(expr)
        print(f"  {expr}")
        print(f"    复杂度分数: {score:.1f}/100")


def example_2_evaluate_performance():
    """
    示例 2: 评估因子性能
    """
    print("\n" + "=" * 60)
    print("示例 2: 评估因子性能")
    print("=" * 60)

    # 创建模拟数据
    factor_df, return_df = create_mock_data()

    # 创建评估器
    evaluator = PerformanceEvaluator(use_log_return=False)

    # 计算所有指标
    print("\n计算性能指标...")
    metrics = evaluator.calculate_all(
        pred=factor_df['factor'],
        label=return_df['return'],
        quantile=0.2
    )

    # 打印结果
    print("\n性能指标:")
    print("-" * 60)

    print("\n1. 预测能力指标")
    print(f"   IC 均值:     {metrics['ic_mean']:>10.4f}")
    print(f"   IC 标准差:   {metrics['ic_std']:>10.4f}")
    print(f"   ICIR:        {metrics['icir']:>10.4f}")
    print(f"   Rank IC 均值: {metrics['rank_ic_mean']:>10.4f}")
    print(f"   Rank IC 标准差: {metrics['rank_ic_std']:>10.4f}")
    print(f"   Rank ICIR:    {metrics['rank_icir']:>10.4f}")

    print("\n2. 交易效果指标")
    print(f"   年化收益率:   {metrics['annual_return']:>10.2%}")
    print(f"   夏普比率:    {metrics['sharpe_ratio']:>10.2f}")
    print(f"   最大回撤:    {metrics['max_drawdown']:>10.2%}")
    print(f"   胜率:       {metrics['win_rate']:>10.2%}")

    # 打印完整报告
    print("\n" + "=" * 60)
    print("完整性能报告")
    print("=" * 60)
    report = evaluator.generate_report(metrics, "模拟动量因子")
    print(report)


def example_3_compare_factors():
    """
    示例 3: 比较多个因子
    """
    print("\n" + "=" * 60)
    print("示例 3: 比较多个因子")
    print("=" * 60)

    # 创建模拟数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    instruments = [f'SH{i:06d}' for i in range(600000, 600005)]

    index = pd.MultiIndex.from_product(
        [dates, instruments],
        names=['datetime', 'instrument']
    )

    # 创建三个不同质量的因子
    factors = {}

    # 高质量因子（高 IC，高 ICIR）
    factors['高质量'] = (
        np.random.randn(len(index)) * 0.1 +  # 基础因子
        np.random.randn(len(index)) * 0.02  # 噪声
    )

    # 中等质量因子（中等 IC，中等 ICIR）
    factors['中等质量'] = (
        np.random.randn(len(index)) * 0.05 +
        np.random.randn(len(index)) * 0.1
    )

    # 低质量因子（低 IC，低 ICIR）
    factors['低质量'] = (
        np.random.randn(len(index)) * 0.01 +
        np.random.randn(len(index)) * 0.15
    )

    # 创建收益率
    returns = factors['高质量'] * 0.05 + np.random.randn(len(index)) * 0.1

    # 评估每个因子
    evaluator = PerformanceEvaluator()

    print("\n因子对比:")
    print("-" * 100)
    print(f"{'因子名称':<15} {'IC均值':<10} {'ICIR':<10} {'年化收益':<12} {'夏普比率':<10} {'评级':<10}")
    print("-" * 100)

    results = {}
    for factor_name, factor_values in factors.items():
        factor_series = pd.Series(factor_values, index=index)
        return_series = pd.Series(returns, index=index)

        metrics = evaluator.calculate_all(factor_series, return_series)
        results[factor_name] = metrics

        # 计算评级
        ic_abs = abs(metrics['ic_mean'])
        if ic_abs > 0.05:
            grade = 'A'
        elif ic_abs > 0.03:
            grade = 'B'
        elif ic_abs > 0.01:
            grade = 'C'
        else:
            grade = 'D'

        print(f"{factor_name:<15} {metrics['ic_mean']:<10.4f} "
              f"{metrics['icir']:<10.4f} {metrics['annual_return']:<12.2%} "
              f"{metrics['sharpe_ratio']:<10.2f} {grade:<10}")

    print("-" * 100)

    # 推荐
    best_factor = max(results.keys(), key=lambda k: results[k]['icir'])
    print(f"\n推荐: {best_factor} 因子表现最佳（ICIR: {results[best_factor]['icir']:.4f}）")


def example_4_log_vs_simple_return():
    """
    示例 4: 对数收益率 vs 简单收益率
    """
    print("\n" + "=" * 60)
    print("示例 4: 对数收益率 vs 简单收益率")
    print("=" * 60)

    # 创建模拟价格数据
    np.random.seed(42)
    dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
    instruments = ['SH600000', 'SH600001']

    index = pd.MultiIndex.from_product(
        [dates, instruments],
        names=['datetime', 'instrument']
    )

    # 生成价格序列（随机游走）
    price_df = pd.DataFrame(
        {
            'close': 100 + np.random.randn(len(index)).cumsum()
        },
        index=index
    )

    evaluator = PerformanceEvaluator()

    # 计算简单收益率
    simple_return = evaluator.calculate_simple_return(price_df, shift=2)

    # 计算对数收益率
    log_return = evaluator.calculate_log_return(price_df, shift=2)

    print("\n收益率对比（前5行）:")
    print("-" * 60)
    comparison = pd.DataFrame({
        '价格': price_df['close'].head(5),
        '简单收益率': simple_return['close'].head(5),
        '对数收益率': log_return['close'].head(5),
    })
    print(comparison)

    print("\n统计特性:")
    print("-" * 60)
    print(f"简单收益率均值:   {simple_return['close'].mean():>.6f}")
    print(f"对数收益率均值:   {log_return['close'].mean():>.6f}")
    print(f"简单收益率标准差: {simple_return['close'].std():>.6f}")
    print(f"对数收益率标准差: {log_return['close'].std():>.6f}")

    print("\n使用建议:")
    print("  - 简单收益率: 更直观，适合短期策略")
    print("  - 对数收益率: 可加性好，适合长期策略或高波动资产")


def main():
    """
    运行所有示例
    """
    print("\n" + "=" * 60)
    print("因子管理和性能评估综合示例")
    print("=" * 60)

    # 运行示例
    example_1_validate_expression()
    example_2_evaluate_performance()
    example_3_compare_factors()
    example_4_log_vs_simple_return()

    print("\n" + "=" * 60)
    print("所有示例运行完成!")
    print("=" * 60)


if __name__ == "__main__":
    main()
