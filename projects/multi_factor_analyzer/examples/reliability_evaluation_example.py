"""
可靠性评估示例

演示如何使用可靠性评估器和相关性分析器评估因子。
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd

from core.performance_eval import PerformanceEvaluator
from core.strategy_analyzer import StrategyAnalyzer
from core.reliability import ReliabilityEvaluator, evaluate_factor_reliability
from core.correlation_analyzer import FactorCorrelationAnalyzer, analyze_factor_correlation


def generate_sample_data():
    """生成示例数据"""
    print("=" * 70)
    print("生成示例数据")
    print("=" * 70)

    np.random.seed(42)

    # 日期和股票
    dates = pd.date_range('2020-01-01', '2020-12-31', freq='D')
    instruments = [f'SH60000{i}' for i in range(10)]

    # 创建多索引
    index = pd.MultiIndex.from_product(
        [dates, instruments],
        names=['datetime', 'instrument']
    )

    n = len(index)

    # 生成价格数据
    price = 100 + np.cumsum(np.random.randn(n) * 0.5)
    price_df = pd.Series(price, index=index).to_frame('close')

    # 生成因子数据
    # MA20: 优秀因子
    ma20_signal = np.random.randn(n) * 0.5 + np.sin(np.arange(n) / 50) * 2
    ma20_df = pd.Series(ma20_signal, index=index)

    # MA60: 与 MA20 高度相关
    ma60_signal = ma20_signal * 0.9 + np.random.randn(n) * 0.2
    ma60_df = pd.Series(ma60_signal, index=index)

    # RSI: 中等因子
    rsi_signal = np.random.randn(n) * 0.3
    rsi_df = pd.Series(rsi_signal, index=index)

    # MACD: 较差因子
    macd_signal = np.random.randn(n) * 0.1
    macd_df = pd.Series(macd_signal, index=index)

    # 生成未来收益率（T+1 到 T+2）
    future_return = pd.Series(
        np.random.randn(n) * 0.02,
        index=index
    )

    # 让优秀因子与收益率有一定相关性
    future_return = future_return + ma20_signal * 0.01

    print(f"\n数据概览:")
    print(f"  日期范围: {dates[0].date()} 到 {dates[-1].date()}")
    print(f"  股票数量: {len(instruments)}")
    print(f"  总观测值: {n}")
    print(f"  因子数量: 4 (MA20, MA60, RSI, MACD)")

    return {
        'price': price_df,
        'factors': {
            'MA20': ma20_df,
            'MA60': ma60_df,
            'RSI': rsi_df,
            'MACD': macd_df,
        },
        'returns': future_return,
    }


def example_1_single_factor_reliability(data):
    """示例 1: 单因子可靠性评估"""
    print("\n" + "=" * 70)
    print("示例 1: 单因子可靠性评估 (MA20)")
    print("=" * 70)

    # 1. 计算性能指标
    print("\n步骤 1: 计算性能指标...")
    perf_evaluator = PerformanceEvaluator()
    metrics = perf_evaluator.calculate_all(
        pred=data['factors']['MA20'],
        label=data['returns']
    )

    print(f"  IC 均值: {metrics['ic_mean']:.4f}")
    print(f"  ICIR: {metrics['icir']:.2f}")
    print(f"  年化收益: {metrics['annual_return']:.2%}")
    print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"  胜率: {metrics['win_rate']:.2%}")

    # 2. 计算策略场景
    print("\n步骤 2: 计算策略场景...")
    strategy_analyzer = StrategyAnalyzer()
    scenario_results = strategy_analyzer.analyze_all_scenarios(
        data['factors']['MA20'].to_frame('factor'),
        data['returns'].to_frame('return'),
        price_df=data['price']
    )

    # 注意：strategy_analyzer 需要 DataFrame，所以这里转换一下

    print(f"  看涨策略年化收益: {scenario_results['bull']['annual_return']:.2%}")
    print(f"  多空策略年化收益: {scenario_results['long_short']['annual_return']:.2%}")

    # 3. 可靠性评估（默认配置）
    print("\n步骤 3: 可靠性评估（默认配置）...")
    default_evaluator = ReliabilityEvaluator()
    default_result = default_evaluator.evaluate(
        metrics,
        scenario_results,
        factor_name="MA20"
    )

    print(f"\n评估结果:")
    print(f"  综合评分: {default_result['total_score']:.2f}")
    print(f"  可靠性等级: {default_result['reliability']}")

    # 4. 可靠性评估（保守型配置）
    print("\n步骤 4: 可靠性评估（保守型配置）...")
    conservative_evaluator = ReliabilityEvaluator(strategy_type='conservative')
    conservative_result = conservative_evaluator.evaluate(
        metrics,
        scenario_results,
        factor_name="MA20"
    )

    print(f"\n评估结果:")
    print(f"  综合评分: {conservative_result['total_score']:.2f}")
    print(f"  可靠性等级: {conservative_result['reliability']}")

    # 5. 生成详细报告
    print("\n步骤 5: 生成详细报告...")
    report = default_evaluator.generate_report(default_result, "MA20")
    print(report)


def example_2_multiple_factors_comparison(data):
    """示例 2: 多因子比较"""
    print("\n" + "=" * 70)
    print("示例 2: 多因子可靠性比较")
    print("=" * 70)

    perf_evaluator = PerformanceEvaluator()
    strategy_analyzer = StrategyAnalyzer()
    reliability_evaluator = ReliabilityEvaluator()

    results = {}

    for factor_name, factor_df in data['factors'].items():
        print(f"\n评估因子: {factor_name}")

        # 计算性能指标
        metrics = perf_evaluator.calculate_all(
            pred=factor_df,
            label=data['returns']
        )

        # 计算策略场景
        scenario_results = strategy_analyzer.analyze_bull_strategy(
            factor_df.to_frame('factor'),
            data['returns'].to_frame('return')
        )

        # 可靠性评估
        result = reliability_evaluator.evaluate(
            metrics,
            {'bull': scenario_results},
            factor_name=factor_name
        )

        results[factor_name] = result

        print(f"  综合评分: {result['total_score']:.2f}")
        print(f"  可靠性等级: {result['reliability']}")

    # 排序并显示结果
    print("\n" + "-" * 70)
    print("因子可靠性排名:")
    print("-" * 70)

    sorted_factors = sorted(
        results.items(),
        key=lambda x: x[1]['total_score'],
        reverse=True
    )

    for rank, (factor_name, result) in enumerate(sorted_factors, 1):
        print(
            f"  {rank}. {factor_name:8s} - "
            f"评分: {result['total_score']:.2f}, "
            f"等级: {result['reliability']}"
        )


def example_3_correlation_analysis(data):
    """示例 3: 因子相关性分析"""
    print("\n" + "=" * 70)
    print("示例 3: 因子相关性分析")
    print("=" * 70)

    # 1. 分析相关性
    print("\n步骤 1: 分析因子相关性...")
    analyzer = FactorCorrelationAnalyzer(method='spearman', threshold=0.7)
    result = analyzer.analyze_correlation(data['factors'])

    # 2. 显示相关性矩阵
    print("\n相关性矩阵:")
    print(result['correlation_matrix'].round(3))

    # 3. 显示高度相关的因子对
    print(f"\n高度相关的因子对 (阈值={analyzer.threshold}):")
    if result['high_correlation_pairs']:
        for pair in result['high_correlation_pairs']:
            print(
                f"  - {pair['factor1']} 和 {pair['factor2']}: "
                f"{pair['correlation']:.3f} ({pair['level']})"
            )
    else:
        print("  无高度相关的因子对")

    # 4. 显示统计信息
    print("\n相关性统计:")
    stats = result['statistics']
    print(f"  平均相关系数: {stats['mean_correlation']:.3f}")
    print(f"  最大相关系数: {stats['max_correlation']:.3f}")
    print(f"  最小相关系数: {stats['min_correlation']:.3f}")
    print(f"  高相关因子对数量: {stats['high_correlation_count']}")

    # 5. 显示建议
    print("\n建议:")
    print(result['recommendation'])


def example_4_custom_weights(data):
    """示例 4: 自定义权重配置"""
    print("\n" + "=" * 70)
    print("示例 4: 自定义权重配置")
    print("=" * 70)

    # 自定义权重：更注重收益，不太关注稳定性
    custom_weights = {
        'ic_stability': 0.25,      # 降低稳定性权重
        'ic_absolute': 0.15,
        'ir': 0.15,
        'long_short_return': 0.35,  # 大幅提高收益权重
        'win_rate': 0.10,
    }

    print("\n自定义权重配置:")
    for key, value in custom_weights.items():
        print(f"  {key}: {value:.2%}")

    # 使用自定义权重评估
    evaluator = ReliabilityEvaluator(weights=custom_weights)

    # 评估 MA20
    perf_evaluator = PerformanceEvaluator()
    metrics = perf_evaluator.calculate_all(
        pred=data['factors']['MA20'],
        label=data['returns']
    )

    result = evaluator.evaluate(metrics, factor_name="MA20")

    print(f"\n评估结果:")
    print(f"  综合评分: {result['total_score']:.2f}")
    print(f"  可靠性等级: {result['reliability']}")

    print("\n各维度得分:")
    for key, score in result['scores'].items():
        print(f"  {evaluator._get_dimension_name(key)}: {score:.3f}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("可靠性评估模块示例")
    print("=" * 70)

    # 生成示例数据
    data = generate_sample_data()

    # 运行各个示例
    example_1_single_factor_reliability(data)
    example_2_multiple_factors_comparison(data)
    example_3_correlation_analysis(data)
    example_4_custom_weights(data)

    print("\n" + "=" * 70)
    print("所有示例运行完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
