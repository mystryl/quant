"""
可靠性评估模块简化示例

演示可靠性评估器和相关性分析器的基本功能。
"""

import sys
from pathlib import Path

# 添加 src 目录到路径
src_path = Path(__file__).parent.parent / 'src'
sys.path.insert(0, str(src_path))

import numpy as np
import pandas as pd

from core.reliability import ReliabilityEvaluator
from core.correlation_analyzer import FactorCorrelationAnalyzer
from core.config import (
    get_weights,
    get_thresholds,
    get_reliability_grade,
    DEFAULT_WEIGHTS,
    CONSERVATIVE_WEIGHTS,
    AGGRESSIVE_WEIGHTS,
)


def demo_1_reliability_evaluation():
    """演示 1: 因子可靠性评估"""
    print("=" * 70)
    print("演示 1: 因子可靠性评估")
    print("=" * 70)

    # 创建示例性能指标
    print("\n创建示例性能指标...")
    metrics = {
        'ic_mean': 0.05,           # IC 均值
        'icir': 0.8,               # IC 信息比率
        'rank_ic_mean': 0.06,      # Rank IC 均值
        'rank_icir': 0.9,          # Rank IC 信息比率
        'long_short_return': pd.Series([0.01, 0.02, -0.01, 0.03, 0.01]),
        'annual_return': 0.12,      # 年化收益率 12%
        'sharpe_ratio': 2.5,       # 夏普比率
        'max_drawdown': -0.08,     # 最大回撤 8%
        'win_rate': 0.65,          # 胜率 65%
        'calmar_ratio': 1.5,       # 卡玛比率
    }

    print(f"  IC 均值: {metrics['ic_mean']:.3f}")
    print(f"  ICIR: {metrics['icir']:.2f}")
    print(f"  年化收益: {metrics['annual_return']:.2%}")
    print(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")
    print(f"  胜率: {metrics['win_rate']:.2%}")

    # 使用默认配置评估
    print("\n" + "-" * 70)
    print("使用默认配置评估:")
    print("-" * 70)

    evaluator = ReliabilityEvaluator()
    result = evaluator.evaluate(metrics, factor_name="MA20")

    print(f"\n综合评分: {result['total_score']:.2f} / 1.00")
    print(f"可靠性等级: {result['reliability']}")

    print("\n各维度得分:")
    for key, score in result['scores'].items():
        print(f"  {evaluator._get_dimension_name(key):12s}: {score:.3f}")

    # 使用保守型配置评估
    print("\n" + "-" * 70)
    print("使用保守型配置评估:")
    print("-" * 70)

    conservative_evaluator = ReliabilityEvaluator(strategy_type='conservative')
    conservative_result = conservative_evaluator.evaluate(metrics, factor_name="MA20")

    print(f"\n综合评分: {conservative_result['total_score']:.2f} / 1.00")
    print(f"可靠性等级: {conservative_result['reliability']}")

    # 使用激进型配置评估
    print("\n" + "-" * 70)
    print("使用激进型配置评估:")
    print("-" * 70)

    aggressive_evaluator = ReliabilityEvaluator(strategy_type='aggressive')
    aggressive_result = aggressive_evaluator.evaluate(metrics, factor_name="MA20")

    print(f"\n综合评分: {aggressive_result['total_score']:.2f} / 1.00")
    print(f"可靠性等级: {aggressive_result['reliability']}")


def demo_2_correlation_analysis():
    """演示 2: 因子相关性分析"""
    print("\n" + "=" * 70)
    print("演示 2: 因子相关性分析")
    print("=" * 70)

    # 创建示例因子数据
    print("\n创建示例因子数据...")
    np.random.seed(42)

    dates = pd.date_range('2020-01-01', '2020-01-31', freq='D')
    instruments = ['SH600000', 'SH600001', 'SH600002']

    index = pd.MultiIndex.from_product(
        [dates, instruments],
        names=['datetime', 'instrument']
    )

    n = len(index)

    # 创建相关和不相关的因子
    factor_dict = {
        # MA20 和 MA60 高度相关
        'MA20': pd.Series(np.random.randn(n), index=index),
        'MA60': None,
        # RSI 独立
        'RSI': pd.Series(np.random.randn(n), index=index),
        # MACD 独立
        'MACD': pd.Series(np.random.randn(n), index=index),
    }

    # MA60 与 MA20 高度相关
    factor_dict['MA60'] = factor_dict['MA20'] * 0.9 + np.random.randn(n) * 0.1

    print(f"  因子数量: {len(factor_dict)}")
    print(f"  每个因子观测值: {n}")

    # 分析相关性
    print("\n" + "-" * 70)
    print("分析因子相关性:")
    print("-" * 70)

    analyzer = FactorCorrelationAnalyzer(method='spearman', threshold=0.7)
    result = analyzer.analyze_correlation(factor_dict)

    # 显示相关性矩阵
    print("\n相关性矩阵:")
    print(result['correlation_matrix'].round(3))

    # 显示高度相关的因子对
    print(f"\n高度相关的因子对 (阈值={analyzer.threshold}):")
    if result['high_correlation_pairs']:
        for pair in result['high_correlation_pairs']:
            print(
                f"  - {pair['factor1']} 和 {pair['factor2']}: "
                f"{pair['correlation']:.3f} ({pair['level']})"
            )
    else:
        print("  无高度相关的因子对")

    # 显示统计信息
    print("\n相关性统计:")
    stats = result['statistics']
    print(f"  平均相关系数: {stats['mean_correlation']:.3f}")
    print(f"  最大相关系数: {stats['max_correlation']:.3f}")
    print(f"  最小相关系数: {stats['min_correlation']:.3f}")
    print(f"  高相关因子对数量: {stats['high_correlation_count']}")

    # 显示建议
    print("\n建议:")
    print(result['recommendation'])


def demo_3_custom_weights():
    """演示 3: 自定义权重配置"""
    print("\n" + "=" * 70)
    print("演示 3: 自定义权重配置")
    print("=" * 70)

    # 查看预定义权重
    print("\n预定义权重配置:")

    print("\n默认权重:")
    for key, value in DEFAULT_WEIGHTS.items():
        print(f"  {key:20s}: {value:.2%}")

    print("\n保守型权重:")
    for key, value in CONSERVATIVE_WEIGHTS.items():
        print(f"  {key:20s}: {value:.2%}")

    print("\n激进型权重:")
    for key, value in AGGRESSIVE_WEIGHTS.items():
        print(f"  {key:20s}: {value:.2%}")

    # 创建自定义权重
    print("\n" + "-" * 70)
    print("创建自定义权重（更注重收益）:")
    print("-" * 70)

    custom_weights = {
        'ic_stability': 0.25,      # 降低稳定性权重
        'ic_absolute': 0.15,
        'ir': 0.15,
        'long_short_return': 0.35,  # 大幅提高收益权重
        'win_rate': 0.10,
    }

    print("\n自定义权重:")
    for key, value in custom_weights.items():
        print(f"  {key:20s}: {value:.2%}")

    # 使用自定义权重评估
    metrics = {
        'ic_mean': 0.05,
        'icir': 0.8,
        'rank_ic_mean': 0.06,
        'rank_icir': 0.9,
        'long_short_return': pd.Series([0.01, 0.02, -0.01, 0.03, 0.01]),
        'annual_return': 0.12,
        'sharpe_ratio': 2.5,
        'max_drawdown': -0.08,
        'win_rate': 0.65,
    }

    evaluator = ReliabilityEvaluator(weights=custom_weights)
    result = evaluator.evaluate(metrics, factor_name="MA20")

    print(f"\n评估结果:")
    print(f"  综合评分: {result['total_score']:.2f}")
    print(f"  可靠性等级: {result['reliability']}")


def demo_4_grade_system():
    """演示 4: 可靠性等级系统"""
    print("\n" + "=" * 70)
    print("演示 4: 可靠性等级系统")
    print("=" * 70)

    # 显示等级定义
    print("\n可靠性等级定义:")
    print("-" * 70)

    grades_info = [
        ('A+', 0.95, '优秀'),
        ('A', 0.85, '良好'),
        ('B', 0.75, '中等'),
        ('C', 0.65, '一般'),
        ('D', 0.55, '较差'),
        ('F', 0.45, '失败'),
    ]

    for grade, score, description in grades_info:
        print(f"  {grade:2s} ({score:.2f}): {description}")

    # 测试评分到等级的映射
    print("\n评分到等级的映射:")
    print("-" * 70)

    test_scores = [0.95, 0.85, 0.75, 0.65, 0.55, 0.45]
    for score in test_scores:
        grade = get_reliability_grade(score)
        print(f"  评分 {score:.2f} -> 等级 {grade}")


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("可靠性评估模块简化示例")
    print("=" * 70)

    # 运行各个演示
    demo_1_reliability_evaluation()
    demo_2_correlation_analysis()
    demo_3_custom_weights()
    demo_4_grade_system()

    print("\n" + "=" * 70)
    print("所有演示运行完成！")
    print("=" * 70)


if __name__ == "__main__":
    main()
