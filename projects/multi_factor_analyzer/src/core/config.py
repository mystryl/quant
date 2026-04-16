"""
可靠性评估配置文件

定义不同策略类型的权重配置和评分阈值。

权重配置理论依据：
1. Grinold & Kahn, "Active Portfolio Management" - IC 和 IR 的理论基础
2. Sharpe, "Information Ratio and Performance" - 信息比率的应用
3. Jacobs & Levy, "Long-Short Equity Strategies" - 多空收益的重要性
4. Kaufman, "Trading Systems and Methods" - 胜率对交易心理的影响
"""

from typing import Dict

# ==================== 权重配置 ====================

DEFAULT_WEIGHTS: Dict[str, float] = {
    "ic_stability": 0.40,  # IC 稳定性 (ICIR)
    "ic_absolute": 0.20,  # IC 绝对值
    "ir": 0.20,  # 信息比率
    "long_short_return": 0.10,  # 多空收益
    "win_rate": 0.10,  # 胜率
}

"""
默认权重配置（基于学术研究和实践经验）

理论依据：
- IC 稳定性 (40%): ICIR 是因子稳定性的核心指标，直接影响因子的长期可靠性
  参考文献: Grinold & Kahn, "Active Portfolio Management", Chapter 6

- IC 绝对值 (20%): IC 均值反映因子的预测能力，是因子的基本属性
  参考文献: Grinold & Kahn, "Active Portfolio Management", Chapter 5

- IR (20%): 信息比率综合考虑收益和风险，是实际交易效果的关键指标
  参考文献: Sharpe, "Information Ratio and Performance", Journal of Portfolio Management

- 多空收益 (10%): 实际交易效果，反映因子的经济价值
  参考文献: Jacobs & Levy, "Long-Short Equity Strategies", Journal of Portfolio Management

- 胜率 (10%): 胜率影响策略的心理承受能力和资金管理
  参考文献: Kaufman, "Trading Systems and Methods", Chapter 12
"""


CONSERVATIVE_WEIGHTS: Dict[str, float] = {
    "ic_stability": 0.50,  # IC 稳定性权重更高
    "ic_absolute": 0.15,
    "ir": 0.20,
    "long_short_return": 0.10,
    "win_rate": 0.05,  # 胜率权重降低
}

"""
保守型策略权重（更注重稳定性）

适用场景：
- 养老金、保险资金等风险厌恶型投资者
- 需要长期稳定收益的策略
- 对回撤敏感的资金

设计理念：
- 大幅提高 IC 稳定性权重（50%），确保因子在不同市场环境下都能保持稳定
- 降低胜率权重，因为保守策略更关注长期表现而非短期胜率
- 保持 IR 和多空收益权重，平衡风险和收益
"""


AGGRESSIVE_WEIGHTS: Dict[str, float] = {
    "ic_stability": 0.30,  # IC 稳定性权重降低
    "ic_absolute": 0.20,
    "ir": 0.15,
    "long_short_return": 0.25,  # 多空收益权重更高
    "win_rate": 0.10,
}

"""
激进型策略权重（更注重收益）

适用场景：
- 对冲基金、私募基金等追求高收益的策略
- 能够承受较大波动的资金
- 短期交易策略

设计理念：
- 大幅提高多空收益权重（25%），追求更高的绝对收益
- 降低 IC 稳定性权重，愿意接受一定的不稳定性以换取高收益
- 略微降低 IR 权重，接受更高的风险
"""


HIGH_FREQUENCY_WEIGHTS: Dict[str, float] = {
    "ic_stability": 0.35,
    "ic_absolute": 0.25,  # IC 绝对值权重更高（短期预测更敏感）
    "ir": 0.15,
    "long_short_return": 0.15,
    "win_rate": 0.10,
}

"""
高频交易策略权重

适用场景：
- 日内交易、高频交易
- 短期持仓策略
- 程序化交易

设计理念：
- 提高 IC 绝对值权重（25%），高频策略更关注短期预测能力
- 适度提高 IC 稳定性权重，确保高频交易的稳定性
- 保持多空收益权重，高频交易的收益累积效应
"""


# ==================== 评分阈值 ====================

DEFAULT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ic_mean": {
        "excellent": 0.05,
        "good": 0.03,
        "warning": 0.01,
    },
    "icir": {
        "excellent": 0.70,
        "good": 0.50,
        "warning": 0.30,
    },
    "rank_ic_mean": {
        "excellent": 0.06,
        "good": 0.04,
        "warning": 0.02,
    },
    "rank_icir": {
        "excellent": 0.80,
        "good": 0.60,
        "warning": 0.40,
    },
    "annual_return": {
        "excellent": 0.10,  # 10% 年化收益
        "good": 0.05,  # 5% 年化收益
        "warning": 0.0,  # 0% 年化收益
    },
    "sharpe_ratio": {
        "excellent": 2.0,
        "good": 1.5,
        "warning": 1.0,
    },
    "max_drawdown": {
        "excellent": 0.10,  # 10% 最大回撤
        "good": 0.15,  # 15% 最大回撤
        "warning": 0.25,  # 25% 最大回撤
    },
    "win_rate": {
        "excellent": 0.60,  # 60% 胜率
        "good": 0.55,  # 55% 胜率
        "warning": 0.50,  # 50% 胜率
    },
    "calmar_ratio": {
        "excellent": 1.5,
        "good": 1.0,
        "warning": 0.5,
    },
}

"""
默认评分阈值

阈值设置依据：
1. IC 均值:
   - 优秀 (0.05): 因子具有强预测能力
   - 良好 (0.03): 因子具有中等预测能力
   - 警告 (0.01): 因子预测能力较弱

2. ICIR:
   - 优秀 (0.70): 因子稳定性非常好
   - 良好 (0.50): 因子稳定性良好
   - 警告 (0.30): 因子稳定性不足

3. 年化收益:
   - 优秀 (10%): 显著超越市场
   - 良好 (5%): 超越市场
   - 警告 (0%): 与市场持平

4. 夏普比率:
   - 优秀 (2.0): 风险调整后收益极佳
   - 良好 (1.5): 风险调整后收益良好
   - 警告 (1.0): 风险调整后收益一般

参考文献：
- Qlib 量化框架的因子评估标准
- 学术界普遍接受的因子评估阈值
- 实践中常用的行业标准
"""


STRICT_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ic_mean": {
        "excellent": 0.07,
        "good": 0.05,
        "warning": 0.03,
    },
    "icir": {
        "excellent": 1.0,
        "good": 0.70,
        "warning": 0.50,
    },
    "rank_ic_mean": {
        "excellent": 0.08,
        "good": 0.06,
        "warning": 0.04,
    },
    "rank_icir": {
        "excellent": 1.2,
        "good": 0.80,
        "warning": 0.60,
    },
    "annual_return": {
        "excellent": 0.15,
        "good": 0.10,
        "warning": 0.05,
    },
    "sharpe_ratio": {
        "excellent": 2.5,
        "good": 2.0,
        "warning": 1.5,
    },
    "max_drawdown": {
        "excellent": 0.08,
        "good": 0.10,
        "warning": 0.15,
    },
    "win_rate": {
        "excellent": 0.65,
        "good": 0.60,
        "warning": 0.55,
    },
    "calmar_ratio": {
        "excellent": 2.0,
        "good": 1.5,
        "warning": 1.0,
    },
}

"""
严格评分阈值

适用场景：
- 对因子质量要求极高的机构
- 需要从大量因子中筛选出最优因子
- 追求卓越表现的策略

特点：
- 所有阈值都显著提高
- 只有真正优秀的因子才能达到"优秀"评级
"""


RELAXED_THRESHOLDS: Dict[str, Dict[str, float]] = {
    "ic_mean": {
        "excellent": 0.03,
        "good": 0.02,
        "warning": 0.01,
    },
    "icir": {
        "excellent": 0.50,
        "good": 0.30,
        "warning": 0.20,
    },
    "rank_ic_mean": {
        "excellent": 0.04,
        "good": 0.03,
        "warning": 0.02,
    },
    "rank_icir": {
        "excellent": 0.60,
        "good": 0.40,
        "warning": 0.30,
    },
    "annual_return": {
        "excellent": 0.08,
        "good": 0.03,
        "warning": 0.0,
    },
    "sharpe_ratio": {
        "excellent": 1.5,
        "good": 1.0,
        "warning": 0.5,
    },
    "max_drawdown": {
        "excellent": 0.15,
        "good": 0.20,
        "warning": 0.30,
    },
    "win_rate": {
        "excellent": 0.55,
        "good": 0.52,
        "warning": 0.48,
    },
    "calmar_ratio": {
        "excellent": 1.0,
        "good": 0.7,
        "warning": 0.3,
    },
}

"""
宽松评分阈值

适用场景：
- 新兴市场或波动较大的市场
- 创新性因子的初步评估
- 对因子表现容忍度较高的策略

特点：
- 所有阈值都适当降低
- 更容易获得"良好"或"优秀"评级
"""


# ==================== 可靠性等级定义 ====================

RELIABILITY_GRADES = {
    "A+": {
        "score_range": (0.90, 1.0),
        "description": "优秀",
        "color": "green",
        "recommendation": "该因子表现优秀，建议重点使用。",
    },
    "A": {
        "score_range": (0.80, 0.90),
        "description": "良好",
        "color": "blue",
        "recommendation": "该因子表现良好，建议使用。",
    },
    "B": {
        "score_range": (0.70, 0.80),
        "description": "中等",
        "color": "yellow",
        "recommendation": "该因子表现中等，可以与其他因子组合使用。",
    },
    "C": {
        "score_range": (0.60, 0.70),
        "description": "一般",
        "color": "orange",
        "recommendation": "该因子表现一般，建议谨慎使用或优化后再使用。",
    },
    "D": {
        "score_range": (0.50, 0.60),
        "description": "较差",
        "color": "red",
        "recommendation": "该因子表现较差，不建议单独使用，可考虑作为辅助因子。",
    },
    "F": {
        "score_range": (0.0, 0.50),
        "description": "失败",
        "color": "darkred",
        "recommendation": "该因子不可靠，不建议使用。",
    },
}

"""
可靠性等级定义

等级划分依据：
- A+ (90-100分): 因子在各方面都表现优秀
- A (80-90分): 因子整体表现良好
- B (70-80分): 因子表现中等，有改进空间
- C (60-70分): 因子表现一般，需要优化
- D (50-60分): 因子表现较差，不推荐使用
- F (0-50分): 因子失败，不应使用
"""


# ==================== 相关性分析阈值 ====================

CORRELATION_THRESHOLDS = {
    "high": 0.70,
    "medium": 0.50,
    "low": 0.30,
}

"""
因子相关性阈值

- high (0.70): 高度相关，建议谨慎组合或去重
- medium (0.50): 中度相关，可以组合但需注意
- low (0.30): 低度相关，适合组合使用

参考文献：
- 现代投资组合理论（Markowitz）
- 多因子模型实践
"""


# ==================== 便捷函数 ====================


def get_weights(strategy_type: str = "default") -> Dict[str, float]:
    """
    获取指定策略类型的权重配置

    Args:
        strategy_type: 策略类型 ('default', 'conservative', 'aggressive', 'high_frequency')

    Returns:
        权重配置字典

    Raises:
        ValueError: 如果策略类型不支持

    Examples:
        >>> weights = get_weights('conservative')
        >>> print(weights['ic_stability'])
        0.5
    """
    weights_map = {
        "default": DEFAULT_WEIGHTS,
        "conservative": CONSERVATIVE_WEIGHTS,
        "aggressive": AGGRESSIVE_WEIGHTS,
        "high_frequency": HIGH_FREQUENCY_WEIGHTS,
    }

    if strategy_type not in weights_map:
        raise ValueError(f"不支持的策略类型: {strategy_type}. " f"支持的类型: {list(weights_map.keys())}")

    return weights_map[strategy_type].copy()


def get_thresholds(strictness: str = "default") -> Dict[str, Dict[str, float]]:
    """
    获取指定严格程度的评分阈值

    Args:
        strictness: 严格程度 ('default', 'strict', 'relaxed')

    Returns:
        评分阈值字典

    Raises:
        ValueError: 如果严格程度不支持

    Examples:
        >>> thresholds = get_thresholds('strict')
        >>> print(thresholds['ic_mean']['excellent'])
        0.07
    """
    thresholds_map = {
        "default": DEFAULT_THRESHOLDS,
        "strict": STRICT_THRESHOLDS,
        "relaxed": RELAXED_THRESHOLDS,
    }

    if strictness not in thresholds_map:
        raise ValueError(f"不支持的严格程度: {strictness}. " f"支持的类型: {list(thresholds_map.keys())}")

    return thresholds_map[strictness].copy()


def get_reliability_grade(score: float) -> str:
    """
    根据评分获取可靠性等级

    Args:
        score: 综合评分 (0-1)

    Returns:
        可靠性等级 ('A+', 'A', 'B', 'C', 'D', 'F')

    Examples:
        >>> grade = get_reliability_grade(0.85)
        >>> print(grade)
        'A'
    """
    # 确保 score 在 [0, 1] 范围内
    score = max(0.0, min(1.0, score))

    for grade, info in RELIABILITY_GRADES.items():
        min_score, max_score = info["score_range"]
        # 使用左闭右开区间，但处理 1.0 的特殊情况
        if grade == "A+" and score >= 0.90:
            return grade
        elif min_score <= score < max_score:
            return grade

    return "F"


def validate_weights(weights: Dict[str, float]) -> bool:
    """
    验证权重配置是否有效

    Args:
        weights: 权重配置字典

    Returns:
        是否有效

    Raises:
        ValueError: 如果权重配置无效

    Examples:
        >>> validate_weights(DEFAULT_WEIGHTS)
        True
    """
    # 检查权重和是否为 1
    total_weight = sum(weights.values())
    if not (0.99 <= total_weight <= 1.01):  # 允许浮点误差
        raise ValueError(f"权重总和必须为 1.0，当前为 {total_weight:.4f}")

    # 检查所有权重是否在 [0, 1] 范围内
    for key, value in weights.items():
        if not (0 <= value <= 1):
            raise ValueError(f"权重 '{key}' 的值必须在 [0, 1] 范围内，当前为 {value}")

    return True


if __name__ == "__main__":
    # 示例：使用配置
    print("=" * 60)
    print("可靠性评估配置示例")
    print("=" * 60)

    # 1. 获取默认权重
    print("\n默认权重配置:")
    weights = get_weights("default")
    for key, value in weights.items():
        print(f"  {key}: {value:.2f}")

    # 2. 获取保守型权重
    print("\n保守型权重配置:")
    weights = get_weights("conservative")
    for key, value in weights.items():
        print(f"  {key}: {value:.2f}")

    # 3. 获取评分阈值
    print("\n默认评分阈值:")
    thresholds = get_thresholds("default")
    print(f"  IC 均值 - 优秀: {thresholds['ic_mean']['excellent']:.3f}")
    print(f"  ICIR - 优秀: {thresholds['icir']['excellent']:.2f}")
    print(f"  年化收益 - 优秀: {thresholds['annual_return']['excellent']:.2%}")

    # 4. 获取可靠性等级
    print("\n可靠性等级示例:")
    for score in [0.95, 0.85, 0.75, 0.65, 0.55, 0.45]:
        grade = get_reliability_grade(score)
        info = RELIABILITY_GRADES[grade]
        print(f"  评分 {score:.2f} -> {grade} ({info['description']})")

    # 5. 验证权重
    print("\n权重验证:")
    try:
        validate_weights(DEFAULT_WEIGHTS)
        print("  默认权重配置有效")
    except ValueError as e:
        print(f"  错误: {e}")
