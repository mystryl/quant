"""
核心模块

包含多因子分析系统的核心功能：
- 因子表达式解析器 (FactorExpressionParser)
- 因子计算引擎 (FactorManager)
- 周期对齐模块 (CycleAligner)
- 性能评估引擎 (PerformanceEvaluator)
- 策略场景分析器 (StrategyAnalyzer)
- 可靠性评估器 (ReliabilityEvaluator)
"""

from .factor_expression_parser import (
    FactorExpressionParser,
    validate_factor_expression,
)
from .factor_engine import FactorManager, create_factor_manager
from .performance_eval import (
    PerformanceEvaluator,
    evaluate_factor_performance,
)

# 已有的模块（如果存在）
try:
    from .cycle_aligner import CycleAligner, align_factor_returns  # noqa: F401

    _has_cycle_aligner = True
except ImportError:
    _has_cycle_aligner = False

try:
    from .strategy_analyzer import StrategyAnalyzer, analyze_factor_strategies  # noqa: F401

    _has_strategy_analyzer = True
except ImportError:
    _has_strategy_analyzer = False

try:
    from .reliability import (  # noqa: F401
        ReliabilityEvaluator,
        evaluate_factor_reliability,
    )

    _has_reliability = True
except ImportError:
    _has_reliability = False

try:
    from .correlation_analyzer import (  # noqa: F401
        FactorCorrelationAnalyzer,
        analyze_factor_correlation,
    )

    _has_correlation_analyzer = True
except ImportError:
    _has_correlation_analyzer = False

try:
    from .config import (  # noqa: F401
        get_weights,
        get_thresholds,
        get_reliability_grade,
        validate_weights,
        DEFAULT_WEIGHTS,
        CONSERVATIVE_WEIGHTS,
        AGGRESSIVE_WEIGHTS,
        DEFAULT_THRESHOLDS,
        RELIABILITY_GRADES,
    )

    _has_config = True
except ImportError:
    _has_config = False

__all__ = [
    # 因子表达式解析
    "FactorExpressionParser",
    "validate_factor_expression",
    # 因子管理
    "FactorManager",
    "create_factor_manager",
    # 性能评估
    "PerformanceEvaluator",
    "evaluate_factor_performance",
]

# 动态添加已实现的模块
if _has_cycle_aligner:
    __all__.extend(["CycleAligner", "align_factor_returns"])

if _has_strategy_analyzer:
    __all__.extend(["StrategyAnalyzer", "analyze_factor_strategies"])

if _has_reliability:
    __all__.extend(
        [
            "ReliabilityEvaluator",
            "evaluate_factor_reliability",
        ]
    )

if _has_correlation_analyzer:
    __all__.extend(
        [
            "FactorCorrelationAnalyzer",
            "analyze_factor_correlation",
        ]
    )

if _has_config:
    __all__.extend(
        [
            "get_weights",
            "get_thresholds",
            "get_reliability_grade",
            "validate_weights",
            "DEFAULT_WEIGHTS",
            "CONSERVATIVE_WEIGHTS",
            "AGGRESSIVE_WEIGHTS",
            "DEFAULT_THRESHOLDS",
            "RELIABILITY_GRADES",
        ]
    )
