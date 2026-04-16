"""
工具模块

提供因子表达式解析、未来函数检测和各种辅助函数。
"""

from .guard import (
    FactorExpressionParser,
    FutureFunctionError,
    FutureFunctionMatch,
    validate_expression,
    analyze_expression,
)

from .helpers import (
    # 数据处理工具
    remove_outliers,
    normalize,
    neutralize,
    # 时间序列工具
    calculate_returns,
    calculate_forward_returns,
    resample_data,
    # 验证工具
    validate_data_format,
    check_missing_values,
    check_infinite_values,
    # 性能计算工具
    calculate_ic,
    calculate_long_short_return,
    align_data,
    split_data,
    # 其他工具
    save_results,
)

__all__ = [
    # guard.py
    "FactorExpressionParser",
    "FutureFunctionError",
    "FutureFunctionMatch",
    "validate_expression",
    "analyze_expression",
    # helpers.py - 数据处理
    "remove_outliers",
    "normalize",
    "neutralize",
    # helpers.py - 时间序列
    "calculate_returns",
    "calculate_forward_returns",
    "resample_data",
    # helpers.py - 验证
    "validate_data_format",
    "check_missing_values",
    "check_infinite_values",
    # helpers.py - 性能计算
    "calculate_ic",
    "calculate_long_short_return",
    "align_data",
    "split_data",
    # helpers.py - 其他
    "save_results",
]
