#!/usr/bin/env python3
"""
V03混合策略优化模块
提供参数优化功能，包括：
- Optuna贝叶斯优化
- 网格搜索
- 参数稳定性测试
- 过拟合检测
"""

from .config import (
    PARAM_SPACE,
    OPTIMIZATION_CONFIG,
    get_param_space_for_optimization
)

__all__ = [
    'PARAM_SPACE',
    'OPTIMIZATION_CONFIG',
    'get_param_space_for_optimization'
]
