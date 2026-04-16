"""
数据访问层模块

提供数据加载、验证和处理功能，复用 qlib_backtest 的 SmartDataProvider。

主要组件:
- FactorDataProvider: 统一数据访问接口
- DataLoader: 数据加载和预处理
- DataValidator: 数据验证和因子处理
"""

from .provider import FactorDataProvider, get_factor_data
from .loader import DataLoader, load_data
from .validator import DataValidator, check_data_quality, standardize_factor, neutralize_factor

__all__ = [
    # Provider
    "FactorDataProvider",
    "get_factor_data",
    # Loader
    "DataLoader",
    "load_data",
    # Validator
    "DataValidator",
    "check_data_quality",
    "standardize_factor",
    "neutralize_factor",
]
