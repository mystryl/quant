"""
统一数据管理系统

提供统一的数据访问接口，支持 Parquet 和 Qlib 格式。
"""

from .unified_data_provider import (
    ParquetDataProvider,
    get_data,
    DataProviderError,
    InstrumentNotFoundError,
    DataConversionError,
    InvalidFieldError
)

from .config import (
    PROJECT_ROOT,
    PARQUET_DATA_DIR,
    QLIB_CACHE_DIR,
    STANDARD_FIELDS,
    get_parquet_file_path,
    get_qlib_instrument_file,
    normalize_field_name
)

__all__ = [
    # 主要类
    "ParquetDataProvider",

    # 便捷函数
    "get_data",

    # 异常
    "DataProviderError",
    "InstrumentNotFoundError",
    "DataConversionError",
    "InvalidFieldError",

    # 配置
    "PROJECT_ROOT",
    "PARQUET_DATA_DIR",
    "QLIB_CACHE_DIR",
    "STANDARD_FIELDS",
    "get_parquet_file_path",
    "get_qlib_instrument_file",
    "normalize_field_name"
]

__version__ = "1.0.0"
__author__ = "Quant Team"
