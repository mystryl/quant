"""
统一数据管理系统 - 配置文件

提供路径配置和常量定义，使用相对路径提高跨系统兼容性。
"""

from pathlib import Path
from typing import Dict, List


# =============================================================================
# 路径配置
# =============================================================================

def get_project_root() -> Path:
    """
    获取项目根目录

    自动检测项目根目录，支持从不同位置调用。

    Returns:
        Path: 项目根目录路径
    """
    # 从当前文件向上查找项目根目录
    # 项目根目录特征：包含 .git 目录
    current_dir = Path(__file__).resolve().parent

    # 向上查找，最多 6 层
    for _ in range(6):
        if (current_dir / ".git").exists():
            return current_dir
        current_dir = current_dir.parent

    # 如果找不到 .git，使用默认目录结构
    # 从 scripts/data 向上 4 层到达项目根目录
    # __file__ = .../projects/qlib_backtest/scripts/data/config.py
    # parent.parent.parent.parent = Quant/
    return Path(__file__).resolve().parent.parent.parent.parent


# 项目根目录
PROJECT_ROOT = get_project_root()

# Parquet 数据目录（主存储）
PARQUET_DATA_DIR = PROJECT_ROOT / "K线数据库/期货商品指数_parquet"

# Qlib 缓存目录（转换后的数据）
QLIB_CACHE_DIR = PROJECT_ROOT / "data/qlib_cache"

# 临时目录
TEMP_DIR = PROJECT_ROOT / "data/temp"

# 日志目录
LOG_DIR = PROJECT_ROOT / "data/logs"


# =============================================================================
# 数据字段配置
# =============================================================================

# 标准 OHLCV 字段
STANDARD_FIELDS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "money",
    "avg",
    "open_interest"
]

# Qlib 字段映射（带 $ 前缀）
QLIB_FIELD_PREFIX = "$"

# 字段别名映射（用于不同数据源的字段名转换）
FIELD_ALIASES: Dict[str, List[str]] = {
    "open": ["open", "Open", "OPEN", "开"],
    "high": ["high", "High", "HIGH", "高"],
    "low": ["low", "Low", "LOW", "低"],
    "close": ["close", "Close", "CLOSE", "收"],
    "volume": ["volume", "Volume", "VOLUME", "量", "vol"],
    "money": ["money", "Money", "MONEY", "金额", "amount"],
    "avg": ["avg", "Avg", "AVG", "均价", "vwap"],
    "open_interest": ["open_interest", "OpenInterest", "持仓量", "oi"]
}


# =============================================================================
# Qlib 配置
# =============================================================================

# Qlib 区域配置
QLIB_REGION = "cn"

# Qlib 频率配置
QLIB_FREQ = "1min"

# Qlib 日历配置
CALENDAR_DIR = QLIB_CACHE_DIR / "calendars"


# =============================================================================
# 缓存配置
# =============================================================================

# 是否启用内存缓存
ENABLE_MEMORY_CACHE = True

# 内存缓存大小限制（MB）
MEMORY_CACHE_SIZE_MB = 512

# LRU 缓存最大条目数
LRU_CACHE_MAX_SIZE = 100

# 缓存过期时间（天）
CACHE_EXPIRE_DAYS = 30


# =============================================================================
# 转换配置
# =============================================================================

# 并行转换的工作进程数
CONVERT_PARALLEL_WORKERS = 4

# 批量转换的批次大小
CONVERT_BATCH_SIZE = 10

# 是否启用增量转换（只转换新增或修改的文件）
CONVERT_INCREMENTAL = True


# =============================================================================
# 文件名模式
# =============================================================================

# Parquet 文件扩展名
PARQUET_EXTENSION = ".parquet"

# CSV 文件扩展名
CSV_EXTENSION = ".csv"

# Qlib 目录结构
QLIB_INSTRUMENTS_DIR = QLIB_CACHE_DIR / "instruments" / QLIB_FREQ

# Qlib 字段目录前缀
QLIB_FIELD_DIR_PREFIX = "$"


# =============================================================================
# 日志配置
# =============================================================================

# 日志级别
LOG_LEVEL = "INFO"

# 日志文件前缀
LOG_FILE_PREFIX = "unified_data_provider"

# 日志文件格式
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"


# =============================================================================
# 验证配置
# =============================================================================

def validate_paths() -> None:
    """
    验证路径配置的有效性

    检查关键目录是否存在，如果不存在则创建。

    Raises:
        RuntimeError: 如果项目根目录无效
    """
    if not PROJECT_ROOT.exists():
        raise RuntimeError(f"项目根目录不存在: {PROJECT_ROOT}")

    # 创建必要的目录
    directories = [
        QLIB_CACHE_DIR,
        QLIB_INSTRUMENTS_DIR,
        CALENDAR_DIR,
        TEMP_DIR,
        LOG_DIR
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 辅助函数
# =============================================================================

def get_parquet_file_path(instrument: str) -> Path:
    """
    获取合约的 Parquet 文件路径

    Args:
        instrument: 合约代码，如 "HC8888.XSGE"

    Returns:
        Path: Parquet 文件路径
    """
    return PARQUET_DATA_DIR / f"{instrument}{PARQUET_EXTENSION}"


def get_qlib_field_dir(field: str) -> Path:
    """
    获取 Qlib 字段目录路径

    Args:
        field: 字段名，如 "close"

    Returns:
        Path: Qlib 字段目录路径
    """
    field_dir_name = f"{QLIB_FIELD_DIR_PREFIX}{field}"
    return QLIB_INSTRUMENTS_DIR / field_dir_name


def get_qlib_instrument_file(instrument: str, field: str) -> Path:
    """
    获取 Qlib 合约字段文件路径

    Args:
        instrument: 合约代码
        field: 字段名

    Returns:
        Path: Qlib 文件路径
    """
    field_dir = get_qlib_field_dir(field)
    return field_dir / f"{instrument}{CSV_EXTENSION}"


def normalize_field_name(field: str) -> str:
    """
    标准化字段名

    将不同数据源的字段名映射到标准字段名。

    Args:
        field: 原始字段名

    Returns:
        str: 标准化后的字段名
    """
    field_lower = field.lower()

    for standard_name, aliases in FIELD_ALIASES.items():
        if field_lower in [alias.lower() for alias in aliases]:
            return standard_name

    return field


# =============================================================================
# 初始化
# =============================================================================

# 模块导入时验证路径
validate_paths()


if __name__ == "__main__":
    # 测试配置
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Parquet 数据目录: {PARQUET_DATA_DIR}")
    print(f"Qlib 缓存目录: {QLIB_CACHE_DIR}")
    print(f"Qlib instruments 目录: {QLIB_INSTRUMENTS_DIR}")
    print(f"日历目录: {CALENDAR_DIR}")

    # 测试路径函数
    print("\n测试路径函数:")
    print(f"HC8888.XSGE Parquet: {get_parquet_file_path('HC8888.XSGE')}")
    print(f"HC8888.XSGE Close (Qlib): {get_qlib_instrument_file('HC8888.XSGE', 'close')}")

    # 测试字段标准化
    print("\n测试字段标准化:")
    print(f"'Open' -> {normalize_field_name('Open')}")
    print(f"'持仓量' -> {normalize_field_name('持仓量')}")
    print(f"'VOL' -> {normalize_field_name('VOL')}")
