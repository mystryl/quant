"""
统一数据提供者 - SmartDataProvider

提供统一的数据访问接口，支持多种数据格式和智能路由。
主要特性：
- 透明缓存：自动管理 Parquet → Qlib 转换
- 按需转换：只在需要时转换为 Qlib 格式
- 智能路由：根据使用场景选择最优格式
- 统一接口：一个 API 支持多种使用场景
"""

import logging
import pickle
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import pandas as pd
from tqdm import tqdm

from .config import (
    PARQUET_DATA_DIR,
    QLIB_CACHE_DIR,
    QLIB_INSTRUMENTS_DIR,
    CALENDAR_DIR,
    STANDARD_FIELDS,
    QLIB_FIELD_PREFIX,
    QLIB_FREQ,
    ENABLE_MEMORY_CACHE,
    LRU_CACHE_MAX_SIZE,
    CACHE_EXPIRE_DAYS,
    CONVERT_PARALLEL_WORKERS,
    get_parquet_file_path,
    get_qlib_field_dir,
    get_qlib_instrument_file,
    normalize_field_name,
    validate_paths
)


# =============================================================================
# 日志配置
# =============================================================================

logger = logging.getLogger(__name__)


# =============================================================================
# 异常类
# =============================================================================

class DataProviderError(Exception):
    """数据提供者基础异常"""
    pass


class InstrumentNotFoundError(DataProviderError):
    """合约数据不存在异常"""
    pass


class DataConversionError(DataProviderError):
    """数据转换异常"""
    pass


class InvalidFieldError(DataProviderError):
    """无效字段异常"""
    pass


# =============================================================================
# 缓存装饰器
# =============================================================================

class SimpleLRUCache:
    """简单的 LRU 缓存实现"""

    def __init__(self, max_size: int = 100):
        self.max_size = max_size
        self.cache: Dict[str, Any] = {}
        self.access_order: List[str] = []

    def get(self, key: str) -> Optional[Any]:
        """获取缓存项"""
        if key in self.cache:
            # 更新访问顺序
            self.access_order.remove(key)
            self.access_order.append(key)
            return self.cache[key]
        return None

    def put(self, key: str, value: Any) -> None:
        """存入缓存项"""
        if key in self.cache:
            # 更新现有项
            self.access_order.remove(key)
        elif len(self.cache) >= self.max_size:
            # 淘汰最久未使用的项
            oldest = self.access_order.pop(0)
            del self.cache[oldest]

        self.cache[key] = value
        self.access_order.append(key)

    def clear(self) -> None:
        """清空缓存"""
        self.cache.clear()
        self.access_order.clear()

    def size(self) -> int:
        """获取缓存大小"""
        return len(self.cache)


# =============================================================================
# 核心数据提供者类
# =============================================================================

class ParquetDataProvider:
    """
    统一的 Parquet 数据提供者

    提供统一的数据访问接口，支持 Parquet 和 Qlib 格式。
    自动管理数据转换和缓存。
    """

    def __init__(
        self,
        project_root: Optional[Path] = None,
        parquet_dir: Optional[Path] = None,
        qlib_cache_dir: Optional[Path] = None,
        enable_memory_cache: bool = ENABLE_MEMORY_CACHE,
        memory_cache_size: int = LRU_CACHE_MAX_SIZE
    ):
        """
        初始化数据提供者

        Args:
            project_root: 项目根目录（已弃用，使用自动检测）
            parquet_dir: Parquet 数据目录（已弃用，使用 config.py）
            qlib_cache_dir: Qlib 缓存目录（已弃用，使用 config.py）
            enable_memory_cache: 是否启用内存缓存
            memory_cache_size: 内存缓存大小
        """
        # 使用配置文件中的路径
        self.parquet_dir = PARQUET_DATA_DIR
        self.qlib_cache_dir = QLIB_CACHE_DIR
        self.qlib_instruments_dir = QLIB_INSTRUMENTS_DIR

        # 初始化内存缓存
        self.enable_memory_cache = enable_memory_cache
        self.memory_cache = SimpleLRUCache(max_size=memory_cache_size) if enable_memory_cache else None

        # 验证路径
        validate_paths()

        logger.info(f"数据提供者初始化完成")
        logger.info(f"Parquet 目录: {self.parquet_dir}")
        logger.info(f"Qlib 缓存目录: {self.qlib_cache_dir}")

    # ========================================================================
    # 公共 API
    # ========================================================================

    def get_data(
        self,
        instrument: str,
        start_time: Union[str, datetime, pd.Timestamp],
        end_time: Union[str, datetime, pd.Timestamp],
        fields: Optional[List[str]] = None,
        format: str = 'auto',
        use_cache: bool = True
    ) -> pd.DataFrame:
        """
        获取数据，自动选择最优格式

        Args:
            instrument: 合约代码，如 "HC8888.XSGE"
            start_time: 开始时间
            end_time: 结束时间
            fields: 字段列表，如 ["open", "high", "low", "close"]
            format: 数据格式 ('auto', 'parquet', 'qlib')
                   - auto: 自动选择
                   - parquet: 直接读取 Parquet
                   - qlib: 读取或转换为 Qlib 格式
            use_cache: 是否使用缓存

        Returns:
            pd.DataFrame: 数据框

        Raises:
            InstrumentNotFoundError: 合约不存在
            InvalidFieldError: 字段无效
        """
        # 标准化字段名
        if fields is None:
            fields = STANDARD_FIELDS
        else:
            fields = [normalize_field_name(f) for f in fields]

        # 验证字段
        invalid_fields = [f for f in fields if f not in STANDARD_FIELDS]
        if invalid_fields:
            raise InvalidFieldError(f"无效字段: {invalid_fields}")

        # 转换时间格式
        start_time = pd.to_datetime(start_time)
        end_time = pd.to_datetime(end_time)

        # 选择格式
        if format == 'auto':
            format = self._select_format()

        logger.debug(f"获取数据: {instrument}, {start_time} - {end_time}, 格式: {format}")

        # 根据格式获取数据
        if format == 'parquet':
            return self._load_from_parquet(instrument, start_time, end_time, fields, use_cache)
        elif format == 'qlib':
            return self._load_from_qlib(instrument, start_time, end_time, fields, use_cache)
        else:
            raise ValueError(f"不支持的格式: {format}")

    def list_instruments(self, pattern: str = "*") -> List[str]:
        """
        列出可用的合约

        Args:
            pattern: 文件匹配模式，如 "HC*" 或 "8888*"

        Returns:
            List[str]: 合约代码列表
        """
        parquet_files = list(self.parquet_dir.glob(f"{pattern}.parquet"))
        instruments = [f.stem for f in parquet_files]
        return sorted(instruments)

    def get_calendar(self, instrument: str) -> List[Union[str, pd.Timestamp]]:
        """
        获取合约的交易日历

        Args:
            instrument: 合约代码

        Returns:
            List: 交易日历列表
        """
        parquet_file = get_parquet_file_path(instrument)

        if not parquet_file.exists():
            raise InstrumentNotFoundError(f"合约不存在: {instrument}")

        df = pd.read_parquet(parquet_file, columns=['close'])
        return df.index.tolist()

    def convert_to_qlib(
        self,
        instrument: str,
        force: bool = False,
        progress: bool = True
    ) -> None:
        """
        将 Parquet 数据转换为 Qlib 格式

        Args:
            instrument: 合约代码
            force: 是否强制转换（覆盖已存在的缓存）
            progress: 是否显示进度条
        """
        parquet_file = get_parquet_file_path(instrument)

        if not parquet_file.exists():
            raise InstrumentNotFoundError(f"合约不存在: {instrument}")

        # 检查是否已转换
        if not force:
            close_file = get_qlib_instrument_file(instrument, 'close')
            if close_file.exists():
                logger.debug(f"合约已转换，跳过: {instrument}")
                return

        logger.info(f"转换合约: {instrument}")

        # 读取 Parquet 数据
        df = pd.read_parquet(parquet_file)

        # 转换为 Qlib 格式
        fields_to_convert = [f for f in STANDARD_FIELDS if f in df.columns]

        if progress:
            fields_iter = tqdm(fields_to_convert, desc=f"转换 {instrument}")
        else:
            fields_iter = fields_to_convert

        for field in fields_iter:
            # 创建字段目录
            field_dir = get_qlib_field_dir(field)
            field_dir.mkdir(parents=True, exist_ok=True)

            # 保存字段数据
            series = df[field]
            output_file = get_qlib_instrument_file(instrument, field)
            series.to_csv(
                output_file,
                header=[f"${field}"],
                date_format='%Y-%m-%d %H:%M:%S'
            )

        logger.debug(f"合约转换完成: {instrument}")

    def convert_batch(
        self,
        instruments: Optional[List[str]] = None,
        pattern: str = "*",
        force: bool = False,
        parallel: bool = True,
        max_workers: int = CONVERT_PARALLEL_WORKERS
    ) -> None:
        """
        批量转换为 Qlib 格式

        Args:
            instruments: 合约列表，None 表示使用 pattern 查找所有
            pattern: 文件匹配模式
            force: 是否强制转换
            parallel: 是否并行处理
            max_workers: 最大并行工作数
        """
        # 获取合约列表
        if instruments is None:
            instruments = self.list_instruments(pattern=pattern)

        if not instruments:
            logger.warning("没有找到需要转换的合约")
            return

        logger.info(f"批量转换 {len(instruments)} 个合约")

        # 串行转换
        for instrument in tqdm(instruments, desc="批量转换"):
            try:
                self.convert_to_qlib(instrument, force=force, progress=False)
            except Exception as e:
                logger.error(f"转换失败 {instrument}: {e}")

        logger.info("批量转换完成")

    def clear_cache(self) -> None:
        """清空内存缓存"""
        if self.memory_cache:
            self.memory_cache.clear()
            logger.info("内存缓存已清空")

    def get_cache_stats(self) -> Dict[str, Any]:
        """获取缓存统计信息"""
        if not self.memory_cache:
            return {"enabled": False}

        return {
            "enabled": True,
            "size": self.memory_cache.size(),
            "max_size": self.memory_cache.max_size
        }

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _select_format(self) -> str:
        """
        智能选择数据格式

        根据使用场景选择最优格式：
        - Qlib 回测：使用 qlib 格式
        - 数据分析：使用 parquet 格式

        Returns:
            str: 'parquet' 或 'qlib'
        """
        # 检测是否在 Qlib 上下文中
        import inspect
        for frame in inspect.stack():
            filename = frame.filename.lower()
            if 'qlib' in filename and 'backtest' in filename:
                return 'qlib'

        # 默认使用 Parquet（更快）
        return 'parquet'

    def _load_from_parquet(
        self,
        instrument: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        fields: List[str],
        use_cache: bool
    ) -> pd.DataFrame:
        """从 Parquet 读取数据"""
        # 检查内存缓存
        cache_key = f"parquet_{instrument}_{start_time}_{end_time}_{fields}"
        if use_cache and self.memory_cache:
            cached = self.memory_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"从内存缓存读取: {instrument}")
                return cached

        # 读取 Parquet 文件
        parquet_file = get_parquet_file_path(instrument)

        if not parquet_file.exists():
            raise InstrumentNotFoundError(f"合约不存在: {instrument}")

        df = pd.read_parquet(parquet_file, columns=fields)

        # 时间范围过滤
        mask = (df.index >= start_time) & (df.index <= end_time)
        df = df.loc[mask]

        # 存入缓存
        if use_cache and self.memory_cache:
            self.memory_cache.put(cache_key, df)

        return df

    def _load_from_qlib(
        self,
        instrument: str,
        start_time: pd.Timestamp,
        end_time: pd.Timestamp,
        fields: List[str],
        use_cache: bool
    ) -> pd.DataFrame:
        """从 Qlib 缓存读取或自动转换"""
        # 检查是否已转换
        close_file = get_qlib_instrument_file(instrument, 'close')
        if not close_file.exists():
            # 自动转换
            logger.info(f"自动转换为 Qlib 格式: {instrument}")
            self.convert_to_qlib(instrument, force=False, progress=True)

        # 检查内存缓存
        cache_key = f"qlib_{instrument}_{start_time}_{end_time}_{fields}"
        if use_cache and self.memory_cache:
            cached = self.memory_cache.get(cache_key)
            if cached is not None:
                logger.debug(f"从内存缓存读取: {instrument}")
                return cached

        # 读取 Qlib 格式数据
        data = {}
        for field in fields:
            field_file = get_qlib_instrument_file(instrument, field)
            if not field_file.exists():
                logger.warning(f"字段不存在: {instrument}.{field}")
                continue

            series = pd.read_csv(
                field_file,
                index_col=0,
                parse_dates=True,
                header=0,  # 第一行是 header
                dtype={f"${field}": 'float64'}
            )
            data[field] = series.iloc[:, 0]

        if not data:
            raise DataProviderError(f"无法读取数据: {instrument}")

        # 合并字段
        df = pd.DataFrame(data)

        # 时间范围过滤
        mask = (df.index >= start_time) & (df.index <= end_time)
        df = df.loc[mask]

        # 存入缓存
        if use_cache and self.memory_cache:
            self.memory_cache.put(cache_key, df)

        return df


# =============================================================================
# 便捷函数
# =============================================================================

def get_data(
    instrument: str,
    start_time: Union[str, datetime],
    end_time: Union[str, datetime],
    fields: Optional[List[str]] = None,
    format: str = 'auto'
) -> pd.DataFrame:
    """
    便捷函数：获取数据

    使用默认配置的数据提供者。

    Args:
        instrument: 合约代码
        start_time: 开始时间
        end_time: 结束时间
        fields: 字段列表
        format: 数据格式

    Returns:
        pd.DataFrame: 数据框
    """
    provider = ParquetDataProvider()
    return provider.get_data(instrument, start_time, end_time, fields, format)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 测试数据提供者
    provider = ParquetDataProvider()

    print("\n=== 测试1: 列出合约 ===")
    instruments = provider.list_instruments(pattern="HC*")
    print(f"找到 {len(instruments)} 个 HC 开头的合约")
    if instruments:
        print(f"示例: {instruments[:3]}")

    print("\n=== 测试2: 获取数据（Parquet） ===")
    if instruments:
        test_instrument = instruments[0]
        try:
            df = provider.get_data(
                test_instrument,
                "2024-01-01",
                "2024-01-31",
                fields=["open", "high", "low", "close", "volume"],
                format='parquet'
            )
            print(f"读取 {len(df)} 条数据")
            print(df.head())
            print(f"\n数据统计:")
            print(df.describe())
        except Exception as e:
            print(f"读取失败: {e}")

    print("\n=== 测试3: 转换为 Qlib 格式 ===")
    if instruments:
        test_instrument = instruments[0]
        try:
            provider.convert_to_qlib(test_instrument, force=False, progress=True)
            print(f"转换完成: {test_instrument}")
        except Exception as e:
            print(f"转换失败: {e}")

    print("\n=== 测试4: 获取数据（Qlib） ===")
    if instruments:
        test_instrument = instruments[0]
        try:
            df = provider.get_data(
                test_instrument,
                "2024-01-01",
                "2024-01-31",
                fields=["open", "high", "low", "close"],
                format='qlib'
            )
            print(f"读取 {len(df)} 条数据")
            print(df.head())
        except Exception as e:
            print(f"读取失败: {e}")

    print("\n=== 测试5: 缓存统计 ===")
    stats = provider.get_cache_stats()
    print(f"缓存统计: {stats}")

    print("\n=== 测试完成 ===")
