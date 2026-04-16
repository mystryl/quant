"""
数据提供者模块 - FactorDataProvider

该模块实现因子数据提供者，复用 qlib_backtest 的 SmartDataProvider，
为多因子分析系统提供统一的数据访问接口。

主要特性：
- 复用 ParquetDataProvider 的数据访问能力
- 提供因子计算所需的基础数据
- 支持多合约、多字段批量获取
- 自动缓存管理
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import List, Optional, Union, Dict, Any

import pandas as pd

logger = logging.getLogger(__name__)


# =============================================================================
# 导入 qlib_backtest 的 ParquetDataProvider
# =============================================================================


def _import_parquet_provider():
    """
    动态导入 ParquetDataProvider

    Returns:
        ParquetDataProvider 类或 None
    """
    import sys
    import types

    # 找到 qlib_backtest 项目路径
    current_path = Path(__file__).resolve().parent
    project_root = current_path.parent.parent.parent
    qlib_backtest_path = project_root / "qlib_backtest"

    if not qlib_backtest_path.exists():
        logger.warning(f"未找到 qlib_backtest 目录: {qlib_backtest_path}")
        return None

    scripts_path = qlib_backtest_path / "scripts"

    # 添加到 sys.path
    if str(scripts_path) not in sys.path:
        sys.path.insert(0, str(scripts_path))

    try:
        # 读取 unified_data_provider.py 文件
        provider_file = scripts_path / "data" / "unified_data_provider.py"

        if not provider_file.exists():
            logger.warning(f"文件不存在: {provider_file}")
            return None

        # 读取文件内容并执行
        with open(provider_file, "r", encoding="utf-8") as f:
            code = f.read()

        # 创建临时模块
        temp_module = types.ModuleType("_qlib_backtest_temp")

        # 修改代码中的相对导入为绝对导入
        code = code.replace("from .config import", "from qlib_backtest_config import")

        # 导入 config 模块
        config_file = scripts_path / "data" / "config.py"
        if config_file.exists():
            with open(config_file, "r", encoding="utf-8") as f:
                config_code = f.read()

            config_module = types.ModuleType("qlib_backtest_config")
            exec(config_code, config_module.__dict__)
            sys.modules["qlib_backtest_config"] = config_module

        # 执行 provider 代码
        exec(code, temp_module.__dict__)

        logger.info("成功导入 ParquetDataProvider")
        return temp_module.ParquetDataProvider

    except Exception as e:
        logger.warning(f"导入 ParquetDataProvider 失败: {e}")
        return None


# 尝试导入 ParquetDataProvider
ParquetDataProvider = _import_parquet_provider()


# 如果导入失败，使用存根类
if ParquetDataProvider is None:
    logger.warning("使用存根类替代 ParquetDataProvider")

    class ParquetDataProvider:  # type: ignore
        """存根类，用于开发测试"""

        def __init__(self, **kwargs):
            self.parquet_dir = None
            self.qlib_cache_dir = None
            logger.warning("使用存根类，数据提供功能不可用")

        def get_data(self, **kwargs):
            raise NotImplementedError("数据提供功能不可用。请确保 qlib_backtest 项目正确配置。")

        def list_instruments(self, pattern: str = "*") -> List[str]:
            return []

        def get_calendar(self, instrument: str) -> List[Union[str, pd.Timestamp]]:
            return []

        def clear_cache(self):
            pass

        def get_cache_stats(self) -> Dict[str, Any]:
            return {"enabled": False}


# =============================================================================
# 核心数据提供者类
# =============================================================================


class FactorDataProvider:
    """
    因子数据提供者

    包装 ParquetDataProvider，为多因子分析系统提供统一的数据访问接口。

    Examples:
        >>> provider = FactorDataProvider()
        >>> # 获取单个合约数据
        >>> data = provider.get_factor_data(
        ...     instruments="HC8888.XSGE",
        ...     fields=["open", "high", "low", "close", "volume"],
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31"
        ... )
        >>> # 获取多个合约数据
        >>> data = provider.get_factor_data(
        ...     instruments=["HC8888.XSGE", "RB8888.XSGE"],
        ...     fields=["close", "volume"],
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31"
        ... )
    """

    # 默认支持的字段
    DEFAULT_FIELDS = ["open", "high", "low", "close", "volume", "money", "avg", "open_interest"]

    def __init__(
        self, parquet_dir: Optional[Path] = None, qlib_cache_dir: Optional[Path] = None, enable_cache: bool = True
    ):
        """
        初始化因子数据提供者

        Args:
            parquet_dir: Parquet 数据目录（可选，默认使用配置文件中的路径）
            qlib_cache_dir: Qlib 缓存目录（可选，默认使用配置文件中的路径）
            enable_cache: 是否启用缓存
        """
        # 初始化底层 ParquetDataProvider
        self.provider = ParquetDataProvider(
            parquet_dir=parquet_dir, qlib_cache_dir=qlib_cache_dir, enable_memory_cache=enable_cache
        )

        logger.info("因子数据提供者初始化完成")

    def get_factor_data(
        self,
        instruments: Union[str, List[str]],
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        fields: Optional[List[str]] = None,
        format: str = "parquet",
    ) -> pd.DataFrame:
        """
        获取因子计算所需的基础数据

        Args:
            instruments: 合约代码或合约列表
                       - 单个合约: "HC8888.XSGE"
                       - 多个合约: ["HC8888.XSGE", "RB8888.XSGE"]
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表，如 ["open", "high", "low", "close", "volume"]
                   默认为 None，使用 DEFAULT_FIELDS
            format: 数据格式 ('parquet' 或 'qlib')

        Returns:
            pd.DataFrame: 因子数据
                         - 单个合约: 索引为日期，列为字段
                         - 多个合约: MultiIndex (datetime, instrument)，列为字段

        Raises:
            ValueError: 参数无效
            Exception: 数据加载失败

        Examples:
            >>> provider = FactorDataProvider()
            >>> # 获取单个合约的收盘价和成交量
            >>> data = provider.get_factor_data(
            ...     instruments="HC8888.XSGE",
            ...     start_date="2024-01-01",
            ...     end_date="2024-01-31",
            ...     fields=["close", "volume"]
            ... )
            >>> print(data.head())
        """
        # 标准化参数
        if fields is None:
            fields = self.DEFAULT_FIELDS

        # 转换时间格式
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)

        # 处理单个合约的情况
        if isinstance(instruments, str):
            return self._get_single_instrument_data(instruments, start_date, end_date, fields, format)

        # 处理多个合约的情况
        elif isinstance(instruments, list):
            return self._get_multiple_instruments_data(instruments, start_date, end_date, fields, format)

        else:
            raise ValueError(f"instruments 参数类型错误: {type(instruments)}。" "应为 str 或 List[str]。")

    def get_price_data(
        self,
        instruments: Union[str, List[str]],
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        price_field: str = "close",
        format: str = "parquet",
    ) -> pd.DataFrame:
        """
        获取价格数据（便捷方法）

        Args:
            instruments: 合约代码或合约列表
            start_date: 开始日期
            end_date: 结束日期
            price_field: 价格字段，默认为 'close'。
                        可选: 'open', 'high', 'low', 'close', 'avg'
            format: 数据格式 ('parquet' 或 'qlib')

        Returns:
            pd.DataFrame: 价格数据

        Examples:
            >>> provider = FactorDataProvider()
            >>> # 获取收盘价
            >>> close_prices = provider.get_price_data(
            ...     instruments="HC8888.XSGE",
            ...     start_date="2024-01-01",
            ...     end_date="2024-01-31",
            ...     price_field='close'
            ... )
        """
        return self.get_factor_data(
            instruments=instruments, start_date=start_date, end_date=end_date, fields=[price_field], format=format
        )

    def list_instruments(self, pattern: str = "*") -> List[str]:
        """
        列出可用的合约

        Args:
            pattern: 文件匹配模式，如 "HC*" 或 "8888*"

        Returns:
            List[str]: 合约代码列表

        Examples:
            >>> provider = FactorDataProvider()
            >>> # 列出所有 HC 开头的合约
            >>> hc_instruments = provider.list_instruments(pattern="HC*")
            >>> print(f"找到 {len(hc_instruments)} 个 HC 合约")
        """
        return self.provider.list_instruments(pattern=pattern)

    def get_calendar(self, instrument: str) -> List[Union[str, pd.Timestamp]]:
        """
        获取合约的交易日历

        Args:
            instrument: 合约代码

        Returns:
            List: 交易日历列表

        Examples:
            >>> provider = FactorDataProvider()
            >>> calendar = provider.get_calendar("HC8888.XSGE")
            >>> print(f"该合约有 {len(calendar)} 个交易日")
        """
        return self.provider.get_calendar(instrument)

    def clear_cache(self) -> None:
        """
        清空缓存

        Examples:
            >>> provider = FactorDataProvider()
            >>> provider.clear_cache()
        """
        self.provider.clear_cache()
        logger.info("因子数据提供者缓存已清空")

    def get_cache_stats(self) -> Dict[str, Any]:
        """
        获取缓存统计信息

        Returns:
            Dict: 缓存统计信息

        Examples:
            >>> provider = FactorDataProvider()
            >>> stats = provider.get_cache_stats()
            >>> print(f"缓存大小: {stats.get('size', 0)}")
        """
        return self.provider.get_cache_stats()

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _get_single_instrument_data(
        self, instrument: str, start_date: pd.Timestamp, end_date: pd.Timestamp, fields: List[str], format: str
    ) -> pd.DataFrame:
        """
        获取单个合约的数据

        Args:
            instrument: 合约代码
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表
            format: 数据格式

        Returns:
            pd.DataFrame: 数据框
        """
        try:
            df = self.provider.get_data(
                instrument=instrument, start_time=start_date, end_time=end_date, fields=fields, format=format
            )

            logger.debug(f"获取数据成功: {instrument}, " f"{len(df)} 条记录, {len(fields)} 个字段")

            return df

        except Exception as e:
            logger.error(f"获取数据失败: {instrument}, 错误: {e}")
            raise

    def _get_multiple_instruments_data(
        self, instruments: List[str], start_date: pd.Timestamp, end_date: pd.Timestamp, fields: List[str], format: str
    ) -> pd.DataFrame:
        """
        获取多个合约的数据

        Args:
            instruments: 合约列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表
            format: 数据格式

        Returns:
            pd.DataFrame: 多索引数据框 (datetime, instrument)
        """
        all_data = []

        for instrument in instruments:
            try:
                df = self._get_single_instrument_data(instrument, start_date, end_date, fields, format)

                # 添加 instrument 列
                df = df.copy()
                df["instrument"] = instrument

                all_data.append(df)

            except Exception as e:
                logger.warning(f"跳过合约 {instrument}: {e}")
                continue

        if not all_data:
            raise ValueError("未能获取任何数据，请检查合约代码和时间范围")

        # 合并所有合约数据
        combined = pd.concat(all_data, ignore_index=False)

        # 设置 MultiIndex
        combined.set_index("instrument", append=True, inplace=True)
        combined.index.names = ["datetime", "instrument"]

        # 重新排序索引：datetime 在内层，instrument 在外层
        combined = combined.reorder_levels(["instrument", "datetime"])

        logger.info(f"获取 {len(all_data)}/{len(instruments)} 个合约的数据, " f"共 {len(combined)} 条记录")

        return combined

    def validate_fields(self, fields: List[str]) -> bool:
        """
        验证字段是否有效

        Args:
            fields: 字段列表

        Returns:
            bool: 是否全部有效

        Raises:
            ValueError: 包含无效字段
        """
        invalid_fields = [f for f in fields if f not in self.DEFAULT_FIELDS]

        if invalid_fields:
            raise ValueError(f"无效字段: {invalid_fields}。" f"支持的字段: {self.DEFAULT_FIELDS}")

        return True


# =============================================================================
# 便捷函数
# =============================================================================


def get_factor_data(
    instruments: Union[str, List[str]],
    start_date: Union[str, datetime, pd.Timestamp],
    end_date: Union[str, datetime, pd.Timestamp],
    fields: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    便捷函数：获取因子数据

    使用默认配置的数据提供者。

    Args:
        instruments: 合约代码或合约列表
        start_date: 开始日期
        end_date: 结束日期
        fields: 字段列表

    Returns:
        pd.DataFrame: 因子数据

    Examples:
        >>> # 获取多个合约的收盘价
        >>> data = get_factor_data(
        ...     instruments=["HC8888.XSGE", "RB8888.XSGE"],
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31",
        ...     fields=["close", "volume"]
        ... )
    """
    provider = FactorDataProvider()
    return provider.get_factor_data(instruments, start_date, end_date, fields)
