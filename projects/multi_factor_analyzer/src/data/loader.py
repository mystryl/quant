"""
数据加载器模块 - DataLoader

该模块实现数据加载和预处理功能，支持批量加载、缺失值处理、
数据清洗等操作，为因子计算提供高质量的数据。

主要特性：
- 批量加载多个合约的数据
- 自动处理缺失值（前向填充、删除等）
- 数据质量检查
- 时间对齐
"""

import logging
from datetime import datetime
from typing import List, Optional, Union, Dict

import numpy as np
import pandas as pd

from .provider import FactorDataProvider

logger = logging.getLogger(__name__)


class DataLoader:
    """
    数据加载器

    负责数据加载、预处理和质量检查，为因子计算提供高质量的数据。

    Examples:
        >>> loader = DataLoader()
        >>> # 加载单个合约数据
        >>> data = loader.load_data(
        ...     instruments="HC8888.XSGE",
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31",
        ...     fields=["close", "volume"]
        ... )
        >>> # 加载并处理缺失值
        >>> data = loader.load_data(
        ...     instruments=["HC8888.XSGE", "RB8888.XSGE"],
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31",
        ...     fill_method='ffill'
        ... )
    """

    # 缺失值处理方法
    FILL_METHODS = ["ffill", "bfill", "interpolate", "drop", "none"]

    # 数据质量检查阈值
    DEFAULT_MISSING_THRESHOLD = 0.1  # 缺失率超过 10% 则警告
    DEFAULT_ZERO_THRESHOLD = 0.5  # 零值超过 50% 则警告

    def __init__(
        self,
        data_provider: Optional[FactorDataProvider] = None,
        fill_method: str = "none",
        missing_threshold: float = DEFAULT_MISSING_THRESHOLD,
        zero_threshold: float = DEFAULT_ZERO_THRESHOLD,
    ):
        """
        初始化数据加载器

        Args:
            data_provider: 数据提供者实例，默认创建新的实例
            fill_method: 缺失值填充方法
                        - 'ffill': 前向填充
                        - 'bfill': 后向填充
                        - 'interpolate': 线性插值
                        - 'drop': 删除包含缺失值的行
                        - 'none': 不处理（默认）
            missing_threshold: 缺失值警告阈值（0-1）
            zero_threshold: 零值警告阈值（0-1）
        """
        if fill_method not in self.FILL_METHODS:
            raise ValueError(f"无效的 fill_method: {fill_method}。" f"支持的方法: {self.FILL_METHODS}")

        self.data_provider = data_provider or FactorDataProvider()
        self.fill_method = fill_method
        self.missing_threshold = missing_threshold
        self.zero_threshold = zero_threshold

        logger.info(
            f"数据加载器初始化完成: fill_method={fill_method}, "
            f"missing_threshold={missing_threshold}, "
            f"zero_threshold={zero_threshold}"
        )

    def load_data(
        self,
        instruments: Union[str, List[str]],
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        fields: Optional[List[str]] = None,
        fill_method: Optional[str] = None,
        check_quality: bool = True,
        drop_zero_volume: bool = False,
    ) -> pd.DataFrame:
        """
        加载数据（支持预处理）

        Args:
            instruments: 合约代码或合约列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表
            fill_method: 缺失值填充方法（覆盖实例默认值）
            check_quality: 是否检查数据质量
            drop_zero_volume: 是否删除成交量为 0 的行

        Returns:
            pd.DataFrame: 加载并处理后的数据

        Raises:
            ValueError: 参数无效或数据质量问题

        Examples:
            >>> loader = DataLoader()
            >>> # 基本用法
            >>> data = loader.load_data(
            ...     instruments="HC8888.XSGE",
            ...     start_date="2024-01-01",
            ...     end_date="2024-01-31"
            ... )
            >>> # 前向填充缺失值
            >>> data = loader.load_data(
            ...     instruments="HC8888.XSGE",
            ...     start_date="2024-01-01",
            ...     end_date="2024-01-31",
            ...     fill_method='ffill'
            ... )
        """
        # 使用指定的 fill_method 或实例默认值
        if fill_method is None:
            fill_method = self.fill_method

        # 加载原始数据
        logger.debug(f"加载数据: instruments={instruments}, " f"{start_date} - {end_date}")

        data = self.data_provider.get_factor_data(
            instruments=instruments, start_date=start_date, end_date=end_date, fields=fields
        )

        if data.empty:
            raise ValueError(
                f"加载的数据为空。请检查合约代码和时间范围: "
                f"instruments={instruments}, "
                f"{start_date} - {end_date}"
            )

        # 处理缺失值
        if fill_method != "none":
            data = self._fill_missing_values(data, fill_method)

        # 删除零成交量的行
        if drop_zero_volume and "volume" in data.columns:
            data = data[data["volume"] > 0]

        # 数据质量检查
        if check_quality:
            self._check_data_quality(data)

        logger.debug(f"数据加载完成: {len(data)} 条记录, " f"{len(data.columns)} 个字段")

        return data

    def load_batch(
        self,
        instruments: List[str],
        start_date: Union[str, datetime, pd.Timestamp],
        end_date: Union[str, datetime, pd.Timestamp],
        fields: Optional[List[str]] = None,
        fill_method: Optional[str] = None,
        parallel: bool = False,
        show_progress: bool = True,
    ) -> Dict[str, pd.DataFrame]:
        """
        批量加载多个合约的数据

        Args:
            instruments: 合约代码列表
            start_date: 开始日期
            end_date: 结束日期
            fields: 字段列表
            fill_method: 缺失值填充方法
            parallel: 是否并行加载（暂未实现，保留接口）
            show_progress: 是否显示进度

        Returns:
            Dict[str, pd.DataFrame]: 合约代码到数据的映射

        Examples:
            >>> loader = DataLoader()
            >>> instruments = ["HC8888.XSGE", "RB8888.XSGE", "CU8888.XSGE"]
            >>> data_dict = loader.load_batch(
            ...     instruments=instruments,
            ...     start_date="2024-01-01",
            ...     end_date="2024-01-31",
            ...     fields=["close", "volume"]
            ... )
            >>> for instrument, data in data_dict.items():
            ...     print(f"{instrument}: {len(data)} 条记录")
        """
        data_dict = {}

        if show_progress:
            from tqdm import tqdm

            instruments_iter = tqdm(instruments, desc="批量加载数据")
        else:
            instruments_iter = instruments

        for instrument in instruments_iter:
            try:
                data = self.load_data(
                    instruments=instrument,
                    start_date=start_date,
                    end_date=end_date,
                    fields=fields,
                    fill_method=fill_method,
                    check_quality=False,  # 批量加载时跳过质量检查以提高速度
                )
                data_dict[instrument] = data

            except Exception as e:
                logger.warning(f"加载失败: {instrument}, 错误: {e}")
                continue

        logger.info(f"批量加载完成: {len(data_dict)}/{len(instruments)} 个合约成功")

        return data_dict

    def resample_data(self, data: pd.DataFrame, freq: str = "1D", agg_method: Dict[str, str] = None) -> pd.DataFrame:
        """
        重采样数据到指定频率

        Args:
            data: 原始数据
            freq: 目标频率（如 '1D', '1W', '1M'）
            agg_method: 聚合方法字典
                       默认: {
                           'open': 'first',
                           'high': 'max',
                           'low': 'min',
                           'close': 'last',
                           'volume': 'sum',
                           'money': 'sum',
                           'open_interest': 'last'
                       }

        Returns:
            pd.DataFrame: 重采样后的数据

        Examples:
            >>> loader = DataLoader()
            >>> data = loader.load_data(
            ...     instruments="HC8888.XSGE",
            ...     start_date="2024-01-01",
            ...     end_date="2024-01-31"
            ... )
            >>> # 转换为日线数据
            >>> daily_data = loader.resample_data(data, freq='1D')
        """
        # 默认聚合方法
        if agg_method is None:
            agg_method = {
                "open": "first",
                "high": "max",
                "low": "min",
                "close": "last",
                "volume": "sum",
                "money": "sum",
                "open_interest": "last",
                "avg": "mean",
            }

        # 根据索引类型选择重采样方法
        if isinstance(data.index, pd.MultiIndex):
            # MultiIndex: 按 instrument 分组重采样
            resampled = data.groupby(level="instrument").apply(lambda x: self._resample_single(x, freq, agg_method))
            resampled.index.names = ["instrument", "datetime"]
        else:
            # 单一索引
            resampled = self._resample_single(data, freq, agg_method)

        logger.info(f"数据重采样: {len(data)} -> {len(resampled)} 条记录")

        return resampled

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _fill_missing_values(self, data: pd.DataFrame, method: str) -> pd.DataFrame:
        """
        处理缺失值

        Args:
            data: 数据框
            method: 填充方法

        Returns:
            pd.DataFrame: 处理后的数据框
        """
        original_len = len(data)
        missing_count = data.isnull().sum().sum()

        if missing_count == 0:
            return data

        logger.debug(f"处理缺失值: {missing_count} 个缺失值, 方法={method}")

        if method == "ffill":
            # 前向填充
            data = data.ffill().bfill()

        elif method == "bfill":
            # 后向填充
            data = data.bfill().ffill()

        elif method == "interpolate":
            # 线性插值
            data = data.interpolate(method="linear", limit_direction="both")

        elif method == "drop":
            # 删除包含缺失值的行
            data = data.dropna()

        # 检查是否还有缺失值
        remaining_missing = data.isnull().sum().sum()
        if remaining_missing > 0:
            logger.warning(f"处理后仍有 {remaining_missing} 个缺失值, " "将被填充为 0")
            data = data.fillna(0)

        logger.debug(f"缺失值处理完成: {original_len} -> {len(data)} 条记录")

        return data

    def _check_data_quality(self, data: pd.DataFrame) -> None:
        """
        检查数据质量

        Args:
            data: 数据框

        Raises:
            ValueError: 数据质量问题
        """
        # 检查缺失值比例
        missing_ratio = data.isnull().sum() / len(data)
        high_missing_fields = missing_ratio[missing_ratio > self.missing_threshold].index.tolist()

        if high_missing_fields:
            logger.warning(f"以下字段缺失值比例超过 {self.missing_threshold:.1%}: " f"{high_missing_fields}")

        # 检查零值比例
        if "volume" in data.columns:
            zero_volume_ratio = (data["volume"] == 0).sum() / len(data)
            if zero_volume_ratio > self.zero_threshold:
                logger.warning(f"零成交量比例过高: {zero_volume_ratio:.1%} " f"(阈值: {self.zero_threshold:.1%})")

        # 检查数据范围
        for col in ["open", "high", "low", "close"]:
            if col in data.columns:
                if (data[col] <= 0).any():
                    logger.warning(f"{col} 字段存在非正值")

        # 检查 high/low 逻辑
        if all(col in data.columns for col in ["high", "low"]):
            invalid_hl = (data["high"] < data["low"]).sum()
            if invalid_hl > 0:
                raise ValueError(f"数据逻辑错误: high < low 出现 {invalid_hl} 次")

        logger.debug("数据质量检查完成")

    def _resample_single(self, data: pd.DataFrame, freq: str, agg_method: Dict[str, str]) -> pd.DataFrame:
        """
        重采样单个数据框

        Args:
            data: 数据框
            freq: 目标频率
            agg_method: 聚合方法

        Returns:
            pd.DataFrame: 重采样后的数据框
        """
        # 过滤聚合方法中存在的字段
        valid_agg = {col: method for col, method in agg_method.items() if col in data.columns}

        if not valid_agg:
            logger.warning("没有有效的字段进行聚合")
            return data

        # 重采样
        resampled = data.resample(freq).agg(valid_agg)

        # 删除全为 NaN 的行
        resampled = resampled.dropna(how="all")

        return resampled


# =============================================================================
# 便捷函数
# =============================================================================


def load_data(
    instruments: Union[str, List[str]],
    start_date: Union[str, datetime, pd.Timestamp],
    end_date: Union[str, datetime, pd.Timestamp],
    fields: Optional[List[str]] = None,
    fill_method: str = "none",
) -> pd.DataFrame:
    """
    便捷函数：加载数据

    使用默认配置的数据加载器。

    Args:
        instruments: 合约代码或合约列表
        start_date: 开始日期
        end_date: 结束日期
        fields: 字段列表
        fill_method: 缺失值填充方法

    Returns:
        pd.DataFrame: 加载的数据

    Examples:
        >>> # 加载单个合约数据
        >>> data = load_data(
        ...     instruments="HC8888.XSGE",
        ...     start_date="2024-01-01",
        ...     end_date="2024-01-31",
        ...     fields=["close", "volume"],
        ...     fill_method='ffill'
        ... )
    """
    loader = DataLoader(fill_method=fill_method)
    return loader.load_data(
        instruments=instruments, start_date=start_date, end_date=end_date, fields=fields, fill_method=fill_method
    )


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 测试数据加载器
    loader = DataLoader(fill_method="ffill")

    print("\n=== 测试1: 加载单个合约数据 ===")
    provider = FactorDataProvider()
    instruments = provider.list_instruments(pattern="HC*")

    if instruments:
        test_instrument = instruments[0]
        try:
            data = loader.load_data(
                instruments=test_instrument,
                start_date="2024-01-01",
                end_date="2024-01-31",
                fields=["open", "high", "low", "close", "volume"],
                check_quality=True,
            )
            print(f"读取 {len(data)} 条数据")
            print(data.head())
            print("\n数据统计:")
            print(data.describe())
        except Exception as e:
            print(f"加载失败: {e}")

    print("\n=== 测试2: 批量加载 ===")
    if len(instruments) >= 2:
        try:
            data_dict = loader.load_batch(
                instruments=instruments[:3],
                start_date="2024-01-01",
                end_date="2024-01-31",
                fields=["close", "volume"],
                show_progress=True,
            )
            print(f"成功加载 {len(data_dict)} 个合约")
            for instrument, data in data_dict.items():
                print(f"  {instrument}: {len(data)} 条记录")
        except Exception as e:
            print(f"批量加载失败: {e}")

    print("\n=== 测试3: 数据重采样 ===")
    if instruments:
        try:
            # 加载分钟级数据
            data = loader.load_data(
                instruments=instruments[0],
                start_date="2024-01-01",
                end_date="2024-01-07",
                fields=["open", "high", "low", "close", "volume"],
            )

            # 重采样为日线
            daily_data = loader.resample_data(data, freq="1D")

            print(f"原始数据: {len(data)} 条")
            print(f"日线数据: {len(daily_data)} 条")
            print(daily_data.head())
        except Exception as e:
            print(f"重采样失败: {e}")

    print("\n=== 测试4: 缺失值处理 ===")
    # 创建包含缺失值的测试数据
    test_data = pd.DataFrame(
        {"close": [100, 101, np.nan, 103, np.nan, 105], "volume": [1000, np.nan, 1200, 1300, 1400, np.nan]}
    )

    print("原始数据:")
    print(test_data)

    print("\n前向填充:")
    print(loader._fill_missing_values(test_data.copy(), "ffill"))

    print("\n线性插值:")
    print(loader._fill_missing_values(test_data.copy(), "interpolate"))

    print("\n=== 测试完成 ===")
