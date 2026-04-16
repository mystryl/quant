"""
周期对齐模块 (Cycle Aligner)

该模块实现了因子数据和未来收益率的对齐功能，支持多种对齐策略：
1. 默认对齐（Qlib T+1 到 T+2）
2. 灵活对齐（自定义偏移量）
3. 自动检测最优对齐方式
4. 周期验证机制

设计理念：
- 符合中国 T+1 交易规则
- 避免未来数据泄露
- 支持不同因子的周期特性
"""

import numpy as np
import pandas as pd
from typing import Tuple, Dict, Union


class CycleAligner:
    """
    周期对齐器

    用于对齐因子数据和未来收益率，支持多种对齐策略：

    1. **默认对齐 (Qlib 方式)**: T+1 到 T+2
       - T 日因子 -> 预测 T+2 的收益率
       - 符合中国 T+1 交易规则
       - T 日收盘时无法买入，只能 T+1 买入，T+2 卖出

    2. **灵活对齐**: 支持自定义偏移量
       - shift=1: jqfactor_analyzer 方式 (T to T+1)
       - shift=N: 自定义持有期

    3. **自动检测**: 自动选择最优对齐方式
       - 基于 IC 最优化
       - 适合探索因子周期特性

    Examples:
        >>> from core.cycle_aligner import CycleAligner
        >>>
        >>> # 创建对齐器
        >>> aligner = CycleAligner()
        >>>
        >>> # 默认对齐 (T+1 to T+2)
        >>> factor_aligned, returns_aligned = aligner.align(
        ...     factor_df,
        ...     price_df,
        ...     method='default'
        ... )
        >>>
        >>> # 自定义对齐 (T to T+1)
        >>> factor_aligned, returns_aligned = aligner.align(
        ...     factor_df,
        ...     price_df,
        ...     method='flexible',
        ...     shift=1
        ... )
        >>>
        >>> # 自动检测最优对齐
        >>> factor_aligned, returns_aligned, best_shift = aligner.align(
        ...     factor_df,
        ...     price_df,
        ...     method='auto'
        ... )
    """

    def __init__(self, use_log_return: bool = False):
        """
        初始化周期对齐器

        Args:
            use_log_return: 是否使用对数收益率
                - False: 使用简单收益率 (默认)
                - True: 使用对数收益率 (适合长期持有策略)
        """
        self.use_log_return = use_log_return

    def align(
        self,
        factor_df: pd.DataFrame,
        price_df: pd.DataFrame,
        method: str = "default",
        shift: int = 2,
        auto_search_range: Tuple[int, int] = (1, 10),
        ic_calc_method: str = "pearson",
    ) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, int]]:
        """
        对齐因子数据和未来收益率

        Args:
            factor_df: 因子数据 (MultiIndex: datetime, instrument)
            price_df: 价格数据 (MultiIndex: datetime, instrument)
            method: 对齐方法
                - 'default': 默认对齐 (T+1 to T+2)
                - 'flexible': 灵活对齐 (自定义 shift)
                - 'auto': 自动检测最优对齐
            shift: 偏移量（仅用于 method='flexible'）
                - 1: T to T+1 (jqfactor_analyzer 方式)
                - 2: T+1 to T+2 (Qlib 默认方式)
                - N: T+(N-1) to T+N
            auto_search_range: 自动检测的搜索范围 (min_shift, max_shift)
            ic_calc_method: IC 计算方法
                - 'pearson': Pearson IC
                - 'spearman': Spearman Rank IC

        Returns:
            method='default' or 'flexible':
                (factor_aligned, returns_aligned)
            method='auto':
                (factor_aligned, returns_aligned, best_shift)

        Raises:
            ValueError: 如果参数不合法
        """
        # 验证输入数据
        self._validate_data(factor_df, price_df)

        # 根据方法选择对齐策略
        if method == "default":
            return self._align_default(factor_df, price_df)
        elif method == "flexible":
            return self._align_flexible(factor_df, price_df, shift)
        elif method == "auto":
            return self._align_auto(factor_df, price_df, search_range=auto_search_range, ic_method=ic_calc_method)
        else:
            raise ValueError(f"不合法的对齐方法: {method}\n" f"支持的方法: 'default', 'flexible', 'auto'")

    def _validate_data(self, factor_df: pd.DataFrame, price_df: pd.DataFrame):
        """验证输入数据的合法性"""
        # 检查是否为 MultiIndex
        if not isinstance(factor_df.index, pd.MultiIndex):
            raise ValueError("factor_df 必须是 MultiIndex (datetime, instrument)")

        if not isinstance(price_df.index, pd.MultiIndex):
            raise ValueError("price_df 必须是 MultiIndex (datetime, instrument)")

        # 检查索引名称
        factor_index_names = factor_df.index.names
        price_index_names = price_df.index.names

        if not all(name in factor_index_names for name in ["datetime", "instrument"]):
            raise ValueError("factor_df 索引必须包含 'datetime' 和 'instrument'")

        if not all(name in price_index_names for name in ["datetime", "instrument"]):
            raise ValueError("price_df 索引必须包含 'datetime' 和 'instrument'")

    def _align_default(self, factor_df: pd.DataFrame, price_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        默认对齐方式 (Qlib T+1 to T+2)

        T 日因子 -> 预测 T+2 的收益率

        计算方式：
        - T 日收盘时计算因子
        - T+1 日开盘时买入（使用 T+1 开盘价）
        - T+2 日收盘时卖出（使用 T+2 收盘价）
        - 收益率 = (T+2 收盘价 / T+1 收盘价) - 1

        Args:
            factor_df: 因子数据
            price_df: 价格数据

        Returns:
            (factor_aligned, returns_aligned)
        """
        # 计算 T+1 to T+2 的收益率
        # price_df.shift(-2): T+2 收盘价
        # price_df.shift(-1): T+1 收盘价
        if self.use_log_return:
            returns = np.log(price_df.shift(-2) / price_df.shift(-1))
        else:
            returns = price_df.shift(-2) / price_df.shift(-1) - 1

        # 对齐因子和收益率
        # 因子需要去掉最后两天（因为没有对应的未来收益率）
        factor_aligned = factor_df.iloc[:-2]
        returns_aligned = returns.iloc[:-2]

        return factor_aligned, returns_aligned

    def _align_flexible(
        self, factor_df: pd.DataFrame, price_df: pd.DataFrame, shift: int
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        灵活对齐方式（自定义偏移量）

        Args:
            factor_df: 因子数据
            price_df: 价格数据
            shift: 偏移量
                - shift=1: T to T+1 (jqfactor_analyzer 方式)
                - shift=2: T+1 to T+2 (Qlib 默认方式)
                - shift=N: T+(N-1) to T+N

        Returns:
            (factor_aligned, returns_aligned)

        Raises:
            ValueError: 如果 shift < 1
        """
        if shift < 1:
            raise ValueError(f"shift 必须 >= 1, 当前值: {shift}")

        # 计算未来收益率
        # T+(N-1) to T+N 的收益率
        if self.use_log_return:
            returns = np.log(price_df.shift(-shift) / price_df.shift(-(shift - 1)))
        else:
            returns = price_df.shift(-shift) / price_df.shift(-(shift - 1)) - 1

        # 对齐因子和收益率
        factor_aligned = factor_df.iloc[:-shift]
        returns_aligned = returns.iloc[:-shift]

        return factor_aligned, returns_aligned

    def _align_auto(
        self, factor_df: pd.DataFrame, price_df: pd.DataFrame, search_range: Tuple[int, int], ic_method: str = "pearson"
    ) -> Tuple[pd.DataFrame, pd.DataFrame, int]:
        """
        自动检测最优对齐方式

        通过网格搜索找到使 IC 绝对值最大的偏移量

        Args:
            factor_df: 因子数据
            price_df: 价格数据
            search_range: 搜索范围 (min_shift, max_shift)
            ic_method: IC 计算方法 ('pearson' or 'spearman')

        Returns:
            (factor_aligned, returns_aligned, best_shift)
        """
        min_shift, max_shift = search_range

        if min_shift < 1:
            raise ValueError(f"search_range[0] 必须 >= 1, 当前值: {min_shift}")

        if max_shift > len(factor_df) - 10:
            raise ValueError(f"search_range[1] ({max_shift}) 过大，" f"数据长度不足: {len(factor_df)}")

        # 网格搜索最优偏移量
        ic_values = {}
        for shift in range(min_shift, max_shift + 1):
            factor_aligned, returns_aligned = self._align_flexible(factor_df, price_df, shift)

            # 计算 IC
            ic = self._calculate_ic(factor_aligned, returns_aligned, method=ic_method)
            ic_values[shift] = abs(ic.mean())  # 使用绝对值

        # 找到最优偏移量
        best_shift = max(ic_values, key=ic_values.get)

        # 使用最优偏移量对齐
        factor_aligned, returns_aligned = self._align_flexible(factor_df, price_df, best_shift)

        return factor_aligned, returns_aligned, best_shift

    def _calculate_ic(self, factor_df: pd.DataFrame, returns_df: pd.DataFrame, method: str = "pearson") -> pd.Series:
        """
        计算 IC (Information Coefficient)

        Args:
            factor_df: 因子数据
            returns_df: 收益率数据
            method: 计算方法
                - 'pearson': Pearson IC
                - 'spearman': Spearman Rank IC

        Returns:
            IC 序列 (按日期索引)
        """
        # 确保索引对齐
        common_index = factor_df.index.intersection(returns_df.index)
        factor_aligned = factor_df.loc[common_index]
        returns_aligned = returns_df.loc[common_index]

        # 按日期计算 IC
        dates = factor_aligned.index.get_level_values("datetime").unique()
        ic_series = pd.Series(index=dates, dtype=float)

        for date in dates:
            factor_values = factor_aligned.loc[date].values.flatten()
            return_values = returns_aligned.loc[date].values.flatten()

            # 移除 NaN
            mask = ~(np.isnan(factor_values) | np.isnan(return_values))
            factor_values = factor_values[mask]
            return_values = return_values[mask]

            if len(factor_values) < 2:
                ic_series[date] = np.nan
                continue

            # 计算 IC
            if method == "pearson":
                ic = np.corrcoef(factor_values, return_values)[0, 1]
            elif method == "spearman":
                from scipy.stats import spearmanr

                ic, _ = spearmanr(factor_values, return_values)
            else:
                raise ValueError(f"不支持的 IC 计算方法: {method}")

            ic_series[date] = ic

        return ic_series

    def validate_alignment(
        self, factor_df: pd.DataFrame, returns_df: pd.DataFrame, max_nan_ratio: float = 0.1
    ) -> Dict[str, any]:
        """
        验证对齐结果的质量

        检查项：
        1. NaN 比例
        2. 数据对齐情况
        3. 因子分布统计

        Args:
            factor_df: 对齐后的因子数据
            returns_df: 对齐后的收益率数据
            max_nan_ratio: 最大允许的 NaN 比例

        Returns:
            验证结果字典
        """
        result = {"is_valid": True, "warnings": [], "errors": [], "statistics": {}}

        # 检查索引对齐
        if not factor_df.index.equals(returns_df.index):
            result["is_valid"] = False
            result["errors"].append("因子和收益率索引不对齐")
            return result

        # 检查 NaN 比例
        factor_nan_ratio = factor_df.isna().sum().sum() / factor_df.size
        returns_nan_ratio = returns_df.isna().sum().sum() / returns_df.size

        result["statistics"]["factor_nan_ratio"] = factor_nan_ratio
        result["statistics"]["returns_nan_ratio"] = returns_nan_ratio

        if factor_nan_ratio > max_nan_ratio:
            result["warnings"].append(f"因子 NaN 比例过高: {factor_nan_ratio:.2%} > {max_nan_ratio:.2%}")

        if returns_nan_ratio > max_nan_ratio:
            result["warnings"].append(f"收益率 NaN 比例过高: {returns_nan_ratio:.2%} > {max_nan_ratio:.2%}")

        # 检查收益率异常值
        returns_min = returns_df.min().min()
        returns_max = returns_df.max().max()

        result["statistics"]["returns_min"] = returns_min
        result["statistics"]["returns_max"] = returns_max

        # 检查是否有极端收益率（如涨跌停）
        if returns_max > 0.15:  # 涨幅 > 15%
            result["warnings"].append(f"检测到极端正向收益率: {returns_max:.2%}")

        if returns_min < -0.15:  # 跌幅 < -15%
            result["warnings"].append(f"检测到极端负向收益率: {returns_min:.2%}")

        # 因子分布统计
        factor_flat = factor_df.values.flatten()
        factor_flat = factor_flat[~np.isnan(factor_flat)]

        if len(factor_flat) > 0:
            result["statistics"]["factor_mean"] = np.mean(factor_flat)
            result["statistics"]["factor_std"] = np.std(factor_flat)
            result["statistics"]["factor_min"] = np.min(factor_flat)
            result["statistics"]["factor_max"] = np.max(factor_flat)

            # 检查因子是否有足够的变异
            if result["statistics"]["factor_std"] < 1e-6:
                result["warnings"].append("因子标准差过小，可能缺乏变异")

        return result

    def calculate_forward_returns(self, price_df: pd.DataFrame, shift: int = 2, method: str = "simple") -> pd.DataFrame:
        """
        计算未来收益率（辅助方法）

        Args:
            price_df: 价格数据
            shift: 偏移量
            method: 收益率计算方法
                - 'simple': 简单收益率 (默认)
                - 'log': 对数收益率

        Returns:
            收益率数据
        """
        if method == "log":
            returns = np.log(price_df.shift(-shift) / price_df.shift(-(shift - 1)))
        elif method == "simple":
            returns = price_df.shift(-shift) / price_df.shift(-(shift - 1)) - 1
        else:
            raise ValueError(f"不支持的收益率计算方法: {method}")

        return returns

    def get_alignment_summary(self, factor_df: pd.DataFrame, price_df: pd.DataFrame, shift: int) -> Dict[str, any]:
        """
        获取对齐方案摘要

        Args:
            factor_df: 原始因子数据
            price_df: 原始价格数据
            shift: 偏移量

        Returns:
            对齐摘要信息
        """
        factor_aligned, returns_aligned = self._align_flexible(factor_df, price_df, shift)

        summary = {
            "shift": shift,
            "description": f"T+{shift-1} to T+{shift}",
            "original_dates": len(factor_df.index.get_level_values("datetime").unique()),
            "aligned_dates": len(factor_aligned.index.get_level_values("datetime").unique()),
            "data_loss": (len(factor_df) - len(factor_aligned)) / len(factor_df) if len(factor_df) > 0 else 0,
        }

        return summary


# 便捷函数
def align_factor_returns(
    factor_df: pd.DataFrame, price_df: pd.DataFrame, method: str = "default", shift: int = 2, **kwargs
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[pd.DataFrame, pd.DataFrame, int]]:
    """
    对齐因子和收益率的便捷函数

    Args:
        factor_df: 因子数据
        price_df: 价格数据
        method: 对齐方法 ('default', 'flexible', 'auto')
        shift: 偏移量（用于 flexible 方法）
        **kwargs: 其他参数

    Returns:
        对齐后的数据和可能的偏移量

    Examples:
        >>> from core.cycle_aligner import align_factor_returns
        >>>
        >>> # 默认对齐
        >>> factor, returns = align_factor_returns(factor_df, price_df)
        >>>
        >>> # 自定义对齐
        >>> factor, returns = align_factor_returns(
        ...     factor_df, price_df, method='flexible', shift=1
        ... )
        >>>
        >>> # 自动检测
        >>> factor, returns, best_shift = align_factor_returns(
        ...     factor_df, price_df, method='auto'
        ... )
    """
    aligner = CycleAligner(**kwargs)
    return aligner.align(factor_df, price_df, method=method, shift=shift)
