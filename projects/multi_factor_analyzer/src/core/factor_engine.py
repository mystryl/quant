"""
因子管理器模块

本模块提供因子的注册、计算和管理功能，是整个多因子分析系统的核心组件。

主要功能：
1. 因子注册 - 注册自定义因子函数或表达式
2. 因子计算 - 计算单个或批量因子
3. 因子缓存 - 缓存已计算的因子结果
4. 未来函数检测 - 确保因子计算不使用未来数据
5. 因子管理 - 查询、删除、更新已注册的因子
"""

import pickle
from pathlib import Path
from typing import Callable, Dict, List, Optional, Union

import pandas as pd

from .factor_expression_parser import FactorExpressionParser


class FactorManager:
    """
    因子管理器

    管理因子的注册、计算和缓存，支持多种因子定义方式：
    1. 表达式字符串：使用 Qlib 表达式语法
    2. Python 函数：自定义计算逻辑
    3. 预计算数据：直接加载已计算的因子数据

    Attributes:
        data_provider: 数据提供者实例
        factors: 已注册的因子字典 {name: definition}
        factor_cache: 因子计算缓存
        expr_parser: 表达式解析器，用于检测未来函数
        cache_enabled: 是否启用缓存

    Examples:
        >>> from qlid_backtest.scripts.data import SmartDataProvider
        >>>
        >>> # 初始化因子管理器
        >>> provider = SmartDataProvider("/path/to/data")
        >>> manager = FactorManager(provider)
        >>>
        >>> # 注册表达式因子
        >>> manager.register_factor(
        ...     "MA20",
        ...     "Ref($close, 20) / $close - 1"
        ... )
        >>>
        >>> # 注册函数因子
        >>> def custom_factor(provider, instruments, start, end):
        ...     data = provider.get_data(instruments, ["$close"], start, end)
        ...     return data["$close"].pct_change()
        >>>
        >>> manager.register_factor("custom", custom_factor)
        >>>
        >>> # 计算因子
        >>> factor_data = manager.calculate_factor(
        ...     "MA20",
        ...     instruments=["SH600000"],
        ...     start_date="2020-01-01",
        ...     end_date="2020-12-31"
        ... )
    """

    def __init__(self, data_provider, cache_enabled: bool = True, cache_dir: Optional[str] = None):
        """
        初始化因子管理器

        Args:
            data_provider: 数据提供者实例（支持 get_data 方法）
            cache_enabled: 是否启用因子计算缓存
            cache_dir: 缓存目录路径，默认为 ./factor_cache

        Examples:
            >>> from qlib_backtest.scripts.data import SmartDataProvider
            >>> provider = SmartDataProvider("/path/to/data")
            >>> manager = FactorManager(
            ...     provider,
            ...     cache_enabled=True,
            ...     cache_dir="./cache"
            ... )
        """
        self.data_provider = data_provider
        self.factors: Dict[str, Union[str, Callable]] = {}
        self.factor_cache: Dict[str, pd.DataFrame] = {}
        self.expr_parser = FactorExpressionParser()
        self.cache_enabled = cache_enabled

        # 设置缓存目录
        if cache_dir is None:
            cache_dir = "./factor_cache"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # 加载已有的缓存
        if cache_enabled:
            self._load_cache()

    def register_factor(self, name: str, factor_def: Union[str, Callable], metadata: Optional[Dict] = None) -> None:
        """
        注册因子

        支持两种因子定义方式：
        1. 表达式字符串：使用 Qlib 表达式语法（会自动检测未来函数）
        2. Python 函数：自定义计算逻辑

        Args:
            name: 因子名称（唯一标识符）
            factor_def: 因子定义（表达式字符串或函数）
            metadata: 因子元数据（如描述、作者、版本等）

        Raises:
            ValueError: 如果因子名称已存在或表达式包含未来函数
            TypeError: 如果因子定义类型不支持

        Examples:
            >>> # 注册表达式因子
            >>> manager.register_factor(
            ...     "MA20",
            ...     "Ref($close, 20) / $close - 1",
            ...     metadata={"description": "20日均线偏离度"}
            ... )
            >>>
            >>> # 注册函数因子
            >>> def my_factor(provider, instruments, start, end):
            ...     # 自定义计算逻辑
            ...     return factor_data
            >>>
            >>> manager.register_factor("my_factor", my_factor)
        """
        # 检查因子名称是否已存在
        if name in self.factors:
            raise ValueError(f"因子 '{name}' 已存在。" f"如需更新，请先使用 unregister_factor() 删除旧因子。")

        # 如果是表达式，验证未来函数
        if isinstance(factor_def, str):
            self.expr_parser.validate_no_future_functions(factor_def)
        elif not callable(factor_def):
            raise TypeError(f"不支持的因子定义类型: {type(factor_def)}。" f"仅支持 str 或 Callable。")

        # 注册因子
        self.factors[name] = {
            "definition": factor_def,
            "metadata": metadata or {},
            "type": "expression" if isinstance(factor_def, str) else "function",
        }

    def unregister_factor(self, name: str) -> None:
        """
        删除已注册的因子

        Args:
            name: 要删除的因子名称

        Raises:
            KeyError: 如果因子不存在

        Examples:
            >>> manager.unregister_factor("MA20")
        """
        if name not in self.factors:
            raise KeyError(f"因子 '{name}' 不存在")

        del self.factors[name]

        # 清除缓存
        if name in self.factor_cache:
            del self.factor_cache[name]

        # 删除缓存文件
        cache_file = self.cache_dir / f"{name}.pkl"
        if cache_file.exists():
            cache_file.unlink()

    def calculate_factor(
        self, name: str, instruments: List[str], start_date: str, end_date: str, use_cache: bool = True, **kwargs
    ) -> pd.DataFrame:
        """
        计算因子

        Args:
            name: 因子名称
            instruments: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            use_cache: 是否使用缓存
            **kwargs: 传递给数据提供者的额外参数

        Returns:
            pd.DataFrame: 因子数据，索引为 (datetime, instrument)

        Raises:
            KeyError: 如果因子未注册
            Exception: 如果因子计算失败

        Examples:
            >>> factor_data = manager.calculate_factor(
            ...     "MA20",
            ...     instruments=["SH600000", "SH600001"],
            ...     start_date="2020-01-01",
            ...     end_date="2020-12-31"
            ... )
            >>> print(factor_data.head())
        """
        # 检查因子是否已注册
        if name not in self.factors:
            raise KeyError(f"因子 '{name}' 未注册。" f"请先使用 register_factor() 注册该因子。")

        # 生成缓存键
        cache_key = self._generate_cache_key(name, instruments, start_date, end_date)

        # 尝试从缓存加载
        if use_cache and self.cache_enabled and cache_key in self.factor_cache:
            return self.factor_cache[cache_key].copy()

        # 获取因子定义
        factor_info = self.factors[name]
        factor_def = factor_info["definition"]
        factor_type = factor_info["type"]

        # 根据类型计算因子
        try:
            if factor_type == "expression":
                factor_data = self._calculate_expression_factor(factor_def, instruments, start_date, end_date, **kwargs)
            else:
                factor_data = self._calculate_function_factor(factor_def, instruments, start_date, end_date, **kwargs)

            # 缓存结果
            if self.cache_enabled:
                self.factor_cache[cache_key] = factor_data.copy()
                self._save_cache(name, cache_key, factor_data)

            return factor_data

        except Exception as e:
            raise Exception(f"因子 '{name}' 计算失败: {str(e)}\n" f"因子定义: {factor_def}")

    def _calculate_expression_factor(
        self, expression: str, instruments: List[str], start_date: str, end_date: str, **kwargs
    ) -> pd.DataFrame:
        """
        计算表达式因子

        Args:
            expression: Qlib 表达式
            instruments: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 额外参数

        Returns:
            pd.DataFrame: 计算结果
        """
        # 提取表达式中的字段
        fields = self.expr_parser.extract_fields(expression)
        required_fields = [f"${f}" for f in fields]

        # 获取数据
        _data = self.data_provider.get_data(  # noqa: F841
            instrument=instruments, fields=required_fields, start_date=start_date, end_date=end_date, **kwargs
        )

        # 这里应该使用 Qlib 的表达式引擎
        # 为了简化，我们假设 data_provider 已经处理了表达式
        # 实际实现中，应该使用 qlib.expression.Expression
        # 或 qlib.contrib.strategy.signal_strategy.create_signal_from_expression

        # TODO: 实现表达式计算逻辑
        # 这里需要集成 Qlib 的表达式引擎
        raise NotImplementedError(
            "表达式计算需要集成 Qlib 表达式引擎。" "请参考 qlib.contrib.strategy.signal_strategy 的实现。"
        )

    def _calculate_function_factor(
        self, factor_func: Callable, instruments: List[str], start_date: str, end_date: str, **kwargs
    ) -> pd.DataFrame:
        """
        计算函数因子

        Args:
            factor_func: 因子计算函数
            instruments: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 额外参数

        Returns:
            pd.DataFrame: 计算结果
        """
        # 调用因子函数
        return factor_func(self.data_provider, instruments, start_date, end_date, **kwargs)

    def calculate_batch_factors(
        self, names: List[str], instruments: List[str], start_date: str, end_date: str, **kwargs
    ) -> Dict[str, pd.DataFrame]:
        """
        批量计算多个因子

        Args:
            names: 因子名称列表
            instruments: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期
            **kwargs: 传递给 calculate_factor 的额外参数

        Returns:
            Dict[str, pd.DataFrame]: 因子数据字典

        Examples:
            >>> factors = manager.calculate_batch_factors(
            ...     ["MA20", "MA60", "VOL20"],
            ...     instruments=["SH600000"],
            ...     start_date="2020-01-01",
            ...     end_date="2020-12-31"
            ... )
            >>> print(factors.keys())
            dict_keys(['MA20', 'MA60', 'VOL20'])
        """
        results = {}
        for name in names:
            results[name] = self.calculate_factor(name, instruments, start_date, end_date, **kwargs)
        return results

    def list_factors(self) -> List[Dict]:
        """
        列出所有已注册的因子

        Returns:
            List[Dict]: 因子信息列表

        Examples:
            >>> factors = manager.list_factors()
            >>> for factor in factors:
            ...     print(f"{factor['name']}: {factor['type']}")
        """
        return [
            {"name": name, "type": info["type"], "metadata": info["metadata"]} for name, info in self.factors.items()
        ]

    def get_factor_info(self, name: str) -> Dict:
        """
        获取因子详细信息

        Args:
            name: 因子名称

        Returns:
            Dict: 因子信息

        Raises:
            KeyError: 如果因子不存在

        Examples:
            >>> info = manager.get_factor_info("MA20")
            >>> print(info['metadata']['description'])
        """
        if name not in self.factors:
            raise KeyError(f"因子 '{name}' 不存在")

        return self.factors[name].copy()

    def clear_cache(self, name: Optional[str] = None) -> None:
        """
        清除缓存

        Args:
            name: 要清除的因子名称，如果为 None 则清除所有缓存

        Examples:
            >>> # 清除特定因子的缓存
            >>> manager.clear_cache("MA20")
            >>>
            >>> # 清除所有缓存
            >>> manager.clear_cache()
        """
        if name is None:
            # 清除所有缓存
            self.factor_cache.clear()
            # 删除所有缓存文件
            for cache_file in self.cache_dir.glob("*.pkl"):
                cache_file.unlink()
        else:
            # 清除特定因子的缓存
            keys_to_delete = [key for key in self.factor_cache if key.startswith(f"{name}_")]
            for key in keys_to_delete:
                del self.factor_cache[key]

            # 删除缓存文件
            cache_file = self.cache_dir / f"{name}.pkl"
            if cache_file.exists():
                cache_file.unlink()

    def _generate_cache_key(self, name: str, instruments: List[str], start_date: str, end_date: str) -> str:
        """
        生成缓存键

        Args:
            name: 因子名称
            instruments: 股票代码列表
            start_date: 开始日期
            end_date: 结束日期

        Returns:
            str: 缓存键
        """
        # 将股票列表排序后连接，确保相同集合生成相同的键
        instruments_str = "_".join(sorted(instruments))
        return f"{name}_{instruments_str}_{start_date}_{end_date}"

    def _save_cache(self, name: str, cache_key: str, data: pd.DataFrame) -> None:
        """
        保存缓存到磁盘

        Args:
            name: 因子名称
            cache_key: 缓存键
            data: 要缓存的数据
        """
        cache_file = self.cache_dir / f"{name}.pkl"
        try:
            with open(cache_file, "wb") as f:
                pickle.dump({cache_key: data}, f)
        except Exception as e:
            print(f"警告: 无法保存缓存文件 {cache_file}: {e}")

    def _load_cache(self) -> None:
        """
        从磁盘加载缓存
        """
        for cache_file in self.cache_dir.glob("*.pkl"):
            try:
                with open(cache_file, "rb") as f:
                    cached_data = pickle.load(f)
                    self.factor_cache.update(cached_data)
            except Exception as e:
                print(f"警告: 无法加载缓存文件 {cache_file}: {e}")


# 便捷函数
def create_factor_manager(data_provider, cache_enabled: bool = True, cache_dir: Optional[str] = None) -> FactorManager:
    """
    创建因子管理器的便捷函数

    Args:
        data_provider: 数据提供者实例
        cache_enabled: 是否启用缓存
        cache_dir: 缓存目录

    Returns:
        FactorManager: 因子管理器实例

    Examples:
        >>> from qlib_backtest.scripts.data import SmartDataProvider
        >>> provider = SmartDataProvider("/path/to/data")
        >>> manager = create_factor_manager(provider)
    """
    return FactorManager(data_provider, cache_enabled, cache_dir)


if __name__ == "__main__":
    # 示例：使用因子管理器
    print("=" * 60)
    print("因子管理器示例")
    print("=" * 60)

    # 注意：这个示例需要数据提供者，实际使用时需要提供真实的数据提供者
    print("\n提示：此示例需要数据提供者实例。")
    print("实际使用时，请参考以下代码：")
    print("""
    from qlib_backtest.scripts.data import SmartDataProvider
    from src.core.factor_engine import FactorManager

    # 初始化
    provider = SmartDataProvider("/path/to/data")
    manager = FactorManager(provider)

    # 注册表达式因子
    manager.register_factor(
        "MA20",
        "Ref($close, 20) / $close - 1",
        metadata={"description": "20日均线偏离度"}
    )

    # 注册函数因子
    def custom_factor(provider, instruments, start, end):
        data = provider.get_data(instruments, ["$close"], start, end)
        return data["$close"].pct_change()

    manager.register_factor("custom", custom_factor)

    # 计算因子
    factor_data = manager.calculate_factor(
        "MA20",
        instruments=["SH600000"],
        start_date="2020-01-01",
        end_date="2020-12-31"
    )

    # 列出所有因子
    factors = manager.list_factors()
    for factor in factors:
        print(f"{factor['name']}: {factor['type']}")
    """)
