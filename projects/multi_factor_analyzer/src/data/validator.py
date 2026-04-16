"""
数据验证器模块 - DataValidator

该模块实现数据验证和因子处理功能，包括数据质量检查、
因子标准化、因子中性化等操作。

主要特性：
- 检查 NaN、异常值
- 因子标准化（z-score、min-max）
- 因子中性化（行业中性、市值中性）
- 数据质量报告生成
"""

import logging
from typing import Optional, Dict, Any, Tuple

import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


class DataValidator:
    """
    数据验证器

    提供数据质量检查、因子标准化、因子中性化等功能。

    Examples:
        >>> validator = DataValidator()
        >>> # 检查数据质量
        >>> report = validator.check_data_quality(data)
        >>> print(report)
        >>> # 标准化因子
        >>> standardized = validator.standardize_factor(data['close'], method='zscore')
        >>> # 中性化因子
        >>> neutralized = validator.neutralize_factor(
        ...     factor=data['factor'],
        ...     industry=data['industry'],
        ...     market_cap=data['market_cap']
        ... )
    """

    # 标准化方法
    STANDARDIZE_METHODS = ["zscore", "minmax", "rank", "robust"]

    # 异常值检测方法
    OUTLIER_METHODS = ["zscore", "iqr", "isolation", "none"]

    def __init__(self, nan_threshold: float = 0.1, outlier_method: str = "zscore", outlier_threshold: float = 3.0):
        """
        初始化数据验证器

        Args:
            nan_threshold: NaN 比例阈值（超过则警告）
            outlier_method: 异常值检测方法
                           - 'zscore': 基于 z-score（默认）
                           - 'iqr': 基于 IQR（四分位距）
                           - 'isolation': 基于隔离森林
                           - 'none': 不检测
            outlier_threshold: 异常值阈值
                              - zscore 方法: 默认 3.0
                              - iqr 方法: 默认 1.5
        """
        self.nan_threshold = nan_threshold
        self.outlier_method = outlier_method
        self.outlier_threshold = outlier_threshold

        logger.info(
            f"数据验证器初始化完成: nan_threshold={nan_threshold}, "
            f"outlier_method={outlier_method}, "
            f"outlier_threshold={outlier_threshold}"
        )

    def check_data_quality(self, data: pd.DataFrame, return_report: bool = True) -> Optional[Dict[str, Any]]:
        """
        检查数据质量

        Args:
            data: 数据框
            return_report: 是否返回报告

        Returns:
            Dict: 数据质量报告（如果 return_report=True）

        Examples:
            >>> validator = DataValidator()
            >>> data = pd.DataFrame({
            ...     'close': [100, 101, 102, np.nan, 104],
            ...     'volume': [1000, 1100, 1200, 1300, 1400]
            ... })
            >>> report = validator.check_data_quality(data)
            >>> print(report)
        """
        report = {
            "total_rows": len(data),
            "total_columns": len(data.columns),
            "missing_values": {},
            "outliers": {},
            "statistics": {},
            "warnings": [],
            "errors": [],
        }

        # 检查每一列
        for col in data.columns:
            col_data = data[col]

            # 跳过非数值列
            if not np.issubdtype(col_data.dtype, np.number):
                continue

            # 1. 检查缺失值
            missing_count = col_data.isnull().sum()
            missing_ratio = missing_count / len(col_data)

            report["missing_values"][col] = {"count": int(missing_count), "ratio": float(missing_ratio)}

            if missing_ratio > self.nan_threshold:
                report["warnings"].append(
                    f"{col}: 缺失值比例 {missing_ratio:.1%} " f"超过阈值 {self.nan_threshold:.1%}"
                )

            # 2. 检查异常值
            if self.outlier_method != "none":
                outlier_mask = self._detect_outliers(col_data.dropna())
                outlier_count = outlier_mask.sum()
                outlier_ratio = outlier_count / len(col_data.dropna())

                report["outliers"][col] = {"count": int(outlier_count), "ratio": float(outlier_ratio)}

                if outlier_ratio > 0.05:  # 5% 阈值
                    report["warnings"].append(f"{col}: 异常值比例 {outlier_ratio:.1%}")

            # 3. 统计信息
            report["statistics"][col] = {
                "mean": float(col_data.mean()) if not col_data.isnull().all() else np.nan,
                "std": float(col_data.std()) if not col_data.isnull().all() else np.nan,
                "min": float(col_data.min()) if not col_data.isnull().all() else np.nan,
                "max": float(col_data.max()) if not col_data.isnull().all() else np.nan,
                "count": int(col_data.count()),
            }

        # 检查逻辑错误
        if all(col in data.columns for col in ["high", "low"]):
            invalid_hl = (data["high"] < data["low"]).sum()
            if invalid_hl > 0:
                report["errors"].append(f"逻辑错误: high < low 出现 {invalid_hl} 次")

        if all(col in data.columns for col in ["open", "high", "low", "close"]):
            invalid_ohlc = (
                (data["high"] < data["open"])
                | (data["high"] < data["close"])
                | (data["low"] > data["open"])
                | (data["low"] > data["close"])
            ).sum()
            if invalid_ohlc > 0:
                report["errors"].append(f"逻辑错误: OHLC 数据不一致 {invalid_ohlc} 次")

        # 输出报告
        if return_report:
            self._print_quality_report(report)
            return report
        else:
            return None

    def standardize_factor(
        self, factor: pd.Series, method: str = "zscore", clip_range: Optional[Tuple[float, float]] = None
    ) -> pd.Series:
        """
        标准化因子

        Args:
            factor: 因子序列
            method: 标准化方法
                   - 'zscore': z-score 标准化（默认）
                   - 'minmax': min-max 标准化到 [0, 1]
                   - 'rank': 秩标准化（排名标准化）
                   - 'robust': 鲁棒标准化（基于中位数和 IQR）
            clip_range: 裁剪范围 (min, max)，如 (-3, 3) 用于 z-score

        Returns:
            pd.Series: 标准化后的因子

        Examples:
            >>> validator = DataValidator()
            >>> factor = pd.Series([1, 2, 3, 4, 5, 100])  # 包含异常值
            >>> # z-score 标准化
            >>> standardized = validator.standardize_factor(factor, method='zscore')
            >>> # min-max 标准化
            >>> normalized = validator.standardize_factor(factor, method='minmax')
            >>> # 秩标准化
            >>> ranked = validator.standardize_factor(factor, method='rank')
        """
        if method not in self.STANDARDIZE_METHODS:
            raise ValueError(f"无效的 method: {method}。" f"支持的方法: {self.STANDARDIZE_METHODS}")

        factor_copy = factor.copy()

        # 处理 NaN
        nan_mask = factor_copy.isnull()
        valid_data = factor_copy.dropna()

        if len(valid_data) == 0:
            logger.warning("因子全为 NaN，无法标准化")
            return factor_copy

        # 标准化
        if method == "zscore":
            # z-score 标准化: (x - mean) / std
            mean = valid_data.mean()
            std = valid_data.std()
            if std > 0:
                factor_copy[~nan_mask] = (valid_data - mean) / std
            else:
                logger.warning("因子标准差为 0，无法 z-score 标准化")
                factor_copy[~nan_mask] = 0

        elif method == "minmax":
            # min-max 标准化: (x - min) / (max - min)
            min_val = valid_data.min()
            max_val = valid_data.max()
            range_val = max_val - min_val
            if range_val > 0:
                factor_copy[~nan_mask] = (valid_data - min_val) / range_val
            else:
                logger.warning("因子范围为 0，无法 min-max 标准化")
                factor_copy[~nan_mask] = 0.5

        elif method == "rank":
            # 秩标准化: (rank - 1) / (n - 1)
            ranks = valid_data.rank(method="average")
            factor_copy[~nan_mask] = (ranks - 1) / (len(valid_data) - 1)

        elif method == "robust":
            # 鲁棒标准化: (x - median) / IQR
            median = valid_data.median()
            q75 = valid_data.quantile(0.75)
            q25 = valid_data.quantile(0.25)
            iqr = q75 - q25
            if iqr > 0:
                factor_copy[~nan_mask] = (valid_data - median) / iqr
            else:
                logger.warning("因子 IQR 为 0，无法鲁棒标准化")
                factor_copy[~nan_mask] = 0

        # 裁剪
        if clip_range is not None:
            min_val, max_val = clip_range
            factor_copy = factor_copy.clip(lower=min_val, upper=max_val)

        logger.debug(f"因子标准化完成: method={method}, " f"范围=[{factor_copy.min():.2f}, {factor_copy.max():.2f}]")

        return factor_copy

    def neutralize_factor(
        self,
        factor: pd.Series,
        industry: Optional[pd.Series] = None,
        market_cap: Optional[pd.Series] = None,
        method: str = "regression",
    ) -> pd.Series:
        """
        因子中性化

        去除因子对行业、市值等风格因子的暴露。

        Args:
            factor: 因子序列
            industry: 行业分类序列（可选）
            market_cap: 市值序列（可选）
            method: 中性化方法
                   - 'regression': 回归中性化（默认）
                   - 'orthogonal': 正交化

        Returns:
            pd.Series: 中性化后的因子

        Examples:
            >>> validator = DataValidator()
            >>> factor = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
            >>> market_cap = pd.Series([100, 200, 300, 400, 500])
            >>> # 市值中性化
            >>> neutralized = validator.neutralize_factor(
            ...     factor=factor,
            ...     market_cap=market_cap
            ... )
        """
        factor_copy = factor.copy()

        # 处理 NaN
        nan_mask = factor_copy.isnull()
        valid_mask = ~nan_mask

        # 市值中性化
        if market_cap is not None:
            # 市值取对数
            log_mcap = np.log(market_cap[valid_mask])

            # 标准化市值
            log_mcap = (log_mcap - log_mcap.mean()) / log_mcap.std()

            # 回归去市值
            if method == "regression":
                # 简单线性回归
                valid_factor = factor_copy[valid_mask]

                # 计算回归系数
                cov = np.cov(valid_factor, log_mcap)[0, 1]
                var = np.var(log_mcap)

                if var > 0:
                    beta = cov / var
                    residual = valid_factor - beta * log_mcap
                    factor_copy[valid_mask] = residual
                else:
                    logger.warning("市值方差为 0，无法市值中性化")

        # 行业中性化
        if industry is not None:
            # 按行业分组，计算行业均值
            industry_mean = factor_copy.groupby(industry).transform("mean")
            factor_copy = factor_copy - industry_mean

        logger.debug(f"因子中性化完成: method={method}")

        return factor_copy

    def winsorize_factor(self, factor: pd.Series, limits: Tuple[float, float] = (0.05, 0.05)) -> pd.Series:
        """
        去极值（缩尾处理）

        Args:
            factor: 因子序列
            limits: 裁剪比例 (lower, upper)，如 (0.05, 0.05) 表示裁剪上下 5%

        Returns:
            pd.Series: 去极值后的因子

        Examples:
            >>> validator = DataValidator()
            >>> factor = pd.Series([1, 2, 3, 4, 5, 100])
            >>> # 去除上下 5% 的极值
            >>> winsorized = validator.winsorize_factor(factor, limits=(0.05, 0.05))
        """
        lower_limit, upper_limit = limits

        # 使用 scipy.stats.mstats.winsorize
        from scipy.stats.mstats import winsorize

        # 处理 NaN
        nan_mask = factor.isnull()
        valid_data = factor[~nan_mask]

        if len(valid_data) == 0:
            logger.warning("因子全为 NaN，无法去极值")
            return factor

        # 去极值
        winsorized_values = winsorize(valid_data.values, limits=(lower_limit, upper_limit))

        # 重建 Series
        result = factor.copy()
        result[~nan_mask] = winsorized_values

        logger.debug(f"因子去极值完成: limits={limits}, " f"范围=[{result.min():.2f}, {result.max():.2f}]")

        return result

    # ========================================================================
    # 内部方法
    # ========================================================================

    def _detect_outliers(self, data: pd.Series) -> pd.Series:
        """
        检测异常值

        Args:
            data: 数据序列

        Returns:
            pd.Series: 布尔序列，True 表示异常值
        """
        if self.outlier_method == "zscore":
            # z-score 方法
            z_scores = np.abs(stats.zscore(data))
            return pd.Series(z_scores > self.outlier_threshold, index=data.index)

        elif self.outlier_method == "iqr":
            # IQR 方法
            q75 = data.quantile(0.75)
            q25 = data.quantile(0.25)
            iqr = q75 - q25

            lower_bound = q25 - self.outlier_threshold * iqr
            upper_bound = q75 + self.outlier_threshold * iqr

            return (data < lower_bound) | (data > upper_bound)

        elif self.outlier_method == "isolation":
            # 隔离森林方法
            from sklearn.ensemble import IsolationForest

            clf = IsolationForest(contamination=0.05, random_state=42)
            outliers = clf.fit_predict(data.values.reshape(-1, 1))

            return pd.Series(outliers == -1, index=data.index)

        else:
            # 不检测
            return pd.Series(False, index=data.index)

    def _print_quality_report(self, report: Dict[str, Any]) -> None:
        """
        打印数据质量报告

        Args:
            report: 质量报告字典
        """
        print("\n" + "=" * 60)
        print("数据质量报告".center(60))
        print("=" * 60)

        print(f"\n数据维度: {report['total_rows']} 行 × {report['total_columns']} 列")

        # 缺失值
        if report["missing_values"]:
            print("\n缺失值统计:")
            for col, col_stats in report["missing_values"].items():
                print(f"  {col}: {col_stats['count']} ({col_stats['ratio']:.1%})")

        # 异常值
        if report["outliers"]:
            print("\n异常值统计:")
            for col, col_stats in report["outliers"].items():
                print(f"  {col}: {col_stats['count']} ({col_stats['ratio']:.1%})")

        # 统计信息
        if report["statistics"]:
            print("\n基本统计:")
            for col, col_stats in list(report["statistics"].items())[:5]:  # 只显示前5列
                print(f"  {col}:")
                print(f"    均值:   {col_stats['mean']:.2f}")
                print(f"    标准差:   {col_stats['std']:.2f}")
                print(f"    范围: [  {col_stats['min']:.2f},   {col_stats['max']:.2f}]")

        # 警告
        if report["warnings"]:
            print("\n警告:")
            for warning in report["warnings"]:
                print(f"  ⚠️  {warning}")

        # 错误
        if report["errors"]:
            print("\n错误:")
            for error in report["errors"]:
                print(f"  ❌ {error}")

        print("\n" + "=" * 60 + "\n")


# =============================================================================
# 便捷函数
# =============================================================================


def check_data_quality(data: pd.DataFrame) -> Dict[str, Any]:
    """
    便捷函数：检查数据质量

    Args:
        data: 数据框

    Returns:
        Dict: 数据质量报告
    """
    validator = DataValidator()
    return validator.check_data_quality(data)


def standardize_factor(factor: pd.Series, method: str = "zscore") -> pd.Series:
    """
    便捷函数：标准化因子

    Args:
        factor: 因子序列
        method: 标准化方法

    Returns:
        pd.Series: 标准化后的因子
    """
    validator = DataValidator()
    return validator.standardize_factor(factor, method=method)


def neutralize_factor(factor: pd.Series, market_cap: Optional[pd.Series] = None) -> pd.Series:
    """
    便捷函数：因子中性化

    Args:
        factor: 因子序列
        market_cap: 市值序列

    Returns:
        pd.Series: 中性化后的因子
    """
    validator = DataValidator()
    return validator.neutralize_factor(factor, market_cap=market_cap)


# =============================================================================
# 测试代码
# =============================================================================

if __name__ == "__main__":
    # 配置日志
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")

    # 创建测试数据
    print("\n=== 测试1: 数据质量检查 ===")
    test_data = pd.DataFrame(
        {
            "close": [100, 101, 102, np.nan, 104, 105, 1000],  # 包含 NaN 和异常值
            "volume": [1000, 1100, 1200, 1300, 1400, 1500, 1600],
            "high": [102, 103, 104, 105, 106, 107, 1005],
            "low": [99, 100, 101, 102, 103, 104, 995],
        }
    )

    validator = DataValidator()
    report = validator.check_data_quality(test_data)

    print("\n=== 测试2: 因子标准化 ===")
    factor = pd.Series([1, 2, 3, 4, 5, 100])

    print("原始因子:")
    print(factor)
    print(f"均值: {factor.mean():.2f}, 标准差: {factor.std():.2f}")

    print("\nz-score 标准化:")
    zscore = validator.standardize_factor(factor, method="zscore")
    print(zscore)
    print(f"均值: {zscore.mean():.2f}, 标准差: {zscore.std():.2f}")

    print("\nmin-max 标准化:")
    minmax = validator.standardize_factor(factor, method="minmax")
    print(minmax)
    print(f"范围: [{minmax.min():.2f}, {minmax.max():.2f}]")

    print("\n秩标准化:")
    rank = validator.standardize_factor(factor, method="rank")
    print(rank)
    print(f"范围: [{rank.min():.2f}, {rank.max():.2f}]")

    print("\n=== 测试3: 因子中性化 ===")
    factor = pd.Series([0.1, 0.2, 0.3, 0.4, 0.5])
    market_cap = pd.Series([100, 200, 300, 400, 500])

    print("原始因子:")
    print(factor)

    print("\n市值中性化后:")
    neutralized = validator.neutralize_factor(factor=factor, market_cap=market_cap)
    print(neutralized)

    print("\n=== 测试4: 去极值 ===")
    factor = pd.Series([1, 2, 3, 4, 5, 100, -50])

    print("原始因子:")
    print(factor)

    print("\n去极值后 (5%):")
    winsorized = validator.winsorize_factor(factor, limits=(0.05, 0.05))
    print(winsorized)

    print("\n=== 测试5: 异常值检测 ===")
    factor = pd.Series([1, 2, 3, 4, 5, 100])

    print("因子:")
    print(factor)

    print("\nz-score 方法检测异常值:")
    validator_zscore = DataValidator(outlier_method="zscore")
    outliers = validator_zscore._detect_outliers(factor)
    print(outliers)

    print("\nIQR 方法检测异常值:")
    validator_iqr = DataValidator(outlier_method="iqr", outlier_threshold=1.5)
    outliers = validator_iqr._detect_outliers(factor)
    print(outliers)

    print("\n=== 测试完成 ===")
