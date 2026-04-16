"""
辅助函数模块

提供通用的辅助函数，包括数据处理、时间序列工具、验证工具等。
"""

import numpy as np
import pandas as pd
from typing import Tuple, Optional
from datetime import datetime
import warnings

# ========================================
# 数据处理工具
# ========================================


def remove_outliers(
    data: pd.Series,
    method: str = "quantile",
    n_std: float = 3.0,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    replace_with: str = "nan",
) -> pd.Series:
    """
    移除异常值

    Args:
        data: 输入数据
        method: 异常值检测方法
            - 'quantile': 基于分位数
            - 'std': 基于标准差
            - 'iqr': 基于四分位距
        n_std: 标准差倍数（用于 'std' 方法）
        lower_quantile: 下分位数（用于 'quantile' 方法）
        upper_quantile: 上分位数（用于 'quantile' 方法）
        replace_with: 替换方式
            - 'nan': 替换为 NaN
            - 'boundary': 替换为边界值
            - 'median': 替换为中位数

    Returns:
        处理后的数据

    Examples:
        >>> data = pd.Series([1, 2, 3, 100, 5])
        >>> remove_outliers(data, method='quantile')
    """
    data = data.copy()

    if method == "quantile":
        lower = data.quantile(lower_quantile)
        upper = data.quantile(upper_quantile)
        mask = (data < lower) | (data > upper)

    elif method == "std":
        mean = data.mean()
        std = data.std()
        lower = mean - n_std * std
        upper = mean + n_std * std
        mask = (data < lower) | (data > upper)

    elif method == "iqr":
        q1 = data.quantile(0.25)
        q3 = data.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        mask = (data < lower) | (data > upper)

    else:
        raise ValueError(f"未知的异常值检测方法: {method}")

    if replace_with == "nan":
        data[mask] = np.nan
    elif replace_with == "boundary":
        data[data < lower] = lower
        data[data > upper] = upper
    elif replace_with == "median":
        median = data.median()
        data[mask] = median
    else:
        raise ValueError(f"未知的替换方式: {replace_with}")

    return data


def normalize(data: pd.Series, method: str = "zscore", window: Optional[int] = None) -> pd.Series:
    """
    标准化数据

    Args:
        data: 输入数据
        method: 标准化方法
            - 'zscore': Z-score 标准化
            - 'minmax': Min-Max 标准化
            - 'rank': 排名标准化
            - 'rolling_zscore': 滚动 Z-score 标准化
        window: 滚动窗口大小（用于 'rolling_zscore' 方法）

    Returns:
        标准化后的数据

    Examples:
        >>> data = pd.Series([1, 2, 3, 4, 5])
        >>> normalize(data, method='zscore')
    """
    data = data.copy()

    if method == "zscore":
        return (data - data.mean()) / data.std()

    elif method == "minmax":
        return (data - data.min()) / (data.max() - data.min())

    elif method == "rank":
        return data.rank(pct=True)

    elif method == "rolling_zscore":
        if window is None:
            raise ValueError("rolling_zscore 方法需要指定 window 参数")
        rolling_mean = data.rolling(window, min_periods=1).mean()
        rolling_std = data.rolling(window, min_periods=1).std()
        return (data - rolling_mean) / rolling_std

    else:
        raise ValueError(f"未知的标准化方法: {method}")


def neutralize(
    factor: pd.DataFrame,
    industry: Optional[pd.DataFrame] = None,
    market_cap: Optional[pd.DataFrame] = None,
    method: str = "orthogonal",
) -> pd.DataFrame:
    """
    因子中性化

    Args:
        factor: 因子数据，MultiIndex (datetime, instrument)
        industry: 行业分类，MultiIndex (datetime, instrument)
        market_cap: 市值数据，MultiIndex (datetime, instrument)
        method: 中性化方法
            - 'orthogonal': 正交化（线性回归残差）
            - 'group': 分组标准化

    Returns:
        中性化后的因子数据

    Examples:
        >>> factor = pd.DataFrame(...)
        >>> industry = pd.DataFrame(...)
        >>> neutralized = neutralize(factor, industry=industry)
    """
    factor = factor.copy()

    if method == "orthogonal":
        # 对每个时间点进行中性化（使用 numpy lstsq 代替 sklearn 逐列拟合）
        neutralized_data = []
        for date in factor.index.get_level_values("datetime").unique():
            factor_date = factor.loc[date]
            neutralized_date = factor_date.copy()

            # 构建 X 矩阵（行业哑变量 + 可选的 log市值）
            X_parts = []
            align_index = factor_date.index

            if industry is not None:
                industry_date = industry.loc[date]
                industry_dummies = pd.get_dummies(industry_date)
                # 对齐索引，缺失行业填 0
                industry_dummies = industry_dummies.reindex(align_index, fill_value=0)
                X_parts.append(industry_dummies)

            if market_cap is not None:
                market_cap_date = market_cap.loc[date]
                log_mc = np.log(market_cap_date)
                log_mc = log_mc.reindex(align_index, fill_value=np.nan)
                X_parts.append(log_mc.to_frame("log_market_cap"))

            if X_parts:
                X_full = pd.concat(X_parts, axis=1)
                # 逐列向量化回归：用 numpy lstsq 代替 sklearn
                for col in factor_date.columns:
                    valid = factor_date[col].notna() & X_full.notna().all(axis=1)
                    n_valid = valid.sum()
                    if n_valid == 0:
                        continue
                    X_np = X_full.loc[valid].values.astype(np.float64)
                    # 添加截距列
                    X_np = np.column_stack([np.ones(n_valid), X_np])
                    y_np = factor_date[col].loc[valid].values.astype(np.float64)

                    # lstsq 求解 β = (X’X)^{-1} X’y
                    try:
                        beta, _, _, _ = np.linalg.lstsq(X_np, y_np, rcond=None)
                        residuals = y_np - X_np @ beta
                        neutralized_date.loc[valid, col] = residuals
                    except np.linalg.LinAlgError:
                        pass  # 奇异矩阵则跳过

            neutralized_data.append(neutralized_date)

        return pd.concat(neutralized_data)

    elif method == "group":
        # 分组标准化
        if industry is not None:
            # 按行业分组标准化
            result = []
            for date in factor.index.get_level_values("datetime").unique():
                factor_date = factor.loc[date].copy()
                industry_date = industry.loc[date].iloc[:, 0]  # 获取第一列

                for ind in industry_date.unique():
                    mask = industry_date == ind
                    if mask.sum() > 1:
                        factor_date.loc[mask, :] = normalize(
                            factor_date.loc[mask, :].iloc[:, 0], method="zscore"
                        ).to_frame()

                result.append(factor_date)

            return pd.concat(result) if result else factor
        else:
            return factor

    else:
        raise ValueError(f"未知的中性化方法: {method}")


# ========================================
# 时间序列工具
# ========================================


def calculate_returns(
    prices: pd.DataFrame, method: str = "simple", periods: int = 1, log_returns: bool = False
) -> pd.DataFrame:
    """
    计算收益率

    Args:
        prices: 价格数据，MultiIndex (datetime, instrument)
        method: 计算方法
            - 'simple': 简单收益率 (p_t / p_{t-1} - 1)
            - 'log': 对数收益率 (log(p_t / p_{t-1}))
        periods: 周期数
        log_returns: 是否使用对数收益率（已弃用，使用 method='log'）

    Returns:
        收益率数据

    Examples:
        >>> prices = pd.DataFrame(...)
        >>> returns = calculate_returns(prices, method='simple', periods=1)
    """
    if log_returns:
        warnings.warn("log_returns 参数已弃用，请使用 method='log'", DeprecationWarning)
        method = "log"

    if method == "simple":
        return prices.pct_change(periods=periods)
    elif method == "log":
        return np.log(prices / prices.shift(periods=periods))
    else:
        raise ValueError(f"未知的计算方法: {method}")


def calculate_forward_returns(prices: pd.DataFrame, forward_period: int = 1, method: str = "simple") -> pd.DataFrame:
    """
    计算未来收益率

    注意：这个函数仅用于标签计算，不能用于因子计算！

    Args:
        prices: 价格数据，MultiIndex (datetime, instrument)
        forward_period: 向前周期数
        method: 计算方法
            - 'simple': 简单收益率
            - 'log': 对数收益率

    Returns:
        未来收益率数据

    Examples:
        >>> prices = pd.DataFrame(...)
        >>> forward_returns = calculate_forward_returns(prices, forward_period=2)
    """
    if method == "simple":
        return prices.shift(-forward_period) / prices.shift(-(forward_period - 1)) - 1
    elif method == "log":
        return np.log(prices.shift(-forward_period) / prices.shift(-(forward_period - 1)))
    else:
        raise ValueError(f"未知的计算方法: {method}")


def resample_data(data: pd.DataFrame, freq: str, method: str = "last") -> pd.DataFrame:
    """
    重采样数据

    Args:
        data: 输入数据，MultiIndex (datetime, instrument)
        freq: 目标频率
            - 'D': 日
            - 'W': 周
            - 'M': 月
            - 'Q': 季度
        method: 聚合方法
            - 'last': 最后一个值
            - 'first': 第一个值
            - 'mean': 平均值
            - 'sum': 求和

    Returns:
        重采样后的数据

    Examples:
        >>> data = pd.DataFrame(...)
        >>> monthly_data = resample_data(data, freq='M', method='last')
    """
    if data.index.names != ["datetime", "instrument"]:
        raise ValueError("数据必须是 MultiIndex (datetime, instrument)")

    # 重置索引以便重采样
    data_reset = data.reset_index()

    # 按日期和工具分组
    grouped = data_reset.groupby("instrument")

    # 对每个工具进行重采样
    resampled_list = []
    for instrument, group in grouped:
        group = group.set_index("datetime").sort_index()

        if method == "last":
            resampled = group.resample(freq).last()
        elif method == "first":
            resampled = group.resample(freq).first()
        elif method == "mean":
            resampled = group.resample(freq).mean()
        elif method == "sum":
            resampled = group.resample(freq).sum()
        else:
            raise ValueError(f"未知的聚合方法: {method}")

        resampled["instrument"] = instrument
        resampled_list.append(resampled)

    # 合并结果
    result = pd.concat(resampled_list)
    result = result.reset_index().set_index(["datetime", "instrument"])

    # 移除全为 NaN 的行
    result = result.dropna(how="all")

    return result


# ========================================
# 验证工具
# ========================================


def validate_data_format(data: pd.DataFrame, name: str = "数据") -> None:
    """
    验证数据格式

    Args:
        data: 输入数据
        name: 数据名称（用于错误消息）

    Raises:
        ValueError: 当数据格式不正确时

    Examples:
        >>> validate_data_format(data, "因子数据")
    """
    if not isinstance(data, pd.DataFrame):
        raise ValueError(f"{name} 必须是 pandas.DataFrame")

    if data.index.names != ["datetime", "instrument"]:
        raise ValueError(f"{name} 的索引必须是 MultiIndex (datetime, instrument), " f"当前是: {data.index.names}")

    if data.empty:
        raise ValueError(f"{name} 不能为空")


def check_missing_values(data: pd.DataFrame, threshold: float = 0.5) -> dict:
    """
    检查缺失值

    Args:
        data: 输入数据
        threshold: 缺失率阈值（超过此阈值会警告）

    Returns:
        缺失值统计信息

    Examples:
        >>> stats = check_missing_values(data)
    """
    missing_count = data.isnull().sum()
    missing_rate = missing_count / len(data)

    stats = {
        "total_count": len(data),
        "missing_count": missing_count.to_dict(),
        "missing_rate": missing_rate.to_dict(),
        "high_missing_columns": missing_rate[missing_rate > threshold].index.tolist(),
    }

    if stats["high_missing_columns"]:
        warnings.warn(f"以下列的缺失率超过 {threshold:.0%}: {stats['high_missing_columns']}")

    return stats


def check_infinite_values(data: pd.DataFrame) -> dict:
    """
    检查无穷值

    Args:
        data: 输入数据

    Returns:
        无穷值统计信息

    Examples:
        >>> stats = check_infinite_values(data)
    """
    pos_inf = (data == np.inf).sum()
    neg_inf = (data == -np.inf).sum()

    stats = {
        "positive_infinite": pos_inf.to_dict(),
        "negative_infinite": neg_inf.to_dict(),
        "total_infinite": (pos_inf + neg_inf).to_dict(),
    }

    if stats["total_infinite"]:
        warnings.warn(f"检测到无穷值: {stats['total_infinite']}")

    return stats


# ========================================
# 性能计算工具
# ========================================


def calculate_ic(pred: pd.DataFrame, label: pd.DataFrame, method: str = "pearson") -> Tuple[pd.Series, pd.Series]:
    """
    计算 IC (Information Coefficient) 和 Rank IC

    Args:
        pred: 预测值（因子值），MultiIndex (datetime, instrument)
        label: 真实值，MultiIndex (datetime, instrument)
        method: 相关系数计算方法
            - 'pearson': 皮尔逊相关系数
            - 'spearman': 斯皮尔曼相关系数

    Returns:
        (IC, Rank IC)

    Examples:
        >>> ic, rank_ic = calculate_ic(factor_df, return_df)
    """
    # 确保索引对齐
    common_index = pred.index.intersection(label.index)
    pred_aligned = pred.loc[common_index]
    label_aligned = label.loc[common_index]

    # 按日期分组计算相关系数
    ic_values = []
    rank_ic_values = []

    for date in pred_aligned.index.get_level_values("datetime").unique():
        pred_date = pred_aligned.loc[date].iloc[:, 0]
        label_date = label_aligned.loc[date].iloc[:, 0]

        # 移除 NaN
        valid_mask = pred_date.notna() & label_date.notna()
        pred_valid = pred_date[valid_mask]
        label_valid = label_date[valid_mask]

        if len(pred_valid) < 2:
            ic_values.append(np.nan)
            rank_ic_values.append(np.nan)
            continue

        # 计算 IC
        if method == "pearson":
            ic = pred_valid.corr(label_valid)
        elif method == "spearman":
            ic = pred_valid.corr(label_valid, method="spearman")
        else:
            raise ValueError(f"未知的相关系数计算方法: {method}")

        # 计算 Rank IC (使用斯皮尔曼相关系数)
        rank_ic = pred_valid.corr(label_valid, method="spearman")

        ic_values.append(ic)
        rank_ic_values.append(rank_ic)

    ic_series = pd.Series(ic_values, index=pred_aligned.index.get_level_values("datetime").unique())
    rank_ic_series = pd.Series(rank_ic_values, index=pred_aligned.index.get_level_values("datetime").unique())

    return ic_series, rank_ic_series


def calculate_long_short_return(
    pred: pd.DataFrame, label: pd.DataFrame, quantile: float = 0.2
) -> Tuple[pd.Series, pd.Series]:
    """
    计算多空收益

    Args:
        pred: 预测值（因子值），MultiIndex (datetime, instrument)
        label: 真实值（收益率），MultiIndex (datetime, instrument)
        quantile: 分位数（默认 0.2，即 top 20% 和 bottom 20%）

    Returns:
        (多空收益, 平均收益)

    Examples:
        >>> ls_return, avg_return = calculate_long_short_return(factor_df, return_df)
    """
    # 确保索引对齐
    common_index = pred.index.intersection(label.index)
    pred_aligned = pred.loc[common_index]
    label_aligned = label.loc[common_index]

    long_short_returns = []
    average_returns = []

    for date in pred_aligned.index.get_level_values("datetime").unique():
        pred_date = pred_aligned.loc[date].iloc[:, 0]
        label_date = label_aligned.loc[date].iloc[:, 0]

        # 移除 NaN
        valid_mask = pred_date.notna() & label_date.notna()
        pred_valid = pred_date[valid_mask]
        label_valid = label_date[valid_mask]

        if len(pred_valid) < 10:
            long_short_returns.append(np.nan)
            average_returns.append(np.nan)
            continue

        # 计算分位数
        upper_threshold = pred_valid.quantile(1 - quantile)
        lower_threshold = pred_valid.quantile(quantile)

        # 多头组合
        long_mask = pred_valid >= upper_threshold
        long_return = label_valid[long_mask].mean()

        # 空头组合
        short_mask = pred_valid <= lower_threshold
        short_return = label_valid[short_mask].mean()

        # 多空收益
        ls_return = long_return - short_return
        long_short_returns.append(ls_return)

        # 平均收益
        average_returns.append(label_valid.mean())

    ls_series = pd.Series(long_short_returns, index=pred_aligned.index.get_level_values("datetime").unique())
    avg_series = pd.Series(average_returns, index=pred_aligned.index.get_level_values("datetime").unique())

    return ls_series, avg_series


# ========================================
# 其他工具
# ========================================


def align_data(data1: pd.DataFrame, data2: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    对齐两个数据集的索引

    Args:
        data1: 第一个数据集
        data2: 第二个数据集

    Returns:
        对齐后的 (data1, data2)

    Examples:
        >>> factor_aligned, return_aligned = align_data(factor_df, return_df)
    """
    common_index = data1.index.intersection(data2.index)
    return data1.loc[common_index], data2.loc[common_index]


def split_data(
    data: pd.DataFrame, train_ratio: float = 0.7, val_ratio: float = 0.15, test_ratio: float = 0.15
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    分割数据集（按时间）

    Args:
        data: 输入数据，MultiIndex (datetime, instrument)
        train_ratio: 训练集比例
        val_ratio: 验证集比例
        test_ratio: 测试集比例

    Returns:
        (训练集, 验证集, 测试集)

    Examples:
        >>> train, val, test = split_data(data)
    """
    if abs(train_ratio + val_ratio + test_ratio - 1.0) > 1e-6:
        raise ValueError("训练集、验证集、测试集比例之和必须为 1")

    # 获取所有唯一日期
    dates = sorted(data.index.get_level_values("datetime").unique())
    n_dates = len(dates)

    # 计算分割点
    train_end = int(n_dates * train_ratio)
    val_end = int(n_dates * (train_ratio + val_ratio))

    # 分割日期
    train_dates = dates[:train_end]
    val_dates = dates[train_end:val_end]
    test_dates = dates[val_end:]

    # 分割数据
    train_data = data.loc[train_dates]
    val_data = data.loc[val_dates]
    test_data = data.loc[test_dates]

    return train_data, val_data, test_data


def save_results(results: dict, filepath: str, format: str = "json") -> None:
    """
    保存结果到文件

    Args:
        results: 结果字典
        filepath: 文件路径
        format: 文件格式
            - 'json': JSON 格式
            - 'csv': CSV 格式（仅适用于部分结果）
            - 'pickle': Python pickle 格式

    Examples:
        >>> save_results(results, 'results.json')
    """
    import pickle
    from pathlib import Path

    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if format == "json":
        import json

        # 转换 numpy 类型为 Python 类型

        def convert(obj):
            if isinstance(obj, (np.integer, np.floating)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, pd.Series):
                return obj.to_dict()
            elif isinstance(obj, pd.DataFrame):
                return obj.to_dict()
            elif isinstance(obj, (pd.Timestamp, datetime)):
                return obj.isoformat()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert(item) for item in obj]
            else:
                return obj

        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(convert(results), f, indent=2, ensure_ascii=False)

    elif format == "pickle":
        with open(filepath, "wb") as f:
            pickle.dump(results, f)

    else:
        raise ValueError(f"不支持的文件格式: {format}")


# 测试用例
if __name__ == "__main__":
    print("辅助函数模块测试")
    print("=" * 80)

    # 测试数据处理工具
    print("\n1. 测试数据标准化")
    data = pd.Series([1, 2, 3, 4, 5, 100])
    print(f"原始数据: {data.tolist()}")
    print(f"Z-score 标准化: {normalize(data, method='zscore').tolist()}")
    print(f"Min-Max 标准化: {normalize(data, method='minmax').tolist()}")

    # 测试异常值移除
    print("\n2. 测试异常值移除")
    print(f"移除异常值（quantile）: {remove_outliers(data, method='quantile').tolist()}")

    # 测试收益率计算
    print("\n3. 测试收益率计算")
    prices = pd.DataFrame({"A": [100, 102, 101, 103, 105], "B": [50, 51, 52, 51, 53]})
    print(f"价格数据:\n{prices}")
    print(f"简单收益率:\n{calculate_returns(prices, method='simple')}")

    # 测试数据验证
    print("\n4. 测试数据验证")
    try:
        validate_data_format(prices, "测试数据")
    except ValueError as e:
        print(f"验证失败（预期）: {e}")

    print("\n所有测试完成！")
