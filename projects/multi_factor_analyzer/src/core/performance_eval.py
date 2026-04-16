"""
性能评估引擎模块

本模块提供因子性能评估功能，计算 IC、IR、ICIR、多空收益等关键指标。

主要功能：
1. IC 和 Rank IC 计算 - 评估因子预测能力
2. ICIR 和 Rank ICIR 计算 - 评估因子稳定性
3. 多空收益计算 - 评估实际交易效果
4. 风险指标计算 - 夏普比率、最大回撤等
5. 支持对数收益率选项

依赖：
- qlib.contrib.eva.alpha.calc_ic
- qlib.contrib.eva.alpha.calc_long_short_return
"""

from typing import Dict, Tuple, Union

import numpy as np
import pandas as pd
from qlib.contrib.eva.alpha import calc_ic, calc_long_short_return


class PerformanceEvaluator:
    """
    性能评估器

    评估因子的预测能力和实际交易效果，提供全面的性能指标分析。

    核心指标：
    1. IC (Information Coefficient): 信息系数，衡量因子值与未来收益率的相关性
    2. Rank IC: 排名 IC，衡量因子排名与未来收益率排名的相关性
    3. ICIR: IC 信息比率，衡量 IC 的稳定性（IC 均值 / IC 标准差）
    4. 多空收益: 基于因子分组的交易策略收益
    5. 夏普比率: 风险调整后的收益指标
    6. 最大回撤: 最大损失幅度

    Attributes:
        use_log_return: 是否使用对数收益率
        date_col: 日期列名

    Examples:
        >>> evaluator = PerformanceEvaluator(use_log_return=False)
        >>>
        >>> # 准备数据
        >>> pred = factor_df  # 因子值
        >>> label = return_df  # 未来收益率
        >>>
        >>> # 计算所有指标
        >>> metrics = evaluator.calculate_all(pred, label)
        >>> print(f"IC 均值: {metrics['ic_mean']:.4f}")
        >>> print(f"ICIR: {metrics['icir']:.4f}")
        >>> print(f"年化收益: {metrics['annual_return']:.2%}")
    """

    def __init__(self, use_log_return: bool = False, date_col: str = "datetime"):
        """
        初始化性能评估器

        Args:
            use_log_return: 是否使用对数收益率
                           - False: 使用简单收益率 (P_t / P_{t-1} - 1)
                           - True: 使用对数收益率 (log(P_t / P_{t-1}))
            date_col: 日期列名，默认为 'datetime'

        Examples:
            >>> # 使用简单收益率（默认）
            >>> evaluator = PerformanceEvaluator()
            >>>
            >>> # 使用对数收益率
            >>> evaluator = PerformanceEvaluator(use_log_return=True)
        """
        self.use_log_return = use_log_return
        self.date_col = date_col

    def calculate_all(
        self, pred: pd.DataFrame, label: pd.DataFrame, quantile: float = 0.2
    ) -> Dict[str, Union[float, pd.Series]]:
        """
        计算所有性能指标

        这是最常用的方法，一次性计算所有关键指标。

        Args:
            pred: 因子值 DataFrame，索引为 (datetime, instrument)
            label: 未来收益率 DataFrame，索引为 (datetime, instrument)
            quantile: 多空分组分位数，默认为 0.2（前20%和后20%）

        Returns:
            Dict: 包含所有性能指标的字典
                - ic: 每日 IC 序列
                - rank_ic: 每日 Rank IC 序列
                - ic_mean: IC 均值
                - ic_std: IC 标准差
                - icir: IC 信息比率
                - rank_ic_mean: Rank IC 均值
                - rank_ic_std: Rank IC 标准差
                - rank_icir: Rank IC 信息比率
                - long_short_return: 每日多空收益序列
                - annual_return: 年化收益率
                - sharpe_ratio: 夏普比率
                - max_drawdown: 最大回撤
                - win_rate: 胜率

        Examples:
            >>> evaluator = PerformanceEvaluator()
            >>> metrics = evaluator.calculate_all(pred, label)
            >>>
            >>> print(f"IC 均值: {metrics['ic_mean']:.4f}")
            >>> print(f"ICIR: {metrics['icir']:.4f}")
            >>> print(f"年化收益: {metrics['annual_return']:.2%}")
            >>> print(f"夏普比率: {metrics['sharpe_ratio']:.2f}")
        """
        # 计算 IC 和 Rank IC
        ic, rank_ic = self.calculate_ic(pred, label)

        # 计算多空收益
        long_short_return, _ = self.calculate_long_short_return(pred, label, quantile=quantile)

        # 计算 ICIR 和 Rank ICIR
        icir = self.calculate_icir(ic)
        rank_icir = self.calculate_icir(rank_ic)

        # 计算风险指标
        annual_return = self.calculate_annual_return(long_short_return)
        sharpe_ratio = self.calculate_sharpe_ratio(long_short_return)
        max_drawdown = self.calculate_max_drawdown(long_short_return)
        win_rate = self.calculate_win_rate(long_short_return)

        return {
            "ic": ic,
            "rank_ic": rank_ic,
            "ic_mean": ic.mean(),
            "ic_std": ic.std(),
            "icir": icir,
            "rank_ic_mean": rank_ic.mean(),
            "rank_ic_std": rank_ic.std(),
            "rank_icir": rank_icir,
            "long_short_return": long_short_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
        }

    def calculate_ic(self, pred: pd.DataFrame, label: pd.DataFrame) -> Tuple[pd.Series, pd.Series]:
        """
        计算 IC 和 Rank IC

        IC (Information Coefficient) 是因子值与未来收益率的相关系数。
        Rank IC 是因子排名与未来收益率排名的相关系数（Spearman 相关）。

        Args:
            pred: 因子值 DataFrame，索引为 (datetime, instrument)
            label: 未来收益率 DataFrame，索引为 (datetime, instrument)

        Returns:
            Tuple[pd.Series, pd.Series]: (IC 序列, Rank IC 序列)

        Examples:
            >>> ic, rank_ic = evaluator.calculate_ic(pred, label)
            >>> print(f"IC 均值: {ic.mean():.4f}")
            >>> print(f"IC 标准差: {ic.std():.4f}")
        """
        # 使用 qlib 的 calc_ic 函数
        ic, rank_ic = calc_ic(pred=pred, label=label, date_col=self.date_col, dropna=False)

        return ic, rank_ic

    def calculate_icir(self, ic: pd.Series) -> float:
        """
        计算 ICIR (IC Information Ratio)

        ICIR = IC 均值 / IC 标准差

        ICIR 衡量 IC 的稳定性，值越大表示因子预测能力越稳定。
        通常认为 ICIR > 0.5 表示因子稳定性较好。

        Args:
            ic: IC 序列

        Returns:
            float: ICIR 值

        Examples:
            >>> icir = evaluator.calculate_icir(ic_series)
            >>> print(f"ICIR: {icir:.4f}")
        """
        if ic.empty or len(ic) < 2:
            return 0.0

        ic_mean = ic.mean()
        ic_std = ic.std()

        if ic_std == 0:
            return 0.0

        return ic_mean / ic_std

    def calculate_long_short_return(
        self, pred: pd.DataFrame, label: pd.DataFrame, quantile: float = 0.2
    ) -> Tuple[pd.Series, pd.Series]:
        """
        计算多空收益

        基于因子值分组，做多因子值最高的组，做空因子值最低的组。

        Args:
            pred: 因子值 DataFrame，索引为 (datetime, instrument)
            label: 未来收益率 DataFrame，索引为 (datetime, instrument)
            quantile: 分组分位数，默认为 0.2（前20%和后20%）

        Returns:
            Tuple[pd.Series, pd.Series]:
                - 多空收益序列（每日收益）
                - 平均收益序列

        Examples:
            >>> long_short, avg_return = evaluator.calculate_long_short_return(
            ...     pred, label, quantile=0.2
            ... )
            >>> print(f"多空收益均值: {long_short.mean():.4f}")
        """
        # 使用 qlib 的 calc_long_short_return 函数
        long_short_return, avg_return = calc_long_short_return(
            pred=pred, label=label, date_col=self.date_col, quantile=quantile, dropna=False
        )

        return long_short_return, avg_return

    def calculate_annual_return(self, returns: pd.Series, periods_per_year: int = 252) -> float:
        """
        计算年化收益率

        Args:
            returns: 收益率序列
            periods_per_year: 每年周期数，默认为 252（交易日）

        Returns:
            float: 年化收益率

        Examples:
            >>> annual_ret = evaluator.calculate_annual_return(daily_returns)
            >>> print(f"年化收益率: {annual_ret:.2%}")
        """
        if len(returns) == 0:
            return 0.0

        # 计算累计收益
        total_return = (1 + returns).prod() - 1

        # 计算年数
        years = len(returns) / periods_per_year

        if years == 0:
            return 0.0

        # 年化收益率
        annual_return = (1 + total_return) ** (1 / years) - 1

        return annual_return

    def calculate_sharpe_ratio(
        self, returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252
    ) -> float:
        """
        计算夏普比率

        Sharpe Ratio = (收益率 - 无风险收益率) / 收益率标准差

        Args:
            returns: 收益率序列
            risk_free_rate: 无风险收益率（年化），默认为 0
            periods_per_year: 每年周期数，默认为 252

        Returns:
            float: 夏普比率

        Examples:
            >>> sharpe = evaluator.calculate_sharpe_ratio(daily_returns)
            >>> print(f"夏普比率: {sharpe:.2f}")
        """
        if len(returns) == 0:
            return 0.0

        # 计算超额收益
        excess_returns = returns - risk_free_rate / periods_per_year

        # 计算均值和标准差
        mean = excess_returns.mean()
        std = excess_returns.std()

        if std == 0:
            return 0.0

        # 年化夏普比率
        sharpe = mean / std * np.sqrt(periods_per_year)

        return sharpe

    def calculate_max_drawdown(self, returns: pd.Series) -> float:
        """
        计算最大回撤

        最大回撤是从峰值到谷值的最大跌幅。

        Args:
            returns: 收益率序列

        Returns:
            float: 最大回撤（正数，如 0.15 表示 15% 的回撤）

        Examples:
            >>> max_dd = evaluator.calculate_max_drawdown(daily_returns)
            >>> print(f"最大回撤: {max_dd:.2%}")
        """
        if len(returns) == 0:
            return 0.0

        # 计算累计收益
        cumulative = (1 + returns).cumprod()

        # 计算历史最高点
        running_max = cumulative.expanding().max()

        # 计算回撤
        drawdown = (cumulative - running_max) / running_max

        # 最大回撤
        max_drawdown = abs(drawdown.min())

        return max_drawdown

    def calculate_win_rate(self, returns: pd.Series) -> float:
        """
        计算胜率

        胜率 = 正收益天数 / 总天数

        Args:
            returns: 收益率序列

        Returns:
            float: 胜率（0-1 之间）

        Examples:
            >>> win_rate = evaluator.calculate_win_rate(daily_returns)
            >>> print(f"胜率: {win_rate:.2%}")
        """
        if len(returns) == 0:
            return 0.0

        win_days = (returns > 0).sum()
        total_days = len(returns)

        return win_days / total_days

    def calculate_log_return(self, price_df: pd.DataFrame, shift: int = 2) -> pd.DataFrame:
        """
        计算对数收益率

        对数收益率具有可加性，适合长期持有策略或高波动资产。

        log_return = log(P_{t+shift} / P_t)

        Args:
            price_df: 价格 DataFrame，索引为 (datetime, instrument)
            shift: 向前移动的周期数，默认为 2（T+1 到 T+2）

        Returns:
            pd.DataFrame: 对数收益率 DataFrame

        Examples:
            >>> log_ret = evaluator.calculate_log_return(price_df, shift=2)
            >>> print(log_ret.head())
        """
        return np.log(price_df.shift(-shift) / price_df.shift(-(shift - 1)))

    def calculate_simple_return(self, price_df: pd.DataFrame, shift: int = 2) -> pd.DataFrame:
        """
        计算简单收益率

        simple_return = P_{t+shift} / P_t - 1

        Args:
            price_df: 价格 DataFrame，索引为 (datetime, instrument)
            shift: 向前移动的周期数，默认为 2（T+1 到 T+2）

        Returns:
            pd.DataFrame: 简单收益率 DataFrame

        Examples:
            >>> simple_ret = evaluator.calculate_simple_return(price_df, shift=2)
            >>> print(simple_ret.head())
        """
        return price_df.shift(-shift) / price_df.shift(-(shift - 1)) - 1

    def generate_report(self, metrics: Dict[str, Union[float, pd.Series]], factor_name: str = "Factor") -> str:
        """
        生成性能评估报告

        Args:
            metrics: calculate_all 返回的指标字典
            factor_name: 因子名称

        Returns:
            str: 格式化的报告文本

        Examples:
            >>> report = evaluator.generate_report(metrics, "MA20")
            >>> print(report)
        """
        report = f"""
{'=' * 60}
因子性能评估报告: {factor_name}
{'=' * 60}

1. 预测能力指标
{'-' * 60}
   IC 均值:     {metrics['ic_mean']:>10.4f}
   IC 标准差:   {metrics['ic_std']:>10.4f}
   ICIR:        {metrics['icir']:>10.4f}
   Rank IC 均值: {metrics['rank_ic_mean']:>10.4f}
   Rank IC 标准差: {metrics['rank_ic_std']:>10.4f}
   Rank ICIR:    {metrics['rank_icir']:>10.4f}

2. 交易效果指标
{'-' * 60}
   年化收益率:   {metrics['annual_return']:>10.2%}
   夏普比率:    {metrics['sharpe_ratio']:>10.2f}
   最大回撤:    {metrics['max_drawdown']:>10.2%}
   胜率:       {metrics['win_rate']:>10.2%}

3. 综合评估
{'-' * 60}
"""

        # 综合评估
        ic_abs = abs(metrics["ic_mean"])
        icir_val = metrics["icir"]
        annual_ret = metrics["annual_return"]
        sharpe = metrics["sharpe_ratio"]  # noqa: F841
        max_dd = metrics["max_drawdown"]  # noqa: F841

        if ic_abs > 0.05 and icir_val > 0.5 and annual_ret > 0.05:
            grade = "A (优秀)"
            comment = "因子表现优秀，预测能力强且稳定性好，建议重点使用。"
        elif ic_abs > 0.03 and icir_val > 0.3 and annual_ret > 0.02:
            grade = "B (良好)"
            comment = "因子表现良好，具有一定的预测能力和稳定性。"
        elif ic_abs > 0.02 and icir_val > 0.2:
            grade = "C (一般)"
            comment = "因子表现一般，建议谨慎使用或考虑优化。"
        else:
            grade = "D (较差)"
            comment = "因子表现较差，不建议使用，建议重新设计。"

        report += f"   可靠性等级: {grade}\n"
        report += f"   评价: {comment}\n"
        report += f"{'=' * 60}\n"

        return report


# 便捷函数
def evaluate_factor_performance(
    pred: pd.DataFrame, label: pd.DataFrame, use_log_return: bool = False, quantile: float = 0.2
) -> Dict:
    """
    评估因子性能的便捷函数

    Args:
        pred: 因子值 DataFrame
        label: 未来收益率 DataFrame
        use_log_return: 是否使用对数收益率
        quantile: 多空分组分位数

    Returns:
        Dict: 性能指标字典

    Examples:
        >>> metrics = evaluate_factor_performance(pred, label)
        >>> print(f"IC 均值: {metrics['ic_mean']:.4f}")
    """
    evaluator = PerformanceEvaluator(use_log_return=use_log_return)
    return evaluator.calculate_all(pred, label, quantile=quantile)


if __name__ == "__main__":
    # 示例：使用性能评估器
    print("=" * 60)
    print("性能评估器示例")
    print("=" * 60)

    # 创建模拟数据
    print("\n生成模拟数据...")
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    instruments = ["SH600000", "SH600001", "SH600002"]

    # 创建多索引
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

    # 生成模拟因子值（正态分布）
    pred = pd.Series(np.random.randn(len(index)), index=index)
    pred.name = "factor"

    # 生成模拟收益率（与因子有一定相关性）
    label = pd.Series(pred.values * 0.05 + np.random.randn(len(index)) * 0.1, index=index)
    label.name = "return"

    print(f"因子值: {len(pred)} 个观测值")
    print(f"收益率: {len(label)} 个观测值")

    # 创建评估器
    evaluator = PerformanceEvaluator(use_log_return=False)

    # 计算所有指标
    print("\n计算性能指标...")
    metrics = evaluator.calculate_all(pred, label)

    # 打印关键指标
    print("\n关键指标:")
    print(f"  IC 均值:     {metrics['ic_mean']:.4f}")
    print(f"  ICIR:        {metrics['icir']:.4f}")
    print(f"  年化收益率:   {metrics['annual_return']:.2%}")
    print(f"  夏普比率:    {metrics['sharpe_ratio']:.2f}")
    print(f"  最大回撤:    {metrics['max_drawdown']:.2%}")
    print(f"  胜率:       {metrics['win_rate']:.2%}")

    # 生成报告
    print("\n完整报告:")
    print(evaluator.generate_report(metrics, "模拟因子"))

    # 演示对数收益率
    print("\n对数收益率示例:")
    price_data = pd.DataFrame({"close": 100 + np.random.randn(len(index)).cumsum()}, index=index)
    log_return = evaluator.calculate_log_return(price_data, shift=2)
    print(f"对数收益率:\n{log_return.head()}")
