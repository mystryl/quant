"""
策略场景分析器 (Strategy Analyzer)

该模块实现了因子在不同策略场景下的性能分析：
1. 看涨策略（做多因子值高的股票）
2. 看跌策略（做多因子值低的股票或做空）
3. 波动率策略（根据市场波动率调整仓位）
4. 其他场景（牛熊市、行业轮动、市值分组）

设计理念：
- 评估因子在不同市场环境下的表现
- 提供多维度的策略分析
- 支持灵活的分组和配置
"""

import numpy as np
import pandas as pd
from typing import Dict, Optional


class StrategyAnalyzer:
    """
    策略场景分析器

    用于分析因子在不同策略场景下的表现，包括：

    1. **看涨策略 (Bull Strategy)**
       - 选取因子值最高的股票做多
       - 评估标准：正向 IC、正向多空收益

    2. **看跌策略 (Bear Strategy)**
       - 选取因子值最低的股票做多（反向策略）
       - 评估标准：负向 IC、负向多空收益

    3. **波动率策略 (Volatility Strategy)**
       - 根据市场波动率调整仓位
       - 高波动时降低仓位，低波动时提高仓位

    4. **其他场景**
       - 牛熊市分组
       - 行业轮动
       - 市值分组

    Examples:
        >>> from core.strategy_analyzer import StrategyAnalyzer
        >>>
        >>> # 创建分析器
        >>> analyzer = StrategyAnalyzer()
        >>>
        >>> # 分析看涨策略
        >>> bull_result = analyzer.analyze_bull_strategy(
        ...     factor_df, returns_df, top_pct=0.2
        ... )
        >>>
        >>> # 分析看跌策略
        >>> bear_result = analyzer.analyze_bear_strategy(
        ...     factor_df, returns_df, bottom_pct=0.2
        ... )
        >>>
        >>> # 分析波动率策略
        >>> vol_result = analyzer.analyze_volatility_strategy(
        ...     factor_df, returns_df, price_df
        ... )
        >>>
        >>> # 分析所有场景
        >>> all_results = analyzer.analyze_all_scenarios(
        ...     factor_df, returns_df, price_df
        ... )
    """

    def __init__(self, annualization_factor: int = 252):
        """
        初始化策略分析器

        Args:
            annualization_factor: 年化因子（默认 252 个交易日）
        """
        self.annualization_factor = annualization_factor

    def analyze_bull_strategy(
        self, factor_df: pd.DataFrame, returns_df: pd.DataFrame, top_pct: float = 0.2, quantile: Optional[int] = None
    ) -> Dict[str, any]:
        """
        分析看涨策略

        选取因子值最高的 top_pct 比例股票做多，计算策略收益

        Args:
            factor_df: 因子数据 (MultiIndex: datetime, instrument)
            returns_df: 收益率数据 (MultiIndex: datetime, instrument)
            top_pct: 选取比例 (0-1)，如 0.2 表示选取前 20%
            quantile: 分组数量（如果指定，则按分位数分组）

        Returns:
            包含策略收益和指标的字典
                - strategy_returns: 每日策略收益序列
                - total_return: 总收益率
                - annual_return: 年化收益率
                - sharpe_ratio: 夏普比率
                - max_drawdown: 最大回撤
                - win_rate: 胜率
                - calmar_ratio: 卡玛比率
        """
        # 确保索引对齐
        common_index = factor_df.index.intersection(returns_df.index)
        factor_aligned = factor_df.loc[common_index]
        returns_aligned = returns_df.loc[common_index]

        # 选择因子值最高的股票
        selected = self._select_top_factor(
            factor_aligned, top_pct=top_pct, quantile=quantile, ascending=False  # 降序，选最大的
        )

        # 计算策略收益
        strategy_returns = self._calculate_strategy_returns(selected, returns_aligned)

        # 计算策略指标
        metrics = self._calculate_strategy_metrics(strategy_returns)

        return {"strategy_returns": strategy_returns, "selection": selected, **metrics}

    def analyze_bear_strategy(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        top_pct: float = 0.2,
        quantile: Optional[int] = None,
        bottom_pct: Optional[float] = None,
    ) -> Dict[str, any]:
        """
        分析看跌策略（反向策略）

        选取因子值最低的 bottom_pct 比例股票做多，计算策略收益

        Args:
            factor_df: 因子数据
            returns_df: 收益率数据
            top_pct: 选取比例（为了与看涨策略保持一致，使用 top_pct）
            quantile: 分组数量（如果指定，则按分位数分组）
            bottom_pct: 别名，如果指定则使用此参数

        Returns:
            包含策略收益和指标的字典
        """
        # 兼容两种参数名
        if bottom_pct is not None:
            selection_pct = bottom_pct
        else:
            selection_pct = top_pct

        # 确保索引对齐
        common_index = factor_df.index.intersection(returns_df.index)
        factor_aligned = factor_df.loc[common_index]
        returns_aligned = returns_df.loc[common_index]

        # 选择因子值最低的股票
        selected = self._select_top_factor(
            factor_aligned, top_pct=selection_pct, quantile=quantile, ascending=True  # 升序，选最小的
        )

        # 计算策略收益
        strategy_returns = self._calculate_strategy_returns(selected, returns_aligned)

        # 计算策略指标
        metrics = self._calculate_strategy_metrics(strategy_returns)

        return {"strategy_returns": strategy_returns, "selection": selected, **metrics}

    def analyze_long_short_strategy(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        top_pct: float = 0.2,
        bottom_pct: Optional[float] = None,
        quantile: Optional[int] = None,
    ) -> Dict[str, any]:
        """
        分析多空策略

        买入因子值最高的股票，卖出（做空）因子值最低的股票

        Args:
            factor_df: 因子数据
            returns_df: 收益率数据
            top_pct: 多头选取比例
            bottom_pct: 空头选取比例（默认等于 top_pct）
            quantile: 分组数量

        Returns:
            包含策略收益和指标的字典
        """
        if bottom_pct is None:
            bottom_pct = top_pct

        # 确保索引对齐
        common_index = factor_df.index.intersection(returns_df.index)
        factor_aligned = factor_df.loc[common_index]
        returns_aligned = returns_df.loc[common_index]

        # 选择多头和空头
        long_selected = self._select_top_factor(factor_aligned, top_pct=top_pct, quantile=quantile, ascending=False)
        short_selected = self._select_top_factor(factor_aligned, top_pct=bottom_pct, quantile=quantile, ascending=True)

        # 计算多空收益
        long_returns = self._calculate_strategy_returns(long_selected, returns_aligned)
        short_returns = self._calculate_strategy_returns(short_selected, returns_aligned)

        # 多空收益 = 多头收益 - 空头收益
        strategy_returns = long_returns - short_returns

        # 计算策略指标
        metrics = self._calculate_strategy_metrics(strategy_returns)

        return {
            "strategy_returns": strategy_returns,
            "long_returns": long_returns,
            "short_returns": short_returns,
            "long_selection": long_selected,
            "short_selection": short_selected,
            **metrics,
        }

    def analyze_volatility_strategy(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        price_df: pd.DataFrame,
        top_pct: float = 0.2,
        vol_window: int = 20,
        position_method: str = "inverse",
    ) -> Dict[str, any]:
        """
        分析波动率策略

        根据市场波动率调整仓位：
        - 高波动时降低仓位
        - 低波动时提高仓位

        Args:
            factor_df: 因子数据
            returns_df: 收益率数据
            price_df: 价格数据（用于计算市场波动率）
            top_pct: 选股比例
            vol_window: 波动率计算窗口
            position_method: 仓位调整方法
                - 'inverse': 反向（波动率越高，仓位越低）
                - 'threshold': 阈值（高于阈值降低仓位）

        Returns:
            包含策略收益和指标的字典
        """
        # 计算基础看涨策略
        bull_result = self.analyze_bull_strategy(factor_df, returns_df, top_pct=top_pct)
        base_returns = bull_result["strategy_returns"]

        # 计算市场波动率
        market_vol = self._calculate_market_volatility(returns_df, window=vol_window)

        # 计算动态仓位
        if position_method == "inverse":
            # 反向关系：波动率越高，仓位越低
            position = 1 / (1 + market_vol * 10)  # 乘以 10 是为了放大效果
            position = position.clip(0.3, 1.0)  # 限制在 [0.3, 1.0]
        elif position_method == "threshold":
            # 阈值方法：高于阈值降低仓位
            vol_median = market_vol.median()
            position = np.where(market_vol > vol_median, 0.5, 1.0)
            position = pd.Series(position, index=market_vol.index)
        else:
            raise ValueError(f"不支持的仓位调整方法: {position_method}")

        # 对齐索引
        common_index = base_returns.index.intersection(position.index)
        aligned_returns = base_returns.loc[common_index]
        aligned_position = position.loc[common_index]

        # 应用动态仓位
        adjusted_returns = aligned_returns * aligned_position

        # 计算策略指标
        metrics = self._calculate_strategy_metrics(adjusted_returns)

        return {
            "strategy_returns": adjusted_returns,
            "base_returns": base_returns,
            "position": position,
            "market_volatility": market_vol,
            **metrics,
        }

    def analyze_market_regime(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        market_index: pd.Series,
        bull_threshold: float = 0.0,
        window: int = 20,
    ) -> Dict[str, any]:
        """
        分析牛熊市场景

        根据市场趋势分组，分析因子在牛市和熊市中的表现

        Args:
            factor_df: 因子数据
            returns_df: 收益率数据
            market_index: 市场指数序列
            bull_threshold: 牛市阈值（累计收益率）
            window: 市场趋势计算窗口

        Returns:
            包含牛市和熊市表现的字典
        """
        # 计算市场趋势
        market_return = market_index.pct_change(window)
        market_regime = market_return > bull_threshold

        # 分析看涨策略
        bull_result = self.analyze_bull_strategy(factor_df, returns_df)
        strategy_returns = bull_result["strategy_returns"]

        # 分组统计
        regime_performance = {}
        for regime_name, regime_mask in [("牛市", market_regime), ("熊市", ~market_regime)]:
            # 对齐索引
            common_index = strategy_returns.index.intersection(regime_mask[regime_mask].index)

            if len(common_index) == 0:
                regime_performance[regime_name] = {
                    "total_return": np.nan,
                    "sharpe_ratio": np.nan,
                    "win_rate": np.nan,
                    "days": 0,
                }
                continue

            regime_returns = strategy_returns.loc[common_index]

            regime_performance[regime_name] = {
                "total_return": (1 + regime_returns).prod() - 1,
                "sharpe_ratio": (
                    regime_returns.mean() / regime_returns.std() * np.sqrt(self.annualization_factor)
                    if regime_returns.std() > 0
                    else np.nan
                ),
                "win_rate": (regime_returns > 0).mean(),
                "days": len(regime_returns),
                "avg_daily_return": regime_returns.mean(),
            }

        return regime_performance

    def analyze_industry_rotation(
        self, factor_df: pd.DataFrame, returns_df: pd.DataFrame, industry_df: pd.DataFrame, top_pct: float = 0.2
    ) -> Dict[str, any]:
        """
        分析行业轮动场景

        按行业分组，分析因子在不同行业中的表现

        Args:
            factor_df: 因子数据
            returns_df: 收益率数据
            industry_df: 行业分类数据 (MultiIndex: datetime, instrument)
            top_pct: 选股比例

        Returns:
            包含各行业表现的字典
        """
        # 确保索引对齐
        common_index = factor_df.index.intersection(returns_df.index)
        common_index = common_index.intersection(industry_df.index)

        factor_aligned = factor_df.loc[common_index]
        returns_aligned = returns_df.loc[common_index]
        industry_aligned = industry_df.loc[common_index]

        # 获取所有行业
        industries = industry_aligned.unique()

        industry_performance = {}
        for industry in industries:
            # 选择该行业的股票
            industry_mask = industry_aligned == industry
            industry_factor = factor_aligned[industry_mask]
            industry_returns = returns_aligned[industry_mask]

            # 分析看涨策略
            try:
                result = self.analyze_bull_strategy(industry_factor, industry_returns, top_pct=top_pct)
                industry_performance[industry] = {
                    "total_return": result["total_return"],
                    "sharpe_ratio": result["sharpe_ratio"],
                    "win_rate": result["win_rate"],
                    "max_drawdown": result["max_drawdown"],
                }
            except Exception as e:
                # 某些行业可能数据不足
                industry_performance[industry] = {"error": str(e)}

        return industry_performance

    def analyze_market_cap_groups(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        market_cap_df: pd.DataFrame,
        top_pct: float = 0.2,
        n_groups: int = 3,
    ) -> Dict[str, any]:
        """
        分析市值分组场景

        按市值分组（大盘、中盘、小盘），分析因子在不同市值组中的表现

        Args:
            factor_df: 因子数据
            returns_df: 收益率数据
            market_cap_df: 市值数据 (MultiIndex: datetime, instrument)
            top_pct: 选股比例
            n_groups: 分组数量（默认 3：大中小盘）

        Returns:
            包含各市值组表现的字典
        """
        # 确保索引对齐
        common_index = factor_df.index.intersection(returns_df.index)
        common_index = common_index.intersection(market_cap_df.index)

        factor_aligned = factor_df.loc[common_index]
        returns_aligned = returns_df.loc[common_index]
        market_cap_aligned = market_cap_df.loc[common_index]

        # 按市值分组
        cap_groups = pd.qcut(
            market_cap_aligned,
            q=n_groups,
            labels=["小盘" if i == 0 else "中盘" if i == 1 else "大盘" for i in range(n_groups)],
            duplicates="drop",
        )

        group_performance = {}
        for group_name in cap_groups.unique():
            # 选择该市值组的股票
            group_mask = cap_groups == group_name
            group_factor = factor_aligned[group_mask]
            group_returns = returns_aligned[group_mask]

            # 分析看涨策略
            try:
                result = self.analyze_bull_strategy(group_factor, group_returns, top_pct=top_pct)
                group_performance[group_name] = {
                    "total_return": result["total_return"],
                    "sharpe_ratio": result["sharpe_ratio"],
                    "win_rate": result["win_rate"],
                    "max_drawdown": result["max_drawdown"],
                }
            except Exception as e:
                # 某些组可能数据不足
                group_performance[group_name] = {"error": str(e)}

        return group_performance

    def analyze_all_scenarios(
        self,
        factor_df: pd.DataFrame,
        returns_df: pd.DataFrame,
        price_df: Optional[pd.DataFrame] = None,
        market_index: Optional[pd.Series] = None,
        industry_df: Optional[pd.DataFrame] = None,
        market_cap_df: Optional[pd.DataFrame] = None,
        top_pct: float = 0.2,
    ) -> Dict[str, any]:
        """
        分析所有策略场景

        Args:
            factor_df: 因子数据
            returns_df: 收益率数据
            price_df: 价格数据（用于波动率策略）
            market_index: 市场指数（用于牛熊市分析）
            industry_df: 行业分类（用于行业轮动分析）
            market_cap_df: 市值数据（用于市值分组分析）
            top_pct: 选股比例

        Returns:
            包含所有场景分析结果的字典
        """
        results = {}

        # 1. 看涨策略
        results["bull"] = self.analyze_bull_strategy(factor_df, returns_df, top_pct=top_pct)

        # 2. 看跌策略
        results["bear"] = self.analyze_bear_strategy(factor_df, returns_df, top_pct=top_pct)

        # 3. 多空策略
        results["long_short"] = self.analyze_long_short_strategy(factor_df, returns_df, top_pct=top_pct)

        # 4. 波动率策略（需要价格数据）
        if price_df is not None:
            results["volatility"] = self.analyze_volatility_strategy(factor_df, returns_df, price_df, top_pct=top_pct)

        # 5. 牛熊市场景（需要市场指数）
        if market_index is not None:
            results["market_regime"] = self.analyze_market_regime(factor_df, returns_df, market_index)

        # 6. 行业轮动（需要行业分类）
        if industry_df is not None:
            results["industry_rotation"] = self.analyze_industry_rotation(
                factor_df, returns_df, industry_df, top_pct=top_pct
            )

        # 7. 市值分组（需要市值数据）
        if market_cap_df is not None:
            results["market_cap_groups"] = self.analyze_market_cap_groups(
                factor_df, returns_df, market_cap_df, top_pct=top_pct
            )

        return results

    # ========== 辅助方法 ==========

    def _select_top_factor(
        self, factor_df: pd.DataFrame, top_pct: float = 0.2, quantile: Optional[int] = None, ascending: bool = False
    ) -> pd.DataFrame:
        """
        选择因子值最高或最低的股票

        Args:
            factor_df: 因子数据
            top_pct: 选取比例
            quantile: 分组数量
            ascending: 是否升序（True 选最小的，False 选最大的）

        Returns:
            选中的股票掩码（DataFrame，值为 True/False）
        """
        selected = pd.DataFrame(False, index=factor_df.index, columns=["selected"])

        dates = factor_df.index.get_level_values("datetime").unique()

        for date in dates:
            daily_factor = factor_df.loc[date].dropna()

            # 如果是 DataFrame，转换为 Series
            if isinstance(daily_factor, pd.DataFrame):
                if daily_factor.shape[1] == 1:
                    daily_factor = daily_factor.iloc[:, 0]
                else:
                    # 多列情况，取第一列
                    daily_factor = daily_factor.iloc[:, 0]

            if len(daily_factor) == 0:
                continue

            # 确定选取数量
            if quantile is not None:
                n_select = max(1, len(daily_factor) // quantile)
            else:
                n_select = max(1, int(len(daily_factor) * top_pct))

            # 选择
            if ascending:
                # 选最小的
                threshold = daily_factor.nsmallest(n_select).iloc[-1]
                mask = daily_factor <= threshold
            else:
                # 选最大的
                threshold = daily_factor.nlargest(n_select).iloc[-1]
                mask = daily_factor >= threshold

            # 更新选中标记
            for idx in mask[mask].index:
                selected.loc[(date, idx), "selected"] = True

        return selected

    def _calculate_strategy_returns(self, selected: pd.DataFrame, returns_df: pd.DataFrame) -> pd.Series:
        """
        计算策略收益

        Args:
            selected: 选股掩码
            returns_df: 收益率数据

        Returns:
            策略每日收益序列
        """
        # 确保索引对齐
        common_index = selected.index.intersection(returns_df.index)
        selected_aligned = selected.loc[common_index]
        returns_aligned = returns_df.loc[common_index]

        # 如果 returns_df 是 DataFrame，转换为 Series（取第一列）
        if isinstance(returns_aligned, pd.DataFrame):
            if returns_aligned.shape[1] == 1:
                returns_aligned = returns_aligned.iloc[:, 0]
            else:
                # 多列情况，取第一列
                returns_aligned = returns_aligned.iloc[:, 0]

        # 计算每日平均收益
        strategy_returns = []

        dates = selected_aligned.index.get_level_values("datetime").unique()

        for date in dates:
            daily_selected = selected_aligned.loc[date]
            daily_returns = returns_aligned.loc[date]

            # 选择被选中的股票的收益
            mask = daily_selected["selected"].values
            if mask.sum() == 0:
                strategy_returns.append(np.nan)
            else:
                selected_returns = daily_returns.values[mask]
                # 去除 NaN
                selected_returns = selected_returns[~np.isnan(selected_returns)]

                if len(selected_returns) == 0:
                    strategy_returns.append(np.nan)
                else:
                    strategy_returns.append(selected_returns.mean())

        return pd.Series(strategy_returns, index=dates).dropna()

    def _calculate_strategy_metrics(self, strategy_returns: pd.Series) -> Dict[str, float]:
        """
        计算策略指标

        Args:
            strategy_returns: 策略收益序列

        Returns:
            包含各项指标的字典
        """
        if len(strategy_returns) == 0:
            return {
                "total_return": np.nan,
                "annual_return": np.nan,
                "sharpe_ratio": np.nan,
                "max_drawdown": np.nan,
                "win_rate": np.nan,
                "calmar_ratio": np.nan,
            }

        # 总收益率
        total_return = (1 + strategy_returns).prod() - 1

        # 年化收益率
        n_days = len(strategy_returns)
        annual_return = (1 + total_return) ** (self.annualization_factor / n_days) - 1

        # 夏普比率
        mean_return = float(strategy_returns.mean())
        std_return = float(strategy_returns.std())

        if not np.isnan(std_return) and std_return > 0:
            sharpe_ratio = mean_return / std_return * np.sqrt(self.annualization_factor)
        else:
            sharpe_ratio = np.nan

        # 最大回撤
        cumulative_returns = (1 + strategy_returns).cumprod()
        running_max = cumulative_returns.expanding().max()
        drawdown = (cumulative_returns - running_max) / running_max
        max_drawdown = float(drawdown.min())

        # 胜率
        win_rate = float((strategy_returns > 0).mean())

        # 卡玛比率（年化收益 / 最大回撤的绝对值）
        if not np.isnan(max_drawdown) and abs(max_drawdown) > 1e-6:
            calmar_ratio = annual_return / abs(max_drawdown)
        else:
            calmar_ratio = np.nan

        return {
            "total_return": total_return,
            "annual_return": annual_return,
            "sharpe_ratio": sharpe_ratio,
            "max_drawdown": max_drawdown,
            "win_rate": win_rate,
            "calmar_ratio": calmar_ratio,
        }

    def _calculate_market_volatility(self, returns_df: pd.DataFrame, window: int = 20) -> pd.Series:
        """
        计算市场波动率

        Args:
            returns_df: 收益率数据
            window: 计算窗口

        Returns:
            市场波动率序列
        """
        # 如果 returns_df 是 DataFrame，转换为 Series（取第一列）
        if isinstance(returns_df, pd.DataFrame):
            if returns_df.shape[1] == 1:
                returns_series = returns_df.iloc[:, 0]
            else:
                # 多列情况，取第一列
                returns_series = returns_df.iloc[:, 0]
        else:
            returns_series = returns_df

        # 计算每日所有股票的收益率标准差
        daily_vol = returns_series.groupby(level="datetime").std()

        # 平滑处理
        market_vol = daily_vol.rolling(window=window, min_periods=1).mean()

        # 去除 NaN
        market_vol = market_vol.dropna()

        return market_vol


# 便捷函数
def analyze_factor_strategies(
    factor_df: pd.DataFrame,
    returns_df: pd.DataFrame,
    price_df: Optional[pd.DataFrame] = None,
    top_pct: float = 0.2,
    **kwargs,
) -> Dict[str, any]:
    """
    分析因子策略的便捷函数

    Args:
        factor_df: 因子数据
        returns_df: 收益率数据
        price_df: 价格数据（可选）
        top_pct: 选股比例
        **kwargs: 其他参数

    Returns:
        策略分析结果

    Examples:
        >>> from core.strategy_analyzer import analyze_factor_strategies
        >>>
        >>> # 快速分析
        >>> results = analyze_factor_strategies(
        ...     factor_df, returns_df, price_df, top_pct=0.2
        ... )
        >>>
        >>> # 查看结果
        >>> print(f"看涨策略年化收益: {results['bull']['annual_return']:.2%}")
        >>> print(f"多空策略夏普比率: {results['long_short']['sharpe_ratio']:.2f}")
    """
    analyzer = StrategyAnalyzer(**kwargs)
    return analyzer.analyze_all_scenarios(factor_df, returns_df, price_df, top_pct=top_pct)
