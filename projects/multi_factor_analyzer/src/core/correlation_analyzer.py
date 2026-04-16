"""
因子相关性分析器 (Factor Correlation Analyzer)

该模块实现了因子之间的相关性分析功能：
1. 计算因子相关性矩阵
2. 识别高度相关的因子对
3. 生成因子去重建议
4. 提供可视化支持

设计理念：
- 多因子组合时需要避免高度相关的因子
- 高度相关的因子会导致多重共线性问题
- 相关性分析有助于因子筛选和组合优化
"""

from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .config import CORRELATION_THRESHOLDS


class FactorCorrelationAnalyzer:
    """
    因子相关性分析器

    用于分析多个因子之间的相关性，识别高度相关的因子对，
    并提供去重建议。

    核心功能：
    1. 计算因子相关性矩阵（Pearson、Spearman、Kendall）
    2. 识别高度相关的因子对
    3. 生成因子去重和组合建议
    4. 可视化相关性矩阵

    Attributes:
        method: 相关性计算方法 ('pearson', 'spearman', 'kendall')
        threshold: 高相关性阈值

    Examples:
        >>> from core.correlation_analyzer import FactorCorrelationAnalyzer
        >>>
        >>> # 准备因子数据
        >>> factor_dict = {
        ...     'MA20': ma20_df,
        ...     'MA60': ma60_df,
        ...     'RSI': rsi_df,
        ... }
        >>>
        >>> # 创建分析器
        >>> analyzer = FactorCorrelationAnalyzer(method='spearman')
        >>>
        >>> # 分析相关性
        >>> result = analyzer.analyze_correlation(factor_dict)
        >>>
        >>> # 查看结果
        >>> print(result['correlation_matrix'])
        >>> print(result['recommendation'])
    """

    def __init__(self, method: str = "spearman", threshold: Optional[float] = None, min_periods: int = 10):
        """
        初始化相关性分析器

        Args:
            method: 相关性计算方法
                - 'pearson': 皮尔逊相关系数（线性相关）
                - 'spearman': 斯皮尔曼相关系数（单调相关，推荐）
                - 'kendall': 肯德尔相关系数（秩相关）
            threshold: 高相关性阈值，默认使用配置文件中的值
            min_periods: 计算相关性的最小观测数

        Examples:
            >>> # 使用斯皮尔曼相关系数
            >>> analyzer = FactorCorrelationAnalyzer(method='spearman')
            >>>
            >>> # 自定义阈值
            >>> analyzer = FactorCorrelationAnalyzer(threshold=0.8)
        """
        if method not in ["pearson", "spearman", "kendall"]:
            raise ValueError(f"不支持的相关性方法: {method}. " "支持的方法: 'pearson', 'spearman', 'kendall'")

        self.method = method
        self.threshold = threshold or CORRELATION_THRESHOLDS["high"]
        self.min_periods = min_periods

    def analyze_correlation(
        self, factor_dict: Dict[str, pd.DataFrame], threshold: Optional[float] = None
    ) -> Dict[str, any]:
        """
        分析因子之间的相关性

        这是最常用的方法，一次性完成所有相关性分析。

        Args:
            factor_dict: 因子字典 {factor_name: factor_df}
                        因子 DataFrame 的索引应为 (datetime, instrument)
            threshold: 高相关性阈值（覆盖初始化时的值）

        Returns:
            包含分析结果的字典：
                - correlation_matrix: 相关性矩阵 (DataFrame)
                - high_correlation_pairs: 高度相关因子对列表
                - recommendation: 去重建议 (str)
                - statistics: 相关性统计信息

        Examples:
            >>> result = analyzer.analyze_correlation(factor_dict)
            >>>
            >>> # 查看相关性矩阵
            >>> print(result['correlation_matrix'])
            >>>
            >>> # 查看高度相关的因子对
            >>> for pair in result['high_correlation_pairs']:
            ...     print(f"{pair['factor1']} - {pair['factor2']}: "
            ...           f"{pair['correlation']:.2f}")
            >>>
            >>> # 查看建议
            >>> print(result['recommendation'])
        """
        # 1. 计算相关性矩阵
        corr_matrix = self.calculate_correlation_matrix(factor_dict)

        # 2. 找出高度相关的因子对
        high_corr_threshold = threshold or self.threshold
        high_corr_pairs = self.find_high_correlation(corr_matrix, threshold=high_corr_threshold)

        # 3. 生成建议
        recommendation = self.generate_recommendation(high_corr_pairs, corr_matrix)

        # 4. 计算统计信息
        statistics = self.calculate_statistics(corr_matrix)

        return {
            "correlation_matrix": corr_matrix,
            "high_correlation_pairs": high_corr_pairs,
            "recommendation": recommendation,
            "statistics": statistics,
        }

    def calculate_correlation_matrix(self, factor_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        计算因子相关性矩阵

        Args:
            factor_dict: 因子字典 {factor_name: factor_df}

        Returns:
            相关性矩阵 DataFrame

        Raises:
            ValueError: 如果因子数据为空或格式不正确

        Examples:
            >>> corr_matrix = analyzer.calculate_correlation_matrix(factor_dict)
            >>> print(corr_matrix)
                    MA20      MA60       RSI
            MA20  1.000000  0.852341  0.123456
            MA60  0.852341  1.000000  0.098765
            RSI   0.123456  0.098765  1.000000
        """
        if not factor_dict:
            raise ValueError("因子字典不能为空")

        # 对齐所有因子的索引
        aligned_factors = self._align_factors(factor_dict)

        # 合并所有因子
        combined_df = pd.concat(aligned_factors, axis=1)

        # 计算相关性矩阵
        corr_matrix = combined_df.corr(method=self.method, min_periods=self.min_periods)

        return corr_matrix

    def find_high_correlation(
        self, corr_matrix: pd.DataFrame, threshold: Optional[float] = None
    ) -> List[Dict[str, any]]:
        """
        找出高度相关的因子对

        Args:
            corr_matrix: 相关性矩阵
            threshold: 相关系数阈值（默认使用初始化时的值）

        Returns:
            高度相关因子对列表，每个元素包含：
                - factor1: 第一个因子名称
                - factor2: 第二个因子名称
                - correlation: 相关系数
                - level: 相关性等级 ('high', 'medium', 'low')

        Examples:
            >>> pairs = analyzer.find_high_correlation(corr_matrix, threshold=0.7)
            >>> for pair in pairs:
            ...     print(f"{pair['factor1']} 和 {pair['factor2']}: "
            ...           f"{pair['correlation']:.2f}")
        """
        threshold = threshold or self.threshold
        high_corr_pairs = []

        # 获取因子名称
        factors = corr_matrix.columns.tolist()

        # 遍历上三角矩阵（避免重复）
        for i in range(len(factors)):
            for j in range(i + 1, len(factors)):
                factor1 = factors[i]
                factor2 = factors[j]
                correlation = corr_matrix.iloc[i, j]

                # 只保留高度相关的因子对
                if abs(correlation) >= threshold:
                    # 确定相关性等级
                    if abs(correlation) >= CORRELATION_THRESHOLDS["high"]:
                        level = "high"
                    elif abs(correlation) >= CORRELATION_THRESHOLDS["medium"]:
                        level = "medium"
                    else:
                        level = "low"

                    high_corr_pairs.append(
                        {
                            "factor1": factor1,
                            "factor2": factor2,
                            "correlation": correlation,
                            "level": level,
                        }
                    )

        # 按相关系数绝对值降序排序
        high_corr_pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)

        return high_corr_pairs

    def generate_recommendation(self, high_corr_pairs: List[Dict[str, any]], corr_matrix: pd.DataFrame) -> str:
        """
        生成因子去重和组合建议

        Args:
            high_corr_pairs: 高度相关因子对列表
            corr_matrix: 相关性矩阵

        Returns:
            建议文本

        Examples:
            >>> recommendation = analyzer.generate_recommendation(pairs, corr_matrix)
            >>> print(recommendation)
        """
        if not high_corr_pairs:
            return (
                "✅ 所有因子相关性较低，可以安全组合使用。\n\n"
                "建议：\n"
                "- 可以直接使用所有因子构建多因子模型\n"
                "- 因子之间具有良好的互补性"
            )

        recommendation_lines = ["⚠️  发现高度相关的因子对，建议进行去重处理：\n"]

        for i, pair in enumerate(high_corr_pairs, 1):
            factor1 = pair["factor1"]
            factor2 = pair["factor2"]
            corr = pair["correlation"]
            level = pair["level"]

            # 相关性等级标识
            if level == "high":
                level_icon = "🔴"
            elif level == "medium":
                level_icon = "🟡"
            else:
                level_icon = "🟢"

            recommendation_lines.append(f"{i}. {level_icon} {factor1} 和 {factor2}: " f"相关系数 {corr:.3f}")

        recommendation_lines.append("\n建议处理方案：")

        # 分析高度相关因子对，提供具体建议
        unique_factors = set()
        for pair in high_corr_pairs:
            unique_factors.add(pair["factor1"])
            unique_factors.add(pair["factor2"])

        # 建议保留策略
        if len(high_corr_pairs) >= 3:
            recommendation_lines.append(
                "\n1. 因子聚类分析：\n"
                "   - 建议使用层次聚类或 K-means 聚类对因子进行分组\n"
                "   - 从每个聚类中选择代表性因子\n"
                "   - 可以使用 IC、IR 等指标作为选择依据"
            )

        recommendation_lines.append(
            "\n2. 因子筛选：\n"
            "   - 比较高度相关因子的 IC、IR、多空收益等指标\n"
            "   - 保留表现更好的因子\n"
            "   - 或者通过主成分分析（PCA）提取综合因子"
        )

        recommendation_lines.append(
            "\n3. 正交化处理：\n"
            "   - 对高度相关的因子进行正交化处理\n"
            "   - 使用回归方法剔除因子间的线性相关性\n"
            "   - 保留正交化后的残差作为新因子"
        )

        recommendation_lines.append(
            "\n4. 分组使用：\n"
            "   - 将高度相关的因子分为不同组\n"
            "   - 在不同的市场环境或策略中分别使用\n"
            "   - 避免在同一模型中同时使用"
        )

        return "\n".join(recommendation_lines)

    def calculate_statistics(self, corr_matrix: pd.DataFrame) -> Dict[str, float]:
        """
        计算相关性统计信息

        Args:
            corr_matrix: 相关性矩阵

        Returns:
            统计信息字典

        Examples:
            >>> stats = analyzer.calculate_statistics(corr_matrix)
            >>> print(f"平均相关系数: {stats['mean_correlation']:.3f}")
        """
        # 提取上三角矩阵（排除对角线）
        upper_triangle = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]

        if len(upper_triangle) == 0:
            return {
                "mean_correlation": 0.0,
                "median_correlation": 0.0,
                "std_correlation": 0.0,
                "max_correlation": 0.0,
                "min_correlation": 0.0,
                "high_correlation_count": 0,
                "high_correlation_ratio": 0.0,
            }

        statistics = {
            "mean_correlation": float(np.mean(upper_triangle)),
            "median_correlation": float(np.median(upper_triangle)),
            "std_correlation": float(np.std(upper_triangle)),
            "max_correlation": float(np.max(upper_triangle)),
            "min_correlation": float(np.min(upper_triangle)),
            "high_correlation_count": int(np.sum(np.abs(upper_triangle) >= self.threshold)),
            "high_correlation_ratio": float(np.sum(np.abs(upper_triangle) >= self.threshold) / len(upper_triangle)),
        }

        return statistics

    def plot_correlation_matrix(
        self,
        corr_matrix: pd.DataFrame,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = "RdYlGn",
        annot: bool = True,
        save_path: Optional[str] = None,
        **kwargs,
    ) -> plt.Axes:
        """
        绘制相关性矩阵热力图

        Args:
            corr_matrix: 相关性矩阵
            figsize: 图形大小
            cmap: 颜色映射
            annot: 是否显示数值标注
            save_path: 保存路径（如果为 None，则不保存）
            **kwargs: 传递给 sns.heatmap 的其他参数

        Returns:
            matplotlib Axes 对象

        Examples:
            >>> ax = analyzer.plot_correlation_matrix(
            ...     corr_matrix,
            ...     save_path='correlation_matrix.png'
            ... )
            >>> plt.show()
        """
        # 创建图形
        fig, ax = plt.subplots(figsize=figsize)

        # 绘制热力图
        sns.heatmap(
            corr_matrix,
            cmap=cmap,
            annot=annot,
            fmt=".3f",
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax,
            **kwargs,
        )

        ax.set_title(f"Factor Correlation Matrix ({self.method.capitalize()})", fontsize=14, fontweight="bold", pad=20)

        plt.tight_layout()

        # 保存图形
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            print(f"✅ 相关性矩阵已保存到: {save_path}")

        return ax

    def _align_factors(self, factor_dict: Dict[str, pd.DataFrame]) -> Dict[str, pd.Series]:
        """
        对齐所有因子的索引

        Args:
            factor_dict: 因子字典

        Returns:
            对齐后的因子字典（转换为 Series）
        """
        # 获取所有因子的公共索引
        common_index = None

        for factor_name, factor_df in factor_dict.items():
            # 如果是 DataFrame，转换为 Series（取第一列）
            if isinstance(factor_df, pd.DataFrame):
                if factor_df.shape[1] == 1:
                    factor_series = factor_df.iloc[:, 0]
                else:
                    # 多列情况，取第一列
                    factor_series = factor_df.iloc[:, 0]
            else:
                factor_series = factor_df

            # 更新公共索引
            if common_index is None:
                common_index = factor_series.index
            else:
                common_index = common_index.intersection(factor_series.index)

        if common_index is None or len(common_index) == 0:
            raise ValueError("因子之间没有公共索引，无法计算相关性")

        # 对齐所有因子
        aligned_factors = {}
        for factor_name, factor_df in factor_dict.items():
            # 转换为 Series
            if isinstance(factor_df, pd.DataFrame):
                if factor_df.shape[1] == 1:
                    factor_series = factor_df.iloc[:, 0]
                else:
                    factor_series = factor_df.iloc[:, 0]
            else:
                factor_series = factor_df

            # 对齐索引
            aligned_factors[factor_name] = factor_series.loc[common_index]

        return aligned_factors


# 便捷函数
def analyze_factor_correlation(
    factor_dict: Dict[str, pd.DataFrame],
    method: str = "spearman",
    threshold: float = 0.7,
    plot: bool = False,
    save_path: Optional[str] = None,
) -> Dict[str, any]:
    """
    分析因子相关性的便捷函数

    Args:
        factor_dict: 因子字典 {factor_name: factor_df}
        method: 相关性计算方法
        threshold: 高相关性阈值
        plot: 是否绘制相关性矩阵
        save_path: 图形保存路径

    Returns:
        相关性分析结果

    Examples:
        >>> from core.correlation_analyzer import analyze_factor_correlation
        >>>
        >>> # 快速分析
        >>> result = analyze_factor_correlation(
        ...     factor_dict,
        ...     method='spearman',
        ...     threshold=0.7,
        ...     plot=True,
        ...     save_path='correlation.png'
        ... )
        >>>
        >>> # 查看建议
        >>> print(result['recommendation'])
    """
    analyzer = FactorCorrelationAnalyzer(method=method, threshold=threshold)
    result = analyzer.analyze_correlation(factor_dict)

    # 绘制图形
    if plot:
        analyzer.plot_correlation_matrix(result["correlation_matrix"], save_path=save_path)

    return result


if __name__ == "__main__":
    # 示例：使用相关性分析器
    print("=" * 60)
    print("因子相关性分析器示例")
    print("=" * 60)

    # 创建模拟数据
    print("\n生成模拟因子数据...")
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")
    instruments = ["SH600000", "SH600001", "SH600002", "SH600003", "SH600004"]

    # 创建多索引
    index = pd.MultiIndex.from_product([dates, instruments], names=["datetime", "instrument"])

    # 生成模拟因子数据
    n = len(index)

    # MA20 和 MA60 高度相关
    ma20 = pd.Series(np.random.randn(n), index=index)
    ma60 = ma20 * 0.9 + np.random.randn(n) * 0.1  # 高度相关

    # RSI 独立
    rsi = pd.Series(np.random.randn(n), index=index)

    # MACD 独立
    macd = pd.Series(np.random.randn(n), index=index)

    factor_dict = {
        "MA20": ma20,
        "MA60": ma60,
        "RSI": rsi,
        "MACD": macd,
    }

    print(f"生成了 {len(factor_dict)} 个因子")
    print(f"每个因子有 {n} 个观测值")

    # 创建分析器
    analyzer = FactorCorrelationAnalyzer(method="spearman", threshold=0.7)

    # 分析相关性
    print("\n分析因子相关性...")
    result = analyzer.analyze_correlation(factor_dict)

    # 打印相关性矩阵
    print("\n相关性矩阵:")
    print(result["correlation_matrix"].round(3))

    # 打印高度相关的因子对
    print(f"\n高度相关的因子对 (阈值={analyzer.threshold}):")
    if result["high_correlation_pairs"]:
        for pair in result["high_correlation_pairs"]:
            print(f"  - {pair['factor1']} 和 {pair['factor2']}: " f"{pair['correlation']:.3f} ({pair['level']})")
    else:
        print("  无高度相关的因子对")

    # 打印统计信息
    print("\n相关性统计:")
    stats = result["statistics"]
    print(f"  平均相关系数: {stats['mean_correlation']:.3f}")
    print(f"  中位数相关系数: {stats['median_correlation']:.3f}")
    print(f"  最大相关系数: {stats['max_correlation']:.3f}")
    print(f"  最小相关系数: {stats['min_correlation']:.3f}")
    print(f"  高相关因子对数量: {stats['high_correlation_count']}")
    print(f"  高相关因子对比例: {stats['high_correlation_ratio']:.2%}")

    # 打印建议
    print("\n建议:")
    print(result["recommendation"])
