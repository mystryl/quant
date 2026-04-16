"""
可视化工具模块 (Visualizer)

该模块提供因子分析的可视化功能，包括：
1. IC 时间序列图
2. IC 分布直方图
3. 多空收益曲线
4. 策略场景对比图
5. 周期对齐效果对比图
6. 累计收益曲线
7. 回撤图

设计理念：
- 使用 matplotlib 和 seaborn 创建专业图表
- 统一的配色方案和样式
- 支持中文字体
- 支持多种输出格式（PNG、SVG、PDF）
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")  # 使用非交互式后端
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.font_manager import fontManager
import seaborn as sns

# Module-level cache: avoid scanning all system fonts on every call
_CJK_FONT_INITIALIZED = False


def _setup_cjk_font() -> None:
    """
    配置 matplotlib 中文字体，优先使用 macOS 内置字体。

    按优先级依次检测可用字体：
    - macOS: PingFang SC, Heiti SC, Heiti TC, STHeiti
    - 跨平台: Arial Unicode MS, Noto Sans CJK SC
    - Linux: WenQuanYi Micro Hei, SimHei
    - 兜底: DejaVu Sans（不支持 CJK，但保证不报错）

    结果会被缓存，重复调用是空操作（除非 force=True）。
    """
    _setup_cjk_font_impl(force=False)


def _setup_cjk_font_impl(force: bool = False) -> None:
    """
    内部实现：配置 matplotlib 中文字体。

    Args:
        force: 强制重新扫描字体（例如 plt.style.use 重置后）。
    """
    global _CJK_FONT_INITIALIZED

    if _CJK_FONT_INITIALIZED and not force:
        # 快速路径：恢复已缓存的字体配置
        plt.rcParams["axes.unicode_minus"] = False
        return

    # 所有候选字体（按平台优先级排列）
    _CJK_FONT_CANDIDATES = [
        # macOS 内置（Apple Silicon / Intel 通用）
        "PingFang SC",
        "Heiti SC",
        "Heiti TC",
        "STHeiti",
        # 跨平台
        "Arial Unicode MS",
        "Noto Sans CJK SC",
        # Windows / WSL
        "Microsoft YaHei",
        "SimHei",
        # Linux
        "WenQuanYi Micro Hei",
        "Noto Sans CJK JP",  # 日文字体也覆盖大量 CJK 字符
        # 兜底（非 CJK，保证列表不为空）
        "DejaVu Sans",
    ]

    # 从 matplotlib 已注册字体中筛选实际可用的
    available_names = {f.name for f in fontManager.ttflist}
    resolved_fonts = [name for name in _CJK_FONT_CANDIDATES if name in available_names]

    if not resolved_fonts:
        warnings.warn("无法设置中文字体，图表中文可能显示为方块。" "建议安装 Noto Sans CJK 或 PingFang 字体。")
        resolved_fonts = ["DejaVu Sans"]

    # 将 CJK 字体插入 font.sans-serif 列表最前面，保留其余系统字体
    existing = list(plt.rcParams["font.sans-serif"])
    # 去重：去掉 existing 中已出现在 resolved_fonts 里的
    existing_deduped = [f for f in existing if f not in resolved_fonts]
    plt.rcParams["font.sans-serif"] = resolved_fonts + existing_deduped

    # Also set monospace fallback chain so CJK glyphs render
    # in text that explicitly uses family="monospace" (e.g. code blocks).
    monospace_cjk = ["Heiti TC", "PingFang SC", "Menlo", "DejaVu Sans Mono", "monospace"]
    existing_mono = list(plt.rcParams["font.monospace"])
    existing_mono_deduped = [f for f in existing_mono if f not in monospace_cjk]
    plt.rcParams["font.monospace"] = monospace_cjk + existing_mono_deduped

    plt.rcParams["axes.unicode_minus"] = False
    _CJK_FONT_INITIALIZED = True


# 设置 seaborn 样式
sns.set_style("whitegrid")
sns.set_context("paper", font_scale=1.2)

# 配置中文字体（必须在 seaborn 样式设置之后，否则会被覆盖）
try:
    _setup_cjk_font()
except (RuntimeError, ValueError, OSError) as exc:
    warnings.warn(f"中文字体初始化失败 ({exc})，图表中文可能显示为方块")


# 统一配色方案
COLOR_SCHEME = {
    "primary": "#2c3e50",  # 深蓝
    "secondary": "#3498db",  # 蓝
    "success": "#27ae60",  # 绿
    "warning": "#f39c12",  # 橙
    "danger": "#c0392b",  # 红
    "info": "#16a085",  # 青
    "purple": "#8e44ad",  # 紫
    "gray": "#95a5a6",  # 灰
}


class Visualizer:
    """
    可视化工具类

    生成因子分析的各种可视化图表。

    支持的图表类型：
    1. IC 时间序列图：展示 IC 随时间的变化
    2. IC 分布直方图：展示 IC 的分布特征
    3. 累计收益曲线：展示策略的累计收益
    4. 回撤图：展示策略的回撤情况
    5. 策略对比图：对比不同策略的表现
    6. 周期对比图：对比不同对齐周期的效果
    7. 滚动 IC 图：展示 IC 的滚动统计

    Attributes:
        output_dir: 图表输出目录
        dpi: 图表分辨率
        style: 图表样式
        figsize: 图表默认大小

    Examples:
        >>> from report.visualizer import Visualizer
        >>>
        >>> # 创建可视化工具
        >>> viz = Visualizer(output_dir="./charts")
        >>>
        >>> # 生成 IC 时间序列图
        >>> fig = viz.plot_ic_timeseries(metrics['ic'], factor_name="MA20")
        >>> viz.save_figure(fig, "MA20_ic_timeseries.png")
        >>>
        >>> # 生成累计收益曲线
        >>> fig = viz.plot_cumulative_return(
        ...     scenario_results['bull']['strategy_returns'],
        ...     factor_name="MA20"
        ... )
        >>> viz.save_figure(fig, "MA20_cumulative_return.png")
    """

    def __init__(
        self,
        output_dir: str = "./charts",
        dpi: int = 100,
        style: str = "seaborn-v0_8-whitegrid",
        figsize: Tuple[int, int] = (12, 6),
    ):
        """
        初始化可视化工具

        Args:
            output_dir: 图表输出目录
            dpi: 图表分辨率（每英寸点数）
            style: 图表样式（seaborn 或 matplotlib 样式）
            figsize: 图表默认大小（宽，高）英寸
        """
        self.output_dir = Path(output_dir)
        self.dpi = dpi
        self.style = style
        self.figsize = figsize

        # 创建输出目录
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 设置样式
        try:
            plt.style.use(style)
        except (OSError, ValueError):
            # 如果样式不存在，使用默认
            sns.set_style("whitegrid")

        # plt.style.use / sns.set_style 会重置字体，重新配置 CJK
        _setup_cjk_font_impl(force=True)

    def plot_ic_timeseries(
        self,
        ic_series: pd.Series,
        factor_name: str = "Factor",
        window: int = 20,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        绘制 IC 时间序列图

        展示 IC 随时间的变化，包括原始 IC 和滚动均值。

        Args:
            ic_series: IC 序列（索引为日期）
            factor_name: 因子名称
            window: 滚动窗口大小
            figsize: 图表大小

        Returns:
            matplotlib Figure 对象

        Examples:
            >>> fig = viz.plot_ic_timeseries(
            ...     metrics['ic'],
            ...     factor_name="MA20",
            ...     window=20
            ... )
        """
        if figsize is None:
            figsize = self.figsize

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制原始 IC
        ax.plot(
            ic_series.index, ic_series.values, color=COLOR_SCHEME["secondary"], alpha=0.5, linewidth=1, label="原始 IC"
        )

        # 绘制滚动均值
        if len(ic_series) >= window:
            rolling_mean = ic_series.rolling(window=window).mean()
            ax.plot(
                rolling_mean.index,
                rolling_mean.values,
                color=COLOR_SCHEME["primary"],
                linewidth=2,
                label=f"{window}期滚动均值",
            )

        # 绘制零线
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        # 标记正负区域
        ax.fill_between(
            ic_series.index,
            ic_series.values,
            0,
            where=ic_series.values >= 0,
            color=COLOR_SCHEME["success"],
            alpha=0.2,
            label="正 IC",
        )
        ax.fill_between(
            ic_series.index,
            ic_series.values,
            0,
            where=ic_series.values < 0,
            color=COLOR_SCHEME["danger"],
            alpha=0.2,
            label="负 IC",
        )

        # 设置标题和标签
        ax.set_title(f"{factor_name} - IC 时间序列", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("IC 值", fontsize=12)

        # 格式化 x 轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)

        # 添加图例
        ax.legend(loc="upper left", framealpha=0.9)

        # 添加网格
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_ic_distribution(
        self, ic_series: pd.Series, factor_name: str = "Factor", figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        绘制 IC 分布直方图

        展示 IC 的分布特征，包括直方图和核密度估计。

        Args:
            ic_series: IC 序列
            factor_name: 因子名称
            figsize: 图表大小

        Returns:
            matplotlib Figure 对象

        Examples:
            >>> fig = viz.plot_ic_distribution(
            ...     metrics['ic'],
            ...     factor_name="MA20"
            ... )
        """
        if figsize is None:
            figsize = self.figsize

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制直方图
        ax.hist(
            ic_series.dropna().values,
            bins=30,
            color=COLOR_SCHEME["secondary"],
            alpha=0.6,
            edgecolor="white",
            density=True,
        )

        # 绘制核密度估计
        from scipy.stats import gaussian_kde

        kde = gaussian_kde(ic_series.dropna().values)
        x_range = np.linspace(ic_series.min(), ic_series.max(), 200)
        ax.plot(x_range, kde(x_range), color=COLOR_SCHEME["primary"], linewidth=2, label="核密度估计")

        # 绘制均值线
        ic_mean = ic_series.mean()
        ax.axvline(ic_mean, color=COLOR_SCHEME["success"], linestyle="--", linewidth=2, label=f"均值: {ic_mean:.4f}")

        # 绘制零线
        ax.axvline(0, color="gray", linestyle="-", linewidth=1, alpha=0.5)

        # 设置标题和标签
        ax.set_title(f"{factor_name} - IC 分布", fontsize=14, fontweight="bold")
        ax.set_xlabel("IC 值", fontsize=12)
        ax.set_ylabel("密度", fontsize=12)

        # 添加图例
        ax.legend(loc="upper right", framealpha=0.9)

        # 添加网格
        ax.grid(True, alpha=0.3, axis="y")

        # 添加统计信息文本框
        stats_text = (
            f"均值: {ic_mean:.4f}\n"
            f"标准差: {ic_series.std():.4f}\n"
            f"偏度: {ic_series.skew():.4f}\n"
            f"峰度: {ic_series.kurtosis():.4f}"
        )
        ax.text(
            0.02,
            0.98,
            stats_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )

        plt.tight_layout()
        return fig

    def plot_cumulative_return(
        self,
        returns: pd.Series,
        factor_name: str = "Factor",
        benchmark: Optional[pd.Series] = None,
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        绘制累计收益曲线

        展示策略的累计收益变化。

        Args:
            returns: 收益率序列
            factor_name: 因子名称
            benchmark: 基准收益率序列（可选）
            figsize: 图表大小

        Returns:
            matplotlib Figure 对象

        Examples:
            >>> fig = viz.plot_cumulative_return(
            ...     scenario_results['bull']['strategy_returns'],
            ...     factor_name="MA20"
            ... )
        """
        if figsize is None:
            figsize = (12, 6)

        fig, ax = plt.subplots(figsize=figsize)

        # 计算累计收益
        cumulative = (1 + returns).cumprod()

        # 绘制策略收益
        ax.plot(
            cumulative.index, cumulative.values, color=COLOR_SCHEME["primary"], linewidth=2, label=f"{factor_name} 策略"
        )

        # 绘制基准
        if benchmark is not None:
            benchmark_cumulative = (1 + benchmark).cumprod()
            # 对齐索引
            common_index = cumulative.index.intersection(benchmark_cumulative.index)
            ax.plot(
                common_index,
                benchmark_cumulative.loc[common_index],
                color=COLOR_SCHEME["gray"],
                linewidth=2,
                linestyle="--",
                label="基准",
            )

        # 绘制零线
        ax.axhline(y=1, color="gray", linestyle="-", linewidth=1, alpha=0.5)

        # 设置标题和标签
        total_return = cumulative.iloc[-1] - 1
        ax.set_title(f"{factor_name} - 累计收益曲线 (总收益: {total_return:.2%})", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("累计收益", fontsize=12)

        # 格式化 y 轴为百分比
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

        # 格式化 x 轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)

        # 添加图例
        ax.legend(loc="upper left", framealpha=0.9)

        # 添加网格
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_drawdown(
        self, returns: pd.Series, factor_name: str = "Factor", figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        绘制回撤图

        展示策略的回撤情况。

        Args:
            returns: 收益率序列
            factor_name: 因子名称
            figsize: 图表大小

        Returns:
            matplotlib Figure 对象

        Examples:
            >>> fig = viz.plot_drawdown(
            ...     scenario_results['bull']['strategy_returns'],
            ...     factor_name="MA20"
            ... )
        """
        if figsize is None:
            figsize = (12, 6)

        fig, ax = plt.subplots(figsize=figsize)

        # 计算累计收益
        cumulative = (1 + returns).cumprod()

        # 计算回撤
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max

        # 绘制回撤
        ax.fill_between(drawdown.index, drawdown.values, 0, color=COLOR_SCHEME["danger"], alpha=0.3, label="回撤")
        ax.plot(drawdown.index, drawdown.values, color=COLOR_SCHEME["danger"], linewidth=1.5)

        # 标记最大回撤
        max_dd_idx = drawdown.idxmin()
        max_dd_value = drawdown.min()
        ax.annotate(
            f"最大回撤: {max_dd_value:.2%}",
            xy=(max_dd_idx, max_dd_value),
            xytext=(10, 20),
            textcoords="offset points",
            fontsize=10,
            bbox=dict(boxstyle="round,pad=0.5", fc="yellow", alpha=0.5),
            arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
        )

        # 设置标题和标签
        ax.set_title(f"{factor_name} - 回撤图", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("回撤", fontsize=12)

        # 格式化 y 轴为百分比
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.1%}".format(y)))

        # 格式化 x 轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)

        # 添加网格
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_strategy_comparison(
        self, scenario_results: Dict[str, Dict], factor_name: str = "Factor", figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        绘制策略对比图

        对比不同策略场景的累计收益。

        Args:
            scenario_results: 策略场景分析结果字典
            factor_name: 因子名称
            figsize: 图表大小

        Returns:
            matplotlib Figure 对象

        Examples:
            >>> fig = viz.plot_strategy_comparison(
            ...     scenario_results,
            ...     factor_name="MA20"
            ... )
        """
        if figsize is None:
            figsize = (12, 6)

        fig, ax = plt.subplots(figsize=figsize)

        # 策略中文名称映射
        strategy_names = {"bull": "看涨策略", "bear": "看跌策略", "long_short": "多空策略", "volatility": "波动率策略"}

        # 策略颜色映射
        strategy_colors = {
            "bull": COLOR_SCHEME["success"],
            "bear": COLOR_SCHEME["danger"],
            "long_short": COLOR_SCHEME["primary"],
            "volatility": COLOR_SCHEME["warning"],
        }

        # 绘制各策略的累计收益
        for strategy_key, strategy_data in scenario_results.items():
            if "strategy_returns" not in strategy_data:
                continue

            returns = strategy_data["strategy_returns"]
            cumulative = (1 + returns).cumprod()

            strategy_name = strategy_names.get(strategy_key, strategy_key)
            strategy_color = strategy_colors.get(strategy_key, COLOR_SCHEME["gray"])

            ax.plot(
                cumulative.index,
                cumulative.values,
                color=strategy_color,
                linewidth=2,
                label=f'{strategy_name} (年化: {strategy_data.get("annual_return", 0):.2%})',
            )

        # 绘制零线
        ax.axhline(y=1, color="gray", linestyle="-", linewidth=1, alpha=0.5)

        # 设置标题和标签
        ax.set_title(f"{factor_name} - 策略对比", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("累计收益", fontsize=12)

        # 格式化 y 轴为百分比
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: "{:.0%}".format(y)))

        # 格式化 x 轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)

        # 添加图例
        ax.legend(loc="upper left", framealpha=0.9)

        # 添加网格
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_cycle_comparison(
        self, cycle_analysis: Dict[str, Dict], factor_name: str = "Factor", figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        绘制周期对齐效果对比图

        对比不同对齐周期的 IC 均值和 ICIR。

        Args:
            cycle_analysis: 周期分析结果字典
                格式: {shift: {'ic_mean': float, 'icir': float}}
            factor_name: 因子名称
            figsize: 图表大小

        Returns:
            matplotlib Figure 对象

        Examples:
            >>> cycle_data = {
            ...     1: {'ic_mean': 0.02, 'icir': 0.3},
            ...     2: {'ic_mean': 0.03, 'icir': 0.5},
            ...     3: {'ic_mean': 0.025, 'icir': 0.4}
            ... }
            >>> fig = viz.plot_cycle_comparison(cycle_data, factor_name="MA20")
        """
        if figsize is None:
            figsize = (12, 5)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # 提取数据
        shifts = sorted(cycle_analysis.keys())
        ic_means = [cycle_analysis[s]["ic_mean"] for s in shifts]
        icirs = [cycle_analysis[s].get("icir", 0) for s in shifts]

        # 绘制 IC 均值对比
        bars1 = ax1.bar(
            [f"T+{s}" for s in shifts], ic_means, color=COLOR_SCHEME["secondary"], alpha=0.7, edgecolor="white"
        )
        ax1.set_title(f"{factor_name} - 不同周期的 IC 均值", fontsize=12, fontweight="bold")
        ax1.set_xlabel("对齐周期", fontsize=11)
        ax1.set_ylabel("IC 均值", fontsize=11)
        ax1.grid(True, alpha=0.3, axis="y")

        # 标记最佳周期
        if ic_means:
            best_idx = np.argmax(np.abs(ic_means))
            bars1[best_idx].set_color(COLOR_SCHEME["success"])
            bars1[best_idx].set_alpha(1.0)

        # 绘制 ICIR 对比
        bars2 = ax2.bar([f"T+{s}" for s in shifts], icirs, color=COLOR_SCHEME["info"], alpha=0.7, edgecolor="white")
        ax2.set_title(f"{factor_name} - 不同周期的 ICIR", fontsize=12, fontweight="bold")
        ax2.set_xlabel("对齐周期", fontsize=11)
        ax2.set_ylabel("ICIR", fontsize=11)
        ax2.grid(True, alpha=0.3, axis="y")

        # 标记最佳周期
        if icirs:
            best_idx = np.argmax(np.abs(icirs))
            bars2[best_idx].set_color(COLOR_SCHEME["success"])
            bars2[best_idx].set_alpha(1.0)

        plt.tight_layout()
        return fig

    def plot_rolling_ic(
        self,
        ic_series: pd.Series,
        factor_name: str = "Factor",
        windows: List[int] = [20, 40, 60],
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        绘制滚动 IC 图

        展示不同窗口期的滚动 IC 均值。

        Args:
            ic_series: IC 序列
            factor_name: 因子名称
            windows: 滚动窗口列表
            figsize: 图表大小

        Returns:
            matplotlib Figure 对象

        Examples:
            >>> fig = viz.plot_rolling_ic(
            ...     metrics['ic'],
            ...     factor_name="MA20",
            ...     windows=[20, 40, 60]
            ... )
        """
        if figsize is None:
            figsize = (12, 6)

        fig, ax = plt.subplots(figsize=figsize)

        # 绘制原始 IC（淡色）
        ax.plot(ic_series.index, ic_series.values, color=COLOR_SCHEME["gray"], alpha=0.3, linewidth=1, label="原始 IC")

        # 绘制不同窗口的滚动均值
        colors = [COLOR_SCHEME["primary"], COLOR_SCHEME["success"], COLOR_SCHEME["warning"]]
        for i, window in enumerate(windows):
            if len(ic_series) >= window:
                rolling_mean = ic_series.rolling(window=window).mean()
                color = colors[i % len(colors)]
                ax.plot(rolling_mean.index, rolling_mean.values, color=color, linewidth=2, label=f"{window}期滚动均值")

        # 绘制零线
        ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5)

        # 设置标题和标签
        ax.set_title(f"{factor_name} - 滚动 IC", fontsize=14, fontweight="bold")
        ax.set_xlabel("日期", fontsize=12)
        ax.set_ylabel("IC 值", fontsize=12)

        # 格式化 x 轴日期
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))
        plt.xticks(rotation=45)

        # 添加图例
        ax.legend(loc="upper left", framealpha=0.9)

        # 添加网格
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def plot_monthly_ic_heatmap(
        self, ic_series: pd.Series, factor_name: str = "Factor", figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        绘制月度 IC 热力图

        展示 IC 在不同月份的表现。

        Args:
            ic_series: IC 序列（索引为日期）
            factor_name: 因子名称
            figsize: 图表大小

        Returns:
            matplotlib Figure 对象

        Examples:
            >>> fig = viz.plot_monthly_ic_heatmap(
            ...     metrics['ic'],
            ...     factor_name="MA20"
            ... )
        """
        if figsize is None:
            figsize = (12, 8)

        fig, ax = plt.subplots(figsize=figsize)

        # 将 IC 序列转换为月度数据
        ic_monthly = ic_series.resample("ME").mean()

        # 创建年月矩阵
        ic_monthly.name = "ic"
        df = ic_monthly.to_frame()
        df["year"] = df.index.year
        df["month"] = df.index.month

        # 透视表
        heatmap_data = df.pivot(index="year", columns="month", values="ic")

        # 绘制热力图
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".3f",
            cmap="RdYlGn",
            center=0,
            cbar_kws={"label": "IC 值"},
            ax=ax,
            linewidths=0.5,
        )

        # 设置标题和标签
        ax.set_title(f"{factor_name} - 月度 IC 热力图", fontsize=14, fontweight="bold")
        ax.set_xlabel("月份", fontsize=12)
        ax.set_ylabel("年份", fontsize=12)

        # 设置月份标签
        month_labels = ["1月", "2月", "3月", "4月", "5月", "6月", "7月", "8月", "9月", "10月", "11月", "12月"]
        ax.set_xticklabels(month_labels, rotation=0)

        plt.tight_layout()
        return fig

    def plot_ic_qq(
        self, ic_series: pd.Series, factor_name: str = "Factor", figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """
        绘制 IC Q-Q 图

        检验 IC 是否服从正态分布。

        Args:
            ic_series: IC 序列
            factor_name: 因子名称
            figsize: 图表大小

        Returns:
            matplotlib Figure 对象

        Examples:
            >>> fig = viz.plot_ic_qq(metrics['ic'], factor_name="MA20")
        """
        if figsize is None:
            figsize = (8, 8)

        fig, ax = plt.subplots(figsize=figsize)

        from scipy import stats

        # 绘制 Q-Q 图
        stats.probplot(ic_series.dropna(), dist="norm", plot=ax)

        # 设置标题
        ax.set_title(f"{factor_name} - IC Q-Q 图 (正态分布检验)", fontsize=14, fontweight="bold")
        ax.set_xlabel("理论分位数", fontsize=12)
        ax.set_ylabel("样本分位数", fontsize=12)

        # 添加网格
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    def create_full_report_chart(
        self,
        metrics: Dict[str, Any],
        scenario_results: Dict[str, Dict],
        factor_name: str = "Factor",
        figsize: Optional[Tuple[int, int]] = None,
    ) -> plt.Figure:
        """
        创建完整的分析报告图表组合

        生成包含多个子图的综合图表。

        Args:
            metrics: 性能指标字典
            scenario_results: 策略场景分析结果
            factor_name: 因子名称
            figsize: 图表大小

        Returns:
            matplotlib Figure 对象

        Examples:
            >>> fig = viz.create_full_report_chart(
            ...     metrics,
            ...     scenario_results,
            ...     factor_name="MA20"
            ... )
        """
        if figsize is None:
            figsize = (16, 12)

        fig = plt.figure(figsize=figsize)

        # 创建网格布局
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)

        # 1. IC 时间序列
        ax1 = fig.add_subplot(gs[0, :])
        ic_series = metrics.get("ic")
        if ic_series is not None:
            ax1.plot(ic_series.index, ic_series.values, color=COLOR_SCHEME["secondary"], alpha=0.5)
            if len(ic_series) >= 20:
                rolling_mean = ic_series.rolling(20).mean()
                ax1.plot(rolling_mean.index, rolling_mean.values, color=COLOR_SCHEME["primary"], linewidth=2)
            ax1.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
            ax1.set_title(f"{factor_name} - IC 时间序列", fontweight="bold")
            ax1.set_ylabel("IC 值")
            ax1.grid(True, alpha=0.3)

        # 2. IC 分布
        ax2 = fig.add_subplot(gs[1, 0])
        if ic_series is not None:
            ax2.hist(ic_series.dropna(), bins=30, color=COLOR_SCHEME["secondary"], alpha=0.6, density=True)
            ax2.axvline(ic_series.mean(), color=COLOR_SCHEME["success"], linestyle="--", linewidth=2)
            ax2.set_title("IC 分布", fontweight="bold")
            ax2.set_xlabel("IC 值")
            ax2.set_ylabel("密度")
            ax2.grid(True, alpha=0.3, axis="y")

        # 3. 累计收益对比
        ax3 = fig.add_subplot(gs[1, 1])
        for strategy_key, strategy_data in scenario_results.items():
            if "strategy_returns" in strategy_data:
                returns = strategy_data["strategy_returns"]
                cumulative = (1 + returns).cumprod()
                ax3.plot(cumulative.index, cumulative.values, linewidth=2, label=strategy_key)
        ax3.axhline(y=1, color="gray", linestyle="-", alpha=0.5)
        ax3.set_title("策略累计收益对比", fontweight="bold")
        ax3.set_ylabel("累计收益")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

        # 4. 回撤
        ax4 = fig.add_subplot(gs[2, 0])
        if "long_short_return" in metrics:
            returns = metrics["long_short_return"]
            cumulative = (1 + returns).cumprod()
            running_max = cumulative.expanding().max()
            drawdown = (cumulative - running_max) / running_max
            ax4.fill_between(drawdown.index, drawdown.values, 0, color=COLOR_SCHEME["danger"], alpha=0.3)
            ax4.plot(drawdown.index, drawdown.values, color=COLOR_SCHEME["danger"], linewidth=1.5)
            ax4.set_title("回撤", fontweight="bold")
            ax4.set_ylabel("回撤")
            ax4.grid(True, alpha=0.3)

        # 5. 关键指标文本
        ax5 = fig.add_subplot(gs[2, 1])
        ax5.axis("off")

        # 提取关键指标
        ic_mean = metrics.get("ic_mean", 0)
        icir = metrics.get("icir", 0)
        annual_return = metrics.get("annual_return", 0)
        sharpe = metrics.get("sharpe_ratio", 0)
        max_dd = metrics.get("max_drawdown", 0)

        # 创建指标文本
        metrics_text = f"""
        关键指标
        ═════════════

        IC 均值:      {ic_mean:.4f}
        ICIR:         {icir:.4f}
        年化收益:     {annual_return:.2%}
        夏普比率:     {sharpe:.2f}
        最大回撤:     {max_dd:.2%}
        """

        ax5.text(
            0.1,
            0.5,
            metrics_text,
            fontsize=12,
            family="monospace",
            verticalalignment="center",
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3),
        )

        fig.suptitle(f"{factor_name} - 综合分析报告", fontsize=16, fontweight="bold", y=0.995)

        return fig

    def save_figure(
        self,
        fig: plt.Figure,
        filename: str,
        format: Optional[str] = None,
        dpi: Optional[int] = None,
        bbox_inches: str = "tight",
    ) -> Path:
        """
        保存图表

        Args:
            fig: matplotlib Figure 对象
            filename: 文件名（可包含路径）
            format: 文件格式（png、svg、pdf）
                     如果为 None，则从文件扩展名推断
            dpi: 分辨率（仅用于位图格式）
            bbox_inches: 边界框设置

        Returns:
            保存的文件路径

        Examples:
            >>> fig = viz.plot_ic_timeseries(metrics['ic'])
            >>> path = viz.save_figure(fig, "MA20_ic.png", dpi=300)
        """
        if dpi is None:
            dpi = self.dpi

        # 推断格式
        if format is None:
            ext = Path(filename).suffix.lower()
            format_map = {".png": "png", ".svg": "svg", ".pdf": "pdf", ".jpg": "jpg", ".jpeg": "jpg"}
            format = format_map.get(ext, "png")

        # 完整路径
        filepath = self.output_dir / filename

        # 保存
        fig.savefig(
            filepath,
            format=format,
            dpi=dpi if format in ["png", "jpg", "jpeg"] else None,
            bbox_inches=bbox_inches,
            facecolor="white",
            edgecolor="none",
        )

        # 关闭图形，释放内存
        plt.close(fig)

        return filepath

    def save_all_charts(
        self, metrics: Dict[str, Any], scenario_results: Dict[str, Dict], factor_name: str = "Factor"
    ) -> Dict[str, Path]:
        """
        保存所有图表

        生成并保存所有标准图表。

        Args:
            metrics: 性能指标字典
            scenario_results: 策略场景分析结果
            factor_name: 因子名称

        Returns:
            Dict: 文件名到文件路径的映射

        Examples:
            >>> paths = viz.save_all_charts(metrics, scenario_results, "MA20")
            >>> print(f"已保存 {len(paths)} 个图表")
        """
        paths = {}
        safe_name = factor_name.replace(" ", "_").replace("/", "_")

        # 1. IC 时间序列图
        try:
            if "ic" in metrics and metrics["ic"] is not None:
                fig = self.plot_ic_timeseries(metrics["ic"], factor_name)
                path = self.save_figure(fig, f"{safe_name}_ic_timeseries.png")
                paths["ic_timeseries"] = path
        except Exception as e:
            warnings.warn(f"无法生成 IC 时间序列图: {e}")

        # 2. IC 分布图
        try:
            if "ic" in metrics and metrics["ic"] is not None:
                fig = self.plot_ic_distribution(metrics["ic"], factor_name)
                path = self.save_figure(fig, f"{safe_name}_ic_distribution.png")
                paths["ic_distribution"] = path
        except Exception as e:
            warnings.warn(f"无法生成 IC 分布图: {e}")

        # 3. 策略对比图
        try:
            fig = self.plot_strategy_comparison(scenario_results, factor_name)
            path = self.save_figure(fig, f"{safe_name}_strategy_comparison.png")
            paths["strategy_comparison"] = path
        except Exception as e:
            warnings.warn(f"无法生成策略对比图: {e}")

        # 4. 累计收益图
        try:
            if "bull" in scenario_results:
                fig = self.plot_cumulative_return(scenario_results["bull"]["strategy_returns"], factor_name)
                path = self.save_figure(fig, f"{safe_name}_cumulative_return.png")
                paths["cumulative_return"] = path
        except Exception as e:
            warnings.warn(f"无法生成累计收益图: {e}")

        # 5. 回撤图
        try:
            if "bull" in scenario_results:
                fig = self.plot_drawdown(scenario_results["bull"]["strategy_returns"], factor_name)
                path = self.save_figure(fig, f"{safe_name}_drawdown.png")
                paths["drawdown"] = path
        except Exception as e:
            warnings.warn(f"无法生成回撤图: {e}")

        # 6. 滚动 IC 图
        try:
            if "ic" in metrics and metrics["ic"] is not None:
                fig = self.plot_rolling_ic(metrics["ic"], factor_name)
                path = self.save_figure(fig, f"{safe_name}_rolling_ic.png")
                paths["rolling_ic"] = path
        except Exception as e:
            warnings.warn(f"无法生成滚动 IC 图: {e}")

        # 7. 综合报告图
        try:
            fig = self.create_full_report_chart(metrics, scenario_results, factor_name)
            path = self.save_figure(fig, f"{safe_name}_full_report.png")
            paths["full_report"] = path
        except Exception as e:
            warnings.warn(f"无法生成综合报告图: {e}")

        return paths


# 便捷函数
def create_factor_charts(
    metrics: Dict[str, Any], scenario_results: Dict[str, Dict], factor_name: str, output_dir: str = "./charts", **kwargs
) -> Dict[str, Path]:
    """
    创建因子分析图表的便捷函数

    Args:
        metrics: 性能指标字典
        scenario_results: 策略场景分析结果
        factor_name: 因子名称
        output_dir: 输出目录
        **kwargs: 其他参数传递给 Visualizer

    Returns:
        Dict: 文件名到文件路径的映射

    Examples:
        >>> from report.visualizer import create_factor_charts
        >>>
        >>> # 快速生成所有图表
        >>> paths = create_factor_charts(
        ...     metrics=metrics,
        ...     scenario_results=scenario_results,
        ...     factor_name="MA20"
        ... )
        >>>
        >>> # 查看生成的图表
        >>> for name, path in paths.items():
        ...     print(f"{name}: {path}")
    """
    viz = Visualizer(output_dir=output_dir, **kwargs)
    return viz.save_all_charts(metrics, scenario_results, factor_name)


if __name__ == "__main__":
    # 示例：使用可视化工具
    print("=" * 60)
    print("可视化工具示例")
    print("=" * 60)

    # 创建模拟数据
    print("\n生成模拟数据...")
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", "2020-12-31", freq="D")

    # 模拟 IC 序列
    ic_series = pd.Series(np.random.randn(366) * 0.05 + 0.03, index=dates)

    # 模拟收益率序列
    returns = pd.Series(np.random.randn(366) * 0.01 + 0.001, index=dates)

    # 模拟策略结果
    scenario_results = {
        "bull": {"strategy_returns": returns, "annual_return": 0.12, "sharpe_ratio": 2.1},
        "bear": {"strategy_returns": -returns * 0.5, "annual_return": -0.02, "sharpe_ratio": -0.5},
    }

    # 模拟性能指标
    metrics = {
        "ic": ic_series,
        "ic_mean": 0.035,
        "icir": 0.673,
        "annual_return": 0.085,
        "sharpe_ratio": 1.85,
        "max_drawdown": -0.12,
        "long_short_return": returns,
    }

    # 创建可视化工具
    viz = Visualizer(output_dir="./charts")

    # 生成并保存所有图表
    print("\n生成图表...")
    paths = viz.save_all_charts(metrics=metrics, scenario_results=scenario_results, factor_name="MA20")

    # 打印结果
    print("\n已生成的图表:")
    for name, path in paths.items():
        print(f"  - {name}: {path}")

    # 生成单个图表示例
    print("\n生成单个图表示例...")
    fig = viz.plot_ic_timeseries(ic_series, factor_name="MA20")
    path = viz.save_figure(fig, "MA20_ic_timeseries_example.png", dpi=300)
    print(f"IC 时间序列图已保存到: {path}")

    print("\n完成!")
