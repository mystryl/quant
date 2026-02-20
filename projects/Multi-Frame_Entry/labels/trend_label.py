"""
趋势标签计算模块

基于 60min 数据计算未来收益率，并生成三分类趋势标签：
- 1 = 上涨趋势
- 0 = 震荡
- -1 = 下跌趋势

重要：严格防止未来函数污染
"""
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, List
import logging

logger = logging.getLogger(__name__)


class TrendLabelGenerator:
    """趋势标签生成器"""

    def __init__(
        self,
        future_windows: List[int] = None,
        up_threshold: float = 0.003,
        down_threshold: float = -0.003
    ):
        """
        初始化标签生成器

        Args:
            future_windows: 未来K线窗口列表，默认 [5, 10, 15, 20, 30, 40]
            up_threshold: 上涨阈值（收益率 > 此值标记为上涨）
            down_threshold: 下跌阈值（收益率 < 此值标记为下跌）
        """
        self.future_windows = future_windows or [5, 10, 15, 20, 30, 40]
        self.up_threshold = up_threshold
        self.down_threshold = down_threshold

        logger.info(f"趋势标签生成器初始化:")
        logger.info(f"  测试窗口: {self.future_windows} 根K线")
        logger.info(f"  上涨阈值: {up_threshold:.2%}")
        logger.info(f"  下跌阈值: {down_threshold:.2%}")

    def calculate_future_returns(
        self,
        df: pd.DataFrame,
        window: int,
        price_col: str = 'close'
    ) -> pd.Series:
        """
        计算未来 N 根 K 线的收益率

        重要：使用 shift(-window) 确保使用未来数据

        Args:
            df: 价格数据（index为datetime）
            window: 未来K线数量
            price_col: 价格列名

        Returns:
            未来收益率序列
        """
        close_price = df[price_col]

        # 计算未来收益率：未来价格 / 当前价格 - 1
        # shift(-window) 表示取未来 window 期的值
        future_price = close_price.shift(-window)
        future_return = (future_price - close_price) / close_price

        logger.info(f"  计算 {window} 根K线未来收益率:")
        logger.info(f"    有效样本: {future_return.notna().sum()} / {len(future_return)}")
        logger.info(f"    平均收益率: {future_return.mean():.4f}")
        logger.info(f"    标准差: {future_return.std():.4f}")

        return future_return

    def generate_labels(
        self,
        future_return: pd.Series,
        window: int
    ) -> pd.Series:
        """
        根据未来收益率生成三分类标签

        Args:
            future_return: 未来收益率序列
            window: 窗口大小（用于日志）

        Returns:
            标签序列（1=上涨, 0=震荡, -1=下跌）
        """
        # 初始化为震荡
        labels = pd.Series(0, index=future_return.index)

        # 上涨标签
        labels[future_return > self.up_threshold] = 1

        # 下跌标签
        labels[future_return < self.down_threshold] = -1

        # 统计标签分布
        label_counts = labels.value_counts()
        total = len(labels[labels.notna()])

        logger.info(f"  {window}根K线标签分布:")
        for label_type in [1, 0, -1]:
            count = label_counts.get(label_type, 0)
            pct = count / total * 100 if total > 0 else 0
            label_name = {1: '上涨', 0: '震荡', -1: '下跌'}[label_type]
            logger.info(f"    {label_name}: {count} ({pct:.1f}%)")

        return labels

    def analyze_window(
        self,
        df: pd.DataFrame,
        window: int,
        price_col: str = 'close'
    ) -> Dict:
        """
        分析单个窗口的表现

        Args:
            df: 价格数据
            window: 未来K线数量
            price_col: 价格列名

        Returns:
            分析结果字典
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"分析窗口: {window} 根K线")
        logger.info(f"{'='*60}")

        # 计算未来收益率
        future_return = self.calculate_future_returns(df, window, price_col)

        # 生成标签
        labels = self.generate_labels(future_return, window)

        # 统计指标
        valid_return = future_return[future_return.notna()]
        valid_labels = labels[labels.notna()]

        analysis = {
            'window': window,
            'window_hours': window,  # 60min一根，所以数值相等
            'window_days': window / 13.5,  # 假设每个交易日13.5根60minK线

            # 样本统计
            'total_samples': len(df),
            'valid_samples': len(valid_return),
            'sample_coverage': len(valid_return) / len(df),

            # 收益率统计
            'mean_return': valid_return.mean(),
            'std_return': valid_return.std(),
            'min_return': valid_return.min(),
            'max_return': valid_return.max(),
            'median_return': valid_return.median(),

            # 信噪比
            'signal_to_noise': abs(valid_return.mean()) / valid_return.std() if valid_return.std() > 0 else 0,

            # 标签分布
            'label_up_count': (valid_labels == 1).sum(),
            'label_range_count': (valid_labels == 0).sum(),
            'label_down_count': (valid_labels == -1).sum(),
            'label_up_pct': (valid_labels == 1).sum() / len(valid_labels) * 100 if len(valid_labels) > 0 else 0,
            'label_range_pct': (valid_labels == 0).sum() / len(valid_labels) * 100 if len(valid_labels) > 0 else 0,
            'label_down_pct': (valid_labels == -1).sum() / len(valid_labels) * 100 if len(valid_labels) > 0 else 0,

            # 标签平稳性（切换频率）
            'label_changes': valid_labels.astype(float).diff().abs().sum(),

            # 原始数据
            'future_return': future_return,
            'labels': labels
        }

        # 计算平衡性得分（越接近33.3%越好）
        target_pct = 33.33
        up_diff = abs(analysis['label_up_pct'] - target_pct)
        range_diff = abs(analysis['label_range_pct'] - target_pct)
        down_diff = abs(analysis['label_down_pct'] - target_pct)
        analysis['balance_score'] = 100 - (up_diff + range_diff + down_diff) / 3

        logger.info(f"\n  关键指标:")
        logger.info(f"    信噪比: {analysis['signal_to_noise']:.4f}")
        logger.info(f"    平衡性得分: {analysis['balance_score']:.2f}/100")
        logger.info(f"    标签切换次数: {analysis['label_changes']}")

        return analysis

    def compare_windows(
        self,
        df: pd.DataFrame,
        price_col: str = 'close'
    ) -> pd.DataFrame:
        """
        对比分析多个窗口

        Args:
            df: 价格数据
            price_col: 价格列名

        Returns:
            对比结果DataFrame
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"多窗口对比分析")
        logger.info(f"{'='*60}")

        results = []
        for window in self.future_windows:
            try:
                analysis = self.analyze_window(df, window, price_col)
                results.append(analysis)
            except Exception as e:
                logger.error(f"分析窗口 {window} 时出错: {e}")
                continue

        # 转换为DataFrame
        df_results = pd.DataFrame(results)

        # 保存详细结果
        logger.info(f"\n{'='*60}")
        logger.info(f"窗口对比总结")
        logger.info(f"{'='*60}")

        # 打印关键指标对比表
        display_cols = [
            'window', 'window_hours', 'window_days',
            'mean_return', 'std_return', 'signal_to_noise',
            'balance_score', 'label_changes',
            'label_up_pct', 'label_range_pct', 'label_down_pct'
        ]

        logger.info(f"\n关键指标对比:")
        logger.info(df_results[display_cols].to_string())

        # 计算综合得分（越高越好）
        # 权重：信噪比30%，平衡性40%，平稳性30%
        df_results['overall_score'] = (
            df_results['signal_to_noise'] * 30 +
            df_results['balance_score'] * 0.4 +
            (1 - df_results['label_changes'] / df_results['valid_samples']) * 100 * 30
        )

        logger.info(f"\n综合得分排名:")
        df_results_sorted = df_results.sort_values('overall_score', ascending=False)
        for idx, row in df_results_sorted.iterrows():
            logger.info(f"  {int(row['window'])}根K线: {row['overall_score']:.2f}分")

        return df_results_sorted

    def save_results(
        self,
        df_results: pd.DataFrame,
        output_dir: Path
    ):
        """
        保存分析结果到文件

        Args:
            df_results: 对比结果DataFrame
            output_dir: 输出目录
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 1. 保存对比表格
        table_file = output_dir / 'window_comparison.csv'
        df_results.to_csv(table_file, index=False)
        logger.info(f"\n✓ 对比表格已保存: {table_file}")

        # 2. 保存标签数据（每个窗口）
        labels_dir = output_dir / 'labels_raw'
        labels_dir.mkdir(parents=True, exist_ok=True)

        for _, row in df_results.iterrows():
            window = int(row['window'])
            labels = row['labels']

            # 只保存有效标签
            valid_labels = labels[labels.notna()]
            labels_file = labels_dir / f'labels_{window}bars.csv'
            valid_labels.to_csv(labels_file)
            logger.info(f"✓ {window}根K线标签已保存: {labels_file}")

        # 3. 保存未来收益率数据
        returns_dir = output_dir / 'returns_raw'
        returns_dir.mkdir(parents=True, exist_ok=True)

        for _, row in df_results.iterrows():
            window = int(row['window'])
            future_return = row['future_return']

            # 只保存有效收益率
            valid_return = future_return[future_return.notna()]
            return_file = returns_dir / f'return_{window}bars.csv'
            valid_return.to_csv(return_file)
            logger.info(f"✓ {window}根K线收益率已保存: {return_file}")

        logger.info(f"\n所有结果已保存到: {output_dir}")

    def recommend_window(self, df_results: pd.DataFrame) -> Dict:
        """
        根据分析结果推荐最优窗口

        Args:
            df_results: 对比结果DataFrame

        Returns:
            推荐结果字典
        """
        best = df_results.iloc[0]

        recommendation = {
            'recommended_window': int(best['window']),
            'recommended_hours': float(best['window_hours']),
            'recommended_days': float(best['window_days']),
            'overall_score': float(best['overall_score']),
            'reasoning': [
                f"综合得分最高: {best['overall_score']:.2f}分",
                f"信噪比: {best['signal_to_noise']:.4f}",
                f"平衡性得分: {best['balance_score']:.2f}/100",
                f"标签分布: 上涨{best['label_up_pct']:.1f}%, 震荡{best['label_range_pct']:.1f}%, 下跌{best['label_down_pct']:.1f}%"
            ]
        }

        logger.info(f"\n{'='*60}")
        logger.info(f"推荐窗口: {recommendation['recommended_window']} 根K线")
        logger.info(f"{'='*60}")
        for reason in recommendation['reasoning']:
            logger.info(f"  - {reason}")

        return recommendation
