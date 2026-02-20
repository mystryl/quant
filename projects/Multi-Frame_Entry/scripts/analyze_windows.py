#!/usr/bin/env python3
"""
窗口对比分析脚本

直接从CSV文件加载数据，对比分析多个前瞻窗口的表现，
找出最优的标签生成窗口。
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from labels.trend_label import TrendLabelGenerator
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False


def load_60min_data_from_csv(
    data_dir: Path = None,
    instrument: str = 'HC8888.XSGE'
) -> pd.DataFrame:
    """
    从CSV文件加载 60min 数据

    Args:
        data_dir: 数据目录
        instrument: 合约代码

    Returns:
        DataFrame with columns: datetime, open, high, low, close, volume
    """
    if data_dir is None:
        data_dir = Path('/Users/mystryl/Documents/Quant/data/qlib_data_multi_freq')

    logger.info(f"从CSV加载 {instrument} 60min 数据...")

    # 读取各个字段
    data_dict = {}
    freq_dir = data_dir / 'instruments' / '60min'

    for field in ['open', 'high', 'low', 'close', 'volume']:
        field_file = freq_dir / f'${field}' / f'{instrument}.csv'

        if not field_file.exists():
            logger.error(f"文件不存在: {field_file}")
            raise FileNotFoundError(f"无法找到 {field} 数据文件")

        df_field = pd.read_csv(field_file, index_col=0, parse_dates=True)
        data_dict[field] = df_field.iloc[:, 0]  # 获取第一列
        logger.info(f"  读取 {field}: {len(df_field)} 行")

    # 合并为单个DataFrame
    df = pd.DataFrame(data_dict)
    df = df.reset_index()
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    # 过滤时间范围（2022-2025）
    df = df[(df['datetime'] >= '2022-01-01') & (df['datetime'] <= '2025-12-31')]

    logger.info(f"✓ 数据加载完成: {df.shape}")
    logger.info(f"  时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")
    logger.info(f"  总行数: {len(df)}")

    return df


def create_visualizations(
    df_results: pd.DataFrame,
    df_data: pd.DataFrame,
    output_dir: Path
):
    """
    创建可视化图表

    Args:
        df_results: 窗口对比结果
        df_data: 原始数据
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"\n生成可视化图表...")

    # 1. 综合得分对比
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # 1.1 综合得分
    ax = axes[0, 0]
    bars = ax.bar(df_results['window'], df_results['overall_score'], color='steelblue')
    ax.set_xlabel('窗口（根K线）', fontsize=12)
    ax.set_ylabel('综合得分', fontsize=12)
    ax.set_title('各窗口综合得分对比', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    # 标注数值
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # 1.2 信噪比
    ax = axes[0, 1]
    bars = ax.bar(df_results['window'], df_results['signal_to_noise'], color='coral')
    ax.set_xlabel('窗口（根K线）', fontsize=12)
    ax.set_ylabel('信噪比', fontsize=12)
    ax.set_title('各窗口信噪比对比', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9)

    # 1.3 平衡性得分
    ax = axes[1, 0]
    bars = ax.bar(df_results['window'], df_results['balance_score'], color='lightgreen')
    ax.set_xlabel('窗口（根K线）', fontsize=12)
    ax.set_ylabel('平衡性得分', fontsize=12)
    ax.set_title('各窗口标签平衡性对比', fontsize=14, fontweight='bold')
    ax.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='良好线(80分)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=9)

    # 1.4 标签分布堆叠柱状图
    ax = axes[1, 1]
    x = np.arange(len(df_results))
    width = 0.6

    ax.bar(x, df_results['label_up_pct'], width, label='上涨', color='#2ca02c')
    ax.bar(x, df_results['label_range_pct'], width,
           bottom=df_results['label_up_pct'], label='震荡', color='gray')
    ax.bar(x, df_results['label_down_pct'], width,
           bottom=df_results['label_up_pct'] + df_results['label_range_pct'],
           label='下跌', color='#d62728')

    ax.set_xlabel('窗口（根K线）', fontsize=12)
    ax.set_ylabel('标签占比 (%)', fontsize=12)
    ax.set_title('各窗口标签分布对比', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(df_results['window'])
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig1 = output_dir / 'window_comparison_overview.png'
    plt.savefig(fig1, dpi=150, bbox_inches='tight')
    logger.info(f"✓ 图表已保存: {fig1}")
    plt.close()

    # 2. 详细对比雷达图
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # 选择前3个窗口进行对比
    top3 = df_results.head(3)

    categories = ['信噪比\n(归一化)', '平衡性\n(归一化)', '平稳性\n(归一化)',
                  '样本覆盖\n(归一化)', '收益强度\n(归一化)']
    N = len(categories)

    angles = [n / float(N) * 2 * np.pi for n in range(N)]
    angles += angles[:1]

    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

    for idx, (_, row) in enumerate(top3.iterrows()):
        # 归一化数据到 0-1
        values = [
            row['signal_to_noise'] / df_results['signal_to_noise'].max(),
            row['balance_score'] / 100,
            (1 - row['label_changes'] / row['valid_samples']),
            row['sample_coverage'],
            abs(row['mean_return']) / df_results['mean_return'].abs().max()
        ]
        values += values[:1]

        ax = plt.subplot(111, projection='polar')
        ax.plot(angles, values, 'o-', linewidth=2,
                label=f"{int(row['window'])}根K线", color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories, fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title('Top 3 窗口多维度对比', fontsize=14, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.tight_layout()
    fig2 = output_dir / 'window_radar_chart.png'
    plt.savefig(fig2, dpi=150, bbox_inches='tight')
    logger.info(f"✓ 雷达图已保存: {fig2}")
    plt.close()

    # 3. 收益率分布对比
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (_, row) in enumerate(df_results.iterrows()):
        if idx >= 6:
            break

        ax = axes[idx]
        future_return = row['future_return']
        valid_return = future_return[future_return.notna()]

        # 绘制直方图
        ax.hist(valid_return, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='零线')
        ax.axvline(x=0.003, color='green', linestyle=':', linewidth=1.5, label='上涨阈值')
        ax.axvline(x=-0.003, color='orange', linestyle=':', linewidth=1.5, label='下跌阈值')

        ax.set_xlabel('收益率', fontsize=10)
        ax.set_ylabel('频数', fontsize=10)
        ax.set_title(f'{int(row["window"])}根K线 - 收益率分布\n(均值={row["mean_return"]:.4f}, 标准差={row["std_return"]:.4f})',
                     fontsize=11, fontweight='bold')
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig3 = output_dir / 'return_distribution_comparison.png'
    plt.savefig(fig3, dpi=150, bbox_inches='tight')
    logger.info(f"✓ 收益率分布图已保存: {fig3}")
    plt.close()

    logger.info(f"✓ 所有图表已保存到: {output_dir}")


def create_summary_report(
    df_results: pd.DataFrame,
    recommendation: dict,
    output_dir: Path
):
    """
    创建总结报告（Markdown格式）

    Args:
        df_results: 窗口对比结果
        recommendation: 推荐结果
        output_dir: 输出目录
    """
    output_dir = Path(output_dir)

    report = f"""# 窗口对比分析报告

## 分析概况

- **测试窗口**: {', '.join([f"{int(w)}根" for w in df_results['window']])}
- **合约**: HC8888.XSGE (热卷连续合约)
- **数据频率**: 60min
- **阈值设置**: 上涨 > 0.3%, 下跌 < -0.3%
- **分析时间**: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 推荐结果 🎯

**最优窗口: {recommendation['recommended_window']} 根K线**

**原因:**
"""
    for reason in recommendation['reasoning']:
        report += f"- {reason}\n"

    report += f"""

**时间跨度**: {recommendation['recommended_hours']:.1f}小时 ({recommendation['recommended_days']:.2f}个交易日)

---

## 详细对比表

| 排名 | 窗口(根) | 时间(小时) | 时间(天) | 综合得分 | 信噪比 | 平衡性 | 平稳性 | 标签分布(涨/震/跌) |
|------|----------|-----------|---------|---------|--------|--------|--------|-------------------|
"""
    for idx, (_, row) in enumerate(df_results.head(10).iterrows(), 1):
        report += f"| {idx} | {int(row['window'])} | {row['window_hours']:.1f} | {row['window_days']:.2f} | {row['overall_score']:.2f} | {row['signal_to_noise']:.4f} | {row['balance_score']:.1f} | {row['label_changes']/row['valid_samples']*100:.2f}% | {row['label_up_pct']:.1f}% / {row['label_range_pct']:.1f}% / {row['label_down_pct']:.1f}% |\n"

    report += """

---

## 关键发现

### 1. 信噪比对比

信噪比越高，说明收益率信号越强，相对噪音越小。

- **最高信噪比**: {int(df_results.loc[df_results['signal_to_noise'].idxmax(), 'window'])}根K线 ({df_results['signal_to_noise'].max():.4f})
- **最低信噪比**: {int(df_results.loc[df_results['signal_to_noise'].idxmin(), 'window'])}根K线 ({df_results['signal_to_noise'].min():.4f})

### 2. 标签平衡性

平衡性得分越高，说明三类标签分布越均衡（理想值：100分）

- **最平衡**: {int(df_results.loc[df_results['balance_score'].idxmax(), 'window'])}根K线 ({df_results['balance_score'].max():.1f}分)
- **最不平衡**: {int(df_results.loc[df_results['balance_score'].idxmin(), 'window'])}根K线 ({df_results['balance_score'].min():.1f}分)

### 3. 标签平稳性

切换频率越低，说明标签越平稳，不易频繁变化。

- **最平稳**: {int(df_results.loc[df_results['label_changes'].idxmin(), 'window'])}根K线 ({df_results['label_changes'].min():.0f}次)
- **最不稳定**: {int(df_results.loc[df_results['label_changes'].idxmax(), 'window'])}根K线 ({df_results['label_changes'].max():.0f}次)

---

## 使用建议

1. **推荐使用 {recommendation['recommended_window']} 根K线窗口**进行趋势标签生成
2. 该窗口在信噪比、平衡性和平稳性之间达到最佳平衡
3. 后续特征工程和模型训练将基于此窗口

---

## 数据文件

所有原始数据已保存在 `data/labels/` 目录：

- `window_comparison.csv` - 完整对比表格
- `labels_raw/labels_*bars.csv` - 各窗口标签数据
- `returns_raw/return_*bars.csv` - 各窗口收益率数据
- `*.png` - 可视化图表

---

*本报告由窗口对比分析模块自动生成*
"""

    report_file = output_dir / 'ANALYSIS_REPORT.md'
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write(report)

    logger.info(f"✓ 分析报告已保存: {report_file}")


def main():
    """主函数"""
    logger.info("="*60)
    logger.info("窗口对比分析开始")
    logger.info("="*60)

    # 路径配置
    output_dir = project_root / 'data' / 'labels'

    # 1. 加载 60min 数据（直接从CSV）
    df = load_60min_data_from_csv(
        data_dir=Path('/Users/mystryl/Documents/Quant/data/qlib_data_multi_freq'),
        instrument='HC8888.XSGE'
    )

    # 2. 创建标签生成器
    generator = TrendLabelGenerator(
        future_windows=[5, 10, 15, 20, 30, 40],
        up_threshold=0.003,
        down_threshold=-0.003
    )

    # 3. 对比分析多个窗口
    df_results = generator.compare_windows(
        df=df,
        price_col='close'
    )

    # 4. 保存结果
    generator.save_results(df_results, output_dir)

    # 5. 生成推荐
    recommendation = generator.recommend_window(df_results)

    # 6. 创建可视化
    create_visualizations(df_results, df, output_dir)

    # 7. 生成报告
    create_summary_report(df_results, recommendation, output_dir)

    logger.info("\n" + "="*60)
    logger.info("窗口对比分析完成！")
    logger.info("="*60)
    logger.info(f"\n推荐窗口: {recommendation['recommended_window']} 根K线")
    logger.info(f"结果目录: {output_dir}")


if __name__ == '__main__':
    main()
