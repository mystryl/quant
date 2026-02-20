"""
二分类趋势标签生成模块

简化为二分类问题：
- 1 = 有趋势（上涨或下跌超过阈值）
- 0 = 震荡（价格变化在阈值内）

优势：
- 减少类别数量，降低模型复杂度
- 更容易预测"是否有趋势"而非"趋势方向"
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# 二分类标签配置
BINARY_LABEL_CONFIG = {
    'window': 20,          # 前瞻窗口（根K线）
    'threshold': 0.005,    # 趋势阈值 ±0.5%
    'data_file': Path('/Users/mystryl/Documents/Quant/data/qlib_data_multi_freq/instruments/60min/$close/HC8888.XSGE.csv'),
    'output_file': Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data/labels/binary_labels.csv')
}


def generate_binary_labels(
    window: int = 20,
    threshold: float = 0.005,
    data_file: Path = None,
    output_file: Path = None
) -> pd.DataFrame:
    """
    生成二分类趋势标签

    Args:
        window: 前瞻窗口（根K线）
        threshold: 趋势阈值（如0.005表示±0.5%）
        data_file: 60min收盘价数据文件
        output_file: 输出文件路径

    Returns:
        包含标签的DataFrame
    """
    if data_file is None:
        data_file = BINARY_LABEL_CONFIG['data_file']
    if output_file is None:
        output_file = BINARY_LABEL_CONFIG['output_file']

    logger.info("="*60)
    logger.info("生成二分类趋势标签")
    logger.info("="*60)
    logger.info(f"参数:")
    logger.info(f"  前瞻窗口: {window} 根K线")
    logger.info(f"  趋势阈值: ±{threshold*100:.2f}%")

    # 加载60min收盘价数据
    logger.info(f"\n加载数据: {data_file}")
    df_price = pd.read_csv(data_file, index_col=0, parse_dates=True)
    logger.info(f"  数据形状: {df_price.shape}")
    logger.info(f"  时间范围: {df_price.index.min()} ~ {df_price.index.max()}")

    # 获取收盘价
    close_price = df_price.iloc[:, 0]  # 第一列是收盘价

    # 计算未来收益率（严格防未来函数）
    future_price = close_price.shift(-window)
    future_return = (future_price - close_price) / close_price

    # 生成二分类标签
    # 1 = 有趋势（涨幅或跌幅超过阈值）
    # 0 = 震荡（涨跌幅在阈值内）
    labels = pd.Series(np.nan, index=df_price.index)

    # 有趋势：上涨或下跌超过阈值
    has_trend = (future_return.abs() > threshold) & (future_return.notna())
    labels[has_trend] = 1

    # 震荡：涨跌幅在阈值内
    is_range = (future_return.abs() <= threshold) & (future_return.notna())
    labels[is_range] = 0

    # 创建结果DataFrame
    df_result = pd.DataFrame({
        'datetime': df_price.index,
        'close': close_price.values,
        'future_return': future_return.values,
        'trend_label': labels.values
    })

    # 统计标签分布
    label_counts = df_result['trend_label'].value_counts().sort_index()
    logger.info(f"\n标签分布:")
    for label, count in label_counts.items():
        if pd.notna(label):
            label_name = {0: '震荡', 1: '有趋势'}[int(label)]
            logger.info(f"  {label_name}: {count} ({count/df_result['trend_label'].notna().sum()*100:.1f}%)")

    # 验证最后window个样本
    logger.info(f"\n防未来函数验证:")
    logger.info(f"  最后{window}个样本应全部为NaN")
    last_window = df_result['trend_label'].iloc[-window:]
    logger.info(f"  实际: {last_window.isna().sum()}/{window} 为NaN")

    if last_window.isna().sum() != window:
        logger.error("❌ 防未来函数验证失败！")
        raise ValueError("最后window个标签应全部为NaN")

    logger.info("✓ 防未来函数验证通过")

    # 保存结果
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(output_file, index=False)
    logger.info(f"\n✓ 结果已保存: {output_file}")

    # 额外统计
    valid_labels = df_result['trend_label'].dropna()
    logger.info(f"\n数据统计:")
    logger.info(f"  总样本数: {len(df_result)}")
    logger.info(f"  有效标签数: {len(valid_labels)}")
    logger.info(f"  无效样本数: {df_result['trend_label'].isna().sum()}")

    logger.info(f"\n{'='*60}")
    logger.info(f"二分类标签生成完成！")
    logger.info(f"{'='*60}")

    return df_result


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 生成二分类标签
    df = generate_binary_labels(
        window=20,
        threshold=0.005,  # ±0.5%
        output_file=Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data/labels/binary_labels.csv')
    )

    print(f"\n最终数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
