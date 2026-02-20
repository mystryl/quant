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
    'window': 20,
    'sigma_threshold': 1.5,  # 标准差倍数阈值（波动率归一化）
    'data_file': Path('/Users/mystryl/Documents/Quant/data/qlib_data_multi_freq/instruments/60min/$close/HC8888.XSGE.csv'),
    'output_file': Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data/labels/binary_labels.csv')
}


def generate_binary_labels(
    window: int = 20,
    sigma_threshold: float = 1.5,
    data_file: Path = None,
    output_file: Path = None
) -> pd.DataFrame:
    """
    生成二分类趋势标签（波动率归一化版本）

    使用波动率归一化收益定义趋势，使不同时期数据可比。
    标签定义：|return/σ| > 1.5σ 为有趋势，否则为震荡。

    Args:
        window: 前瞻窗口（根K线）
        sigma_threshold: 标准差倍数阈值（如1.5表示1.5σ）
        data_file: 60min收盘价数据文件
        output_file: 输出文件路径

    Returns:
        包含标签、归一化收益和波动率的DataFrame
    """
    if data_file is None:
        data_file = BINARY_LABEL_CONFIG['data_file']
    if output_file is None:
        output_file = BINARY_LABEL_CONFIG['output_file']

    logger.info("="*60)
    logger.info("生成二分类趋势标签（波动率归一化版本）")
    logger.info("="*60)
    logger.info(f"参数:")
    logger.info(f"  前瞻窗口: {window} 根K线")
    logger.info(f"  趋势阈值: ±{sigma_threshold}σ (波动率归一化)")

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

    # 计算滚动波动率（百分比标准差）
    # 方法：先计算价格的百分比变化，再求标准差
    price_pct_change = close_price.pct_change()
    rolling_volatility_1bar = price_pct_change.rolling(window=window).std()

    # 缩放1bar波动率到20bar波动率（假设独立，sqrt(20)倍）
    rolling_volatility_20bar = rolling_volatility_1bar * np.sqrt(window)

    # 避免除零
    rolling_volatility_20bar = rolling_volatility_20bar.replace(0, np.nan)

    # 波动率归一化收益
    normalized_return = future_return / rolling_volatility_20bar
    normalized_return = normalized_return.clip(-10, 10)  # 限制极值

    # 生成二分类标签（使用标准差倍数阈值）
    # 1 = 有趋势（归一化收益超过1.5倍标准差）
    # 0 = 震荡（归一化收益在1.5倍标准差内）
    labels = pd.Series(np.nan, index=df_price.index)

    # 有趋势：归一化收益超过阈值
    has_trend = (normalized_return.abs() > sigma_threshold) & (normalized_return.notna())
    labels[has_trend] = 1

    # 震荡：归一化收益在阈值内
    is_range = (normalized_return.abs() <= sigma_threshold) & (normalized_return.notna())
    labels[is_range] = 0

    # 创建结果DataFrame
    df_result = pd.DataFrame({
        'datetime': df_price.index,
        'close': close_price.values,
        'future_return': future_return.values,
        'normalized_return': normalized_return.values,  # 归一化收益
        'rolling_volatility': rolling_volatility_20bar.values,  # 滚动波动率（20bar缩放）
        'trend_label': labels.values
    })

    # 统计标签分布
    label_counts = df_result['trend_label'].value_counts().sort_index()
    logger.info(f"\n标签分布:")
    for label, count in label_counts.items():
        if pd.notna(label):
            label_name = {0: '震荡', 1: '有趋势'}[int(label)]
            logger.info(f"  {label_name}: {count} ({count/df_result['trend_label'].notna().sum()*100:.1f}%)")

    # 输出归一化收益统计
    logger.info(f"\n归一化收益统计:")
    logger.info(f"  均值: {df_result['normalized_return'].mean():.4f}")
    logger.info(f"  标准差: {df_result['normalized_return'].std():.4f}")
    logger.info(f"  最小值: {df_result['normalized_return'].min():.4f}")
    logger.info(f"  最大值: {df_result['normalized_return'].max():.4f}")
    logger.info(f"  |R| > {1.5}σ: {(df_result['normalized_return'].abs() > 1.5).sum()} ({(df_result['normalized_return'].abs() > 1.5).sum()/len(df_result)*100:.1f}%)")

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
        sigma_threshold=1.5,  # 1.5倍标准差
        output_file=Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data/labels/binary_labels.csv')
    )

    print(f"\n最终数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
