"""
波动率Regime分类模块

使用10根K线窗口预测未来波动率状态：
- 1 = 高波动Regime（未来波动率 > 历史中位数）
- 0 = 低波动Regime（未来波动率 ≤ 历史中位数）

思路：
1. 先识别市场波动率状态
2. 在高波动Regime中训练趋势模型（趋势更容易在波动中形成）
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# 波动率Regime配置
VOLATILITY_REGIME_CONFIG = {
    'window': 10,  # 10根K线窗口
    'data_file': Path('/Users/mystryl/Documents/Quant/data/qlib_data_multi_freq/instruments/60min/$close/HC8888.XSGE.csv'),
    'output_file': Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data/labels/volatility_regime_labels.csv')
}


def generate_volatility_regime_labels(
    window: int = 10,
    data_file: Path = None,
    output_file: Path = None,
    train_start_year: int = 2020
) -> pd.DataFrame:
    """
    生成波动率Regime标签

    计算未来10根K线的波动率，与历史中位数比较，识别高/低波动状态。

    Args:
        window: 前瞻窗口（根K线）
        data_file: 60min收盘价数据文件
        output_file: 输出文件路径
        train_start_year: 训练起始年份（用于计算历史中位数）

    Returns:
        包含波动率Regime标签的DataFrame
    """
    if data_file is None:
        data_file = VOLATILITY_REGIME_CONFIG['data_file']
    if output_file is None:
        output_file = VOLATILITY_REGIME_CONFIG['output_file']

    logger.info("="*60)
    logger.info("生成波动率Regime标签")
    logger.info("="*60)
    logger.info(f"参数:")
    logger.info(f"  前瞻窗口: {window} 根K线")
    logger.info(f"  训练起始年份: {train_start_year}")

    # 加载60min收盘价数据
    logger.info(f"\n加载数据: {data_file}")
    df_price = pd.read_csv(data_file, index_col=0, parse_dates=True)
    logger.info(f"  数据形状: {df_price.shape}")
    logger.info(f"  时间范围: {df_price.index.min()} ~ {df_price.index.max()}")

    # 获取收盘价
    close_price = df_price.iloc[:, 0]

    # 计算历史波动率中位数（用于定义Regime阈值）
    # 只使用训练起始年份之前的数据计算中位数
    train_start_mask = df_price.index.year >= train_start_year
    pre_train_data = df_price[~train_start_mask]

    logger.info(f"\n计算历史波动率中位数:")
    logger.info(f"  训练起始年之前数据: {len(pre_train_data)} 样本")
    logger.info(f"  时间范围: {pre_train_data.index.min()} ~ {pre_train_data.index.max()}")

    if len(pre_train_data) < window * 2:
        logger.warning(f"  ⚠️  训练前数据不足({len(pre_train_data)} < {window*2})，使用全部数据计算中位数")
        pre_train_data = df_price

    # 计算1分钟收益率的历史波动率（滚动标准差）
    pre_train_returns = pre_train_data.iloc[:, 0].pct_change()
    historical_volatility_median = pre_train_returns.rolling(window=window).std().median()

    logger.info(f"  历史波动率中位数: {historical_volatility_median:.6f}")

    # 计算未来波动率（未来window根K线的收益率标准差）
    logger.info(f"\n计算未来{window}根K线波动率...")

    future_volatility = pd.Series(np.nan, index=df_price.index)

    for i in range(len(df_price) - window):
        # 获取未来window根K线的收盘价
        future_prices = close_price.iloc[i+1:i+1+window]

        if len(future_prices) == window:
            # 计算未来收益率的标准差（已实现波动率）
            future_returns = future_prices.pct_change().dropna()
            if len(future_returns) > 0:
                future_volatility.iloc[i] = future_returns.std()

    # 生成Regime标签
    # 1 = 高波动Regime（未来波动率 > 历史中位数）
    # 0 = 低波动Regime（未来波动率 ≤ 历史中位数）
    labels = pd.Series(np.nan, index=df_price.index)

    high_vol = (future_volatility > historical_volatility_median) & (future_volatility.notna())
    labels[high_vol] = 1

    low_vol = (future_volatility <= historical_volatility_median) & (future_volatility.notna())
    labels[low_vol] = 0

    # 创建结果DataFrame
    df_result = pd.DataFrame({
        'datetime': df_price.index,
        'close': close_price.values,
        'future_volatility': future_volatility.values,
        'volatility_median': historical_volatility_median,
        'regime_label': labels.values
    })

    # 统计标签分布
    label_counts = df_result['regime_label'].value_counts().sort_index()
    logger.info(f"\n标签分布:")
    for label, count in label_counts.items():
        if pd.notna(label):
            label_name = {0: '低波动', 1: '高波动'}[int(label)]
            logger.info(f"  {label_name}: {count} ({count/df_result['regime_label'].notna().sum()*100:.1f}%)")

    # 输出未来波动率统计
    logger.info(f"\n未来波动率统计:")
    logger.info(f"  均值: {df_result['future_volatility'].mean():.6f}")
    logger.info(f"  标准差: {df_result['future_volatility'].std():.6f}")
    logger.info(f"  最小值: {df_result['future_volatility'].min():.6f}")
    logger.info(f"  最大值: {df_result['future_volatility'].max():.6f}")
    logger.info(f"  中位数阈值: {historical_volatility_median:.6f}")

    # 验证最后window个样本
    logger.info(f"\n防未来函数验证:")
    logger.info(f"  最后{window}个样本应全部为NaN")
    last_window = df_result['regime_label'].iloc[-window:]
    logger.info(f"  实际: {last_window.isna().sum()}/{window} 为NaN")

    if last_window.isna().sum() != window:
        logger.error("❌ 防未来函数验证失败！")
        raise ValueError(f"最后{window}个标签应全部为NaN")

    logger.info("✓ 防未来函数验证通过")

    # 保存结果
    output_file = Path(output_file)
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df_result.to_csv(output_file, index=False)
    logger.info(f"\n✓ 结果已保存: {output_file}")

    # 额外统计
    valid_labels = df_result['regime_label'].dropna()
    logger.info(f"\n数据统计:")
    logger.info(f"  总样本数: {len(df_result)}")
    logger.info(f"  有效标签数: {len(valid_labels)}")
    logger.info(f"  无效样本数: {df_result['regime_label'].isna().sum()}")

    logger.info(f"\n{'='*60}")
    logger.info(f"波动率Regime标签生成完成！")
    logger.info(f"{'='*60}")

    return df_result


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 生成波动率Regime标签
    df = generate_volatility_regime_labels(
        window=10,
        train_start_year=2020,
        output_file=Path('data/labels/volatility_regime_labels.csv')
    )

    print(f"\n最终数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
