"""
最终标签生成模块

基于窗口对比分析结果，使用20根K线窗口生成趋势标签。
配置：
- 窗口：20根60min K线（约20小时，1.5个交易日）
- 上涨阈值：> 0.3%
- 下跌阈值：< -0.3%
- 标签：1=上涨, 0=震荡, -1=下跌
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging

logger = logging.getLogger(__name__)


# 最终配置（基于窗口对比分析结果）
TREND_LABEL_CONFIG = {
    'window': 20,  # 20根60min K线
    'window_hours': 20,
    'window_days': 1.48,
    'up_threshold': 0.003,  # 0.3%
    'down_threshold': -0.003,  # -0.3%
    'data_freq': '60min',
    'instrument': 'HC8888.XSGE'
}


def generate_trend_labels(
    df: pd.DataFrame,
    price_col: str = 'close'
) -> pd.DataFrame:
    """
    生成趋势标签（20根K线窗口）

    Args:
        df: 包含价格数据的DataFrame（必须有datetime索引和close列）
        price_col: 价格列名

    Returns:
        添加了标签列的DataFrame
    """
    logger.info(f"生成趋势标签（{TREND_LABEL_CONFIG['window']}根K线窗口）...")

    # 计算未来收益率
    close_price = df[price_col]
    future_price = close_price.shift(-TREND_LABEL_CONFIG['window'])
    future_return = (future_price - close_price) / close_price

    # 生成标签（只在有效收益率处生成标签）
    labels = pd.Series(np.nan, index=df.index)  # 初始化为NaN
    labels[future_return > TREND_LABEL_CONFIG['up_threshold']] = 1
    labels[future_return < TREND_LABEL_CONFIG['down_threshold']] = -1
    labels[(future_return >= TREND_LABEL_CONFIG['down_threshold']) &
           (future_return <= TREND_LABEL_CONFIG['up_threshold']) &
           (future_return.notna())] = 0

    # 添加到DataFrame
    df = df.copy()
    df['future_return'] = future_return
    df['trend_label'] = labels

    # 统计信息
    valid_labels = labels[labels.notna()]
    label_counts = valid_labels.value_counts()

    logger.info(f"  总样本: {len(df)}")
    logger.info(f"  有效标签: {len(valid_labels)} ({len(valid_labels)/len(df)*100:.1f}%)")
    logger.info(f"  标签分布:")
    logger.info(f"    上涨 (1): {label_counts.get(1, 0)} ({label_counts.get(1, 0)/len(valid_labels)*100:.1f}%)")
    logger.info(f"    震荡 (0): {label_counts.get(0, 0)} ({label_counts.get(0, 0)/len(valid_labels)*100:.1f}%)")
    logger.info(f"    下跌 (-1): {label_counts.get(-1, 0)} ({label_counts.get(-1, 0)/len(valid_labels)*100:.1f}%)")

    return df


def load_and_generate_labels(
    data_dir: Path = None,
    instrument: str = None,
    start_date: str = '2022-01-01',
    end_date: str = '2025-12-31',
    save_path: Path = None
) -> pd.DataFrame:
    """
    加载60min数据并生成趋势标签

    Args:
        data_dir: 数据目录
        instrument: 合约代码
        start_date: 开始日期
        end_date: 结束日期
        save_path: 保存路径（如果提供，则保存结果）

    Returns:
        包含标签的DataFrame
    """
    if data_dir is None:
        data_dir = Path('/Users/mystryl/Documents/Quant/data/qlib_data_multi_freq')
    if instrument is None:
        instrument = TREND_LABEL_CONFIG['instrument']

    logger.info(f"\n{'='*60}")
    logger.info(f"生成趋势标签: {instrument}")
    logger.info(f"{'='*60}")

    # 读取60min数据
    logger.info(f"读取60min数据...")
    freq_dir = data_dir / 'instruments' / '60min'

    data_dict = {}
    for field in ['open', 'high', 'low', 'close', 'volume']:
        field_file = freq_dir / f'${field}' / f'{instrument}.csv'
        df_field = pd.read_csv(field_file, index_col=0, parse_dates=True)
        data_dict[field] = df_field.iloc[:, 0]

    # 合并数据
    df = pd.DataFrame(data_dict)
    df = df.reset_index()
    df.columns = ['datetime', 'open', 'high', 'low', 'close', 'volume']

    # 过滤日期范围
    df = df[(df['datetime'] >= start_date) & (df['datetime'] <= end_date)]

    logger.info(f"  数据加载完成: {df.shape}")
    logger.info(f"  时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")

    # 生成标签
    df = generate_trend_labels(df, price_col='close')

    # 保存结果
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(save_path, index=False)
        logger.info(f"\n✓ 结果已保存: {save_path}")

    return df


def validate_no_lookahead(df: pd.DataFrame) -> bool:
    """
    验证标签是否存在未来函数污染

    检查点：
    1. 标签计算是否使用了shift(-window)
    2. 最后window个样本应该无标签（NaN）
    3. 标签变化规律是否符合预期

    Args:
        df: 包含标签的DataFrame

    Returns:
        True表示无未来函数，False表示有问题
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"验证无未来函数污染...")
    logger.info(f"{'='*60}")

    issues = []

    # 检查1：最后window个样本应该无标签
    window = TREND_LABEL_CONFIG['window']
    last_labels = df['trend_label'].iloc[-window:]
    if last_labels.notna().any():
        issues.append(f"最后{window}个样本不应有标签，但发现了{last_labels.notna().sum()}个")
    else:
        logger.info(f"  ✓ 最后{window}个样本正确为NaN")

    # 检查2：标签数量应符合预期
    expected_valid = len(df) - window
    actual_valid = df['trend_label'].notna().sum()
    if actual_valid != expected_valid:
        issues.append(f"有效标签数量不符: 预期{expected_valid}, 实际{actual_valid}")
    else:
        logger.info(f"  ✓ 有效标签数量正确: {actual_valid}")

    # 检查3：验证shift(-window)的正确性
    # 手动计算几个样本验证
    sample_idx = len(df) - window - 5  # 选择倒数第window+5个样本
    if sample_idx >= 0:
        current_price = df['close'].iloc[sample_idx]
        future_price = df['close'].iloc[sample_idx + window]
        expected_return = (future_price - current_price) / current_price
        actual_return = df['future_return'].iloc[sample_idx]

        if abs(expected_return - actual_return) < 1e-10:
            logger.info(f"  ✓ 未来收益率计算正确（样本{sample_idx}）")
        else:
            issues.append(f"未来收益率计算错误（样本{sample_idx}）: 预期{expected_return}, 实际{actual_return}")

    # 检查4：标签与收益率的一致性
    sample_with_label = df[df['trend_label'].notna()].iloc[0]
    return_val = sample_with_label['future_return']
    label = sample_with_label['trend_label']

    if label == 1 and return_val <= TREND_LABEL_CONFIG['up_threshold']:
        issues.append(f"标签为1但收益率{return_val:.4f}未超过阈值{TREND_LABEL_CONFIG['up_threshold']}")
    elif label == -1 and return_val >= TREND_LABEL_CONFIG['down_threshold']:
        issues.append(f"标签为-1但收益率{return_val:.4f}未低于阈值{TREND_LABEL_CONFIG['down_threshold']}")
    elif label == 0:
        if return_val > TREND_LABEL_CONFIG['up_threshold'] or return_val < TREND_LABEL_CONFIG['down_threshold']:
            issues.append(f"标签为0但收益率{return_val:.4f}超出震荡区间")
    else:
        logger.info(f"  ✓ 标签与收益率一致")

    if issues:
        logger.error("发现以下问题:")
        for issue in issues:
            logger.error(f"  ✗ {issue}")
        return False
    else:
        logger.info(f"\n{'='*60}")
        logger.info(f"✓ 所有检查通过，无未来函数污染")
        logger.info(f"{'='*60}")
        return True


def main():
    """主函数：生成并保存趋势标签"""
    # 配置
    output_dir = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data/labels')
    output_file = output_dir / 'final_labels_20bars.csv'

    # 生成标签
    df = load_and_generate_labels(
        data_dir=Path('/Users/mystryl/Documents/Quant/data/qlib_data_multi_freq'),
        instrument='HC8888.XSGE',
        start_date='2022-01-01',
        end_date='2025-12-31',
        save_path=output_file
    )

    # 验证无未来函数
    is_valid = validate_no_lookahead(df)

    if is_valid:
        logger.info(f"\n{'='*60}")
        logger.info(f"趋势标签生成完成！")
        logger.info(f"{'='*60}")
        logger.info(f"配置:")
        logger.info(f"  窗口: {TREND_LABEL_CONFIG['window']}根K线")
        logger.info(f"  时间跨度: {TREND_LABEL_CONFIG['window_hours']}小时 ({TREND_LABEL_CONFIG['window_days']}天)")
        logger.info(f"  阈值: ±{abs(TREND_LABEL_CONFIG['up_threshold'])*100:.1f}%")
        logger.info(f"\n输出文件: {output_file}")
        logger.info(f"数据形状: {df.shape}")
        logger.info(f"列: {list(df.columns)}")
    else:
        logger.error(f"\n标签验证失败！请检查代码逻辑。")
        return 1

    return 0


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )
    exit(main())
