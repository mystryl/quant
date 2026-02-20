"""
为二分类标签生成特征

1. 加载60min OHLCV数据
2. 筛选2022-2025数据
3. 合并标签
4. 计算57个特征
5. 保存最终特征文件
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from features.trend_features import TrendFeatures

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_60min_ohlcv():
    """加载60min OHLCV数据"""
    logger.info("加载60min OHLCV数据...")

    data_dir = Path('/Users/mystryl/Documents/Quant/data/qlib_data_multi_freq/instruments/60min')
    instrument = 'HC8888.XSGE'

    # 加载各个字段
    fields = {
        'open': '$open',
        'high': '$high',
        'low': '$low',
        'close': '$close',
        'volume': '$volume'
    }

    dfs = []
    for field_name, folder_name in fields.items():
        field_file = data_dir / folder_name / f'{instrument}.csv'
        logger.info(f"  加载 {field_name}: {field_file}")
        df_field = pd.read_csv(field_file, index_col=0, parse_dates=True)
        dfs.append(df_field)

    # 合并所有字段
    df_ohlcv = pd.concat(dfs, axis=1)
    df_ohlcv.columns = fields.keys()

    # 重置索引
    df_ohlcv = df_ohlcv.reset_index()
    df_ohlcv.columns = ['datetime'] + list(fields.keys())

    logger.info(f"✓ 数据形状: {df_ohlcv.shape}")
    logger.info(f"✓ 时间范围: {df_ohlcv['datetime'].min()} ~ {df_ohlcv['datetime'].max()}")

    return df_ohlcv


def generate_binary_features():
    """生成二分类特征"""
    logger.info("="*60)
    logger.info("生成二分类特征")
    logger.info("="*60)

    # 1. 加载OHLCV数据
    df_ohlcv = load_60min_ohlcv()

    # 2. 加载二分类标签
    logger.info("\n加载二分类标签...")
    df_labels = pd.read_csv('data/labels/binary_labels_2022_2025.csv')
    df_labels['datetime'] = pd.to_datetime(df_labels['datetime'])
    logger.info(f"  标签数据: {df_labels.shape}")

    # 3. 合并数据
    logger.info("\n合并OHLCV和标签...")
    df = df_ohlcv.merge(df_labels[['datetime', 'trend_label']], on='datetime', how='inner')
    logger.info(f"  合并后数据: {df.shape}")

    # 4. 计算特征
    logger.info("\n计算趋势特征...")
    calculator = TrendFeatures(
        ema_short=20,
        ema_long=60,
        adx_period=14,
        atr_period=14
    )

    df = calculator.compute_all_features(df, shift=True)

    # 5. 保存结果
    output_file = Path('data/features/binary_features.csv')
    output_file.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_file, index=False)
    logger.info(f"\n✓ 特征已保存: {output_file}")

    # 6. 统计信息
    logger.info(f"\n数据统计:")
    logger.info(f"  总样本数: {len(df)}")
    logger.info(f"  特征数: {len(df.columns) - 7}")  # 减去datetime, OHLCV, trend_label

    valid_mask = df['trend_label'].notna()
    logger.info(f"  有效标签: {valid_mask.sum()}")

    logger.info(f"\n标签分布:")
    label_counts = df.loc[valid_mask, 'trend_label'].value_counts().sort_index()
    for label, count in label_counts.items():
        label_name = {0: '震荡', 1: '有趋势'}[int(label)]
        pct = count / valid_mask.sum() * 100
        logger.info(f"  {label_name}: {count} ({pct:.1f}%)")

    logger.info(f"\n{'='*60}")
    logger.info(f"特征生成完成！")
    logger.info(f"{'='*60}")

    return df


if __name__ == '__main__':
    df = generate_binary_features()
    print(f"\n最终数据形状: {df.shape}")
