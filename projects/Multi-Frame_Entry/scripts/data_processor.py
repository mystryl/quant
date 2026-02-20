#!/usr/bin/env python3
"""
数据预处理模块：多周期重采样
将 1min 数据重采样到 5min, 15min, 60min, 1day

重要：
- OHLC 聚合规则：open(first), high(max), low(min), close(last)
- 成交量求和
- VWAP 重算：amount / volume
- 严格保持时间对齐
"""
import pandas as pd
import numpy as np
from pathlib import Path
import shutil
from typing import Dict, List
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class MultiFrameDataProcessor:
    """多周期数据处理器"""

    # 字段映射（原始字段 → qlib 特征名）
    FIELD_MAPPING = {
        'open': '$open',
        'high': '$high',
        'low': '$low',
        'close': '$close',
        'volume': '$volume',
        'amount': '$amount',
        'vwap': '$vwap',
        'open_interest': '$open_interest'
    }

    # 重采样规则（遵循标准 OHLC 聚合）
    RESAMPLE_RULES = {
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
        'amount': 'sum',
        'open_interest': 'last'
    }

    # 目标频率（pandas 频率字符串）
    TARGET_FREQS = ['5min', '15min', '60min', 'D']

    def __init__(self, data_dir: Path, output_dir: Path = None):
        """
        初始化数据处理器

        Args:
            data_dir: 统一数据目录（包含 1min 数据）
            output_dir: 输出目录，默认为 data_dir/qlib_data_multi_freq
        """
        self.data_dir = Path(data_dir)
        self.output_dir = Path(output_dir) if output_dir else self.data_dir / 'qlib_data_multi_freq'

        # 路径配置
        self.instruments_dir = self.data_dir / 'qlib_data_multi_freq' / 'instruments'
        self.calendars_dir = self.data_dir / 'qlib_data_multi_freq' / 'calendars'

        logger.info(f"数据处理器初始化:")
        logger.info(f"  数据目录: {self.data_dir}")
        logger.info(f"  输出目录: {self.output_dir}")

    def read_1min_data(self, instrument: str) -> pd.DataFrame:
        """
        读取指定合约的 1min 数据

        Args:
            instrument: 合约代码，如 'HC8888.XSGE'

        Returns:
            包含所有字段的 DataFrame
        """
        logger.info(f"读取合约 {instrument} 的 1min 数据...")

        instrument_dir = self.instruments_dir / '1min'
        if not instrument_dir.exists():
            raise FileNotFoundError(f"1min 数据目录不存在: {instrument_dir}")

        # 读取所有字段
        data_1min = {}
        for field in ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'open_interest']:
            field_dir = instrument_dir / f'${field}'
            field_file = field_dir / f'{instrument}.csv'

            if field_file.exists():
                df = pd.read_csv(field_file, index_col=0, parse_dates=True)
                data_1min[field] = df.iloc[:, 0]
                logger.info(f"  ✓ 读取 {field}: {len(df)} 行")
            else:
                logger.warning(f"  ✗ 字段 {field} 不存在")

        if not data_1min:
            raise ValueError(f"合约 {instrument} 没有可用数据")

        # 合并为单个 DataFrame
        df_1min = pd.DataFrame(data_1min)
        df_1min = df_1min.sort_index()

        logger.info(f"  1min 数据统计:")
        logger.info(f"    总行数: {len(df_1min)}")
        logger.info(f"    时间范围: {df_1min.index.min()} ~ {df_1min.index.max()}")
        logger.info(f"    字段: {list(df_1min.columns)}")

        return df_1min

    def resample_data(self, df_1min: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        重采样数据到指定频率

        Args:
            df_1min: 1min 数据
            freq: 目标频率 ('5min', '15min', '60min', '1day')

        Returns:
            重采样后的 DataFrame
        """
        # 显示友好的频率名称
        freq_display = '1day' if freq == 'D' else freq
        logger.info(f"  重采样到 {freq_display}...")

        # 创建重采样规则字典
        resample_dict = {}
        for field, rule in self.RESAMPLE_RULES.items():
            if field in df_1min.columns:
                resample_dict[field] = rule

        # 执行重采样
        df_resampled = df_1min.resample(freq).agg(resample_dict)

        # 重新计算 VWAP（如果存在 amount 和 volume）
        if 'amount' in df_resampled.columns and 'volume' in df_resampled.columns:
            df_resampled['vwap'] = df_resampled['amount'] / df_resampled['volume']
            # 处理除零情况
            df_resampled['vwap'] = df_resampled['vwap'].replace([np.inf, -np.inf], np.nan)
            df_resampled['vwap'] = df_resampled['vwap'].ffill()

        # 删除全 NaN 行（非交易时间段）
        df_resampled = df_resampled.dropna(subset=['close'])

        freq_display = '1day' if freq == 'D' else freq
        logger.info(f"    {freq_display} 数据: {len(df_resampled)} 行")
        logger.info(f"    时间范围: {df_resampled.index.min()} ~ {df_resampled.index.max()}")

        return df_resampled

    def save_to_qlib_format(self, df: pd.DataFrame, instrument: str, freq: str):
        """
        保存数据到 Qlib 格式

        Args:
            df: 重采样后的数据
            instrument: 合约代码
            freq: 频率
        """
        freq_dir = self.output_dir / 'instruments' / freq

        for field, feature_name in self.FIELD_MAPPING.items():
            if field in df.columns:
                feature_dir = freq_dir / feature_name
                feature_dir.mkdir(parents=True, exist_ok=True)

                # 保存单列数据
                df_feature = df[[field]].copy()
                df_feature.to_csv(feature_dir / f'{instrument}.csv')

        logger.info(f"    ✓ 保存到: {freq_dir}")

    def generate_calendar(self, freq: str):
        """
        生成指定频率的日历文件

        Args:
            freq: 频率
        """
        # 读取 1min 日历
        cal_1min_file = self.calendars_dir / '1min.txt'
        if not cal_1min_file.exists():
            raise FileNotFoundError(f"1min 日历文件不存在: {cal_1min_file}")

        cal_1min = pd.read_csv(cal_1min_file, header=None, names=['datetime'])
        cal_1min['datetime'] = pd.to_datetime(cal_1min['datetime'])

        # 重采样日历
        if freq == '1min':
            cal_resampled = cal_1min
        elif freq == 'D':
            # 日线使用已有的 day.txt
            cal_day_file = self.calendars_dir / 'day.txt'
            cal_output_file = self.output_dir / 'calendars' / 'day.txt'
            if cal_day_file.exists():
                # 检查是否是同一个文件
                if cal_day_file != cal_output_file:
                    shutil.copy(cal_day_file, cal_output_file)
                logger.info(f"  ✓ 使用现有日线日历")
                return
            cal_resampled = cal_1min.set_index('datetime').resample('D').first().dropna()
            freq = 'day'  # 保存文件时使用 day 而不是 D
        else:
            cal_resampled = cal_1min.set_index('datetime').resample(freq).first().dropna()

        cal_resampled = cal_resampled.reset_index()

        # 保存日历
        calendars_dir = self.output_dir / 'calendars'
        calendars_dir.mkdir(parents=True, exist_ok=True)

        cal_file = calendars_dir / f'{freq}.txt'
        cal_resampled['datetime'].dt.strftime('%Y-%m-%d %H:%M:%S').to_csv(
            cal_file, index=False, header=False
        )

        freq_display = '1day' if freq == 'day' else freq
        logger.info(f"  ✓ 生成 {freq_display} 日历: {len(cal_resampled)} 个时间点")

    def process_instrument(self, instrument: str):
        """
        处理单个合约：重采样并保存所有频率

        Args:
            instrument: 合约代码
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"处理合约: {instrument}")
        logger.info(f"{'='*60}\n")

        # 读取 1min 数据
        df_1min = self.read_1min_data(instrument)

        # 对每个目标频率进行重采样
        for freq in self.TARGET_FREQS:
            df_resampled = self.resample_data(df_1min, freq)
            self.save_to_qlib_format(df_resampled, instrument, freq)

        # 生成日历文件
        logger.info(f"\n生成日历文件...")
        for freq in ['1min'] + self.TARGET_FREQS:
            self.generate_calendar(freq)

        logger.info(f"\n{'='*60}")
        logger.info(f"合约 {instrument} 处理完成！")
        logger.info(f"{'='*60}\n")

    def process_all(self, instruments: List[str] = None):
        """
        处理所有合约

        Args:
            instruments: 合约列表，如果为 None 则处理 instruments 目录下的所有合约
        """
        if instruments is None:
            # 自动发现合约
            freq_dir = self.instruments_dir / '1min'
            if freq_dir.exists():
                # 从任意特征目录读取合约列表
                for feature_dir in freq_dir.iterdir():
                    if feature_dir.is_dir() and feature_dir.name.startswith('$'):
                        instruments = [f.stem for f in feature_dir.glob('*.csv')]
                        break

        if not instruments:
            raise ValueError("未找到合约数据")

        logger.info(f"\n{'='*60}")
        logger.info(f"开始批量处理 {len(instruments)} 个合约")
        logger.info(f"{'='*60}\n")

        for instrument in instruments:
            try:
                self.process_instrument(instrument)
            except Exception as e:
                logger.error(f"处理合约 {instrument} 时出错: {e}")
                continue

        logger.info(f"\n{'='*60}")
        logger.info(f"所有合约处理完成！")
        logger.info(f"{'='*60}")


def main():
    """主函数"""
    # 配置路径
    DATA_DIR = Path("/Users/mystryl/Documents/Quant/data")

    # 创建处理器
    processor = MultiFrameDataProcessor(data_dir=DATA_DIR)

    # 处理单个合约
    processor.process_instrument('HC8888.XSGE')

    # 或处理所有合约
    # processor.process_all()


if __name__ == '__main__':
    main()
