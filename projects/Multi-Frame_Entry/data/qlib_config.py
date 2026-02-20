"""
Qlib 配置模块

配置 Qlib 数据加载和多频率数据访问
"""
import qlib
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


# Qlib 全局配置
QLIB_CONFIG = {
    # 数据目录
    'provider_uri': '/Users/mystryl/Documents/Quant/data/qlib_data_multi_freq',

    # 区域配置
    'region': 'cn',

    # 缓存目录
    'kv_cache': '/Users/mystryl/Documents/Quant/data/qlib_cache',

    # 默认频率
    'default_freq': '1min',

    # 可用频率
    'available_freqs': ['1min', '5min', '15min', '60min', 'day']
}


def init_qlib(
    provider_uri: str = None,
    region: str = None,
    kv_cache: str = None
):
    """
    初始化 Qlib

    Args:
        provider_uri: 数据目录路径
        region: 区域配置 ('cn' 或 'us')
        kv_cache: 缓存目录路径
    """
    provider_uri = provider_uri or QLIB_CONFIG['provider_uri']
    region = region or QLIB_CONFIG['region']
    kv_cache = kv_cache or QLIB_CONFIG['kv_cache']

    logger.info(f"初始化 Qlib:")
    logger.info(f"  数据目录: {provider_uri}")
    logger.info(f"  区域: {region}")
    logger.info(f"  缓存目录: {kv_cache}")

    qlib.init(
        provider_uri=provider_uri,
        region=region,
        kv_cache=kv_cache
    )

    logger.info("✓ Qlib 初始化完成")


def get_instruments(market: str = 'all') -> List[str]:
    """
    获取可用合约列表

    Args:
        market: 市场名称

    Returns:
        合约代码列表
    """
    from qlib.data import D

    instruments = D.instruments(market=market)
    logger.info(f"可用合约 ({market}): {len(instruments)} 个")
    return instruments


def load_data(
    instruments: List[str],
    fields: List[str],
    start_time: str,
    end_time: str,
    freq: str = '1min'
):
    """
    加载多频率数据

    Args:
        instruments: 合约列表
        fields: 字段列表（如 ['$open', '$high', '$low', '$close', '$volume']）
        start_time: 开始时间
        end_time: 结束时间
        freq: 频率 ('1min', '5min', '15min', '60min', 'day')

    Returns:
        DataFrame (multi-index: instrument, datetime)
    """
    from qlib.data import D

    logger.info(f"加载数据:")
    logger.info(f"  合约: {instruments}")
    logger.info(f"  频率: {freq}")
    logger.info(f"  时间范围: {start_time} ~ {end_time}")
    logger.info(f"  字段: {fields}")

    df = D.features(
        instruments=instruments,
        fields=fields,
        start_time=start_time,
        end_time=end_time,
        freq=freq
    )

    logger.info(f"✓ 数据加载完成: {df.shape}")
    return df


def load_multi_freq_data(
    instruments: List[str],
    fields: List[str],
    start_time: str,
    end_time: str,
    freqs: List[str] = None
) -> dict:
    """
    加载多周期数据

    Args:
        instruments: 合约列表
        fields: 字段列表
        start_time: 开始时间
        end_time: 结束时间
        freqs: 频率列表，默认为所有可用频率

    Returns:
        字典 {freq: DataFrame}
    """
    if freqs is None:
        freqs = QLIB_CONFIG['available_freqs']

    logger.info(f"加载多周期数据...")

    data = {}
    for freq in freqs:
        try:
            df = load_data(
                instruments=instruments,
                fields=fields,
                start_time=start_time,
                end_time=end_time,
                freq=freq
            )
            data[freq] = df
        except Exception as e:
            logger.warning(f"加载 {freq} 数据失败: {e}")
            continue

    logger.info(f"✓ 多周期数据加载完成，共 {len(data)} 个频率")
    return data


# 使用示例
if __name__ == '__main__':
    # 初始化 Qlib
    init_qlib()

    # 获取合约列表
    instruments = get_instruments()
    print(f"\n可用合约: {instruments}")

    # 加载 60min 数据
    df_60min = load_data(
        instruments=['HC8888.XSGE'],
        fields=['$open', '$high', '$low', '$close', '$volume'],
        start_time='2024-01-01',
        end_time='2024-12-31',
        freq='60min'
    )
    print(f"\n60min 数据:")
    print(df_60min.head())

    # 加载多周期数据
    multi_freq_data = load_multi_freq_data(
        instruments=['HC8888.XSGE'],
        fields=['$close'],
        start_time='2024-01-01',
        end_time='2024-01-31',
        freqs=['5min', '15min', '60min']
    )
    print(f"\n多周期数据:")
    for freq, df in multi_freq_data.items():
        print(f"  {freq}: {df.shape}")
