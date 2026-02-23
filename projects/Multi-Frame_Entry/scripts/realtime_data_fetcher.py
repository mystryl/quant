#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
实时数据获取模块

使用efinance/akshare获取期货60分钟K线数据
支持螺纹钢(RB)、热卷(HC)、铁矿石(I)、黄金(AU)、郑棉(CF)等品种
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Optional, Dict, List
import efinance as ef
import akshare as ak
from datetime import datetime, timedelta

logger = logging.getLogger(__name__)


class RealtimeDataFetcher:
    """实时数据获取器 - 使用efinance和akshare获取期货数据"""

    # 品种代码映射（efinance代码 -> 项目代码）
    SYMBOL_MAPPING = {
        'RB0': 'HC888',   # 螺纹钢 -> 使用热卷模型
        'HC0': 'HC888',   # 热卷
        'I0': 'I888',     # 铁矿石
        'AU0': 'AU888',   # 黄金
        'CF0': 'CF888',   # 郑棉
    }

    # 品种名称映射
    SYMBOL_NAMES = {
        'RB0': '螺纹钢',
        'HC0': '热卷',
        'I0': '铁矿石',
        'AU0': '黄金',
        'CF0': '郑棉',
    }

    # 本地数据目录（使用项目内相对路径）
    # 数据文件应位于项目根目录下的 data/60min/ 目录
    LOCAL_DATA_DIR = Path(__file__).parent.parent / 'data' / '60min'

    def __init__(self, preferred_source: str = 'local'):
        """
        初始化数据获取器

        Args:
            preferred_source: 首选数据源 ('local', 'efinance' 或 'akshare')
        """
        self.preferred_source = preferred_source
        logger.info(f"初始化实时数据获取器 (首选数据源: {preferred_source})")

    def fetch_latest_data_efinance(
        self,
        symbol: str,
        bars: int = 100,
        period: str = '60min'
    ) -> Optional[pd.DataFrame]:
        """
        使用efinance获取最新60分钟K线数据

        Args:
            symbol: 品种代码（如 'RB0' for 螺纹钢）
            bars: 获取K线数量
            period: K线周期（'60min' for 60分钟）

        Returns:
            包含OHLCV数据的DataFrame，失败返回None
        """
        try:
            logger.info(f"使用efinance获取数据: {symbol}, 数量: {bars}")

            # 获取期货基础信息
            futures_base_info = ef.futures.get_futures_base_info()

            # 筛选对应品种的合约
            if symbol == 'RB0':  # 螺纹钢
                contract_pattern = 'RB'
                market = 'XSGE'
            elif symbol == 'HC0':  # 热卷
                contract_pattern = 'HC'
                market = 'XSGE'
            elif symbol == 'I0':  # 铁矿石
                contract_pattern = 'I'
                market = 'XDCE'
            elif symbol == 'AU0':  # 黄金
                contract_pattern = 'AU'
                market = 'XSGE'
            elif symbol == 'CF0':  # 郑棉
                contract_pattern = 'CF'
                market = 'XZCE'
            else:
                logger.error(f"不支持的品种: {symbol}")
                return None

            # 查找连续合约或主力合约
            symbol_futures = futures_base_info[
                futures_base_info['期货代码'].str.contains(contract_pattern, case=False, na=False)
            ].copy()

            # 优先查找连续合约
            continuous = symbol_futures[
                symbol_futures['期货代码'].str.contains('连续|000|888', case=False, na=False)
            ]

            if not continuous.empty:
                quote_id = continuous.iloc[0]['行情ID']
                logger.info(f"使用连续合约: {continuous.iloc[0]['期货代码']} (行情ID: {quote_id})")
            else:
                # 查找主力合约
                main = symbol_futures[
                    symbol_futures['期货代码'].str.contains('主力|m|LZH', case=False, na=False)
                ]
                if not main.empty:
                    quote_id = main.iloc[0]['行情ID']
                    logger.info(f"使用主力合约: {main.iloc[0]['期货代码']} (行情ID: {quote_id})")
                else:
                    logger.warning(f"未找到{symbol}的连续或主力合约，使用第一个合约")
                    quote_id = symbol_futures.iloc[0]['行情ID']

            # 计算时间范围（向前推足够的天数以获取足够的K线）
            # 60分钟K线，每天最多约4小时交易 = 4根
            # 获取100根约需要25个交易日
            end_date = datetime.now()
            start_date = end_date - timedelta(days=60)  # 多获取一些确保有足够数据

            beg = start_date.strftime('%Y%m%d')
            end = end_date.strftime('%Y%m%d')

            # klt参数: 101=1分钟, 102=5分钟, 103=15分钟, 104=30分钟, 105=60分钟
            klt_map = {
                '1min': 101,
                '5min': 102,
                '15min': 103,
                '30min': 104,
                '60min': 105,
            }
            klt = klt_map.get(period, 105)

            logger.info(f"请求时间范围: {beg} - {end}, K线周期: {period}")

            # 获取数据
            data = ef.futures.get_quote_history(
                quote_ids=quote_id,
                beg=beg,
                end=end,
                klt=klt,
                fqt=0,  # 不复权
                return_df=True
            )

            if data is None or len(data) == 0:
                logger.error(f"未获取到数据: {symbol}")
                return None

            # 数据格式转换
            df = self._format_efinance_data(data)

            # 只保留最新的bars根K线
            if len(df) > bars:
                df = df.tail(bars).copy()

            logger.info(f"✓ 成功获取 {len(df)} 根K线 (时间范围: {df.index[0]} 到 {df.index[-1]})")

            return df

        except Exception as e:
            logger.error(f"efinance获取数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def fetch_latest_data_akshare(
        self,
        symbol: str,
        bars: int = 100,
        period: str = '60min'
    ) -> Optional[pd.DataFrame]:
        """
        使用akshare获取最新60分钟K线数据

        Args:
            symbol: 品种代码（如 'RB0' for 螺纹钢）
            bars: 获取K线数量
            period: K线周期

        Returns:
            包含OHLCV数据的DataFrame，失败返回None
        """
        try:
            logger.info(f"使用akshare获取数据: {symbol}")

            # akshare品种代码映射
            ak_symbol_map = {
                'RB0': 'RB2501',  # 螺纹钢主力合约
                'HC0': 'HC2501',  # 热卷主力合约
                'I0': 'I2501',    # 铁矿石主力合约
                'AU0': 'AU2501',  # 黄金主力合约
                'CF0': 'CF501',   # 郑棉主力合约
            }

            ak_symbol = ak_symbol_map.get(symbol, symbol)

            # period映射
            period_map = {
                '1min': '1',
                '5min': '5',
                '15min': '15',
                '30min': '30',
                '60min': '60',
            }
            ak_period = period_map.get(period, '60')

            # 使用sina源获取期货数据
            df = ak.futures_zh_hist_sina_symbol(
                symbol=ak_symbol,
                period=ak_period
            )

            if df is None or len(df) == 0:
                logger.error(f"未获取到数据: {symbol}")
                return None

            # 数据格式转换
            df = self._format_akshare_data(df)

            # 只保留最新的bars根K线
            if len(df) > bars:
                df = df.tail(bars).copy()

            logger.info(f"✓ 成功获取 {len(df)} 根K线")

            return df

        except Exception as e:
            logger.error(f"akshare获取数据失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def fetch_latest_data_local(
        self,
        symbol: str,
        bars: int = 100
    ) -> Optional[pd.DataFrame]:
        """
        从本地文件加载最新数据

        Args:
            symbol: 品种代码（如 'RB0' for 螺纹钢）
            bars: 获取K线数量

        Returns:
            包含OHLCV数据的DataFrame，失败返回None
        """
        try:
            # 映射到本地文件名
            model_code = self.SYMBOL_MAPPING.get(symbol, 'HC888')
            file_path = self.LOCAL_DATA_DIR / f'{model_code}.parquet'

            logger.info(f"从本地加载数据: {file_path}")

            if not file_path.exists():
                logger.error(f"本地文件不存在: {file_path}")
                return None

            # 读取数据
            df = pd.read_parquet(file_path)
            df['datetime'] = pd.to_datetime(df['date'])
            df = df.set_index('datetime')

            # 添加估算字段（如果缺失）
            if 'money' not in df.columns:
                df['money'] = df['close'] * df['volume']
            if 'open_interest' not in df.columns:
                df['open_interest'] = df['volume']

            # 只保留最新的bars根K线
            if len(df) > bars:
                df = df.tail(bars).copy()

            logger.info(f"✓ 成功加载 {len(df)} 根K线 (时间范围: {df.index[0]} 到 {df.index[-1]})")

            return df

        except Exception as e:
            logger.error(f"本地数据加载失败: {e}")
            import traceback
            traceback.print_exc()
            return None

    def fetch_latest_data(
        self,
        symbol: str,
        bars: int = 100,
        period: str = '60min'
    ) -> Optional[pd.DataFrame]:
        """
        获取最新K线数据（自动选择数据源）

        Args:
            symbol: 品种代码（如 'RB0' for 螺纹钢）
            bars: 获取K线数量（默认100根）
            period: K线周期（默认'60min'）

        Returns:
            包含OHLCV数据的DataFrame，失败返回None
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"获取最新数据: {symbol}")
        logger.info(f"  K线数量: {bars}")
        logger.info(f"  K线周期: {period}")
        logger.info(f"{'='*60}")

        # 优先使用本地数据
        if self.preferred_source == 'local':
            df = self.fetch_latest_data_local(symbol, bars)
            if df is not None:
                return df

            # 本地数据失败，尝试efinance
            logger.warning("本地数据失败，尝试使用efinance...")
            df = self.fetch_latest_data_efinance(symbol, bars, period)
            if df is not None:
                return df

            # efinance失败，尝试akshare
            logger.warning("efinance失败，尝试使用akshare...")
            df = self.fetch_latest_data_akshare(symbol, bars, period)
            return df

        # 优先使用efinance
        elif self.preferred_source == 'efinance':
            df = self.fetch_latest_data_efinance(symbol, bars, period)
            if df is not None:
                return df

            # efinance失败，尝试本地数据
            logger.warning("efinance失败，尝试使用本地数据...")
            df = self.fetch_latest_data_local(symbol, bars)
            if df is not None:
                return df

            # 本地数据失败，尝试akshare
            logger.warning("本地数据失败，尝试使用akshare...")
            df = self.fetch_latest_data_akshare(symbol, bars, period)
            return df

        else:  # akshare
            df = self.fetch_latest_data_akshare(symbol, bars, period)
            if df is not None:
                return df

            # akshare失败，尝试本地数据
            logger.warning("akshare失败，尝试使用本地数据...")
            df = self.fetch_latest_data_local(symbol, bars)
            if df is not None:
                return df

            # 本地数据失败，尝试efinance
            logger.warning("本地数据失败，尝试使用efinance...")
            df = self.fetch_latest_data_efinance(symbol, bars, period)
            return df

    def _format_efinance_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        格式化efinance数据

        Args:
            data: efinance原始数据

        Returns:
            格式化后的DataFrame
        """
        df = data.copy()

        # 重命名列（efinance的列名是中文）
        column_map = {
            '日期': 'datetime',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume',
            '持仓量': 'open_interest',
            '成交额': 'money',
        }

        df = df.rename(columns=column_map)

        # 设置时间索引
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')

        # 确保必需的列存在
        required_cols = ['open', 'high', 'low', 'close', 'volume']
        for col in required_cols:
            if col not in df.columns:
                logger.error(f"缺少必需列: {col}")
                raise ValueError(f"缺少必需列: {col}")

        # 添加估算字段（如果缺失）
        if 'money' not in df.columns:
            df['money'] = df['close'] * df['volume']
            logger.info("  估算成交额 = close * volume")

        if 'open_interest' not in df.columns:
            df['open_interest'] = df['volume']
            logger.info("  估算持仓量 = volume")

        # 按时间排序
        df = df.sort_index()

        # 删除重复数据
        df = df[~df.index.duplicated(keep='last')]

        return df

    def _format_akshare_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        格式化akshare数据

        Args:
            data: akshare原始数据

        Returns:
            格式化后的DataFrame
        """
        df = data.copy()

        # akshare的列名通常是英文
        # 确保列名小写
        df.columns = [col.lower() for col in df.columns]

        # 处理时间列
        if 'date' in df.columns:
            df['datetime'] = pd.to_datetime(df['date'])
            df = df.set_index('datetime')
            df = df.drop(columns=['date'])
        elif 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])
            df = df.set_index('datetime')
        elif 'time' in df.columns:
            df['datetime'] = pd.to_datetime(df['time'])
            df = df.set_index('datetime')
            df = df.drop(columns=['time'])

        # 添加估算字段（如果缺失）
        if 'money' not in df.columns:
            df['money'] = df['close'] * df['volume']

        if 'open_interest' not in df.columns:
            df['open_interest'] = df['volume']

        # 按时间排序
        df = df.sort_index()

        return df

    @staticmethod
    def get_project_symbol(efinance_symbol: str) -> str:
        """
        将efinance品种代码转换为项目品种代码

        Args:
            efinance_symbol: efinance品种代码（如 'RB0'）

        Returns:
            项目品种代码（如 'HC888'）
        """
        return RealtimeDataFetcher.SYMBOL_MAPPING.get(efinance_symbol, 'HC888')

    @staticmethod
    def get_symbol_name(symbol: str) -> str:
        """
        获取品种中文名称

        Args:
            symbol: 品种代码（如 'RB0'）

        Returns:
            品种中文名（如 '螺纹钢'）
        """
        return RealtimeDataFetcher.SYMBOL_NAMES.get(symbol, symbol)


if __name__ == '__main__':
    """测试数据获取功能"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 创建数据获取器
    fetcher = RealtimeDataFetcher(preferred_source='efinance')

    # 测试获取螺纹钢数据
    print("\n" + "="*60)
    print("测试: 获取螺纹钢 (RB0) 60分钟K线")
    print("="*60)

    df = fetcher.fetch_latest_data(symbol='RB0', bars=100, period='60min')

    if df is not None:
        print(f"\n✓ 数据获取成功!")
        print(f"  数据形状: {df.shape}")
        print(f"  时间范围: {df.index[0]} 到 {df.index[-1]}")
        print(f"  列名: {list(df.columns)}")
        print(f"\n最新5根K线:")
        print(df.tail(5).to_string())
    else:
        print("\n✗ 数据获取失败")
