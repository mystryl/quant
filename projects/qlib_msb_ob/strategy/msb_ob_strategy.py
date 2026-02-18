#!/usr/bin/env python3
"""
MSB+OB Strategy Implementation
市场结构突破 + 订单块策略

基于 LuxAlgo - Market Structure Break & OB Probability Toolkit 源码分析
完全按照Pine源码的计算逻辑实现
"""

import pandas as pd
import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
from datetime import datetime
import uuid


# ========================================================================
# 数据结构定义
# ========================================================================

@dataclass
class OrderBlock:
    """
    订单块数据结构

    对应Pine源码第79-89行的OrderBlock类型
    """
    id: str
    ob_type: str  # 'bullish' or 'bearish' (源码中的isBull)
    timestamp: datetime  # MSB发生时间
    ob_index: int  # OB对应的K线索引
    top: float  # OB顶部价格
    bottom: float  # OB底部价格
    poc: float  # Point of Control (中点)
    width: float  # OB宽度
    quality_score: float  # 质量分数 (0-100)
    is_hpz: bool  # 是否为高概率区域 (HPZ)
    is_mitigated: bool = False  # 是否已失效
    mitigation_timestamp: Optional[datetime] = None  # 失效时间
    mitigation_bar_index: Optional[int] = None  # 失效K线索引

    def __post_init__(self):
        """初始化后计算POC和宽度"""
        if self.poc is None:
            self.poc = (self.top + self.bottom) / 2
        if self.width is None:
            self.width = self.top - self.bottom


@dataclass
class Position:
    """
    持仓数据结构
    """
    id: str
    pos_type: str  # 'long' or 'short'
    entry_price: float
    entry_timestamp: datetime
    entry_bar_index: int
    stop_loss: float
    take_profits: Dict[str, float]  # {'tp1': price, 'tp2': price, 'tp3': price}
    position_size: float
    ob_id: str  # 关联的订单块ID
    exit_price: Optional[float] = None
    exit_timestamp: Optional[datetime] = None
    exit_reason: Optional[str] = None  # 'stop', 'tp1', 'tp2', 'tp3', 'manual'
    pnl: Optional[float] = None
    is_active: bool = True


@dataclass
class MSBSignal:
    """
    MSB信号数据结构
    """
    id: str
    signal_type: str  # 'bullish' or 'bearish'
    timestamp: datetime
    bar_index: int
    pivot_price: float
    pivot_index: int
    momentum_z: float
    volume_percentile: float


# ========================================================================
# 辅助函数
# ========================================================================

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """
    计算平均真实波幅(ATR)

    用于止损设置

    Args:
        high: 最高价序列
        low: 最低价序列
        close: 收盘价序列
        period: 周期 (默认14)

    Returns:
        ATR序列
    """
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_value = tr.rolling(period).mean()
    return atr_value


def calculate_pivot_points(df: pd.DataFrame, pivot_len: int = 7) -> Tuple[pd.Series, pd.Series]:
    """
    计算枢轴点

    对应Pine源码第149-163行：
    float ph = ta.pivothigh(high, pivotLenInput, pivotLenInput)
    float pl = ta.pivotlow(low, pivotLenInput, pivotLenInput)

    Pivot High: 当前K线的high是左右各pivot_len根K线的最高点
    Pivot Low: 当前K线的low是左右各pivot_len根K线的最低点

    Args:
        df: 包含 high, low 的 DataFrame
        pivot_len: 回看长度 (默认7)

    Returns:
        (pivot_high, pivot_low): 枢轴高点和枢轴低点序列
    """
    high = df['high'].values
    low = df['low'].values

    pivot_high = pd.Series(np.nan, index=df.index)
    pivot_low = pd.Series(np.nan, index=df.index)

    # 需要足够的数据才能计算枢轴点
    # 位置i需要满足: i >= pivot_len 且 i < len(df) - pivot_len
    for i in range(pivot_len, len(df) - pivot_len):
        # 检查是否为枢轴高点
        left_high = high[i - pivot_len:i]
        right_high = high[i + 1:i + pivot_len + 1]
        if high[i] > np.max(left_high) and high[i] > np.max(right_high):
            pivot_high.iloc[i] = high[i]

        # 检查是否为枢轴低点
        left_low = low[i - pivot_len:i]
        right_low = low[i + 1:i + pivot_len + 1]
        if low[i] < np.min(left_low) and low[i] < np.min(right_low):
            pivot_low.iloc[i] = low[i]

    return pivot_high, pivot_low


# ========================================================================
# 主策略类
# ========================================================================

class MSBOBStrategy:
    """
    MSB+OB 策略实现

    基于Pine源码的完整实现，包括：
    1. 动量计算 (momentumZ, volPercent)
    2. 枢轴点检测 (pivot high/low)
    3. MSB信号识别 (Market Structure Break)
    4. 订单块识别 (Order Block)
    5. 质量评分 (Quality Score & HPZ)
    6. OB失效检测 (Mitigation)
    7. 入场/出场信号生成
    8. 持仓管理
    """

    def __init__(
        self,
        pivot_len: int = 7,
        msb_zscore: float = 0.5,
        ob_max_count: int = 10,
        hpz_threshold: float = 80,
        atr_period: int = 14,
        atr_multiplier: float = 1.0,
        tp1_multiplier: float = 0.5,
        tp2_multiplier: float = 1.0,
        tp3_multiplier: float = 1.5,
        entry_buffer: float = 0.001,
        hpz_only: bool = False,
        max_positions: int = 3
    ):
        """
        初始化策略参数

        Args:
            pivot_len: 枢轴点回看长度 (默认7，对应Pine源码 pivotLenInput)
            msb_zscore: MSB动量Z-Score阈值 (默认0.5，对应Pine源码 msbZScoreInput)
            ob_max_count: 最大活跃OB数量 (默认10，对应Pine源码 obCountInput)
            hpz_threshold: HPZ质量分数阈值 (默认80，对应Pine源码 score > 80)
            atr_period: ATR计算周期 (默认14)
            atr_multiplier: ATR止损倍数 (默认1.0)
            tp1_multiplier: 目标1止盈倍数 (默认0.5倍OB宽度)
            tp2_multiplier: 目标2止盈倍数 (默认1.0倍OB宽度)
            tp3_multiplier: 目标3止盈倍数 (默认1.5倍OB宽度)
            entry_buffer: 入场价格缓冲 (默认0.001 = 0.1%)
            hpz_only: 是否只交易HPZ (默认False)
            max_positions: 最大同时持仓数 (默认3)
        """
        # Pine源码参数
        self.pivot_len = pivot_len
        self.msb_zscore = msb_zscore
        self.ob_max_count = ob_max_count
        self.hpz_threshold = hpz_threshold

        # 止损止盈参数
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.tp1_multiplier = tp1_multiplier
        self.tp2_multiplier = tp2_multiplier
        self.tp3_multiplier = tp3_multiplier

        # 过滤参数
        self.entry_buffer = entry_buffer
        self.hpz_only = hpz_only
        self.max_positions = max_positions

        # 状态变量
        self.active_obs: List[OrderBlock] = []
        self.mitigated_obs: List[OrderBlock] = []
        self.active_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.msb_signals: List[MSBSignal] = []

        # 统计变量 (对应Pine源码第237-241行)
        self.total_obs = 0
        self.total_mitigated = 0

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算所有技术指标

        对应Pine源码第142-147行：
        float priceChange = ta.change(close)
        float avgChange   = ta.sma(priceChange, 50)
        float stdChange   = ta.stdev(priceChange, 50)
        float momentumZ   = (priceChange - avgChange) / stdChange
        float volPercent  = ta.percentrank(volume, 100)

        Args:
            df: 包含 OHLCV 的 DataFrame

        Returns:
            添加了指标的 DataFrame
        """
        df = df.copy()

        # Pine源码第142行：价格变化
        # ta.change(close) = close - close[1]
        df['price_change'] = df['close'].diff()

        # Pine源码第143-146行：动量Z-Score计算
        df['avg_change'] = df['price_change'].rolling(50).mean()
        df['std_change'] = df['price_change'].rolling(50).std()
        df['momentum_z'] = (df['price_change'] - df['avg_change']) / df['std_change']

        # Pine源码第147行：成交量百分位
        # ta.percentrank(volume, 100) = 当前成交量在过去100期的百分位排名
        df['vol_percentile'] = df['volume'].rolling(100).rank(pct=True) * 100

        # 计算枢轴点 (Pine源码第149-163行)
        pivot_high, pivot_low = calculate_pivot_points(df, self.pivot_len)
        df['pivot_high'] = pivot_high
        df['pivot_low'] = pivot_low

        # 计算ATR (用于止损)
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], self.atr_period)

        # 添加K线索引
        df['bar_index'] = range(len(df))

        return df

    def detect_msb_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        检测MSB信号

        对应Pine源码第173-174行：
        bool isMSBBull = close > lastPh and close[1] <= lastPh and momentumZ > msbZScoreInput
        bool isMSBBear = close < lastPl and close[1] >= lastPl and momentumZ < -msbZScoreInput

        MSB条件：
        1. 价格突破枢轴点
        2. 前一根K线未突破 (确认真正的突破)
        3. 动量Z-Score超过阈值 (过滤弱突破)

        Args:
            df: 包含指标的 DataFrame

        Returns:
            添加了MSB信号的 DataFrame
        """
        df = df.copy()

        # 跟踪最近的枢轴点 (Pine源码第153-163行)
        last_ph = np.nan
        last_pl = np.nan
        last_ph_idx = -1
        last_pl_idx = -1

        # 初始化信号列
        df['msb_bullish'] = False
        df['msb_bearish'] = False
        df['last_ph'] = np.nan
        df['last_pl'] = np.nan

        for i in range(len(df)):
            # 更新最近的枢轴点
            if not pd.isna(df['pivot_high'].iloc[i]):
                last_ph = df['pivot_high'].iloc[i]
                last_ph_idx = df['bar_index'].iloc[i] - self.pivot_len

            if not pd.isna(df['pivot_low'].iloc[i]):
                last_pl = df['pivot_low'].iloc[i]
                last_pl_idx = df['bar_index'].iloc[i] - self.pivot_len

            # 保存当前的枢轴点
            df.loc[df.index[i], 'last_ph'] = last_ph
            df.loc[df.index[i], 'last_pl'] = last_pl

            # 检测MSB信号 (需要i > 0才能获取close[1])
            if i > 0 and not pd.isna(last_ph):
                close = df['close'].iloc[i]
                close_prev = df['close'].iloc[i-1]
                momentum_z = df['momentum_z'].iloc[i]

                # 看涨MSB (Pine源码第173行)
                if (close > last_ph and close_prev <= last_ph and
                    momentum_z > self.msb_zscore):
                    df.loc[df.index[i], 'msb_bullish'] = True

            if i > 0 and not pd.isna(last_pl):
                close = df['close'].iloc[i]
                close_prev = df['close'].iloc[i-1]
                momentum_z = df['momentum_z'].iloc[i]

                # 看跌MSB (Pine源码第174行)
                if (close < last_pl and close_prev >= last_pl and
                    momentum_z < -self.msb_zscore):
                    df.loc[df.index[i], 'msb_bearish'] = True

        return df

    def find_order_block_kline(self, df: pd.DataFrame, msb_idx: int, msb_type: str) -> Optional[int]:
        """
        在MSB发生后回溯找订单块K线

        对应Pine源码第187-192行：
        if isMSBBull or isMSBBear
            int obIdx = 0
            for i = 1 to 10
                if (isMSBBull and close[i] < open[i]) or (isMSBBear and close[i] > open[i])
                    obIdx := i
                    break

        逻辑：
        - 看涨MSB: 回溯找阴线 (close < open) - 机构在卖出区域做多
        - 看跌MSB: 回溯找阳线 (close > open) - 机构在买入区域做空

        Args:
            df: DataFrame
            msb_idx: MSB发生的K线索引
            msb_type: MSB类型 ('bullish' or 'bearish')

        Returns:
            OB K线的偏移量 (1-10)，如果没找到返回None
        """
        # 回溯最多10根K线 (Pine源码第189行)
        for offset in range(1, 11):
            kline_idx = msb_idx - offset
            if kline_idx < 0:
                break

            close = df['close'].iloc[kline_idx]
            open_price = df['open'].iloc[kline_idx]

            # 看涨MSB找阴线，看跌MSB找阳线 (Pine源码第190行)
            if (msb_type == 'bullish' and close < open_price) or \
               (msb_type == 'bearish' and close > open_price):
                return offset

        return None

    def calculate_ob_quality(self, momentum_z: float, vol_percentile: float) -> Tuple[float, bool]:
        """
        计算OB质量分数

        对应Pine源码第199行：
        float score = math.min(100, (math.abs(momentumZ) * 20) + (volPercent * 0.5))
        bool isHPZ = score > 80

        评分公式：
        score = min(100, |Z-Score| * 20 + volPercentile * 0.5)

        Args:
            momentum_z: 动量Z-Score
            vol_percentile: 成交量百分位

        Returns:
            (score, is_hpz): 质量分数和是否为高概率区域
        """
        score = min(100, abs(momentum_z) * 20 + vol_percentile * 0.5)
        is_hpz = score > self.hpz_threshold
        return score, is_hpz

    def create_order_block(
        self,
        df: pd.DataFrame,
        msb_idx: int,
        msb_type: str,
        ob_offset: int
    ) -> OrderBlock:
        """
        创建订单块

        对应Pine源码第194-228行

        Args:
            df: DataFrame
            msb_idx: MSB发生的K线索引
            msb_type: MSB类型
            ob_offset: OB K线相对MSB的偏移量

        Returns:
            OrderBlock对象
        """
        ob_idx = msb_idx - ob_offset

        # Pine源码第194-196行：OB价格范围
        ob_top = df['high'].iloc[ob_idx]
        ob_bottom = df['low'].iloc[ob_idx]
        ob_poc = (ob_top + ob_bottom) / 2

        # Pine源码第198-200行：质量评分
        momentum_z = df['momentum_z'].iloc[msb_idx]
        vol_percentile = df['vol_percentile'].iloc[msb_idx]
        score, is_hpz = self.calculate_ob_quality(momentum_z, vol_percentile)

        # 创建OB对象
        ob = OrderBlock(
            id=str(uuid.uuid4()),
            ob_type=msb_type,
            timestamp=df.index[msb_idx],
            ob_index=ob_idx,
            top=ob_top,
            bottom=ob_bottom,
            poc=ob_poc,
            width=ob_top - ob_bottom,
            quality_score=score,
            is_hpz=is_hpz
        )

        return ob

    def check_ob_mitigation(self, ob: OrderBlock, current_bar: pd.Series) -> bool:
        """
        检查OB是否失效

        对应Pine源码第252行：
        bool isMitigated = ob.isBull ? low < ob.bottom : high > ob.top

        失效条件：
        - 看涨OB: 价格跌破OB底部 (low < ob.bottom)
        - 看跌OB: 价格突破OB顶部 (high > ob.top)

        Args:
            ob: OrderBlock对象
            current_bar: 当前K线数据

        Returns:
            True表示OB已失效
        """
        if ob.ob_type == 'bullish':
            # 看涨OB失效：价格跌破OB底部
            return current_bar['low'] < ob.bottom
        else:
            # 看跌OB失效：价格突破OB顶部
            return current_bar['high'] > ob.top

    def check_overlap(self, new_ob: OrderBlock) -> bool:
        """
        检查新OB是否与现有OB重叠

        对应Pine源码第210-216行

        Args:
            new_ob: 新的OrderBlock

        Returns:
            True表示有重叠
        """
        for ob in self.active_obs:
            if not ob.is_mitigated:
                # 检查重叠 (Pine源码第214行)
                overlap = (
                    (new_ob.top <= ob.top and new_ob.top >= ob.bottom) or
                    (new_ob.bottom <= ob.top and new_ob.bottom >= ob.bottom)
                )
                if overlap:
                    return True
        return False

    def identify_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        识别和管理订单块

        对应Pine源码第186-234行

        Args:
            df: 包含MSB信号的DataFrame

        Returns:
            更新后的DataFrame
        """
        df = df.copy()

        # 初始化OB相关列
        df['ob_created'] = False
        df['ob_id'] = None
        df['ob_top'] = np.nan
        df['ob_bottom'] = np.nan
        df['ob_score'] = np.nan
        df['ob_is_hpz'] = False

        # 遍历每根K线，检测MSB并创建OB
        for i in range(len(df)):
            # 获取当前K线索引
            bar_idx = df['bar_index'].iloc[i]

            # 检查是否有MSB信号
            is_msb_bull = df['msb_bullish'].iloc[i]
            is_msb_bear = df['msb_bearish'].iloc[i]

            if is_msb_bull or is_msb_bear:
                msb_type = 'bullish' if is_msb_bull else 'bearish'

                # Pine源码第187-192行：回溯找OB K线
                ob_offset = self.find_order_block_kline(df, i, msb_type)

                if ob_offset is not None:
                    # Pine源码第194-228行：创建OB
                    ob = self.create_order_block(df, i, msb_type, ob_offset)

                    # Pine源码第210-216行：检查重叠
                    if not self.check_overlap(ob):
                        # Pine源码第228-229行：添加到OB数组
                        self.active_obs.append(ob)
                        self.total_obs += 1

                        # 记录到DataFrame
                        df.loc[df.index[i], 'ob_created'] = True
                        df.loc[df.index[i], 'ob_id'] = ob.id
                        df.loc[df.index[i], 'ob_top'] = ob.top
                        df.loc[df.index[i], 'ob_bottom'] = ob.bottom
                        df.loc[df.index[i], 'ob_score'] = ob.quality_score
                        df.loc[df.index[i], 'ob_is_hpz'] = ob.is_hpz

                        # 保存MSB信号
                        msb_signal = MSBSignal(
                            id=str(uuid.uuid4()),
                            signal_type=msb_type,
                            timestamp=df.index[i],
                            bar_index=bar_idx,
                            pivot_price=df['last_ph'].iloc[i] if msb_type == 'bullish' else df['last_pl'].iloc[i],
                            pivot_index=bar_idx,  # 简化处理
                            momentum_z=df['momentum_z'].iloc[i],
                            volume_percentile=df['vol_percentile'].iloc[i]
                        )
                        self.msb_signals.append(msb_signal)

                    # Pine源码第289-293行：限制OB数量
                    if len(self.active_obs) > self.ob_max_count:
                        old_ob = self.active_obs.pop(0)
                        # 可以在这里清理旧OB的可视化元素

        return df

    def update_ob_states(self, df: pd.DataFrame) -> None:
        """
        更新所有OB的失效状态

        对应Pine源码第243-287行

        Args:
            df: DataFrame
        """
        # 从后往前遍历（与Pine源码一致）
        for ob in reversed(self.active_obs):
            # 获取OB创建后的K线
            ob_bars = df[df['bar_index'] > ob.ob_index]

            for idx, bar in ob_bars.iterrows():
                if ob.is_mitigated:
                    break

                # Pine源码第252行：检查失效
                if self.check_ob_mitigation(ob, bar):
                    ob.is_mitigated = True
                    ob.mitigation_timestamp = idx
                    ob.mitigation_bar_index = bar['bar_index']
                    self.total_mitigated += 1
                    break

        # 将失效的OB移到mitigated列表
        still_active = []
        for ob in self.active_obs:
            if ob.is_mitigated:
                self.mitigated_obs.append(ob)
            else:
                still_active.append(ob)
        self.active_obs = still_active

    def generate_entry_signals(self, df: pd.DataFrame) -> List[Dict]:
        """
        生成入场信号

        入场条件：
        1. OB未失效
        2. 价格到达OB区域
        3. 可选：只交易HPZ
        4. 未超过最大持仓数

        Args:
            df: DataFrame

        Returns:
            入场信号列表
        """
        entry_signals = []

        # 如果已达到最大持仓数，不再开仓
        if len(self.active_positions) >= self.max_positions:
            return entry_signals

        for ob in self.active_obs:
            # 检查是否可以交易此OB
            if ob.is_mitigated:
                continue

            # 可选过滤：只交易HPZ
            if self.hpz_only and not ob.is_hpz:
                continue

            # 检查是否已经基于此OB开仓
            already_traded = any(pos.ob_id == ob.id for pos in self.active_positions)
            if already_traded:
                continue

            # 获取OB创建后的K线
            ob_bars = df[df['bar_index'] > ob.ob_index]

            for idx, bar in ob_bars.iterrows():
                entry_price = None
                entry_triggered = False

                if ob.ob_type == 'bullish':
                    # 看涨OB：等待价格回调到OB区域
                    # 入场价格 = OB底部 + 缓冲
                    target_price = ob.bottom * (1 + self.entry_buffer)

                    if bar['low'] <= target_price and bar['close'] > ob.bottom:
                        entry_price = min(bar['open'], target_price)
                        entry_triggered = True

                else:  # bearish
                    # 看跌OB：等待价格反弹到OB区域
                    # 入场价格 = OB顶部 - 缓冲
                    target_price = ob.top * (1 - self.entry_buffer)

                    if bar['high'] >= target_price and bar['close'] < ob.top:
                        entry_price = max(bar['open'], target_price)
                        entry_triggered = True

                if entry_triggered:
                    # 计算止损和止盈
                    atr = bar['atr']

                    if ob.ob_type == 'bullish':
                        stop_loss = ob.bottom - atr * self.atr_multiplier
                        tp1 = entry_price + ob.width * self.tp1_multiplier
                        tp2 = entry_price + ob.width * self.tp2_multiplier
                        tp3 = entry_price + ob.width * self.tp3_multiplier
                        pos_type = 'long'
                    else:
                        stop_loss = ob.top + atr * self.atr_multiplier
                        tp1 = entry_price - ob.width * self.tp1_multiplier
                        tp2 = entry_price - ob.width * self.tp2_multiplier
                        tp3 = entry_price - ob.width * self.tp3_multiplier
                        pos_type = 'short'

                    signal = {
                        'timestamp': idx,
                        'bar_index': bar['bar_index'],
                        'ob_id': ob.id,
                        'ob_type': ob.ob_type,
                        'entry_price': entry_price,
                        'stop_loss': stop_loss,
                        'take_profits': {'tp1': tp1, 'tp2': tp2, 'tp3': tp3},
                        'pos_type': pos_type,
                        'quality_score': ob.quality_score,
                        'is_hpz': ob.is_hpz
                    }

                    entry_signals.append(signal)
                    break

        return entry_signals

    def generate_exit_signals(self, df: pd.DataFrame, position: Position) -> Optional[Dict]:
        """
        生成出场信号

        出场条件：
        1. 止损触发
        2. 止盈触发 (TP1/TP2/TP3)

        Args:
            df: DataFrame
            position: 持仓对象

        Returns:
            出场信号字典，如果没有触发返回None
        """
        # 获取持仓后的K线
        entry_bars = df[df['bar_index'] > position.entry_bar_index]

        for idx, bar in entry_bars.iterrows():
            if not position.is_active:
                break

            exit_triggered = False
            exit_reason = None
            exit_price = None

            if position.pos_type == 'long':
                # 多头持仓
                if bar['low'] <= position.stop_loss:
                    # 止损
                    exit_triggered = True
                    exit_reason = 'stop'
                    exit_price = min(bar['open'], position.stop_loss)

                elif bar['high'] >= position.take_profits['tp1']:
                    # TP1触发
                    exit_triggered = True
                    exit_reason = 'tp1'
                    exit_price = max(bar['open'], position.take_profits['tp1'])

            else:  # short
                # 空头持仓
                if bar['high'] >= position.stop_loss:
                    # 止损
                    exit_triggered = True
                    exit_reason = 'stop'
                    exit_price = max(bar['open'], position.stop_loss)

                elif bar['low'] <= position.take_profits['tp1']:
                    # TP1触发
                    exit_triggered = True
                    exit_reason = 'tp1'
                    exit_price = min(bar['open'], position.take_profits['tp1'])

            if exit_triggered:
                # 计算盈亏
                if position.pos_type == 'long':
                    pnl = exit_price - position.entry_price
                else:
                    pnl = position.entry_price - exit_price

                return {
                    'timestamp': idx,
                    'bar_index': bar['bar_index'],
                    'exit_price': exit_price,
                    'exit_reason': exit_reason,
                    'pnl': pnl
                }

        return None

    def manage_positions(self, df: pd.DataFrame) -> None:
        """
        管理持仓，处理出场

        Args:
            df: DataFrame
        """
        positions_to_close = []

        for position in self.active_positions:
            exit_signal = self.generate_exit_signals(df, position)

            if exit_signal is not None:
                position.exit_price = exit_signal['exit_price']
                position.exit_timestamp = exit_signal['timestamp']
                position.exit_reason = exit_signal['exit_reason']
                position.pnl = exit_signal['pnl']
                position.is_active = False
                positions_to_close.append(position)

        # 将平仓的持仓移到closed列表
        for position in positions_to_close:
            self.active_positions.remove(position)
            self.closed_positions.append(position)

    def execute_entry_signals(self, entry_signals: List[Dict], df: pd.DataFrame) -> None:
        """
        执行入场信号，创建持仓

        Args:
            entry_signals: 入场信号列表
            df: DataFrame
        """
        for signal in entry_signals:
            # 如果已达到最大持仓数，停止开仓
            if len(self.active_positions) >= self.max_positions:
                break

            # 计算仓位大小 (简化为固定大小)
            position_size = 1.0

            # 创建持仓
            position = Position(
                id=str(uuid.uuid4()),
                pos_type=signal['pos_type'],
                entry_price=signal['entry_price'],
                entry_timestamp=signal['timestamp'],
                entry_bar_index=signal['bar_index'],
                stop_loss=signal['stop_loss'],
                take_profits=signal['take_profits'],
                position_size=position_size,
                ob_id=signal['ob_id'],
                is_active=True
            )

            self.active_positions.append(position)

    def generate_signal(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        主信号生成函数

        整合所有逻辑：
        1. 计算指标
        2. 检测MSB信号
        3. 识别订单块
        4. 更新OB状态
        5. 生成入场信号
        6. 管理持仓

        Args:
            df: 包含 OHLCV 的 DataFrame

        Returns:
            添加了信号的 DataFrame
        """
        # 重置状态
        self.active_obs = []
        self.mitigated_obs = []
        self.active_positions = []
        self.closed_positions = []
        self.msb_signals = []
        self.total_obs = 0
        self.total_mitigated = 0

        # 1. 计算指标
        df = self.calculate_indicators(df)

        # 2. 检测MSB信号
        df = self.detect_msb_signals(df)

        # 3. 识别订单块
        df = self.identify_order_blocks(df)

        # 4. 更新OB状态 (检查失效)
        self.update_ob_states(df)

        # 5. 生成入场信号
        entry_signals = self.generate_entry_signals(df)

        # 6. 执行入场
        self.execute_entry_signals(entry_signals, df)

        # 7. 管理持仓 (处理出场)
        self.manage_positions(df)

        # 8. 将持仓信号添加到DataFrame
        df['position'] = 0
        for position in self.active_positions:
            mask = df.index >= position.entry_timestamp
            if position.exit_timestamp is not None:
                mask = mask & (df.index <= position.exit_timestamp)

            pos_value = 1 if position.pos_type == 'long' else -1
            df.loc[mask, 'position'] = pos_value

        # 添加持仓详情
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan

        for position in self.active_positions + self.closed_positions:
            mask = df.index >= position.entry_timestamp
            if position.exit_timestamp is not None:
                mask = mask & (df.index <= position.exit_timestamp)

            df.loc[mask, 'entry_price'] = position.entry_price
            df.loc[mask, 'stop_loss'] = position.stop_loss
            df.loc[mask, 'take_profit'] = position.take_profits['tp1']

        # 添加统计信息
        df['total_ob'] = self.total_obs
        df['active_ob_count'] = len(self.active_obs)
        df['mitigated_ob_count'] = len(self.mitigated_obs)
        df['hpz_count'] = sum(1 for ob in self.active_obs if ob.is_hpz and not ob.is_mitigated)

        return df

    def get_statistics(self) -> Dict:
        """
        获取策略统计信息

        对应Pine源码第306-327行的Dashboard统计

        Returns:
            统计信息字典
        """
        # Pine源码第307行：Reliability = 已失效OB数 / 总OB数 * 100%
        efficiency = (self.total_mitigated / self.total_obs * 100) if self.total_obs > 0 else 0

        # Pine源码第308-312行：HP-OB数量
        hpz_count = sum(1 for ob in self.active_obs if ob.is_hpz and not ob.is_mitigated)

        # 交易统计
        total_trades = len(self.closed_positions) + len(self.active_positions)
        winning_trades = sum(1 for p in self.closed_positions if p.pnl and p.pnl > 0)
        losing_trades = sum(1 for p in self.closed_positions if p.pnl and p.pnl < 0)
        win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

        total_pnl = sum(p.pnl for p in self.closed_positions if p.pnl)

        stats = {
            # Pine源码统计
            'total_ob_created': self.total_obs,
            'total_ob_mitigated': self.total_mitigated,
            'ob_reliability': efficiency,  # 对应源码的Reliability
            'hpz_active_count': hpz_count,  # 对应源码的HP-OB Count
            'active_ob_count': len(self.active_obs),

            # 交易统计
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate,
            'total_pnl': total_pnl,
            'active_positions': len(self.active_positions)
        }

        return stats


# ========================================================================
# 便捷函数
# ========================================================================

def create_msb_ob_strategy(
    pivot_len: int = 7,
    msb_zscore: float = 0.5,
    hpz_only: bool = False,
    **kwargs
) -> MSBOBStrategy:
    """
    创建MSB+OB策略实例的便捷函数

    Args:
        pivot_len: 枢轴点回看长度
        msb_zscore: MSB动量阈值
        hpz_only: 是否只交易HPZ
        **kwargs: 其他参数

    Returns:
        MSBOBStrategy实例
    """
    return MSBOBStrategy(
        pivot_len=pivot_len,
        msb_zscore=msb_zscore,
        hpz_only=hpz_only,
        **kwargs
    )


def analyze_signals(df: pd.DataFrame, strategy: MSBOBStrategy) -> pd.DataFrame:
    """
    分析信号并返回详细结果

    Args:
        df: OHLCV数据
        strategy: 策略实例

    Returns:
        带信号的DataFrame
    """
    df_with_signals = strategy.generate_signal(df)
    stats = strategy.get_statistics()

    print("\n" + "="*60)
    print("MSB+OB 策略统计")
    print("="*60)
    print(f"\n订单块统计 (对应Pine源码Dashboard):")
    print(f"  总OB数: {stats['total_ob_created']}")
    print(f"  已失效OB数: {stats['total_ob_mitigated']}")
    print(f"  可靠性 (Reliability): {stats['ob_reliability']:.2f}%")
    print(f"  活跃HPZ数量 (HP-OB Count): {stats['hpz_active_count']}")
    print(f"  活跃OB数量: {stats['active_ob_count']}")

    print(f"\n交易统计:")
    print(f"  总交易数: {stats['total_trades']}")
    print(f"  盈利交易: {stats['winning_trades']}")
    print(f"  亏损交易: {stats['losing_trades']}")
    print(f"  胜率: {stats['win_rate']:.2f}%")
    print(f"  总盈亏: {stats['total_pnl']:.2f}")

    return df_with_signals
