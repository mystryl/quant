"""
趋势特征工程模块

基于 60min 数据计算趋势特征，用于预测未来20根K线的趋势方向。

重要：所有特征必须 shift(1) 避免未来函数污染！

特征分类：
1. 斜率类：EMA60/20 slope, TWAP slope, 线性回归斜率
2. 趋势强度：ADX, ADX 变化率, ATR, ATR/price
3. 结构类：金叉死叉, K线突破, 高低点突破, 均线排列
4. 波动率：rolling std, Parkinson 波动率
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class TrendFeatures:
    """趋势特征计算器"""

    def __init__(self, ema_short=20, ema_long=60, adx_period=14, atr_period=14):
        """
        初始化特征计算器

        Args:
            ema_short: 短期EMA周期
            ema_long: 长期EMA周期
            adx_period: ADX计算周期
            atr_period: ATR计算周期
        """
        self.ema_short = ema_short
        self.ema_long = ema_long
        self.adx_period = adx_period
        self.atr_period = atr_period

        logger.info(f"趋势特征计算器初始化:")
        logger.info(f"  EMA短期: {ema_short}, EMA长期: {ema_long}")
        logger.info(f"  ADX周期: {adx_period}, ATR周期: {atr_period}")

    def calculate_ema(self, prices: pd.Series, period: int) -> pd.Series:
        """计算指数移动平均"""
        return prices.ewm(span=period, adjust=False).mean()

    def calculate_slope(self, series: pd.Series, period: int = 5) -> pd.Series:
        """
        计算序列的斜率（使用线性回归）

        Args:
            series: 价格序列
            period: 计算斜率的窗口期

        Returns:
            斜率序列
        """
        slopes = pd.Series(index=series.index, dtype=float)

        for i in range(period - 1, len(series)):
            y = series.iloc[i - period + 1:i + 1].values
            x = np.arange(period)
            slope, _ = np.polyfit(x, y, 1)
            slopes.iloc[i] = slope

        return slopes

    # ==================== 1. 斜率类特征 ====================

    def compute_slope_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算斜率类特征

        特征列表：
        - ema60_slope: EMA60的斜率
        - ema20_slope: EMA20的斜率
        - twap_slope: TWAP的斜率
        - linear_reg_slope: 价格的线性回归斜率
        """
        logger.info("\n计算斜率类特征...")

        close = df['close']

        # 1. EMA斜率
        ema60 = self.calculate_ema(close, self.ema_long)
        ema20 = self.calculate_ema(close, self.ema_short)

        df['ema60_slope'] = self.calculate_slope(ema60, period=5)
        df['ema20_slope'] = self.calculate_slope(ema20, period=5)

        # 归一化斜率（除以价格，使其无量纲）
        df['ema60_slope_norm'] = df['ema60_slope'] / close
        df['ema20_slope_norm'] = df['ema20_slope'] / close

        logger.info(f"  ✓ EMA60斜率: 均值={df['ema60_slope_norm'].mean():.6f}")
        logger.info(f"  ✓ EMA20斜率: 均值={df['ema20_slope_norm'].mean():.6f}")

        # 2. TWAP斜率
        twap = df['close']  # 简化：使用收盘价
        df['twap_slope'] = self.calculate_slope(twap, period=5)
        df['twap_slope_norm'] = df['twap_slope'] / close

        logger.info(f"  ✓ TWAP斜率: 均值={df['twap_slope_norm'].mean():.6f}")

        # 3. 线性回归斜率
        df['linear_reg_slope'] = self.calculate_slope(close, period=10)
        df['linear_reg_slope_norm'] = df['linear_reg_slope'] / close

        logger.info(f"  ✓ 线性回归斜率: 均值={df['linear_reg_slope_norm'].mean():.6f}")

        return df

    # ==================== 2. 趋势强度特征 ====================

    def compute_adx(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算 ADX (Average Directional Index)

        Args:
            df: 包含high, low, close的DataFrame
            period: 计算周期

        Returns:
            ADX值序列
        """
        high = df['high']
        low = df['low']
        close = df['close']

        # 计算+DM和-DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # 计算TR (True Range)
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # 计算+DI和-DI
        plus_di = 100 * (plus_dm.rolling(window=period).sum() /
                         tr.rolling(window=period).sum())
        minus_di = 100 * (minus_dm.rolling(window=period).sum() /
                          tr.rolling(window=period).sum())

        # 计算DX和ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(window=period).mean()

        return adx

    def compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """
        计算 ATR (Average True Range)

        Args:
            df: 包含high, low, close的DataFrame
            period: 计算周期

        Returns:
            ATR值序列
        """
        high = df['high']
        low = df['low']
        close = df['close']

        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.rolling(window=period).mean()

        return atr

    def compute_strength_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算趋势强度特征

        特征列表：
        - adx: ADX指标
        - adx_change: ADX变化率
        - atr: ATR指标
        - atr_ratio: ATR / price
        """
        logger.info("\n计算趋势强度特征...")

        close = df['close']

        # 1. ADX
        df['adx'] = self.compute_adx(df, self.adx_period)
        df['adx_change'] = df['adx'].diff()
        df['adx_change_pct'] = df['adx'].pct_change(fill_method=None)

        logger.info(f"  ✓ ADX: 均值={df['adx'].mean():.2f}")
        logger.info(f"  ✓ ADX变化率: 均值={df['adx_change_pct'].mean():.4f}")

        # 2. ATR
        df['atr'] = self.compute_atr(df, self.atr_period)
        df['atr_ratio'] = df['atr'] / close

        logger.info(f"  ✓ ATR: 均值={df['atr'].mean():.2f}")
        logger.info(f"  ✓ ATR/价格: 均值={df['atr_ratio'].mean():.4f}")

        return df

    # ==================== 3. 结构类特征 ====================

    def compute_structure_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算结构类特征

        特征列表：
        - golden_cross: EMA5/60 金叉死叉
        - close_ema60_ratio: 收盘价与EMA60比率（突破）
        - high_low_break: 高低点突破距离
        - ma_alignment: 均线排列
        - consecutive_bars: 多头/空头连续排列
        """
        logger.info("\n计算结构类特征...")

        close = df['close']
        high = df['high']
        low = df['low']

        # 计算EMA
        ema5 = self.calculate_ema(close, 5)
        ema20 = self.calculate_ema(close, 20)
        ema60 = self.calculate_ema(close, 60)

        # 1. 金叉死叉
        df['ema5_ema60_ratio'] = ema5 / ema60
        df['golden_cross'] = (df['ema5_ema60_ratio'] > 1).astype(int)  # 1=金叉, 0=死叉

        logger.info(f"  ✓ 金叉信号: {(df['golden_cross'] == 1).sum()} 个")
        logger.info(f"  ✓ 死叉信号: {(df['golden_cross'] == 0).sum()} 个")

        # 2. K线突破
        df['close_ema60_ratio'] = close / ema60
        df['close_above_ema60'] = (close > ema60).astype(int)

        # 3. 高低点突破
        df['high_20_high'] = high.rolling(window=20).max()
        df['low_20_low'] = low.rolling(window=20).min()
        df['high_break_ratio'] = (high - df['high_20_high'].shift(1)) / df['high_20_high'].shift(1)
        df['low_break_ratio'] = (df['low_20_low'].shift(1) - low) / df['low_20_low'].shift(1)

        logger.info(f"  ✓ 高点突破距离: 均值={df['high_break_ratio'].mean():.4f}")
        logger.info(f"  ✓ 低点突破距离: 均值={df['low_break_ratio'].mean():.4f}")

        # 4. 均线排列
        # 多头排列: EMA5 > EMA20 > EMA60
        ma_bullish = (ema5 > ema20) & (ema20 > ema60)
        # 空头排列: EMA5 < EMA20 < EMA60
        ma_bearish = (ema5 < ema20) & (ema20 < ema60)

        df['ma_alignment'] = 0
        df.loc[ma_bullish, 'ma_alignment'] = 1   # 多头排列
        df.loc[ma_bearish, 'ma_alignment'] = -1  # 空头排列

        logger.info(f"  ✓ 多头排列: {(df['ma_alignment'] == 1).sum()} 个")
        logger.info(f"  ✓ 空头排列: {(df['ma_alignment'] == -1).sum()} 个")
        logger.info(f"  ✓ 无序排列: {(df['ma_alignment'] == 0).sum()} 个")

        # 5. 连续排列（多头/空头连续K线数）
        df['consecutive_bullish'] = (close > close.shift(1)).astype(int)
        df['consecutive_bearish'] = (close < close.shift(1)).astype(int)

        df['bullish_streak'] = df['consecutive_bullish'].groupby(
            (df['consecutive_bullish'] != df['consecutive_bullish'].shift()).cumsum()
        ).cumsum()

        df['bearish_streak'] = df['consecutive_bearish'].groupby(
            (df['consecutive_bearish'] != df['consecutive_bearish'].shift()).cumsum()
        ).cumsum()

        logger.info(f"  ✓ 多头连续K线: 平均={df['bullish_streak'].mean():.2f}")
        logger.info(f"  ✓ 空头连续K线: 平均={df['bearish_streak'].mean():.2f}")

        return df

    # ==================== 4. 波动率特征 ====================

    def compute_volatility_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算波动率特征

        特征列表：
        - rolling_std: 收盘价滚动标准差
        - parkinson_vol: Parkinson波动率
        """
        logger.info("\n计算波动率特征...")

        close = df['close']
        high = df['high']
        low = df['low']

        # 1. 滚动标准差
        for period in [5, 10, 20]:
            df[f'rolling_std_{period}'] = close.rolling(window=period).std()
            df[f'rolling_std_{period}_ratio'] = df[f'rolling_std_{period}'] / close

        logger.info(f"  ✓ 滚动标准差(20): 均值={df['rolling_std_20_ratio'].mean():.4f}")

        # 2. Parkinson波动率
        # 使用高低价计算，比收盘价更准确
        for period in [5, 10, 20]:
            log_hl = np.log(high / low)
            parkinson = np.sqrt((log_hl ** 2).rolling(window=period).mean() / (4 * np.log(2)))
            df[f'parkinson_vol_{period}'] = parkinson

        logger.info(f"  ✓ Parkinson波动率(20): 均值={df['parkinson_vol_20'].mean():.4f}")

        return df

    # ==================== 5. 技术指标特征 ====================

    def compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """
        计算 RSI (Relative Strength Index)

        Args:
            prices: 价格序列
            period: 计算周期

        Returns:
            RSI值序列
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()

        rs = gain / loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def compute_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple:
        """
        计算 MACD (Moving Average Convergence Divergence)

        Args:
            prices: 价格序列
            fast: 快线周期
            slow: 慢线周期
            signal: 信号线周期

        Returns:
            (MACD线, 信号线, 柱状图)
        """
        ema_fast = prices.ewm(span=fast, adjust=False).mean()
        ema_slow = prices.ewm(span=slow, adjust=False).mean()

        macd_line = ema_fast - ema_slow
        signal_line = macd_line.ewm(span=signal, adjust=False).mean()
        histogram = macd_line - signal_line

        return macd_line, signal_line, histogram

    def compute_bollinger_bands(self, prices: pd.Series, period: int = 20, num_std: float = 2) -> tuple:
        """
        计算布林带

        Args:
            prices: 价格序列
            period: 周期
            num_std: 标准差倍数

        Returns:
            (上轨, 中轨, 下轨)
        """
        sma = prices.rolling(window=period).mean()
        std = prices.rolling(window=period).std()

        upper_band = sma + (std * num_std)
        lower_band = sma - (std * num_std)

        return upper_band, sma, lower_band

    def compute_technical_indicator_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        计算技术指标特征

        特征列表：
        - MACD: MACD线, 信号线, 柱状图
        - RSI: 相对强弱指标
        - 布林带位置: 价格相对于布林带的位置
        - 成交量变化率: volume的百分比变化
        - 价格加速度: 价格的二阶导数
        """
        logger.info("\n计算技术指标特征...")

        close = df['close']
        high = df['high']
        low = df['low']
        volume = df['volume']

        # 1. MACD (12, 26, 9)
        macd_line, signal_line, histogram = self.compute_macd(close, fast=12, slow=26, signal=9)

        df['macd'] = macd_line
        df['macd_signal'] = signal_line
        df['macd_histogram'] = histogram
        df['macd_histogram_norm'] = histogram / close  # 归一化

        logger.info(f"  ✓ MACD柱状图: 均值={df['macd_histogram_norm'].mean():.6f}")

        # 2. RSI (14期)
        for period in [14, 6, 24]:
            df[f'rsi_{period}'] = self.compute_rsi(close, period=period)
            # RSI归一化到[-1, 1]（50为中性）
            df[f'rsi_{period}_norm'] = (df[f'rsi_{period}'] - 50) / 50

        logger.info(f"  ✓ RSI(14): 均值={df['rsi_14_norm'].mean():.4f}")

        # 3. 布林带位置 (20期, 2倍标准差)
        upper_band, middle_band, lower_band = self.compute_bollinger_bands(close, period=20, num_std=2)

        df['bb_upper'] = upper_band
        df['bb_middle'] = middle_band
        df['bb_lower'] = lower_band

        # 布林带宽度（波动率指标）
        df['bb_width'] = (upper_band - lower_band) / middle_band

        # 价格在布林带中的位置（0=下轨, 1=上轨）
        df['bb_position'] = (close - lower_band) / (upper_band - lower_band)

        logger.info(f"  ✓ 布林带位置: 均值={df['bb_position'].mean():.4f}")
        logger.info(f"  ✓ 布林带宽度: 均值={df['bb_width'].mean():.4f}")

        # 4. 成交量变化率
        for period in [1, 5, 10]:
            df[f'volume_change_{period}'] = volume.pct_change(periods=period, fill_method=None)

        logger.info(f"  ✓ 成交量变化率(5): 均值={df['volume_change_5'].mean():.4f}")

        # 5. 价格加速度（二阶导数）
        # 一阶导数：价格变化率
        price_velocity = close.diff()
        # 二阶导数：加速度（变化率的变化）
        price_acceleration = price_velocity.diff()

        df['price_velocity'] = price_velocity
        df['price_acceleration'] = price_acceleration

        # 归一化
        df['price_velocity_norm'] = price_velocity / close
        df['price_acceleration_norm'] = price_acceleration / close

        logger.info(f"  ✓ 价格加速度: 均值={df['price_acceleration_norm'].mean():.6f}")

        return df

    # ==================== 主函数 ====================

    def compute_all_features(self, df: pd.DataFrame, shift: bool = True) -> pd.DataFrame:
        """
        计算所有趋势特征

        Args:
            df: 包含OHLCV数据的DataFrame
            shift: 是否对特征进行shift(1)以避免未来函数

        Returns:
            添加了所有特征的DataFrame
        """
        logger.info("\n" + "="*60)
        logger.info("开始计算趋势特征")
        logger.info("="*60)
        logger.info(f"输入数据: {df.shape}")

        # 记录原始列
        original_cols = df.columns.tolist()

        # 计算各类特征
        df = self.compute_slope_features(df)
        df = self.compute_strength_features(df)
        df = self.compute_structure_features(df)
        df = self.compute_volatility_features(df)
        df = self.compute_technical_indicator_features(df)

        # 防未来函数：shift(1)
        if shift:
            logger.info("\n执行 shift(1) 防止未来函数污染...")
            feature_cols = [col for col in df.columns if col not in original_cols]
            df[feature_cols] = df[feature_cols].shift(1)
            logger.info(f"  ✓ 已对 {len(feature_cols)} 个特征执行 shift(1)")

        # 统计特征数量
        new_features = [col for col in df.columns if col not in original_cols]
        logger.info(f"\n{'='*60}")
        logger.info(f"特征计算完成！")
        logger.info(f"{'='*60}")
        logger.info(f"新增特征数量: {len(new_features)}")
        logger.info(f"总特征数量: {len(df.columns)}")
        logger.info(f"有效特征样本: {df[new_features].notna().all(axis=1).sum()}")

        return df


def generate_trend_features(
    data_file: Path = None,
    output_file: Path = None
) -> pd.DataFrame:
    """
    加载标签数据并生成趋势特征

    Args:
        data_file: 标签数据文件路径
        output_file: 输出文件路径（如果提供则保存）

    Returns:
        包含特征和标签的DataFrame
    """
    if data_file is None:
        data_file = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data/labels/final_labels_20bars.csv')

    logger.info(f"\n加载标签数据: {data_file}")
    df = pd.read_csv(data_file)
    df['datetime'] = pd.to_datetime(df['datetime'])
    logger.info(f"  数据形状: {df.shape}")

    # 计算特征
    calculator = TrendFeatures(
        ema_short=20,
        ema_long=60,
        adx_period=14,
        atr_period=14
    )

    df = calculator.compute_all_features(df, shift=True)

    # 保存结果
    if output_file:
        output_file = Path(output_file)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_file, index=False)
        logger.info(f"\n✓ 结果已保存: {output_file}")

    return df


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 生成特征
    output_file = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data/features/trend_features.csv')
    df = generate_trend_features(output_file=output_file)

    print(f"\n最终数据形状: {df.shape}")
    print(f"列名: {list(df.columns)}")
