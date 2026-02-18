#!/usr/bin/env python3
"""
V03改进方案1+2：ATR分层止盈
- 改用ATR止盈（符合Pine原文）
- 实现分层平仓（TP1/TP2/TP3各1/3）

2023-2025年回测
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
import uuid
from dataclasses import dataclass
from typing import Optional, Dict, List

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


# ========================================================================
# 数据结构定义
# ========================================================================

@dataclass
class OrderBlock:
    """订单块数据结构"""
    id: str
    ob_type: str  # 'bullish' or 'bearish'
    timestamp: pd.Timestamp
    bar_index: int
    top: float
    bottom: float
    width: float
    width_ratio: float
    atr_value: float


@dataclass
class Position:
    """持仓数据结构（支持分层平仓）"""
    id: str
    pos_type: str  # 'long' or 'short'
    entry_price: float
    entry_timestamp: pd.Timestamp
    entry_bar_index: int
    stop_loss: float
    take_profits: Dict[str, float]
    ob_id: str

    # 分层止盈状态
    tp1_filled: bool = False  # TP1是否已触发
    tp2_filled: bool = False  # TP2是否已触发
    tp3_filled: bool = False  # TP3是否已触发

    # 剩余仓位（用1.0表示满仓）
    remaining_position: float = 1.0


# ========================================================================
# 指标计算函数
# ========================================================================

def calculate_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """计算平均真实波幅(ATR)"""
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_value = tr.rolling(period).mean()
    return atr_value


def calculate_momentum_z(close, window=50):
    """计算动量Z-Score"""
    price_change = close.diff()
    avg_change = price_change.rolling(window).mean()
    std_change = price_change.rolling(window).std()
    return (price_change - avg_change) / std_change


def detect_pivots(high, low, pivot_len=7):
    """检测枢轴点"""
    pivot_high = pd.Series(np.nan, index=high.index)
    pivot_low = pd.Series(np.nan, index=low.index)

    for i in range(pivot_len, len(high) - pivot_len):
        if high.iloc[i] == high.iloc[i-pivot_len:i+pivot_len+1].max():
            pivot_high.iloc[i] = high.iloc[i]
        if low.iloc[i] == low.iloc[i-pivot_len:i+pivot_len+1].min():
            pivot_low.iloc[i] = low.iloc[i]

    return pivot_high, pivot_low


def find_order_block_kline(df, msb_idx, msb_type, max_lookback=10):
    """在MSB发生后回溯找订单块K线"""
    for offset in range(1, max_lookback + 1):
        kline_idx = msb_idx - offset
        if kline_idx < 0:
            break

        close = df['close'].iloc[kline_idx]
        open_price = df['open'].iloc[kline_idx]

        if (msb_type == 'bullish' and close < open_price) or \
           (msb_type == 'bearish' and close > open_price):
            return offset

    return None


# ========================================================================
# ATR分层止盈策略类
# ========================================================================

class HybridATRPartialStrategy:
    """
    ATR分层止盈策略

    改进点：
    1. 止盈使用ATR倍数（符合Pine原文）
    2. 实现分层平仓（TP1/TP2/TP3各平1/3）
    """

    def __init__(
        self,
        pivot_len: int = 7,
        msb_zscore: float = 0.3,
        atr_period: int = 14,
        atr_multiplier: float = 1.0,
        tp1_multiplier: float = 0.5,
        tp2_multiplier: float = 1.0,
        tp3_multiplier: float = 1.5
    ):
        self.pivot_len = pivot_len
        self.msb_zscore = msb_zscore
        self.atr_period = atr_period
        self.atr_multiplier = atr_multiplier
        self.tp1_multiplier = tp1_multiplier
        self.tp2_multiplier = tp2_multiplier
        self.tp3_multiplier = tp3_multiplier

        # 状态变量
        self.active_obs: List[OrderBlock] = []
        self.active_positions: List[Position] = []
        self.closed_positions: List[Position] = []

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """计算所有技术指标"""
        df = df.copy()

        df['price_change'] = df['close'].diff()
        df['avg_change'] = df['price_change'].rolling(50).mean()
        df['std_change'] = df['price_change'].rolling(50).std()
        df['momentum_z'] = (df['price_change'] - df['avg_change']) / df['std_change']

        df['vol_percentile'] = df['volume'].rolling(100).rank(pct=True) * 100
        df['ema20'] = df['close'].ewm(span=20, adjust=False).mean()
        df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()

        df['pivot_high'], df['pivot_low'] = detect_pivots(df['high'], df['low'], self.pivot_len)
        df['atr'] = calculate_atr(df['high'], df['low'], df['close'], self.atr_period)
        df['bar_index'] = range(len(df))

        return df

    def detect_msb_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """检测MSB信号"""
        df = df.copy()

        last_ph = np.nan
        last_pl = np.nan

        df['msb_bullish'] = False
        df['msb_bearish'] = False

        for i in range(len(df)):
            # 更新枢轴点
            if not pd.isna(df['pivot_high'].iloc[i]):
                last_ph = df['pivot_high'].iloc[i]
            if not pd.isna(df['pivot_low'].iloc[i]):
                last_pl = df['pivot_low'].iloc[i]

            # 检测MSB信号
            if i > 0 and not pd.isna(last_ph):
                close = df['close'].iloc[i]
                close_prev = df['close'].iloc[i-1]
                momentum_z = df['momentum_z'].iloc[i]

                if (close > last_ph and close_prev <= last_ph and
                    momentum_z > self.msb_zscore):
                    df.loc[df.index[i], 'msb_bullish'] = True

            if i > 0 and not pd.isna(last_pl):
                close = df['close'].iloc[i]
                close_prev = df['close'].iloc[i-1]
                momentum_z = df['momentum_z'].iloc[i]

                if (close < last_pl and close_prev >= last_pl and
                    momentum_z < -self.msb_zscore):
                    df.loc[df.index[i], 'msb_bearish'] = True

        return df

    def identify_order_blocks(self, df: pd.DataFrame) -> pd.DataFrame:
        """识别订单块"""
        df = df.copy()

        df['ob_created'] = False
        df['ob_id'] = None
        df['ob_top'] = np.nan
        df['ob_bottom'] = np.nan

        for i in range(len(df)):
            is_msb_bull = df['msb_bullish'].iloc[i]
            is_msb_bear = df['msb_bearish'].iloc[i]

            if is_msb_bull or is_msb_bear:
                msb_type = 'bullish' if is_msb_bull else 'bearish'

                # 回溯找OB K线
                ob_offset = find_order_block_kline(df, i, msb_type)

                if ob_offset is not None:
                    ob_idx = i - ob_offset
                    atr = df['atr'].iloc[i]
                    width = df['high'].iloc[ob_idx] - df['low'].iloc[ob_idx]
                    width_ratio = width / atr if atr > 0 else 0

                    ob = OrderBlock(
                        id=str(uuid.uuid4()),
                        ob_type=msb_type,
                        timestamp=df.index[i],
                        bar_index=df['bar_index'].iloc[i],
                        top=df['high'].iloc[ob_idx],
                        bottom=df['low'].iloc[ob_idx],
                        width=width,
                        width_ratio=width_ratio,
                        atr_value=atr
                    )

                    self.active_obs.append(ob)

                    df.loc[df.index[i], 'ob_created'] = True
                    df.loc[df.index[i], 'ob_id'] = ob.id
                    df.loc[df.index[i], 'ob_top'] = ob.top
                    df.loc[df.index[i], 'ob_bottom'] = ob.bottom

        return df

    def generate_signals(self, df: pd.DataFrame) -> pd.DataFrame:
        """生成交易信号"""
        df = df.copy()
        df['position'] = 0.0  # 改为float类型，支持部分仓位
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        df['signal_ob_id'] = None
        df['signal_ema20'] = np.nan

        for i in range(2, len(df)):
            # 1. 检查分层止盈
            for pos in self.active_positions[:]:
                exit_triggered = False
                exit_type = None

                if pos.pos_type == 'long':
                    # 多头检查
                    current_price = df['close'].iloc[i]
                    high_price = df['high'].iloc[i]
                    low_price = df['low'].iloc[i]

                    # 检查止损
                    if low_price <= pos.stop_loss:
                        exit_triggered = True
                        exit_type = 'stop_loss'
                    # 检查TP3（剩余仓位全部平仓）
                    elif not pos.tp3_filled and high_price >= pos.take_profits['tp3']:
                        exit_triggered = True
                        exit_type = 'tp3'
                    # 检查TP2（平1/3）
                    elif not pos.tp2_filled and high_price >= pos.take_profits['tp2']:
                        # 记录TP2触发，但保留仓位
                        pos.tp2_filled = True
                        pos.remaining_position *= 2/3  # 剩余2/3仓位
                    # 检查TP1（平1/3）
                    elif not pos.tp1_filled and high_price >= pos.take_profits['tp1']:
                        # 记录TP1触发，但保留仓位
                        pos.tp1_filled = True
                        pos.remaining_position *= 2/3  # 剩余2/3仓位
                else:
                    # 空头检查
                    current_price = df['close'].iloc[i]
                    high_price = df['high'].iloc[i]
                    low_price = df['low'].iloc[i]

                    # 检查止损
                    if high_price >= pos.stop_loss:
                        exit_triggered = True
                        exit_type = 'stop_loss'
                    # 检查TP3
                    elif not pos.tp3_filled and low_price <= pos.take_profits['tp3']:
                        exit_triggered = True
                        exit_type = 'tp3'
                    # 检查TP2
                    elif not pos.tp2_filled and low_price <= pos.take_profits['tp2']:
                        pos.tp2_filled = True
                        pos.remaining_position *= 2/3
                    # 检查TP1
                    elif not pos.tp1_filled and low_price <= pos.take_profits['tp1']:
                        pos.tp1_filled = True
                        pos.remaining_position *= 2/3

                if exit_triggered:
                    self.active_positions.remove(pos)
                    self.closed_positions.append(pos)

            # 2. 检查入场（简化版）
            if len(self.active_positions) == 0:
                prev_idx = i - 1
                if prev_idx >= 0:
                    is_msb_bull = df['msb_bullish'].iloc[prev_idx]
                    is_msb_bear = df['msb_bearish'].iloc[prev_idx]

                    ob_id = df['ob_id'].iloc[prev_idx] if df['ob_created'].iloc[prev_idx] else None
                    ob = next((o for o in self.active_obs if o.id == ob_id), None)

                    if ob is not None:
                        if is_msb_bull:
                            if df['close'].iloc[i] < df['open'].iloc[i]:
                                self._open_long_position(df, i, ob)
                        elif is_msb_bear:
                            if df['close'].iloc[i] > df['open'].iloc[i]:
                                self._open_short_position(df, i, ob)

            # 3. 更新当前持仓状态（显示剩余仓位）
            total_position = 0
            for pos in self.active_positions:
                if pos.pos_type == 'long':
                    total_position += pos.remaining_position
                    df.loc[df.index[i], 'position'] = total_position
                else:
                    total_position -= pos.remaining_position
                    df.loc[df.index[i], 'position'] = -total_position

        return df

    def _open_long_position(self, df, bar_idx, ob):
        """开多头仓位"""
        entry_price = df['close'].iloc[bar_idx]
        atr = df['atr'].iloc[bar_idx]

        # 止损：OB底部 - 1*ATR
        stop_loss = ob.bottom - ob.atr_value * self.atr_multiplier

        # 止盈：**改用ATR倍数**（符合Pine原文）
        tp1 = entry_price + atr * self.tp1_multiplier
        tp2 = entry_price + atr * self.tp2_multiplier
        tp3 = entry_price + atr * self.tp3_multiplier

        pos = Position(
            id=str(uuid.uuid4()),
            pos_type='long',
            entry_price=entry_price,
            entry_timestamp=df.index[bar_idx],
            entry_bar_index=bar_idx,
            stop_loss=stop_loss,
            take_profits={'tp1': tp1, 'tp2': tp2, 'tp3': tp3},
            ob_id=ob.id,
            remaining_position=1.0  # 满仓
        )
        self.active_positions.append(pos)

    def _open_short_position(self, df, bar_idx, ob):
        """开空头仓位"""
        entry_price = df['close'].iloc[bar_idx]
        atr = df['atr'].iloc[bar_idx]

        # 止损：OB顶部 + 1*ATR
        stop_loss = ob.top + ob.atr_value * self.atr_multiplier

        # 止盈：**改用ATR倍数**
        tp1 = entry_price - atr * self.tp1_multiplier
        tp2 = entry_price - atr * self.tp2_multiplier
        tp3 = entry_price - atr * self.tp3_multiplier

        pos = Position(
            id=str(uuid.uuid4()),
            pos_type='short',
            entry_price=entry_price,
            entry_timestamp=df.index[bar_idx],
            entry_bar_index=bar_idx,
            stop_loss=stop_loss,
            take_profits={'tp1': tp1, 'tp2': tp2, 'tp3': tp3},
            ob_id=ob.id,
            remaining_position=1.0
        )
        self.active_positions.append(pos)

    def run_strategy(self, df: pd.DataFrame) -> pd.DataFrame:
        """运行完整的策略流程"""
        # 重置状态
        self.active_obs = []
        self.active_positions = []
        self.closed_positions = []

        # 1. 计算指标
        df = self.calculate_indicators(df)

        # 2. 检测MSB信号
        df = self.detect_msb_signals(df)

        # 3. 识别订单块
        df = self.identify_order_blocks(df)

        # 4. 生成交易信号
        df = self.generate_signals(df)

        return df


# ========================================================================
# 回测指标计算
# ========================================================================

def _identify_trades_from_positions(closed_positions):
    """从平仓记录中提取交易"""
    trades = []
    for pos in closed_positions:
        # 计算实际持仓时间和平均退出价格
        # 这里简化处理，实际应该记录每个退出点的价格
        entry_price = pos.entry_price
        exit_price = pos.entry_price  # 简化：应该根据TP1/TP2/TP3触发情况计算

        if pos.pos_type == 'long':
            pnl = (exit_price - entry_price) / entry_price
        else:
            pnl = (entry_price - exit_price) / entry_price

        trades.append({
            'entry': pos.entry_bar_index,
            'exit': pos.entry_bar_index,  # 简化
            'pnl': pnl,
            'type': pos.pos_type,
            'tp1_filled': pos.tp1_filled,
            'tp2_filled': pos.tp2_filled,
            'tp3_filled': pos.tp3_filled
        })

    return trades


def calculate_metrics(df, freq='60min'):
    """计算回测指标（简化版）"""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df = df.dropna(subset=['position', 'strategy_returns'])

    if len(df) == 0 or df['strategy_returns'].sum() == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'cumulative_return': 0,
            'max_drawdown': 0,
            'sharpe_ratio': 0
        }

    cumulative_return = (1 + df['strategy_returns']).prod() - 1
    cum_returns = (1 + df['strategy_returns']).cumprod()
    max_drawdown = ((cum_returns.expanding().max() - cum_returns) / cum_returns.expanding().max()).max()

    trading_periods = {'5min': 48*252, '15min': 16*252, '60min': 4*252}
    periods = trading_periods.get(freq, 4*252)
    sharpe = (df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(periods)
              if df['strategy_returns'].std() > 0 else 0)

    return {
        'total_trades': 0,  # 暂时返回0，后续从position记录中统计
        'win_rate': 0,
        'cumulative_return': cumulative_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe
    }


def load_data(year, freq='60min'):
    """加载指定年份的数据"""
    data_file = project_root / "data" / f"RB9999_{year}_{freq}.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    df = pd.read_csv(data_file, parse_dates=['datetime'])
    df = df.sort_values('datetime').reset_index(drop=True)
    return df


def run_backtest_for_year(year, freq='60min'):
    """对单年数据运行回测"""
    print(f"\n{year}年...", end=' ')

    # 加载数据
    df = load_data(year, freq)

    # 创建策略（使用原始参数）
    strategy = HybridATRPartialStrategy(
        pivot_len=7,
        msb_zscore=0.3,
        atr_period=14,
        atr_multiplier=1.0,
        tp1_multiplier=0.5,
        tp2_multiplier=1.0,
        tp3_multiplier=1.5
    )

    # 运行策略
    df_sig = strategy.run_strategy(df)
    metrics = calculate_metrics(df_sig, freq=freq)

    metrics['year'] = year
    metrics['freq'] = freq
    metrics['total_positions'] = len(strategy.closed_positions)

    print(f"仓位{metrics['total_positions']:3}  "
          f"收益{metrics['cumulative_return']:6.2f}%  "
          f"回撤{metrics['max_drawdown']:6.2f}%  "
          f"夏普{metrics['sharpe_ratio']:6.2f}")

    return metrics, strategy


# ========================================================================
# 主程序
# ========================================================================

def main():
    """主函数"""
    print("="*80)
    print("V03改进方案1+2：ATR分层止盈")
    print("  - 改用ATR止盈（符合Pine原文）")
    print("  - 实现分层平仓（TP1/TP2/TP3各1/3）")
    print("="*80)

    years = [2023, 2024, 2025]
    freq = '15min'

    all_metrics = []
    all_strategies = []

    print(f"\n周期: {freq}")
    print("-"*80)

    for year in years:
        try:
            metrics, strategy = run_backtest_for_year(year, freq)
            all_metrics.append(metrics)
            all_strategies.append(strategy)
        except Exception as e:
            print(f"\n{year}年: 失败 - {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    if all_metrics:
        print(f"\n{'='*80}")
        print("汇总")
        print(f"{'='*80}\n")

        total_positions = sum(m['total_positions'] for m in all_metrics)
        total_return = sum(m['cumulative_return'] for m in all_metrics)
        max_dd = max(m['max_drawdown'] for m in all_metrics)
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in all_metrics])

        print(f"总持仓数: {total_positions}")
        print(f"总收益率: {total_return:.2%}")
        print(f"最大回撤: {max_dd:.2%}")
        print(f"平均夏普: {avg_sharpe:.2f}")

        # 保存结果
        output = Path(__file__).parent / "results_atr_partial.csv"
        df_res = pd.DataFrame(all_metrics)
        df_res.to_csv(output, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存: {output}")

        return all_metrics


if __name__ == "__main__":
    main()
