#!/usr/bin/env python3
"""
V03: 混合策略
- 入场：简化版（MSB信号后下一根反向K线入场）
- 出场：原Pine源码逻辑（止损：OB边界外1倍ATR，TP1/TP2/TP3：OB宽度的0.5/1.0/1.5倍）

2023-2025年 多时间周期回测
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
    width_ratio: float  # OB宽度与价格的比率
    atr_value: float  # 入场时的ATR值


@dataclass
class Position:
    """持仓数据结构"""
    id: str
    pos_type: str  # 'long' or 'short'
    entry_price: float
    entry_timestamp: pd.Timestamp
    entry_bar_index: int
    stop_loss: float
    take_profits: Dict[str, float]
    ob_id: str


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
    """
    在MSB发生后回溯找订单块K线
    - 看涨MSB: 回溯找阴线 (close < open)
    - 看跌MSB: 回溯找阳线 (close > open)
    """
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
# 混合策略类
# ========================================================================

class HybridMSBOBStrategy:
    """
    混合MSB+OB策略

    入场：简化版（MSB信号后下一根反向K线入场）
    出场：Pine源码逻辑（止损：OB边界外1倍ATR，TP1/TP2/TP3：OB宽度的0.5/1.0/1.5倍）
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

        # EMA指标
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
        """识别订单块（简化版：MSB后直接回溯找OB）"""
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

                    # 计算OB宽度和比率
                    ob_high = df['high'].iloc[ob_idx]
                    ob_low = df['low'].iloc[ob_idx]
                    ob_width = ob_high - ob_low
                    close_price = df['close'].iloc[i]
                    width_ratio = ob_width / close_price if close_price > 0 else 0

                    ob = OrderBlock(
                        id=str(uuid.uuid4()),
                        ob_type=msb_type,
                        timestamp=df.index[i],
                        bar_index=df['bar_index'].iloc[i],
                        top=ob_high,
                        bottom=ob_low,
                        width=ob_width,
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
        df['position'] = 0
        df['entry_price'] = np.nan
        df['stop_loss'] = np.nan
        df['take_profit'] = np.nan
        df['signal_ob_id'] = None  # 记录信号对应的OB ID
        df['signal_ema20'] = np.nan  # 记录信号时的EMA20值

        for i in range(2, len(df)):
            # 1. 检查是否已平仓
            for pos in self.active_positions[:]:
                if not self._check_position_exit(pos, df.iloc[i], i):
                    self.active_positions.remove(pos)
                    self.closed_positions.append(pos)

            # 2. 检查入场（简化版：MSB信号后，下一根反向K线入场）
            if len(self.active_positions) == 0:
                # 检查上一根K线是否有MSB信号
                prev_idx = i - 1
                if prev_idx >= 0:
                    is_msb_bull = df['msb_bullish'].iloc[prev_idx]
                    is_msb_bear = df['msb_bearish'].iloc[prev_idx]

                    # 找到对应的OB
                    ob_id = df['ob_id'].iloc[prev_idx] if df['ob_created'].iloc[prev_idx] else None
                    ob = next((o for o in self.active_obs if o.id == ob_id), None)

                    if ob is not None:
                        if is_msb_bull:
                            # 看涨：找阴线入场
                            if df['close'].iloc[i] < df['open'].iloc[i]:
                                self._open_long_position(df, i, ob)
                        elif is_msb_bear:
                            # 看跌：找阳线入场
                            if df['close'].iloc[i] > df['open'].iloc[i]:
                                self._open_short_position(df, i, ob)

            # 3. 更新当前持仓状态
            for pos in self.active_positions:
                # 查找对应的OB
                ob = next((o for o in self.active_obs if o.id == pos.ob_id), None)
                ema20 = df['ema20'].iloc[i] if 'ema20' in df.columns else np.nan

                if pos.pos_type == 'long':
                    df.loc[df.index[i], 'position'] = 1
                    df.loc[df.index[i], 'entry_price'] = pos.entry_price
                    df.loc[df.index[i], 'stop_loss'] = pos.stop_loss
                    df.loc[df.index[i], 'take_profit'] = pos.take_profits['tp1']
                    df.loc[df.index[i], 'signal_ob_id'] = pos.ob_id
                    df.loc[df.index[i], 'signal_ema20'] = ema20
                else:
                    df.loc[df.index[i], 'position'] = -1
                    df.loc[df.index[i], 'entry_price'] = pos.entry_price
                    df.loc[df.index[i], 'stop_loss'] = pos.stop_loss
                    df.loc[df.index[i], 'take_profit'] = pos.take_profits['tp1']
                    df.loc[df.index[i], 'signal_ob_id'] = pos.ob_id
                    df.loc[df.index[i], 'signal_ema20'] = ema20

        return df

    def _open_long_position(self, df, bar_idx, ob):
        """开多头仓位"""
        entry_price = df['close'].iloc[bar_idx]

        # 止损：OB底部 - 1*ATR
        stop_loss = ob.bottom - ob.atr_value * self.atr_multiplier

        # 止盈：OB宽度的0.5/1.0/1.5倍
        tp1 = entry_price + ob.width * self.tp1_multiplier
        tp2 = entry_price + ob.width * self.tp2_multiplier
        tp3 = entry_price + ob.width * self.tp3_multiplier

        pos = Position(
            id=str(uuid.uuid4()),
            pos_type='long',
            entry_price=entry_price,
            entry_timestamp=df.index[bar_idx],
            entry_bar_index=bar_idx,
            stop_loss=stop_loss,
            take_profits={'tp1': tp1, 'tp2': tp2, 'tp3': tp3},
            ob_id=ob.id
        )
        self.active_positions.append(pos)

    def _open_short_position(self, df, bar_idx, ob):
        """开空头仓位"""
        entry_price = df['close'].iloc[bar_idx]

        # 止损：OB顶部 + 1*ATR
        stop_loss = ob.top + ob.atr_value * self.atr_multiplier

        # 止盈：OB宽度的0.5/1.0/1.5倍
        tp1 = entry_price - ob.width * self.tp1_multiplier
        tp2 = entry_price - ob.width * self.tp2_multiplier
        tp3 = entry_price - ob.width * self.tp3_multiplier

        pos = Position(
            id=str(uuid.uuid4()),
            pos_type='short',
            entry_price=entry_price,
            entry_timestamp=df.index[bar_idx],
            entry_bar_index=bar_idx,
            stop_loss=stop_loss,
            take_profits={'tp1': tp1, 'tp2': tp2, 'tp3': tp3},
            ob_id=ob.id
        )
        self.active_positions.append(pos)

    def _check_position_exit(self, pos, bar, bar_idx):
        """检查持仓是否需要平仓"""
        if pos.pos_type == 'long':
            # 多头：检查止损或TP1
            if bar['low'] <= pos.stop_loss:
                return False  # 触发止损
            elif bar['high'] >= pos.take_profits['tp1']:
                return False  # 触发TP1
        else:
            # 空头：检查止损或TP1
            if bar['high'] >= pos.stop_loss:
                return False  # 触发止损
            elif bar['low'] <= pos.take_profits['tp1']:
                return False  # 触发TP1
        return True  # 继续持有

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

def _identify_trades(df):
    """识别每笔完整交易"""
    trades = []
    in_position = False
    entry_idx = None
    entry_price = None
    pos_type = None

    for i in range(len(df)):
        if not in_position and df['position'].iloc[i] != 0:
            # 开仓
            in_position = True
            entry_idx = i
            entry_price = df['close'].iloc[i]
            pos_type = 'long' if df['position'].iloc[i] > 0 else 'short'
        elif in_position and df['position'].iloc[i] == 0:
            # 平仓
            exit_price = df['close'].iloc[i]
            if pos_type == 'long':
                pnl = (exit_price - entry_price) / entry_price
            else:
                pnl = (entry_price - exit_price) / entry_price
            trades.append({'entry': entry_idx, 'exit': i, 'pnl': pnl, 'type': pos_type})
            in_position = False

    return trades


def calculate_metrics(df, freq='60min'):
    """计算回测指标"""
    df = df.copy()
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df = df.dropna(subset=['position', 'strategy_returns'])

    if len(df) == 0:
        return {
            'total_trades': 0,
            'win_rate': 0,
            'avg_win': 0,
            'avg_loss': 0,
            'profit_loss_ratio': 0,
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

    # 修正：基于完整交易计算胜率
    trades = _identify_trades(df)
    winning_trades = sum(1 for t in trades if t['pnl'] > 0)
    losing_trades = sum(1 for t in trades if t['pnl'] < 0)
    total_trades = len(trades)
    win_rate = (winning_trades / total_trades * 100) if total_trades > 0 else 0

    # 计算平均盈利和亏损（基于完整交易）
    winning_pnls = [t['pnl'] for t in trades if t['pnl'] > 0]
    losing_pnls = [t['pnl'] for t in trades if t['pnl'] < 0]
    avg_win = np.mean(winning_pnls) if winning_pnls else 0
    avg_loss = np.mean(losing_pnls) if losing_pnls else 0

    # 盈亏比
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
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

    # 创建策略
    strategy = HybridMSBOBStrategy(
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

    msb_count = df_sig['msb_bullish'].sum() + df_sig['msb_bearish'].sum()
    ob_count = df_sig['ob_created'].sum()

    print(f"MSB{int(msb_count):3}  OB{int(ob_count):3}  交易{metrics['total_trades']:3}  "
          f"胜率{metrics['win_rate']:5.2f}%  "
          f"收益{metrics['cumulative_return']:6.2f}%  "
          f"回撤{metrics['max_drawdown']:6.2f}%")

    return metrics


# ========================================================================
# 主程序
# ========================================================================

def main():
    """主函数"""
    print("="*80)
    print("V03: 混合策略 - MSB+OB")
    print("  入场：简化版（MSB信号后下一根反向K线入场）")
    print("  出场：Pine源码逻辑（止损：OB边界外1倍ATR，TP1/TP2/TP3：OB宽度的0.5/1.0/1.5倍）")
    print("="*80)

    years = [2023, 2024, 2025]
    freqs = ['60min', '15min', '5min']

    all_results = []

    for freq in freqs:
        print(f"\n{'='*80}")
        print(f"时间周期: {freq}")
        print(f"{'='*80}")

        for year in years:
            try:
                result = run_backtest_for_year(year, freq)
                all_results.append(result)
            except Exception as e:
                print(f"\n{year}年: 失败 - {e}")
                import traceback
                traceback.print_exc()

    # 汇总
    if all_results:
        print(f"\n{'='*80}")
        print("汇总对比")
        print(f"{'='*80}\n")

        print(f"{'周期':>6}  {'交易':>6}  {'胜率':>8}  {'总收益':>8}  {'回撤':>8}  {'夏普':>8}")
        print("-"*60)

        for freq in freqs:
            freq_res = [r for r in all_results if r['freq'] == freq]
            if not freq_res:
                continue

            total_trades = sum(r['total_trades'] for r in freq_res)
            avg_win = np.mean([r['win_rate'] for r in freq_res])
            total_ret = sum(r['cumulative_return'] for r in freq_res)
            max_dd = max(r['max_drawdown'] for r in freq_res)
            avg_sharpe = np.mean([r['sharpe_ratio'] for r in freq_res])

            print(f"{freq:>6}  {total_trades:6}  {avg_win:7.2f}%  {total_ret:8.2f}%  {max_dd:8.2f}%  {avg_sharpe:8.2f}")

        # 保存
        df_res = pd.DataFrame(all_results)
        output = Path(__file__).parent / "report_v03_hybrid.csv"
        output.parent.mkdir(parents=True, exist_ok=True)
        df_res.to_csv(output, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存: {output}")

        return all_results


if __name__ == "__main__":
    main()
