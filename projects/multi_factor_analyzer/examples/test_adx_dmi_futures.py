"""
ADX/DMI Factor Backtest — Futures-Grade Engine
===============================================

Properly designed for SINGLE-INSTRUMENT FUTURES trading:
  - Hourly bars resampled from 1-min data
  - T+0 execution (signal at bar T close → execute at bar T+1 open)
  - Transaction costs (commission + slippage per trade)
  - Position sizing based on factor signal MAGNITUDE (not binary)
  - Stop-loss / exit simulation
  - Margin-adjusted return calculation
  - Futures-specific metrics: per-trade stats, drawdown, Sharpe, Calmar

Does NOT use the cross-sectional equity framework (StrategyAnalyzer, ReliabilityEvaluator)
which was designed for sorting 3000+ stocks — fundamentally wrong for single futures.
"""

import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════
# ADX / DMI Core Indicator
# ═══════════════════════════════════════════════════════════════════════

def calculate_adx_dmi(df: pd.DataFrame, n: int = 14, m: int = 6):
    """Calculate ADX/DMI components. Returns: (pdi, mdi, adx, adxr) as pd.Series."""
    high = df['high']
    low = df['low']
    close = df['close']

    tr1 = high - low
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    hd = high - high.shift(1)
    ld = low.shift(1) - low

    dmp = hd.where((hd > 0) & (hd >= ld), 0.0)
    dmm = ld.where((ld > 0) & (ld > hd), 0.0)

    tr_sum = tr.rolling(n, min_periods=n).sum()
    dmp_sum = dmp.rolling(n, min_periods=n).sum()
    dmm_sum = dmm.rolling(n, min_periods=n).sum()

    pdi = (dmp_sum / tr_sum) * 100
    mdi = (dmm_sum / tr_sum) * 100

    dx_sum = pdi + mdi
    dx = pd.Series(0.0, index=df.index)
    nonzero = dx_sum > 0
    dx[nonzero] = ((mdi[nonzero] - pdi[nonzero]).abs() / dx_sum[nonzero]) * 100

    adx = dx.rolling(m, min_periods=m).mean()
    adxr = (adx + adx.shift(m)) / 2

    return pdi, mdi, adx, adxr


def _bars_since(condition: pd.Series) -> pd.Series:
    """Count bars since condition was last True. 0 on the signal bar itself."""
    groups = condition.cumsum()
    result = groups.groupby(groups).cumcount()
    result[condition] = 0
    return result


# ═══════════════════════════════════════════════════════════════════════
# ADX/DMI Factor — 8-Rule System (unchanged logic)
# ═══════════════════════════════════════════════════════════════════════

def calculate_adx_factor(df: pd.DataFrame, n: int = 14, m: int = 6) -> pd.Series:
    """
    ADX/DMI factor — strictly following the 8-rule trading system.

    Signal encoding (continuous, range ~ [-1, +1]):
      +1.0  Trend launch: ADX crosses 20 up + DI+ on top
      +0.8  Standard long: golden cross within 5 bars + ADX > 20
      +0.5  Trend continuation: DI+ > DI- + ADX > 20
      +0.3  Trend weakening: ADX declining or DI converging → reduce
       0.0  No trade: ADX < 20 (choppy market)
      -0.7  Short reversal: death cross + ADX > 20 + ADX declining
             (BLOCKED when ADX > 25 and DI+ > DI-)
    """
    pdi, mdi, adx, adxr = calculate_adx_dmi(df, n, m)

    golden_cross = (pdi > mdi) & (pdi.shift(1) <= mdi.shift(1))
    death_cross = (mdi > pdi) & (mdi.shift(1) <= pdi.shift(1))
    adx_cross_up_20 = (adx > 20) & (adx.shift(1) <= 20)
    adx_declining = adx < adx.shift(1)

    bars_since_golden = _bars_since(golden_cross)
    bars_since_death = _bars_since(death_cross)

    di_spread = (pdi - mdi).abs()
    di_converging = di_spread < di_spread.shift(1)

    factor = pd.Series(0.0, index=df.index)

    # RULE 七: ADX < 20 → no trade (factor stays 0)

    # RULE 五: Strongest — trend launch
    strongest = adx_cross_up_20 & (pdi > mdi)
    factor[strongest] = 1.0

    # RULE 二: Standard long
    long_signal = (bars_since_golden <= 5) & (adx > 20)
    factor[long_signal] = np.maximum(factor[long_signal], 0.8)

    # RULE 七-3: Trend continuation
    trend_bull = (pdi > mdi) & (adx > 20) & (bars_since_golden > 5)
    factor[trend_bull] = np.maximum(factor[trend_bull], 0.5)

    # RULE 三: No-short zone
    no_short_zone = (adx > 25) & (pdi > mdi)

    # RULE 四: Short reversal
    short_signal = (
        (bars_since_death <= 5) &
        (adx > 20) &
        adx_declining &
        ~no_short_zone
    )
    factor[short_signal] = -0.7

    # RULE 六: Exit / risk control — attenuate
    exit_zone = adx_declining & (factor > 0) & (factor < 1.0)
    factor[exit_zone] *= 0.3

    converge_exit = di_converging & (factor > 0) & (factor < 1.0)
    factor[converge_exit] = np.minimum(factor[converge_exit], 0.3)

    # FINAL: ADX < 20 → zero
    factor[adx < 20] = 0.0

    return factor


# ═══════════════════════════════════════════════════════════════════════
# Data Loading — Hourly Resampling
# ═══════════════════════════════════════════════════════════════════════

def load_hourly_data(parquet_path: str, start_date: str = '2023-01-01'):
    """
    Load 1-min parquet, resample to HOURLY bars.
    Filters to date range after resampling.
    """
    logger.info(f"Loading parquet: {parquet_path}")
    df = pd.read_parquet(parquet_path)
    df.index = pd.to_datetime(df.index)
    logger.info(f"  Raw rows: {len(df):,}, range: {df.index[0]} ~ {df.index[-1]}")

    # Resample to hourly bars
    hourly = df.resample('1h').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna()

    # Remove overnight gaps (hours with no trading: 15:01-20:59, weekends, holidays)
    # Rebar futures night session: 21:00-23:00, day session: 09:00-11:00, 13:30-15:00
    # After resampling we have bars at each hour, but many will have zero volume
    # Keep only bars that actually have trading activity
    hourly = hourly[hourly['volume'] > 0]

    logger.info(f"  Hourly bars (with volume): {len(hourly):,}, "
                f"range: {hourly.index[0]} ~ {hourly.index[-1]}")

    # Filter to date range
    hourly = hourly[start_date:]
    logger.info(f"  Filtered to {start_date}: {len(hourly):,} bars, "
                f"{hourly.index[0]} ~ {hourly.index[-1]}")
    logger.info(f"  Price range: {hourly['close'].min():.0f} ~ {hourly['close'].max():.0f}")

    return hourly


# ═══════════════════════════════════════════════════════════════════════
# Futures-Grade Backtest Engine
# ═══════════════════════════════════════════════════════════════════════

class FuturesBacktest:
    """
    Single-instrument futures backtest engine.

    Execution model:
      - Signal generated at bar T close (factor value known)
      - Execute at bar T+1 open price
      - Position magnitude = |factor| (0 to 1, representing fraction of full position)
      - Position direction = sign(factor)
      - Transaction cost deducted on each trade (entry + exit)
      - Optional: signal confirmation (consecutive bars required), stop-loss

    This is fundamentally different from the equity cross-sectional framework
    which sorts stocks by factor and selects top/bottom quintiles.
    """

    def __init__(
        self,
        commission_rate: float = 0.0001,   # 1 bp per side (螺纹钢约万分之一)
        slippage_ticks: int = 1,            # 1 tick slippage per trade
        tick_size: float = 1.0,             # 螺纹钢最小变动价位 1元/吨
        margin_rate: float = 0.10,          # 10%保证金
        confirm_bars: int = 0,              # signal confirmation: require N consecutive bars
        stop_loss_pct: float = 0.0,         # stop-loss: exit if unrealized loss > X% of price (0=disabled)
        trailing_stop_pct: float = 0.0,     # trailing stop from peak (0=disabled)
    ):
        self.commission_rate = commission_rate
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size
        self.margin_rate = margin_rate
        self.confirm_bars = confirm_bars
        self.stop_loss_pct = stop_loss_pct
        self.trailing_stop_pct = trailing_stop_pct

    def _smooth_signal(self, factor_vals: np.ndarray) -> np.ndarray:
        """Apply signal confirmation: only change position after N consecutive bars agree."""
        if self.confirm_bars <= 1:
            return factor_vals

        smoothed = np.zeros_like(factor_vals)
        n = len(factor_vals)
        # Track how many consecutive bars agree with current signal
        current_signal = 0.0
        consec_count = 0

        for i in range(n):
            target = factor_vals[i]

            if abs(target - current_signal) < 0.01:
                # Same signal direction — increment confirmation
                consec_count += 1
                if consec_count >= self.confirm_bars:
                    smoothed[i] = current_signal
                else:
                    # Still in confirmation phase — keep previous smoothed value
                    smoothed[i] = smoothed[i - 1] if i > 0 else 0.0
            else:
                # Signal changed — reset confirmation counter
                current_signal = target
                consec_count = 1
                smoothed[i] = smoothed[i - 1] if i > 0 else 0.0

        return smoothed

    def run(self, df: pd.DataFrame, factor: pd.Series) -> dict:
        """
        Run backtest.

        Args:
            df: OHLCV DataFrame with DatetimeIndex
            factor: Signal Series aligned to df index.
                    -1.0 to +1.0, 0 = no position.

        Returns:
            dict with trades, equity curve, metrics
        """
        # Align factor and data
        common = df.index.intersection(factor.index)
        df = df.loc[common].copy()
        factor = factor.loc[common].copy()

        # Drop NaN rows (warmup period)
        valid = factor.notna() & df['open'].notna() & df['close'].notna()
        df = df[valid]
        factor = factor[valid]

        if len(df) < 10:
            return {'error': 'Insufficient data'}

        # Signal at bar T → execute at bar T+1 open
        factor_shifted = factor.shift(1).values
        smoothed = self._smooth_signal(factor_shifted)

        opens = df['open'].values
        closes = df['close'].values
        timestamps = df.index
        n = len(df)

        # State
        position = 0.0       # current signed position (-1 to +1)
        entry_price = 0.0    # fill price at position entry
        peak_price = 0.0     # highest favorable price since entry (for trailing stop)
        trades = []
        equity = np.ones(n)
        active_entry_idx = -1
        stop_exits = 0       # count exits via stop-loss

        slippage = self.slippage_ticks * self.tick_size

        for i in range(1, n):
            target_pos = smoothed[i]

            # ---- Stop-loss check (only when holding) ----
            stopped = False
            if abs(position) > 0.01 and entry_price > 0:
                # Fixed stop-loss
                if self.stop_loss_pct > 0:
                    if position > 0 and closes[i] < entry_price * (1 - self.stop_loss_pct):
                        stopped = True
                    elif position < 0 and closes[i] > entry_price * (1 + self.stop_loss_pct):
                        stopped = True

                # Trailing stop
                if not stopped and self.trailing_stop_pct > 0:
                    if position > 0:
                        peak_price = max(peak_price, closes[i])
                        if closes[i] < peak_price * (1 - self.trailing_stop_pct):
                            stopped = True
                    elif position < 0:
                        peak_price = min(peak_price, closes[i])
                        if closes[i] > peak_price * (1 + self.trailing_stop_pct):
                            stopped = True

            if stopped:
                # Exit at next bar open (realistic: stop triggers, filled at open)
                exit_price = opens[i] - slippage if position > 0 else opens[i] + slippage
                gross_pnl = np.sign(position) * (exit_price - entry_price)
                commission = (
                    self.commission_rate * (entry_price + exit_price) * abs(position)
                )
                net_pnl = gross_pnl - commission
                pnl_pct = net_pnl / entry_price
                equity[i] = equity[i - 1] * (1 + pnl_pct * abs(position) / self.margin_rate)

                holding = i - active_entry_idx
                trades.append({
                    'entry_time': timestamps[active_entry_idx],
                    'exit_time': timestamps[i],
                    'direction': 'long' if position > 0 else 'short',
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': net_pnl * abs(position),
                    'pnl_pct': pnl_pct * 100,
                    'holding_bars': max(holding, 1),
                    'position_size': abs(position),
                    'exit_reason': 'stop_loss',
                })
                stop_exits += 1
                position = 0.0
                entry_price = 0.0
                peak_price = 0.0
                active_entry_idx = -1

                # If target_pos is non-zero, re-enter after stop (at same bar open)
                if abs(target_pos) > 0.01:
                    entry_price = opens[i] + slippage if target_pos > 0 else opens[i] - slippage
                    position = target_pos
                    peak_price = entry_price
                    active_entry_idx = i
                continue

            # ---- Normal position transition ----
            if abs(target_pos - position) > 0.01:
                # Close existing position
                if abs(position) > 0.01:
                    exit_price = opens[i] - slippage if position > 0 else opens[i] + slippage
                    gross_pnl = np.sign(position) * (exit_price - entry_price)
                    commission = (
                        self.commission_rate * (entry_price + exit_price) * abs(position)
                    )
                    net_pnl = gross_pnl - commission
                    pnl_pct = net_pnl / entry_price
                    equity[i] = equity[i - 1] * (1 + pnl_pct * abs(position) / self.margin_rate)

                    holding = i - active_entry_idx
                    trades.append({
                        'entry_time': timestamps[active_entry_idx],
                        'exit_time': timestamps[i],
                        'direction': 'long' if position > 0 else 'short',
                        'entry_price': entry_price,
                        'exit_price': exit_price,
                        'pnl': net_pnl * abs(position),
                        'pnl_pct': pnl_pct * 100,
                        'holding_bars': max(holding, 1),
                        'position_size': abs(position),
                        'exit_reason': 'signal',
                    })

                    if abs(target_pos) <= 0.01:
                        position = 0.0
                        entry_price = 0.0
                        peak_price = 0.0
                        active_entry_idx = -1
                        continue

                # Open new position
                if abs(target_pos) > 0.01:
                    entry_price = opens[i] + slippage if target_pos > 0 else opens[i] - slippage
                    position = target_pos
                    peak_price = entry_price
                    active_entry_idx = i
                    equity[i] = equity[i - 1]

            elif abs(position) > 0.01:
                # Holding — mark to market at close
                unrealized_pct = np.sign(position) * (closes[i] - entry_price) / entry_price
                cost_pct = self.commission_rate * (entry_price + closes[i]) * abs(position) / entry_price
                net_pct = unrealized_pct * abs(position) - cost_pct
                equity[i] = equity[i - 1] * (1 + net_pct / self.margin_rate)
            else:
                equity[i] = equity[i - 1]

        equity_series = pd.Series(equity, index=timestamps)
        return self._compute_metrics(
            trades, equity_series, timestamps, smoothed, stop_exits
        )

    def _compute_metrics(self, trades, equity, timestamps, factor_vals, stop_exits=0):
        """Compute comprehensive futures metrics."""
        result = {}
        result['stop_exits'] = stop_exits

        # --- Equity curve metrics ---
        if len(equity) > 1:
            returns = equity.pct_change().dropna()
            result['total_return'] = (equity.iloc[-1] / equity.iloc[0]) - 1
            # Annualize: count actual trading days from timestamps
            unique_days = timestamps.to_series().dt.date.nunique()
            years = unique_days / 365
            if years > 0:
                result['annualized_return'] = (1 + result['total_return']) ** (1 / years) - 1
            else:
                result['annualized_return'] = 0.0

            result['sharpe'] = (
                returns.mean() / returns.std() * np.sqrt(252 * 7)
                if returns.std() > 0 else 0.0
            )

            # Max drawdown
            cum = (1 + returns).cumprod()
            running_max = cum.cummax()
            drawdown = (cum - running_max) / running_max
            result['max_drawdown'] = drawdown.min()
            result['max_drawdown_duration'] = (
                drawdown[drawdown == 0].index.to_series().diff().max().total_seconds() / 3600
                if len(drawdown[drawdown == 0]) > 1 else 0
            )

            # Calmar ratio
            if result['max_drawdown'] != 0:
                result['calmar'] = result['annualized_return'] / abs(result['max_drawdown'])
            else:
                result['calmar'] = 0.0

        # --- Trade statistics ---
        if trades:
            df_trades = pd.DataFrame(trades)
            result['total_trades'] = len(df_trades)
            result['long_trades'] = len(df_trades[df_trades['direction'] == 'long'])
            result['short_trades'] = len(df_trades[df_trades['direction'] == 'short'])
            result['win_rate'] = (df_trades['pnl'] > 0).mean()
            result['avg_pnl_pct'] = df_trades['pnl_pct'].mean()
            result['avg_win_pct'] = df_trades[df_trades['pnl'] > 0]['pnl_pct'].mean() if (df_trades['pnl'] > 0).any() else 0
            result['avg_loss_pct'] = df_trades[df_trades['pnl'] <= 0]['pnl_pct'].mean() if (df_trades['pnl'] <= 0).any() else 0
            result['profit_factor'] = (
                df_trades[df_trades['pnl'] > 0]['pnl'].sum() /
                abs(df_trades[df_trades['pnl'] <= 0]['pnl'].sum())
                if (df_trades['pnl'] <= 0).sum() > 0 else float('inf')
            )
            result['avg_holding_bars'] = df_trades['holding_bars'].mean()
            result['max_consecutive_losses'] = self._max_consecutive(
                df_trades['pnl'] <= 0
            )
            result['trades'] = df_trades
        else:
            result['total_trades'] = 0
            result['win_rate'] = 0.0
            for k in ['long_trades', 'short_trades', 'avg_pnl_pct',
                       'avg_win_pct', 'avg_loss_pct', 'profit_factor',
                       'avg_holding_bars', 'max_consecutive_losses']:
                result[k] = 0.0
            result['trades'] = pd.DataFrame()

        # --- Factor signal statistics ---
        nonzero = np.abs(factor_vals) > 0.01
        result['signal_coverage'] = nonzero.sum() / len(factor_vals) if len(factor_vals) > 0 else 0
        result['avg_signal_strength'] = np.abs(factor_vals[nonzero]).mean() if nonzero.any() else 0

        # Long vs short coverage
        result['long_coverage'] = (factor_vals > 0.01).sum() / len(factor_vals) if len(factor_vals) > 0 else 0
        result['short_coverage'] = (factor_vals < -0.01).sum() / len(factor_vals) if len(factor_vals) > 0 else 0

        return result

    @staticmethod
    def _max_consecutive(series: pd.Series) -> int:
        """Max consecutive True values in series."""
        if not series.any():
            return 0
        groups = (series != series.shift()).cumsum()
        return series.groupby(groups).sum().max()


# ═══════════════════════════════════════════════════════════════════════
# Report Output
# ═══════════════════════════════════════════════════════════════════════

def print_metrics(name: str, m: dict):
    """Print backtest results in a readable format."""
    print(f"\n{'═'*70}")
    print(f"  {name}")
    print(f"{'═'*70}")

    print(f"\n  📊 收益指标")
    print(f"    总收益:        {m.get('total_return', 0):>10.2%}")
    print(f"    年化收益:      {m.get('annualized_return', 0):>10.2%}")
    print(f"    夏普比率:      {m.get('sharpe', 0):>10.2f}")
    print(f"    最大回撤:      {m.get('max_drawdown', 0):>10.2%}")
    print(f"    Calmar比率:    {m.get('calmar', 0):>10.2f}")

    print(f"\n  📈 交易统计")
    print(f"    总交易次数:    {m.get('total_trades', 0):>10d}")
    print(f"    做多次数:      {m.get('long_trades', 0):>10d}")
    print(f"    做空次数:      {m.get('short_trades', 0):>10d}")
    print(f"    胜率:          {m.get('win_rate', 0):>10.2%}")
    print(f"    盈亏比:        {m.get('profit_factor', 0):>10.2f}")
    print(f"    平均盈利(笔):  {m.get('avg_win_pct', 0):>10.3f}%")
    print(f"    平均亏损(笔):  {m.get('avg_loss_pct', 0):>10.3f}%")
    print(f"    平均持仓(根):  {m.get('avg_holding_bars', 0):>10.1f}")
    print(f"    最大连亏次数:  {m.get('max_consecutive_losses', 0):>10d}")

    if m.get('stop_exits', 0) > 0:
        print(f"\n  🛑 止损统计")
        print(f"    止损触发次数:  {m.get('stop_exits', 0):>10d}")
        print(f"    止损占比:      {m.get('stop_exits', 0) / max(m.get('total_trades', 1), 1):>10.2%}")

    print(f"\n  📡 信号统计")
    print(f"    信号覆盖率:    {m.get('signal_coverage', 0):>10.2%}")
    print(f"    做多占比:      {m.get('long_coverage', 0):>10.2%}")
    print(f"    做空占比:      {m.get('short_coverage', 0):>10.2%}")
    print(f"    平均信号强度:  {m.get('avg_signal_strength', 0):>10.3f}")

    print()


# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

VARIANTS = [
    {'name': 'ADX_N7_Fast',   'n': 7,  'm': 3},
    {'name': 'ADX_N14_Std',   'n': 14, 'm': 6},
    {'name': 'ADX_N21_Slow',  'n': 21, 'm': 9},
]


def main():
    print("\n" + "=" * 70)
    print("  ADX/DMI 因子回测 — 螺纹钢期货 (小时线, 期货规则)")
    print("  多场景对比: 原始信号 / 信号确认 / 加止损")
    print("=" * 70)

    # 1. Load hourly data
    parquet_path = os.path.join(
        os.path.dirname(__file__), '..', 'data', 'RB9999.XSGE.parquet'
    )
    hourly = load_hourly_data(parquet_path, start_date='2023-01-01')

    # 2. Calculate all factor variants
    factors = {}
    for v in VARIANTS:
        name, n, m = v['name'], v['n'], v['m']
        factors[name] = calculate_adx_factor(hourly, n=n, m=m)
        logger.info(f"  {name}: nonzero signals = {(factors[name].abs() > 0.01).sum()}")

    # 3. Define backtest scenarios
    scenarios = [
        {
            'label': 'A-原始信号',
            'params': dict(
                commission_rate=0.0001,
                slippage_ticks=1,
                tick_size=1.0,
                margin_rate=0.10,
            ),
        },
        {
            'label': 'B-3根确认',
            'params': dict(
                commission_rate=0.0001,
                slippage_ticks=1,
                tick_size=1.0,
                margin_rate=0.10,
                confirm_bars=3,
            ),
        },
        {
            'label': 'C-3根确认+2%止损',
            'params': dict(
                commission_rate=0.0001,
                slippage_ticks=1,
                tick_size=1.0,
                margin_rate=0.10,
                confirm_bars=3,
                stop_loss_pct=0.02,
            ),
        },
        {
            'label': 'D-5根确认+3%止损+移动止损',
            'params': dict(
                commission_rate=0.0001,
                slippage_ticks=1,
                tick_size=1.0,
                margin_rate=0.10,
                confirm_bars=5,
                stop_loss_pct=0.03,
                trailing_stop_pct=0.015,
            ),
        },
    ]

    # 4. Run all combinations
    all_results = {}
    for scenario in scenarios:
        slabel = scenario['label']
        print(f"\n{'█'*70}")
        print(f"  场景: {slabel}")
        print(f"  参数: {scenario['params']}")
        print(f"{'█'*70}")

        bt = FuturesBacktest(**scenario['params'])

        for v in VARIANTS:
            name = v['name']
            combo_key = f"{slabel}_{name}"

            print(f"\n{'─'*70}")
            print(f"  {combo_key}")
            print(f"{'─'*70}")

            result = bt.run(hourly, factors[name])

            if 'error' in result:
                logger.error(f"  {combo_key}: {result['error']}")
                continue

            all_results[combo_key] = result
            print_metrics(combo_key, result)

    # 5. Comparison table
    if not all_results:
        logger.error("No variants completed")
        return

    print(f"\n{'═'*70}")
    print("  全场景对比汇总")
    print(f"{'═'*70}")
    print(f"\n{'指标':<16}", end='')
    for key in all_results:
        print(f" {key:>18}", end='')
    print()
    print("─" * 16 + "─" * 18 * len(all_results))

    rows = [
        ('年化收益', 'annualized_return', '{:.2%}'),
        ('夏普', 'sharpe', '{:.2f}'),
        ('最大回撤', 'max_drawdown', '{:.2%}'),
        ('Calmar', 'calmar', '{:.2f}'),
        ('总交易', 'total_trades', '{:d}'),
        ('胜率', 'win_rate', '{:.2%}'),
        ('盈亏比', 'profit_factor', '{:.2f}'),
        ('平均持仓', 'avg_holding_bars', '{:.1f}'),
        ('信号覆盖', 'signal_coverage', '{:.2%}'),
    ]

    for label, metric_key, fmt in rows:
        print(f"{label:<16}", end='')
        for key, r in all_results.items():
            val = r.get(metric_key, 0)
            if val == float('inf'):
                print(f" {'inf':>18}", end='')
            else:
                print(f" {fmt.format(val):>18}", end='')
        print()

    print(f"\n  数据: 螺纹钢主力连续 RB9999.XSGE, 小时线")
    print(f"  区间: {hourly.index[0]} ~ {hourly.index[-1]}")
    print(f"  K线数: {len(hourly):,}")
    print(f"{'═'*70}\n")

    # 6. Save trade logs
    output_dir = os.path.join(
        os.path.dirname(__file__), '..', 'output', 'adx_futures_hourly'
    )
    os.makedirs(output_dir, exist_ok=True)

    for key, r in all_results.items():
        if not r['trades'].empty:
            filepath = os.path.join(output_dir, f'{key}_trades.csv')
            r['trades'].to_csv(filepath, index=False)
            logger.info(f"  Saved: {filepath} ({len(r['trades'])} trades)")


if __name__ == '__main__':
    main()
