"""
回测引擎模块
实现简单的事件驱动回测框架
"""

import pandas as pd
import numpy as np
from typing import List, Optional
from dataclasses import dataclass, field
from enum import Enum
import yaml
from pathlib import Path
import sys

# 添加项目根目录到路径
sys.path.append(str(Path(__file__).parent.parent))

from indicators.trend import TrendState
from indicators.signal import EntrySignal


class PositionType(Enum):
    """持仓类型"""
    LONG = "long"
    SHORT = "short"


class ExitReason(Enum):
    """出场原因"""
    STOP_LOSS = "stop"
    TAKE_PROFIT_1 = "tp1"
    TAKE_PROFIT_2 = "tp2"
    TAKE_PROFIT_3 = "tp3"
    TREND_REVERSAL = "reversal"
    MANUAL = "manual"


@dataclass
class Trade:
    """交易记录"""
    id: str
    instrument: str
    entry_type: PositionType
    entry_price: float
    entry_timestamp: pd.Timestamp
    entry_bar_index: int
    entry_signal: EntrySignal

    # 止损止盈
    stop_loss: float
    initial_risk: float  # 入场价到止损的距离
    tp1_price: float
    tp2_price: float
    tp3_price: float

    # 出场信息
    exit_price: Optional[float] = None
    exit_timestamp: Optional[pd.Timestamp] = None
    exit_bar_index: Optional[int] = None
    exit_reason: Optional[ExitReason] = None

    # 盈亏信息
    pnl: float = 0.0
    pnl_pct: float = 0.0
    commission: float = 0.0
    slippage: float = 0.0

    # 持仓信息
    position_size: float = 0.0
    duration_bars: int = 0
    max_profit: float = 0.0
    max_drawdown: float = 0.0

    # 止损跟踪
    trailing_stop: Optional[float] = None
    breakeven_triggered: bool = False


@dataclass
class BacktestResult:
    """回测结果"""
    trades: List[Trade] = field(default_factory=list)

    # 交易指标
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float = 0.0
    avg_win: float = 0.0
    avg_loss: float = 0.0
    profit_factor: float = 0.0
    expectancy: float = 0.0

    # 收益指标
    total_pnl: float = 0.0
    total_pnl_pct: float = 0.0
    max_drawdown: float = 0.0
    max_drawdown_pct: float = 0.0
    total_commission: float = 0.0
    total_slippage: float = 0.0

    # 时间指标
    start_date: Optional[pd.Timestamp] = None
    end_date: Optional[pd.Timestamp] = None
    total_bars: int = 0
    avg_days_per_trade: float = 0.0


class BacktestEngine:
    """回测引擎类"""

    def __init__(self, config: dict):
        """
        初始化回测引擎

        Args:
            config: 配置字典
        """
        self.config = config
        self.data_config = config['data']
        self.cost_config = config['costs']
        self.risk_config = config['risk']

        # 账户状态
        self.account_balance = 100000.0  # 初始资金
        self.initial_balance = 100000.0
        self.current_positions: List[Trade] = []
        self.trades: List[Trade] = []
        self.equity_curve: List[float] = []
        self.drawdown_curve: List[float] = []

        # 回测状态
        self.active = False
        self.paused = False
        self.consecutive_losses = 0

    def run(self, df: pd.DataFrame, signals: List[EntrySignal]) -> BacktestResult:
        """
        运行回测

        Args:
            df: K线数据（必须包含所有指标）
            signals: 入场信号列表

        Returns:
            BacktestResult: 回测结果
        """
        print(f"[BacktestEngine] 开始回测...")
        print(f"  初始资金: {self.account_balance:,.2f}")

        self.active = True
        self.trades = []
        self.equity_curve = [self.account_balance]
        self.drawdown_curve = [0.0]

        # 将信号转换为字典以便快速查找
        signal_dict = {s.bar_index: s for s in signals}

        # 按时间遍历K线
        for i in range(len(df)):
            current_time = df.index[i]
            current_bar = df.iloc[i]

            # 1. 检查是否入场
            if i in signal_dict:
                signal = signal_dict[i]
                self._check_entry(signal, df, i)

            # 2. 管理现有持仓
            self._manage_positions(df, i)

            # 3. 更新权益曲线
            self._update_equity(df, i)

            # 4. 检查最大回撤
            self._check_drawdown()

        # 5. 平掉所有剩余持仓
        self._close_all_positions(df, len(df)-1, "手动平仓")

        # 6. 计算回测结果
        result = self._calculate_results()

        print(f"[BacktestEngine] 回测完成")
        print(f"  总交易数: {result.total_trades}")
        print(f"  胜率: {result.win_rate:.1%}")
        print(f"  总收益: {result.total_pnl:,.2f} ({result.total_pnl_pct:.1%})")
        print(f"  最大回撤: {result.max_drawdown_pct:.1%}")

        return result

    def _check_entry(self, signal: EntrySignal, df: pd.DataFrame, index: int):
        """
        检查是否可以入场

        Args:
            signal: 入场信号
            df: K线数据
            index: 当前K线索引
        """
        # 检查是否暂停交易
        if self.paused:
            return

        # 检查是否超过最大持仓数
        if len(self.current_positions) >= self.risk_config['max_positions']:
            return

        # 检查是否超过最大回撤
        current_equity = self.equity_curve[-1]
        drawdown = (self.initial_balance - current_equity) / self.initial_balance
        if drawdown > self.risk_config['max_drawdown_pct']:
            self.paused = True
            print(f"[BacktestEngine] 最大回撤触发，暂停交易 (回撤: {drawdown:.1%})")
            return

        # 检查连续亏损
        if self.consecutive_losses >= self.risk_config['max_consecutive_losses']:
            self.paused = True
            print(f"[BacktestEngine] 连续亏损触发，暂停交易 (亏损次数: {self.consecutive_losses})")
            return

        # 计算仓位大小
        entry_price = signal.entry_price
        stop_loss = self._calculate_stop_loss(signal, df, index)
        initial_risk = abs(entry_price - stop_loss)

        # 计算仓位
        risk_amount = self.account_balance * self.risk_config['risk_per_trade']
        position_size = risk_amount / initial_risk

        if position_size <= 0:
            return

        # 创建交易记录
        trade = Trade(
            id=f"trade_{len(self.trades)}_{index}",
            instrument=self.data_config['instrument'],
            entry_type=PositionType.LONG if signal.trend_direction == 'bullish' else PositionType.SHORT,
            entry_price=entry_price,
            entry_timestamp=signal.timestamp,
            entry_bar_index=index,
            entry_signal=signal,
            stop_loss=stop_loss,
            initial_risk=initial_risk,
            tp1_price=entry_price + initial_risk * self.config['take_profit']['tp1_rr'],
            tp2_price=entry_price + initial_risk * self.config['take_profit']['tp2_rr'],
            tp3_price=entry_price + initial_risk * self.config['take_profit']['tp3_rr'],
            position_size=position_size,
            commission=self.cost_config['commission_per_unit'] * position_size,
            slippage=entry_price * self.cost_config['slippage_pct'] * position_size
        )

        # 如果是看跌，止损止盈方向相反
        if signal.trend_direction == 'bearish':
            trade.stop_loss = entry_price + initial_risk
            trade.tp1_price = entry_price - initial_risk * self.config['take_profit']['tp1_rr']
            trade.tp2_price = entry_price - initial_risk * self.config['take_profit']['tp2_rr']
            trade.tp3_price = entry_price - initial_risk * self.config['take_profit']['tp3_rr']

        # 扣除手续费和滑点
        self.account_balance -= (trade.commission + trade.slippage)

        # 添加到持仓列表
        self.current_positions.append(trade)
        self.trades.append(trade)

        print(f"[BacktestEngine] 入场: {trade.entry_type.value} "
              f"@ {entry_price:.2f} 仓位: {position_size:.0f}")

    def _calculate_stop_loss(self, signal: EntrySignal, df: pd.DataFrame, index: int) -> float:
        """
        计算止损价格

        Args:
            signal: 入场信号
            df: K线数据（包含supertrend列）
            index: 当前K线索引

        Returns:
            float: 止损价格
        """
        # 使用Supertrend作为动态止损
        if 'supertrend_lower' in df.columns and 'supertrend_upper' in df.columns:
            if signal.trend_direction == 'bullish':
                # 看涨：止损在Supertrend下线
                stop_loss = df['supertrend_lower'].iloc[index]
            else:
                # 看跌：止损在Supertrend上线
                stop_loss = df['supertrend_upper'].iloc[index]

            # 确保止损在入场价的另一侧
            if signal.trend_direction == 'bullish' and stop_loss >= signal.entry_price:
                stop_loss = signal.entry_price * 0.99  # 保守止损
            elif signal.trend_direction == 'bearish' and stop_loss <= signal.entry_price:
                stop_loss = signal.entry_price * 1.01  # 保守止损

            return stop_loss

        # 备选：固定ATR止损
        atr = df.get('atr', pd.Series(0, index=df.index)).iloc[index]
        if signal.trend_direction == 'bullish':
            return signal.entry_price - (atr * self.config['supertrend']['atr_sl_multiplier'])
        else:
            return signal.entry_price + (atr * self.config['supertrend']['atr_sl_multiplier'])

    def _manage_positions(self, df: pd.DataFrame, index: int):
        """
        管理现有持仓（检查止损止盈）

        Args:
            df: K线数据
            index: 当前K线索引
        """
        for trade in self.current_positions[:]:  # 复制列表以安全迭代
            current_bar = df.iloc[index]
            current_price = current_bar['close']

            # 更新持仓信息
            trade.duration_bars += 1
            trade.max_profit = max(trade.max_profit,
                                   (current_price - trade.entry_price) * trade.position_size)
            trade.max_drawdown = max(trade.max_drawdown,
                                     (trade.entry_price - current_price) * trade.position_size)

            # 更新Supertrend止损
            if 'supertrend_lower' in df.columns and 'supertrend_upper' in df.columns:
                if trade.entry_type == PositionType.LONG:
                    # 看涨：止损跟随Supertrend下线
                    new_stop = df['supertrend_lower'].iloc[index]
                    if new_stop > trade.stop_loss:  # 只向上移动
                        trade.stop_loss = new_stop
                else:
                    # 看跌：止损跟随Supertrend上线
                    new_stop = df['supertrend_upper'].iloc[index]
                    if new_stop < trade.stop_loss:  # 只向下移动
                        trade.stop_loss = new_stop

            # 盈利保护：2倍风险后移至盈亏平衡点
            profit = (current_price - trade.entry_price) * trade.position_size
            if profit >= trade.initial_risk * self.config['take_profit']['breakeven_rr']:
                if not trade.breakeven_triggered:
                    trade.stop_loss = trade.entry_price
                    trade.breakeven_triggered = True
                    # print(f"[BacktestEngine] 盈利保护触发: {trade.id}")

            # 检查出场条件
            exit_triggered = False
            exit_reason = None

            if trade.entry_type == PositionType.LONG:
                # 看涨持仓
                if current_price <= trade.stop_loss:
                    # 止损触发
                    exit_triggered = True
                    exit_reason = ExitReason.STOP_LOSS
                elif current_price >= trade.tp1_price and not exit_triggered:
                    # 目标1触发
                    exit_triggered = True
                    exit_reason = ExitReason.TAKE_PROFIT_1
                elif current_price >= trade.tp2_price and not exit_triggered:
                    # 目标2触发
                    exit_triggered = True
                    exit_reason = ExitReason.TAKE_PROFIT_2
                elif current_price >= trade.tp3_price and not exit_triggered:
                    # 目标3触发
                    exit_triggered = True
                    exit_reason = ExitReason.TAKE_PROFIT_3
            else:
                # 看跌持仓
                if current_price >= trade.stop_loss:
                    # 止损触发
                    exit_triggered = True
                    exit_reason = ExitReason.STOP_LOSS
                elif current_price <= trade.tp1_price and not exit_triggered:
                    # 目标1触发
                    exit_triggered = True
                    exit_reason = ExitReason.TAKE_PROFIT_1
                elif current_price <= trade.tp2_price and not exit_triggered:
                    # 目标2触发
                    exit_triggered = True
                    exit_reason = ExitReason.TAKE_PROFIT_2
                elif current_price <= trade.tp3_price and not exit_triggered:
                    # 目标3触发
                    exit_triggered = True
                    exit_reason = ExitReason.TAKE_PROFIT_3

            # 如果触发出场
            if exit_triggered:
                self._close_position(trade, df, index, current_price, exit_reason)

    def _close_position(self, trade: Trade, df: pd.DataFrame, index: int,
                       exit_price: float, exit_reason: ExitReason):
        """
        平仓

        Args:
            trade: 交易记录
            df: K线数据
            index: 当前K线索引
            exit_price: 出场价格
            exit_reason: 出场原因
        """
        # 计算盈亏
        if trade.entry_type == PositionType.LONG:
            pnl = (exit_price - trade.entry_price) * trade.position_size
        else:
            pnl = (trade.entry_price - exit_price) * trade.position_size

        # 计算手续费和滑点
        exit_commission = self.cost_config['commission_per_unit'] * trade.position_size
        exit_slippage = exit_price * self.cost_config['slippage_pct'] * trade.position_size

        # 更新交易记录
        trade.exit_price = exit_price
        trade.exit_timestamp = df.index[index]
        trade.exit_bar_index = index
        trade.exit_reason = exit_reason
        trade.pnl = pnl - (trade.commission + trade.slippage + exit_commission + exit_slippage)
        trade.pnl_pct = trade.pnl / (trade.entry_price * trade.position_size) * 100
        trade.commission += exit_commission
        trade.slippage += exit_slippage

        # 更新账户余额
        self.account_balance += (pnl - exit_commission - exit_slippage)

        # 从持仓列表移除
        if trade in self.current_positions:
            self.current_positions.remove(trade)

        # 更新连续亏损计数
        if trade.pnl < 0:
            self.consecutive_losses += 1
        else:
            self.consecutive_losses = 0

        # 恢复交易（如果因连续亏损暂停）
        if self.paused and self.consecutive_losses == 0:
            self.paused = False
            # print(f"[BacktestEngine] 恢复交易")

        print(f"[BacktestEngine] 出场: {trade.entry_type.value} "
              f"@ {exit_price:.2f} {exit_reason.value} "
              f"盈亏: {trade.pnl:,.2f}")

    def _close_all_positions(self, df: pd.DataFrame, index: int, reason: str):
        """
        平掉所有持仓

        Args:
            df: K线数据
            index: 当前K线索引
            reason: 平仓原因
        """
        for trade in self.current_positions[:]:
            exit_price = df['close'].iloc[index]
            self._close_position(trade, df, index, exit_price, ExitReason.MANUAL)

    def _update_equity(self, df: pd.DataFrame, index: int):
        """
        更新权益曲线

        Args:
            df: K线数据
            index: 当前K线索引
        """
        # 计算浮动盈亏
        floating_pnl = 0
        for trade in self.current_positions:
            current_price = df['close'].iloc[index]
            if trade.entry_type == PositionType.LONG:
                floating_pnl += (current_price - trade.entry_price) * trade.position_size
            else:
                floating_pnl += (trade.entry_price - current_price) * trade.position_size

        # 更新权益
        total_equity = self.account_balance + floating_pnl
        self.equity_curve.append(total_equity)

    def _check_drawdown(self):
        """检查最大回撤"""
        current_equity = self.equity_curve[-1]
        peak = max(self.equity_curve)

        drawdown = (peak - current_equity) / peak if peak > 0 else 0
        self.drawdown_curve.append(drawdown)

    def _calculate_results(self) -> BacktestResult:
        """
        计算回测结果

        Returns:
            BacktestResult: 回测结果
        """
        result = BacktestResult()

        # 交易列表
        result.trades = self.trades
        result.total_trades = len(self.trades)

        if result.total_trades == 0:
            return result

        # 计算交易指标
        winning_trades = [t for t in self.trades if t.pnl > 0]
        losing_trades = [t for t in self.trades if t.pnl < 0]

        result.winning_trades = len(winning_trades)
        result.losing_trades = len(losing_trades)
        result.win_rate = result.winning_trades / result.total_trades

        result.avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
        result.avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0

        total_win = sum([t.pnl for t in winning_trades]) if winning_trades else 0
        total_loss = abs(sum([t.pnl for t in losing_trades])) if losing_trades else 0

        result.profit_factor = total_win / total_loss if total_loss > 0 else float('inf')

        result.expectancy = (total_win - total_loss) / result.total_trades

        # 计算收益指标
        result.total_pnl = sum([t.pnl for t in self.trades])
        result.total_pnl_pct = result.total_pnl / self.initial_balance * 100

        result.max_drawdown = max(self.drawdown_curve) * self.initial_balance
        result.max_drawdown_pct = max(self.drawdown_curve) * 100

        result.total_commission = sum([t.commission for t in self.trades])
        result.total_slippage = sum([t.slippage for t in self.trades])

        # 时间指标
        if len(self.equity_curve) > 1:
            result.start_date = pd.Timestamp.min  # 需要从数据获取
            result.end_date = pd.Timestamp.max  # 需要从数据获取
            result.total_bars = len(self.equity_curve)

        return result


if __name__ == "__main__":
    # 测试回测引擎
    print("回测引擎测试")
