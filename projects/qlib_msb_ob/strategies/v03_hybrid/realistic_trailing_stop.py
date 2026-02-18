#!/usr/bin/env python3
"""
真实交易回测系统 - 移动止盈版本
包含：
1. 仓位控制（固定金额分配）
2. 保证金机制（螺纹钢10%）
3. 手续费（万一：0.01%）
4. 滑点模拟
5. 基础仓位10万
6. 移动止盈（Trailing Stop）
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import uuid
from dataclasses import dataclass

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.v03_hybrid.strat_v03_hybrid import HybridMSBOBStrategy, OrderBlock


# ========================================================================
# 交易配置
# ========================================================================

class TradingConfig:
    """交易配置"""
    # 基础配置
    INITIAL_CAPITAL = 100_000.0  # 初始资金 10万
    BASE_POSITION_VALUE = 50_000.0  # 单次开仓基础金额 5万

    # 螺纹钢合约参数
    CONTRACT_SIZE = 10.0  # 螺纹钢每手10吨
    MARGIN_RATE = 0.10  # 保证金比例 10%
    COMMISSION_RATE = 0.0001  # 手续费率 万一（开平各收一次）

    # 滑点设置
    SLIPPAGE_TICKS = 1  # 滑点1个tick（螺纹钢1元/吨）

    # 风控参数
    MAX_POSITION_RATIO = 0.8  # 最大仓位占用比例（80%）
    MAX_MARGIN_RATIO = 0.5  # 最大保证金占用比例（50%）
    STOP_LOSS_RATIO = 0.05  # 单笔最大亏损比例（5%）

    # 移动止盈参数
    TRAILING_ACTIVATION_ATR = 1.0  # 盈利多少倍ATR后激活移动止损
    TRAILING_DISTANCE_ATR = 0.5  # 移动止损距离（ATR倍数）


# ========================================================================
# 数据结构
# ========================================================================

@dataclass
class TradeRecord:
    """交易记录"""
    trade_id: str
    entry_time: pd.Timestamp
    exit_time: pd.Timestamp
    direction: str  # 'long' or 'short'
    entry_price: float
    exit_price: float
    quantity: int  # 手数
    entry_cost: float  # 开仓成本（含手续费+滑点）
    exit_cost: float  # 平仓成本（含手续费+滑点）
    margin_used: float  # 占用保证金
    pnl: float  # 净利润（扣除所有费用）
    pnl_ratio: float  # 收益率
    exit_reason: str  # 'tp1', 'stop_loss', 'trailing_stop', 'tp2', 'tp3'

    # 移动止盈相关
    trailing_stop_activated: bool = False  # 是否激活移动止损
    max_profit: float = 0.0  # 最大浮动盈利
    max_profit_ratio: float = 0.0  # 最大浮动盈利率


@dataclass
class AccountSnapshot:
    """账户快照"""
    timestamp: pd.Timestamp
    total_capital: float  # 总资金
    available_cash: float  # 可用现金
    margin_used: float  # 占用保证金
    position_value: float  # 持仓市值
    total_cost: float  # 总成本
    unrealized_pnl: float  # 浮动盈亏
    net_worth: float  # 净值


# ========================================================================
# 移动止盈数据结构
# ========================================================================

class TrailingPosition:
    """带移动止损的持仓"""
    def __init__(self, direction, entry_price, stop_loss, take_profit,
                 quantity, entry_time, margin, entry_commission):
        self.direction = direction
        self.entry_price = entry_price
        self.stop_loss = stop_loss
        self.initial_stop_loss = stop_loss
        self.take_profit = take_profit
        self.quantity = quantity
        self.entry_time = entry_time
        self.margin = margin
        self.entry_commission = entry_commission

        # 移动止盈状态
        self.highest_price = entry_price if direction == 'long' else None
        self.lowest_price = entry_price if direction == 'short' else None
        self.trailing_stop_activated = False

        # 最大浮动盈利追踪
        self.max_profit = 0.0
        self.max_profit_ratio = 0.0


# ========================================================================
# 真实回测引擎（移动止盈版本）
# ========================================================================

class RealisticTrailingStopBacktestEngine:
    """真实交易回测引擎 - 移动止盈版本"""

    def __init__(self, strategy: HybridMSBOBStrategy, config: TradingConfig = None):
        self.strategy = strategy
        self.config = config or TradingConfig()

        # 账户状态
        self.total_capital = self.config.INITIAL_CAPITAL
        self.available_cash = self.config.INITIAL_CAPITAL
        self.margin_used = 0.0
        self.position_value = 0.0

        # 交易记录
        self.trades: List[TradeRecord] = []
        self.snapshots: List[AccountSnapshot] = []

        # 当前持仓（使用TrailingPosition）
        self.current_position: TrailingPosition = None

    def calculate_position_size(self, price: float) -> int:
        """计算开仓手数"""
        # 单手合约价值
        single_contract_value = price * self.config.CONTRACT_SIZE

        # 单手所需保证金
        margin_per_contract = single_contract_value * self.config.MARGIN_RATE

        # 可用保证金
        available_margin = self.total_capital * self.config.MAX_MARGIN_RATIO

        # 基于可用保证金计算最大手数
        max_contracts_by_margin = int(available_margin / margin_per_contract)

        # 基于目标仓位价值计算手数
        target_contracts = int(self.config.BASE_POSITION_VALUE / single_contract_value)

        # 取较小值，但至少1手
        contracts = min(max_contracts_by_margin, target_contracts)
        contracts = max(1, contracts)

        return contracts

    def calculate_commission(self, price: float, quantity: int) -> float:
        """计算手续费（开平各收一次）"""
        contract_value = price * quantity * self.config.CONTRACT_SIZE
        return contract_value * self.config.COMMISSION_RATE

    def calculate_slippage(self, price: float, direction: str) -> float:
        """计算滑点成本"""
        # 螺纹钢最小跳动1元/吨
        slippage_per_unit = self.config.SLIPPAGE_TICKS * 1.0

        if direction == 'long':
            # 买入时滑点不利
            return price + slippage_per_unit
        else:
            # 卖出时滑点不利
            return price - slippage_per_unit

    def open_position(self, timestamp: pd.Timestamp, price: float, direction: str,
                     stop_loss: float, take_profits: Dict[str, float],
                     atr_value: float) -> bool:
        """开仓"""
        if self.current_position is not None:
            return False  # 已有持仓，不开新仓

        # 计算手数
        quantity = self.calculate_position_size(price)

        # 计算实际成交价（含滑点）
        entry_price = self.calculate_slippage(price, direction)

        # 计算开仓手续费
        entry_commission = self.calculate_commission(entry_price, quantity)

        # 计算保证金
        contract_value = entry_price * quantity * self.config.CONTRACT_SIZE
        margin = contract_value * self.config.MARGIN_RATE

        # 检查可用资金（需要同时支付手续费和保证金）
        total_required = entry_commission + margin

        if total_required > self.available_cash:
            return False  # 资金不足

        # 检查保证金占用
        MAX_MARGIN_RATIO = 0.5
        if self.margin_used + margin > self.total_capital * MAX_MARGIN_RATIO:
            return False  # 保证金超限

        # 更新账户状态
        self.available_cash -= total_required
        self.margin_used += margin
        self.position_value = contract_value

        # 记录持仓（使用TrailingPosition）
        self.current_position = TrailingPosition(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profits,
            quantity=quantity,
            entry_time=timestamp,
            margin=margin,
            entry_commission=entry_commission
        )

        return True

    def close_position(self, timestamp: pd.Timestamp, price: float, reason: str,
                       atr_value: float = None) -> bool:
        """平仓"""
        if self.current_position is None:
            return False

        pos = self.current_position
        direction = pos.direction
        quantity = pos.quantity

        # 计算实际成交价（含滑点）
        exit_price = self.calculate_slippage(price, 'short' if direction == 'long' else 'long')

        # 计算平仓手续费
        exit_commission = self.calculate_commission(exit_price, quantity)

        # 计算盈亏
        if direction == 'long':
            # 多头：卖出价 - 买入价
            profit_per_unit = exit_price - pos.entry_price
        else:
            # 空头：买入价 - 卖出价
            profit_per_unit = pos.entry_price - exit_price

        gross_profit = profit_per_unit * quantity * self.config.CONTRACT_SIZE

        # 净利润（扣除手续费）
        total_commission = pos.entry_commission + exit_commission
        net_profit = gross_profit - total_commission

        # 收益率（相对于占用保证金）
        pnl_ratio = net_profit / pos.margin

        # 计算最大浮动盈利
        max_profit = getattr(pos, 'max_profit', 0)
        max_profit_ratio = getattr(pos, 'max_profit_ratio', 0)

        # 更新账户
        self.available_cash += (pos.margin + net_profit)
        self.margin_used -= pos.margin
        self.position_value = 0.0
        self.total_capital = self.available_cash + self.margin_used

        # 记录交易
        trade = TradeRecord(
            trade_id=str(uuid.uuid4()),
            entry_time=pos.entry_time,
            exit_time=timestamp,
            direction=direction,
            entry_price=pos.entry_price,
            exit_price=exit_price,
            quantity=quantity,
            entry_cost=pos.entry_commission,
            exit_cost=exit_commission,
            margin_used=pos.margin,
            pnl=net_profit,
            pnl_ratio=pnl_ratio,
            exit_reason=reason,
            trailing_stop_activated=getattr(pos, 'trailing_stop_activated', False),
            max_profit=max_profit,
            max_profit_ratio=max_profit_ratio
        )
        self.trades.append(trade)

        # 清空持仓
        self.current_position = None

        return True

    def update_trailing_stop(self, current_price: float, atr_value: float) -> bool:
        """更新移动止损（返回是否触发了止损）"""
        if self.current_position is None:
            return False

        pos = self.current_position

        if pos.direction == 'long':
            # === 多头移动止损逻辑 ===

            # 更新最高价
            if pos.highest_price is None or current_price > pos.highest_price:
                pos.highest_price = current_price

            # 计算盈利距离
            profit_distance = pos.highest_price - pos.entry_price

            # 检查是否应该激活移动止损
            if not pos.trailing_stop_activated:
                activation_distance = atr_value * self.config.TRAILING_ACTIVATION_ATR
                if profit_distance >= activation_distance:
                    pos.trailing_stop_activated = True

            # 如果已激活，更新移动止损
            if pos.trailing_stop_activated:
                trailing_distance = atr_value * self.config.TRAILING_DISTANCE_ATR
                new_stop = pos.highest_price - trailing_distance

                # 移动止损只能上移，不能下移
                if new_stop > pos.stop_loss:
                    pos.stop_loss = new_stop

            # 检查止损
            if current_price <= pos.stop_loss:
                return True  # 触发止损

            # 更新最大浮动盈利
            unrealized_pnl = (current_price - pos.entry_price) * pos.quantity * self.config.CONTRACT_SIZE
            if unrealized_pnl > pos.max_profit:
                pos.max_profit = unrealized_pnl
                pos.max_profit_ratio = unrealized_pnl / pos.margin

        else:
            # === 空头移动止损逻辑 ===

            # 更新最低价
            if pos.lowest_price is None or current_price < pos.lowest_price:
                pos.lowest_price = current_price

            # 计算盈利距离
            profit_distance = pos.entry_price - pos.lowest_price

            # 检查是否应该激活移动止损
            if not pos.trailing_stop_activated:
                activation_distance = atr_value * self.config.TRAILING_ACTIVATION_ATR
                if profit_distance >= activation_distance:
                    pos.trailing_stop_activated = True

            # 如果已激活，更新移动止损
            if pos.trailing_stop_activated:
                trailing_distance = atr_value * self.config.TRAILING_DISTANCE_ATR
                new_stop = pos.lowest_price + trailing_distance

                # 移动止损只能下移，不能上移
                if new_stop < pos.stop_loss:
                    pos.stop_loss = new_stop

            # 检查止损
            if current_price >= pos.stop_loss:
                return True  # 触发止损

            # 更新最大浮动盈利
            unrealized_pnl = (pos.entry_price - current_price) * pos.quantity * self.config.CONTRACT_SIZE
            if unrealized_pnl > pos.max_profit:
                pos.max_profit = unrealized_pnl
                pos.max_profit_ratio = unrealized_pnl / pos.margin

        return False

    def update_unrealized_pnl(self, current_price: float) -> float:
        """更新浮动盈亏"""
        if self.current_position is None:
            return 0.0

        pos = self.current_position
        direction = pos.direction
        quantity = pos.quantity

        if direction == 'long':
            unrealized_pnl = (current_price - pos.entry_price) * quantity * self.config.CONTRACT_SIZE
        else:
            unrealized_pnl = (pos.entry_price - current_price) * quantity * self.config.CONTRACT_SIZE

        # 更新持仓市值
        self.position_value = current_price * quantity * self.config.CONTRACT_SIZE

        return unrealized_pnl

    def take_snapshot(self, timestamp: pd.Timestamp, current_price: float):
        """记录账户快照"""
        unrealized_pnl = self.update_unrealized_pnl(current_price)

        # 净值 = 可用现金 + 保证金 + 浮动盈亏
        net_worth = self.available_cash + self.margin_used + unrealized_pnl

        snapshot = AccountSnapshot(
            timestamp=timestamp,
            total_capital=self.total_capital,
            available_cash=self.available_cash,
            margin_used=self.margin_used,
            position_value=self.position_value,
            total_cost=sum(t.entry_cost + t.exit_cost for t in self.trades),
            unrealized_pnl=unrealized_pnl,
            net_worth=net_worth
        )

        self.snapshots.append(snapshot)

    def run_backtest(self, df: pd.DataFrame) -> pd.DataFrame:
        """运行回测"""
        # 运行策略生成信号
        df_sig = self.strategy.run_strategy(df)

        # 初始化
        self.total_capital = self.config.INITIAL_CAPITAL
        self.available_cash = self.config.INITIAL_CAPITAL
        self.margin_used = 0.0
        self.position_value = 0.0

        self.trades = []
        self.snapshots = []
        self.current_position = None

        # 逐K线处理
        for i in range(len(df_sig)):
            row = df_sig.iloc[i]
            timestamp = df_sig.index[i]
            atr_value = row['atr'] if 'atr' in row else df_sig['atr'].iloc[i]

            # 如果有持仓，检查平仓（包括移动止损）
            if self.current_position is not None:
                pos = self.current_position
                should_close = False
                close_reason = None

                # 检查移动止损
                if self.update_trailing_stop(row['close'], atr_value):
                    should_close = True
                    close_reason = 'trailing_stop'
                # 检查固定止损（初始止损）
                elif pos.direction == 'long' and row['low'] <= pos.initial_stop_loss:
                    should_close = True
                    close_reason = 'initial_stop_loss'
                elif pos.direction == 'short' and row['high'] >= pos.initial_stop_loss:
                    should_close = True
                    close_reason = 'initial_stop_loss'
                # 检查TP1止盈
                elif pos.direction == 'long' and row['high'] >= pos.take_profit['tp1']:
                    should_close = True
                    close_reason = 'tp1'
                elif pos.direction == 'short' and row['low'] <= pos.take_profit['tp1']:
                    should_close = True
                    close_reason = 'tp1'

                if should_close:
                    # 使用触发价平仓
                    if close_reason in ['trailing_stop', 'initial_stop_loss']:
                        close_price = pos.stop_loss
                    else:
                        close_price = pos.take_profit['tp1']

                    self.close_position(timestamp, close_price, close_reason, atr_value)

            # 如果无持仓，检查开仓
            if self.current_position is None and row['position'] != 0:
                direction = 'long' if row['position'] > 0 else 'short'

                # 从策略中获取止损止盈
                stop_loss = row['stop_loss']
                tp1 = row['take_profit']

                take_profits = {'tp1': tp1, 'tp2': tp1, 'tp3': tp1}

                self.open_position(
                    timestamp=timestamp,
                    price=row['close'],
                    direction=direction,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    atr_value=atr_value
                )

            # 记录快照
            self.take_snapshot(timestamp, row['close'])

        # 最后如果有持仓，强制平仓
        if self.current_position is not None:
            last_row = df_sig.iloc[-1]
            self.close_position(df_sig.index[-1], last_row['close'], 'force_close')

        # 添加快照到原始数据
        snapshot_data = {
            'account_total_capital': [s.total_capital for s in self.snapshots],
            'available_cash': [s.available_cash for s in self.snapshots],
            'margin_used': [s.margin_used for s in self.snapshots],
            'position_value': [s.position_value for s in self.snapshots],
            'unrealized_pnl': [s.unrealized_pnl for s in self.snapshots],
            'net_worth': [s.net_worth for s in self.snapshots]
        }

        for key, values in snapshot_data.items():
            df_sig[key] = values

        return df_sig


# ========================================================================
# 真实回测指标计算
# ========================================================================

def calculate_realistic_metrics(engine: RealisticTrailingStopBacktestEngine,
                                 df: pd.DataFrame, freq='60min'):
    """计算真实回测指标"""
    if not engine.trades:
        return _empty_metrics()

    # 基本统计
    total_trades = len(engine.trades)
    winning_trades = [t for t in engine.trades if t.pnl > 0]
    losing_trades = [t for t in engine.trades if t.pnl < 0]

    win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

    # 盈亏统计
    avg_win = np.mean([t.pnl for t in winning_trades]) if winning_trades else 0
    avg_loss = np.mean([t.pnl for t in losing_trades]) if losing_trades else 0
    profit_loss_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else 0

    # 收益统计
    total_pnl = sum(t.pnl for t in engine.trades)
    total_commission = sum(t.entry_cost + t.exit_cost for t in engine.trades)
    gross_profit = sum(t.pnl for t in winning_trades)
    gross_loss = sum(t.pnl for t in losing_trades)

    # 初始和最终净值
    initial_capital = engine.config.INITIAL_CAPITAL
    final_net_worth = engine.snapshots[-1].net_worth

    # 总收益率
    total_return = (final_net_worth - initial_capital) / initial_capital

    # 最大回撤
    net_worth_series = [s.net_worth for s in engine.snapshots]
    cummax = pd.Series(net_worth_series).expanding().max()
    drawdown_series = (cummax - pd.Series(net_worth_series)) / cummax
    max_drawdown = drawdown_series.max()

    # 夏普比率
    returns = pd.Series([s.net_worth for s in engine.snapshots]).pct_change().dropna()
    trading_periods = {'5min': 48*252, '15min': 16*252, '60min': 4*252}
    periods = trading_periods.get(freq, 4*252)

    sharpe_ratio = 0
    if returns.std() > 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(periods)

    # 盈亏比（金额）
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0

    # 移动止盈统计
    trailing_stop_exits = sum(1 for t in engine.trades if t.exit_reason == 'trailing_stop')
    tp1_exits = sum(1 for t in engine.trades if t.exit_reason == 'tp1')
    stop_loss_exits = sum(1 for t in engine.trades if 'stop_loss' in t.exit_reason)

    activated_count = sum(1 for t in engine.trades if t.trailing_stop_activated)
    activation_rate = activated_count / total_trades * 100 if total_trades > 0 else 0

    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_loss_ratio': profit_loss_ratio,
        'total_return': total_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'total_pnl': total_pnl,
        'total_commission': total_commission,
        'gross_profit': gross_profit,
        'gross_loss': gross_loss,
        'profit_factor': profit_factor,
        'initial_capital': initial_capital,
        'final_net_worth': final_net_worth,
        'commission_ratio': total_commission / initial_capital,
        'trailing_stop_exits': trailing_stop_exits,
        'tp1_exits': tp1_exits,
        'stop_loss_exits': stop_loss_exits,
        'trailing_activation_rate': activation_rate,
        'avg_max_profit_ratio': np.mean([t.max_profit_ratio for t in engine.trades]) if engine.trades else 0,
        'long_trades': sum(1 for t in engine.trades if t.direction == 'long'),
        'short_trades': sum(1 for t in engine.trades if t.direction == 'short')
    }


def _empty_metrics():
    """返回空指标"""
    return {
        'total_trades': 0,
        'win_rate': 0,
        'avg_win': 0,
        'avg_loss': 0,
        'profit_loss_ratio': 0,
        'total_return': 0,
        'max_drawdown': 0,
        'sharpe_ratio': 0,
        'total_pnl': 0,
        'total_commission': 0,
        'gross_profit': 0,
        'gross_loss': 0,
        'profit_factor': 0,
        'initial_capital': 100_000,
        'final_net_worth': 100_000,
        'commission_ratio': 0,
        'trailing_stop_exits': 0,
        'tp1_exits': 0,
        'stop_loss_exits': 0,
        'trailing_activation_rate': 0,
        'avg_max_profit_ratio': 0,
        'long_trades': 0,
        'short_trades': 0
    }


# ========================================================================
# 主程序
# ========================================================================

def run_realistic_trailing_stop_backtest(year: int, freq: str = '15min') -> Tuple[dict, str]:
    """运行真实移动止盈回测"""
    # 加载数据
    data_file = project_root / "data" / f"RB9999_{year}_{freq}.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    df = pd.read_csv(data_file, parse_dates=['datetime'])
    df = df.sort_values('datetime').set_index('datetime')

    print(f"\n{year}年 {freq}周期:")
    print(f"  数据行数: {len(df)}")
    print(f"  时间范围: {df.index.min()} ~ {df.index.max()}")

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

    # 创建回测引擎
    config = TradingConfig()
    engine = RealisticTrailingStopBacktestEngine(strategy, config)

    # 运行回测
    df_result = engine.run_backtest(df)

    # 计算指标
    metrics = calculate_realistic_metrics(engine, df_result, freq=freq)
    metrics['year'] = year
    metrics['freq'] = freq

    # 打印结果
    print(f"  交易: {metrics['total_trades']}笔  "
          f"胜率: {metrics['win_rate']:.1f}%  "
          f"收益: {metrics['total_return']:.2%}  "
          f"回撤: {metrics['max_drawdown']:.2%}  "
          f"夏普: {metrics['sharpe_ratio']:.2f}")

    print(f"  移动止盈: {metrics['trailing_stop_exits']}笔  "
          f"激活率: {metrics['trailing_activation_rate']:.1f}%  "
          f"最大浮盈: {metrics['avg_max_profit_ratio']:.1%}")

    return metrics, engine


def main():
    """主函数"""
    import sys
    import io
    # 设置UTF-8编码输出
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 120)
    print("真实交易回测系统 - 移动止盈版本（含交易成本）")
    print("=" * 120)
    print("\n交易配置:")
    print(f"  初始资金: {TradingConfig.INITIAL_CAPITAL:,.2f}元")
    print(f"  单次开仓: {TradingConfig.BASE_POSITION_VALUE:,.2f}元")
    print(f"  合约乘数: {TradingConfig.CONTRACT_SIZE}")
    print(f"  保证金: {TradingConfig.MARGIN_RATE:.1%}")
    print(f"  手续费: {TradingConfig.COMMISSION_RATE:.4%} (万一)")
    print(f"  滑点: {TradingConfig.SLIPPAGE_TICKS} tick")
    print(f"  移动止盈激活: {TradingConfig.TRAILING_ACTIVATION_ATR}倍ATR")
    print(f"  移动止损距离: {TradingConfig.TRAILING_DISTANCE_ATR}倍ATR")

    years = [2023, 2024, 2025]
    freq = '15min'

    all_metrics = []

    for year in years:
        try:
            metrics, _ = run_realistic_trailing_stop_backtest(year, freq)
            all_metrics.append(metrics)
        except Exception as e:
            print(f"\n{year}年: 失败 - {e}")
            import traceback
            traceback.print_exc()

    # 汇总
    if all_metrics:
        print(f"\n{'='*120}")
        print("汇总")
        print(f"{'='*120}\n")

        total_trades = sum(m['total_trades'] for m in all_metrics)
        total_pnl = sum(m['total_pnl'] for m in all_metrics)
        total_commission = sum(m['total_commission'] for m in all_metrics)

        weighted_win_rate = sum(m['win_rate'] * m['total_trades'] for m in all_metrics) / total_trades
        total_return = sum(m['total_return'] for m in all_metrics)
        max_dd = max(m['max_drawdown'] for m in all_metrics)
        avg_sharpe = np.mean([m['sharpe_ratio'] for m in all_metrics])

        trailing_exits = sum(m['trailing_stop_exits'] for m in all_metrics)
        activation_rate = np.mean([m['trailing_activation_rate'] for m in all_metrics])

        print(f"总交易: {total_trades}笔")
        print(f"总净利润: {total_pnl:,.2f}元")
        print(f"总手续费: {total_commission:,.2f}元 (占比: {total_commission/100000:.2%})")
        print(f"加权胜率: {weighted_win_rate:.2f}%")
        print(f"总收益率: {total_return:.2%}")
        print(f"最大回撤: {max_dd:.2%}")
        print(f"平均夏普: {avg_sharpe:.2f}")
        print(f"\n移动止盈统计:")
        print(f"  移动止盈出场: {trailing_exits}笔 ({trailing_exits/total_trades*100:.1f}%)")
        print(f"  激活率: {activation_rate:.1f}% (激活移动止损的交易占比)")

        # 保存汇总
        output = Path(__file__).parent / "realistic_trailing_results.csv"
        df_metrics = pd.DataFrame(all_metrics)
        df_metrics.to_csv(output, index=False, encoding='utf-8-sig')
        print(f"\n结果已保存: {output}")

        return all_metrics


if __name__ == "__main__":
    main()
