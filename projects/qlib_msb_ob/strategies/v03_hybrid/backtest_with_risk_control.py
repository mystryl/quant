#!/usr/bin/env python3
"""
带风险控制的回测脚本
使用15分钟周期优化参数进行2023-2025年回测

优化参数（基于2025年测试，2023-2024验证）:
- pivot_len=7
- msb_zscore=0.3018
- atr_period=12
- atr_multiplier=1.143
- tp1_multiplier=0.300
- tp2_multiplier=0.888
- tp3_multiplier=1.456
"""
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import uuid
from dataclasses import dataclass
import json

# 添加项目根目录到路径
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from strategies.v03_hybrid.strat_v03_hybrid import HybridMSBOBStrategy


# ========================================================================
# 交易配置（使用realistic_backtest.py的配置）
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
    exit_reason: str  # 'tp1', 'stop_loss', 'tp2', 'tp3', 'force_close'
    signal_quality: float = 0.0  # 信号质量评分（如果有）


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
# 回测引擎（基于realistic_backtest.py）
# ========================================================================

class RiskControlledBacktestEngine:
    """带风险控制的回测引擎"""

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

        # 当前持仓
        self.current_position = None

    def calculate_position_size(self, price: float, signal_quality: float = 1.0) -> int:
        """
        计算开仓手数
        使用信号质量评分调整仓位大小（如果有）
        """
        # 单手合约价值
        single_contract_value = price * self.config.CONTRACT_SIZE

        # 单手所需保证金
        margin_per_contract = single_contract_value * self.config.MARGIN_RATE

        # 可用保证金
        available_margin = self.total_capital * self.config.MAX_MARGIN_RATIO

        # 基于可用保证金计算最大手数
        max_contracts_by_margin = int(available_margin / margin_per_contract)

        # 基于目标仓位价值计算手数（使用信号质量调整）
        target_value = self.config.BASE_POSITION_VALUE * signal_quality
        target_contracts = int(target_value / single_contract_value)

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
        slippage_per_unit = self.config.SLIPPAGE_TICKS * 1.0

        if direction == 'long':
            return price + slippage_per_unit
        else:
            return price - slippage_per_unit

    def open_position(self, timestamp: pd.Timestamp, price: float, direction: str,
                     stop_loss: float, take_profits: Dict[str, float],
                     signal_quality: float = 1.0) -> bool:
        """开仓"""
        if self.current_position is not None:
            return False

        # 计算手数（使用信号质量调整）
        quantity = self.calculate_position_size(price, signal_quality)

        # 计算实际成交价（含滑点）
        entry_price = self.calculate_slippage(price, direction)

        # 计算开仓手续费
        entry_commission = self.calculate_commission(entry_price, quantity)

        # 计算保证金
        contract_value = entry_price * quantity * self.config.CONTRACT_SIZE
        margin = contract_value * self.config.MARGIN_RATE

        # 检查可用资金
        total_required = entry_commission + margin

        if total_required > self.available_cash:
            return False

        if self.margin_used + margin > self.total_capital * self.config.MAX_MARGIN_RATIO:
            return False

        # 更新账户状态
        self.available_cash -= total_required
        self.margin_used += margin
        self.position_value = contract_value

        # 记录持仓
        self.current_position = {
            'direction': direction,
            'quantity': quantity,
            'entry_price': entry_price,
            'entry_time': timestamp,
            'entry_commission': entry_commission,
            'stop_loss': stop_loss,
            'take_profits': take_profits,
            'margin': margin,
            'signal_quality': signal_quality
        }

        return True

    def close_position(self, timestamp: pd.Timestamp, price: float, reason: str) -> bool:
        """平仓"""
        if self.current_position is None:
            return False

        pos = self.current_position
        direction = pos['direction']
        quantity = pos['quantity']

        # 计算实际成交价（含滑点）
        exit_price = self.calculate_slippage(price, 'short' if direction == 'long' else 'long')

        # 计算平仓手续费
        exit_commission = self.calculate_commission(exit_price, quantity)

        # 计算盈亏
        if direction == 'long':
            profit_per_unit = exit_price - pos['entry_price']
        else:
            profit_per_unit = pos['entry_price'] - exit_price

        gross_profit = profit_per_unit * quantity * self.config.CONTRACT_SIZE

        # 净利润（扣除手续费）
        total_commission = pos['entry_commission'] + exit_commission
        net_profit = gross_profit - total_commission

        # 收益率（相对于占用保证金）
        pnl_ratio = net_profit / pos['margin']

        # 更新账户
        self.available_cash += (pos['margin'] + net_profit)
        self.margin_used -= pos['margin']
        self.position_value = 0.0
        self.total_capital = self.available_cash + self.margin_used

        # 记录交易
        trade = TradeRecord(
            trade_id=str(uuid.uuid4()),
            entry_time=pos['entry_time'],
            exit_time=timestamp,
            direction=direction,
            entry_price=pos['entry_price'],
            exit_price=exit_price,
            quantity=quantity,
            entry_cost=pos['entry_commission'],
            exit_cost=exit_commission,
            margin_used=pos['margin'],
            pnl=net_profit,
            pnl_ratio=pnl_ratio,
            exit_reason=reason,
            signal_quality=pos.get('signal_quality', 1.0)
        )
        self.trades.append(trade)

        # 清空持仓
        self.current_position = None

        return True

    def update_unrealized_pnl(self, current_price: float) -> float:
        """更新浮动盈亏"""
        if self.current_position is None:
            return 0.0

        pos = self.current_position
        direction = pos['direction']
        quantity = pos['quantity']

        if direction == 'long':
            profit_per_unit = current_price - pos['entry_price']
        else:
            profit_per_unit = pos['entry_price'] - current_price

        unrealized_pnl = profit_per_unit * quantity * self.config.CONTRACT_SIZE
        self.position_value = current_price * quantity * self.config.CONTRACT_SIZE

        return unrealized_pnl

    def take_snapshot(self, timestamp: pd.Timestamp, current_price: float):
        """记录账户快照"""
        unrealized_pnl = self.update_unrealized_pnl(current_price)
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

            # 如果有持仓，检查平仓
            if self.current_position is not None:
                pos = self.current_position
                should_close = False
                close_reason = None

                if pos['direction'] == 'long':
                    if row['low'] <= pos['stop_loss']:
                        should_close = True
                        close_reason = 'stop_loss'
                    elif row['high'] >= pos['take_profits']['tp1']:
                        should_close = True
                        close_reason = 'tp1'
                else:
                    if row['high'] >= pos['stop_loss']:
                        should_close = True
                        close_reason = 'stop_loss'
                    elif row['low'] <= pos['take_profits']['tp1']:
                        should_close = True
                        close_reason = 'tp1'

                if should_close:
                    if close_reason == 'stop_loss':
                        close_price = pos['stop_loss']
                    else:
                        close_price = pos['take_profits']['tp1']

                    self.close_position(timestamp, close_price, close_reason)

            # 如果无持仓，检查开仓
            if self.current_position is None and row['position'] != 0:
                direction = 'long' if row['position'] > 0 else 'short'

                stop_loss = row['stop_loss']
                tp1 = row['take_profit']
                take_profits = {'tp1': tp1, 'tp2': tp1, 'tp3': tp1}

                # 使用默认信号质量=1.0（如果后续实现质量评分系统可传入）
                self.open_position(
                    timestamp=timestamp,
                    price=row['close'],
                    direction=direction,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    signal_quality=1.0
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
# 回测指标计算
# ========================================================================

def calculate_metrics(engine: RiskControlledBacktestEngine, df: pd.DataFrame, freq='15min') -> dict:
    """计算回测指标"""
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
    periods = trading_periods.get(freq, 16*252)

    sharpe_ratio = 0
    if returns.std() > 0:
        sharpe_ratio = returns.mean() / returns.std() * np.sqrt(periods)

    # 盈亏比（金额）
    profit_factor = abs(gross_profit / gross_loss) if gross_loss != 0 else 0

    # 平均收益/风险
    avg_trade_pnl = total_pnl / total_trades if total_trades > 0 else 0
    avg_trade_pnl_ratio = avg_trade_pnl / initial_capital

    # 最大单笔盈利/亏损
    max_single_win = max([t.pnl for t in winning_trades]) if winning_trades else 0
    max_single_loss = min([t.pnl for t in losing_trades]) if losing_trades else 0

    # 手续费占比
    commission_ratio = total_commission / initial_capital if initial_capital > 0 else 0

    # 交易统计
    tp1_exits = sum(1 for t in engine.trades if t.exit_reason == 'tp1')
    sl_exits = sum(1 for t in engine.trades if t.exit_reason == 'stop_loss')
    fc_exits = sum(1 for t in engine.trades if t.exit_reason == 'force_close')

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
        'avg_trade_pnl': avg_trade_pnl,
        'avg_trade_pnl_ratio': avg_trade_pnl_ratio,
        'max_single_win': max_single_win,
        'max_single_loss': max_single_loss,
        'commission_ratio': commission_ratio,
        'tp1_exits': tp1_exits,
        'sl_exits': sl_exits,
        'force_close_exits': fc_exits,
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
        'avg_trade_pnl': 0,
        'avg_trade_pnl_ratio': 0,
        'max_single_win': 0,
        'max_single_loss': 0,
        'commission_ratio': 0,
        'tp1_exits': 0,
        'sl_exits': 0,
        'force_close_exits': 0,
        'long_trades': 0,
        'short_trades': 0
    }


# ========================================================================
# 报告生成
# ========================================================================

def generate_report(engine: RiskControlledBacktestEngine, metrics: dict,
                   year: int, freq: str, params: dict) -> str:
    """生成详细报表"""
    report = []
    report.append("=" * 100)
    report.append(f"风险控制回测报告 - {year}年 {freq}周期")
    report.append("=" * 100)

    # 参数配置
    report.append("\n【优化参数配置】")
    report.append("-" * 100)
    report.append(f"  pivot_len: {params['pivot_len']}")
    report.append(f"  msb_zscore: {params['msb_zscore']:.4f}")
    report.append(f"  atr_period: {params['atr_period']}")
    report.append(f"  atr_multiplier: {params['atr_multiplier']:.3f}")
    report.append(f"  tp1_multiplier: {params['tp1_multiplier']:.3f}")
    report.append(f"  tp2_multiplier: {params['tp2_multiplier']:.3f}")
    report.append(f"  tp3_multiplier: {params['tp3_multiplier']:.3f}")

    # 账户概况
    report.append("\n【账户概况】")
    report.append("-" * 100)
    report.append(f"  初始资金: {metrics['initial_capital']:>12,.2f}元")
    report.append(f"  最终净值: {metrics['final_net_worth']:>12,.2f}元")
    report.append(f"  净利润: {metrics['total_pnl']:>12,.2f}元")
    report.append(f"  总收益率: {metrics['total_return']:>10.2%}")
    report.append(f"  总手续费: {metrics['total_commission']:>12,.2f}元  (占比: {metrics['commission_ratio']:.2%})")

    # 交易统计
    report.append("\n【交易统计】")
    report.append("-" * 100)
    report.append(f"  总交易次数: {metrics['total_trades']}")
    report.append(f"  多头交易: {metrics['long_trades']}")
    report.append(f"  空头交易: {metrics['short_trades']}")
    report.append(f"  胜率: {metrics['win_rate']:.2f}%")
    report.append(f"  TP1止盈: {metrics['tp1_exits']}笔")
    report.append(f"  止损出场: {metrics['sl_exits']}笔")
    report.append(f"  强制平仓: {metrics['force_close_exits']}笔")

    # 盈亏分析
    report.append("\n【盈亏分析】")
    report.append("-" * 100)
    report.append(f"  平均盈利: {metrics['avg_win']:>10,.2f}元")
    report.append(f"  平均亏损: {metrics['avg_loss']:>10,.2f}元")
    report.append(f"  盈亏比: {metrics['profit_loss_ratio']:.2f}")
    report.append(f"  盈利因子: {metrics['profit_factor']:.2f}")
    report.append(f"  平均每笔收益: {metrics['avg_trade_pnl']:>10,.2f}元  ({metrics['avg_trade_pnl_ratio']:.2%})")
    report.append(f"  最大单笔盈利: {metrics['max_single_win']:>10,.2f}元")
    report.append(f"  最大单笔亏损: {metrics['max_single_loss']:>10,.2f}元")

    # 风险指标
    report.append("\n【风险指标】")
    report.append("-" * 100)
    report.append(f"  最大回撤: {metrics['max_drawdown']:.2%}")
    report.append(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")

    # 详细交易记录（前10笔和后10笔）
    if engine.trades:
        report.append("\n【交易明细（前10笔）】")
        report.append("-" * 100)
        report.append(f"{'序号':<4} {'方向':<6} {'开仓时间':<19} {'平仓时间':<19} "
                     f"{'开仓价':<8} {'平仓价':<8} {'手数':<4} {'净利润':>10} {'收益率':>8} {'原因':<10}")
        report.append("-" * 100)

        for i, trade in enumerate(engine.trades[:10], 1):
            direction_symbol = "多" if trade.direction == 'long' else "空"
            pnl_str = f"{trade.pnl:>+,.2f}"

            report.append(
                f"{i:<4} {direction_symbol:<6} "
                f"{trade.entry_time.strftime('%Y-%m-%d %H:%M'):<19} "
                f"{trade.exit_time.strftime('%Y-%m-%d %H:%M'):<19} "
                f"{trade.entry_price:<8.2f} {trade.exit_price:<8.2f} "
                f"{trade.quantity:<4} {pnl_str:>10} "
                f"{trade.pnl_ratio:>7.2%} {trade.exit_reason:<10}"
            )

        if len(engine.trades) > 10:
            report.append(f"\n... 省略 {len(engine.trades) - 20} 笔交易 ...\n")

            report.append("\n【交易明细（后10笔）】")
            report.append("-" * 100)
            for i, trade in enumerate(engine.trades[-10:], len(engine.trades) - 9):
                direction_symbol = "多" if trade.direction == 'long' else "空"
                pnl_str = f"{trade.pnl:>+,.2f}"

                report.append(
                    f"{i:<4} {direction_symbol:<6} "
                    f"{trade.entry_time.strftime('%Y-%m-%d %H:%M'):<19} "
                    f"{trade.exit_time.strftime('%Y-%m-%d %H:%M'):<19} "
                    f"{trade.entry_price:<8.2f} {trade.exit_price:<8.2f} "
                    f"{trade.quantity:<4} {pnl_str:>10} "
                    f"{trade.pnl_ratio:>7.2%} {trade.exit_reason:<10}"
                )

    report.append("\n" + "=" * 100)

    return "\n".join(report)


def save_trades_to_csv(engine: RiskControlledBacktestEngine, output_path: Path):
    """保存交易记录到CSV"""
    if not engine.trades:
        return

    trades_data = []
    for t in engine.trades:
        trades_data.append({
            'trade_id': t.trade_id,
            'direction': t.direction,
            'entry_time': t.entry_time,
            'exit_time': t.exit_time,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'quantity': t.quantity,
            'entry_cost': t.entry_cost,
            'exit_cost': t.exit_cost,
            'margin_used': t.margin_used,
            'pnl': t.pnl,
            'pnl_ratio': t.pnl_ratio,
            'exit_reason': t.exit_reason,
            'signal_quality': t.signal_quality
        })

    df_trades = pd.DataFrame(trades_data)
    df_trades.to_csv(output_path, index=False, encoding='utf-8-sig')


# ========================================================================
# 主程序
# ========================================================================

# 优化参数（15分钟周期）
OPTIMIZED_PARAMS_15MIN = {
    'pivot_len': 7,
    'msb_zscore': 0.3018,
    'atr_period': 12,
    'atr_multiplier': 1.143,
    'tp1_multiplier': 0.300,
    'tp2_multiplier': 0.888,
    'tp3_multiplier': 1.456
}

# 原始参数（用于对比）
ORIGINAL_PARAMS = {
    'pivot_len': 7,
    'msb_zscore': 0.3,
    'atr_period': 14,
    'atr_multiplier': 1.0,
    'tp1_multiplier': 0.5,
    'tp2_multiplier': 1.0,
    'tp3_multiplier': 1.5
}


def run_backtest_with_params(year: int, freq: str, params: dict) -> Tuple[dict, str]:
    """使用指定参数运行回测"""
    # 加载数据
    data_file = project_root / "data" / f"RB9999_{year}_{freq}.csv"
    if not data_file.exists():
        raise FileNotFoundError(f"数据文件不存在: {data_file}")

    df = pd.read_csv(data_file, parse_dates=['datetime'])
    df = df.sort_values('datetime').set_index('datetime')

    print(f"\n{year}年 {freq}周期:")
    print(f"  数据行数: {len(df)}")
    print(f"  时间范围: {df.index.min()} ~ {df.index.max()}")

    # 创建策略（使用指定参数）
    strategy = HybridMSBOBStrategy(**params)

    # 创建回测引擎
    engine = RiskControlledBacktestEngine(strategy, TradingConfig())

    # 运行回测
    df_result = engine.run_backtest(df)

    # 计算指标
    metrics = calculate_metrics(engine, df_result, freq=freq)
    metrics['year'] = year
    metrics['freq'] = freq

    # 生成报表
    report = generate_report(engine, metrics, year, freq, params)

    # 保存交易记录
    output_dir = Path(__file__).parent / "risk_control_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    param_suffix = "_optimized" if params == OPTIMIZED_PARAMS_15MIN else "_original"
    trades_file = output_dir / f"trades_{year}_{freq}{param_suffix}.csv"
    save_trades_to_csv(engine, trades_file)

    # 保存账户快照
    snapshots_file = output_dir / f"snapshots_{year}_{freq}{param_suffix}.csv"
    snapshot_data = {
        'timestamp': [s.timestamp for s in engine.snapshots],
        'net_worth': [s.net_worth for s in engine.snapshots],
        'available_cash': [s.available_cash for s in engine.snapshots],
        'margin_used': [s.margin_used for s in engine.snapshots],
        'position_value': [s.position_value for s in engine.snapshots],
        'unrealized_pnl': [s.unrealized_pnl for s in engine.snapshots]
    }
    df_snapshots = pd.DataFrame(snapshot_data)
    df_snapshots.to_csv(snapshots_file, index=False, encoding='utf-8-sig')

    print(f"  交易记录已保存: {trades_file}")
    print(f"  账户快照已保存: {snapshots_file}")

    return metrics, report


def generate_comparison_report(optimized_metrics: list, original_metrics: list) -> str:
    """生成对比报告"""
    report = []
    report.append("\n")
    report.append("=" * 100)
    report.append("优化前后对比报告 - 15分钟周期 2023-2025年")
    report.append("=" * 100)

    # 汇总优化参数结果
    report.append("\n【优化参数汇总】")
    report.append("-" * 100)

    opt_total_trades = sum(m['total_trades'] for m in optimized_metrics)
    opt_total_pnl = sum(m['total_pnl'] for m in optimized_metrics)
    opt_total_return = sum(m['total_return'] for m in optimized_metrics)
    opt_max_dd = max(m['max_drawdown'] for m in optimized_metrics)
    opt_avg_sharpe = np.mean([m['sharpe_ratio'] for m in optimized_metrics])

    # 加权胜率
    opt_weighted_win_rate = sum(m['win_rate'] * m['total_trades'] for m in optimized_metrics if m['total_trades'] > 0)
    opt_weighted_win_rate = opt_weighted_win_rate / opt_total_trades if opt_total_trades > 0 else 0

    report.append(f"  总交易次数: {opt_total_trades}")
    report.append(f"  加权胜率: {opt_weighted_win_rate:.2f}%")
    report.append(f"  总净利润: {opt_total_pnl:,.2f}元")
    report.append(f"  总收益率: {opt_total_return:.2%}")
    report.append(f"  最大回撤: {opt_max_dd:.2%}")
    report.append(f"  平均夏普: {opt_avg_sharpe:.2f}")

    # 汇总原始参数结果
    report.append("\n【原始参数汇总】")
    report.append("-" * 100)

    orig_total_trades = sum(m['total_trades'] for m in original_metrics)
    orig_total_pnl = sum(m['total_pnl'] for m in original_metrics)
    orig_total_return = sum(m['total_return'] for m in original_metrics)
    orig_max_dd = max(m['max_drawdown'] for m in original_metrics)
    orig_avg_sharpe = np.mean([m['sharpe_ratio'] for m in original_metrics])

    orig_weighted_win_rate = sum(m['win_rate'] * m['total_trades'] for m in original_metrics if m['total_trades'] > 0)
    orig_weighted_win_rate = orig_weighted_win_rate / orig_total_trades if orig_total_trades > 0 else 0

    report.append(f"  总交易次数: {orig_total_trades}")
    report.append(f"  加权胜率: {orig_weighted_win_rate:.2f}%")
    report.append(f"  总净利润: {orig_total_pnl:,.2f}元")
    report.append(f"  总收益率: {orig_total_return:.2%}")
    report.append(f"  最大回撤: {orig_max_dd:.2%}")
    report.append(f"  平均夏普: {orig_avg_sharpe:.2f}")

    # 对比
    report.append("\n【改进效果】")
    report.append("-" * 100)

    pnl_improve = ((opt_total_pnl - orig_total_pnl) / abs(orig_total_pnl) * 100) if orig_total_pnl != 0 else 0
    return_improve = ((opt_total_return - orig_total_return) / abs(orig_total_return) * 100) if orig_total_return != 0 else 0
    sharpe_improve = ((opt_avg_sharpe - orig_avg_sharpe) / abs(orig_avg_sharpe) * 100) if orig_avg_sharpe != 0 else 0
    dd_improve = ((orig_max_dd - opt_max_dd) / orig_max_dd * 100) if orig_max_dd != 0 else 0

    report.append(f"  收益率改进: {return_improve:+.2f}%")
    report.append(f"  夏普比率改进: {sharpe_improve:+.2f}%")
    report.append(f"  最大回撤改进: {dd_improve:+.2f}%")

    # 详细对比表
    report.append("\n【详细对比表】")
    report.append("-" * 100)
    report.append(f"{'年份':<8} {'参数':<12} {'交易':<6} {'胜率':<8} {'收益率':<10} {'回撤':<10} {'夏普':<8}")
    report.append("-" * 100)

    for i, year in enumerate([2023, 2024, 2025]):
        opt = optimized_metrics[i]
        orig = original_metrics[i]

        report.append(f"{year:<8} {'优化':<12} {opt['total_trades']:<6} "
                     f"{opt['win_rate']:>6.2f}% {opt['total_return']:>8.2%}   "
                     f"{opt['max_drawdown']:>8.2%}   {opt['sharpe_ratio']:>6.2f}")

        report.append(f"{year:<8} {'原始':<12} {orig['total_trades']:<6} "
                     f"{orig['win_rate']:>6.2f}% {orig['total_return']:>8.2%}   "
                     f"{orig['max_drawdown']:>8.2%}   {orig['sharpe_ratio']:>6.2f}")

        report.append("-" * 100)

    report.append("\n" + "=" * 100)

    return "\n".join(report)


def main():
    """主函数"""
    import sys
    import io
    # 设置UTF-8编码输出
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 100)
    print("风险控制回测系统 - 使用15分钟优化参数")
    print("=" * 100)
    print("\n优化参数（基于2025测试，2023-2024验证）:")
    print(f"  pivot_len={OPTIMIZED_PARAMS_15MIN['pivot_len']}")
    print(f"  msb_zscore={OPTIMIZED_PARAMS_15MIN['msb_zscore']:.4f}")
    print(f"  atr_period={OPTIMIZED_PARAMS_15MIN['atr_period']}")
    print(f"  atr_multiplier={OPTIMIZED_PARAMS_15MIN['atr_multiplier']:.3f}")
    print(f"  tp1_multiplier={OPTIMIZED_PARAMS_15MIN['tp1_multiplier']:.3f}")
    print(f"  tp2_multiplier={OPTIMIZED_PARAMS_15MIN['tp2_multiplier']:.3f}")
    print(f"  tp3_multiplier={OPTIMIZED_PARAMS_15MIN['tp3_multiplier']:.3f}")

    years = [2023, 2024, 2025]
    freq = '15min'

    # 运行优化参数回测
    print(f"\n{'='*100}")
    print(f"使用优化参数回测 ({freq})")
    print(f"{'='*100}")

    optimized_metrics = []
    optimized_reports = []

    for year in years:
        try:
            metrics, report = run_backtest_with_params(year, freq, OPTIMIZED_PARAMS_15MIN)
            optimized_metrics.append(metrics)
            optimized_reports.append(report)

            # 打印简要结果
            print(f"  交易: {metrics['total_trades']}笔  "
                 f"胜率: {metrics['win_rate']:.1f}%  "
                 f"收益: {metrics['total_return']:.2%}  "
                 f"回撤: {metrics['max_drawdown']:.2%}  "
                 f"夏普: {metrics['sharpe_ratio']:.2f}")

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    # 运行原始参数回测（用于对比）
    print(f"\n{'='*100}")
    print(f"使用原始参数回测 ({freq})")
    print(f"{'='*100}")

    original_metrics = []
    original_reports = []

    for year in years:
        try:
            metrics, report = run_backtest_with_params(year, freq, ORIGINAL_PARAMS)
            original_metrics.append(metrics)
            original_reports.append(report)

            # 打印简要结果
            print(f"  交易: {metrics['total_trades']}笔  "
                 f"胜率: {metrics['win_rate']:.1f}%  "
                 f"收益: {metrics['total_return']:.2%}  "
                 f"回撤: {metrics['max_drawdown']:.2%}  "
                 f"夏普: {metrics['sharpe_ratio']:.2f}")

        except Exception as e:
            print(f"  错误: {e}")
            import traceback
            traceback.print_exc()

    # 保存汇总结果
    output_dir = Path(__file__).parent / "risk_control_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    # 保存指标汇总
    if optimized_metrics:
        df_opt = pd.DataFrame(optimized_metrics)
        opt_summary_file = output_dir / "summary_optimized.csv"
        df_opt.to_csv(opt_summary_file, index=False, encoding='utf-8-sig')
        print(f"\n优化参数汇总已保存: {opt_summary_file}")

    if original_metrics:
        df_orig = pd.DataFrame(original_metrics)
        orig_summary_file = output_dir / "summary_original.csv"
        df_orig.to_csv(orig_summary_file, index=False, encoding='utf-8-sig')
        print(f"原始参数汇总已保存: {orig_summary_file}")

    # 生成并保存对比报告
    if optimized_metrics and original_metrics:
        comparison = generate_comparison_report(optimized_metrics, original_metrics)
        comparison_file = output_dir / "comparison_report.txt"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            f.write(comparison)
        print(f"对比报告已保存: {comparison_file}")

        # 打印对比报告
        print(comparison)

    # 保存完整详细报告
    all_reports = []
    all_reports.extend(optimized_reports)
    all_reports.extend(original_reports)

    detailed_file = output_dir / "detailed_report.txt"
    with open(detailed_file, 'w', encoding='utf-8') as f:
        f.write("\n\n".join(all_reports))
    print(f"\n详细报告已保存: {detailed_file}")

    # 保存参数配置
    config_file = output_dir / "parameters.json"
    with open(config_file, 'w', encoding='utf-8') as f:
        json.dump({
            'optimized_params': OPTIMIZED_PARAMS_15MIN,
            'original_params': ORIGINAL_PARAMS,
            'trading_config': {
                'initial_capital': TradingConfig.INITIAL_CAPITAL,
                'base_position_value': TradingConfig.BASE_POSITION_VALUE,
                'contract_size': TradingConfig.CONTRACT_SIZE,
                'margin_rate': TradingConfig.MARGIN_RATE,
                'commission_rate': TradingConfig.COMMISSION_RATE,
                'slippage_ticks': TradingConfig.SLIPPAGE_TICKS,
                'max_margin_ratio': TradingConfig.MAX_MARGIN_RATIO
            }
        }, f, indent=2)
    print(f"参数配置已保存: {config_file}")

    print(f"\n{'='*100}")
    print(f"所有结果已保存到: {output_dir}")
    print(f"{'='*100}")


if __name__ == "__main__":
    main()
