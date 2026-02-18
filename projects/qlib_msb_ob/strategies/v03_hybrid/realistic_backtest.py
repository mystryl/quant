#!/usr/bin/env python3
"""
真实交易回测系统
包含：
1. 仓位控制（固定金额分配）
2. 保证金机制（螺纹钢10%保证金）
3. 手续费（万一：0.01%）
4. 滑点模拟
5. 基础仓位10万
6. 真实收益计算
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
    exit_reason: str  # 'tp1', 'stop_loss', 'tp2', 'tp3'
    quality_score: float = None  # 信号质量评分


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
# 真实回测引擎
# ========================================================================

class RealisticBacktestEngine:
    """真实交易回测引擎"""

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
        self.current_position = None  # {'type': 'long'/'short', 'quantity': int, 'entry_price': float, ...}

    def calculate_signal_quality(self, df: pd.DataFrame, bar_idx: int, ob: OrderBlock, ema20: float) -> float:
        """
        计算信号质量评分 (0-100分)

        评分维度：
        - 动量强度(40%)：|momentum_z| 越大越好，最高40分
        - 成交量确认(30%)：vol_percentile 越高越好
        - OB宽度质量(20%)：width适中（ATR的0.5-2倍）
        - 趋势一致性(10%)：与EMA20趋势方向一致

        Args:
            df: 包含指标数据的DataFrame
            bar_idx: 当前K线索引
            ob: 订单块对象
            ema20: EMA20均线值

        Returns:
            信号质量评分 (0-100)
        """
        score = 0.0

        # 1. 动量强度 (40分)
        momentum_z = df['momentum_z'].iloc[bar_idx]
        momentum_score = min(abs(momentum_z) / 3.0 * 40, 40)  # |z|>=3 得满分
        score += momentum_score

        # 2. 成交量确认 (30分)
        vol_percentile = df['vol_percentile'].iloc[bar_idx]
        volume_score = vol_percentile / 100 * 30  # 百分位越高越好
        score += volume_score

        # 3. OB宽度质量 (20分)
        # 理想宽度：ATR的0.5-2倍之间
        width_ratio = ob.width / ob.atr_value if ob.atr_value > 0 else 0
        if 0.5 <= width_ratio <= 2.0:
            # 在理想范围内，越接近1越好
            distance_from_ideal = abs(width_ratio - 1.0)
            width_score = max(20 - distance_from_ideal * 10, 10)  # 10-20分
        else:
            # 超出理想范围，扣分
            width_score = max(10 - abs(width_ratio - 1.0) * 5, 0)
        score += width_score

        # 4. 趋势一致性 (10分)
        # 检查价格与EMA20的关系
        current_price = df['close'].iloc[bar_idx]
        if not pd.isna(ema20):
            if ob.ob_type == 'bullish':
                # 看涨OB：价格应在EMA20之上或接近突破
                trend_score = 10 if current_price > ema20 else 5
            else:
                # 看跌OB：价格应在EMA20之下或接近突破
                trend_score = 10 if current_price < ema20 else 5
            score += trend_score
        else:
            # 无EMA数据，给部分分数
            score += 5

        return min(score, 100.0)  # 确保不超过100分

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

    def calculate_position_size_with_risk_control(
        self,
        price: float,
        stop_loss: float,
        quality_score: float
    ) -> int:
        """基于风险的动态仓位计算

        风控规则:
        - 单次开仓不超过30%资金
        - 最大持仓不超过50%资金
        - 止损亏损不超过账户的2%
        - 根据信号质量调整仓位大小

        Args:
            price: 当前价格
            stop_loss: 止损价格
            quality_score: 信号质量分数 (0-100)

        Returns:
            计算后的手数，质量分数<50时返回0
        """
        # 1. 质量分数过滤：低于50不开仓
        if quality_score < 50:
            return 0

        # 2. 单手合约价值
        single_contract_value = price * self.config.CONTRACT_SIZE

        # 3. 单手所需保证金
        margin_per_contract = single_contract_value * self.config.MARGIN_RATE

        # 4. 计算止损亏损比例
        stop_loss_distance = abs(price - stop_loss)
        loss_per_contract = stop_loss_distance * self.config.CONTRACT_SIZE

        # 5. 基于止损限额计算最大手数（止损不超过账户2%）
        max_loss_allowed = self.total_capital * 0.02  # 2%
        max_contracts_by_stop_loss = int(max_loss_allowed / loss_per_contract) if loss_per_contract > 0 else 999

        # 6. 基于单次开仓限额计算最大手数（不超过30%资金）
        max_position_value = self.total_capital * 0.30  # 30%
        max_contracts_by_position = int(max_position_value / single_contract_value)

        # 7. 基于最大持仓限制计算最大手数（不超过50%资金）
        max_margin_allowed = self.total_capital * 0.50  # 50%
        max_contracts_by_margin = int(max_margin_allowed / margin_per_contract)

        # 8. 取各限制的最小值
        max_contracts = min(
            max_contracts_by_stop_loss,
            max_contracts_by_position,
            max_contracts_by_margin
        )
        max_contracts = max(1, max_contracts)  # 至少1手

        # 9. 根据信号质量调整仓位倍率
        if quality_score >= 80:
            # 高质量信号：满仓 (100%)
            size_multiplier = 1.0
        elif quality_score >= 65:
            # 中质量信号：标准仓 (75%)
            size_multiplier = 0.75
        else:  # 50-64
            # 低质量信号：半仓 (50%)
            size_multiplier = 0.50

        # 10. 应用质量调整
        final_contracts = int(max_contracts * size_multiplier)
        final_contracts = max(1, final_contracts)  # 至少1手

        return final_contracts

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
                     quality_score: float = None) -> bool:
        """开仓

        Args:
            quality_score: 信号质量评分 (0-100)，如果提供则使用风险控制仓位计算
        """
        if self.current_position is not None:
            return False  # 已有持仓，不开新仓

        # 计算手数（如果提供了质量评分，使用风险控制仓位计算）
        if quality_score is not None:
            # 使用风险控制
            if quality_score < 50:
                return False  # 质量评分太低，拒绝开仓
            quantity = self.calculate_position_size_with_risk_control(
                price, stop_loss, quality_score
            )
            if quantity == 0:
                return False  # 风控限制，不开仓
        else:
            # 使用原始的固定仓位计算
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

        if self.margin_used + margin > self.total_capital * self.config.MAX_MARGIN_RATIO:
            return False  # 保证金超限

        # 更新账户状态
        # 从可用现金中扣除手续费和保证金
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
            'quality_score': quality_score
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
        # 平仓时，多头卖出价格偏低，空头买入价格偏高
        exit_price = self.calculate_slippage(price, 'short' if direction == 'long' else 'long')

        # 计算平仓手续费
        exit_commission = self.calculate_commission(exit_price, quantity)

        # 计算盈亏
        if direction == 'long':
            # 多头：卖出价 - 买入价
            profit_per_unit = exit_price - pos['entry_price']
        else:
            # 空头：买入价 - 卖出价
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
            quality_score=pos.get('quality_score')
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

        # 更新持仓市值
        self.position_value = current_price * quantity * self.config.CONTRACT_SIZE

        return unrealized_pnl

    def take_snapshot(self, timestamp: pd.Timestamp, current_price: float):
        """记录账户快照"""
        unrealized_pnl = self.update_unrealized_pnl(current_price)

        # 净值 = 可用现金 + 保证金 + 浮动盈亏
        # 注意：position_value是合约面值（名义价值），不计入净值
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
                    # 多头检查
                    if row['low'] <= pos['stop_loss']:
                        should_close = True
                        close_reason = 'stop_loss'
                    elif row['high'] >= pos['take_profits']['tp1']:
                        should_close = True
                        close_reason = 'tp1'
                else:
                    # 空头检查
                    if row['high'] >= pos['stop_loss']:
                        should_close = True
                        close_reason = 'stop_loss'
                    elif row['low'] <= pos['take_profits']['tp1']:
                        should_close = True
                        close_reason = 'tp1'

                if should_close:
                    # 使用触发价平仓
                    if close_reason == 'stop_loss':
                        close_price = pos['stop_loss']
                    else:
                        close_price = pos['take_profits']['tp1']

                    self.close_position(timestamp, close_price, close_reason)

            # 如果无持仓，检查开仓
            if self.current_position is None and row['position'] != 0:
                direction = 'long' if row['position'] > 0 else 'short'

                # 从策略中获取止损止盈
                stop_loss = row['stop_loss']
                tp1 = row['take_profit']

                # 简化：只记录TP1
                take_profits = {'tp1': tp1, 'tp2': tp1, 'tp3': tp1}

                # 计算信号质量评分（如果策略提供了ob_id和ema20）
                quality_score = None
                if 'signal_ob_id' in row and pd.notna(row['signal_ob_id']):
                    ob_id = row['signal_ob_id']
                    # 从策略的active_obs中查找对应的OB
                    ob = next((o for o in self.strategy.active_obs if o.id == ob_id), None)
                    if ob is not None:
                        ema20 = row.get('signal_ema20', np.nan)
                        quality_score = self.calculate_signal_quality(
                            df_sig, i, ob, ema20
                        )

                self.open_position(
                    timestamp=timestamp,
                    price=row['close'],
                    direction=direction,
                    stop_loss=stop_loss,
                    take_profits=take_profits,
                    quality_score=quality_score
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

def calculate_realistic_metrics(engine: RealisticBacktestEngine, df: pd.DataFrame, freq='60min'):
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
        'long_trades': 0,
        'short_trades': 0
    }


# ========================================================================
# 生成详细报表
# ========================================================================

def generate_detailed_report(engine: RealisticBacktestEngine, metrics: dict,
                            year: int, freq: str) -> str:
    """生成详细报表"""
    report = []
    report.append("=" * 120)
    report.append(f"真实交易回测报告 - {year}年 {freq}周期")
    report.append("=" * 120)

    # 配置信息
    report.append("\n【交易配置】")
    report.append("-" * 120)
    report.append(f"  初始资金: {engine.config.INITIAL_CAPITAL:,.2f}元")
    report.append(f"  单次开仓金额: {engine.config.BASE_POSITION_VALUE:,.2f}元")
    report.append(f"  合约乘数: {engine.config.CONTRACT_SIZE}")
    report.append(f"  保证金比例: {engine.config.MARGIN_RATE:.1%}")
    report.append(f"  手续费率: {engine.config.COMMISSION_RATE:.4%} (万一)")
    report.append(f"  滑点: {engine.config.SLIPPAGE_TICKS} tick/t")
    report.append(f"  最大保证金占用: {engine.config.MAX_MARGIN_RATIO:.1%}")

    # 账户概况
    report.append("\n【账户概况】")
    report.append("-" * 120)
    report.append(f"  初始资金: {metrics['initial_capital']:>12,.2f}元")
    report.append(f"  最终净值: {metrics['final_net_worth']:>12,.2f}元")
    report.append(f"  净利润: {metrics['total_pnl']:>12,.2f}元")
    report.append(f"  总收益率: {metrics['total_return']:>10.2%}")
    report.append(f"  总手续费: {metrics['total_commission']:>12,.2f}元  (占比: {metrics['commission_ratio']:.2%})")

    # 交易统计
    report.append("\n【交易统计】")
    report.append("-" * 120)
    report.append(f"  总交易次数: {metrics['total_trades']}")
    report.append(f"  多头交易: {metrics['long_trades']}")
    report.append(f"  空头交易: {metrics['short_trades']}")
    report.append(f"  胜率: {metrics['win_rate']:.2f}%")
    report.append(f"  TP1止盈: {metrics['tp1_exits']}笔")
    report.append(f"  止损出场: {metrics['sl_exits']}笔")

    # 盈亏分析
    report.append("\n【盈亏分析】")
    report.append("-" * 120)
    report.append(f"  平均盈利: {metrics['avg_win']:>10,.2f}元")
    report.append(f"  平均亏损: {metrics['avg_loss']:>10,.2f}元")
    report.append(f"  盈亏比: {metrics['profit_loss_ratio']:.2f}")
    report.append(f"  盈利因子: {metrics['profit_factor']:.2f}")
    report.append(f"  平均每笔收益: {metrics['avg_trade_pnl']:>10,.2f}元  ({metrics['avg_trade_pnl_ratio']:.2%})")
    report.append(f"  最大单笔盈利: {metrics['max_single_win']:>10,.2f}元")
    report.append(f"  最大单笔亏损: {metrics['max_single_loss']:>10,.2f}元")

    # 风险指标
    report.append("\n【风险指标】")
    report.append("-" * 120)
    report.append(f"  最大回撤: {metrics['max_drawdown']:.2%}")
    report.append(f"  夏普比率: {metrics['sharpe_ratio']:.2f}")

    # 详细交易记录
    if engine.trades:
        report.append("\n【交易明细】")
        report.append("-" * 120)
        report.append(f"{'序号':<4} {'方向':<6} {'开仓时间':<19} {'平仓时间':<19} "
                     f"{'开仓价':<8} {'平仓价':<8} {'手数':<4} {'净利润':>10} {'收益率':>8} {'原因':<10}")
        report.append("-" * 120)

        for i, trade in enumerate(engine.trades, 1):
            direction_symbol = "多" if trade.direction == 'long' else "空"
            pnl_str = f"{trade.pnl:>8,.2f}"
            pnl_color = "+" if trade.pnl > 0 else ""
            pnl_str = f"{pnl_color}{trade.pnl:,.2f}"

            report.append(
                f"{i:<4} {direction_symbol:<6} "
                f"{trade.entry_time.strftime('%Y-%m-%d %H:%M'):<19} "
                f"{trade.exit_time.strftime('%Y-%m-%d %H:%M'):<19} "
                f"{trade.entry_price:<8.2f} {trade.exit_price:<8.2f} "
                f"{trade.quantity:<4} {pnl_str:>10} "
                f"{trade.pnl_ratio:>7.2%} {trade.exit_reason:<10}"
            )

    report.append("\n" + "=" * 120)

    return "\n".join(report)


def save_trades_to_csv(engine: RealisticBacktestEngine, output_path: Path):
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
            'quality_score': t.quality_score
        })

    df_trades = pd.DataFrame(trades_data)
    df_trades.to_csv(output_path, index=False, encoding='utf-8-sig')


# ========================================================================
# 主程序
# ========================================================================

def run_realistic_backtest(year: int, freq: str) -> Tuple[dict, str]:
    """运行真实回测"""
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
    engine = RealisticBacktestEngine(strategy, TradingConfig())

    # 运行回测
    df_result = engine.run_backtest(df)

    # 计算指标
    metrics = calculate_realistic_metrics(engine, df_result, freq=freq)
    metrics['year'] = year
    metrics['freq'] = freq

    # 生成报表
    report = generate_detailed_report(engine, metrics, year, freq)

    # 保存交易记录
    output_dir = Path(__file__).parent / "realistic_results"
    output_dir.mkdir(parents=True, exist_ok=True)

    trades_file = output_dir / f"trades_{year}_{freq}.csv"
    save_trades_to_csv(engine, trades_file)

    # 保存账户快照
    snapshots_file = output_dir / f"snapshots_{year}_{freq}.csv"
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


def main():
    """主函数"""
    import sys
    import io
    # 设置UTF-8编码输出
    if sys.platform == 'win32':
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

    print("=" * 120)
    print("真实交易回测系统 - V03混合策略")
    print("=" * 120)
    print("\n交易配置:")
    print(f"  初始资金: {TradingConfig.INITIAL_CAPITAL:,.2f}元")
    print(f"  单次开仓: {TradingConfig.BASE_POSITION_VALUE:,.2f}元")
    print(f"  保证金: {TradingConfig.MARGIN_RATE:.1%}")
    print(f"  手续费: {TradingConfig.COMMISSION_RATE:.4%} (万一)")
    print(f"  滑点: {TradingConfig.SLIPPAGE_TICKS} tick")

    years = [2023, 2024, 2025]
    freqs = ['60min', '15min', '5min']

    all_metrics = []
    all_reports = []

    for freq in freqs:
        print(f"\n{'='*120}")
        print(f"周期: {freq}")
        print(f"{'='*120}")

        freq_metrics = []

        for year in years:
            try:
                metrics, report = run_realistic_backtest(year, freq)
                freq_metrics.append(metrics)
                all_metrics.append(metrics)
                all_reports.append(report)

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

        # 周期汇总
        if freq_metrics:
            print(f"\n{'-'*120}")
            print(f"{freq}周期汇总:")
            print("-" * 120)

            total_trades = sum(m['total_trades'] for m in freq_metrics)
            total_pnl = sum(m['total_pnl'] for m in freq_metrics)
            total_commission = sum(m['total_commission'] for m in freq_metrics)

            weighted_win_rate = sum(m['win_rate'] * m['total_trades'] for m in freq_metrics if m['total_trades'] > 0)
            weighted_win_rate = weighted_win_rate / total_trades if total_trades > 0 else 0

            total_return = sum(m['total_return'] for m in freq_metrics)
            max_dd = max(m['max_drawdown'] for m in freq_metrics)
            avg_sharpe = np.mean([m['sharpe_ratio'] for m in freq_metrics])

            print(f"  总交易: {total_trades}笔")
            print(f"  总净利润: {total_pnl:,.2f}元")
            print(f"  总手续费: {total_commission:,.2f}元")
            print(f"  加权胜率: {weighted_win_rate:.2f}%")
            print(f"  总收益率: {total_return:.2%}")
            print(f"  最大回撤: {max_dd:.2%}")
            print(f"  平均夏普: {avg_sharpe:.2f}")

    # 保存汇总结果
    if all_metrics:
        output_dir = Path(__file__).parent / "realistic_results"
        output_dir.mkdir(parents=True, exist_ok=True)

        df_metrics = pd.DataFrame(all_metrics)
        summary_file = output_dir / "summary.csv"
        df_metrics.to_csv(summary_file, index=False, encoding='utf-8-sig')

        # 保存详细报告
        report_file = output_dir / "detailed_report.txt"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(all_reports))

        print(f"\n{'='*120}")
        print(f"汇总结果已保存: {summary_file}")
        print(f"详细报告已保存: {report_file}")
        print(f"{'='*120}")

    # 打印最终对比表
    if all_metrics:
        print(f"\n{'='*120}")
        print("最终对比表")
        print(f"{'='*120}\n")

        print(f"{'年份':<8} {'周期':<10} {'交易':<6} {'胜率':<8} {'净利润':>12} {'收益率':<10} "
             f"{'回撤':<10} {'夏普':<8} {'盈亏比':<8}")
        print("-" * 110)

        for m in all_metrics:
            print(f"{m['year']:<8} {m['freq']:<10} {m['total_trades']:<6} "
                 f"{m['win_rate']:>6.2f}% "
                 f"{m['total_pnl']:>10,.2f}元"
                 f"{m['total_return']:>8.2%}   "
                 f"{m['max_drawdown']:>8.2%}   "
                 f"{m['sharpe_ratio']:>6.2f}   "
                 f"{m['profit_loss_ratio']:>6.2f}")

    print(f"\n{'='*120}")


if __name__ == "__main__":
    main()
