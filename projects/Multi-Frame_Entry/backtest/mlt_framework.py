#!/usr/bin/env python3
"""
多层时间框架回测框架 (Multi-Timeframe Backtesting Framework)

架构：
- 大级别（60min）：ML模型预测趋势概率，判断交易环境
- 小级别（5min）：MSB+OB策略寻找精确入场点
- 出场管理：4层动态止损系统

作者：Claude Code
日期：2026-02-21
"""

import pandas as pd
import numpy as np
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass, field
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# ========================================================================
# 数据结构定义
# ========================================================================

@dataclass
class Position:
    """持仓数据结构"""
    id: str
    pos_type: str  # 'long' or 'short'
    entry_price: float
    entry_timestamp: pd.Timestamp
    entry_bar_index: int
    stop_loss: float
    initial_stop: float
    take_profits: Dict[str, float] = field(default_factory=dict)
    position_size: float = 1.0
    pnl: float = 0.0
    pnl_r: float = 0.0  # 盈亏（以R为单位）
    exit_price: Optional[float] = None
    exit_timestamp: Optional[pd.Timestamp] = None
    exit_reason: Optional[str] = None  # 'initial_stop', 'break_even', 'trailing', 'structure_break'
    exit_bar_index: Optional[int] = None
    is_active: bool = True
    break_even_triggered: bool = False
    hold_bars: int = 0

    # 出场配置
    trailing_method: str = 'ema'  # 'ema', 'structure', 'chandelier'
    ema_period: int = 20
    atr_mult: float = 0.5
    break_even_r: float = 1.0
    min_hold_bars: int = 5


@dataclass
class Trade:
    """交易记录数据结构"""
    id: str
    pos_type: str
    entry_timestamp: pd.Timestamp
    entry_price: float
    exit_timestamp: pd.Timestamp
    exit_price: float
    exit_reason: str
    pnl: float
    pnl_r: float
    hold_bars: int
    ml_model_id: str  # 使用的ML模型窗口ID
    msb_ob_id: Optional[str] = None  # 关联的MSB+OB信号ID


@dataclass
class DailyState:
    """每日状态记录"""
    date: pd.Timestamp
    equity: float
    n_positions: int
    exposure: float  # 市场暴露度（持仓占用资金比例）
    drawdown: float
    trading_mode: str  # 'long', 'short', 'wait'
    trend_proba: float
    regime: str


# ========================================================================
# 主回测框架类
# ========================================================================

class MultiTimeframeBacktest:
    """
    多层时间框架回测系统

    功能：
    1. 大级别（60min）：ML模型预测趋势
    2. 小级别（5min）：MSB+OB入场
    3. 4层出场管理
    """

    def __init__(
        self,
        symbol: str = 'AU8888.XSGE',
        start_date: str = '2021-01-01',
        end_date: str = '2025-12-31',
        initial_capital: float = 100000.0,
        entry_threshold: float = 0.6,
        max_positions: int = 3,
        min_hold_bars: int = 5,
        commission: float = 0.0001,  # 万分之一
        slippage_ticks: int = 1,
        tick_size: float = 0.01
    ):
        """
        初始化回测框架

        Args:
            symbol: 品种代码
            start_date: 回测开始日期
            end_date: 回测结束日期
            initial_capital: 初始资金
            entry_threshold: 入场概率阈值
            max_positions: 最大持仓数
            min_hold_bars: 最小持仓K线数
            commission: 手续费率
            slippage_ticks: 滑点（ticks）
            tick_size: 最小变动价位
        """
        self.symbol = symbol
        self.start_date = pd.Timestamp(start_date)
        self.end_date = pd.Timestamp(end_date)
        self.initial_capital = initial_capital
        self.entry_threshold = entry_threshold
        self.max_positions = max_positions
        self.min_hold_bars = min_hold_bars
        self.commission = commission
        self.slippage_ticks = slippage_ticks
        self.tick_size = tick_size

        # 状态变量
        self.current_capital = initial_capital
        self.active_positions: List[Position] = []
        self.closed_positions: List[Position] = []
        self.trades: List[Trade] = []
        self.daily_states: List[DailyState] = []

        # 数据缓存
        self.df_60min = None
        self.df_5min = None
        self.ml_models = {}  # {window_id: model_dict}

        logger.info("="*80)
        logger.info("多层时间框架回测系统初始化")
        logger.info("="*80)
        logger.info(f"品种: {symbol}")
        logger.info(f"回测期间: {start_date} ~ {end_date}")
        logger.info(f"初始资金: {initial_capital:,.0f}")
        logger.info(f"入场阈值: {entry_threshold}")
        logger.info(f"最小持仓: {min_hold_bars}根K线")
        logger.info("="*80)

    def load_data(self) -> bool:
        """
        加载回测所需数据

        Returns:
            是否加载成功
        """
        logger.info("\n加载回测数据...")

        project_root = Path(__file__).parent.parent
        qlib_data_dir = project_root / 'data' / 'multi_symbol' / self.symbol / 'qlib_data' / 'instruments'

        # 使用qlib加载数据
        try:
            import qlib
            from qlib.data import D
        except ImportError:
            logger.error("Qlib未安装，请先安装: pip install pyqlib")
            return False

        # 初始化qlib
        qlib.init(provider_uri='local')

        # 加载5min和60min数据
        try:
            self.df_5min = self.load_qlib_data(qlib_data_dir, '5min')
            self.df_60min = self.load_qlib_data(qlib_data_dir, '60min')
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            import traceback
            traceback.print_exc()
            return False

        # 筛选时间范围
        self.df_5min = self.df_5min.loc[self.start_date:self.end_date]
        self.df_60min = self.df_60min.loc[self.start_date:self.end_date]

        logger.info(f"  ✓ 60min数据: {len(self.df_60min)} 根K线")
        logger.info(f"  ✓ 5min数据:  {len(self.df_5min)} 根K线")

        # 2. 加载ML模型
        model_dir = project_root / 'models' / 'rolling_3month'
        if not model_dir.exists():
            logger.error(f"模型目录不存在: {model_dir}")
            return False

        self.ml_models = self.load_rolling_models(model_dir)
        logger.info(f"  ✓ 加载了 {len(self.ml_models)} 个ML模型")

        return True

    def load_qlib_data(self, qlib_data_dir: Path, freq: str) -> pd.DataFrame:
        """
        从qlib格式加载数据

        Args:
            qlib_data_dir: qlib数据目录
            freq: 频率 ('5min', '60min')

        Returns:
            DataFrame
        """
        from qlib.constant import REG_CN

        # 读取qlib数据
        df = D.features(
            instruments=[self.symbol],
            fields=['open', 'high', 'low', 'close', 'volume', 'money'],
            start_time=self.start_date,
            end_time=self.end_date,
            freq=freq,
            adjust_price=False
        )

        return df

    def resample_ohlcv(self, df: pd.DataFrame, freq: str) -> pd.DataFrame:
        """
        重采样OHLC数据

        Args:
            df: 原始DataFrame（1min）
            freq: 目标频率 ('5min', '60min')

        Returns:
            重采样后的DataFrame
        """
        df = df.copy()

        # 确保索引是datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # OHLC重采样规则
        df_resampled = df.resample(freq).agg({
            'open': 'first',
            'high': 'max',
            'low': 'min',
            'close': 'last',
            'volume': 'sum',
            'money': 'sum'
        }).dropna()

        return df_resampled

    def load_rolling_models(self, model_dir: Path) -> Dict:
        """
        加载所有季度滚动模型

        Args:
            model_dir: 模型目录

        Returns:
            {window_id: model_dict}
        """
        models = {}

        for model_file in sorted(model_dir.glob(f'{self.symbol}_window*.pkl')):
            with open(model_file, 'rb') as f:
                model_data = pickle.load(f)

            window_id = model_data['window_id']
            models[window_id] = model_data

        return models

    def get_ml_prediction(self, timestamp: pd.Timestamp) -> Tuple[float, str]:
        """
        获取ML模型的预测（趋势概率和Regime）

        Args:
            timestamp: 当前时间

        Returns:
            (trend_proba, regime)
        """
        # 找到对应的模型窗口
        target_window = None
        for window_id, model_data in sorted(self.ml_models.items()):
            test_start = pd.Timestamp(model_data['test_start'])
            test_end = pd.Timestamp(model_data['test_end'])

            if test_start <= timestamp <= test_end:
                target_window = window_id
                break

        if target_window is None:
            return 0.5, 'unknown'  # 默认返回中性值

        # 使用模型预测
        model_data = self.ml_models[target_window]
        model = model_data['model']
        features = model_data['features']

        # 获取当前特征（简化处理，实际需要从数据中提取）
        # 这里返回模型的历史平均概率作为示例
        metrics = model_data.get('metrics', {})
        trend_proba = metrics.get('auc', 0.5)  # 简化：使用AUC作为概率估计

        # Regime判断（基于波动率）
        regime = 'high_volatility'  # 简化处理

        return trend_proba, regime

    def determine_trading_mode(
        self,
        trend_proba: float,
        regime: str
    ) -> str:
        """
        确定交易模式

        Args:
            trend_proba: 趋势概率
            regime: Regime类型

        Returns:
            'long', 'short', or 'wait'
        """
        if regime != 'high_volatility':
            return 'wait'

        if trend_proba > self.entry_threshold:
            return 'long'
        elif trend_proba < (1 - self.entry_threshold):
            return 'short'
        else:
            return 'wait'

    def run_backtest(self) -> Dict:
        """
        执行回测

        Returns:
            回测结果统计
        """
        logger.info("\n" + "="*80)
        logger.info("开始执行回测")
        logger.info("="*80)

        start_time = datetime.now()

        # 按日期遍历
        current_date = self.start_date

        while current_date <= self.end_date:
            # 获取当天的5min数据
            day_start = current_date
            day_end = current_date + timedelta(days=1) - timedelta(minutes=5)

            df_day = self.df_5min.loc[day_start:day_end]

            if len(df_day) == 0:
                current_date += timedelta(days=1)
                continue

            # 获取60min的ML预测（当天开盘时）
            trend_proba, regime = self.get_ml_prediction(current_date)

            # 确定交易模式
            trading_mode = self.determine_trading_mode(trend_proba, regime)

            # 记录每日状态
            daily_state = DailyState(
                date=current_date,
                equity=self.current_capital,
                n_positions=len(self.active_positions),
                exposure=0.0,  # TODO: 计算实际暴露度
                drawdown=0.0,  # TODO: 计算回撤
                trading_mode=trading_mode,
                trend_proba=trend_proba,
                regime=regime
            )

            # 管理现有持仓
            self.manage_positions(df_day)

            # 尝试入场（如果不在观望模式）
            if trading_mode != 'wait':
                self.try_entry(df_day, trading_mode)

            # 更新每日状态
            self.daily_states.append(daily_state)

            current_date += timedelta(days=1)

            # 进度显示
            if current_date.day == 1 or current_date == self.start_date:
                logger.info(f"  {current_date.strftime('%Y-%m-%d')}: "
                           f"权益={self.current_capital:,.0f}, "
                           f"持仓={len(self.active_positions)}, "
                           f"模式={trading_mode}")

        elapsed = (datetime.now() - start_time).total_seconds()

        logger.info("\n" + "="*80)
        logger.info("回测完成")
        logger.info("="*80)
        logger.info(f"耗时: {elapsed:.1f}秒")
        logger.info(f"总交易数: {len(self.trades)}")
        logger.info(f"最终权益: {self.current_capital:,.0f}")
        logger.info(f"总收益率: {(self.current_capital / self.initial_capital - 1) * 100:.2f}%")

        return self.calculate_statistics()

    def manage_positions(self, df_day: pd.DataFrame):
        """
        管理持仓（检查出场）

        Args:
            df_day: 当天的5min数据
        """
        positions_to_close = []

        for position in self.active_positions:
            position.hold_bars += 1

            for idx, row in df_day.iterrows():
                exit_signal = self.check_exit(position, row)

                if exit_signal:
                    # 平仓
                    self.close_position(position, row, exit_signal)
                    positions_to_close.append(position)
                    break

        # 移除已平仓的持仓
        for pos in positions_to_close:
            if pos in self.active_positions:
                self.active_positions.remove(pos)
            self.closed_positions.append(pos)

    def check_exit(self, position: Position, bar: pd.Series) -> Optional[str]:
        """
        检查是否触发出场

        Args:
            position: 持仓对象
            bar: 当前K线数据

        Returns:
            出场原因或None
        """
        # 获取当前价格
        current_price = bar['close']

        # 层1：初始止损（绝对保护）
        if position.pos_type == 'long':
            if current_price <= position.stop_loss:
                return 'initial_stop'
        else:
            if current_price >= position.stop_loss:
                return 'initial_stop'

        # 层2：保本机制（盈利≥1R后）
        if not position.break_even_triggered:
            pnl_r = self.calculate_pnl_r(position, current_price)
            if pnl_r >= position.break_even_r:
                position.stop_loss = position.entry_price
                position.break_even_triggered = True

        # 层3：趋势追踪止损（保本后才启用）
        if position.break_even_triggered:
            trailing_stop = self.calculate_trailing_stop(position, bar)
            if position.pos_type == 'long':
                if trailing_stop > position.stop_loss:
                    position.stop_loss = trailing_stop
                if current_price <= position.stop_loss:
                    return 'trailing_stop'
            else:
                if trailing_stop < position.stop_loss:
                    position.stop_loss = trailing_stop
                if current_price >= position.stop_loss:
                    return 'trailing_stop'

        # 层4：结构破坏（最小持仓时间后）
        if position.hold_bars >= position.min_hold_bars:
            # 简化版：如果盈利回撤超过一定比例
            if self.check_structure_break(position, bar):
                return 'structure_break'

        return None

    def calculate_pnl_r(self, position: Position, current_price: float) -> float:
        """
        计算盈亏（以R为单位）

        Args:
            position: 持仓对象
            current_price: 当前价格

        Returns:
            盈亏（R倍数）
        """
        if position.pos_type == 'long':
            pnl = current_price - position.entry_price
        else:
            pnl = position.entry_price - current_price

        initial_r = abs(position.entry_price - position.initial_stop)
        return pnl / initial_r if initial_r > 0 else 0

    def calculate_trailing_stop(self, position: Position, bar: pd.Series) -> float:
        """
        计算追踪止损价格

        Args:
            position: 持仓对象
            bar: 当前K线数据

        Returns:
            新的止损价格
        """
        # 简化版：使用固定百分位追踪
        if position.pos_type == 'long':
            # 多单：当前价的95%
            return bar['close'] * 0.95
        else:
            # 空单：当前价的105%
            return bar['close'] * 1.05

    def check_structure_break(self, position: Position, bar: pd.Series) -> bool:
        """
        检查结构破坏

        Args:
            position: 持仓对象
            bar: 当前K线数据

        Returns:
            是否触发结构破坏
        """
        # 简化版：如果盈利回撤超过50%
        current_pnl_r = self.calculate_pnl_r(position, bar['close'])

        if current_pnl_r >= 1.0:
            # 曾经盈利1R以上，现在回撤到0.5R以下
            if current_pnl_r < 0.5:
                return True

        return False

    def close_position(self, position: Position, bar: pd.Series, reason: str):
        """
        平仓

        Args:
            position: 持仓对象
            bar: 当前K线数据
            reason: 出场原因
        """
        # 计算出场价格（考虑滑点）
        if position.pos_type == 'long':
            exit_price_raw = bar['close']
            exit_price = min(bar['open'], bar['low']) - self.slippage_ticks * self.tick_size
        else:
            exit_price_raw = bar['close']
            exit_price = max(bar['open'], bar['high']) + self.slippage_ticks * self.tick_size

        # 计算盈亏
        if position.pos_type == 'long':
            pnl = exit_price - position.entry_price
        else:
            pnl = position.entry_price - exit_price

        # 扣除手续费
        commission = exit_price * self.commission
        pnl -= commission

        # 更新持仓
        position.exit_price = exit_price
        position.exit_timestamp = bar.name
        position.exit_reason = reason
        position.pnl = pnl
        position.is_active = False

        # 更新资金
        self.current_capital += pnl * position.position_size

        # 记录交易
        trade = Trade(
            id=str(uuid.uuid4()),
            pos_type=position.pos_type,
            entry_timestamp=position.entry_timestamp,
            entry_price=position.entry_price,
            exit_timestamp=position.exit_timestamp,
            exit_price=position.exit_price,
            exit_reason=position.exit_reason,
            pnl=position.pnl,
            pnl_r=self.calculate_pnl_r(position, exit_price),
            hold_bars=position.hold_bars,
            ml_model_id='unknown',  # TODO: 记录实际使用的模型
            msb_ob_id=None
        )

        self.trades.append(trade)

        logger.debug(f"平仓: {position.pos_type}, "
                    f"原因={reason}, "
                    f"盈亏={pnl:.2f}, "
                    f"持有={position.hold_bars}根K线")

    def try_entry(self, df_day: pd.DataFrame, trading_mode: str):
        """
        尝试入场

        Args:
            df_day: 当天的5min数据
            trading_mode: 交易模式（'long' or 'short'）
        """
        # 检查是否可以开仓
        if len(self.active_positions) >= self.max_positions:
            return

        # TODO: 实现MSB+OB入场逻辑
        # 这里使用简化版：随机入场作为占位符
        if len(df_day) > 100 and np.random.random() < 0.05:  # 5%概率
            # 随机选择一个入场点
            entry_idx = np.random.randint(0, len(df_day))
            entry_bar = df_day.iloc[entry_idx]

            if trading_mode == 'long':
                pos_type = 'long'
                entry_price = entry_bar['close']
                stop_loss = entry_price * 0.97  # 3%止损
            else:
                pos_type = 'short'
                entry_price = entry_bar['close']
                stop_loss = entry_price * 1.03

            # 创建持仓
            position = Position(
                id=str(uuid.uuid4()),
                pos_type=pos_type,
                entry_price=entry_price,
                entry_timestamp=entry_bar.name,
                entry_bar_index=entry_idx,
                stop_loss=stop_loss,
                initial_stop=stop_loss,
                min_hold_bars=self.min_hold_bars
            )

            self.active_positions.append(position)

            logger.debug(f"开仓: {pos_type}, "
                        f"价格={entry_price:.2f}, "
                        f"止损={stop_loss:.2f}")

    def calculate_statistics(self) -> Dict:
        """
        计算回测统计数据

        Returns:
            统计字典
        """
        if len(self.trades) == 0:
            return {
                'total_trades': 0,
                'total_return': 0.0,
                'annual_return': 0.0,
                'sharpe_ratio': 0.0,
                'max_drawdown': 0.0,
                'win_rate': 0.0,
                'avg_pnl': 0.0,
                'final_capital': self.current_capital
            }

        # 计算基本统计
        total_trades = len(self.trades)
        winning_trades = sum(1 for t in self.trades if t.pnl > 0)
        losing_trades = total_trades - winning_trades

        win_rate = winning_trades / total_trades if total_trades > 0 else 0

        total_pnl = sum(t.pnl for t in self.trades)
        avg_pnl = total_pnl / total_trades if total_trades > 0 else 0

        total_return = (self.current_capital / self.initial_capital - 1) * 100
        years = (self.end_date - self.start_date).days / 365.25
        annual_return = ((self.current_capital / self.initial_capital) ** (1/years) - 1) * 100

        # 计算最大回撤
        equity_curve = [state.equity for state in self.daily_states]
        if len(equity_curve) > 0:
            max_equity = np.maximum.accumulate(equity_curve)
            drawdown = (max_equity - equity_curve) / max_equity
            max_drawdown = np.max(drawdown) * 100
        else:
            max_drawdown = 0.0

        # 计算夏普比率（简化版）
        if len(equity_curve) > 1:
            returns = pd.Series(equity_curve).pct_change().dropna()
            sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252) if returns.std() > 0 else 0
        else:
            sharpe_ratio = 0.0

        stats = {
            'total_trades': total_trades,
            'winning_trades': winning_trades,
            'losing_trades': losing_trades,
            'win_rate': win_rate * 100,
            'total_return': total_return,
            'annual_return': annual_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'avg_pnl': avg_pnl,
            'final_capital': self.current_capital
        }

        return stats

    def generate_report(self, stats: Dict):
        """
        生成回测报告

        Args:
            stats: 统计字典
        """
        logger.info("\n" + "="*80)
        logger.info("回测报告")
        logger.info("="*80)

        logger.info(f"\n交易统计:")
        logger.info(f"  总交易数: {stats['total_trades']}")
        logger.info(f"  盈利交易: {stats['winning_trades']}")
        logger.info(f"  亏损交易: {stats['losing_trades']}")
        logger.info(f"  胜率: {stats['win_rate']:.2f}%")

        logger.info(f"\n收益统计:")
        logger.info(f"  总收益率: {stats['total_return']:.2f}%")
        logger.info(f"  年化收益: {stats['annual_return']:.2f}%")
        logger.info(f"  平均盈亏: {stats['avg_pnl']:.2f}")

        logger.info(f"\n风险指标:")
        logger.info(f"  最大回撤: {stats['max_drawdown']:.2f}%")
        logger.info(f"  夏普比率: {stats['sharpe_ratio']:.2f}")

        logger.info(f"\n资金变化:")
        logger.info(f"  初始资金: {self.initial_capital:,.0f}")
        logger.info(f"  最终资金: {stats['final_capital']:,.0f}")

        logger.info("="*80)


# ========================================================================
# 便捷函数
# ========================================================================

def run_multi_timeframe_backtest(
    symbol: str = 'AU8888.XSGE',
    start_date: str = '2021-01-01',
    end_date: str = '2025-12-31'
) -> Dict:
    """
    运行多层时间框架回测

    Args:
        symbol: 品种代码
        start_date: 开始日期
        end_date: 结束日期

    Returns:
        回测统计
    """
    backtest = MultiTimeframeBacktest(
        symbol=symbol,
        start_date=start_date,
        end_date=end_date
    )

    # 加载数据
    if not backtest.load_data():
        logger.error("数据加载失败")
        return {}

    # 执行回测
    stats = backtest.run_backtest()

    # 生成报告
    backtest.generate_report(stats)

    return stats


if __name__ == '__main__':
    # 运行回测
    stats = run_multi_timeframe_backtest()
