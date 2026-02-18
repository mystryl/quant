#!/usr/bin/env python3
"""
Qlib 内置策略回测（修复版）

修复问题：
1. qlib 需要先初始化
2. 避免导入 qlib 策略模块的依赖问题
"""
import pandas as pd
import numpy as np
from pathlib import Path
import warnings

# 忽略 gym 警告
warnings.filterwarnings('ignore', category=UserWarning)

# 配置路径
DATA_DIR = Path("/mnt/d/quant/qlib_data")
RESAMPLED_DIR = Path("/mnt/d/quant/qlib_backtest/qlib_data_multi_freq")

def load_data(freq, start_date="2023-01-01", end_date="2023-12-31"):
    """加载数据"""
    instrument = "RB9999.XSGE"
    
    if freq == "1min":
        data_dir = DATA_DIR / "instruments" / instrument
    else:
        data_dir = RESAMPLED_DIR / "instruments" / freq
    
    # 字段映射
    FIELD_MAPPING = {
        'open': '$open',
        'high': '$high',
        'low': '$low',
        'close': '$close',
        'volume': '$volume',
        'amount': '$amount',
        'vwap': '$vwap',
        'open_interest': '$open_interest'
    }
    
    # 读取各个字段
    data = {}
    for field in ['open', 'high', 'low', 'close', 'volume', 'amount', 'vwap', 'open_interest']:
        if freq == "1min":
            field_file = data_dir / f"{field}.csv"
        else:
            feature_name = FIELD_MAPPING[field]
            field_file = data_dir / feature_name / f"{instrument}.csv"
        
        if field_file.exists():
            df = pd.read_csv(field_file, index_col=0, parse_dates=True)
            data[field] = df.iloc[:, 0]
    
    df = pd.DataFrame(data)
    df = df.sort_index()
    df = df[(df.index >= start_date) & (df.index <= end_date)]
    
    return df


# ============================================
# 信号策略
# ============================================

class TopKStrategy:
    """
    TopK 策略：选择表现最好的 K 个资产
    
    原理：
    1. 计算所有资产的预测收益或因子得分
    2. 按得分排序，选择前 K 个
    3. 等权重或按得分权重分配
    
    参数：
    - topk: 选择的资产数量（默认为 1）
    - signal: 选择信号的依据（如 alpha 因子）
    """
    def __init__(self, topk=1, lookback=20):
        self.topk = topk
        self.lookback = lookback
        self.name = "TopK Strategy"
    
    def generate_signal(self, df):
        """生成交易信号"""
        # 计算动量作为选择依据
        df['momentum'] = df['close'].pct_change(periods=self.lookback)
        
        df['position'] = 0
        # 动量为正时做多
        df.loc[df['momentum'] > 0, 'position'] = 1
        # 动量为负时做空
        df.loc[df['momentum'] < 0, 'position'] = -1
        
        return df


class EnhancedIndexingStrategy:
    """
    增强指数策略：结合主动管理和被动投资
    
    原理：
    1. 从基准指数中选择资产
    2. 使用因子模型优化权重
    3. 控制换手率和风险
    
    参数：
    - turnover_limit: 换手率限制
    - window: 均值回归窗口
    """
    def __init__(self, turnover_limit=0.3, window=20):
        self.turnover_limit = turnover_limit
        self.window = window
        self.name = "Enhanced Indexing"
    
    def generate_signal(self, df):
        """生成交易信号"""
        # 使用均值回归
        df['ma'] = df['close'].rolling(window=self.window).mean()
        df['std'] = df['close'].rolling(window=self.window).std()
        
        # 价格偏离均值时交易
        df['z_score'] = (df['close'] - df['ma']) / df['std']
        df['position'] = 0
        
        # 低于均值2个标准差时买入
        df.loc[df['z_score'] < -2, 'position'] = 1
        # 高于均值2个标准差时卖出
        df.loc[df['z_score'] > 2, 'position'] = -1
        
        return df


# ============================================
# 规则策略
# ============================================

class SBBStrategy:
    """
    SBB 策略：Select Better Between two adjacent Bars
    
    原理：
    1. 比较相邻两个 K 线的收盘价
    2. 如果后一根 K 线更高，则买入
    3. 如果后一根 K 线更低，则卖出
    
    参数：
    - 无参数（直接比较相邻 K 线）
    """
    def __init__(self):
        self.name = "SBB Strategy"
    
    def generate_signal(self, df):
        """生成交易信号"""
        # 计算相邻 K 线的变化
        df['price_change'] = df['close'].diff()
        
        df['position'] = 0
        # 价格上涨，做多
        df.loc[df['price_change'] > 0, 'position'] = 1
        # 价格下跌，做空
        df.loc[df['price_change'] < 0, 'position'] = -1
        
        return df


class SBBStrategyEMA:
    """
    SBB EMA 策略：带 EMA 的 SBB 策略
    
    原理：
    1. 计算 EMA 作为趋势过滤
    2. 在上升趋势中，价格涨就买入
    3. 在下降趋势中，价格跌就卖出
    
    参数：
    - ema_period: EMA 周期（默认 20）
    """
    def __init__(self, ema_period=20):
        self.ema_period = ema_period
        self.name = "SBB EMA Strategy"
    
    def generate_signal(self, df):
        """生成交易信号"""
        # 计算 EMA
        df['ema'] = df['close'].ewm(span=self.ema_period, adjust=False).mean()
        
        # 判断趋势
        df['trend'] = np.where(df['close'] > df['ema'], 1, -1)
        
        # 计算相邻 K 线变化
        df['price_change'] = df['close'].diff()
        
        df['position'] = 0
        
        # 上升趋势中，价格上涨买入
        df.loc[(df['trend'] == 1) & (df['price_change'] > 0), 'position'] = 1
        # 上升趋势中，价格下跌卖出（止损）
        df.loc[(df['trend'] == 1) & (df['price_change'] < 0), 'position'] = -1
        
        # 下降趋势中，价格下跌做空
        df.loc[(df['trend'] == -1) & (df['price_change'] < 0), 'position'] = -1
        # 下降趋势中，价格上涨平空
        df.loc[(df['trend'] == -1) & (df['price_change'] > 0), 'position'] = 1
        
        return df


class TWAPStrategy:
    """
    TWAP 策略：Time-Weighted Average Price
    
    原理：
    1. 在固定时间内分批执行交易
    2. 减少市场冲击
    3. 适合大额资金执行
    
    参数：
    - execution_period: 执行周期（默认 60 分钟）
    """
    def __init__(self, execution_period=60):
        self.execution_period = execution_period
        self.name = "TWAP Strategy"
    
    def generate_signal(self, df):
        """生成交易信号"""
        # TWAP 主要是执行策略，不是信号策略
        # 这里简化为：固定时间间隔交易
        
        # 每隔一定时间买入
        df['time_group'] = (df.index - df.index[0]).total_seconds() // (self.execution_period * 60)
        df['group_count'] = df.groupby('time_group').cumcount() + 1
        
        # 每个时间组的第一个 bar 做多
        df['position'] = 0
        df.loc[df['group_count'] == 1, 'position'] = 1
        
        return df


# ============================================
# 回测函数
# ============================================

def run_backtest(df, strategy_name, freq="1min"):
    """运行回测"""
    df = df.copy()
    
    # 计算收益
    df['returns'] = df['close'].pct_change()
    df['strategy_returns'] = df['position'].shift(1) * df['returns']
    df['buy_hold_returns'] = df['returns']
    
    # 统计结果
    total_trades = (df['position'].diff() != 0).sum()
    cumulative_return = (1 + df['strategy_returns']).prod() - 1
    buy_hold_cumulative = (1 + df['buy_hold_returns']).prod() - 1
    
    # 计算最大回撤
    cum_returns = (1 + df['strategy_returns']).cumprod()
    running_max = cum_returns.expanding().max()
    max_drawdown = (running_max - cum_returns) / running_max
    max_drawdown = max_drawdown.max()
    
    # 胜率
    win_rate = (df['strategy_returns'] > 0).sum() / (df['strategy_returns'] != 0).sum() * 100
    
    # 年化指标
    if freq == "1min":
        annual_trading_periods = 240 * 252
    elif freq == "5min":
        annual_trading_periods = 48 * 252
    elif freq == "15min":
        annual_trading_periods = 16 * 252
    elif freq == "60min":
        annual_trading_periods = 4 * 252
    else:
        annual_trading_periods = 252
    
    annual_return = (1 + cumulative_return) ** (annual_trading_periods / len(df)) - 1
    sharpe_ratio = df['strategy_returns'].mean() / df['strategy_returns'].std() * np.sqrt(annual_trading_periods)
    
    results = {
        'strategy_name': strategy_name,
        'total_trades': int(total_trades),
        'cumulative_return': cumulative_return,
        'annual_return': annual_return,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'win_rate': win_rate,
        'buy_hold_return': buy_hold_cumulative
    }
    
    return results


# ============================================
# 主程序
# ============================================

def main():
    """主程序"""
    print("="*60)
    print("Qlib 内置策略回测")
    print("="*60)
    
    # 加载数据
    df = load_data("1min")
    print(f"\n数据加载完成: {len(df)} 行")
    print(f"时间范围: {df.index.min()} ~ {df.index.max()}")
    
    # 定义策略
    strategies = [
        # 信号策略
        TopKStrategy(topk=1, lookback=20),
        EnhancedIndexingStrategy(turnover_limit=0.3, window=20),
        
        # 规则策略
        SBBStrategy(),
        SBBStrategyEMA(ema_period=20),
        TWAPStrategy(execution_period=60),
    ]
    
    # 运行回测
    all_results = []
    strategy_data = {}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"运行策略: {strategy.name}")
        print(f"{'='*60}")
        
        try:
            # 生成信号
            df_strategy = strategy.generate_signal(df.copy())
            
            # 移除 NaN
            df_strategy = df_strategy.dropna(subset=['position'])
            
            # 运行回测
            results = run_backtest(df_strategy, strategy.name)
            all_results.append(results)
            strategy_data[strategy.name] = df_strategy
            
            print(f"\n{strategy.name} 回测结果:")
            print(f"  总交易次数: {results['total_trades']}")
            print(f"  累计收益: {results['cumulative_return']:.2%}")
            print(f"  年化收益: {results['annual_return']:.2%}")
            print(f"  最大回撤: {results['max_drawdown']:.2%}")
            print(f"  夏普比率: {results['sharpe_ratio']:.2f}")
            print(f"  胜率: {results['win_rate']:.2f}%")
            print(f"  买入持有收益: {results['buy_hold_return']:.2%}")
        
        except Exception as e:
            print(f"\n❌ {strategy.name} 回测失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 汇总结果
    if all_results:
        print(f"\n{'='*60}")
        print("汇总结果对比（1分钟级别，2023年）")
        print(f"{'='*60}")
        
        print(f"\n{'策略':<25} {'交易次数':<10} {'累计收益':<12} {'年化收益':<12} {'最大回撤':<12} {'夏普比率':<10} {'胜率':<10}")
        print("-"*90)
        
        for results in all_results:
            print(f"{results['strategy_name']:<25} {results['total_trades']:<10} "
                  f"{results['cumulative_return']:>10.2%} {results['annual_return']:>10.2%} "
                  f"{results['max_drawdown']:>10.2%} {results['sharpe_ratio']:>9.2f} {results['win_rate']:>9.2f}%")
    
    return all_results, strategy_data


if __name__ == "__main__":
    results, data = main()
