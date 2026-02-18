#!/usr/bin/env python3
"""
使用 Qlib 优化流程优化增强版 SuperTrend 策略

优化参数：
- period: ATR 周期
- multiplier: ATR 倍数
- n: 突破确认系数
- trailing_stop_rate: 止损幅度
- max_holding_period: 最大持仓周期
- min_liqka: 最小止损系数
- max_liqka: 最大止损系数
"""
import pandas as pd
import numpy as np
from pathlib import Path
import itertools
import time
from qlib_supertrend_enhanced import SupertrendEnhancedStrategy, run_backtest


def load_data_multi_freq(freq="1min", start_date="2023-01-01", end_date="2023-12-31"):
    """
    加载 qlib 数据（支持多频率）
    """
    DATA_DIR = Path("/mnt/d/quant/qlib_data")
    RESAMPLED_DIR = Path("/mnt/d/quant/qlib_backtest/qlib_data_multi_freq")

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

    instrument = "RB9999.XSGE"

    if freq == "1min":
        data_dir = DATA_DIR / "instruments" / instrument
    else:
        data_dir = RESAMPLED_DIR / "instruments" / freq

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

    if len(data) == 0:
        return None

    df = pd.DataFrame(data)
    df = df.sort_index()
    df = df[(df.index >= start_date) & (df.index <= end_date)]

    return df


def grid_search_enhanced(df, freq, year, param_grid):
    """
    网格搜索增强版 SuperTrend 参数

    Args:
        df: 数据 DataFrame
        freq: 频率
        year: 年份
        param_grid: 参数网格字典

    Returns:
        最优结果和所有结果列表
    """
    all_results = []
    total_combinations = 1
    
    # 计算总组合数
    for key, values in param_grid.items():
        total_combinations *= len(values)
    
    print(f"   参数网格搜索开始，总共 {total_combinations} 种参数组合...")
    
    best_sharpe = -float('inf')
    best_params = None
    best_result = None
    
    current_idx = 0
    
    # 生成所有参数组合
    for period in param_grid['period']:
        for multiplier in param_grid['multiplier']:
            for n in param_grid['n']:
                for ts in param_grid['trailing_stop_rate']:
                    for max_hold in param_grid['max_holding_period']:
                        for min_liqka in param_grid['min_liqka']:
                            for max_liqka in param_grid['max_liqka']:
                                current_idx += 1
                                
                                if current_idx % 10 == 0:
                                    progress = current_idx / total_combinations * 100
                                    print(f"   进度: {progress:.1f}% ({current_idx}/{total_combinations})")
                                
                                # 创建策略
                                strategy = SupertrendEnhancedStrategy(
                                    period=period,
                                    multiplier=multiplier,
                                    n=n,
                                    trailing_stop_rate=ts,
                                    max_holding_period=max_hold,
                                    min_liqka=min_liqka,
                                    max_liqka=max_liqka
                                )
                                
                                # 生成信号
                                df_strategy = strategy.generate_signal(df.copy())
                                df_strategy = df_strategy.dropna(subset=['position'])
                                
                                if len(df_strategy) == 0:
                                    continue
                                
                                # 运行回测
                                results = run_backtest(df_strategy, strategy.name)
                                
                                # 添加参数信息
                                results['freq'] = freq
                                results['year'] = year
                                results['data_length'] = len(df_strategy)
                                results['period'] = period
                                results['multiplier'] = multiplier
                                results['n'] = n
                                results['trailing_stop_rate'] = ts
                                results['max_holding_period'] = max_hold
                                results['min_liqka'] = min_liqka
                                results['max_liqka'] = max_liqka
                                
                                all_results.append(results)
                                
                                # 记录最优参数（按夏普比率）
                                if results['sharpe_ratio'] > best_sharpe:
                                    best_sharpe = results['sharpe_ratio']
                                    best_params = {
                                        'period': period,
                                        'multiplier': multiplier,
                                        'n': n,
                                        'trailing_stop_rate': ts,
                                        'max_holding_period': max_hold,
                                        'min_liqka': min_liqka,
                                        'max_liqka': max_liqka
                                    }
                                    best_result = results
    
    return all_results, best_result, best_params


def main():
    """主程序"""
    print("="*80)
    print("增强版 SuperTrend 策略参数优化（网格搜索）")
    print("="*80)
    
    # 测试配置
    years = [2023]  # 先测试2023年
    freqs = ["15min"]  # 先测试15分钟
    
    # 参数网格搜索空间
    # 基于回测结果，我们缩小搜索范围
    param_grid = {
        'period': [7, 10, 12, 14],           # ATR 周期
        'multiplier': [2.5, 3.0, 3.5],         # ATR 倍数
        'n': [1, 2],                           # 突破确认系数（降低）
        'trailing_stop_rate': [70, 80, 90],    # 止损幅度
        'max_holding_period': [100, 150],    # 最大持仓周期
        'min_liqka': [0.5, 0.6, 0.7],          # 最小止损系数
        'max_liqka': [0.9, 1.0]               # 最大止损系数
    }
    
    print(f"\n参数网格搜索空间：")
    for key, values in param_grid.items():
        print(f"   {key}: {values}")
    
    total_combinations = 1
    for key, values in param_grid.items():
        total_combinations *= len(values)
    print(f"\n总组合数: {total_combinations}")
    
    all_results = []
    
    for year in years:
        for freq in freqs:
            print(f"\n{'='*80}")
            print(f"优化年份: {year}, 频率: {freq}")
            print(f"{'='*80}")
            
            # 加载数据
            print(f"\n加载数据...")
            df = load_data_multi_freq(freq=freq, start_date=f"{year}-01-01", end_date=f"{year}-12-31")
            
            if df is None or len(df) == 0:
                print(f"   ⚠️  {year} 年 {freq} 数据不存在，跳过")
                continue
            
            print(f"   数据加载完成: {len(df)} 行")
            print(f"   时间范围: {df.index.min()} ~ {df.index.max()}")
            
            # 运行网格搜索
            print(f"\n开始网格搜索优化...")
            start_time = time.time()
            
            results, best_result, best_params = grid_search_enhanced(df, freq, year, param_grid)
            
            elapsed = time.time() - start_time
            print(f"   网格搜索完成，耗时: {elapsed:.2f}秒")
            
            # 输出最优结果
            print(f"\n{'='*80}")
            print(f"{year} 年 {freq} 最优参数：")
            print(f"{'='*80}")
            print(f"   period: {best_params['period']}")
            print(f"   multiplier: {best_params['multiplier']}")
            print(f"   n: {best_params['n']}")
            print(f"   trailing_stop_rate: {best_params['trailing_stop_rate']}")
            print(f"   max_holding_period: {best_params['max_holding_period']}")
            print(f"   min_liqka: {best_params['min_liqka']}")
            print(f"   max_liqka: {best_params['max_liqka']}")
            
            print(f"\n最优回测结果：")
            print(f"   累计收益: {best_result['cumulative_return']:.2%}")
            print(f"   年化收益: {best_result['annual_return']:.2%}")
            print(f"   最大回撤: {best_result['max_drawdown']:.2%}")
            print(f"   夏普比率: {best_result['sharpe_ratio']:.2f}")
            print(f"   胜率: {best_result['win_rate']:.2f}%")
            print(f"   总交易次数: {best_result['total_trades']}")
            print(f"   买入持有收益: {best_result['buy_hold_return']:.2%}")
            print(f"   止损平仓次数: {best_result['stopped_out_count']}")
            
            all_results.extend(results)
    
    # 保存所有结果
    if len(all_results) > 0:
        print(f"\n{'='*80}")
        print("保存优化结果...")
        print(f"{'='*80}")
        
        results_df = pd.DataFrame(all_results)
        
        # 调整列顺序
        cols_order = ['strategy_name', 'period', 'multiplier', 'n', 'trailing_stop_rate',
                     'max_holding_period', 'min_liqka', 'max_liqka',
                     'total_trades', 'cumulative_return', 'annual_return',
                     'max_drawdown', 'sharpe_ratio', 'win_rate', 'buy_hold_return',
                     'stopped_out_count', 'data_length', 'freq', 'year']
        
        existing_cols = [col for col in cols_order if col in results_df.columns]
        results_df = results_df[existing_cols]
        
        # 按夏普比率排序
        results_df = results_df.sort_values('sharpe_ratio', ascending=False)
        
        output_file = Path("/mnt/d/quant/qlib_backtest/supertrend_enhanced_optimization.csv")
        results_df.to_csv(output_file, index=False, encoding='utf-8-sig')
        print(f"   结果已保存到: {output_file}")
        
        # 输出Top 10结果
        print(f"\n{'='*80}")
        print("Top 10 优化结果（按夏普比率排序）")
        print(f"{'='*80}")
        
        print(f"\n{'参数':<50} {'收益':<8} {'回撤':<8} {'夏普':<6} {'交易':<6}")
        print("-"*90)
        
        top_10 = results_df.head(10)
        for idx, row in top_10.iterrows():
            param_str = f"({row['period']},{row['multiplier']:.1f},n={row['n']},ts={row['trailing_stop_rate']})"
            print(f"{param_str:<50} {row['cumulative_return']:>6.2%} {row['max_drawdown']:>6.2%} "
                  f"{row['sharpe_ratio']:>5.2f} {row['total_trades']:>6}")
    
    print(f"\n{'='*80}")
    print("参数优化完成!")
    print(f"{'='*80}")
    
    return all_results


if __name__ == "__main__":
    results = main()
