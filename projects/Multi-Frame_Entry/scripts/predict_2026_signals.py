#!/usr/bin/env python3
"""
2026年趋势信号预测脚本

功能：
1. 使用window20模型对2025-10-01至2026-02-13的60分钟数据进行预测
2. 检测趋势信号变化点
3. 输出Excel格式结果

品种：HC(热卷)、I(铁矿石)、AU(黄金)、CF(郑棉)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
import sys
project_root = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry')
sys.path.insert(0, str(project_root))

from features.trend_features import TrendFeatures

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 品种配置
SYMBOL_CONFIG = {
    'HC888': {'full_code': 'HC8888.XSGE', 'name': '热卷'},
    'I888': {'full_code': 'I8888.XDCE', 'name': '铁矿石'},
    'AU888': {'full_code': 'AU8888.XSGE', 'name': '黄金'},
    'CF888': {'full_code': 'CF8888.XZCE', 'name': '郑棉'},
}

# 路径配置
DATA_DIR = Path('/Users/mystryl/Library/CloudStorage/Dropbox/润富/钢铁/code/期货相关代码/futures_data_fetcher/futures_data/60min')
MODEL_DIR = project_root / 'models' / 'rolling_3month'
OUTPUT_DIR = project_root / 'predictions' / '2026_signals'
THRESHOLD = 0.5  # 二分类概率阈值


def load_60min_data(symbol: str) -> pd.DataFrame:
    """加载60分钟数据"""
    file_path = DATA_DIR / f'{symbol}.parquet'
    logger.info(f"加载数据: {file_path}")
    
    df = pd.read_parquet(file_path)
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.set_index('datetime')
    
    # 添加模型需要的列（open_interest, money）
    # 如果数据中没有，使用成交量估算
    if 'money' not in df.columns:
        df['money'] = df['close'] * df['volume']  # 估算成交额
    if 'open_interest' not in df.columns:
        df['open_interest'] = df['volume']  # 估算持仓量
    
    # 筛选2025-10-01之后的数据
    df = df[df.index >= '2025-10-01']
    
    logger.info(f"  数据范围: {df.index.min()} 至 {df.index.max()}, 共 {len(df)} 条")
    
    return df


def generate_features(df: pd.DataFrame) -> pd.DataFrame:
    """生成趋势特征"""
    # 需要保留足够的历史数据用于特征计算
    # 确保有足够的历史数据（至少60根K线）
    df_full = df.copy()
    
    # 使用TrendFeatures计算特征
    calculator = TrendFeatures(
        ema_short=20,
        ema_long=60,
        adx_period=14,
        atr_period=14
    )
    
    # 计算所有特征（不shift，因为这是用于预测的）
    df_features = calculator.compute_all_features(df_full, shift=True)
    
    # 只保留2025-10-01之后的数据
    df_features = df_features[df_features.index >= '2025-10-01']
    
    return df_features


def load_model(symbol: str):
    """加载二分类模型"""
    full_code = SYMBOL_CONFIG[symbol]['full_code']
    model_file = MODEL_DIR / f'{full_code}_window20.pkl'

    logger.info(f"加载模型: {model_file}")

    with open(model_file, 'rb') as f:
        model_data = pickle.load(f)

    return model_data


def predict_three_class_signals_simple(
    df_features: pd.DataFrame,
    model_data: dict
) -> pd.DataFrame:
    """
    三分类信号预测（简化版：二分类+MACD）

    Args:
        df_features: 特征DataFrame
        model_data: 二分类模型数据

    Returns:
        结果DataFrame，包含三分类信号和概率
    """
    model = model_data['model']
    features = model_data['features']

    # 提取特征
    X = df_features[features].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())

    # 预测概率
    binary_proba = model.predict_proba(X)[:, 1]  # P(有趋势)

    # 使用MACD直方图判断方向
    macd_hist = df_features['macd_histogram'].values

    # 组合预测
    signals = []
    p_range = []  # P(震荡)
    p_up = []     # P(上涨)
    p_down = []   # P(下跌)

    for i in range(len(df_features)):
        p_trend = binary_proba[i]  # P(有趋势)
        p_no_trend = 1 - p_trend  # P(震荡)

        if p_trend < THRESHOLD:
            # 震荡
            signals.append('震荡')
            p_range.append(p_no_trend)
            p_up.append(0)
            p_down.append(0)
        else:
            # 有趋势，用MACD判断方向
            if macd_hist[i] > 0:
                signals.append('上涨')
                p_range.append(0)
                p_up.append(p_trend)
                p_down.append(0)
            else:
                signals.append('下跌')
                p_range.append(0)
                p_up.append(0)
                p_down.append(p_trend)

    # 创建结果DataFrame
    df_result = pd.DataFrame({
        'datetime': df_features.index,
        'signal': signals,
        'P(震荡)': p_range,
        'P(上涨)': p_up,
        'P(下跌)': p_down,
        'binary_proba': binary_proba,
        'macd_histogram': macd_hist
    })

    return df_result


def detect_signal_changes(df_result: pd.DataFrame) -> pd.DataFrame:
    """检测信号变化点"""
    # 检测信号变化
    df_result['prev_signal'] = df_result['signal'].shift(1)

    # 只保留信号变化的行
    df_changes = df_result[
        (df_result['signal'] != df_result['prev_signal']) &
        (df_result['prev_signal'].notna())
    ].copy()

    # 添加信号变化类型（直接使用当前信号）
    df_changes['signal_change_type'] = df_changes['signal']

    # 只保留有效的信号变化
    df_changes = df_changes[df_changes['signal_change_type'].notna()]

    return df_changes


def main():
    """主函数"""
    logger.info("="*80)
    logger.info("2026年趋势信号预测（三分类：上涨/下跌/震荡）")
    logger.info("="*80)
    logger.info(f"数据目录: {DATA_DIR}")
    logger.info(f"模型目录: {MODEL_DIR}")
    logger.info(f"输出目录: {OUTPUT_DIR}")
    logger.info(f"方法: 二分类模型 + MACD方向判断")
    logger.info(f"信号阈值: {THRESHOLD}")
    logger.info("="*80 + "\n")
    
    # 创建输出目录
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = []
    all_klines = []  # 存储所有K线信号
    
    for symbol, config in SYMBOL_CONFIG.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"处理品种: {symbol} ({config['name']})")
        logger.info(f"{'='*60}")
        
        try:
            # 1. 加载数据
            df = load_60min_data(symbol)
            
            # 2. 生成特征
            logger.info("生成特征...")
            df_features = generate_features(df)
            logger.info(f"  特征数据: {len(df_features)} 条")
            
            # 3. 加载模型
            model_data = load_model(symbol)

            # 4. 预测信号
            logger.info("预测信号...")
            logger.info("  使用二分类+MACD判断方向")

            df_result = predict_three_class_signals_simple(df_features, model_data)

            # 5. 保存所有K线信号（包含Close价格）
            df_all = df_result[['datetime', 'signal', 'P(震荡)', 'P(上涨)', 'P(下跌)']].copy()
            df_all['Close'] = df_features['close'].values  # 添加Close价格（按顺序对应）
            df_all['品种'] = config['name']
            df_all['合约代码'] = config['full_code']
            all_klines.append(df_all)
            
            # 6. 检测信号变化
            logger.info("检测信号变化...")
            df_changes = detect_signal_changes(df_result)

            logger.info(f"  检测到 {len(df_changes)} 个信号变化点")

            # 7. 添加品种信息和Close价格
            df_changes['品种'] = config['name']
            df_changes['合约代码'] = config['full_code']

            # 通过datetime索引匹配Close价格
            close_prices = []
            for dt in df_changes['datetime']:
                close_price = df_features.loc[dt, 'close']
                close_prices.append(close_price)
            df_changes['Close'] = close_prices
            
            all_results.append(df_changes)
            
            # 打印信号变化
            if len(df_changes) > 0:
                logger.info("\n信号变化详情:")
                for _, row in df_changes.iterrows():
                    p_max = max(row['P(震荡)'], row['P(上涨)'], row['P(下跌)'])
                    logger.info(f"  {row['datetime']} | {row['signal_change_type']} | 收盘价: {row['Close']:.2f} | 最高概率: {p_max:.4f}")
            
        except Exception as e:
            logger.error(f"处理 {symbol} 时出错: {e}")
            import traceback
            traceback.print_exc()
    
    # 合并所有结果（信号变化点）
    if all_results:
        df_final = pd.concat(all_results, ignore_index=True)
        
        # 整理列顺序
        df_final = df_final[['品种', '合约代码', 'datetime', 'signal_change_type', 'Close', 'P(震荡)', 'P(上涨)', 'P(下跌)']]
        df_final = df_final.rename(columns={
            'datetime': '信号变化时间',
            'signal_change_type': '信号类型',
            'Close': '收盘价'
        })
        
        # 按时间排序
        df_final = df_final.sort_values('信号变化时间')
        
        # 保存Excel - 信号变化点
        output_file = OUTPUT_DIR / '趋势信号变化_2026.xlsx'
        df_final.to_excel(output_file, index=False)
        logger.info(f"信号变化结果已保存: {output_file}")
        logger.info(f"共 {len(df_final)} 条信号变化记录")
        
        # 保存所有K线信号明细（包含Close价格）
        df_all_klines = pd.concat(all_klines, ignore_index=True)

        # 按时间排序
        df_all_klines = df_all_klines.sort_values('datetime')

        # 重命名列（中文列名）
        df_all_klines = df_all_klines.rename(columns={
            'datetime': '时间',
            'signal': '信号类型',
            'Close': '收盘价',
            'P(震荡)': 'P(震荡)',
            'P(上涨)': 'P(上涨)',
            'P(下跌)': 'P(下跌)'
        })

        # 计算最高概率
        df_all_klines['最高概率'] = df_all_klines[['P(震荡)', 'P(上涨)', 'P(下跌)']].max(axis=1)

        # 调整列顺序
        df_all_klines = df_all_klines[[
            '品种', '合约代码', '时间', '收盘价',
            '信号类型', 'P(震荡)', 'P(上涨)', 'P(下跌)', '最高概率'
        ]]

        output_file_all = OUTPUT_DIR / '所有K线信号_2026.xlsx'
        df_all_klines.to_excel(output_file_all, index=False)
        logger.info(f"\n所有K线信号已保存: {output_file_all}")
        logger.info(f"共 {len(df_all_klines)} 条K线信号记录")
        logger.info(f"列: {list(df_all_klines.columns)}")
        logger.info(f"{'='*80}")
        
        # 打印汇总
        print("\n" + "="*80)
        print("信号变化汇总（三分类）")
        print("="*80)
        for symbol, config in SYMBOL_CONFIG.items():
            symbol_data = df_final[df_final['品种'] == config['name']]
            if len(symbol_data) > 0:
                print(f"\n{config['name']} ({config['full_code']}):")
                for _, row in symbol_data.iterrows():
                    print(f"  {row['信号变化时间']} | {row['信号类型']} | 收盘价: {row['收盘价']:.2f}")
        
        return df_final
    else:
        logger.warning("没有生成任何结果")
        return None


if __name__ == '__main__':
    main()
