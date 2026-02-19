#!/usr/bin/env python3
"""
KLineChart Pro 测试服务器
直接从 Parquet 数据库读取合约K线数据并提供API
"""

import json
from pathlib import Path
from flask import Flask, jsonify, send_from_directory, request
from flask_cors import CORS
import pandas as pd
from datetime import datetime

app = Flask(__name__, static_folder='.')
CORS(app)

# 数据路径
PARQUET_DIR = Path('/Users/mystryl/Documents/Quant/K线数据库/期货主力连续_parquet')
OUTPUT_DIR = Path(__file__).parent.parent / 'frontend' / 'public' / 'data'

def load_contracts_from_parquet():
    """从 Parquet 目录扫描合约列表"""
    contracts = []

    if not PARQUET_DIR.exists():
        print(f"⚠️  Parquet 目录不存在: {PARQUET_DIR}")
        return contracts

    # 品种名称映射
    name_map = {
        'CU': '铜主力连续',
        'AL': '铝主力连续',
        'AU': '黄金主力连续',
        'AG': '白银主力连续',
        'ZN': '锌主力连续',
        'NI': '镍主力连续',
        'SN': '锡主力连续',
        'PB': '铅主力连续',
        'RB': '螺纹钢主力连续',
        'HC': '热卷主力连续',
        'BU': '沥青主力连续',
        'RU': '橡胶主力连续',
        'A': '豆一主力连续',
        'M': '豆粕主力连续',
        'Y': '豆油主力连续',
        'P': '棕榈油主力连续',
        'C': '玉米主力连续',
        'MA': '甲醇主力连续',
        'SR': '白糖主力连续',
        'CF': '棉花主力连续',
        'TA': 'PTA主力连续',
        'FG': '玻璃主力连续',
        'OI': '菜油主力连续',
        'RM': '菜粕主力连续'
    }

    # 扫描所有 parquet 文件
    for parquet_file in PARQUET_DIR.glob('*.parquet'):
        symbol = parquet_file.stem  # 文件名不带扩展名

        # 提取交易所信息
        if '.' in symbol:
            exchange = symbol.split('.')[-1]
            code = symbol.split('9')[0]
        else:
            continue

        name = name_map.get(code, f'{code}主力连续')

        contracts.append({
            'symbol': symbol,
            'name': name,
            'exchange': exchange,
            'dataAvailable': True
        })

    return sorted(contracts, key=lambda x: x['symbol'])

def load_kline_from_parquet(symbol, start_date=None, end_date=None):
    """
    从 Parquet 文件加载K线数据

    参数:
        symbol: 合约代码
        start_date: 开始日期 (datetime 或 None)
        end_date: 结束日期 (datetime 或 None)

    返回:
        DataFrame with columns [timestamp, open, high, low, close, volume]
    """
    parquet_file = PARQUET_DIR / f'{symbol}.parquet'

    if not parquet_file.exists():
        print(f"❌ 文件不存在: {parquet_file}")
        return None

    try:
        # 读取 Parquet 文件
        df = pd.read_parquet(parquet_file)

        # 确保索引是datetime类型
        if not isinstance(df.index, pd.DatetimeIndex):
            df.index = pd.to_datetime(df.index)

        # 按日期范围过滤
        if start_date:
            df = df[df.index >= start_date]
        if end_date:
            df = df[df.index <= end_date]

        # 重置索引
        df = df.reset_index()

        # 转换时间戳为毫秒
        df['timestamp'] = df['index'].astype('int64') // 10**6

        # 选择需要的列
        df = df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

        return df

    except Exception as e:
        print(f"❌ 读取 Parquet 失败: {e}")
        return None

def resample_kline_data(df, period):
    """
    将1分钟K线数据重采样为指定周期

    参数:
        df: DataFrame with columns [timestamp, open, high, low, close, volume]
        period: 周期字符串 ('1m', '5m', '15m', '1D', '1W')

    返回:
        重采样后的 DataFrame
    """
    if period == '1m':
        return df

    # 转换时间戳为 datetime
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='ms')

    # 设置时间戳为索引
    df.set_index('datetime', inplace=True)

    # 根据周期确定重采样规则
    period_map = {
        '5m': '5T',
        '15m': '15T',
        '1D': '1D',
        '1W': '1W'
    }

    if period not in period_map:
        return df

    rule = period_map[period]

    # 重采样数据
    resampled = df.resample(rule).agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum'
    }).dropna()

    # 转换回原始格式
    resampled.reset_index(inplace=True)
    resampled['timestamp'] = resampled['datetime'].astype('int64') // 10**6
    resampled = resampled[['timestamp', 'open', 'high', 'low', 'close', 'volume']]

    return resampled

@app.route('/')
def index():
    """返回主页面"""
    return send_from_directory('.', 'index.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """提供静态文件"""
    return send_from_directory('static', filename)

@app.route('/api/contracts')
def get_contracts():
    """获取合约列表"""
    contracts = load_contracts_from_parquet()
    return jsonify(contracts)

@app.route('/api/kline/<symbol>')
def get_kline(symbol):
    """
    获取K线数据

    参数:
        symbol: 合约代码
        period: 周期 (1m, 5m, 15m, 1D, 1W)，默认 1m
        start_date: 开始日期 (YYYY-MM-DD)，默认 2023-01-01
        end_date: 结束日期 (YYYY-MM-DD)，默认 2025-12-31
        limit: 返回数据条数，默认不限制
    """
    # 获取请求参数
    period = request.args.get('period', '1m')
    start_date_str = request.args.get('start_date', '2023-01-01')
    end_date_str = request.args.get('end_date', '2025-12-31')
    limit = request.args.get('limit', None)

    # 转换日期
    start_date = pd.to_datetime(start_date_str)
    end_date = pd.to_datetime(end_date_str)

    print(f"📊 请求 K线数据: {symbol}, 周期: {period}, 日期: {start_date_str} ~ {end_date_str}")

    # 从 Parquet 加载数据
    df = load_kline_from_parquet(symbol, start_date, end_date)

    if df is None or df.empty:
        return jsonify({'error': 'Data not found'}), 404

    # 应用限制
    if limit:
        limit = int(limit)
        df = df.tail(limit)

    # 重采样数据
    df_resampled = resample_kline_data(df, period)

    # 转换为列表
    kline_data = df_resampled.to_dict('records')

    print(f"✅ 返回 {len(kline_data)} 条 {period} 数据 (原始: {len(df)} 条)")

    return jsonify({
        'data': kline_data,
        'count': len(kline_data),
        'period': period,
        'dateRange': {
            'start': start_date_str,
            'end': end_date_str
        }
    })

if __name__ == '__main__':
    print("=" * 60)
    print("🚀 KLineChart Pro 测试服务器 (Parquet 直接读取)")
    print("=" * 60)
    print(f"📂 Parquet 目录: {PARQUET_DIR}")
    print(f"🌐 访问地址: http://localhost:5001")
    print(f"📅 默认日期范围: 2023-01-01 ~ 2025-12-31")
    print("=" * 60)

    # 检查数据目录
    if not PARQUET_DIR.exists():
        print(f"⚠️  警告: Parquet 目录不存在: {PARQUET_DIR}")
    else:
        contracts = load_contracts_from_parquet()
        print(f"✅ 找到 {len(contracts)} 个合约")
        for contract in contracts[:5]:
            print(f"   - {contract['name']} ({contract['symbol']})")
        if len(contracts) > 5:
            print(f"   ... 还有 {len(contracts) - 5} 个")

    app.run(debug=False, port=5001)
