"""
准备前端演示用的K线数据

从Parquet文件读取数据并导出为JSON格式
"""

import pandas as pd
import json
from pathlib import Path
from datetime import datetime

# 配置
# 获取项目根目录（从scripts目录向上两级）
PROJECT_ROOT = Path(__file__).parent.parent.parent
PARQUET_DIR = PROJECT_ROOT / "../K线数据库/期货主力连续_parquet"
OUTPUT_DIR = Path(__file__).parent.parent / "frontend/public/data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# 要导出的合约列表
CONTRACTS = [
    'CU9999.XSGE',  # 铜
    'AL9999.XSGE',  # 铝
    'AU9999.XSGE',  # 黄金
    'AG9999.XSGE',  # 白银
    'RB9999.XSGE',  # 螺纹钢
    'MA9999.XZCE',  # 甲醇
    'SR9999.XZCE',  # 白糖
]

# 数据条数限制（每个合约导出的K线条数）
DATA_LIMIT = 1000


def convert_parquet_to_json(contract_symbol: str, limit: int = DATA_LIMIT) -> dict:
    """
    将Parquet数据转换为JSON格式

    Args:
        contract_symbol: 合约代码，如 CU9999.XSGE
        limit: 导出的数据条数

    Returns:
        包含K线数据的字典
    """
    parquet_file = PARQUET_DIR / f"{contract_symbol}.parquet"

    if not parquet_file.exists():
        print(f"❌ 文件不存在: {parquet_file}")
        return None

    print(f"📖 读取 {contract_symbol}...")

    # 读取Parquet文件
    df = pd.read_parquet(parquet_file)

    # 取最近的N条数据
    df = df.tail(limit).copy()

    # 确保索引是datetime类型
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # 转换为KLineChart需要的格式
    kline_data = []
    for timestamp, row in df.iterrows():
        kline_data.append({
            'timestamp': int(timestamp.timestamp() * 1000),  # 转换为毫秒时间戳
            'open': float(row['open']),
            'high': float(row['high']),
            'low': float(row['low']),
            'close': float(row['close']),
            'volume': float(row['volume'])
        })

    # 计算基础统计信息
    stats = {
        'count': len(kline_data),
        'start_date': kline_data[0]['timestamp'] if kline_data else None,
        'end_date': kline_data[-1]['timestamp'] if kline_data else None,
        'price_range': {
            'min': float(df['low'].min()),
            'max': float(df['high'].max())
        } if not df.empty else None
    }

    return {
        'symbol': contract_symbol,
        'data': kline_data,
        'stats': stats
    }


def export_contract_data(contract_symbol: str) -> bool:
    """导出单个合约数据"""
    try:
        result = convert_parquet_to_json(contract_symbol)

        if result is None:
            return False

        # 导出为JSON文件
        output_file = OUTPUT_DIR / f"{contract_symbol}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)

        print(f"✅ 导出成功: {output_file}")
        print(f"   数据条数: {result['stats']['count']}")
        print(f"   价格区间: {result['stats']['price_range']['min']:.2f} - {result['stats']['price_range']['max']:.2f}")
        return True

    except Exception as e:
        print(f"❌ 导出失败 {contract_symbol}: {e}")
        return False


def generate_contracts_list():
    """生成合约列表元数据"""
    contracts_info = []

    for symbol in CONTRACTS:
        parquet_file = PARQUET_DIR / f"{symbol}.parquet"

        if not parquet_file.exists():
            continue

        try:
            df = pd.read_parquet(parquet_file)
            df = df.tail(DATA_LIMIT)

            # 提取交易所信息
            exchange = symbol.split('.')[-1]

            # 根据品种代码生成中文名称
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

            code = symbol.split('9')[0]
            name = name_map.get(code, f'{code}主力连续')

            contracts_info.append({
                'symbol': symbol,
                'name': name,
                'exchange': exchange,
                'dataAvailable': True
            })

        except Exception as e:
            print(f"⚠️  读取元数据失败 {symbol}: {e}")

    # 导出合约列表
    contracts_file = OUTPUT_DIR / "contracts.json"
    with open(contracts_file, 'w', encoding='utf-8') as f:
        json.dump({'contracts': contracts_info}, f, ensure_ascii=False, indent=2)

    print(f"\n✅ 合约列表已导出: {contracts_file}")
    print(f"   共 {len(contracts_info)} 个合约")


def main():
    """主函数"""
    print("=" * 60)
    print("准备前端K线数据")
    print("=" * 60)
    print(f"\n📂 数据源: {PARQUET_DIR}")
    print(f"📂 输出目录: {OUTPUT_DIR}")
    print(f"📊 每个合约导出: {DATA_LIMIT} 条数据\n")

    # 检查数据目录
    if not PARQUET_DIR.exists():
        print(f"❌ 错误: 数据目录不存在 {PARQUET_DIR}")
        print("   请确保在项目根目录运行此脚本")
        return

    # 生成合约列表
    print("=" * 60)
    print("生成合约列表")
    print("=" * 60)
    generate_contracts_list()

    # 导出各个合约数据
    print("\n" + "=" * 60)
    print("导出K线数据")
    print("=" * 60)

    success_count = 0
    for contract in CONTRACTS:
        if export_contract_data(contract):
            success_count += 1
        print()

    print("=" * 60)
    print(f"✅ 完成! 成功导出 {success_count}/{len(CONTRACTS)} 个合约")
    print("=" * 60)
    print(f"\n📂 数据文件位置: {OUTPUT_DIR.absolute()}")
    print("\n前端可以通过以下方式访问:")
    print("  /data/contracts.json       - 合约列表")
    print("  /data/CU9999.XSGE.json     - K线数据示例")


if __name__ == '__main__':
    main()
