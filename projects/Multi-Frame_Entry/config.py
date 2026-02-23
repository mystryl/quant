"""
项目配置文件

统一管理项目中的路径和配置参数
"""
from pathlib import Path

# 项目根目录（自动检测）
# 获取本文件所在目录的父目录
PROJECT_ROOT = Path(__file__).parent.absolute()

# 数据目录配置
DATA_DIR = PROJECT_ROOT / 'data'
FEATURES_DIR = DATA_DIR / 'features'
LABELS_DIR = DATA_DIR / 'labels'
KLINE_DATA_DIR = DATA_DIR / '60min'  # 60分钟K线数据目录

# 模型目录配置
MODELS_DIR = PROJECT_ROOT / 'models'
ROLLING_MODEL_DIR = MODELS_DIR / 'rolling_3month'

# 脚本目录配置
SCRIPTS_DIR = PROJECT_ROOT / 'scripts'
FEATURES_DIR = PROJECT_ROOT / 'features'

# 预测结果目录
PREDICTIONS_DIR = PROJECT_ROOT / 'predictions'

# 日志配置
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'

# 数据获取配置
DATA_FETCH_CONFIG = {
    'period': '60min',      # 60分钟K线
    'bars': 100,            # 获取100根K线
    'lookback_bars': 10,    # 检测最近10根K线
}

# 模型配置
MODEL_CONFIG = {
    'threshold': 0.5,       # 二分类概率阈值
    'window': 20,           # 预测窗口（根K线）
}

# 品种配置
SYMBOL_CONFIG = {
    'RB0': {
        'full_code': 'HC8888.XSGE',
        'name': '螺纹钢',
        'model_code': 'HC888',
        'data_file': 'HC888.parquet'
    },
    'HC0': {
        'full_code': 'HC8888.XSGE',
        'name': '热卷',
        'model_code': 'HC888',
        'data_file': 'HC888.parquet'
    },
    'I0': {
        'full_code': 'I8888.XDCE',
        'name': '铁矿石',
        'model_code': 'I888',
        'data_file': 'I888.parquet'
    },
    'AU0': {
        'full_code': 'AU8888.XSGE',
        'name': '黄金',
        'model_code': 'AU888',
        'data_file': 'AU888.parquet'
    },
    'CF0': {
        'full_code': 'CF8888.XZCE',
        'name': '郑棉',
        'model_code': 'CF888',
        'data_file': 'CF888.parquet'
    },
}


def get_model_file(symbol: str) -> Path:
    """获取品种对应的模型文件路径"""
    config = SYMBOL_CONFIG.get(symbol)
    if not config:
        raise ValueError(f"不支持的品种: {symbol}")
    return ROLLING_MODEL_DIR / f"{config['full_code']}_window20.pkl"


def get_data_file(symbol: str) -> Path:
    """获取品种对应的数据文件路径"""
    config = SYMBOL_CONFIG.get(symbol)
    if not config:
        raise ValueError(f"不支持的品种: {symbol}")
    return KLINE_DATA_DIR / config['data_file']


def get_symbol_name(symbol: str) -> str:
    """获取品种中文名称"""
    config = SYMBOL_CONFIG.get(symbol)
    if not config:
        raise ValueError(f"不支持的品种: {symbol}")
    return config['name']


if __name__ == '__main__':
    """测试配置"""
    print("项目配置信息")
    print("=" * 60)
    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"数据目录: {DATA_DIR}")
    print(f"K线数据目录: {KLINE_DATA_DIR}")
    print(f"模型目录: {ROLLING_MODEL_DIR}")
    print(f"脚本目录: {SCRIPTS_DIR}")
    print()
    print("支持的品种:")
    for symbol, config in SYMBOL_CONFIG.items():
        print(f"  {symbol}: {config['name']} - {config['full_code']}")
    print()
    print("数据文件检查:")
    for symbol in SYMBOL_CONFIG.keys():
        data_file = get_data_file(symbol)
        exists = "✓" if data_file.exists() else "✗"
        print(f"  {exists} {symbol}: {data_file}")
