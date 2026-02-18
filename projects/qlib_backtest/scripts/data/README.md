# 统一数据管理系统

智能数据提供者（SmartDataProvider）- 统一的 Parquet 数据管理解决方案。

## 特性

- ✅ **节省空间**：Parquet 格式压缩率 81%（40MB vs 216MB）
- ✅ **跨系统兼容**：使用相对路径，自动检测项目根目录
- ✅ **智能路由**：根据使用场景自动选择最优格式
- ✅ **透明缓存**：自动管理 Parquet → Qlib 转换
- ✅ **按需转换**：只在需要时转换为 Qlib 格式
- ✅ **统一接口**：一个 API 支持多种使用场景

## 快速开始

### 1. 数据探索和分析

```python
from projects.qlib_backtest.scripts.data import ParquetDataProvider

# 创建数据提供者（自动检测路径）
provider = ParquetDataProvider()

# 读取 Parquet 数据（快速）
df = provider.get_data(
    "HC8888.XSGE",
    "2024-01-01",
    "2024-12-31",
    fields=["open", "high", "low", "close", "volume"],
    format='parquet'
)

# 分析数据
print(df.describe())
df['close'].plot()
```

### 2. Qlib 回测

```python
import qlib
from projects.qlib_backtest.scripts.data import ParquetDataProvider

provider = ParquetDataProvider()

# 首次运行：自动转换为 Qlib 格式
qlib.init(provider_uri=str(provider.qlib_cache_dir), region="cn")

from qlib.data import D
df = D.features(
    instruments=["HC8888.XSGE"],
    fields=["$close", "$volume"]
)

# 第二次运行：直接使用缓存（无需重新转换）
```

### 3. 便捷函数

```python
from projects.qlib_backtest.scripts.data import get_data

# 快速读取数据
df = get_data(
    "HC8888.XSGE",
    "2024-01-01",
    "2024-12-31",
    fields=["close"]
)
```

## 命令行工具

### 批量转换为 Qlib 格式

```bash
# 转换单个合约
python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib \\
    --instrument HC8888.XSGE

# 转换所有合约
python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib --all

# 增量转换（只转换新增或修改的）
python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib \\
    --all --incremental

# 使用模式匹配
python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib \\
    --all --pattern "HC*"

# 强制重新转换
python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib \\
    --all --force

# 试运行（查看将要转换的合约）
python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib \\
    --all --dry-run
```

## API 参考

### ParquetDataProvider

主要数据提供者类。

#### 初始化

```python
provider = ParquetDataProvider(
    enable_memory_cache=True,    # 启用内存缓存
    memory_cache_size=100        # 缓存大小
)
```

#### 主要方法

##### get_data()

获取数据，自动选择最优格式。

```python
df = provider.get_data(
    instrument="HC8888.XSGE",    # 合约代码
    start_time="2024-01-01",     # 开始时间
    end_time="2024-12-31",       # 结束时间
    fields=["close", "volume"],  # 字段列表
    format='auto'                # 格式: 'auto', 'parquet', 'qlib'
)
```

**参数**：
- `instrument` (str): 合约代码
- `start_time` (str/datetime): 开始时间
- `end_time` (str/datetime): 结束时间
- `fields` (List[str]): 字段列表，可选：`open`, `high`, `low`, `close`, `volume`, `money`, `avg`, `open_interest`
- `format` (str): 数据格式
  - `'auto'`: 自动选择（默认）
  - `'parquet'`: 直接读取 Parquet（快速）
  - `'qlib'`: 读取或转换为 Qlib 格式
- `use_cache` (bool): 是否使用缓存

**返回**：
- `pd.DataFrame`: 数据框

##### list_instruments()

列出可用的合约。

```python
instruments = provider.list_instruments(pattern="HC*")
# 返回: ['HC8888.XSGE', ...]
```

##### get_calendar()

获取合约的交易日历。

```python
calendar = provider.get_calendar("HC8888.XSGE")
# 返回: List[pd.Timestamp]
```

##### convert_to_qlib()

将 Parquet 数据转换为 Qlib 格式。

```python
provider.convert_to_qlib(
    instrument="HC8888.XSGE",
    force=False,        # 是否强制转换
    progress=True       # 是否显示进度
)
```

##### convert_batch()

批量转换为 Qlib 格式。

```python
provider.convert_batch(
    instruments=None,   # None 表示使用 pattern 查找所有
    pattern="*",
    force=False
)
```

##### clear_cache()

清空内存缓存。

```python
provider.clear_cache()
```

##### get_cache_stats()

获取缓存统计信息。

```python
stats = provider.get_cache_stats()
# 返回: {'enabled': True, 'size': 10, 'max_size': 100}
```

### 便捷函数

#### get_data()

快速获取数据的便捷函数。

```python
from projects.qlib_backtest.scripts.data import get_data

df = get_data(
    instrument="HC8888.XSGE",
    start_time="2024-01-01",
    end_time="2024-12-31",
    fields=["close", "volume"],
    format='auto'
)
```

## 配置

路径配置在 `config.py` 中：

```python
from projects.qlib_backtest.scripts.data.config import (
    PROJECT_ROOT,
    PARQUET_DATA_DIR,
    QLIB_CACHE_DIR,
    STANDARD_FIELDS
)

# 自动检测项目根目录
print(PROJECT_ROOT)
# /Users/mystryl/Documents/Quant

# Parquet 数据目录
print(PARQUET_DATA_DIR)
# /Users/mystryl/Documents/Quant/K线数据库/期货商品索引_parquet

# Qlib 缓存目录
print(QLIB_CACHE_DIR)
# /Users/mystryl/Documents/Quant/data/qlib_cache
```

## 使用场景

### 场景 1：Jupyter Notebook 数据探索

```python
# 在项目根目录的 notebook 中
from projects.qlib_backtest.scripts.data import ParquetDataProvider
import matplotlib.pyplot as plt

provider = ParquetDataProvider()

# 查看可用合约
instruments = provider.list_instruments()
print(f"可用合约: {len(instruments)} 个")

# 获取数据
df = provider.get_data(
    "HC8888.XSGE",
    "2024-01-01",
    "2024-12-31",
    fields=["open", "high", "low", "close", "volume"]
)

# 可视化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# 价格
df[['close']].plot(ax=ax1)
ax1.set_title('HC8888.XSGE 价格走势')

# 成交量
df[['volume']].plot(ax=ax2)
ax2.set_title('HC8888.XSGE 成交量')

plt.tight_layout()
plt.show()
```

### 场景 2：与现有回测脚本集成

```python
# 现有的回测脚本（无需修改）
import qlib
from projects.qlib_backtest.scripts.data import ParquetDataProvider

# 初始化数据提供者
provider = ParquetDataProvider()

# 初始化 Qlib（指向缓存目录）
qlib.init(provider_uri=str(provider.qlib_cache_dir), region="cn")

# 正常使用 Qlib API
from qlib.data import D

# 获取数据（自动转换）
df = D.features(
    instruments=["HC8888.XSGE"],
    fields=["$close", "$high", "$low", "$volume"],
    start_time="2024-01-01",
    end_time="2024-12-31"
)

# 回测逻辑...
```

### 场景 3：批量数据处理

```python
from projects.qlib_backtest.scripts.data import ParquetDataProvider
import pandas as pd

provider = ParquetDataProvider()

# 获取所有合约
instruments = provider.list_instruments()

# 批量处理
results = {}
for instrument in instruments:
    try:
        df = provider.get_data(
            instrument,
            "2024-01-01",
            "2024-12-31",
            fields=["close"],
            format='parquet'  # 使用 Parquet 快速读取
        )

        # 计算收益率
        df['returns'] = df['close'].pct_change()
        results[instrument] = df['returns'].describe()

    except Exception as e:
        print(f"处理失败 {instrument}: {e}")

# 汇总结果
summary = pd.DataFrame(results).T
print(summary)
```

## 性能对比

| 操作 | Parquet | Qlib .bin | 说明 |
|------|---------|-----------|------|
| **存储空间** | 40 MB | 160 MB | Parquet 节省 75% |
| **首次读取** | ~0.5s | ~2s | Parquet 更快 |
| **随机访问** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Qlib 更优 |
| **范围查询** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | 相当 |
| **回测性能** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | Qlib 更优 |

**建议**：
- 数据分析：用 Parquet
- 回测：用 Qlib（自动转换）

## 数据结构

### Parquet 格式（主存储）

```
K线数据库/期货商品索引_parquet/
├── HC8888.XSGE.parquet     # 单个文件，包含所有字段
├── RB8888.XSHG.parquet
└── ...
```

文件结构：
- 索引：DatetimeIndex
- 列：`open`, `high`, `low`, `close`, `volume`, `money`, `avg`, `open_interest`

### Qlib 缓存（按需生成）

```
data/qlib_cache/instruments/1min/
├── $open/
│   ├── HC8888.XSGE.csv
│   └── ...
├── $close/
│   ├── HC8888.XSGE.csv
│   └── ...
└── ...
```

文件格式（CSV）：
```csv
$close
2014-03-21 09:01:00,3262.749
2014-03-21 09:02:00,3277.958
...
```

## 异常处理

```python
from projects.qlib_backtest.scripts.data import (
    ParquetDataProvider,
    InstrumentNotFoundError,
    InvalidFieldError
)

provider = ParquetDataProvider()

try:
    df = provider.get_data(
        "INVALID.XSGE",  # 不存在的合约
        "2024-01-01",
        "2024-12-31"
    )
except InstrumentNotFoundError as e:
    print(f"合约不存在: {e}")

try:
    df = provider.get_data(
        "HC8888.XSGE",
        "2024-01-01",
        "2024-12-31",
        fields=["invalid_field"]  # 无效字段
    )
except InvalidFieldError as e:
    print(f"字段无效: {e}")
```

## 最佳实践

### 1. 使用相对路径

```python
# ✅ 推荐：自动检测项目根目录
provider = ParquetDataProvider()

# ❌ 不推荐：硬编码绝对路径
provider = ParquetDataProvider(
    parquet_dir="/Users/mystryl/Documents/K线数据库/..."
)
```

### 2. 选择合适的格式

```python
# 数据分析：使用 Parquet（快速）
df = provider.get_data(..., format='parquet')

# Qlib 回测：使用 Qlib（高性能）
df = provider.get_data(..., format='qlib')

# 让系统自动选择
df = provider.get_data(..., format='auto')
```

### 3. 利用缓存

```python
# 启用内存缓存（默认）
provider = ParquetDataProvider(enable_memory_cache=True)

# 多次读取相同数据（使用缓存）
for _ in range(10):
    df = provider.get_data("HC8888.XSGE", ...)  # 第二次开始使用缓存
```

### 4. 批量转换

```python
# 使用命令行工具批量转换
# 而不是在 Python 中逐个转换

# ✅ 推荐
python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib --all

# ❌ 不推荐
for inst in instruments:
    provider.convert_to_qlib(inst)
```

## 故障排除

### 问题：找不到合约

```python
# 检查合约是否存在
instruments = provider.list_instruments()
print(instruments)

# 检查文件是否存在
from pathlib import Path
parquet_file = PARQUET_DATA_DIR / "HC8888.XSGE.parquet"
print(parquet_file.exists())
```

### 问题：转换失败

```python
# 查看详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

provider = ParquetDataProvider()
provider.convert_to_qlib("HC8888.XSGE", progress=True)
```

### 问题：路径错误

```python
# 检查项目根目录
from projects.qlib_backtest.scripts.data.config import PROJECT_ROOT
print(f"项目根目录: {PROJECT_ROOT}")

# 检查数据目录
from projects.qlib_backtest.scripts.data.config import PARQUET_DATA_DIR
print(f"Parquet 目录: {PARQUET_DATA_DIR}")
print(f"目录存在: {PARQUET_DATA_DIR.exists()}")
```

## 扩展

### 添加新的数据源

1. 在 `config.py` 中添加新的数据源路径
2. 在 `ParquetDataProvider` 中添加新的读取方法
3. 更新 `_select_format()` 方法以支持新格式

### 自定义字段映射

```python
# 在 config.py 中修改
FIELD_ALIASES = {
    "close": ["close", "Close", "收盘价", "your_custom_field"],
    ...
}
```

## 更多信息

- **配置文件**：`config.py`
- **核心实现**：`unified_data_provider.py`
- **转换工具**：`convert_parquet_to_qlib.py`
- **包初始化**：`__init__.py`

## 更新日志

### v1.0.0 (2024-02-19)

- ✅ 初始版本
- ✅ Parquet 数据提供者
- ✅ 自动转换为 Qlib 格式
- ✅ 命令行转换工具
- ✅ 内存缓存支持
- ✅ 相对路径支持
