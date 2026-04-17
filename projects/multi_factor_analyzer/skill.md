# 数据转换工具使用指南

## 概述

本项目使用统一的 Parquet 数据管理系统，提供智能的数据访问接口。

**核心特性：**
- Parquet 格式存储（节省 81% 空间）
- 自动转换为 Qlib 格式（按需）
- 相对路径支持（跨系统兼容）
- 统一 API（一个接口支持多种场景）

---

## 何时使用数据转换工具

### 场景 1：数据分析和探索

**使用 Parquet 格式** - 速度快，不占用额外空间

```python
from projects.qlib_backtest.scripts.data import ParquetDataProvider

provider = ParquetDataProvider()
df = provider.get_data(
    "HC8888.XSGE",
    "2024-01-01",
    "2024-12-31",
    fields=["open", "high", "low", "close", "volume"],
    format='parquet'  # 显式使用 Parquet
)
```

### 场景 2：Qlib 回测

**使用 Qlib 格式** - 高性能随机访问

```python
import qlib
from projects.qlib_backtest.scripts.data import ParquetDataProvider

provider = ParquetDataProvider()

# 首次回测：自动转换为 Qlib 格式（如需要）
qlib.init(provider_uri=str(provider.qlib_cache_dir), region="cn")

from qlib.data import D
df = D.features(
    instruments=["HC8888.XSGE"],
    fields=["$close", "$volume"]
)
```

### 场景 3：快速读取数据

**使用便捷函数**

```python
from projects.qlib_backtest.scripts.data import get_data

# 自动选择最优格式
df = get_data(
    "HC8888.XSGE",
    "2024-01-01",
    "2024-12-31",
    fields=["close"]
)
```

### 场景 4：批量转换为 Qlib 格式

**使用命令行工具**

```bash
# 转换所有合约
python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib --all

# 转换单个合约
python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib \
    --instrument HC8888.XSGE

# 增量转换（只转换新增或修改的）
python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib \
    --all --incremental

# 试运行（查看将要转换的合约）
python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib \
    --all --dry-run
```

---

## 重要规则

### ✅ DO - 推荐做法

1. **使用统一接口**
   ```python
   # ✅ 推荐：使用统一接口
   from projects.qlib_backtest.scripts.data import ParquetDataProvider
   provider = ParquetDataProvider()
   df = provider.get_data("HC8888.XSGE", "2024-01-01", "2024-12-31")
   ```

2. **利用自动路径检测**
   ```python
   # ✅ 推荐：让系统自动检测路径
   provider = ParquetDataProvider()
   ```

3. **数据分析用 Parquet**
   ```python
   # ✅ 推荐：数据分析使用 Parquet（快速）
   df = provider.get_data(..., format='parquet')
   ```

4. **回测前批量转换**
   ```bash
   # ✅ 推荐：回测前批量转换所有合约
   python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib --all
   ```

### ❌ DON'T - 避免做法

1. **不要硬编码路径**
   ```python
   # ❌ 避免：硬编码绝对路径
   provider = ParquetDataProvider(
       parquet_dir="/Users/mystryl/Documents/K线数据库/..."
   )
   ```

2. **不要重复转换**
   ```python
   # ❌ 避免：每次都转换
   for inst in instruments:
       provider.convert_to_qlib(inst)  # 效率低
   ```

3. **不要直接读取原始 CSV**
   ```python
   # ❌ 避免：直接读取 CSV（慢且占用空间）
   df = pd.read_csv("/path/to/data.csv")
   ```

---

## 常见任务示例

### 任务 1：列出可用合约

```python
from projects.qlib_backtest.scripts.data import ParquetDataProvider

provider = ParquetDataProvider()

# 列出所有合约
all_instruments = provider.list_instruments()

# 使用模式匹配
hc_instruments = provider.list_instruments(pattern="HC*")
```

### 任务 2：获取交易日历

```python
provider = ParquetDataProvider()
calendar = provider.get_calendar("HC8888.XSGE")
```

### 任务 3：批量处理多个合约

```python
provider = ParquetDataProvider()
instruments = provider.list_instruments()

results = {}
for instrument in instruments:
    df = provider.get_data(
        instrument,
        "2024-01-01",
        "2024-12-31",
        format='parquet'  # 使用 Parquet 快速读取
    )
    # 处理数据...
```

### 任务 4：在回测脚本中使用

```python
import qlib
from projects.qlib_backtest.scripts.data import ParquetDataProvider

# 初始化数据提供者
provider = ParquetDataProvider()

# 初始化 Qlib
qlib.init(provider_uri=str(provider.qlib_cache_dir), region="cn")

# 正常使用 Qlib API
from qlib.data import D
from qlib.contrib.evaluate import risk_analysis

# 获取数据（自动转换或使用缓存）
df = D.features(
    instruments=["HC8888.XSGE"],
    fields=["$close", "$high", "$low", "$volume"]
)

# 执行回测...
```

### 任务 5：在 Jupyter Notebook 中使用

```python
# 在项目根目录的 notebook 中
from projects.qlib_backtest.scripts.data import ParquetDataProvider
import matplotlib.pyplot as plt

provider = ParquetDataProvider()

# 读取数据
df = provider.get_data(
    "HC8888.XSGE",
    "2024-01-01",
    "2024-12-31",
    fields=["close", "volume"]
)

# 可视化
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
df[['close']].plot(ax=ax1)
df[['volume']].plot(ax=ax2)
plt.show()
```

---

## 数据格式说明

### Parquet 格式（主存储）

- **位置**：`K线数据库/期货商品索引_parquet/`
- **大小**：40 MB/合约
- **优势**：压缩率高、读取速度快
- **适用**：数据分析、数据探索

### Qlib 格式（缓存）

- **位置**：`data/qlib_cache/instruments/1min/`
- **大小**：160 MB/合约
- **优势**：高性能随机访问
- **适用**：Qlib 回测

---

## 字段映射

系统支持自动字段标准化，支持以下字段：

| 标准字段 | 别名 |
|---------|------|
| open | Open, 开 |
| high | High, 高 |
| low | Low, 低 |
| close | Close, 收 |
| volume | Volume, VOL, 量 |
| money | Money, 金额, amount |
| avg | Avg, 均价, vwap |
| open_interest | OpenInterest, 持仓量, oi |

示例：
```python
# 以下写法等效
df = provider.get_data(..., fields=["close"])
df = provider.get_data(..., fields=["Close"])
df = provider.get_data(..., fields=["收"])
```

---

## 故障排除

### 问题：合约不存在

```python
# 检查合约是否存在
provider = ParquetDataProvider()
instruments = provider.list_instruments()
print(instruments)  # 查看所有可用合约
```

### 问题：路径错误

```python
# 检查项目根目录
from projects.qlib_backtest.scripts.data.config import PROJECT_ROOT
print(f"项目根目录: {PROJECT_ROOT}")
```

### 问题：转换失败

```bash
# 查看详细日志
python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib \
    --instrument HC8888.XSGE --verbose
```

---

## 性能提示

1. **首次转换**：需要 ~56秒/合约（一次性）
2. **后续使用**：<0.1秒（使用缓存）
3. **批量转换**：使用命令行工具，而非 Python 循环
4. **内存缓存**：自动启用 LRU 缓存，避免重复读取

---

## 更多信息

- **完整文档**：`projects/qlib_backtest/scripts/data/README.md`
- **代码示例**：`projects/qlib_backtest/scrip ts/data/examples.py`
- **配置文件**：`projects/qlib_backtest/scripts/data/config.py`

---

## 快速参考

```python
# 导入
from projects.qlib_backtest.scripts.data import (
    ParquetDataProvider,
    get_data
)

# 创建数据提供者
provider = ParquetDataProvider()

# 读取数据
df = provider.get_data(
    instrument="HC8888.XSGE",
    start_time="2024-01-01",
    end_time="2024-12-31",
    fields=["close", "volume"],
    format='auto'  # 'auto', 'parquet', 'qlib'
)

# 列出合约
instruments = provider.list_instruments()

# 转换为 Qlib
provider.convert_to_qlib("HC8888.XSGE")

# 获取日历
calendar = provider.get_calendar("HC8888.XSGE")
```

---

**记住**：使用统一接口，让系统自动选择最优格式！
