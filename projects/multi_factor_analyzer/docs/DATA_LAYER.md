# 数据访问层文档

数据访问层提供了统一的数据访问接口，支持数据加载、验证和处理。

## 模块结构

```
src/data/
├── __init__.py          # 模块导出
├── provider.py          # 数据提供者
├── loader.py            # 数据加载器
└── validator.py         # 数据验证器
```

## 核心组件

### 1. FactorDataProvider

数据提供者，封装底层 ParquetDataProvider，提供统一的数据访问接口。

#### 基本用法

```python
from src.data import FactorDataProvider

# 初始化
provider = FactorDataProvider()

# 获取单个合约数据
data = provider.get_factor_data(
    instruments="HC8888.XSGE",
    start_date="2024-01-01",
    end_date="2024-01-31",
    fields=["close", "volume"]
)

# 获取多个合约数据
data = provider.get_factor_data(
    instruments=["HC8888.XSGE", "RB8888.XSGE"],
    start_date="2024-01-01",
    end_date="2024-01-31",
    fields=["close", "volume"]
)

# 获取收盘价（便捷方法）
close_prices = provider.get_price_data(
    instruments="HC8888.XSGE",
    start_date="2024-01-01",
    end_date="2024-01-31"
)

# 列出可用合约
instruments = provider.list_instruments(pattern="HC*")

# 获取交易日历
calendar = provider.get_calendar("HC8888.XSGE")
```

### 2. DataLoader

数据加载器，提供数据加载、预处理和质量检查功能。

#### 基本用法

```python
from src.data import DataLoader

# 初始化（启用前向填充）
loader = DataLoader(fill_method='ffill')

# 加载数据
data = loader.load_data(
    instruments="HC8888.XSGE",
    start_date="2024-01-01",
    end_date="2024-01-31",
    fields=["open", "high", "low", "close", "volume"],
    check_quality=True  # 启用数据质量检查
)

# 批量加载
data_dict = loader.load_batch(
    instruments=["HC8888.XSGE", "RB8888.XSGE"],
    start_date="2024-01-01",
    end_date="2024-01-31",
    show_progress=True
)

# 数据重采样（转换为日线）
daily_data = loader.resample_data(data, freq='1D')
```

#### 缺失值处理方法

- `'ffill'`: 前向填充（默认）
- `'bfill'`: 后向填充
- `'interpolate'`: 线性插值
- `'drop'`: 删除包含缺失值的行
- `'none'`: 不处理

### 3. DataValidator

数据验证器，提供数据质量检查、因子标准化、因子中性化等功能。

#### 数据质量检查

```python
from src.data import DataValidator

validator = DataValidator()

# 检查数据质量
report = validator.check_data_quality(data)
# 报告包含：
# - 缺失值统计
# - 异常值统计
# - 基本统计信息
# - 警告和错误
```

#### 因子标准化

```python
# z-score 标准化
standardized = validator.standardize_factor(factor, method='zscore')

# min-max 标准化
normalized = validator.standardize_factor(factor, method='minmax')

# 秩标准化
ranked = validator.standardize_factor(factor, method='rank')

# 鲁棒标准化
robust = validator.standardize_factor(factor, method='robust')

# 带裁剪的标准化
standardized = validator.standardize_factor(
    factor,
    method='zscore',
    clip_range=(-3, 3)  # 裁剪到 [-3, 3]
)
```

#### 因子中性化

```python
# 市值中性化
neutralized = validator.neutralize_factor(
    factor=factor,
    market_cap=market_cap
)

# 行业中性化
neutralized = validator.neutralize_factor(
    factor=factor,
    industry=industry
)

# 同时市值和行业中性化
neutralized = validator.neutralize_factor(
    factor=factor,
    industry=industry,
    market_cap=market_cap
)
```

#### 去极值

```python
# 去除上下 5% 的极值
winsorized = validator.winsorize_factor(factor, limits=(0.05, 0.05))

# 去除上下 10% 的极值
winsorized = validator.winsorize_factor(factor, limits=(0.1, 0.1))
```

## 快捷函数

```python
from src.data import get_factor_data, load_data, check_data_quality

# 快捷加载数据
data = get_factor_data(
    instruments="HC8888.XSGE",
    start_date="2024-01-01",
    end_date="2024-01-31"
)

# 快捷加载（带缺失值处理）
data = load_data(
    instruments="HC8888.XSGE",
    start_date="2024-01-01",
    end_date="2024-01-31",
    fill_method='ffill'
)

# 快捷检查数据质量
report = check_data_quality(data)
```

## 完整示例

```python
from src.data import FactorDataProvider, DataLoader, DataValidator

# 1. 初始化组件
provider = FactorDataProvider()
loader = DataLoader(fill_method='ffill')
validator = DataValidator()

# 2. 加载数据
data = loader.load_data(
    instruments="HC8888.XSGE",
    start_date="2024-01-01",
    end_date="2024-12-31",
    fields=["open", "high", "low", "close", "volume"],
    check_quality=True
)

# 3. 计算因子
data['momentum'] = data['close'].pct_change(5)

# 4. 标准化因子
data['momentum_std'] = validator.standardize_factor(
    data['momentum'],
    method='zscore',
    clip_range=(-3, 3)
)

# 5. 去极值
data['momentum_final'] = validator.winsorize_factor(
    data['momentum_std'],
    limits=(0.05, 0.05)
)

# 6. 验证最终因子
report = validator.check_data_quality(data[['momentum_final']])
```

## 配置说明

### qlib_backtest 配置

FactorDataProvider 依赖 qlib_backtest 的 ParquetDataProvider。

#### 1. 确保 qlib_backtest 项目在正确位置

```
projects/
├── qlib_backtest/          # 必须在这里
└── multi_factor_analyzer/
```

#### 2. 如果 qlib_backtest 在其他位置

修改 `src/data/provider.py` 中的路径查找逻辑：

```python
# 找到 qlib_backtest 项目路径
current_path = Path(__file__).resolve().parent
project_root = current_path.parent.parent.parent
qlib_backtest_path = project_root / "qlib_backtest"  # 修改这里
```

### 数据目录配置

ParquetDataProvider 会自动查找数据文件：

- **Parquet 数据目录**: `{qlib_backtest}/../K线数据库/期货商品指数_parquet/`
- **Qlib 缓存目录**: `{qlib_backtest}/data/qlib_cache/`

## 注意事项

1. **数据格式**: 返回的数据框索引为 datetime，列名为字段名
2. **多合约数据**: 多合约数据返回 MultiIndex (instrument, datetime)
3. **缺失值**: 建议使用 `fill_method='ffill'` 处理缺失值
4. **因子标准化**: 推荐使用 z-score 标准化
5. **去极值**: 推荐在标准化后进行去极值处理

## 常见问题

### Q: 如何处理合约代码不存在？

A: 使用 `list_instruments()` 先检查可用合约，或捕获异常：

```python
try:
    data = provider.get_factor_data(instruments="HC8888.XSGE", ...)
except ValueError as e:
    print(f"合约不存在: {e}")
```

### Q: 如何批量加载大量合约？

A: 使用 `load_batch()` 方法：

```python
data_dict = loader.load_batch(
    instruments=instruments,  # 可以是几百个合约
    start_date="2024-01-01",
    end_date="2024-12-31",
    show_progress=True
)
```

### Q: 如何选择标准化方法？

A:
- **z-score**: 最常用，适合大多数情况
- **min-max**: 需要 [0, 1] 范围时使用
- **rank**: 非线性因子或存在异常值时使用
- **robust**: 存在较多异常值时使用

### Q: 如何判断因子是否需要中性化？

A: 检查因子与市值、行业的相关性：

```python
# 计算相关性
mcap_corr = factor.corr(market_cap)
if abs(mcap_corr) > 0.1:
    # 需要市值中性化
    factor = validator.neutralize_factor(factor, market_cap=market_cap)
```
