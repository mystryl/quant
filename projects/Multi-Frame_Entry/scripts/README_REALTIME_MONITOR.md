# 实时趋势监控系统

## 概述

实时趋势监控系统使用最新训练的window20模型计算期货趋势方向，并检测最近10根K线的趋势变化。

## 功能特性

- ✅ 使用window20模型预测趋势方向（上涨/下跌/震荡）
- ✅ 检测最近10根K线的趋势变化
- ✅ 支持多个品种同时监控
- ✅ 优先使用本地数据（速度更快）
- ✅ 自动回退到在线数据源（efinance/akshare）
- ✅ 输出格式化的监控报告

## 支持的品种

| 品种代码 | 品种名称 | 模型文件 |
|---------|---------|----------|
| RB0 | 螺纹钢 | HC8888.XSGE_window20.pkl |
| HC0 | 热卷 | HC8888.XSGE_window20.pkl |
| I0 | 铁矿石 | I8888.XDCE_window20.pkl |
| AU0 | 黄金 | AU8888.XSGE_window20.pkl |
| CF0 | 郑棉 | CF8888.XZCE_window20.pkl |

## 安装依赖

```bash
pip install pandas numpy efinance akshare pyarrow
```

## 使用方法

### 1. 监控单个品种

```bash
# 监控螺纹钢
python scripts/realtime_monitor.py --symbol RB0

# 监控热卷
python scripts/realtime_monitor.py --symbol HC0

# 监控铁矿石
python scripts/realtime_monitor.py --symbol I0

# 监控黄金
python scripts/realtime_monitor.py --symbol AU0

# 监控郑棉
python scripts/realtime_monitor.py --symbol CF0
```

### 2. 监控所有品种

```bash
python scripts/realtime_monitor.py --all
```

### 3. 自定义参数

```bash
# 获取150根K线，检测最近15根的变化
python scripts/realtime_monitor.py --symbol RB0 --bars 150 --lookback 15
```

### 参数说明

- `--symbol`: 品种代码（RB0, HC0, I0, AU0, CF0）
- `--all`: 监控所有品种
- `--bars`: 获取K线数量（默认100）
- `--lookback`: 回溯K线数量（默认10）

## 输出示例

```
============================================================
实时趋势监控报告
时间: 2026-02-23 08:49:35
品种: 热卷 (HC0)
============================================================

📊 当前状态:
  信号: 震荡
  价格: 3222.00
  P(上涨): 0.00, P(震荡): 0.95, P(下跌): 0.00

🔔 最近10根K线内的趋势变化:
  无趋势变化

============================================================
```

### 有趋势变化时的输出

```
🔔 最近10根K线内的趋势变化:

  [2根K线前] 15:00:00
  类型: 趋势启动
  变化: 震荡 → 上涨
  价格: 3575.00 → 3580.00 (+5.00, +0.14%)

  [5根K线前] 12:00:00
  类型: 趋势反转
  变化: 下跌 → 上涨
  价格: 3560.00 → 3570.00 (+10.00, +0.28%)

============================================================
```

## 趋势变化类型

| 类型 | 说明 | 示例 |
|------|------|------|
| 趋势启动 | 震荡 → 有趋势 | 震荡 → 上涨、震荡 → 下跌 |
| 趋势反转 | 趋势方向改变 | 上涨 → 下跌、下跌 → 上涨 |
| 趋势结束 | 有趋势 → 震荡 | 上涨 → 震荡、下跌 → 震荡 |

## 文件结构

```
scripts/
├── realtime_monitor.py           # 主监控脚本
├── realtime_data_fetcher.py      # 数据获取模块
├── trend_change_detector.py      # 趋势变化检测模块
└── README_REALTIME_MONITOR.md    # 本文档
```

## 数据源优先级

1. **本地数据** (最快) - 从Dropbox本地文件夹加载
2. **efinance** (次优) - 在线获取期货数据
3. **akshare** (备用) - 如果efinance失败则使用

## 定时任务设置

### Linux/Mac (crontab)

```bash
# 编辑crontab
crontab -e

# 添加每小时执行一次的任务
0 * * * * cd /Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry && python scripts/realtime_monitor.py --all >> /tmp/realtime_monitor.log 2>&1
```

### Windows (任务计划程序)

1. 打开"任务计划程序"
2. 创建基本任务
3. 设置触发器（例如：每小时）
4. 设置操作：运行python脚本
   - 程序：`python`
   - 参数：`scripts/realtime_monitor.py --all`
   - 起始于：`/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry`

## 故障排除

### 问题1: 模型文件不存在

**错误信息**: `FileNotFoundError: 模型文件不存在: xxx_window20.pkl`

**解决方法**:
- 检查`models/rolling_3month/`目录下是否有对应的模型文件
- 运行训练脚本生成模型

### 问题2: 本地数据不存在

**错误信息**: `本地文件不存在: xxx.parquet`

**解决方法**:
- 检查Dropbox同步是否正常
- 系统会自动回退到在线数据源（efinance/akshare）

### 问题3: 在线数据获取失败

**错误信息**: `efinance获取数据失败` 或 `akshare获取数据失败`

**解决方法**:
- 检查网络连接
- 确认数据源API是否正常
- 系统会自动尝试其他数据源

## 性能优化

### 使用本地数据

默认情况下，系统优先使用本地数据，速度最快：

```python
fetcher = RealtimeDataFetcher(preferred_source='local')
```

### 使用在线数据

如果需要获取最新的在线数据：

```python
fetcher = RealtimeDataFetcher(preferred_source='efinance')
```

## 扩展功能

### 添加新品种

1. 在`SYMBOL_CONFIG`中添加新品种配置
2. 确保模型文件存在于`models/rolling_3month/`
3. 确保数据文件存在于本地数据目录

示例：

```python
SYMBOL_CONFIG = {
    'RB0': {'full_code': 'HC8888.XSGE', 'name': '螺纹钢', 'model_code': 'HC888'},
    # 添加新品种
    'CU0': {'full_code': 'CU8888.XSGE', 'name': '铜', 'model_code': 'CU888'},
}
```

### 自定义报告格式

修改`format_monitor_report()`函数来自定义输出格式。

## 技术细节

### 特征计算

- 使用`TrendFeatures`类计算57个技术指标
- 特征经过`shift(1)`处理避免未来函数污染
- 有效特征样本约70条（100根K线减去特征计算损失）

### 模型预测

- 使用window20二分类模型预测P(有趋势)
- 使用MACD直方图判断趋势方向
- 组合生成三分类信号（上涨/下跌/震荡）

### 变化检测

- 检测最近N根K线的信号变化
- 只关注有意义的变化类型（趋势启动、反转、结束）
- 忽略"震荡延续"类型

## 相关文档

- [特征工程](../features/trend_features.py)
- [模型训练](../models/rolling_3month/)
- [数据获取参考](/Users/mystryl/Library/CloudStorage/Dropbox/润富/钢铁/code/期货相关代码/futures_data_fetcher/)

## 更新日志

### 2026-02-23
- ✅ 初始版本发布
- ✅ 支持本地数据优先加载
- ✅ 支持5个品种监控
- ✅ 实现趋势变化检测
- ✅ 输出格式化报告

## 许可证

内部使用
