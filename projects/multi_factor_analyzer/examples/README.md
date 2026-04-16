# CLI 使用示例

本目录包含多因子量化分析系统的命令行接口使用示例。

## 文件说明

- **factors_config.yaml**: 批量因子分析配置文件示例
- **instruments.txt**: 股票代码列表示例
- **cli_demo.sh**: CLI 命令演示脚本

## 快速开始

### 1. 验证因子表达式

```bash
# 验证动量因子
python -m src.cli.main validate "\$close / Ref(\$close, 20) - 1"

# 验证波动率因子
python -m src.cli.main validate "Std(\$close, 20) / Mean(\$close, 20)"

# 测试未来函数检测（应该报错）
python -m src.cli.main validate "Ref(\$close, -5)"
```

### 2. 分析单个因子

```bash
# 使用表达式分析
python -m src.cli.main analyze \
  --factor "\$close / Ref(\$close, 20) - 1" \
  --instruments instruments.txt \
  --start 2020-01-01 \
  --end 2020-12-31

# 分析看涨策略并保存结果
python -m src.cli.main analyze \
  --factor MA20 \
  --instruments instruments.txt \
  --start 2020-01-01 \
  --end 2020-12-31 \
  --strategy bull \
  --output ma20_results.json
```

### 3. 批量分析多个因子

```bash
# 使用配置文件批量分析
python -m src.cli.main batch --config factors_config.yaml

# 指定输出目录
python -m src.cli.main batch \
  --config factors_config.yaml \
  --output output/results
```

### 4. 生成分析报告

```bash
# 生成 HTML 报告
python -m src.cli.main report \
  --input output/results \
  --output report.html

# 生成 JSON 报告
python -m src.cli.main report \
  --input output/results \
  --output report.json \
  --format json
```

## 配置文件说明

### factors_config.yaml

批量分析配置文件，包含以下部分：

1. **factors**: 因子定义列表
   - `name`: 因子名称
   - `expression`: 因子表达式
   - `description`: 因子描述

2. **analysis**: 分析参数
   - `instruments`: 股票列表文件
   - `start_date`: 开始日期
   - `end_date`: 结束日期
   - `quantile`: 多空分组分位数
   - `top_pct`: 选股比例

3. **output**: 输出选项（可选）
   - `dir`: 结果保存目录
   - `format`: 输出格式
   - `generate_report`: 是否生成报告

### instruments.txt

股票代码列表文件，每行一个代码：

```
SH600000
SH600001
SH600004
SH600519
SZ000001
SZ000002
```

## 示例因子

### 1. 均线偏离度因子

```python
# 20日均线偏离度
MA20 = "Ref($close, 20) / $close - 1"

# 60日均线偏离度
MA60 = "Ref($close, 60) / $close - 1"
```

### 2. 动量因子

```python
# 20日动量
Momentum20 = "$close / Ref($close, 20) - 1"

# 60日动量
Momentum60 = "$close / Ref($close, 60) - 1"
```

### 3. 波动率因子

```python
# 20日波动率
Volatility20 = "Std($close, 20) / Mean($close, 20)"

# 60日波动率
Volatility60 = "Std($close, 60) / Mean($close, 60)"
```

### 4. 成交量因子

```python
# 成交量变化率
VolumeChange = "$volume / Ref($volume, 5) - 1"

# 成交量均值比
VolumeRatio = "$volume / Mean($volume, 20)"
```

## 运行演示脚本

```bash
# 运行完整演示
./cli_demo.sh

# 或使用 bash
bash cli_demo.sh
```

## 常见使用场景

### 场景 1: 因子验证

开发新因子时，首先验证表达式：

```bash
# 验证表达式语法
python -m src.cli.main validate "your_expression_here"

# 查看详细信息
python -m src.cli.main validate "your_expression_here" --verbose
```

### 场景 2: 因子筛选

批量测试多个因子，筛选最优：

```bash
# 1. 创建配置文件 my_factors.yaml
# 2. 运行批量分析
python -m src.cli.main batch --config my_factors.yaml

# 3. 生成对比报告
python -m src.cli.main report \
  -i output/batch_results \
  -o comparison_report.html
```

### 场景 3: 参数优化

测试不同参数组合：

```bash
# 测试不同的动量周期
for period in 5 10 15 20 30; do
  python -m src.cli.main analyze \
    --factor "\$close / Ref(\$close, $period) - 1" \
    --instruments instruments.txt \
    --start 2020-01-01 \
    --end 2020-12-31 \
    --output "momentum_${period}d.json"
done
```

### 场景 4: 回测验证

在不同时间段验证因子表现：

```bash
# 2020年
python -m src.cli.main analyze \
  --factor MA20 \
  --instruments instruments.txt \
  --start 2020-01-01 \
  --end 2020-12-31 \
  --output ma20_2020.json

# 2021年
python -m src.cli.main analyze \
  --factor MA20 \
  --instruments instruments.txt \
  --start 2021-01-01 \
  --end 2021-12-31 \
  --output ma20_2021.json
```

## 输出文件说明

### JSON 格式

包含完整的分析结果：

```json
{
  "factor": "MA20",
  "instruments": ["SH600000", "SH600001"],
  "start_date": "2020-01-01",
  "end_date": "2020-12-31",
  "metrics": {
    "ic_mean": 0.0523,
    "icir": 0.4238,
    "annual_return": 0.0856,
    ...
  },
  "strategy_results": {...}
}
```

### HTML 报告

美观的交互式报告，包含：
- 汇总统计
- 因子对比表格
- 每个因子的详细指标
- 可视化图表

## 更多信息

详细的使用文档请参考：[CLI_GUIDE.md](../docs/CLI_GUIDE.md)

## 注意事项

1. 确保数据源正确配置
2. 股票代码格式正确（如：SH600000）
3. 日期格式为 YYYY-MM-DD
4. 表达式验证通过后再进行分析
5. 大批量分析建议使用配置文件
6. 定期清理缓存以释放磁盘空间
