# 多因子量化分析系统 - CLI 使用指南

## 目录

- [安装](#安装)
- [快速开始](#快速开始)
- [命令详解](#命令详解)
- [配置文件](#配置文件)
- [使用示例](#使用示例)
- [常见问题](#常见问题)

---

## 安装

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

主要依赖：
- click: 命令行框架
- rich: 终端美化
- yaml: 配置文件解析
- pandas, numpy: 数据处理

### 2. 验证安装

```bash
python -m src.cli.main --version
```

预期输出：
```
Multi-Factor Analyzer version 1.0.0
```

---

## 快速开始

### 1. 查看帮助

```bash
# 查看主帮助
python -m src.cli.main --help

# 查看子命令帮助
python -m src.cli.main analyze --help
python -m src.cli.main batch --help
python -m src.cli.main report --help
python -m src.cli.main validate --help
```

### 2. 验证因子表达式

```bash
python -m src.cli.main validate "Ref($close, 20) / $close - 1"
```

预期输出：
```
成功: 表达式验证通过！
  - 未检测到未来函数
  - 语法检查通过
```

### 3. 分析单个因子

```bash
python -m src.cli.main analyze \
  --factor "Ref($close, 20) / $close - 1" \
  --instruments SH600000 \
  --start 2020-01-01 \
  --end 2020-12-31
```

### 4. 批量分析因子

```bash
python -m src.cli.main batch --config examples/factors_config.yaml
```

---

## 命令详解

### 主命令 `mfa`

```bash
python -m src.cli.main [OPTIONS] COMMAND [ARGS]
```

**全局选项：**

| 选项 | 说明 |
|------|------|
| `--version` | 显示版本信息 |
| `--verbose, -v` | 显示详细输出 |
| `--help` | 显示帮助信息 |

**可用命令：**

| 命令 | 说明 |
|------|------|
| `analyze` | 分析单个因子的性能 |
| `batch` | 批量分析多个因子 |
| `report` | 生成分析报告 |
| `validate` | 验证因子表达式 |

---

### `analyze` 命令

分析单个因子的性能。

**语法：**

```bash
python -m src.cli.main analyze [OPTIONS]
```

**必需参数：**

| 参数 | 说明 |
|------|------|
| `--factor, -f TEXT` | 因子名称或表达式 |
| `--instruments, -i TEXT` | 股票代码或代码文件路径 |
| `--start, -s TEXT` | 开始日期 (YYYY-MM-DD) |
| `--end, -e TEXT` | 结束日期 (YYYY-MM-DD) |

**可选参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output, -o TEXT` | 输出文件路径（JSON 或 CSV） | - |
| `--quantile, -q FLOAT` | 多空分组分位数 | 0.2 |
| `--top-pct FLOAT` | 选股比例 | 0.2 |
| `--strategy TEXT` | 策略类型 | all |
| `--no-cache` | 禁用缓存 | False |
| `--verbose, -v` | 显示详细输出 | False |

**策略类型：**

- `all`: 分析所有策略（默认）
- `bull`: 看涨策略
- `bear`: 看跌策略
- `long_short`: 多空策略
- `volatility`: 波动率策略

**示例：**

```bash
# 1. 使用表达式分析
python -m src.cli.main analyze \
  -f "Ref($close, 20) / $close - 1" \
  -i SH600000 \
  -s 2020-01-01 \
  -e 2020-12-31

# 2. 使用已注册的因子
python -m src.cli.main analyze \
  -f MA20 \
  -i instruments.txt \
  -s 2020-01-01 \
  -e 2020-12-31

# 3. 只分析看涨策略
python -m src.cli.main analyze \
  -f MA20 \
  -i SH600000 \
  -s 2020-01-01 \
  -e 2020-12-31 \
  --strategy bull

# 4. 保存结果到文件
python -m src.cli.main analyze \
  -f MA20 \
  -i SH600000 \
  -s 2020-01-01 \
  -e 2020-12-31 \
  -o results.json

# 5. 自定义分位数和选股比例
python -m src.cli.main analyze \
  -f MA20 \
  -i SH600000 \
  -s 2020-01-01 \
  -e 2020-12-31 \
  --quantile 0.3 \
  --top-pct 0.1
```

---

### `batch` 命令

批量分析多个因子。

**语法：**

```bash
python -m src.cli.main batch [OPTIONS]
```

**必需参数：**

| 参数 | 说明 |
|------|------|
| `--config, -c TEXT` | 配置文件路径（YAML 格式） |

**可选参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--output, -o TEXT` | 输出目录路径 | - |
| `--parallel, -p INT` | 并行任务数 | 1 |
| `--verbose, -v` | 显示详细输出 | False |

**示例：**

```bash
# 1. 使用配置文件批量分析
python -m src.cli.main batch --config examples/factors_config.yaml

# 2. 指定输出目录
python -m src.cli.main batch \
  -c examples/factors_config.yaml \
  -o output/results

# 3. 启用并行处理（4个进程）
python -m src.cli.main batch \
  -c examples/factors_config.yaml \
  -p 4
```

**配置文件格式：**

参见 [配置文件](#配置文件) 章节。

---

### `report` 命令

生成分析报告。

**语法：**

```bash
python -m src.cli.main report [OPTIONS]
```

**必需参数：**

| 参数 | 说明 |
|------|------|
| `--input, -i TEXT` | 输入目录或文件路径 |
| `--output, -o TEXT` | 输出报告文件路径 |

**可选参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--format, -f TEXT` | 报告格式（html/pdf/json） | html |
| `--title, -t TEXT` | 报告标题 | 因子分析报告 |

**示例：**

```bash
# 1. 生成 HTML 报告
python -m src.cli.main report \
  -i output/results \
  -o report.html

# 2. 生成 PDF 报告（通过 HTML 转换）
python -m src.cli.main report \
  -i output/results \
  -o report.pdf

# 3. 生成 JSON 报告
python -m src.cli.main report \
  -i output/results \
  -o report.json \
  --format json

# 4. 自定义报告标题
python -m src.cli.main report \
  -i output/results \
  -o report.html \
  --title "2024年度因子分析报告"
```

---

### `validate` 命令

验证因子表达式。

**语法：**

```bash
python -m src.cli.main validate EXPRESSION [OPTIONS]
```

**必需参数：**

| 参数 | 说明 |
|------|------|
| `EXPRESSION` | 因子表达式 |

**可选参数：**

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--verbose, -v` | 显示详细检查信息 | False |

**示例：**

```bash
# 1. 验证简单表达式
python -m src.cli.main validate "Ref($close, 20) / $close - 1"

# 2. 验证复杂表达式
python -m src.cli.main validate "($close - Mean($close, 20)) / Std($close, 20)"

# 3. 显示详细信息
python -m src.cli.main validate "Ref($close, 20) / $close - 1" --verbose

# 4. 验证未来函数（会报错）
python -m src.cli.main validate "Ref($close, -5)"  # 错误：包含未来引用
```

---

## 配置文件

### YAML 配置文件格式

批量分析使用 YAML 格式的配置文件。

**完整示例：**

```yaml
# 因子定义列表
factors:
  - name: MA20
    expression: "Ref($close, 20) / $close - 1"
    description: "20日均线偏离度"

  - name: MA60
    expression: "Ref($close, 60) / $close - 1"
    description: "60日均线偏离度"

# 分析参数配置
analysis:
  # 股票列表文件路径
  instruments: "examples/instruments.txt"

  # 分析时间范围
  start_date: "2020-01-01"
  end_date: "2020-12-31"

  # 性能评估参数
  quantile: 0.2        # 多空分组分位数
  top_pct: 0.2         # 选股比例

# 输出选项
output:
  # 结果保存目录
  dir: "output/batch_results"

  # 输出格式
  format: ["json", "csv"]

  # 是否生成HTML报告
  generate_report: true
```

**配置项说明：**

1. **factors**: 因子列表
   - `name`: 因子名称（唯一标识符）
   - `expression`: 因子表达式字符串
   - `description`: 因子描述（可选）

2. **analysis**: 分析参数
   - `instruments`: 股票代码文件路径
   - `start_date`: 分析开始日期
   - `end_date`: 分析结束日期
   - `quantile`: 多空分组分位数
   - `top_pct`: 选股比例

3. **output**: 输出选项（可选）
   - `dir`: 结果保存目录
   - `format`: 输出格式列表
   - `generate_report`: 是否生成报告

---

## 使用示例

### 示例 1: 快速因子验证

开发新因子时，首先验证表达式是否正确：

```bash
# 验证动量因子
python -m src.cli.main validate "$close / Ref($close, 20) - 1"

# 验证波动率因子
python -m src.cli.main validate "Std($close, 20) / Mean($close, 20)"

# 验证复合因子
python -m src.cli.main validate "($close / Ref($close, 5) - 1) * ($volume / Mean($volume, 20))"
```

### 示例 2: 单因子完整分析

分析一个动量因子在不同策略下的表现：

```bash
python -m src.cli.main analyze \
  --factor "$close / Ref($close, 20) - 1" \
  --instruments examples/instruments.txt \
  --start 2020-01-01 \
  --end 2020-12-31 \
  --strategy all \
  --quantile 0.2 \
  --top-pct 0.2 \
  --output momentum_analysis.json
```

### 示例 3: 批量因子筛选

批量测试多个因子，筛选出表现最好的：

**步骤 1**: 创建配置文件 `screening.yaml`

```yaml
factors:
  - name: MA5
    expression: "Ref($close, 5) / $close - 1"
    description: "5日均线偏离度"

  - name: MA10
    expression: "Ref($close, 10) / $close - 1"
    description: "10日均线偏离度"

  - name: MA20
    expression: "Ref($close, 20) / $close - 1"
    description: "20日均线偏离度"

  - name: MA60
    expression: "Ref($close, 60) / $close - 1"
    description: "60日均线偏离度"

analysis:
  instruments: "examples/instruments.txt"
  start_date: "2020-01-01"
  end_date: "2020-12-31"
  quantile: 0.2
  top_pct: 0.2
```

**步骤 2**: 运行批量分析

```bash
python -m src.cli.main batch --config screening.yaml --output output/screening
```

**步骤 3**: 生成对比报告

```bash
python -m src.cli.main report \
  -i output/screening \
  -o output/screening/report.html \
  --title "均线因子筛选报告"
```

### 示例 4: 因子优化

调整因子参数，找到最优配置：

```bash
# 测试不同的动量周期
for period in 5 10 15 20 30 60; do
  python -m src.cli.main analyze \
    --factor "$close / Ref($close, $period) - 1" \
    --instruments examples/instruments.txt \
    --start 2020-01-01 \
    --end 2020-12-31 \
    --output "momentum_${period}d.json"
done
```

### 示例 5: 回测验证

在历史数据上验证因子表现：

```bash
# 2020年表现
python -m src.cli.main analyze \
  --factor MA20 \
  --instruments examples/instruments.txt \
  --start 2020-01-01 \
  --end 2020-12-31 \
  --output ma20_2020.json

# 2021年表现
python -m src.cli.main analyze \
  --factor MA20 \
  --instruments examples/instruments.txt \
  --start 2021-01-01 \
  --end 2021-12-31 \
  --output ma20_2021.json

# 生成对比报告
python -m src.cli.main report \
  -i . \
  -o ma20_comparison.html \
  --title "MA20因子年度对比"
```

---

## 常见问题

### 1. 如何处理大量股票？

使用股票代码文件，每行一个代码：

```bash
# 创建 instruments.txt
cat > instruments.txt << EOF
SH600000
SH600001
SH600004
...
EOF

# 使用文件
python -m src.cli.main analyze \
  --factor MA20 \
  --instruments instruments.txt \
  --start 2020-01-01 \
  --end 2020-12-31
```

### 2. 如何加速批量分析？

使用并行处理（注意：需要足够的数据源支持）：

```bash
python -m src.cli.main batch \
  --config factors.yaml \
  --parallel 4
```

### 3. 如何禁用缓存？

对于调试或数据更新频繁的场景：

```bash
python -m src.cli.main analyze \
  --factor MA20 \
  --instruments SH600000 \
  --start 2020-01-01 \
  --end 2020-12-31 \
  --no-cache
```

### 4. 输出结果格式说明

**JSON 格式：**

```json
{
  "factor": "MA20",
  "instruments": ["SH600000", "SH600001"],
  "start_date": "2020-01-01",
  "end_date": "2020-12-31",
  "metrics": {
    "ic_mean": 0.0523,
    "ic_std": 0.1234,
    "icir": 0.4238,
    "rank_ic_mean": 0.0567,
    "rank_ic_std": 0.1156,
    "rank_icir": 0.4905,
    "annual_return": 0.0856,
    "sharpe_ratio": 1.2345,
    "max_drawdown": 0.1234,
    "win_rate": 0.5432
  },
  "strategy_results": {
    "bull": {
      "annual_return": 0.0856,
      "sharpe_ratio": 1.2345,
      "max_drawdown": 0.1234,
      "win_rate": 0.5432
    }
  }
}
```

**CSV 格式：**

包含因子基本信息和主要指标，适合 Excel 打开。

### 5. 错误处理

**常见错误及解决方案：**

| 错误信息 | 原因 | 解决方案 |
|---------|------|---------|
| `因子未注册` | 因子名称不存在 | 使用正确的因子名或表达式 |
| `表达式包含未来函数` | 检测到未来数据引用 | 修改表达式，避免使用 `Ref($field, -N)` |
| `数据加载失败` | 数据不存在或格式错误 | 检查数据路径和股票代码 |
| `配置文件格式错误` | YAML 语法错误 | 检查 YAML 缩进和语法 |

### 6. 性能优化建议

1. **使用缓存**: 默认启用缓存，重复分析同一因子会更快
2. **批量分析**: 对于多个因子，使用 `batch` 命令而非多次调用 `analyze`
3. **合理选择时间范围**: 避免过大的时间范围导致内存占用过高
4. **并行处理**: 在数据源支持的情况下使用 `--parallel` 选项

---

## 高级技巧

### 1. 使用管道处理结果

```bash
# 提取 ICIR > 0.5 的因子
python -m src.cli.main batch --config factors.yaml \
  | jq '.results[] | select(.metrics.icir > 0.5) | .name'
```

### 2. 定时任务

使用 cron 定期执行因子分析：

```bash
# 每天凌晨 2 点执行
0 2 * * * cd /path/to/mfa && python -m src.cli.main batch --config daily_factors.yaml
```

### 3. 日志记录

重定向输出到日志文件：

```bash
python -m src.cli.main analyze \
  --factor MA20 \
  --instruments instruments.txt \
  --start 2020-01-01 \
  --end 2020-12-31 \
  > analysis.log 2>&1
```

---

## 附录

### A. 支持的字段

| 字段 | 说明 |
|------|------|
| `$open` | 开盘价 |
| `$high` | 最高价 |
| `$low` | 最低价 |
| `$close` | 收盘价 |
| `$volume` | 成交量 |
| `$money` | 成交额 |
| `$avg` | 均价 |
| `$open_interest` | 持仓量（期货） |

### B. 支持的函数

| 函数 | 说明 | 示例 |
|------|------|------|
| `Ref($field, N)` | 引用 N 周期前的值 | `Ref($close, 20)` |
| `Mean($field, N)` | N 周期均值 | `Mean($close, 20)` |
| `Std($field, N)` | N 周期标准差 | `Std($close, 20)` |
| `Max($field, N)` | N 周期最大值 | `Max($high, 20)` |
| `Min($field, N)` | N 周期最小值 | `Min($low, 20)` |

### C. 性能指标说明

| 指标 | 说明 | 优秀标准 |
|------|------|---------|
| IC 均值 | 因子预测能力 | > 0.03 |
| ICIR | IC 稳定性 | > 0.5 |
| Rank IC 均值 | 排名预测能力 | > 0.04 |
| 年化收益率 | 策略年化收益 | > 5% |
| 夏普比率 | 风险调整收益 | > 1.5 |
| 最大回撤 | 最大损失幅度 | < 15% |
| 胜率 | 盈利天数占比 | > 55% |

---

**文档版本**: 1.0.0
**最后更新**: 2024-03-19
