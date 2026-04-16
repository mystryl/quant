# 报告生成和可视化模块

本模块提供因子分析报告的生成和可视化功能。

## 模块结构

```
src/report/
├── __init__.py           # 模块导出
├── generator.py          # 报告生成器
└── visualizer.py         # 可视化工具
```

## 主要功能

### 1. 报告生成器 (ReportGenerator)

生成因子分析报告，支持多种格式：

- **Markdown**: 适合文档编写和版本控制
- **HTML**: 适合网页展示和在线查看
- **纯文本**: 适合终端输出和日志记录
- **JSON**: 适合程序化处理和数据交换

#### 报告内容

1. **执行摘要**: 可靠性等级、关键指标
2. **IC 分析**: IC 时间序列、IC 分布、ICIR
3. **IR 分析**: 多空收益、夏普比率、最大回撤
4. **策略场景**: 看涨/看跌/波动率环境表现
5. **周期分析**: 最佳周期、周期对齐效果
6. **稳定性分析**: 时间稳定性、市场环境稳定性
7. **建议**: 是否使用、如何优化

### 2. 可视化工具 (Visualizer)

生成专业的因子分析图表：

- **IC 时间序列图**: 展示 IC 随时间的变化
- **IC 分布直方图**: 展示 IC 的分布特征
- **累计收益曲线**: 展示策略的累计收益
- **回撤图**: 展示策略的回撤情况
- **策略对比图**: 对比不同策略的表现
- **周期对比图**: 对比不同对齐周期的效果
- **滚动 IC 图**: 展示 IC 的滚动统计
- **月度 IC 热力图**: 展示 IC 在不同月份的表现
- **IC Q-Q 图**: 检验 IC 是否服从正态分布
- **综合报告图**: 包含多个子图的综合图表

## 快速开始

### 安装依赖

```bash
pip install matplotlib seaborn scipy
```

### 基本使用

#### 1. 生成报告

```python
from report.generator import ReportGenerator

# 创建报告生成器
generator = ReportGenerator(output_dir="./reports")

# 生成完整报告
reports = generator.generate_full_report(
    factor_name="MA20",
    metrics=metrics,                    # 性能指标
    scenario_results=scenario_results,  # 策略场景结果
    reliability_result=reliability_result,  # 可靠性评估（可选）
    cycle_analysis=cycle_analysis,      # 周期分析（可选）
    factor_description="基于20日移动平均线的趋势因子",
    backtest_period=('2020-01-01', '2020-12-31')
)

# 保存不同格式的报告
generator.save_markdown(reports['markdown'], "MA20_report.md")
generator.save_html(reports['html'], "MA20_report.html")
generator.save_text(reports['text'], "MA20_report.txt")
generator.save_json(reports['json'], "MA20_report.json")
```

#### 2. 生成图表

```python
from report.visualizer import Visualizer

# 创建可视化工具
viz = Visualizer(output_dir="./charts")

# 生成单个图表
fig = viz.plot_ic_timeseries(
    metrics['ic'],
    factor_name="MA20",
    window=20
)
viz.save_figure(fig, "MA20_ic_timeseries.png", dpi=300)

# 生成所有图表
chart_paths = viz.save_all_charts(
    metrics=metrics,
    scenario_results=scenario_results,
    factor_name="MA20"
)
```

#### 3. 快捷方式

```python
from report import generate_factor_report, create_factor_charts

# 快速生成报告
reports = generate_factor_report(
    factor_name="MA20",
    metrics=metrics,
    scenario_results=scenario_results,
    output_dir="./reports"
)

# 快速生成图表
charts = create_factor_charts(
    metrics=metrics,
    scenario_results=scenario_results,
    factor_name="MA20",
    output_dir="./charts"
)
```

## 完整示例

```python
import numpy as np
import pandas as pd
from report.generator import ReportGenerator
from report.visualizer import Visualizer

# 1. 准备数据
# 假设你已经通过因子分析得到了性能指标
metrics = {
    'ic': pd.Series([...]),      # IC 时间序列
    'ic_mean': 0.035,            # IC 均值
    'ic_std': 0.052,             # IC 标准差
    'icir': 0.673,               # ICIR
    'annual_return': 0.12,       # 年化收益
    'sharpe_ratio': 1.95,        # 夏普比率
    'max_drawdown': -0.11,       # 最大回撤
    'win_rate': 0.58,            # 胜率
    # ... 更多指标
}

scenario_results = {
    'bull': {
        'strategy_returns': pd.Series([...]),
        'annual_return': 0.15,
        'sharpe_ratio': 2.2,
        # ... 更多指标
    },
    # ... 其他策略
}

# 2. 生成报告
generator = ReportGenerator(output_dir="./reports")
reports = generator.generate_full_report(
    factor_name="MA20 移动平均线因子",
    metrics=metrics,
    scenario_results=scenario_results,
    factor_description="基于20日移动平均线的趋势因子"
)

# 保存报告
generator.save_html(reports['html'], "MA20_report.html")

# 3. 生成图表
viz = Visualizer(output_dir="./charts")

# IC 时间序列图
fig = viz.plot_ic_timeseries(metrics['ic'], factor_name="MA20")
viz.save_figure(fig, "MA20_ic.png")

# 累计收益曲线
fig = viz.plot_cumulative_return(
    scenario_results['bull']['strategy_returns'],
    factor_name="MA20"
)
viz.save_figure(fig, "MA20_return.png")

# 或一次性生成所有图表
chart_paths = viz.save_all_charts(
    metrics=metrics,
    scenario_results=scenario_results,
    factor_name="MA20"
)
```

## 图表样式

### 配色方案

```python
COLOR_SCHEME = {
    'primary': '#2c3e50',      # 深蓝
    'secondary': '#3498db',    # 蓝
    'success': '#27ae60',      # 绿
    'warning': '#f39c12',      # 橙
    'danger': '#c0392b',       # 红
    'info': '#16a085',         # 青
    'purple': '#8e44ad',       # 紫
    'gray': '#95a5a6',         # 灰
}
```

### 中文字体支持

图表支持中文显示，会尝试使用以下字体：

- Arial Unicode MS
- SimHei
- DejaVu Sans

如果系统中没有这些字体，中文可能显示为方块。可以通过安装中文字体或修改 `visualizer.py` 中的字体配置来解决。

## 输出格式

### 图片格式

支持多种图片格式：

- **PNG**: 默认格式，适合网络传输
- **SVG**: 矢量图，适合打印和编辑
- **PDF**: 适合文档嵌入
- **JPG**: 有损压缩，文件较小

```python
viz.save_figure(fig, "chart.png", format="png", dpi=300)
viz.save_figure(fig, "chart.svg", format="svg")
viz.save_figure(fig, "chart.pdf", format="pdf")
```

### DPI 设置

```python
# 标准分辨率 (100 DPI)
viz.save_figure(fig, "chart.png")

# 高分辨率 (300 DPI)
viz.save_figure(fig, "chart.png", dpi=300)

# 超高分辨率 (600 DPI)
viz.save_figure(fig, "chart.png", dpi=600)
```

## API 参考

### ReportGenerator

#### 初始化

```python
generator = ReportGenerator(
    output_dir="./reports",      # 输出目录
    include_charts=True,         # HTML 报告中是否包含图表
    date_format="%Y-%m-%d"       # 日期格式
)
```

#### 主要方法

- `generate_full_report()`: 生成完整报告
- `save_markdown()`: 保存 Markdown 报告
- `save_html()`: 保存 HTML 报告
- `save_text()`: 保存纯文本报告
- `save_json()`: 保存 JSON 报告

### Visualizer

#### 初始化

```python
viz = Visualizer(
    output_dir="./charts",       # 输出目录
    dpi=100,                     # 默认分辨率
    style='seaborn-whitegrid',   # 图表样式
    figsize=(12, 6)             # 默认图表大小
)
```

#### 主要方法

- `plot_ic_timeseries()`: 绘制 IC 时间序列图
- `plot_ic_distribution()`: 绘制 IC 分布直方图
- `plot_cumulative_return()`: 绘制累计收益曲线
- `plot_drawdown()`: 绘制回撤图
- `plot_strategy_comparison()`: 绘制策略对比图
- `plot_cycle_comparison()`: 绘制周期对比图
- `plot_rolling_ic()`: 绘制滚动 IC 图
- `plot_monthly_ic_heatmap()`: 绘制月度 IC 热力图
- `plot_ic_qq()`: 绘制 IC Q-Q 图
- `create_full_report_chart()`: 创建综合报告图
- `save_figure()`: 保存图表
- `save_all_charts()`: 保存所有标准图表

## 测试

运行测试：

```bash
# 测试报告生成器
pytest tests/report/test_generator.py -v

# 测试可视化工具
pytest tests/report/test_visualizer.py -v

# 测试所有报告模块
pytest tests/report/ -v
```

## 示例

查看完整示例：

```bash
python examples/report/example_report_generation.py
```

这将生成示例报告和图表，保存在 `output/report_example/` 目录。

## 注意事项

1. **中文字体**: 如果中文显示为方块，需要安装中文字体或修改字体配置
2. **内存管理**: 生成大量图表时注意内存使用，建议及时关闭不再使用的 Figure 对象
3. **文件路径**: 输出目录会自动创建，如果已存在同名文件会被覆盖
4. **数据格式**: 确保输入的数据格式正确，特别是时间序列的索引应为 DatetimeIndex
5. **性能**: 生成大量高分辨率图表可能较慢，可以降低 DPI 或使用多进程加速

## 扩展

### 自定义图表样式

```python
# 修改默认图表大小
viz = Visualizer(figsize=(16, 8))

# 修改默认样式
viz = Visualizer(style='ggplot')

# 修改配色方案
from report.visualizer import COLOR_SCHEME
COLOR_SCHEME['primary'] = '#your-color'
```

### 自定义报告模板

```python
# 生成报告数据
reports = generator.generate_full_report(...)

# 自定义处理
markdown_content = reports['markdown']
# 添加自定义内容
markdown_content += "\n## 自定义分析\n\n..."

# 保存
with open("custom_report.md", "w") as f:
    f.write(markdown_content)
```

## 常见问题

### Q: 如何在 HTML 报告中嵌入图表？

A: 目前需要手动将生成的图表嵌入 HTML 报告。可以在生成 HTML 后，使用编辑器或脚本将图片链接添加到 HTML 中。

### Q: 如何批量生成多个因子的报告？

A: 使用循环处理每个因子：

```python
factors = {
    'MA20': (metrics1, scenario_results1),
    'MA60': (metrics2, scenario_results2),
    # ...
}

for factor_name, (metrics, scenario_results) in factors.items():
    reports = generator.generate_full_report(
        factor_name=factor_name,
        metrics=metrics,
        scenario_results=scenario_results
    )
    generator.save_html(reports['html'], f"{factor_name}_report.html")
```

### Q: 图表中文显示为方块怎么办？

A: 安装中文字体，例如：

```bash
# macOS
brew install --cask font-source-han-sans

# Ubuntu/Debian
sudo apt-get install fonts-wqy-zenhei

# Windows
# 通常已安装中文字体，只需修改配置
```

然后在 `visualizer.py` 中修改字体配置：

```python
plt.rcParams['font.sans-serif'] = ['Your Font Name', 'SimHei']
```

## 更新日志

### v1.0.0 (2024-03-19)

- 初始版本发布
- 支持多种报告格式（Markdown、HTML、文本、JSON）
- 支持 10+ 种图表类型
- 完整的单元测试覆盖
- 详细的文档和示例
