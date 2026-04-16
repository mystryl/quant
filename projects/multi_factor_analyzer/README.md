# 多因子量化分析系统

## 项目简介

这是一个基于 Qlib 的多因子量化分析系统，能够：
- 给定因子，根据回测数据计算 IC、IR 等性能指标
- 评估因子可靠性，考虑周期对齐
- 支持多种策略场景（看涨、看跌、波动率策略）
- 避免未来函数，仅在性能评估时使用未来收益率

## 核心特性

### 1. 未来函数检测
- 静态分析验证因子表达式
- 自动检测 Ref($close, -N) 等未来引用
- 在计算前进行验证，避免数据泄露

### 2. 性能评估
- IC (Information Coefficient) - 信息系数
- IR (Information Ratio) - 信息比率
- ICIR - IC 信息比率
- 多空收益、胜率、最大回撤

### 3. 周期对齐
- 默认采用 Qlib 的 T+1 到 T+2 设计
- 支持自动检测最优周期
- 支持手动指定对齐方式

### 4. 策略场景分析
- 看涨策略（做多高因子值股票）
- 看跌策略（做空低因子值股票）
- 波动率策略（根据市场波动率调整仓位）
- 牛熊市、行业轮动、市值分组

### 5. 可靠性评估
- 可配置的权重系统（默认/保守/激进）
- 因子相关性分析
- 综合评分和建议

## 安装

```bash
# 克隆项目
git clone <repo_url>
cd multi_factor_analyzer

# 安装依赖
pip install -r requirements.txt

# 安装项目
pip install -e .
```

## 快速开始

### Python API

```python
from multi_factor_analyzer import FactorManager, PerformanceEvaluator

# 创建因子管理器
factor_manager = FactorManager(data_provider)

# 计算因子
factor_values = factor_manager.calculate_factor(
    name="momentum_20d",
    instruments=["SH600000", "SH600001"],
    start_date="2020-01-01",
    end_date="2020-12-31"
)

# 评估性能
evaluator = PerformanceEvaluator()
metrics = evaluator.calculate_all(factor_values, forward_returns)

print(f"IC Mean: {metrics['ic_mean']:.4f}")
print(f"ICIR: {metrics['icir']:.4f}")
```

### 命令行接口 (CLI)

#### 快速开始

```bash
# 查看帮助
python -m src.cli.main --help

# 验证因子表达式
python -m src.cli.main validate "Ref(\$close, 20) / \$close - 1"

# 分析单个因子
python -m src.cli.main analyze \
  --factor "Ref(\$close, 20) / \$close - 1" \
  --instruments examples/instruments.txt \
  --start 2020-01-01 \
  --end 2020-12-31

# 批量分析
python -m src.cli.main batch --config examples/factors_config.yaml

# 生成报告
python -m src.cli.main report \
  --input output/batch_results \
  --output report.html
```

#### 主要命令

| 命令 | 说明 | 示例 |
|------|------|------|
| `analyze` | 分析单个因子 | `python -m src.cli.main analyze -f MA20 -i stocks.txt -s 2020-01-01 -e 2020-12-31` |
| `batch` | 批量分析多个因子 | `python -m src.cli.main batch --config factors.yaml` |
| `report` | 生成分析报告 | `python -m src.cli.main report -i results/ -o report.html` |
| `validate` | 验证因子表达式 | `python -m src.cli.main validate "Ref(\$close, 20) / \$close - 1"` |

#### 更多信息

详细使用文档请参考：[CLI 使用指南](docs/CLI_GUIDE.md) | [CLI 实现总结](CLI_IMPLEMENTATION_SUMMARY.md)

## 文档

### 系统文档
- [系统设计文档](SYSTEM_DESIGN.md) - 完整的系统架构和设计决策
- [设计优化总结](DESIGN_IMPROVEMENTS.md) - 最近的优化和改进
- [CLI 实现总结](CLI_IMPLEMENTATION_SUMMARY.md) - 命令行接口实现详情

### API 文档
- [API 文档](docs/API.md) - API 使用说明
- [用户指南](docs/USER_GUIDE.md) - 详细使用教程
- [CLI 使用指南](docs/CLI_GUIDE.md) - 命令行接口完整指南 (700+ 行)

### 示例和教程
- [CLI 示例](examples/README.md) - 命令行使用示例
- [快速开始](quickstart.py) - Python API 快速入门

## 项目结构

```
multi_factor_analyzer/
├── src/                    # 源代码
│   ├── core/              # 核心模块
│   │   ├── factor_engine.py
│   │   ├── cycle_aligner.py
│   │   ├── performance_eval.py
│   │   ├── strategy_analyzer.py
│   │   └── reliability.py
│   ├── data/              # 数据层
│   │   ├── provider.py
│   │   ├── loader.py
│   │   └── validator.py
│   ├── report/            # 报告生成
│   │   ├── generator.py
│   │   └── visualizer.py
│   ├── cli/               # 命令行接口
│   │   └── main.py
│   └── utils/             # 工具函数
│       ├── guard.py
│       └── helpers.py
├── tests/                 # 测试
├── examples/              # 示例
└── docs/                  # 文档
```

## 开发

```bash
# 运行测试
pytest tests/

# 代码格式化
black src/

# 代码检查
flake8 src/
```

## 贡献

欢迎提交 Issue 和 Pull Request！

## 许可证

MIT License
