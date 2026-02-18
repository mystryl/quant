# Quant

量化交易项目集合，包含回测系统、数据处理和策略开发。

## ⚠️ 重要提示

**所有回测必须考虑真实交易成本！** 请查看 [BACKTEST_CONFIG.md](BACKTEST_CONFIG.md) 了解回测配置标准，包括：
- 交易手续费（开仓/平仓）
- 滑点成本
- 仓位控制（开仓比例、最大持仓比例）
- 保证金和杠杆
- 风险管理要求

## 项目结构

```
Quant/
├── projects/          # 交易策略和回测项目
│   ├── scripts/       # Python脚本（策略、回测、数据处理）
│   ├── docs/          # 文档
│   ├── data/          # Qlib数据文件
│   ├── backtest_reports/  # 回测报告
│   └── results/       # 结果文件
├── data/              # 数据集和配置
├── frameworks/        # 外部框架依赖（使用 git submodule）
└── TradingAgents-CN/  # 第三方项目（参考）
```

## 子模块

### Projects
交易策略回测系统，使用 Qlib 框架进行多频率回测。

- **策略**: SuperTrend 及其增强版本
- **回测**: 支持多时间周期（1分钟、5分钟、15分钟、60分钟、日线）
- **优化**: 使用 Optuna 进行参数优化

### Data
数据存储和管理。

## 快速开始

### 克隆仓库

```bash
git clone --recurse-submodules https://github.com/mystryl/quant.git
cd quant
```

如果已经克隆但没有获取子模块：

```bash
git submodule update --init --recursive
```

### 环境要求

- Python 3.10+
- Qlib
- 其他依赖见各项目的 requirements.txt

## 开发

### 添加新的 Framework

使用 git submodule 添加外部依赖：

```bash
git submodule add <repository-url> frameworks/<name>
git commit -m "Add framework: <name>"
```

### 提交更改

```bash
git add .
git commit -m "Your commit message"
git push
```

## 许可证

MIT License
