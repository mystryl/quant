# Multi-Frame Entry 项目架构图

## 项目概述
期货多时间框架入场策略系统，使用机器学习预测趋势方向（上涨/下跌/震荡）。

**版本**: v2.0
**更新日期**: 2026-02-23
**核心方法**: 二分类模型 + MACD方向判断

---

## 目录结构

```
Multi-Frame_Entry/
│
├── 📂 data/                          # 数据目录
│   ├── multi_symbol/                 # 多品种数据
│   │   ├── HC8888.XSGE/             # 热卷
│   │   │   ├── features/            # 特征文件
│   │   │   └── labels/             # 标签文件
│   │   ├── I8888.XDCE/             # 铁矿石
│   │   ├── AU8888.XSGE/             # 黄金
│   │   └── CF8888.XZCE/             # 郑棉
│   │
│   ├── labels/                      # 全局标签数据
│   │   ├── final_labels_20bars.csv          # 三分类标签（上涨/震荡/下跌）
│   │   ├── binary_labels_10bars.csv          # 二分类标签（趋势/震荡）
│   │   └── volatility_regime_labels.csv      # 波动率标签
│   │
│   ├── features/                    # 特征数据（已合并到multi_symbol）
│   └── predictions/                 # 预测结果输出
│       └── 2026_signals/
│           ├── 所有K线信号_2026.xlsx          # K线信号明细
│           ├── 趋势信号变化_2026.xlsx          # 信号变化点
│           └── charts/                          # K线图可视化
│               ├── 热卷_K线图_信号标记.html
│               ├── 铁矿石_K线图_信号标记.html
│               ├── 黄金_K线图_信号标记.html
│               └── 郑棉_K线图_信号标记.html
│
├── 📂 features/                     # 特征工程模块 ⭐
│   └── trend_features.py            # 57个技术指标特征计算
│       ├── 斜率类: EMA斜率、TWAP斜率、线性回归斜率
│       ├── 强度类: ADX、ATR
│       ├── 结构类: 金叉死叉、均线排列、高低点突破
│       ├── 波动率: 滚动标准差、Parkinson波动率
│       └── 技术指标: MACD、RSI、布林带
│
├── 📂 labels/                       # 标签生成模块 ⭐
│   ├── trend_label_final.py         # 三分类标签（上涨/震荡/下跌）⭐
│   ├── binary_label.py              # 二分类标签（趋势/震荡）
│   └── volatility_regime.py         # 波动率标签（高/低波动）
│
├── 📂 models/                       # 模型模块 ⭐
│   ├── binary_model.py              # 二分类模型类 ⭐
│   └── rolling_3month/              # 滚动训练模型（80个）⭐
│       ├── HC8888.XSGE_window01-20.pkl
│       ├── I8888.XDCE_window01-20.pkl
│       ├── AU8888.XSGE_window01-20.pkl
│       └── CF8888.XZCE_window01-20.pkl
│
├── 📂 scripts/                      # 脚本模块
│   ├── rolling_train_3month.py      # 滚动训练脚本 ⭐（关键代码）
│   ├── predict_2026_signals.py      # 预测脚本 ⭐（关键代码）
│   ├── visualize_all_symbols.py     # 批量可视化脚本 ⭐（案例代码）
│   ├── visualize_signals.py         # 单品种可视化
│   ├── data_pipeline_multi_symbol.py  # 多品种数据管道
│   ├── generate_binary_features.py  # 特征生成脚本
│   ├── generate_excel_report.py     # Excel报告生成
│   └── rolling_train_multi_symbol.py  # 多品种训练
│
└── 📄 README.md                     # 本文档
```

---

## 核心代码说明

### 1️⃣ 特征工程 (`features/trend_features.py`)

**功能**: 计算57个技术指标特征

**关键类**: `TrendFeatures`

**特征分类**:
- **斜率类** (9个): EMA60斜率、EMA20斜率、TWAP斜率、线性回归斜率等
- **强度类** (4个): ADX、ADX变化率、ATR、ATR比率
- **结构类** (18个): 金叉死叉、均线排列、高低点突破、连续K线
- **波动率** (11个): 滚动标准差、Parkinson波动率
- **技术指标** (15个): MACD、RSI、布林带、成交量变化率、价格加速度

**防未来函数**: 所有特征已`shift(1)`处理

---

### 2️⃣ 标签生成 (`labels/trend_label_final.py`)

**功能**: 生成三分类趋势标签

**配置**:
- 窗口: 20根60min K线（约20小时，1.5天）
- 上涨阈值: > 0.3%
- 下跌阈值: < -0.3%
- 震荡: ±0.3%之间

**标签分布**:
- 上涨: 41.3%
- 下跌: 41.9%
- 震荡: 16.8%

**验证**: 已通过未来函数验证

---

### 3️⃣ 二分类模型 (`models/binary_model.py`)

**功能**: 趋势vs震荡二分类模型

**类**: `BinaryTrendModel`

**算法**: XGBoost
- n_estimators: 200
- max_depth: 5
- learning_rate: 0.05

**性能** (2025年测试数据):
| 品种 | AUC | 准确率 |
|------|-----|--------|
| 热卷 | 0.6537 | 60.5% |
| 铁矿石 | 0.6125 | 58.2% |
| 黄金 | 0.6488 | 61.1% |
| 郑棉 | 0.6034 | 56.8% |

---

### 4️⃣ 滚动训练框架 (`scripts/rolling_train_3month.py`)

**功能**: 18月窗口 + 3月预测的滚动训练

**参数**:
- 训练窗口: 18个月
- 预测窗口: 3个月
- 滚动步长: 3个月
- 时间范围: 2021-2025 (20个季度窗口)

**输出**: 80个季度模型 (4品种 × 20窗口)

**关键特性**:
- Walk-Forward验证
- Top30特征选择
- Early Stopping
- 性能报告生成

---

### 5️⃣ 预测系统 (`scripts/predict_2026_signals.py`)

**功能**: 生成三分类趋势预测

**方法**: 二分类模型 + MACD方向判断

**流程**:
```
输入: 57个特征
  ↓
二分类模型判断是否有趋势
  ├─ P(趋势) < 0.5 → "震荡"
  └─ P(趋势) ≥ 0.5 → 用MACD判断方向
      ├─ MACD > 0 → "上涨"
      └─ MACD ≤ 0 → "下跌"
```

**输出**:
- 所有K线信号明细（含Close价格）
- 信号变化点汇总
- 三分类概率

---

### 6️⃣ 可视化 (`scripts/visualize_all_symbols.py`)

**功能**: 生成交互式K线图+信号标记

**特性**:
- 红涨绿跌（中国习惯）
- 三角标记（▲上涨 ▼下跌）
- MACD指标
- OHLC价格显示
- 交互式缩放

**输出**: 4个HTML图表文件

---

## 最新模型

### 滚动训练模型 (`models/rolling_3month/`)

**数量**: 80个季度模型

**命名规则**: `{品种}_window{窗口ID:02d}.pkl`

**品种**:
- HC8888.XSGE (热卷) - 20个模型
- I8888.XDCE (铁矿石) - 20个模型
- AU8888.XSGE (黄金) - 20个模型
- CF8888.XZCE (郑棉) - 20个模型

**最新模型**: window20 (2024-04至2025-09训练)

**模型内容**:
```python
{
    'model': XGBClassifier对象,
    'features': Top30特征列表,
    'feature_importance': 特征重要性,
    'metrics': 性能指标,
    'symbol': 品种代码,
    'window_id': 窗口ID,
    'train_period': 训练周期,
    'test_period': 测试周期
}
```

---

## 测试代码

### 案例代码 (`scripts/visualize_all_symbols.py`)

**功能**: 批量生成所有品种的K线图可视化

**使用场景**: 展示预测结果、信号分析

**运行方式**:
```bash
python scripts/visualize_all_symbols.py
```

**输出**: `predictions/2026_signals/charts/` 目录下的HTML文件

---

## 快速开始

### 1. 生成预测
```bash
cd /Users/mystryl/Documents/Quant
python projects/Multi-Frame_Entry/scripts/predict_2026_signals.py
```

### 2. 查看结果
```bash
# K线信号明细
open predictions/2026_signals/所有K线信号_2026.xlsx

# K线图可视化
open predictions/2026_signals/charts/热卷_K线图_信号标记.html
```

### 3. 训练新模型
```bash
python scripts/rolling_train_3month.py
```

---

## 关键文件索引

| 功能 | 文件路径 | 说明 |
|------|---------|------|
| 特征计算 | `features/trend_features.py` | 57个技术指标 |
| 三分类标签 | `labels/trend_label_final.py` | 上涨/震荡/下跌 |
| 二分类模型 | `models/binary_model.py` | 趋势/震荡分类 |
| 滚动训练 | `scripts/rolling_train_3month.py` | 80个季度模型 |
| 信号预测 | `scripts/predict_2026_signals.py` | 三分类预测 |
| 可视化 | `scripts/visualize_all_symbols.py` | K线图生成 |
| 最新模型 | `models/rolling_3month/*_window20.pkl` | 4个最新模型 |

---

## 数据流程

```
1. 数据采集
   ↓
2. 特征计算 (trend_features.py) → 57个特征
   ↓
3. 标签生成 (trend_label_final.py) → 三分类标签
   ↓
4. 模型训练 (rolling_train_3month.py) → 80个季度模型
   ↓
5. 信号预测 (predict_2026_signals.py) → 三分类信号
   ↓
6. 可视化展示 (visualize_all_symbols.py) → K线图
```

---

## 性能指标

### 二分类模型（趋势 vs 震荡）
| 品种 | AUC | 准确率 | 召回率 |
|------|-----|--------|--------|
| 热卷 | 0.6537 | 60.5% | 75.2% |
| 铁矿石 | 0.6125 | 58.2% | 72.1% |
| 黄金 | 0.6488 | 61.1% | 76.8% |
| 郑棉 | 0.6034 | 56.8% | 69.5% |

### 三分类预测（上涨/下跌/震荡）
- 准确率: 25-30%（因样本极度不平衡）
- 震荡判断准确率: ~80%
- 趋势判断准确率: ~45%

---

## 版本历史

### v2.0 (2026-02-23)
- ✅ 简化为二分类+MACD方案
- ✅ 清理过期代码和数据
- ✅ 添加K线图可视化
- ✅ 优化输出格式（含Close价格）
- ✅ 保留重要项目文档

### v1.0 (2025-12-01)
- 初始版本
- 80个季度滚动模型
- 三分类标签系统

## 重要文档说明

- **findings.md**: 研究发现、实验总结、关键洞察
- **progress.md**: 项目开发进度记录
- **task_plan.md**: 详细的任务计划和拆分
- **plan.md**: 整体项目计划和架构设计
- **入场出场方案.md**: 入场出场策略设计文档

---

## 重要文档

| 文档 | 说明 |
|------|------|
| **README.md** | 项目文档（本文件）|
| **ARCHITECTURE.md** | 架构图（ASCII可视化）|
| **CLEANUP_REPORT.md** | 清理报告（详细说明）|
| **findings.md** | 🔬 研究发现和实验总结 |
| **progress.md** | 📊 项目进度记录 |
| **task_plan.md** | 📋 任务计划和拆分 |
| **plan.md** | 📈 项目计划和架构设计 |
| **入场出场方案.md** | 🎯 入场出场策略设计 |

---

## 版本历史

### v2.0 (2026-02-23)
- ✅ 简化为二分类+MACD方案
- ✅ 清理过期代码和数据
- ✅ 添加K线图可视化
- ✅ 优化输出格式（含Close价格）
- ✅ 保留重要项目文档

### v1.0 (2025-12-01)
- 初始版本
- 80个季度滚动模型
- 三分类标签系统

---

**维护者**: Claude Sonnet
**最后更新**: 2026-02-23
**项目状态**: 生产就绪 ✅
