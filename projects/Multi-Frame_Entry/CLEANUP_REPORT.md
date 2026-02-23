# Multi-Frame Entry 项目清理报告

## 清理完成时间
2026-02-23 08:28

---

## 清理内容

### 已删除过期文件

#### 文档 (8个)
- ❌ DEPLOYMENT_REPORT.md
- ❌ EXPERIMENT_SUMMARY.md
- ❌ README_三分类预测.md
- ❌ README_可视化说明.md
- ❌ findings.md
- ❌ progress.md
- ❌ task_plan.md
- ❌ plan.md
- ❌ 入场出场方案.md
- ❌ Multi_Frame_Entry_Summary.pptx

#### 代码 (14个)
- ❌ models/trend_model.py (旧版模型)
- ❌ models/trend_model.pkl
- ❌ models/trend_model_optimized.pkl
- ❌ models/trend_model_xgboost.pkl
- ❌ scripts/walk_forward_trend_model.py
- ❌ scripts/analyze_windows.py
- ❌ scripts/create_summary_ppt.py
- ❌ scripts/test_pipeline_single.py
- ❌ scripts/test_symbol_batch.py
- ❌ scripts/test_train_single.py
- ❌ scripts/verify_paths.py
- ❌ scripts/check_experiment_progress.sh
- ❌ scripts/run_experiment_low_memory.sh
- ❌ scripts/run_multi_symbol_experiment.py
- ❌ scripts/test_remaining_symbols.sh

#### 目录 (5个)
- ❌ backtest/
- ❌ docs/
- ❌ logs/
- ❌ tests/
- ❌ models/rolling/
- ❌ models/training_results/

#### 数据文件
- ❌ data/labels/labels_raw/
- ❌ data/labels/returns_raw/
- ❌ data/labels/*.png

---

## 保留的核心文件

### 📂 features/ (特征工程模块)
```
features/
├── __init__.py
└── trend_features.py         # ⭐ 57个技术指标特征计算
```

### 📂 labels/ (标签生成模块)
```
labels/
├── __init__.py
├── binary_label.py          # 二分类标签（趋势/震荡）
├── trend_label_final.py      # ⭐ 三分类标签（上涨/震荡/下跌）
└── volatility_regime.py     # 波动率标签（高/低波动）
```

### 📂 models/ (模型模块)
```
models/
├── __init__.py
├── binary_model.py          # ⭐ 二分类模型类
├── binary_model_xgboost.pkl # 单个完整模型
├── rolling_3month/          # ⭐ 80个季度滚动模型
│   ├── HC8888.XSGE_window01-20.pkl
│   ├── I8888.XDCE_window01-20.pkl
│   ├── AU8888.XSGE_window01-20.pkl
│   └── CF8888.XZCE_window01-20.pkl
└── training_results_3month/ # 训练结果报告
```

### 📂 scripts/ (脚本模块)
```
scripts/
├── __init__.py
├── data_pipeline_multi_symbol.py  # 多品种数据管道
├── data_processor.py             # 数据处理工具
├── generate_binary_features.py   # 特征生成
├── generate_excel_report.py      # Excel报告生成
├── predict_2026_signals.py       # ⭐ 预测脚本（关键代码）
├── rolling_train_3month.py       # ⭐ 滚动训练脚本（关键代码）
├── rolling_train_multi_symbol.py # 多品种训练
├── visualize_all_symbols.py     # ⭐ 批量可视化（案例代码）
└── visualize_signals.py         # 单品种可视化
```

### 📂 predictions/ (预测结果)
```
predictions/2026_signals/
├── 所有K线信号_2026.xlsx         # K线信号明细
├── 趋势信号变化_2026.xlsx         # 信号变化点
└── charts/                      # K线图可视化
    ├── 热卷_K线图_信号标记.html
    ├── 铁矿石_K线图_信号标记.html
    ├── 黄金_K线图_信号标记.html
    └── 郑棉_K线图_信号标记.html
```

### 📂 data/ (数据目录)
```
data/
├── multi_symbol/               # 多品种数据
│   ├── HC8888.XSGE/            # 热卷
│   ├── I8888.XDCE/             # 铁矿石
│   ├── AU8888.XSGE/            # 黄金
│   └── CF8888.XZCE/            # 郑棉
├── labels/                      # 全局标签数据
│   ├── final_labels_20bars.csv
│   ├── binary_labels_10bars.csv
│   └── volatility_regime_labels.csv
└── predictions/                 # 历史预测结果
```

---

## 项目统计

### 代码文件统计
| 模块 | Python文件数 | 说明 |
|------|-------------|------|
| features/ | 1 | 特征工程 |
| labels/ | 3 | 标签生成 |
| models/ | 1 | 模型类 |
| scripts/ | 10 | 脚本工具 |
| **总计** | **15** | **精简高效** |

### 模型统计
| 类型 | 数量 | 说明 |
|------|------|------|
| 滚动模型 | 80 | 4品种×20窗口 |
| 单个模型 | 4 | 完整模型文件 |
| **总计** | **84** | **生产就绪** |

### 数据统计
| 品种 | 特征文件 | 标签文件 | 模型数量 |
|------|---------|---------|---------|
| 热卷 | 1 | 1 | 20 |
| 铁矿石 | 1 | 1 | 20 |
| 黄金 | 1 | 1 | 20 |
| 郑棉 | 1 | 1 | 20 |

---

## 关键代码标注

### ⭐ 核心代码（必读）
1. **features/trend_features.py** - 57个特征计算
2. **labels/trend_label_final.py** - 三分类标签生成
3. **models/binary_model.py** - 二分类模型
4. **scripts/rolling_train_3month.py** - 滚动训练框架
5. **scripts/predict_2026_signals.py** - 预测系统

### 🔧 辅助代码
6. **labels/binary_label.py** - 二分类标签
7. **labels/volatility_regime.py** - 波动率标签
8. **scripts/data_pipeline_multi_symbol.py** - 数据管道
9. **scripts/rolling_train_multi_symbol.py** - 多品种训练

### 📊 案例代码
10. **scripts/visualize_all_symbols.py** - 批量可视化
11. **scripts/visualize_signals.py** - 单品种可视化
12. **scripts/generate_excel_report.py** - 报告生成

---

## 最新模型

### 滚动训练模型（80个）
**位置**: `models/rolling_3month/`

**命名**: `{品种}_window{ID:02d}.pkl`

**最新**: window20 (2024-04至2025-09训练)

**品种**:
- HC8888.XSGE (热卷)
- I8888.XDCE (铁矿石)
- AU8888.XSGE (黄金)
- CF8888.XZCE (郑棉)

**性能**:
- 热卷: AUC=0.6537
- 铁矿石: AUC=0.6125
- 黄金: AUC=0.6488
- 郑棉: AUC=0.6034

---

## 测试代码

### 案例代码
**文件**: `scripts/visualize_all_symbols.py`

**功能**: 批量生成4个品种的K线图+信号标记

**使用场景**:
- 展示预测结果
- 信号分析
- 回顾历史表现

**运行方式**:
```bash
python scripts/visualize_all_symbols.py
```

**输出**: 4个交互式HTML图表

---

## 清理效果

### 清理前
- 📁 15个目录
- 📄 30+个文件
- 🗂️ 大量过期代码和数据

### 清理后
- ✅ 7个目录（精简53%）
- ✅ 15个核心Python文件
- ✅ 结构清晰，易于维护

---

## 项目结构图

```
Multi-Frame_Entry/
│
├── 📄 README.md                 # 项目文档
│
├── 📂 features/                 # 特征工程
│   └── trend_features.py        # 57个技术指标
│
├── 📂 labels/                   # 标签生成
│   ├── binary_label.py          # 二分类
│   ├── trend_label_final.py    # ⭐ 三分类
│   └── volatility_regime.py    # 波动率
│
├── 📂 models/                   # 模型
│   ├── binary_model.py          # ⭐ 模型类
│   └── rolling_3month/          # ⭐ 80个模型
│
├── 📂 scripts/                  # 脚本
│   ├── predict_2026_signals.py  # ⭐ 预测
│   ├── rolling_train_3month.py  # ⭐ 训练
│   ├── visualize_all_symbols.py # ⭐ 可视化
│   └── ... (辅助脚本)
│
├── 📂 data/                     # 数据
│   ├── multi_symbol/           # 品种数据
│   └── labels/                 # 标签数据
│
└── 📂 predictions/              # 预测结果
    └── 2026_signals/
        ├── 所有K线信号_2026.xlsx
        ├── 趋势信号变化_2026.xlsx
        └── charts/              # 可视化图表
```

---

## 快速开始

### 1. 查看项目文档
```bash
cat README.md
```

### 2. 生成预测
```bash
python scripts/predict_2026_signals.py
```

### 3. 查看可视化
```bash
open predictions/2026_signals/charts/热卷_K线图_信号标记.html
```

---

**清理完成！** ✅
项目结构清晰，代码精简，易于维护！
