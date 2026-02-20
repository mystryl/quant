# Phase 5: 多品种滚动训练框架设计

**日期**: 2025-02-21
**目标**: 实现5个品种的独立滚动训练和对比评估

---

## 1. 整体架构

### 目标品种（5个）
- HC8888.XSGE (热卷)
- I8888.XDCE (铁矿石)
- AU8888.XSGE (黄金)
- CF8888.XZCE (郑棉)
- IF8888.CCFX (股指期货) ⚠️ 缺失值24.59%

### 数据策略
- **统一时间范围**: 2020-01-01 至 2025-12-31
- **IF品种**: 额外处理缺失值（前向填充 + 删除连续缺失>100行）
- **独立处理**: 每个品种独立完成数据重采样、标签生成、特征计算、模型训练

### 核心改进
1. **并行处理**: 5个品种可以并行生成特征
2. **统一接口**: 所有品种使用相同的特征工程和标签逻辑
3. **对比分析**: 生成跨品种性能对比表
4. **模型管理**: 按`品种_日期.pkl`格式保存模型

### 工作流程
```
For each symbol in [HC8888, I8888, AU8888, CF8888, IF8888]:
  1. 数据重采样 (1min → 5/15/60min/day)
  2. 生成Regime标签 + 趋势标签 (10根K线窗口)
  3. 计算57个特征
  4. Walk Forward训练 (2020→2021→...→2025)
  5. 保存模型和评估结果

6. 生成多品种对比汇总Excel表
```

### 输出结构
```
data/
├── multi_symbol/
│   ├── HC8888/  # 每个品种独立目录
│   │   ├── features/
│   │   ├── labels/
│   │   └── predictions/
│   ├── I8888/
│   ├── AU8888/
│   ├── CF8888/
│   └── IF8888/
models/
├── rolling/
│   ├── HC8888_2020_2024.pkl
│   ├── I8888_2020_2024.pkl
│   └── ...
results/
└── multi_symbol_comparison_2025.xlsx  # 汇总Excel表
    ├── HC8888
    ├── I8888
    ├── AU8888
    ├── CF8888
    ├── IF8888
    └── Summary  # 跨品种对比
```

---

## 2. 数据处理管道

### 核心模块
`scripts/data_pipeline_multi_symbol.py`

### 功能
1. **批量数据重采样**
   - 从`期货商品指数_parquet`读取5个品种的1min数据
   - 统一时间范围：2020-2025
   - 生成5/15/60min/day周期数据
   - 输出到`qlib_data_multi_freq/`

2. **缺失值处理**
   - IF品种：前向填充 + 删除连续缺失超过100行的样本
   - 其他品种：标准处理（已有）

3. **并行化**
   - 使用`joblib`并行处理5个品种
   - 每个品种独立的进程，互不干扰
   - 进度条显示整体进度

### 接口设计
```python
class MultiSymbolDataPipeline:
    def process_all_symbols(symbol_list, start_date, end_date):
        """批量处理所有品种"""

    def process_single_symbol(symbol):
        """处理单个品种（可并行）"""

    def validate_data_quality():
        """验证数据质量"""
```

### 输出
- 5个品种的多周期数据（约20分钟完成）
- 数据质量报告CSV文件

---

## 3. 滚动训练框架（年度版本）

### 核心模块
`scripts/rolling_train_multi_symbol.py`

### 训练策略
- **Walk Forward**: 逐年滚动训练（2020→2021→2022→2023→2024→2025）
- **训练窗口**: 使用前12-18个月数据
- **验证策略**: 在下一年的高波动Regime上测试
- **重训频率**: 年度版本（本次实现）

### 训练流程
```python
for symbol in [HC8888, I8888, AU8888, CF8888, IF8888]:
    # 1. 加载该品种的标签和特征
    labels = load_labels(symbol)  # 10根K线窗口
    features = load_features(symbol)  # 57个特征

    # 2. Walk Forward训练
    for year in [2021, 2022, 2023, 2024, 2025]:
        train_data = filter_range(symbol, start=f"{year-2}-01-01", end=f"{year-1}-12-31")
        test_data = filter_range(symbol, year=year)

        # 3. Regime过滤：只用高波动数据训练
        train_data = filter_high_volatility(train_data)

        # 4. 训练XGBoost模型
        model = train_xgboost(train_data, top_k_features=30)

        # 5. 评估
        metrics = evaluate(model, test_data)

        # 6. 保存模型
        save_model(model, f"{symbol}_{year}.pkl")
```

### 并行化
- 5个品种可以完全并行训练（5个进程同时运行）
- 总耗时：约30-45分钟（取决于CPU核心数）

---

## 4. 评估与报告系统

### 核心模块
`scripts/evaluate_multi_symbol.py`

### 评估指标
- **基础指标**: 准确率、AUC-ROC、精确率、召回率、F1-score
- **分类指标**: 震荡召回率、有趋势召回率、Regime分布
- **稳定性指标**: 逐年性能变化、品种间差异

### 报告格式
- **Excel文件**: `results/multi_symbol_comparison_2025.xlsx`
- **结构**:
  - Sheet1: HC8888 详细结果（逐年性能）
  - Sheet2: I8888 详细结果
  - Sheet3: AU8888 详细结果
  - Sheet4: CF8888 详细结果
  - Sheet5: IF8888 详细结果
  - Sheet6: **Summary汇总表**（重点）

### Summary汇总表内容
```
品种 | 2021_AUC | 2022_AUC | 2023_AUC | 2024_AUC | 2025_AUC | 平均_AUC | 最佳年份 | 最差年份
HC8888 | 0.50 | 0.54 | 0.46 | 0.66 | 0.53 | 0.538 | 2024 | 2023
I8888  | ...
```

### 可视化图表
- 每个品种：AUC逐年折线图
- 多品种对比：箱线图、热力图
- 保存为PNG图片

---

## 5. 实施步骤（6步，预计2-3小时）

### Step 1: 数据准备 (30分钟)
- 创建`scripts/data_pipeline_multi_symbol.py`
- 实现5个品种的多周期重采样
- 生成Regime标签和趋势标签（10根K线窗口）
- 计算特征（复用现有57个特征）

### Step 2: 训练框架 (45分钟)
- 创建`scripts/rolling_train_multi_symbol.py`
- 实现Walk Forward年度滚动训练
- 集成XGBoost模型和Top30特征选择
- 添加进度条和并行处理

### Step 3: 评估系统 (30分钟)
- 创建`scripts/evaluate_multi_symbol.py`
- 实现多维度评估指标计算
- 生成逐年性能对比表

### Step 4: Excel报告 (30分钟)
- 创建`scripts/generate_report.py`
- 使用`openpyxl`生成多sheet Excel文件
- 添加Summary汇总表和格式化

### Step 5: 可视化 (30分钟)
- 生成AUC逐年折线图
- 生成多品种对比图表

### Step 6: 主控脚本 (15分钟)
- 创建`scripts/run_multi_symbol_experiment.py`
- 一键运行完整流程

### 验收标准
- ✅ 5个品种全部完成训练
- ✅ Excel报告包含所有品种结果
- ✅ 至少3个品种AUC > 0.5

---

## 6. 技术细节

### 数据配置
```python
SYMBOL_CONFIG = {
    'HC8888.XSGE': {'name': '热卷', 'exchange': 'XSGE'},
    'I8888.XDCE': {'name': '铁矿石', 'exchange': 'XDCE'},
    'AU8888.XSGE': {'name': '黄金', 'exchange': 'XSGE'},
    'CF8888.XZCE': {'name': '郑棉', 'exchange': 'XZCE'},
    'IF8888.CCFX': {'name': '股指期货', 'exchange': 'CCFX'},
}

TRAINING_YEARS = [2021, 2022, 2023, 2024, 2025]
TRAIN_WINDOW_MONTHS = 18  # 训练窗口18个月
LABEL_WINDOW_BARS = 10    # 10根K线窗口
VOLATILITY_THRESHOLD = 1.5  # 1.5σ波动率阈值
```

### 关键参数
- 特征数: Top 30
- 模型: XGBoost (max_depth=5, n_estimators=200)
- Regime过滤: 仅高波动数据
- 评估指标: AUC-ROC（主要）
