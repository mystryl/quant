# Task Plan: Multi-Frame Entry Strategy
<!--
  WHAT: 构建多周期市场状态识别 + 条件入场模型 + Walk-forward 回测框架
  WHY: 这是一个复杂的量化策略项目，需要严格的数据管道和防未来函数处理
  WHEN: 2026-02-20 创建，随着项目进展持续更新
-->

## Goal
实现多品种滚动训练框架，使用18-24个月滚动窗口训练模型，预测未来3个月，对比各品种在滚动验证下的表现，评估模型稳定性。

## Current Phase
Phase 5: 滚动训练框架优化 (18-24月窗口 + 3月预测)

## Phases

### Phase 1: Data Preprocessing (数据预处理) ✅
- [x] Task 1.1: 验证现有 1min 数据质量
  - ✓ 检查 /Users/mystryl/Documents/Quant/data/qlib_data_multi_freq 目录
  - ✓ 确认 1min 数据完整性（HC8888.XSGE: 988,140 行，2014-2025）
  - ✓ 检查时间索引对齐（所有字段一致）
- [x] Task 1.2: 实现多周期重采样模块
  - ✓ 从 1min 构造 5min (208,774 行), 15min (77,022 行), 60min (25,464 行), 1day (2,926 行)
  - ✓ OHLC 正确聚合（open:first, high:max, low:min, close:last）
  - ✓ 成交量求和
  - ✓ 输出到统一数据目录
  - ✓ 生成各频率日历文件
- [x] Task 1.3: 构建 Qlib DataHandler
  - ✓ 创建 data/qlib_config.py 配置模块
  - ✓ 支持多频率数据加载
  - ✓ 创建测试脚本 tests/test_qlib_config.py
- [x] 单元测试：验证重采样正确性
  - ✓ 验证 5min, 60min 数据正确性
- **Status:** complete

**Phase 1 成果：**
- scripts/data_processor.py: 多周期数据处理器
- data/qlib_config.py: Qlib 配置和数据加载模块
- tests/test_qlib_config.py: Qlib 配置测试
- 多频率数据已生成：5min, 15min, 60min, 1day

### Phase 2: Trend Label Construction (趋势标签构建) ✅
- [x] 实现窗口对比分析模块
  - ✓ 测试6个窗口（5, 10, 15, 20, 30, 40根K线）
  - ✓ 计算各窗口信噪比、平衡性、平稳性
  - ✓ 生成对比表格和可视化
  - ✓ 数据驱动推荐：40根K线最优，20根K线第3
- [x] 用户决策：采用20根K线窗口
  - ✓ 理由：时间跨度适中（1.5天），平衡性更好（89.1分）
- [x] 实现三分类标签（1=上涨, 0=震荡, -1=下跌）
- [x] 基于 60min 周期计算未来20根K线收益率
- [x] 按阈值贴标签（±0.3%）
- [x] 严格防止未来函数污染（shift(-20)）
- [x] 输出字段: trend_label, future_return
- [x] 单元测试：验证标签无未来函数
  - ✓ 5个测试全部通过
  - ✓ 验证最后20个样本正确为NaN
  - ✓ 验证shift(-20)计算正确
- **Status:** complete

**Phase 2 成果：**
- labels/trend_label.py: 窗口对比分析模块
- scripts/analyze_windows.py: 窗口对比分析脚本
- labels/trend_label_final.py: 最终标签生成模块（20根K线）
- tests/test_trend_label.py: 完整的单元测试
- data/labels/final_labels_20bars.csv: 最终标签数据（8617个有效标签）
- data/labels/window_comparison.csv: 完整窗口对比表格
- data/labels/ANALYSIS_REPORT.md: 分析报告
- data/labels/*.png: 可视化图表（3张）

### Phase 3: Trend Feature Engineering (趋势特征工程)
- [ ] 实现斜率类特征（EMA60/20 slope, TWAP slope, 线性回归斜率）
- [ ] 实现趋势强度特征（ADX, ADX 变化率, ATR, ATR/price）
- [ ] 实现结构类特征（金叉死叉, K线突破, 高低点突破, 均线排列）
- [ ] 实现波动率特征（rolling std, Parkinson 波动率）
- [ ] 所有特征 shift(1) 避免未来函数
- [ ] 单元测试：验证特征无 look-ahead bias
- [ ] **Status:** pending

### Phase 4: Trend Model Training (趋势模型训练)
- [ ] 选择模型：RandomForestClassifier 或 XGBoost
- [ ] 时间序列分割（train: 2022-23, valid: 2024, test: 2025）
- [ ] 训练趋势分类模型
- [ ] 输出概率：trend_prob_up, trend_prob_down, trend_prob_range
- [ ] 保存模型到 models/trend_model.pkl
- [ ] 验证：确保不使用 shuffle，严格时间分割
- [ ] **Status:** pending

### Phase 5: Entry Model (入场模型)
- [ ] 实现入场标签（未来 15min 最大涨幅 > 0.2%, 最大回撤 < 0.15%）
- [ ] 实现入场特征（RSI, MACD diff, 布林带位置, VWAP 偏离）
- [ ] 条件过滤：只在 trend_prob_up > 0.6 或 trend_prob_down > 0.6 时入场
- [ ] 训练入场模型
- [ ] 单元测试：验证条件过滤逻辑
- [ ] **Status:** pending

### Phase 6: Strategy Implementation (策略实现)
- [ ] 实现策略逻辑（趋势+入场信号组合）
- [ ] 实现止损模块（固定止损, ATR 止损, 时间止损）
- [ ] 实现 Qlib 执行器
- [ ] **Status:** pending

### Phase 5: 滚动训练框架优化 (18-24月窗口 + 3月预测) 🔄
- [x] **Task 5.1: 设计新的滚动训练参数**
  - [x] 训练窗口：18-24个月（可调节参数）
  - [x] 预测窗口：3个月（固定）
  - [x] 滚动步长：3个月（每季度重训）
  - [x] 时间范围：2021-2025（共5年，20个窗口）
- [x] **Task 5.2: 实现滚动训练引擎**
  - [x] 创建 `scripts/rolling_train_3month.py`
  - [x] 支持多品种批量处理
  - [x] 自动滚动切分训练/测试集
  - [x] 保存每个滚动窗口的模型和指标
- [x] **Task 5.3: 批量训练各品种**
  - [x] HC8888.XSGE (热卷) - 20个窗口全部成功
  - [x] I8888.XDCE (铁矿石) - 20个窗口全部成功
  - [x] AU8888.XSGE (黄金) - 20个窗口全部成功 ⭐ 最佳
  - [x] CF8888.XZCE (郑棉) - 20个窗口全部成功
- [x] **Task 5.4: 结果对比分析**
  - [x] 汇总各品种滚动验证指标
  - [x] 生成对比图表
  - [x] 分析模型稳定性（各窗口AUC波动）
  - [x] 输出Excel报告
- [ ] **Task 5.5: 24个月窗口对比测试（可选）**
  - [ ] 测试24个月训练窗口
  - [ ] 对比18月vs24月效果
- **Status:** complete

### Phase 8: Risk Control Enhancement (风险控制增强)
- [ ] 实现概率加权仓位
- [ ] 实现 regime 切换仓位调整
- [ ] 实现高波动仓位降低
- [ ] **Status:** pending

### Phase 9: Code Review & Documentation (代码审查和文档)
- [ ] 代码审查：使用 code-reviewer skill
- [ ] 验证所有关键风控点（无未来函数, 含手续费, 时间分割）
- [ ] 编写项目文档
- [ ] **Status:** pending

## Key Questions
1. 训练窗口选18个月还是24个月？（影响：更长窗口=更多数据，但可能包含过时的市场模式）
2. 滚动步长是否需要调整？（当前3个月，是否需要更频繁重训？）
3. 是否需要对不同品种使用不同的参数？（各品种特性不同）
4. 如何评估模型稳定性？（各窗口AUC标准差、连续失败窗口数）

## Decisions Made
| Decision | Rationale |
|----------|-----------|
| 使用18-24月训练窗口 | 平衡模型稳定性和适应性，避免用过多历史数据（市场模式会变化） |
| 预测3个月 | 季度级别预测更实用，符合实际交易场景 |
| 滚动步长3个月 | 与预测窗口一致，每季度重训模型 |
| 先测试单一窗口参数 | 先固定18个月训练窗口测试流程，成功后再测试24个月对比 |

## Errors Encountered
| Error | Attempt | Resolution |
|-------|---------|------------|
| 18月滚动训练无错误 | 1 | 一次性成功完成，所有4品种×20窗口=80个模型训练成功 |

## Notes
- 所有特征必须 shift(1) 避免未来函数污染
- 标签必须使用未来数据
- 回测必须包含手续费和滑点
- 禁止 shuffle，必须时间分割
- 每完成一个阶段进行 code review 和验证
- 当任务复杂化后使用多 Agent 并行开发
