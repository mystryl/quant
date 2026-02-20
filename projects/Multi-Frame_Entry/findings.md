# Findings & Decisions
<!--
  WHAT: Multi-Frame Entry 项目的知识库和发现记录
  WHY: 持久化所有发现和决策，避免在上下文窗口中丢失
  WHEN: 每次发现新信息时更新（遵循 2-Action Rule）
-->

## Requirements
<!-- 从 plan.md 捕获的用户需求 -->
- **当前任务**: 实现18-24月滚动窗口训练框架，预测未来3个月
- 对比各品种（HC8888, I8888, AU8888, CF8888）的滚动验证效果
- 评估模型稳定性（各窗口AUC波动情况）
- 生成对比分析报告和可视化图表
- 严格防止未来函数污染
- 时间序列分割（禁止 shuffle）

## Research Findings

### 已完成的训练框架对比（2026-02-21）

#### 1. 年度滚动训练（2026-02-20）
- **训练方法**: 逐年滚动（用1年训练，预测下1年）
- **已完成品种**：
  - HC8888.XSGE (热卷): 平均AUC=0.5658
  - I8888.XDCE (铁矿石): 平均AUC=0.5678
  - AU8888.XSGE (黄金): 平均AUC=0.6446 ⭐ (最佳)
  - CF8888.XZCE (郑棉): 平均AUC=0.5823

#### 2. 季度滚动训练（2026-02-21）⭐ 推荐
- **训练方法**: 18个月训练窗口，预测未来3个月，每季度滚动
- **总窗口数**: 20个窗口（2021-2025，每年4个季度）
- **结果对比**:

| 品种 | 名称 | 18月滚动AUC | 年度滚动AUC | 改善 | 标准差 |
|------|------|-------------|-------------|------|--------|
| **AU8888.XSGE** | **黄金** | **0.6537** | 0.6446 | **+1.4%** | **0.0526** |
| CF8888.XZCE | 郑棉 | 0.5840 | 0.5823 | +0.3% | 0.0542 |
| I8888.XDCE | 铁矿石 | 0.5812 | 0.5678 | +2.4% | 0.0768 |
| HC8888.XSGE | 热卷 | 0.5758 | 0.5658 | +1.8% | 0.0787 |

#### 3. 关键发现
- **18月滚动优于年度滚动**: 所有品种都有改善（+0.3% ~ +2.4%）
- **AU8888 (黄金) 最优**: AUC=0.6537，标准差最小(0.0526)，最稳定
- **季度预测更实用**: 3个月预测窗口比1年更符合实际交易场景
- **模型稳定性提升**: 更频繁的季度滚动使模型更好适应市场变化

#### 4. 各年份表现（AU8888为例）
- 2021年: AUC=0.6930, 标准差=0.0419
- 2022年: AUC=0.6325, 标准差=0.0402
- 2023年: AUC=0.6556, 标准差=0.0527
- 2024年: AUC=0.6607, 标准差=0.0170 ⭐ 最稳定年份
- 2025年: AUC=0.6265, 标准差=0.0678
<!-- 探索过程中的关键发现 -->

### 数据基础设施（2026-02-20）
- 统一数据目录已建立：`/Users/mystryl/Documents/Quant/data/`
- 多频率数据目录：`qlib_data_multi_freq/` 已存在
  - 包含 calendars 目录（1min, day）
  - 包含 instruments 目录（需要确认具体频率）
- 现有数据重采样代码：`/Users/mystryl/Documents/Quant/projects/qlib_backtest/scripts/data/resample_data.py`
  - 支持 5min, 15min, 60min 重采样
  - OHLC 聚合规则正确
  - 已有 vwap 重算逻辑

### 现有代码资源（2026-02-20）
- Qlib 集成代码：`/Users/mystryl/Documents/Quant/frameworks/qlib/`
- 数据准备脚本：
  - `prepare_data.py` / `prepare_data_v2.py`
  - `resample_data.py`（已读取，可直接复用）
- Qlib 数据加载器：`qlib_data_loader_explained.py`

## Technical Decisions
<!-- 技术和架构决策 -->
| Decision | Rationale |
|----------|-----------|
| 复用现有 resample_data.py | 已有成熟的重采样逻辑，直接移植到项目中避免重复开发 |
| 使用统一数据目录 | 项目已建立 /Users/mystryl/Documents/Quant/data，避免数据重复存储 |
| 先数据管道后模型 | 数据是基础，先建立可靠的数据管道，再进行特征工程和建模 |
| 分阶段开发 + Code Review | 每个阶段完成后审查，避免累积问题，符合用户要求 |
| Phase 1-3 串行，Phase 4+ 可并行 | 前期依赖数据管道，后期模块相对独立可并行开发 |

## Issues Encountered
<!-- 遇到的问题和解决方案 -->
| Issue | Resolution |
|-------|------------|
| | |

## Resources
<!-- 有用的链接、文件路径、API 参考 -->
- 项目计划：`/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/plan.md`
- 统一数据目录：`/Users/mystryl/Documents/Quant/data/`
- 现有重采样代码：`/Users/mystryl/Documents/Quant/projects/qlib_backtest/scripts/data/resample_data.py`
- Qlib 框架：`/Users/mystryl/Documents/Quant/frameworks/qlib/`
- Qlib 数据加载示例：`/Users/mystryl/Documents/Quant/projects/qlib_backtest/scripts/other/qlib_data_loader_explained.py`

## Visual/Browser Findings
<!-- 从图片、PDF 或浏览器结果中学到的信息 -->
-

---
<!-- 遵循 2-Action Rule：每 2 次 view/browser/search 操作后更新此文件 -->
*Update this file after every 2 view/browser/search operations*
*This prevents visual information from being lost*
