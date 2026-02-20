# Findings & Decisions
<!--
  WHAT: Multi-Frame Entry 项目的知识库和发现记录
  WHY: 持久化所有发现和决策，避免在上下文窗口中丢失
  WHEN: 每次发现新信息时更新（遵循 2-Action Rule）
-->

## Requirements
<!-- 从 plan.md 捕获的用户需求 -->
- 构建多周期市场状态识别 + 条件入场模型 + Walk-forward 回测框架
- 数据源：1min 主力连续合约
- 框架：Qlib
- 模型：RandomForest / XGBoost
- 严格防止未来函数污染
- 所有特征 shift(1)，标签使用未来数据
- 回测包含手续费滑点
- 时间序列分割（禁止 shuffle）

## Research Findings
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
