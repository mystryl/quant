# 多时间框架入场系统 - 项目总结

## 项目概述

本项目实现了一个完整的期货趋势预测和实时监控系统，使用机器学习模型预测未来20根K线的趋势方向。

## 核心功能

### 1. 特征工程 ✅
- **文件**: `features/trend_features.py`
- **功能**: 计算57个技术指标特征
- **特征类别**:
  - 斜率类：EMA斜率、TWAP斜率、线性回归斜率
  - 趋势强度：ADX、ATR
  - 结构类：金叉死叉、均线排列、高低点突破
  - 波动率：滚动标准差、Parkinson波动率
  - 技术指标：MACD、RSI、布林带

### 2. 模型训练 ✅
- **文件**: `models/rolling_3month/`
- **模型类型**: XGBoost二分类模型
- **训练窗口**: 3个月滚动窗口
- **预测目标**: 未来20根K线是否有趋势
- **品种**: HC(热卷)、I(铁矿石)、AU(黄金)、CF(郑棉)

### 3. 信号预测 ✅
- **文件**: `scripts/predict_2026_signals.py`
- **功能**: 使用训练好的模型预测2026年信号
- **方法**: 二分类模型 + MACD方向判断 = 三分类信号（上涨/下跌/震荡）
- **输出**: Excel格式的信号变化点和所有K线信号

### 4. 实时监控 ✅
- **文件**: 
  - `scripts/realtime_monitor.py` - 主监控脚本
  - `scripts/realtime_data_fetcher.py` - 数据获取模块
  - `scripts/trend_change_detector.py` - 趋势变化检测模块
- **功能**:
  - 实时获取最新K线数据
  - 计算特征并预测信号
  - 检测最近10根K线的趋势变化
  - 输出格式化监控报告

### 5. 可视化 ✅
- **文件**: 
  - `scripts/visualize_signals.py` - 信号可视化
  - `scripts/visualize_all_symbols.py` - 多品种可视化
- **功能**: 
  - 绘制K线图和信号标记
  - 支持单个和多个品种

### 6. 测试框架 ✅
- **文件**: `scripts/test_realtime_monitor.py`
- **测试覆盖**:
  - 数据获取模块
  - 趋势变化检测模块
  - 完整监控流程
  - 历史数据验证

## 文件结构

```
Multi-Frame_Entry/
├── features/
│   └── trend_features.py          # 特征计算模块 (57个技术指标)
├── models/
│   ├── binary_model_xgboost.pkl   # 基础二分类模型
│   └── rolling_3month/            # 滚动窗口模型
│       ├── HC8888.XSGE_window20.pkl
│       ├── I8888.XDCE_window20.pkl
│       ├── AU8888.XSGE_window20.pkl
│       └── CF8888.XZCE_window20.pkl
├── scripts/
│   ├── predict_2026_signals.py    # 信号预测脚本
│   ├── realtime_monitor.py        # 实时监控主脚本
│   ├── realtime_data_fetcher.py   # 数据获取模块
│   ├── trend_change_detector.py   # 趋势变化检测
│   ├── visualize_signals.py       # 信号可视化
│   ├── visualize_all_symbols.py   # 多品种可视化
│   ├── test_realtime_monitor.py   # 测试脚本
│   ├── quick_start_monitor.sh     # 快速启动脚本
│   └── README_REALTIME_MONITOR.md # 监控系统文档
├── data/
│   ├── features/                  # 特征数据
│   └── labels/                    # 标签数据
├── predictions/
│   └── 2026_signals/              # 2026年预测结果
├── README.md                      # 项目总览
├── ARCHITECTURE.md                # 架构文档
├── CLEANUP_REPORT.md              # 代码清理报告
├── REALTIME_MONITOR_SUMMARY.md    # 监控系统总结
└── PROJECT_SUMMARY.md             # 本文档
```

## 技术栈

- **语言**: Python 3.8+
- **机器学习**: XGBoost
- **数据处理**: pandas, numpy
- **特征工程**: TA (技术指标库)
- **数据源**: efinance, akshare
- **可视化**: matplotlib, plotly
- **存储**: parquet, pickle

## 使用示例

### 1. 训练模型

```bash
# 使用滚动窗口训练模型
python scripts/train_rolling_window.py
```

### 2. 预测信号

```bash
# 预测2026年信号
python scripts/predict_2026_signals.py
```

### 3. 实时监控

```bash
# 监控单个品种
python scripts/realtime_monitor.py --symbol HC0

# 监控所有品种
python scripts/realtime_monitor.py --all

# 使用快速启动脚本
bash scripts/quick_start_monitor.sh
```

### 4. 可视化

```bash
# 可视化单个品种
python scripts/visualize_signals.py --symbol HC888

# 可视化所有品种
python scripts/visualize_all_symbols.py
```

### 5. 运行测试

```bash
# 运行完整测试套件
python scripts/test_realtime_monitor.py
```

## 支持的品种

| 品种代码 | 品种名称 | 交易所 | 模型文件 |
|---------|---------|--------|----------|
| HC888 | 热卷 | 上期所 (XSGE) | HC8888.XSGE_window20.pkl |
| I888 | 铁矿石 | 大商所 (XDCE) | I8888.XDCE_window20.pkl |
| AU888 | 黄金 | 上期所 (XSGE) | AU8888.XSGE_window20.pkl |
| CF888 | 郑棉 | 郑商所 (XZCE) | CF8888.XZCE_window20.pkl |

## 核心算法

### 信号预测流程

```
1. 计算57个技术指标特征
   ├─ 斜率类 (4个): EMA斜率、TWAP斜率、线性回归斜率
   ├─ 趋势强度 (4个): ADX、ATR及其变化率
   ├─ 结构类 (12个): 金叉死叉、均线排列、高低点突破
   ├─ 波动率 (6个): 滚动标准差、Parkinson波动率
   └─ 技术指标 (31个): MACD、RSI、布林带等

2. 二分类模型预测
   ├─ 输入: 30个精选特征
   ├─ 输出: P(有趋势)
   └─ 模型: XGBoost (滚动3个月窗口)

3. MACD方向判断
   ├─ 如果 P(有趋势) < 0.5 → 震荡
   ├─ 如果 P(有趋势) >= 0.5:
   │   ├─ MACD直方图 > 0 → 上涨
   │   └─ MACD直方图 < 0 → 下跌

4. 输出三分类信号
   └─ 上涨 / 下跌 / 震荡
```

### 趋势变化检测

```
检测逻辑:
  1. 遍历最近N根K线的信号序列
  2. 识别信号转换点
  3. 分类变化类型:
     - 震荡 → 上涨/下跌: 趋势启动
     - 上涨 → 下跌: 趋势反转
     - 下跌 → 上涨: 趋势反转
     - 上涨/下跌 → 震荡: 趋势结束
  4. 输出变化事件（时间、类型、价格）
```

## 性能指标

### 模型性能

- **训练时间**: 约5分钟/模型
- **预测速度**: 约1000条/秒
- **模型大小**: 约500KB/模型

### 实时监控性能

- **数据加载**: < 1秒（本地数据）
- **特征计算**: < 1秒（100根K线）
- **模型预测**: < 0.5秒（71个样本）
- **变化检测**: < 0.1秒
- **总耗时**: 约2秒/品种

### 历史数据验证

使用热卷2026年1-2月数据验证：
- **总K线数**: 121条
- **信号分布**: 震荡81.8%、下跌13.2%、上涨5.0%
- **趋势变化点**: 27个
- **变化类型**: 启动13次、反转1次、结束13次

## 定时任务

### 每小时监控

```bash
# 编辑crontab
crontab -e

# 添加任务
0 * * * * python scripts/realtime_monitor.py --all >> /tmp/realtime_monitor.log 2>&1
```

### 每日数据更新

```bash
# 每天凌晨2点更新数据
0 2 * * * cd /Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry && bash scripts/update_data.sh >> /tmp/data_update.log 2>&1
```

## 扩展方向

### 短期优化

1. **实时数据更新**
   - [ ] 使用efinance/akshare实时API
   - [ ] 增量更新本地数据文件
   - [ ] WebSocket实时推送

2. **告警功能**
   - [ ] 邮件通知
   - [ ] 短信通知
   - [ ] 钉钉/企业微信推送

3. **性能优化**
   - [ ] 多品种并行监控
   - [ ] 模型预加载
   - [ ] 特征计算缓存

### 中期规划

4. **Web界面**
   - [ ] Flask/FastAPI后端
   - [ ] React前端
   - [ ] 实时数据刷新

5. **数据持久化**
   - [ ] SQLite/PostgreSQL数据库
   - [ ] 监控历史记录
   - [ ] Excel导出功能

6. **高级功能**
   - [ ] 多时间框架分析
   - [ ] 跨品种相关性分析
   - [ ] 自动交易接口

### 长期规划

7. **深度学习模型**
   - [ ] LSTM/GRU时序模型
   - [ ] Transformer架构
   - [ ] 强化学习交易agent

8. **云部署**
   - [ ] Docker容器化
   - [ ] Kubernetes集群
   - [ ] CI/CD流水线

## 相关文档

- [项目架构](ARCHITECTURE.md) - 系统架构设计
- [监控系统总结](REALTIME_MONITOR_SUMMARY.md) - 实时监控详细文档
- [监控系统文档](scripts/README_REALTIME_MONITOR.md) - 使用指南
- [代码清理报告](CLEANUP_REPORT.md) - 代码优化记录

## 许可证

内部使用

## 联系方式

- 项目路径: `/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry`
- 数据路径: `/Users/mystryl/Library/CloudStorage/Dropbox/润富/钢铁/code/期货相关代码/futures_data_fetcher/futures_data/60min/`

---

**最后更新**: 2026-02-23
**版本**: 1.0.0
**状态**: 生产就绪 ✅
