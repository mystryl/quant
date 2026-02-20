# K线回测数据可视化系统

基于 Nuxt 3 + FastAPI 的期货回测数据可视化平台

## 📋 项目概述

为量化交易策略研究提供专业的 K 线数据可视化工具，支持技术指标分析、回测信号标注、多周期切换等功能。

### 核心特性

- ✅ K 线图表展示（OHLC）
- ✅ 技术指标叠加（MA、EMA、MACD、RSI、布林带等）
- 🔄 回测信号标注（待实现）
- 🔄 多周期切换（待实现）
- ✅ 双数据源支持（Parquet + SmartDataProvider）

### 使用场景

**策略研究辅助工具** - 支持手动调整参数、测试指标、辅助策略开发

---

## 🚀 快速开始

### 前置要求

- Node.js 18+
- Python 3.10+
- npm 或 yarn

### 安装与运行

#### 前端（Phase 1）

```bash
cd frontend
npm install
npm run dev
```

访问 http://localhost:3000

#### 后端（Phase 2 - 待实现）

```bash
cd backend
pip install -r requirements.txt
python main.py
```

---

## 📂 项目结构

```
projects/backtest_visualizer/
├── plan.md              # 详细项目规划
├── README.md            # 本文件
├── frontend/            # Nuxt 3 前端
│   ├── src/
│   │   ├── components/ # Vue 组件
│   │   ├── composables/# 组合式函数
│   │   └── types/      # TypeScript 类型
│   └── package.json
└── backend/             # Python 后端（待创建）
    ├── main.py
    ├── services/
    └── requirements.txt
```

---

## 🎯 开发阶段

### ✅ Phase 1: 静态页面 + 模拟数据（已完成）
- [x] 初始化 Nuxt 3 项目
- [x] 集成 KLineChart 9.8
- [x] 创建基础页面布局
- [x] 实现合约选择器（静态）
- [x] 实现日期范围选择器（静态）
- [x] 实现指标配置面板（静态）
- [x] 使用模拟数据渲染 K 线图
- [x] 添加基础技术指标

### 🔄 Phase 2: 后端数据服务（待开始）
- [ ] 初始化 FastAPI 项目
- [ ] 实现合约列表接口
- [ ] 实现 Parquet 数据读取
- [ ] 集成 SmartDataProvider
- [ ] 实现 K线数据查询接口
- [ ] 实现技术指标计算服务

### 📋 Phase 3-5: 后续阶段
详见 [plan.md](./plan.md)

---

## 🛠️ 技术栈

### 前端
- **Nuxt 3** - Vue 3 全栈框架
- **TypeScript** - 类型安全
- **KLineChart 9.8** - K线图表库
- **Pinia** - 状态管理
- **VueUse** - 组合式工具库

### 后端（计划）
- **FastAPI** - Python Web 框架
- **pyarrow** - Parquet 文件读取
- **pandas** - 数据处理
- **SmartDataProvider** - 统一数据接口

---

## 📝 使用说明

### 1. 选择合约
从下拉菜单中选择要查看的期货合约（如：铜主力连续 CU9999.XSGE）

### 2. 设置时间段
选择开始和结束日期，点击"加载数据"按钮

### 3. 配置指标
在技术指标面板中开启/关闭各类技术指标

### 4. 查看 K 线
图表区域显示 K线走势、技术指标和成交量

---

## 📊 数据来源

- **Parquet 文件**: 直接读取 `K线数据库/期货主力连续_parquet/`
- **SmartDataProvider**: 复用 `projects/qlib_backtest` 的统一数据接口

---

## 🤝 贡献

本项目处于开发初期，欢迎提出建议和反馈。

---

## 📄 许可证

MIT License

---

**最后更新**: 2026-02-19
**当前版本**: v0.1.0 (Phase 1)
