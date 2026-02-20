# K线回测数据可视化系统 - 项目规划

## 📋 项目概述

构建一个基于 Nuxt 3 + FastAPI 的 K 线数据可视化系统，用于期货回测数据的分析和策略研究。

### 核心特性
- ✅ K 线图表展示（OHLC）
- ✅ 技术指标叠加（MA、EMA、MACD、RSI、布林带等）
- ✅ 回测信号标注
- ✅ 多周期切换
- ✅ 双数据源支持（Parquet 直接读取 + SmartDataProvider）

### 使用场景
**策略研究辅助工具** - 支持手动调整参数、测试指标、辅助策略开发

---

## 🏗️ 技术架构

### 前端与 API 层
- **Nuxt 3** + TypeScript：Vue 3 全栈框架
- **Nitro Server**：Nuxt 内置的服务器引擎，提供 API Routes
- **KLineChart 9.8**：K 线图表渲染（从 `frameworks/klinecharts` 引入）
- **Nuxt UI** 或 **Element Plus**：UI 组件库
- **VueUse**：Vue 组合式工具库
- **Pinia**：状态管理

### 数据服务层
- **FastAPI**（Python）：高性能数据处理微服务
- **SmartDataProvider**：复用现有统一数据接口
- **pyarrow** + **pandas**：Parquet 文件读取和数据处理

### 架构流程图
```
浏览器 (Nuxt 3 前端)
    ↕ HTTP/JSON
Nuxt Server API (Nitro)
    ↕ HTTP/gRPC
Python 数据服务 (FastAPI)
    ↕
Parquet 数据文件
```

---

## 📂 项目结构

```
projects/backtest_visualizer/
├── plan.md                    # 本文件
├── README.md                  # 项目说明
├── frontend/                  # Nuxt 3 前端 + API 层
│   ├── .nuxt/                # Nuxt 构建输出
│   ├── node_modules/         # 依赖
│   ├── server/               # Nitro 服务器 API
│   │   └── api/              # API 路由
│   │       ├── contracts.ts  # 合约列表接口
│   │       ├── kline.ts      # K线数据接口
│   │       └── indicators.ts # 技术指标接口
│   ├── src/
│   │   ├── components/       # Vue 组件
│   │   │   ├── KLineChart.vue
│   │   │   ├── ContractSelector.vue
│   │   │   ├── DateRangePicker.vue
│   │   │   └── IndicatorPanel.vue
│   │   ├── composables/      # 组合式函数
│   │   │   └── useKLine.ts
│   │   ├── stores/           # Pinia 状态管理
│   │   │   └── chart.ts
│   │   ├── types/            # TypeScript 类型
│   │   │   └── kline.ts
│   │   └── assets/           # 静态资源
│   ├── public/               # 公共静态文件
│   ├── nuxt.config.ts        # Nuxt 配置
│   ├── package.json          # 前端依赖
│   └── tsconfig.json         # TypeScript 配置
├── backend/                  # Python 数据服务
│   ├── main.py               # FastAPI 应用入口
│   ├── services/
│   │   ├── data_provider.py  # 数据提供者
│   │   ├── indicator_calc.py # 技术指标计算
│   │   └── resampler.py      # 周期重采样
│   ├── models/
│   │   └── schemas.py        # Pydantic 模型
│   └── requirements.txt      # Python 依赖
└── docs/                     # 文档
    ├── API.md                # API 文档
    └── DEPLOY.md             # 部署文档
```

---

## 🚀 分阶段实施计划

### ✅ Phase 1: 静态页面 + 模拟数据（当前阶段）
**目标**：搭建基础框架，实现静态页面展示

**任务清单**：
- [ ] 初始化 Nuxt 3 项目
- [ ] 配置 TypeScript 和开发环境
- [ ] 集成 KLineChart 9.8
- [ ] 创建基础页面布局
- [ ] 实现合约选择器（静态）
- [ ] 实现日期范围选择器（静态）
- [ ] 实现指标配置面板（静态）
- [ ] 使用模拟数据渲染 K 线图
- [ ] 添加基础技术指标（MA5、MA10、MA20）

**交付物**：
- 可运行的前端页面
- 静态组件完整
- 模拟数据显示正常

---

### 🔄 Phase 2: 后端数据服务
**目标**：实现 Python 数据服务，提供真实数据

**任务清单**：
- [ ] 初始化 FastAPI 项目
- [ ] 实现合约列表接口
- [ ] 实现直接读取 Parquet 的数据提供者
- [ ] 集成 SmartDataProvider（可选）
- [ ] 实现 K线数据查询接口
- [ ] 实现技术指标计算服务
- [ ] 实现周期重采样功能
- [ ] 添加 CORS 和错误处理
- [ ] 编写 API 文档

**交付物**：
- FastAPI 服务可运行
- 提供完整的 RESTful API
- 支持双数据源切换

---

### 🌐 Phase 3: 前后端联调
**目标**：前端接入真实数据

**任务清单**：
- [ ] 实现 Nuxt Server API Routes
- [ ] 封装 API 调用函数
- [ ] 实现合约选择器真实数据
- [ ] 实现日期范围查询
- [ ] 实现数据源切换功能
- [ ] 添加加载状态和错误处理
- [ ] 优化大数据量渲染性能

**交付物**：
- 前后端打通
- 可以查看真实 K线数据

---

### 📊 Phase 4: 高级功能
**目标**：完善策略研究功能

**任务清单**：
- [ ] 实现多周期切换
- [ ] 添加更多技术指标（MACD、RSI、布林带等）
- [ ] 实现回测信号标注
- [ ] 添加数据导出功能
- [ ] 实现图表截图保存
- [ ] 添加快捷键支持

**交付物**：
- 功能完整的可视化系统
- 支持策略研究所需的所有功能

---

### 🎨 Phase 5: 优化与部署
**目标**：内网服务器部署

**任务清单**：
- [ ] 性能优化（虚拟滚动、数据分片）
- [ ] 响应式布局优化
- [ ] 编写部署文档
- [ ] 配置生产环境
- [ ] 设置自动启动脚本

**交付物**：
- 可在内网访问的生产环境
- 完整的使用文档

---

## 📝 数据接口设计

### 合约列表接口
```
GET /api/contracts
Response:
{
  "contracts": [
    { "symbol": "CU9999.XSGE", "name": "铜主力连续", "exchange": "XSGE" },
    { "symbol": "AL9999.XSGE", "name": "铝主力连续", "exchange": "XSGE" }
  ]
}
```

### K线数据接口
```
GET /api/kline?symbol=CU9999.XSGE&start=2024-01-01&end=2024-12-31&period=1m&source=parquet
Response:
{
  "data": [
    { "timestamp": "2024-01-01T09:00:00", "open": 68500, "high": 68700, "low": 68400, "close": 68650, "volume": 12345 },
    ...
  ],
  "symbol": "CU9999.XSGE",
  "period": "1m"
}
```

### 技术指标接口
```
GET /api/indicators?symbol=CU9999.XSGE&indicators=MA5,MA10,MA20,MACD
Response:
{
  "MA5": [ ... ],
  "MA10": [ ... ],
  "MA20": [ ... ],
  "MACD": { "diff": [...], "dea": [...], "histogram": [...] }
}
```

---

## 📦 依赖清单

### 前端依赖 (package.json)
```json
{
  "dependencies": {
    "nuxt": "^3.x",
    "vue": "^3.x",
    "@klinecharts/pro": "^9.8.x",
    "@vueuse/core": "^11.x",
    "@pinia/nuxt": "^0.5.x",
    "date-fns": "^3.x"
  }
}
```

### 后端依赖 (requirements.txt)
```
fastapi>=0.115.0
uvicorn[standard]>=0.32.0
pyarrow>=18.0.0
pandas>=2.2.0
numpy>=2.0.0
python-multipart>=0.0.9
pydantic>=2.0.0
```

---

## 🎯 当前阶段：Phase 1

### 下一步行动
1. 初始化 Nuxt 3 项目
2. 安装 KLineChart
3. 创建基础页面
4. 使用模拟数据测试

---

**最后更新**：2026-02-19
**当前状态**：🚀 准备开始 Phase 1
