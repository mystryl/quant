# K线回测数据可视化系统 - Frontend

Nuxt 3 + TypeScript + KLineChart 9.8

## ✨ Phase 1 已完成

- ✅ 真实历史数据加载
- ✅ 7个期货合约（铜、铝、黄金、白银、螺纹钢、甲醇、白糖）
- ✅ 每个合约1000条真实K线数据
- ✅ KLineChart 9.8 集成
- ✅ 响应式UI设计

## 🚀 快速开始

### 1. 安装依赖

```bash
cd frontend
npm install
```

### 2. 运行开发服务器

```bash
npm run dev
```

访问 http://localhost:3000

### 3. 使用说明

1. 页面加载后会自动从 `/data/contracts.json` 加载合约列表
2. 从下拉菜单选择合约（如：铜主力连续 CU9999.XSGE）
3. 自动加载该合约的真实历史数据（1000条）
4. 查看K线图表、技术指标和成交量

## 📊 数据来源

数据存储在 `public/data/` 目录：

```
public/data/
├── contracts.json       # 合约列表元数据
├── CU9999.XSGE.json     # 铜主力连续K线数据
├── AL9999.XSGE.json     # 铝主力连续K线数据
├── AU9999.XSGE.json     # 黄金主力连续K线数据
├── AG9999.XSGE.json     # 白银主力连续K线数据
├── RB9999.XSGE.json     # 螺纹钢主力连续K线数据
├── MA9999.XZCE.json     # 甲醇主力连续K线数据
└── SR9999.XZCE.json     # 白糖主力连续K线数据
```

### 数据格式

每个JSON文件包含：
- `symbol`: 合约代码
- `data`: K线数据数组（timestamp, open, high, low, close, volume）
- `stats`: 统计信息（条数、时间范围、价格区间）

## 🔧 添加更多合约数据

在项目根目录运行数据准备脚本：

```bash
cd ../..
python projects/backtest_visualizer/scripts/prepare_sample_data.py
```

编辑 `scripts/prepare_sample_data.py` 中的 `CONTRACTS` 列表来添加更多合约。

## 🛠️ 技术栈

- **Nuxt 3**: Vue 3 全栈框架
- **TypeScript**: 类型安全
- **KLineChart 9.8**: 专业 K线图表库（从本地 frameworks 引入）
- **Pinia**: 状态管理
- **VueUse**: Vue 组合式工具库

## 📂 项目结构

```
frontend/
├── src/
│   ├── components/          # Vue 组件
│   │   ├── KLineChart.vue      # K线图表组件
│   │   ├── ContractSelector.vue # 合约选择器
│   │   └── IndicatorPanel.vue   # 指标配置面板
│   ├── composables/         # 组合式函数
│   │   ├── useRealData.ts      # 真实数据加载器
│   │   └── useMockData.ts      # 模拟数据生成器（备用）
│   ├── types/              # TypeScript 类型
│   │   └── kline.ts            # K线相关类型定义
│   └── assets/            # 静态资源
│       └── css/main.css        # 全局样式
├── public/
│   └── data/              # 真实数据文件
│       ├── contracts.json
│       └── *.json         # 各合约K线数据
├── app.vue                # 主页面
├── nuxt.config.ts        # Nuxt 配置
└── package.json          # 依赖管理
```

## 🎯 下一步

- [ ] Phase 2: Python 后端数据服务（FastAPI）
- [ ] Phase 3: 前后端联调
- [ ] Phase 4: 高级功能（多周期切换、回测信号标注等）
- [ ] Phase 5: 优化与内网部署

详见项目根目录的 [plan.md](../plan.md)

---

**当前版本**: v0.1.0 (Phase 1 真实数据)
**最后更新**: 2026-02-19
