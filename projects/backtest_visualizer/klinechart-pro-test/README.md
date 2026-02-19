# KLineChart Pro 测试项目

这是一个简单的测试项目，用于测试 KLineChart Pro 框架。

## 功能

- 使用 KLineChart Pro 展示K线图表
- 从现有数据库读取合约数据
- 支持多合约切换查看

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 启动服务器

```bash
python server.py
```

### 3. 访问页面

打开浏览器访问: http://localhost:5000

## 数据来源

项目会自动读取 `../frontend/public/data/` 目录下的：
- `contracts.json` - 合约列表
- `{SYMBOL}.json` - 各合约的K线数据

## 技术栈

- **前端**: 纯 HTML + JavaScript，使用 KLineChart Pro (CDN)
- **后端**: Flask (Python)
- **数据**: 从项目的 JSON 文件读取

## 目录结构

```
klinechart-pro-test/
├── index.html          # 主页面
├── server.py           # Flask 服务器
├── requirements.txt    # Python 依赖
└── README.md          # 说明文档
```
