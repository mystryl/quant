# 调试指南

## 🔍 K线数据不显示的问题排查

### 步骤1：打开浏览器开发者工具

1. 在浏览器中打开 http://localhost:3000/
2. 按 `F12` 或 `Cmd+Option+I`（Mac）打开开发者工具
3. 切换到 **Console（控制台）** 标签页

### 步骤2：查看调试信息

页面加载后，控制台应该显示：

```
🔄 页面已挂载，开始加载合约列表...
🔄 fetch /data/contracts.json
✅ 合约列表加载成功: 7 个
✅ 已加载合约列表: 7 个
```

### 步骤3：选择合约

1. 在页面上从下拉菜单选择一个合约（如：铜主力连续 CU9999.XSGE）
2. 控制台应该显示：

```
合约切换: {symbol: "CU9999.XSGE", name: "铜主力连续", exchange: "XSGE"}
开始加载 CU9999.XSGE 数据...
🔄 fetch /data/CU9999.XSGE.json
✅ CU9999.XSGE 数据加载成功: 1000 条
✅ 加载 CU9999.XSGE 数据成功: 1000 条
数据示例: {timestamp: 1767016260000, open: 101790, ...}
klineData.value 长度: 1000
KLineChart: 数据更新 1000 条
KLineChart: 应用数据到图表 {timestamp: 1767016260000, ...}
```

### 步骤4：检查可能的错误

#### 错误1：合约下拉菜单为空或显示 disabled

**原因**：合约列表加载失败

**解决**：
- 检查控制台是否有 "❌ 加载合约列表失败" 错误
- 确认 `public/data/contracts.json` 文件存在
- 尝试直接访问 http://localhost:3000/data/contracts.json

#### 错误2：选择合约后没有反应

**原因**：事件处理未触发

**解决**：
- 检查控制台是否有 "合约切换:" 日志
- 刷新页面重试

#### 错误3：数据显示但图表为空白

**原因**：KLineChart 初始化失败

**解决**：
- 检查控制台是否有 KLineChart 相关错误
- 确认 klinecharts 包已正确安装：`npm list klinecharts`
- 查看是否有任何 JavaScript 错误

#### 错误4：数据加载失败

**原因**：数据文件路径错误或文件不存在

**解决**：
- 检查控制台错误信息
- 确认数据文件存在：`ls public/data/*.json`
- 尝试直接访问数据文件 URL

### 步骤5：网络请求检查

1. 切换到 **Network（网络）** 标签页
2. 刷新页面
3. 检查以下请求：
   - `/data/contracts.json` - 应该返回 200 OK
   - `/data/{SYMBOL}.json` - 选择合约后应该返回 200 OK

### 常见问题

**Q: 控制台显示 404 错误**
- A: 检查文件路径，确保数据文件在 `public/data/` 目录下

**Q: 控制台显示 CORS 错误**
- A: 这是开发服务器的正常行为，不应该有 CORS 问题

**Q: 合约选择器一直是 disabled 状态**
- A: 等待合约列表加载完成（通常1-2秒）

**Q: 图表区域显示 "请选择合约开始查看 K线数据"**
- A: 这是正常的初始状态，选择合约后应该显示图表

### 手动测试数据访问

在浏览器新标签页中直接访问：
- http://localhost:3000/data/contracts.json
- http://localhost:3000/data/CU9999.XSGE.json

应该能看到 JSON 数据。

### 获取更多帮助

如果问题仍然存在，请提供：
1. 浏览器控制台的完整输出（截图或复制文本）
2. Network 标签页的请求状态
3. 使用的浏览器类型和版本

## 📝 调试日志说明

日志前缀含义：
- 🔄 = 进行中的操作
- ✅ = 成功的操作
- ❌ = 失败的操作
- ⚠️ = 警告信息

## 🔧 快速修复

如果一切都失败了，尝试：

```bash
# 停止服务器（Ctrl+C）
# 清理缓存
rm -rf .nuxt node_modules/.vite

# 重新启动
npm run dev
```

然后在浏览器中硬刷新页面（Cmd+Shift+R 或 Ctrl+Shift+R）。
