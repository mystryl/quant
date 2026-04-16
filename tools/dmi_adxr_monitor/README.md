# 价格监控系统使用说明

## 文件结构

```
dmi_adxr_monitor/
├── price_monitor.py    # 主程序
├── config.yaml         # 配置文件
├── start.sh           # 启动脚本
└── logs/              # 日志目录（自动创建）
```

## 配置说明

编辑 `config.yaml` 文件来配置监控参数：

```yaml
# 监控品种列表
monitors:
  - symbol: "HC0"      # 期货合约代码
    name: "热卷主力"    # 品种名称
    targets:           # 目标价位列表
      - 3267.40
      - 3248.72
      - 3237.58
    threshold: 0.2     # 触发阈值（点）

# 全局参数
interval: 300          # 检测间隔（秒）

# 日志配置
log:
  enabled: true
  file: "logs/price_monitor.log"
```

## 基本使用

### 1. 手动启动监控

```bash
cd /Users/mystryl/Documents/Quant/tools/dmi_adxr_monitor
python3 price_monitor.py
```

或使用启动脚本：

```bash
cd /Users/mystryl/Documents/Quant/tools/dmi_adxr_monitor
bash start.sh
```

### 2. 添加监控品种

编辑 `config.yaml`，在 `monitors` 列表中添加新品种：

```yaml
monitors:
  - symbol: "HC0"
    name: "热卷主力"
    targets: [3267.40, 3248.72, 3237.58]
    threshold: 0.2

  - symbol: "RB0"  # 新增品种
    name: "螺纹钢主力"
    targets: [3500.00, 3480.00]
    threshold: 0.5
```

## 定时监控方案

### 方案一：使用 nohup 后台运行（推荐）

在服务器或持续运行的机器上使用：

```bash
cd /Users/mystryl/Documents/Quant/tools/dmi_adxr_monitor
nohup python3 price_monitor.py > monitor.out 2>&1 &

# 查看进程
ps aux | grep price_monitor

# 停止监控
kill <进程ID>
```

**优点**：
- 简单易用
- 持续运行
- 输出到文件便于查看

### 方案二：使用 systemd 服务（Linux）

1. 创建服务文件 `/etc/systemd/system/price-monitor.service`：

```ini
[Unit]
Description=Price Monitor Service
After=network.target

[Service]
Type=simple
User=your_username
WorkingDirectory=/Users/mystryl/Documents/Quant/tools/dmi_adxr_monitor
ExecStart=/usr/bin/python3 /Users/mystryl/Documents/Quant/tools/dmi_adxr_monitor/price_monitor.py
Restart=always
RestartSec=10

[Install]
WantedBy=multi-user.target
```

2. 启动服务：

```bash
# 重载配置
sudo systemctl daemon-reload

# 启动服务
sudo systemctl start price-monitor

# 设置开机自启
sudo systemctl enable price-monitor

# 查看状态
sudo systemctl status price-monitor

# 查看日志
sudo journalctl -u price-monitor -f

# 停止服务
sudo systemctl stop price-monitor
```

**优点**：
- 开机自动启动
- 崩溃自动重启
- 系统级管理

### 方案三：使用 cron 定时执行（不推荐）

**注意**：cron 方案不适合此场景，因为：

1. cron 最小间隔是 1 分钟，但本程序已经内置了循环
2. 使用 cron 会导致每次启动新进程，无法持续监控
3. 会产生大量重复日志

**不推荐的配置**（仅作参考）：
```bash
# 编辑 crontab
crontab -e

# 每小时执行一次（不推荐！）
# 0 * * * * cd /Users/mystryl/Documents/Quant/tools/dmi_adxr_monitor && python3 price_monitor.py
```

### 方案四：使用 tmux/screen 会话

在 tmux 会话中运行，断开 SSH 也不会停止：

```bash
# 创建新会话
tmux new -s price-monitor

# 在会话中启动监控
cd /Users/mystryl/Documents/Quant/tools/dmi_adxr_monitor
python3 price_monitor.py

# 按 Ctrl+B 然后按 D 断开会话

# 重新连接
tmux attach -t price-monitor

# 查看所有会话
tmux ls
```

或使用 screen：

```bash
# 创建新会话
screen -S price-monitor

# 在会话中启动监控
cd /Users/mystryl/Documents/Quant/tools/dmi_adxr_monitor
python3 price_monitor.py

# 按 Ctrl+A 然后按 D 断开会话

# 重新连接
screen -r price-monitor

# 查看所有会话
screen -ls
```

## 日志查看

### 实时日志

```bash
# 如果使用 tmux/screen
# 直接查看控制台输出

# 如果使用 nohup
tail -f monitor.out

# 查看程序日志
tail -f logs/price_monitor.log
```

### 日志搜索

```bash
# 查找触发预警的记录
grep "触发预警" logs/price_monitor.log

# 查看特定品种的记录
grep "HC0" logs/price_monitor.log

# 查看今天的日志
grep "$(date +%Y-%m-%d)" logs/price_monitor.log
```

## 依赖安装

```bash
pip install akshare pyyaml
```

## 注意事项

1. **交易时间监控**：期货只在交易时间段有实时数据，非交易时间会获取失败
2. **网络稳定**：确保网络连接稳定，否则可能获取价格失败
3. **日志清理**：定期清理日志文件，避免占用过多磁盘空间
4. **配置热更新**：修改 `config.yaml` 后需要重启程序才能生效

## 推荐方案总结

| 场景 | 推荐方案 |
|------|---------|
| 本地开发测试 | 直接运行或 tmux |
| 服务器长期运行 | systemd 服务 |
| 简单快速部署 | nohup 后台运行 |
| 需要频繁查看输出 | tmux/screen 会话 |
