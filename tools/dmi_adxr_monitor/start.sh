#!/bin/bash
# 价格监控启动脚本

# 切换到脚本所在目录
cd "$(dirname "$0")"

# 激活虚拟环境（如果使用）
# source venv/bin/activate

# 启动监控
python3 price_monitor.py
