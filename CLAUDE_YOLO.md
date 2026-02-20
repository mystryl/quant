# Claude YOLO 模式使用指南

## 🎯 什么是 YOLO 模式？

YOLO (You Only Live Once) 模式让 Claude 在执行操作时**自动确认所有提示**，无需手动回答确认问题。

### 主要特性

- 🔥 **自动确认** - 跳过所有 "Are you sure?" 类型的确认
- ⚡ **快速执行** - 减少 Claude 提问次数
- 🚀 **高效批量操作** - 适合批量任务
- ⚠️ **谨慎使用** - 可能会执行危险操作

---

## 🚀 使用方法

### 基本用法

```bash
# 启用 YOLO 模式
claude --yolo "你的任务"

# 示例
claude --yolo "批量转换所有合约数据"
claude --yolo "删除所有临时文件"
claude --yolo "重构这个模块"
```

### 使用场景

#### ✅ 推荐使用场景

1. **批量数据转换**
   ```bash
   claude --yolo "将所有合约数据转换为 Qlib 格式"
   ```

2. **代码重构**
   ```bash
   claude --yolo "重构所有代码，使用新的数据接口"
   ```

3. **自动化脚本**
   ```bash
   # 在脚本中使用
   #!/bin/bash
   claude --yolo "执行每日数据更新"
   ```

4. **重复性任务**
   ```bash
   claude --yolo "统一修改所有文件的导入语句"
   ```

#### ❌ 不推荐使用场景

1. **删除操作** - 可能误删重要文件
2. **生产环境** - 需要谨慎检查
3. **不确定的操作** - 不清楚会执行什么

---

## 🔧 工作原理

当你使用 `--yolo` 参数时：

1. 设置环境变量 `YOLO_MODE=1`
2. 传递给你的工具和脚本
3. 工具可以检测到 YOLO 模式并自动确认操作

### 在代码中检测 YOLO 模式

```python
import os

if os.getenv('YOLO_MODE') == '1':
    # YOLO 模式：自动确认所有操作
    force = True
    skip_warnings = True
else:
    # 正常模式：需要用户确认
    force = False
    skip_warnings = False
```

---

## 📝 实际示例

### 示例 1：数据转换

```bash
# 正常模式 - Claude 会询问是否覆盖
claude "转换所有合约数据"

# YOLO 模式 - 自动确认覆盖
claude --yolo "转换所有合约数据"
```

### 示例 2：批量文件操作

```bash
# YOLO 模式 - 自动确认删除
claude --yolo "删除所有 .tmp 文件"
```

### 示例 3：代码重构

```bash
# YOLO 模式 - 自动应用所有重构建议
claude --yolo "将所有代码迁移到新的数据接口"
```

---

## 🛡️ 安全建议

### 使用 YOLO 模式前的检查清单

- [ ] 确认你知道将要执行什么操作
- [ ] 已备份重要数据
- [ ] 不是在删除生产数据
- [ ] 代码已提交到 Git（可以回退）

### 最佳实践

1. **先在测试环境使用**
   ```bash
   # 先测试
   claude --yolo "转换单个合约"

   # 确认无误后再批量
   claude --yolo "转换所有合约"
   ```

2. **使用 Git 版本控制**
   ```bash
   # 操作前提交
   git commit -m "Before YOLO operation"

   # 使用 YOLO 模式
   claude --yolo "重构代码"

   # 检查变更
   git diff

   # 不满意可以回退
   git reset --hard HEAD
   ```

3. **结合 --verbose 使用**
   ```bash
   # 查看详细日志
   claude --yolo "批量转换" --verbose
   ```

---

## 🎨 高级用法

### 组合使用其他参数

```bash
# YOLO + 详细输出
claude --yolo "任务" --verbose

# YOLO + 指定模型
claude --yolo "任务" --model opus
```

### 在 Shell 脚本中使用

```bash
#!/bin/bash

# 数据更新脚本
echo "开始数据更新..."

# 使用 YOLO 模式自动确认
claude --yolo "转换所有新增的合约数据"

# 检查结果
if [ $? -eq 0 ]; then
    echo "✅ 数据更新成功"
else
    echo "❌ 数据更新失败"
    exit 1
fi
```

### 创建快捷别名

```bash
# 在 ~/.zshrc 中添加
alias cy='claude --yolo'

# 使用
cy "批量转换数据"
```

---

## 🔍 确认 YOLO 模式已启用

当你使用 `--yolo` 时，会看到提示：

```
🔥 YOLO 模式已启用！自动确认所有操作
```

---

## 🆘 故障排除

### Q: YOLO 模式没有生效？

**A**: 检查是否重新加载了 shell 配置：

```bash
# 重新加载 zsh 配置
source ~/.zshrc

# 或者打开新的终端窗口
```

### Q: 如何临时禁用 YOLO 模式？

**A**: 不使用 `--yolo` 参数即可：

```bash
# 正常模式
claude "你的任务"
```

### Q: YOLO 模式会影响所有命令吗？

**A**: 不会。只在使用 `--yolo` 参数的命令中生效。

---

## 📚 相关资源

- **数据转换工具**: `projects/qlib_backtest/scripts/data/README.md`
- **使用指南**: `skill.md`
- **Shell 配置**: `~/.zshrc`

---

## ⚡ 快速参考

```bash
# 基本用法
claude --yolo "任务描述"

# 批量数据转换
claude --yolo "转换所有合约到 Qlib 格式"

# 代码重构
claude --yolo "重构代码使用新的数据接口"

# 创建快捷别名
alias cy='claude --yolo'
cy "快速任务"
```

---

**🔥 YOLO 模式 - 让 Claude 更高效！**

⚠️ **记住**: 使用前确认操作，重要数据先备份！
