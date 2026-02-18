# External Frameworks

这个目录用于存放外部依赖框架。

## 推荐使用 Git Submodule

对于成熟的外部项目，建议使用 git submodule 来管理，这样不会将第三方代码直接提交到你的仓库中。

### 添加新的 framework

```bash
# 例如添加 qlib
git submodule add https://github.com/microsoft/qlib.git frameworks/qlib
git commit -m "Add framework: qlib"
```

### 克隆包含 submodule 的仓库

```bash
# 克隆时同时获取 submodule
git clone --recurse-submodules <quant-repo-url>

# 或者在克隆后初始化
git submodule update --init --recursive
```

### 更新 submodule

```bash
# 更新所有 submodule 到最新版本
git submodule update --remote

# 更新特定的 submodule
cd frameworks/qlib
git pull origin main
cd ../..
git add frameworks/qlib
git commit -m "Update qlib submodule"
```

## 当前 Frameworks

目前本目录为空。如需添加外部框架，请使用 git submodule。
