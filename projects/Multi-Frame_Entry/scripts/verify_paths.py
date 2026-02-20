#!/usr/bin/env python3
"""
启动前路径验证脚本

检查所有关键路径和依赖，确保实验可以正常运行
"""
import sys
from pathlib import Path
import logging

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

logger = logging.getLogger(__name__)

# 颜色输出
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
RESET = '\033[0m'


def check_path(path, description, must_exist=True):
    """检查单个路径"""
    print(f"检查 {description}:", end=" ")

    if must_exist:
        if path.exists():
            size = path.stat().st_size if path.is_file() else sum(f.stat().st_size for f in path.rglob('*') if f.is_file())
            print(f"{GREEN}✓ 存在{RESET} ({path})")
            if path.is_dir():
                print(f"  大小: {size / 1024 / 1024:.1f} MB")
            return True
        else:
            print(f"{RED}✗ 不存在{RESET} ({path})")
            return False
    else:
        if not path.exists():
            print(f"{YELLOW}⚠️  不存在（将自动创建）{RESET} ({path})")
        else:
            print(f"{GREEN}✓ 存在{RESET} ({path})")
        return True


def verify_dependencies():
    """验证Python依赖"""
    print("\n" + "="*60)
    print("检查Python依赖")
    print("="*60)

    dependencies = [
        ('pandas', 'pandas'),
        ('numpy', 'numpy'),
        ('xgboost', 'xgboost'),
        ('sklearn', 'scikit-learn'),
        ('openpyxl', 'openpyxl'),
        ('joblib', 'joblib'),
    ]

    all_ok = True
    for module_name, package_name in dependencies:
        try:
            __import__(module_name)
            print(f"{GREEN}✓{RESET} {package_name}")
        except ImportError:
            print(f"{RED}✗{RESET} {package_name} - 未安装")
            all_ok = False

    return all_ok


def verify_paths():
    """验证所有路径"""
    print("\n" + "="*60)
    print("检查路径")
    print("="*60)

    results = []

    # 源数据目录
    results.append(check_path(
        Path('/Users/mystryl/Documents/Quant/K线数据库/期货商品指数_parquet'),
        "源数据目录（期货商品指数_parquet）"
    ))

    # 项目目录结构
    results.append(check_path(
        project_root / 'scripts',
        "脚本目录"
    ))

    results.append(check_path(
        project_root / 'labels',
        "标签目录"
    ))

    results.append(check_path(
        project_root / 'features',
        "特征目录"
    ))

    # 输出目录（可以不存在）
    check_path(
        project_root / 'data',
        "数据输出目录",
        must_exist=False
    )

    check_path(
        project_root / 'models',
        "模型输出目录",
        must_exist=False
    )

    check_path(
        project_root / 'logs',
        "日志目录",
        must_exist=False
    )

    # 检查关键脚本
    critical_scripts = [
        'scripts/data_pipeline_multi_symbol.py',
        'scripts/rolling_train_multi_symbol.py',
        'scripts/run_multi_symbol_experiment.py',
    ]

    print("\n" + "-"*60)
    print("检查关键脚本")
    print("-"*60)

    for script in critical_scripts:
        results.append(check_path(
            project_root / script,
            f"脚本 {script}"
        ))

    return all(results)


def verify_source_data():
    """验证源数据文件"""
    print("\n" + "="*60)
    print("检查源数据文件")
    print("="*60)

    source_dir = Path('/Users/mystryl/Documents/Quant/K线数据库/期货商品指数_parquet')

    required_files = {
        'HC8888.XSGE.parquet': '热卷',
        'I8888.XDCE.parquet': '铁矿石',
        'AU8888.XSGE.parquet': '黄金',
        'CF8888.XZCE.parquet': '郑棉',
        'IF8888.CCFX.parquet': '股指期货'
    }

    all_ok = True
    for filename, name in required_files.items():
        filepath = source_dir / filename
        exists = check_path(filepath, f"{name} ({filename})")
        if not exists:
            all_ok = False

    return all_ok


def main():
    """主函数"""
    print("\n" + "="*60)
    print("多品种滚动训练实验 - 启动前验证")
    print("="*60)
    print(f"项目根目录: {project_root}")

    # 1. 检查依赖
    deps_ok = verify_dependencies()

    # 2. 检查路径
    paths_ok = verify_paths()

    # 3. 检查源数据
    data_ok = verify_source_data()

    # 总结
    print("\n" + "="*60)
    print("验证结果")
    print("="*60)

    all_ok = deps_ok and paths_ok and data_ok

    if all_ok:
        print(f"{GREEN}✓ 所有检查通过！{RESET}")
        print("\n可以安全启动实验：")
        print("  python3 scripts/run_multi_symbol_experiment.py")
        return 0
    else:
        print(f"{RED}✗ 存在问题，请先修复上述错误{RESET}")
        return 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
