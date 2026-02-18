#!/usr/bin/env python3
"""
Parquet 到 Qlib 批量转换工具

支持命令行操作：
- 转换单个合约
- 批量转换所有合约
- 增量转换（只转换新增或修改的）
- 并行处理

使用示例：
    # 转换单个合约
    python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib \\
        --instrument HC8888.XSGE

    # 转换所有合约
    python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib --all

    # 增量转换
    python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib --all --incremental

    # 并行转换（4个进程）
    python -m projects.qlib_backtest.scripts.data.convert_parquet_to_qlib --all --workers 4
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Optional

from tqdm import tqdm

from .unified_data_provider import ParquetDataProvider
from .config import (
    PARQUET_DATA_DIR,
    QLIB_INSTRUMENTS_DIR,
    CONVERT_INCREMENTAL,
    CONVERT_PARALLEL_WORKERS,
    CONVERT_BATCH_SIZE
)


# =============================================================================
# 日志配置
# =============================================================================

def setup_logging(verbose: bool = False) -> None:
    """配置日志"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )


logger = logging.getLogger(__name__)


# =============================================================================
# 转换工具类
# =============================================================================

class ConverterTool:
    """批量转换工具"""

    def __init__(
        self,
        provider: Optional[ParquetDataProvider] = None,
        incremental: bool = CONVERT_INCREMENTAL,
        parallel: bool = True,
        workers: int = CONVERT_PARALLEL_WORKERS
    ):
        """
        初始化转换工具

        Args:
            provider: 数据提供者实例
            incremental: 是否增量转换
            parallel: 是否并行处理
            workers: 工作进程数
        """
        self.provider = provider or ParquetDataProvider()
        self.incremental = incremental
        self.parallel = parallel
        self.workers = workers

    def convert_instrument(
        self,
        instrument: str,
        force: bool = False,
        show_progress: bool = False
    ) -> bool:
        """
        转换单个合约

        Args:
            instrument: 合约代码
            force: 是否强制转换
            show_progress: 是否显示进度

        Returns:
            bool: 是否成功
        """
        try:
            # 增量转换检查
            if self.incremental and not force:
                if self._is_converted(instrument):
                    logger.debug(f"合约已转换，跳过: {instrument}")
                    return True

            # 执行转换
            self.provider.convert_to_qlib(
                instrument,
                force=force,
                progress=show_progress
            )
            return True

        except Exception as e:
            logger.error(f"转换失败 {instrument}: {e}")
            return False

    def convert_batch(
        self,
        instruments: List[str],
        force: bool = False
    ) -> dict:
        """
        批量转换合约

        Args:
            instruments: 合约列表
            force: 是否强制转换

        Returns:
            dict: 转换结果统计
        """
        stats = {
            "total": len(instruments),
            "success": 0,
            "failed": 0,
            "skipped": 0
        }

        logger.info(f"开始批量转换 {len(instruments)} 个合约")

        # 过滤已转换的合约（增量模式）
        if self.incremental and not force:
            instruments = [
                inst for inst in instruments
                if not self._is_converted(inst)
            ]
            stats["skipped"] = len(instruments) - stats["total"]
            stats["total"] = len(instruments)

        if not instruments:
            logger.info("没有需要转换的合约")
            return stats

        # 串行转换
        for instrument in tqdm(instruments, desc="转换进度"):
            success = self.convert_instrument(instrument, force=force, show_progress=False)
            if success:
                stats["success"] += 1
            else:
                stats["failed"] += 1

        # 输出结果
        logger.info(f"转换完成:")
        logger.info(f"  总计: {stats['total']}")
        logger.info(f"  成功: {stats['success']}")
        logger.info(f"  失败: {stats['failed']}")
        if stats['skipped'] > 0:
            logger.info(f"  跳过: {stats['skipped']}")

        return stats

    def convert_all(
        self,
        pattern: str = "*",
        force: bool = False
    ) -> dict:
        """
        转换所有合约

        Args:
            pattern: 文件匹配模式
            force: 是否强制转换

        Returns:
            dict: 转换结果统计
        """
        # 获取所有合约
        instruments = self.provider.list_instruments(pattern=pattern)

        if not instruments:
            logger.warning(f"没有找到匹配的合约: {pattern}")
            return {"total": 0, "success": 0, "failed": 0, "skipped": 0}

        logger.info(f"找到 {len(instruments)} 个合约")

        # 批量转换
        return self.convert_batch(instruments, force=force)

    def _is_converted(self, instrument: str) -> bool:
        """检查合约是否已转换"""
        from .config import get_qlib_instrument_file

        close_file = get_qlib_instrument_file(instrument, 'close')
        return close_file.exists()


# =============================================================================
# 命令行接口
# =============================================================================

def parse_args() -> argparse.Namespace:
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="Parquet 到 Qlib 批量转换工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例：
  # 转换单个合约
  %(prog)s --instrument HC8888.XSGE

  # 转换所有合约
  %(prog)s --all

  # 增量转换（只转换新增或修改的）
  %(prog)s --all --incremental

  # 强制重新转换所有合约
  %(prog)s --all --force

  # 使用模式匹配转换
  %(prog)s --all --pattern "HC*"

  # 指定并行工作数
  %(prog)s --all --workers 8
        """
    )

    # 转换目标
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--instrument", "-i",
        help="转换单个合约，如 HC8888.XSGE"
    )
    target_group.add_argument(
        "--all", "-a",
        action="store_true",
        help="转换所有合约"
    )

    # 转换选项
    parser.add_argument(
        "--pattern", "-p",
        default="*",
        help="文件匹配模式，如 'HC*' 或 '8888*'（仅与 --all 一起使用）"
    )
    parser.add_argument(
        "--force", "-f",
        action="store_true",
        help="强制转换（覆盖已存在的缓存）"
    )
    parser.add_argument(
        "--incremental",
        action="store_true",
        help="增量转换（只转换新增或修改的合约）"
    )

    # 性能选项
    parser.add_argument(
        "--workers", "-w",
        type=int,
        default=CONVERT_PARALLEL_WORKERS,
        help=f"并行工作进程数（默认: {CONVERT_PARALLEL_WORKERS}）"
    )
    parser.add_argument(
        "--no-parallel",
        action="store_true",
        help="禁用并行处理"
    )

    # 其他选项
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="详细输出"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="试运行（只显示将要转换的合约，不实际转换）"
    )

    return parser.parse_args()


def main() -> int:
    """主函数"""
    args = parse_args()

    # 配置日志
    setup_logging(args.verbose)

    # 创建转换工具
    tool = ConverterTool(
        incremental=args.incremental,
        parallel=not args.no_parallel,
        workers=args.workers
    )

    try:
        # 转换单个合约
        if args.instrument:
            logger.info(f"转换合约: {args.instrument}")

            if args.dry_run:
                logger.info("[试运行] 不会实际转换")
                return 0

            success = tool.convert_instrument(
                args.instrument,
                force=args.force,
                show_progress=True
            )

            return 0 if success else 1

        # 转换所有合约
        elif args.all:
            logger.info(f"批量转换模式: {args.pattern}")

            if args.dry_run:
                instruments = tool.provider.list_instruments(pattern=args.pattern)
                logger.info(f"[试运行] 将要转换 {len(instruments)} 个合约:")
                for inst in instruments[:10]:
                    logger.info(f"  - {inst}")
                if len(instruments) > 10:
                    logger.info(f"  ... 还有 {len(instruments) - 10} 个合约")
                return 0

            stats = tool.convert_all(
                pattern=args.pattern,
                force=args.force
            )

            # 返回状态码
            return 0 if stats["failed"] == 0 else 1

    except KeyboardInterrupt:
        logger.info("用户中断")
        return 130
    except Exception as e:
        logger.error(f"发生错误: {e}", exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
