"""
价格监控脚本
定时检测期货价格是否突破或接近设定价位
配置从 config.yaml 读取
"""

import akshare as ak
import time
import yaml
import os
from datetime import datetime
from pathlib import Path
from typing import List, Dict


class PriceMonitor:
    """价格监控类"""

    def __init__(self, config_path: str = None):
        """
        初始化监控器

        Args:
            config_path: 配置文件路径
        """
        # 获取配置文件路径
        if config_path is None:
            config_path = Path(__file__).parent / 'config.yaml'

        self.config_path = Path(config_path)

        # 加载配置
        self.config = self._load_config()

        # 创建日志目录
        if self.config.get('log', {}).get('enabled'):
            log_file = self.config['log']['file']
            log_dir = Path(log_file).parent
            log_dir.mkdir(parents=True, exist_ok=True)

    def _load_config(self) -> dict:
        """加载配置文件"""
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        except FileNotFoundError:
            print(f"配置文件不存在: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            print(f"配置文件格式错误: {e}")
            raise

    def get_latest_price(self, symbol: str) -> dict:
        """
        获取期货最新价格

        Args:
            symbol: 期货合约代码

        Returns:
            包含最新价格的字典
        """
        try:
            df = ak.futures_zh_spot(symbol=symbol, market='CF')

            if df.empty:
                return None

            # 获取最新数据
            row = df.iloc[0]
            price = row['current_price'] if 'current_price' in row else row['last_close']

            return {
                'symbol': symbol,
                'price': float(price),
                'time': row['time'] if 'time' in row else datetime.now().strftime('%H%M%S'),
                'open': float(row['open']) if 'open' in row else None,
                'high': float(row['high']) if 'high' in row else None,
                'low': float(row['low']) if 'low' in row else None
            }

        except Exception as e:
            print(f"获取价格失败 ({symbol}): {e}")
            return None

    def check_price_trigger(self, current_price: float, targets: List[float],
                           threshold: float) -> List[Dict]:
        """
        检查价格是否触发预警

        Args:
            current_price: 当前价格
            targets: 目标价位列表
            threshold: 触发阈值

        Returns:
            触发的目标列表
        """
        triggered = []

        for target in targets:
            distance = current_price - target

            # 突破或接近检测
            if abs(distance) <= threshold:
                status = "接近" if abs(distance) <= threshold else "突破"
                triggered.append({
                    'target': target,
                    'distance': distance,
                    'status': status
                })

        return triggered

    def log_message(self, message: str):
        """记录日志"""
        print(message)

        if self.config.get('log', {}).get('enabled'):
            log_file = self.config['log']['file']
            try:
                with open(log_file, 'a', encoding='utf-8') as f:
                    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                    f.write(f"[{timestamp}] {message}\n")
            except Exception as e:
                print(f"写入日志失败: {e}")

    def run_once(self):
        """执行一次监控检测"""
        monitors = self.config.get('monitors', [])

        if not monitors:
            self.log_message("⚠️  没有配置监控品种")
            return

        for monitor in monitors:
            symbol = monitor['symbol']
            name = monitor.get('name', symbol)
            targets = monitor['targets']
            threshold = monitor.get('threshold', 0.2)

            # 获取最新价格
            data = self.get_latest_price(symbol)

            if data:
                message = f"[{data['time']}] {name}({symbol}) 价格: {data['price']:.2f}"
                self.log_message(message)

                # 检测触发
                triggered = self.check_price_trigger(data['price'], targets, threshold)

                if triggered:
                    alert_msg = f"  ⚠️  触发预警！"
                    self.log_message(alert_msg)

                    for t in triggered:
                        self.log_message(f"    目标: {t['target']:.2f} | "
                                       f"当前: {data['price']:.2f} | "
                                       f"距离: {t['distance']:+.2f} | "
                                       f"状态: {t['status']}")
                else:
                    # 显示距离所有目标价位的距离
                    for target in targets:
                        distance = data['price'] - target
                        self.log_message(f"    目标 {target:.2f}: {distance:+.2f}")
            else:
                error_msg = f"[{datetime.now().strftime('%H:%M:%S')}] {name}({symbol}) 获取价格失败"
                self.log_message(error_msg)

    def run(self):
        """持续监控"""
        interval = self.config.get('interval', 300)

        monitors = self.config.get('monitors', [])

        print("=" * 80)
        print("价格监控系统")
        print("=" * 80)
        print(f"配置文件: {self.config_path}")
        print(f"监控品种数量: {len(monitors)}")
        for m in monitors:
            print(f"  - {m.get('name', m['symbol'])}({m['symbol']}): {m['targets']}")
        print(f"检测间隔: {interval} 秒")
        if self.config.get('log', {}).get('enabled'):
            print(f"日志文件: {self.config['log']['file']}")
        print("=" * 80)
        print("按 Ctrl+C 停止监控\n")

        try:
            while True:
                self.run_once()
                print(f"\n下次检测: {interval} 秒后...")
                time.sleep(interval)

        except KeyboardInterrupt:
            print("\n\n监控已停止")


def main():
    """主函数"""
    monitor = PriceMonitor()
    monitor.run()


if __name__ == '__main__':
    main()
