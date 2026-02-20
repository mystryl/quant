#!/usr/bin/env python3
"""
季度滚动训练框架 (18-24月窗口 + 3月预测)

功能：
1. 使用18-24个月滚动窗口训练
2. 预测未来3个月
3. 每季度滚动一次
4. 评估模型在各时间窗口的稳定性
5. 对比各品种表现

参数：
- 训练窗口：18个月（可配置为24个月）
- 预测窗口：3个月
- 滚动步长：3个月
- 时间范围：2021-2025

品种：
- HC8888.XSGE (热卷)
- I8888.XDCE (铁矿石)
- AU8888.XSGE (黄金)
- CF8888.XZCE (郑棉)
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import sys
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 模型库
try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score
)

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# 品种配置
SYMBOL_CONFIG = {
    'HC8888.XSGE': {'name': '热卷'},
    'I8888.XDCE': {'name': '铁矿石'},
    'AU8888.XSGE': {'name': '黄金'},
    'CF8888.XZCE': {'name': '郑棉'}
}

# 训练配置
TRAIN_WINDOW_MONTHS = 18  # 训练窗口18个月（可改为24）
PREDICT_WINDOW_MONTHS = 3  # 预测窗口3个月
ROLL_STEP_MONTHS = 3  # 滚动步长3个月
START_YEAR = 2021  # 开始年份
END_YEAR = 2025  # 结束年份
TOP_K_FEATURES = 30  # 选择Top30特征

# XGBoost参数
XGBOOST_PARAMS = {
    'n_estimators': 200,
    'max_depth': 5,
    'learning_rate': 0.05,
    'eval_metric': 'logloss',
    'random_state': 42,
    'n_jobs': -1
}


def generate_rolling_windows(
    start_year: int,
    end_year: int,
    train_months: int,
    predict_months: int,
    roll_step: int
) -> List[Dict]:
    """
    生成滚动训练窗口

    Args:
        start_year: 开始年份
        end_year: 结束年份
        train_months: 训练窗口月数
        predict_months: 预测窗口月数
        roll_step: 滚动步长（月数）

    Returns:
        窗口列表，每个窗口包含train_start, train_end, test_start, test_end
    """
    windows = []

    # 从start_year开始，每roll_step个月滚动一次
    current_date = pd.Timestamp(f"{start_year}-01-01")

    while current_date.year < end_year + 1:
        # 训练窗口：当前日期往前推train_months个月
        train_start = current_date - pd.DateOffset(months=train_months)
        train_end = current_date - pd.DateOffset(days=1)

        # 测试窗口：从current_date开始的predict_months个月
        test_start = current_date
        test_end = current_date + pd.DateOffset(months=predict_months) - pd.DateOffset(days=1)

        windows.append({
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'window_id': len(windows) + 1
        })

        # 滚动到下一个窗口
        current_date = current_date + pd.DateOffset(months=roll_step)

    return windows


class QuarterlyRollingTrain:
    """季度滚动训练框架"""

    def __init__(
        self,
        data_base_dir: Path,
        model_output_dir: Path,
        symbols: List[str] = None,
        train_months: int = 18
    ):
        """
        初始化训练框架

        Args:
            data_base_dir: 数据基础目录（multi_symbol目录）
            model_output_dir: 模型输出目录
            symbols: 要训练的品种列表
            train_months: 训练窗口月数
        """
        self.data_base_dir = Path(data_base_dir)
        self.model_output_dir = Path(model_output_dir)
        self.symbols = symbols or list(SYMBOL_CONFIG.keys())
        self.train_months = train_months

        # 创建输出目录
        self.model_output_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir = self.model_output_dir.parent / 'training_results_3month'
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # 生成滚动窗口
        self.windows = generate_rolling_windows(
            START_YEAR,
            END_YEAR,
            self.train_months,
            PREDICT_WINDOW_MONTHS,
            ROLL_STEP_MONTHS
        )

        logger.info("="*80)
        logger.info("季度滚动训练框架 (18月窗口 + 3月预测)")
        logger.info("="*80)
        logger.info(f"数据目录: {self.data_base_dir}")
        logger.info(f"模型输出目录: {self.model_output_dir}")
        logger.info(f"训练品种: {self.symbols}")
        logger.info(f"训练窗口: {self.train_months}个月")
        logger.info(f"预测窗口: {PREDICT_WINDOW_MONTHS}个月")
        logger.info(f"滚动步长: {ROLL_STEP_MONTHS}个月")
        logger.info(f"总窗口数: {len(self.windows)}")
        logger.info("="*80 + "\n")

        if not XGBOOST_AVAILABLE:
            logger.warning("⚠️  XGBoost不可用，将跳过训练")

    def load_symbol_data(self, symbol: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        加载品种的数据（特征、趋势标签、Regime标签）

        Args:
            symbol: 品种代码

        Returns:
            (features, trend_labels, regime_labels)
        """
        logger.info(f"\n加载 {symbol} 的数据...")

        symbol_dir = self.data_base_dir / symbol

        # 1. 加载特征
        feature_file = symbol_dir / 'features' / f'trend_features_{symbol}.csv'
        if not feature_file.exists():
            # 尝试其他可能的文件名
            feature_file = symbol_dir / 'features_binary.csv'
        if not feature_file.exists():
            raise FileNotFoundError(f"特征文件不存在: {symbol_dir}/features/")

        df_features = pd.read_csv(feature_file, index_col=0, parse_dates=True)
        logger.info(f"  ✓ 特征: {df_features.shape}")

        # 2. 加载趋势标签
        trend_label_file = symbol_dir / 'labels' / f'binary_labels_10bars_{symbol}.csv'
        if not trend_label_file.exists():
            trend_label_file = symbol_dir / 'labels' / 'binary_labels_10bars.csv'
        if not trend_label_file.exists():
            raise FileNotFoundError(f"趋势标签文件不存在")

        df_trend_labels = pd.read_csv(trend_label_file, index_col=0, parse_dates=True)
        logger.info(f"  ✓ 趋势标签: {df_trend_labels.shape}")

        # 3. 加载Regime标签
        regime_label_file = symbol_dir / 'labels' / f'volatility_regime_labels_{symbol}.csv'
        if not regime_label_file.exists():
            regime_label_file = symbol_dir / 'labels' / 'volatility_regime_labels.csv'
        if not regime_label_file.exists():
            raise FileNotFoundError(f"Regime标签文件不存在")

        df_regime_labels = pd.read_csv(regime_label_file, index_col=0, parse_dates=True)
        logger.info(f"  ✓ Regime标签: {df_regime_labels.shape}")

        return df_features, df_trend_labels, df_regime_labels

    def prepare_training_data(
        self,
        df_features: pd.DataFrame,
        df_trend_labels: pd.DataFrame,
        df_regime_labels: pd.DataFrame,
        train_start: pd.Timestamp,
        train_end: pd.Timestamp,
        test_start: pd.Timestamp,
        test_end: pd.Timestamp
    ) -> Tuple[Tuple, Tuple]:
        """
        准备训练和测试数据

        Args:
            df_features: 特征DataFrame
            df_trend_labels: 趋势标签
            df_regime_labels: Regime标签
            train_start: 训练开始日期
            train_end: 训练结束日期
            test_start: 测试开始日期
            test_end: 测试结束日期

        Returns:
            (X_train, y_train), (X_test, y_test)
        """
        # 合并数据
        df = df_features.copy()

        # 添加标签（确保列名不冲突）
        if 'trend_label' in df_trend_labels.columns:
            df['trend_label'] = df_trend_labels['trend_label']
        elif 'label' in df_trend_labels.columns:
            df['trend_label'] = df_trend_labels['label']

        if 'regime_label' in df_regime_labels.columns:
            df['regime_label'] = df_regime_labels['regime_label']

        # 删除包含NaN的行
        df = df.dropna(subset=['trend_label', 'regime_label'])

        # 筛选时间范围
        train_data = df.loc[train_start:train_end]
        test_data = df.loc[test_start:test_end]

        # 只在高波动Regime中训练
        train_data = train_data[train_data['regime_label'] == 1]

        # 分离特征和标签
        feature_cols = [col for col in df.columns
                       if col not in ['trend_label', 'regime_label']]

        X_train = train_data[feature_cols].copy()
        y_train = train_data['trend_label']

        X_test = test_data[feature_cols].copy()
        y_test = test_data['trend_label']

        # 清理数据：替换inf和NaN
        X_train = X_train.replace([np.inf, -np.inf], np.nan)
        X_train = X_train.fillna(X_train.median())

        X_test = X_test.replace([np.inf, -np.inf], np.nan)
        X_test = X_test.fillna(X_test.median())

        return (X_train, y_train), (X_test, y_test)

    def train_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_test: pd.DataFrame,
        y_test: pd.Series
    ) -> Tuple:
        """
        训练XGBoost模型

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_test: 测试特征
            y_test: 测试标签

        Returns:
            (model, feature_importance, metrics, top_features)
        """
        # 选择Top K特征（基于特征重要性）
        # 首先用所有特征训练一个初步模型
        preliminary_model = XGBClassifier(**XGBOOST_PARAMS)
        preliminary_model.fit(X_train, y_train, verbose=False)

        # 获取特征重要性
        feature_importance = pd.DataFrame({
            'feature': X_train.columns,
            'importance': preliminary_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 选择Top K特征
        top_features = feature_importance.head(TOP_K_FEATURES)['feature'].tolist()

        # 用Top K特征重新训练
        X_train_top = X_train[top_features]
        X_test_top = X_test[top_features]

        # 训练最终模型
        model = XGBClassifier(**XGBOOST_PARAMS)
        model.fit(X_train_top, y_train, verbose=False)

        # 评估
        y_pred = model.predict(X_test_top)
        y_pred_proba = model.predict_proba(X_test_top)[:, 1]

        metrics = self.calculate_metrics(y_test, y_pred, y_pred_proba)

        # 更新特征重要性（只保留Top K）
        feature_importance = feature_importance.head(TOP_K_FEATURES)

        return model, feature_importance, metrics, top_features

    def calculate_metrics(
        self,
        y_true: pd.Series,
        y_pred: np.ndarray,
        y_pred_proba: np.ndarray
    ) -> Dict:
        """
        计算评估指标

        Args:
            y_true: 真实标签
            y_pred: 预测标签
            y_pred_proba: 预测概率

        Returns:
            指标字典
        """
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='binary', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='binary', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='binary', zero_division=0),
            'auc': roc_auc_score(y_true, y_pred_proba)
        }

        # 震荡召回率（label=0的召回率）
        y_true_array = np.array(y_true)
        y_pred_array = np.array(y_pred)

        range_mask = y_true_array == 0
        if range_mask.sum() > 0:
            metrics['recall_range'] = (y_pred_array[range_mask] == 0).sum() / range_mask.sum()
        else:
            metrics['recall_range'] = 0.0

        # 有趋势召回率（label=1的召回率）
        trend_mask = y_true_array == 1
        if trend_mask.sum() > 0:
            metrics['recall_trend'] = (y_pred_array[trend_mask] == 1).sum() / trend_mask.sum()
        else:
            metrics['recall_trend'] = 0.0

        return metrics

    def train_single_window(
        self,
        symbol: str,
        window: Dict,
        df_features: pd.DataFrame,
        df_trend_labels: pd.DataFrame,
        df_regime_labels: pd.DataFrame
    ) -> Dict:
        """
        训练单个品种单个窗口

        Args:
            symbol: 品种代码
            window: 窗口信息
            df_features: 特征
            df_trend_labels: 趋势标签
            df_regime_labels: Regime标签

        Returns:
            训练结果字典
        """
        window_id = window['window_id']
        train_start = window['train_start'].strftime('%Y-%m-%d')
        train_end = window['train_end'].strftime('%Y-%m-%d')
        test_start = window['test_start'].strftime('%Y-%m-%d')
        test_end = window['test_end'].strftime('%Y-%m-%d')

        logger.info(f"\n{'='*60}")
        logger.info(f"{symbol} - 窗口{window_id}: {test_start[:7]}")
        logger.info(f"  训练: {train_start} ~ {train_end}")
        logger.info(f"  测试: {test_start} ~ {test_end}")
        logger.info(f"{'='*60}")

        result = {
            'symbol': symbol,
            'window_id': window_id,
            'train_start': train_start,
            'train_end': train_end,
            'test_start': test_start,
            'test_end': test_end,
            'success': False,
            'error': None,
            'metrics': {},
            'top_features': []
        }

        try:
            # 准备数据
            (X_train, y_train), (X_test, y_test) = self.prepare_training_data(
                df_features, df_trend_labels, df_regime_labels,
                window['train_start'], window['train_end'],
                window['test_start'], window['test_end']
            )

            # 检查数据是否足够
            if len(X_train) < 100 or len(X_test) < 20:
                raise ValueError(f"数据不足: 训练集={len(X_train)}, 测试集={len(X_test)}")

            # 训练模型
            model, feature_importance, metrics, top_features = self.train_model(
                X_train, y_train, X_test, y_test
            )

            # 保存模型
            model_file = self.model_output_dir / f'{symbol}_window{window_id:02d}.pkl'
            with open(model_file, 'wb') as f:
                pickle.dump({
                    'model': model,
                    'features': top_features,
                    'feature_importance': feature_importance,
                    'metrics': metrics,
                    'symbol': symbol,
                    'window_id': window_id,
                    'train_period': f"{train_start} ~ {train_end}",
                    'test_period': f"{test_start} ~ {test_end}"
                }, f)

            logger.info(f"  ✓ AUC: {metrics['auc']:.4f}, Acc: {metrics['accuracy']:.4f}")

            result['success'] = True
            result['metrics'] = metrics
            result['top_features'] = top_features

        except Exception as e:
            result['error'] = str(e)
            logger.warning(f"  ✗ 训练失败: {e}")

        return result

    def train_single_symbol(self, symbol: str) -> List[Dict]:
        """
        训练单个品种的所有窗口

        Args:
            symbol: 品种代码

        Returns:
            所有窗口的结果列表
        """
        logger.info("\n" + "="*80)
        logger.info(f"训练品种: {symbol} ({SYMBOL_CONFIG[symbol]['name']})")
        logger.info(f"滚动窗口数: {len(self.windows)}")
        logger.info("="*80)

        start_time = datetime.now()
        results = []

        try:
            # 加载数据
            df_features, df_trend_labels, df_regime_labels = self.load_symbol_data(symbol)

            # 逐窗口训练
            for window in self.windows:
                result = self.train_single_window(
                    symbol, window,
                    df_features, df_trend_labels, df_regime_labels
                )
                results.append(result)

            elapsed = (datetime.now() - start_time).total_seconds()
            logger.info(f"\n✓ {symbol} 训练完成！耗时: {elapsed:.1f}秒")

            # 统计成功率
            success_count = sum(1 for r in results if r['success'])
            logger.info(f"  成功窗口: {success_count}/{len(results)}")

        except Exception as e:
            logger.error(f"\n✗ {symbol} 训练失败: {e}")
            import traceback
            traceback.print_exc()

        return results

    def train_all_symbols(self) -> Dict[str, List[Dict]]:
        """
        训练所有品种

        Returns:
            所有品种的结果字典
        """
        logger.info("\n" + "="*80)
        logger.info("开始批量训练所有品种")
        logger.info("="*80)

        all_results = {}

        for i, symbol in enumerate(self.symbols, 1):
            logger.info(f"\n{'='*80}")
            logger.info(f"进度: {i}/{len(self.symbols)} - {symbol}")
            logger.info(f"{'='*80}")

            results = self.train_single_symbol(symbol)
            all_results[symbol] = results

            # 显式清理内存
            import gc
            gc.collect()

        # 汇总结果
        self._summarize_results(all_results)

        # 保存汇总结果
        self._save_summary(all_results)

        return all_results

    def _summarize_results(self, all_results: Dict[str, List[Dict]]):
        """汇总训练结果"""
        logger.info("\n" + "="*80)
        logger.info("滚动训练结果汇总")
        logger.info("="*80)

        # 按品种汇总
        for symbol, results in all_results.items():
            success_results = [r for r in results if r['success']]

            if success_results:
                aucs = [r['metrics']['auc'] for r in success_results]
                accs = [r['metrics']['accuracy'] for r in success_results]

                logger.info(f"\n{symbol} ({SYMBOL_CONFIG[symbol]['name']}):")
                logger.info(f"  成功窗口: {len(success_results)}/{len(results)}")
                logger.info(f"  AUC: 均值={np.mean(aucs):.4f}, 标准差={np.std(aucs):.4f}, 范围=[{np.min(aucs):.4f}, {np.max(aucs):.4f}]")
                logger.info(f"  Acc: 均值={np.mean(accs):.4f}, 标准差={np.std(accs):.4f}")

    def _save_summary(self, all_results: Dict[str, List[Dict]]):
        """保存详细结果"""
        import json

        # 保存JSON格式的详细结果
        json_file = self.results_dir / f'rolling_results_{self.train_months}months.json'

        # 转换为可序列化的格式
        serializable_results = {}
        for symbol, results in all_results.items():
            serializable_results[symbol] = []
            for result in results:
                serializable_result = {
                    'symbol': result['symbol'],
                    'window_id': result['window_id'],
                    'train_start': result['train_start'],
                    'train_end': result['train_end'],
                    'test_start': result['test_start'],
                    'test_end': result['test_end'],
                    'success': result['success'],
                    'error': result['error'],
                    'metrics': result['metrics'],
                    'top_features': result['top_features']
                }
                serializable_results[symbol].append(serializable_result)

        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_results, f, indent=2, ensure_ascii=False)

        logger.info(f"\n✓ 详细结果已保存: {json_file}")

        # 生成CSV汇总
        summary_data = []
        for symbol, results in all_results.items():
            for result in results:
                if result['success']:
                    summary_data.append({
                        '品种': symbol,
                        '名称': SYMBOL_CONFIG[symbol]['name'],
                        '窗口ID': result['window_id'],
                        '训练开始': result['train_start'],
                        '训练结束': result['train_end'],
                        '测试开始': result['test_start'],
                        '测试结束': result['test_end'],
                        '准确率': f"{result['metrics']['accuracy']:.4f}",
                        'AUC': f"{result['metrics']['auc']:.4f}",
                        '震荡召回率': f"{result['metrics'].get('recall_range', 0):.4f}",
                        '趋势召回率': f"{result['metrics'].get('recall_trend', 0):.4f}",
                        'F1': f"{result['metrics']['f1']:.4f}"
                    })

        if summary_data:
            df_summary = pd.DataFrame(summary_data)
            csv_file = self.results_dir / f'rolling_summary_{self.train_months}months.csv'
            df_summary.to_csv(csv_file, index=False)
            logger.info(f"✓ 汇总CSV已保存: {csv_file}")


def main():
    """主函数"""
    # 路径配置（基于脚本位置）
    project_root = Path(__file__).parent.parent
    data_base_dir = project_root / 'data' / 'multi_symbol'
    model_output_dir = project_root / 'models' / 'rolling_3month'

    # 验证数据目录
    if not data_base_dir.exists():
        raise FileNotFoundError(f"数据目录不存在: {data_base_dir}")

    # 创建训练框架（18个月训练窗口）
    trainer = QuarterlyRollingTrain(
        data_base_dir=data_base_dir,
        model_output_dir=model_output_dir,
        train_months=18  # 可改为24测试
    )

    # 训练所有品种
    all_results = trainer.train_all_symbols()

    return all_results


if __name__ == '__main__':
    main()
