"""
趋势模型训练模块

使用 RandomForest 或 XGBoost 训练趋势分类模型。

重要：
- 严格时间序列分割（禁止 shuffle）
- 输出三类概率：P(上涨), P(震荡), P(下跌)
- 保存模型供后续使用
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from typing import Dict, Tuple, List
from datetime import datetime

# 模型库
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    RandomForestClassifier = None

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

logger = logging.getLogger(__name__)


class TrendModel:
    """趋势分类模型"""

    def __init__(
        self,
        model_type: str = 'random_forest',
        n_estimators: int = 100,
        max_depth: int = 5,
        min_samples_split: int = 50,
        min_samples_leaf: int = 20,
        learning_rate: float = 0.05,
        random_state: int = 42,
        top_k_features: int = 20
    ):
        """
        初始化模型（优化参数，减少过拟合）

        Args:
            model_type: 模型类型 ('random_forest' 或 'xgboost')
            n_estimators: 树的数量
            max_depth: 最大深度（降低以减少过拟合）
            min_samples_split: 最小分割样本数（增加以减少过拟合）
            min_samples_leaf: 最小叶子节点样本数
            learning_rate: 学习率（XGBoost专用）
            random_state: 随机种子
            top_k_features: 特征选择数量
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_samples_leaf = min_samples_leaf
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.top_k_features = top_k_features
        self.model = None
        self.feature_names = None
        self.label_names = [-1, 0, 1]  # 下跌, 震荡, 上涨
        self.selected_features = None  # 特征选择后的特征

        logger.info(f"趋势模型初始化（优化版本）:")
        logger.info(f"  模型类型: {model_type}")
        logger.info(f"  参数: n_estimators={n_estimators}, max_depth={max_depth}")
        if model_type == 'xgboost':
            logger.info(f"  学习率: learning_rate={learning_rate}")
        logger.info(f"  正则化: min_samples_split={min_samples_split}, min_samples_leaf={min_samples_leaf}")
        logger.info(f"  特征选择: Top {top_k_features}")

        if not SKLEARN_AVAILABLE:
            raise ImportError("sklearn 未安装，请运行: pip install scikit-learn")
        if model_type == 'xgboost' and not XGBOOST_AVAILABLE:
            raise ImportError("XGBoost 未安装，请运行: pip install xgboost")

    def select_features(self, X_train, y_train):
        """
        特征选择：只保留 Top K 特征

        这里使用简单的基于训练集的特征重要性排序
        实际应用中应使用验证集进行特征选择

        Args:
            X_train: 训练特征
            y_train: 训练标签
        """
        logger.info(f"\n执行特征选择（保留 Top {self.top_k_features}）...")

        # 先用所有特征训练一个临时模型获取特征重要性
        if self.model_type == 'random_forest':
            temp_model = RandomForestClassifier(
                n_estimators=50,  # 较少的树
                max_depth=5,
                min_samples_split=50,
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == 'xgboost':
            temp_model = XGBClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

        temp_model.fit(X_train, y_train)

        # 获取特征重要性
        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': temp_model.feature_importances_
        }).sort_values('importance', ascending=False)

        # 选择 Top K 特征
        selected = importances.head(self.top_k_features)
        self.selected_features = selected['feature'].tolist()

        logger.info(f"  选择的特征:")
        for idx, row in selected.iterrows():
            logger.info(f"    {idx+1}. {row['feature']}: {row['importance']:.4f}")

        return self.selected_features

    def build_model(self):
        """构建模型"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                random_state=self.random_state,
                n_jobs=-1,  # 使用所有CPU核心
                class_weight='balanced'  # 自动平衡类别权重
            )
            logger.info("✓ 使用 RandomForestClassifier（优化参数）")
        elif self.model_type == 'xgboost':
            # 计算类别权重
            # XGBoost需要手动设置scale_pos_weight，但我们有多分类问题
            # 使用sample_weight或让XGBoost自动处理
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='mlogloss',
                # 多类别平衡参数
                objective='multi:softprob',
                num_class=3
            )
            logger.info("✓ 使用 XGBClassifier（优化参数）")
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def load_data(
        self,
        data_file: Path = None
    ) -> pd.DataFrame:
        """
        加载特征数据

        Args:
            data_file: 数据文件路径

        Returns:
            DataFrame
        """
        if data_file is None:
            data_file = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data/features/trend_features.csv')

        logger.info(f"\n加载数据: {data_file}")
        df = pd.read_csv(data_file)
        df['datetime'] = pd.to_datetime(df['datetime'])

        logger.info(f"  数据形状: {df.shape}")
        logger.info(f"  时间范围: {df['datetime'].min()} ~ {df['datetime'].max()}")

        return df

    def prepare_features(
        self,
        df: pd.DataFrame,
        feature_cols: List[str] = None
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """
        准备特征和标签

        Args:
            df: 数据DataFrame
            feature_cols: 特征列名列表

        Returns:
            (X, y) 特征和标签
        """
        # 如果没有指定特征列，自动识别
        if feature_cols is None:
            exclude_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume',
                           'future_return', 'trend_label']
            feature_cols = [col for col in df.columns if col not in exclude_cols]

        logger.info(f"\n准备特征和标签:")
        logger.info(f"  特征数量: {len(feature_cols)}")
        logger.info(f"  特征列表: {feature_cols[:10]}...")  # 显示前10个

        # 删除缺失值
        df_clean = df[feature_cols + ['trend_label', 'datetime']].dropna()
        logger.info(f"  删除缺失值后: {len(df_clean)} 行 (原{len(df)} 行)")

        # 提取特征和标签
        X = df_clean[feature_cols]
        y = df_clean['trend_label']

        # 标签转换为 0, 1, 2 (sklearn 要求)
        # -1(下跌) -> 0, 0(震荡) -> 1, 1(上涨) -> 2
        y = y.map({-1: 0, 0: 1, 1: 2})

        self.feature_names = feature_cols

        # 标签分布
        label_counts = y.value_counts()
        logger.info(f"  标签分布:")
        for label, count in label_counts.items():
            label_name = {0: '下跌', 1: '震荡', 2: '上涨'}[label]
            logger.info(f"    {label_name}: {count} ({count/len(y)*100:.1f}%)")

        return X, y, df_clean

    def time_series_split(
        self,
        df: pd.DataFrame,
        train_years: Tuple[int, int] = (2022, 2023),
        valid_year: int = 2024,
        test_year: int = 2025
    ) -> Dict:
        """
        时间序列分割（严格按照年份）

        重要：禁止 shuffle！

        Args:
            df: 数据DataFrame
            train_years: 训练集年份
            valid_year: 验证集年份
            test_year: 测试集年份

        Returns:
            字典包含 X_train, X_valid, X_test, y_train, y_valid, y_test
        """
        logger.info(f"\n执行时间序列分割:")
        logger.info(f"  训练集: {train_years[0]}-{train_years[1]}")
        logger.info(f"  验证集: {valid_year}")
        logger.info(f"  测试集: {test_year}")

        # 提取年份
        df['year'] = df['datetime'].dt.year

        # 分割数据
        train_mask = df['year'].isin(train_years)
        valid_mask = df['year'] == valid_year
        test_mask = df['year'] == test_year

        X_train = df[train_mask][self.feature_names]
        y_train = df[train_mask]['trend_label'].map({-1: 0, 0: 1, 1: 2})

        X_valid = df[valid_mask][self.feature_names]
        y_valid = df[valid_mask]['trend_label'].map({-1: 0, 0: 1, 1: 2})

        X_test = df[test_mask][self.feature_names]
        y_test = df[test_mask]['trend_label'].map({-1: 0, 0: 1, 1: 2})

        # 删除缺失值
        X_train = X_train.dropna()
        y_train = y_train[X_train.index]

        X_valid = X_valid.dropna()
        y_valid = y_valid[X_valid.index]

        X_test = X_test.dropna()
        y_test = y_test[X_test.index]

        logger.info(f"\n分割结果:")
        logger.info(f"  训练集: {len(X_train)} 样本")
        logger.info(f"  验证集: {len(X_valid)} 样本")
        logger.info(f"  测试集: {len(X_test)} 样本")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_valid': X_valid, 'y_valid': y_valid,
            'X_test': X_test, 'y_test': y_test
        }

    def train(self, X_train, y_train, X_valid, y_valid):
        """
        训练模型（带特征选择）

        Args:
            X_train: 训练特征
            y_train: 训练标签
            X_valid: 验证特征
            y_valid: 验证标签
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"开始训练模型（优化版本）")
        logger.info(f"{'='*60}")

        # 特征选择
        selected_features = self.select_features(X_train, y_train)

        # 只使用选中的特征
        X_train_selected = X_train[selected_features]
        X_valid_selected = X_valid[selected_features]

        logger.info(f"\n使用特征: {len(selected_features)}")
        logger.info(f"训练数据: {X_train_selected.shape}")

        # 构建模型
        self.build_model()

        # 训练
        self.model.fit(X_train_selected, y_train)

        # 训练集评估
        train_score = self.model.score(X_train_selected, y_train)
        logger.info(f"✓ 训练集准确率: {train_score:.4f}")

        # 验证集评估
        valid_score = self.model.score(X_valid_selected, y_valid)
        logger.info(f"✓ 验证集准确率: {valid_score:.4f}")

        # 过拟合检查
        overfitting = train_score - valid_score
        logger.info(f"\n过拟合程度: {overfitting:.4f}")
        if overfitting > 0.15:
            logger.warning(f"⚠️  过拟合严重 ({overfitting:.2%})")
        elif overfitting > 0.10:
            logger.warning(f"⚠️  存在过拟合 ({overfitting:.2%})")
        else:
            logger.info(f"✓ 过拟合控制良好 ({overfitting:.2%})")

        # 特征重要性
        if hasattr(self.model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': selected_features,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)

            logger.info(f"\n前10个重要特征:")
            for idx, row in importances.head(10).iterrows():
                logger.info(f"  {row['feature']}: {row['importance']:.4f}")

            self.feature_importance_ = importances

        logger.info(f"\n{'='*60}")
        logger.info(f"模型训练完成！")
        logger.info(f"{'='*60}")

    def predict_proba(self, X, use_selected_features=True) -> pd.DataFrame:
        """
        预测概率

        Args:
            X: 特征数据
            use_selected_features: 是否使用特征选择后的特征

        Returns:
            包含三类概率的DataFrame
        """
        if use_selected_features and self.selected_features is not None:
            X = X[self.selected_features]

        proba = self.model.predict_proba(X)  # shape: (n_samples, 3)

        # 转换为DataFrame
        # sklearn输出顺序: 0(下跌), 1(震荡), 2(上涨)
        df_proba = pd.DataFrame(proba, columns=['prob_down', 'prob_range', 'prob_up'])

        return df_proba

    def evaluate(self, X, y, dataset_name: str = 'Test', use_selected_features=True):
        """
        评估模型

        Args:
            X: 特征数据
            y: 真实标签
            dataset_name: 数据集名称
            use_selected_features: 是否使用特征选择后的特征
        """
        logger.info(f"\n{'='*60}")
        logger.info(f"{dataset_name} 评估")
        logger.info(f"{'='*60}")

        if use_selected_features and self.selected_features is not None:
            X = X[self.selected_features]

        # 预测
        y_pred = self.model.predict(X)
        y_proba = self.predict_proba(X, use_selected_features=False)

        # 准确率
        accuracy = accuracy_score(y, y_pred)
        logger.info(f"准确率: {accuracy:.4f}")

        # 分类报告
        target_names = ['下跌(-1)', '震荡(0)', '上涨(1)']
        report = classification_report(y, y_pred, target_names=target_names)
        logger.info(f"\n分类报告:\n{report}")

        # 混淆矩阵
        cm = confusion_matrix(y, y_pred)
        logger.info(f"混淆矩阵:")
        logger.info(f"  {cm}")

        # 概率统计
        logger.info(f"\n概率统计:")
        logger.info(f"  上涨概率均值: {y_proba['prob_up'].mean():.4f}")
        logger.info(f"  震荡概率均值: {y_proba['prob_range'].mean():.4f}")
        logger.info(f"  下跌概率均值: {y_proba['prob_down'].mean():.4f}")

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report,
            'probabilities': y_proba
        }

    def save_predictions(
        self,
        df: pd.DataFrame,
        predictions: pd.DataFrame,
        output_path: Path = None
    ):
        """
        保存预测结果

        Args:
            df: 原始数据
            predictions: 预测概率
            output_path: 输出路径
        """
        if output_path is None:
            output_path = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data/predictions/trend_predictions_optimized.csv')

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # 合并数据
        result = df[['datetime', 'trend_label']].copy()
        result['pred_label'] = predictions[['prob_up', 'prob_range', 'prob_down']].idxmax(axis=1)
        result['pred_label'] = result['pred_label'].map({'prob_up': 1, 'prob_range': 0, 'prob_down': -1})
        result = pd.concat([result, predictions], axis=1)

        result.to_csv(output_path, index=False)
        logger.info(f"✓ 预测结果已保存: {output_path}")
        logger.info(f"{'='*60}")

    def save_model(self, model_path: Path = None):
        """
        保存模型

        Args:
            model_path: 模型保存路径
        """
        if model_path is None:
            model_path = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/models/trend_model.pkl')

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        # 保存模型和元数据
        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'label_names': self.label_names,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'min_samples_split': self.min_samples_split,
                'random_state': self.random_state
            },
            'trained_at': datetime.now().isoformat()
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"\n✓ 模型已保存: {model_path}")

    def load_model(self, model_path: Path = None):
        """
        加载模型

        Args:
            model_path: 模型文件路径
        """
        if model_path is None:
            model_path = Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/models/trend_model.pkl')

        with open(model_path, 'rb') as f:
            model_data = pickle.load(f)

        self.model = model_data['model']
        self.model_type = model_data['model_type']
        self.feature_names = model_data['feature_names']
        self.label_names = model_data['label_names']

        logger.info(f"✓ 模型已加载: {model_path}")
        logger.info(f"  类型: {self.model_type}")
        logger.info(f"  训练时间: {model_data['trained_at']}")


def train_and_evaluate():
    """训练和评估趋势模型的完整流程（优化版本）"""
    logger.info("="*60)
    logger.info("趋势模型训练流程（优化版本）")
    logger.info("="*60)

    # 创建模型（优化参数）
    model = TrendModel(
        model_type='random_forest',
        n_estimators=100,
        max_depth=5,              # 降低深度减少过拟合
        min_samples_split=50,       # 增加分割样本数
        min_samples_leaf=20,        # 增加叶子节点样本数
        random_state=42,
        top_k_features=20          # 特征选择
    )

    # 加载数据
    df = model.load_data()

    # 准备特征和标签
    X, y, df_clean = model.prepare_features(df)

    # 时间序列分割
    splits = model.time_series_split(df_clean)

    # 训练模型
    model.train(
        splits['X_train'], splits['y_train'],
        splits['X_valid'], splits['y_valid']
    )

    # 评估测试集
    test_results = model.evaluate(
        splits['X_test'], splits['y_test'],
        'Test (优化后)'
    )

    # 预测测试集概率
    test_proba = model.predict_proba(splits['X_test'])

    # 保存模型
    model.save_model(
        Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/models/trend_model_optimized.pkl')
    )

    # 保存预测结果
    # 获取测试集对应的datetime
    test_df = df_clean[df_clean['datetime'].dt.year == 2025].copy()
    test_df = test_df.iloc[:len(test_proba)]  # 对齐长度
    test_df['datetime'] = test_df['datetime'].values
    model.save_predictions(test_df, test_proba)

    logger.info(f"\n{'='*60}")
    logger.info(f"训练流程完成！")
    logger.info(f"{'='*60}")

    return model, test_results


def train_xgboost_model():
    """训练和评估XGBoost趋势模型的完整流程（新特征）"""
    logger.info("="*60)
    logger.info("XGBoost趋势模型训练流程（新特征版本）")
    logger.info("="*60)

    # 创建XGBoost模型
    model = TrendModel(
        model_type='xgboost',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        min_samples_split=50,
        min_samples_leaf=20,
        random_state=42,
        top_k_features=30  # 使用更多特征
    )

    # 加载数据
    df = model.load_data()

    # 准备特征和标签
    X, y, df_clean = model.prepare_features(df)

    # 时间序列分割
    splits = model.time_series_split(df_clean)

    # 训练模型
    model.train(
        splits['X_train'], splits['y_train'],
        splits['X_valid'], splits['y_valid']
    )

    # 评估测试集
    test_results = model.evaluate(
        splits['X_test'], splits['y_test'],
        'Test (XGBoost + 新特征)'
    )

    # 预测测试集概率
    test_proba = model.predict_proba(splits['X_test'])

    # 保存模型
    model.save_model(
        Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/models/trend_model_xgboost.pkl')
    )

    # 保存预测结果
    test_df = df_clean[df_clean['datetime'].dt.year == 2025].copy()
    test_df = test_df.iloc[:len(test_proba)]
    test_df['datetime'] = test_df['datetime'].values
    model.save_predictions(
        test_df,
        test_proba,
        Path('/Users/mystryl/Documents/Quant/projects/Multi-Frame_Entry/data/predictions/trend_predictions_xgboost.csv')
    )

    logger.info(f"\n{'='*60}")
    logger.info(f"XGBoost训练流程完成！")
    logger.info(f"{'='*60}")

    return model, test_results


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    # 训练XGBoost模型（新特征）
    model, results = train_xgboost_model()
