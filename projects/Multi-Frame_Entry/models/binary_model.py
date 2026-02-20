"""
二分类趋势模型训练

数据划分策略：
- 训练集：2022-2025，每年随机抽取4个月
- 验证集：再随机抽取4个月
- 测试集：剩下的4个月
"""
import pandas as pd
import numpy as np
from pathlib import Path
import pickle
import logging
from typing import Dict, List, Tuple
from datetime import datetime
import random

# 模型库
try:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score
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


class BinaryTrendModel:
    """二分类趋势模型"""

    def __init__(
        self,
        model_type: str = 'xgboost',
        n_estimators: int = 200,
        max_depth: int = 5,
        learning_rate: float = 0.05,
        random_state: int = 42,
        top_k_features: int = 30
    ):
        """
        初始化模型

        Args:
            model_type: 'random_forest' 或 'xgboost'
            n_estimators: 树的数量
            max_depth: 最大深度
            learning_rate: 学习率（XGBoost）
            random_state: 随机种子
            top_k_features: 特征选择数量
        """
        self.model_type = model_type
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.random_state = random_state
        self.top_k_features = top_k_features
        self.model = None
        self.feature_names = None
        self.selected_features = None

        logger.info(f"二分类趋势模型初始化:")
        logger.info(f"  模型类型: {model_type}")
        logger.info(f"  参数: n_estimators={n_estimators}, max_depth={max_depth}")
        if model_type == 'xgboost':
            logger.info(f"  学习率: {learning_rate}")

    def random_monthly_split(
        self,
        df: pd.DataFrame,
        train_months_per_year: int = 4,
        valid_months_per_year: int = 4,
        random_seed: int = 42
    ) -> Dict:
        """
        按月随机划分数据集

        Args:
            df: 包含datetime列的数据
            train_months_per_year: 每年训练集月数
            valid_months_per_year: 每年验证集月数
            random_seed: 随机种子

        Returns:
            数据划分字典
        """
        logger.info("="*60)
        logger.info("按月随机划分数据集")
        logger.info("="*60)
        logger.info(f"每年训练月数: {train_months_per_year}")
        logger.info(f"每年验证月数: {valid_months_per_year}")
        logger.info(f"每年测试月数: {12 - train_months_per_year - valid_months_per_year}")

        # 提取年月信息
        df = df.copy()
        df['year'] = df['datetime'].dt.year
        df['month'] = df['datetime'].dt.month

        # 设置随机种子
        random.seed(random_seed)
        np.random.seed(random_seed)

        train_indices = []
        valid_indices = []
        test_indices = []

        # 每年单独随机分配
        for year in sorted(df['year'].unique()):
            year_data = df[df['year'] == year]
            months_in_year = sorted(year_data['month'].unique())

            logger.info(f"\n{year}年数据: {len(year_data)} 样本, {len(months_in_year)} 个月")

            # 随机打乱月份
            shuffled_months = months_in_year.copy()
            random.shuffle(shuffled_months)

            # 分配月份
            train_months = shuffled_months[:train_months_per_year]
            valid_months = shuffled_months[train_months_per_year:train_months_per_year + valid_months_per_year]
            test_months = shuffled_months[train_months_per_year + valid_months_per_year:]

            logger.info(f"  训练月份: {sorted(train_months)}")
            logger.info(f"  验证月份: {sorted(valid_months)}")
            logger.info(f"  测试月份: {sorted(test_months)}")

            # 收集索引
            for month in train_months:
                train_indices.extend(year_data[year_data['month'] == month].index.tolist())
            for month in valid_months:
                valid_indices.extend(year_data[year_data['month'] == month].index.tolist())
            for month in test_months:
                test_indices.extend(year_data[year_data['month'] == month].index.tolist())

        # 创建数据集
        X_train = df.loc[train_indices].filter(regex='^(?!datetime|year|month|open|high|low|close|volume|trend_label).*$')
        y_train = df.loc[train_indices, 'trend_label']

        X_valid = df.loc[valid_indices].filter(regex='^(?!datetime|year|month|open|high|low|close|volume|trend_label).*$')
        y_valid = df.loc[valid_indices, 'trend_label']

        X_test = df.loc[test_indices].filter(regex='^(?!datetime|year|month|open|high|low|close|volume|trend_label).*$')
        y_test = df.loc[test_indices, 'trend_label']

        # 删除标签缺失的样本
        valid_train = y_train.notna()
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        valid_valid = y_valid.notna()
        X_valid = X_valid[valid_valid]
        y_valid = y_valid[valid_valid]

        valid_test = y_test.notna()
        X_test = X_test[valid_test]
        y_test = y_test[valid_test]

        # 删除特征缺失的样本
        X_train = X_train.dropna()
        y_train = y_train[X_train.index]

        X_valid = X_valid.dropna()
        y_valid = y_valid[X_valid.index]

        X_test = X_test.dropna()
        y_test = y_test[X_test.index]

        # 确保标签为整数类型
        y_train = y_train.astype(int)
        y_valid = y_valid.astype(int)
        y_test = y_test.astype(int)

        self.feature_names = X_train.columns.tolist()

        logger.info(f"\n划分结果:")
        logger.info(f"  训练集: {len(X_train)} 样本 ({len(X_train)/len(df)*100:.1f}%)")
        logger.info(f"  验证集: {len(X_valid)} 样本 ({len(X_valid)/len(df)*100:.1f}%)")
        logger.info(f"  测试集: {len(X_test)} 样本 ({len(X_test)/len(df)*100:.1f}%)")

        # 标签分布
        logger.info(f"\n标签分布:")
        logger.info(f"  训练集 - 震荡: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.1f}%), 有趋势: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.1f}%)")
        logger.info(f"  验证集 - 震荡: {(y_valid==0).sum()} ({(y_valid==0).sum()/len(y_valid)*100:.1f}%), 有趋势: {(y_valid==1).sum()} ({(y_valid==1).sum()/len(y_valid)*100:.1f}%)")
        logger.info(f"  测试集 - 震荡: {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.1f}%), 有趋势: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.1f}%)")

        return {
            'X_train': X_train, 'y_train': y_train,
            'X_valid': X_valid, 'y_valid': y_valid,
            'X_test': X_test, 'y_test': y_test
        }

    def select_features(self, X_train, y_train):
        """特征选择"""
        logger.info(f"\n执行特征选择（保留 Top {self.top_k_features}）...")

        if self.model_type == 'random_forest':
            temp_model = RandomForestClassifier(
                n_estimators=50,
                max_depth=5,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            temp_model = XGBClassifier(
                n_estimators=50,
                max_depth=5,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss'
            )

        temp_model.fit(X_train, y_train)

        importances = pd.DataFrame({
            'feature': self.feature_names,
            'importance': temp_model.feature_importances_
        }).sort_values('importance', ascending=False)

        selected = importances.head(self.top_k_features)
        self.selected_features = selected['feature'].tolist()

        logger.info(f"  Top 10 特征:")
        for idx, row in selected.head(10).iterrows():
            logger.info(f"    {row['feature']}: {row['importance']:.4f}")

        return self.selected_features

    def build_model(self):
        """构建模型"""
        if self.model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state,
                n_jobs=-1,
                class_weight='balanced'
            )
        elif self.model_type == 'xgboost':
            self.model = XGBClassifier(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                learning_rate=self.learning_rate,
                random_state=self.random_state,
                n_jobs=-1,
                eval_metric='logloss',
                scale_pos_weight=1  # 将根据类别比例调整
            )
        else:
            raise ValueError(f"不支持的模型类型: {self.model_type}")

    def train(self, X_train, y_train, X_valid, y_valid):
        """训练模型"""
        logger.info("\n" + "="*60)
        logger.info("开始训练模型")
        logger.info("="*60)

        # 特征选择
        selected_features = self.select_features(X_train, y_train)
        X_train_selected = X_train[selected_features]
        X_valid_selected = X_valid[selected_features]

        # 计算类别权重（XGBoost）
        if self.model_type == 'xgboost':
            neg_count = (y_train == 0).sum()
            pos_count = (y_train == 1).sum()
            scale_pos_weight = neg_count / pos_count
            logger.info(f"  XGBoost scale_pos_weight: {scale_pos_weight:.2f}")

            # 先构建模型（不含scale_pos_weight）
            self.build_model()

            # 然后设置参数
            self.model.set_params(scale_pos_weight=scale_pos_weight)
        else:
            self.build_model()

        # 训练模型
        self.model.fit(X_train_selected, y_train)

        # 评估
        train_score = self.model.score(X_train_selected, y_train)
        valid_score = self.model.score(X_valid_selected, y_valid)

        logger.info(f"\n✓ 训练集准确率: {train_score:.4f}")
        logger.info(f"✓ 验证集准确率: {valid_score:.4f}")

        overfitting = train_score - valid_score
        logger.info(f"\n过拟合程度: {overfitting:.4f}")
        if overfitting > 0.15:
            logger.warning(f"⚠️  过拟合严重 ({overfitting:.2%})")
        else:
            logger.info(f"✓ 过拟合控制良好 ({overfitting:.2%})")

        logger.info(f"\n{'='*60}")
        logger.info(f"模型训练完成！")
        logger.info(f"{'='*60}")

    def evaluate(self, X, y, dataset_name: str = 'Test'):
        """评估模型"""
        logger.info(f"\n{'='*60}")
        logger.info(f"{dataset_name} 评估")
        logger.info(f"{'='*60}")

        X_selected = X[self.selected_features]

        # 预测
        y_pred = self.model.predict(X_selected)
        y_proba = self.model.predict_proba(X_selected)[:, 1]  # 正类概率

        # 准确率
        accuracy = accuracy_score(y, y_pred)
        logger.info(f"准确率: {accuracy:.4f}")

        # AUC-ROC
        try:
            auc = roc_auc_score(y, y_proba)
            logger.info(f"AUC-ROC: {auc:.4f}")
        except:
            logger.info(f"AUC-ROC: 无法计算")

        # 分类报告
        target_names = ['震荡(0)', '有趋势(1)']
        report = classification_report(y, y_pred, target_names=target_names)
        logger.info(f"\n分类报告:\n{report}")

        # 混淆矩阵
        cm = confusion_matrix(y, y_pred)
        logger.info(f"混淆矩阵:")
        logger.info(f"  TN={cm[0,0]}, FP={cm[0,1]}")
        logger.info(f"  FN={cm[1,0]}, TP={cm[1,1]}")

        # 召回率
        recall_0 = cm[0,0] / (cm[0,0] + cm[0,1]) if (cm[0,0] + cm[0,1]) > 0 else 0
        recall_1 = cm[1,1] / (cm[1,1] + cm[1,0]) if (cm[1,1] + cm[1,0]) > 0 else 0
        logger.info(f"\n召回率:")
        logger.info(f"  震荡: {recall_0:.4f}")
        logger.info(f"  有趋势: {recall_1:.4f}")

        return {
            'accuracy': accuracy,
            'confusion_matrix': cm,
            'classification_report': report
        }

    def save_model(self, model_path: Path = None):
        """保存模型"""
        if model_path is None:
            model_path = Path('models/binary_model.pkl')

        model_path = Path(model_path)
        model_path.parent.mkdir(parents=True, exist_ok=True)

        model_data = {
            'model': self.model,
            'model_type': self.model_type,
            'feature_names': self.feature_names,
            'selected_features': self.selected_features,
            'params': {
                'n_estimators': self.n_estimators,
                'max_depth': self.max_depth,
                'learning_rate': self.learning_rate
            },
            'trained_at': datetime.now().isoformat()
        }

        with open(model_path, 'wb') as f:
            pickle.dump(model_data, f)

        logger.info(f"\n✓ 模型已保存: {model_path}")


def train_binary_model():
    """训练二分类模型"""
    logger.info("="*60)
    logger.info("二分类趋势模型训练")
    logger.info("="*60)

    # 加载数据
    logger.info("\n加载特征数据...")
    df = pd.read_csv('data/features/binary_features.csv')
    df['datetime'] = pd.to_datetime(df['datetime'])
    logger.info(f"  数据形状: {df.shape}")

    # 创建模型
    model = BinaryTrendModel(
        model_type='xgboost',
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        top_k_features=30
    )

    # 数据划分
    splits = model.random_monthly_split(df, random_seed=42)

    # 训练
    model.train(
        splits['X_train'], splits['y_train'],
        splits['X_valid'], splits['y_valid']
    )

    # 评估
    test_results = model.evaluate(splits['X_test'], splits['y_test'], '测试集')

    # 保存模型
    model.save_model(Path('models/binary_model_xgboost.pkl'))

    logger.info(f"\n{'='*60}")
    logger.info(f"训练完成！")
    logger.info(f"{'='*60}")

    return model, test_results


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s'
    )

    model, results = train_binary_model()
