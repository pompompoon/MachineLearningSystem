import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.model_selection import GridSearchCV
from imblearn.under_sampling import RandomUnderSampler
from imblearn.ensemble import RUSBoostClassifier

class RUSBoostModel(BaseEstimator, ClassifierMixin):
    """
    RUSBoostModelクラス
    不均衡データに対応するためのRUSBoost(Random Under-Sampling Boost)アルゴリズムを実装
    
    Parameters:
    -----------
    n_estimators : int, default=50
        ブースティングで使用する弱学習器の数
    
    learning_rate : float, default=1.0
        各弱学習器の貢献度を抑制する学習率
        
    algorithm : {'SAMME', 'SAMME.R'}, default='SAMME.R'
        ブースティングアルゴリズム
        
    sampling_strategy : float or str, default='auto'
        少数クラスに対する多数クラスのサンプリング比率
        
    random_state : int, default=None
        乱数シード
    """
    
    def __init__(self, n_estimators=50, learning_rate=1.0, algorithm='SAMME.R', 
                 sampling_strategy='auto', random_state=None):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.algorithm = algorithm
        self.sampling_strategy = sampling_strategy
        self.random_state = random_state
        self.model = None
        self.feature_importances_ = None
    
    def fit(self, X, y):
        """
        モデルの学習
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            学習データの特徴量
        
        y : array-like, shape (n_samples,)
            学習データのラベル
            
        Returns:
        --------
        self : object
            学習済みモデル
        """
        # RUSBoostClassifierの作成
        self.model = RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state
        )
        
        # モデルの学習
        self.model.fit(X, y)
        
        # 特徴量重要度を取得（あれば）
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        else:
            # 各弱学習器の特徴量重要度の平均を計算
            try:
                importances = np.zeros(X.shape[1])
                for estimator in self.model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances += estimator.feature_importances_
                self.feature_importances_ = importances / len(self.model.estimators_)
            except:
                self.feature_importances_ = np.ones(X.shape[1]) / X.shape[1]
        
        return self
    
    def predict(self, X):
        """
        予測の実行
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            予測する特徴量
            
        Returns:
        --------
        y_pred : array, shape (n_samples,)
            予測ラベル
        """
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        確率予測の実行
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            予測する特徴量
            
        Returns:
        --------
        y_proba : array, shape (n_samples, n_classes)
            クラスごとの確率値
        """
        return self.model.predict_proba(X)
    
    def get_model(self):
        """
        内部モデルの取得
        
        Returns:
        --------
        model : object
            学習済みの内部モデル
        """
        if self.model is None:
            self.model = RUSBoostClassifier(
                estimator=DecisionTreeClassifier(max_depth=3),
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                algorithm=self.algorithm,
                sampling_strategy=self.sampling_strategy,
                random_state=self.random_state
            )
        return self.model
    
    def perform_grid_search(self, X_train, y_train, X_val=None, y_val=None, cv=5):
        """
        グリッドサーチによるハイパーパラメータ最適化
        
        Parameters:
        -----------
        X_train : array-like, shape (n_samples, n_features)
            学習データの特徴量
        
        y_train : array-like, shape (n_samples,)
            学習データのラベル
            
        X_val : array-like, shape (n_samples, n_features), optional
            検証データの特徴量（指定しない場合はクロスバリデーション）
            
        y_val : array-like, shape (n_samples,), optional
            検証データのラベル（指定しない場合はクロスバリデーション）
            
        cv : int, default=5
            クロスバリデーションの分割数
            
        Returns:
        --------
        best_params : dict
            最適なハイパーパラメータ
            
        best_model : object
            最適なハイパーパラメータで学習したモデル
        """
        # グリッドサーチのパラメータ範囲
        param_grid = {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.1, 0.5, 1.0],
            'sampling_strategy': ['auto', 0.5, 0.75]
        }
        
        # 基本モデルの作成
        base_model = RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            random_state=self.random_state
        )
        
        # グリッドサーチの実行
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=cv,
            scoring='f1_weighted',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        # 最適なパラメータでモデルを再構築
        self.n_estimators = grid_search.best_params_['n_estimators']
        self.learning_rate = grid_search.best_params_['learning_rate']
        self.sampling_strategy = grid_search.best_params_['sampling_strategy']
        
        # 最適なパラメータでモデルを作成して学習
        self.model = RUSBoostClassifier(
            estimator=DecisionTreeClassifier(max_depth=3),
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
            algorithm=self.algorithm,
            sampling_strategy=self.sampling_strategy,
            random_state=self.random_state
        )
        
        self.model.fit(X_train, y_train)
        
        # 特徴量重要度を更新
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importances_ = self.model.feature_importances_
        else:
            # 各弱学習器の特徴量重要度の平均を計算
            try:
                importances = np.zeros(X_train.shape[1])
                for estimator in self.model.estimators_:
                    if hasattr(estimator, 'feature_importances_'):
                        importances += estimator.feature_importances_
                self.feature_importances_ = importances / len(self.model.estimators_)
            except:
                self.feature_importances_ = np.ones(X_train.shape[1]) / X_train.shape[1]
        
        return grid_search.best_params_, self
    
    @classmethod
    def run_cv(cls, X, y, n_splits=5, random_state=42):
        """
        クロスバリデーションを実行するクラスメソッド
        
        Parameters:
        -----------
        X : array-like, shape (n_samples, n_features)
            特徴量データ
        
        y : array-like, shape (n_samples,)
            ラベルデータ
            
        n_splits : int, default=5
            クロスバリデーションの分割数
            
        random_state : int, default=42
            乱数シード
            
        Returns:
        --------
        oof_preds : array, shape (n_samples,)
            Out-of-foldの予測値
            
        scores : dict
            評価指標のスコア
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        # Out-of-foldの予測値を格納する配列
        oof_preds = np.zeros(len(y))
        oof_probs = np.zeros(len(y))
        
        # 評価指標のスコアを格納する辞書
        scores = {
            'accuracy': [],
            'precision': [],
            'recall': [],
            'specificity': [],
            'f1': [],
            'auc': []
        }
        
        # 交差検証の分割
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # 各foldでの学習と評価
        for fold, (train_idx, val_idx) in enumerate(kf.split(X, y)):
            print(f"\nFold {fold+1}/{n_splits}")
            
            # データの分割
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            # モデルの初期化と学習
            model = cls(random_state=random_state)
            model.fit(X_train, y_train)
            
            # Out-of-foldの予測
            oof_preds[val_idx] = model.predict(X_val)
            
            # 確率予測（AUC計算用）
            if hasattr(model, 'predict_proba'):
                oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
            
            # 評価指標の計算
            scores['accuracy'].append(accuracy_score(y_val, oof_preds[val_idx]))
            scores['precision'].append(precision_score(y_val, oof_preds[val_idx], average='weighted'))
            scores['recall'].append(recall_score(y_val, oof_preds[val_idx], average='weighted'))
            
            # 特異度の計算
            from sklearn.metrics import confusion_matrix
            cm = confusion_matrix(y_val, oof_preds[val_idx])
            if cm.shape == (2, 2):
                tn, fp = cm[0, 0], cm[0, 1]
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                # 多クラス分類の場合
                specificities = []
                n_classes = cm.shape[0]
                for i in range(n_classes):
                    tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
                    fp = np.sum(np.delete(cm, i, axis=0)[:, i])
                    specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
                specificity = np.mean(specificities)
            
            scores['specificity'].append(specificity)
            scores['f1'].append(f1_score(y_val, oof_preds[val_idx], average='weighted'))
            
            # AUCの計算（二値分類の場合のみ）
            if len(np.unique(y)) == 2 and hasattr(model, 'predict_proba'):
                try:
                    scores['auc'].append(roc_auc_score(y_val, oof_probs[val_idx]))
                except Exception as e:
                    print(f"AUC計算エラー: {e}")
                    scores['auc'].append(0.5)
            
            # このfoldの結果を表示
            print(f"Fold {fold+1} Accuracy: {scores['accuracy'][-1]:.4f}")
            print(f"Fold {fold+1} Precision: {scores['precision'][-1]:.4f}")
            print(f"Fold {fold+1} Recall: {scores['recall'][-1]:.4f}")
            print(f"Fold {fold+1} Specificity: {scores['specificity'][-1]:.4f}")
            print(f"Fold {fold+1} F1 Score: {scores['f1'][-1]:.4f}")
            if 'auc' in scores and len(scores['auc']) > fold:
                print(f"Fold {fold+1} AUC: {scores['auc'][-1]:.4f}")
        
        # 評価指標の平均値を計算
        for metric in scores:
            if scores[metric]:  # リストが空でない場合
                scores[metric] = np.mean(scores[metric])
            else:
                scores[metric] = 0
        
        return oof_preds, scores