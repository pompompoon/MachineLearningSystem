import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool

class UndersamplingModel(BaseEstimator, ClassifierMixin):
    """
    不均衡データセット用のアンダーサンプリングモデル
    
    多数派クラスからランダムにサンプリングして均衡データセットを作成し、
    ひとつのモデルを学習するシンプルなアプローチを実装しています。
    """
    
    def __init__(self, base_model='lightgbm', random_state=42, base_params=None, categorical_features=None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        base_model : str, default='lightgbm'
            使用するベースモデル。'lightgbm', 'xgboost', 'random_forest', 'catboost'から選択
        random_state : int, default=42
            再現性のための乱数シード
        base_params : dict, default=None
            ベースモデル用のパラメータ
        categorical_features : list, default=None
            カテゴリカル変数のインデックスリスト（CatBoost用）
        """
        self.base_model = base_model
        self.random_state = random_state
        self.base_params = base_params or {}
        self.categorical_features = categorical_features
        self.model = None
        self.feature_importances_ = None
        self.feature_names = None

    def _get_lightgbm_model(self, X_train, y_train, X_eval=None, y_eval=None):
        """LightGBMモデルを作成して学習するメソッド"""
        # デフォルトパラメータ
        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbose': -1,
            'random_state': self.random_state
        }
        
        # ユーザー指定パラメータで更新
        params.update(self.base_params)
        
        # データセット作成
        lgb_train = lgb.Dataset(X_train, y_train)
        lgb_eval = None
        if X_eval is not None and y_eval is not None:
            lgb_eval = lgb.Dataset(X_eval, y_eval, reference=lgb_train)
        
        # モデル学習
        model = lgb.train(
            params, 
            lgb_train, 
            valid_sets=lgb_eval if lgb_eval else None,
            num_boost_round=10000,
            callbacks=[lgb.early_stopping(stopping_rounds=50)] if lgb_eval else None
        )
        
        return model
    
    def _get_xgboost_model(self, X_train, y_train, X_eval=None, y_eval=None):
        """XGBoostモデルを作成して学習するメソッド"""
        # デフォルトパラメータ
        params = {
            'objective': 'binary:logistic',
            'eval_metric': 'logloss',
            'verbosity': 0,
            'seed': self.random_state
        }
        
        # ユーザー指定パラメータで更新
        params.update(self.base_params)
        
        # データセット作成
        dtrain = xgb.DMatrix(X_train, label=y_train)
        deval = None
        if X_eval is not None and y_eval is not None:
            deval = xgb.DMatrix(X_eval, label=y_eval)
        
        # モデル学習
        watchlist = [(dtrain, 'train')]
        if deval:
            watchlist.append((deval, 'eval'))
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=10000,
            evals=watchlist,
            early_stopping_rounds=50 if deval else None,
            verbose_eval=False
        )
        
        return model
    
    def _get_random_forest_model(self, X_train, y_train, X_eval=None, y_eval=None):
        """RandomForestモデルを作成して学習するメソッド"""
        # デフォルトパラメータ
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': self.random_state
        }
        
        # ユーザー指定パラメータで更新
        params.update(self.base_params)
        
        # モデル作成と学習
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def _get_catboost_model(self, X_train, y_train, X_eval=None, y_eval=None):
        """CatBoostモデルを作成して学習するメソッド"""
        # デフォルトパラメータ
        params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'Logloss',
            'random_seed': self.random_state,
            'verbose': False
        }
        
        # ユーザー指定パラメータで更新
        params.update(self.base_params)
        
        # モデル作成
        model = CatBoostClassifier(**params)
        
        # カテゴリカル変数がある場合はPoolを使用
        if self.categorical_features is not None:
            train_pool = Pool(X_train, y_train, cat_features=self.categorical_features)
            eval_pool = None
            if X_eval is not None and y_eval is not None:
                eval_pool = Pool(X_eval, y_eval, cat_features=self.categorical_features)
            
            # モデル学習
            model.fit(
                train_pool,
                eval_set=eval_pool,
                early_stopping_rounds=20,
                verbose=False
            )
        else:
            # カテゴリカル変数がない場合は通常の学習
            eval_set = None
            if X_eval is not None and y_eval is not None:
                eval_set = (X_eval, y_eval)
            
            # モデル学習
            model.fit(
                X_train, y_train,
                eval_set=eval_set,
                early_stopping_rounds=20,
                verbose=False
            )
        
        return model
    
    def fit(self, X, y, X_eval=None, y_eval=None):
        """
        アンダーサンプリングを適用してモデルを学習するメソッド
        
        Parameters:
        -----------
        X : array-like or pandas DataFrame
            学習用特徴量
        y : array-like
            学習用目的変数
        X_eval : array-like or pandas DataFrame, optional
            評価用特徴量（早期停止用）
        y_eval : array-like, optional
            評価用目的変数（早期停止用）
        
        Returns:
        --------
        self : object
            自身を返す
        """
        # 必要に応じてnumpy配列に変換
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            self.feature_names = X.columns.tolist()
        else:
            X_values = X
            self.feature_names = None
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        if X_eval is not None:
            if isinstance(X_eval, pd.DataFrame):
                X_eval_values = X_eval.values
            else:
                X_eval_values = X_eval
        else:
            X_eval_values = None
            
        if y_eval is not None:
            if isinstance(y_eval, pd.Series):
                y_eval_values = y_eval.values
            else:
                y_eval_values = y_eval
        else:
            y_eval_values = None
        
        # ランダムアンダーサンプリングの実行
        print("アンダーサンプリングを適用...")
        rus = RandomUnderSampler(
            sampling_strategy='auto',
            random_state=self.random_state
        )
        X_resampled, y_resampled = rus.fit_resample(X_values, y_values)
        
        # サンプリング後のクラス分布を表示
        print("サンプリング後のクラス分布:")
        unique, counts = np.unique(y_resampled, return_counts=True)
        for u, c in zip(unique, counts):
            print(f"クラス {u}: {c} サンプル")
        
        # 指定されたベースモデルに基づいてモデルを学習
        print(f"{self.base_model}モデルを学習...")
        if self.base_model == 'lightgbm':
            self.model = self._get_lightgbm_model(X_resampled, y_resampled, X_eval_values, y_eval_values)
        elif self.base_model == 'xgboost':
            self.model = self._get_xgboost_model(X_resampled, y_resampled, X_eval_values, y_eval_values)
        elif self.base_model == 'random_forest':
            self.model = self._get_random_forest_model(X_resampled, y_resampled, X_eval_values, y_eval_values)
        elif self.base_model == 'catboost':
            self.model = self._get_catboost_model(X_resampled, y_resampled, X_eval_values, y_eval_values)
        else:
            raise ValueError(f"サポートされていないベースモデル: {self.base_model}")
        
        # 特徴量重要度の計算
        self._calculate_feature_importance()
        
        return self
    
    def _calculate_feature_importance(self):
        """特徴量重要度を計算するメソッド"""
        if self.model is None:
            return None
        
        if self.base_model == 'lightgbm':
            self.feature_importances_ = self.model.feature_importance(importance_type='gain')
        elif self.base_model == 'xgboost':
            importances = self.model.get_score(importance_type='gain')
            # XGBoostは特徴量名をキーとした辞書を返す
            if self.feature_names:
                self.feature_importances_ = np.zeros(len(self.feature_names))
                for i, name in enumerate(self.feature_names):
                    f_name = f'f{i}'
                    if f_name in importances:
                        self.feature_importances_[i] = importances[f_name]
            else:
                # 特徴量名がない場合は順序づけられた配列に変換
                n_features = max([int(k[1:]) for k in importances.keys()]) + 1
                self.feature_importances_ = np.zeros(n_features)
                for f, imp in importances.items():
                    feature_idx = int(f[1:])
                    self.feature_importances_[feature_idx] = imp
        elif self.base_model == 'random_forest':
            self.feature_importances_ = self.model.feature_importances_
        elif self.base_model == 'catboost':
            self.feature_importances_ = self.model.get_feature_importance()
        
        return self.feature_importances_
    
    def predict_proba(self, X):
        """
        クラス確率を予測するメソッド
        
        Parameters:
        -----------
        X : array-like or pandas DataFrame
            特徴量
        
        Returns:
        --------
        probas : array-like
            予測クラス確率
        """
        if self.model is None:
            raise ValueError("モデルがまだ学習されていません。")
        
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        # モデルに基づいて予測
        if self.base_model == 'lightgbm':
            # LightGBMは正例クラスの確率のみ返す
            pos_proba = self.model.predict(X_values)
            probas = np.vstack((1 - pos_proba, pos_proba)).T
        elif self.base_model == 'xgboost':
            # XGBoostも正例クラスの確率のみ返す
            dtest = xgb.DMatrix(X_values)
            pos_proba = self.model.predict(dtest)
            probas = np.vstack((1 - pos_proba, pos_proba)).T
        elif self.base_model == 'random_forest':
            # RandomForestは両クラスの確率を返す
            probas = self.model.predict_proba(X_values)
        elif self.base_model == 'catboost':
            # CatBoostの予測処理
            if self.categorical_features is not None:
                test_pool = Pool(X_values, cat_features=self.categorical_features)
                probas = self.model.predict_proba(test_pool)
            else:
                probas = self.model.predict_proba(X_values)
        else:
            raise ValueError(f"サポートされていないベースモデル: {self.base_model}")
        
        return probas
    
    def predict(self, X, threshold=0.5):
        """
        クラスラベルを予測するメソッド
        
        Parameters:
        -----------
        X : array-like or pandas DataFrame
            特徴量
        threshold : float, default=0.5
            正例クラスの閾値
        
        Returns:
        --------
        labels : array-like
            予測クラスラベル
        """
        probas = self.predict_proba(X)
        return (probas[:, 1] > threshold).astype(int)
    
    def perform_grid_search(self, X_train, y_train, X_test, y_test):
        """
        簡易的なパラメータ選択を行うメソッド
        
        Parameters:
        -----------
        X_train : array-like or pandas DataFrame
            学習用特徴量
        y_train : array-like
            学習用目的変数
        X_test : array-like or pandas DataFrame
            テスト用特徴量
        y_test : array-like
            テスト用目的変数
        
        Returns:
        --------
        best_params : dict
            最適なパラメータ
        best_model : object
            最適なパラメータで学習したモデル
        """
        # アンダーサンプリングモデルではグリッドサーチを省略
        print("アンダーサンプリングモデルでは詳細なグリッドサーチは実行せず、基本パラメータで学習します。")
        self.fit(X_train, y_train)
        
        # 空の辞書を返す（互換性のため）
        return {}, self
    
    def get_model(self):
        """互換性のために自身を返すメソッド"""
        return self
    
    def plot_feature_importance(self, top_n=20, figsize=(12, 8)):
        """
        特徴量重要度をプロットするメソッド
        
        Parameters:
        -----------
        top_n : int, default=20
            表示する上位特徴量の数
        figsize : tuple, default=(12, 8)
            図のサイズ
        """
        import matplotlib.pyplot as plt
        import seaborn as sns
        
        if self.feature_importances_ is None:
            print("モデルがまだ学習されていないか、特徴量重要度が計算されていません。")
            return
        
        # 特徴量名がない場合はインデックスを使用
        if self.feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(self.feature_importances_))]
        else:
            feature_names = self.feature_names
        
        # 特徴量重要度のデータフレーム作成
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': self.feature_importances_
        })
        
        # 上位N個の特徴量を選択
        top_features = importance_df.sort_values('importance', ascending=False).head(top_n)
        
        # プロット
        plt.figure(figsize=figsize)
        sns.barplot(x='importance', y='feature', data=top_features)
        plt.title(f'{self.base_model.upper()} - 特徴量重要度（上位{top_n}）', fontsize=14)
        plt.xlabel('重要度', fontsize=12)
        plt.ylabel('特徴量', fontsize=12)
        plt.tight_layout()
        plt.show()
    
    @staticmethod
    def run_cv(X, y, base_model='lightgbm', n_splits=5, random_state=42, 
              base_params=None, categorical_features=None):
        """
        交差検証を実行する静的メソッド
        
        Parameters:
        -----------
        X : array-like or pandas DataFrame
            特徴量
        y : array-like
            目的変数
        base_model : str, default='lightgbm'
            ベースモデルの種類
        n_splits : int, default=5
            交差検証の分割数
        random_state : int, default=42
            乱数シード
        base_params : dict, default=None
            ベースモデル用のパラメータ
        categorical_features : list, default=None
            カテゴリカル特徴量のインデックス（CatBoost用）
            
        Returns:
        --------
        oof_preds : array-like
            Out-of-foldの予測値
        scores : dict
            評価指標のスコア
        """
        from sklearn.model_selection import StratifiedKFold
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
        
        # 必要に応じてnumpy配列に変換
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
            
        if isinstance(y, pd.Series):
            y_values = y.values
        else:
            y_values = y
            
        # 交差検証の設定
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Out-of-fold予測の初期化
        oof_preds = np.zeros(len(y_values))
        oof_probs = np.zeros(len(y_values))
        
        # 交差検証ループ
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_values, y_values)):
            print(f"\n===== Fold {fold+1}/{n_splits} =====")
            
            # データの分割
            X_train, X_val = X_values[train_idx], X_values[val_idx]
            y_train, y_val = y_values[train_idx], y_values[val_idx]
            
            # モデルの作成と学習
            model = UndersamplingModel(
                base_model=base_model,
                random_state=random_state + fold,
                base_params=base_params,
                categorical_features=categorical_features
            )
            model.fit(X_train, y_train)
            
            # バリデーションデータでの予測
            oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = model.predict(X_val)
        
        # 特異度（Specificity）の計算のための混同行列
        cm = confusion_matrix(y_values, oof_preds)
        # 2クラス分類の場合: TN=cm[0,0], FP=cm[0,1], FN=cm[1,0], TP=cm[1,1]
        if cm.shape == (2, 2):
            tn, fp = cm[0, 0], cm[0, 1]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        else:
            # 多クラス分類の場合は特異度の計算が複雑になるため、マクロ平均を使用
            specificities = []
            n_classes = cm.shape[0]
            for i in range(n_classes):
                tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
                fp = np.sum(np.delete(cm, i, axis=0)[:, i])
                specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
            specificity = np.mean(specificities)
        
        # 評価指標の計算
        scores = {
            'accuracy': accuracy_score(y_values, oof_preds),
            'precision': precision_score(y_values, oof_preds, average='weighted'),
            'recall': recall_score(y_values, oof_preds, average='weighted'),
            'specificity': specificity,  # 特異度を追加
            'f1': f1_score(y_values, oof_preds, average='weighted'),
            'roc_auc': roc_auc_score(y_values, oof_probs)
        }
        
        # 結果の表示
        print("\n===== 交差検証結果 =====")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
        
        return oof_preds, scores