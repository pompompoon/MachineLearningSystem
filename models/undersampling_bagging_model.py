# Save this file as: G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\models\undersampling_bagging_model.py

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.model_selection import StratifiedKFold
from imblearn.under_sampling import RandomUnderSampler
import lightgbm as lgb
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier, Pool
import matplotlib.pyplot as plt
import seaborn as sns

class UndersamplingBaggingModel(BaseEstimator, ClassifierMixin):
    """
    不均衡データセット用のアンダーサンプリング+バギングモデル
    
    多数派クラスからランダムにサンプリングして複数の均衡データセットを作成し、
    それぞれでモデルを学習して予測を平均化するアプローチを実装しています。
    """
    
    def __init__(self, base_model='lightgbm', n_bags=10, replacement=True, 
                 random_state=42, base_params=None, categorical_features=None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        base_model : str, default='lightgbm'
            使用するベースモデル。'lightgbm', 'xgboost', 'random_forest', 'catboost'から選択
        n_bags : int, default=10
            作成するバッグ（サブモデル）の数
        replacement : bool, default=True
            アンダーサンプリング時に置換を使用するかどうか
        random_state : int, default=42
            再現性のための乱数シード
        base_params : dict, default=None
            ベースモデル用のパラメータ
        categorical_features : list, default=None
            カテゴリカル変数のインデックスリスト（CatBoost用）
        """
        self.base_model = base_model
        self.n_bags = n_bags
        self.replacement = replacement
        self.random_state = random_state
        self.base_params = base_params or {}
        self.categorical_features = categorical_features
        self.models = []
        self.feature_importances_ = None

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
        
        # モデル学習 - early_stopping_rounds を early_stopping に変更
        model = lgb.train(
            params, 
            lgb_train, 
            valid_sets=lgb_eval if lgb_eval else None,
            num_boost_round=10000,
            # early_stopping_rounds=50 if lgb_eval else None,  # 古いバージョン
            callbacks=[lgb.early_stopping(stopping_rounds=50)] if lgb_eval else None,  # 新しいバージョン
            # verbose=False
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
    
    def _undersample_and_train(self, X_train, y_train, X_eval=None, y_eval=None):
        """訓練データをアンダーサンプリングしてモデルを学習するメソッド"""
        # ランダムアンダーサンプリングの実行
        rus = RandomUnderSampler(
            sampling_strategy='auto',
            replacement=self.replacement,
            random_state=np.random.randint(0, 10000)  # バッグごとに異なるシード
        )
        X_resampled, y_resampled = rus.fit_resample(X_train, y_train)
        
        # 指定されたベースモデルに基づいてモデルを学習
        if self.base_model == 'lightgbm':
            model = self._get_lightgbm_model(X_resampled, y_resampled, X_eval, y_eval)
        elif self.base_model == 'xgboost':
            model = self._get_xgboost_model(X_resampled, y_resampled, X_eval, y_eval)
        elif self.base_model == 'random_forest':
            model = self._get_random_forest_model(X_resampled, y_resampled, X_eval, y_eval)
        elif self.base_model == 'catboost':
            model = self._get_catboost_model(X_resampled, y_resampled, X_eval, y_eval)
        else:
            raise ValueError(f"サポートされていないベースモデル: {self.base_model}")
        
        return model
    
    def fit(self, X, y, X_eval=None, y_eval=None):
        """
        複数のアンダーサンプリングモデルを学習するメソッド
        
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
        
        # 複数のアンダーサンプリングモデルを学習
        self.models = []
        for i in range(self.n_bags):
            print(f"バッグ {i+1}/{self.n_bags} を学習中...")
            model = self._undersample_and_train(X_values, y_values, X_eval_values, y_eval_values)
            self.models.append(model)
        
        # 全モデルの平均から特徴量重要度を計算
        self._calculate_feature_importance(X_values)
        
        return self
    
    def _calculate_feature_importance(self, X):
        """全バッグの平均から特徴量重要度を計算するメソッド"""
        # 特徴量の数を取得
        n_features = X.shape[1]
        feature_importances = np.zeros(n_features)
        
        # 全モデルの特徴量重要度を合計
        for model in self.models:
            if self.base_model == 'lightgbm':
                importances = model.feature_importance(importance_type='gain')
            elif self.base_model == 'xgboost':
                importances = model.get_score(importance_type='gain')
                # XGBoostは特徴量名をキーとした辞書を返す
                for i, imp in importances.items():
                    feature_idx = int(i.replace('f', ''))
                    feature_importances[feature_idx] += imp
                continue  # XGBoostは別処理のため、平均化をスキップ
            elif self.base_model == 'random_forest':
                importances = model.feature_importances_
            elif self.base_model == 'catboost':
                importances = model.get_feature_importance()
            else:
                return None
            
            # 合計に追加
            feature_importances += importances
        
        # 特徴量重要度の平均を計算
        if self.base_model != 'xgboost':  # XGBoostは別処理
            feature_importances /= len(self.models)
        
        self.feature_importances_ = feature_importances
        return feature_importances
    
    def predict_proba(self, X):
        """
        全バッグの予測を平均化してクラス確率を予測するメソッド
        
        Parameters:
        -----------
        X : array-like or pandas DataFrame
            特徴量
        
        Returns:
        --------
        probas : array-like
            予測クラス確率
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        # 予測配列の初期化
        n_samples = X_values.shape[0]
        probas = np.zeros((n_samples, 2))
        
        # 全モデルの予測を合計
        for model in self.models:
            if self.base_model == 'lightgbm':
                # LightGBMは正例クラスの確率のみ返す
                pos_proba = model.predict(X_values)
                model_probas = np.vstack((1 - pos_proba, pos_proba)).T
            elif self.base_model == 'xgboost':
                # XGBoostも正例クラスの確率のみ返す
                dtest = xgb.DMatrix(X_values)
                pos_proba = model.predict(dtest)
                model_probas = np.vstack((1 - pos_proba, pos_proba)).T
            elif self.base_model == 'random_forest':
                # RandomForestは両クラスの確率を返す
                model_probas = model.predict_proba(X_values)
            elif self.base_model == 'catboost':
                # CatBoostの予測処理
                if self.categorical_features is not None:
                    test_pool = Pool(X_values, cat_features=self.categorical_features)
                    pos_proba = model.predict_proba(test_pool)[:, 1]
                else:
                    pos_proba = model.predict_proba(X_values)[:, 1]
                model_probas = np.vstack((1 - pos_proba, pos_proba)).T
            else:
                raise ValueError(f"サポートされていないベースモデル: {self.base_model}")
            
            probas += model_probas
        
        # 予測の平均化
        probas /= len(self.models)
        
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
        注: 本格的なグリッドサーチにはsklearnのGridSearchCVを使用してください
        
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
        # 簡易版では、バッグの数を変えてみる
        param_grid = {'n_bags': [5, 10, 15]}
        
        best_score = 0
        best_params = {}
        
        for n_bags in param_grid['n_bags']:
            print(f"\nバッグ数 {n_bags} でテスト中...")
            self.n_bags = n_bags
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)
            from sklearn.metrics import f1_score
            score = f1_score(y_test, y_pred)
            
            print(f"F1スコア: {score:.4f}")
            
            if score > best_score:
                best_score = score
                best_params = {'n_bags': n_bags}
        
        # 最適パラメータでモデルを再学習
        print(f"\n最適なバッグ数: {best_params['n_bags']}")
        self.n_bags = best_params['n_bags']
        self.fit(X_train, y_train)
        
        return best_params, self
    
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
    def run_cv(X, y, base_model='lightgbm', n_bags=10, n_splits=5, random_state=42, 
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
        n_bags : int, default=10
            バッグの数
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
        # 必要に応じてnumpy配列に変換
        if isinstance(X, pd.DataFrame):
            X_values = X.values
            feature_names = X.columns.tolist()
        else:
            X_values = X
            feature_names = None
            
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
            model = UndersamplingBaggingModel(
                base_model=base_model,
                n_bags=n_bags,
                random_state=random_state + fold,
                base_params=base_params,
                categorical_features=categorical_features
            )
            model.fit(X_train, y_train)
            
            # バリデーションデータでの予測
            oof_probs[val_idx] = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = model.predict(X_val)
            
        # 評価指標の計算
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        scores = {
            'accuracy': accuracy_score(y_values, oof_preds),
            'precision': precision_score(y_values, oof_preds),
            'recall': recall_score(y_values, oof_preds),
            'f1': f1_score(y_values, oof_preds),
            'roc_auc': roc_auc_score(y_values, oof_probs)
        }
        
        # 結果の表示
        print("\n===== 交差検証結果 =====")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
        
        return oof_preds, scores