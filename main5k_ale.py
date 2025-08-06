# mainコードです。回帰コード
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import os
import datetime

# 現在のファイルのディレクトリを取得してPythonパスに追加
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, current_dir)

# matplotlib設定
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import japanize_matplotlib

# 基本ライブラリ
import pandas as pd
import numpy as np
from scipy.stats import rankdata, pearsonr

# scikit-learn関連
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor

# 機械学習ライブラリ
import lightgbm
import xgboost
from catboost import CatBoostRegressor

try:
    from models.ale_plotter import AccumulatedLocalEffects, analyze_accumulated_local_effects
except ImportError:
    print("Warning: ale_plotter module not found")
    AccumulatedLocalEffects = None
    analyze_accumulated_local_effects = None

# データ可視化
import seaborn as sns

# プロジェクト内モジュール
try:
    import models.RegressionResultManager as RegressionModelEvaluator
except ImportError:
    print("Warning: models.RegressionResultManager not found")

# SMOTE関連のインポート（冒頭部分）
try:
    from models.regression_smote import RegressionSMOTE, visualize_smote_effect, IntegerRegressionSMOTE
except ImportError:
    try:
        # フォルダ名が違う場合を考慮
        from regression_smotebackup import RegressionSMOTE
        from regression_smote import IntegerRegressionSMOTE  # 新しいクラスをインポート
        print("Warning: visualize_smote_effect function not found")
    except ImportError:
        print("Warning: RegressionSMOTE not found")
        # 代替実装またはスキップの処理

# 可視化関連
try:
    from visualization.smote_visualization import (
        visualize_smote_effect_with_pdp,
        analyze_smote_effect_comprehensive,
        visualize_smote_data_distribution,
        compare_model_predictions,
        run_smote_analysis_pipeline
    )
except ImportError as e:
    print(f"Warning: Some visualization functions not found: {e}")

try:
    from visualization.regression_visualizer import RegressionVisualizer, EyeTrackingVisualizer
except ImportError:
    print("Warning: RegressionVisualizer not found")

# 部分依存プロット関連
try:
    from models.partial_dependence_plotter_kaikidev_ale import PartialDependencePlotter, analyze_partial_dependence
except ImportError:
    try:
        from models.partial_dependence_plotter_kaikidev_ale import PartialDependencePlotter
        print("Warning: analyze_partial_dependence function not found")
    except ImportError:
        print("Warning: PartialDependencePlotter not found")

# その他のライブラリ
import argparse
# エラー修正パッチ（メインコードの最初に追加）
import warnings
warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# matplotlib のバージョン確認
import matplotlib
import sklearn
print(f"matplotlib: {matplotlib.__version__}, sklearn: {sklearn.__version__}")

def setup_matplotlib_japanese_font():
    """
    matplotlibで日本語を表示するための設定を行う関数
    """
    import matplotlib.pyplot as plt
    import matplotlib
    import platform
    
    # OSの判定
    system = platform.system()
    
    # フォント設定
    if system == 'Windows':
        # Windowsの場合
        font_family = 'MS Gothic'
        matplotlib.rcParams['font.family'] = font_family
    elif system == 'Darwin':
        # macOSの場合
        font_family = 'AppleGothic'
        matplotlib.rcParams['font.family'] = font_family
    else:
        # Linux/その他の場合
        try:
            # IPAフォントがインストールされていることを前提
            font_family = 'IPAGothic'
            matplotlib.rcParams['font.family'] = font_family
        except:
            print("日本語フォントの設定に失敗しました。デフォルトフォントを使用します。")
    
    # matplotlibのグローバル設定
    plt.rcParams['axes.unicode_minus'] = False  # マイナス記号を正しく表示
    
    print(f"matplotlibのフォントを'{font_family}'に設定しました")
    return font_family

# RegressionBaggingModelを回帰用に修正したクラス
class RegressionBaggingModel:
    """
    回帰問題用のバギングモデル
    
    複数のサブモデルを作成し、それぞれの予測を平均化するアプローチを実装しています。
    """
    
    def __init__(self, base_model='lightgbm', n_bags=10, sample_size=0.8, 
                 random_state=42, base_params=None, categorical_features=None):
        """
        初期化メソッド
        
        Parameters:
        -----------
        base_model : str, default='lightgbm'
            使用するベースモデル。'lightgbm', 'xgboost', 'random_forest', 'catboost'から選択
        n_bags : int, default=10
            作成するバッグ（サブモデル）の数
        sample_size : float, default=0.8
            各バッグに使用するデータサンプルの割合
        random_state : int, default=42
            再現性のための乱数シード
        base_params : dict, default=None
            ベースモデル用のパラメータ
        categorical_features : list, default=None
            カテゴリカル変数のインデックスリスト（CatBoost用）
        """
        self.base_model = base_model
        self.n_bags = n_bags
        self.sample_size = sample_size
        self.random_state = random_state
        self.base_params = base_params or {}
        self.categorical_features = categorical_features
        self.models = []
        self.feature_importances_ = None

    def _get_lightgbm_model(self, X_train, y_train, X_eval=None, y_eval=None):
        """LightGBMモデルを作成して学習するメソッド（回帰用）"""
        import lightgbm as lgb
        
        # デフォルトパラメータ（回帰用に変更）
        params = {
            'objective': 'regression',
            'metric': 'rmse',
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
            callbacks=[lgb.early_stopping(stopping_rounds=50)] if lgb_eval else None,
        )
        
        return model
    
    def _get_xgboost_model(self, X_train, y_train, X_eval=None, y_eval=None):
        """XGBoostモデルを作成して学習するメソッド（回帰用）"""
        import xgboost as xgb
        
        # デフォルトパラメータ（回帰用に変更）
        params = {
            'objective': 'reg:squarederror',
            'eval_metric': 'rmse',
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
        """RandomForestモデルを作成して学習するメソッド（回帰用）"""
        from sklearn.ensemble import RandomForestRegressor
        
        # デフォルトパラメータ
        params = {
            'n_estimators': 100,
            'max_depth': 10,
            'random_state': self.random_state
        }
        
        # ユーザー指定パラメータで更新
        params.update(self.base_params)
        
        # モデル作成と学習（回帰器に変更）
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        return model
    
    def _get_catboost_model(self, X_train, y_train, X_eval=None, y_eval=None):
        """CatBoostモデルを作成して学習するメソッド（回帰用）"""
        from catboost import CatBoostRegressor, Pool
        
        # デフォルトパラメータ（回帰用に変更）
        params = {
            'iterations': 1000,
            'learning_rate': 0.05,
            'depth': 6,
            'loss_function': 'RMSE',  # 回帰用に変更
            'random_seed': self.random_state,
            'verbose': False
        }
        
        # ユーザー指定パラメータで更新
        params.update(self.base_params)
        
        # モデル作成（回帰器に変更）
        model = CatBoostRegressor(**params)
        
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
    
    def _sample_and_train(self, X_train, y_train, X_eval=None, y_eval=None):
        """訓練データからサンプリングしてモデルを学習するメソッド"""
        # データのインデックスからランダムサンプリング
        n_samples = X_train.shape[0]
        sample_size = int(n_samples * self.sample_size)
        
        np.random.seed(np.random.randint(0, 10000))  # バッグごとに異なるシード
        indices = np.random.choice(n_samples, sample_size, replace=True)
        
        X_sampled = X_train[indices]
        y_sampled = y_train[indices]
        
        # 指定されたベースモデルに基づいてモデルを学習
        if self.base_model == 'lightgbm':
            model = self._get_lightgbm_model(X_sampled, y_sampled, X_eval, y_eval)
        elif self.base_model == 'xgboost':
            model = self._get_xgboost_model(X_sampled, y_sampled, X_eval, y_eval)
        elif self.base_model == 'random_forest':
            model = self._get_random_forest_model(X_sampled, y_sampled, X_eval, y_eval)
        elif self.base_model == 'catboost':
            model = self._get_catboost_model(X_sampled, y_sampled, X_eval, y_eval)
        else:
            raise ValueError(f"サポートされていないベースモデル: {self.base_model}")
        
        return model
    
    def fit(self, X, y, X_eval=None, y_eval=None):
        """
        複数のモデルを学習するメソッド
        
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
        
        # 複数のモデルを学習
        self.models = []
        for i in range(self.n_bags):
            print(f"バッグ {i+1}/{self.n_bags} を学習中...")
            model = self._sample_and_train(X_values, y_values, X_eval_values, y_eval_values)
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
    
    def predict(self, X):
        """
        全バッグの予測を平均化して予測値を返すメソッド
        
        Parameters:
        -----------
        X : array-like or pandas DataFrame
            特徴量
        
        Returns:
        --------
        predictions : array-like
            予測値
        """
        if isinstance(X, pd.DataFrame):
            X_values = X.values
        else:
            X_values = X
        
        # 予測配列の初期化
        n_samples = X_values.shape[0]
        predictions = np.zeros(n_samples)
        
        # 全モデルの予測を合計
        for model in self.models:
            if self.base_model == 'lightgbm':
                preds = model.predict(X_values)
            elif self.base_model == 'xgboost':
                import xgboost as xgb
                dtest = xgb.DMatrix(X_values)
                preds = model.predict(dtest)
            elif self.base_model == 'random_forest':
                preds = model.predict(X_values)
            elif self.base_model == 'catboost':
                from catboost import Pool
                if self.categorical_features is not None:
                    test_pool = Pool(X_values, cat_features=self.categorical_features)
                    preds = model.predict(test_pool)
                else:
                    preds = model.predict(X_values)
            else:
                raise ValueError(f"サポートされていないベースモデル: {self.base_model}")
            
            predictions += preds
        
        # 予測の平均化
        predictions /= len(self.models)
        
        return predictions
    
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
        # 簡易版では、バッグの数を変えてみる
        param_grid = {'n_bags': [5, 10, 15]}
        
        best_score = float('inf')  # RMSEを最小化したいので無限大から開始
        best_params = {}
        
        for n_bags in param_grid['n_bags']:
            print(f"\nバッグ数 {n_bags} でテスト中...")
            self.n_bags = n_bags
            self.fit(X_train, y_train)
            y_pred = self.predict(X_test)
            
            # RMSEで評価（回帰問題なので）
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            print(f"RMSE: {rmse:.4f}")
            
            if rmse < best_score:
                best_score = rmse
                best_params = {'n_bags': n_bags}
        
        # 最適パラメータでモデルを再学習
        print(f"\n最適なバッグ数: {best_params['n_bags']}")
        self.n_bags = best_params['n_bags']
        self.fit(X_train, y_train)
        
        return best_params, self
    
    def get_model(self):
        """互換性のために自身を返すメソッド"""
        return self
    
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
        from sklearn.model_selection import KFold
        
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
            
        # 交差検証の設定（回帰なのでKFoldを使用）
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        
        # Out-of-fold予測の初期化
        oof_preds = np.zeros(len(y_values))
        
        # 交差検証ループ
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_values)):
            print(f"\n===== Fold {fold+1}/{n_splits} =====")
            
            # データの分割
            X_train, X_val = X_values[train_idx], X_values[val_idx]
            y_train, y_val = y_values[train_idx], y_values[val_idx]
            
            # モデルの作成と学習
            model = RegressionBaggingModel(
                base_model=base_model,
                n_bags=n_bags,
                random_state=random_state + fold,
                base_params=base_params,
                categorical_features=categorical_features
            )
            model.fit(X_train, y_train)
            
            # バリデーションデータでの予測
            oof_preds[val_idx] = model.predict(X_val)
            
        # 評価指標の計算
        mse = mean_squared_error(y_values, oof_preds)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_values, oof_preds)
        r2 = r2_score(y_values, oof_preds)
        
        scores = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        # 結果の表示
        print("\n===== 交差検証結果 =====")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
        
        return oof_preds, scores

# メイン関数

class AccumulatedLocalEffects:
    """Accumulated Local Effects Plot (ALE) の実装"""
    
    def __init__(self, estimator, X):
        self.estimator = estimator
        if isinstance(X, pd.DataFrame):
            self.X = X.values
            self.feature_names = X.columns.tolist()
        else:
            self.X = X
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
        
        print(f"✓ ALE初期化: {len(self.feature_names)}特徴量, {len(self.X)}サンプル")
    
    def _predict_average(self, X_subset, j, value):
        """特定の特徴量を固定値に設定した場合の平均予測値を計算"""
        if len(X_subset) == 0:
            return 0.0
        
        X_modified = X_subset.copy()
        X_modified[:, j] = value
        
        try:
            predictions = self.estimator.predict(X_modified)
            return np.mean(predictions)
        except Exception as e:
            print(f"予測エラー: {e}")
            return 0.0
    
    def calculate_ale(self, feature_idx, n_grid=30):
        """特定の特徴量のALEを計算"""
        print(f"特徴量{feature_idx}({self.feature_names[feature_idx]})のALE計算中...")
        
        # 特徴量の値を取得
        feature_values = self.X[:, feature_idx]
        
        # 無効値のチェック
        valid_mask = np.isfinite(feature_values)
        if not np.any(valid_mask):
            print(f"エラー: 特徴量{feature_idx}に有効な値がありません")
            return np.array([]), np.array([])
        
        # 分位点の計算
        try:
            quantiles = np.linspace(0, 1, n_grid + 1)
            xjks = np.quantile(feature_values[valid_mask], quantiles)
            xjks = np.unique(xjks)  # 重複除去
            
            if len(xjks) < 2:
                print(f"エラー: 特徴量{feature_idx}の分位点が不十分")
                return np.array([]), np.array([])
            
            n_grid = len(xjks) - 1
            
        except Exception as e:
            print(f"分位点計算エラー: {e}")
            return np.array([]), np.array([])
        
        # 区間ごとの効果計算
        local_effects = np.zeros(n_grid)
        
        for k in range(1, n_grid + 1):
            # 区間内のデータを選択
            if k == 1:
                mask = (self.X[:, feature_idx] >= xjks[k-1]) & (self.X[:, feature_idx] <= xjks[k])
            else:
                mask = (self.X[:, feature_idx] > xjks[k-1]) & (self.X[:, feature_idx] <= xjks[k])
            
            if np.sum(mask) > 0:
                try:
                    upper_pred = self._predict_average(self.X[mask], feature_idx, xjks[k])
                    lower_pred = self._predict_average(self.X[mask], feature_idx, xjks[k-1])
                    local_effects[k-1] = upper_pred - lower_pred
                except Exception as e:
                    local_effects[k-1] = 0
            else:
                local_effects[k-1] = 0
        
        # 累積和とセンタリング
        accumulated_effects = np.cumsum(local_effects)
        accumulated_effects -= np.mean(accumulated_effects)
        
        # グリッドポイント
        grid_points = (xjks[:-1] + xjks[1:]) / 2
        
        print(f"✓ ALE計算完了: {len(grid_points)}点")
        return grid_points, accumulated_effects
    
    def plot_single_feature(self, feature_idx, n_grid=30, ax=None, figsize=(8, 6)):
        """単一特徴量のALEプロット"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # ALE計算
        grid_values, ale_values = self.calculate_ale(feature_idx, n_grid)
        
        if len(grid_values) == 0:
            ax.text(0.5, 0.5, f"ALE計算失敗\n{self.feature_names[feature_idx]}", 
                   ha='center', va='center', transform=ax.transAxes)
            return fig
        
        # プロット
        ax.plot(grid_values, ale_values, 'b-', linewidth=2, label='ALE')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # データ分布
        y_offset = -0.1 * (np.max(ale_values) - np.min(ale_values)) if len(ale_values) > 0 else -0.1
        ax.scatter(self.X[:, feature_idx], np.full(len(self.X), y_offset),
                  alpha=0.1, s=10, color='gray', label='Data')
        
        # ラベル
        ax.set_xlabel(self.feature_names[feature_idx])
        ax.set_ylabel("ALE")
        ax.set_title(f"ALE: {self.feature_names[feature_idx]}")
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        return fig
    
    def plot_multiple_features(self, feature_indices, n_grid=30, n_cols=3, figsize=None):
        """複数特徴量のALEプロット"""
        n_features = len(feature_indices)
        if n_features == 0:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "表示する特徴量がありません", ha='center', va='center')
            return fig
        
        n_rows = (n_features + n_cols - 1) // n_cols
        if figsize is None:
            figsize = (5 * n_cols, 4 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # axes処理
        if n_rows == 1 and n_cols == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        # 各特徴量をプロット
        for idx, feature_idx in enumerate(feature_indices):
            if n_rows == 1 and n_cols > 1:
                ax = axes[idx]
            elif n_cols == 1 and n_rows > 1:
                ax = axes[idx]
            else:
                row, col = idx // n_cols, idx % n_cols
                ax = axes[row, col]
            
            try:
                self.plot_single_feature(feature_idx, n_grid, ax=ax)
            except Exception as e:
                ax.text(0.5, 0.5, f"エラー\n{str(e)}", ha='center', va='center', transform=ax.transAxes)
        
        # 余分なサブプロットを削除
        if n_rows > 1 and n_cols > 1:
            for idx in range(n_features, n_rows * n_cols):
                row, col = idx // n_cols, idx % n_cols
                fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        return fig


def analyze_accumulated_local_effects_fixed(
    model, X, feature_importances=None, output_dir='ale_plots',
    n_top_features=6, n_grid=30, compare_with_pdp=False, pdp_calculator=None
):
    """ALE分析の修正版"""
    
    print(f"\n=== ALE分析開始（修正版） ===")
    print(f"出力ディレクトリ: {output_dir}")
    print(f"対象特徴量数: {n_top_features}")
    
    # 出力ディレクトリ作成
    try:
        os.makedirs(output_dir, exist_ok=True)
        print(f"✓ 出力ディレクトリ確認: {output_dir}")
    except Exception as e:
        print(f"出力ディレクトリ作成エラー: {e}")
        output_dir = '.'
    
    # ALE初期化
    try:
        ale = AccumulatedLocalEffects(model, X)
    except Exception as e:
        print(f"ALE初期化エラー: {e}")
        return None
    
    # 特徴量選択
    feature_indices = []
    if feature_importances is not None and len(feature_importances) > 0:
        print("特徴量重要度による選択...")
        top_features = feature_importances.head(n_top_features)['feature'].tolist()
        
        for feature in top_features:
            if feature in ale.feature_names:
                feature_indices.append(ale.feature_names.index(feature))
        
        print(f"選択特徴量: {[ale.feature_names[i] for i in feature_indices]}")
    else:
        print("デフォルト特徴量選択...")
        feature_indices = list(range(min(n_top_features, len(ale.feature_names))))
    
    if not feature_indices:
        print("エラー: 有効な特徴量がありません")
        return ale
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 1. 複数特徴量プロット
    try:
        print("複数特徴量ALEプロット生成中...")
        fig = ale.plot_multiple_features(feature_indices, n_grid=n_grid)
        
        multi_path = os.path.join(output_dir, f'ale_top{len(feature_indices)}_{timestamp}.png')
        fig.savefig(multi_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print(f"✓ 保存成功: {multi_path}")
        
        # ファイル存在確認
        if os.path.exists(multi_path):
            size = os.path.getsize(multi_path)
            print(f"✓ ファイルサイズ: {size} bytes")
        else:
            print("✗ ファイルの保存に失敗")
            
    except Exception as e:
        print(f"複数特徴量プロット生成エラー: {e}")
        import traceback
        traceback.print_exc()
    
    # # 2. 個別特徴量プロット（上位3つ）
    # for i, feature_idx in enumerate(feature_indices[:3]):
    #     try:
    #         feature_name = ale.feature_names[feature_idx]
    #         print(f"個別ALEプロット生成: {feature_name}")
            
    #         fig = ale.plot_single_feature(feature_idx, n_grid=n_grid)
            
    #         # ファイル名の安全化
    #         safe_name = feature_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
    #         single_path = os.path.join(output_dir, f'ale_{safe_name}_{timestamp}.png')
            
    #         fig.savefig(single_path, dpi=300, bbox_inches='tight')
    #         plt.close(fig)
            
    #         if os.path.exists(single_path):
    #             print(f"✓ 保存成功: {single_path}")
    #         else:
    #             print(f"✗ 保存失敗: {single_path}")
                
    #     except Exception as e:
    #         print(f"個別プロット生成エラー（{feature_idx}）: {e}")
    #         import traceback
    #         traceback.print_exc()
    
    # 3. 出力確認
    try:
        if os.path.exists(output_dir):
            files = [f for f in os.listdir(output_dir) if f.startswith('ale_') and f.endswith('.png')]
            print(f"\n✓ 生成されたALEファイル: {len(files)}個")
            for f in files:
                print(f"  - {f}")
        else:
            print("✗ 出力ディレクトリが存在しません")
    except Exception as e:
        print(f"出力確認エラー: {e}")
    
    print("ALE分析完了!")
    return ale


def safe_convert_array(x):
    """安全に配列文字列を数値に変換する関数"""
    if not isinstance(x, str):
        return np.nan
    
    if '...' in x:
        return np.nan
    
    try:
        x = x.strip('[]')
        numbers = []
        for num in x.split():
            try:
                numbers.append(float(num))
            except ValueError:
                continue
        
        if not numbers:
            return np.nan
        
        return np.mean(numbers)
    except:
        return np.nan



# ファイルの上部に以下の関数を追加
def mean_absolute_percentage_error(y_true, y_pred):
    """
    平均絶対パーセント誤差(MAPE)を計算する関数
    
    Parameters:
    -----------
    y_true : array-like
        実測値
    y_pred : array-like
        予測値
    
    Returns:
    --------
    mape : float
        平均絶対パーセント誤差（%単位）
    """
    import numpy as np
    
    # 0による除算を防ぐため、小さな値を追加
    epsilon = np.finfo(np.float64).eps
    
    # 実測値が0または非常に小さい場合の対処
    mask = np.abs(y_true) > epsilon
    
    if np.sum(mask) == 0:
        return np.nan  # 全ての実測値が0または非常に小さい場合
    
    # マスクを適用して計算
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    # MAPEの計算（%で表示）
    mape = np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
    
    return mape
def spearman_correlation(x, y):
    """
    スピアマンの順位相関係数を計算する関数
    
    Parameters:
    -----------
    x : array-like
        第1変数のデータ
    y : array-like
        第2変数のデータ
        
    Returns:
    --------
    float
        スピアマンの順位相関係数
    """
    # NaNを含むデータを除外する
    mask = ~(np.isnan(x) | np.isnan(y))
    x_clean = np.array(x)[mask]
    y_clean = np.array(y)[mask]
    
    # データが空の場合はNaNを返す
    if len(x_clean) == 0 or len(y_clean) == 0:
        return np.nan
    
    # データを順位に変換
    # method='average'は同順位の場合に平均順位を割り当てる
    rank_x = rankdata(x_clean, method='average')
    rank_y = rankdata(y_clean, method='average')
    
    # 順位の差の二乗和を計算
    n = len(x_clean)
    d_squared_sum = np.sum((rank_x - rank_y) ** 2)
    
    # スピアマンの順位相関係数を計算
    # 公式: rho = 1 - (6 * Σd²) / (n * (n² - 1))
    rho = 1 - (6 * d_squared_sum) / (n * (n**2 - 1))
    
    return rho


def prepare_data_for_regression(df, target_column='target'):
    """データの前処理を行う関数（回帰用）"""
    print("\nデータの形状:", df.shape)
    
    # 重要: ID列の取得と保存
    id_column = None
    if 'InspectionDateAndId' in df.columns:
        id_column = df['InspectionDateAndId'].copy()
    elif 'Id' in df.columns:
        id_column = df['Id'].copy()
    
    # IDカラムを除外し、特徴量として使用するカラムを選択
    drop_cols = []
    if 'InspectionDateAndId' in df.columns:
        drop_cols.append('InspectionDateAndId')
    if 'Id' in df.columns:
        drop_cols.append('Id')
    if target_column in df.columns:
        drop_cols.append(target_column)
    
    features = df.drop(drop_cols, axis=1, errors='ignore')
    
    # 目的変数はそのまま使用（回帰なので変換は不要）
    target = df[target_column].copy()
    
    # 配列文字列の処理
    for col in ['freq', 'power_spectrum']:
        if col in features.columns:
            features[col] = features[col].apply(safe_convert_array)
    
    # 欠損値の削除
    features = features.dropna()
    target = target[features.index]
    
    # ID列も同様にフィルタリング
    if id_column is not None:
        id_column = id_column[features.index]
    
    print(f"\n処理後のデータ数: {len(features)}")
    print("\n目的変数の統計量:")
    print(target.describe())
    
    # featuresとtargetに加えて、id_columnも返す
    return features, target, id_column

def train_regression_model_with_smote(features, target, id_column=None, model_class=None, 
                                     use_bagging=False, random_state=42,
                                     # 新しいSMOTE関連パラメータ
                                     use_smote=False, smote_method='density', smote_kwargs=None,
                                     use_integer_smote=True, target_min=9, target_max=30,
                                     **kwargs):
    """
    適切なSMOTE適用を行うモデル学習関数
    
    処理順序:
    1. データを訓練・テストに分割
    2. 訓練データのみにSMOTEを適用
    3. テストデータは元データのまま評価
    
    Parameters:
    -----------
    features : pandas.DataFrame
        特徴量データ
    target : pandas.Series
        目的変数
    id_column : pandas.Series, optional
        ID列
    model_class : callable, optional
        モデルクラスを作成する関数
    use_bagging : bool, default=False
        バギングモデルを使用するかどうか
    random_state : int, default=42
        乱数シード
    use_smote : bool, default=False
        SMOTEを使用するかどうか
    smote_method : str, default='density'
        SMOTEの手法 ('binning', 'density', 'outliers')
    smote_kwargs : dict, optional
        SMOTE手法固有のパラメータ
    use_integer_smote : bool, default=True
        整数値対応SMOTEを使用するかどうか
    target_min : int, default=9
        目的変数の最小値（MoCAスコア用）
    target_max : int, default=30
        目的変数の最大値（MoCAスコア用）
    **kwargs : dict
        その他のパラメータ
    
    Returns:
    --------
    dict
        学習結果の辞書
        - model: 学習済みモデル
        - predictions: テストデータでの予測値
        - true_values: テストデータの真値
        - feature_importance: 特徴量重要度
        - test_indices: テストデータのインデックス
        - best_params: 最適パラメータ
        - features: 元の特徴量データ
        - target: 元の目的変数
        - X_train: スケーリング済み訓練データ
        - X_train_orig: SMOTE適用前の元の訓練データ
        - X_train_after_smote: SMOTE適用後の訓練データ（未スケーリング）
        - y_train: SMOTE適用後の訓練ターゲット
        - y_train_original: SMOTE適用前の元の訓練ターゲット
        - X_test: スケーリング済みテストデータ
        - X_test_orig: 元のテストデータ（未スケーリング）
        - id_test: テストデータのID列
        - scaler: 使用したスケーラー
        - smote_applied: SMOTEが実際に適用されたかどうか
        - smote_method: 使用したSMOTE手法
        - training_data_size: SMOTE適用後の訓練データサイズ
        - original_training_size: 元の訓練データサイズ
    """
    # 1. 最初にデータを分割（SMOTE適用前）
    print(f"\n=== データ分割 ===")
    if id_column is not None:
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            features, target, id_column, test_size=0.1, random_state=random_state
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.1, random_state=random_state
        )
        id_train, id_test = None, None
    
    print(f"データ分割結果:")
    print(f"  訓練データ: {X_train.shape}")
    print(f"  テストデータ: {X_test.shape}")
    print(f"  訓練データの目的変数範囲: {y_train.min():.2f} - {y_train.max():.2f}")
    print(f"  テストデータの目的変数範囲: {y_test.min():.2f} - {y_test.max():.2f}")
    
    # 元の訓練・テストデータを保存（可視化やSMOTE効果確認用）
    X_train_original = X_train.copy()
    y_train_original = y_train.copy()
    X_test_original = X_test.copy()
    
    # 2. 訓練データのみにSMOTEを適用
    print(f"\n=== SMOTE適用 ===")
    smote_applied = False
    if use_smote:
        print(f"訓練データのみにSMOTE（{smote_method}）を適用します...")
        print("※テストデータは元データのまま保持（データリーク防止）")
        
        # SMOTE用パラメータの設定
        if smote_kwargs is None:
            smote_kwargs = {
                'binning': {'sampling_strategy': 'auto', 'n_bins': 10},
                'density': {'density_threshold': 0.3},
                'outliers': {'outlier_threshold': 0.15}
            }.get(smote_method, {})
        
        print(f"SMOTE設定: {smote_kwargs}")
        
        # 整数値対応SMOTEまたは通常のSMOTEを選択
        smote_instance = None
        if use_integer_smote:
            try:
                from regression_smote import IntegerRegressionSMOTE
                smote_instance = IntegerRegressionSMOTE(
                    method=smote_method, 
                    k_neighbors=5, 
                    random_state=random_state,
                    target_min=target_min,
                    target_max=target_max
                )
                print(f"整数値対応SMOTEを使用します（範囲: {target_min}-{target_max}）")
            except ImportError:
                print("整数値対応SMOTEが見つかりません。通常のSMOTEを使用します")
                try:
                    from regression_smotebackup import RegressionSMOTE
                    smote_instance = RegressionSMOTE(method=smote_method, k_neighbors=5, random_state=random_state)
                    use_integer_smote = False
                except ImportError:
                    print("SMOTEモジュールが見つかりません。SMOTEをスキップします。")
                    use_smote = False
        else:
            try:
                from regression_smotebackup import RegressionSMOTE
                smote_instance = RegressionSMOTE(method=smote_method, k_neighbors=5, random_state=random_state)
                print("通常のSMOTEを使用します")
            except ImportError:
                print("SMOTEモジュールが見つかりません。SMOTEをスキップします。")
                use_smote = False
        
        # SMOTEの実際の適用
        if use_smote and smote_instance is not None:
            try:
                # 訓練データのみリサンプリング
                X_train_values = X_train.values if hasattr(X_train, 'values') else X_train
                y_train_values = y_train.values if hasattr(y_train, 'values') else y_train
                
                print(f"SMOTE適用前の訓練データ:")
                print(f"  サンプル数: {len(X_train_values)}")
                print(f"  目的変数統計: 平均={y_train_values.mean():.2f}, 標準偏差={y_train_values.std():.2f}")
                
                # SMOTE適用
                X_train_resampled, y_train_resampled = smote_instance.fit_resample(
                    X_train_values, y_train_values, **smote_kwargs
                )
                
                # フォールバック処理：通常のSMOTEを使った場合の整数化
                if use_integer_smote and 'IntegerRegressionSMOTE' not in str(type(smote_instance)):
                    print("合成データの整数化を実行中...")
                    n_original = len(y_train_values)
                    
                    # 合成されたサンプルのみ整数化
                    for i in range(n_original, len(y_train_resampled)):
                        y_train_resampled[i] = round(y_train_resampled[i])
                        y_train_resampled[i] = np.clip(y_train_resampled[i], target_min, target_max)
                    
                    # 整数化の確認
                    synthetic_targets = y_train_resampled[n_original:]
                    non_integer_count = np.sum(synthetic_targets % 1 != 0)
                    if non_integer_count > 0:
                        print(f"警告: {non_integer_count}個の非整数値が残っています")
                    else:
                        print("確認: すべての合成目的変数は整数値です")
                
                # データフレームに戻す
                X_train = pd.DataFrame(X_train_resampled, columns=features.columns)
                y_train = pd.Series(y_train_resampled, name=target.name)
                
                # ID列の処理（訓練データのSMOTE適用時）
                if id_train is not None:
                    original_ids = id_train.values
                    n_synthetic = len(X_train_resampled) - len(original_ids)
                    synthetic_ids = [f"synthetic_train_{i:06d}" for i in range(n_synthetic)]
                    id_train = pd.Series(list(original_ids) + synthetic_ids, name=id_train.name)
                    print(f"ID列も拡張しました（合成ID: {n_synthetic}個）")
                
                smote_applied = True
                
                # SMOTE適用結果の詳細表示
                print(f"\nSMOTE適用結果:")
                print(f"  元の訓練データ数: {len(X_train_values)}")
                print(f"  SMOTE後訓練データ数: {len(X_train_resampled)}")
                print(f"  合成データ数: {len(X_train_resampled) - len(X_train_values)}")
                print(f"  データ増加率: {((len(X_train_resampled) - len(X_train_values)) / len(X_train_values) * 100):.2f}%")
                print(f"  テストデータ: 元データのまま（{len(X_test)}サンプル）")
                
                # 合成後の統計
                print(f"\nSMOTE適用後の訓練データ統計:")
                print(f"  目的変数範囲: {y_train_resampled.min():.2f} - {y_train_resampled.max():.2f}")
                print(f"  目的変数統計: 平均={y_train_resampled.mean():.2f}, 標準偏差={y_train_resampled.std():.2f}")
                
                # 合成データのみの統計
                if len(X_train_resampled) > len(X_train_values):
                    synthetic_targets = y_train_resampled[len(X_train_values):]
                    print(f"  合成データの目的変数: 平均={synthetic_targets.mean():.2f}, 標準偏差={synthetic_targets.std():.2f}")
                
                # SMOTE効果の可視化（デバッグモード時）
                if os.environ.get('DEBUG_SMOTE', '0') == '1':
                    try:
                        if use_integer_smote and 'IntegerRegressionSMOTE' in str(type(smote_instance)):
                            from regression_smote import visualize_integer_smote_effect
                            print("整数値SMOTE効果を可視化中...")
                            visualize_integer_smote_effect(
                                X_train_values, y_train_values, 
                                X_train_resampled, y_train_resampled
                            )
                        else:
                            from regression_smotebackup import visualize_smote_effect
                            print("SMOTE効果を可視化中...")
                            visualize_smote_effect(
                                X_train_values, y_train_values, 
                                X_train_resampled, y_train_resampled
                            )
                    except Exception as e:
                        print(f"SMOTE効果の可視化中にエラーが発生しました: {e}")
                        
            except Exception as e:
                print(f"SMOTE適用中にエラーが発生しました: {e}")
                print("SMOTEなしで処理を続行します...")
                import traceback
                traceback.print_exc()
                use_smote = False
                smote_applied = False
    else:
        print("SMOTEは使用しません")
    
    # 3. スケーリング（訓練データで学習、テストデータに適用）
    print(f"\n=== 特徴量スケーリング ===")
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=features.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=features.columns
    )
    
    print(f"スケーリング完了:")
    print(f"  訓練データ: {X_train_scaled.shape}")
    print(f"  テストデータ: {X_test_scaled.shape}")
    
    # 4. モデル学習（SMOTE適用済みの訓練データで学習）
    print(f"\n=== モデル学習 ===")
    best_model = None
    best_params = {}
    feature_importance = None
    
    if use_bagging:
        print("RegressionBaggingModelを使用します...")
        base_model = kwargs.get('base_model', 'lightgbm')
        n_bags = kwargs.get('n_bags', 10)
        print(f"  ベースモデル: {base_model}")
        print(f"  バッグ数: {n_bags}")
        
        model = RegressionBaggingModel(
            base_model=base_model,
            n_bags=n_bags,
            random_state=random_state
        )
        
        try:
            print("グリッドサーチを実行中...")
            best_params, best_model = model.perform_grid_search(
                X_train_scaled, y_train,
                X_test_scaled, y_test
            )
            print(f"グリッドサーチ完了: {best_params}")
        except Exception as e:
            print(f"グリッドサーチでエラーが発生しました: {e}")
            print("基本モデルを使用します...")
            best_model = model
            best_model.fit(X_train_scaled, y_train)
            best_params = {'n_bags': n_bags}
        
        # 特徴量重要度を設定
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"特徴量重要度を取得しました（上位5特徴量）:")
            print(feature_importance.head())
        else:
            feature_importance = pd.DataFrame({'feature': features.columns, 'importance': 0})
            print("特徴量重要度が取得できませんでした")
    
    else:
        # 通常のモデルを使用する場合
        print("通常のモデルを使用します...")
        try:
            model = model_class()
            model_name = model.__class__.__name__
            print(f"  モデル: {model_name}")
            
            # scikit-learnベースのシンプルなグリッドサーチ
            from sklearn.model_selection import GridSearchCV
            
            # モデルタイプごとに適切なパラメータグリッドを設定
            param_grid = {}
            if isinstance(model, lightgbm.LGBMRegressor):
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [-1, 10, 20]
                }
            elif isinstance(model, xgboost.XGBRegressor):
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'max_depth': [3, 6, 10]
                }
            elif isinstance(model, RandomForestRegressor):
                param_grid = {
                    'n_estimators': [50, 100, 200],
                    'max_depth': [None, 10, 20],
                    'min_samples_split': [2, 5, 10]
                }
            elif isinstance(model, CatBoostRegressor):
                param_grid = {
                    'iterations': [50, 100, 200],
                    'learning_rate': [0.01, 0.05, 0.1],
                    'depth': [4, 6, 8]
                }
            
            if param_grid:
                print(f"グリッドサーチを実行中（パラメータ組み合わせ数: {len(list(param_grid.values())[0]) ** len(param_grid)}）...")
                grid_search = GridSearchCV(
                    model, 
                    param_grid,
                    cv=3,
                    scoring='neg_mean_squared_error',
                    n_jobs=-1,
                    verbose=0
                )
                grid_search.fit(X_train_scaled, y_train)
                best_model = grid_search.best_estimator_
                best_params = grid_search.best_params_
                print(f"グリッドサーチ完了")
                print(f"最適パラメータ: {best_params}")
                print(f"最適CVスコア: {-grid_search.best_score_:.4f}")
            else:
                print("パラメータグリッドが定義されていません。デフォルトパラメータで学習します...")
                model.fit(X_train_scaled, y_train)
                best_model = model
                best_params = {}
        except Exception as e:
            print(f"グリッドサーチでエラーが発生しました: {e}")
            print("基本モデルを使用します...")
            import traceback
            traceback.print_exc()
            
            try:
                model = model_class()
                model.fit(X_train_scaled, y_train)
                best_model = model
                best_params = {}
            except Exception as e2:
                print(f"基本モデルの学習でもエラーが発生しました: {e2}")
                raise
        
        # 特徴量の重要度
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
            print(f"特徴量重要度を取得しました（上位5特徴量）:")
            print(feature_importance.head())
        else:
            feature_importance = pd.DataFrame({'feature': features.columns, 'importance': 0})
            print("特徴量重要度が取得できませんでした")
    
    # 5. テストデータで予測・評価（元データで評価）
    print(f"\n=== 予測・評価 ===")
    print("テストデータで予測を実行中...")
    y_pred = best_model.predict(X_test_scaled)
    
    # 基本的な評価指標の計算
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"テストデータでの性能:")
    print(f"  RMSE: {rmse:.4f}")
    print(f"  MAE: {mae:.4f}")
    print(f"  R²: {r2:.4f}")
    
    # 予測値の範囲チェック
    print(f"\n予測値の統計:")
    print(f"  範囲: {y_pred.min():.2f} - {y_pred.max():.2f}")
    print(f"  平均: {y_pred.mean():.2f}")
    print(f"  標準偏差: {y_pred.std():.2f}")
    
    print(f"\n実測値の統計:")
    print(f"  範囲: {y_test.min():.2f} - {y_test.max():.2f}")
    print(f"  平均: {y_test.mean():.2f}")
    print(f"  標準偏差: {y_test.std():.2f}")
    
    # 6. 結果の辞書を作成
    print(f"\n=== 結果のまとめ ===")
    result_dict = {
        'model': best_model,
        'predictions': y_pred,
        'true_values': y_test,
        'feature_importance': feature_importance,
        'test_indices': X_test.index,
        'best_params': best_params,
        'features': features,
        'target': target,
        'X_train': X_train_scaled,
        'X_train_orig': X_train_original,  # SMOTE適用前の元の訓練データ
        'X_train_after_smote': X_train,    # SMOTE適用後の訓練データ（未スケーリング）
        'y_train': y_train,  # SMOTE適用後の訓練ターゲット
        'y_train_original': y_train_original,  # SMOTE適用前の元の訓練ターゲット
        'X_test': X_test_scaled,
        'X_test_orig': X_test_original,  # 元のテストデータ（未スケーリング）
        'id_test': id_test,
        'scaler': scaler,
        'smote_applied': smote_applied,
        'smote_method': smote_method if smote_applied else None,
        'training_data_size': len(X_train),  # SMOTE適用後のサイズ
        'original_training_size': len(X_train_original),  # 元の訓練データサイズ
        'test_rmse': rmse,
        'test_mae': mae,
        'test_r2': r2
    }
    
    # SMOTEが適用された場合の追加情報
    if smote_applied:
        print(f"SMOTE適用情報:")
        print(f"  手法: {smote_method}")
        print(f"  元訓練データ数: {len(X_train_original)}")
        print(f"  SMOTE後訓練データ数: {len(X_train)}")
        print(f"  合成データ数: {len(X_train) - len(X_train_original)}")
        print(f"  増加率: {((len(X_train) - len(X_train_original)) / len(X_train_original) * 100):.2f}%")
        print(f"  テストデータ: 元データのまま評価")
        
        result_dict['smote_info'] = {
            'method': smote_method,
            'original_size': len(X_train_original),
            'resampled_size': len(X_train),
            'synthetic_count': len(X_train) - len(X_train_original),
            'increase_ratio': ((len(X_train) - len(X_train_original)) / len(X_train_original) * 100),
            'parameters': smote_kwargs
        }
    else:
        print("SMOTEは適用されませんでした")
    
    print("\n学習完了!")
    return result_dict

def evaluate_regression_results(results):
    """回帰結果の評価を行う関数"""
    y_true = results['true_values']
    y_pred = results['predictions']
    
    # 評価指標の計算
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    mape = mean_absolute_percentage_error(y_true, y_pred)
    # ピアソン相関係数
    pearson_corr, p_value = pearsonr(y_true, y_pred)
     # スピアマン相関係数
    spearman_corr = spearman_correlation(y_true, y_pred)

    metrics = {
        'MSE': mse,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape,
        'R2': r2,
        'Pearson相関係数': pearson_corr,
        'P値': p_value
        
    }
    # スピアマン相関係数を追加
    metrics['Spearman_Corr'] = spearman_corr
    # テストデータの元のインデックスを取得
    test_indices = results['test_indices']
    
    # 予測値と実測値の比較データフレーム
    pred_df = pd.DataFrame({
        'True_Value': results['true_values'],
        'Predicted_Value': results['predictions'],
        'Error': results['true_values'] - results['predictions'],
        'Abs_Error': np.abs(results['true_values'] - results['predictions'])
    }, index=test_indices)
    
    # ID列の取得
    id_test = results.get('id_test')
    
    # IDカラムがある場合は先頭に追加
    if id_test is not None:
        # ID列名を特定（InspectionDateAndIdかIdか）
        if isinstance(id_test, pd.Series):
            id_name = id_test.name if hasattr(id_test, 'name') and id_test.name else 'InspectionDateAndId'
        else:
            id_name = 'InspectionDateAndId'  # デフォルト名
        
        # ID列をデータフレームに追加
        id_df = pd.DataFrame({id_name: id_test}, index=test_indices)
        pred_df = id_df.join(pred_df)
        
        # インデックスをリセット
        pred_df = pred_df.reset_index(drop=True)
    
    return pred_df, metrics
def organize_regression_result_files(data_file_name, output_dir):
    """
    現在の実行で生成された回帰分析の結果ファイルのみを新しいディレクトリに整理する関数
    
    Parameters:
    -----------
    data_file_name : str
        データファイル名
    output_dir : str
        出力ディレクトリ
    
    Returns:
    --------
    str
        新しい出力ディレクトリのパス
    """
    # データファイル名から拡張子を除去したベース名を取得
    base_name = os.path.splitext(os.path.basename(data_file_name))[0]
    
    # 現在の時刻を取得（実行開始時刻として扱う）
    current_time = datetime.datetime.now()
    # 少し前の時刻（例：30分前）を計算し、その時刻以降に生成・更新されたファイルのみをコピー
    filter_time = current_time - datetime.timedelta(minutes=30)
    
    # 新しい出力ディレクトリを作成
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    new_output_dir = f"{base_name}_{timestamp}"
    
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)
        print(f"\n新しい出力ディレクトリを作成しました: {new_output_dir}")
    
    file_count = 0
    
    # output_dirが存在する場合のみ処理を実行
    if os.path.exists(output_dir):
        # 重要なパターンのファイルをコピー
        important_patterns = [
            'pdp_*.png',                    # 部分依存プロット
            'true_vs_predicted.png',        # 真値と予測値の散布図
            'residuals.png',                # 残差プロット
            'residual_distribution.png',    # 残差分布
            'feature_importance.png',       # 特徴量重要度
            'feature_importance_*.csv',     # 特徴量重要度CSV
            'correlation_heatmap.png',      # 相関ヒートマップ
            'predictions_*.csv',            # 予測結果
            'metrics_*.txt'                 # 評価指標
        ]
        
        # パターンに一致する最近のファイルをコピー
        for pattern in important_patterns:
            import glob
            pattern_path = os.path.join(output_dir, pattern)
            for source_file in glob.glob(pattern_path):
                if os.path.isfile(source_file):
                    # ファイルの更新時刻を取得
                    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(source_file))
                    # フィルタ時間より新しいファイルのみコピー
                    if mod_time >= filter_time:
                        filename = os.path.basename(source_file)
                        dest_file = os.path.join(new_output_dir, filename)
                        try:
                            shutil.copy2(source_file, dest_file)
                            file_count += 1
                            print(f"コピーしました: {dest_file}")
                        except Exception as e:
                            print(f"ファイルのコピー中にエラーが発生しました: {source_file} -> {e}")
        
        # saved_modelディレクトリの処理
        saved_model_dir = os.path.join(output_dir, 'saved_model')
        if os.path.exists(saved_model_dir) and os.path.isdir(saved_model_dir):
            # 新しいディレクトリ内にsaved_modelディレクトリを作成
            new_saved_model_dir = os.path.join(new_output_dir, 'saved_model')
            if not os.path.exists(new_saved_model_dir):
                os.makedirs(new_saved_model_dir)
            
            # saved_modelディレクトリ内の最近のファイルをコピー
            for filename in os.listdir(saved_model_dir):
                source_path = os.path.join(saved_model_dir, filename)
                if os.path.isfile(source_path):
                    # base_nameが含まれるファイルのみを対象にする
                    if base_name in filename:
                        # ファイルの更新時刻を取得
                        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(source_path))
                        # フィルタ時間より新しいファイルのみコピー
                        if mod_time >= filter_time:
                            dest_path = os.path.join(new_saved_model_dir, filename)
                            try:
                                shutil.copy2(source_path, dest_path)
                                file_count += 1
                                print(f"モデルファイルをコピーしました: {dest_path}")
                            except Exception as e:
                                print(f"モデルファイルのコピー中にエラーが発生しました: {source_path} -> {e}")
    
    else:
        print(f"警告: 出力ディレクトリ {output_dir} が見つかりません")
    
    print(f"{file_count}個の新しいファイルを {new_output_dir} にコピーしました")
    return new_output_dir
def save_regression_model(model, features, target, scaler, output_path, model_name):
    """
    回帰モデルと関連するデータを保存する関数
    
    Parameters:
    -----------
    model : object
        保存する学習済みモデル
    features : pandas.DataFrame
        学習に使用した特徴量
    target : pandas.Series or numpy.ndarray
        学習に使用した目的変数
    scaler : object
        特徴量のスケーリングに使用したスケーラー
    output_path : str
        モデルを保存するディレクトリのパス
    model_name : str
        保存するモデルの名前
    """
    # 出力ディレクトリの作成
    if not os.path.exists(output_path):
        os.makedirs(output_path)
        print(f"モデル保存ディレクトリを作成しました: {output_path}")
    
    # saved_modelサブディレクトリの作成
    saved_model_dir = os.path.join(output_path, 'saved_model')
    if not os.path.exists(saved_model_dir):
        os.makedirs(saved_model_dir)
    
    import pickle
    
    # モデルを保存
    model_path = os.path.join(saved_model_dir, f"{model_name}_model.pkl")
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"モデルを保存しました: {model_path}")
    
    # スケーラーを保存
    scaler_path = os.path.join(saved_model_dir, f"{model_name}_scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"スケーラーを保存しました: {scaler_path}")
    
    # 特徴量の列名を保存
    feature_names = features.columns.tolist()
    feature_names_path = os.path.join(saved_model_dir, f"{model_name}_feature_names.pkl")
    with open(feature_names_path, 'wb') as f:
        pickle.dump(feature_names, f)
    print(f"特徴量名を保存しました: {feature_names_path}")
    
    # 学習に使用したデータのサンプルを保存
    data_sample = pd.DataFrame(features.head(100))
    data_sample['target'] = target.iloc[:100] if hasattr(target, 'iloc') else target[:100]
    data_sample_path = os.path.join(saved_model_dir, f"{model_name}_data_sample.csv")
    data_sample.to_csv(data_sample_path, index=False)
    print(f"学習データのサンプルを保存しました: {data_sample_path}")
    
    # モデル情報のテキストファイル作成
    info_path = os.path.join(saved_model_dir, f"{model_name}_info.txt")
    with open(info_path, 'w') as f:
        f.write(f"モデル名: {model_name}\n")
        f.write(f"モデルタイプ: {type(model).__name__}\n")
        f.write(f"特徴量数: {features.shape[1]}\n")
        f.write(f"学習データ数: {len(features)}\n")
        f.write(f"保存日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        # モデルのパラメータがある場合は追加
        if hasattr(model, 'get_params'):
            f.write("\nモデルパラメータ:\n")
            for param, value in model.get_params().items():
                f.write(f"  {param}: {value}\n")
    
    print(f"モデル情報を保存しました: {info_path}")
    
    return model_path

# メインコードの上部にインポートを追加
from regression_smotebackup import RegressionSMOTE, visualize_smote_effect



def run_regression_analysis(df, model_class=None, use_bagging=False, 
                         random_state=42, output_dir=None, target_column='target', 
                         data_file_name=None, organize_files=True,
                         # SMOTE関連のパラメータ
                         use_smote=False, smote_method='density', smote_kwargs=None,
                         # 整数値対応SMOTEのパラメータ
                         use_integer_smote=True, target_min=9, target_max=30,
                         # 部分依存プロット関連のパラメータを追加
                         generate_pdp=True, pdp_n_features=6, pdp_grid_resolution=50,
                         # ALE関連のパラメータを追加
                         generate_ale=True, ale_n_features=6, ale_grid_resolution=30,
                         compare_ale_pdp=True,
                         **kwargs):
    """
    回帰分析を実行する関数（修正版：適切なSMOTE適用）
    
    """
    import datetime
    import shutil
    
    # ============================================
    # 1. dfパラメータの初期確認と設定
    # ============================================
    
    # ログ出力でデータの基本情報を確認
    print(f"\n{'='*50}")
    print(f"回帰分析開始")
    print(f"{'='*50}")
    print(f"\n=== データ情報 ===")
    print(f"データ形状: {df.shape}")
    print(f"カラム数: {len(df.columns)}")
    print(f"行数: {len(df)}")
    print(f"目的変数: {target_column}")
    print(f"dfのカラム: {df.columns.tolist()}")
    
    # dfの基本統計を出力
    if len(df) > 0:
        print(f"\n目的変数 '{target_column}' の統計:")
        if target_column in df.columns:
            print(df[target_column].describe())
            
            # 目的変数の分布確認
            print(f"\n目的変数の分布:")
            print(f"  欠損値: {df[target_column].isnull().sum()}個")
            print(f"  ユニーク値数: {df[target_column].nunique()}個")
            
            # MoCAスコアの場合の詳細
            if target_min <= df[target_column].min() <= target_max and target_min <= df[target_column].max() <= target_max:
                unique_values = sorted(df[target_column].dropna().unique())
                print(f"  ユニーク値: {unique_values}")
        else:
            print(f"警告: 目的変数 '{target_column}' が見つかりません!")
            print(f"利用可能なカラム: {df.columns.tolist()}")
            raise ValueError(f"目的変数 '{target_column}' が見つかりません")
    
    # SMOTE用パラメータのデフォルト値設定
    if smote_kwargs is None:
        smote_kwargs = {
            'binning': {'sampling_strategy': 'auto', 'n_bins': 10},
            'density': {'density_threshold': 0.3},
            'outliers': {'outlier_threshold': 0.15}
        }.get(smote_method, {})
    
    # ============================================
    # 2. 出力ディレクトリの設定（dfの情報を使用）
    # ============================================
    
    # 新しい出力ディレクトリをここで作成
    original_output_dir = output_dir
    new_output_dir = None
    
    print(f"\n=== 出力ディレクトリ設定 ===")
    if output_dir and data_file_name:
        base_name = os.path.splitext(os.path.basename(data_file_name))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # SMOTE使用時はファイル名に含める
        smote_suffix = f"_smote_{smote_method}" if use_smote else ""
        
        # dfのサイズ情報も含める（オプション）
        size_info = f"_n{len(df)}"
        new_output_dir = f"{base_name}{smote_suffix}{size_info}_{timestamp}"
        
        if not os.path.exists(new_output_dir):
            os.makedirs(new_output_dir)
            print(f"新しい出力ディレクトリを作成しました: {new_output_dir}")
            
        # saved_modelサブディレクトリも事前に作成
        saved_model_dir = os.path.join(new_output_dir, "saved_model")
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)
            
        # 処理結果の出力先を新しいディレクトリに設定
        output_dir = new_output_dir
    elif output_dir:
        print(f"既存の出力ディレクトリを使用: {output_dir}")
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"出力ディレクトリを作成しました: {output_dir}")
    else:
        print("出力ディレクトリが指定されていません")
    
    # ============================================
    # 3. dfとモデル情報の表示
    # ============================================
    
    print(f"\n=== モデル・SMOTE設定 ===")
    # モデル名の表示
    if use_bagging:
        base_model = kwargs.get('base_model', 'lightgbm')
        n_bags = kwargs.get('n_bags', 10)
        print(f"モデル: RegressionBaggingModel")
        print(f"  ベースモデル: {base_model}")
        print(f"  バッグ数: {n_bags}")
    elif model_class is not None:
        try:
            model_name = model_class().__class__.__name__
        except:
            model_name = "回帰モデル"
        print(f"モデル: {model_name}")
    else:
        raise ValueError("model_classまたはuse_baggingのいずれかを指定してください。")
    
    if use_smote:
        print(f"\nSMOTE設定:")
        print(f"  手法: {smote_method}")
        print(f"  パラメータ: {smote_kwargs}")
        print(f"  整数値対応: {'有効' if use_integer_smote else '無効'}")
        print(f"  目的変数範囲: {target_min}-{target_max}")
        print(f"  ※SMOTEは訓練データのみに適用されます（データリーク防止）")
    else:
        print("SMOTE: 使用しない")
    
    # ============================================
    # 4. dfの前処理（SMOTE適用なし）
    # ============================================
    
    print(f"\n=== データ前処理 ===")
    print(f"元のdf形状: {df.shape}")
    
    # SMOTEを適用しない通常の前処理のみ実行
    try:
        features, target, id_column = prepare_data_for_regression(df, target_column=target_column)
    except Exception as e:
        print(f"前処理中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    print(f"前処理結果:")
    print(f"  特徴量形状: {features.shape}")
    print(f"  ターゲット形状: {target.shape}")
    if id_column is not None:
        print(f"  ID列: {id_column.name} ({len(id_column)}件)")
    
    # ============================================
    # 5. 前処理後のデータチェック
    # ============================================
    
    # データが空の場合はエラー
    if len(features) == 0 or len(target) == 0:
        print(f"\nエラー: 前処理後のデータが空です")
        print(f"元のdfサイズ: {df.shape}")
        print(f"前処理後のfeatures: {features.shape}")
        print(f"前処理後のtarget: {target.shape}")
        raise ValueError("前処理後のデータが空です。データを確認してください。")
    
    # 前処理の結果をログ出力
    print(f"\n=== 前処理完了 ===")
    print(f"最終的な特徴量数: {features.shape[1]}")
    print(f"最終的なサンプル数: {features.shape[0]}")
    print(f"特徴量名: {features.columns.tolist()}")
    print(f"削除されたサンプル数: {len(df) - len(features)}")
    print(f"削除率: {((len(df) - len(features)) / len(df) * 100):.2f}%")
    
    # 目的変数の最終的な分布
    print(f"\n前処理後の目的変数統計:")
    print(target.describe())
    
    # ============================================
    # 6. 可視化準備（dfから生成されたデータを使用）
    # ============================================
    
    print(f"\n=== 可視化準備 ===")
    # 可視化のためのインスタンス作成 - 出力ディレクトリを使用
    visualizer = EyeTrackingVisualizer(output_dir=output_dir)  # 修正：args.output_dir → output_dir
    
    # 特徴量と目的変数の関係の可視化（前処理済みデータを使用）
    print("特徴量と目的変数の関係を分析中...")
    try:
        visualizer.plot_feature_correlations(features, target)
        print("特徴量相関プロットを生成しました")
    except Exception as e:
        print(f"特徴量相関プロットの生成中にエラー: {e}")
    
    # ============================================
    # 7. モデル学習（SMOTE適用は学習関数内で実行）
    # ============================================
    
    print(f"\n=== モデル学習開始 ===")
    print(f"学習用データ形状: features={features.shape}, target={target.shape}")
    if use_smote:
        print("注意: SMOTEが有効な場合、訓練データのみに適用されます")
    
    try:
        # 修正版のtrain_regression_model_with_smoteを呼び出し
        results = train_regression_model_with_smote(
            features, target, id_column,
            model_class=model_class, 
            use_bagging=use_bagging,
            random_state=random_state,
            # SMOTE関連パラメータを追加
            use_smote=use_smote,
            smote_method=smote_method,
            smote_kwargs=smote_kwargs,
            use_integer_smote=use_integer_smote,
            target_min=target_min,
            target_max=target_max,
            **kwargs
        )
    except Exception as e:
        print(f"学習中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # ============================================
    # 8. モデル名の決定とログ出力
    # ============================================
    
    # モデル名の決定
    if use_bagging:
        model_name = f"reg_bagging_{kwargs.get('base_model', 'lightgbm')}"
    else:
        try:
            model_name = results['model'].__class__.__name__.lower()
        except:
            model_name = "regression_model"
    
    # SMOTE使用時はモデル名に追加
    if use_smote:
        model_name += f"_smote_{smote_method}"
    
    print(f"\n=== モデル学習完了 ===")
    print(f"最終モデル名: {model_name}")
    
    # ============================================
    # 9. 結果の評価（dfから派生したデータを使用）
    # ============================================
    
    print(f"\n=== 結果評価 ===")
    try:
        predictions_df, metrics = evaluate_regression_results(results)
        print("評価完了")
    except Exception as e:
        print(f"評価中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        raise
    
    # 結果の表示
    print(f"\n最適なパラメータ:")
    for param, value in results['best_params'].items():
        print(f"  {param}: {value}")
    
    if use_smote:
        print(f"\nSMOTE設定:")
        print(f"  手法: {smote_method}")
        for key, value in smote_kwargs.items():
            print(f"  {key}: {value}")
        
        # SMOTE適用結果の表示
        if 'smote_applied' in results and results['smote_applied']:
            print(f"\nSMOTE適用結果:")
            print(f"  元の訓練データ数: {results['original_training_size']}")
            print(f"  SMOTE後訓練データ数: {results['training_data_size']}")
            print(f"  合成データ数: {results['training_data_size'] - results['original_training_size']}")
            print(f"  データ増加率: {((results['training_data_size'] - results['original_training_size']) / results['original_training_size'] * 100):.2f}%")
            print(f"  テストデータ: 元データのまま評価")
        else:
            print("\nSMOTE: 適用されませんでした")
        
        # 元のdfとの比較情報
        print(f"\n元のdf情報:")
        print(f"  元データ数: {len(df)}")
        print(f"  前処理後データ数: {len(features)}")
    
    print(f"\n性能指標:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"  {metric}: {value:.4f}")
        else:
            print(f"  {metric}: {value}")
    
    print(f"\n特徴量の重要度（上位10件）:")
    if results['feature_importance'] is not None and len(results['feature_importance']) > 0:
        print(results['feature_importance'].head(10).to_string())
    else:
        print("  特徴量重要度が取得できませんでした")
    
    # ============================================
    # 10. 結果の可視化
    # ============================================
    
    print(f"\n=== 結果可視化 ===")
    try:
        print("モデル評価の可視化を実行中...")
        visualizer.plot_true_vs_predicted(results['true_values'], results['predictions'])
        print("  真値vs予測値プロット: 完了")
        
        visualizer.plot_residuals(results['true_values'], results['predictions'])
        print("  残差プロット: 完了")
        
        visualizer.plot_feature_importance(results['feature_importance'])
        print("  特徴量重要度プロット: 完了")
        
        visualizer.plot_residual_distribution(results['true_values'], results['predictions'])
        print("  残差分布プロット: 完了")
    except Exception as e:
        print(f"可視化中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
# ============================================
# 11. 安全な部分依存プロット・ALE統合実装
# ============================================

# ===== クラス定義部分 =====


    class SafePartialDependencePlotter:
        """エラー処理を強化した部分依存プロッター"""
        
        def __init__(self, estimator, X, feature_names=None):
            self.estimator = estimator
            self.X = X
            if feature_names is not None:
                self.feature_names = feature_names
            elif hasattr(X, 'columns'):
                self.feature_names = X.columns.tolist()
            else:
                self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
            
            print(f"✓ 安全PDP初期化: {len(self.feature_names)}特徴量")
        
        def plot_single_feature(self, feature_name, grid_resolution=50, ax=None):
            """安全な単一特徴量部分依存プロット"""
            
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            else:
                fig = ax.figure
            
            # 特徴量インデックスを取得
            if feature_name in self.feature_names:
                feature_idx = self.feature_names.index(feature_name)
            else:
                ax.text(0.5, 0.5, f"特徴量 '{feature_name}' が見つかりません", 
                    ha='center', va='center', transform=ax.transAxes)
                return fig
            
            try:
                # 手動で部分依存を計算（より安全）
                pd_result = partial_dependence(
                    self.estimator,
                    self.X,
                    features=[feature_idx],
                    grid_resolution=grid_resolution,
                    kind='average'
                )
                
                values = pd_result['values'][0]
                grid = pd_result['grid'][0]
                
                # プロット
                ax.plot(grid, values, 'b-', linewidth=2, label='部分依存')
                ax.set_xlabel(feature_name)
                ax.set_ylabel('Partial Dependence')
                ax.set_title(f"部分依存プロット: {feature_name}")
                ax.grid(True, alpha=0.3)
                ax.legend()
                
                print(f"✓ PDP生成成功: {feature_name}")
                
            except Exception as e:
                print(f"✗ PDP生成エラー ({feature_name}): {e}")
                ax.text(0.5, 0.5, f"プロット生成失敗\n{feature_name}\n\nエラー: {str(e)[:50]}...", 
                    ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f"部分依存プロット: {feature_name} (エラー)")
            
            return fig
        
        def plot_interaction(self, feature1, feature2, grid_resolution=25):
            """安全な特徴量相互作用プロット"""
            
            fig, ax = plt.subplots(figsize=(10, 8))
            
            try:
                idx1 = self.feature_names.index(feature1)
                idx2 = self.feature_names.index(feature2)
            except ValueError as e:
                ax.text(0.5, 0.5, f"特徴量が見つかりません\n{str(e)}", 
                    ha='center', va='center', transform=ax.transAxes)
                return fig
            
            try:
                # 部分依存値を計算
                pd_result = partial_dependence(
                    self.estimator,
                    self.X,
                    features=[[idx1, idx2]],
                    grid_resolution=grid_resolution,
                    kind='average'
                )
                
                values = pd_result['values'][0]
                grid1, grid2 = pd_result['grid']
                
                # メッシュグリッド作成
                X_grid, Y_grid = np.meshgrid(grid1, grid2)
                
                # 安全なプロット（pcolormeshを使用）
                im = ax.pcolormesh(X_grid, Y_grid, values, shading='auto', cmap='viridis')
                plt.colorbar(im, ax=ax, label='Partial Dependence')
                
                # 等高線を追加（エラーが出ても続行）
                try:
                    contours = ax.contour(X_grid, Y_grid, values, levels=8, colors='white', alpha=0.6, linewidths=0.8)
                    ax.clabel(contours, inline=True, fontsize=8)
                except:
                    pass
                
                ax.set_xlabel(feature1)
                ax.set_ylabel(feature2)
                ax.set_title(f"特徴量相互作用: {feature1} × {feature2}")
                
                print(f"✓ 相互作用プロット生成成功: {feature1} × {feature2}")
                
            except Exception as e:
                print(f"✗ 相互作用プロット生成エラー: {e}")
                ax.text(0.5, 0.5, f"相互作用プロット生成失敗\n{feature1} × {feature2}\n\nエラー: {str(e)[:100]}...", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=10)
                ax.set_title(f"相互作用プロット: {feature1} × {feature2} (エラー)")
            
            return fig


    class AccumulatedLocalEffects:
        """Accumulated Local Effects Plot (ALE) の実装"""
        
        def __init__(self, estimator, X):
            self.estimator = estimator
            if isinstance(X, pd.DataFrame):
                self.X = X.values
                self.feature_names = X.columns.tolist()
            else:
                self.X = X
                self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
            
            print(f"✓ ALE初期化: {len(self.feature_names)}特徴量, {len(self.X)}サンプル")
        
        def _predict_average(self, X_subset, j, value):
            """特定の特徴量を固定値に設定した場合の平均予測値を計算"""
            if len(X_subset) == 0:
                return 0.0
            
            X_modified = X_subset.copy()
            X_modified[:, j] = value
            
            try:
                predictions = self.estimator.predict(X_modified)
                return np.mean(predictions)
            except Exception as e:
                print(f"予測エラー: {e}")
                return 0.0
        
        def calculate_ale(self, feature_idx, n_grid=30):
            """特定の特徴量のALEを計算"""
            print(f"特徴量{feature_idx}({self.feature_names[feature_idx]})のALE計算中...")
            
            # 特徴量の値を取得
            feature_values = self.X[:, feature_idx]
            
            # 無効値のチェック
            valid_mask = np.isfinite(feature_values)
            if not np.any(valid_mask):
                print(f"エラー: 特徴量{feature_idx}に有効な値がありません")
                return np.array([]), np.array([])
            
            # 分位点の計算
            try:
                quantiles = np.linspace(0, 1, n_grid + 1)
                xjks = np.quantile(feature_values[valid_mask], quantiles)
                xjks = np.unique(xjks)
                
                if len(xjks) < 2:
                    print(f"エラー: 特徴量{feature_idx}の分位点が不十分")
                    return np.array([]), np.array([])
                
                n_grid = len(xjks) - 1
                
            except Exception as e:
                print(f"分位点計算エラー: {e}")
                return np.array([]), np.array([])
            
            # 区間ごとの効果計算
            local_effects = np.zeros(n_grid)
            
            for k in range(1, n_grid + 1):
                if k == 1:
                    mask = (self.X[:, feature_idx] >= xjks[k-1]) & (self.X[:, feature_idx] <= xjks[k])
                else:
                    mask = (self.X[:, feature_idx] > xjks[k-1]) & (self.X[:, feature_idx] <= xjks[k])
                
                if np.sum(mask) > 0:
                    try:
                        upper_pred = self._predict_average(self.X[mask], feature_idx, xjks[k])
                        lower_pred = self._predict_average(self.X[mask], feature_idx, xjks[k-1])
                        local_effects[k-1] = upper_pred - lower_pred
                    except Exception as e:
                        local_effects[k-1] = 0
                else:
                    local_effects[k-1] = 0
            
            # 累積和とセンタリング
            accumulated_effects = np.cumsum(local_effects)
            accumulated_effects -= np.mean(accumulated_effects)
            
            # グリッドポイント
            grid_points = (xjks[:-1] + xjks[1:]) / 2
            
            print(f"✓ ALE計算完了: {len(grid_points)}点")
            return grid_points, accumulated_effects
        
        def plot_single_feature(self, feature_idx, n_grid=30, ax=None):
            """単一特徴量のALEプロット"""
            if ax is None:
                fig, ax = plt.subplots(figsize=(8, 6))
            else:
                fig = ax.figure
            
            # ALE計算
            grid_values, ale_values = self.calculate_ale(feature_idx, n_grid)
            
            if len(grid_values) == 0:
                ax.text(0.5, 0.5, f"ALE計算失敗\n{self.feature_names[feature_idx]}", 
                    ha='center', va='center', transform=ax.transAxes)
                return fig
        
            # プロット
            ax.plot(grid_values, ale_values, 'b-', linewidth=2, label='ALE')
            ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            
            # データ分布
            y_offset = -0.1 * (np.max(ale_values) - np.min(ale_values)) if len(ale_values) > 0 else -0.1
            ax.scatter(self.X[:, feature_idx], np.full(len(self.X), y_offset),
                    alpha=0.1, s=10, color='gray', label='Data')
            
            # ラベル
            ax.set_xlabel(self.feature_names[feature_idx])
            ax.set_ylabel("ALE")
            ax.set_title(f"ALE: {self.feature_names[feature_idx]}")
            ax.grid(True, alpha=0.3)
            ax.legend()
            
            return fig
        
        def plot_multiple_features(self, feature_indices, n_grid=30, n_cols=3, figsize=None):
            """複数特徴量のALEプロット"""
            n_features = len(feature_indices)
            if n_features == 0:
                fig, ax = plt.subplots(figsize=(8, 6))
                ax.text(0.5, 0.5, "表示する特徴量がありません", ha='center', va='center')
                return fig
            
            n_rows = (n_features + n_cols - 1) // n_cols
            if figsize is None:
                figsize = (5 * n_cols, 4 * n_rows)
            
            fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
            
            # axes処理
            if n_rows == 1 and n_cols == 1:
                axes = [axes]
            elif n_rows == 1:
                axes = axes.reshape(1, -1)
            elif n_cols == 1:
                axes = axes.reshape(-1, 1)
            
            # 各特徴量をプロット
            for idx, feature_idx in enumerate(feature_indices):
                if n_rows == 1 and n_cols > 1:
                    ax = axes[idx]
                elif n_cols == 1 and n_rows > 1:
                    ax = axes[idx]
                else:
                    row, col = idx // n_cols, idx % n_cols
                    ax = axes[row, col]
                
                try:
                    self.plot_single_feature(feature_idx, n_grid, ax=ax)
                except Exception as e:
                    ax.text(0.5, 0.5, f"エラー\n{str(e)}", ha='center', va='center', transform=ax.transAxes)
            
            # 余分なサブプロットを削除
            if n_rows > 1 and n_cols > 1:
                for idx in range(n_features, n_rows * n_cols):
                    row, col = idx // n_cols, idx % n_cols
                    fig.delaxes(axes[row, col])
            
            plt.tight_layout()
            return fig


        # ===== 実行部分 =====

# 重要度上位の特徴量を取得
    if results['feature_importance'] is not None and len(results['feature_importance']) > 0:
            top_features = results['feature_importance'].head(max(pdp_n_features, ale_n_features))['feature'].tolist()
    else:
            # 特徴量重要度がない場合は最初の特徴量を使用
        feature_names = results['X_train'].columns.tolist() if hasattr(results['X_train'], 'columns') else [f"Feature_{i}" for i in range(results['X_train'].shape[1])]
        top_features = feature_names[:max(pdp_n_features, ale_n_features)]

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    smote_suffix = f"_smote_{smote_method}" if use_smote else ""

    # ============================================
    # 11.1 部分依存プロット（PDP）生成
    # ============================================

    if generate_pdp:
        print(f"\n=== 安全な部分依存プロット生成 ===")
        print(f"使用特徴量数: {pdp_n_features}")
        print(f"グリッド解像度: {pdp_grid_resolution}")
                
        try:
                    # 安全な部分依存プロッター
            safe_plotter = SafePartialDependencePlotter(
                estimator=results['model'],
                X=results['X_train'],
                feature_names=results['X_train'].columns.tolist() if hasattr(results['X_train'], 'columns') else None
            )
                    
                    # PDP出力ディレクトリ
            pdp_output_dir = os.path.join(output_dir, 'pdp_plots') if output_dir else 'pdp_plots'
            os.makedirs(pdp_output_dir, exist_ok=True)
                    
                    # 統合プロット（複数特徴量を一つの図に）
            try:
                print("統合部分依存プロット生成中...")
                        
                        # 複数特徴量プロット用のmatplotlibサブプロット作成
                n_features_plot = min(pdp_n_features, len(top_features))
                n_cols = 3
                n_rows = (n_features_plot + n_cols - 1) // n_cols
                        
                fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 5*n_rows))
                if n_rows == 1:
                            axes = axes.reshape(1, -1) if n_cols > 1 else [axes]
                        
                for i, feature in enumerate(top_features[:n_features_plot]):
                    row, col = i // n_cols, i % n_cols
                    ax = axes[row, col] if n_rows > 1 else axes[col]
                    
                    try:
                        safe_plotter.plot_single_feature(feature, grid_resolution=pdp_grid_resolution, ax=ax)
                    except Exception as e:
                        ax.text(0.5, 0.5, f"エラー\n{feature}\n{str(e)[:30]}", 
                            ha='center', va='center', transform=ax.transAxes)
                
                # 余分なサブプロットを非表示
                for i in range(n_features_plot, n_rows * n_cols):
                    row, col = i // n_cols, i % n_cols
                    ax = axes[row, col] if n_rows > 1 else axes[col]
                    ax.set_visible(False)
                
                plt.suptitle(f'部分依存プロット (上位{n_features_plot}特徴量){smote_suffix}', fontsize=16)
                plt.tight_layout()
                
                multi_path = os.path.join(pdp_output_dir, f'pdp_top{n_features_plot}{smote_suffix}_{timestamp}.png')
                fig.savefig(multi_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ 統合PDP保存: {multi_path}")
                
            except Exception as e:
                print(f"✗ 統合PDP生成エラー: {e}")
            
            # 特徴量相互作用プロット
            if len(top_features) >= 2:
                try:
                    print("特徴量相互作用プロット生成中...")
                    fig = safe_plotter.plot_interaction(
                        top_features[0], 
                        top_features[1], 
                        grid_resolution=max(20, pdp_grid_resolution//2)
                    )
                    
                    interaction_path = os.path.join(pdp_output_dir, f'pdp_interaction{smote_suffix}_{timestamp}.png')
                    fig.savefig(interaction_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"✓ PDP相互作用保存: {interaction_path}")
                    
                except Exception as e:
                    print(f"✗ PDP相互作用エラー: {e}")
                    
        except Exception as e:
            print(f"PDPプロット生成中にエラー: {e}")
            import traceback
            traceback.print_exc()

    # ============================================
    # 11.2 ALE（Accumulated Local Effects）生成
    # ============================================

    if generate_ale:
        print(f"\n=== ALE (Accumulated Local Effects) プロット生成 ===")
        print(f"使用特徴量数: {ale_n_features}")
        print(f"グリッド解像度: {ale_grid_resolution}")
        
        try:
            # ALE初期化
            ale = AccumulatedLocalEffects(results['model'], results['X_train'])
            
            # ALE出力ディレクトリ
            ale_output_dir = os.path.join(output_dir, 'ale_plots') if output_dir else 'ale_plots'
            os.makedirs(ale_output_dir, exist_ok=True)
            
            # 特徴量インデックスを取得
            feature_indices = []
            for feature in top_features[:ale_n_features]:
                if feature in ale.feature_names:
                    feature_indices.append(ale.feature_names.index(feature))
            
            # 複数特徴量のALEプロット
            if feature_indices:
                try:
                    fig = ale.plot_multiple_features(feature_indices, n_grid=ale_grid_resolution)
                    
                    multi_path = os.path.join(ale_output_dir, f'ale_top{len(feature_indices)}{smote_suffix}_{timestamp}.png')
                    fig.savefig(multi_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"✓ ALE統合保存: {multi_path}")
                    
                except Exception as e:
                    print(f"✗ ALE統合エラー: {e}")
            
            # 個別特徴量のALE（上位3つ）
            for i, feature_idx in enumerate(feature_indices[:3]):
                try:
                    fig = ale.plot_single_feature(feature_idx, n_grid=ale_grid_resolution)
                    
                    safe_name = ale.feature_names[feature_idx].replace('/', '_').replace('\\', '_')
                    single_path = os.path.join(ale_output_dir, f'ale_{safe_name}{smote_suffix}_{timestamp}.png')
                    fig.savefig(single_path, dpi=300, bbox_inches='tight')
                    plt.close(fig)
                    print(f"✓ ALE個別保存: {single_path}")
                    
                except Exception as e:
                    print(f"✗ ALE個別エラー ({feature_idx}): {e}")
                    
        except Exception as e:
            print(f"ALEプロット生成中にエラー: {e}")
            import traceback
            traceback.print_exc()

    print("\n✓ 安全なPDP・ALE生成処理完了")

        
    # ============================================
    # 12. 結果をまとめる
    # ============================================
    
    print(f"\n=== 結果まとめ ===")
    # 結果をまとめる
    results['predictions_df'] = predictions_df

    # 元のdfとの比較情報を結果に追加
    results['original_df_shape'] = df.shape
    results['preprocessed_features_shape'] = features.shape
    results['preprocessed_target_shape'] = target.shape if hasattr(target, 'shape') else (len(target),)

    # ============================================
    # 13. 結果の保存（dfに関する情報も含む）
    # ============================================
    
    if output_dir:
        print(f"\n=== 結果保存 ===")
        # 予測結果の保存
        predictions_path = os.path.join(output_dir, f'predictions_{model_name}.csv')
        try:
            predictions_df.to_csv(predictions_path, index=False)
            print(f"予測結果: {predictions_path}")
        except Exception as e:
            print(f"予測結果の保存中にエラーが発生しました: {e}")
        
        # 特徴量の重要度の保存
        importance_path = os.path.join(output_dir, f'feature_importance_{model_name}.csv')
        try:
            results['feature_importance'].to_csv(importance_path, index=False)
            print(f"特徴量重要度: {importance_path}")
        except Exception as e:
            print(f"特徴量重要度の保存中にエラーが発生しました: {e}")
        
        # 実行設定の保存（dfの情報も含む）
        config_path = os.path.join(output_dir, f'config_{model_name}.txt')
        try:
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write("=== 実行設定 ===\n")
                f.write(f"モデル: {model_name}\n")
                f.write(f"使用したデータファイル: {data_file_name}\n")
                f.write(f"目的変数: {target_column}\n")
                f.write(f"乱数シード: {random_state}\n")
                
                # dfの情報を追加
                f.write(f"\n=== 入力データ情報 ===\n")
                f.write(f"元のdfのサイズ: {df.shape}\n")
                f.write(f"元のdfのカラム: {df.columns.tolist()}\n")
                f.write(f"元のdfの目的変数統計:\n")
                if target_column in df.columns:
                    f.write(f"  平均: {df[target_column].mean():.4f}\n")
                    f.write(f"  標準偏差: {df[target_column].std():.4f}\n")
                    f.write(f"  最小値: {df[target_column].min():.4f}\n")
                    f.write(f"  最大値: {df[target_column].max():.4f}\n")
                
                # 前処理後の情報
                f.write(f"\n=== 前処理後のデータ情報 ===\n")
                f.write(f"前処理後の特徴量数: {features.shape[1]}\n")
                f.write(f"前処理後のサンプル数: {features.shape[0]}\n")
                f.write(f"使用された特徴量: {features.columns.tolist()}\n")
                
                if use_smote and 'smote_applied' in results and results['smote_applied']:
                    f.write(f"\n=== SMOTE設定 ===\n")
                    f.write(f"SMOTE手法: {smote_method}\n")
                    f.write(f"整数値対応: {'有効' if use_integer_smote else '無効'}\n")
                    f.write(f"目的変数範囲: {target_min}-{target_max}\n")
                    for key, value in smote_kwargs.items():
                        f.write(f"{key}: {value}\n")
                    
                    # SMOTEによるデータ増加情報
                    original_training_size = results['original_training_size']
                    training_size = results['training_data_size']
                    f.write(f"\n=== 訓練データの変化（SMOTE適用） ===\n")
                    f.write(f"元の訓練データ数: {original_training_size}\n")
                    f.write(f"SMOTE適用後: {training_size}\n")
                    f.write(f"合成データ数: {training_size - original_training_size}\n")
                    f.write(f"データ増加率: {((training_size - original_training_size) / original_training_size * 100):.2f}%\n")
                    f.write(f"テストデータ: 元データのまま評価\n")
                
                f.write(f"\n=== 最適パラメータ ===\n")
                for param, value in results['best_params'].items():
                    f.write(f"{param}: {value}\n")
                
                f.write(f"\n=== 性能指標 ===\n")
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        f.write(f"{metric}: {value:.4f}\n")
                    else:
                        f.write(f"{metric}: {value}\n")
                
                if generate_pdp:
                    f.write(f"\n=== 部分依存プロット設定 ===\n")
                    f.write(f"特徴量数: {pdp_n_features}\n")
                    f.write(f"グリッド解像度: {pdp_grid_resolution}\n")
                
                f.write(f"\n=== 実行日時 ===\n")
                f.write(f"{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"実行設定: {config_path}")
        except Exception as e:
            print(f"実行設定の保存中にエラーが発生しました: {e}")
    
    # ============================================
    # 14. モデルの保存（dfから生成された最終モデル）
    # ============================================
    
    if output_dir and data_file_name:
        print(f"\n=== モデル保存 ===")
        model_base_name = os.path.splitext(os.path.basename(data_file_name))[0]
        if use_smote:
            model_base_name += f"_smote_{smote_method}"
        saved_model_dir = os.path.join(output_dir, 'saved_model')
        
        try:
            # モデル保存時は前処理後のfeaturesを使用する
            save_regression_model(
                model=results['model'],
                features=features,  # 前処理後の特徴量を使用
                target=target,      # 前処理後のターゲットを使用
                scaler=results['scaler'],
                output_path=output_dir,
                model_name=model_base_name
            )
            print(f"モデル保存: {saved_model_dir}/{model_base_name}")
            
            # モデル保存時の情報もログ出力
            print(f"保存されたモデルの情報:")
            print(f"  学習データ数: {results['training_data_size']}")
            print(f"  特徴量数: {features.shape[1]}")
            print(f"  元のdfサイズ: {df.shape}")
            if use_smote and 'smote_applied' in results and results['smote_applied']:
                print(f"  SMOTE適用: あり（{smote_method}）")
                print(f"  元の訓練データ数: {results['original_training_size']}")
            
        except Exception as e:
            print(f"モデルの保存中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    # ============================================
    # 15. 結果ファイルの整理（オプション）
    # ============================================
    
    if organize_files and original_output_dir:
        print(f"\n=== ファイル整理 ===")
        try:
            organized_dir = organize_regression_result_files(data_file_name, original_output_dir)
            print(f"結果ファイルを整理しました: {organized_dir}")
        except Exception as e:
            print(f"結果ファイルの整理中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()

    # ============================================
    # 16. 最終的な結果レポート
    # ============================================
    
    print(f"\n{'='*50}")
    print(f"分析完了レポート")
    print(f"{'='*50}")
    print(f"分析が完了しました。")
    if output_dir:
        print(f"すべての結果は {output_dir} に保存されました。")
    
    # dfに関する統計情報
    print(f"\n=== データ処理統計 ===")
    print(f"入力データ（df）:")
    print(f"  - サイズ: {df.shape}")
    print(f"  - カラム数: {len(df.columns)}")
    print(f"  - 行数: {len(df)}")
    
    print(f"\n前処理後データ:")
    print(f"  - 特徴量数: {features.shape[1]}")
    print(f"  - サンプル数: {features.shape[0]}")
    print(f"  - 削除率: {(1 - features.shape[0] / len(df)) * 100:.2f}%")
    
    if use_smote and 'smote_applied' in results and results['smote_applied']:
        original_training_size = results['original_training_size']
        final_training_size = results['training_data_size']
        synthetic_size = final_training_size - original_training_size
        
        print(f"\nSMOTE効果（訓練データのみ）:")
        print(f"  - 元の訓練データサイズ: {original_training_size}")
        print(f"  - SMOTE後の訓練データサイズ: {final_training_size}")
        print(f"  - 合成サンプル数: {synthetic_size}")
        print(f"  - データ増加率: {(synthetic_size / original_training_size * 100):.2f}%")
        print(f"  - テストデータ: 元データのまま（データリーク防止）")
    
    print(f"\n最終性能指標:")
    for metric, value in metrics.items():
        if isinstance(value, float):
            print(f"   {metric}: {value:.4f}")
        else:
            print(f"   {metric}: {value}")
    
    # ============================================
    # 17. 関数の戻り値
    # ============================================
    
    # results辞書に追加情報を格納
    results['df_info'] = {
        'original_shape': df.shape,
        'original_columns': df.columns.tolist(),
        'final_features_shape': features.shape,
        'final_target_shape': target.shape if hasattr(target, 'shape') else (len(target),),
        'used_features': features.columns.tolist(),
        'smote_applied': use_smote and results.get('smote_applied', False),
        'smote_method': smote_method if use_smote else None,
        'data_increase_ratio': ((results['training_data_size'] - results['original_training_size']) / results['original_training_size'] * 100) if use_smote and 'smote_applied' in results and results['smote_applied'] else 0
    }
    
    # 詳細なメトリクスを結果に追加
    results['metrics'] = metrics
    
    print(f"\n{'='*50}")
    print(f"分析処理完了！")
    print(f"{'='*50}")
    
    # 必ず2つの値を返す
    return results, new_output_dir


def run_regression_cv_analysis(df, model_class=None, n_splits=5, use_bagging=False,
                            output_dir='result', random_state=42, target_column='target', **kwargs):
    """
    回帰問題用の交差検証を実行する関数
    """
    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    bagging_str = "（バギングあり）" if use_bagging else ""
    
    print(f"\n回帰モデルでの {n_splits}分割交差検証を開始...{bagging_str}")
    
    # データの前処理
    print("データの前処理を開始...")
    features, target , _ = prepare_data_for_regression(df, target_column=target_column)
    
    # データが空の場合はエラー
    if len(features) == 0 or len(target) == 0:
        raise ValueError("前処理後のデータが空です。データを確認してください。")
    
    # 可視化のためのインスタンス作成 - 修正：args.output_dir → output_dir
    visualizer = EyeTrackingVisualizer(output_dir=output_dir)
    
    # 特徴量と目的変数の関係の可視化
    print("\n特徴量と目的変数の関係を分析します...")
    visualizer.plot_feature_correlations(features, target)
    
    # RegressionBaggingModelを使用する場合
    if use_bagging:
        base_model = kwargs.get('base_model', 'lightgbm')
        n_bags = kwargs.get('n_bags', 10)
        print(f"RegressionBaggingModelを使用した交差検証を実行します（ベースモデル: {base_model}、バッグ数: {n_bags}）")
        
        # RegressionBaggingModelのクラスメソッドを使用
        oof_preds, scores = RegressionBaggingModel.run_cv(
            X=features, 
            y=target, 
            base_model=base_model,
            n_bags=n_bags,
            n_splits=n_splits,
            random_state=random_state
        )
        
        # 結果の表示
        print("\n交差検証の結果:")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
            
        # 結果を保存
        model_name = f"reg_bagging_{base_model}"
        cv_results_df = pd.DataFrame({
            'true': target,
            'pred': oof_preds
        })
        
        if output_dir:
            # 予測結果の保存
            cv_results_df.to_csv(f'{output_dir}/cv_{model_name}_{n_splits}fold_predictions.csv')
            
            # 評価指標の保存
            with open(f'{output_dir}/cv_{model_name}_{n_splits}fold_metrics.txt', 'w') as f:
                for metric, score in scores.items():
                    f.write(f"{metric}: {score:.4f}\n")
        
        # 予測値と実測値の散布図
        visualizer.plot_true_vs_predicted(target, oof_preds)
        
        return None, scores
    
    # 通常のモデルを使用する場合（交差検証）
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    print(f"通常の回帰モデル {model_class.__name__} を使用した交差検証を実行します")
    
    # 特徴量とターゲットの準備
    X = features.values
    y = target.values
    
    # 交差検証の設定
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 結果の格納用
    fold_metrics = []
    oof_predictions = np.zeros(len(y))
    feature_importances = []
    
    # 各分割でモデルを学習・評価
    for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        
        # データの分割
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # モデルの作成と学習
        model = model_class()
        model.fit(X_train_scaled, y_train)
        
        # 予測
        y_pred = model.predict(X_test_scaled)
        oof_predictions[test_idx] = y_pred
        
        # 評価指標の計算
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        pearson_corr, p_value = pearsonr(y_test, y_pred)
        spearman_corr = spearman_correlation(y_test, y_pred)

        # 結果の保存
        fold_metrics.append({
            'fold': fold + 1,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'pearson_corr': pearson_corr,
            'spearman_corr': spearman_corr 
        })
        
        # 特徴量重要度の保存（あれば）
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': features.columns,
                'importance': model.feature_importances_,
                'fold': fold + 1
            })
            feature_importances.append(importances)
        
        print(f"Fold {fold+1} - RMSE: {rmse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}")
    
    # 結果のまとめ
    metrics_df = pd.DataFrame(fold_metrics)
    
    # 平均指標の計算
    avg_metrics = {
        'mse': metrics_df['mse'].mean(),
        'rmse': metrics_df['rmse'].mean(),
        'mae': metrics_df['mae'].mean(),
        'r2': metrics_df['r2'].mean(),
        'pearson_corr': metrics_df['pearson_corr'].mean(),
         'spearman_corr': metrics_df['spearman_corr'].mean() 
    }
    
    # 標準偏差の計算
    std_metrics = {
        'mse_std': metrics_df['mse'].std(),
        'rmse_std': metrics_df['rmse'].std(),
        'mae_std': metrics_df['mae'].std(),
        'r2_std': metrics_df['r2'].std(),
        'pearson_corr_std': metrics_df['pearson_corr'].std(),
        'spearman_corr_std': metrics_df['spearman_corr'].std()  # スピアマン相関の標準偏差を追加
}

    
    # 平均特徴量重要度の計算（特徴量重要度がある場合）
    avg_importance = None
    if feature_importances:
        importances_df = pd.concat(feature_importances)
        avg_importance = importances_df.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
    
    # 結果の表示
    print("\n=== 交差検証の結果 ===")
    print("\n平均性能指標:")
    for metric, value in avg_metrics.items():
        std = std_metrics[f'{metric}_std']
        print(f"{metric}: {value:.4f} ± {std:.4f}")
    
    # 特徴量重要度の表示（あれば）
    if avg_importance is not None:
        print("\n特徴量の平均重要度（上位10件）:")
        print(avg_importance.head(10))
    
    # 予測値と実測値の可視化
    visualizer.plot_true_vs_predicted(y, oof_predictions)
    visualizer.plot_residuals(y, oof_predictions)
    
    # 結果の保存
    model_name = model_class.__name__.lower()
    if output_dir:
        # 予測結果の保存
        pd.DataFrame({
            'true': y,
            'pred': oof_predictions
        }).to_csv(f'{output_dir}/cv_{model_name}_{n_splits}fold_predictions.csv')
        
        # 評価指標の保存
        with open(f'{output_dir}/cv_{model_name}_{n_splits}fold_metrics.txt', 'w') as f:
            for metric, value in avg_metrics.items():
                std = std_metrics[f'{metric}_std']
                f.write(f"{metric}: {value:.4f} ± {std:.4f}\n")
        
        # 特徴量重要度の保存（あれば）
        if avg_importance is not None:
            avg_importance.to_csv(f'{output_dir}/cv_{model_name}_{n_splits}fold_importance.csv', index=False)
    
    # 結果のまとめ
    results = {
        'oof_predictions': oof_predictions,
        'metrics': avg_metrics,
        'metrics_std': std_metrics,
        'fold_metrics': metrics_df,
        'feature_importance': avg_importance
    }
    
    return results, avg_metrics
def run_regression_cv_analysis_with_smote(df, model_class=None, n_splits=5, use_bagging=False,
                                         output_dir='result', random_state=42, target_column='target',
                                         # SMOTE関連パラメータ
                                         use_smote=False, smote_method='density', smote_kwargs=None,
                                         use_integer_smote=True, target_min=9, target_max=30,
                                         **kwargs):
    """
    SMOTE対応の回帰問題用交差検証を実行する関数
    """
    
    from sklearn.model_selection import KFold
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.preprocessing import StandardScaler
    import numpy as np
    import pandas as pd
    import os
    
    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # SMOTE設定の表示
    smote_info = ""
    if use_smote:
        smote_info = f"（SMOTE: {smote_method}）"
        print(f"\n=== SMOTE対応交差検証開始 ===")
        print(f"SMOTE手法: {smote_method}")
        print(f"整数値対応: {'有効' if use_integer_smote else '無効'}")
        if use_integer_smote:
            print(f"目的変数範囲: {target_min}-{target_max}")
        print("※各フォールドで訓練データのみにSMOTE適用（データリーク防止）")
    else:
        print(f"\n=== 通常の交差検証開始 ===")
    
    bagging_str = "（バギングあり）" if use_bagging else ""
    print(f"回帰モデルでの {n_splits}分割交差検証{smote_info}{bagging_str}")
    
    # データの前処理
    print("データの前処理を開始...")
    features, target, id_column = prepare_data_for_regression(df, target_column=target_column)
    
    # データが空の場合はエラー
    if len(features) == 0 or len(target) == 0:
        raise ValueError("前処理後のデータが空です。データを確認してください。")
    
    # 可視化のためのインスタンス作成 - 修正：args.output_dir → output_dir
    visualizer = EyeTrackingVisualizer(output_dir=output_dir)
    
    # 特徴量と目的変数の関係の可視化
    print("\n特徴量と目的変数の関係を分析します...")
    visualizer.plot_feature_correlations(features, target)
    
    # SMOTE用パラメータの設定
    if smote_kwargs is None:
        smote_kwargs = {
            'binning': {'sampling_strategy': 'auto', 'n_bins': 10},
            'density': {'density_threshold': 0.3},
            'outliers': {'outlier_threshold': 0.15}
        }.get(smote_method, {})
    
    # 特徴量とターゲットの準備
    X = features.values
    y = target.values
    
    # 交差検証の設定（回帰なのでKFoldを使用）
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 結果の格納用
    fold_metrics = []
    oof_predictions = np.zeros(len(y))
    feature_importances = []
    smote_statistics = []  # SMOTE統計情報を格納
    
    print(f"\n交差検証開始（{n_splits}分割）...")
    
    # 各分割でモデルを学習・評価
    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        print(f"\n=== Fold {fold+1}/{n_splits} ===")
        
        # データの分割
        X_train, X_val = X[train_idx], X[val_idx]
        y_train, y_val = y[train_idx], y[val_idx]
        
        print(f"分割結果:")
        print(f"  訓練データ: {X_train.shape}")
        print(f"  検証データ: {X_val.shape}")
        print(f"  訓練ターゲット範囲: {y_train.min():.2f} - {y_train.max():.2f}")
        print(f"  検証ターゲット範囲: {y_val.min():.2f} - {y_val.max():.2f}")
        
        # 元の訓練データサイズを記録
        original_train_size = len(X_train)
        
        # SMOTE適用（訓練データのみ）
        smote_applied = False
        if use_smote:
            print(f"Fold {fold+1}: 訓練データにSMOTE（{smote_method}）を適用中...")
            try:
                # SMOTE instance の作成
                if use_integer_smote:
                    try:
                        from regression_smote import IntegerRegressionSMOTE
                        smote_instance = IntegerRegressionSMOTE(
                            method=smote_method, 
                            k_neighbors=5, 
                            random_state=random_state + fold,  # フォールドごとに異なるシード
                            target_min=target_min,
                            target_max=target_max
                        )
                        print(f"  整数値対応SMOTEを使用（範囲: {target_min}-{target_max}）")
                    except ImportError:
                        from regression_smotebackup import RegressionSMOTE
                        smote_instance = RegressionSMOTE(
                            method=smote_method, 
                            k_neighbors=5, 
                            random_state=random_state + fold
                        )
                        print(f"  通常のSMOTEを使用（整数値対応なし）")
                        use_integer_smote = False
                else:
                    from regression_smotebackup import RegressionSMOTE
                    smote_instance = RegressionSMOTE(
                        method=smote_method, 
                        k_neighbors=5, 
                        random_state=random_state + fold
                    )
                    print(f"  通常のSMOTEを使用")
                
                # SMOTE適用前の統計
                print(f"  SMOTE適用前:")
                print(f"    サンプル数: {len(X_train)}")
                print(f"    目的変数: 平均={y_train.mean():.2f}, 標準偏差={y_train.std():.2f}")
                
                # SMOTE実行
                X_train_resampled, y_train_resampled = smote_instance.fit_resample(
                    X_train, y_train, **smote_kwargs
                )
                
                # フォールバック処理：通常のSMOTEを使った場合の整数化
                if use_integer_smote and 'IntegerRegressionSMOTE' not in str(type(smote_instance)):
                    print("  合成データの整数化を実行中...")
                    n_original = len(y_train)
                    
                    # 合成されたサンプルのみ整数化
                    for i in range(n_original, len(y_train_resampled)):
                        y_train_resampled[i] = round(y_train_resampled[i])
                        y_train_resampled[i] = np.clip(y_train_resampled[i], target_min, target_max)
                
                # SMOTE適用後の統計
                X_train = X_train_resampled
                y_train = y_train_resampled
                smote_applied = True
                
                synthetic_count = len(X_train) - original_train_size
                increase_ratio = (synthetic_count / original_train_size) * 100
                
                print(f"  SMOTE適用後:")
                print(f"    サンプル数: {len(X_train)} (+{synthetic_count})")
                print(f"    データ増加率: {increase_ratio:.2f}%")
                print(f"    目的変数: 平均={y_train.mean():.2f}, 標準偏差={y_train.std():.2f}")
                
                # SMOTE統計を記録
                smote_stats = {
                    'fold': fold + 1,
                    'method': smote_method,
                    'original_size': original_train_size,
                    'resampled_size': len(X_train),
                    'synthetic_count': synthetic_count,
                    'increase_ratio': increase_ratio,
                    'original_target_mean': y[train_idx].mean(),
                    'original_target_std': y[train_idx].std(),
                    'resampled_target_mean': y_train.mean(),
                    'resampled_target_std': y_train.std()
                }
                smote_statistics.append(smote_stats)
                
            except Exception as e:
                print(f"  SMOTE適用中にエラーが発生: {e}")
                print("  SMOTEなしで処理を続行します...")
                smote_applied = False
                import traceback
                traceback.print_exc()
        
        # 検証データは元データのまま使用（重要：データリーク防止）
        print(f"  検証データ: 元データのまま使用（{len(X_val)}サンプル）")
        
        # スケーリング（SMOTE適用後の訓練データで学習、検証データに適用）
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_val_scaled = scaler.transform(X_val)
        
        # モデル学習
        print(f"  モデル学習中...")
        if use_bagging:
            base_model = kwargs.get('base_model', 'lightgbm')
            n_bags = kwargs.get('n_bags', 10)
            print(f"    RegressionBaggingModel使用（{base_model}, {n_bags}バッグ）")
            
            model = RegressionBaggingModel(
                base_model=base_model,
                n_bags=n_bags,
                random_state=random_state + fold
            )
            model.fit(X_train_scaled, y_train)
            
            # 特徴量重要度
            if hasattr(model, 'feature_importances_'):
                importances = pd.DataFrame({
                    'feature': features.columns,
                    'importance': model.feature_importances_,
                    'fold': fold + 1
                })
                feature_importances.append(importances)
            
        else:
            # 通常のモデル
            model = model_class()
            model.fit(X_train_scaled, y_train)
            
            # 特徴量重要度
            if hasattr(model, 'feature_importances_'):
                importances = pd.DataFrame({
                    'feature': features.columns,
                    'importance': model.feature_importances_,
                    'fold': fold + 1
                })
                feature_importances.append(importances)
        
        # 検証データで予測（元データで評価）
        y_pred = model.predict(X_val_scaled)
        oof_predictions[val_idx] = y_pred
        
        # 評価指標の計算（元の検証データで評価）
        mse = mean_squared_error(y_val, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val, y_pred)
        r2 = r2_score(y_val, y_pred)
        mape = mean_absolute_percentage_error(y_val, y_pred)
        pearson_corr, p_value = pearsonr(y_val, y_pred)
        spearman_corr = spearman_correlation(y_val, y_pred)
        
        # 結果の保存
        fold_result = {
            'fold': fold + 1,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'pearson_corr': pearson_corr,
            'p_value': p_value,
            'spearman_corr': spearman_corr,
            'smote_applied': smote_applied,
            'train_size_original': original_train_size,
            'train_size_final': len(X_train),
            'val_size': len(X_val)
        }
        fold_metrics.append(fold_result)
        
        print(f"  Fold {fold+1} 結果:")
        print(f"    RMSE: {rmse:.4f}")
        print(f"    MAE: {mae:.4f}")
        print(f"    R²: {r2:.4f}")
        print(f"    ピアソン相関: {pearson_corr:.4f}")
        print(f"    スピアマン相関: {spearman_corr:.4f}")
        if smote_applied:
            print(f"    SMOTE効果: +{len(X_train) - original_train_size}サンプル")
    
    # 結果のまとめ
    print(f"\n=== 交差検証結果のまとめ ===")
    metrics_df = pd.DataFrame(fold_metrics)
    
    # 平均指標の計算
    avg_metrics = {
        'mse': metrics_df['mse'].mean(),
        'rmse': metrics_df['rmse'].mean(),
        'mae': metrics_df['mae'].mean(),
        'mape': metrics_df['mape'].mean(),
        'r2': metrics_df['r2'].mean(),
        'pearson_corr': metrics_df['pearson_corr'].mean(),
        'spearman_corr': metrics_df['spearman_corr'].mean()
    }
    
    # 標準偏差の計算
    std_metrics = {
        'mse_std': metrics_df['mse'].std(),
        'rmse_std': metrics_df['rmse'].std(),
        'mae_std': metrics_df['mae'].std(),
        'mape_std': metrics_df['mape'].std(),
        'r2_std': metrics_df['r2'].std(),
        'pearson_corr_std': metrics_df['pearson_corr'].std(),
        'spearman_corr_std': metrics_df['spearman_corr'].std()
    }
    
    # 平均特徴量重要度の計算
    avg_importance = None
    if feature_importances:
        importances_df = pd.concat(feature_importances)
        avg_importance = importances_df.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
    
    # SMOTE統計のまとめ
    smote_summary = None
    if use_smote and smote_statistics:
        smote_df = pd.DataFrame(smote_statistics)
        smote_summary = {
            'total_folds': n_splits,
            'smote_applied_folds': len(smote_df),
            'avg_original_size': smote_df['original_size'].mean(),
            'avg_resampled_size': smote_df['resampled_size'].mean(),
            'avg_synthetic_count': smote_df['synthetic_count'].mean(),
            'avg_increase_ratio': smote_df['increase_ratio'].mean(),
            'method': smote_method,
            'parameters': smote_kwargs
        }
    
    # 結果の表示
    print("\n平均性能指標:")
    for metric, value in avg_metrics.items():
        std = std_metrics[f'{metric}_std']
        print(f"  {metric}: {value:.4f} ± {std:.4f}")
    
    if use_smote and smote_summary:
        print(f"\nSMOTE適用統計:")
        print(f"  手法: {smote_summary['method']}")
        print(f"  適用フォールド数: {smote_summary['smote_applied_folds']}/{smote_summary['total_folds']}")
        print(f"  平均元データ数: {smote_summary['avg_original_size']:.0f}")
        print(f"  平均リサンプル後データ数: {smote_summary['avg_resampled_size']:.0f}")
        print(f"  平均合成データ数: {smote_summary['avg_synthetic_count']:.0f}")
        print(f"  平均データ増加率: {smote_summary['avg_increase_ratio']:.2f}%")
        print(f"  パラメータ: {smote_summary['parameters']}")
    
    # 特徴量重要度の表示
    if avg_importance is not None:
        print("\n特徴量の平均重要度（上位10件）:")
        print(avg_importance.head(10).to_string(index=False))
    
    # 予測値と実測値の可視化
    print("\n可視化を実行中...")
    visualizer.plot_true_vs_predicted(y, oof_predictions)
    visualizer.plot_residuals(y, oof_predictions)
    
    # 結果の保存
    # 結果の保存
    if output_dir:
        # モデル名の決定（lambda関数対応）
        if use_bagging:
            model_name = f"reg_bagging_{kwargs.get('base_model', 'lightgbm')}"
        else:
            # kwargs経由でモデルタイプ名が渡されていればそれを使用
            if 'model_type_name' in kwargs:
                model_name = kwargs['model_type_name']
            else:
                # デフォルトでモデルタイプを推測
                model_name = kwargs.get('model', 'regression_model')
        
        smote_suffix = f"_smote_{smote_method}" if use_smote else ""
        file_prefix = f'cv_{model_name}{smote_suffix}_{n_splits}fold'
            
        print(f"\n結果をファイルに保存中...")
        
        # Out-of-fold予測結果の保存
        oof_df = pd.DataFrame({
            'true': y,
            'pred': oof_predictions,
            'residual': y - oof_predictions,
            'abs_residual': np.abs(y - oof_predictions)
        })
        
        # ID列がある場合は追加
        if id_column is not None:
            oof_df.insert(0, id_column.name, id_column.values)
        
        oof_path = os.path.join(output_dir, f'{file_prefix}_predictions.csv')
        oof_df.to_csv(oof_path, index=False)
        print(f"  Out-of-fold予測: {oof_path}")
        
        # 評価指標の保存
        metrics_path = os.path.join(output_dir, f'{file_prefix}_metrics.txt')
        with open(metrics_path, 'w', encoding='utf-8') as f:
            f.write("=== 交差検証結果 ===\n")
            f.write(f"分割数: {n_splits}\n")
            f.write(f"データ数: {len(features)}\n")
            f.write(f"特徴量数: {features.shape[1]}\n")
            
            if use_smote:
                f.write(f"\n=== SMOTE設定 ===\n")
                f.write(f"手法: {smote_method}\n")
                f.write(f"整数値対応: {'有効' if use_integer_smote else '無効'}\n")
                if use_integer_smote:
                    f.write(f"目的変数範囲: {target_min}-{target_max}\n")
                for key, value in smote_kwargs.items():
                    f.write(f"{key}: {value}\n")
                
                if smote_summary:
                    f.write(f"\n=== SMOTE統計 ===\n")
                    f.write(f"適用フォールド数: {smote_summary['smote_applied_folds']}/{smote_summary['total_folds']}\n")
                    f.write(f"平均元データ数: {smote_summary['avg_original_size']:.0f}\n")
                    f.write(f"平均リサンプル後データ数: {smote_summary['avg_resampled_size']:.0f}\n")
                    f.write(f"平均合成データ数: {smote_summary['avg_synthetic_count']:.0f}\n")
                    f.write(f"平均データ増加率: {smote_summary['avg_increase_ratio']:.2f}%\n")
            
            f.write(f"\n=== 平均性能指標 ===\n")
            for metric, value in avg_metrics.items():
                std = std_metrics[f'{metric}_std']
                f.write(f"{metric}: {value:.4f} ± {std:.4f}\n")
            
            f.write(f"\n=== 各フォールドの詳細 ===\n")
            for _, row in metrics_df.iterrows():
                f.write(f"Fold {row['fold']}:\n")
                f.write(f"  RMSE: {row['rmse']:.4f}\n")
                f.write(f"  MAE: {row['mae']:.4f}\n")
                f.write(f"  R²: {row['r2']:.4f}\n")
                f.write(f"  SMOTE適用: {'Yes' if row['smote_applied'] else 'No'}\n")
                if row['smote_applied']:
                    f.write(f"  訓練データ増加: {row['train_size_original']} → {row['train_size_final']}\n")
                f.write("\n")
        
        print(f"  評価指標: {metrics_path}")
        
        # 特徴量重要度の保存
        if avg_importance is not None:
            importance_path = os.path.join(output_dir, f'{file_prefix}_importance.csv')
            avg_importance.to_csv(importance_path, index=False)
            print(f"  特徴量重要度: {importance_path}")
        
        # SMOTE統計の保存
        if use_smote and smote_statistics:
            smote_stats_path = os.path.join(output_dir, f'{file_prefix}_smote_stats.csv')
            pd.DataFrame(smote_statistics).to_csv(smote_stats_path, index=False)
            print(f"  SMOTE統計: {smote_stats_path}")
    
    # 結果のまとめ
    results = {
        'oof_predictions': oof_predictions,
        'true_values': y,
        'metrics': avg_metrics,
        'metrics_std': std_metrics,
        'fold_metrics': metrics_df,
        'feature_importance': avg_importance,
        'smote_statistics': smote_statistics,
        'smote_summary': smote_summary,
        'features': features,
        'target': target
    }
    
    print(f"\n{'='*50}")
    print(f"SMOTE対応交差検証完了！")
    if use_smote:
        print(f"各フォールドで訓練データのみにSMOTE適用")
        print(f"検証データは元データのまま評価（データリーク防止）")
    print(f"{'='*50}")
    
    return results, avg_metrics
def create_regression_model(model_type='lightgbm', random_state=42):
    """回帰モデルを作成するヘルパー関数"""
    if model_type == 'lightgbm':
        import lightgbm as lgb
        return lgb.LGBMRegressor(objective='regression', random_state=random_state)
    elif model_type == 'xgboost':
        import xgboost as xgb
        return xgb.XGBRegressor(objective='reg:squarederror', random_state=random_state)
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(random_state=random_state)
    elif model_type == 'catboost':
        from catboost import CatBoostRegressor
        return CatBoostRegressor(loss_function='RMSE', random_seed=random_state, verbose=False)
    else:
        raise ValueError(f"サポートされていないモデルタイプ: {model_type}")


def create_regression_bagging_model(base_model='lightgbm', n_bags=10, random_state=42):
    """RegressionBaggingModelを作成するヘルパー関数"""
    return {
        'model': RegressionBaggingModel(
            base_model=base_model,
            n_bags=n_bags,
            random_state=random_state
        ),
        'name': f'reg_bagging_{base_model}',
        'params': {
            'base_model': base_model,
            'n_bags': n_bags
        }
    }

# メイン実行部分の修正（if __name__ == "__main__": 以下の部分）

if __name__ == "__main__":
    setup_matplotlib_japanese_font()
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='視線追跡データを用いた回帰モデルの実行')
    parser.add_argument('--cv', action='store_true', help='交差検証を実行する')
    parser.add_argument('--bagging', action='store_true', help='バギングモデルを使用する')
    parser.add_argument('--base-model', type=str, default='catboost', 
                        choices=['lightgbm', 'xgboost', 'random_forest', 'catboost'], 
                        help='バギングで使用するベースモデル')
    parser.add_argument('--n-bags', type=int, default=10, help='バギングのバッグ数')
    parser.add_argument('--splits', type=int, default=5, help='交差検証の分割数')
    parser.add_argument('--model', type=str, default='catboost', 
                        choices=['lightgbm', 'xgboost', 'random_forest', 'catboost'],
                        help='使用するモデル')
    parser.add_argument('--random-state', type=int, default=42, help='乱数シード')
    parser.add_argument('--data-path', type=str, 
                       default="data",
                       help='データファイルのパス')
    parser.add_argument('--data-file', type=str, 
                       default="minirocket長文字正規化点_実行.csv",
                       help='データファイル名')
    
    parser.add_argument('--target-column', type=str,
                       default="target",
                       help='目的変数のカラム名')
    parser.add_argument('--output-dir', type=str, 
                       default="result",
                       help='結果出力ディレクトリ')
    parser.add_argument('--no-save', dest='save_plots', action='store_false', help='プロットをファイルに保存しない')
    parser.add_argument('--no-organize', dest='organize_files', action='store_false', 
                        help='結果ファイルを整理しない')
    
    # SMOTE関連の引数を追加
    smote_group = parser.add_argument_group('SMOTE オプション', '回帰用SMOTEのパラメータ')
    smote_group.add_argument('--smote', action='store_true', 
                           help='回帰用SMOTEを使用する')
    smote_group.add_argument('--smote-method', type=str, default='density',        
                           choices=['binning', 'density', 'outliers'],
                           help='SMOTEの手法を選択 (default: density)')
    smote_group.add_argument('--smote-k-neighbors', type=int, default=5,
                           help='SMOTE用の近傍数 (default: 5)')
    smote_group.add_argument('--smote-n-bins', type=int, default=10,
                           help='binning手法でのビン数 (default: 10)')
    smote_group.add_argument('--smote-density-threshold', type=float, default=0.3,
                           help='density手法での密度閾値 (default: 0.3)')
    smote_group.add_argument('--smote-outlier-threshold', type=float, default=0.15,
                           help='outliers手法での外れ値閾値 (default: 0.15)')
    smote_group.add_argument('--integer-smote', action='store_true', default=True,
                       help='整数値対応SMOTEを使用する（MoCAスコア用）')
    smote_group.add_argument('--target-min', type=int, default=9,
                        help='目的変数の最小値 (default: 9)')
    smote_group.add_argument('--target-max', type=int, default=30,
                        help='目的変数の最大値 (default: 30)')
    
    # 部分依存プロット関連の引数
    pdp_group = parser.add_argument_group('部分依存プロット オプション', '部分依存プロットの生成に関するパラメータ')
    pdp_group.add_argument('--no-pdp', dest='generate_pdp', action='store_false', 
                          help='部分依存プロットを生成しない')
    pdp_group.add_argument('--pdp-n-features', type=int, default=6,
                          help='部分依存プロットで表示する特徴量数 (default: 6)')
    pdp_group.add_argument('--pdp-grid-resolution', type=int, default=50,
                          help='部分依存プロットのグリッド解像度 (default: 50)')
    
    
    
    
    ale_group = parser.add_argument_group('ALE オプション', 'Accumulated Local Effectsプロットの生成に関するパラメータ')
    ale_group.add_argument('--no-ale', dest='generate_ale', action='store_false', 
                        help='ALEプロットを生成しない')
    ale_group.add_argument('--ale-n-features', type=int, default=6,
                        help='ALEプロットで表示する特徴量数 (default: 6)')
    ale_group.add_argument('--ale-grid-resolution', type=int, default=30,
                        help='ALEプロットのグリッド解像度 (default: 30)')
    ale_group.add_argument('--compare-ale-pdp', action='store_true', default=True,
                        help='ALEとPDPの比較プロットを生成する')
    # parser.set_defaultsにgenerate_ale=Trueを追加
    parser.set_defaults(bagging=False, save_plots=True, organize_files=True, generate_pdp=True, generate_ale=True)
    args = parser.parse_args()
    # データの読み込み
    data_file_path = os.path.join(args.data_path, args.data_file)
    print(f"データファイルを読み込みます: {data_file_path}")
    try:
        df = pd.read_csv(data_file_path)
    except Exception as e:
        print(f"データファイルの読み込みに失敗しました: {e}")
        print("ファイルパスと形式を確認してください。")
        exit(1)
    
    # SMOTEのパラメータ設定
    smote_kwargs = {}
    if args.smote:
        if args.smote_method == 'binning':
            smote_kwargs = {
                'sampling_strategy': 'auto',
                'n_bins': args.smote_n_bins
            }
        elif args.smote_method == 'density':
            smote_kwargs = {
                'density_threshold': args.smote_density_threshold
            }
        elif args.smote_method == 'outliers':
            smote_kwargs = {
                'outlier_threshold': args.smote_outlier_threshold
            }
    
    # モデルの選択
    model_class = None
    if not args.bagging:
        if args.model == 'lightgbm':
            import lightgbm as lgb
            model_class = lambda: lgb.LGBMRegressor(objective='regression', random_state=args.random_state)
        elif args.model == 'xgboost':
            import xgboost as xgb
            model_class = lambda: xgb.XGBRegressor(objective='reg:squarederror', random_state=args.random_state)
        elif args.model == 'random_forest':
            from sklearn.ensemble import RandomForestRegressor
            model_class = lambda: RandomForestRegressor(random_state=args.random_state)
        elif args.model == 'catboost':
            from catboost import CatBoostRegressor
            model_class = lambda: CatBoostRegressor(loss_function='RMSE', random_seed=args.random_state, verbose=False)
        
        model_name = args.model
    else:
        model_class = None
        model_name = f"reg_bagging_{args.base_model}"
    
    # 実行設定の表示
    print(f"実行設定:")
    if args.bagging:
        print(f"- モデル: RegressionBagging ({args.base_model})")
        print(f"- バッグ数: {args.n_bags}")
    else:
        print(f"- モデル: {args.model}")
    print(f"- 交差検証: {'あり' if args.cv else 'なし'}")
    if args.cv:
        print(f"- 交差検証分割数: {args.splits}")
    print(f"- 乱数シード: {args.random_state}")
    print(f"- データファイル: {args.data_file}")
    print(f"- 目的変数: {args.target_column}")
    print(f"- 結果ファイル整理: {'あり' if args.organize_files else 'なし'}")
    
    # SMOTE設定の表示
    if args.smote:
        print(f"- SMOTE: あり ({args.smote_method})")
        print(f"- SMOTE近傍数: {args.smote_k_neighbors}")
        print(f"- 整数値対応: {'あり' if args.integer_smote else 'なし'}")
        if args.integer_smote:
            print(f"- 目的変数範囲: {args.target_min}-{args.target_max}")
        if args.smote_method == 'binning':
            print(f"- SMOTEビン数: {args.smote_n_bins}")
        elif args.smote_method == 'density':
            print(f"- SMOTE密度閾値: {args.smote_density_threshold}")
        elif args.smote_method == 'outliers':
            print(f"- SMOTE外れ値閾値: {args.smote_outlier_threshold}")
        
        if args.cv:
            print(f"- 交差検証でのSMOTE適用: 各フォールドの訓練データのみ（データリーク防止）")
    else:
        print(f"- SMOTE: なし")
    
    # 部分依存プロット設定の表示
    if not args.cv:  # 単一モデル時のみ
        print(f"- 部分依存プロット: {'あり' if args.generate_pdp else 'なし'}")
        if args.generate_pdp:
            print(f"- PDP特徴量数: {args.pdp_n_features}")
            print(f"- PDPグリッド解像度: {args.pdp_grid_resolution}")
    if not args.cv:  # 単一モデル時のみ
        print(f"- ALE: {'あり' if args.generate_ale else 'なし'}")
        if args.generate_ale:
            print(f"- ALE特徴量数: {args.ale_n_features}")
            print(f"- ALEグリッド解像度: {args.ale_grid_resolution}")
            print(f"- ALE-PDP比較: {'あり' if args.compare_ale_pdp else 'なし'}")
    try:
        # 分析実行
        if args.cv:
            # **修正**: SMOTE対応交差検証関数を使用
            print("\n=== SMOTE対応交差検証を実行します ===")
            
            results, cv_metrics = run_regression_cv_analysis_with_smote(
                df, 
                model_class=model_class if not args.bagging else None, 
                n_splits=args.splits,
                use_bagging=args.bagging,
                output_dir=args.output_dir,
                random_state=args.random_state,
                target_column=args.target_column,
                use_smote=args.smote,
                smote_method=args.smote_method,
                smote_kwargs=smote_kwargs,
                use_integer_smote=args.integer_smote,
                target_min=args.target_min,
                target_max=args.target_max,
                base_model=args.base_model,
                n_bags=args.n_bags
            )
            
            # 結果ファイルの整理
            if args.organize_files and args.output_dir:
                new_dir = organize_regression_result_files(args.data_file, args.output_dir)
                print(f"\nSMOTE対応交差検証による分析が完了しました。結果は {new_dir} に保存されています。")
            else:
                print(f"\nSMOTE対応交差検証による分析が完了しました。")
                if args.output_dir:
                    print(f"結果は {args.output_dir} ディレクトリに保存されました。")
            
            # SMOTE効果のサマリー表示
            if args.smote and 'smote_summary' in results and results['smote_summary']:
                smote_summary = results['smote_summary']
                print(f"\n=== SMOTE効果サマリー ===")
                print(f"手法: {smote_summary['method']}")
                print(f"パラメータ: {smote_summary['parameters']}")
                print(f"適用フォールド数: {smote_summary['smote_applied_folds']}/{smote_summary['total_folds']}")
                print(f"平均データ増加率: {smote_summary['avg_increase_ratio']:.2f}%")
                print(f"平均合成データ数: {smote_summary['avg_synthetic_count']:.0f}")
                print("各フォールドで訓練データのみに適用（データリーク防止）")
        
        else:
            # 単一の学習・評価による分析（既存のコード）
            results, new_dir = run_regression_analysis(
                df, 
                model_class=model_class if not args.bagging else None,
                use_bagging=args.bagging,
                random_state=args.random_state,
                output_dir=args.output_dir,
                target_column=args.target_column,
                data_file_name=args.data_file,
                organize_files=args.organize_files,
                base_model=args.base_model,
                n_bags=args.n_bags,
                # SMOTE関連のパラメータを渡す
                use_smote=args.smote,
                smote_method=args.smote_method,
                smote_kwargs=smote_kwargs,
                # 整数値対応 SMOTE のパラメータ
                use_integer_smote=args.integer_smote,
                target_min=args.target_min,
                target_max=args.target_max,
                # 部分依存プロット関連のパラメータ
                generate_pdp=args.generate_pdp,
                pdp_n_features=args.pdp_n_features,
                pdp_grid_resolution=args.pdp_grid_resolution,
                # ALE関連のパラメータを追加
                generate_ale=args.generate_ale,
                ale_n_features=args.ale_n_features,
                ale_grid_resolution=args.ale_grid_resolution,
                compare_ale_pdp=args.compare_ale_pdp
            )
            
            if new_dir:
                print(f"\n単一モデルによる分析が完了しました。結果は {new_dir} に保存されています。")
            else:
                print("\n単一モデルによる分析が完了しました。")
                if args.output_dir:
                    print(f"結果は {args.output_dir} ディレクトリに保存されました。")
            
            # SMOTEのサマリー表示
            if args.smote:
                print(f"\n=== SMOTE適用サマリー ===")
                print(f"手法: {args.smote_method}")
                print(f"パラメータ: {smote_kwargs}")
                if 'df_info' in results:
                    df_info = results['df_info']
                    print(f"元データサイズ: {df_info['original_shape']}")
                    print(f"最終データサイズ: {df_info['final_features_shape']}")
                    print(f"データ増加率: {df_info['data_increase_ratio']:.2f}%")
    
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        print("\n実行を中断します。エラーの詳細を確認し、設定を見直してください。")