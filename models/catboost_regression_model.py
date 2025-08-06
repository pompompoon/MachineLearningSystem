import numpy as np
import pandas as pd
from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import GridSearchCV, KFold
import matplotlib.pyplot as plt
import re
import unicodedata

class CatBoostRegressionModel:
    """
    CatBoost model class for regression tasks
    
    This class implements CatBoost for regression, which is efficient
    for categorical variables and can handle missing values automatically.
    """
    
    def __init__(self, random_state=42):
        """
        Initialize the model
        
        Parameters:
        -----------
        random_state : int, default=42
            Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.best_params = None
        self.feature_importances_ = None  # mainコードとの連携のために追加
    
    def _sanitize_feature_names(self, feature_names):
        if feature_names is None:
            return None
            
        # より強力なサニタイズ処理
        sanitized_names = []
        for name in feature_names:
            try:
                # Unicode正規化を適用
                normalized = unicodedata.normalize('NFKD', str(name))
                # ASCII文字のみに制限
                sanitized = ''.join(c for c in normalized if ord(c) < 128)
                # 英数字とアンダースコアのみに制限
                sanitized = re.sub(r'[^\w]+', '_', sanitized)
                # 空文字列の場合はプレースホルダー
                if not sanitized:
                    sanitized = f"feature_{len(sanitized_names)}"
                # 数字で始まる場合はプレフィックスを追加
                if sanitized[0].isdigit():
                    sanitized = f"f_{sanitized}"
                # 重複を避けるため必要に応じて連番を付与
                if sanitized in sanitized_names:
                    i = 1
                    while f"{sanitized}_{i}" in sanitized_names:
                        i += 1
                    sanitized = f"{sanitized}_{i}"
                sanitized_names.append(sanitized)
            except Exception as e:
                # エラーが発生した場合は代替名を使用
                sanitized = f"feature_{len(sanitized_names)}"
                sanitized_names.append(sanitized)
                print(f"特徴量名 '{name}' のサニタイズ中にエラーが発生しました: {e}")
        
        return sanitized_names
    
    def get_model(self):
        """
        Return the model instance
        
        Returns:
        --------
        model : CatBoostRegressor
            CatBoostRegressor instance
        """
        if self.model is None:
            self.model = CatBoostRegressor(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                loss_function='RMSE',  # 回帰用に変更
                random_seed=self.random_state,
                verbose=False
            )
        return self.model
    def perform_grid_search_simplified(self, X_train, y_train, X_test, y_test, categorical_features=None):
        """
        単純化されたグリッドサーチを実行する
        """
        print("\nCatBoostモデルの単純化されたグリッドサーチを開始します...")
        
        # NumPy配列に変換（インデックスの問題を避けるため）
        X_train_np = X_train.values if isinstance(X_train, pd.DataFrame) else X_train
        y_train_np = y_train.values if isinstance(y_train, (pd.Series, pd.DataFrame)) else y_train
        X_test_np = X_test.values if isinstance(X_test, pd.DataFrame) else X_test
        y_test_np = y_test.values if isinstance(y_test, (pd.Series, pd.DataFrame)) else y_test
        
        # 少ないパラメータ候補でグリッドを作成
        param_combinations = [
            {'learning_rate': 0.05, 'depth': 3, 'iterations': 500},
            {'learning_rate': 0.05, 'depth': 5, 'iterations': 500},
            {'learning_rate': 0.05, 'depth': 6, 'iterations': 1000},
            {'learning_rate': 0.05, 'depth': 7, 'iterations': 1000}
        ]
        
        best_score = float('inf')  # RMSEを最小化
        best_params = None
        best_model = None
        
        # 直接パラメータの組み合わせをテスト
        for params in param_combinations:
            print(f"パラメータをテスト中: {params}")
            
            try:
                # モデルを初期化
                model = CatBoostRegressor(
                    loss_function='RMSE',
                    random_seed=self.random_state,
                    verbose=False,
                    **params
                )
                
                # 単純にトレーニング
                if categorical_features is not None:
                    model.fit(
                        X_train_np, y_train_np,
                        cat_features=categorical_features,
                        eval_set=(X_test_np, y_test_np),
                        early_stopping_rounds=20,
                        verbose=False
                    )
                else:
                    model.fit(
                        X_train_np, y_train_np,
                        eval_set=(X_test_np, y_test_np),
                        early_stopping_rounds=20,
                        verbose=False
                    )
                
                # テストデータで評価
                y_pred = model.predict(X_test_np)
                from sklearn.metrics import mean_squared_error
                rmse = np.sqrt(mean_squared_error(y_test_np, y_pred))
                
                print(f"RMSE: {rmse:.4f}")
                
                # より良いスコアなら更新
                if rmse < best_score:
                    best_score = rmse
                    best_params = params
                    best_model = model
                    print(f"新しい最良スコア: {best_score:.4f}")
            
            except Exception as e:
                print(f"エラーが発生しました: {e}")
                continue
        
        # 最良のモデルが見つからなかった場合
        if best_model is None:
            print("どのパラメータセットも機能しませんでした。デフォルトモデルを使用します。")
            best_model = self.get_model()
            best_model.fit(X_train_np, y_train_np)
            best_params = {
                'learning_rate': 0.05,
                'depth': 6,
                'iterations': 1000
            }
        
        self.model = best_model
        self.best_params = best_params
        self.feature_importances_ = best_model.get_feature_importance()
        
        print("\n最良のパラメータ:")
        for param, value in self.best_params.items():
            print(f"{param}: {value}")
        
        return self.best_params, self.model
    """
    def simple_grid_search(self, X_train, y_train, categorical_features=None):
        # より単純なGridSearchCVを使用してグリッドサーチを実行
        from sklearn.model_selection import GridSearchCV
        
        param_grid = {
            'learning_rate': [0.05, 0.1],
            'depth': [3, 5, 6],
            'iterations': [500, 1000]
        }
        
        # 基本モデルを作成
        base_model = CatBoostRegressor(
            loss_function='RMSE',
            random_seed=self.random_state,
            verbose=False
        )
        
        # グリッドサーチオブジェクトを作成
        grid = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=1
        )
        
        # cat_featuresパラメータのフィットを処理
        if categorical_features is not None:
            grid.fit(X_train, y_train, cat_features=categorical_features)
        else:
            grid.fit(X_train, y_train)
        
        return grid.best_params_, grid.best_estimator_
        """

    def fit(self, X, y, eval_set=None, cat_features=None, verbose=False):
        """
        Train the model - mainコードとの連携のために追加
        
        Parameters:
        -----------
        X : array-like
            Training features
        y : array-like
            Target variable
        eval_set : tuple, default=None
            Validation set for early stopping
        cat_features : list, default=None
            List of categorical feature indices
        verbose : bool, default=False
            Verbose output
            
        Returns:
        --------
        self : object
            Trained model instance
        """
        model = self.get_model()
        
        # カテゴリカル変数があれば指定
        fit_params = {}
        if cat_features is not None:
            fit_params['cat_features'] = cat_features
        
        # 評価セットがあれば指定
        if eval_set is not None:
            fit_params['eval_set'] = eval_set
        
        # モデルの学習
        model.fit(X, y, verbose=verbose, **fit_params)
        
        # 特徴量重要度を設定
        self.feature_importances_ = model.get_feature_importance()
        
        return self
    
    def predict(self, X):
        """
        Make predictions - mainコードとの連携のために追加
        
        Parameters:
        -----------
        X : array-like
            Features
            
        Returns:
        --------
        array
            Predicted values
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)
    
    def plot_feature_importance(self, feature_names=None, top_n=20, figsize=(12, 8)):
        """
        Plot feature importance
        
        Parameters:
        -----------
        feature_names : list, default=None
            List of feature names
        top_n : int, default=20
            Number of top features to display
        figsize : tuple, default=(12, 8)
            Plot size
        """
        if self.model is None:
            print("Model has not been trained yet.")
            return
        
        # Get feature importance
        feature_importance = self.model.get_feature_importance()
        
        # Sanitize feature names if provided
        if feature_names is not None:
            feature_names = self._sanitize_feature_names(feature_names)
        else:
            feature_names = [f"feature_{i}" for i in range(len(feature_importance))]
        
        # Select top N features
        indices = np.argsort(feature_importance)[-top_n:]
        top_features = [feature_names[i] for i in indices]
        top_importance = feature_importance[indices]
        
        # Create plot
        plt.figure(figsize=figsize)
        plt.barh(range(len(top_features)), top_importance, align='center')
        plt.yticks(range(len(top_features)), top_features)
        plt.title('CatBoost - Feature Importance (Top {})'.format(top_n))
        plt.xlabel('Importance')
        plt.tight_layout()
        plt.show()
    
    def plot_learning_curve(self):
        """Plot learning curve"""
        if self.model is None:
            print("Model has not been trained yet.")
            return
        
        try:
            evals_result = self.model.get_evals_result()
            
            # Create plot
            plt.figure(figsize=(12, 6))
            
            # Training loss curve
            if 'learn' in evals_result:
                plt.plot(evals_result['learn']['RMSE'], label='Training')  # 回帰用にRMSEに変更
            
            # Validation loss curve
            if 'validation' in evals_result:
                plt.plot(evals_result['validation']['RMSE'], label='Validation')  # 回帰用にRMSEに変更
            
            plt.title('CatBoost - Learning Curve')
            plt.xlabel('Iterations')
            plt.ylabel('RMSE')  # 回帰用にRMSEに変更
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting learning curve: {e}")