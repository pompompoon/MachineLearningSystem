import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import re
import unicodedata

class CatBoostModel:
    """
    CatBoost model class for classification tasks
    
    This class implements CatBoost for classification, which is efficient
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
        """
        Sanitize feature names to avoid encoding issues
        
        Parameters:
        -----------
        feature_names : list
            List of feature names
            
        Returns:
        --------
        list
            List of sanitized feature names
        """
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
                # エラーが発生した場合はフォールバック
                sanitized = f"feature_{len(sanitized_names)}"
                sanitized_names.append(sanitized)
                print(f"警告: 特徴量名 '{name}' のサニタイズ中にエラーが発生しました: {e}")
        
        return sanitized_names
    
    def get_model(self):
        """
        Return the model instance
        
        Returns:
        --------
        model : CatBoostClassifier
            CatBoostClassifier instance
        """
        if self.model is None:
            self.model = CatBoostClassifier(
                iterations=1000,
                learning_rate=0.05,
                depth=6,
                loss_function='Logloss',
                random_seed=self.random_state,
                verbose=False
            )
        return self.model
    
    def perform_grid_search(self, X_train, y_train, X_test, y_test, categorical_features=None):
        """
        Perform grid search to optimize hyperparameters
        
        Parameters:
        -----------
        X_train : array-like
            Training features
        y_train : array-like
            Training target variable
        X_test : array-like
            Test features
        y_test : array-like
            Test target variable
        categorical_features : list, default=None
            List of categorical feature indices
            
        Returns:
        --------
        best_params : dict
            Best parameters
        best_model : CatBoostClassifier
            Model trained with best parameters
        """
        print("\nStarting grid search for CatBoost model...")
        
        # データフレームへの変換を確実に行う
        if not isinstance(X_train, pd.DataFrame):
            X_train = pd.DataFrame(X_train)
        if not isinstance(X_test, pd.DataFrame):
            X_test = pd.DataFrame(X_test)
        
        # 特徴量名のサニタイズ
        print("Sanitizing feature names to prevent encoding issues...")
        original_columns = X_train.columns.tolist()
        sanitized_columns = self._sanitize_feature_names(original_columns)
        
        # サニタイズ後の特徴量名をデータフレームに適用
        X_train_safe = X_train.copy()
        X_train_safe.columns = sanitized_columns
        
        X_test_safe = X_test.copy()
        X_test_safe.columns = sanitized_columns
        
        # カテゴリカル特徴量のインデックスを調整（必要な場合）
        cat_features_safe = None
        if categorical_features is not None:
            if isinstance(categorical_features[0], str):
                # 名前でカテゴリカル特徴量が指定されている場合
                cat_features_safe = []
                for cat_feature in categorical_features:
                    if cat_feature in original_columns:
                        idx = original_columns.index(cat_feature)
                        cat_features_safe.append(sanitized_columns[idx])
            else:
                # インデックスでカテゴリカル特徴量が指定されている場合
                cat_features_safe = categorical_features
        
        # Parameter grid for grid search - 単純化した少数のパラメータセット
        param_grid = {
            'learning_rate': [0.05, 0.1],
            'depth': [3,5,6],
            'l2_leaf_reg': [3, 5],
            'iterations': [500, 1000]
        }
        
        # Cross-validation settings
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        try:
            # 手動グリッドサーチを実装（CatBoostClassifierのみを使用）
            print("Running manual grid search to avoid encoding issues...")
            best_score = 0
            best_params = {}
            
            for lr in param_grid['learning_rate']:
                for depth in param_grid['depth']:
                    for l2 in param_grid['l2_leaf_reg']:
                        for iters in param_grid['iterations']:
                            try:
                                # 現在のパラメータを設定
                                current_params = {
                                    'learning_rate': lr,
                                    'depth': depth,
                                    'l2_leaf_reg': l2,
                                    'iterations': iters,
                                    'loss_function': 'Logloss',
                                    'random_seed': self.random_state,
                                    'verbose': False
                                }
                                
                                print(f"Testing parameters: lr={lr}, depth={depth}, l2={l2}, iters={iters}")
                                
                                # 交差検証スコアを計算
                                cv_scores = []
                                for train_idx, val_idx in cv.split(X_train_safe, y_train):
                                    X_cv_train = X_train_safe.iloc[train_idx]
                                    y_cv_train = y_train.iloc[train_idx]
                                    X_cv_val = X_train_safe.iloc[val_idx]
                                    y_cv_val = y_train.iloc[val_idx]
                                    
                                    # モデルを初期化して学習
                                    cv_model = CatBoostClassifier(**current_params)
                                    
                                    # カテゴリカル特徴量があれば指定
                                    fit_params = {}
                                    if cat_features_safe is not None:
                                        fit_params['cat_features'] = cat_features_safe
                                    
                                    # モデルをフィット
                                    cv_model.fit(X_cv_train, y_cv_train, **fit_params)
                                    
                                    # バリデーションスコアを計算
                                    score = cv_model.score(X_cv_val, y_cv_val)
                                    cv_scores.append(score)
                                
                                # 平均スコアを計算
                                mean_score = np.mean(cv_scores)
                                print(f"Mean CV score: {mean_score:.4f}")
                                
                                # より良いスコアが得られた場合、パラメータを更新
                                if mean_score > best_score:
                                    best_score = mean_score
                                    best_params = {
                                        'learning_rate': lr,
                                        'depth': depth,
                                        'l2_leaf_reg': l2,
                                        'iterations': iters
                                    }
                                    print(f"New best score: {best_score:.4f} with params: {best_params}")
                            
                            except Exception as e:
                                print(f"Error during CV with params lr={lr}, depth={depth}, l2={l2}, iters={iters}: {e}")
                                continue
            
            if not best_params:
                raise ValueError("No valid parameters found during grid search")
            
            # 最終モデルをベストパラメータで学習
            print(f"\nTraining final model with best parameters: {best_params}")
            best_model = CatBoostClassifier(
                loss_function='Logloss',
                random_seed=self.random_state,
                verbose=False,
                **best_params
            )
            
            # カテゴリカル特徴量があれば指定して学習
            fit_params = {}
            if cat_features_safe is not None:
                fit_params['cat_features'] = cat_features_safe
            
            if X_test is not None and y_test is not None:
                fit_params['eval_set'] = (X_test_safe, y_test)
            
            best_model.fit(X_train_safe, y_train, **fit_params)
            
            self.model = best_model
            self.best_params = best_params
            
            # 特徴量重要度を設定
            self.feature_importances_ = best_model.get_feature_importance()
            
        except Exception as e:
            print(f"Grid search error: {e}")
            print("Using default parameters...")
            
            # デフォルトモデルを作成して学習
            model = self.get_model()
            
            # カテゴリカル特徴量があれば指定して学習
            fit_params = {}
            if cat_features_safe is not None:
                fit_params['cat_features'] = cat_features_safe
            
            if X_test is not None and y_test is not None:
                fit_params['eval_set'] = (X_test_safe, y_test)
            
            model.fit(X_train_safe, y_train, **fit_params)
            
            # パラメータを取得
            params = model.get_params()
            best_params = {
                'learning_rate': params.get('learning_rate', 0.05),
                'depth': params.get('depth', 6),
                'l2_leaf_reg': params.get('l2_leaf_reg', 3),
                'iterations': params.get('iterations', 1000)
            }
            
            self.model = model
            self.best_params = best_params
            
            # 特徴量重要度を設定
            self.feature_importances_ = model.get_feature_importance()
        
        print("\nBest parameters:")
        for param, value in self.best_params.items():
            print(f"{param}: {value}")
        
        return self.best_params, self.model
    
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
        if hasattr(model, 'feature_importances_'):
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
            Predicted class labels
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict(X)
    
    def predict_proba(self, X):
        """
        Predict class probabilities - mainコードとの連携のために追加
        
        Parameters:
        -----------
        X : array-like
            Features
            
        Returns:
        --------
        array
            Predicted class probabilities
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet.")
        return self.model.predict_proba(X)
    
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
                plt.plot(evals_result['learn']['Logloss'], label='Training')
            
            # Validation loss curve
            if 'validation' in evals_result:
                plt.plot(evals_result['validation']['Logloss'], label='Validation')
            
            plt.title('CatBoost - Learning Curve')
            plt.xlabel('Iterations')
            plt.ylabel('Logloss')
            plt.legend()
            plt.grid(True)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"Error plotting learning curve: {e}")