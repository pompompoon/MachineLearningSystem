# -*- coding: utf-8 -*-
# Save this file as: G:\共有ドライブ\GAP_長寿研\user\iwamoto\視線の動きの俊敏さ\models\catboost_model.py

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import matplotlib.pyplot as plt
import re

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
            
        # Replace non-ASCII characters and special characters with underscores
        sanitized_names = []
        for name in feature_names:
            # Replace non-ASCII characters and special characters with underscores
            sanitized = re.sub(r'[^\x00-\x7F]+', '_', name)
            sanitized = re.sub(r'[^\w]+', '_', sanitized)
            sanitized_names.append(sanitized)
        
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
        
        # Sanitize column names to avoid encoding issues
        if isinstance(X_train, pd.DataFrame):
            # Save original column names
            original_columns = X_train.columns.tolist()
            sanitized_columns = self._sanitize_feature_names(original_columns)
            
            # Create a copy with sanitized column names
            X_train_safe = X_train.copy()
            X_train_safe.columns = sanitized_columns
            
            X_test_safe = X_test.copy()
            X_test_safe.columns = sanitized_columns
        else:
            X_train_safe = X_train
            X_test_safe = X_test
        
        # Default parameter grid
        param_grid = {
            'learning_rate': [0.05],
            'depth': [6],
            'l2_leaf_reg': [3],
            'iterations': [1000]
        }
        
        # Cross-validation settings
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
        
        try:
            # Initialize model
            model = CatBoostClassifier(
                loss_function='Logloss',
                random_seed=self.random_state,
                verbose=False
            )
            
            # Simple grid search without using Pool for safety
            best_score = 0
            best_params = {}
            
            for lr in [0.05]:  # Simplify to reduce computation time
                for depth in [5]:  # Simplify to reduce computation time
                    for l2 in [3]:  # Simplify to reduce computation time
                        for iters in [500]:  # Simplify to reduce computation time
                            current_params = {
                                'learning_rate': lr,
                                'depth': depth,
                                'l2_leaf_reg': l2,
                                'iterations': iters
                            }
                            
                            # Create and fit model with current parameters
                            current_model = CatBoostClassifier(
                                loss_function='Logloss',
                                random_seed=self.random_state,
                                verbose=False,
                                **current_params
                            )
                            
                            # Simple cross-validation
                            scores = []
                            for train_idx, val_idx in cv.split(X_train_safe, y_train):
                                X_tr, X_val = X_train_safe.iloc[train_idx], X_train_safe.iloc[val_idx]
                                y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
                                
                                # Fit and evaluate
                                current_model.fit(X_tr, y_tr, verbose=False)
                                score = current_model.score(X_val, y_val)
                                scores.append(score)
                                
                            avg_score = np.mean(scores)
                            print(f"Parameters: lr={lr}, depth={depth}, l2={l2}, iters={iters} - Score: {avg_score:.4f}")
                            
                            if avg_score > best_score:
                                best_score = avg_score
                                best_params = current_params
            
            # Train final model with best parameters
            model = CatBoostClassifier(
                loss_function='Logloss',
                random_seed=self.random_state,
                verbose=False,
                **best_params
            )
            model.fit(X_train_safe, y_train, eval_set=(X_test_safe, y_test), verbose=False)
            
        except Exception as e:
            print(f"Grid search error: {e}")
            print("Using default parameters...")
            
            model = self.get_model()
            model.fit(X_train_safe, y_train, eval_set=(X_test_safe, y_test), verbose=False)
            
            # Get parameters safely
            params = model.get_params()
            best_params = {
                'learning_rate': params.get('learning_rate', 0.05),
                'depth': params.get('depth', 6),
                'l2_leaf_reg': params.get('l2_leaf_reg', 3),
                'iterations': params.get('iterations', 1000)
            }
        
        self.model = model
        self.best_params = best_params
        
        print("\nBest parameters:")
        for param, value in best_params.items():
            print(f"{param}: {value}")
        
        return best_params, model
    
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