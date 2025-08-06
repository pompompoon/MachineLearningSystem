from abc import ABC, abstractmethod
from sklearn.model_selection import KFold
import numpy as np

class BaseModel:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.model = None
        self.best_params = None
    
    @abstractmethod
    def get_model(self):
        pass
    
    @abstractmethod
    def get_param_grid(self):
        pass
    
    def perform_grid_search(self, X_train, y_train, X_test, y_test):
        param_grid = self.get_param_grid()
        best_score = -1
        best_params = None
        best_model = None
        
        # K分割交差検証の準備
        kf = KFold(n_splits=5, shuffle=True, random_state=self.random_state)
        total_combinations = np.prod([len(v) for v in param_grid.values()])
        print(f"Total parameter combinations: {total_combinations}")
        
        # 各パラメータの組み合わせを試す
        for i, params in enumerate(self._get_param_combinations(param_grid)):
            print(f"\nTrying combination {i+1}/{total_combinations}")
            print(f"Parameters: {params}")
            
            # 交差検証のスコアを計算
            scores = []
            for train_idx, val_idx in kf.split(X_train):
                # データの分割
                X_fold_train = X_train.iloc[train_idx]
                y_fold_train = y_train.iloc[train_idx]
                X_fold_val = X_train.iloc[val_idx]
                y_fold_val = y_train.iloc[val_idx]
                
                # モデルの学習と評価
                model = self.get_model()
                model.set_params(**params)
                model.fit(X_fold_train, y_fold_train)
                score = model.score(X_fold_val, y_fold_val)
                scores.append(score)
            
            # 平均スコアを計算
            mean_score = np.mean(scores)
            print(f"Mean CV score: {mean_score:.4f}")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = params
                best_model = self.get_model()
                best_model.set_params(**params)
        
        # 最良のモデルを全訓練データで学習
        best_model.fit(X_train, y_train)
        self.model = best_model
        self.best_params = best_params
        
        return best_params, best_model
    
    def _get_param_combinations(self, param_grid):
        """パラメータの全組み合わせを生成"""
        from itertools import product
        keys = param_grid.keys()
        values = param_grid.values()
        for instance in product(*values):
            yield dict(zip(keys, instance))