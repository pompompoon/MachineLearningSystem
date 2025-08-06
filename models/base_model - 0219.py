from abc import ABC, abstractmethod
from sklearn.model_selection import cross_val_score
# from sklearn.model_selection import GridSearchCV
# from sklearn.base import BaseEstimator, ClassifierMixin  # 追加

# class BaseModel (ABC, BaseEstimator, ClassifierMixin):  # 継承を追加
class BaseModel:
    def __init__(self, random_state=42):
        # super().__init__()  # 親クラスの初期化
        self.random_state = random_state
        self.model = None
        self.best_params = None
    
    @abstractmethod
    def get_model(self):
        pass
    
    @abstractmethod
    def get_param_grid(self):
        pass
    """
    def perform_grid_search(self, X_train, y_train, X_test, y_test):
        model = self.get_model()
        param_grid = self.get_param_grid()
    """  
    def perform_grid_search(self, X_train, y_train, X_test, y_test):
        param_grid = self.get_param_grid()
        best_score = -1
        best_params = None
        best_model = None
        """    
        grid_search = GridSearchCV(
            estimator=model,
            param_grid=param_grid,
            cv=5,
            scoring='accuracy',
            n_jobs=-1,
            verbose=1
        )
        """
        # 手動でグリッドサーチを実装
        for params in self._get_param_combinations(param_grid):
            model = self.get_model()
            model.set_params(**params)

            # クロスバリデーションでスコアを計算
            scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
            avg_score = scores.mean()
            
            if avg_score > best_score:
                best_score = avg_score
                best_params = params
                best_model = model

        # 最良のパラメータでモデルを訓練
        best_model.fit(X_train, y_train)
        self.model = best_model
        self.best_params = best_params

        """        
        grid_search.fit(X_train, y_train)
        self.best_params = grid_search.best_params_
        self.model = grid_search.best_estimator_
        """
        # return self.best_params, self.model
        return best_params, best_model
    def _get_param_combinations(self, param_grid):
        """パラメータの全組み合わせを生成"""
        from itertools import product
        keys = param_grid.keys()
        values = param_grid.values()
        for instance in product(*values):
            yield dict(zip(keys, instance))