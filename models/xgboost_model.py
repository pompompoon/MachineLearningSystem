import xgboost as xgb
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def get_model(self):
        return xgb.XGBClassifier(
            random_state=self.random_state,
            use_label_encoder=False,
            eval_metric='logloss',
           # enable_categorical=True  # カテゴリカル変数のサポートを有効化
        )
    
    def get_param_grid(self):
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0],
            'gamma': [0, 0.1, 0.2]
        }
"""
    def fit(self, X, y):
        # scikit-learnの互換性のために必要
        self.model = self.get_model()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        # scikit-learnの互換性のために必要
        return self.model.predict(X)

    def predict_proba(self, X):
        # scikit-learnの互換性のために必要
        return self.model.predict_proba(X)
"""