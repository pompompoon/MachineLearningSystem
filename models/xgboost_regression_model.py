import xgboost as xgb
from .base_model import BaseModel

class XGBoostModel(BaseModel):
    def get_model(self):
        return xgb.XGBRegressor(
            random_state=self.random_state,
            objective='reg:squarederror',  # 回帰用の目的関数
            eval_metric='rmse',  # 回帰用の評価指標
            # enable_categorical=True  # カテゴリカル変数のサポートを有効化
        )
    
    def get_param_grid(self):
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7],
            'min_child_weight': [1, 3, 5],
            'subsample': [0.8, 0.9, 1.0],
            'colsample_bytree': [0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2],
            'reg_alpha': [0, 0.1, 0.5],  # L1正則化（回帰で有用）
            'reg_lambda': [1, 1.5, 2]    # L2正則化（回帰で有用）
        }

    def fit(self, X, y):
        # scikit-learnの互換性のために必要
        self.model = self.get_model()
        self.model.fit(X, y)
        return self

    def predict(self, X):
        # scikit-learnの互換性のために必要（回帰では予測値を直接返す）
        return self.model.predict(X)

    # 回帰では predict_proba は不要（分類用のメソッド）
    # def predict_proba(self, X):
    #     # 回帰では確率ではなく予測値を返すため、このメソッドは不要
    #     return self.model.predict(X)