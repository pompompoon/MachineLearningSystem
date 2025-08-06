import lightgbm as lgb
from .base_model import BaseModel

class LightGBMModel(BaseModel):
    def get_model(self):
        return lgb.LGBMClassifier(
            random_state=self.random_state,
            verbose=-1
        )
    
    def get_param_grid(self):
        return {
            'n_estimators': [50, 100, 200],
            'learning_rate': [0.01,0.05,0.1],
            'max_depth': [3, 5, 7],
            'min_child_samples': [10, 20, 30],
            'min_split_gain': [1e-4, 1e-3, 1e-2],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }