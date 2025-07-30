from .xgboost_model import XGBoostModel
from .lightgbm_model import LightGBMModel
from .random_forest_model import RandomForestModel

MODEL_MAPPING = {
    'xgboost': XGBoostModel,
    'lightgbm': LightGBMModel,
    'random_forest': RandomForestModel
}

def get_model_class(model_name):
    """モデル名からモデルクラスを取得する関数"""
    model_class = MODEL_MAPPING.get(model_name.lower())
    if model_class is None:
        raise ValueError(f"Unknown model: {model_name}. Available models: {list(MODEL_MAPPING.keys())}")
    return model_class