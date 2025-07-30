"""
回帰モデル統合ファイル（改良版）

このファイルには以下のクラスが含まれています：
- BaseModel: 回帰モデルの基底クラス
- XGBoostRegressionModel: XGBoost回帰モデル
- LightGBMRegressionModel: LightGBM回帰モデル
- CatBoostRegressionModel: CatBoost回帰モデル
- RandomForestRegressionModel: RandomForest回帰モデル

改良点：
- より堅牢なエラーハンドリング
- ライブラリ存在チェック機能
- 拡張パラメータグリッド
- 詳細なドキュメンテーション
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ========================================
# 依存関係チェック関数
# ========================================

def check_library_availability():
    """
    必要なライブラリの利用可能性をチェック
    
    Returns:
    --------
    availability : dict
        各ライブラリの利用可能性
    """
    availability = {
        'sklearn': False,
        'xgboost': False,
        'lightgbm': False,
        'catboost': False,
        'numpy': False,
        'pandas': False
    }
    
    try:
        import sklearn
        availability['sklearn'] = True
    except ImportError:
        pass
    
    try:
        import xgboost
        availability['xgboost'] = True
    except ImportError:
        pass
    
    try:
        import lightgbm
        availability['lightgbm'] = True
    except ImportError:
        pass
    
    try:
        import catboost
        availability['catboost'] = True
    except ImportError:
        pass
    
    try:
        import numpy
        availability['numpy'] = True
    except ImportError:
        pass
    
    try:
        import pandas
        availability['pandas'] = True
    except ImportError:
        pass
    
    return availability

# ライブラリ利用可能性を事前チェック
_LIBRARY_AVAILABILITY = check_library_availability()

# ========================================
# BaseModel: 回帰モデルの基底クラス
# ========================================

class BaseModel:
    """
    回帰モデルの基底クラス
    すべての回帰モデルクラスはこのクラスを継承する
    """
    def __init__(self, random_state=42):
        """
        初期化
        
        Parameters:
        -----------
        random_state : int, default=42
            乱数シード
        """
        self.random_state = random_state
        self.model = None
        self._model_type = self.__class__.__name__.lower().replace('regressionmodel', '')
    
    def get_model(self):
        """
        具体的なモデルインスタンスを返す
        サブクラスで実装必須
        
        Returns:
        --------
        model : sklearn-compatible model
            学習可能なモデルインスタンス
        """
        raise NotImplementedError("Subclasses must implement get_model method")
    
    def get_param_grid(self):
        """
        グリッドサーチ用のパラメータグリッドを返す
        サブクラスで実装必須
        
        Returns:
        --------
        param_grid : dict
            グリッドサーチ用のパラメータ辞書
        """
        raise NotImplementedError("Subclasses must implement get_param_grid method")
    
    def get_enhanced_param_grid(self):
        """
        拡張パラメータグリッドを返す（デフォルトでは通常のパラメータグリッドと同じ）
        
        Returns:
        --------
        param_grid : dict
            拡張グリッドサーチ用のパラメータ辞書
        """
        return self.get_param_grid()
    
    def fit(self, X, y):
        """
        モデルの学習
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            特徴量
        y : array-like or pandas.Series
            目的変数
            
        Returns:
        --------
        self : object
            自身を返す
        """
        if self.model is None:
            self.model = self.get_model()
        self.model.fit(X, y)
        return self
    
    def predict(self, X):
        """
        予測の実行
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            特徴量
            
        Returns:
        --------
        predictions : array-like
            予測値
        """
        if self.model is None:
            raise ValueError("Model has not been fitted yet")
        return self.model.predict(X)
    
    def get_params(self, deep=True):
        """
        パラメータの取得（scikit-learn互換）
        
        Parameters:
        -----------
        deep : bool, default=True
            深い階層のパラメータも取得するかどうか
            
        Returns:
        --------
        params : dict
            パラメータ辞書
        """
        if self.model is not None:
            return self.model.get_params(deep)
        else:
            return self.get_model().get_params(deep)
    
    def set_params(self, **params):
        """
        パラメータの設定（scikit-learn互換）
        
        Parameters:
        -----------
        **params : dict
            設定するパラメータ
            
        Returns:
        --------
        self : object
            パラメータが設定されたモデル
        """
        if self.model is not None:
            return self.model.set_params(**params)
        else:
            model = self.get_model()
            return model.set_params(**params)
    
    def check_dependencies(self):
        """
        このモデルに必要な依存関係をチェック
        
        Returns:
        --------
        available : bool
            依存関係が満たされているかどうか
        """
        required_libs = self.get_required_libraries()
        for lib in required_libs:
            if not _LIBRARY_AVAILABILITY.get(lib, False):
                return False
        return True
    
    def get_required_libraries(self):
        """
        このモデルに必要なライブラリのリストを返す
        サブクラスでオーバーライド可能
        
        Returns:
        --------
        libraries : list
            必要なライブラリ名のリスト
        """
        return ['sklearn', 'numpy']

# ========================================
# XGBoostRegressionModel: XGBoost回帰モデル
# ========================================

class XGBoostRegressionModel(BaseModel):
    """
    XGBoost回帰モデルクラス
    
    回帰問題に特化したXGBoostの実装
    L1/L2正則化パラメータを含む拡張パラメータグリッドを提供
    """
    
    def get_required_libraries(self):
        """XGBoostモデルに必要なライブラリ"""
        return ['sklearn', 'numpy', 'xgboost']
    
    def get_model(self):
        """
        XGBoost回帰モデルインスタンスを作成
        
        Returns:
        --------
        model : xgboost.XGBRegressor
            XGBoost回帰モデル
        """
        if not self.check_dependencies():
            raise ImportError("XGBoost がインストールされていません。'pip install xgboost' でインストールしてください。")
        
        import xgboost as xgb
        
        return xgb.XGBRegressor(
            random_state=self.random_state,
            objective='reg:squarederror',  # 回帰用の目的関数
            eval_metric='rmse',           # 回帰用の評価指標
            n_jobs=-1,                    # 並列処理を有効化
            verbosity=0                   # ログを抑制
        )
    
    def get_param_grid(self):
        """
        XGBoost用の基本パラメータグリッド
        
        Returns:
        --------
        param_grid : dict
            グリッドサーチ用のパラメータ辞書
        """
        return {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 6],
            'min_child_weight': [1, 3],
            'subsample': [0.8, 1.0],
            'colsample_bytree': [0.8, 1.0]
        }
    
    def get_enhanced_param_grid(self):
        """
        XGBoost用の拡張パラメータグリッド
        
        Returns:
        --------
        param_grid : dict
            拡張グリッドサーチ用のパラメータ辞書
        """
        return {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [3, 5, 7, 9],
            'min_child_weight': [1, 3, 5, 7],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'gamma': [0, 0.1, 0.2, 0.3],
            'reg_alpha': [0, 0.1, 0.5, 1.0],  # L1正則化
            'reg_lambda': [1, 1.5, 2, 3]      # L2正則化
        }

# ========================================
# LightGBMRegressionModel: LightGBM回帰モデル
# ========================================

class LightGBMRegressionModel(BaseModel):
    """
    LightGBM回帰モデルクラス
    
    高速な勾配ブースティング回帰の実装
    メモリ効率とスピードに優れている
    """
    
    def get_required_libraries(self):
        """LightGBMモデルに必要なライブラリ"""
        return ['sklearn', 'numpy', 'lightgbm']
    
    def get_model(self):
        """
        LightGBM回帰モデルインスタンスを作成
        
        Returns:
        --------
        model : lightgbm.LGBMRegressor
            LightGBM回帰モデル
        """
        if not self.check_dependencies():
            raise ImportError("LightGBM がインストールされていません。'pip install lightgbm' でインストールしてください。")
        
        import lightgbm as lgb
        
        return lgb.LGBMRegressor(
            random_state=self.random_state,
            objective='regression',
            metric='rmse',
            verbosity=-1,
            n_jobs=-1
        )
    
    def get_param_grid(self):
        """
        LightGBM用の基本パラメータグリッド
        
        Returns:
        --------
        param_grid : dict
            グリッドサーチ用のパラメータ辞書
        """
        return {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [-1, 10],
            'num_leaves': [31, 50],
            'min_child_samples': [20, 30],
            'subsample': [0.8, 1.0]
        }
    
    def get_enhanced_param_grid(self):
        """
        LightGBM用の拡張パラメータグリッド
        
        Returns:
        --------
        param_grid : dict
            拡張グリッドサーチ用のパラメータ辞書
        """
        return {
            'n_estimators': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'max_depth': [-1, 5, 10, 20],
            'num_leaves': [20, 31, 50, 100],
            'min_child_samples': [10, 20, 30, 50],
            'subsample': [0.7, 0.8, 0.9, 1.0],
            'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
            'reg_alpha': [0, 0.1, 0.5, 1.0],
            'reg_lambda': [0, 0.1, 0.5, 1.0]
        }

# ========================================
# CatBoostRegressionModel: CatBoost回帰モデル
# ========================================

class CatBoostRegressionModel(BaseModel):
    """
    CatBoost回帰モデルクラス
    
    カテゴリカル変数の処理に優れた勾配ブースティング回帰
    前処理が少なくて済むのが特徴
    """
    
    def get_required_libraries(self):
        """CatBoostモデルに必要なライブラリ"""
        return ['sklearn', 'numpy', 'catboost']
    
    def get_model(self):
        """
        CatBoost回帰モデルインスタンスを作成
        
        Returns:
        --------
        model : catboost.CatBoostRegressor
            CatBoost回帰モデル
        """
        if not self.check_dependencies():
            raise ImportError("CatBoost がインストールされていません。'pip install catboost' でインストールしてください。")
        
        from catboost import CatBoostRegressor
        
        return CatBoostRegressor(
            random_seed=self.random_state,
            loss_function='RMSE',
            verbose=False,
            thread_count=-1
        )
    
    def get_param_grid(self):
        """
        CatBoost用の基本パラメータグリッド
        
        Returns:
        --------
        param_grid : dict
            グリッドサーチ用のパラメータ辞書
        """
        return {
            'iterations': [100, 200],
            'learning_rate': [0.05, 0.1],
            'depth': [4, 6],
            'l2_leaf_reg': [1, 3],
            'border_count': [32, 64]
        }
    
    def get_enhanced_param_grid(self):
        """
        CatBoost用の拡張パラメータグリッド
        
        Returns:
        --------
        param_grid : dict
            拡張グリッドサーチ用のパラメータ辞書
        """
        return {
            'iterations': [50, 100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1, 0.2],
            'depth': [3, 4, 6, 8],
            'l2_leaf_reg': [1, 3, 5, 7],
            'border_count': [32, 64, 128, 200],
            'bagging_temperature': [0, 0.5, 1, 1.5],
            'random_strength': [1, 2, 3]
        }

# ========================================
# RandomForestRegressionModel: RandomForest回帰モデル
# ========================================

class RandomForestRegressionModel(BaseModel):
    """
    RandomForest回帰モデルクラス
    
    アンサンブル学習による回帰モデル
    過学習に強く、解釈しやすい特徴量重要度を提供
    """
    
    def get_required_libraries(self):
        """RandomForestモデルに必要なライブラリ"""
        return ['sklearn', 'numpy']
    
    def get_model(self):
        """
        RandomForest回帰モデルインスタンスを作成
        
        Returns:
        --------
        model : sklearn.ensemble.RandomForestRegressor
            RandomForest回帰モデル
        """
        if not self.check_dependencies():
            raise ImportError("scikit-learn がインストールされていません。'pip install scikit-learn' でインストールしてください。")
        
        from sklearn.ensemble import RandomForestRegressor
        
        return RandomForestRegressor(
            random_state=self.random_state,
            n_jobs=-1
        )
    
    def get_param_grid(self):
        """
        RandomForest用の基本パラメータグリッド
        
        Returns:
        --------
        param_grid : dict
            グリッドサーチ用のパラメータ辞書
        """
        return {
            'n_estimators': [100, 200],
            'max_depth': [None, 10],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2],
            'max_features': ['sqrt', 'log2']
        }
    
    def get_enhanced_param_grid(self):
        """
        RandomForest用の拡張パラメータグリッド
        
        Returns:
        --------
        param_grid : dict
            拡張グリッドサーチ用のパラメータ辞書
        """
        return {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [None, 5, 10, 20],
            'min_samples_split': [2, 5, 10, 15],
            'min_samples_leaf': [1, 2, 4, 6],
            'max_features': ['sqrt', 'log2', None],
            'bootstrap': [True, False],
            'max_samples': [None, 0.8, 0.9]
        }

# ========================================
# ヘルパー関数
# ========================================

def get_available_models():
    """
    利用可能なモデルの一覧を取得
    
    Returns:
    --------
    models : dict
        モデル名とクラスの辞書
    """
    all_models = {
        'xgboost': XGBoostRegressionModel,
        'lightgbm': LightGBMRegressionModel,
        'catboost': CatBoostRegressionModel,
        'random_forest': RandomForestRegressionModel
    }
    
    # 利用可能なモデルのみを返す
    available_models = {}
    for name, model_class in all_models.items():
        try:
            test_instance = model_class(random_state=42)
            if test_instance.check_dependencies():
                available_models[name] = model_class
        except Exception:
            # 依存関係エラーの場合はスキップ
            pass
    
    return available_models

def create_model(model_type, random_state=42):
    """
    指定されたタイプのモデルを作成
    
    Parameters:
    -----------
    model_type : str
        モデルタイプ ('xgboost', 'lightgbm', 'catboost', 'random_forest')
    random_state : int, default=42
        乱数シード
        
    Returns:
    --------
    model : BaseModel
        指定されたタイプのモデルインスタンス
    """
    available_models = get_available_models()
    
    if model_type not in available_models:
        # 利用可能なモデルの情報を提供
        available_names = list(available_models.keys())
        raise ValueError(f"モデルタイプ '{model_type}' は利用できません。"
                        f"利用可能なモデル: {available_names}")
    
    return available_models[model_type](random_state=random_state)

def check_all_dependencies():
    """
    全ての依存関係をチェックし、詳細な情報を返す
    
    Returns:
    --------
    report : dict
        依存関係の詳細レポート
    """
    availability = check_library_availability()
    
    model_availability = {}
    all_models = {
        'xgboost': XGBoostRegressionModel,
        'lightgbm': LightGBMRegressionModel, 
        'catboost': CatBoostRegressionModel,
        'random_forest': RandomForestRegressionModel
    }
    
    for name, model_class in all_models.items():
        try:
            test_instance = model_class(random_state=42)
            model_availability[name] = {
                'available': test_instance.check_dependencies(),
                'required_libs': test_instance.get_required_libraries(),
                'error': None
            }
        except Exception as e:
            model_availability[name] = {
                'available': False,
                'required_libs': [],
                'error': str(e)
            }
    
    return {
        'libraries': availability,
        'models': model_availability,
        'summary': {
            'total_libs': len(availability),
            'available_libs': sum(availability.values()),
            'total_models': len(model_availability),
            'available_models': sum(1 for m in model_availability.values() if m['available'])
        }
    }

def test_all_models():
    """
    全モデルの動作テスト用関数
    """
    print("=== 回帰モデル統合ファイルのテスト（改良版） ===")
    
    # 依存関係レポート
    dependency_report = check_all_dependencies()
    
    print(f"\n📊 依存関係レポート:")
    print(f"  利用可能ライブラリ: {dependency_report['summary']['available_libs']}/{dependency_report['summary']['total_libs']}")
    print(f"  利用可能モデル: {dependency_report['summary']['available_models']}/{dependency_report['summary']['total_models']}")
    
    # ライブラリ詳細
    print(f"\n📚 ライブラリ状況:")
    for lib, available in dependency_report['libraries'].items():
        status = "✅" if available else "❌"
        print(f"  {status} {lib}")
    
    # モデル詳細テスト
    print(f"\n🤖 モデル詳細テスト:")
    
    try:
        # サンプルデータの作成（numpyが利用可能な場合のみ）
        if dependency_report['libraries']['numpy']:
            import numpy as np
            from sklearn.datasets import make_regression
            
            X, y = make_regression(n_samples=100, n_features=5, random_state=42)
            
            # 利用可能なモデルをテスト
            available_models = get_available_models()
            
            for model_name, model_class in available_models.items():
                print(f"\n--- {model_name.upper()} ---")
                
                try:
                    # モデルインスタンスの作成
                    model_instance = create_model(model_name, random_state=42)
                    
                    # 実際のモデルを取得
                    model = model_instance.get_model()
                    
                    # 学習
                    model.fit(X, y)
                    
                    # 予測
                    predictions = model.predict(X[:5])
                    
                    # パラメータグリッドの取得
                    basic_grid = model_instance.get_param_grid()
                    enhanced_grid = model_instance.get_enhanced_param_grid()
                    
                    print(f"  ✅ 学習成功")
                    print(f"  予測値例: {predictions[:3]}")
                    print(f"  基本パラメータグリッド: {len(basic_grid)}項目")
                    print(f"  拡張パラメータグリッド: {len(enhanced_grid)}項目")
                    
                    # パラメータ組み合わせ数の計算
                    basic_combinations = 1
                    for param_values in basic_grid.values():
                        basic_combinations *= len(param_values)
                    
                    enhanced_combinations = 1
                    for param_values in enhanced_grid.values():
                        enhanced_combinations *= len(param_values)
                    
                    print(f"  基本組み合わせ数: {basic_combinations:,}")
                    print(f"  拡張組み合わせ数: {enhanced_combinations:,}")
                    
                except Exception as e:
                    print(f"  ❌ エラー: {e}")
                    
        else:
            print("NumPyが利用できないため、実際の学習テストはスキップします")
            
            # それでも基本機能はテスト
            available_models = get_available_models()
            for model_name in available_models:
                try:
                    model_instance = create_model(model_name, random_state=42)
                    print(f"  ✅ {model_name.upper()}: インスタンス作成成功")
                except Exception as e:
                    print(f"  ❌ {model_name.upper()}: {e}")
        
        print(f"\n=== テスト完了 ===")
        
    except Exception as e:
        print(f"テスト中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

# ========================================
# モジュール情報
# ========================================

__all__ = [
    'BaseModel',
    'XGBoostRegressionModel',
    'LightGBMRegressionModel',
    'CatBoostRegressionModel',
    'RandomForestRegressionModel',
    'get_available_models',
    'create_model',
    'test_all_models',
    'check_library_availability',
    'check_all_dependencies'
]

# バージョン情報
__version__ = "1.1.0"
__author__ = "Regression Models Integration (Enhanced)"

if __name__ == "__main__":
    # モジュールが直接実行された場合のテスト
    test_all_models()