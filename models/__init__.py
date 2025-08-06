"""
回帰モデル統合パッケージ (完全修正版)

このパッケージには以下が含まれています：
- 回帰モデルクラス群
- グリッドサーチ対応モデル作成関数
- モデル情報取得関数
- 統合管理関数

使用例:
    from models import create_model, get_regression_model
    model = create_model('xgboost', random_state=42)
"""

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

# ========================================
# 基本インポート
# ========================================

import sys
import os
from typing import Dict, List, Tuple, Any, Optional, Union

# ========================================
# 利用可能性フラグ（最優先で定義）
# ========================================

# 初期値を設定
REGRESSION_MODELS_AVAILABLE = False
CLASSIFICATION_MODELS_AVAILABLE = False

# ========================================
# フォールバック用の基本クラス定義
# ========================================

class BaseModel:
    """フォールバック用の基本モデルクラス"""
    def __init__(self, random_state=42):
        self.random_state = random_state

class XGBoostRegressionModel(BaseModel):
    """フォールバック用のXGBoostモデルクラス"""
    pass

class LightGBMRegressionModel(BaseModel):
    """フォールバック用のLightGBMモデルクラス"""
    pass

class CatBoostRegressionModel(BaseModel):
    """フォールバック用のCatBoostモデルクラス"""
    pass

class RandomForestRegressionModel(BaseModel):
    """フォールバック用のRandomForestモデルクラス"""
    pass

# ========================================
# フォールバック関数（常に定義）
# ========================================

def check_library_availability():
    """ライブラリ利用可能性チェック（フォールバック版）"""
    availability = {}
    for lib in ['sklearn', 'xgboost', 'lightgbm', 'catboost', 'numpy', 'pandas']:
        try:
            __import__(lib)
            availability[lib] = True
        except ImportError:
            availability[lib] = False
    return availability

def check_all_dependencies():
    """依存関係チェック（フォールバック版）"""
    libraries = check_library_availability()
    return {
        'libraries': libraries,
        'models': {},
        'summary': {
            'total_libs': len(libraries),
            'available_libs': sum(libraries.values()),
            'total_models': 4,
            'available_models': sum(libraries.values()) // 2  # 概算
        }
    }

# ========================================
# regression_models.py からの段階的インポート
# ========================================

print("🔄 regression_models.py からのインポートを開始...")

# ステップ1: 基本クラスのインポート
try:
    from .regression_models import (
        BaseModel,
        XGBoostRegressionModel,
        LightGBMRegressionModel,
        CatBoostRegressionModel,
        RandomForestRegressionModel
    )
    print("✅ 基本クラスのインポート成功")
    classes_imported = True
except ImportError as e:
    print(f"⚠️  基本クラスのインポート失敗: {e}")
    classes_imported = False

# ステップ2: 基本関数のインポート
try:
    from .regression_models import (
        get_available_models as _get_available_models,
        create_model as _create_model,
        test_all_models as _test_all_models
    )
    print("✅ 基本関数のインポート成功")
    functions_imported = True
except ImportError as e:
    print(f"⚠️  基本関数のインポート失敗: {e}")
    functions_imported = False
    
    # フォールバック関数を定義
    def _get_available_models():
        if classes_imported:
            return {
                'xgboost': XGBoostRegressionModel,
                'lightgbm': LightGBMRegressionModel,
                'catboost': CatBoostRegressionModel,
                'random_forest': RandomForestRegressionModel
            }
        return {}
    
    def _create_model(model_type, random_state=42):
        models_map = _get_available_models()
        if model_type in models_map:
            return models_map[model_type](random_state=random_state)
        else:
            raise ValueError(f"サポートされていないモデル: {model_type}")
    
    def _test_all_models():
        print("フォールバック版テストを実行中...")

# ステップ3: オプション関数のインポート（失敗しても続行）
try:
    from .regression_models import check_library_availability as _check_library_availability
    check_library_availability = _check_library_availability
    print("✅ check_library_availability のインポート成功")
except ImportError:
    print("ℹ️  check_library_availability は内蔵版を使用")

try:
    from .regression_models import check_all_dependencies as _check_all_dependencies
    check_all_dependencies = _check_all_dependencies
    print("✅ check_all_dependencies のインポート成功")
except ImportError:
    print("ℹ️  check_all_dependencies は内蔵版を使用")

# ========================================
# インポート結果の評価
# ========================================

# 成功条件：基本クラスまたは基本関数のいずれかが成功
if classes_imported or functions_imported:
    REGRESSION_MODELS_AVAILABLE = True
    success_items = []
    if classes_imported:
        success_items.append("クラス")
    if functions_imported:
        success_items.append("関数")
    
    print(f"✅ regression_models.py から正常にインポートしました ({', '.join(success_items)})")
else:
    REGRESSION_MODELS_AVAILABLE = False
    print("❌ regression_models.py からのインポートが完全に失敗しました")

# ========================================
# 高度なモデル作成関数
# ========================================

def get_regression_model(model_type: str, random_state: int = 42, **kwargs):
    """
    標準的な回帰モデルを作成
    
    Parameters:
    -----------
    model_type : str
        モデルタイプ ('xgboost', 'lightgbm', 'catboost', 'random_forest')
    random_state : int, default=42
        乱数シード
    **kwargs : dict
        追加のモデルパラメータ
        
    Returns:
    --------
    model : sklearn-compatible model
        学習可能なモデルインスタンス
    """
    if not REGRESSION_MODELS_AVAILABLE:
        raise ImportError("回帰モデルが利用できません。regression_models.py を確認してください。")
    
    try:
        # 基本モデルを作成
        model_instance = _create_model(model_type, random_state=random_state)
        actual_model = model_instance.get_model()
        
        # 追加パラメータがある場合は設定
        if kwargs:
            actual_model.set_params(**kwargs)
        
        return actual_model
        
    except Exception as e:
        raise ValueError(f"モデル '{model_type}' の作成に失敗しました: {e}")

def get_regression_model_with_grid(model_type: str, random_state: int = 42, **kwargs):
    """
    グリッドサーチ対応の拡張パラメータ回帰モデルを作成
    
    Parameters:
    -----------
    model_type : str
        モデルタイプ
    random_state : int, default=42
        乱数シード
    **kwargs : dict
        追加のモデルパラメータ
        
    Returns:
    --------
    model : sklearn-compatible model
        拡張パラメータグリッドを持つモデルインスタンス
    """
    if not REGRESSION_MODELS_AVAILABLE:
        raise ImportError("回帰モデルが利用できません。")
    
    try:
        # 拡張パラメータ版モデルを作成
        model_instance = _create_model(model_type, random_state=random_state)
        actual_model = model_instance.get_model()
        
        # 拡張パラメータグリッドを取得
        if hasattr(model_instance, 'get_enhanced_param_grid'):
            param_grid = model_instance.get_enhanced_param_grid()
        elif hasattr(model_instance, 'get_param_grid'):
            param_grid = model_instance.get_param_grid()
        else:
            # 基本パラメータグリッドを定義
            param_grid = {
                'n_estimators': [100, 200],
                'learning_rate': [0.05, 0.1] if model_type in ['xgboost', 'lightgbm', 'catboost'] else [],
                'max_depth': [None, 10] if model_type == 'random_forest' else [3, 6]
            }
        
        # モデルに拡張情報を添付
        actual_model._enhanced_param_grid = param_grid
        actual_model._model_type = model_type
        actual_model._is_enhanced = True
        
        # 追加パラメータがある場合は設定
        if kwargs:
            actual_model.set_params(**kwargs)
        
        return actual_model
        
    except Exception as e:
        raise ValueError(f"拡張モデル '{model_type}' の作成に失敗しました: {e}")

# ========================================
# モデル情報取得関数
# ========================================

def get_model_info(model_name: str, task_type: str = 'regression') -> Dict[str, Any]:
    """
    指定されたモデルの詳細情報を取得
    
    Parameters:
    -----------
    model_name : str
        モデル名
    task_type : str, default='regression'
        タスクタイプ（現在は'regression'のみサポート）
        
    Returns:
    --------
    info : dict
        モデルの詳細情報
    """
    if task_type != 'regression':
        return {
            'name': model_name,
            'available': False,
            'error': f"タスクタイプ '{task_type}' はサポートされていません"
        }
    
    if not REGRESSION_MODELS_AVAILABLE:
        return {
            'name': model_name,
            'available': False,
            'error': "回帰モデルが利用できません"
        }
    
    try:
        # モデルインスタンスを作成して情報を取得
        model_instance = _create_model(model_name, random_state=42)
        
        # パラメータグリッドの取得
        if hasattr(model_instance, 'get_param_grid'):
            param_grid = model_instance.get_param_grid()
        else:
            param_grid = {'n_estimators': [100, 200]}  # デフォルト
        
        # 拡張パラメータグリッドがある場合はそれも取得
        enhanced_param_grid = None
        if hasattr(model_instance, 'get_enhanced_param_grid'):
            enhanced_param_grid = model_instance.get_enhanced_param_grid()
        
        # パラメータグリッドのサイズ計算
        param_grid_size = len(param_grid)
        
        # 全組み合わせ数の計算
        total_combinations = 1
        for param_values in param_grid.values():
            if isinstance(param_values, (list, tuple)):
                total_combinations *= len(param_values)
        
        # 拡張版の組み合わせ数
        enhanced_combinations = None
        if enhanced_param_grid:
            enhanced_combinations = 1
            for param_values in enhanced_param_grid.values():
                if isinstance(param_values, (list, tuple)):
                    enhanced_combinations *= len(param_values)
        
        # クラス名の取得
        class_name = model_instance.__class__.__name__
        
        info = {
            'name': model_name,
            'available': True,
            'class_name': class_name,
            'param_grid': param_grid,
            'param_grid_size': param_grid_size,
            'total_combinations': total_combinations,
            'task_type': task_type
        }
        
        # 拡張情報があれば追加
        if enhanced_param_grid:
            info['enhanced_param_grid'] = enhanced_param_grid
            info['enhanced_combinations'] = enhanced_combinations
        
        return info
        
    except Exception as e:
        return {
            'name': model_name,
            'available': False,
            'error': str(e)
        }

def list_available_models() -> Dict[str, List[str]]:
    """
    利用可能なモデルの一覧を取得
    
    Returns:
    --------
    models : dict
        タスクタイプ別のモデル一覧
    """
    result = {
        'classification': [],  # 将来の拡張用
        'regression': []
    }
    
    if REGRESSION_MODELS_AVAILABLE:
        try:
            available_models = _get_available_models()
            result['regression'] = list(available_models.keys())
        except Exception as e:
            print(f"回帰モデル一覧の取得に失敗: {e}")
            # デフォルトモデルを返す
            result['regression'] = ['xgboost', 'lightgbm', 'catboost', 'random_forest']
    
    return result

# ========================================
# テスト・検証関数
# ========================================

def test_models_integration():
    """
    統合パッケージの動作テスト
    """
    print("🧪 === 統合パッケージ詳細テスト ===")
    
    print(f"回帰モデル利用可能: {'✅' if REGRESSION_MODELS_AVAILABLE else '❌'}")
    print(f"分類モデル利用可能: {'✅' if CLASSIFICATION_MODELS_AVAILABLE else '❌'}")
    
    if not REGRESSION_MODELS_AVAILABLE:
        print("❌ 回帰モデルが利用できません")
        return
    
    try:
        available_models = list_available_models()
        regression_models = available_models['regression']
        
        print(f"📊 利用可能な回帰モデル数: {len(regression_models)}")
        print(f"📋 モデルリスト: {regression_models}")
        
        for model_name in regression_models[:2]:  # 最初の2つをテスト
            print(f"\n--- {model_name.upper()} ---")
            
            try:
                # 基本モデルのテスト
                basic_model = get_regression_model(model_name, random_state=42)
                print(f"  基本モデル: ✅ 作成成功")
                
                # 拡張モデルのテスト
                enhanced_model = get_regression_model_with_grid(model_name, random_state=42)
                print(f"  拡張モデル: ✅ 作成成功")
                
                # モデル情報の取得
                model_info = get_model_info(model_name, task_type='regression')
                if model_info['available']:
                    print(f"  パラメータグリッド: {model_info['param_grid_size']}個")
                    print(f"  組み合わせ数: {model_info['total_combinations']:,}通り")
                    print(f"  クラス名: {model_info['class_name']}")
                    
                    if 'enhanced_combinations' in model_info:
                        print(f"  拡張組み合わせ数: {model_info['enhanced_combinations']:,}通り")
                else:
                    print(f"  情報取得: ❌ {model_info['error']}")
                
            except ImportError as e:
                print(f"  ⚠️  スキップ: 必要なライブラリが未インストール ({e})")
            except Exception as e:
                print(f"  ❌ エラー: {e}")
        
        print(f"\n=== テスト完了 ===")
        
    except Exception as e:
        print(f"統合テスト中にエラー: {e}")
        import traceback
        traceback.print_exc()

def test_all_models():
    """
    全モデルの基本動作テスト
    """
    if REGRESSION_MODELS_AVAILABLE:
        _test_all_models()
    else:
        print("回帰モデルが利用できないため、テストをスキップします")

# ========================================
# 便利な統合関数
# ========================================

def create_model(model_type: str, random_state: int = 42, enhanced: bool = False, **kwargs):
    """
    統合モデル作成関数（基本・拡張対応）
    
    Parameters:
    -----------
    model_type : str
        モデルタイプ
    random_state : int, default=42
        乱数シード
    enhanced : bool, default=False
        拡張パラメータグリッド版を使用するかどうか
    **kwargs : dict
        追加のモデルパラメータ
        
    Returns:
    --------
    model : sklearn-compatible model
        学習可能なモデルインスタンス
    """
    if enhanced:
        return get_regression_model_with_grid(model_type, random_state=random_state, **kwargs)
    else:
        return get_regression_model(model_type, random_state=random_state, **kwargs)

def get_available_models():
    """
    回帰モデルの辞書を取得（regression_models.pyとの互換性用）
    
    Returns:
    --------
    models : dict
        モデル名とクラスの辞書
    """
    if REGRESSION_MODELS_AVAILABLE:
        return _get_available_models()
    else:
        return {}

# ========================================
# パッケージ情報とエクスポート
# ========================================

__version__ = "1.2.0"
__author__ = "Unified Regression Models Package (Robust Version)"

# エクスポートする関数・クラスの定義
__all__ = [
    # 基本クラス
    'BaseModel',
    'XGBoostRegressionModel',
    'LightGBMRegressionModel', 
    'CatBoostRegressionModel',
    'RandomForestRegressionModel',
    
    # モデル作成関数
    'get_regression_model',
    'get_regression_model_with_grid',
    'create_model',
    'get_available_models',
    
    # 情報取得関数
    'get_model_info',
    'list_available_models',
    
    # テスト関数
    'test_models_integration',
    'test_all_models',
    
    # 依存関係チェック関数
    'check_library_availability',
    'check_all_dependencies',
    
    # フラグ
    'REGRESSION_MODELS_AVAILABLE',
    'CLASSIFICATION_MODELS_AVAILABLE'
]

# ========================================
# 初期化完了メッセージ
# ========================================

if __name__ == "__main__":
    print("models パッケージが直接実行されました")
    print(f"回帰モデル利用可能: {'✅' if REGRESSION_MODELS_AVAILABLE else '❌'}")
    print(f"分類モデル利用可能: {'✅' if CLASSIFICATION_MODELS_AVAILABLE else '❌'}")
    
    if REGRESSION_MODELS_AVAILABLE:
        test_models_integration()
else:
    # パッケージがインポートされた時の情報表示
    if REGRESSION_MODELS_AVAILABLE:
        try:
            available = list_available_models()
            regression_count = len(available['regression'])
            print(f"📦 models パッケージが読み込まれました（回帰モデル: {regression_count}個利用可能）")
        except:
            print(f"📦 models パッケージが読み込まれました（回帰モデル: 利用可能）")
    else:
        print("⚠️  models パッケージが読み込まれましたが、回帰モデルは利用できません")

# デバッグ用：現在の状態を表示
print(f"🔧 DEBUG: REGRESSION_MODELS_AVAILABLE = {REGRESSION_MODELS_AVAILABLE}")
print(f"🔧 DEBUG: CLASSIFICATION_MODELS_AVAILABLE = {CLASSIFICATION_MODELS_AVAILABLE}")