
# main5.py 分類コード
import matplotlib
matplotlib.use('TkAgg')  # バックエンドをTkAggに設定（ウィンドウを表示できるもの）
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import matplotlib as mpl
import japanize_matplotlib

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import sys
import os
# カレントディレクトリをPythonパスに追加
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import os
from scipy.stats import chi2_contingency
import pickle  # モデルの保存・読み込みに使用
import datetime
import shutil
# from models.borderline_smote_processor import BorderlineSMOTEProcessor
# 既存のモデルのインポート
try:
    from models.lightgbm_model import LightGBMModel
    from models.xgboost_model import XGBoostModel
    from models.random_forest_model import RandomForestModel
    from models.catboost_model import CatBoostModel
    from models.undersampling_bagging_model import UndersamplingBaggingModel
    from models.undersampling_model import UndersamplingModel
    from models.cross_validator import CrossValidator, run_cv_analysis, safe_convert_array
    CLASSIFICATION_MODELS_AVAILABLE = True
    print("🔧 DEBUG: CLASSIFICATION_MODELS_AVAILABLE = True")
except ImportError as e:
    CLASSIFICATION_MODELS_AVAILABLE = False
    print(f"🔧 DEBUG: CLASSIFICATION_MODELS_AVAILABLE = False")
    print(f"🔧 DEBUG: モデルインポートエラー: {e}")

# 拡張機能のインポート（修正版）
try:
    from models.smotetomek_processor import SMOTETomekProcessor
    from visualization.aleplotter import ALEPlotter
    from models.enhanced_analysis_integration import EnhancedAnalysisRunner
    ENHANCED_FEATURES_AVAILABLE = True
    print("✓ 拡張機能（SMOTETomek & ALE）が利用可能です")
except ImportError as e:
    ENHANCED_FEATURES_AVAILABLE = False
    print(f"⚠ 拡張機能が利用できません: {e}")
    print("  models/smotetomek_processor.py, models/ale_plotter.py, models/enhanced_analysis_integration.py を確認してください")

# 視線データ可視化クラスのインポート
try:
    from visualization.eye_tracking_visualizer import EyeTrackingVisualizer
    from visualization.threshold_evaluator import ThresholdEvaluator
    VISUALIZATION_AVAILABLE = True
except ImportError as e:
    VISUALIZATION_AVAILABLE = False
    print(f"⚠ 可視化機能が利用できません: {e}")

try:
    from prediction import save_model  # prediction.pyから関数をインポート
    PREDICTION_AVAILABLE = True
except ImportError:
    PREDICTION_AVAILABLE = False

try:
    from models.holdout_bunrui import ClassificationHoldoutCVAnalyzer, run_classification_holdout_cv_analysis
    HOLDOUT_CV_AVAILABLE = True
except ImportError:
    print("🔧 DEBUG: HOLDOUT_CV_AVAILABLE = False")
    HOLDOUT_CV_AVAILABLE = False
def run_enhanced_comprehensive_analysis(df, model_class=None, output_dir='enhanced_result', 
                                       random_state=42, data_file_name=None, **kwargs):
    """
    SMOTETomek + ALE + 従来手法の包括的比較分析を実行
    """
    if not ENHANCED_FEATURES_AVAILABLE:
        print("エラー: 拡張機能が利用できません。")
        print("models/smotetomek_processor.py, models/ale_plotter.py, models/enhanced_analysis_integration.py を確認してください。")
        return None
    
    print("=== 包括的拡張分析開始 ===")
    
    # EnhancedAnalysisRunnerを使用
    runner = EnhancedAnalysisRunner(output_dir, random_state)
    
    results = runner.run_comprehensive_analysis(
        df=df,
        model_class=model_class,
        data_file_name=data_file_name,
        use_baseline=True,
        use_smote=True,
        use_smotetomek=True,
        use_ale=True,
        **kwargs
    )
    
    return results

def calculate_specificity(y_true, y_pred):
    """特異度（Specificity）を計算する関数"""
    cm = confusion_matrix(y_true, y_pred)
    # 2クラス分類の場合: TN=cm[0,0], FP=cm[0,1], FN=cm[1,0], TP=cm[1,1]
    if cm.shape == (2, 2):
        tn, fp = cm[0, 0], cm[0, 1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        # 多クラス分類の場合は特異度の計算が複雑になるため、マクロ平均を使用
        specificities = []
        n_classes = cm.shape[0]
        for i in range(n_classes):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm, i, axis=0)[:, i])
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        specificity = np.mean(specificities)
    
    return specificity

def prepare_data(df):
    """データの前処理を行う関数"""
    print("\nデータの形状:", df.shape)
    
    # 目的変数のカラム名を確認
    if 'target' in df.columns:
        target_column = 'target'
    elif 'Target' in df.columns:
        target_column = 'Target'
    else:
        raise ValueError("目的変数の列 'target' または 'Target' が見つかりません")
        
    # 元のデータの値を確認
    original_values = df[target_column].unique()
    print(f"元の{target_column}の一意な値:", original_values)
        
    # 重要: IDカラムの取得と保存
    id_column = None
    if 'InspectionDateAndId' in df.columns:
        id_column = df['InspectionDateAndId'].copy()
    elif 'Id' in df.columns:
        id_column = df['Id'].copy()
    
    # InspectionDateAndId、Id、および目的変数を除外した特徴量を抽出
    drop_cols = []
    if 'InspectionDateAndId' in df.columns:
        drop_cols.append('InspectionDateAndId')
    if 'Id' in df.columns:
        drop_cols.append('Id')
    drop_cols.append(target_column)

    features = df.drop(drop_cols, axis=1)
    
    # 特徴量名のクリーニング - LightGBM用の対応
    # JSON特殊文字や日本語を含む列名を修正
    new_columns = {}
    for col in features.columns:
        # 特殊文字や日本語を含む列名をアルファベット、数字、アンダースコアのみに置換
        new_col = ''.join(c if c.isascii() and (c.isalnum() or c == '_') else '_' for c in col)
        # 空の名前や数字から始まる名前の場合はプレフィックスを追加
        if not new_col or new_col[0].isdigit():
            new_col = 'f_' + new_col
        # 同じ名前になってしまう場合は連番を付与
        if new_col in new_columns.values():
            i = 1
            while f"{new_col}_{i}" in new_columns.values():
                i += 1
            new_col = f"{new_col}_{i}"
        
        if new_col != col:
            new_columns[col] = new_col
    
    # 列名の置換が必要な場合は置換
    if new_columns:
        print("\n特殊文字を含む列名を修正します:")
        for old_col, new_col in new_columns.items():
            print(f"  {old_col} -> {new_col}")
        
        features = features.rename(columns=new_columns)
    
    # 元のデータが数値の場合はそのまま使用
    if set(original_values).issubset({0, 1}):
        target = df[target_column].copy()
        print("数値ラベルをそのまま使用します (0, 1)")
    else:
        # 文字列の場合はマッピング
        target = df[target_column].map({'intact': 0, 'mci': 1})
        print("文字列ラベルをマッピングします (intact->0, mci->1)")
    
    # ターゲット変数の確認
    print("ターゲット変数のNaN数:", target.isna().sum())
    print("ターゲット変数の一意な値:", target.unique())
    
    for col in ['freq', 'power_spectrum']:
        if col in features.columns:
            features[col] = features[col].apply(safe_convert_array)
    
    # 欠損値の削除
    features = features.dropna()
    target = target[features.index]
    
    # ID列も同様にフィルタリング
    if id_column is not None:
        id_column = id_column[features.index]
    
    print(f"\n処理後のデータ数: {len(features)}")
    print("\nクラス分布:")
    print(target.value_counts())
    
    # NaNがある場合は警告
    if target.isna().any():
        print("警告: ターゲット変数にNaNが含まれています。これらの行を削除します。")
        valid_idx = ~target.isna()
        features = features[valid_idx]
        target = target[valid_idx]
        if id_column is not None:
            id_column = id_column[valid_idx]
        print(f"NaN削除後のデータ数: {len(features)}")
        print("修正後のクラス分布:")
        print(target.value_counts())
    
    # featuresとtargetに加えて、id_columnも返す
    return features, target, id_column

def train_model(features, target, id_column=None, model_class=None, use_smote=False, 
                use_undersampling=False, use_simple_undersampling=False, 
                use_smotetomek=False,  # 新しいパラメータ
                random_state=42, **kwargs):
    """モデルの学習を行う関数（SMOTETomek対応版）"""
    # データの分割
    if id_column is not None:
        # ID列も一緒に分割する
        X_train, X_test, y_train, y_test, id_train, id_test = train_test_split(
            features, target, id_column, test_size=0.2, random_state=random_state, stratify=target
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            features, target, test_size=0.2, random_state=random_state, stratify=target
        )
        id_train, id_test = None, None
    
    # 特徴量のスケーリング
    scaler = StandardScaler()
    X_train_scaled = pd.DataFrame(
        scaler.fit_transform(X_train),
        columns=features.columns
    )
    X_test_scaled = pd.DataFrame(
        scaler.transform(X_test),
        columns=features.columns
    )
    
    # オリジナルデータを保存（可視化のため）
    X_train_orig = X_train.copy()
    X_test_orig = X_test.copy()
    
    # SMOTETomekを適用する場合（新しい処理）
    if use_smotetomek:
        if not ENHANCED_FEATURES_AVAILABLE:
            print("警告: SMOTETomek機能が利用できません。通常のSMOTEを使用します。")
            use_smote = True
            use_smotetomek = False
        else:
            print("SMOTETomekを適用してデータをバランシングします...")
            
            # SMOTETomekプロセッサーの作成
            processor = SMOTETomekProcessor(
                smote_sampling_strategy=kwargs.get('smotetomek_strategy', 'auto'),
                smote_k_neighbors=kwargs.get('smotetomek_k_neighbors', 5),
                tomek_sampling_strategy=kwargs.get('tomek_strategy', 'auto'),
                random_state=random_state
            )
            
            # リサンプリング実行
            X_train_resampled, y_train_resampled, statistics = processor.fit_resample(
                X_train_scaled, y_train, verbose=False
            )
            
            X_train_scaled = pd.DataFrame(X_train_resampled, columns=features.columns)
            y_train = pd.Series(y_train_resampled)
            
            print("SMOTETomek適用後のクラス分布:")
            print(pd.Series(y_train).value_counts())
            
            # 統計情報を結果に含める
            kwargs['smotetomek_statistics'] = statistics
    
    # SMOTEを適用する場合（既存の処理）
    elif use_smote:
        from imblearn.over_sampling import SMOTE
        print("SMOTEを適用してデータをバランシングします...")
        smote = SMOTE(random_state=random_state)
        X_train_resampled, y_train_resampled = smote.fit_resample(X_train_scaled, y_train)
        X_train_scaled = pd.DataFrame(X_train_resampled, columns=features.columns)
        y_train = pd.Series(y_train_resampled)
        print("SMOTE適用後のクラス分布:")
        print(pd.Series(y_train).value_counts())
    
    # UndersamplingBaggingを使用する場合（既存の処理）
    if use_undersampling:
        print("UndersamplingBaggingモデルを使用します...")
        base_model = kwargs.get('base_model', 'lightgbm')
        n_bags = kwargs.get('n_bags', 10)
        model = UndersamplingBaggingModel(
            base_model=base_model,
            n_bags=n_bags,
            random_state=random_state
        )
        
        try:
            best_params, best_model = model.perform_grid_search(
                X_train_scaled, y_train,
                X_test_scaled, y_test
            )
        except Exception as e:
            print(f"グリッドサーチでエラーが発生しました: {e}")
            print("基本モデルを使用します...")
            best_model = model
            best_model.fit(X_train_scaled, y_train)
            best_params = {'n_bags': n_bags}
        
        # 特徴量重要度を設定
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = pd.DataFrame({'feature': features.columns, 'importance': 0})
    
    # シンプルなアンダーサンプリングを使用する場合
    elif use_simple_undersampling:
        print("シンプルアンダーサンプリングモデルを使用します...")
        base_model = kwargs.get('base_model', 'lightgbm')
        model = UndersamplingModel(
            base_model=base_model,
            random_state=random_state
        )
        
        try:
            best_params, best_model = model.perform_grid_search(
                X_train_scaled, y_train,
                X_test_scaled, y_test
            )
        except Exception as e:
            print(f"グリッドサーチでエラーが発生しました: {e}")
            print("基本モデルを使用します...")
            best_model = model
            best_model.fit(X_train_scaled, y_train)
            best_params = {}
        
        # 特徴量重要度を設定
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = pd.DataFrame({'feature': features.columns, 'importance': 0})
    
    else:
        # 通常のモデルを使用する場合
        model = model_class()
        try:
            best_params, best_model = model.perform_grid_search(
                X_train_scaled, y_train,
                X_test_scaled, y_test
            )
        except Exception as e:
            print(f"グリッドサーチでエラーが発生しました: {e}")
            print("基本モデルを使用します...")
            best_model = model.get_model()
            best_model.fit(X_train_scaled, y_train)
            best_params = {}
        
        # 特徴量の重要度
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': features.columns,
                'importance': best_model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            feature_importance = pd.DataFrame({'feature': features.columns, 'importance': 0})
    
    # 予測
    y_pred = best_model.predict(X_test_scaled)
    y_pred_proba = best_model.predict_proba(X_test_scaled)
    
    # 結果辞書の作成
    result = {
        'model': best_model,
        'predictions': y_pred,
        'prediction_proba': y_pred_proba,
        'true_values': y_test,
        'feature_importance': feature_importance,
        'test_indices': X_test.index,
        'best_params': best_params,
        'features': features,
        'target': target,  # 全データのターゲット
        'X_train': X_train_scaled,
        'X_train_orig': X_train_orig,  # スケーリング前のトレーニングデータ
        'y_train': y_train,  # トレーニングデータのターゲット
        'X_test': X_test_scaled,
        'X_test_orig': X_test_orig,  # スケーリング前のテストデータ
        'id_test': id_test,  # テストデータのID列
        'scaler': scaler
    }
    
    # SMOTETomek統計情報を追加
    if use_smotetomek and 'smotetomek_statistics' in kwargs:
        result['smotetomek_statistics'] = kwargs['smotetomek_statistics']
    
    return result

def evaluate_results(results):
    """結果の評価を行う関数（特異度と詳細評価を追加）"""
    # 特異度（Specificity）の計算
    cm = confusion_matrix(results['true_values'], results['predictions'])
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        # 多クラス分類の場合
        specificities = []
        n_classes = cm.shape[0]
        for i in range(n_classes):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm, i, axis=0)[:, i])
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        specificity = np.mean(specificities)
    
    metrics = {
        'Accuracy': accuracy_score(results['true_values'], results['predictions']),
        'Precision': precision_score(results['true_values'], results['predictions'], average='weighted'),
        'Recall': recall_score(results['true_values'], results['predictions'], average='weighted'),
        'Specificity': specificity,  # 特異度を追加
        'F1': f1_score(results['true_values'], results['predictions'], average='weighted')
    }
    
    # テストデータの元のインデックスを使用
    test_indices = results['test_indices']
    
    # 予測結果のデータフレームを作成
    pred_df = pd.DataFrame({
        'True_Label': pd.Series(results['true_values']).map({0: 'intact', 1: 'mci'}),
        'Predicted_Label': pd.Series(results['predictions']).map({0: 'intact', 1: 'mci'}),
        'Probability_mci': results['prediction_proba'][:, 1]
    }, index=test_indices)
    
    # 正誤判定列を追加
    pred_df['Correct'] = pred_df['True_Label'] == pred_df['Predicted_Label']
    
    # ID列の取得方法を変更
    id_test = results.get('id_test')
    
    # IDカラムがある場合は先頭に追加
    if id_test is not None:
        # ID列をデータフレームに追加してリセット
        id_name = 'InspectionDateAndId' if isinstance(id_test, pd.Series) and id_test.name == 'InspectionDateAndId' else 'Id'
        pred_df = pd.DataFrame({id_name: id_test}).join(pred_df)
        
        # インデックスをリセット
        pred_df = pred_df.reset_index(drop=True)
    
    # 詳細評価の追加
    print("\nテストデータに対する詳細評価:")
    print("混同行列:")
    print(cm)
    
    # 2クラス分類の場合、TN, FP, FN, TPの詳細を表示
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        print(f"真陰性(TN): {tn}, 偽陽性(FP): {fp}, 偽陰性(FN): {fn}, 真陽性(TP): {tp}")
        
        # 各種評価指標の詳細表示
        accuracy = (tp + tn) / (tp + tn + fp + fn)
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        print(f"正解率(Accuracy): {accuracy:.4f}")
        print(f"適合率(Precision): {precision:.4f}")
        print(f"再現率(Recall): {recall:.4f}")
        print(f"特異度(Specificity): {specificity:.4f}")
        print(f"F値(F1 Score): {f1:.4f}")
    
    return pred_df, metrics

def organize_result_files(data_file_name, output_dir):
    """
    現在の実行で生成されたファイルのみを新しいディレクトリに整理する関数
    1次元部分依存プロットとICEプロットも含める
    
    Parameters:
    -----------
    data_file_name : str
        データファイル名
    output_dir : str
        出力ディレクトリ
    
    Returns:
    --------
    str
        新しい出力ディレクトリのパス
    """
    # データファイル名から拡張子を除去したベース名を取得
    base_name = os.path.splitext(os.path.basename(data_file_name))[0]
    
    # 現在の時刻を取得（実行開始時刻として扱う）
    current_time = datetime.datetime.now()
    # 少し前の時刻（例：30分前）を計算し、その時刻以降に生成・更新されたファイルのみをコピー
    filter_time = current_time - datetime.timedelta(minutes=30)
    
    # 新しい出力ディレクトリを作成
    timestamp = current_time.strftime("%Y%m%d_%H%M%S")
    new_output_dir = f"{base_name}_{timestamp}"
    
    if not os.path.exists(new_output_dir):
        os.makedirs(new_output_dir)
        print(f"\n新しい出力ディレクトリを作成しました: {new_output_dir}")
    
    file_count = 0
    
    # output_dirが存在する場合のみ処理を実行
    if os.path.exists(output_dir):
        # interactionsディレクトリの処理
        interactions_dir = os.path.join(output_dir, "interactions")
        if os.path.exists(interactions_dir) and os.path.isdir(interactions_dir):
            new_interactions_dir = os.path.join(new_output_dir, "interactions")
            if not os.path.exists(new_interactions_dir):
                os.makedirs(new_interactions_dir)
            
            # 最近更新されたファイルのみをコピー
            for filename in os.listdir(interactions_dir):
                source_path = os.path.join(interactions_dir, filename)
                if os.path.isfile(source_path):
                    # ファイルの更新時刻を取得
                    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(source_path))
                    # フィルタ時間より新しいファイルのみコピー
                    if mod_time >= filter_time:
                        dest_path = os.path.join(new_interactions_dir, filename)
                        try:
                            shutil.copy2(source_path, dest_path)
                            file_count += 1
                            print(f"コピーしました: {dest_path}")
                        except Exception as e:
                            print(f"ファイルのコピー中にエラーが発生しました: {source_path} -> {e}")
        
        # rootディレクトリ内の重要なファイルをコピー
        important_patterns = [
            'top5_pdp_ice_1d.png',          # 部分依存プロット（全体）
            'manual_pdp_1d.png',            # 手動部分依存プロット
            'manual_pdp_ice_1d.png',        # 手動部分依存プロット（ICE付き）
            'pdp_*.png',                    # その他の部分依存プロット
            'confusion_matrix.png',         # 混同行列
            'roc_curve.png',                # ROC曲線
            'prediction_distribution.png',  # 予測分布
            'feature_importance.png',       # 特徴量重要度
            'feature_importance_*.csv',     # 特徴量重要度CSV
            'correlation_heatmap.png'       # 相関ヒートマップ
        ]
        
        # パターンに一致する最近のファイルをコピー
        for pattern in important_patterns:
            import glob
            pattern_path = os.path.join(output_dir, pattern)
            for source_file in glob.glob(pattern_path):
                if os.path.isfile(source_file):
                    # ファイルの更新時刻を取得
                    mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(source_file))
                    # フィルタ時間より新しいファイルのみコピー
                    if mod_time >= filter_time:
                        filename = os.path.basename(source_file)
                        dest_file = os.path.join(new_output_dir, filename)
                        try:
                            shutil.copy2(source_file, dest_file)
                            file_count += 1
                            print(f"コピーしました: {dest_file}")
                        except Exception as e:
                            print(f"ファイルのコピー中にエラーが発生しました: {source_file} -> {e}")
                            
        # その他の重要なディレクトリの処理
        other_important_dirs = ['centered_ice']
        for dir_name in other_important_dirs:
            source_dir = os.path.join(output_dir, dir_name)
            if os.path.exists(source_dir) and os.path.isdir(source_dir):
                new_dir = os.path.join(new_output_dir, dir_name)
                if not os.path.exists(new_dir):
                    os.makedirs(new_dir)
                
                # 最近更新されたファイルのみをコピー
                for filename in os.listdir(source_dir):
                    source_path = os.path.join(source_dir, filename)
                    if os.path.isfile(source_path):
                        # ファイルの更新時刻を取得
                        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(source_path))
                        # フィルタ時間より新しいファイルのみコピー
                        if mod_time >= filter_time:
                            dest_path = os.path.join(new_dir, filename)
                            try:
                                shutil.copy2(source_path, dest_path)
                                file_count += 1
                                print(f"コピーしました: {dest_path}")
                            except Exception as e:
                                print(f"ファイルのコピー中にエラーが発生しました: {source_path} -> {e}")
    
        # saved_modelディレクトリの処理
        saved_model_dir = os.path.join(output_dir, 'saved_model')
        if os.path.exists(saved_model_dir) and os.path.isdir(saved_model_dir):
            # 新しいディレクトリ内にsaved_modelディレクトリを作成
            new_saved_model_dir = os.path.join(new_output_dir, 'saved_model')
            if not os.path.exists(new_saved_model_dir):
                os.makedirs(new_saved_model_dir)
            
            # saved_modelディレクトリ内の最近のファイルをコピー
            for filename in os.listdir(saved_model_dir):
                source_path = os.path.join(saved_model_dir, filename)
                if os.path.isfile(source_path):
                    # base_nameが含まれるファイルのみを対象にする
                    if base_name in filename:
                        # ファイルの更新時刻を取得
                        mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(source_path))
                        # フィルタ時間より新しいファイルのみコピー
                        if mod_time >= filter_time:
                            dest_path = os.path.join(new_saved_model_dir, filename)
                            try:
                                shutil.copy2(source_path, dest_path)
                                file_count += 1
                                print(f"モデルファイルをコピーしました: {dest_path}")
                            except Exception as e:
                                print(f"モデルファイルのコピー中にエラーが発生しました: {source_path} -> {e}")
            
            # 同名のモデルフォルダがあれば同様に処理
            model_dir = os.path.join(saved_model_dir, base_name)
            if os.path.exists(model_dir) and os.path.isdir(model_dir):
                # ディレクトリの更新時刻を確認
                dir_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(model_dir))
                if dir_mod_time >= filter_time:
                    new_model_dir = os.path.join(new_saved_model_dir, base_name)
                    if not os.path.exists(new_model_dir):
                        os.makedirs(new_model_dir)
                    
                    # フォルダ内の最近のファイルをコピー
                    try:
                        for sub_filename in os.listdir(model_dir):
                            sub_source_path = os.path.join(model_dir, sub_filename)
                            if os.path.isfile(sub_source_path):
                                # ファイルの更新時刻を取得
                                sub_mod_time = datetime.datetime.fromtimestamp(os.path.getmtime(sub_source_path))
                                # フィルタ時間より新しいファイルのみコピー
                                if sub_mod_time >= filter_time:
                                    sub_dest_path = os.path.join(new_model_dir, sub_filename)
                                    try:
                                        shutil.copy2(sub_source_path, sub_dest_path)
                                        file_count += 1
                                        print(f"モデルサブファイルをコピーしました: {sub_dest_path}")
                                    except Exception as e:
                                        print(f"モデルサブファイルのコピー中にエラーが発生しました: {sub_source_path} -> {e}")
                    except Exception as e:
                        print(f"モデルディレクトリの処理中にエラーが発生しました: {model_dir} -> {e}")
    
    else:
        print(f"警告: 出力ディレクトリ {output_dir} が見つかりません")
    
    print(f"{file_count}個の新しいファイルを {new_output_dir} にコピーしました")
    return new_output_dir


def run_simple_analysis(df, model_class=None, use_smote=False, use_undersampling=False, 
                       use_simple_undersampling=False, use_smotetomek=False, 
                       random_state=42, output_dir=None, data_file_name=None, 
                       organize_files=True, evaluate_thresholds=True,
                       create_ale_plots=False, ale_features=5, ale_include_2d=False,
                       ale_vs_pdp=False, **kwargs):
    """単一の学習・評価を行う分析を実行する関数（拡張機能対応版）"""
    
    # 新しい出力ディレクトリをここで作成
    original_output_dir = output_dir  # 元の出力ディレクトリを保存
    new_output_dir = None
    
    if output_dir and data_file_name:
        base_name = os.path.splitext(os.path.basename(data_file_name))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        new_output_dir = ("G:/共有ドライブ/GAP_長寿研/user/iwamoto/視線の動きの俊敏さ/result2/"+f"{base_name}_{timestamp}")
        
        if not os.path.exists(new_output_dir):
            os.makedirs(new_output_dir)
            print(f"\n新しい出力ディレクトリを作成しました: {new_output_dir}")
            
        # interactionsとsaved_modelサブディレクトリも事前に作成
        interactions_dir = os.path.join(new_output_dir, "interactions")
        saved_model_dir = os.path.join(new_output_dir, "saved_model")
        ale_dir = os.path.join(new_output_dir, "ale_analysis")  # ALE用ディレクトリも追加
        if not os.path.exists(interactions_dir):
            os.makedirs(interactions_dir)
        if not os.path.exists(saved_model_dir):
            os.makedirs(saved_model_dir)
        if not os.path.exists(ale_dir):
            os.makedirs(ale_dir)
            
        # 処理結果の出力先を新しいディレクトリに設定
        output_dir = new_output_dir
    
    # データの前処理
    if 'Target' in df.columns and 'target' not in df.columns:
        df = df.rename(columns={'Target': 'target'})
    
    # 使用手法の表示
    method_description = []
    if use_undersampling:
        base_model = kwargs.get('base_model', 'lightgbm')
        n_bags = kwargs.get('n_bags', 10)
        method_description.append(f"UndersamplingBagging ({base_model}, {n_bags} bags)")
    elif use_simple_undersampling:
        base_model = kwargs.get('base_model', 'lightgbm')
        method_description.append(f"SimpleUndersampling ({base_model})")
    elif use_smotetomek:
        if ENHANCED_FEATURES_AVAILABLE:
            method_description.append("SMOTETomek (SMOTE + Tomek Links)")
        else:
            method_description.append("SMOTETomek (機能無効 - 通常SMOTE使用)")
            use_smote = True
            use_smotetomek = False
    elif model_class is not None:
        method_description.append(f"{model_class.__name__}")
    else:
        raise ValueError("model_class、use_undersampling、use_simple_undersampling、またはuse_smotetomekのいずれかを指定してください。")
    
    if use_smote and not use_smotetomek:
        method_description.append("SMOTE")
    
    print(f"\nUsing {', '.join(method_description)}...")
    
    print("データの前処理を開始...")
    try:
        features, target, id_column = prepare_data(df)  # id_columnも取得するように変更
    except ValueError as e:
        if "too many values to unpack" in str(e):
            # 古い関数と互換性を保つ
            features, target = prepare_data(df)
            id_column = None
            print("警告: IDカラムが取得できませんでした。古い関数シグネチャを使用します。")
        else:
            raise
    
    # データが空の場合はエラー
    if len(features) == 0 or len(target) == 0:
        raise ValueError("前処理後のデータが空です。データを確認してください。")
    
    # ターゲットに一意な値が1つしかない場合はエラー
    if len(target.unique()) < 2:
        raise ValueError(f"ターゲット変数に一意な値が{len(target.unique())}種類しかありません。少なくとも2クラス必要です。")
    
    # 可視化のためのインスタンス作成 - 新しい出力ディレクトリを使用
    visualizer = EyeTrackingVisualizer(output_dir=output_dir)
    
    # 特徴量と目的変数の関係の可視化
    print("\n特徴量と目的変数の関係を分析します...")
    # 親クラスのメソッドを使用するように変更
    if hasattr(visualizer, 'plot_feature_correlations'):
        # RegressionVisualizerのメソッドを使用
        visualizer.plot_feature_correlations(features, target)
    elif hasattr(visualizer, 'visualize_feature_target_relationships'):
        # 旧メソッドが存在する場合
        visualizer.visualize_feature_target_relationships(features, target)
    else:
        print("警告: 特徴量と目的変数の関係を分析するメソッドが見つかりません。このステップをスキップします。")
    
    # 元のデータに視線追跡データが含まれている場合はサッカードの可視化も行う
    has_saccade = any('saccade' in col.lower() for col in df.columns)
    has_eye_tracking = any('EyeCenterAngle' in col for col in df.columns)
    
    if has_saccade:
        print("\nサッカード特徴量の分析を行います...")
        # すでにtarget列があるのでそのまま使用
        visualizer.visualize_saccade_features(df)
    
    if has_eye_tracking:
        print("\n視線軌跡の分析を行います...")
        try:
            eye_data_sample = df.sample(min(1000, len(df)), random_state=random_state)
            visualizer.visualize_eye_movement_trajectories(eye_data_sample)
        except Exception as e:
            print(f"視線軌跡の可視化中にエラーが発生しました: {e}")
    
    print("\nモデルの学習を開始...")
    try:
        # id_columnも渡すように変更、SMOTETomekパラメータも追加
        results = train_model(
        features, target, id_column,
        model_class=model_class, 
        use_smote=use_smote, 
        use_undersampling=use_undersampling,
        use_simple_undersampling=use_simple_undersampling,
        use_smotetomek=use_smotetomek,  # use_smotetomekを直接使用
        random_state=random_state,
        smotetomek_strategy=kwargs.get('smotetomek_strategy', 'auto'),
        smotetomek_k_neighbors=kwargs.get('smotetomek_k_neighbors', 5),
        tomek_strategy=kwargs.get('tomek_strategy', 'auto'),
        base_model=kwargs.get('base_model', 'lightgbm'),
        n_bags=kwargs.get('n_bags', 5)
        )   
        
    except TypeError as e:
        # 古い関数シグネチャとの互換性を保つ
        if "got an unexpected keyword argument" in str(e) or "takes" in str(e) and "positional argument" in str(e):
            print("警告: 古い関数シグネチャを使用します（id_columnまたはSMOTETomekなし）")
            results = train_model(
                features, target,
                model_class=model_class, 
                use_smote=use_smote, 
                use_undersampling=use_undersampling,
                use_simple_undersampling=use_simple_undersampling,
                random_state=random_state,
                **kwargs
            )
            
            # ID列を取得して追加
            if id_column is not None:
                # テストインデックスを使用してIDをフィルタリング
                test_indices = results['test_indices']
                results['id_test'] = id_column.loc[test_indices]
        else:
            raise
    
    # モデル名の決定 - 先に決定して他の場所で使用できるようにする
    if use_undersampling:
        model_name = f"usbag_{kwargs.get('base_model', 'lightgbm')}"
    elif use_simple_undersampling:
        model_name = f"usimple_{kwargs.get('base_model', 'lightgbm')}"
    elif use_smotetomek:
        if ENHANCED_FEATURES_AVAILABLE:
            model_name = f"smotetomek_{results['model'].__class__.__name__.lower()}"
        else:
            model_name = f"smote_{results['model'].__class__.__name__.lower()}"
    else:
        model_name = results['model'].__class__.__name__.lower()
        
    smote_suffix = "_smote" if use_smote and not use_smotetomek else ""
    
    print("\n結果の評価を開始...")
    predictions_df, metrics = evaluate_results(results)
    
    # ここに閾値評価のコードを追加 (evaluate_thresholdsがTrueの場合)
    if evaluate_thresholds:
        from visualization.threshold_evaluator import ThresholdEvaluator
        
        print("\n閾値の評価を開始...")
        threshold_evaluator = ThresholdEvaluator(output_dir=output_dir)
        
        # 実際のラベルとMCIクラスの予測確率を使用
        y_true = results['true_values']
        y_proba = results['prediction_proba'][:, 1]  # MCIクラスの予測確率
        
        try:
            optimal_thresholds, metrics_df = threshold_evaluator.create_threshold_recommendation_report(
                y_true, y_proba
            )
            
            # 最適閾値の表示
            print("\n最適閾値の推奨値:")
            for method, values in optimal_thresholds.items():
                print(f"\n最適化方法: {method}")
                print(f"  閾値: {values['threshold']:.3f}")
                print(f"  適合率: {values['precision']:.3f}")
                print(f"  再現率: {values['recall']:.3f}")
                print(f"  キュー率: {values['queue_rate']:.3f}")
                print(f"  F1スコア: {values['f1']:.3f}")
        except Exception as e:
            print(f"閾値評価中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    # 再現率と特異度の閾値による変化の可視化
    print("\n再現率と特異度の閾値による変化を可視化...")
    try:
        # データの準備
        y_true = results['true_values']
        y_proba = results['prediction_proba'][:, 1]  # 陽性クラスの確率
        
        # グラフの作成と保存
        threshold_evaluator.plot_recall_specificity_curve(
            y_true, 
            y_proba, 
            output_dir=output_dir,
            model_name=model_name,
            smote_suffix=smote_suffix
        )
        
    except Exception as e:
        print(f"再現率と特異度の閾値による変化の可視化中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

    # ROC曲線と混同行列の作成・保存
    print("\nROC曲線と混同行列の作成・保存...")
    try:
        # データの準備
        y_true = results['true_values']
        y_pred = results['predictions']
        y_proba = results['prediction_proba'][:, 1]  # 陽性クラスの確率

        # 混同行列の計算
        cm = confusion_matrix(y_true, y_pred)
        
        # 評価指標の計算
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred)
        recall = recall_score(y_true, y_pred)
        f1 = f1_score(y_true, y_pred)
        
        # 特異度の計算
        tn, fp, fn, tp = cm.ravel()
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        
        # 評価指標をファイルに保存
        metrics_path = os.path.join(output_dir, f'evaluation_metrics_{model_name}{smote_suffix}.txt')
        with open(metrics_path, 'w') as f:
            f.write(f"正解率(Accuracy): {accuracy:.4f}\n")
            f.write(f"適合率(Precision): {precision:.4f}\n")
            f.write(f"再現率(Recall): {recall:.4f}\n")
            f.write(f"特異度(Specificity): {specificity:.4f}\n")
            f.write(f"F値(F1 Score): {f1:.4f}\n")
            f.write(f"\n混同行列:\n")
            f.write(f"TN: {tn}, FP: {fp}\n")
            f.write(f"FN: {fn}, TP: {tp}\n")
        print(f"評価指標を保存しました: {metrics_path}")
        
        # ROC曲線の作成
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        # ROC曲線のプロット
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.3f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('偽陽性率 (False Positive Rate)')
        plt.ylabel('真陽性率 (True Positive Rate)')
        plt.title('ROC曲線')
        plt.legend(loc="lower right")
        
        # ROC曲線の保存
        roc_path = os.path.join(output_dir, f'roc_curve_{model_name}{smote_suffix}.png')
        plt.savefig(roc_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"ROC曲線を保存しました: {roc_path}")
        
        # 混同行列のヒートマップ作成
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                  xticklabels=['Negative (0)', 'Positive (1)'],
                  yticklabels=['Negative (0)', 'Positive (1)'])
        plt.xlabel('予測ラベル')
        plt.ylabel('真のラベル')
        plt.title('混同行列')
        
        # 混同行列の保存
        cm_path = os.path.join(output_dir, f'confusion_matrix_{model_name}{smote_suffix}.png')
        plt.savefig(cm_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"混同行列を保存しました: {cm_path}")
        
    except Exception as e:
        print(f"ROC曲線と混同行列の作成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    # 予測確率分布とヒートマップの保存を追加
    print("\n予測確率分布と相関ヒートマップの作成・保存...")
    try:
        # 1. 予測確率分布の作成
        plt.figure(figsize=(12, 8))
        
        # データの準備
        y_true = results['true_values']
        y_proba = results['prediction_proba'][:, 1]  # 陽性クラスの確率
        
        # ラベルをマッピング
        true_labels = pd.Series(y_true).map({0: 'intact', 1: 'mci'})
        
        # ヒストグラムの作成
        sns.histplot(
            data=pd.DataFrame({'Probability_mci': y_proba, 'True Label': true_labels}),
            x='Probability_mci',
            hue='True Label',
            element='bars',
            stat='count',
            bins=10,
            alpha=0.7,
            palette={'intact': 'skyblue', 'mci': 'lightsalmon'}
        )
        
        # タイトルと軸ラベルの設定
        plt.title('Prediction Probability Distribution by True Label', fontsize=16)
        plt.xlabel('Probability of MCI', fontsize=12)
        plt.ylabel('Count', fontsize=12)
        
        # 予測確率分布の保存
        dist_path = os.path.join(output_dir, f'prediction_distribution_{model_name}{smote_suffix}.png')
        plt.savefig(dist_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"予測確率分布を保存しました: {dist_path}")
        
        # 2. 相関ヒートマップの作成
        # 元データの取得
        features = results.get('features')
        target = results.get('target')
        
        if features is not None and target is not None:
            # 特徴量と目的変数を結合
            data = features.copy()
            data['target'] = target
            
            # 相関行列の計算
            corr = data.corr()
            
            # ヒートマップの作成
            plt.figure(figsize=(20, 18))
            mask = np.zeros_like(corr, dtype=bool)
            mask[np.triu_indices_from(mask, k=1)] = True
            
            # ヒートマップの描画
            sns.heatmap(
                corr, 
                mask=mask,
                cmap=sns.diverging_palette(220, 10, as_cmap=True),
                vmax=1.0, 
                vmin=-1.0,
                center=0,
                square=True, 
                linewidths=.5, 
                annot=True,
                fmt='.2f',
                annot_kws={"size": 8}
            )
            
            # タイトルの設定
            plt.title('特徴量間と目的変数の相関ヒートマップ', fontsize=16)
            
            # 相関ヒートマップの保存
            corr_path = os.path.join(output_dir, f'correlation_heatmap_{model_name}{smote_suffix}.png')
            plt.savefig(corr_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"相関ヒートマップを保存しました: {corr_path}")
            
    except Exception as e:
        print(f"予測確率分布と相関ヒートマップの作成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
    
    # ALEプロット作成（新機能）
    if create_ale_plots and ENHANCED_FEATURES_AVAILABLE:
        print("\nALE（Accumulated Local Effects）プロット作成中...")
        try:
            # ALEプロッター作成
            ale_dir = os.path.join(output_dir, 'ale_analysis')
            ale_plotter = ALEPlotter(
                model=results['model'],
                X=results['X_test'],
                feature_names=list(results['features'].columns),
                output_dir=ale_dir
            )
            
            # ALE分析実行
            ale_results = ale_plotter.create_ale_dashboard(
                n_features=ale_features,
                include_2d=ale_include_2d
            )
            
            # ALE vs PDP比較
            if ale_vs_pdp:
                print("ALE vs PDP 比較分析中...")
                top_features = ale_results['features_analyzed'][:min(3, len(ale_results['features_analyzed']))]
                for feature in top_features:
                    try:
                        fig = ale_plotter.compare_ale_vs_pdp(feature)
                        if fig:
                            plt.close(fig)
                    except Exception as e:
                        print(f"ALE vs PDP比較エラー ({feature}): {e}")
            
            # 結果に追加
            results['ale_analysis'] = ale_results
            
            # ALEレポート作成
            from ale_plotter import create_ale_analysis_report
            create_ale_analysis_report(ale_plotter, ale_results['features_analyzed'], ale_dir)
            
            print("ALE分析完了")
            print(f"  1D プロット: {len(ale_results['ale_1d_plots'])}個")
            print(f"  2D プロット: {len(ale_results['ale_2d_plots'])}個")
            
        except Exception as e:
            print(f"ALE分析中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
    
    elif create_ale_plots and not ENHANCED_FEATURES_AVAILABLE:
        print("警告: ALE機能が利用できません。ale_plotter.pyを確認してください。")
    
    # 結果の表示
    print("\n最適なパラメータ:")
    for param, value in results['best_params'].items():
        print(f"{param}: {value}")
    
    print("\n性能指標:")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    print("\n特徴量の重要度（上位10件）:")
    print(results['feature_importance'].head(10))
    
    # SMOTETomek統計の表示
    if use_smotetomek and 'smotetomek_statistics' in results:
        print("\nSMOTETomek統計:")
        stats = results['smotetomek_statistics']
        print(f"  元データサイズ: {stats['original_size']}")
        print(f"  処理後サイズ: {stats['resampled_size']}")
        print(f"  変化: {stats['size_change']:+d}")
        print(f"  元クラス分布: {stats['original_class_counts']}")
        print(f"  処理後クラス分布: {stats['resampled_class_counts']}")
    
    # 結果の可視化
    print("\nモデル評価の可視化...")
    
    # 上位特徴量の分布可視化の実行
    visualize_top_features(results, visualizer, n_top=5)
    
    # 部分依存グラフの作成
    print("\n部分依存グラフの作成...")
    try:
        # 直接create_partial_dependence_plotsメソッドを呼び出す
        visualizer.create_partial_dependence_plots(results)
    except Exception as e:
        print(f"部分依存グラフの作成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()  # エラーの詳細を表示
    
    # 結果をまとめる
    results['predictions_df'] = predictions_df
    
    # 出力ディレクトリがある場合は結果を保存 - 新しい出力ディレクトリに直接保存
    if output_dir:
        # 予測結果の保存
        predictions_path = os.path.join(output_dir, f'predictions_{model_name}{smote_suffix}.csv')
        try:
            predictions_df.to_csv(predictions_path)
            print(f"予測結果を保存しました: {predictions_path}")
        except Exception as e:
            print(f"予測結果の保存中にエラーが発生しました: {e}")
        
        # 特徴量の重要度の保存
        importance_path = os.path.join(output_dir, f'feature_importance_{model_name}{smote_suffix}.csv')
        try:
            results['feature_importance'].to_csv(importance_path)
            print(f"特徴量重要度を保存しました: {importance_path}")
        except Exception as e:
            print(f"特徴量重要度の保存中にエラーが発生しました: {e}")
        
        # 最適パラメータを保存
        params_path = os.path.join(output_dir, f'best_parameters_{model_name}{smote_suffix}.txt')
        try:
            with open(params_path, 'w') as f:
                for param, value in results['best_params'].items():
                    f.write(f"{param}: {value}\n")
                
                # SMOTETomek統計も保存
                if use_smotetomek and 'smotetomek_statistics' in results:
                    f.write(f"\n=== SMOTETomek統計 ===\n")
                    stats = results['smotetomek_statistics']
                    f.write(f"元データサイズ: {stats['original_size']}\n")
                    f.write(f"処理後サイズ: {stats['resampled_size']}\n")
                    f.write(f"変化: {stats['size_change']}\n")
                    f.write(f"元クラス分布: {stats['original_class_counts']}\n")
                    f.write(f"処理後クラス分布: {stats['resampled_class_counts']}\n")
                    f.write(f"SMOTE設定: {stats['smote_settings']}\n")
                    f.write(f"Tomek設定: {stats['tomek_settings']}\n")
            
            print(f"最適パラメータを保存しました: {params_path}")
        except Exception as e:
            print(f"最適パラメータの保存中にエラーが発生しました: {e}")
    
    # モデルの保存 - 新しい出力ディレクトリのsaved_modelサブディレクトリに保存
    if output_dir and data_file_name:
        model_base_name = os.path.splitext(os.path.basename(data_file_name))[0]
        saved_model_dir = os.path.join(output_dir, 'saved_model')
        
        try:
            save_model(
                model=results['model'],
                features=results['features'],
                target=results['target'],
                scaler=results['scaler'],
                output_path=saved_model_dir,
                model_name=model_base_name
            )
            print(f"モデルを保存しました: {saved_model_dir}/{model_base_name}")
        except Exception as e:
            print(f"モデルの保存中にエラーが発生しました: {e}")

    print(f"\n分析が完了しました。すべての結果は {output_dir} に保存されました。")
    
    # 必ず2つの値を返す
    return results, new_output_dir

def run_enhanced_comprehensive_analysis(df, model_class=None, output_dir='enhanced_result', 
                                       random_state=42, data_file_name=None, **kwargs):
    """
    SMOTETomek + ALE + 従来手法の包括的比較分析を実行
    """
    if not ENHANCED_FEATURES_AVAILABLE:
        print("エラー: 拡張機能が利用できません。")
        print("smotetomek_processor.py, ale_plotter.py, enhanced_analysis_integration.py を確認してください。")
        return None
    
    print("=== 包括的拡張分析開始 ===")
    
    # EnhancedAnalysisRunnerを使用
    runner = EnhancedAnalysisRunner(output_dir, random_state)
    
    results = runner.run_comprehensive_analysis(
        df=df,
        model_class=model_class,
        data_file_name=data_file_name,
        use_baseline=True,
        use_smote=True,
        use_smotetomek=True,
        use_ale=True,
        **kwargs
    )
    
    return results

def visualize_top_features(results, visualizer, n_top=5):
    """
    上位n_top個の特徴量の分布を可視化（表示なし、保存のみ）
    
    Parameters:
    -----------
    results : dict
        結果データを含む辞書
    visualizer : EyeTrackingVisualizer
        可視化用のインスタンス
    n_top : int, default=5
        可視化する上位特徴量の数
    """
    print(f"\n上位{n_top}個の特徴量の分布を可視化します（保存のみ）...")
    
    try:
        # 必要なデータの取得
        X_test = results.get('X_test')
        y_true = results.get('true_values')
        feature_importance = results.get('feature_importance')
        
        if X_test is None or y_true is None or feature_importance is None:
            print("警告: 必要なデータが不足しています")
            return
        
        # 特徴量重要度を確認
        print("特徴量重要度の情報:")
        print(feature_importance.head(n_top))
        
        # 上位n個の特徴量を取得
        top_features = feature_importance.head(n_top)['feature'].tolist()
        top_features = [f for f in top_features if f in X_test.columns]
        
        if not top_features:
            print("警告: 有効な特徴量が見つかりません")
            return
            
        print(f"可視化する特徴量: {top_features}")
        
        # 出力ディレクトリの確認
        if not visualizer.output_dir:
            print("警告: 出力ディレクトリが設定されていないため、保存をスキップします")
            return
        
        # 各特徴量の分布をプロット
        for feature in top_features:
            # データの準備
            feature_data = X_test[feature]
            
            # 欠損値のチェック
            if feature_data.isna().any():
                print(f"警告: 特徴量 '{feature}' に欠損値が含まれています。欠損値を除外します。")
                mask = ~feature_data.isna()
                feature_data = feature_data[mask]
                y_feature = y_true[mask] if isinstance(y_true, pd.Series) else y_true[mask.values]
            else:
                y_feature = y_true
            
            # ランク表示用
            rank = top_features.index(feature) + 1
            
            # 図の作成（表示しない）
            plt.figure(figsize=(10, 6))
            
            # ヒストグラムの作成
            df = pd.DataFrame({
                'feature_value': feature_data,
                'true_label': y_feature
            })
            
            sns.histplot(
                data=df, 
                x='feature_value',
                hue='true_label',
                element='bars',
                stat='count',
                bins=30,
                alpha=0.7,
                palette={0: 'skyblue', 1: 'sandybrown'}
            )
            
            # タイトルと軸ラベルの設定
            plt.title(f'{feature} 分布 (上位{rank}位)', fontsize=16)
            plt.xlabel(feature, fontsize=12)
            plt.ylabel('Count', fontsize=12)
            
            # 凡例の設定
            plt.legend(title='Target', labels=['target=0', 'target=1'], fontsize=12)
            
            # 図を保存
            clean_name = feature.replace(' ', '_').replace('/', '_').replace('\\', '_')
            clean_name = ''.join(c for c in clean_name if c.isalnum() or c in '_-.')
            
            save_path = os.path.join(visualizer.output_dir, f'top_feature_{rank}_{clean_name}_distribution.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"保存しました: {save_path}")
            
            # 図を閉じる（表示しない）
            plt.close()
        
        print(f"上位{len(top_features)}個の特徴量の分布図を保存しました")
        
    except Exception as e:
        print(f"上位特徴量の分布図作成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

def run_cv_analysis_with_smote(df, model_class=None, n_splits=5, use_smote=False, 
                              use_undersampling=False, use_simple_undersampling=False,
                              output_dir='result', random_state=42, **kwargs):
    """
    CrossValidator を使用した分析を実行するラッパー関数（SMOTEオプション付き）
    """
    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    smote_str = "（SMOTEあり）" if use_smote else ""
    undersampling_str = ""
    if use_undersampling:
        undersampling_str = "（UndersamplingBaggingあり）"
    elif use_simple_undersampling:
        undersampling_str = "（シンプルアンダーサンプリングあり）"
    
    print(f"\nCrossValidator を使用した {n_splits}分割交差検証を開始...{smote_str}{undersampling_str}")
    
    # データの前処理（修正版：3つの値を受け取る）
    print("データの前処理を開始...")
    try:
        features, target, id_column = prepare_data(df)
    except ValueError as e:
        if "too many values to unpack" in str(e):
            # 古いprepare_data関数との互換性
            print("警告: 古いprepare_data関数を使用しています")
            features, target = prepare_data(df)
            id_column = None
        else:
            raise
    
    # データが空の場合はエラー
    if len(features) == 0 or len(target) == 0:
        raise ValueError("前処理後のデータが空です。データを確認してください。")
    
    # ターゲットに一意な値が1つしかない場合はエラー
    if len(target.unique()) < 2:
        raise ValueError(f"ターゲット変数に一意な値が{len(target.unique())}種類しかありません。少なくとも2クラス必要です。")
    
    # 可視化のためのインスタンス作成
    visualizer = EyeTrackingVisualizer(output_dir=output_dir)
    
    # 特徴量と目的変数の関係の可視化
    print("\n特徴量と目的変数の関係を分析します...")
    try:
        if hasattr(visualizer, 'plot_feature_correlations'):
            visualizer.plot_feature_correlations(features, target)
        elif hasattr(visualizer, 'visualize_feature_target_relationships'):
            visualizer.visualize_feature_target_relationships(features, target)
        else:
            print("警告: 特徴量と目的変数の関係を分析するメソッドが見つかりません。")
    except Exception as e:
        print(f"特徴量可視化中にエラー: {e}")
    
    # 元のデータに視線追跡データが含まれている場合はサッカードの可視化も行う
    has_saccade = any('saccade' in col.lower() for col in df.columns)
    has_eye_tracking = any('EyeCenterAngle' in col for col in df.columns)
    
    if has_saccade:
        print("\nサッカード特徴量の分析を行います...")
        try:
            visualizer.visualize_saccade_features(df)
        except Exception as e:
            print(f"サッカード分析中にエラー: {e}")
    
    if has_eye_tracking:
        print("\n視線軌跡の分析を行います...")
        try:
            eye_data_sample = df.sample(min(1000, len(df)), random_state=random_state)
            visualizer.visualize_eye_movement_trajectories(eye_data_sample)
        except Exception as e:
            print(f"視線軌跡の可視化中にエラーが発生しました: {e}")
    
    # UndersamplingBaggingModelを使用する場合
    if use_undersampling:
        base_model = kwargs.get('base_model', 'lightgbm')
        n_bags = kwargs.get('n_bags', 10)
        print(f"UndersamplingBaggingModelを使用した交差検証を実行します（ベースモデル: {base_model}、バッグ数: {n_bags}）")
        
        # UndersamplingBaggingModelのクラスメソッドを使用
        try:
            from models.undersampling_bagging_model import UndersamplingBaggingModel
            oof_preds, scores = UndersamplingBaggingModel.run_cv(
                X=features, 
                y=target, 
                base_model=base_model,
                n_bags=n_bags,
                n_splits=n_splits,
                random_state=random_state
            )
        except ImportError as e:
            print(f"UndersamplingBaggingModelのインポートエラー: {e}")
            return None, None
        
        # 結果の表示
        print("\n交差検証の結果:")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
            
        # 結果を保存
        model_name = f"usbag_{base_model}"
        smote_suffix = "_smote" if use_smote else ""
        cv_results_df = pd.DataFrame({
            'true': target,
            'pred': oof_preds
        })
        
        if output_dir:
            # 予測結果の保存
            cv_results_df.to_csv(f'{output_dir}/cv_{model_name}{smote_suffix}_{n_splits}fold_predictions.csv')
            
            # 評価指標の保存
            with open(f'{output_dir}/cv_{model_name}{smote_suffix}_{n_splits}fold_metrics.txt', 'w') as f:
                for metric, score in scores.items():
                    f.write(f"{metric}: {score:.4f}\n")
        
        return None, scores  # CrossValidatorの代わりに結果のみを返す
    
    # SimpleUndersamplingModelを使用する場合
    elif use_simple_undersampling:
        base_model = kwargs.get('base_model', 'lightgbm')
        print(f"シンプルアンダーサンプリングモデルを使用した交差検証を実行します（ベースモデル: {base_model}）")
        
        # UndersamplingModelのクラスメソッドを使用
        try:
            from models.undersampling_model import UndersamplingModel
            oof_preds, scores = UndersamplingModel.run_cv(
                X=features, 
                y=target, 
                base_model=base_model,
                n_splits=n_splits,
                random_state=random_state
            )
        except ImportError as e:
            print(f"UndersamplingModelのインポートエラー: {e}")
            return None, None
        
        # 結果の表示
        print("\n交差検証の結果:")
        for metric, score in scores.items():
            print(f"{metric}: {score:.4f}")
            
        # 結果を保存
        model_name = f"usimple_{base_model}"
        smote_suffix = "_smote" if use_smote else ""
        cv_results_df = pd.DataFrame({
            'true': target,
            'pred': oof_preds
        })
        
        if output_dir:
            # 予測結果の保存
            cv_results_df.to_csv(f'{output_dir}/cv_{model_name}{smote_suffix}_{n_splits}fold_predictions.csv')
            
            # 評価指標の保存
            with open(f'{output_dir}/cv_{model_name}{smote_suffix}_{n_splits}fold_metrics.txt', 'w') as f:
                for metric, score in scores.items():
                    f.write(f"{metric}: {score:.4f}\n")
        
        return None, scores  # CrossValidatorの代わりに結果のみを返す
    
    # 通常のCrossValidatorを使用する場合
    try:
        from models.cross_validator import CrossValidator
        
        # CrossValidator の設定と実行
        validator = CrossValidator(n_splits=n_splits, random_state=random_state)
        
        # SMOTEを使用する場合は、CrossValidatorクラスに処理を追加する必要があります
        # このコードでは簡易的にrun_cross_validationメソッドを直接呼び出します
        if use_smote:
            print("注意: CrossValidatorクラスでSMOTEはサポートされていません。")
            print("SMOTEを適用するには、CrossValidatorクラスを修正する必要があります。")
        
        # 交差検証の実行
        if model_class is None:
            print("エラー: model_classが指定されていません。")
            return None, None
            
        results = validator.run_cross_validation(features, target, model_class)
        
        # 結果の表示と可視化
        validator.print_results()
        validator.plot_results()
        
        # 結果の保存
        model_name = model_class.__name__.lower()
        smote_suffix = "_smote" if use_smote else ""
        validator.save_results(
            prefix=f'cv_{model_name}{smote_suffix}_{n_splits}fold',
            output_dir=output_dir
        )
        
        return validator, results
        
    except ImportError as e:
        print(f"CrossValidatorのインポートエラー: {e}")
        return None, None
    except Exception as e:
        print(f"CrossValidator実行中にエラー: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def create_usbag_model(base_model='lightgbm', n_bags=10, random_state=42):
    """UndersamplingBaggingModelを作成するヘルパー関数"""
    return {
        'model': UndersamplingBaggingModel(
            base_model=base_model,
            n_bags=n_bags,
            random_state=random_state
        ),
        'name': f'usbag_{base_model}',
        'params': {
            'base_model': base_model,
            'n_bags': n_bags
        }
    }


def create_simple_us_model(base_model='lightgbm', random_state=42):
    """シンプルなUndersamplingModelを作成するヘルパー関数"""
    return {
        'model': UndersamplingModel(
            base_model=base_model,
            random_state=random_state
        ),
        'name': f'usimple_{base_model}',
        'params': {
            'base_model': base_model
        }
    }


if __name__ == "__main__":
    # コマンドライン引数の設定
    parser = argparse.ArgumentParser(description='視線追跡データを用いた機械学習モデルの実行')
    
    # 既存の基本引数
    parser.add_argument('--cv', action='store_true', help='交差検証を実行する')
    parser.add_argument('--smote', action='store_true', help='SMOTEを使用する')
    parser.add_argument('--no-smote', dest='smote', action='store_false', help='SMOTEを使用しない')
    parser.add_argument('--undersampling', action='store_true', help='UndersamplingBaggingを使用する')
    parser.add_argument('--simple-undersampling', action='store_true', help='シンプルなアンダーサンプリング（バギングなし）を使用する')
    parser.add_argument('--base-model', type=str, default='catboost', 
                        choices=['lightgbm', 'xgboost', 'random_forest', 'catboost'], 
                        help='UndersamplingBaggingで使用するベースモデル')
    parser.add_argument('--n-bags', type=int, default=5, help='UndersamplingBaggingのバッグ数')
    parser.add_argument('--splits', type=int, default=5, help='交差検証の分割数')
    parser.add_argument('--model', type=str, default='catboost', 
                        choices=['lightgbm', 'xgboost', 'random_forest', 'catboost'],
                        help='使用するモデル')
    parser.add_argument('--random-state', type=int, default=42, help='乱数シード')
    parser.add_argument('--data-path', type=str, 
                       default="data",
                       help='データファイルのパス')
    parser.add_argument('--data-file', type=str, 
                       default="cyozyuken_b_noeye.csv",
                       help='データファイル名')
    parser.add_argument('--output-dir', type=str, 
                       default="result",
                       help='結果出力ディレクトリ')
    parser.add_argument('--viz-only', action='store_true', help='可視化のみを実行（学習なし）')
    parser.add_argument('--no-pdp', dest='pdp', action='store_false', help='部分依存グラフを作成しない')
    parser.add_argument('--no-save', dest='save_plots', action='store_false', help='プロットをファイルに保存しない')
    parser.add_argument('--no-organize', dest='organize_files', action='store_false', 
                        help='結果ファイルを整理しない')
    parser.add_argument('--holdout-cv', action='store_true', help='テストデータ保持+CVを実行')
    
    # SMOTETomek関連の引数（新機能）
    parser.add_argument('--smotetomek', action='store_true', 
                       help='SMOTETomekを使用する（SMOTEとTomek Linksの組み合わせ）')
    parser.add_argument('--smotetomek-k-neighbors', type=int, default=5,
                       help='SMOTETomekのSMOTE部分で使用する近傍数 (default: 5)')
    parser.add_argument('--smotetomek-strategy', type=str, default='auto',
                       choices=['auto', 'minority', 'majority'],
                       help='SMOTETomekのサンプリング戦略 (default: auto)')
    parser.add_argument('--tomek-strategy', type=str, default='auto',
                       choices=['auto', 'majority', 'all'],
                       help='Tomek Linksのサンプリング戦略 (default: auto)')
    
    # ALE関連の引数（新機能）
    parser.add_argument('--ale-plots', action='store_true',
                       help='ALE（Accumulated Local Effects）プロットを作成する')
    parser.add_argument('--ale-features', type=int, default=5,
                       help='ALE分析する上位特徴量数 (default: 5)')
    parser.add_argument('--ale-include-2d', action='store_true',
                       help='2次元ALEプロットも作成する')
    parser.add_argument('--ale-vs-pdp', action='store_true',
                       help='ALEとPDPの比較プロットを作成する')
    
    # 拡張分析関連の引数（新機能）
    parser.add_argument('--enhanced-analysis', action='store_true',
                       help='SMOTETomek + ALE + 従来手法の包括的比較分析を実行')
    parser.add_argument('--compare-all-methods', action='store_true',
                       help='ベースライン、SMOTE、SMOTETomekの全手法を比較')
    
    # デフォルト設定
    parser.set_defaults(smote=False, undersampling=False, simple_undersampling=False, 
                       pdp=True, save_plots=True, viz_only=False, organize_files=True)
    # コマンドライン引数の追加
    parser.add_argument('--borderline-smote', action='store_true',
                    help='Borderline SMOTEを使用する')
    parser.add_argument('--borderline-type', type=int, default=1, choices=[1, 2],
                    help='Borderline-SMOTEのタイプ (1 or 2)')
    parser.add_argument('--borderline-k-neighbors', type=int, default=5,
                    help='Borderline SMOTEのk近傍数 (default: 5)')
    parser.add_argument('--borderline-m-neighbors', type=int, default=10,
                    help='Borderline SMOTE境界判定用近傍数 (default: 10)')
        
    args = parser.parse_args()
    
    # 出力ディレクトリの設定
    output_dir = args.output_dir if args.save_plots else None
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # データの読み込み
    data_file_path = os.path.join(args.data_path, args.data_file)
    print(f"データファイルを読み込みます: {data_file_path}")
    try:
        df = pd.read_csv(data_file_path)
    except Exception as e:
        print(f"データファイルの読み込みに失敗しました: {e}")
        print("ファイルパスと形式を確認してください。")
        exit(1)
    
    # 可視化のみのモードの場合
    if args.viz_only:
        print("可視化のみのモードで実行します...")
        features, target = prepare_data(df)
        
        # 可視化インスタンスの作成
        visualizer = EyeTrackingVisualizer(output_dir=output_dir)
        
        # 特徴量と目的変数の関係の可視化
        visualizer.visualize_feature_target_relationships(features, target)
        
        # サッカード特徴量の可視化
        has_saccade = any('saccade' in col.lower() for col in df.columns)
        if has_saccade:
            print("\nサッカード特徴量の分析を行います...")
            visualizer.visualize_saccade_features(df)
        
        # 視線追跡データの可視化
        has_eye_tracking = any('EyeCenterAngle' in col for col in df.columns)
        if has_eye_tracking:
            print("\n視線軌跡の分析を行います...")
            try:
                eye_data_sample = df.sample(min(1000, len(df)), random_state=args.random_state)
                visualizer.visualize_eye_movement_trajectories(eye_data_sample)
            except Exception as e:
                print(f"視線軌跡の可視化中にエラーが発生しました: {e}")
        
        # 結果ファイルの整理（可視化のみの場合）
        if args.organize_files and output_dir:
            new_dir = organize_result_files(args.data_file, output_dir)
            print(f"可視化処理が完了しました。結果は {new_dir} に保存されています。")
        else:
            print("可視化処理が完了しました。")
        exit(0)
    
    # モデルの選択
    model_mapping = {
        'lightgbm': LightGBMModel,
        'xgboost': XGBoostModel,
        'random_forest': RandomForestModel,
        'catboost': CatBoostModel
    }
    
    # アンダーサンプリング系のモデルを使用する場合はmodel_classをNoneに設定
    if args.undersampling:
        model_class = None
        model_name = f"usbag_{args.base_model}"
    elif args.simple_undersampling:
        model_class = None
        model_name = f"usimple_{args.base_model}"
    elif args.smotetomek:
        model_class = model_mapping[args.model]
        model_name = f"smotetomek_{args.model}"
    else:
        model_class = model_mapping[args.model]
        model_name = args.model
    
    # 実行設定の表示
    print(f"実行設定:")
    if args.undersampling:
        print(f"- モデル: UndersamplingBagging ({args.base_model})")
        print(f"- バッグ数: {args.n_bags}")
    elif args.simple_undersampling:
        print(f"- モデル: シンプルアンダーサンプリング ({args.base_model})")
    elif args.smotetomek:
        print(f"- モデル: {args.model} with SMOTETomek")
    else:
        print(f"- モデル: {args.model}")
    
    print(f"- SMOTE: {'あり' if args.smote and not args.smotetomek else 'なし'}")
    print(f"- SMOTETomek: {'あり' if args.smotetomek else 'なし'}")
    if args.smotetomek:
        print(f"  - SMOTE近傍数: {args.smotetomek_k_neighbors}")
        print(f"  - SMOTETomek戦略: {args.smotetomek_strategy}")
        print(f"  - Tomek戦略: {args.tomek_strategy}")
    
    print(f"- 交差検証: {'あり' if args.cv else 'なし'}")
    if args.cv:
        print(f"- 交差検証分割数: {args.splits}")
    
    print(f"- ALEプロット: {'あり' if args.ale_plots else 'なし'}")
    if args.ale_plots:
        print(f"  - 分析特徴量数: {args.ale_features}")
        print(f"  - 2次元ALE: {'あり' if args.ale_include_2d else 'なし'}")
        print(f"  - ALE vs PDP比較: {'あり' if args.ale_vs_pdp else 'なし'}")
    
    print(f"- 拡張分析: {'あり' if args.enhanced_analysis else 'なし'}")
    print(f"- 部分依存グラフ: {'あり' if args.pdp else 'なし'}")
    print(f"- 乱数シード: {args.random_state}")
    print(f"- データファイル: {args.data_file}")
    print(f"- 結果ファイル整理: {'あり' if args.organize_files else 'なし'}")
    
    # 拡張機能の利用可能性チェック
    if (args.smotetomek or args.ale_plots or args.enhanced_analysis) and not ENHANCED_FEATURES_AVAILABLE:
        print("\n警告: 拡張機能が要求されていますが利用できません")
        print("必要なファイルを確認してください:")
        print("- smotetomek_processor.py")
        print("- ale_plotter.py") 
        print("- enhanced_analysis_integration.py")
        
        if args.smotetomek:
            print("SMOTETomekの代わりに通常のSMOTEを使用します")
            args.smote = True
            args.smotetomek = False
        
        if args.ale_plots:
            print("ALEプロット機能は無効化されます")
            args.ale_plots = False
        
        if args.enhanced_analysis:
            print("拡張分析は実行できません")
            args.enhanced_analysis = False
    
    try:
        # 分析実行
        if args.enhanced_analysis:
            # 包括的拡張分析（全手法比較 + ALE）
            if not ENHANCED_FEATURES_AVAILABLE:
                print("❌ 拡張分析が利用できません。通常分析を実行します。")
                args.enhanced_analysis = False
            else:
                print("包括的拡張分析を実行します...")
                results = run_enhanced_comprehensive_analysis(
                    df=df,
                    model_class=model_class,
                    output_dir=output_dir,
                    random_state=args.random_state,
                    data_file_name=args.data_file,
                    smotetomek_strategy=args.smotetomek_strategy,
                    smotetomek_k_neighbors=args.smotetomek_k_neighbors,
                    tomek_strategy=args.tomek_strategy
                )
                
                if results and results.get('output_directory'):
                    print(f"\n包括的拡張分析が完了しました。")
                    print(f"結果は {results['output_directory']} に保存されています。")
                    
                    # サマリー表示
                    summary = results.get('summary', {})
                    if summary.get('best_method'):
                        print(f"\n最良手法: {summary['best_method'].upper()}")
                        print(f"最良F1スコア: {summary['best_f1_score']:.4f}")
                exit(0)
        
        if args.holdout_cv:
            # テストデータ保持+クロスバリデーション分析
            if not HOLDOUT_CV_AVAILABLE:
                print("❌ テストデータ保持+CVが利用できません。models/holdout_bunrui.pyファイルを確認してください。")
                exit(1)
            
            print("テストデータ保持+クロスバリデーション分析を実行します...")
            results = run_classification_holdout_cv_analysis(
                df=df, 
                model_class=model_class if not (args.undersampling or args.simple_undersampling) else None,
                use_smote=args.smote,
                use_undersampling=args.undersampling,
                use_simple_undersampling=args.simple_undersampling,
                n_splits=args.splits,
                use_smotetomek=args.smotetomek,
                smotetomek_strategy=args.smotetomek_strategy,
                smotetomek_k_neighbors=args.smotetomek_k_neighbors,
                tomek_strategy=args.tomek_strategy,
                test_size=0.2,  # テストデータ割合
                random_state=args.random_state,
                data_file_name=args.data_file,
                output_dir=output_dir,
                evaluate_thresholds=True,  # 閾値評価を有効化
                save_splits=True,  # 分割データの保存
                base_model=args.base_model,
                n_bags=args.n_bags,
                use_borderline_smote=args.borderline_smote,
                borderline_type=args.borderline_type,
                borderline_k_neighbors=args.borderline_k_neighbors,
                borderline_m_neighbors=args.borderline_m_neighbors,
            )
            
            print(f"\nテストデータ保持+クロスバリデーション分析が完了しました。")
            if results.get('output_dir'):
                print(f"結果は {results['output_dir']} に保存されています。")
                
        elif args.cv:
            # CrossValidator を使用した交差検証
            validator, cv_results = run_cv_analysis_with_smote(
                df, 
                model_class=model_class if not (args.undersampling or args.simple_undersampling) else None, 
                n_splits=args.splits,
                use_smote=args.smote,
                use_undersampling=args.undersampling,
                use_simple_undersampling=args.simple_undersampling,
                output_dir=output_dir,
                random_state=args.random_state,
                base_model=args.base_model,
                n_bags=args.n_bags
            )
            
            # 結果ファイルの整理
            if args.organize_files and output_dir:
                new_dir = organize_result_files(args.data_file, output_dir)
                print(f"\n交差検証による分析が完了しました。結果は {new_dir} に保存されています。")
            else:
                print("\n交差検証による分析が完了しました。")
                if output_dir:
                    print(f"結果は {output_dir} ディレクトリに保存されました。")
        else:
            # 単一の学習・評価による分析（拡張機能対応）
            results, new_dir = run_simple_analysis(
                df, 
                model_class=model_class if not (args.undersampling or args.simple_undersampling) else None,
                use_smote=args.smote,
                use_undersampling=args.undersampling,
                use_simple_undersampling=args.simple_undersampling,
                use_smotetomek=args.smotetomek,  # 新パラメータ
                random_state=args.random_state,
                output_dir=output_dir,
                data_file_name=args.data_file,
                organize_files=args.organize_files,
                create_ale_plots=args.ale_plots,  # ALE関連パラメータ
                ale_features=args.ale_features,
                ale_include_2d=args.ale_include_2d,
                ale_vs_pdp=args.ale_vs_pdp,
                smotetomek_strategy=args.smotetomek_strategy,  # SMOTETomekパラメータ
                smotetomek_k_neighbors=args.smotetomek_k_neighbors,
                tomek_strategy=args.tomek_strategy,
                base_model=args.base_model,
                n_bags=args.n_bags,
                use_borderline_smote=args.borderline_smote,
                borderline_type=args.borderline_type,
                borderline_k_neighbors=args.borderline_k_neighbors,
                borderline_m_neighbors=args.borderline_m_neighbors,
            )
            
            if new_dir:
                print(f"\n単一モデルによる分析が完了しました。結果は {new_dir} に保存されています。")
                
                # 使用した手法のサマリー表示
                print(f"\n使用した手法:")
                if args.smotetomek:
                    print(f"  - SMOTETomek (データバランシング + 境界クリーニング)")
                elif args.smote:
                    print(f"  - SMOTE (データバランシング)")
                else:
                    print(f"  - 元データ")
                
                if args.ale_plots:
                    print(f"  - ALE分析 ({args.ale_features}特徴量)")
                    if 'ale_analysis' in results:
                        ale_info = results['ale_analysis']
                        print(f"    * 1D プロット: {len(ale_info['ale_1d_plots'])}個")
                        if args.ale_include_2d:
                            print(f"    * 2D プロット: {len(ale_info['ale_2d_plots'])}個")
                
                # 性能サマリー表示
                if 'predictions_df' in results:
                    pred_df = results['predictions_df']
                    accuracy = (pred_df['True_Label'] == pred_df['Predicted_Label']).mean()
                    print(f"\n最終性能: 精度 = {accuracy:.4f}")
            else:
                print("\n単一モデルによる分析が完了しました。")
                if output_dir:
                    print(f"結果は {output_dir} ディレクトリに保存されました。")
    
    except Exception as e:
        print(f"\nエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()
        
        # エラー時の推奨事項
        print(f"\n推奨事項:")
        print(f"1. データファイルの形式を確認してください")
        print(f"2. 必要なライブラリがインストールされているか確認してください")
        if args.smotetomek or args.ale_plots:
            print(f"3. 拡張機能ファイルが正しく配置されているか確認してください:")
            print(f"   - smotetomek_processor.py")
            print(f"   - ale_plotter.py")
            print(f"   - enhanced_analysis_integration.py")
        print(f"4. --viz-only オプションで基本的な可視化のみ試してください")
        
        exit(1)