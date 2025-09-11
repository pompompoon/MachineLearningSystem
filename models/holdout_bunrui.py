"""
分類タスク用テストデータ保持+クロスバリデーション分析器
"""

import numpy as np
import pandas as pd
import os
import datetime
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, 
    confusion_matrix, roc_curve, auc, roc_auc_score,
    classification_report, precision_recall_curve
)

# scipy
from scipy.stats import chi2_contingency

# パス設定
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 安全なインポート用のヘルパー関数
def safe_import_from_main():
    """main5.pyから安全にインポート"""
    try:
        from main5 import prepare_data, calculate_specificity
        return prepare_data, calculate_specificity
    except ImportError as e:
        print(f"Warning: Could not import from main5.py: {e}")
        
        # フォールバック実装
        def calculate_specificity(y_true, y_pred):
            """特異度（Specificity）を計算する関数"""
            cm = confusion_matrix(y_true, y_pred)
            if cm.shape == (2, 2):
                tn, fp = cm[0, 0], cm[0, 1]
                specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            else:
                specificities = []
                n_classes = cm.shape[0]
                for i in range(n_classes):
                    tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
                    fp = np.sum(np.delete(cm, i, axis=0)[:, i])
                    specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
                specificity = np.mean(specificities)
            return specificity
        
        def prepare_data(df):
            """データの前処理を行う関数（フォールバック）"""
            # 目的変数のカラム名を確認
            if 'target' in df.columns:
                target_column = 'target'
            elif 'Target' in df.columns:
                target_column = 'Target'
            else:
                raise ValueError("目的変数の列 'target' または 'Target' が見つかりません")
            
            # IDカラムの取得
            id_column = None
            if 'InspectionDateAndId' in df.columns:
                id_column = df['InspectionDateAndId'].copy()
            elif 'Id' in df.columns:
                id_column = df['Id'].copy()
            
            # 特徴量とターゲットの分離
            drop_cols = []
            if 'InspectionDateAndId' in df.columns:
                drop_cols.append('InspectionDateAndId')
            if 'Id' in df.columns:
                drop_cols.append('Id')
            drop_cols.append(target_column)
            
            features = df.drop(drop_cols, axis=1)
            target = df[target_column]
            
            # ターゲットの変換（必要に応じて）
            if target.dtype == 'object':
                target = target.map({'intact': 0, 'mci': 1})
            
            # 欠損値の削除
            features = features.dropna()
            target = target[features.index]
            if id_column is not None:
                id_column = id_column[features.index]
            
            return features, target, id_column
        
        return prepare_data, calculate_specificity

# グローバルにインポート
prepare_data, calculate_specificity = safe_import_from_main()


class ClassificationHoldoutCVAnalyzer:
    """
    分類タスク用テストデータ保持+クロスバリデーション分析器
    
    使用例:
    -------
    analyzer = ClassificationHoldoutCVAnalyzer(
        model_class=LightGBMModel,
        n_splits=5,
        test_size=0.2,
        use_smote=True
    )
    
    results = analyzer.run_analysis(df, target_column='target')
    """
    
    def __init__(self, model_class=None, use_bagging=False, n_splits=5, 
                test_size=0.2, random_state=42, output_dir='result',
                # SMOTE関連パラメータ
                use_smote=False, smote_sampling_strategy='auto',
                use_undersampling=False, use_simple_undersampling=False,
                use_smotetomek=False,  # 新パラメータ
                smotetomek_strategy='auto',  # 新パラメータ
                smotetomek_k_neighbors=5,  # 新パラメータ  
                tomek_strategy='auto', 
                use_borderline_smote=False,
                borderline_type=1,
                borderline_k_neighbors=5,
                borderline_m_neighbors=10, # 新パラメータ
                # 閾値評価パラメータ
                evaluate_thresholds=True, threshold_methods=None,
                # その他のパラメータ
                save_splits=False, **kwargs):
        """
        初期化
        
        Parameters:
        -----------
        model_class : class
            使用するモデルクラス
        use_bagging : bool
            バギングモデルを使用するか
        n_splits : int
            クロスバリデーションの分割数
        test_size : float
            テストデータの割合
        random_state : int
            乱数シード
        output_dir : str
            結果出力ディレクトリ
        use_smote : bool
            SMOTEを使用するか
        smote_sampling_strategy : str or dict
            SMOTEのサンプリング戦略
        use_undersampling : bool
            アンダーサンプリングバギングを使用するか
        use_simple_undersampling : bool
            シンプルアンダーサンプリングを使用するか
        evaluate_thresholds : bool
            閾値評価を実行するか
        threshold_methods : list
            評価する閾値最適化手法のリスト
        save_splits : bool
            分割データを保存するか
        **kwargs : dict
            その他のパラメータ
        """
        self.model_class = model_class
        self.use_bagging = use_bagging
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.output_dir = output_dir
        
        # SMOTE設定
        self.use_smote = use_smote
        self.smote_sampling_strategy = smote_sampling_strategy
        self.use_undersampling = use_undersampling
        self.use_simple_undersampling = use_simple_undersampling
        
        # 閾値評価設定
        self.enable_threshold_evaluation = evaluate_thresholds
        self.threshold_methods = threshold_methods or ['youden', 'f1', 'balanced_accuracy']
        
        # その他の設定
        self.save_splits = save_splits
        self.kwargs = kwargs
        
        # 結果保存用
        self.results = {}
        self.X_dev = None
        self.X_test = None
        self.y_dev = None
        self.y_test = None
        self.id_dev = None
        self.id_test = None
        self.features = None
        self.target = None
        
        # 中間結果保存用
        self.fold_metrics = []
        self.oof_predictions = None
        self.oof_probabilities = None
        self.feature_importances = []
        self.smote_statistics = []
        self.final_model = None
        self.final_scaler = None
        
        self.use_smotetomek = use_smotetomek
        self.smotetomek_strategy = smotetomek_strategy
        self.smotetomek_k_neighbors = smotetomek_k_neighbors
        self.tomek_strategy = tomek_strategy
        self.use_borderline_smote = use_borderline_smote
        self.borderline_type = borderline_type
        # デバッグ情報を追加
        print(f"DEBUG: 初期化完了")
        print(f"  use_smotetomek: {self.use_smotetomek}")
        print(f"  use_smote: {self.use_smote}")
        print(f"  smotetomek_strategy: {getattr(self, 'smotetomek_strategy', 'NOT_SET')}")
        print(f"DEBUG: Borderline SMOTE初期化")
        print(f"  use_borderline_smote: {getattr(self, 'use_borderline_smote', 'NOT_SET')}")
        print(f"  borderline_type: {getattr(self, 'borderline_type', 'NOT_SET')}")
        print(f"  borderline_k_neighbors: {getattr(self, 'borderline_k_neighbors', 'NOT_SET')}")
        print(f"  borderline_m_neighbors: {getattr(self, 'borderline_m_neighbors', 'NOT_SET')}")
        
    def _setup_output_directory(self, data_file_name=None):
        """出力ディレクトリの設定"""
        if self.output_dir and data_file_name:
            base_name = os.path.splitext(os.path.basename(data_file_name))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # モデルタイプ識別
            model_type = "bagging" if self.use_bagging else "standard"
            smote_suffix = "_smote" if self.use_smote else ""
            undersampling_suffix = ""
            if self.use_undersampling:
                undersampling_suffix = "_usbag"
            elif self.use_simple_undersampling:
                undersampling_suffix = "_usimple"
            
            self.output_dir = f"{base_name}_clf_holdout_cv_{model_type}{smote_suffix}{undersampling_suffix}_{timestamp}"
            
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                print(f"出力ディレクトリを作成: {self.output_dir}")
                
                # サブディレクトリも作成
                subdirs = ['cv_results', 'visualizations', 'saved_model', 'data_splits', 'threshold_analysis']
                for subdir in subdirs:
                    os.makedirs(os.path.join(self.output_dir, subdir), exist_ok=True)
        
        return self.output_dir
    
    def split_data(self, features, target, id_column=None):
        """
        データを開発用とテスト用に分割（層化分割）
        
        Parameters:
        -----------
        features : pd.DataFrame
            特徴量
        target : pd.Series
            目的変数
        id_column : pd.Series, optional
            ID列
            
        Returns:
        --------
        dict
            分割結果の情報
        """
        print(f"\n=== データ分割（層化分割） ===")
        print("テストデータを分離して保持...")
        
        self.features = features
        self.target = target
        
        # クラス分布の確認
        class_counts = target.value_counts().sort_index()
        print(f"全体のクラス分布:")
        for class_val, count in class_counts.items():
            print(f"  クラス {class_val}: {count}サンプル ({count/len(target)*100:.1f}%)")
        
        # 層化分割
        if id_column is not None:
            self.X_dev, self.X_test, self.y_dev, self.y_test, self.id_dev, self.id_test = train_test_split(
                features, target, id_column,
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=target  # 層化分割
            )
        else:
            self.X_dev, self.X_test, self.y_dev, self.y_test = train_test_split(
                features, target,
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=target  # 層化分割
            )
            self.id_dev, self.id_test = None, None
        
        dev_ratio = len(self.X_dev) / len(features) * 100
        test_ratio = len(self.X_test) / len(features) * 100
        
        print(f"\n分割結果:")
        print(f"  開発データ: {len(self.X_dev)}サンプル ({dev_ratio:.1f}%)")
        print(f"  テストデータ: {len(self.X_test)}サンプル ({test_ratio:.1f}%)")
        
        # 各分割でのクラス分布確認
        print(f"\n開発データのクラス分布:")
        dev_class_counts = self.y_dev.value_counts().sort_index()
        for class_val, count in dev_class_counts.items():
            print(f"  クラス {class_val}: {count}サンプル ({count/len(self.y_dev)*100:.1f}%)")
        
        print(f"\nテストデータのクラス分布:")
        test_class_counts = self.y_test.value_counts().sort_index()
        for class_val, count in test_class_counts.items():
            print(f"  クラス {class_val}: {count}サンプル ({count/len(self.y_test)*100:.1f}%)")
        
        print(f"  テストデータは最終評価まで使用しません")
        
        # 分割データの保存（オプション）
        if self.save_splits and self.output_dir:
            self._save_split_data()
        
        return {
            'dev_size': len(self.X_dev),
            'test_size': len(self.X_test),
            'dev_ratio': dev_ratio,
            'test_ratio': test_ratio,
            'dev_class_distribution': dev_class_counts.to_dict(),
            'test_class_distribution': test_class_counts.to_dict()
        }
    
    def _save_split_data(self):
        """分割データをファイルに保存"""
        print(f"\n分割データを保存中...")
        splits_dir = os.path.join(self.output_dir, 'data_splits')
        
        # 開発データを保存
        dev_df = self.X_dev.copy()
        dev_df[self.target.name] = self.y_dev
        if self.id_dev is not None:
            dev_df.insert(0, self.id_dev.name, self.id_dev)
        dev_path = os.path.join(splits_dir, 'development_split.csv')
        dev_df.to_csv(dev_path, index=False)
        
        # テストデータを保存
        test_df = self.X_test.copy()
        test_df[self.target.name] = self.y_test
        if self.id_test is not None:
            test_df.insert(0, self.id_test.name, self.id_test)
        test_path = os.path.join(splits_dir, 'test_split.csv')
        test_df.to_csv(test_path, index=False)
        
        print(f"  開発データ: {dev_path}")
        print(f"  テストデータ: {test_path}")
    
    def _apply_resampling(self, X_train, y_train, fold=None):
        """
        リサンプリング手法を適用（Borderline SMOTE → SMOTETomek → SMOTE の優先順位）
        """
        # Borderline SMOTE（最優先）
        if getattr(self, 'use_borderline_smote', False):
            try:
                from models.borderline_smote_processor import BorderlineSMOTEProcessor
                
                borderline_type = getattr(self, 'borderline_type', 1)
                k_neighbors = getattr(self, 'borderline_k_neighbors', 5)
                m_neighbors = getattr(self, 'borderline_m_neighbors', 10)
                
                print(f"    Borderline-SMOTE{borderline_type}を適用中...")
                print(f"      設定: k_neighbors={k_neighbors}, m_neighbors={m_neighbors}")
                
                processor = BorderlineSMOTEProcessor(
                    sampling_strategy=self.smote_sampling_strategy,
                    k_neighbors=k_neighbors,
                    m_neighbors=m_neighbors,
                    borderline_type=borderline_type,
                    random_state=self.random_state + (fold or 0)
                )
                
                X_resampled, y_resampled, statistics = processor.fit_resample(
                    X_train, y_train, verbose=False  # 詳細ログは無効化
                )
                
                print(f"    Borderline-SMOTE{borderline_type}適用完了: {len(X_train)} -> {len(X_resampled)} サンプル")
                
                # 統計情報をインスタンス変数に保存
                if not hasattr(self, 'borderline_statistics'):
                    self.borderline_statistics = []
                self.borderline_statistics.append(statistics)
                
                return X_resampled, y_resampled
                
            except ImportError as e:
                print(f"    Borderline SMOTEインポートエラー: {e}")
                print(f"    models/borderline_smote_processor.py を確認してください")
                print(f"    通常のSMOTEにフォールバック...")
                self.use_smote = True
                self.use_borderline_smote = False
                
            except Exception as e:
                print(f"    Borderline SMOTEエラー: {e}")
                import traceback
                traceback.print_exc()
                print(f"    通常のSMOTEにフォールバック...")
                self.use_smote = True
                self.use_borderline_smote = False
        
        # SMOTETomek（第2優先）
        if self.use_smotetomek:
            try:
                from models.smotetomek_processor import SMOTETomekProcessor
                
                print(f"    SMOTETomekを適用中...")
                
                processor = SMOTETomekProcessor(
                    smote_sampling_strategy=self.smote_sampling_strategy,
                    smote_k_neighbors=self.smotetomek_k_neighbors,
                    tomek_sampling_strategy=self.tomek_strategy,
                    random_state=self.random_state + (fold or 0)
                )
                
                X_resampled, y_resampled, statistics = processor.fit_resample(
                    X_train, y_train, verbose=False
                )
                
                print(f"    SMOTETomek適用完了: {len(X_train)} -> {len(X_resampled)} サンプル")
                return X_resampled, y_resampled
                
            except ImportError as e:
                print(f"    SMOTETomekインポートエラー: {e}")
                print(f"    通常のSMOTEにフォールバック...")
                self.use_smote = True
                self.use_smotetomek = False
            except Exception as e:
                print(f"    SMOTETomekエラー: {e}")
                print(f"    通常のSMOTEにフォールバック...")
                self.use_smote = True
                self.use_smotetomek = False
        
        # 通常のSMOTE（第3優先）
        if self.use_smote:
            try:
                from imblearn.over_sampling import SMOTE
                
                print(f"    SMOTEを適用中...")
                
                # 少数クラスのサンプル数を確認
                unique, counts = np.unique(y_train, return_counts=True)
                min_samples = min(counts)
                
                # k_neighborsを調整（少数クラスのサンプル数-1、最小1）
                k_neighbors = min(5, max(1, min_samples - 1))
                
                smote = SMOTE(
                    sampling_strategy=self.smote_sampling_strategy,
                    k_neighbors=k_neighbors,
                    random_state=self.random_state + (fold or 0)
                )
                
                X_resampled, y_resampled = smote.fit_resample(X_train, y_train)
                print(f"    SMOTE適用完了: {len(X_train)} -> {len(X_resampled)} サンプル")
                return X_resampled, y_resampled
                
            except ImportError:
                print(f"    imbalanced-learnライブラリが見つかりません。リサンプリングをスキップします。")
                return X_train, y_train
            except Exception as e:
                print(f"    SMOTEエラー: {e}")
                return X_train, y_train
        
        return X_train, y_train
    
    def run_cross_validation(self):
        """
        開発データで層化クロスバリデーションを実行
        
        Returns:
        --------
        dict
            クロスバリデーション結果
        """
        print(f"\n=== 層化クロスバリデーション実行 ===")
        print(f"開発データ（{len(self.X_dev)}サンプル）で{self.n_splits}分割層化クロスバリデーション")
        
        # デバッグ情報
        print(f"DEBUG: self.use_borderline_smote = {getattr(self, 'use_borderline_smote', False)}")
        print(f"DEBUG: self.use_smotetomek = {self.use_smotetomek}")
        print(f"DEBUG: self.use_smote = {self.use_smote}")
        if getattr(self, 'use_borderline_smote', False):
            print(f"DEBUG: borderline_type = {getattr(self, 'borderline_type', 'NOT_SET')}")
            print(f"DEBUG: borderline_k_neighbors = {getattr(self, 'borderline_k_neighbors', 'NOT_SET')}")
            print(f"DEBUG: borderline_m_neighbors = {getattr(self, 'borderline_m_neighbors', 'NOT_SET')}")
        
        # 開発データの配列変換
        X_dev_values = self.X_dev.values
        y_dev_values = self.y_dev.values
        
        # 層化交差検証の設定
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # 結果の初期化
        self.fold_metrics = []
        self.oof_predictions = np.zeros(len(y_dev_values))
        self.oof_probabilities = np.zeros((len(y_dev_values), 2))  # 2クラス分類想定
        self.feature_importances = []
        self.smote_statistics = []
        
        # リサンプリング手法の決定と表示
        resampling_method = "なし"
        if getattr(self, 'use_borderline_smote', False):
            borderline_type = getattr(self, 'borderline_type', 1)
            resampling_method = f"Borderline-SMOTE{borderline_type}"
        elif self.use_smotetomek:
            resampling_method = "SMOTETomek"
        elif self.use_smote:
            resampling_method = "SMOTE"
        
        print(f"各フォールドで訓練データにリサンプリング適用: {resampling_method}")
        
        # 各フォールドでの処理
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev_values, y_dev_values)):
            print(f"\n--- Fold {fold+1}/{self.n_splits} ---")
            
            # データ分割
            X_train_fold = X_dev_values[train_idx]
            X_val_fold = X_dev_values[val_idx]
            y_train_fold = y_dev_values[train_idx]
            y_val_fold = y_dev_values[val_idx]
            
            print(f"  訓練: {len(X_train_fold)}サンプル, 検証: {len(X_val_fold)}サンプル")
            
            # クラス分布表示
            train_class_counts = pd.Series(y_train_fold).value_counts().sort_index()
            val_class_counts = pd.Series(y_val_fold).value_counts().sort_index()
            print(f"  訓練データクラス分布: {dict(train_class_counts)}")
            print(f"  検証データクラス分布: {dict(val_class_counts)}")
            
            # 元の訓練データサイズを記録
            original_train_size = len(X_train_fold)
            
            # リサンプリング適用（訓練データのみ）
            any_resampling = (getattr(self, 'use_borderline_smote', False) or 
                            self.use_smotetomek or self.use_smote)
            
            if any_resampling:
                print(f"  {resampling_method}を訓練データに適用中...")
                X_train_fold, y_train_fold = self._apply_resampling(X_train_fold, y_train_fold, fold)
                
                synthetic_count = len(X_train_fold) - original_train_size
                print(f"    リサンプリング後: {len(X_train_fold)}サンプル (+{synthetic_count})")
                
                # 統計を記録
                resampling_stats = {
                    'fold': fold + 1,
                    'method': resampling_method,
                    'original_size': original_train_size,
                    'resampled_size': len(X_train_fold),
                    'synthetic_count': synthetic_count
                }
                self.smote_statistics.append(resampling_stats)
                
                # リサンプリング後のクラス分布表示
                resampled_class_counts = pd.Series(y_train_fold).value_counts().sort_index()
                print(f"    リサンプリング後クラス分布: {dict(resampled_class_counts)}")
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # モデル学習
            model = self._create_model(fold)
            model.fit(X_train_scaled, y_train_fold)
            
            # 検証データで予測
            y_pred_fold = model.predict(X_val_scaled)
            y_pred_proba_fold = model.predict_proba(X_val_scaled)
            
            self.oof_predictions[val_idx] = y_pred_fold
            self.oof_probabilities[val_idx] = y_pred_proba_fold
            
            # 評価指標計算
            fold_result = self._calculate_classification_metrics(y_val_fold, y_pred_fold, y_pred_proba_fold, fold + 1)
            fold_result.update({
                'train_size': len(X_train_fold),
                'val_size': len(X_val_fold)
            })
            self.fold_metrics.append(fold_result)
            
            # 特徴量重要度
            if hasattr(model, 'feature_importances_'):
                importances = pd.DataFrame({
                    'feature': self.features.columns,
                    'importance': model.feature_importances_,
                    'fold': fold + 1
                })
                self.feature_importances.append(importances)
            
            # 結果表示
            metrics_str = ", ".join([f"{k}={v:.4f}" for k, v in fold_result.items() 
                                if k not in ['fold', 'train_size', 'val_size']])
            print(f"  結果: {metrics_str}")
        
        # クロスバリデーション結果のまとめ
        cv_results = self._summarize_cv_results()
        
        print(f"\n=== クロスバリデーション結果 ===")
        print(f"開発データでのクロスバリデーション性能:")
        for metric, value in cv_results['cv_metrics'].items():
            std = cv_results['cv_std'][f'{metric}_std']
            print(f"  {metric.upper()}: {value:.4f} ± {std:.4f}")
        
        return cv_results
        
    def _create_model(self, fold=None):
        """モデルを作成"""
        if self.use_undersampling:
            # UndersamplingBaggingModelの使用
            try:
                from models.undersampling_bagging_model import UndersamplingBaggingModel
                model = UndersamplingBaggingModel(
                    base_model=self.kwargs.get('base_model', 'lightgbm'),
                    n_bags=self.kwargs.get('n_bags', 10),
                    random_state=self.random_state + (fold or 0)
                )
            except ImportError:
                print("UndersamplingBaggingModelが見つかりません。標準モデルを使用します。")
                model = self.model_class()
        elif self.use_simple_undersampling:
            # UndersamplingModelの使用
            try:
                from models.undersampling_model import UndersamplingModel
                model = UndersamplingModel(
                    base_model=self.kwargs.get('base_model', 'lightgbm'),
                    random_state=self.random_state + (fold or 0)
                )
            except ImportError:
                print("UndersamplingModelが見つかりません。標準モデルを使用します。")
                model = self.model_class()
        else:
            # CatBoost特有の設定
            if hasattr(self.model_class, '__name__') and 'CatBoost' in self.model_class.__name__:
                # CatBoostの場合、カテゴリカル特徴量を自動検出
                cat_features = self.kwargs.get('cat_features', None)
                if cat_features is None and hasattr(self, 'features'):
                    # 文字列型やobject型の列をカテゴリカルとして自動検出
                    cat_features = []
                    for i, col in enumerate(self.features.columns):
                        if self.features[col].dtype == 'object' or self.features[col].dtype.name == 'category':
                            cat_features.append(i)
                
                # CatBoostModelが独自の初期化パラメータを持つ場合
                try:
                    # クロスバリデーション中はuse_best_model=Falseに設定
                    model = self.model_class(
                        cat_features=cat_features,
                        use_best_model=False,  # CVでは無効化
                        random_state=self.random_state + (fold or 0)
                    )
                except TypeError:
                    # 標準的な初期化の場合
                    model = self.model_class(random_state=self.random_state + (fold or 0))
                    
                    # モデルがCatBoostの場合、use_best_modelを無効化
                    if hasattr(model, 'model') and hasattr(model.model, 'set_params'):
                        try:
                            model.model.set_params(use_best_model=False)
                        except:
                            pass
            else:
                # CatBoost以外のモデル
                try:
                    model = self.model_class(random_state=self.random_state + (fold or 0))
                except TypeError:
                    model = self.model_class()
        
        return model
    
    def _calculate_classification_metrics(self, y_true, y_pred, y_pred_proba, fold=None):
        """分類評価指標を計算"""
        
        # 基本指標
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # 特異度
        specificity = calculate_specificity(y_true, y_pred)
        
        # AUC（2クラス分類の場合）
        try:
            if y_pred_proba.shape[1] == 2:
                auc_score = roc_auc_score(y_true, y_pred_proba[:, 1])
            else:
                auc_score = roc_auc_score(y_true, y_pred_proba, multi_class='ovr', average='weighted')
        except:
            auc_score = 0.0
        
        metrics = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'specificity': specificity,
            'f1': f1,
            'auc': auc_score
        }
        
        if fold is not None:
            metrics['fold'] = fold
        
        return metrics
    
    def _summarize_cv_results(self):
        """クロスバリデーション結果をまとめる"""
        metrics_df = pd.DataFrame(self.fold_metrics)
        
        # 平均指標
        cv_metrics = {
            'accuracy': metrics_df['accuracy'].mean(),
            'precision': metrics_df['precision'].mean(),
            'recall': metrics_df['recall'].mean(),
            'specificity': metrics_df['specificity'].mean(),
            'f1': metrics_df['f1'].mean(),
            'auc': metrics_df['auc'].mean()
        }
        
        # 標準偏差
        cv_std = {
            'accuracy_std': metrics_df['accuracy'].std(),
            'precision_std': metrics_df['precision'].std(),
            'recall_std': metrics_df['recall'].std(),
            'specificity_std': metrics_df['specificity'].std(),
            'f1_std': metrics_df['f1'].std(),
            'auc_std': metrics_df['auc'].std()
        }
        
        # 特徴量重要度の平均
        avg_importance = None
        if self.feature_importances:
            importances_df = pd.concat(self.feature_importances)
            avg_importance = importances_df.groupby('feature')['importance'].mean().reset_index()
            avg_importance = avg_importance.sort_values('importance', ascending=False)
        
        return {
            'cv_metrics': cv_metrics,
            'cv_std': cv_std,
            'fold_metrics': metrics_df,
            'avg_importance': avg_importance
        }
    
    def train_final_model(self):
        """
        開発データ全体で最終モデルを学習
        
        Returns:
        --------
        dict
            最終モデルの情報
        """
        print(f"\n=== 最終モデル学習 ===")
        print("開発データ全体で最終モデルを学習...")
        
        # 開発データ全体にリサンプリング適用
        X_dev_values = self.X_dev.values
        y_dev_values = self.y_dev.values
        
        # リサンプリング手法の確認
        any_resampling = (getattr(self, 'use_borderline_smote', False) or 
                        self.use_smotetomek or self.use_smote)
        
        if any_resampling:
            # 使用する手法を決定
            if getattr(self, 'use_borderline_smote', False):
                borderline_type = getattr(self, 'borderline_type', 1)
                method_name = f"Borderline-SMOTE{borderline_type}"
            elif self.use_smotetomek:
                method_name = "SMOTETomek"
            else:
                method_name = "SMOTE"
            
            print(f"開発データ全体に{method_name}を適用中...")
            X_dev_final, y_dev_final = self._apply_resampling(X_dev_values, y_dev_values)
            
            synthetic_count = len(X_dev_final) - len(X_dev_values)
            print(f"  リサンプリング適用前: {len(X_dev_values)}サンプル")
            print(f"  リサンプリング適用後: {len(X_dev_final)}サンプル (+{synthetic_count})")
            
            # 最終的なクラス分布表示
            final_class_counts = pd.Series(y_dev_final).value_counts().sort_index()
            print(f"  最終クラス分布: {dict(final_class_counts)}")
        else:
            X_dev_final = X_dev_values
            y_dev_final = y_dev_values
            print("リサンプリングは適用しません")
        
        # 最終スケーリング
        self.final_scaler = StandardScaler()
        X_dev_scaled = self.final_scaler.fit_transform(X_dev_final)
        
        # 最終モデル学習
        self.final_model = self._create_model()
        self.final_model.fit(X_dev_scaled, y_dev_final)
        
        print(f"最終モデル学習完了")
        
        return {
            'model': self.final_model,
            'scaler': self.final_scaler,
            'training_size': len(X_dev_final),
            'original_size': len(X_dev_values),
            'resampling_applied': any_resampling
        }
    def evaluate_test(self):
        """
        テストデータで最終評価
        
        Returns:
        --------
        dict
            テスト評価結果
        """
        print(f"\n=== テストデータで最終評価 ===")
        print("保持していたテストデータで最終性能を評価...")
        
        # テストデータで予測
        X_test_scaled = self.final_scaler.transform(self.X_test.values)
        test_pred = self.final_model.predict(X_test_scaled)
        test_pred_proba = self.final_model.predict_proba(X_test_scaled)
        
        # テスト性能の計算
        test_metrics = self._calculate_classification_metrics(self.y_test, test_pred, test_pred_proba)
        
        print(f"テストデータでの最終性能:")
        for metric, value in test_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        return {
            'test_metrics': test_metrics,
            'test_predictions': test_pred,
            'test_probabilities': test_pred_proba,
            'test_true': self.y_test.values
        }
    
    def _evaluate_thresholds(self, test_results):
        """
        閾値評価を実行
        
        Parameters:
        -----------
        test_results : dict
            テスト結果
            
        Returns:
        --------
        dict
            閾値評価結果
        """
        if not self.enable_threshold_evaluation:
            return None
        
        print(f"\n=== 閾値評価 ===")
        
        try:
            from visualization.threshold_evaluator import ThresholdEvaluator
            
            threshold_dir = os.path.join(self.output_dir, 'threshold_analysis')
            threshold_evaluator = ThresholdEvaluator(output_dir=threshold_dir)
            
            # Out-of-fold予測での閾値評価
            y_true_oof = self.y_dev.values
            y_proba_oof = self.oof_probabilities[:, 1]  # 陽性クラスの確率
            
            print("Out-of-fold予測による閾値最適化...")
            oof_optimal_thresholds, oof_metrics_df = threshold_evaluator.create_threshold_recommendation_report(
                y_true_oof, y_proba_oof
            )
            
            # テストデータでの閾値評価
            y_true_test = test_results['test_true']
            y_proba_test = test_results['test_probabilities'][:, 1]
            
            print("テストデータによる閾値評価...")
            test_optimal_thresholds, test_metrics_df = threshold_evaluator.create_threshold_recommendation_report(
                y_true_test, y_proba_test
            )
            
            # 閾値による性能変化の可視化
            threshold_evaluator.plot_recall_specificity_curve(
                y_true_test, 
                y_proba_test, 
                output_dir=threshold_dir,
                model_name="holdout_cv_classifier"
            )
            
            print("閾値評価完了")
            
            return {
                'oof_optimal_thresholds': oof_optimal_thresholds,
                'test_optimal_thresholds': test_optimal_thresholds,
                'oof_metrics_df': oof_metrics_df,
                'test_metrics_df': test_metrics_df
            }
            
        except Exception as e:
            print(f"閾値評価中にエラー: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def save_results(self, cv_results, test_results, threshold_results=None, data_file_name=None):
        """
        結果を保存
        
        Parameters:
        -----------
        cv_results : dict
            クロスバリデーション結果
        test_results : dict
            テスト結果
        threshold_results : dict, optional
            閾値評価結果
        data_file_name : str, optional
            データファイル名
        """
        if not self.output_dir:
            return
        
        print(f"\n=== 結果の保存 ===")
        
        # 可視化
        self._create_visualizations(cv_results, test_results, threshold_results)
        
        # 結果をCSVに保存
        self._save_csv_results(cv_results, test_results)
        
        # 設定情報保存
        self._save_config(cv_results, test_results, threshold_results, data_file_name)
        
        print(f"結果保存完了: {self.output_dir}")
    
    def _create_visualizations(self, cv_results, test_results, threshold_results=None):
        """可視化を作成"""
        try:
            viz_dir = os.path.join(self.output_dir, 'visualizations')
            
            # 混同行列
            cm = confusion_matrix(test_results['test_true'], test_results['test_predictions'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                      xticklabels=['Negative (0)', 'Positive (1)'],
                      yticklabels=['Negative (0)', 'Positive (1)'])
            plt.xlabel('予測ラベル')
            plt.ylabel('真のラベル')
            plt.title('混同行列（テストデータ）')
            plt.savefig(os.path.join(viz_dir, 'confusion_matrix_test.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # ROC曲線
            fpr, tpr, _ = roc_curve(test_results['test_true'], test_results['test_probabilities'][:, 1])
            auc_score = auc(fpr, tpr)
            
            plt.figure(figsize=(8, 6))
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {auc_score:.3f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('偽陽性率 (False Positive Rate)')
            plt.ylabel('真陽性率 (True Positive Rate)')
            plt.title('ROC曲線（テストデータ）')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(viz_dir, 'roc_curve_test.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 予測確率分布
            plt.figure(figsize=(10, 6))
            for class_label in [0, 1]:
                mask = test_results['test_true'] == class_label
                probabilities = test_results['test_probabilities'][mask, 1]
                plt.hist(probabilities, bins=20, alpha=0.7, 
                        label=f'クラス {class_label}', density=True)
            plt.xlabel('陽性クラス予測確率')
            plt.ylabel('密度')
            plt.title('予測確率分布（テストデータ）')
            plt.legend()
            plt.savefig(os.path.join(viz_dir, 'probability_distribution_test.png'), dpi=300, bbox_inches='tight')
            plt.close()
            
            # 特徴量重要度
            if cv_results['avg_importance'] is not None:
                plt.figure(figsize=(10, 8))
                top_features = cv_results['avg_importance'].head(20)
                sns.barplot(data=top_features, x='importance', y='feature')
                plt.title('特徴量重要度（上位20）')
                plt.xlabel('重要度')
                plt.tight_layout()
                plt.savefig(os.path.join(viz_dir, 'feature_importance.png'), dpi=300, bbox_inches='tight')
                plt.close()
            
            print(f"可視化完了")
            
        except Exception as e:
            print(f"可視化エラー: {e}")
            import traceback
            traceback.print_exc()
    
    def _save_csv_results(self, cv_results, test_results):
        """結果をCSVに保存"""
        # Out-of-fold結果
        oof_results_df = pd.DataFrame({
            'true': self.y_dev.values,
            'predicted': self.oof_predictions,
            'probability_class_0': self.oof_probabilities[:, 0],
            'probability_class_1': self.oof_probabilities[:, 1],
            'correct': self.y_dev.values == self.oof_predictions
        })
        
        if self.id_dev is not None:
            oof_results_df.insert(0, self.id_dev.name, self.id_dev.values)
        
        oof_path = os.path.join(self.output_dir, 'oof_results.csv')
        oof_results_df.to_csv(oof_path, index=False)
        
        # テスト結果
        test_results_df = pd.DataFrame({
            'true': test_results['test_true'],
            'predicted': test_results['test_predictions'],
            'probability_class_0': test_results['test_probabilities'][:, 0],
            'probability_class_1': test_results['test_probabilities'][:, 1],
            'correct': test_results['test_true'] == test_results['test_predictions']
        })
        
        if self.id_test is not None:
            test_results_df.insert(0, self.id_test.name, self.id_test.values)
        
        test_path = os.path.join(self.output_dir, 'test_results.csv')
        test_results_df.to_csv(test_path, index=False)
        
        print(f"Out-of-fold結果: {oof_path}")
        print(f"テスト結果: {test_path}")
    
    def _save_config(self, cv_results, test_results, threshold_results, data_file_name):
        """設定情報を保存"""
        config_path = os.path.join(self.output_dir, 'config_classification_holdout_cv.txt')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("=== 分類用テストデータ保持+クロスバリデーション設定 ===\n")
            f.write(f"データファイル: {data_file_name}\n")
            f.write(f"開発データ: {(1-self.test_size)*100:.1f}% ({len(self.X_dev)}サンプル)\n")
            f.write(f"テストデータ: {self.test_size*100:.1f}% ({len(self.X_test)}サンプル)\n")
            f.write(f"CV分割数: {self.n_splits}\n")
            
            if self.use_smote:
                f.write(f"\n=== SMOTE設定 ===\n")
                f.write(f"サンプリング戦略: {self.smote_sampling_strategy}\n")
            
            if self.use_undersampling:
                f.write(f"\n=== アンダーサンプリングバギング設定 ===\n")
                f.write(f"ベースモデル: {self.kwargs.get('base_model', 'lightgbm')}\n")
                f.write(f"バッグ数: {self.kwargs.get('n_bags', 10)}\n")
            
            f.write(f"\n=== クロスバリデーション性能（開発データ） ===\n")
            for metric, value in cv_results['cv_metrics'].items():
                std = cv_results['cv_std'][f'{metric}_std']
                f.write(f"{metric.upper()}: {value:.4f} ± {std:.4f}\n")
            
            f.write(f"\n=== テストデータでの最終性能 ===\n")
            for metric, value in test_results['test_metrics'].items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
            
            # 性能差
            f.write(f"\n=== 性能差（テスト - CV） ===\n")
            for metric in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']:
                diff = test_results['test_metrics'][metric] - cv_results['cv_metrics'][metric]
                f.write(f"{metric.upper()}差: {diff:+.4f}\n")
            
            # 閾値評価結果
            if threshold_results is not None:
                f.write(f"\n=== 最適閾値（テストデータ） ===\n")
                for method, values in threshold_results.get('test_optimal_thresholds', {}).items():
                    f.write(f"\n{method}最適化:\n")
                    f.write(f"  閾値: {values.get('threshold', 0):.3f}\n")
                    f.write(f"  精度: {values.get('precision', 0):.3f}\n")
                    f.write(f"  再現率: {values.get('recall', 0):.3f}\n")
                    f.write(f"  F1スコア: {values.get('f1', 0):.3f}\n")
                    
            f.write(f"\n実行日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"設定情報: {config_path}")
    
    def run_analysis(self, df, target_column='target', data_file_name=None):
        """
        完全な分析を実行
        
        Parameters:
        -----------
        df : pd.DataFrame
            分析対象のデータフレーム
        target_column : str
            目的変数のカラム名
        data_file_name : str, optional
            データファイル名
            
        Returns:
        --------
        dict
            分析結果
        """
        print(f"\n{'='*70}")
        print(f"分類用テストデータ保持+クロスバリデーション分析開始")
        print(f"{'='*70}")
        
        # 出力ディレクトリ設定
        self._setup_output_directory(data_file_name)
        
        # データ前処理
        print(f"\n=== データ前処理 ===")
        features, target, id_column = prepare_data(df)
        
        print(f"前処理後のデータ:")
        print(f"  特徴量数: {features.shape[1]}")
        print(f"  サンプル数: {features.shape[0]}")
        print(f"  クラス数: {target.nunique()}")
        
        # ステップ1: データ分割
        split_info = self.split_data(features, target, id_column)
        
        # ステップ2: クロスバリデーション
        cv_results = self.run_cross_validation()
        
        # ステップ3: 最終モデル学習
        model_info = self.train_final_model()
        
        # ステップ4: テスト評価
        test_results = self.evaluate_test()
        
        # ステップ5: 閾値評価
        threshold_results = self._evaluate_thresholds(test_results) if self.enable_threshold_evaluation else None
        
        # ステップ6: 結果保存
        self.save_results(cv_results, test_results, threshold_results, data_file_name)
        
        # 結果のまとめ
        results = {
            'cv_metrics': cv_results['cv_metrics'],
            'cv_std': cv_results['cv_std'],
            'test_metrics': test_results['test_metrics'],
            'fold_metrics': cv_results['fold_metrics'],
            'oof_predictions': self.oof_predictions,
            'oof_probabilities': self.oof_probabilities,
            'test_predictions': test_results['test_predictions'],
            'test_probabilities': test_results['test_probabilities'],
            'test_true': test_results['test_true'],
            'dev_true': self.y_dev.values,
            'final_model': self.final_model,
            'scaler': self.final_scaler,
            'feature_importance': cv_results['avg_importance'],
            'threshold_results': threshold_results,
            'smote_statistics': self.smote_statistics,
            'output_dir': self.output_dir,
            'data_splits': split_info
        }
        
        # サマリー表示
        self._print_summary(cv_results, test_results, threshold_results)
        
        return results
    
    def _print_summary(self, cv_results, test_results, threshold_results):
        """結果サマリーを表示"""
        print(f"\n{'='*70}")
        print(f"分類用テストデータ保持+クロスバリデーション完了")
        
        # CV性能表示
        print(f"\nCV性能（開発データ）:")
        for metric, value in cv_results['cv_metrics'].items():
            std = cv_results['cv_std'][f'{metric}_std']
            print(f"  {metric.upper()}: {value:.4f} ± {std:.4f}")
        
        # テスト性能表示
        print(f"\n最終性能（テストデータ）:")
        for metric, value in test_results['test_metrics'].items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        # 性能差表示
        print(f"\n性能差（テスト - CV）:")
        for metric in ['accuracy', 'precision', 'recall', 'specificity', 'f1', 'auc']:
            diff = test_results['test_metrics'][metric] - cv_results['cv_metrics'][metric]
            print(f"  {metric.upper()}差: {diff:+.4f}")
        
        # 特徴量重要度
        if cv_results['avg_importance'] is not None:
            print(f"\n特徴量重要度（上位10）:")
            print(cv_results['avg_importance'].head(10).to_string(index=False))
        
        # 最適閾値
        if threshold_results is not None and threshold_results.get('test_optimal_thresholds'):
            print(f"\n最適閾値（テストデータ）:")
            for method, values in threshold_results['test_optimal_thresholds'].items():
                print(f"  {method}: 閾値={values.get('threshold', 0):.3f}, F1={values.get('f1', 0):.3f}")
        
        if self.output_dir:
            print(f"\n結果保存先: {self.output_dir}")
        print(f"{'='*70}")


def run_classification_holdout_cv_analysis(df, model_class=None, use_bagging=False,
                                         n_splits=5, test_size=0.2, 
                                         output_dir='result', random_state=42, 
                                         target_column='target', data_file_name=None,
                                         save_splits=False,
                                         # SMOTE関連パラメータ
                                         use_smote=False, smote_sampling_strategy='auto',
                                         # アンダーサンプリング関連パラメータ
                                         use_undersampling=False, use_simple_undersampling=False,
                                         # SMOTETomek関連パラメータ（新規追加）
                                         use_smotetomek=False,
                                         smotetomek_strategy='auto',
                                         smotetomek_k_neighbors=5,
                                         tomek_strategy='auto',
                                         use_borderline_smote=False,
                                         borderline_type=1,
                                         borderline_k_neighbors=5,
                                         borderline_m_neighbors=10,
                                         # 閾値評価パラメータ
                                         evaluate_thresholds=True,
                                         # モデル保存パラメータ
                                         save_final_model=True,
                                         **kwargs):
    """
    分類タスク用テストデータ保持+クロスバリデーション分析の実行関数
    
    Parameters:
    -----------
    df : pd.DataFrame
        分析対象データ
    model_class : class
        使用するモデルクラス
    use_bagging : bool
        バギングを使用するか
    n_splits : int
        CV分割数
    test_size : float
        テストデータ割合
    output_dir : str
        出力ディレクトリ
    random_state : int
        乱数シード
    target_column : str
        目的変数列名
    data_file_name : str
        データファイル名
    save_splits : bool
        分割データを保存するか
    use_smote : bool
        SMOTEを使用するか
    smote_sampling_strategy : str or dict
        SMOTEサンプリング戦略
    use_undersampling : bool
        アンダーサンプリングバギングを使用するか
    use_simple_undersampling : bool
        シンプルアンダーサンプリングを使用するか
    evaluate_thresholds : bool
        閾値評価を実行するか
    save_final_model : bool
        最終モデルを保存するか
    **kwargs : dict
        その他のパラメータ
        
    Returns:
    --------
    dict
        分析結果
    """
    print(f"DEBUG: 関数に渡されたパラメータ")
    print(f"  use_smotetomek: {use_smotetomek}")
    print(f"  smotetomek_strategy: {smotetomek_strategy}")
    print(f"DEBUG: 関数に渡されたパラメータ（Borderline SMOTE）")
    print(f"  use_borderline_smote: {locals().get('use_borderline_smote', 'NOT_FOUND')}")
    print(f"  borderline_type: {locals().get('borderline_type', 'NOT_FOUND')}")
    analyzer = ClassificationHoldoutCVAnalyzer(
        model_class=model_class,
        use_bagging=use_bagging,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
        output_dir=output_dir,
        use_smote=use_smote,
        smote_sampling_strategy=smote_sampling_strategy,
        use_undersampling=use_undersampling,
        use_simple_undersampling=use_simple_undersampling,
        use_smotetomek=use_smotetomek,  # 新パラメータ
        smotetomek_strategy=smotetomek_strategy,  # 新パラメータ
        smotetomek_k_neighbors=smotetomek_k_neighbors,  # 新パラメータ
        tomek_strategy=tomek_strategy,  # 新パラメータ
        use_borderline_smote=use_borderline_smote,
        borderline_type=borderline_type,
        evaluate_thresholds=evaluate_thresholds,
        save_splits=save_splits,
        **kwargs
    )
    
    return analyzer.run_analysis(df, target_column, data_file_name)