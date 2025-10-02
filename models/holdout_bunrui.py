# 分類用テストデータ保持+クロスバリデーション分析器（ALE対応版）
import numpy as np
import pandas as pd
import os
import datetime
import sys
import matplotlib.pyplot as plt

# scikit-learn
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score

# scipy
from scipy.stats import chi2_contingency
import warnings

# ALE関連のインポート（安全）
try:
    from visualization.aleplotter import ALEPlotter
    ALE_AVAILABLE = True
    print("✓ ALE機能が利用可能です")
except ImportError:
    ALE_AVAILABLE = False
    print("⚠ ALE機能が利用できません: visualization/aleplotter.py を確認してください")

# 既存のインポート部分
try:
    from models.undersampling_bagging_model import UndersamplingBaggingModel
    from models.undersampling_model import UndersamplingModel
    UNDERSAMPLING_AVAILABLE = True
except ImportError:
    UNDERSAMPLING_AVAILABLE = False
    print("⚠ アンダーサンプリングモデルが利用できません")


class ClassificationHoldoutCVAnalyzer:
    """
    分類問題用のテストデータ保持+クロスバリデーション分析クラス（ALE対応版）
    """
    
    def __init__(self, model_class=None, use_undersampling=False, use_simple_undersampling=False,
                 n_splits=5, test_size=0.2, random_state=42, output_dir='result',
                 # SMOTE関連パラメータ
                 use_smote=False, use_smotetomek=False, smote_kwargs=None,
                 # ALE関連パラメータ
                 generate_ale=True, ale_n_features=6, ale_grid_resolution=30,
                 ale_include_2d=False, ale_vs_pdp=False,
                 # その他のパラメータ
                 save_splits=False, **kwargs):
        """
        初期化
        """
        self.model_class = model_class
        self.use_undersampling = use_undersampling
        self.use_simple_undersampling = use_simple_undersampling
        self.n_splits = n_splits
        self.test_size = test_size
        self.random_state = random_state
        self.output_dir = output_dir
        
        # ALE設定
        self.generate_ale = generate_ale and ALE_AVAILABLE
        self.ale_n_features = ale_n_features
        self.ale_grid_resolution = ale_grid_resolution
        self.ale_include_2d = ale_include_2d
        self.ale_vs_pdp = ale_vs_pdp
        
        # SMOTE設定
        self.use_smote = use_smote
        self.use_smotetomek = use_smotetomek
        self.smote_kwargs = smote_kwargs or {}
        
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
        self.final_model = None
        self.final_scaler = None

    def generate_ale_plots(self, cv_results):
        """ALE（Accumulated Local Effects）プロットを生成（分類版・エラー対応強化）"""
        if not self.generate_ale or not ALE_AVAILABLE:
            return
        
        print(f"\n=== 分類用ALE プロット生成 ===")
        
        # ALE出力ディレクトリ
        ale_output_dir = os.path.join(self.output_dir, 'ale_plots') if self.output_dir else 'ale_plots'
        os.makedirs(ale_output_dir, exist_ok=True)
        
        try:
            # 開発データ全体でALEを計算
            X_dev_for_ale = self.X_dev
            
            print(f"ALE計算用データ: {X_dev_for_ale.shape}")
            print(f"使用モデル: {type(self.final_model).__name__}")
            print(f"対象特徴量数: {self.ale_n_features}")
            
            # ALEプロッター作成（エラー対応強化版）
            try:
                ale_plotter = ALEPlotter(
                    model=self.final_model,
                    X=X_dev_for_ale,
                    feature_names=list(X_dev_for_ale.columns),
                    output_dir=ale_output_dir
                )
            except Exception as e:
                print(f"ALEプロッター作成エラー: {e}")
                return
            
            # 特徴量重要度がある場合は上位特徴量を選択
            feature_names_for_ale = []
            if cv_results.get('avg_importance') is not None and len(cv_results['avg_importance']) > 0:
                try:
                    top_features = cv_results['avg_importance'].head(self.ale_n_features)
                    feature_names_for_ale = [f for f in top_features['feature'].tolist() 
                                           if f in X_dev_for_ale.columns]
                except Exception as e:
                    print(f"特徴量重要度取得エラー: {e}")
            
            if not feature_names_for_ale:
                # 特徴量重要度がない場合は最初の特徴量を使用
                feature_names_for_ale = list(X_dev_for_ale.columns)[:self.ale_n_features]
            
            print(f"選択された特徴量: {feature_names_for_ale[:self.ale_n_features]}")
            
            # 1次元ALEプロット（上位特徴量）
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            successful_plots = 0
            
            for i, feature_name in enumerate(feature_names_for_ale[:self.ale_n_features]):
                print(f"1D ALEプロット生成: {feature_name} ({i+1}/{len(feature_names_for_ale)})")
                try:
                    fig = ale_plotter.plot_ale_1d(
                        feature_name,
                        bins=self.ale_grid_resolution,
                        title=f'ALE Plot: {feature_name} (分類用)',
                        save=True
                    )
                    if fig:
                        # 個別保存パスも設定
                        safe_name = feature_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
                        individual_path = os.path.join(
                            ale_output_dir, 
                            f'ale_1d_{safe_name}_classification_{timestamp}.png'
                        )
                        fig.savefig(individual_path, dpi=300, bbox_inches='tight')
                        plt.close(fig)
                        successful_plots += 1
                        print(f"✓ 1D ALEプロット保存: {individual_path}")
                except Exception as e:
                    print(f"✗ 1D ALEプロット生成エラー ({feature_name}): {e}")
            
            # 2次元ALEプロット（オプション）
            if self.ale_include_2d and len(feature_names_for_ale) >= 2:
                print("2D ALEプロット生成中...")
                n_pairs = min(2, len(feature_names_for_ale) * (len(feature_names_for_ale) - 1) // 2)
                
                pair_count = 0
                for i in range(len(feature_names_for_ale)):
                    for j in range(i + 1, len(feature_names_for_ale)):
                        if pair_count >= n_pairs:
                            break
                        
                        feature1 = feature_names_for_ale[i]
                        feature2 = feature_names_for_ale[j]
                        
                        try:
                            print(f"2D ALEプロット生成: {feature1} vs {feature2}")
                            fig = ale_plotter.plot_ale_2d(
                                feature1, feature2,
                                bins=min(20, self.ale_grid_resolution // 2),
                                title=f'2D ALE: {feature1} vs {feature2} (分類用)',
                                save=True
                            )
                            if fig:
                                plt.close(fig)
                                successful_plots += 1
                                print(f"✓ 2D ALEプロット生成完了")
                            pair_count += 1
                        except Exception as e:
                            print(f"✗ 2D ALEプロット生成エラー ({feature1}, {feature2}): {e}")
                    
                    if pair_count >= n_pairs:
                        break
            
            # ALE vs PDP 比較（オプション）
            if self.ale_vs_pdp:
                print("ALE vs PDP 比較分析中...")
                for feature_name in feature_names_for_ale[:min(3, len(feature_names_for_ale))]:
                    try:
                        fig = ale_plotter.compare_ale_vs_pdp(
                            feature_name,
                            bins=self.ale_grid_resolution
                        )
                        if fig:
                            plt.close(fig)
                            successful_plots += 1
                            print(f"✓ ALE vs PDP比較完了: {feature_name}")
                    except Exception as e:
                        print(f"✗ ALE vs PDP比較エラー ({feature_name}): {e}")
            
            # ALEダッシュボード作成
            print("ALEダッシュボード作成中...")
            try:
                dashboard_info = ale_plotter.create_ale_dashboard(
                    features=feature_names_for_ale[:self.ale_n_features],
                    include_2d=self.ale_include_2d,
                    bins_1d=self.ale_grid_resolution,
                    bins_2d=min(20, self.ale_grid_resolution // 2)
                )
                
                successful_plots += len(dashboard_info['ale_1d_plots'])
                successful_plots += len(dashboard_info['ale_2d_plots'])
                
                print(f"✓ ALEダッシュボード作成完了")
                print(f"  1D プロット: {len(dashboard_info['ale_1d_plots'])}個")
                print(f"  2D プロット: {len(dashboard_info['ale_2d_plots'])}個")
                
            except Exception as e:
                print(f"✗ ALEダッシュボード作成エラー: {e}")
            
            print(f"ALE プロット生成完了: {successful_plots}個のプロットを作成")
            
            # ALEレポート作成
            try:
                summary_path = os.path.join(ale_output_dir, f'ale_summary_{timestamp}.txt')
                with open(summary_path, 'w', encoding='utf-8') as f:
                    f.write("=== ALE分析サマリー ===\n")
                    f.write(f"実行日時: {datetime.datetime.now()}\n")
                    f.write(f"対象特徴量数: {len(feature_names_for_ale)}\n")
                    f.write(f"成功したプロット数: {successful_plots}\n")
                    f.write(f"出力ディレクトリ: {ale_output_dir}\n")
                    f.write(f"分析特徴量: {', '.join(feature_names_for_ale)}\n")
                
                print(f"ALEサマリー保存: {summary_path}")
            except Exception as e:
                print(f"ALEサマリー作成エラー: {e}")
            
        except Exception as e:
            print(f"ALE プロット生成中に重大なエラー: {e}")
            import traceback
            traceback.print_exc()

    def split_data(self, features, target, id_column=None):
        """
        データを開発用とテスト用に分割（層化サンプリング使用）
        """
        print(f"\n=== データ分割（分類用） ===")
        print("テストデータを分離して保持...")
        
        self.features = features
        self.target = target
        
        # 層化サンプリングでデータ分割
        if id_column is not None:
            self.X_dev, self.X_test, self.y_dev, self.y_test, self.id_dev, self.id_test = train_test_split(
                features, target, id_column,
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=target  # 分類問題では層化
            )
        else:
            self.X_dev, self.X_test, self.y_dev, self.y_test = train_test_split(
                features, target,
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=target  # 分類問題では層化
            )
            self.id_dev, self.id_test = None, None
        
        dev_ratio = len(self.X_dev) / len(features) * 100
        test_ratio = len(self.X_test) / len(features) * 100
        
        print(f"分割結果:")
        print(f"  開発データ: {len(self.X_dev)}サンプル ({dev_ratio:.1f}%)")
        print(f"  テストデータ: {len(self.X_test)}サンプル ({test_ratio:.1f}%)")
        
        # クラス分布の確認
        print(f"  開発データのクラス分布:")
        dev_counts = self.y_dev.value_counts().sort_index()
        for class_label, count in dev_counts.items():
            print(f"    クラス{class_label}: {count}サンプル ({count/len(self.y_dev)*100:.1f}%)")
        
        print(f"  テストデータのクラス分布:")
        test_counts = self.y_test.value_counts().sort_index()
        for class_label, count in test_counts.items():
            print(f"    クラス{class_label}: {count}サンプル ({count/len(self.y_test)*100:.1f}%)")
        
        # 分割データの保存（オプション）
        if self.save_splits and self.output_dir:
            self._save_split_data()
        
        return {
            'dev_size': len(self.X_dev),
            'test_size': len(self.X_test),
            'dev_ratio': dev_ratio,
            'test_ratio': test_ratio,
            'dev_class_distribution': dev_counts.to_dict(),
            'test_class_distribution': test_counts.to_dict()
        }

    def run_cross_validation(self):
        """
        開発データで分類用クロスバリデーションを実行
        """
        print(f"\n=== 分類用クロスバリデーション実行 ===")
        print(f"開発データ（{len(self.X_dev)}サンプル）で{self.n_splits}分割交差検証")
        
        # 開発データの配列変換
        X_dev_values = self.X_dev.values
        y_dev_values = self.y_dev.values
        
        # 層化交差検証の設定
        skf = StratifiedKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # 結果の初期化
        self.fold_metrics = []
        self.oof_predictions = np.zeros(len(y_dev_values))
        self.oof_probabilities = np.zeros(len(y_dev_values))
        self.feature_importances = []
        
        print(f"各フォールドでの処理: 層化サンプリング使用")
        
        # 各フォールドでの処理
        for fold, (train_idx, val_idx) in enumerate(skf.split(X_dev_values, y_dev_values)):
            print(f"\n--- Fold {fold+1}/{self.n_splits} ---")
            
            # データ分割
            X_train_fold = X_dev_values[train_idx]
            X_val_fold = X_dev_values[val_idx]
            y_train_fold = y_dev_values[train_idx]
            y_val_fold = y_dev_values[val_idx]
            
            print(f"  訓練: {len(X_train_fold)}サンプル, 検証: {len(X_val_fold)}サンプル")
            
            # クラス分布の確認
            train_class_counts = pd.Series(y_train_fold).value_counts().sort_index()
            val_class_counts = pd.Series(y_val_fold).value_counts().sort_index()
            print(f"  訓練データクラス分布: {train_class_counts.to_dict()}")
            print(f"  検証データクラス分布: {val_class_counts.to_dict()}")
            
            # SMOTE適用（必要に応じて）
            if self.use_smote:
                print(f"  SMOTEを訓練データに適用中...")
                try:
                    from imblearn.over_sampling import SMOTE
                    smote = SMOTE(random_state=self.random_state + fold)
                    X_train_fold, y_train_fold = smote.fit_resample(X_train_fold, y_train_fold)
                    print(f"    SMOTE後: {len(X_train_fold)}サンプル")
                except Exception as e:
                    print(f"    SMOTEエラー: {e}")
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # モデル学習
            model = self._create_model(fold)
            
            # CatBoostモデルの場合は特別な処理
            if hasattr(model, '__class__') and 'catboost' in model.__class__.__name__.lower():
                try:
                    eval_set = (X_val_scaled, y_val_fold)
                    model.fit(X_train_scaled, y_train_fold, eval_set=eval_set, verbose=False)
                except Exception as e:
                    print(f"  CatBoost評価セット付き学習でエラー: {e}")
                    print(f"  use_best_model=Falseで再試行...")
                    
                    if hasattr(model, 'model') and model.model is not None:
                        params = model.model.get_params()
                        params['use_best_model'] = False
                        params['early_stopping_rounds'] = None
                        
                        from catboost import CatBoostClassifier
                        new_catboost = CatBoostClassifier(**params)
                        new_catboost.fit(X_train_scaled, y_train_fold, verbose=False)
                        model.model = new_catboost
                    else:
                        model.fit(X_train_scaled, y_train_fold)
            else:
                if self.use_undersampling or self.use_simple_undersampling:
                    model.fit(X_train_scaled, y_train_fold)
                else:
                    model.fit(X_train_scaled, y_train_fold)
            
            # 検証データで予測
            if hasattr(model, 'predict_proba'):
                y_proba_fold = model.predict_proba(X_val_scaled)[:, 1]
                self.oof_probabilities[val_idx] = y_proba_fold
            else:
                y_proba_fold = model.predict(X_val_scaled).astype(float)
                self.oof_probabilities[val_idx] = y_proba_fold
            
            y_pred_fold = model.predict(X_val_scaled)
            self.oof_predictions[val_idx] = y_pred_fold
            
            # 評価指標計算
            fold_result = self._calculate_classification_metrics(y_val_fold, y_pred_fold, y_proba_fold, fold + 1)
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
        cv_results = self._summarize_classification_cv_results()
        
        print(f"\n=== 分類用クロスバリデーション結果 ===")
        print(f"開発データでのクロスバリデーション性能:")
        for metric, value in cv_results['cv_metrics'].items():
            std = cv_results['cv_std'][f'{metric}_std']
            print(f"  {metric.upper()}: {value:.4f} ± {std:.4f}")
        
        return cv_results

    def _create_model(self, fold=None):
        """モデルを作成"""
        if self.use_undersampling and UNDERSAMPLING_AVAILABLE:
            model = UndersamplingBaggingModel(
                base_model=self.kwargs.get('base_model', 'catboost'),
                n_bags=self.kwargs.get('n_bags', 5),
                random_state=self.random_state + (fold or 0)
            )
        elif self.use_simple_undersampling and UNDERSAMPLING_AVAILABLE:
            model = UndersamplingModel(
                base_model=self.kwargs.get('base_model', 'catboost'),
                random_state=self.random_state + (fold or 0)
            )
        else:
            model = self.model_class()
        
        return model

    def _calculate_classification_metrics(self, y_true, y_pred, y_proba, fold=None):
        """分類用評価指標を計算"""
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1': f1_score(y_true, y_pred, zero_division=0),
        }
        
        # AUCも計算（確率予測がある場合）
        if y_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
            except ValueError:
                metrics['roc_auc'] = 0.0
        
        if fold is not None:
            metrics['fold'] = fold
        
        return metrics

    def _summarize_classification_cv_results(self):
        """分類用クロスバリデーション結果をまとめる"""
        metrics_df = pd.DataFrame(self.fold_metrics)
        
        # 平均指標
        cv_metrics = {
            'accuracy': metrics_df['accuracy'].mean(),
            'precision': metrics_df['precision'].mean(),
            'recall': metrics_df['recall'].mean(),
            'f1': metrics_df['f1'].mean(),
        }
        
        # ROC-AUCがある場合は追加
        if 'roc_auc' in metrics_df.columns:
            cv_metrics['roc_auc'] = metrics_df['roc_auc'].mean()
        
        # 標準偏差
        cv_std = {
            'accuracy_std': metrics_df['accuracy'].std(),
            'precision_std': metrics_df['precision'].std(),
            'recall_std': metrics_df['recall'].std(),
            'f1_std': metrics_df['f1'].std(),
        }
        
        if 'roc_auc' in metrics_df.columns:
            cv_std['roc_auc_std'] = metrics_df['roc_auc'].std()
        
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
        """
        print(f"\n=== 最終モデル学習（分類用） ===")
        print("開発データ全体で最終モデルを学習...")
        
        # 開発データ全体の準備
        X_dev_values = self.X_dev.values
        y_dev_values = self.y_dev.values
        
        # SMOTE適用（必要に応じて）
        if self.use_smote:
            print(f"開発データ全体にSMOTEを適用中...")
            try:
                from imblearn.over_sampling import SMOTE
                smote = SMOTE(random_state=self.random_state)
                X_dev_values, y_dev_values = smote.fit_resample(X_dev_values, y_dev_values)
                print(f"  SMOTE適用後: {len(X_dev_values)}サンプル")
            except Exception as e:
                print(f"  SMOTEエラー: {e}")
        
        # 最終スケーリング
        self.final_scaler = StandardScaler()
        X_dev_scaled = self.final_scaler.fit_transform(X_dev_values)
        
        # 最終モデル学習
        self.final_model = self._create_model()
        self.final_model.fit(X_dev_scaled, y_dev_values)
        
        print(f"最終モデル学習完了")
        
        return {
            'model': self.final_model,
            'scaler': self.final_scaler,
            'training_size': len(X_dev_values),
            'original_size': len(self.X_dev)
        }

    def evaluate_test(self):
        """
        テストデータで最終評価（分類用）
        """
        print(f"\n=== テストデータで最終評価（分類用） ===")
        print("保持していたテストデータで最終性能を評価...")
        
        # テストデータで予測
        X_test_scaled = self.final_scaler.transform(self.X_test.values)
        test_pred = self.final_model.predict(X_test_scaled)
        
        # 確率予測も取得
        if hasattr(self.final_model, 'predict_proba'):
            test_proba = self.final_model.predict_proba(X_test_scaled)[:, 1]
        else:
            test_proba = None
        
        # テスト性能の計算
        test_metrics = self._calculate_classification_metrics(self.y_test, test_pred, test_proba)
        
        print(f"テストデータでの最終性能:")
        for metric, value in test_metrics.items():
            print(f"  {metric.upper()}: {value:.4f}")
        
        return {
            'test_metrics': test_metrics,
            'test_predictions': test_pred,
            'test_probabilities': test_proba,
            'test_true': self.y_test.values
        }

    def _save_split_data(self):
        """分割データをファイルに保存"""
        print(f"\n分割データを保存中...")
        splits_dir = os.path.join(self.output_dir, 'data_splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        try:
            # 開発データを保存
            dev_df = self.X_dev.copy()
            dev_df['target'] = self.y_dev
            if self.id_dev is not None:
                dev_df.insert(0, self.id_dev.name, self.id_dev)
            dev_path = os.path.join(splits_dir, 'development_split.csv')
            dev_df.to_csv(dev_path, index=False)
            
            # テストデータを保存
            test_df = self.X_test.copy()
            test_df['target'] = self.y_test
            if self.id_test is not None:
                test_df.insert(0, self.id_test.name, self.id_test)
            test_path = os.path.join(splits_dir, 'test_split.csv')
            test_df.to_csv(test_path, index=False)
            
            print(f"  開発データ: {dev_path}")
            print(f"  テストデータ: {test_path}")
            
        except Exception as e:
            print(f"分割データ保存エラー: {e}")

    def _create_classification_visualizations(self, cv_results, test_results):
        """分類用可視化を作成"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # 混同行列の作成
            cm = confusion_matrix(test_results['test_true'], test_results['test_predictions'])
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix (Test Data)')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            
            cm_path = os.path.join(self.output_dir, 'confusion_matrix.png')
            plt.savefig(cm_path, dpi=300, bbox_inches='tight')
            plt.close()
            print(f"混同行列保存: {cm_path}")
            
            # ROC曲線（確率予測がある場合）
            if test_results.get('test_probabilities') is not None:
                from sklearn.metrics import roc_curve, auc
                
                fpr, tpr, _ = roc_curve(test_results['test_true'], test_results['test_probabilities'])
                roc_auc = auc(fpr, tpr)
                
                plt.figure(figsize=(8, 6))
                plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
                plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
                plt.xlim([0.0, 1.0])
                plt.ylim([0.0, 1.05])
                plt.xlabel('False Positive Rate')
                plt.ylabel('True Positive Rate')
                plt.title('ROC Curve (Test Data)')
                plt.legend(loc="lower right")
                
                roc_path = os.path.join(self.output_dir, 'roc_curve.png')
                plt.savefig(roc_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"ROC曲線保存: {roc_path}")
            
            # 特徴量重要度
            if cv_results['avg_importance'] is not None:
                plt.figure(figsize=(10, 8))
                top_features = cv_results['avg_importance'].head(20)
                sns.barplot(data=top_features, y='feature', x='importance')
                plt.title('Feature Importance (Top 20)')
                plt.xlabel('Importance')
                plt.tight_layout()
                
                importance_path = os.path.join(self.output_dir, 'feature_importance.png')
                plt.savefig(importance_path, dpi=300, bbox_inches='tight')
                plt.close()
                print(f"特徴量重要度保存: {importance_path}")
            
            print(f"可視化完了")
            
        except Exception as e:
            print(f"可視化エラー: {e}")

    def run_analysis(self, df, target_column='target', data_file_name=None):
        """
        完全な分類分析を実行（ALE対応版）
        """
        print(f"\n{'='*70}")
        print(f"分類用テストデータ保持+クロスバリデーション分析開始")
        print(f"{'='*70}")
        
        # データ前処理（分類用）
        from main5 import prepare_data
        print(f"\n=== データ前処理（分類用） ===")
        try:
            features, target, id_column = prepare_data(df)
        except ValueError as e:
            if "too many values to unpack" in str(e):
                features, target = prepare_data(df)
                id_column = None
                print("警告: IDカラムが取得できませんでした")
            else:
                raise
        
        print(f"前処理後のデータ:")
        print(f"  特徴量数: {features.shape[1]}")
        print(f"  サンプル数: {features.shape[0]}")
        print(f"  クラス分布: {target.value_counts().to_dict()}")
        
        # ステップ1: データ分割
        split_info = self.split_data(features, target, id_column)
        
        # ステップ2: クロスバリデーション
        cv_results = self.run_cross_validation()
        
        # ステップ3: 最終モデル学習
        model_info = self.train_final_model()
        
        # ステップ4: テスト評価
        test_results = self.evaluate_test()
        
        # ステップ5: ALEプロット生成
        if self.generate_ale:
            self.generate_ale_plots(cv_results)
        
        # ステップ6: 結果保存
        self.save_results(cv_results, test_results, data_file_name)
        
        # 結果のまとめ
        results = {
            'cv_metrics': cv_results['cv_metrics'],
            'cv_std': cv_results['cv_std'],
            'test_metrics': test_results['test_metrics'],
            'fold_metrics': cv_results['fold_metrics'],
            'oof_predictions': self.oof_predictions,
            'oof_probabilities': self.oof_probabilities,
            'test_predictions': test_results['test_predictions'],
            'test_probabilities': test_results.get('test_probabilities'),
            'test_true': test_results['test_true'],
            'dev_true': self.y_dev.values,
            'final_model': self.final_model,
            'scaler': self.final_scaler,
            'feature_importance': cv_results['avg_importance'],
            'output_dir': self.output_dir,
            'data_splits': split_info
        }
        
        # サマリー表示
        self._print_classification_summary(cv_results, test_results)
        
        return results

    def save_results(self, cv_results, test_results, data_file_name=None):
        """
        分類結果を保存
        """
        if not self.output_dir:
            return
        
        print(f"\n=== 分類結果の保存 ===")
        
        # 可視化
        self._create_classification_visualizations(cv_results, test_results)
        
        # 結果をCSVに保存
        self._save_classification_csv_results(cv_results, test_results)
        
        # 設定情報保存
        self._save_classification_config(cv_results, test_results, data_file_name)
        
        print(f"結果保存完了: {self.output_dir}")

    def _save_classification_csv_results(self, cv_results, test_results):
        """分類結果をCSVに保存"""
        # Out-of-fold結果
        oof_results_df = pd.DataFrame({
            'true': self.y_dev.values,
            'predicted': self.oof_predictions,
            'probability': self.oof_probabilities,
            'correct': (self.y_dev.values == self.oof_predictions).astype(int)
        })
        
        if self.id_dev is not None:
            oof_results_df.insert(0, self.id_dev.name, self.id_dev.values)
        
        oof_path = os.path.join(self.output_dir, 'oof_classification_results.csv')
        oof_results_df.to_csv(oof_path, index=False)
        
        # テスト結果
        test_results_df = pd.DataFrame({
            'true': test_results['test_true'],
            'predicted': test_results['test_predictions'],
            'correct': (test_results['test_true'] == test_results['test_predictions']).astype(int)
        })
        
        if test_results.get('test_probabilities') is not None:
            test_results_df['probability'] = test_results['test_probabilities']
        
        if self.id_test is not None:
            test_results_df.insert(0, self.id_test.name, self.id_test.values)
        
        test_path = os.path.join(self.output_dir, 'test_classification_results.csv')
        test_results_df.to_csv(test_path, index=False)
        
        print(f"Out-of-fold結果: {oof_path}")
        print(f"テスト結果: {test_path}")

    def _save_classification_config(self, cv_results, test_results, data_file_name):
        """分類設定情報を保存"""
        config_path = os.path.join(self.output_dir, 'config_classification_holdout_cv.txt')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("=== 分類用テストデータ保持+クロスバリデーション設定 ===\n")
            f.write(f"データファイル: {data_file_name}\n")
            f.write(f"開発データ: {(1-self.test_size)*100:.1f}% ({len(self.X_dev)}サンプル)\n")
            f.write(f"テストデータ: {self.test_size*100:.1f}% ({len(self.X_test)}サンプル)\n")
            f.write(f"CV分割数: {self.n_splits}\n")
            
            if self.use_undersampling:
                f.write(f"\n=== アンダーサンプリング設定 ===\n")
                f.write(f"バギング使用: UndersamplingBagging\n")
                f.write(f"ベースモデル: {self.kwargs.get('base_model', 'catboost')}\n")
                f.write(f"バッグ数: {self.kwargs.get('n_bags', 5)}\n")
            elif self.use_simple_undersampling:
                f.write(f"\n=== シンプルアンダーサンプリング設定 ===\n")
                f.write(f"ベースモデル: {self.kwargs.get('base_model', 'catboost')}\n")
            
            if self.use_smote:
                f.write(f"\n=== SMOTE設定 ===\n")
                f.write(f"SMOTE使用: 有効\n")
            
            if self.generate_ale:
                f.write(f"\n=== ALE設定 ===\n")
                f.write(f"ALE生成: 有効\n")
                f.write(f"分析特徴量数: {self.ale_n_features}\n")
                f.write(f"グリッド解像度: {self.ale_grid_resolution}\n")
                f.write(f"2次元ALE: {'有効' if self.ale_include_2d else '無効'}\n")
                f.write(f"ALE vs PDP: {'有効' if self.ale_vs_pdp else '無効'}\n")
            
            f.write(f"\n=== CV性能（開発データ） ===\n")
            for metric, value in cv_results['cv_metrics'].items():
                std = cv_results['cv_std'][f'{metric}_std']
                f.write(f"{metric.upper()}: {value:.4f} ± {std:.4f}\n")
            
            f.write(f"\n=== テスト性能（最終評価） ===\n")
            for metric, value in test_results['test_metrics'].items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
                
            f.write(f"\n実行日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        print(f"設定情報: {config_path}")

    def _print_classification_summary(self, cv_results, test_results):
        """分類結果サマリーを表示"""
        print(f"\n{'='*70}")
        print(f"分類用テストデータ保持+クロスバリデーション分析完了")
        
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
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in cv_results['cv_metrics'] and metric in test_results['test_metrics']:
                diff = test_results['test_metrics'][metric] - cv_results['cv_metrics'][metric]
                print(f"  {metric.upper()}差: {diff:+.4f}")
        
        if 'roc_auc' in cv_results['cv_metrics'] and 'roc_auc' in test_results['test_metrics']:
            auc_diff = test_results['test_metrics']['roc_auc'] - cv_results['cv_metrics']['roc_auc']
            print(f"  ROC-AUC差: {auc_diff:+.4f}")
        
        # 特徴量重要度の表示
        if cv_results['avg_importance'] is not None and len(cv_results['avg_importance']) > 0:
            print(f"\n特徴量の平均重要度（上位10件）:")
            print(cv_results['avg_importance'].head(10).to_string(index=False))
        else:
            print(f"\n特徴量重要度が取得できませんでした")
        
        # ALEプロット生成状況
        if self.generate_ale:
            print(f"\nALEプロット生成: 完了")
            ale_output_dir = os.path.join(self.output_dir, 'ale_plots')
            if os.path.exists(ale_output_dir):
                ale_files = [f for f in os.listdir(ale_output_dir) if f.endswith('.png')]
                print(f"  生成されたALEプロット: {len(ale_files)}個")
        
        if self.output_dir:
            print(f"\n結果保存先: {self.output_dir}")
        print(f"{'='*70}")


def run_classification_holdout_cv_analysis(df, model_class=None, use_undersampling=False,
                                         use_simple_undersampling=False, n_splits=5, test_size=0.2,
                                         use_smote=False, use_smotetomek=False, random_state=42,
                                         data_file_name=None, output_dir='classification_result',
                                         generate_ale=True, ale_n_features=6, ale_grid_resolution=30,
                                         ale_include_2d=False, ale_vs_pdp=False,
                                         save_splits=False, **kwargs):
    """
    分類用テストデータ保持+クロスバリデーション分析を実行する関数（ALE対応版）
    """
    
    print(f"\n{'='*60}")
    print(f"分類用テストデータ保持+クロスバリデーション分析開始（ALE対応版）")
    print(f"{'='*60}")
    
    # 分割比率の表示
    dev_size = 1 - test_size
    print(f"データ分割比率:")
    print(f"  開発用（学習+検証）: {dev_size*100:.1f}%")
    print(f"  テスト用: {test_size*100:.1f}%")
    print(f"  CV分割数: {n_splits}")
    print(f"  ALE生成: {'有効' if generate_ale else '無効'}")
    
    # 出力ディレクトリの設定
    if output_dir and data_file_name:
        base_name = os.path.splitext(os.path.basename(data_file_name))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        
        method_suffix = ""
        if use_undersampling:
            method_suffix = f"_usbag_{kwargs.get('base_model', 'catboost')}"
        elif use_simple_undersampling:
            method_suffix = f"_usimple_{kwargs.get('base_model', 'catboost')}"
        elif model_class:
            method_suffix = f"_{model_class.__name__.lower()}"
        
        if use_smote:
            method_suffix += "_smote"
        if use_smotetomek:
            method_suffix += "_smotetomek"
        
        output_dir = f"{base_name}_classification_holdout_cv{method_suffix}_{timestamp}"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"出力ディレクトリを作成: {output_dir}")
    
    # 分析器の作成と実行
    analyzer = ClassificationHoldoutCVAnalyzer(
        model_class=model_class,
        use_undersampling=use_undersampling,
        use_simple_undersampling=use_simple_undersampling,
        n_splits=n_splits,
        test_size=test_size,
        random_state=random_state,
        output_dir=output_dir,
        use_smote=use_smote,
        use_smotetomek=use_smotetomek,
        generate_ale=generate_ale,
        ale_n_features=ale_n_features,
        ale_grid_resolution=ale_grid_resolution,
        ale_include_2d=ale_include_2d,
        ale_vs_pdp=ale_vs_pdp,
        save_splits=save_splits,
        **kwargs
    )
    
    # 分析実行
    results = analyzer.run_analysis(df, data_file_name=data_file_name)
    
    return results


if __name__ == "__main__":
    print("分類用テストデータ保持+クロスバリデーション分析器（ALE対応版）")
    print("使用例:")
    print("""
    # 基本的な使用例
    results = run_classification_holdout_cv_analysis(
        df=df,
        model_class=CatBoostModel,
        n_splits=5,
        test_size=0.2,
        data_file_name="data.csv",
        generate_ale=True,
        ale_n_features=6
    )
    
    # アンダーサンプリング + ALE
    results = run_classification_holdout_cv_analysis(
        df=df,
        use_undersampling=True,
        base_model='catboost',
        n_bags=10,
        generate_ale=True,
        ale_include_2d=True,
        ale_vs_pdp=True
    )
    """)