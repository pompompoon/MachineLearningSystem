"""
テストデータ保持+クロスバリデーション分析器
"""

import numpy as np
import pandas as pd
import os
import datetime
import sys

# scikit-learn
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# scipy
from scipy.stats import pearsonr, spearmanr
# ファイル上部のインポート部分に追加
try:
    from ale_plotter import AccumulatedLocalEffects, analyze_accumulated_local_effects
    ALE_AVAILABLE = True
except ImportError:
    print("Warning: ale_plotter not found. ALE plots will be disabled.")
    ALE_AVAILABLE = False

# パス設定（main5k_ale2cvt.pyからのインポート用）
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# 安全なインポート用のヘルパー関数
def safe_import_from_main():
    """main5k_ale2cvt.pyから安全にインポート"""
    try:
        from main5k_ale2cvt import spearman_correlation, mean_absolute_percentage_error, RegressionBaggingModel
        return spearman_correlation, mean_absolute_percentage_error, RegressionBaggingModel
    except ImportError as e:
        print(f"Warning: Could not import from main5k_ale2cvt.py: {e}")
        
        # フォールバック実装
        def mean_absolute_percentage_error(y_true, y_pred):
            epsilon = np.finfo(np.float64).eps
            mask = np.abs(y_true) > epsilon
            if np.sum(mask) == 0:
                return np.nan
            y_true_masked = y_true[mask]
            y_pred_masked = y_pred[mask]
            return np.mean(np.abs((y_true_masked - y_pred_masked) / y_true_masked)) * 100
        
        def spearman_correlation(x, y):
            corr, _ = spearmanr(x, y)
            return corr
        
        # RegressionBaggingModelのダミー実装
        class RegressionBaggingModel:
            def __init__(self, **kwargs):
                raise ImportError("RegressionBaggingModel could not be imported")
        
        return spearman_correlation, mean_absolute_percentage_error, RegressionBaggingModel

# グローバルにインポート
spearman_correlation, mean_absolute_percentage_error, RegressionBaggingModel = safe_import_from_main()


class HoldoutCVAnalyzer:
    """
    テストデータを保持してクロスバリデーションを実行するクラス
    
    使用例:
    -------
    analyzer = HoldoutCVAnalyzer(
        model_class=CatBoostRegressor,
        n_splits=5,
        test_size=0.2,
        use_smote=True,
        smote_method='density'
    )
    
    results = analyzer.run_analysis(df, target_column='target')
    """
    
    def __init__(self, model_class=None, use_bagging=False, n_splits=5, 
                 test_size=0.2, random_state=42, output_dir='result',
                 # SMOTE関連パラメータ
                 use_smote=False, smote_method='density', smote_kwargs=None,
                 use_integer_smote=True, target_min=9, target_max=30,
                 # ALE関連パラメータを追加
                 generate_ale=False, ale_n_features=6, ale_grid_resolution=30,
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
        smote_method : str
            SMOTEの手法
        smote_kwargs : dict
            SMOTEのパラメータ
        use_integer_smote : bool
            整数値対応SMOTEを使用するか
        target_min : int
            目的変数の最小値
        target_max : int
            目的変数の最大値
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
        self.generate_ale = generate_ale and ALE_AVAILABLE
        self.ale_n_features = ale_n_features
        self.ale_grid_resolution = ale_grid_resolution
        # SMOTE設定
        self.use_smote = use_smote
        self.smote_method = smote_method
        self.smote_kwargs = smote_kwargs or self._get_default_smote_kwargs()
        self.use_integer_smote = use_integer_smote
        self.target_min = target_min
        self.target_max = target_max
        
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
        self.feature_importances = []
        self.smote_statistics = []
        self.final_model = None
        self.final_scaler = None
    def generate_ale_plots(self, cv_results):
        """ALE（Accumulated Local Effects）プロットを生成"""
        if not self.generate_ale or not ALE_AVAILABLE:
            return
        
        print(f"\n=== ALE プロット生成 ===")
        
        # ALE出力ディレクトリ
        ale_output_dir = os.path.join(self.output_dir, 'ale_plots') if self.output_dir else 'ale_plots'
        os.makedirs(ale_output_dir, exist_ok=True)
        
        try:
            # 開発データ全体でALEを計算
            X_dev_for_ale = self.X_dev  # pandas DataFrameのまま使用
            
            print(f"ALE計算用データ: {X_dev_for_ale.shape}")
            print(f"使用モデル: {type(self.final_model).__name__}")
            print(f"対象特徴量数: {self.ale_n_features}")
            
            # ALEインスタンス作成
            ale = AccumulatedLocalEffects(self.final_model, X_dev_for_ale)
            
            # 特徴量重要度がある場合は上位特徴量を選択
            feature_indices = []
            if cv_results['avg_importance'] is not None and len(cv_results['avg_importance']) > 0:
                top_features = cv_results['avg_importance'].head(self.ale_n_features)['feature'].tolist()
                for feature in top_features:
                    if feature in ale.feature_names:
                        feature_indices.append(ale.feature_names.index(feature))
            else:
                # 特徴量重要度がない場合は最初の特徴量を使用
                feature_indices = list(range(min(self.ale_n_features, len(ale.feature_names))))
            
            if not feature_indices:
                print("ALE用の有効な特徴量が見つかりません")
                return
            
            print(f"選択された特徴量: {[ale.feature_names[i] for i in feature_indices]}")
            
            # 複数特徴量のALEプロット
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            print("複数特徴量ALEプロット生成中...")
            fig = ale.plot_multiple_features(feature_indices, n_grid=self.ale_grid_resolution)
            
            ale_multi_path = os.path.join(ale_output_dir, f'ale_top{len(feature_indices)}_holdout_cv_{timestamp}.png')
            fig.savefig(ale_multi_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"✓ ALE複数特徴量プロット保存: {ale_multi_path}")
            
            # 個別特徴量のALEプロット（上位3つ）
            for i, feature_idx in enumerate(feature_indices[:3]):
                feature_name = ale.feature_names[feature_idx]
                print(f"個別ALEプロット生成: {feature_name}")
                
                fig = ale.plot_single_feature(feature_idx, n_grid=self.ale_grid_resolution)
                
                safe_name = feature_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
                ale_single_path = os.path.join(ale_output_dir, f'ale_{safe_name}_holdout_cv_{timestamp}.png')
                fig.savefig(ale_single_path, dpi=300, bbox_inches='tight')
                plt.close(fig)
                print(f"✓ ALE個別プロット保存: {ale_single_path}")
            
            print(f"ALE プロット生成完了")
            
        except Exception as e:
            print(f"ALE プロット生成中にエラー: {e}")
            import traceback
            traceback.print_exc()    
    def _get_default_smote_kwargs(self):
        """SMOTEのデフォルトパラメータを取得"""
        return {
            'binning': {'sampling_strategy': 'auto', 'n_bins': 10},
            'density': {'density_threshold': 0.3},
            'outliers': {'outlier_threshold': 0.15}
        }.get(self.smote_method, {})
    
    def _setup_output_directory(self, data_file_name=None):
        """出力ディレクトリの設定"""
        if self.output_dir and data_file_name:
            base_name = os.path.splitext(os.path.basename(data_file_name))[0]
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
            smote_suffix = f"_smote_{self.smote_method}" if self.use_smote else ""
            self.output_dir = f"{base_name}_holdout_cv{smote_suffix}_{timestamp}"
            
            if not os.path.exists(self.output_dir):
                os.makedirs(self.output_dir)
                print(f"出力ディレクトリを作成: {self.output_dir}")
        
        return self.output_dir
    
    def split_data(self, features, target, id_column=None):
        """
        データを開発用とテスト用に分割
        
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
        print(f"\n=== データ分割 ===")
        print("テストデータを分離して保持...")
        
        self.features = features
        self.target = target
        
        # データ分割
        if id_column is not None:
            self.X_dev, self.X_test, self.y_dev, self.y_test, self.id_dev, self.id_test = train_test_split(
                features, target, id_column,
                test_size=self.test_size, 
                random_state=self.random_state,
                stratify=None  # 回帰問題なので層化しない
            )
        else:
            self.X_dev, self.X_test, self.y_dev, self.y_test = train_test_split(
                features, target,
                test_size=self.test_size, 
                random_state=self.random_state
            )
            self.id_dev, self.id_test = None, None
        
        dev_ratio = len(self.X_dev) / len(features) * 100
        test_ratio = len(self.X_test) / len(features) * 100
        
        print(f"分割結果:")
        print(f"  開発データ: {len(self.X_dev)}サンプル ({dev_ratio:.1f}%)")
        print(f"  テストデータ: {len(self.X_test)}サンプル ({test_ratio:.1f}%)")
        print(f"  テストデータは最終評価まで使用しません")
        
        # 分割データの保存（オプション）
        if self.save_splits and self.output_dir:
            self._save_split_data()
        
        return {
            'dev_size': len(self.X_dev),
            'test_size': len(self.X_test),
            'dev_ratio': dev_ratio,
            'test_ratio': test_ratio
        }
    
    def _save_split_data(self):
        """分割データをファイルに保存"""
        print(f"\n分割データを保存中...")
        splits_dir = os.path.join(self.output_dir, 'data_splits')
        os.makedirs(splits_dir, exist_ok=True)
        
        # 開発データを保存
        dev_df = self.X_dev.copy()
        dev_df[self.target.name] = self.y_dev
        if self.id_dev is not None:
            dev_df.insert(0, self.id_dev.name, self.id_dev)
        dev_path = os.path.join(splits_dir, 'dev_data.csv')
        dev_df.to_csv(dev_path, index=False)
        
        # テストデータを保存
        test_df = self.X_test.copy()
        test_df[self.target.name] = self.y_test
        if self.id_test is not None:
            test_df.insert(0, self.id_test.name, self.id_test)
        test_path = os.path.join(splits_dir, 'test_data.csv')
        test_df.to_csv(test_path, index=False)
        
        print(f"  開発データ: {dev_path}")
        print(f"  テストデータ: {test_path}")
    
    def _apply_smote(self, X_train, y_train, fold=None):
        """
        SMOTEを適用
        
        Parameters:
        -----------
        X_train : np.ndarray
            訓練データの特徴量
        y_train : np.ndarray
            訓練データの目的変数
        fold : int, optional
            フォールド番号（乱数シード調整用）
            
        Returns:
        --------
        tuple
            (X_train_resampled, y_train_resampled)
        """
        if not self.use_smote:
            return X_train, y_train
        
        try:
            # SMOTEインスタンス作成
            seed = self.random_state + (fold or 0)
            
            if self.use_integer_smote:
                try:
                    from regression_smote import IntegerRegressionSMOTE
                    smote_instance = IntegerRegressionSMOTE(
                        method=self.smote_method, 
                        k_neighbors=5, 
                        random_state=seed,
                        target_min=self.target_min,
                        target_max=self.target_max
                    )
                except ImportError:
                    from regression_smotebackup import RegressionSMOTE
                    smote_instance = RegressionSMOTE(
                        method=self.smote_method, 
                        k_neighbors=5, 
                        random_state=seed
                    )
            else:
                from regression_smotebackup import RegressionSMOTE
                smote_instance = RegressionSMOTE(
                    method=self.smote_method, 
                    k_neighbors=5, 
                    random_state=seed
                )
            
            # SMOTE適用
            X_resampled, y_resampled = smote_instance.fit_resample(
                X_train, y_train, **self.smote_kwargs
            )
            
            return X_resampled, y_resampled
            
        except Exception as e:
            print(f"    SMOTEエラー: {e}")
            return X_train, y_train
    
    def run_cross_validation(self):
        """
        開発データでクロスバリデーションを実行
        
        Returns:
        --------
        dict
            クロスバリデーション結果
        """
        print(f"\n=== クロスバリデーション実行 ===")
        print(f"開発データ（{len(self.X_dev)}サンプル）で{self.n_splits}分割クロスバリデーション")
        
        # 開発データの配列変換
        X_dev_values = self.X_dev.values
        y_dev_values = self.y_dev.values
        
        # 交差検証の設定
        kf = KFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        
        # 結果の初期化
        self.fold_metrics = []
        self.oof_predictions = np.zeros(len(y_dev_values))
        self.feature_importances = []
        self.smote_statistics = []
        
        print(f"各フォールドで訓練データにSMOTE適用: {'あり' if self.use_smote else 'なし'}")
        
        # 各フォールドでの処理
        for fold, (train_idx, val_idx) in enumerate(kf.split(X_dev_values)):
            print(f"\n--- Fold {fold+1}/{self.n_splits} ---")
            
            # データ分割
            X_train_fold = X_dev_values[train_idx]
            X_val_fold = X_dev_values[val_idx]
            y_train_fold = y_dev_values[train_idx]
            y_val_fold = y_dev_values[val_idx]
            
            print(f"  訓練: {len(X_train_fold)}サンプル, 検証: {len(X_val_fold)}サンプル")
            
            # 元の訓練データサイズを記録
            original_train_size = len(X_train_fold)
            
            # SMOTE適用（訓練データのみ）
            if self.use_smote:
                print(f"  SMOTE（{self.smote_method}）を訓練データに適用中...")
                X_train_fold, y_train_fold = self._apply_smote(X_train_fold, y_train_fold, fold)
                
                synthetic_count = len(X_train_fold) - original_train_size
                print(f"    SMOTE後: {len(X_train_fold)}サンプル (+{synthetic_count})")
                
                # SMOTE統計を記録
                smote_stats = {
                    'fold': fold + 1,
                    'original_size': original_train_size,
                    'resampled_size': len(X_train_fold),
                    'synthetic_count': synthetic_count
                }
                self.smote_statistics.append(smote_stats)
            
            # スケーリング
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train_fold)
            X_val_scaled = scaler.transform(X_val_fold)
            
            # モデル学習
            model = self._create_model(fold)
            model.fit(X_train_scaled, y_train_fold)
            
            # 検証データで予測
            y_pred_fold = model.predict(X_val_scaled)
            self.oof_predictions[val_idx] = y_pred_fold
            
            # 評価指標計算
            fold_result = self._calculate_metrics(y_val_fold, y_pred_fold, fold + 1)
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
            if 'p_value' in metric:
                print(f"  {metric.upper()}: {value:.6f} ± {std:.6f}")  # p値は6桁表示
            else:
                print(f"  {metric.upper()}: {value:.4f} ± {std:.4f}")
        
        return cv_results
    
    def _create_model(self, fold=None):
        """モデルを作成"""
        if self.use_bagging:
            # RegressionBaggingModelをインポート（必要に応じて）
            from main5k_ale2cvt import RegressionBaggingModel
            model = RegressionBaggingModel(
                base_model=self.kwargs.get('base_model', 'lightgbm'),
                n_bags=self.kwargs.get('n_bags', 10),
                random_state=self.random_state + (fold or 0)
            )
        else:
            model = self.model_class()
        
        return model
    
    def _calculate_metrics(self, y_true, y_pred, fold=None):
        """評価指標を計算"""
        # spearman_correlationをインポート（必要に応じて）
        
        
        # ピアソン相関係数とp値
        pearson_corr, pearson_p_value = pearsonr(y_true, y_pred)
    
    # スピアマン相関係数とp値（scipyのspearmanrを使用）
        spearman_corr, spearman_p_value = spearmanr(y_true, y_pred)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mape': mean_absolute_percentage_error(y_true, y_pred),
            'pearson_corr': pearson_corr,
            'pearson_p_value': pearson_p_value, 
            'spearman_corr': spearman_corr,
            'spearman_p_value': spearman_p_value
        }
        
        if fold is not None:
            metrics['fold'] = fold
        
        return metrics
    
    def _summarize_cv_results(self):
        """クロスバリデーション結果をまとめる"""
        metrics_df = pd.DataFrame(self.fold_metrics)
        
        # 平均指標
        cv_metrics = {
            'rmse': metrics_df['rmse'].mean(),
            'mae': metrics_df['mae'].mean(),
            'r2': metrics_df['r2'].mean(),
            'mape': metrics_df['mape'].mean(),
            'pearson_corr': metrics_df['pearson_corr'].mean(),
            'pearson_p_value': metrics_df['pearson_p_value'].mean(),
            'spearman_corr': metrics_df['spearman_corr'].mean(),
            'spearman_p_value': metrics_df['spearman_p_value'].mean() 
        }
        
        # 標準偏差
        cv_std = {
            'rmse_std': metrics_df['rmse'].std(),
            'mae_std': metrics_df['mae'].std(),
            'r2_std': metrics_df['r2'].std(),
            'mape_std': metrics_df['mape'].std(),
            'pearson_corr_std': metrics_df['pearson_corr'].std(),
            'pearson_p_value_std': metrics_df['pearson_p_value'].std(),
            'spearman_corr_std': metrics_df['spearman_corr'].std(),
            'spearman_p_value_std': metrics_df['spearman_p_value'].std()
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
        
        # 開発データ全体にSMOTE適用
        X_dev_values = self.X_dev.values
        y_dev_values = self.y_dev.values
        
        if self.use_smote:
            print(f"開発データ全体にSMOTE（{self.smote_method}）を適用中...")
            X_dev_final, y_dev_final = self._apply_smote(X_dev_values, y_dev_values)
            
            synthetic_count = len(X_dev_final) - len(X_dev_values)
            print(f"  SMOTE適用前: {len(X_dev_values)}サンプル")
            print(f"  SMOTE適用後: {len(X_dev_final)}サンプル (+{synthetic_count})")
        else:
            X_dev_final = X_dev_values
            y_dev_final = y_dev_values
        
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
            'original_size': len(X_dev_values)
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
        
        # テスト性能の計算
        test_metrics = self._calculate_metrics(self.y_test, test_pred)
        
        print(f"テストデータでの最終性能:")
        for metric, value in test_metrics.items():
            if 'p_value' in metric:
                print(f"  {metric.upper()}: {value:.6f}")  # p値は6桁表示
            else:
                print(f"  {metric.upper()}: {value:.4f}")
        
        return {
            'test_metrics': test_metrics,
            'test_predictions': test_pred,
            'test_true': self.y_test.values
        }
    
    def save_results(self, cv_results, test_results, data_file_name=None):
        """
        結果を保存
        
        Parameters:
        -----------
        cv_results : dict
            クロスバリデーション結果
        test_results : dict
            テスト結果
        data_file_name : str, optional
            データファイル名
        """
        if not self.output_dir:
            return
        
        print(f"\n=== 結果の保存 ===")
        
        # 可視化
        self._create_visualizations(cv_results, test_results)
        # ALE プロット生成
        if self.generate_ale:
            self.generate_ale_plots(cv_results)
        # 結果をCSVに保存
        self._save_csv_results(cv_results, test_results)
        
        # 設定情報保存
        self._save_config(cv_results, test_results, data_file_name)
        
        print(f"結果保存完了: {self.output_dir}")
    
    def _create_visualizations(self, cv_results, test_results):
        """可視化を作成"""
        try:
            # EyeTrackingVisualizerをインポート（必要に応じて）
            from visualization.regression_visualizer import EyeTrackingVisualizer
            visualizer = EyeTrackingVisualizer(output_dir=self.output_dir)
            
            # Out-of-fold予測の可視化
            visualizer.plot_true_vs_predicted(self.y_dev.values, self.oof_predictions)
            
            # テストデータの可視化
            visualizer.plot_true_vs_predicted(test_results['test_true'], test_results['test_predictions'])
            
            # 特徴量重要度
            if cv_results['avg_importance'] is not None:
                visualizer.plot_feature_importance(cv_results['avg_importance'])
            
            print(f"可視化完了")
            
        except Exception as e:
            print(f"可視化エラー: {e}")
            print("可視化をスキップして処理を続行します...")
    def _save_csv_results(self, cv_results, test_results):
        """結果をCSVに保存"""
        # Out-of-fold結果
        oof_results_df = pd.DataFrame({
            'true': self.y_dev.values,
            'predicted': self.oof_predictions,
            'residual': self.y_dev.values - self.oof_predictions,
            'abs_residual': np.abs(self.y_dev.values - self.oof_predictions)
        })
        
        if self.id_dev is not None:
            oof_results_df.insert(0, self.id_dev.name, self.id_dev.values)
        
        oof_path = os.path.join(self.output_dir, 'oof_results.csv')
        oof_results_df.to_csv(oof_path, index=False)
        
        # テスト結果
        test_results_df = pd.DataFrame({
            'true': test_results['test_true'],
            'predicted': test_results['test_predictions'],
            'residual': test_results['test_true'] - test_results['test_predictions'],
            'abs_residual': np.abs(test_results['test_true'] - test_results['test_predictions'])
        })
        
        if self.id_test is not None:
            test_results_df.insert(0, self.id_test.name, self.id_test.values)
        
        test_path = os.path.join(self.output_dir, 'test_results.csv')
        test_results_df.to_csv(test_path, index=False)
        
        print(f"Out-of-fold結果: {oof_path}")
        print(f"テスト結果: {test_path}")
    
    def _save_config(self, cv_results, test_results, data_file_name):
        """設定情報を保存"""
        config_path = os.path.join(self.output_dir, 'config_holdout_cv.txt')
        with open(config_path, 'w', encoding='utf-8') as f:
            f.write("=== テストデータ保持+クロスバリデーション設定 ===\n")
            f.write(f"データファイル: {data_file_name}\n")
            f.write(f"開発データ: {(1-self.test_size)*100:.1f}% ({len(self.X_dev)}サンプル)\n")
            f.write(f"テストデータ: {self.test_size*100:.1f}% ({len(self.X_test)}サンプル)\n")
            f.write(f"CV分割数: {self.n_splits}\n")
            
            if self.use_smote:
                f.write(f"\n=== SMOTE設定 ===\n")
                f.write(f"手法: {self.smote_method}\n")
                f.write(f"整数値対応: {'有効' if self.use_integer_smote else '無効'}\n")
                for key, value in self.smote_kwargs.items():
                    f.write(f"{key}: {value}\n")
            
            f.write(f"\n=== クロスバリデーション性能（開発データ） ===\n")
            for metric, value in cv_results['cv_metrics'].items():
                std = cv_results['cv_std'][f'{metric}_std']
                f.write(f"{metric.upper()}: {value:.4f} ± {std:.4f}\n")
            
            f.write(f"\n=== テストデータでの最終性能 ===\n")
            for metric, value in test_results['test_metrics'].items():
                f.write(f"{metric.upper()}: {value:.4f}\n")
                
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
        print(f"テストデータ保持+クロスバリデーション分析開始")
        print(f"{'='*70}")
        
        # 分割比率の表示
        dev_size = 1 - self.test_size
        print(f"データ分割比率:")
        print(f"  開発データ: {dev_size*100:.1f}% (クロスバリデーション用)")
        print(f"  テストデータ: {self.test_size*100:.1f}% (最終評価専用)")
        print(f"  CV分割数: {self.n_splits}")
        
        # 出力ディレクトリ設定
        self._setup_output_directory(data_file_name)
        
        # データ前処理
        from main5k_ale2cvt import prepare_data_for_regression
        print(f"\n=== データ前処理 ===")
        features, target, id_column = prepare_data_for_regression(df, target_column=target_column)
        
        print(f"前処理後のデータ:")
        print(f"  特徴量数: {features.shape[1]}")
        print(f"  サンプル数: {features.shape[0]}")
        print(f"  目的変数範囲: {target.min():.2f} - {target.max():.2f}")
        
        # ステップ1: データ分割
        split_info = self.split_data(features, target, id_column)
        
        # ステップ2: クロスバリデーション
        cv_results = self.run_cross_validation()
        
        # ステップ3: 最終モデル学習
        model_info = self.train_final_model()
        
        # ステップ4: テスト評価
        test_results = self.evaluate_test()
        
        # ステップ5: 結果保存
        self.save_results(cv_results, test_results, data_file_name)
        
        # 結果のまとめ
        results = {
            'cv_metrics': cv_results['cv_metrics'],
            'cv_std': cv_results['cv_std'],
            'test_metrics': test_results['test_metrics'],
            'fold_metrics': cv_results['fold_metrics'],
            'oof_predictions': self.oof_predictions,
            'test_predictions': test_results['test_predictions'],
            'test_true': test_results['test_true'],
            'dev_true': self.y_dev.values,
            'final_model': self.final_model,
            'scaler': self.final_scaler,
            'feature_importance': cv_results['avg_importance'],
            'smote_statistics': self.smote_statistics,
            'output_dir': self.output_dir,
            'data_splits': split_info
        }
        
        # サマリー表示
        self._print_summary(cv_results, test_results)
        
        return results
    
    # _print_summaryメソッドを以下に置き換え
    def _print_summary(self, cv_results, test_results):
        """結果サマリーを表示"""
        print(f"\n{'='*70}")
        print(f"テストデータ保持+クロスバリデーション分析完了")
        
        # CV性能表示
        print(f"\nCV性能（開発データ）:")
        for metric, value in cv_results['cv_metrics'].items():
            std = cv_results['cv_std'][f'{metric}_std']
            if 'p_value' in metric:
                print(f"  {metric.upper()}: {value:.6f} ± {std:.6f}")  # p値は6桁表示
            else:
                print(f"  {metric.upper()}: {value:.4f} ± {std:.4f}")
        
        # テスト性能表示
        print(f"\n最終性能（テストデータ）:")
        for metric, value in test_results['test_metrics'].items():
            if 'p_value' in metric:
                print(f"  {metric.upper()}: {value:.6f}")  # p値は6桁表示
            else:
                print(f"  {metric.upper()}: {value:.4f}")
        
        # 性能差表示
        print(f"\n性能差（テスト - CV）:")
        print(f"  RMSE差: {test_results['test_metrics']['rmse'] - cv_results['cv_metrics']['rmse']:+.4f}")
        print(f"  R²差: {test_results['test_metrics']['r2'] - cv_results['cv_metrics']['r2']:+.4f}")
        print(f"  Pearson相関差: {test_results['test_metrics']['pearson_corr'] - cv_results['cv_metrics']['pearson_corr']:+.4f}")
        print(f"  Pearson p値差: {test_results['test_metrics']['pearson_p_value'] - cv_results['cv_metrics']['pearson_p_value']:+.6f}")
        print(f"  Spearman相関差: {test_results['test_metrics']['spearman_corr'] - cv_results['cv_metrics']['spearman_corr']:+.4f}")
        print(f"  Spearman p値差: {test_results['test_metrics']['spearman_p_value'] - cv_results['cv_metrics']['spearman_p_value']:+.6f}")
        
        # 統計的有意性の解釈を追加
        print(f"\n=== 統計的有意性（α=0.05） ===")
        print(f"CV ピアソン相関: {'有意' if cv_results['cv_metrics']['pearson_p_value'] < 0.05 else '非有意'} (p={cv_results['cv_metrics']['pearson_p_value']:.6f})")
        print(f"CV スピアマン相関: {'有意' if cv_results['cv_metrics']['spearman_p_value'] < 0.05 else '非有意'} (p={cv_results['cv_metrics']['spearman_p_value']:.6f})")
        print(f"テスト ピアソン相関: {'有意' if test_results['test_metrics']['pearson_p_value'] < 0.05 else '非有意'} (p={test_results['test_metrics']['pearson_p_value']:.6f})")
        print(f"テスト スピアマン相関: {'有意' if test_results['test_metrics']['spearman_p_value'] < 0.05 else '非有意'} (p={test_results['test_metrics']['spearman_p_value']:.6f})")
        
        # 特徴量重要度の表示
        if cv_results['avg_importance'] is not None and len(cv_results['avg_importance']) > 0:
            print(f"\n特徴量の平均重要度（上位10件）:")
            print(cv_results['avg_importance'].head(10).to_string(index=False))
        else:
            print(f"\n特徴量重要度が取得できませんでした")
            print("※ モデルが特徴量重要度をサポートしていない可能性があります")
        
        if self.output_dir:
            print(f"\n結果保存先: {self.output_dir}")
        print(f"{'='*70}")


def run_holdout_cv_analysis(df, model_class=None, use_bagging=False,
                           n_splits=5, test_size=0.2, 
                           output_dir='result', random_state=42, 
                           target_column='target', data_file_name=None,
                           save_splits=False,
                           # SMOTE関連パラメータ
                           use_smote=False, smote_method='density', 
                           smote_kwargs=None, use_integer_smote=True, 
                           target_min=9, target_max=30,
                           # ALE関連パラメータ
                           generate_ale=True, ale_n_features=6, ale_grid_resolution=30,
                           # モデル保存パラメータ
                           save_final_model=True,
                           **kwargs):
    """
    テストデータを保持してクロスバリデーションを実行する関数（最終修正版）
    """
    
    # 必要な関数をインポート（main.pyから）
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    import numpy as np
    import pandas as pd
    import os
    import datetime
    
    print(f"\n{'='*60}")
    print(f"テストデータ保持+クロスバリデーション分析開始")
    print(f"{'='*60}")
    
    # 分割比率の表示と確認
    dev_size = 1 - test_size
    print(f"データ分割比率:")
    print(f"  開発用（学習+検証）: {dev_size*100:.1f}%")
    print(f"  テスト用: {test_size*100:.1f}%")
    print(f"  CV分割数: {n_splits}")
    
    # 出力ディレクトリの設定
    if output_dir and data_file_name:
        base_name = os.path.splitext(os.path.basename(data_file_name))[0]
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        smote_suffix = f"_smote_{smote_method}" if use_smote else ""
        output_dir = f"{base_name}_holdout_cv{smote_suffix}_{timestamp}"
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"出力ディレクトリを作成: {output_dir}")
    
    # サブディレクトリの事前作成
    if output_dir:
        subdirs = ['cv_results', 'ale_plots', 'saved_model', 'data_splits']
        for subdir in subdirs:
            subdir_path = os.path.join(output_dir, subdir)
            if not os.path.exists(subdir_path):
                os.makedirs(subdir_path)
                print(f"サブディレクトリを作成: {subdir_path}")
    
    # データの前処理（全データで一度だけ実行）
    print(f"\n=== データ前処理 ===")
    
    # main.pyから必要な関数をインポート
    import sys
    import importlib.util
    
    # main.pyのパスを取得
    main_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'main5k_ale2cvt.py')
    if not os.path.exists(main_path):
        main_path = 'main5k_ale2cvt.py'
    
    # main.pyから必要な関数をインポート
    spec = importlib.util.spec_from_file_location("main", main_path)
    main_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(main_module)
    
    features, target, id_column = main_module.prepare_data_for_regression(df, target_column=target_column)
    
    print(f"前処理後のデータ:")
    print(f"  特徴量数: {features.shape[1]}")
    print(f"  サンプル数: {features.shape[0]}")
    print(f"  目的変数範囲: {target.min():.2f} - {target.max():.2f}")
    
    # データが空でないことを確認
    if len(features) == 0 or len(target) == 0:
        raise ValueError(f"前処理後のデータが空です。元のデータ形状: {df.shape}, 前処理後: features={features.shape}, target={target.shape}")
    
    # フェーズ1: データ分割（テストデータを先に分離）
    print(f"\n=== フェーズ1: データ分割 ===")
    
    if id_column is not None:
        X_dev, X_test, y_dev, y_test, id_dev, id_test = train_test_split(
            features, target, id_column,
            test_size=test_size, 
            random_state=random_state
        )
    else:
        X_dev, X_test, y_dev, y_test = train_test_split(
            features, target,
            test_size=test_size, 
            random_state=random_state
        )
        id_dev, id_test = None, None
    
    print(f"実際のデータ分割結果:")
    print(f"  開発用: {len(X_dev)}サンプル ({len(X_dev)/len(features)*100:.1f}%)")
    print(f"  テスト用: {len(X_test)}サンプル ({len(X_test)/len(features)*100:.1f}%)")
    print(f"  開発データの目的変数範囲: {y_dev.min():.2f} - {y_dev.max():.2f}")
    print(f"  テストデータの目的変数範囲: {y_test.min():.2f} - {y_test.max():.2f}")
    
    # 開発データが空でないことを確認
    if len(X_dev) == 0 or len(y_dev) == 0:
        raise ValueError(f"開発データが空です。開発データ: X_dev={X_dev.shape}, y_dev={y_dev.shape}")
    
    # 分割データの保存（オプション）
    if save_splits and output_dir:
        print(f"\n分割データを保存中...")
        splits_dir = os.path.join(output_dir, 'data_splits')
        
        try:
            # 開発データを保存
            dev_df = X_dev.copy()
            dev_df[target_column] = y_dev
            if id_dev is not None:
                dev_df.insert(0, id_dev.name, id_dev)
            
            dev_path = os.path.join(splits_dir, 'development_split.csv')
            dev_df.to_csv(dev_path, index=False)
            print(f"  開発データ: {dev_path}")
            
            # テストデータを保存
            test_df = X_test.copy()
            test_df[target_column] = y_test
            if id_test is not None:
                test_df.insert(0, id_test.name, id_test)
            
            test_path = os.path.join(splits_dir, 'test_split.csv')
            test_df.to_csv(test_path, index=False)
            print(f"  テストデータ: {test_path}")
            
        except Exception as e:
            print(f"分割データ保存エラー: {e}")
    
    # SMOTE設定
    if smote_kwargs is None:
        smote_kwargs = {
            'binning': {'sampling_strategy': 'auto', 'n_bins': 10},
            'density': {'density_threshold': 0.3},
            'outliers': {'outlier_threshold': 0.15}
        }.get(smote_method, {})
    
    # フェーズ2: 開発データでクロスバリデーション（直接実装版）
    print(f"\n=== フェーズ2: 開発データでクロスバリデーション ===")
    print("開発データのみを使用してクロスバリデーションを実行")
    print("（テストデータは一切使用しない - データリーク防止）")
    
    from sklearn.model_selection import KFold
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # 開発データの準備
    X_dev_values = X_dev.values if hasattr(X_dev, 'values') else X_dev
    y_dev_values = y_dev.values if hasattr(y_dev, 'values') else y_dev
    
    print(f"CV用データ準備:")
    print(f"  X_dev_values形状: {X_dev_values.shape}")
    print(f"  y_dev_values形状: {y_dev_values.shape}")
    
    # 交差検証の設定
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    
    # 結果の格納用
    fold_metrics = []
    oof_predictions = np.zeros(len(y_dev_values))
    feature_importances = []
    smote_statistics = []
    
    print(f"\n開発データでの{n_splits}分割交差検証を開始...")
    
    # 各分割でモデルを学習・評価
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_dev_values)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        # データの分割
        X_train_fold, X_val_fold = X_dev_values[train_idx], X_dev_values[val_idx]
        y_train_fold, y_val_fold = y_dev_values[train_idx], y_dev_values[val_idx]
        
        print(f"Fold {fold+1} データ:")
        print(f"  訓練データ: {X_train_fold.shape}")
        print(f"  検証データ: {X_val_fold.shape}")
        
        # 元の訓練データサイズを記録
        original_train_size = len(X_train_fold)
        
        # SMOTE適用（訓練データのみ）
        smote_applied = False
        if use_smote:
            print(f"Fold {fold+1}: 訓練データにSMOTE（{smote_method}）を適用中...")
            try:
                # SMOTE instance作成
                if use_integer_smote:
                    try:
                        from regression_smote import IntegerRegressionSMOTE
                        smote_instance = IntegerRegressionSMOTE(
                            method=smote_method, 
                            k_neighbors=5, 
                            random_state=random_state + fold,
                            target_min=target_min,
                            target_max=target_max
                        )
                        print(f"  整数値対応SMOTEを使用")
                    except ImportError:
                        from regression_smotebackup import RegressionSMOTE
                        smote_instance = RegressionSMOTE(
                            method=smote_method, 
                            k_neighbors=5, 
                            random_state=random_state + fold
                        )
                        print(f"  通常のSMOTEを使用")
                else:
                    from regression_smotebackup import RegressionSMOTE
                    smote_instance = RegressionSMOTE(
                        method=smote_method, 
                        k_neighbors=5, 
                        random_state=random_state + fold
                    )
                
                print(f"  SMOTE適用前: {len(X_train_fold)}サンプル")
                X_train_resampled, y_train_resampled = smote_instance.fit_resample(
                    X_train_fold, y_train_fold, **smote_kwargs
                )
                
                # 整数化処理
                if use_integer_smote and 'IntegerRegressionSMOTE' not in str(type(smote_instance)):
                    n_original = len(y_train_fold)
                    for i in range(n_original, len(y_train_resampled)):
                        y_train_resampled[i] = round(y_train_resampled[i])
                        y_train_resampled[i] = np.clip(y_train_resampled[i], target_min, target_max)
                
                X_train_fold = X_train_resampled
                y_train_fold = y_train_resampled
                smote_applied = True
                
                synthetic_count = len(X_train_fold) - original_train_size
                print(f"  SMOTE適用後: {len(X_train_fold)}サンプル (+{synthetic_count})")
                
                # SMOTE統計を記録
                smote_stats = {
                    'fold': fold + 1,
                    'method': smote_method,
                    'original_size': original_train_size,
                    'resampled_size': len(X_train_fold),
                    'synthetic_count': synthetic_count,
                    'increase_ratio': (synthetic_count / original_train_size) * 100
                }
                smote_statistics.append(smote_stats)
                
            except Exception as e:
                print(f"  SMOTE適用中にエラー: {e}")
                print("  SMOTEなしで処理を続行...")
                smote_applied = False
        
        # スケーリング
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train_fold)
        X_val_scaled = scaler.transform(X_val_fold)
        
        # モデル学習
        print(f"  モデル学習中...")
        if use_bagging:
            model = main_module.RegressionBaggingModel(
                base_model=kwargs.get('base_model', 'lightgbm'),
                n_bags=kwargs.get('n_bags', 10),
                random_state=random_state + fold
            )
        else:
            model = model_class()
        
        model.fit(X_train_scaled, y_train_fold)
        
        # 特徴量重要度
        if hasattr(model, 'feature_importances_'):
            importances = pd.DataFrame({
                'feature': features.columns,
                'importance': model.feature_importances_,
                'fold': fold + 1
            })
            feature_importances.append(importances)
        
        # 検証データで予測
        y_pred = model.predict(X_val_scaled)
        oof_predictions[val_idx] = y_pred
        
        # 評価指標の計算
        mse = mean_squared_error(y_val_fold, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_val_fold, y_pred)
        r2 = r2_score(y_val_fold, y_pred)
        mape = main_module.mean_absolute_percentage_error(y_val_fold, y_pred)
        pearson_corr, p_value = pearsonr(y_val_fold, y_pred)
        spearman_corr = main_module.spearman_correlation(y_val_fold, y_pred)
        
        # 結果の保存
        fold_result = {
            'fold': fold + 1,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mape': mape,
            'r2': r2,
            'pearson_corr': pearson_corr,
            'p_value': p_value,
            'spearman_corr': spearman_corr,
            'smote_applied': smote_applied,
            'train_size_original': original_train_size,
            'train_size_final': len(X_train_fold),
            'val_size': len(X_val_fold)
        }
        fold_metrics.append(fold_result)
        
        print(f"  Fold {fold+1} 結果: RMSE={rmse:.4f}, R²={r2:.4f}")
    
    # CV結果のまとめ
    metrics_df = pd.DataFrame(fold_metrics)
    
    # 平均指標の計算
    cv_metrics = {
        'mse': metrics_df['mse'].mean(),
        'rmse': metrics_df['rmse'].mean(),
        'mae': metrics_df['mae'].mean(),
        'mape': metrics_df['mape'].mean(),
        'r2': metrics_df['r2'].mean(),
        'pearson_corr': metrics_df['pearson_corr'].mean(),
        'spearman_corr': metrics_df['spearman_corr'].mean()
    }
    
    # 標準偏差の計算
    cv_std = {
        'mse_std': metrics_df['mse'].std(),
        'rmse_std': metrics_df['rmse'].std(),
        'mae_std': metrics_df['mae'].std(),
        'mape_std': metrics_df['mape'].std(),
        'r2_std': metrics_df['r2'].std(),
        'pearson_corr_std': metrics_df['pearson_corr'].std(),
        'spearman_corr_std': metrics_df['spearman_corr'].std()
    }
    
    print(f"\n開発データでのCV結果:")
    for metric, value in cv_metrics.items():
        print(f"  {metric.upper()}: {value:.4f} ± {cv_std[f'{metric}_std']:.4f}")
    
    # 平均特徴量重要度
    avg_importance = None
    if feature_importances:
        importances_df = pd.concat(feature_importances)
        avg_importance = importances_df.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
    
    # ===== CV結果の保存 =====
    if output_dir:
        print(f"\n=== CV結果の保存 ===")
        cv_results_dir = os.path.join(output_dir, 'cv_results')
        
        try:
            # Out-of-fold予測結果の保存
            oof_df = pd.DataFrame({
                'true': y_dev_values,
                'pred': oof_predictions,
                'residual': y_dev_values - oof_predictions,
                'abs_residual': np.abs(y_dev_values - oof_predictions)
            })
            
            # ID列がある場合は追加
            if id_dev is not None:
                oof_df.insert(0, id_dev.name, id_dev.values)
            
            oof_path = os.path.join(cv_results_dir, 'oof_predictions.csv')
            oof_df.to_csv(oof_path, index=False)
            print(f"✓ OOF予測結果を保存: {oof_path}")
            
            # 各フォールドの詳細結果
            fold_details_path = os.path.join(cv_results_dir, 'fold_details.csv')
            metrics_df.to_csv(fold_details_path, index=False)
            print(f"✓ フォールド詳細を保存: {fold_details_path}")
            
            # CV指標の保存
            cv_metrics_path = os.path.join(cv_results_dir, 'cv_metrics.txt')
            with open(cv_metrics_path, 'w', encoding='utf-8') as f:
                f.write("=== クロスバリデーション結果 ===\n")
                f.write(f"分割数: {n_splits}\n")
                f.write(f"開発データ数: {len(X_dev)}\n")
                f.write(f"\n平均性能指標:\n")
                for metric, value in cv_metrics.items():
                    std = cv_std[f'{metric}_std']
                    f.write(f"{metric.upper()}: {value:.4f} ± {std:.4f}\n")
            print(f"✓ CV指標を保存: {cv_metrics_path}")
            
            # SMOTE統計の保存
            if use_smote and smote_statistics:
                smote_stats_path = os.path.join(cv_results_dir, 'smote_statistics.csv')
                pd.DataFrame(smote_statistics).to_csv(smote_stats_path, index=False)
                print(f"✓ SMOTE統計を保存: {smote_stats_path}")
            
            # 特徴量重要度の保存
            if avg_importance is not None:
                importance_path = os.path.join(cv_results_dir, 'feature_importance.csv')
                avg_importance.to_csv(importance_path, index=False)
                print(f"✓ 特徴量重要度を保存: {importance_path}")
                
        except Exception as e:
            print(f"✗ CV結果保存エラー: {e}")
    
    # フェーズ3: 最終モデル学習（開発データ全体）
    print(f"\n=== フェーズ3: 最終モデル学習（開発データ全体）===")
    print("開発データ全体でSMOTE適用→最終モデル学習")
    
    # 開発データ全体の準備
    X_dev_processed = X_dev.copy()
    y_dev_processed = y_dev.copy()
    
    # 開発データ全体にSMOTE適用
    if use_smote:
        print(f"開発データ全体にSMOTE（{smote_method}）を適用中...")
        try:
            # SMOTE instance作成
            if use_integer_smote:
                try:
                    from regression_smote import IntegerRegressionSMOTE
                    smote_instance = IntegerRegressionSMOTE(
                        method=smote_method, 
                        k_neighbors=5, 
                        random_state=random_state + 1000,
                        target_min=target_min,
                        target_max=target_max
                    )
                    print(f"  整数値対応SMOTEを使用")
                except ImportError:
                    from regression_smotebackup import RegressionSMOTE
                    smote_instance = RegressionSMOTE(
                        method=smote_method, 
                        k_neighbors=5, 
                        random_state=random_state + 1000
                    )
                    print(f"  通常のSMOTEを使用")
            else:
                from regression_smotebackup import RegressionSMOTE
                smote_instance = RegressionSMOTE(
                    method=smote_method, 
                    k_neighbors=5, 
                    random_state=random_state + 1000
                )
            
            X_dev_values = X_dev_processed.values
            y_dev_values = y_dev_processed.values
            
            print(f"  SMOTE適用前: {len(X_dev_values)}サンプル")
            X_dev_resampled, y_dev_resampled = smote_instance.fit_resample(
                X_dev_values, y_dev_values, **smote_kwargs
            )
            
            # 整数化処理
            if use_integer_smote and 'IntegerRegressionSMOTE' not in str(type(smote_instance)):
                print("  合成データの整数化を実行中...")
                n_original = len(y_dev_values)
                for i in range(n_original, len(y_dev_resampled)):
                    y_dev_resampled[i] = round(y_dev_resampled[i])
                    y_dev_resampled[i] = np.clip(y_dev_resampled[i], target_min, target_max)
            
            X_dev_processed = pd.DataFrame(X_dev_resampled, columns=features.columns)
            y_dev_processed = pd.Series(y_dev_resampled, name=target.name)
            
            synthetic_count = len(X_dev_resampled) - len(X_dev_values)
            increase_ratio = (synthetic_count / len(X_dev_values)) * 100
            print(f"  SMOTE適用後: {len(X_dev_resampled)}サンプル (+{synthetic_count})")
            print(f"  データ増加率: {increase_ratio:.2f}%")
            
        except Exception as e:
            print(f"  SMOTE適用エラー: {e}")
            print("  SMOTEなしで最終モデルを学習します")
            import traceback
            traceback.print_exc()
    
    # 最終スケーリング
    final_scaler = StandardScaler()
    X_dev_scaled = pd.DataFrame(
        final_scaler.fit_transform(X_dev_processed),
        columns=features.columns
    )
    
    # 最終モデル学習
    print("最終モデル（開発データ全体）を学習中...")
    if use_bagging:
        final_model = main_module.RegressionBaggingModel(
            base_model=kwargs.get('base_model', 'lightgbm'),
            n_bags=kwargs.get('n_bags', 10),
            random_state=random_state + 2000
        )
    else:
        final_model = model_class()
    
    final_model.fit(X_dev_scaled, y_dev_processed)
    print("最終モデル学習完了")
    
    # 特徴量重要度の取得
    final_feature_importance = None
    if hasattr(final_model, 'feature_importances_'):
        final_feature_importance = pd.DataFrame({
            'feature': features.columns,
            'importance': final_model.feature_importances_
        }).sort_values('importance', ascending=False)
        print(f"特徴量重要度を取得しました（上位5特徴量）:")
        print(final_feature_importance.head().to_string(index=False))
    elif avg_importance is not None:
        final_feature_importance = avg_importance
        print(f"CV平均特徴量重要度を使用します")
    
    # フェーズ4: テストデータで最終評価
    print(f"\n=== フェーズ4: テストデータで最終評価 ===")
    print("最終モデルでテストデータを予測（真の未知データでの性能）")
    
    X_test_scaled = pd.DataFrame(
        final_scaler.transform(X_test),
        columns=features.columns
    )
    test_pred = final_model.predict(X_test_scaled)
    
    # テスト評価指標の計算
    test_pearson_corr, test_p_value = pearsonr(y_test, test_pred)
    test_spearman_corr = main_module.spearman_correlation(y_test, test_pred)
    
    test_metrics = {
        'rmse': np.sqrt(mean_squared_error(y_test, test_pred)),
        'mae': mean_absolute_error(y_test, test_pred),
        'r2': r2_score(y_test, test_pred),
        'mape': main_module.mean_absolute_percentage_error(y_test, test_pred),
        'pearson_corr': test_pearson_corr,
        'pearson_p_value': test_p_value,
        'spearman_corr': test_spearman_corr
    }
    
    print(f"\nテストデータでの最終性能:")
    for metric, value in test_metrics.items():
        if metric == 'pearson_p_value':
            print(f"  {metric.upper()}: {value:.6f}")
        else:
            print(f"  {metric.upper()}: {value:.4f}")
    
    # 性能差の計算と表示
    print(f"\n性能差分析（テスト vs CV）:")
    print(f"  RMSE差: {test_metrics['rmse'] - cv_metrics['rmse']:+.4f}")
    print(f"  R²差: {test_metrics['r2'] - cv_metrics['r2']:+.4f}")
    print(f"  MAE差: {test_metrics['mae'] - cv_metrics['mae']:+.4f}")
    print(f"  Pearson相関差: {test_metrics['pearson_corr'] - cv_metrics['pearson_corr']:+.4f}")
    print(f"  Spearman相関差: {test_metrics['spearman_corr'] - cv_metrics['spearman_corr']:+.4f}")
    
    # ===== 最終モデルの保存（強制実行） =====
    print(f"\n=== 最終モデル保存 ===")
    model_saved = False
    
    if output_dir and data_file_name:
        # モデル名の決定
        if use_bagging:
            model_name = f"holdout_cv_{kwargs.get('base_model', 'lightgbm')}_bagging"
        else:
            # モデルクラス名を取得
            if model_class:
                try:
                    test_model = model_class()
                    model_name = f"holdout_cv_{test_model.__class__.__name__.lower()}"
                except:
                    model_name = "holdout_cv_regression_model"
            else:
                model_name = "holdout_cv_regression_model"
        
        # SMOTE使用時はモデル名に追加
        if use_smote:
            model_name += f"_smote_{smote_method}"
        
        # データファイルのベース名を追加
        base_name = os.path.splitext(os.path.basename(data_file_name))[0]
        model_name = f"{base_name}_{model_name}"
        
        print(f"モデル保存を開始: {model_name}")
        print(f"保存先: {output_dir}/saved_model/")
        print(f"学習データ数: {len(X_dev_processed)}")
        print(f"特徴量数: {X_dev_processed.shape[1]}")
        
        try:
            main_module.save_regression_model(
                model=final_model,
                features=X_dev_processed,
                target=y_dev_processed,
                scaler=final_scaler,
                output_path=output_dir,
                model_name=model_name
            )
            model_saved = True
            print(f"✓ 最終モデル保存完了: {model_name}")
            
            # 保存確認
            saved_model_dir = os.path.join(output_dir, 'saved_model')
            if os.path.exists(saved_model_dir):
                saved_files = os.listdir(saved_model_dir)
                print(f"✓ 保存されたファイル数: {len(saved_files)}")
                for f in saved_files:
                    if model_name in f:
                        print(f"  - {f}")
            else:
                print(f"✗ saved_modelディレクトリが見つかりません")
                
        except Exception as e:
            print(f"✗ 最終モデル保存中にエラー: {e}")
            import traceback
            traceback.print_exc()
            model_saved = False
    else:
        print("出力ディレクトリまたはデータファイル名が指定されていません")
    
    # ===== 結果の可視化とファイル保存（修正版） =====
    if output_dir:
        print(f"\n=== 結果の可視化・保存 ===")
        
        try:
            visualizer = main_module.EyeTrackingVisualizer(output_dir=output_dir)
            
            # テストデータの結果可視化
            print("可視化を実行中...")
            visualizer.plot_true_vs_predicted(y_test, test_pred)
            visualizer.plot_residuals(y_test, test_pred)
            
            print("✓ 散布図・残差プロットを生成")
            
            # 特徴量重要度の可視化
            if final_feature_importance is not None:
                visualizer.plot_feature_importance(final_feature_importance)
                print("✓ 特徴量重要度プロットを生成")
                
        except Exception as e:
            print(f"可視化エラー: {e}")
            print("可視化をスキップして処理を続行します...")
        # ===== test_results.csvの確実な保存 =====
        print(f"\ntest_results.csvを保存中...")
        try:
            # テスト結果をCSVに保存
            test_results_df = pd.DataFrame({
                'true': y_test.values if hasattr(y_test, 'values') else y_test,
                'predicted': test_pred,
                'residual': (y_test.values if hasattr(y_test, 'values') else y_test) - test_pred,
                'abs_residual': np.abs((y_test.values if hasattr(y_test, 'values') else y_test) - test_pred)
            })
            
            # ID列がある場合は追加
            if id_test is not None:
                test_results_df.insert(0, id_test.name, id_test.values if hasattr(id_test, 'values') else id_test)
            
            results_path = os.path.join(output_dir, 'test_results.csv')
            test_results_df.to_csv(results_path, index=False)
            
            # ファイル保存の確認
            if os.path.exists(results_path):
                file_size = os.path.getsize(results_path)
                print(f"✓ test_results.csvを保存: {results_path} ({file_size} bytes)")
            else:
                print(f"✗ test_results.csvの保存に失敗: {results_path}")
                
        except Exception as e:
            print(f"✗ test_results.csv保存エラー: {e}")
            import traceback
            traceback.print_exc()
        
        # 設定情報の保存
        print(f"\n設定情報を保存中...")
        try:
            config_path = os.path.join(output_dir, 'config_holdout_cv.txt')
            with open(config_path, 'w', encoding='utf-8') as f:
                f.write("=== テストデータ保持+クロスバリデーション設定 ===\n")
                f.write(f"データファイル: {data_file_name}\n")
                f.write(f"目的変数: {target_column}\n")
                f.write(f"開発用: {len(X_dev)}サンプル ({len(X_dev)/len(features)*100:.1f}%)\n")
                f.write(f"テスト用: {len(X_test)}サンプル ({len(X_test)/len(features)*100:.1f}%)\n")
                f.write(f"CV分割数: {n_splits}\n")
                f.write(f"乱数シード: {random_state}\n")
                
                if use_bagging:
                    f.write(f"\n=== バギング設定 ===\n")
                    f.write(f"ベースモデル: {kwargs.get('base_model', 'lightgbm')}\n")
                    f.write(f"バッグ数: {kwargs.get('n_bags', 10)}\n")
                else:
                    f.write(f"\n=== モデル設定 ===\n")
                    f.write(f"モデルタイプ: {model_class.__name__ if model_class else 'Unknown'}\n")
                
                if use_smote:
                    f.write(f"\n=== SMOTE設定 ===\n")
                    f.write(f"手法: {smote_method}\n")
                    f.write(f"整数値対応: {'有効' if use_integer_smote else '無効'}\n")
                    if use_integer_smote:
                        f.write(f"目的変数範囲: {target_min}-{target_max}\n")
                    for key, value in smote_kwargs.items():
                        f.write(f"{key}: {value}\n")
                
                f.write(f"\n=== CV性能（開発データ） ===\n")
                for metric, value in cv_metrics.items():
                    std = cv_std.get(f'{metric}_std', 0)
                    f.write(f"{metric.upper()}: {value:.4f} ± {std:.4f}\n")
                
                f.write(f"\n=== テスト性能（最終評価） ===\n")
                for metric, value in test_metrics.items():
                    if metric == 'pearson_p_value':
                        f.write(f"{metric.upper()}: {value:.6f}\n")
                    else:
                        f.write(f"{metric.upper()}: {value:.4f}\n")
                
                f.write(f"\n=== 性能差（テスト - CV） ===\n")
                f.write(f"RMSE差: {test_metrics['rmse'] - cv_metrics['rmse']:+.4f}\n")
                f.write(f"R²差: {test_metrics['r2'] - cv_metrics['r2']:+.4f}\n")
                f.write(f"MAE差: {test_metrics['mae'] - cv_metrics['mae']:+.4f}\n")
                f.write(f"Pearson相関差: {test_metrics['pearson_corr'] - cv_metrics['pearson_corr']:+.4f}\n")
                f.write(f"Spearman相関差: {test_metrics['spearman_corr'] - cv_metrics['spearman_corr']:+.4f}\n")
                
                f.write(f"\n=== モデル保存情報 ===\n")
                f.write(f"最終モデル保存: {'成功' if model_saved else '失敗'}\n")
                if model_saved:
                    f.write(f"学習データ数: {len(X_dev_processed)}\n")
                    if use_smote:
                        f.write(f"元開発データ数: {len(X_dev)}\n")
                        f.write(f"合成データ数: {len(X_dev_processed) - len(X_dev)}\n")
                        f.write(f"データ増加率: {((len(X_dev_processed) - len(X_dev)) / len(X_dev) * 100):.2f}%\n")
                    
                f.write(f"\n実行日時: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            print(f"✓ 設定情報を保存: {config_path}")
            
        except Exception as e:
            print(f"✗ 設定情報保存エラー: {e}")
        
        # 特徴量重要度の保存
        if final_feature_importance is not None:
            try:
                importance_path = os.path.join(output_dir, 'feature_importance.csv')
                final_feature_importance.to_csv(importance_path, index=False)
                print(f"✓ 特徴量重要度を保存: {importance_path}")
            except Exception as e:
                print(f"✗ 特徴量重要度保存エラー: {e}")
    
    # ===== ALEプロット生成（修正版） =====
    if generate_ale and final_feature_importance is not None:
        print(f"\n=== ALE (Accumulated Local Effects) プロット生成 ===")
        try:
            ale_output_dir = os.path.join(output_dir, 'ale_plots') if output_dir else 'ale_plots'
            
            print(f"ALE生成開始:")
            print(f"  出力ディレクトリ: {ale_output_dir}")
            print(f"  特徴量数: {ale_n_features}")
            print(f"  グリッド解像度: {ale_grid_resolution}")
            
            # ALE分析実行
            ale_results = main_module.analyze_accumulated_local_effects_fixed(
                model=final_model,
                X=X_dev_scaled,
                feature_importances=final_feature_importance,
                output_dir=ale_output_dir,
                n_top_features=ale_n_features,
                n_grid=ale_grid_resolution
            )
            
            # ALE生成結果の確認
            if os.path.exists(ale_output_dir):
                ale_files = [f for f in os.listdir(ale_output_dir) if f.endswith('.png')]
                if ale_files:
                    print(f"✓ ALEプロット生成完了: {len(ale_files)}個のファイル")
                    for ale_file in ale_files:
                        print(f"  - {ale_file}")
                else:
                    print("✗ ALEプロットファイルが生成されませんでした")
            else:
                print(f"✗ ALEディレクトリが存在しません: {ale_output_dir}")
                
        except Exception as e:
            print(f"✗ ALEプロット生成中にエラー: {e}")
            import traceback
            traceback.print_exc()
            
            # エラーでもディレクトリは作成しておく
            if output_dir:
                ale_output_dir = os.path.join(output_dir, 'ale_plots')
                if not os.path.exists(ale_output_dir):
                    os.makedirs(ale_output_dir)
                    print(f"ALEディレクトリを作成: {ale_output_dir}")
    
    # 結果のまとめ
    results = {
        'cv_metrics': cv_metrics,
        'cv_std': cv_std,
        'test_metrics': test_metrics,
        'final_model': final_model,
        'final_scaler': final_scaler,
        'test_predictions': test_pred,
        'test_true': y_test,
        'feature_importance': final_feature_importance,
        'output_dir': output_dir,
        'development_data': {
            'features': X_dev_processed,
            'target': y_dev_processed,
            'original_features': X_dev,
            'original_target': y_dev
        },
        'test_data': {
            'features': X_test,
            'target': y_test,
            'predictions': test_pred
        },
        'data_splits': {
            'dev_size': len(X_dev),
            'test_size': len(X_test),
            'total_size': len(features)
        },
        'model_saved': model_saved
    }
    
    # 最終確認
    print(f"\n{'='*60}")
    print(f"テストデータ保持+クロスバリデーション完了")
    print(f"CVスコア（開発データ）: RMSE={cv_metrics['rmse']:.4f}, R²={cv_metrics['r2']:.4f}")
    print(f"テストスコア（最終評価）: RMSE={test_metrics['rmse']:.4f}, R²={test_metrics['r2']:.4f}")
    print(f"性能差: RMSE={test_metrics['rmse'] - cv_metrics['rmse']:+.4f}, R²={test_metrics['r2'] - cv_metrics['r2']:+.4f}")
    print(f"モデル保存: {'✓ 成功' if model_saved else '✗ 失敗'}")
    
    # 生成されたファイルの最終確認
    if output_dir and os.path.exists(output_dir):
        all_files = []
        for root, dirs, files in os.walk(output_dir):
            for file in files:
                rel_path = os.path.relpath(os.path.join(root, file), output_dir)
                all_files.append(rel_path)
        
        print(f"\n生成されたファイル一覧 ({len(all_files)}個):")
        for file in sorted(all_files):
            print(f"  - {file}")
        
        # 特に重要なファイルの存在確認
        important_files = ['test_results.csv', 'saved_model/', 'ale_plots/']
        print(f"\n重要なファイル/フォルダの存在確認:")
        for important in important_files:
            full_path = os.path.join(output_dir, important)
            exists = os.path.exists(full_path)
            print(f"  {important}: {'✓ 存在' if exists else '✗ 不在'}")
    
    if output_dir:
        print(f"結果保存先: {output_dir}")
    print(f"{'='*60}")
    
    return results


class HoldoutCVAnalyzer:
    """テストデータ保持+クロスバリデーション分析のクラス"""
    
    def __init__(self, output_dir='result', random_state=42):
        self.output_dir = output_dir
        self.random_state = random_state
        
    def run_analysis(self, *args, **kwargs):
        """分析を実行する"""
        return run_holdout_cv_analysis(*args, **kwargs)