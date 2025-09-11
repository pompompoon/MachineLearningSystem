"""
aleplotter.py
分類
ALE（Accumulated Local Effects）プロット作成クラス
独立したモジュールとして設計されており、他のプロジェクトでも再利用可能
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from sklearn.utils import check_array
import warnings
from datetime import datetime
import os


class ALEPlotter:
    """
    ALE（Accumulated Local Effects）プロット作成クラス
    
    ALEプロットは部分依存プロット（PDP）の改良版で、特徴量間の相関を
    適切に処理し、より正確な特徴量効果を可視化します。
    
    Attributes:
    -----------
    model : sklearn-like model
        予測を行うモデル（predict メソッドを持つ）
    X : pandas.DataFrame
        特徴量データ
    feature_names : list
        特徴量名のリスト
    output_dir : str
        出力ディレクトリ
    ale_cache : dict
        ALEプロット結果のキャッシュ
    """
    
    def __init__(self, model, X, feature_names=None, output_dir=None):
        """
        ALEPlotterの初期化
        
        Parameters:
        -----------
        model : sklearn-like model
            予測を行うモデル（predict メソッドを持つ）
        X : array-like or pandas.DataFrame
            特徴量データ
        feature_names : list, optional
            特徴量名のリスト
        output_dir : str, optional
            出力ディレクトリ
        """
        self.model = model
        self.X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
        
        if feature_names is None:
            if hasattr(X, 'columns'):
                self.feature_names = list(X.columns)
            else:
                self.feature_names = [f'feature_{i}' for i in range(X.shape[1])]
        else:
            self.feature_names = feature_names
        
        self.X.columns = self.feature_names
        self.output_dir = output_dir
        
        # ALEプロット結果のキャッシュ
        self.ale_cache = {}
        
        # 出力ディレクトリの作成
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def set_output_dir(self, output_dir):
        """
        出力ディレクトリを設定
        
        Parameters:
        -----------
        output_dir : str
            新しい出力ディレクトリ
        """
        self.output_dir = output_dir
        if self.output_dir and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
    
    def calculate_ale_1d(self, feature_name, bins=50, quantile_bins=True):
        """
        1次元ALEを計算
        
        Parameters:
        -----------
        feature_name : str
            対象特徴量名
        bins : int, default=50
            ビン数
        quantile_bins : bool, default=True
            分位数ベースのビニングを使用するか
            
        Returns:
        --------
        dict
            ALE計算結果
        """
        if feature_name not in self.feature_names:
            raise ValueError(f"特徴量 '{feature_name}' が見つかりません")
        
        # キャッシュチェック
        cache_key = f"{feature_name}_{bins}_{quantile_bins}"
        if cache_key in self.ale_cache:
            return self.ale_cache[cache_key]
        
        feature_idx = self.feature_names.index(feature_name)
        x_values = self.X.iloc[:, feature_idx].values
        
        # ビンの作成
        if quantile_bins:
            # 分位数ベースのビニング
            bin_edges = np.unique(np.quantile(x_values, np.linspace(0, 1, bins + 1)))
        else:
            # 等間隔ビニング
            bin_edges = np.linspace(x_values.min(), x_values.max(), bins + 1)
        
        if len(bin_edges) < 3:
            # ユニークな値が少ない場合
            bin_edges = np.unique(x_values)
            if len(bin_edges) < 2:
                warnings.warn(f"特徴量 '{feature_name}' のユニーク値が少なすぎます")
                return None
        
        # ALE計算
        ale_values = []
        bin_centers = []
        
        for i in range(len(bin_edges) - 1):
            # 現在のビンに含まれるデータポイントを特定
            mask = (x_values >= bin_edges[i]) & (x_values <= bin_edges[i + 1])
            
            if i == len(bin_edges) - 2:  # 最後のビンは右端も含む
                mask = (x_values >= bin_edges[i]) & (x_values <= bin_edges[i + 1])
            
            if not np.any(mask):
                continue
            
            # ビン内のデータを取得
            X_bin = self.X[mask].copy()
            
            if len(X_bin) == 0:
                continue
            
            # 左端での予測
            X_left = X_bin.copy()
            X_left.iloc[:, feature_idx] = bin_edges[i]
            pred_left = self.model.predict(X_left.values)
            
            # 右端での予測
            X_right = X_bin.copy()
            X_right.iloc[:, feature_idx] = bin_edges[i + 1]
            pred_right = self.model.predict(X_right.values)
            
            # 局所効果の計算
            local_effect = np.mean(pred_right - pred_left)
            ale_values.append(local_effect)
            bin_centers.append((bin_edges[i] + bin_edges[i + 1]) / 2)
        
        # 累積効果の計算
        ale_values = np.array(ale_values)
        cumulative_ale = np.cumsum(ale_values)
        
        # 中央化（平均を0に）
        mean_effect = np.mean(cumulative_ale)
        cumulative_ale = cumulative_ale - mean_effect
        
        result = {
            'feature_name': feature_name,
            'bin_centers': np.array(bin_centers),
            'ale_values': cumulative_ale,
            'local_effects': ale_values,
            'bin_edges': bin_edges,
            'n_bins': len(bin_centers),
            'ale_range': cumulative_ale.max() - cumulative_ale.min() if len(cumulative_ale) > 0 else 0
        }
        
        # キャッシュに保存
        self.ale_cache[cache_key] = result
        
        return result
    
    def plot_ale_1d(self, feature_name, bins=50, quantile_bins=True, 
                   figsize=(10, 6), title=None, save=True, show_data_distribution=True):
        """
        1次元ALEプロットを作成
        
        Parameters:
        -----------
        feature_name : str
            対象特徴量名
        bins : int, default=50
            ビン数
        quantile_bins : bool, default=True
            分位数ベースのビニングを使用するか
        figsize : tuple, default=(10, 6)
            図のサイズ
        title : str, optional
            図のタイトル
        save : bool, default=True
            図を保存するか
        show_data_distribution : bool, default=True
            データ分布を背景に表示するか
            
        Returns:
        --------
        matplotlib.figure.Figure
            作成された図
        """
        # ALE計算
        ale_result = self.calculate_ale_1d(feature_name, bins, quantile_bins)
        
        if ale_result is None:
            print(f"特徴量 '{feature_name}' のALEプロットを作成できませんでした")
            return None
        
        # プロット作成
        fig, ax = plt.subplots(figsize=figsize)
        
        # ALEプロット
        ax.plot(ale_result['bin_centers'], ale_result['ale_values'], 
               linewidth=2, color='blue', marker='o', markersize=4, label='ALE')
        
        # ゼロラインの追加
        ax.axhline(y=0, color='red', linestyle='--', alpha=0.7, linewidth=1)
        
        # データ分布のヒストグラム（背景）
        if show_data_distribution:
            x_values = self.X[feature_name].values
            ax2 = ax.twinx()
            ax2.hist(x_values, bins=30, alpha=0.3, color='gray', density=True)
            ax2.set_ylabel('データ密度', fontsize=12, color='gray')
            ax2.tick_params(axis='y', labelcolor='gray')
        
        # 軸設定
        ax.set_xlabel(feature_name, fontsize=12)
        ax.set_ylabel('ALE (Accumulated Local Effects)', fontsize=12)
        
        if title is None:
            title = f'ALE Plot: {feature_name}'
        ax.set_title(title, fontsize=14)
        
        ax.grid(True, alpha=0.3)
        ax.legend()
        
        # 統計情報の追加
        ale_range = ale_result['ale_range']
        textstr = f'ALE範囲: {ale_range:.4f}\nビン数: {ale_result["n_bins"]}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        plt.tight_layout()
        
        # 保存
        if save and self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = feature_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
            save_path = os.path.join(self.output_dir, f'ale_1d_{safe_name}_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ALEプロットを保存: {save_path}")
        
        return fig
    
    def calculate_ale_2d(self, feature1, feature2, bins=20, quantile_bins=True):
        """
        2次元ALEを計算
        
        Parameters:
        -----------
        feature1 : str
            第1特徴量名
        feature2 : str
            第2特徴量名
        bins : int, default=20
            各次元のビン数
        quantile_bins : bool, default=True
            分位数ベースのビニングを使用するか
            
        Returns:
        --------
        dict
            2次元ALE計算結果
        """
        if feature1 not in self.feature_names or feature2 not in self.feature_names:
            raise ValueError(f"特徴量が見つかりません")
        
        feature1_idx = self.feature_names.index(feature1)
        feature2_idx = self.feature_names.index(feature2)
        
        x1_values = self.X.iloc[:, feature1_idx].values
        x2_values = self.X.iloc[:, feature2_idx].values
        
        # ビンの作成
        if quantile_bins:
            x1_edges = np.unique(np.quantile(x1_values, np.linspace(0, 1, bins + 1)))
            x2_edges = np.unique(np.quantile(x2_values, np.linspace(0, 1, bins + 1)))
        else:
            x1_edges = np.linspace(x1_values.min(), x1_values.max(), bins + 1)
            x2_edges = np.linspace(x2_values.min(), x2_values.max(), bins + 1)
        
        # 2次元ALE計算
        ale_2d = np.zeros((len(x1_edges) - 1, len(x2_edges) - 1))
        
        for i in range(len(x1_edges) - 1):
            for j in range(len(x2_edges) - 1):
                # セルに含まれるデータポイントを特定
                mask = ((x1_values >= x1_edges[i]) & (x1_values <= x1_edges[i + 1]) &
                       (x2_values >= x2_edges[j]) & (x2_values <= x2_edges[j + 1]))
                
                if not np.any(mask):
                    continue
                
                X_cell = self.X[mask].copy()
                
                if len(X_cell) == 0:
                    continue
                
                # 4つの角での予測
                X_00 = X_cell.copy()
                X_00.iloc[:, feature1_idx] = x1_edges[i]
                X_00.iloc[:, feature2_idx] = x2_edges[j]
                
                X_01 = X_cell.copy()
                X_01.iloc[:, feature1_idx] = x1_edges[i]
                X_01.iloc[:, feature2_idx] = x2_edges[j + 1]
                
                X_10 = X_cell.copy()
                X_10.iloc[:, feature1_idx] = x1_edges[i + 1]
                X_10.iloc[:, feature2_idx] = x2_edges[j]
                
                X_11 = X_cell.copy()
                X_11.iloc[:, feature1_idx] = x1_edges[i + 1]
                X_11.iloc[:, feature2_idx] = x2_edges[j + 1]
                
                # 予測値の計算
                pred_00 = np.mean(self.model.predict(X_00.values))
                pred_01 = np.mean(self.model.predict(X_01.values))
                pred_10 = np.mean(self.model.predict(X_10.values))
                pred_11 = np.mean(self.model.predict(X_11.values))
                
                # 2次元局所効果の計算
                local_effect_2d = pred_11 - pred_10 - pred_01 + pred_00
                ale_2d[i, j] = local_effect_2d
        
        # 累積効果の計算（2次元の累積和）
        cumulative_ale_2d = np.cumsum(np.cumsum(ale_2d, axis=0), axis=1)
        
        # 中央化
        mean_effect = np.mean(cumulative_ale_2d)
        cumulative_ale_2d = cumulative_ale_2d - mean_effect
        
        # グリッド中心点の計算
        x1_centers = (x1_edges[:-1] + x1_edges[1:]) / 2
        x2_centers = (x2_edges[:-1] + x2_edges[1:]) / 2
        
        result = {
            'feature1': feature1,
            'feature2': feature2,
            'x1_centers': x1_centers,
            'x2_centers': x2_centers,
            'ale_2d': cumulative_ale_2d,
            'local_effects_2d': ale_2d,
            'x1_edges': x1_edges,
            'x2_edges': x2_edges,
            'ale_range': cumulative_ale_2d.max() - cumulative_ale_2d.min()
        }
        
        return result
    
    def plot_ale_2d(self, feature1, feature2, bins=20, quantile_bins=True,
                   figsize=(12, 10), title=None, save=True):
        """
        2次元ALEプロットを作成
        
        Parameters:
        -----------
        feature1 : str
            第1特徴量名
        feature2 : str
            第2特徴量名
        bins : int, default=20
            各次元のビン数
        quantile_bins : bool, default=True
            分位数ベースのビニングを使用するか
        figsize : tuple, default=(12, 10)
            図のサイズ
        title : str, optional
            図のタイトル
        save : bool, default=True
            図を保存するか
            
        Returns:
        --------
        matplotlib.figure.Figure
            作成された図
        """
        # 2次元ALE計算
        ale_result_2d = self.calculate_ale_2d(feature1, feature2, bins, quantile_bins)
        
        # プロット作成
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # メッシュグリッド作成
        X1, X2 = np.meshgrid(ale_result_2d['x1_centers'], ale_result_2d['x2_centers'], indexing='ij')
        
        # ヒートマップ
        im = axes[0].contourf(X1, X2, ale_result_2d['ale_2d'], levels=20, cmap='RdBu_r')
        axes[0].set_xlabel(feature1, fontsize=12)
        axes[0].set_ylabel(feature2, fontsize=12)
        axes[0].set_title('2D ALE Heatmap', fontsize=14)
        plt.colorbar(im, ax=axes[0], label='ALE Value')
        
        # 等高線プロット
        contour = axes[1].contour(X1, X2, ale_result_2d['ale_2d'], levels=15, colors='black', alpha=0.7)
        axes[1].clabel(contour, inline=True, fontsize=8)
        axes[1].contourf(X1, X2, ale_result_2d['ale_2d'], levels=20, cmap='RdBu_r', alpha=0.6)
        axes[1].set_xlabel(feature1, fontsize=12)
        axes[1].set_ylabel(feature2, fontsize=12)
        axes[1].set_title('2D ALE Contour', fontsize=14)
        
        if title:
            fig.suptitle(title, fontsize=16)
        else:
            fig.suptitle(f'2D ALE Plot: {feature1} vs {feature2}', fontsize=16)
        
        plt.tight_layout()
        
        # 保存
        if save and self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name1 = feature1.replace('/', '_').replace('\\', '_').replace(' ', '_')
            safe_name2 = feature2.replace('/', '_').replace('\\', '_').replace(' ', '_')
            save_path = os.path.join(self.output_dir, f'ale_2d_{safe_name1}_{safe_name2}_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"2次元ALEプロットを保存: {save_path}")
        
        return fig
    
    def create_ale_dashboard(self, features=None, n_features=5, 
                           include_2d=True, bins_1d=50, bins_2d=20):
        """
        複数特徴量のALEダッシュボードを作成
        
        Parameters:
        -----------
        features : list, optional
            対象特徴量のリスト
        n_features : int, default=5
            上位n個の特徴量を選択（featuresが指定されていない場合）
        include_2d : bool, default=True
            2次元ALEプロットを含むか
        bins_1d : int, default=50
            1次元のビン数
        bins_2d : int, default=20
            2次元のビン数
            
        Returns:
        --------
        dict
            作成されたプロットの情報
        """
        if features is None:
            # 特徴量重要度を計算（簡易版）
            if hasattr(self.model, 'feature_importances_'):
                importances = self.model.feature_importances_
                top_indices = np.argsort(importances)[-n_features:][::-1]
                features = [self.feature_names[i] for i in top_indices]
            else:
                # ランダムに選択
                features = self.feature_names[:min(n_features, len(self.feature_names))]
        
        dashboard_info = {
            'ale_1d_plots': [],
            'ale_2d_plots': [],
            'features_analyzed': features,
            'ale_summary': {}
        }
        
        print(f"ALEダッシュボード作成中（対象特徴量: {len(features)}個）...")
        
        # 1次元ALEプロット
        for feature in features:
            print(f"  1D ALE: {feature}")
            try:
                fig = self.plot_ale_1d(feature, bins=bins_1d, save=True)
                if fig is not None:
                    ale_result = self.calculate_ale_1d(feature, bins=bins_1d)
                    dashboard_info['ale_1d_plots'].append({
                        'feature': feature,
                        'figure': fig,
                        'ale_range': ale_result['ale_range'],
                        'n_bins': ale_result['n_bins']
                    })
                    plt.close(fig)
            except Exception as e:
                print(f"    エラー: {e}")
        
        # 2次元ALEプロット（重要な特徴量の組み合わせ）
        if include_2d and len(features) >= 2:
            n_pairs = min(3, len(features) * (len(features) - 1) // 2)
            feature_pairs = []
            
            for i in range(len(features)):
                for j in range(i + 1, len(features)):
                    feature_pairs.append((features[i], features[j]))
                    if len(feature_pairs) >= n_pairs:
                        break
                if len(feature_pairs) >= n_pairs:
                    break
            
            for feature1, feature2 in feature_pairs:
                print(f"  2D ALE: {feature1} vs {feature2}")
                try:
                    fig = self.plot_ale_2d(feature1, feature2, bins=bins_2d, save=True)
                    if fig is not None:
                        ale_result_2d = self.calculate_ale_2d(feature1, feature2, bins=bins_2d)
                        dashboard_info['ale_2d_plots'].append({
                            'feature1': feature1,
                            'feature2': feature2,
                            'figure': fig,
                            'ale_range': ale_result_2d['ale_range']
                        })
                        plt.close(fig)
                except Exception as e:
                    print(f"    エラー: {e}")
        
        # サマリー情報
        dashboard_info['ale_summary'] = {
            'total_features_analyzed': len(features),
            'ale_1d_created': len(dashboard_info['ale_1d_plots']),
            'ale_2d_created': len(dashboard_info['ale_2d_plots']),
            'output_directory': self.output_dir
        }
        
        print(f"ALEダッシュボード作成完了")
        print(f"  1D プロット: {len(dashboard_info['ale_1d_plots'])}個")
        print(f"  2D プロット: {len(dashboard_info['ale_2d_plots'])}個")
        
        return dashboard_info
    
    def compare_ale_vs_pdp(self, feature_name, bins=50, figsize=(15, 6)):
        """
        ALEと部分依存プロット（PDP）を比較
        
        Parameters:
        -----------
        feature_name : str
            対象特徴量名
        bins : int, default=50
            ビン数
        figsize : tuple, default=(15, 6)
            図のサイズ
            
        Returns:
        --------
        matplotlib.figure.Figure
            比較プロット
        """
        # ALE計算
        ale_result = self.calculate_ale_1d(feature_name, bins)
        
        if ale_result is None:
            print(f"特徴量 '{feature_name}' のALE-PDP比較プロットを作成できませんでした")
            return None
        
        # PDPの簡易計算
        feature_idx = self.feature_names.index(feature_name)
        x_values = self.X.iloc[:, feature_idx].values
        x_range = np.linspace(x_values.min(), x_values.max(), bins)
        
        pdp_values = []
        for x_val in x_range:
            X_temp = self.X.copy()
            X_temp.iloc[:, feature_idx] = x_val
            pred = np.mean(self.model.predict(X_temp.values))
            pdp_values.append(pred)
        
        pdp_values = np.array(pdp_values)
        # PDPを中央化
        pdp_values = pdp_values - np.mean(pdp_values)
        
        # 比較プロット作成
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # ALE プロット
        axes[0].plot(ale_result['bin_centers'], ale_result['ale_values'], 
                    linewidth=2, color='blue', marker='o', markersize=4, label='ALE')
        axes[0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[0].set_xlabel(feature_name)
        axes[0].set_ylabel('Effect')
        axes[0].set_title('ALE Plot')
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # PDP プロット
        axes[1].plot(x_range, pdp_values, linewidth=2, color='green', marker='s', markersize=4, label='PDP')
        axes[1].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[1].set_xlabel(feature_name)
        axes[1].set_ylabel('Effect')
        axes[1].set_title('PDP (Partial Dependence Plot)')
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # 比較プロット
        # ALEとPDPを同じスケールに正規化
        ale_normalized = ale_result['ale_values'] / np.std(ale_result['ale_values']) if np.std(ale_result['ale_values']) > 0 else ale_result['ale_values']
        pdp_normalized = pdp_values / np.std(pdp_values) if np.std(pdp_values) > 0 else pdp_values
        
        axes[2].plot(ale_result['bin_centers'], ale_normalized, 
                    linewidth=2, color='blue', marker='o', markersize=4, label='ALE (normalized)')
        
        # PDPを補間してALEと同じx軸で表示
        pdp_interp = np.interp(ale_result['bin_centers'], x_range, pdp_normalized)
        axes[2].plot(ale_result['bin_centers'], pdp_interp, 
                    linewidth=2, color='green', marker='s', markersize=4, label='PDP (normalized)')
        
        axes[2].axhline(y=0, color='red', linestyle='--', alpha=0.7)
        axes[2].set_xlabel(feature_name)
        axes[2].set_ylabel('Normalized Effect')
        axes[2].set_title('ALE vs PDP Comparison')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.suptitle(f'ALE vs PDP Comparison: {feature_name}', fontsize=16)
        plt.tight_layout()
        
        # 保存
        if self.output_dir:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            safe_name = feature_name.replace('/', '_').replace('\\', '_').replace(' ', '_')
            save_path = os.path.join(self.output_dir, f'ale_vs_pdp_{safe_name}_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"ALE-PDP比較プロットを保存: {save_path}")
        
        return fig
    
    def get_ale_summary_report(self):
        """
        ALE分析のサマリーレポートを取得
        
        Returns:
        --------
        dict
            サマリー情報
        """
        summary = {
            'model_info': {
                'model_type': type(self.model).__name__,
                'has_feature_importances': hasattr(self.model, 'feature_importances_')
            },
            'data_info': {
                'n_samples': len(self.X),
                'n_features': len(self.feature_names),
                'feature_names': self.feature_names
            },
            'ale_cache_info': {
                'cached_analyses': len(self.ale_cache),
                'cached_features': list(set([key.split('_')[0] for key in self.ale_cache.keys()]))
            },
            'output_info': {
                'output_directory': self.output_dir,
                'output_enabled': self.output_dir is not None
            }
        }
        
        return summary
    
    def clear_cache(self):
        """ALEキャッシュをクリア"""
        self.ale_cache.clear()
        print("ALEキャッシュをクリアしました")


# 便利関数（クラス外）
def quick_ale_analysis(model, X, feature_names=None, top_features=3, output_dir=None):
    """
    ALEのクイック分析用便利関数
    
    Parameters:
    -----------
    model : sklearn-like model
        学習済みモデル
    X : array-like or pandas.DataFrame
        特徴量データ
    feature_names : list, optional
        特徴量名
    top_features : int, default=3
        分析する上位特徴量数
    output_dir : str, optional
        出力ディレクトリ
        
    Returns:
    --------
    tuple
        (ale_plotter, dashboard_info)
    """
    ale_plotter = ALEPlotter(model, X, feature_names, output_dir)
    dashboard_info = ale_plotter.create_ale_dashboard(n_features=top_features, include_2d=True)
    
    return ale_plotter, dashboard_info


if __name__ == "__main__":
    print("ALEPlotter クラス")
    print("使用例:")
    print("""
    from ale_plotter import ALEPlotter
    
    # インスタンス作成
    ale_plotter = ALEPlotter(
        model=trained_model,
        X=X_test,
        feature_names=feature_names,
        output_dir='ale_output'
    )
    
    # 1次元ALE分析
    ale_plotter.plot_ale_1d('feature_name')
    
    # 2次元ALE分析
    ale_plotter.plot_ale_2d('feature1', 'feature2')
    
    # ダッシュボード作成
    dashboard = ale_plotter.create_ale_dashboard(n_features=5)
    
    # ALE vs PDP比較
    ale_plotter.compare_ale_vs_pdp('feature_name')
    """)