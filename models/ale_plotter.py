"""
Accumulated Local Effects (ALE) プロットの実装

ALEは特徴量間の相関を考慮した、より正確な特徴量の影響を可視化する手法です。
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import japanize_matplotlib
from typing import Optional, List, Tuple, Callable, Union
import os
from datetime import datetime

# 既存のPartialDependencePlotterクラスが必要なので、基底クラスを定義
class PartialDependenceBase:
    """部分依存プロットの基底クラス（簡易版）"""
    
    def __init__(self, estimator, X: Union[np.ndarray, pd.DataFrame]):
        """
        Parameters:
        -----------
        estimator : sklearn互換モデル
            学習済みモデル
        X : array-like
            特徴量データ
        """
        self.estimator = estimator
        if isinstance(X, pd.DataFrame):
            self.X = X.values
            self.feature_names = X.columns.tolist()
        else:
            self.X = X
            self.feature_names = [f"Feature_{i}" for i in range(X.shape[1])]
    
    def _predict_average(self, X_subset: np.ndarray, j: int, value: float) -> float:
        """特定の特徴量を固定値に設定した場合の平均予測値を計算"""
        X_modified = X_subset.copy()
        X_modified[:, j] = value
        predictions = self.estimator.predict(X_modified)
        return np.mean(predictions)


class AccumulatedLocalEffects(PartialDependenceBase):
    """Accumulated Local Effects Plot (ALE) の実装"""
    
    def __init__(self, estimator, X: Union[np.ndarray, pd.DataFrame]):
        """
        Parameters:
        -----------
        estimator : sklearn互換モデル
            学習済みモデル
        X : array-like
            特徴量データ
        """
        super().__init__(estimator, X)
        
    def _estimate_relationship(
        self, j: int, n_grid: int = 30
    ) -> Tuple[np.ndarray, np.ndarray]:
        """ALEを求める
        
        Args:
            j: ALEを計算したい特徴量のインデックス
            n_grid: グリッドを何分割するか
            
        Returns:
            特徴量の値とその場合のALE
        """
        # ターゲットの変数を、取りうる値の最大値から最小値まで動かせるようにする
        # quantileを使うことで、データの分布に応じた区間分割を行う
        xjks = np.quantile(self.X[:, j], q=np.linspace(0, 1, n_grid + 1))
        
        # 区間ごとに両端での予測値の平均的な差分を求める
        local_effects = np.zeros(n_grid)
        
        for k in range(1, n_grid + 1):
            # 現在の区間に含まれるデータポイントを選択
            if k == 1:
                mask = (self.X[:, j] >= xjks[k - 1]) & (self.X[:, j] <= xjks[k])
            else:
                mask = (self.X[:, j] > xjks[k - 1]) & (self.X[:, j] <= xjks[k])
            
            # 区間内にデータがある場合のみ計算
            if np.sum(mask) > 0:
                # 区間の両端での予測値の差分を計算
                upper_pred = self._predict_average(self.X[mask], j, xjks[k])
                lower_pred = self._predict_average(self.X[mask], j, xjks[k - 1])
                local_effects[k - 1] = upper_pred - lower_pred
            else:
                local_effects[k - 1] = 0
        
        # 累積和を計算してALEを得る
        accumulated_local_effects = np.cumsum(local_effects)
        
        # センタリング（平均を0にする）
        accumulated_local_effects -= np.mean(accumulated_local_effects)
        
        # グリッドポイントの中点を返す（可視化用）
        grid_points = (xjks[:-1] + xjks[1:]) / 2
        
        return (grid_points, accumulated_local_effects)
    
    def calculate_ale(self, feature_idx: int, n_grid: int = 30) -> Tuple[np.ndarray, np.ndarray]:
        """特定の特徴量のALEを計算"""
        return self._estimate_relationship(feature_idx, n_grid)
    
    def plot_single_feature(
        self, 
        feature_idx: int, 
        n_grid: int = 30,
        ax: Optional[plt.Axes] = None,
        figsize: Tuple[int, int] = (8, 6),
        title: Optional[str] = None
    ) -> plt.Figure:
        """単一特徴量のALEプロット"""
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        
        # ALEを計算
        grid_values, ale_values = self.calculate_ale(feature_idx, n_grid)
        
        # プロット
        ax.plot(grid_values, ale_values, 'b-', linewidth=2)
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        
        # ラグプロット（データ分布を示す）
        ax.scatter(self.X[:, feature_idx], 
                  np.zeros(len(self.X[:, feature_idx])) - 0.1 * np.ptp(ale_values),
                  alpha=0.1, s=10, color='gray')
        
        # ラベル設定
        feature_name = self.feature_names[feature_idx] if hasattr(self, 'feature_names') else f"Feature {feature_idx}"
        ax.set_xlabel(feature_name)
        ax.set_ylabel("ALE")
        
        if title is None:
            title = f"Accumulated Local Effects: {feature_name}"
        ax.set_title(title)
        
        ax.grid(True, alpha=0.3)
        
        return fig
    
    def plot_multiple_features(
        self,
        feature_indices: List[int],
        n_grid: int = 30,
        n_cols: int = 3,
        figsize: Optional[Tuple[int, int]] = None
    ) -> plt.Figure:
        """複数特徴量のALEプロット"""
        n_features = len(feature_indices)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (5 * n_cols, 4 * n_rows)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_rows == 1:
            axes = axes.reshape(1, -1)
        elif n_cols == 1:
            axes = axes.reshape(-1, 1)
        
        for idx, feature_idx in enumerate(feature_indices):
            row = idx // n_cols
            col = idx % n_cols
            ax = axes[row, col]
            
            self.plot_single_feature(feature_idx, n_grid, ax=ax)
        
        # 余分なサブプロットを削除
        for idx in range(n_features, n_rows * n_cols):
            row = idx // n_cols
            col = idx % n_cols
            fig.delaxes(axes[row, col])
        
        plt.tight_layout()
        return fig
    
    def compare_with_pdp(
        self, 
        feature_idx: int,
        pdp_calculator=None,  # PartialDependencePlotterのインスタンス
        n_grid: int = 30,
        figsize: Tuple[int, int] = (12, 5)
    ) -> plt.Figure:
        """ALEとPDPの比較プロット"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # ALEプロット
        self.plot_single_feature(feature_idx, n_grid, ax=ax1)
        ax1.set_title(f"ALE: {self.feature_names[feature_idx]}")
        
        # PDPプロット（pdp_calculatorが提供されている場合）
        if pdp_calculator is not None:
            try:
                pdp_calculator.plot_single_feature(feature_idx, ax=ax2)
                ax2.set_title(f"PDP: {self.feature_names[feature_idx]}")
            except Exception as e:
                ax2.text(0.5, 0.5, f"PDP計算エラー:\n{str(e)}", 
                        ha='center', va='center', transform=ax2.transAxes)
                ax2.set_title(f"PDP: {self.feature_names[feature_idx]} (エラー)")
        else:
            ax2.text(0.5, 0.5, "PDPCalculatorが\n提供されていません", 
                    ha='center', va='center', transform=ax2.transAxes)
            ax2.set_title(f"PDP: {self.feature_names[feature_idx]} (利用不可)")
        
        plt.tight_layout()
        return fig


def analyze_accumulated_local_effects(
    model,
    X: Union[np.ndarray, pd.DataFrame],
    feature_importances: Optional[pd.DataFrame] = None,
    output_dir: str = 'ale_plots',
    n_top_features: int = 10,
    n_grid: int = 30,
    compare_with_pdp: bool = False,
    pdp_calculator=None
) -> AccumulatedLocalEffects:
    """
    ALEの包括的な分析を実行する関数
    
    Parameters:
    -----------
    model : sklearn互換モデル
        学習済みモデル
    X : array-like
        特徴量データ
    feature_importances : pd.DataFrame, optional
        特徴量重要度（'feature'と'importance'カラムを含む）
    output_dir : str
        出力ディレクトリ
    n_top_features : int
        分析する上位特徴量数
    n_grid : int
        グリッド分割数
    compare_with_pdp : bool
        PDPとの比較を行うかどうか
    pdp_calculator : PartialDependencePlotter, optional
        PDP計算用のインスタンス
        
    Returns:
    --------
    AccumulatedLocalEffects
        ALEインスタンス
    """
    # 出力ディレクトリの作成
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # ALEインスタンスの作成
    ale = AccumulatedLocalEffects(model, X)
    
    # 分析する特徴量の選択
    if feature_importances is not None and len(feature_importances) > 0:
        # 重要度上位の特徴量を選択
        top_features = feature_importances.head(n_top_features)['feature'].tolist()
        feature_indices = []
        for feature in top_features:
            if feature in ale.feature_names:
                feature_indices.append(ale.feature_names.index(feature))
        feature_indices = feature_indices[:n_top_features]
    else:
        # 全特徴量から最初のn_top_features個を選択
        n_features = X.shape[1]
        feature_indices = list(range(min(n_top_features, n_features)))
    
    print(f"\n=== ALE分析開始 ===")
    print(f"対象特徴量数: {len(feature_indices)}")
    
    # 1. 複数特徴量のALEプロット
    if len(feature_indices) > 0:
        print("複数特徴量のALEプロットを生成中...")
        fig = ale.plot_multiple_features(feature_indices, n_grid=n_grid)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        ale_multi_path = os.path.join(output_dir, f'ale_top{len(feature_indices)}_{timestamp}.png')
        fig.savefig(ale_multi_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存: {ale_multi_path}")
    
    # 2. 個別特徴量のALEプロット（上位3つ）
    for i, feature_idx in enumerate(feature_indices[:3]):
        print(f"特徴量 '{ale.feature_names[feature_idx]}' のALEプロットを生成中...")
        fig = ale.plot_single_feature(feature_idx, n_grid=n_grid)
        
        ale_single_path = os.path.join(output_dir, f'ale_{ale.feature_names[feature_idx]}_{timestamp}.png')
        fig.savefig(ale_single_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        print(f"  保存: {ale_single_path}")
    
    # 3. ALEとPDPの比較（オプション）
    if compare_with_pdp and pdp_calculator is not None and len(feature_indices) > 0:
        print("\nALEとPDPの比較プロットを生成中...")
        for i, feature_idx in enumerate(feature_indices[:3]):
            fig = ale.compare_with_pdp(feature_idx, pdp_calculator, n_grid=n_grid)
            
            compare_path = os.path.join(output_dir, f'ale_vs_pdp_{ale.feature_names[feature_idx]}_{timestamp}.png')
            fig.savefig(compare_path, dpi=300, bbox_inches='tight')
            plt.close(fig)
            print(f"  保存: {compare_path}")
    
    print("\nALE分析完了!")
    
    return ale