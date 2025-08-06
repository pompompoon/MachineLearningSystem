"""
部分依存プロットの生成と分析用モジュール
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
from sklearn.preprocessing import StandardScaler
import os
import warnings

class PartialDependencePlotter:
    """
    部分依存グラフを生成・分析するためのクラス
    """
    
    def __init__(self, model=None, features=None, target_names=None):
        """
        初期化関数
        
        Parameters:
        -----------
        model : 学習済みモデル, default=None
            部分依存プロットを生成するための学習済みモデル
        features : pandas.DataFrame, default=None
            特徴量データ
        target_names : list or dict, default=None
            目標変数の名前（クラス名）のリストまたは辞書
            例: ['intact', 'mci'] または {0: 'intact', 1: 'mci'}
        """
        self.model = model
        self.features = features
        self.target_names = target_names
        self.scaler = None
        
    def set_model(self, model):
        """モデルを設定"""
        self.model = model
        return self
    
    def set_features(self, features):
        """特徴量を設定"""
        self.features = features
        return self
    
    def set_target_names(self, target_names):
        """目標変数の名前を設定"""
        self.target_names = target_names
        return self
    
    def scale_features(self, features=None):
        """特徴量をスケーリング"""
        if features is None:
            features = self.features
            
        if self.scaler is None:
            self.scaler = StandardScaler()
            scaled_features = pd.DataFrame(
                self.scaler.fit_transform(features),
                columns=features.columns
            )
        else:
            scaled_features = pd.DataFrame(
                self.scaler.transform(features),
                columns=features.columns
            )
            
        return scaled_features
    
    def plot_single_feature(self, feature_idx, target_idx=0, ax=None, centered=False, grid_resolution=100):
        """
        単一の特徴量に対する部分依存グラフを作成
        
        Parameters:
        -----------
        feature_idx : int or str
            プロットする特徴量のインデックスまたは名前
        target_idx : int, default=0
            プロットするターゲットクラスのインデックス
            (回帰問題では通常0)
        ax : matplotlib.axes.Axes, default=None
            プロット先の軸。Noneの場合は新しい図を作成
        centered : bool, default=False
            プロットを中心化するかどうか
        grid_resolution : int, default=100
            グリッド解像度
            
        Returns:
        --------
        matplotlib.figure.Figure, matplotlib.axes.Axes
            作成された図と軸
        """
        if self.model is None or self.features is None:
            raise ValueError("モデルと特徴量を先に設定してください")
        
        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 6))
        else:
            fig = ax.figure
            
        # 特徴量名の取得
        if isinstance(feature_idx, str):
            feature_name = feature_idx
            if feature_name in self.features.columns:
                feature_idx = list(self.features.columns).index(feature_name)
            else:
                raise ValueError(f"特徴量 '{feature_name}' がデータフレームに存在しません")
        else:
            feature_name = self.features.columns[feature_idx]
        
        # ターゲット名の取得
        target_name = f"予測値"
        if self.target_names is not None:
            if isinstance(self.target_names, dict) and target_idx in self.target_names:
                target_name = self.target_names[target_idx]
            elif isinstance(self.target_names, list) and 0 <= target_idx < len(self.target_names):
                target_name = self.target_names[target_idx]
        
        # スケーリングされた特徴量でプロット
        scaled_features = self.scale_features()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            # 最新のscikit-learnのAPIを使用
            display = PartialDependenceDisplay.from_estimator(
                self.model, scaled_features, [feature_idx], 
                target=target_idx, ax=ax,
                kind='average', centered=centered,
                grid_resolution=grid_resolution
            )
        
        # グラフの装飾
        ax.set_title(f"部分依存プロット: {feature_name}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel(feature_name, fontsize=10)
        ax.set_ylabel(f"部分依存 ({target_name})", fontsize=10)
        
        return fig, ax
        
    def plot_multiple_features(self, feature_indices, target_idx=0, n_cols=3, figsize=None,
                              centered=False, grid_resolution=100):
        """
        複数の特徴量に対する部分依存グラフを作成
        
        Parameters:
        -----------
        feature_indices : list
            プロットする特徴量のインデックスまたは名前のリスト
        target_idx : int, default=0
            プロットするターゲットクラスのインデックス
            (回帰問題では通常0)
        n_cols : int, default=3
            列の数
        figsize : tuple, default=None
            図のサイズ。Noneの場合は自動計算
        centered : bool, default=False
            プロットを中心化するかどうか
        grid_resolution : int, default=100
            グリッド解像度
            
        Returns:
        --------
        matplotlib.figure.Figure
            作成された図
        """
        if self.model is None or self.features is None:
            raise ValueError("モデルと特徴量を先に設定してください")
        
        # 特徴量インデックスの変換
        feature_idxs = []
        feature_names = []
        for idx in feature_indices:
            if isinstance(idx, str):
                if idx in self.features.columns:
                    feature_idxs.append(list(self.features.columns).index(idx))
                    feature_names.append(idx)
                else:
                    raise ValueError(f"特徴量 '{idx}' がデータフレームに存在しません")
            else:
                feature_idxs.append(idx)
                feature_names.append(self.features.columns[idx])
        
        # 図のサイズを計算
        n_features = len(feature_idxs)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        if figsize is None:
            figsize = (n_cols * 5, n_rows * 4)
        
        # スケーリングされた特徴量でプロット
        scaled_features = self.scale_features()
        
        # ターゲット名の取得
        target_name = f"予測値"
        if self.target_names is not None:
            if isinstance(self.target_names, dict) and target_idx in self.target_names:
                target_name = self.target_names[target_idx]
            elif isinstance(self.target_names, list) and 0 <= target_idx < len(self.target_names):
                target_name = self.target_names[target_idx]
        
        # 最新のscikit-learnのAPIを使用
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize)
            display = PartialDependenceDisplay.from_estimator(
                self.model, scaled_features, feature_idxs, 
                target=target_idx, ax=ax.flatten()[:n_features] if n_features > 1 else ax,
                kind='average', centered=centered,
                grid_resolution=grid_resolution
            )
            
        # 使用しない軸を非表示に
        if n_features > 1:
            for i in range(n_features, n_rows * n_cols):
                if i < len(ax.flatten()):
                    ax.flatten()[i].set_visible(False)
        
        # 軸のタイトルを特徴量名に変更
        axes_to_use = display.axes_.flatten() if n_features > 1 else [display.axes_]
        for i, (pdp_ax, name) in enumerate(zip(axes_to_use, feature_names)):
            pdp_ax.set_title(name, fontsize=10)
            pdp_ax.grid(True, linestyle='--', alpha=0.6)
            
        fig.suptitle(f"部分依存プロット", fontsize=14)
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def plot_feature_interaction(self, feature_pair, target_idx=0, figsize=(8, 6),
                            contour_kw=None, ax=None, grid_resolution=50):
        """
        2つの特徴量間の相互作用を2Dプロットで可視化（修正版）
        
        Parameters:
        -----------
        feature_pair : tuple or list
            プロットする2つの特徴量のインデックスまたは名前のペア
        target_idx : int, default=0
            プロットするターゲットクラスのインデックス
        figsize : tuple, default=(8, 6)
            図のサイズ
        contour_kw : dict, default=None
            等高線プロットのキーワード引数
        ax : matplotlib.axes.Axes, default=None
            プロット先の軸。Noneの場合は新しい図を作成
        grid_resolution : int, default=50
            グリッド解像度
            
        Returns:
        --------
        matplotlib.figure.Figure, matplotlib.axes.Axes
            作成された図と軸
        """
        if self.model is None or self.features is None:
            raise ValueError("モデルと特徴量を先に設定してください")
            
        if len(feature_pair) != 2:
            raise ValueError("feature_pairは2つの要素を持つリストかタプルである必要があります")
            
        # 特徴量インデックスの変換
        feature_idxs = []
        feature_names = []
        
        for idx in feature_pair:
            if isinstance(idx, str):
                if idx in self.features.columns:
                    feature_idxs.append(list(self.features.columns).index(idx))
                    feature_names.append(idx)
                else:
                    raise ValueError(f"特徴量 '{idx}' がデータフレームに存在しません")
            else:
                feature_idxs.append(idx)
                feature_names.append(self.features.columns[idx])
                
        # ターゲット名の取得
        target_name = f"予測値"
        if self.target_names is not None:
            if isinstance(self.target_names, dict) and target_idx in self.target_names:
                target_name = self.target_names[target_idx]
            elif isinstance(self.target_names, list) and 0 <= target_idx < len(self.target_names):
                target_name = self.target_names[target_idx]
        
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
            
        # スケーリングされた特徴量でプロット
        scaled_features = self.scale_features()
        
        # 2つの特徴量の組み合わせを計算
        feature_pair_idx = (feature_idxs[0], feature_idxs[1])
        
        print(f"相互作用プロット生成: {feature_names[0]} × {feature_names[1]}")
        
        # === 修正箇所：エラー回避のための複数の方法を試行 ===
        
        try:
            # 方法1: contour_kwパラメータなしで試行
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                display = PartialDependenceDisplay.from_estimator(
                    self.model, scaled_features, [feature_pair_idx], 
                    target=target_idx, ax=ax,
                    kind='average', grid_resolution=grid_resolution
                    # contour_kw は削除してデフォルトに任せる
                )
                
            ax.set_title(f"特徴量相互作用: {feature_names[0]} vs {feature_names[1]}", fontsize=12)
            print(f"✓ 相互作用プロット生成成功: {feature_names[0]} × {feature_names[1]}")
            
        except Exception as e:
            print(f"PartialDependenceDisplay エラー: {e}")
            
            # 方法2: 手動で部分依存を計算してプロット
            try:
                print("手動での相互作用プロット生成を試行...")
                
                # 部分依存値を手動で計算
                from sklearn.inspection import partial_dependence
                pd_result = partial_dependence(
                    self.model,
                    scaled_features,
                    features=[feature_pair_idx],
                    grid_resolution=grid_resolution,
                    kind='average'
                )
                
                # 結果を取得
                values = pd_result['values'][0]
                grid = pd_result['grid']
                
                # 手動でcontourプロット
                X_grid, Y_grid = np.meshgrid(grid[0], grid[1])
                
                # pcolormeshを使用（contourfより安全）
                im = ax.pcolormesh(X_grid, Y_grid, values, shading='auto', cmap='viridis')
                
                # カラーバーを追加
                fig.colorbar(im, ax=ax, label='Partial Dependence')
                
                # 等高線を追加（オプション、エラーが出ても続行）
                try:
                    contour = ax.contour(X_grid, Y_grid, values, levels=10, colors='white', alpha=0.6, linewidths=0.8)
                    ax.clabel(contour, inline=True, fontsize=8)
                except:
                    pass  # 等高線でエラーが出ても無視
                
                # ラベル設定
                ax.set_xlabel(feature_names[0])
                ax.set_ylabel(feature_names[1])
                ax.set_title(f"特徴量相互作用: {feature_names[0]} vs {feature_names[1]}")
                
                print(f"✓ 手動での相互作用プロット生成成功: {feature_names[0]} × {feature_names[1]}")
                
            except Exception as e2:
                print(f"手動プロット生成もエラー: {e2}")
                
                # 方法3: 最もシンプルなフォールバック
                ax.text(0.5, 0.5, 
                    f"相互作用プロット生成失敗\n{feature_names[0]} × {feature_names[1]}\n\n"
                    f"エラー1: {str(e)[:50]}...\n"
                    f"エラー2: {str(e2)[:50]}...", 
                    ha='center', va='center', transform=ax.transAxes, fontsize=10,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.5))
                ax.set_title(f"相互作用プロット: {feature_names[0]} × {feature_names[1]} (エラー)")
                ax.set_xlabel(feature_names[0])
                ax.set_ylabel(feature_names[1])
                
                print(f"✗ 相互作用プロット生成失敗: {feature_names[0]} × {feature_names[1]}")
        
        return fig, ax
    def save_plots(self, feature_importances, output_dir='plots', target_idx=0, 
               n_features=10, n_cols=3, grid_resolution=100, format='png', dpi=300):
        """
        重要度の高い特徴量の部分依存プロットを保存（修正版）
        
        Parameters:
        -----------
        feature_importances : pandas.DataFrame
            'feature'と'importance'列を持つ特徴量重要度のデータフレーム
        output_dir : str, default='plots'
            プロットの保存先ディレクトリ
        target_idx : int, default=0
            プロットするターゲットクラスのインデックス
            (回帰問題では通常0)
        n_features : int, default=10
            プロットする上位n個の特徴量
        n_cols : int, default=3
            列の数
        grid_resolution : int, default=100
            グリッド解像度
        format : str, default='png'
            保存する画像形式（'png', 'pdf', 'svg'など）
        dpi : int, default=300
            画像のDPI（解像度）
            
        Returns:
        --------
        list
            保存されたファイルパスのリスト
        """
        if self.model is None or self.features is None:
            raise ValueError("モデルと特徴量を先に設定してください")
            
        # 出力ディレクトリを作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 重要度の高い特徴量を取得
        top_features = feature_importances.sort_values('importance', ascending=False).head(n_features)
        top_feature_names = top_features['feature'].tolist()
        
        # ターゲット名の取得
        target_name = f"target_{target_idx}"
        if self.target_names is not None:
            if isinstance(self.target_names, dict) and target_idx in self.target_names:
                target_name = self.target_names[target_idx]
            elif isinstance(self.target_names, list) and 0 <= target_idx < len(self.target_names):
                target_name = self.target_names[target_idx]
                
        target_name = target_name.lower().replace(' ', '_')
        
        saved_files = []
        
        print(f"部分依存プロット保存開始: {len(top_feature_names)}個の特徴量")
        
        # 1. 全ての重要な特徴量の部分依存プロットを一つの図に
        try:
            print("統合部分依存プロット生成中...")
            fig = self.plot_multiple_features(
                top_feature_names, target_idx=target_idx, n_cols=n_cols,
                grid_resolution=grid_resolution
            )
            
            filename = os.path.join(output_dir, f"pdp_top{n_features}.{format}")
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(filename)
            print(f"✓ 統合プロット保存: {filename}")
            
        except Exception as e:
            print(f"✗ 統合プロット生成エラー: {e}")
        
        # # 2. 個別の特徴量ごとにプロット
        # print("個別特徴量プロット生成中...")
        # for i, feature_name in enumerate(top_feature_names):
        #     try:
        #         fig, ax = self.plot_single_feature(
        #             feature_name, target_idx=target_idx,
        #             grid_resolution=grid_resolution
        #         )
                
        #         # ファイル名から無効な文字を削除
        #         clean_name = feature_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
        #         clean_name = ''.join(c for c in clean_name if c.isalnum() or c in '_-.')
                
        #         filename = os.path.join(output_dir, f"pdp_{clean_name}.{format}")
        #         fig.savefig(filename, dpi=dpi, bbox_inches='tight')
        #         plt.close(fig)
        #         saved_files.append(filename)
        #         print(f"✓ 個別プロット保存 ({i+1}/{len(top_feature_names)}): {filename}")
                
        #     except Exception as e:
        #         print(f"✗ 個別プロット生成エラー ({feature_name}): {e}")
                
        # 3. 上位2つの特徴量の相互作用プロット（修正版を使用）
        if len(top_feature_names) >= 2:
            try:
                print("特徴量相互作用プロット生成中...")
                fig, ax = self.plot_feature_interaction(
                    top_feature_names[:2], target_idx=target_idx,
                    grid_resolution=min(grid_resolution, 50)  # 2Dプロットは解像度を下げる
                )
                
                clean_name1 = top_feature_names[0].replace(' ', '_').replace('/', '_').replace('\\', '_')
                clean_name1 = ''.join(c for c in clean_name1 if c.isalnum() or c in '_-.')
                
                clean_name2 = top_feature_names[1].replace(' ', '_').replace('/', '_').replace('\\', '_')
                clean_name2 = ''.join(c for c in clean_name2 if c.isalnum() or c in '_-.')
                
                filename = os.path.join(output_dir, f"pdp_interaction_{clean_name1}_{clean_name2}.{format}")
                fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(filename)
                print(f"✓ 相互作用プロット保存: {filename}")
                
            except Exception as e:
                print(f"✗ 相互作用プロット生成エラー: {e}")
                import traceback
                traceback.print_exc()
            
        print(f"✓ 部分依存プロット保存完了: {len(saved_files)}個のファイルを {output_dir} に保存")
        return saved_files


def analyze_partial_dependence(model, X, feature_importances, output_dir='pdp_plots', target_names=None):
    """
    モデルの部分依存分析を実行し、プロットを生成する関数
    
    Parameters:
    -----------
    model : 学習済みモデル
        部分依存プロットを生成するためのモデル
    X : pandas.DataFrame
        特徴量データ
    feature_importances : pandas.DataFrame
        'feature'と'importance'列を持つ特徴量重要度のデータフレーム
    output_dir : str, default='pdp_plots'
        プロットの保存先ディレクトリ
    target_names : list or dict, default=None
        目標変数の名前（クラス名）のリストまたは辞書
        
    Returns:
    --------
    PartialDependencePlotter
        プロッターのインスタンス
    """
    # プロッターの初期化
    plotter = PartialDependencePlotter(
        model=model,
        features=X,
        target_names=target_names
    )
    
    # 部分依存プロットの作成と保存
    plotter.save_plots(
        feature_importances,
        output_dir=output_dir,
        target_idx=0,  # 回帰問題では通常0
        n_features=10,
        n_cols=3,
        grid_resolution=50
    )
    
    return plotter