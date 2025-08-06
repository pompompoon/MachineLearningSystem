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
    
    def plot_single_feature(self, feature_idx, target_idx=1, ax=None, centered=False, grid_resolution=100):
        """
        単一の特徴量に対する部分依存グラフを作成
        
        Parameters:
        -----------
        feature_idx : int or str
            プロットする特徴量のインデックスまたは名前
        target_idx : int, default=1
            プロットするターゲットクラスのインデックス
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
        target_name = f"Class {target_idx}"
        if self.target_names is not None:
            if isinstance(self.target_names, dict) and target_idx in self.target_names:
                target_name = self.target_names[target_idx]
            elif isinstance(self.target_names, list) and 0 <= target_idx < len(self.target_names):
                target_name = self.target_names[target_idx]
        
        # スケーリングされた特徴量でプロット
        scaled_features = self.scale_features()
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # 最新のscikit-learnのAPIを使用
                display = PartialDependenceDisplay.from_estimator(
                    self.model, scaled_features, [feature_idx], 
                    target=target_idx, ax=ax,
                    kind='average', centered=centered,
                    grid_resolution=grid_resolution
                )
            except TypeError:
                # scikit-learn 0.22以前のAPIの場合
                display = PartialDependenceDisplay.from_estimator(
                    self.model, scaled_features, [feature_idx], 
                    target=target_idx, ax=ax,
                    kind='average',  # centeredパラメータなし
                    grid_resolution=grid_resolution
                )
            except Exception as e:
                # 直接部分依存関数を使用して手動でプロット
                print(f"Warning: Using fallback method for PDP: {e}")
                pd_result = partial_dependence(
                    self.model, scaled_features, [feature_idx],
                    grid_resolution=grid_resolution
                )
                
                # APIバージョンによって戻り値の構造が異なる
                if isinstance(pd_result, tuple):
                    if len(pd_result) == 2:
                        pd_values, pd_grid = pd_result
                    else:
                        pd_values, pd_grid, _ = pd_result
                else:
                    pd_values = pd_result.average
                    pd_grid = pd_result.grid_values
                
                ax.plot(pd_grid[0], pd_values[0].T)
                
                # グラフスタイリング
                ax.set_xlabel(feature_name)
                ax.set_ylabel(f'Partial dependence')
        
        # グラフの装飾
        ax.set_title(f"部分依存プロット: {feature_name} → {target_name}", fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.6)
        ax.set_xlabel(feature_name, fontsize=10)
        ax.set_ylabel(f"部分依存 ({target_name})", fontsize=10)
        
        return fig, ax
        
    def plot_multiple_features(self, feature_indices, target_idx=1, n_cols=3, figsize=None,
                              centered=False, grid_resolution=100):
        """
        複数の特徴量に対する部分依存グラフを作成
        
        Parameters:
        -----------
        feature_indices : list
            プロットする特徴量のインデックスまたは名前のリスト
        target_idx : int, default=1
            プロットするターゲットクラスのインデックス
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
        target_name = f"Class {target_idx}"
        if self.target_names is not None:
            if isinstance(self.target_names, dict) and target_idx in self.target_names:
                target_name = self.target_names[target_idx]
            elif isinstance(self.target_names, list) and 0 <= target_idx < len(self.target_names):
                target_name = self.target_names[target_idx]
        
        # 図の作成
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        
        # 個別の特徴量に対してプロット
        for i, (feature_idx, feature_name) in enumerate(zip(feature_idxs, feature_names)):
            # 平坦化された軸の取得
            if n_rows == 1 and n_cols == 1:
                ax = axes
            elif n_rows == 1:
                ax = axes[i % n_cols]
            elif n_cols == 1:
                ax = axes[i // n_cols]
            else:
                ax = axes[i // n_cols, i % n_cols]
                
            try:
                # 個別の特徴量に対して部分依存プロットを作成
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    try:
                        # 最新のscikit-learnのAPIを使用
                        display = PartialDependenceDisplay.from_estimator(
                            self.model, scaled_features, [feature_idx], 
                            target=target_idx, ax=ax,
                            kind='average', centered=centered,
                            grid_resolution=grid_resolution
                        )
                    except TypeError:
                        # centeredパラメータがないバージョン
                        display = PartialDependenceDisplay.from_estimator(
                            self.model, scaled_features, [feature_idx], 
                            target=target_idx, ax=ax,
                            kind='average',
                            grid_resolution=grid_resolution
                        )
                    except Exception as e:
                        # 直接部分依存関数を使用して手動でプロット
                        print(f"Warning: Using fallback method for feature {feature_name}: {e}")
                        pd_result = partial_dependence(
                            self.model, scaled_features, [feature_idx],
                            grid_resolution=grid_resolution
                        )
                        
                        # APIバージョンによって戻り値の構造が異なる
                        if isinstance(pd_result, tuple):
                            if len(pd_result) == 2:
                                pd_values, pd_grid = pd_result
                            else:
                                pd_values, pd_grid, _ = pd_result
                        else:
                            pd_values = pd_result.average
                            pd_grid = pd_result.grid_values
                        
                        ax.plot(pd_grid[0], pd_values[0].T)
                
                # グラフの装飾
                ax.set_title(feature_name, fontsize=10)
                ax.grid(True, linestyle='--', alpha=0.6)
                
            except Exception as e:
                print(f"Error plotting feature {feature_name}: {e}")
                ax.text(0.5, 0.5, f"Error: {str(e)[:50]}...", 
                        ha='center', va='center', transform=ax.transAxes)
        
        # 使用しない軸を非表示に
        for i in range(n_features, n_rows * n_cols):
            if n_rows == 1:
                if i < len(axes):
                    axes[i].set_visible(False)
            elif n_cols == 1:
                if i < len(axes):
                    axes[i].set_visible(False)
            else:
                row, col = i // n_cols, i % n_cols
                if row < axes.shape[0] and col < axes.shape[1]:
                    axes[row, col].set_visible(False)
        
        fig.suptitle(f"部分依存プロット (ターゲット: {target_name})", fontsize=14)
        fig.tight_layout()
        plt.subplots_adjust(top=0.9)
        
        return fig
    
    def plot_feature_interaction(self, feature_pair, target_idx=1, figsize=(8, 6),
                                contour_kw=None, ax=None, grid_resolution=50):
        """
        2つの特徴量間の相互作用を2Dプロットで可視化
        
        Parameters:
        -----------
        feature_pair : tuple or list
            プロットする2つの特徴量のインデックスまたは名前のペア
        target_idx : int, default=1
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
        target_name = f"Class {target_idx}"
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
        
        # デフォルトの等高線プロットオプション
        if contour_kw is None:
            contour_kw = {
                'alpha': 0.75,
                'cmap': 'viridis',
                'levels': 50
            }
            
        # 2つの特徴量の組み合わせを計算
        feature_pair_idx = (feature_idxs[0], feature_idxs[1])
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                # 最新のscikit-learnのAPIを使用
                display = PartialDependenceDisplay.from_estimator(
                    self.model, scaled_features, [feature_pair_idx], 
                    target=target_idx, ax=ax,
                    kind='average', grid_resolution=grid_resolution,
                    contour_kw=contour_kw
                )
            except Exception as e:
                print(f"Warning: Using fallback method for interaction plot: {e}")
                # 手動で2D部分依存プロットを生成
                try:
                    pd_result = partial_dependence(
                        self.model, scaled_features, [feature_pair_idx],
                        grid_resolution=grid_resolution
                    )
                    
                    # APIバージョンによって戻り値の構造が異なる
                    if isinstance(pd_result, tuple):
                        if len(pd_result) == 2:
                            pd_values, pd_grid = pd_result
                        else:
                            pd_values, pd_grid, _ = pd_result
                    else:
                        pd_values = pd_result.average
                        pd_grid = pd_result.grid_values
                    
                    # 等高線プロットの作成
                    X, Y = np.meshgrid(pd_grid[0][0], pd_grid[0][1])
                    Z = pd_values[0].T
                    
                    CS = ax.contourf(X, Y, Z, **contour_kw)
                    fig.colorbar(CS, ax=ax)
                    
                    ax.set_xlabel(feature_names[0])
                    ax.set_ylabel(feature_names[1])
                except Exception as e2:
                    print(f"Failed to create interaction plot: {e2}")
                    ax.text(0.5, 0.5, f"Error creating plot: {str(e2)[:100]}...", 
                            ha='center', va='center', transform=ax.transAxes)
            
        ax.set_title(f"特徴量相互作用: {feature_names[0]} vs {feature_names[1]}\nターゲット: {target_name}", fontsize=12)
        
        return fig, ax
    
    def save_plots(self, feature_importances, output_dir='plots', target_idx=1, 
                   n_features=10, n_cols=3, grid_resolution=100, format='png', dpi=300):
        """
        重要度の高い特徴量の部分依存プロットを保存
        
        Parameters:
        -----------
        feature_importances : pandas.DataFrame
            'feature'と'importance'列を持つ特徴量重要度のデータフレーム
        output_dir : str, default='plots'
            プロットの保存先ディレクトリ
        target_idx : int, default=1
            プロットするターゲットクラスのインデックス
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
        
        # 存在しない特徴量を除外
        valid_features = [f for f in top_feature_names if f in self.features.columns]
        
        if len(valid_features) == 0:
            print("警告: 有効な特徴量が見つかりません。特徴量名を確認してください。")
            # 代わりにすべての特徴量を使用
            valid_features = self.features.columns.tolist()[:n_features]
            print(f"代わりに最初の{len(valid_features)}個の特徴量を使用します。")
        
        # ターゲット名の取得
        target_name = f"class_{target_idx}"
        if self.target_names is not None:
            if isinstance(self.target_names, dict) and target_idx in self.target_names:
                target_name = self.target_names[target_idx]
            elif isinstance(self.target_names, list) and 0 <= target_idx < len(self.target_names):
                target_name = self.target_names[target_idx]
                
        target_name = target_name.lower().replace(' ', '_')
        
        saved_files = []
        
        # 1. 全ての重要な特徴量の部分依存プロットを一つの図に
        try:
            fig = self.plot_multiple_features(
                valid_features, target_idx=target_idx, n_cols=n_cols,
                grid_resolution=grid_resolution
            )
            
            filename = os.path.join(output_dir, f"pdp_top{len(valid_features)}_{target_name}.{format}")
            fig.savefig(filename, dpi=dpi, bbox_inches='tight')
            plt.close(fig)
            saved_files.append(filename)
            print(f"保存しました: {filename}")
        except Exception as e:
            print(f"複数特徴量のプロットの保存中にエラーが発生しました: {e}")
        
        # 2. 個別の特徴量ごとにプロット
        for feature_name in valid_features:
            try:
                fig, ax = self.plot_single_feature(
                    feature_name, target_idx=target_idx,
                    grid_resolution=grid_resolution
                )
                
                # ファイル名から無効な文字を削除
                clean_name = feature_name.replace(' ', '_').replace('/', '_').replace('\\', '_')
                clean_name = ''.join(c for c in clean_name if c.isalnum() or c in '_-.')
                
                filename = os.path.join(output_dir, f"pdp_{clean_name}_{target_name}.{format}")
                fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(filename)
                print(f"保存しました: {filename}")
            except Exception as e:
                print(f"特徴量 '{feature_name}' のプロット保存中にエラーが発生しました: {e}")
            
        # 3. 上位2つの特徴量の相互作用プロット
        if len(valid_features) >= 2:
            try:
                fig, ax = self.plot_feature_interaction(
                    valid_features[:2], target_idx=target_idx,
                    grid_resolution=min(grid_resolution, 50)  # 2Dプロットは解像度を下げる
                )
                
                clean_name1 = valid_features[0].replace(' ', '_').replace('/', '_').replace('\\', '_')
                clean_name1 = ''.join(c for c in clean_name1 if c.isalnum() or c in '_-.')
                
                clean_name2 = valid_features[1].replace(' ', '_').replace('/', '_').replace('\\', '_')
                clean_name2 = ''.join(c for c in clean_name2 if c.isalnum() or c in '_-.')
                
                filename = os.path.join(output_dir, f"pdp_interaction_{clean_name1}_{clean_name2}_{target_name}.{format}")
                fig.savefig(filename, dpi=dpi, bbox_inches='tight')
                plt.close(fig)
                saved_files.append(filename)
                print(f"保存しました: {filename}")
            except Exception as e:
                print(f"特徴量相互作用プロットの保存中にエラーが発生しました: {e}")
            
        print(f"{len(saved_files)}個のプロットを {output_dir} ディレクトリに保存しました")
        return saved_files


# 使用例
def create_partial_dependence_plots(self, results):
    """
    結果オブジェクトから部分依存プロットを作成
    上位5つの特徴量の1次元部分依存プロットと、上位3特徴量の組み合わせで2次元部分依存プロットも作成
    
    Parameters:
    -----------
    results : dict
        モデル学習の結果を含む辞書
    """
    try:
        # 必要なデータの取得
        model = results.get('model')
        X = results.get('X_test')
        feature_importance = results.get('feature_importance')
        
        if model is None or X is None or feature_importance is None:
            print("警告: 部分依存プロットの作成に必要なデータが不足しています")
            return
        
        # UndersamplingBaggingModelの場合は内部モデルを抽出
        pdp_model = model
        if hasattr(model, 'base_models') and len(model.base_models) > 0:
            # 最初のベースモデルを使用
            pdp_model = model.base_models[0]
            print(f"アンサンブルモデルの最初のベースモデルを部分依存プロット用に使用します: {type(pdp_model).__name__}")
        
        # モデルが分類器として学習されているか確認
        if not hasattr(pdp_model, 'predict') or not hasattr(pdp_model, 'predict_proba'):
            print("警告: モデルが分類器として学習されていないか、必要な予測メソッドがありません")
            return
        
        # モデルが適切にfitされているか確認する追加の検証
        try:
            # 小さなサンプルでテスト予測を実行
            test_sample = X.iloc[:1]
            _ = pdp_model.predict_proba(test_sample)
        except Exception as e:
            print(f"エラー: モデルがフィットされていないか、予測できません: {e}")
            return
            
        # PartialDependencePlotterクラスをインポート
        from models.partial_dependence_plotter_kaikidev_ale import PartialDependencePlotter
        
        # 上位6個の特徴量を選択 (1次元プロット用)
        top_features_1d = feature_importance.head(6)['feature'].tolist()
        
        # 特徴量のインデックスを取得
        feature_indices_1d = [list(X.columns).index(feat) for feat in top_features_1d if feat in X.columns]
        
        if not feature_indices_1d:
            print("警告: 有効な特徴量インデックスが見つかりません")
            return
        
        # 分類問題用のターゲット名を設定
        target_names = {0: 'intact', 1: 'mci'}
        
        # 1. 個別の特徴量に対する部分依存プロット (上位6つの特徴量)
        print(f"\n1次元部分依存プロットを作成します (上位{len(feature_indices_1d)}特徴量)...")
        try:
            # 個別特徴量の部分依存プロット - 修正したモデルを使用
            plotter = PartialDependencePlotter(model=pdp_model, features=X, target_names=target_names)
            fig = plotter.plot_multiple_features(feature_indices_1d, target_idx=1, n_cols=3)
            
            if self.output_dir:
                fig.savefig(os.path.join(self.output_dir, 'partial_dependence_1d.png'), dpi=300)
            plt.show()
        except Exception as e:
            print(f"1次元部分依存プロットの作成中にエラーが発生しました: {e}")
            import traceback
            traceback.print_exc()
        
        # 2. 上位3つの特徴量の全ての組み合わせに対する相互作用プロット
        print("\n2次元部分依存プロットを作成します...")
        
        # 上位3個の特徴量を選択 (2次元プロット用)
        top_features_2d = feature_importance.head(3)['feature'].tolist()
        
        # 有効な特徴量だけを抽出
        valid_top_features = [feat for feat in top_features_2d if feat in X.columns]
        
        if len(valid_top_features) < 2:
            print("警告: 2次元部分依存プロットを作成するには少なくとも2つの特徴量が必要です")
            return
        
        # 全ての2つの組み合わせを生成
        from itertools import combinations
        feature_pairs = list(combinations(valid_top_features, 2))
        
        print(f"上位3つの特徴量の組み合わせに対する2次元部分依存プロット({len(feature_pairs)}組)を作成します...")
        
        # プロットを保存するディレクトリ
        if self.output_dir:
            interaction_dir = os.path.join(self.output_dir, 'interactions')
            if not os.path.exists(interaction_dir):
                os.makedirs(interaction_dir)
        else:
            interaction_dir = None
        
        # カラーマップの設定 - 紫から青、緑、黄色へのグラデーション
        from matplotlib.colors import LinearSegmentedColormap
        
        # カスタムカラーマップの定義
        colors = ['#663399', '#336699', '#339999', '#66cc99', '#99cc66', '#cccc33']
        custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', colors, N=256)
        
        # 各組み合わせについてプロット
        for i, (feat1, feat2) in enumerate(feature_pairs):
            try:
                print(f"特徴量の組み合わせ {i+1}/{len(feature_pairs)}: {feat1} と {feat2}")
                
                # contour_kwからlevelsキーワードを削除し、PartialDependenceDisplayが内部で設定するようにする
                contour_kw = {
                    'cmap': custom_cmap,  # カスタムカラーマップ
                    'alpha': 1.0,         # 不透明度を最大に
                    'linewidths': 0.5,    # 等高線の線の太さ
                    'linestyles': '-',    # 等高線の線のスタイル
                    'antialiased': True   # アンチエイリアス
                }
                
                # 特徴量相互作用プロット作成 - 修正したモデルを使用
                plotter = PartialDependencePlotter(model=pdp_model, features=X, target_names=target_names)
                fig, ax = plotter.plot_feature_interaction(
                    [feat1, feat2],       # 直接特徴量名を渡す
                    target_idx=1,         # ターゲットクラスのインデックス (通常MCIクラスは1)
                    figsize=(10, 8),
                    grid_resolution=100,  # 解像度を上げる
                    contour_kw=contour_kw
                )
                
                # 等高線ラベルのフォントサイズ調整
                try:
                    if hasattr(ax, 'collections') and len(ax.collections) > 0:
                        plt.clabel(
                            ax.collections[0], 
                            inline=True, 
                            fontsize=9, 
                            fmt='%.2f'
                        )
                except Exception as cl_err:
                    print(f"等高線ラベルの設定中にエラーが発生しました（スキップします）: {cl_err}")
                
                # タイトルにランクを追加
                rank1 = top_features_2d.index(feat1) + 1
                rank2 = top_features_2d.index(feat2) + 1
                ax.set_title(f"特徴量相互作用: {feat1} (ランク{rank1}) vs {feat2} (ランク{rank2})\nターゲット: MCI", fontsize=12)
                
                # グリッド線を追加
                ax.grid(True, linestyle='--', alpha=0.3)
                
                # プロットを保存
                if interaction_dir:
                    clean_name1 = feat1.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    clean_name2 = feat2.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    clean_name1 = ''.join(c for c in clean_name1 if c.isalnum() or c in '_-.')
                    clean_name2 = ''.join(c for c in clean_name2 if c.isalnum() or c in '_-.')
                    
                    filename = os.path.join(interaction_dir, f"pdp_interaction_{rank1}_{clean_name1}_vs_{rank2}_{clean_name2}.png")
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"保存しました: {filename}")
                
                plt.show()
                plt.close(fig)
                
            except Exception as e:
                print(f"特徴量 '{feat1}' と '{feat2}' の相互作用プロット作成中にエラーが発生しました: {e}")
                import traceback
                traceback.print_exc()
        
        print("すべての2次元部分依存プロットの作成が完了しました")
        
    except Exception as e:
        print(f"部分依存プロットの作成中にエラーが発生しました: {e}")
        import traceback
        traceback.print_exc()