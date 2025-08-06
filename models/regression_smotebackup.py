import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors, KernelDensity
from sklearn.preprocessing import KBinsDiscretizer
from imblearn.over_sampling import SMOTE


class RegressionSMOTE:
    """
    回帰問題用のSMOTE実装クラス
    
    このクラスは回帰問題において、ターゲット値の分布の偏りを補正するために
    合成サンプルを生成します。3つの手法（binning、density、outliers）を提供します。
    """
    
    def __init__(self, method='density', k_neighbors=5, random_state=42):
        """
        回帰用SMOTEクラスの初期化
        
        Parameters:
        -----------
        method : str, default='density'
            使用するSMOTE手法
            - 'binning': ターゲット値を離散化してSMOTEを適用
            - 'density': 密度の低い領域で合成データを生成
            - 'outliers': 外れ値周辺でオーバーサンプリング
        k_neighbors : int, default=5
            近傍探索で使用する近傍数
        random_state : int, default=42
            乱数シード（再現性のため）
        """
        self.method = method
        self.k_neighbors = k_neighbors
        self.random_state = random_state
        self.supported_methods = ['binning', 'density', 'outliers']
        
        if self.method not in self.supported_methods:
            raise ValueError(f"Unsupported method: {self.method}. Supported methods: {self.supported_methods}")
    
    def fit_resample(self, X, y, **kwargs):
        """
        データをリサンプリングする
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            特徴量データ
        y : array-like or pandas.Series
            ターゲット値
        **kwargs : dict
            各手法固有のパラメータ
            - binning: sampling_strategy, n_bins (default=10)
            - density: density_threshold (default=0.3)
            - outliers: outlier_threshold (default=0.15)
        
        Returns:
        --------
        X_resampled : numpy.ndarray
            リサンプル後の特徴量
        y_resampled : numpy.ndarray
            リサンプル後のターゲット値
        """
        # 入力データをnumpy配列に変換
        X = np.array(X)
        y = np.array(y)
        
        # 手法に応じてリサンプリングを実行
        if self.method == 'binning':
            return self._smote_binning(X, y, **kwargs)
        elif self.method == 'density':
            return self._smote_density(X, y, **kwargs)
        elif self.method == 'outliers':
            return self._smote_outliers(X, y, **kwargs)
    
    def _smote_binning(self, X, y, sampling_strategy='auto', n_bins=10):
        """
        ターゲット値を離散化してSMOTEを適用する手法
        
        ターゲット値を等頻度でビンに分割し、各ビンを「クラス」として扱って
        従来のSMOTEを適用します。
        """
        # ターゲット値を離散化
        discretizer = KBinsDiscretizer(n_bins=n_bins, encode='ordinal', strategy='quantile')
        y_binned = discretizer.fit_transform(y.reshape(-1, 1)).ravel()
        
        # SMOTEを適用
        smote = SMOTE(k_neighbors=self.k_neighbors, sampling_strategy=sampling_strategy, 
                      random_state=self.random_state)
        X_resampled, y_binned_resampled = smote.fit_resample(X, y_binned)
        
        # 合成されたサンプルのターゲット値を推定
        # 近傍探索器を元のデータで初期化
        nbrs = NearestNeighbors(n_neighbors=self.k_neighbors).fit(X)
        
        y_resampled = np.zeros(len(X_resampled))
        y_resampled[:len(y)] = y  # 元のデータはそのまま
        
        # 合成されたサンプルのターゲット値を近傍の平均で推定
        for i in range(len(y), len(X_resampled)):
            distances, indices = nbrs.kneighbors(X_resampled[i].reshape(1, -1))
            y_resampled[i] = np.mean(y[indices[0]])
        
        return X_resampled, y_resampled
    
    def _smote_density(self, X, y, density_threshold=0.3):
        """
        ターゲット値の密度が低い領域でSMOTEを適用する手法
        
        ターゲット値の確率密度を推定し、密度が低い（レアな）サンプル周辺で
        合成データを生成します。
        """
        np.random.seed(self.random_state)
        
        # ターゲット値の密度を推定
        kde = KernelDensity(kernel='gaussian', bandwidth=0.2)
        kde.fit(y.reshape(-1, 1))
        log_density = kde.score_samples(y.reshape(-1, 1))
        density = np.exp(log_density)
        
        # 密度が低い（レアな）サンプルを特定
        low_density_mask = density < np.percentile(density, density_threshold * 100)
        rare_indices = np.where(low_density_mask)[0]
        
        if len(rare_indices) == 0:
            print("密度の低いサンプルが見つかりませんでした。")
            return X, y
        
        # レアサンプルに対してSMOTEライクな合成データ生成
        X_rare = X[rare_indices]
        y_rare = y[rare_indices]
        
        # 近傍探索器を初期化
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(X_rare))).fit(X_rare)
        
        X_synthetic = []
        y_synthetic = []
        
        # 各レアサンプルについて合成データを生成
        for i in range(len(X_rare)):
            n_synthetic = np.random.randint(1, 4)  # 1-3個の合成データを生成
            
            # 近傍を取得（自分自身を除く）
            distances, indices = nbrs.kneighbors(X_rare[i].reshape(1, -1))
            neighbor_indices = indices[0][1:]  # 自分自身を除く
            
            for _ in range(n_synthetic):
                if len(neighbor_indices) > 0:
                    # ランダムに近傍を選択
                    neighbor_idx = np.random.choice(neighbor_indices)
                    
                    # 特徴量の補間
                    alpha = np.random.random()
                    x_synthetic = X_rare[i] + alpha * (X_rare[neighbor_idx] - X_rare[i])
                    
                    # ターゲット値の補間
                    y_synthetic_val = y_rare[i] + alpha * (y_rare[neighbor_idx] - y_rare[i])
                    
                    X_synthetic.append(x_synthetic)
                    y_synthetic.append(y_synthetic_val)
        
        # 元のデータと合成データを結合
        if X_synthetic:
            X_resampled = np.vstack([X, np.array(X_synthetic)])
            y_resampled = np.concatenate([y, np.array(y_synthetic)])
        else:
            X_resampled = X
            y_resampled = y
        
        return X_resampled, y_resampled
    
    def _smote_outliers(self, X, y, outlier_threshold=0.15):
        """
        外れ値周辺でSMOTEを適用する手法
        
        ターゲット値の上位・下位のパーセンタイルを外れ値とみなし、
        それらの周辺で合成データを生成します。
        """
        np.random.seed(self.random_state)
        
        # 外れ値を特定（上位・下位のパーセンタイル）
        lower_threshold = np.percentile(y, outlier_threshold * 100)
        upper_threshold = np.percentile(y, (1 - outlier_threshold) * 100)
        
        outlier_mask = (y <= lower_threshold) | (y >= upper_threshold)
        outlier_indices = np.where(outlier_mask)[0]
        
        if len(outlier_indices) == 0:
            print("外れ値が見つかりませんでした。")
            return X, y
        
        X_outliers = X[outlier_indices]
        y_outliers = y[outlier_indices]
        
        # 全データから近傍を探索
        nbrs = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(X))).fit(X)
        
        X_synthetic = []
        y_synthetic = []
        
        # 各外れ値について合成データを生成
        for i in range(len(X_outliers)):
            n_synthetic = np.random.randint(2, 5)  # 外れ値なので多めに生成
            
            # 近傍を取得
            distances, indices = nbrs.kneighbors(X_outliers[i].reshape(1, -1))
            neighbor_indices = indices[0]
            
            for _ in range(n_synthetic):
                # ランダムに近傍を選択（自分自身以外）
                valid_neighbors = [idx for idx in neighbor_indices if idx != outlier_indices[i]]
                if not valid_neighbors:
                    continue
                    
                neighbor_idx = np.random.choice(valid_neighbors)
                
                # 特徴量の補間（外れ値の方向にバイアス）
                alpha = np.random.beta(0.3, 0.7)  # 外れ値寄りの補間
                x_synthetic = X_outliers[i] + alpha * (X[neighbor_idx] - X_outliers[i])
                
                # ターゲット値の補間
                y_synthetic_val = y_outliers[i] + alpha * (y[neighbor_idx] - y_outliers[i])
                
                X_synthetic.append(x_synthetic)
                y_synthetic.append(y_synthetic_val)
        
        # 元のデータと合成データを結合
        if X_synthetic:
            X_resampled = np.vstack([X, np.array(X_synthetic)])
            y_resampled = np.concatenate([y, np.array(y_synthetic)])
        else:
            X_resampled = X
            y_resampled = y
        
        return X_resampled, y_resampled
    
    def get_info(self):
        """
        現在の設定情報を取得
        
        Returns:
        --------
        dict
            設定情報の辞書
        """
        return {
            'method': self.method,
            'k_neighbors': self.k_neighbors,
            'random_state': self.random_state,
            'supported_methods': self.supported_methods
        }


def visualize_smote_effect(X_original, y_original, X_resampled, y_resampled, feature_idx=0):
    """
    SMOTE適用前後の効果を可視化する関数
    
    Parameters:
    -----------
    X_original : array-like
        元の特徴量データ
    y_original : array-like
        元のターゲット値
    X_resampled : array-like
        リサンプル後の特徴量データ
    y_resampled : array-like
        リサンプル後のターゲット値
    feature_idx : int, default=0
        可視化する特徴量のインデックス
    """
    import matplotlib.pyplot as plt
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # 元のデータのプロット
    ax1.scatter(X_original[:, feature_idx], y_original, alpha=0.6, label='元のデータ')
    ax1.set_xlabel(f'特徴量 {feature_idx}')
    ax1.set_ylabel('ターゲット値')
    ax1.set_title('SMOTE適用前')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # リサンプル後のデータのプロット
    n_original = len(X_original)
    ax2.scatter(X_resampled[:n_original, feature_idx], y_resampled[:n_original], 
                alpha=0.6, label='元のデータ', color='blue')
    ax2.scatter(X_resampled[n_original:, feature_idx], y_resampled[n_original:], 
                alpha=0.6, label='合成データ', color='red')
    ax2.set_xlabel(f'特徴量 {feature_idx}')
    ax2.set_ylabel('ターゲット値')
    ax2.set_title('SMOTE適用後')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # 統計情報の表示
    print("\n=== SMOTE適用の効果 ===")
    print(f"元のデータ数: {len(X_original)}")
    print(f"リサンプル後のデータ数: {len(X_resampled)}")
    print(f"追加された合成データ数: {len(X_resampled) - len(X_original)}")
    print(f"データ増加率: {((len(X_resampled) - len(X_original)) / len(X_original) * 100):.2f}%")