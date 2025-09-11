"""
borderline_smote_processor.py

Borderline SMOTEを使用したデータバランシング処理クラス
境界付近のサンプルに重点を置いた合成サンプル生成
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import os
from datetime import datetime
import warnings

try:
    from imblearn.over_sampling import SMOTE
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    warnings.warn("imbalanced-learnライブラリが見つかりません。pip install imbalanced-learnでインストールしてください。")


class BorderlineSMOTEProcessor:
    """
    Borderline SMOTEを使用したデータバランシング処理クラス
    
    Borderline SMOTEは、クラス境界付近のサンプルに重点を置いて
    合成サンプルを生成する手法です。通常のSMOTEよりも効果的な
    バランシングが期待できます。
    
    Attributes:
    -----------
    sampling_strategy : str or dict
        サンプリング戦略
    k_neighbors : int
        近傍数
    m_neighbors : int
        境界判定用近傍数
    borderline_type : int
        Borderline-SMOTEのタイプ（1または2）
    random_state : int
        乱数シード
    statistics : dict
        処理統計情報
    """
    
    def __init__(self, sampling_strategy='auto', k_neighbors=5, m_neighbors=10, 
                 borderline_type=1, random_state=42):
        """
        BorderlineSMOTEProcessorの初期化
        
        Parameters:
        -----------
        sampling_strategy : str or dict, default='auto'
            サンプリング戦略
            - 'auto': 自動バランシング
            - 'minority': 少数クラスのみ
            - dict: {class_label: n_samples}の形式
        k_neighbors : int, default=5
            SMOTE生成で使用する近傍数
        m_neighbors : int, default=10
            境界判定で使用する近傍数
        borderline_type : int, default=1
            Borderline-SMOTEのタイプ
            - 1: Borderline-SMOTE1（同クラスのみ使用）
            - 2: Borderline-SMOTE2（他クラスも使用）
        random_state : int, default=42
            乱数シード
        """
        self.sampling_strategy = sampling_strategy
        self.k_neighbors = k_neighbors
        self.m_neighbors = m_neighbors
        self.borderline_type = borderline_type
        self.random_state = random_state
        
        self.statistics = {}
        self.borderline_samples = {}
        
        # 乱数生成器の初期化
        self.random_generator = np.random.RandomState(random_state)
        
    def _identify_borderline_samples(self, X, y, minority_class):
        """
        境界サンプルを識別する
        
        Parameters:
        -----------
        X : array-like
            特徴量データ
        y : array-like
            ターゲットデータ
        minority_class : int
            少数クラスのラベル
            
        Returns:
        --------
        tuple
            (borderline_indices, danger_indices, safe_indices)
        """
        # 少数クラスのサンプルを取得
        minority_mask = y == minority_class
        minority_X = X[minority_mask]
        minority_indices = np.where(minority_mask)[0]
        
        if len(minority_X) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # 全体データでの近傍探索
        nn = NearestNeighbors(n_neighbors=self.m_neighbors + 1)
        nn.fit(X)
        
        borderline_indices = []
        danger_indices = []
        safe_indices = []
        
        for i, sample_idx in enumerate(minority_indices):
            # m個の最近傍を取得（自分自身を除く）
            distances, indices = nn.kneighbors(X[sample_idx].reshape(1, -1))
            neighbor_indices = indices[0][1:]  # 自分自身を除く
            neighbor_labels = y[neighbor_indices]
            
            # 近傍中の異なるクラスの数を計算
            different_class_count = np.sum(neighbor_labels != minority_class)
            
            if different_class_count == self.m_neighbors:
                # 全ての近傍が異なるクラス → ノイズ（使用しない）
                continue
            elif different_class_count >= self.m_neighbors // 2:
                # 半数以上が異なるクラス → 境界サンプル
                borderline_indices.append(sample_idx)
            elif different_class_count > 0:
                # 一部が異なるクラス → 危険サンプル
                danger_indices.append(sample_idx)
            else:
                # 全て同じクラス → 安全サンプル
                safe_indices.append(sample_idx)
        
        return (np.array(borderline_indices), 
                np.array(danger_indices), 
                np.array(safe_indices))
    
    def _generate_samples_from_borderline(self, X, y, borderline_indices, 
                                        minority_class, n_samples):
        """
        境界サンプルから合成サンプルを生成
        
        Parameters:
        -----------
        X : array-like
            特徴量データ
        y : array-like
            ターゲットデータ
        borderline_indices : array-like
            境界サンプルのインデックス
        minority_class : int
            少数クラスのラベル
        n_samples : int
            生成するサンプル数
            
        Returns:
        --------
        tuple
            (generated_X, generated_y)
        """
        if len(borderline_indices) == 0:
            # 境界サンプルがない場合は通常のSMOTE
            return self._generate_samples_smote(X, y, minority_class, n_samples)
        
        # 境界サンプルのデータ
        borderline_X = X[borderline_indices]
        
        # 少数クラスのサンプルを取得
        minority_mask = y == minority_class
        minority_X = X[minority_mask]
        
        # k近傍探索の準備
        if self.borderline_type == 1:
            # Borderline-SMOTE1: 同じクラスのサンプルのみ使用
            search_X = minority_X
        else:
            # Borderline-SMOTE2: 全サンプルを使用
            search_X = X
        
        nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(search_X)))
        nn.fit(search_X)
        
        generated_samples = []
        
        for _ in range(n_samples):
            # ランダムに境界サンプルを選択
            base_idx = self.random_generator.choice(len(borderline_indices))
            base_sample = borderline_X[base_idx]
            
            # 近傍を探索
            distances, indices = nn.kneighbors(base_sample.reshape(1, -1))
            neighbor_indices = indices[0][1:]  # 自分自身を除く
            
            if len(neighbor_indices) == 0:
                # 近傍がない場合はベースサンプルをそのまま使用
                generated_samples.append(base_sample)
                continue
            
            # ランダムに近傍を選択
            neighbor_idx = self.random_generator.choice(neighbor_indices)
            neighbor_sample = search_X[neighbor_idx]
            
            # Borderline-SMOTE2の場合、異なるクラスの近傍は使用しない
            if self.borderline_type == 2:
                # search_Xが全データの場合、ラベルを確認
                if len(search_X) == len(X):
                    neighbor_original_idx = neighbor_idx
                    if y[neighbor_original_idx] != minority_class:
                        # 異なるクラスの場合は同じクラスの近傍を探す
                        same_class_indices = []
                        for idx in neighbor_indices:
                            if y[idx] == minority_class:
                                same_class_indices.append(idx)
                        
                        if same_class_indices:
                            neighbor_idx = self.random_generator.choice(same_class_indices)
                            neighbor_sample = search_X[neighbor_idx]
                        else:
                            # 同じクラスの近傍がない場合はスキップ
                            generated_samples.append(base_sample)
                            continue
            
            # 線形補間で新しいサンプルを生成
            diff = neighbor_sample - base_sample
            gap = self.random_generator.random()
            synthetic_sample = base_sample + gap * diff
            
            generated_samples.append(synthetic_sample)
        
        generated_X = np.array(generated_samples)
        generated_y = np.full(n_samples, minority_class)
        
        return generated_X, generated_y
    
    def _generate_samples_smote(self, X, y, minority_class, n_samples):
        """
        通常のSMOTEで合成サンプルを生成（フォールバック用）
        
        Parameters:
        -----------
        X : array-like
            特徴量データ
        y : array-like
            ターゲットデータ
        minority_class : int
            少数クラスのラベル
        n_samples : int
            生成するサンプル数
            
        Returns:
        --------
        tuple
            (generated_X, generated_y)
        """
        minority_mask = y == minority_class
        minority_X = X[minority_mask]
        
        if len(minority_X) < 2:
            # サンプルが少なすぎる場合
            return np.array([]), np.array([])
        
        nn = NearestNeighbors(n_neighbors=min(self.k_neighbors + 1, len(minority_X)))
        nn.fit(minority_X)
        
        generated_samples = []
        
        for _ in range(n_samples):
            # ランダムにベースサンプルを選択
            base_idx = self.random_generator.choice(len(minority_X))
            base_sample = minority_X[base_idx]
            
            # 近傍を探索
            distances, indices = nn.kneighbors(base_sample.reshape(1, -1))
            neighbor_indices = indices[0][1:]  # 自分自身を除く
            
            if len(neighbor_indices) == 0:
                generated_samples.append(base_sample)
                continue
            
            # ランダムに近傍を選択
            neighbor_idx = self.random_generator.choice(neighbor_indices)
            neighbor_sample = minority_X[neighbor_idx]
            
            # 線形補間で新しいサンプルを生成
            diff = neighbor_sample - base_sample
            gap = self.random_generator.random()
            synthetic_sample = base_sample + gap * diff
            
            generated_samples.append(synthetic_sample)
        
        generated_X = np.array(generated_samples)
        generated_y = np.full(n_samples, minority_class)
        
        return generated_X, generated_y
    
    def fit_resample(self, X, y, verbose=True):
        """
        Borderline SMOTEを適用してデータをリサンプリング
        
        Parameters:
        -----------
        X : array-like or pandas.DataFrame
            特徴量データ
        y : array-like or pandas.Series
            ターゲットデータ
        verbose : bool, default=True
            詳細な出力を行うか
            
        Returns:
        --------
        tuple
            (X_resampled, y_resampled, statistics)
        """
        # numpyとpandasを明示的にインポート（スコープエラー対策）
        import numpy as np
        import pandas as pd
        
        # データの変換
        X = np.array(X, dtype=float) if not isinstance(X, np.ndarray) else X.astype(float)
        y = np.array(y, dtype=int) if not isinstance(y, np.ndarray) else y.astype(int)
        
        # 元のデータサイズを記録
        original_size = len(X)
        original_class_counts = pd.Series(y).value_counts().sort_index()
        
        if verbose:
            print(f"Borderline-SMOTE{self.borderline_type} 適用前:")
            print(f"  総サンプル数: {original_size}")
            print(f"  クラス分布: {dict(original_class_counts)}")
        
        # 各クラスのサンプル数を取得
        unique_classes, class_counts = np.unique(y, return_counts=True)
        class_count_dict = dict(zip(unique_classes, class_counts))
        
        # サンプリング戦略の決定
        if self.sampling_strategy == 'auto':
            # 最大クラス数に合わせる
            max_samples = max(class_counts)
            target_counts = {cls: max_samples for cls in unique_classes}
        elif self.sampling_strategy == 'minority':
            # 少数クラスのみバランシング
            max_samples = max(class_counts)
            min_samples = min(class_counts)
            target_counts = {}
            for cls in unique_classes:
                if class_count_dict[cls] == min_samples:
                    target_counts[cls] = max_samples
                else:
                    target_counts[cls] = class_count_dict[cls]
        elif isinstance(self.sampling_strategy, dict):
            target_counts = self.sampling_strategy
        else:
            raise ValueError("sampling_strategyは 'auto', 'minority', または辞書である必要があります")
        
        # リサンプリング実行
        X_resampled = X.copy()
        y_resampled = y.copy()
        
        borderline_stats = {}
        
        for class_label in unique_classes:
            current_count = class_count_dict[class_label]
            target_count = target_counts.get(class_label, current_count)
            
            if target_count > current_count:
                n_samples_needed = target_count - current_count
                
                if verbose:
                    print(f"\n  クラス {class_label} の処理:")
                    print(f"    現在: {current_count}, 目標: {target_count}, 生成: {n_samples_needed}")
                
                # 境界サンプルの識別
                borderline_indices, danger_indices, safe_indices = self._identify_borderline_samples(
                    X, y, class_label
                )
                
                borderline_stats[class_label] = {
                    'borderline_samples': len(borderline_indices),
                    'danger_samples': len(danger_indices),
                    'safe_samples': len(safe_indices),
                    'total_original': current_count
                }
                
                if verbose:
                    print(f"    境界サンプル: {len(borderline_indices)}")
                    print(f"    危険サンプル: {len(danger_indices)}")
                    print(f"    安全サンプル: {len(safe_indices)}")
                
                # 合成サンプルの生成
                if len(borderline_indices) > 0:
                    generated_X, generated_y = self._generate_samples_from_borderline(
                        X, y, borderline_indices, class_label, n_samples_needed
                    )
                else:
                    if verbose:
                        print(f"    境界サンプルがないため通常のSMOTEを使用")
                    generated_X, generated_y = self._generate_samples_smote(
                        X, y, class_label, n_samples_needed
                    )
                
                if len(generated_X) > 0:
                    X_resampled = np.vstack([X_resampled, generated_X])
                    y_resampled = np.concatenate([y_resampled, generated_y])
        
        # 結果の統計
        resampled_size = len(X_resampled)
        resampled_class_counts = pd.Series(y_resampled).value_counts().sort_index()
        
        if verbose:
            print(f"\nBorderline-SMOTE{self.borderline_type} 適用後:")
            print(f"  総サンプル数: {resampled_size} (変化: {resampled_size - original_size:+d})")
            print(f"  クラス分布: {dict(resampled_class_counts)}")
        
        # 統計情報を保存
        self.statistics = {
            'original_size': original_size,
            'resampled_size': resampled_size,
            'size_change': resampled_size - original_size,
            'original_class_counts': original_class_counts.to_dict(),
            'resampled_class_counts': resampled_class_counts.to_dict(),
            'borderline_type': self.borderline_type,
            'borderline_stats': borderline_stats,
            'settings': {
                'sampling_strategy': self.sampling_strategy,
                'k_neighbors': self.k_neighbors,
                'm_neighbors': self.m_neighbors,
                'borderline_type': self.borderline_type
            }
        }
        
        self.borderline_samples = borderline_stats
        
        # 返り値の型確認と明示的な変換
        if verbose:
            print(f"DEBUG: 返り値の型確認")
            print(f"  X_resampled type: {type(X_resampled)}")
            print(f"  y_resampled type: {type(y_resampled)}")
        
        # 確実にnumpy配列として返す
        X_resampled = np.array(X_resampled, dtype=float)
        y_resampled = np.array(y_resampled, dtype=int)

        return X_resampled, y_resampled, self.statistics
    
    def visualize_borderline_analysis(self, X_original, y_original, X_resampled, y_resampled,
                                    output_dir=None, figsize=(20, 12)):
        """
        Borderline SMOTE効果とサンプル分布を可視化
        
        Parameters:
        -----------
        X_original : array-like
            元の特徴量データ
        y_original : array-like
            元のターゲットデータ
        X_resampled : array-like
            リサンプリング後の特徴量データ
        y_resampled : array-like
            リサンプリング後のターゲットデータ
        output_dir : str, optional
            出力ディレクトリ
        figsize : tuple, default=(20, 12)
            図のサイズ
            
        Returns:
        --------
        matplotlib.figure.Figure
            作成された図
        """
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        
        # 1. クラス分布の変化
        original_counts = pd.Series(y_original).value_counts().sort_index()
        resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
        
        x = np.arange(len(original_counts))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, original_counts.values, width, 
                      label='元データ', alpha=0.8, color='skyblue')
        axes[0, 0].bar(x + width/2, resampled_counts.values, width, 
                      label=f'Borderline-SMOTE{self.borderline_type}後', alpha=0.8, color='lightcoral')
        axes[0, 0].set_xlabel('クラス')
        axes[0, 0].set_ylabel('サンプル数')
        axes[0, 0].set_title('クラス分布の変化')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(original_counts.index)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 2. 境界サンプル分析
        if self.borderline_samples:
            classes = list(self.borderline_samples.keys())
            borderline_counts = [self.borderline_samples[cls]['borderline_samples'] for cls in classes]
            danger_counts = [self.borderline_samples[cls]['danger_samples'] for cls in classes]
            safe_counts = [self.borderline_samples[cls]['safe_samples'] for cls in classes]
            
            x = np.arange(len(classes))
            width = 0.25
            
            axes[0, 1].bar(x - width, borderline_counts, width, label='境界サンプル', alpha=0.8, color='red')
            axes[0, 1].bar(x, danger_counts, width, label='危険サンプル', alpha=0.8, color='orange')
            axes[0, 1].bar(x + width, safe_counts, width, label='安全サンプル', alpha=0.8, color='green')
            
            axes[0, 1].set_xlabel('クラス')
            axes[0, 1].set_ylabel('サンプル数')
            axes[0, 1].set_title('サンプルタイプ別分布')
            axes[0, 1].set_xticks(x)
            axes[0, 1].set_xticklabels(classes)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
        
        # 3. データサイズの変化
        size_data = ['元データ', f'Borderline-SMOTE{self.borderline_type}後']
        size_values = [len(X_original), len(X_resampled)]
        
        axes[0, 2].bar(size_data, size_values, color=['skyblue', 'lightcoral'], alpha=0.8)
        axes[0, 2].set_ylabel('サンプル数')
        axes[0, 2].set_title('データサイズの変化')
        for i, v in enumerate(size_values):
            axes[0, 2].text(i, v + max(size_values)*0.01, str(v), ha='center', va='bottom')
        axes[0, 2].grid(True, alpha=0.3)
        
        # 4-6. 散布図（最初の2次元で可視化）
        if X_original.shape[1] >= 2:
            # 元データ
            for class_label in np.unique(y_original):
                mask = y_original == class_label
                axes[1, 0].scatter(X_original[mask, 0], X_original[mask, 1], 
                                 label=f'Class {class_label}', alpha=0.6, s=20)
            axes[1, 0].set_title('元データ分布')
            axes[1, 0].set_xlabel('Feature 1')
            axes[1, 0].set_ylabel('Feature 2')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # リサンプリング後
            for class_label in np.unique(y_resampled):
                mask = y_resampled == class_label
                axes[1, 1].scatter(X_resampled[mask, 0], X_resampled[mask, 1], 
                                 label=f'Class {class_label}', alpha=0.6, s=20)
            axes[1, 1].set_title(f'Borderline-SMOTE{self.borderline_type}後分布')
            axes[1, 1].set_xlabel('Feature 1')
            axes[1, 1].set_ylabel('Feature 2')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # 合成サンプルのみ
            synthetic_mask = len(X_original)
            if len(X_resampled) > len(X_original):
                axes[1, 2].scatter(X_resampled[synthetic_mask:, 0], X_resampled[synthetic_mask:, 1], 
                                 c='red', alpha=0.6, s=20, label='合成サンプル')
                # 元の少数クラスも表示
                for class_label in np.unique(y_original):
                    mask = y_original == class_label
                    if np.sum(mask) < len(y_original) * 0.5:  # 少数クラスと推定
                        axes[1, 2].scatter(X_original[mask, 0], X_original[mask, 1], 
                                         alpha=0.3, s=15, label=f'元クラス {class_label}')
                
                axes[1, 2].set_title('合成サンプル分布')
                axes[1, 2].set_xlabel('Feature 1')
                axes[1, 2].set_ylabel('Feature 2')
                axes[1, 2].legend()
                axes[1, 2].grid(True, alpha=0.3)
        
        plt.suptitle(f'Borderline-SMOTE{self.borderline_type} 効果分析', fontsize=16)
        plt.tight_layout()
        
        # 保存処理
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(output_dir, f'borderline_smote{self.borderline_type}_analysis_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Borderline-SMOTE分析プロットを保存: {save_path}")
        
        return fig
    
    def get_statistics_summary(self):
        """統計情報の要約を取得"""
        if not self.statistics:
            return {'error': 'fit_resampleを先に実行してください'}
        
        stats = self.statistics
        
        # バランス改善度計算
        original_counts = pd.Series(stats['original_class_counts'])
        resampled_counts = pd.Series(stats['resampled_class_counts'])
        
        original_imbalance = original_counts.max() / original_counts.min()
        resampled_imbalance = resampled_counts.max() / resampled_counts.min()
        improvement = (original_imbalance - resampled_imbalance) / original_imbalance
        
        return {
            'processing_summary': {
                'borderline_type': stats['borderline_type'],
                'original_size': stats['original_size'],
                'resampled_size': stats['resampled_size'],
                'size_change': stats['size_change'],
                'change_percentage': (stats['size_change'] / stats['original_size']) * 100
            },
            'class_distribution': {
                'before': stats['original_class_counts'],
                'after': stats['resampled_class_counts']
            },
            'balance_improvement': {
                'original_imbalance_ratio': original_imbalance,
                'resampled_imbalance_ratio': resampled_imbalance,
                'improvement_ratio': improvement,
                'improvement_percentage': improvement * 100,
                'is_improved': improvement > 0
            },
            'borderline_analysis': stats['borderline_stats'],
            'settings': stats['settings'],
            'recommendation': 'use_borderline_smote' if improvement > 0 else 'use_original_data'
        }


# 便利関数
def quick_borderline_smote_analysis(X, y, borderline_type=1, output_dir=None, **kwargs):
    """
    Borderline SMOTEのクイック分析用便利関数
    
    Parameters:
    -----------
    X : array-like
        特徴量データ
    y : array-like
        ターゲットデータ
    borderline_type : int, default=1
        Borderline-SMOTEのタイプ
    output_dir : str, optional
        出力ディレクトリ
    **kwargs : dict
        BorderlineSMOTEProcessorのパラメータ
        
    Returns:
    --------
    tuple
        (X_resampled, y_resampled, processor, statistics)
    """
    processor = BorderlineSMOTEProcessor(borderline_type=borderline_type, **kwargs)
    X_resampled, y_resampled, statistics = processor.fit_resample(X, y)
    
    if output_dir:
        processor.visualize_borderline_analysis(X, y, X_resampled, y_resampled, output_dir)
    
    return X_resampled, y_resampled, processor, statistics


if __name__ == "__main__":
    print("BorderlineSMOTEProcessor クラス")
    print("使用例:")
    print("""
    from borderline_smote_processor import BorderlineSMOTEProcessor
    
    # インスタンス作成
    processor = BorderlineSMOTEProcessor(
        sampling_strategy='auto',
        k_neighbors=5,
        m_neighbors=10,
        borderline_type=1,  # 1 or 2
        random_state=42
    )
    
    # データ処理
    X_resampled, y_resampled, stats = processor.fit_resample(X_train, y_train)
    
    # 効果の可視化
    processor.visualize_borderline_analysis(X_train, y_train, X_resampled, y_resampled)
    
    # 統計サマリー
    summary = processor.get_statistics_summary()
    """)