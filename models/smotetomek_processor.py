"""
smotetomek_processor.py

SMOTETomekを使用したデータバランシング処理クラス
独立したモジュールとして設計されており、他のプロジェクトでも再利用可能
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
    from imblearn.combine import SMOTETomek
    from imblearn.over_sampling import SMOTE
    from imblearn.under_sampling import TomekLinks
    IMBALANCED_LEARN_AVAILABLE = True
except ImportError:
    IMBALANCED_LEARN_AVAILABLE = False
    warnings.warn("imbalanced-learnライブラリが見つかりません。pip install imbalanced-learnでインストールしてください。")


class SMOTETomekProcessor:
    """
    SMOTETomekを使用したデータバランシング処理クラス
    
    SMOTEとTomek Linksを組み合わせて、クラス不均衡データの改善と
    境界付近のノイズ除去を同時に行います。
    
    Attributes:
    -----------
    smote_sampling_strategy : str or dict
        SMOTEのサンプリング戦略
    smote_k_neighbors : int
        SMOTEで使用する近傍数
    tomek_sampling_strategy : str or dict
        Tomek Linksのサンプリング戦略
    random_state : int
        乱数シード
    smote_tomek : SMOTETomek
        SMOTETomekインスタンス
    statistics : dict
        処理統計情報
    """
    
    def __init__(self, smote_sampling_strategy='auto', smote_k_neighbors=5, 
                 tomek_sampling_strategy='auto', random_state=42):
        """
        SMOTETomekProcessorの初期化
        
        Parameters:
        -----------
        smote_sampling_strategy : str or dict, default='auto'
            SMOTEのサンプリング戦略
            - 'auto': 自動バランシング
            - 'minority': 少数クラスのみ
            - dict: {class_label: n_samples}の形式
        smote_k_neighbors : int, default=5
            SMOTEで使用する近傍数
        tomek_sampling_strategy : str or dict, default='auto'
            Tomek Linksのサンプリング戦略
            - 'auto': 自動
            - 'majority': 多数クラスのみ
            - 'all': 全クラス
        random_state : int, default=42
            乱数シード
        """
        if not IMBALANCED_LEARN_AVAILABLE:
            raise ImportError("imbalanced-learnライブラリが必要です。pip install imbalanced-learnでインストールしてください。")
        
        self.smote_sampling_strategy = smote_sampling_strategy
        self.smote_k_neighbors = smote_k_neighbors
        self.tomek_sampling_strategy = tomek_sampling_strategy
        self.random_state = random_state
        
        self.smote_tomek = None
        self.statistics = {}
        
    def create_smotetomek(self):
        """SMOTETomekインスタンスを作成"""
        # SMOTEとTomek Linksを個別に設定
        smote = SMOTE(
            sampling_strategy=self.smote_sampling_strategy,
            k_neighbors=self.smote_k_neighbors,
            random_state=self.random_state
        )
        
        tomek = TomekLinks(
            sampling_strategy=self.tomek_sampling_strategy
        )
        
        # SMOTETomekの作成
        self.smote_tomek = SMOTETomek(
            smote=smote,
            tomek=tomek,
            random_state=self.random_state
        )
        
        return self.smote_tomek
    
    def fit_resample(self, X, y, verbose=True):
        """
        SMOTETomekを適用してデータをリサンプリング
        
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
        if self.smote_tomek is None:
            self.create_smotetomek()
        
        # 元のデータサイズを記録
        original_size = len(X)
        original_class_counts = pd.Series(y).value_counts().sort_index()
        
        if verbose:
            print(f"SMOTETomek適用前:")
            print(f"  総サンプル数: {original_size}")
            print(f"  クラス分布: {dict(original_class_counts)}")
        
        # SMOTETomekの適用
        X_resampled, y_resampled = self.smote_tomek.fit_resample(X, y)
        
        # 結果の統計
        resampled_size = len(X_resampled)
        resampled_class_counts = pd.Series(y_resampled).value_counts().sort_index()
        
        if verbose:
            print(f"\nSMOTETomek適用後:")
            print(f"  総サンプル数: {resampled_size} (変化: {resampled_size - original_size:+d})")
            print(f"  クラス分布: {dict(resampled_class_counts)}")
        
        # 統計情報を保存
        self.statistics = {
            'original_size': original_size,
            'resampled_size': resampled_size,
            'size_change': resampled_size - original_size,
            'original_class_counts': original_class_counts.to_dict(),
            'resampled_class_counts': resampled_class_counts.to_dict(),
            'smote_settings': {
                'sampling_strategy': self.smote_sampling_strategy,
                'k_neighbors': self.smote_k_neighbors
            },
            'tomek_settings': {
                'sampling_strategy': self.tomek_sampling_strategy
            }
        }
        
        return X_resampled, y_resampled, self.statistics
    
    def get_balance_improvement(self):
        """
        クラスバランスの改善度を計算
        
        Returns:
        --------
        dict
            改善度の詳細情報
        """
        if not self.statistics:
            raise ValueError("fit_resampleを先に実行してください")
        
        try:
            original_counts = pd.Series(self.statistics['original_class_counts'])
            resampled_counts = pd.Series(self.statistics['resampled_class_counts'])
            
            # インバランス比率の計算（最大クラス数 / 最小クラス数）
            original_imbalance = original_counts.max() / original_counts.min()
            resampled_imbalance = resampled_counts.max() / resampled_counts.min()
            
            improvement = (original_imbalance - resampled_imbalance) / original_imbalance
            
            return {
                'original_imbalance_ratio': original_imbalance,
                'resampled_imbalance_ratio': resampled_imbalance,
                'improvement_ratio': improvement,
                'improvement_percentage': improvement * 100,
                'is_improved': improvement > 0
            }
        except Exception as e:
            return {'error': str(e)}
    
    def visualize_resampling_effect(self, X_original, y_original, X_resampled, y_resampled, 
                                  output_dir=None, figsize=(15, 12)):
        """
        リサンプリング効果を可視化
        
        Parameters:
        -----------
        X_original : array-like or pandas.DataFrame
            元の特徴量データ
        y_original : array-like or pandas.Series
            元のターゲットデータ
        X_resampled : array-like or pandas.DataFrame
            リサンプリング後の特徴量データ
        y_resampled : array-like or pandas.Series
            リサンプリング後のターゲットデータ
        output_dir : str, optional
            出力ディレクトリ
        figsize : tuple, default=(15, 12)
            図のサイズ
            
        Returns:
        --------
        matplotlib.figure.Figure
            作成された図
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        
        # 1-1. クラス分布の棒グラフ
        original_counts = pd.Series(y_original).value_counts().sort_index()
        resampled_counts = pd.Series(y_resampled).value_counts().sort_index()
        
        x = np.arange(len(original_counts))
        width = 0.35
        
        axes[0, 0].bar(x - width/2, original_counts.values, width, label='元データ', alpha=0.8)
        axes[0, 0].bar(x + width/2, resampled_counts.values, width, label='SMOTETomek後', alpha=0.8)
        axes[0, 0].set_xlabel('クラス')
        axes[0, 0].set_ylabel('サンプル数')
        axes[0, 0].set_title('クラス分布の変化')
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(original_counts.index)
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # 1-2. クラス比率の円グラフ（元データ）
        axes[0, 1].pie(original_counts.values, labels=original_counts.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[0, 1].set_title('元データのクラス比率')
        
        # 1-3. クラス比率の円グラフ（SMOTETomek後）
        axes[1, 0].pie(resampled_counts.values, labels=resampled_counts.index, 
                      autopct='%1.1f%%', startangle=90)
        axes[1, 0].set_title('SMOTETomek後のクラス比率')
        
        # 1-4. データサイズの変化
        size_data = ['元データ', 'SMOTETomek後']
        size_values = [len(X_original), len(X_resampled)]
        
        axes[1, 1].bar(size_data, size_values, color=['skyblue', 'lightcoral'], alpha=0.8)
        axes[1, 1].set_ylabel('サンプル数')
        axes[1, 1].set_title('データサイズの変化')
        for i, v in enumerate(size_values):
            axes[1, 1].text(i, v + max(size_values)*0.01, str(v), ha='center', va='bottom')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle('SMOTETomek効果の分析', fontsize=16)
        plt.tight_layout()
        
        # 保存処理
        if output_dir:
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(output_dir, f'smotetomek_analysis_{timestamp}.png')
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"SMOTETomek分析プロットを保存: {save_path}")
        
        return fig
    
    def create_comparison_report(self, output_dir=None):
        """
        SMOTETomek効果の詳細レポートを作成
        
        Parameters:
        -----------
        output_dir : str, optional
            出力ディレクトリ
            
        Returns:
        --------
        str
            レポートファイルのパス（output_dirが指定された場合）
        """
        if not self.statistics:
            raise ValueError("fit_resampleを先に実行してください")
        
        if not output_dir:
            # コンソール出力のみ
            self._print_report()
            return None
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        report_path = os.path.join(output_dir, 'smotetomek_report.txt')
        
        with open(report_path, 'w', encoding='utf-8') as f:
            self._write_report(f)
        
        print(f"SMOTETomekレポートを保存: {report_path}")
        return report_path
    
    def _print_report(self):
        """レポートをコンソールに出力"""
        print("=" * 50)
        print("SMOTETomek効果レポート")
        print("=" * 50)
        print(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        self._write_report_content(print)
    
    def _write_report(self, file_handle):
        """レポートをファイルに書き込み"""
        file_handle.write("=" * 50 + "\n")
        file_handle.write("SMOTETomek効果レポート\n")
        file_handle.write("=" * 50 + "\n")
        file_handle.write(f"作成日時: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        def write_func(text):
            file_handle.write(text + "\n")
        
        self._write_report_content(write_func)
    
    def _write_report_content(self, write_func):
        """レポート内容を書き込み"""
        stats = self.statistics
        
        # 1. 設定情報
        write_func("1. SMOTETomek設定")
        write_func("-" * 30)
        write_func(f"   SMOTE設定:")
        write_func(f"     サンプリング戦略: {stats['smote_settings']['sampling_strategy']}")
        write_func(f"     近傍数: {stats['smote_settings']['k_neighbors']}")
        write_func(f"   Tomek Links設定:")
        write_func(f"     サンプリング戦略: {stats['tomek_settings']['sampling_strategy']}")
        write_func("")
        
        # 2. データサイズの変化
        write_func("2. データサイズの変化")
        write_func("-" * 30)
        write_func(f"   元データ: {stats['original_size']} サンプル")
        write_func(f"   処理後: {stats['resampled_size']} サンプル")
        write_func(f"   変化: {stats['size_change']:+d} サンプル")
        change_ratio = stats['size_change'] / stats['original_size'] * 100
        write_func(f"   変化率: {change_ratio:+.2f}%")
        write_func("")
        
        # 3. クラス分布の変化
        write_func("3. クラス分布の変化")
        write_func("-" * 30)
        write_func("   元データ:")
        for cls, count in stats['original_class_counts'].items():
            write_func(f"     クラス {cls}: {count} サンプル")
        
        write_func("   処理後:")
        for cls, count in stats['resampled_class_counts'].items():
            write_func(f"     クラス {cls}: {count} サンプル")
        write_func("")
        
        # 4. バランス改善度
        write_func("4. クラスバランス改善度")
        write_func("-" * 30)
        balance_improvement = self.get_balance_improvement()
        if 'error' not in balance_improvement:
            write_func(f"   元データのインバランス比率: {balance_improvement['original_imbalance_ratio']:.2f}")
            write_func(f"   処理後のインバランス比率: {balance_improvement['resampled_imbalance_ratio']:.2f}")
            write_func(f"   改善度: {balance_improvement['improvement_percentage']:.2f}%")
            write_func(f"   改善有無: {'改善' if balance_improvement['is_improved'] else '悪化'}")
        else:
            write_func(f"   計算エラー: {balance_improvement['error']}")
        write_func("")
        
        # 5. 推奨事項
        write_func("5. 推奨事項")
        write_func("-" * 30)
        if balance_improvement.get('is_improved', False):
            write_func("   ✓ SMOTETomekによりクラスバランスが改善されました")
            write_func("   ✓ 境界クリーニング効果によりモデル性能の向上が期待できます")
            write_func("   → SMOTETomekの使用を推奨します")
        else:
            write_func("   × SMOTETomekによる明確な改善は見られませんでした")
            write_func("   → 元データでの学習または他の手法を検討してください")
        
        write_func("\n" + "=" * 50)
    
    def get_statistics_summary(self):
        """
        統計情報の要約を取得
        
        Returns:
        --------
        dict
            統計情報の要約
        """
        if not self.statistics:
            return {'error': 'fit_resampleを先に実行してください'}
        
        balance_improvement = self.get_balance_improvement()
        
        return {
            'processing_summary': {
                'original_size': self.statistics['original_size'],
                'resampled_size': self.statistics['resampled_size'],
                'size_change': self.statistics['size_change'],
                'change_percentage': (self.statistics['size_change'] / self.statistics['original_size']) * 100
            },
            'class_distribution': {
                'before': self.statistics['original_class_counts'],
                'after': self.statistics['resampled_class_counts']
            },
            'balance_improvement': balance_improvement,
            'settings': {
                'smote': self.statistics['smote_settings'],
                'tomek': self.statistics['tomek_settings']
            },
            'recommendation': 'use_smotetomek' if balance_improvement.get('is_improved', False) else 'use_original_data'
        }
    
    def reset(self):
        """
        プロセッサーをリセット（新しいデータセット用）
        """
        self.smote_tomek = None
        self.statistics = {}
        print("SMOTETomekProcessorがリセットされました")


# 便利関数（クラス外）
def quick_smotetomek_analysis(X, y, output_dir=None, **kwargs):
    """
    SMOTETomekのクイック分析用便利関数
    
    Parameters:
    -----------
    X : array-like or pandas.DataFrame
        特徴量データ
    y : array-like or pandas.Series
        ターゲットデータ
    output_dir : str, optional
        出力ディレクトリ
    **kwargs : dict
        SMOTETomekProcessorのパラメータ
        
    Returns:
    --------
    tuple
        (X_resampled, y_resampled, processor, statistics)
    """
    processor = SMOTETomekProcessor(**kwargs)
    X_resampled, y_resampled, statistics = processor.fit_resample(X, y)
    
    if output_dir:
        processor.visualize_resampling_effect(X, y, X_resampled, y_resampled, output_dir)
        processor.create_comparison_report(output_dir)
    
    return X_resampled, y_resampled, processor, statistics


if __name__ == "__main__":
    print("SMOTETomekProcessor クラス")
    print("使用例:")
    print("""
    from smotetomek_processor import SMOTETomekProcessor
    
    # インスタンス作成
    processor = SMOTETomekProcessor(
        smote_sampling_strategy='auto',
        smote_k_neighbors=5,
        random_state=42
    )
    
    # データ処理
    X_resampled, y_resampled, stats = processor.fit_resample(X_train, y_train)
    
    # 効果の可視化
    processor.visualize_resampling_effect(X_train, y_train, X_resampled, y_resampled)
    
    # レポート作成
    processor.create_comparison_report('output_dir')
    """)