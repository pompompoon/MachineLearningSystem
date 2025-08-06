import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def safe_convert_array(x):
    """安全に配列文字列を数値に変換する関数"""
    if not isinstance(x, str):
        return np.nan
    
    if '...' in x:
        return np.nan
    
    try:
        x = x.strip('[]')
        numbers = []
        for num in x.split():
            try:
                numbers.append(float(num))
            except ValueError:
                continue
        
        if not numbers:
            return np.nan
        
        return np.mean(numbers)
    except:
        return np.nan

def calculate_specificity(y_true, y_pred):
    """特異度（Specificity）を計算する関数"""
    cm = confusion_matrix(y_true, y_pred)
    # 2クラス分類の場合: TN=cm[0,0], FP=cm[0,1], FN=cm[1,0], TP=cm[1,1]
    if cm.shape == (2, 2):
        tn, fp = cm[0, 0], cm[0, 1]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    else:
        # 多クラス分類の場合は特異度の計算が複雑になるため、マクロ平均を使用
        specificities = []
        n_classes = cm.shape[0]
        for i in range(n_classes):
            tn = np.sum(np.delete(np.delete(cm, i, axis=0), i, axis=1))
            fp = np.sum(np.delete(cm, i, axis=0)[:, i])
            specificities.append(tn / (tn + fp) if (tn + fp) > 0 else 0)
        specificity = np.mean(specificities)
    
    return specificity

class CrossValidator:
    """交差検証を行うクラス"""
    
    def __init__(self, n_splits=5, random_state=42):
        """
        初期化関数
        
        Parameters:
        -----------
        n_splits : int, default=5
            交差検証の分割数
        random_state : int, default=42
            乱数シード
        """
        self.n_splits = n_splits
        self.random_state = random_state
        self.cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
        self.results = {}
        
    def run_cross_validation(self, features, target, model_class):
        """
        交差検証を実行する
        
        Parameters:
        -----------
        features : pandas.DataFrame
            特徴量
        target : pandas.Series
            目標変数
        model_class : class
            使用するモデルのクラス（LightGBMModel、XGBoostModel、RandomForestModelなど）
            
        Returns:
        --------
        dict
            交差検証の結果を含む辞書
        """
        # 結果を格納するリスト
        fold_metrics = []
        feature_importances = []
        best_params_list = []
        all_predictions = pd.DataFrame()
        
        print(f"\n{self.n_splits}分割交差検証を開始...")
        
        # 各分割でモデルを学習・評価
        for fold, (train_idx, test_idx) in enumerate(self.cv.split(features, target)):
            print(f"\n=== Fold {fold+1}/{self.n_splits} ===")
            
            # 訓練データとテストデータの分割
            X_train = features.iloc[train_idx]
            y_train = target.iloc[train_idx]
            X_test = features.iloc[test_idx]
            y_test = target.iloc[test_idx]
            
            # 特徴量のスケーリング
            scaler = StandardScaler()
            X_train_scaled = pd.DataFrame(
                scaler.fit_transform(X_train),
                columns=features.columns
            )
            X_test_scaled = pd.DataFrame(
                scaler.transform(X_test),
                columns=features.columns
            )
            
            # モデルのインスタンス化と学習
            model = model_class()
            best_params, best_model = model.perform_grid_search(
                X_train_scaled, y_train,
                X_test_scaled, y_test
            )
            
            # 予測
            y_pred = best_model.predict(X_test_scaled)
            y_pred_proba = best_model.predict_proba(X_test_scaled)
            
            # 評価指標の計算
            specificity = calculate_specificity(y_test, y_pred)
            
            metrics = {
                'fold': fold + 1,
                'accuracy': accuracy_score(y_test, y_pred),
                'precision': precision_score(y_test, y_pred, average='weighted'),
                'recall': recall_score(y_test, y_pred, average='weighted'),
                'specificity': specificity,  # 特異度を追加
                'f1': f1_score(y_test, y_pred, average='weighted')
            }
            fold_metrics.append(metrics)
            
            # 特徴量の重要度
            importance = pd.DataFrame({
                'feature': features.columns,
                'importance': best_model.feature_importances_,
                'fold': fold + 1
            })
            feature_importances.append(importance)
            
            # 最適パラメータを保存
            best_params_list.append(best_params)
            
            # 予測結果を保存
            fold_preds = pd.DataFrame({
                'fold': fold + 1,
                'true_label': pd.Series(y_test).map({0: 'intact', 1: 'mci'}),
                'predicted_label': pd.Series(y_pred).map({0: 'intact', 1: 'mci'}),
                'probability_mci': y_pred_proba[:, 1]
            }, index=X_test.index)
            fold_preds['correct'] = fold_preds['true_label'] == fold_preds['predicted_label']
            all_predictions = pd.concat([all_predictions, fold_preds])
        
        # 結果をまとめる
        metrics_df = pd.DataFrame(fold_metrics)
        importances_df = pd.concat(feature_importances)
        
        # 平均指標を計算
        avg_metrics = {
            'accuracy': metrics_df['accuracy'].mean(),
            'precision': metrics_df['precision'].mean(),
            'recall': metrics_df['recall'].mean(),
            'specificity': metrics_df['specificity'].mean(),  # 特異度の平均を追加
            'f1': metrics_df['f1'].mean()
        }
        
        # 標準偏差を計算
        std_metrics = {
            'accuracy_std': metrics_df['accuracy'].std(),
            'precision_std': metrics_df['precision'].std(),
            'recall_std': metrics_df['recall'].std(),
            'specificity_std': metrics_df['specificity'].std(),  # 特異度の標準偏差を追加
            'f1_std': metrics_df['f1'].std()
        }
        
        # 平均特徴量重要度を計算
        avg_importance = importances_df.groupby('feature')['importance'].mean().reset_index()
        avg_importance = avg_importance.sort_values('importance', ascending=False)
        
        # 結果を保存
        self.results = {
            'fold_metrics': metrics_df,
            'avg_metrics': avg_metrics,
            'std_metrics': std_metrics,
            'feature_importances': importances_df,
            'avg_importance': avg_importance,
            'predictions': all_predictions,
            'best_params': best_params_list
        }
        
        return self.results
    
    def print_results(self):
        """結果を表示"""
        if not self.results:
            print("まだ交差検証が実行されていません。")
            return
        
        print("\n=== 交差検証の結果 ===")
        print("\n各分割の性能指標:")
        print(self.results['fold_metrics'])
        
        print("\n平均性能指標:")
        for metric, value in self.results['avg_metrics'].items():
            std = self.results['std_metrics'][f'{metric}_std']
            print(f"{metric}: {value:.4f} ± {std:.4f}")
        
        print("\n特徴量の平均重要度（上位10件）:")
        print(self.results['avg_importance'].head(10))
    
    def plot_results(self):
        """結果を可視化"""
        if not self.results:
            print("まだ交差検証が実行されていません。")
            return
        
        # 1. 各分割の性能指標を可視化
        plt.figure(figsize=(14, 7))
        metrics_df = self.results['fold_metrics'].melt(
            id_vars=['fold'],
            value_vars=['accuracy', 'precision', 'recall', 'specificity', 'f1'],  # 特異度を追加
            var_name='metric',
            value_name='value'
        )
        sns.barplot(x='metric', y='value', hue='fold', data=metrics_df)
        plt.title('Performance Metrics by Fold', fontsize=14)
        plt.ylim(0, 1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
        # 2. 特徴量の重要度を可視化
        plt.figure(figsize=(12, 8))
        sns.barplot(
            x='importance',
            y='feature',
            data=self.results['avg_importance'].head(10)
        )
        plt.title('Top 10 Most Important Features (Average)', fontsize=14)
        plt.grid(axis='x', linestyle='--', alpha=0.7)
        plt.tight_layout()
        plt.show()
        
    def save_results(self, prefix='cv_results', output_dir='result'):
        """
        結果をファイルに保存
        
        Parameters:
        -----------
        prefix : str, default='cv_results'
            出力ファイル名の接頭辞
        output_dir : str, default='result'
            結果ファイルの出力先ディレクトリ
        """
        if not self.results:
            print("まだ交差検証が実行されていません。")
            return
        
        import os
        # 出力ディレクトリが存在しない場合は作成
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # ファイルパスの作成
        metrics_path = os.path.join(output_dir, f'{prefix}_metrics.csv')
        importance_path = os.path.join(output_dir, f'{prefix}_avg_importance.csv')
        predictions_path = os.path.join(output_dir, f'{prefix}_predictions.csv')
        params_path = os.path.join(output_dir, f'{prefix}_best_params.txt')
        
        # 各分割の性能指標を保存
        self.results['fold_metrics'].to_csv(metrics_path, index=False)
        
        # 特徴量の重要度を保存
        self.results['avg_importance'].to_csv(importance_path, index=False)
        
        # 予測結果を保存
        self.results['predictions'].to_csv(predictions_path)
        
        # 最適パラメータを保存
        with open(params_path, 'w') as f:
            for fold, params in enumerate(self.results['best_params']):
                f.write(f"=== Fold {fold+1} ===\n")
                for param, value in params.items():
                    f.write(f"{param}: {value}\n")
                f.write("\n")
            
        print(f"結果を {output_dir} ディレクトリに保存しました。")


def run_cv_analysis(df, model_class, n_splits=5, output_dir='result', random_state=42):
    """
    交差検証で分析を実行する関数
    
    Parameters:
    -----------
    df : pandas.DataFrame
        分析対象のデータフレーム
    model_class : class
        使用するモデルのクラス（LightGBMModel、XGBoostModel、RandomForestModelなど）
    n_splits : int, default=5
        交差検証の分割数
    output_dir : str, default='result'
        結果ファイルの出力先ディレクトリ
    random_state : int, default=42
        乱数シード
    
    Returns:
    --------
    tuple
        (CrossValidator インスタンス, 結果の辞書)
    """
    print(f"\nUsing {model_class.__name__} with {n_splits}-fold cross-validation...")
    
    # データの前処理
    print("データの前処理を開始...")
    features = df.drop(['InspectionDateAndId', 'ncggfat_label'], axis=1)
    
    # 元のデータの値を確認
    original_values = df['ncggfat_label'].unique()
    print("元のncggfat_labelの一意な値:", original_values)
    
    # 元のデータが数値の場合はそのまま使用
    if set(original_values).issubset({0, 1}):
        target = df['ncggfat_label'].copy()
        print("数値ラベルをそのまま使用します (0, 1)")
    else:
        # 文字列の場合はマッピング
        target = df['ncggfat_label'].map({'intact': 0, 'mci': 1})
        print("文字列ラベルをマッピングします (intact->0, mci->1)")
    
    # 配列文字列の処理
    for col in ['freq', 'power_spectrum']:
        if col in features.columns:
            features[col] = features[col].apply(safe_convert_array)
    
    # 欠損値の削除
    features = features.dropna()
    target = target[features.index]
    
    print(f"\n処理後のデータ数: {len(features)}")
    print("\nクラスの分布:")
    print(target.value_counts())
    
    # 交差検証の実行
    validator = CrossValidator(n_splits=n_splits, random_state=random_state)
    results = validator.run_cross_validation(features, target, model_class)
    
    # 結果の表示と可視化
    validator.print_results()
    validator.plot_results()
    
    # 結果の保存
    model_name = model_class.__name__.lower()
    validator.save_results(
        prefix=f'cv_{model_name}_{n_splits}fold',
        output_dir=output_dir
    )
    
    return validator, results