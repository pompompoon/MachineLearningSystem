import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from itertools import combinations

def get_proba(model, X):
    """様々なモデルタイプに対応する予測確率取得関数"""
    # 標準のscikit-learnインターフェース
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    # LightGBM Boosterの場合
    elif hasattr(model, 'predict'):
        # LightGBMのBoosterは予測値を直接返す
        preds = model.predict(X)
        # シグモイド関数を適用して確率に変換（二値分類の場合）
        if preds.ndim == 1:
            return 1 / (1 + np.exp(-preds))
        # 複数クラスの場合は最初のクラスの確率を返す
        else:
            return preds[:, 1] if preds.shape[1] > 1 else preds[:, 0]
    # XGBoostのモデルの場合
    elif hasattr(model, 'get_booster'):
        import xgboost as xgb
        dmatrix = xgb.DMatrix(X)
        preds = model.predict(dmatrix)
        return preds
    # その他の場合
    else:
        raise ValueError("サポートされていないモデルタイプです。予測確率を取得できません。")

def create_manual_pdp(model, X, feature_names, feature_importance=None, output_dir=None):
    """
    scikit-learnを使わずに、手動で部分依存プロットを作成する関数
    
    Parameters:
    -----------
    model : 予測モデル
        予測メソッドを持つモデル
    X : pandas.DataFrame
        特徴量データ
    feature_names : list
        プロットする特徴量の名前リスト
    feature_importance : pandas.DataFrame, optional
        'feature'と'importance'列を持つ特徴量重要度
    output_dir : str, optional
        出力ディレクトリ
    """
    print("\n===== 手動部分依存プロット作成 =====")
    
    # 出力ディレクトリの作成
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # 特徴量重要度がない場合は、特徴量名を順番に使用
    if feature_importance is None:
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': range(len(feature_names), 0, -1)
        })
    
    # 上位特徴量を抽出 (最大6個)
    top_features = []
    for feat in feature_importance['feature']:
        if feat in feature_names and feat in X.columns:
            top_features.append(feat)
            if len(top_features) >= 6:
                break
    
    print(f"プロットする特徴量: {top_features}")
    
    # 1. 1次元部分依存プロット
    print("\n1次元部分依存プロットを作成します...")
    
    # 図の準備
    n_cols = 3
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    fig, axs = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(n_cols * 5, n_rows * 4))
    axs = axs.flatten() if hasattr(axs, 'flatten') else [axs]
    
    for i, feature in enumerate(top_features):
        if i >= len(axs):
            break
            
        try:
            print(f"特徴量 {feature} の部分依存プロットを作成中...")
            
            # 特徴量の値の範囲を決定
            x_min, x_max = X[feature].min(), X[feature].max()
            x_range = np.linspace(x_min, x_max, 50)
            
            # 部分依存値の計算
            pdp_values = []
            for x_val in x_range:
                # データのコピーを作成
                X_temp = X.copy()
                # 特徴量の値を固定
                X_temp[feature] = x_val
                # 予測確率の平均を計算
                pred_proba = get_proba(model, X_temp)  # モデルタイプに依存しない予測関数を使用
                pdp_values.append(np.mean(pred_proba))
            
            # プロット
            axs[i].plot(x_range, pdp_values, 'b-')
            axs[i].set_title(feature, fontsize=10)
            axs[i].set_xlabel(feature, fontsize=8)
            axs[i].set_ylabel('Partial dependence', fontsize=8)
            axs[i].grid(True, linestyle='--', alpha=0.6)
            
            print(f"特徴量 {feature} のプロットに成功")
            
        except Exception as e:
            print(f"特徴量 {feature} のプロット中にエラー: {e}")
            axs[i].text(0.5, 0.5, f"Error: {feature}", ha='center', va='center', transform=axs[i].transAxes)
    
    # 使用しない軸を非表示に
    for i in range(len(top_features), len(axs)):
        axs[i].set_visible(False)
        
    plt.tight_layout()
    if output_dir:
        fig.savefig(os.path.join(output_dir, 'manual_pdp_1d.png'), dpi=300, bbox_inches='tight')
    
    # 図を閉じる
    plt.close(fig)
    print("1次元部分依存プロットの作成が完了しました")
    
    # 2. 2次元部分依存プロット (上位3特徴量の組み合わせ)
    if len(top_features) >= 2:
        print("\n2次元部分依存プロットを作成します...")
        
        # 上位3特徴量から組み合わせを生成
        feature_pairs = list(combinations(top_features[:3], 2))
        
        # プロットを保存するディレクトリ
        if output_dir:
            interaction_dir = os.path.join(output_dir, 'interactions')
            if not os.path.exists(interaction_dir):
                os.makedirs(interaction_dir)
        
        # カスタムカラーマップ
        colors = ['#663399', '#336699', '#339999', '#66cc99', '#99cc66', '#cccc33']
        custom_cmap = LinearSegmentedColormap.from_list('custom_viridis', colors, N=256)
        
        for i, (feat1, feat2) in enumerate(feature_pairs):
            try:
                print(f"特徴量の組み合わせ {i+1}/{len(feature_pairs)}: {feat1} と {feat2}")
                
                # 特徴量の値の範囲を決定
                x1_min, x1_max = X[feat1].min(), X[feat1].max()
                x2_min, x2_max = X[feat2].min(), X[feat2].max()
                
                # グリッドポイントの生成 (解像度を低めに)
                resolution = 20
                x1_range = np.linspace(x1_min, x1_max, resolution)
                x2_range = np.linspace(x2_min, x2_max, resolution)
                
                # 2D部分依存値の計算
                pdp_values_2d = np.zeros((resolution, resolution))
                
                for i1, x1_val in enumerate(x1_range):
                    for i2, x2_val in enumerate(x2_range):
                        # データのコピーを作成
                        X_temp = X.copy()
                        # 特徴量の値を固定
                        X_temp[feat1] = x1_val
                        X_temp[feat2] = x2_val
                        # 予測確率の平均を計算
                        pred_proba = get_proba(model, X_temp)  # モデルタイプに依存しない予測関数を使用
                        pdp_values_2d[i2, i1] = np.mean(pred_proba)  # 注: インデックスの順序に注意
                
                # メッシュグリッド作成
                X1, X2 = np.meshgrid(x1_range, x2_range)
                
                # プロット作成
                fig, ax = plt.subplots(figsize=(10, 8))
                cf = ax.contourf(X1, X2, pdp_values_2d, cmap=custom_cmap, levels=15)
                plt.colorbar(cf, ax=ax)
                
                # グラフの装飾
                ax.set_xlabel(feat1)
                ax.set_ylabel(feat2)
                rank1 = top_features.index(feat1) + 1
                rank2 = top_features.index(feat2) + 1
                ax.set_title(f"特徴量相互作用: {feat1} (ランク{rank1}) vs {feat2} (ランク{rank2})\nターゲット: MCI", fontsize=12)
                ax.grid(True, linestyle='--', alpha=0.3)
                
                # 保存
                if interaction_dir:
                    clean_name1 = feat1.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    clean_name2 = feat2.replace(' ', '_').replace('/', '_').replace('\\', '_')
                    clean_name1 = ''.join(c for c in clean_name1 if c.isalnum() or c in '_-.')
                    clean_name2 = ''.join(c for c in clean_name2 if c.isalnum() or c in '_-.')
                    
                    filename = os.path.join(interaction_dir, f"pdp_interaction_{rank1}_{clean_name1}_vs_{rank2}_{clean_name2}.png")
                    fig.savefig(filename, dpi=300, bbox_inches='tight')
                    print(f"保存しました: {filename}")
                
                # 図を閉じる
                plt.close(fig)
                print(f"特徴量 {feat1} と {feat2} の相互作用プロットに成功")
                
            except Exception as e:
                print(f"特徴量 {feat1} と {feat2} の相互作用プロット中にエラー: {e}")
                import traceback
                traceback.print_exc()
    
    print("すべてのプロットの作成が完了しました")