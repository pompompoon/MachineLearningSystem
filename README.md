# 機械学習パイプライン — 分類・回帰分析システム

表形式データに対する分類・回帰タスクを、複数のGBDTモデル（LightGBM / XGBoost / CatBoost / RandomForest）で実行できる機械学習パイプラインです。不均衡データ対策、モデル解釈、交差検証、結果の可視化までを一貫して行えます。
テーブルデータのIDカラムにID,教師ラベルカラムにTargetとつけてください。

## 主な機能

- **複数モデル対応**: LightGBM, XGBoost, CatBoost, RandomForest を統一インターフェースで切り替え可能
- **不均衡データ対策**: SMOTE（Borderline-SMOTE / SMOTETomek 含む）、Undersampling、Undersampling Bagging
- **回帰用SMOTE**: binning / density / outliers の3手法に対応。整数値SMOTEにも対応
- **モデル解釈**: ALE（Accumulated Local Effects）、PDP（Partial Dependence Plot）、ICE
- **交差検証**: Holdout / K-Fold に対応
- **結果管理**: 実行結果の自動整理・保存

## ディレクトリ構成

```
├── data/                        # 入力データ
├── feature_generation/          # 特徴量生成
│   ├── extractioncondition/     # 抽出条件（フィルタ等）
│   └── fdata/                   # 特徴量生成スクリプト群
├── models/                      # モデル実装
├── visualization/               # 可視化・評価ツール
├── pretreatment/                # 前処理（クラスタリング、UMAP等）
├── result/                      # 実行結果出力先
├── main5.py                     # エントリーポイント（分類）
├── main5k_ale2.py               # エントリーポイント（回帰）
└── environment.yml              # conda環境定義
```

## セットアップ

```bash
conda env create -f environment.yml
conda activate new_base_env
```

## 使い方

### 分類（main5.py）

```bash
# 基本実行
python main5.py --model lightgbm --data-path data/features_train.csv --output-dir result

# SMOTE + 交差検証
python main5.py --model catboost --smote --cv --splits 5

# Undersampling Bagging
python main5.py --undersampling --base-model xgboost --n-bags 10

# 可視化のみ（学習なし）
python main5.py --viz-only
```

#### オプション一覧

| オプション | 説明 |
|---|---|
| `--model` | 使用モデル（`lightgbm`, `xgboost`, `random_forest`, `catboost`） |
| `--cv` | 交差検証を実行 |
| `--splits` | 交差検証の分割数 |
| `--smote` / `--no-smote` | SMOTEの使用有無 |
| `--undersampling` | Undersampling Bagging を使用 |
| `--simple-undersampling` | バギングなしのアンダーサンプリング |
| `--base-model` | Undersampling Bagging のベースモデル |
| `--n-bags` | バッグ数 |
| `--random-state` | 乱数シード |
| `--data-path` / `--data-file` | 入力データの指定 |
| `--output-dir` | 結果出力ディレクトリ |
| `--viz-only` | 可視化のみ実行（学習スキップ） |
| `--no-pdp` | PDPを作成しない |
| `--no-save` | プロットを保存しない |
| `--no-organize` | 結果ファイルを整理しない |

### 回帰（main5k_ale2.py）

```bash
# 基本実行
python main5k_ale2.py --model catboost --data-path data/features_train.csv --output-dir result

# SMOTE（density法）+ ALEのみ
python main5k_ale2.py --smote --smote-method density --smote-k-neighbors 5 --smote-density-threshold 0.1 --no-pdp

# SMOTE（binning法）
python main5k_ale2.py --model catboost --smote --smote-method binning

# ALE と PDP を比較
python main5k_ale2.py --compare-ale-pdp
```

#### 回帰固有オプション

| オプション | 説明 |
|---|---|
| `--target-column` | 目的変数のカラム名 |
| `--bagging` | バギングを使用 |
| `--smote-method` | 回帰用SMOTE手法（`binning`, `density`, `outliers`） |
| `--smote-k-neighbors` | SMOTEのk近傍数 |
| `--smote-n-bins` | binning法のビン数 |
| `--smote-density-threshold` | density法の閾値 |
| `--smote-outlier-threshold` | outlier法の閾値 |
| `--integer-smote` | 整数値SMOTEを使用 |
| `--target-min` / `--target-max` | 目的変数の範囲制約 |
| `--no-ale` / `--ale-n-features` | ALEプロットの制御 |
| `--pdp-n-features` / `--pdp-grid-resolution` | PDPの制御 |
| `--compare-ale-pdp` | ALEとPDPを比較表示 |

## モジュール構成

### models/ — モデル・学習ロジック

| ファイル | 役割 |
|---|---|
| `base_model.py` | 全モデルの基底クラス |
| `lightgbm_model.py` | LightGBM分類モデル |
| `xgboost_model.py` / `xgboost_regression_model.py` | XGBoost（分類/回帰） |
| `catboost_model.py` / `catboost_regression_model.py` | CatBoost（分類/回帰） |
| `random_forest_model.py` | RandomForest分類モデル |
| `regression_models.py` | 回帰モデル共通処理 |
| `cross_validator.py` | 交差検証 |
| `regression_smote.py` | 回帰用SMOTE（整数対応） |
| `undersampling_bagging_model.py` | Undersampling Bagging |
| `undersampling_model.py` | シンプルアンダーサンプリング |
| `ale_plotter.py` | ALEプロット生成 |
| `partial_dependence_plotter.py` / `new_pdp.py` | PDPプロット生成 |
| `RegressionResultManager.py` | 回帰結果の管理・保存 |

### visualization/ — 可視化・評価

| ファイル | 役割 |
|---|---|
| `model_evaluator.py` | 分類モデルの評価（混同行列、各種メトリクス等） |
| `regression_model_evaluator.py` | 回帰モデルの評価（RMSE、R²等） |
| `regression_visualizer.py` | 回帰結果の可視化 |
| `regression_pdp_handler.py` / `regression_ice.py` | PDP / ICE プロット |
| `smote_visualization.py` | SMOTE適用結果の可視化 |
| `threshold_evaluator.py` | 閾値評価 |
| `eye_tracking_visualizer.py` | データ可視化 |

### feature_generation/ — 特徴量生成

- `extractioncondition/` — データ抽出条件の定義（フィルタリング）
- `fdata/` — 各種特徴量生成スクリプト

### pretreatment/ — 前処理

- K-Meansクラスタリング、UMAPによる次元削減・データ探索

## 技術スタック

- **Python 3.12**（conda: `new_base_env`）
- **GBDT**: LightGBM, XGBoost, CatBoost
- **scikit-learn**: RandomForest, 前処理、評価指標
- **不均衡データ対策**: imbalanced-learn（SMOTE / SMOTETomek / Borderline-SMOTE）
- **モデル解釈**: ALE, PDP, ICE
- **可視化**: matplotlib
- **次元削減**: UMAP

- <img width="981" height="543" alt="image" src="https://github.com/user-attachments/assets/0421dd9c-8244-4552-b4ce-23729c0fbf75" />

