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

分類の可視化例

<img width="726" height="629" alt="image" src="https://github.com/user-attachments/assets/c4caf22e-c454-4605-b16c-d302ea2df36e" />

<img width="902" height="861" alt="image" src="https://github.com/user-attachments/assets/1a293563-7cfd-46b2-a355-d3086c483b8b" />

<img width="1629" height="863" alt="image" src="https://github.com/user-attachments/assets/a05380a9-6464-4c53-ae76-6a67e0614c1a" />
# タイタニック号生存予測モデル

| 特徴量 | 解釈 |
|--------|------|
| Sex    | 女性（左側）→生存確率約0.65、男性（右側）→約0.27。最も強い影響。"Women and children first"が明確に出ている。 |
| Pclass | 1等→約0.5、2等→0.4、3等→0.4弱。上位クラスほど生存率が高い。階段状の変化。 |
| Age    | 若い（左側）ほど生存率がやや高く、年齢が上がると徐々に低下。子供優先の傾向が見える。 |
| Cabin  | 客室番号（エンコード済み）。ICE線のばらつきが非常に大きく、個人差が強い。欠損が多いカラムなので、欠損有無自体が社会的地位の代理変数になっている可能性がある。 |
| SibSp  | 同乗の兄弟・配偶者数。PDP線はほぼ平坦で影響は弱い。3人以上で若干低下。 |
| Ticket | チケット番号（エンコード済み）。PDPはほぼ平坦で、予測への寄与は小さい。IDと同様、本来意味のない特徴量なので除外を検討すべき。 |

■ 全体的に **Sex > Pclass > Age** が支配的で、歴史的事実と整合する。

<img width="1054" height="860" alt="image" src="https://github.com/user-attachments/assets/d29b7807-c384-491c-9928-f2538e640bdc" />

<img width="1245" height="873" alt="image" src="https://github.com/user-attachments/assets/70246409-5206-47fb-bea6-94bf4f492b6e" />



回帰の可視化例

<img width="938" height="623" alt="image" src="https://github.com/user-attachments/assets/1411c5b3-08ea-4786-99cf-15c628023e6c" />

<img width="1042" height="615" alt="image" src="https://github.com/user-attachments/assets/c06a5620-14fb-4136-9768-083953c4886b" />


<img width="1178" height="615" alt="image" src="https://github.com/user-attachments/assets/f1a78d02-453b-4937-a53e-d778827f4d4c" />
# California Housing 住宅価格予測モデル

| 特徴量             | 解釈 |
|--------------------|------|
| median_income      | 地区の世帯収入の中央値（万ドル単位）。ALEが最も大きく、価格への影響が圧倒的に強い。 |
| ocean_proximity    | 海への近さ（カテゴリ変数をエンコードしたもの）。INLAND, NEAR BAY, NEAR OCEAN, ISLANDなど。 |
| longitude          | 経度。西に行くほど（沿岸部）価格が上がる傾向。 |
| latitude           | 緯度。特定の緯度帯（SF・LA付近）で急上昇。 |
| ID                 | サンプルID。本来予測に使うべきでない特徴量で、データリーケージの可能性がある。ALE変動が大きいのは危険信号。 |
| housing_median_age | 住宅の築年数中央値。古い住宅（都心部に多い）ほどやや高い傾向。 |

■ **median_income** が支配的。**ID** カラムは除外すべき。


# 来客数予測

<img width="1041" height="628" alt="image" src="https://github.com/user-attachments/assets/7f46ce8d-5d9b-4d93-9a66-0168fa261e72" />

<img width="1178" height="621" alt="image" src="https://github.com/user-attachments/assets/2f867302-fa6d-4acb-9b9b-9b197e25c6dd" />

#来訪者数予測モデル ALE解釈

|特徴量|解釈|
|--------------------|------|
|visitors_rolling_mean7|過去7日間の来訪者数移動平均。|ALE幅が最大（-30〜+30）で最も支配的な特徴量。直近の来訪トレンドが高ければ予測も上がる、自然な時系列的関係|
|area_|店舗面積が大きいほど来訪者数が増加する明確な正の関係。店舗のキャパシティや立地規模を反映。|
|visitors_rolling_mean3|過去3日間の移動平均。7日版と同様の傾向だがALE幅がやや小さい（-15〜+15）。短期トレンドの補完的な役割。|
|dow|曜日。左側（平日前半）で-10、右側（週末）で+20。週末に来訪が集中する典型的な小売パターン。|
|promo_budget_|販促予算。低予算域では-25と大きくマイナス、予算増加に伴い+10まで上昇。販促投資と来訪者数の正の関係を捉えている。ただし高予算域（6以上）ではデータが疎でALEが不安定。|
|is_holiday|祝日。ALE値がほ\ぼ0（Y軸スケール0.04）で影響なし。曜日やrolling_meanに祝日効果が吸収されている可能性が高い。|






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



