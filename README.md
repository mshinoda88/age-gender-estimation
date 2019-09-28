# age-gender-estimation
画像から性別、年代を推定するライブラリ

モデル取得先
https://github.com/yu4u/age-gender-estimation/releases/download/v0.5/weights.28-3.73.hdf5

## Requirements

```bash
TODO
```

## Installation
Download pretrained style weight files.

```bash
cd conf/age_gender_estimator
bash download.sh

```

## directories

```bash
 ├─ conf          各種設定ファイル
 │   └─ age_gender_estimator  モデルの重みファイル
 ├─ src           python ファイル格納場所
 └─ images
     ├─ input     画像入力ファイル
     └─ output    画像出力場所
```

