# ForgeCal Post

**2D鍛造FEM解析「ForgeCal」のポスト処理アプリ（Windows用）**

[ForgeCal](https://fem.matsumoto-works.jp) でダウンロードした `results.h5` を読み込み、
応力・ひずみ・速度ベクトル・荷重曲線などを表示・GIF出力できます。
Python のインストールは不要です。

## ダウンロード

👉 **[最新版をダウンロード](https://github.com/matsumoto-works/forgecal/releases/latest)**

- `ForgeCalPost.zip` … アプリ本体（解凍して `ForgeCalPost.exe` を実行）
- `sample_results.h5` … 動作確認用サンプルデータ

## 使い方

1. `ForgeCalPost.zip` を解凍
2. `ForgeCalPost.exe` をダブルクリック（ブラウザが自動で開きます）
3. `results.h5` をドラッグ＆ドロップ（まず試すなら `sample_results.h5`）

## 自分で解析するには

[ForgeCal](https://fem.matsumoto-works.jp) で DXF をアップロードして解析し、
`results.h5` をダウンロードしてください。

## 動作環境

Windows 10 / 11（64bit）。Python のインストール不要。

---

> 初回起動時に Windows SmartScreen の警告が出ることがあります（署名なし EXE のため）。
> 「詳細情報」→「実行」で起動できます。
