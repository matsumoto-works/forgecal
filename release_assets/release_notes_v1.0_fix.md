ForgeCal Post — Post-processor for 2D forming FEM analysis (Windows)

2D鍛造FEM解析「ForgeCal」のポスト処理アプリ（Windows用）

---

[ForgeCal](https://fem.matsumoto-works.jp) で解析した `results.h5` を読み込み、
応力・ひずみ・速度ベクトル・荷重曲線などを表示・GIF出力できます。

Load `results.h5` from [ForgeCal](https://fem.matsumoto-works.jp) to visualize
stress, strain, velocity vectors, load curves, and export GIFs.

---

**■ 使い方 / Usage**

1. `ForgeCalPost.zip` を解凍 / Extract `ForgeCalPost.zip`
2. `ForgeCalPost.exe` をダブルクリック（ブラウザが自動で開きます）  
   Double-click `ForgeCalPost.exe` (browser opens automatically)
3. `results.h5` をドラッグ＆ドロップ  
   Drag & drop `results.h5`  
   （まず試すなら `sample_results.h5` / Try `sample_results.h5` first）

**■ 動作環境 / Requirements**

Windows 10/11 (64-bit). Python のインストール不要 / No Python required.

**■ 自分で解析するには / To run your own analysis**

[ForgeCal](https://fem.matsumoto-works.jp) で DXF をアップロード → 解析 → `results.h5` をダウンロード

Upload DXF to [ForgeCal](https://fem.matsumoto-works.jp), run analysis, download `results.h5`.

---

> 初回起動時に Windows SmartScreen の警告が出ることがあります（「詳細情報」→「実行」）。  
> Windows SmartScreen may warn on first launch (unsigned EXE). Click "More info" → "Run anyway".
