ForgeCal Post — 2D鍛造FEM解析「ForgeCal」のポスト処理アプリ
==============================================================

【これは何？】
ForgeCal (https://fem.matsumoto-works.jp) でダウンロードした results.h5 を
読み込み、応力・ひずみ・速度ベクトル・荷重曲線などを表示・GIF出力する
Windows アプリです。Python のインストールは不要です。

【使い方】
1. このフォルダの ForgeCalPost.exe をダブルクリック
   （黒いウィンドウが出て、ブラウザが自動で開きます）
2. results.h5 をドラッグ＆ドロップ
   （まず試すなら、別途配布の sample_results.h5 を開いてください）
3. 終了するときは、画面左下の「⏻ アプリを終了」ボタン、
   または黒いウィンドウ（コンソール）を閉じてください。

【自分で解析するには】
ForgeCal (https://fem.matsumoto-works.jp) で DXF をアップロードして解析し、
results.h5 をダウンロードしてください。

【動作環境】
Windows 10 / 11 (64bit)

【注意】
- 初回起動時に Windows Defender / SmartScreen の警告が出ることがあります。
  「詳細情報」→「実行」で起動できます（署名なし EXE のため）。
- このフォルダ内のファイル（特に _internal フォルダ）は削除・移動しないで
  ください。EXE の動作に必要です。フォルダごと好きな場所に置けます。
