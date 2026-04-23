#### dog_feel_classify
画像と音（スペクトログラム）で分類、マルチモーダルTransformer.  
犬の画像(動画) と、鳴き声を同時に入力して、犬の気持ちが分類できる、モデル  

##### 開発環境  

  Ubuntu 24.04 PC  
  PyTorch  

##### 実行環境  

  Ubuntu 24.04 PC or Orange pi 5  
  onnx-runtime  

##### 1. 転移学習 or ファインチューニング  

  dog_feel_train.ipynb  

##### 2. 実行  

  Ubuntu 24.04 PC or Orange pi 5  
  onnx-runtime  
  $ python dog_feel_orangepi_onnx.py  

##### 3. 犬の監視 アプリケーション  

  USB カメラから、Video 、 Audio をキャプションして、犬の感情の予測 を行います。  
  キャプションは、常に実行中になりますが、 予測処理 は、音が一定の大きさになったタイミングで、4[秒] ためて行います。  
  この処理は、4回 連続されます。 4[秒] x 4 = 16[秒] その後、再度、入力音待ちになります。  
  予測結果 の class id と score は、ターミナルに表示されます。  
  
  $ python  dog_feel_watch.py  
  
  USB カメラ(音声入力付き)が、必要です。  

##### 4. 参照  
  
  [画像と音（スペクトログラム）で分類、マルチモーダルTransformer.](https://www.netosa.com/blog/2026/04/transformer.html)  
