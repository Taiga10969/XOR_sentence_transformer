# XOR_sentence_transformer
XOR発見を目的として，sentence_transformerを用いた文章埋め込みによるコサイン類似度による推論の挙動確認<br>

-  ```main.ipynb``` : sentence_transformerを用いた論文テキストのランキング問題推論テスト

## 実験環境等
データセット：[SciCap+データセット](https://huggingface.co/datasets/anselyang/SciCapPlus/tree/main)<br>
Dockerイメージ：[taiga10969/basic_image:cuda12.1.0-ubuntu22.04-python3.10](https://hub.docker.com/layers/taiga10969/basic_image/cuda12.1.0-ubuntu22.04-python3.10/images/sha256-076a9005a1daafe2910eda4354921bd852f8611fa70d040313a4504e880f981e?context=repo)<br>
1. ```pip install -r requirements.txt```
2. ```main.ipynb```を実行

## 特徴量の可視化
あるテストデータを入力した際のBERTモデル内の各ポイントでの特徴量を取得し，UMAPで次元削減し2次元のマップとして可視化を行う．<br>
これにより，各ポイントでの特徴量の分布を可視化し，Transformer Encoder内部での特徴空間の変化を確認する．<br>
特徴量の可視化を行う各ポイントについては，以下の通りである．
- Input Embedding後
- PE後
- Multi-Head Attentionの最後のLinear層の処理前
- Multi-Head Attentionの最後のLinear層の処理後（MHAの出力）
- Feed Forward Netwarkの処理後
の計5ポイントで，Input EmbeddingとPE以外はTransformer Encoderの積層数分(12layer)あるため，可視化する特徴量は合計38となる．

作成：Taiga MASUDA <br>
最終更新：2024.3.31 (Sun.)
