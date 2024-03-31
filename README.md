# XOR_sentence_transformer
XOR発見を目的として，sentence_transformerを用いた文章埋め込みによるコサイン類似度による推論の挙動確認<br>

-  ```main.ipynb``` : sentence_transformerを用いた論文テキストのランキング問題推論テスト

## 環境構築等
データセット：[SciCap+データセット](https://huggingface.co/datasets/anselyang/SciCapPlus/tree/main)<br>
Dockerイメージ：[taiga10969/basic_image:cuda12.1.0-ubuntu22.04-python3.10](https://hub.docker.com/layers/taiga10969/basic_image/cuda12.1.0-ubuntu22.04-python3.10/images/sha256-076a9005a1daafe2910eda4354921bd852f8611fa70d040313a4504e880f981e?context=repo)<br>
1. ```pip install -r requirements.txt```
2. ```main.ipynb```を実行

作成：Taiga MASUDA <br>
最終更新：2024.3.31 (Sun.)
