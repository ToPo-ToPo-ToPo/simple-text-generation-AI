# simple-text-generation-AI
Large Language Model(LLM)用のプログラムをまとめたパッケージです。なるべく簡単にLLMを使用、モデルの追加実装ができる環境を目指します。
gradioを用いたLLMとのchatbot機能を使用できます。

## 開発環境
- Python 3.11.7
- gradio

## 仮想環境の構築と起動
1. パッケージ直下のディレクトリにてローカルのPythonのバージョンを設定します。
```
pyenv local 3.11.7
```
2. 仮想環境の情報を保存するディレクトリを作成します。
```
python -m venv env
```
3. Windowsの場合は、以下から仮想環境を起動します。  
```
.\env\Scripts\activate
```
- Macの場合は、以下から仮想環境を起動します。
```
source ./env/bin/activate 
```

## パッケージのインストール
### Windowsの場合
以下をコピーして実行してください。ただし、2行目のPytorchについては自身の環境に合わせて変更してください。
コマンドは[公式サイト](https://pytorch.org/get-started/locally/)から確認できます。
```
pip install transformers
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install ctranslate2
pip install langchain
pip install flask
pip install sentencepiece
pip install gradio 
pip install bitsandbytes
pip install datasets
pip install trl
pip install peft 
```
### Macの場合
以下を実行してください。ただし、Macではcudaを使用できないためPytorchはデフォルト版をインストールします。
コマンドは[公式サイト](https://pytorch.org/get-started/locally/)から確認できます。
```
pip install transformers
pip3 install torch torchvision torchaudio
pip install ctranslate2
pip install langchain
pip install flask
pip install sentencepiece
pip install gradio 
pip install bitsandbytes
pip install datasets
pip install trl
pip install peft
```

## 使用方法
1. 仮想環境の起動
2. パッケージ直下にて以下を実行します。
```
python gradio_main.py
```
3. LLMをロードし、会話します。具体的な操作方法は[Zennの記事](https://zenn.dev/topo/articles/5ddedb7ea81130v)を参照してください。

## その他補足
- [pyenv+venvのコマンド](https://zenn.dev/topo/scraps/d5076b05f5283d)
- Hugging Faceからダウンロードされたモデルのcacheは、models/cache/に保存されます。非常に容量が大きいため、PCの容量が問題なった場合はこのcacheディレクトリを消去してください。

## ライセンス
This project is released under the MIT License, see LICENSE file. However portions of the project are available under separate license terms: Pytorch is licensed under the BSD license.

