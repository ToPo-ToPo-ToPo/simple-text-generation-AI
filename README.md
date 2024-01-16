# simple-text-generation-AI
Large Language Model(LLM)用のプログラムをまとめたパッケージです．なるべく簡単にLLMを使用，モデルの追加実装ができる環境を目指します．
gradioを用いたLLMとのchatbot機能を使用できます．

## 開発環境
- Python 3.11.7
- transformers
- gradio

## インストール方法
1. パッケージ直下のディレクトリにてローカルのPythonのバージョンを設定します．
```
pyenv local 3.11.7
```
2. 仮想環境を構築します．実行は同じパッケージ直下のディレクトリにて実施します．
```
python -m venv env
```
3. 仮想環境を起動します．  
```
.\env\Scripts\activate
```
- Macの場合は，以下の通りです．
```
source .\env\Scripts\activate
```
4. 必要なパッケージをインストールします．
- Windowsの場合は，以下の通りです．なお，pytorchのインストールは別途pytorch_install_command.txtの中身を実行させます．
```
pip install -r requirements_win.txt
```
```
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
- Macの場合は，以下の通りです．
```
pip install -r requirements_apple_silicon.txt
```

## 使用方法
ターミナル起動直後は，仮想環境が起動していないため，最初にパッケージ直下のディレクトリにて，次を実行する必要があります．  
- Windowsの場合は，以下の通りです．
```
.\env\Scripts\activate
```
- Macの場合は，以下の通りです．
```
source .\env\Scripts\activate
```

## その他補足
- pyenv+venvのコマンドは次の記事に整理しています．https://zenn.dev/topo/scraps/d5076b05f5283d
- Hugging Faceのダウンロードされたモデルのcacheは，models/cache/に保存されます．非常に容量が大きいため，PCの容量が問題なった場合は，このcacheを消去してください．

## ライセンス
This project is released under the MIT License, see LICENSE file.

