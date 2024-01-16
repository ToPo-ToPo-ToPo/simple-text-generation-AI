# llmモデルのインポート
import sys
sys.path.append('../src/')

# モデルのキャッシュパスの変更
import os
os.environ['TRANSFORMERS_CACHE'] = '../models/cache/'

import torch
from llm.model_factory import ModelFactory
#============================================================
#プログラムの実行
#============================================================
if __name__ == "__main__":

    # 使用するllmの選択
    print("使用するLLMを以下から選んで入力してください。")
    print("rinna, line, OpenCALM")
    model_name = input("使用するLLM: ")

    # 進捗の表示
    print("LLMを読み込み中です。")

    # llmのオブジェクトを生成
    model_factory = ModelFactory()
    llm = model_factory.create(name=model_name, processor="cpu", load_bit_size=torch.bfloat16, load_in_8bit=False)

    # 対話の開始
    print("会話を始めましょう！何か質問して下さい。")

    while True:
        question = input("ユーザー: ")
        
        # 会話の終了タイミングを設定
        if question == "":
            print("空文字が入力されました。会話を終了します。")
            break

        # プロンプトの生成と内容を表示
        prompt = llm.generate_prompt(question)

        # 入力したinputに対する回答を作成
        response = llm.response(prompt)
        
        # 回答を表示
        print(response)