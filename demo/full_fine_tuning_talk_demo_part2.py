
# モデルのキャッシュパスの変更
import os
os.environ['HF_HOME'] = '../cache/'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer
#======================================================================
# モデルを設定する
#======================================================================
def load_model():    
    
    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(
        "../models/open-calm-small-databricks-dolly-15k-ja-stf"
    ) 
    
    # モデルの準備
    model = AutoModelForCausalLM.from_pretrained(
       "../models/open-calm-small-databricks-dolly-15k-ja-stf",
        device_map="auto"
    )
    
    return tokenizer, model
    
#======================================================================
# prompt check
#======================================================================
def prompt_check(tokenizer, model, prompt):

    # 推論の実行
    for i in range(10):
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            tokens = model.generate(
                **inputs,
                max_new_tokens=64,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.05,
                pad_token_id=tokenizer.pad_token_id,
            )
        
        output = tokenizer.decode(tokens[0], skip_special_tokens=True)
        print(output)
        print("----")


#======================================================================
# メインプログラム
#======================================================================
if __name__ == '__main__':
    
    # トークナイザーとモデルを設定する
    tokenizer, model = load_model()

    # プロンプトを設定する
    question = "日本の首都は?"
    prompt = f"USER: {question}\nASSISTANT: "
    
    # プロンプトの状態を試す
    prompt_check(tokenizer=tokenizer, model=model, prompt=prompt)
    