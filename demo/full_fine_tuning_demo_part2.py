
# モデルのキャッシュパスの変更
import os
os.environ['HF_HOME'] = '../cache/'

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
#======================================================================
# モデルを設定する
#======================================================================
def load_model():    
    
    # トークナイザーの準備
    tokenizer = AutoTokenizer.from_pretrained(
        "cyberagent/open-calm-small"
        #"../models/open-calm-small-multilingual-sentiments-japanese-positive"
    ) 
    
    # モデルの準備
    model = AutoModelForCausalLM.from_pretrained(
        "cyberagent/open-calm-small", 
        #"../models/open-calm-small-multilingual-sentiments-japanese-positive",
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
# プロンプトを作成
#======================================================================   
def formatting_prompts_func(example):
    
    output_texts = []
    
    for i in range(len(example['instruction'])):
        
        text = f"USER: {example['instruction'][i]}\nASSISTANT: {example['output'][i]}<|endoftext|>"
        
        output_texts.append(text)
    
    return output_texts

#======================================================================
# トレーニングデータを作成する
#======================================================================
def create_train_dataset():
    
    # データセットの読み込み読み込み
    dataset = load_dataset("kunishou/databricks-dolly-15k-ja")
    
    # 確認
    print(dataset)
    print(dataset["train"][0])
    
    # データセットをinputが空の要素のみ5000個でフィルタリング
    train_dataset = dataset["train"].filter(lambda data: data["input"] == "").select(range(5000))

    # 確認
    print(train_dataset)
    print(train_dataset[0])
    
    return train_dataset
    
#======================================================================
# フルファインチューニングを行う
#======================================================================
def training(tokenizer, model, train_dataset):

    args = TrainingArguments(
        output_dir="../train_log"
    ) 

    # トレーナーの作成
    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        max_seq_length=64,
        formatting_func=formatting_prompts_func,
        args=args
    )
    
    # 学習の実行
    trainer.train()
    
    return trainer
    
#======================================================================
# メインプログラム
#======================================================================
if __name__ == '__main__':
    
    # トークナイザーとモデルを設定する
    tokenizer, model = load_model()

    # プロンプトを設定する
    #question = "日本の首都は?"
    #prompt = f"USER: {question}\nASSISTANT: "
    
    # プロンプトの状態を試す
    #prompt_check(tokenizer=tokenizer, model=model, prompt=prompt)
    
    # データセットを設定する
    train_dataset = create_train_dataset()
    
    # トレーニングを行う
    trainer = training(tokenizer=tokenizer, model=model, train_dataset=train_dataset)
    
    # モデルの保存
    trainer.save_model("../models/open-calm-small-databricks-dolly-15k-ja-stf")