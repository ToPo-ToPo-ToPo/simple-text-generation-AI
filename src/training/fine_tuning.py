
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset
from trl import SFTTrainer
#======================================================================
# ファインチューニングを行う
#======================================================================
class FineTuning:
    
    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self, tokenizer, model):
        
        self.tokenizer = tokenizer
        self.model = model
    
    #----------------------------------------------------------
    # フルファインチューニングを行う
    #----------------------------------------------------------
    def training(self, tokenizer, model, prompt_format, train_dataset):
        
        # トレーナーの作成
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            dataset_text_field="text",
            max_seq_length=128,
        )
    
        # 学習の実行
        trainer.train()
    
        return trainer
    
    #----------------------------------------------------------
    # トレーニングデータを作成する
    #----------------------------------------------------------
    def create_train_dataset(self, dataset_name, option=""):
        
        # データセットの読み込み読み込み
        dataset = load_dataset("tyqiangz/multilingual-sentiments", "japanese")
    
        # 確認
        print(dataset)
        print(dataset["train"][0])
    
        # データセットをpositiveのみ5000個でフィルタリング
        train_dataset = dataset["train"].filter(lambda data: data["label"] == 0).select(range(5000))

        # 確認
        print(train_dataset)
        print(train_dataset[0])
    
        return train_dataset
        
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