
from transformers import TrainingArguments
from datasets import load_dataset
from trl import SFTTrainer
#======================================================================
# ファインチューニングを行う
#======================================================================
class InstructionSft:
    
    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self):
        pass
    
    #----------------------------------------------------------
    # フルファインチューニングを行う
    #----------------------------------------------------------
    def training(self, tokenizer, model, prompt_format, train_dataset):
        
        args = TrainingArguments(
            output_dir="../temp/train_log"
        )   

        # トレーナーの作成
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            max_seq_length=128,
            formatting_func=prompt_format.formatting_training_prompts_func,
            args=args
        )
    
        # 学習の実行
        trainer.train()
    
        return trainer
    
    #----------------------------------------------------------
    # トレーニングデータを作成する
    #----------------------------------------------------------
    def create_train_dataset(self, dataset_name, option=""):
        
        # データセットの読み込み読み込み
        dataset = load_dataset(dataset_name)
    
        # 確認
        #print(dataset)
        #print(dataset["train"][0])
    
        # データセットをinputが空の要素のみでフィルタリング
        train_dataset = dataset["train"].filter(lambda data: data["input"] == "")

        # 確認
        #print(train_dataset)
        #print(train_dataset[0])
    
        return train_dataset