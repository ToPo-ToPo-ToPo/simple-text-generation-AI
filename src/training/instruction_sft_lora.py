
from transformers import TrainingArguments
from datasets import load_dataset
from peft import LoraConfig, TaskType, prepare_model_for_int8_training
from trl import SFTTrainer
from trl import DataCollatorForCompletionOnlyLM
#===============================================================================
# LoRA学習の基本クラス
#===============================================================================
class InstructionSftLoRA:

    #-------------------------------------------------------------
    # コンストラクタ
    #------------------------------------------------------------- 
    def __init__(self) -> None:
        pass

    #-------------------------------------------------------------
    # LoRAパラメータの準備 
    #-------------------------------------------------------------
    def set_lora_config(self):

        lora_config = LoraConfig(
            r=8, 
            lora_alpha=16,
            target_modules=["query_key_value"],
            lora_dropout=0.05,
            bias="none",
            task_type=TaskType.CAUSAL_LM
        )
        
        return lora_config
    
    #----------------------------------------------------------
    # トレーニングデータを作成する
    #----------------------------------------------------------
    def create_train_dataset(self, dataset_name, VAL_SET_SIZE=4000):
        
        # データセットの準備
        dataset = load_dataset(dataset_name)
        
        # 学習データと検証データの準備
        train_dataset = dataset["train"].filter(lambda data: data["input"] == "")
        
        return train_dataset

    #-------------------------------------------------------------
    # 学習の実行
    #-------------------------------------------------------------
    def training(self, tokenizer, model, prompt_format, train_dataset, NUM_TRAIN_EPOCHS=3, VAL_SET_SIZE=4000):

        # モデルの前処理 8bit in load
        #if model.load_in_8bit() == True:
        #    model = prepare_model_for_int8_training(model)

        args = TrainingArguments(
            output_dir="../temp/train_log",
            num_train_epochs=2,
            gradient_accumulation_steps=8,
            per_device_train_batch_size=8,
            save_strategy="no",
            logging_steps=20,
            lr_scheduler_type="constant",
            save_total_limit=1,
            auto_find_batch_size=True
        )
        
        collator = DataCollatorForCompletionOnlyLM(
            response_template=prompt_format.response_template(), 
            tokenizer=tokenizer
        )

        trainer = SFTTrainer(
            model,
            args=args,
            train_dataset=train_dataset,
            formatting_func=prompt_format.formatting_training_prompts_func,
            max_seq_length=512,
            data_collator=collator,
            peft_config=self.set_lora_config(),
        )

        # 学習の実行
        model.config.use_cache = False
        trainer.train() 
        model.config.use_cache = True
        
        return trainer
        

#===============================================================================
# 学習データのセット
#===============================================================================
class TrainDatasetLoRA:
    
    def __init__(self, train_data, val_data) -> None:
        
        self.train_data = train_data
        self.val_data = val_data
    
    