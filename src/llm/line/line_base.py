
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer
#====================================================================
# lineのベースを管理するクラス
#====================================================================
class LineSft3_6b:

    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self, processor, load_bit_size, load_in_8bit=False, load_in_4bit=False):

        # モデル名の設定
        MODEL_NAME = "line-corporation/japanese-large-lm-3.6b-instruction-sft"

        #
        self.tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, use_fast=False)
        
        # モデルの設定
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            device_map=processor, 
            torch_dtype=load_bit_size, 
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )

    #----------------------------------------------------------
    # プロンプトの設定
    #----------------------------------------------------------
    def generate_prompt(self, question=None):

        prompt = f"ユーザー: {question}\nシステム: "

        return prompt
    
    #----------------------------------------------------------
    # 入力したpromptに対して回答を作成
    #----------------------------------------------------------
    def response(self, prompt=None):

        token_ids = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                do_sample=True,
                max_new_tokens=128,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                #bos_token_id=self.tokenizer.bos_token_id,
                #eos_token_id=self.tokenizer.eos_token_id
            )

        output = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True)

        return output