
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import T5Tokenizer
#====================================================================
# rinnaのベースを管理するクラス
#====================================================================
class RinnaBase:
    
    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self, model_name, processor, load_bit_size, load_in_8bit=False, load_in_4bit=False):

        #
        self.tokenizer = T5Tokenizer.from_pretrained(model_name, use_fast=False)
        
        # モデルの設定
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, 
            device_map=processor, 
            torch_dtype=load_bit_size, 
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )
    
    #----------------------------------------------------------
    # プロンプトの設定
    #----------------------------------------------------------
    def generate_prompt(self, question=None):

        prompt = [
            {
                "speaker": "ユーザー",
                "text": question
            }
        ]

        prompt = [
            f"{uttr['speaker']}: {uttr['text']}"
            for uttr in prompt
        ]

        prompt = "<NL>".join(prompt)
        prompt = (prompt + "<NL>" + "システム: ")

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
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        output = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True)
        output = output.replace("<NL>", "\n")

        return output

#====================================================================
# rinna ppoモデルを定義するクラス
#====================================================================
class RinnaPpo3_6b(RinnaBase):
    
    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self, processor, load_bit_size, load_in_8bit=False, load_in_4bit=False):

        # モデル名の設定
        MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-ppo"

        # スーパークラスのコンストラクタを適用
        super(RinnaPpo3_6b, self).__init__(
            model_name=MODEL_NAME, 
            processor=processor, 
            load_bit_size=load_bit_size, 
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )

#====================================================================
# rinna sft モデルを定義するクラス
#====================================================================
class RinnaSft3_6b(RinnaBase):
    
    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self, processor, load_bit_size, load_in_8bit=False, load_in_4bit=False):

        # モデル名の設定
        MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-sft"

        # スーパークラスのコンストラクタを適用
        super(RinnaSft3_6b, self).__init__(
            model_name=MODEL_NAME, 
            processor=processor, 
            load_bit_size=load_bit_size, 
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )

#====================================================================
# rinna sft v2モデルを定義するクラス
#====================================================================
class RinnaSftV2_3_6b(RinnaBase):
    
    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self, processor, load_bit_size, load_in_8bit=False, load_in_4bit=False):

        # モデル名の設定
        MODEL_NAME = "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2"

        # スーパークラスのコンストラクタを適用
        super(RinnaSftV2_3_6b, self).__init__(
            model_name=MODEL_NAME, 
            processor=processor, 
            load_bit_size=load_bit_size, 
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )


