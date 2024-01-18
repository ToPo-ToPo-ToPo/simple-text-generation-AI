
import torch
from transformers import T5Tokenizer, AutoModelForCausalLM
from llm.prompt import PromptInstructionTuningModel
#====================================================================
# rinnaのベースを管理するクラス
#====================================================================
class RinnaInstructionTuningModel:
    
    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self, model_name, processor, load_bit_size, load_in_8bit=False, load_in_4bit=False):

        #
        self.tokenizer = T5Tokenizer.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            use_fast=False
        )
        
        # モデルの設定
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            device_map=processor, 
            torch_dtype=load_bit_size, 
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit
        )

        # プロンプトの設定
        self.prompt = PromptInstructionTuningModel(
            user_tag="ユーザー:",
            system_tag="システム:",
            new_line_tag="<NL>"
        )
    
    #----------------------------------------------------------
    # プロンプトの設定
    #----------------------------------------------------------
    def generate_prompt(self, question):

        prompt = self.prompt.generate(question=question)

        return prompt
    
    #----------------------------------------------------------
    # 入力したpromptに対して回答を作成
    #----------------------------------------------------------
    def response(self, prompt):

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


