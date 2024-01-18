
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llm.prompt import PromptBaseModel
#====================================================================
# Modelの挙動を管理するクラス
#====================================================================
class CyberagentBaseModel:

    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self, model_name, processor, load_bit_size, load_in_8bit=False, load_in_4bit=False, llm_int8_enable_fp32_cpu_offload=False):

        #
        self.name = model_name

        #
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

        # モデルの設定
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=model_name, 
            device_map=processor, 
            torch_dtype=load_bit_size, 
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit, 
            offload_folder="../temp/offload_folder"
        )
        
        # プロンプトの設定
        self.prompt_format = PromptBaseModel()
    
    #----------------------------------------------------------
    # プロンプトの設定
    #----------------------------------------------------------
    def generate_prompt(self, question=None):

        prompt = self.prompt_format.generate(question=question)

        return prompt
    
    #----------------------------------------------------------
    # 入力したpromptに対して回答を作成
    #----------------------------------------------------------
    def response(self, prompt=None):

        token_ids = self.tokenizer.encode(prompt, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                do_sample=True,
                max_new_tokens=256,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id
            )

        output = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):], skip_special_tokens=True)

        return output