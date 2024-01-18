
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import BitsAndBytesConfig
from llm.prompt import PromptInstructionTuningModel
#====================================================================
# lineのベースを管理するクラス
#====================================================================
class CyberagentInstructionTuningModel:

    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self, model_name, processor, load_bit_size, load_in_8bit=False, load_in_4bit=False, llm_int8_enable_fp32_cpu_offload=False):

        #
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=model_name)

        # 量子化を使用する場合
        if load_in_8bit == True or load_in_4bit == True:
            # 量子化に関する設定
            quantization_config = BitsAndBytesConfig(llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload)

            # モデルの設定
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name, 
                device_map=processor, 
                torch_dtype=load_bit_size, 
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit, 
                quantization_config=quantization_config,
                offload_folder="../temp/offload_folder"
            )
        # 量子化を使用しない場合
        else:
            # モデルの設定
            self.model = AutoModelForCausalLM.from_pretrained(
                pretrained_model_name_or_path=model_name, 
                device_map=processor, 
                torch_dtype=load_bit_size, 
                offload_folder="../temp/offload_folder"
            )
            
        
        # プロンプトの設定
        self.prompt_format = PromptInstructionTuningModel(
            user_tag="USER:",
            system_tag="ASSISTANT:",
            end_of_string="<|endoftext|>"
        )
    
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

        token_ids = self.tokenizer.encode(prompt, add_special_tokens=True, return_tensors="pt")

        with torch.no_grad():
            output_ids = self.model.generate(
                token_ids.to(self.model.device),
                do_sample=True,
                max_new_tokens=256,
                temperature=0.7,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.pad_token_id,
                bos_token_id=self.tokenizer.bos_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )

        output = self.tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
        output = output.replace("<|endoftext|>", "")

        return output