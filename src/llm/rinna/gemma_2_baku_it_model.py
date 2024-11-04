
import platform
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
#====================================================================
# 
#====================================================================
class Gemma2BakuInstructionModel:
    
    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self, model_name, processor, load_bit_size, load_in_8bit=False, load_in_4bit=False):

        #
        self.name = model_name
        self.processor = processor

        #
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # モデルの設定
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map=processor, 
            torch_dtype=load_bit_size,
            attn_implementation="eager",
        )
    
    #----------------------------------------------------------
    # デストラクタ
    #----------------------------------------------------------
    '''def __del__(self):
        
        # CPUに保存されたメモリを解放する
        del self.tokenizer
        del self.model

        # GPUに保存されたメモリを解放する
        if self.processor == "cuda":
            torch.cuda.empty_cache()
        
        elif self.processor == "mps":
            torch.mps.empty_cache()
        
        elif self.processor == "auto":
            if platform.system() == 'Darwin':
                torch.mps.empty_cache()
            else:
                torch.cuda.empty_cache()'''

    
    #----------------------------------------------------------
    # プロンプトの設定
    #----------------------------------------------------------
    def generate_prompt(self, question):

        #prompt = self.prompt_format.generate(question=question)
        chat = [
            {"role": "user", "content": question},
        ]
        
        prompt = self.tokenizer.apply_chat_template(
            chat, 
            tokenize=False, 
            add_generation_prompt=True
        )

        return prompt
    
    #----------------------------------------------------------
    # 入力したpromptに対して回答を作成
    #----------------------------------------------------------
    def response(self, prompt):

        input_ids = self.tokenizer.encode(
            prompt, 
            add_special_tokens=False, 
            return_tensors="pt"
        ).to(self.model.device)
        
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=512,
        )

        response = self.tokenizer.decode(
            outputs[0][input_ids.shape[-1]:], 
            skip_special_tokens=True
        )

        return response
