
from llm.rinna.rinna_gpt_neox_3b import RinnaGptNeox3b
from llm.line.line_base import LineSft3_6b
from llm.cyberagent.open_calm_small import OpenCalmSmall
from llm.cyberagent.calm2_7b import Calm2_7b_Chat
from llm.aituber_mal.aituber_mal_base import AituberMalBase
#====================================================================
# 各種LLMを生成するクラス
#====================================================================
class ModelFactory:

    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self):
        pass
        
    #----------------------------------------------------------
    # LLMのモデルを生成し、返す
    #----------------------------------------------------------
    def create(self, name, processor, load_bit_size, load_in_8bit=False, load_in_4bit=False, llm_int8_enable_fp32_cpu_offload=False):
        
        # 入力されたmodel名からLLMを作成する
        if name == "rinna/japanese-gpt-neox-3.6b-instruction-ppo" or name == "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2" or "rinna/japanese-gpt-neox-3.6b-instruction-sft":
            return RinnaGptNeox3b(
                model_name=name,
                processor=processor, 
                load_bit_size=load_bit_size, 
                load_in_8bit=load_in_8bit, 
                load_in_4bit=load_in_4bit
            )
        
        elif name == "line-corporation/japanese-large-lm-3.6b-instruction-sft":
            return LineSft3_6b(
                model_name=name,
                processor=processor, 
                load_bit_size=load_bit_size, 
                load_in_8bit=load_in_8bit, 
                load_in_4bit=load_in_4bit
            )
        
        elif name == "cyberagent/open-calm-small":
            return OpenCalmSmall(
                model_name=name,
                processor=processor, 
                load_bit_size=load_bit_size, 
                load_in_8bit=load_in_8bit, 
                load_in_4bit=load_in_4bit, 
                llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload
            )

        elif name == "cyberagent/calm2-7b-chat":
            return Calm2_7b_Chat(
                model_name=name,
                processor=processor, 
                load_bit_size=load_bit_size, 
                load_in_8bit=load_in_8bit, 
                load_in_4bit=load_in_4bit, 
                llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload
            )
        
        elif name == "ToPo-ToPo/my-lora-aituber-model-based-line-3.6b-sft-v2":
            return AituberMalBase(
                model_name=name,
                processor=processor, 
                load_bit_size=load_bit_size, 
                load_in_8bit=load_in_8bit, 
                load_in_4bit=load_in_4bit
            )
        
        else:
            print("エラー: 不明なモデル名が入力されました。再確認してください。")
            exit(1)
