
from llm.rinna_instruction_tuning_model import RinnaInstructionTuningModel
from llm.line_instruction_tuning_model import LineInstructionTuningModel
from llm.cyberagent_base_model import CyberagentBaseModel
from llm.cyberagent_instruction_tuning_model import CyberagentInstructionTuningModel
from llm.prompt import PromptInstructionTuningModel
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
    def create(self, model_group, model_name, processor, load_bit_size, load_in_8bit=False, load_in_4bit=False, llm_int8_enable_fp32_cpu_offload=False):
        
        # 入力されたmodel名からLLMを作成する
        if model_group == "rinna-instruction-model":
            return RinnaInstructionTuningModel(
                model_name=model_name,
                processor=processor, 
                load_bit_size=load_bit_size, 
                load_in_8bit=load_in_8bit, 
                load_in_4bit=load_in_4bit
            )
        
        elif model_group == "line-corporation-instruction-model":
            return LineInstructionTuningModel(
                model_name=model_name,
                processor=processor, 
                load_bit_size=load_bit_size, 
                load_in_8bit=load_in_8bit, 
                load_in_4bit=load_in_4bit
            )
        
        elif model_group == "cyberagent-base-model":
            return CyberagentBaseModel(
                model_name=model_name,
                processor=processor, 
                load_bit_size=load_bit_size, 
                load_in_8bit=load_in_8bit, 
                load_in_4bit=load_in_4bit, 
                llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload
            )

        elif model_group == "cyberagent-instruction-model":
            return CyberagentInstructionTuningModel(
                model_name=model_name,
                processor=processor, 
                load_bit_size=load_bit_size, 
                load_in_8bit=load_in_8bit, 
                load_in_4bit=load_in_4bit, 
                llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload
            )
        
        else:
            print("エラー: 不明なモデル名が入力されました。再確認してください。")
            exit(1)
