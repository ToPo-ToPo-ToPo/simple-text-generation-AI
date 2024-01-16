
from llm.rinna.rinna_base import RinnaPpo3_6b, RinnaSft3_6b, RinnaSftV2_3_6b
from llm.line.line_base import LineSft3_6b
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
    def create(self, name, processor, load_bit_size, load_in_8bit=False):
        
        # 入力されたmodel名からLLMを作成する
        if name == "rinna/japanese-gpt-neox-3.6b-instruction-ppo":
            return RinnaPpo3_6b(processor=processor, load_bit_size=load_bit_size, load_in_8bit=load_in_8bit)
        
        elif name == "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2":
            return RinnaSftV2_3_6b(processor=processor, load_bit_size=load_bit_size, load_in_8bit=load_in_8bit)
        
        elif name == "rinna/japanese-gpt-neox-3.6b-instruction-sft":
            return RinnaSft3_6b(processor=processor, load_bit_size=load_bit_size, load_in_8bit=load_in_8bit)
        
        elif name == "line-corporation/japanese-large-lm-3.6b-instruction-sft":
            return LineSft3_6b(processor=processor, load_bit_size=load_bit_size, load_in_8bit=load_in_8bit)
        
        elif name == "cyberagent/calm2-7b-chat":
            return Calm2_7b_Chat(processor=processor, load_bit_size=load_bit_size, load_in_8bit=load_in_8bit)
        
        elif name == "ToPo-ToPo/my-lora-aituber-model-based-line-3.6b-sft-v2":
            return AituberMalBase(processor=processor, load_bit_size=load_bit_size, load_in_8bit=load_in_8bit)
        else:
            print("エラー: 不明なモデル名が入力されました。再確認してください。")
            exit(1)
