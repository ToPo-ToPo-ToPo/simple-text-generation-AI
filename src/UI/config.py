
import platform
import torch
#======================================================================
# 使用できるモデルの種類を指定しておく
#======================================================================
MODEL_LIST = [
    "rinna/japanese-gpt-neox-3.6b-instruction-ppo", 
    "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2",
    "rinna/japanese-gpt-neox-3.6b-instruction-sft",
    "line-corporation/japanese-large-lm-3.6b-instruction-sft",
    "line-corporation/japanese-large-lm-3.6b-instruction-sft-4bit-128g-actorder_False",
    "cyberagent/open-calm-small", 
    "cyberagent/calm2-7b-chat", 
    "ToPo-ToPo/my-lora-aituber-model-based-line-3.6b-sft-v2",
]
#======================================================================
# 使用できるアーキテクチャの種類を指定しておく
#======================================================================
# OSの情報を取得する
pf = platform.system()

PROCESSOR_LIST = []
# macの場合
if pf == 'Darwin':
    if torch.backends.mps.is_available():
        PROCESSOR_LIST = [
            "auto",
            "mps", 
            "cpu"
        ]
    else:
        PROCESSOR_LIST = [
            "auto"
            "cpu"
        ]
# Windows or Linuxの場合
else:
    if torch.cuda.is_available():
        PROCESSOR_LIST = [
            "auto",
            "cuda", 
            "cpu"
        ]
    else:
        PROCESSOR_LIST = [
            "auto",
            "cpu"
        ]
#======================================================================
# 使用できるロード時のbitサイズの種類を指定しておく
# load_in_4bitはWindowsで使用できなかったため削除した
#======================================================================
LOAD_BIT_SIZE_LIST = [
    "float32", 
    "bfloat16", 
    "float16", 
    "load_in_8bit",
    #"load_in_4bit"
]
LOAD_BIT_SIZE_LIST_CPU = [
    "float32", 
    "bfloat16", 
    "float16"
]
LOAD_BIT_SIZE_LIST_MPS = [
    "float32", 
]