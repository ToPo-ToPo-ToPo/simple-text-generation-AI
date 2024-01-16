
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
            "mps", 
            "cpu"
        ]
    else:
        PROCESSOR_LIST = [
            "cpu"
        ]
# Windows or Linuxの場合
else:
    if torch.cuda.is_available():
        PROCESSOR_LIST = [
            "cuda", 
            "cpu"
        ]
    else:
        PROCESSOR_LIST = [
            "cpu"
        ]
#======================================================================
# 使用できるロード時のbitサイズの種類を指定しておく
#======================================================================
LOAD_BIT_SIZE_LIST = [
    "float32", 
    "bfloat16", 
    "float16", 
    "int8"
]
LOAD_BIT_SIZE_LIST_MPS = [
    "float32", 
]