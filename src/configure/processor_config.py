
import platform
import torch
#======================================================================
# 使用できるアーキテクチャの種類を指定しておく
#======================================================================
# OSの情報を取得する
pf = platform.system()

PROCESSOR_LIST = []
# macの場合
if pf == 'Darwin':
    if torch.backends.mps.is_available():
        PROCESSOR_LIST = ["auto", "mps", "cpu"]
    else:
        PROCESSOR_LIST = ["auto", "cpu"]
        
# Windows or Linuxの場合
else:
    if torch.cuda.is_available():
        PROCESSOR_LIST = ["auto", "cuda", "cpu"]
    else:
        PROCESSOR_LIST = ["auto", "cpu"]