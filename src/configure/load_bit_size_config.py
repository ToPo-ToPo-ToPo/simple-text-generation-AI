
import platform
import torch
#======================================================================
# 使用できるロード時のbitサイズの種類を指定しておく
# load_in_4bitはWindowsで使用できなかったため削除した
#======================================================================
# macの場合
if platform.system() == 'Darwin':
    if torch.backends.mps.is_available():
        LOAD_BIT_SIZE_DICT = {
            "auto" : ["float32"],
            "mps" : ["float32"],
            "cpu" : ["float32", "bfloat16", "float16"],
        }
    else:
        LOAD_BIT_SIZE_DICT = {
            "auto" : ["float32"],
            "cpu" : ["float32", "bfloat16", "float16"],
        }    
        
# Windows or Linuxの場合   
else:
    if torch.cuda.is_available():
        LOAD_BIT_SIZE_DICT = {
            "auto" : ["float32", "bfloat16", "float16", "load_in_8bit"],
            "cuda" : ["float32", "bfloat16", "float16", "load_in_8bit"],
            "cpu" : ["float32", "bfloat16", "float16"],
        }
    else:
        LOAD_BIT_SIZE_DICT = {
            "auto" : ["float32", "bfloat16", "float16"],
            "cpu" : ["float32", "bfloat16", "float16"],
        }
        