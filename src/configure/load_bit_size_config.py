
import platform
#======================================================================
# 使用できるロード時のbitサイズの種類を指定しておく
# load_in_4bitはWindowsで使用できなかったため削除した
#======================================================================
# OSの情報を取得する
pf = platform.system()

if pf == 'Darwin':
    LOAD_BIT_SIZE_DICT = {
        "auto" : ["float32"],
        "mps" : ["float32"],
        "cpu" : ["float32", "bfloat16", "float16"],
    }
else:
    LOAD_BIT_SIZE_DICT = {
        "auto" : ["float32", "bfloat16", "float16", "load_in_8bit"],
        "cuda" : ["float32", "bfloat16", "float16", "load_in_8bit"],
        "cpu" : ["float32", "bfloat16", "float16"],
}
"""
LOAD_BIT_SIZE_LIST = ["float32", "bfloat16", "float16", "load_in_8bit"]
LOAD_BIT_SIZE_LIST_CPU = ["float32", "bfloat16", "float16"]
LOAD_BIT_SIZE_LIST_MPS = ["float32"]
"""