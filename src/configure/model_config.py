

#======================================================================
# 使用できるモデルの種類を指定しておく
# 辞書型 {group1: [name1, name2, ...], group2: [name1, name2, ...]}
#======================================================================
MODEL_DICT = {
    "rinna-instruction-model": [
        "rinna/japanese-gpt-neox-3.6b-instruction-ppo", 
        "rinna/japanese-gpt-neox-3.6b-instruction-sft-v2",
        "rinna/japanese-gpt-neox-3.6b-instruction-sft",
    ],
    "line-corporation-instruction-model": [
        "line-corporation/japanese-large-lm-3.6b-instruction-sft",
        "line-corporation/japanese-large-lm-3.6b-instruction-sft-4bit-128g-actorder_False",
        "my-lora-aituber-model-based-line-3.6b-sft-v2",
    ],
    "cyberagent-base-model": [
        "cyberagent/open-calm-small", 
    ],
    "cyberagent-instruction-model": [
        "cyberagent/calm2-7b-chat",
        "../models/open-calm-small-databricks-dolly-15k-ja-sft-v2" 
    ],
}