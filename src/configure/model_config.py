

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
    "line-corporation-base-model": [
        "line-corporation/japanese-large-lm-1.7b",
    ],
    "line-corporation-instruction-model": [
        "ToPo-ToPo/line-japanese-large-lm-1.7b-kunishou-databricks-dolly-15k-ja-full-instruction-sft",
        "line-corporation/japanese-large-lm-3.6b-instruction-sft",
        "line-corporation/japanese-large-lm-3.6b-instruction-sft-4bit-128g-actorder_False",
        "my-lora-aituber-model-based-line-3.6b-sft-v2",
    ],
    "cyberagent-base-model": [
        "cyberagent/open-calm-small", 
    ],
    "cyberagent-instruction-model": [
        "cyberagent/calm2-7b-chat",
        "ToPo-ToPo/cyberagent-open-calm-small-ToPo-ToPo-databricks-dolly-15k-ja-zundamon-full-instruction-tuning",
        "ToPo-ToPo/cyberagent-open-calm-small-kunishou-databricks-dolly-15k-ja-full-instruction-tuning",
        "ToPo-ToPo/cyberagent-open-calm-small-kunishou-databricks-dolly-15k-ja-LoRA-instruction-sft",
    ],
}