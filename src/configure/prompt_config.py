

#======================================================================
# プロンプトの種類を指定しておく
# 辞書型 {group1: [name1, name2, ...], group2: [name1, name2, ...]}
#======================================================================
PROMPT_DICT = {
    # for rinna base model
    "rinna-instruction-prompt": {"user_tag": "ユーザー:", "system_tag": "システム:", "new_line_tag": "<NL>", "end_of_string": ""},
    
    # for line base model
    "line-corporation-instruction-prompt": {"user_tag": "ユーザー:", "system_tag": "システム:", "new_line_tag": "\n", "end_of_string": ""},
    
    # for cyberagent base model
    "cyberagent-base-prompt": {"user_tag": "", "system_tag": "", "new_line_tag": "", "end_of_string": "<|endoftext|>"},
    "cyberagent-instruction-prompt": {"user_tag": "USER:", "system_tag": "ASSISTANT:", "new_line_tag": "\n", "end_of_string": "<|endoftext|>"},
}