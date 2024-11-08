
import ctranslate2
import transformers
import time

base_model = "rinna/gemma-2-baku-2b-it"
quant_model = "./temp/gemma-2-baku-2b-it-Ctranslate2-bfloat16"    #CTranslate2への変換先を指定

#
generator = ctranslate2.Generator(quant_model, device="cuda")

# Rinnaのトークナイザーでは、「use_fast=False」も必要になる
tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, use_fast=False)

# プロンプトを作成する
def prompt(question):
    chat = [
        {"role": "user", "content": question},
    ]
    
    prompt = tokenizer.apply_chat_template(
        chat, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    return prompt

# 返信を作成する
def reply(msg):

    tokens = tokenizer.convert_ids_to_tokens(
        tokenizer.encode(
            prompt(msg),
            add_special_tokens=False,
        )
    )

    results = generator.generate_batch(
        [tokens],
        max_length=256,
        sampling_topk=10,
        sampling_temperature=0.9,
        include_prompt_in_result=False,
    )

    text = tokenizer.decode(
        results[0].sequences_ids[0],
         skip_special_tokens=True
    )
    print("A: " + text)
    return text

if __name__ == "__main__":
    while True:
        msg = input("Q: ")
        start_time = time.time()
        reply(msg)
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Elapsed time: {elapsed_time} seconds")