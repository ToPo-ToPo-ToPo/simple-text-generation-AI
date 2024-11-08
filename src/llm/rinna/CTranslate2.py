
import subprocess

base_model = "rinna/gemma-2-baku-2b-it" #マージモデルの出力先を指定
output_dir = "./temp/gemma-2-baku-2b-it-Ctranslate2"  #CTranslate2への変換先

command = f"ct2-transformers-converter --model {base_model} --output_dir {output_dir}"
subprocess.run(command, shell=True)