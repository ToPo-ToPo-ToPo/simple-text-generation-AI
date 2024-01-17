

#====================================================================
# promptの設定を管理するクラス
#====================================================================
class PromptBaseModel:
    
    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self):
        pass
    
    #----------------------------------------------------------
    # 入力された文章の続きを生成するタイプ
    #----------------------------------------------------------
    def generate(self, question):

        prompt = f"{question}"

        return prompt


#====================================================================
# promptの設定を管理するクラス
#====================================================================
class PromptInstructionTuningModel:
    
    #----------------------------------------------------------
    # コンストラクタ
    #----------------------------------------------------------
    def __init__(self):
        pass
    
    #----------------------------------------------------------
    # プロンプトの設定
    #----------------------------------------------------------
    def generate(self, question, input=None):

        if input == None:
            prompt = f"USER: {question}\nASSISTANT: "
        else:
            prompt = f"INPUT: {input}\nUSER: {question}\nASSISTANT: "

        return prompt
    
    #----------------------------------------------------------
    # 指示文付き学習用のプロンプトの設定
    #----------------------------------------------------------
    def formatting_prompts_func(example):
        
        output_texts = []
    
        for i in range(len(example['instruction'])):
            
            # 前置き(input)がない場合
            if example['input'] == "":
                text = f"USER: {example['instruction'][i]}\nASSISTANT: {example['output'][i]}<|endoftext|>"
            
            # 前置き(input)がある場合
            else:
                text = f"INPUT: {example['input'][i]}\nUSER: {example['instruction'][i]}\nASSISTANT: {example['output'][i]}<|endoftext|>"
            
            # データを追加
            output_texts.append(text)
            
        return output_texts