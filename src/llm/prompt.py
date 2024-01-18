

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
    def __init__(self, user_tag, system_tag, input_tag="INPUT", new_line_tag="\n", end_of_string=""):
        
        self.input_tag = input_tag
        self.user_tag = user_tag
        self.system_tag = system_tag
        self.new_line_tag = new_line_tag
        self.end_of_string = end_of_string
    
    #----------------------------------------------------------
    # プロンプトの設定
    #----------------------------------------------------------
    def generate(self, question, input=""):

        # 前置き(input)がない場合
        if input == "":
            prompt = f"{self.user_tag} {question}{self.new_line_tag}{self.system_tag} "
        
        # 前置き(input)がある場合
        else:
            prompt = f"{self.input_tag} {input}{self.new_line_tag}{self.user_tag} {question}{self.new_line_tag}{self.system_tag} "
        
        print(prompt)

        return prompt
    
    #----------------------------------------------------------
    # 指示文付き学習用のプロンプトの設定
    #----------------------------------------------------------
    def formatting_prompts_func(self, example):
        
        output_texts = []
        
        for i in range(len(example['instruction'])):
            
            # 前置き(input)がない場合
            if example['input'] == "":
                text = f"{self.user_tag} {example['instruction'][i]}{self.new_line_tag}{self.system_tag} {example['output'][i]}{self.end_of_string}"
            
            # 前置き(input)がある場合
            else:
                text = f"{self.input_tag} {example['input'][i]}{self.new_line_tag}{self.user_tag} {example['instruction'][i]}{self.new_line_tag}{self.system_tag} {example['output'][i]}{self.end_of_string}"
            
            # データを追加
            output_texts.append(text)
            
        return output_texts