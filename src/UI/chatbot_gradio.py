
import platform
import time
import torch
import gradio as gr
from llm.model_factory import ModelFactory
from training.fine_tuning import FineTuning
from training.instruction_fine_tuning import InstructionFineTuning
from llm.prompt import PromptInstructionTuningModel
from configure.config import MODEL_DICT, PROCESSOR_LIST, LOAD_BIT_SIZE_LIST, LOAD_BIT_SIZE_LIST_MPS, LOAD_BIT_SIZE_LIST_CPU
#======================================================================
# UIの基本クラス
#======================================================================
class ChatBotGradioUi():

    #-----------------------------------------------------------
    # コンストラクタ
    # 基本情報を設定
    #-----------------------------------------------------------
    def __init__(self):
        
        # 言語モデル関係の変数を初期化
        self.llm = None
        self.info = ""
        self.train_info = ""
        
        self.train_method = None
        self.train_dataset = None
        
        # UIの基本設定
        self.css = """footer {visibility: hidden}"""

        # gradioのメイン画面を生成する
        self.set_window_form()
    
    #----------------------------------------------
    # イベントリスナー
    # テキストボックスに入力された文字を取得し、履歴に追加する
    # 追加された履歴を返す
    #----------------------------------------------
    def user(self, message, chat_history):
        
        # 履歴に入力されたメッセージを追加する
        chat_history.append((message, None))
        
        # テキストボックスmsgには""を返す
        # chatbotには更新した履歴を返す
        return "", chat_history
    
    #----------------------------------------------
    # 回答を返す関数
    #----------------------------------------------
    def bot(self, chat_history):
        
        # 入力した文字を取得
        question = chat_history[-1][0]

        # promptを生成する
        prompt = self.llm.generate_prompt(question=question)
        
        # 入力した文字の感情分析を行う
        #self.interface.emotion_analysis(question)
        
        # 会話文の入力と回答の取得
        response = self.llm.response(prompt=prompt)
        
        # テキストファイルの会話の内容を保存
        #self.interface.memorize(question, response)

        # 履歴に作成した返答を追加する
        chat_history[-1][1] = ""
        for charactor in response:

            # 文字を履歴に追加していく
            chat_history[-1][1] += charactor

            # 文字を返す
            time.sleep(0.05)
            yield chat_history
    
    #-----------------------------------------------------------
    # blocksの作成
    #-----------------------------------------------------------
    def set_window_form(self):

        # UIの定義
        with gr.Blocks(css=self.css) as self.demo:
            # LLM設定タブを定義
            with gr.Tab("Model"):
                self.set_model_ui()
            
            # チャットボットのタブを定義
            with gr.Tab("Chat"):
                chatbot = gr.Chatbot(
                    elem_id="chatbot",
                    elem_classes="chatbot",
                    container=False,
                    height=500,
                    bubble_full_width=False,
                )

                # メッセージを入力するためのテキストボックスを生成
                msg = gr.Textbox(placeholder="Please enter a message.", container=False, scale=1)

                # チャットの内容を全て消去するためのボタンを生成
                clear = gr.ClearButton([msg, chatbot])
            
                # Enterをされた時の動作
                # テキストボックスに何も表示されていない時は空文字が入力される
                msg.submit(fn=self.user, inputs=[msg, chatbot], outputs=[msg, chatbot], queue=False).then(self.bot, chatbot, chatbot)

            # ファインチューニング用のタブを定義
            with gr.Tab("Training"):
                self.set_train_ui()

    #-----------------------------------------------------------
    # 使用するモデルの登録のUI
    #-----------------------------------------------------------
    def set_model_ui(self):

        # UIの設定
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    # 使用するベースのModelを選択
                    model_type_choice = gr.Dropdown(label="1. Model type", info="Please select the LLM.", choices=list(MODEL_DICT.keys()), value=None)

                    # 使用するModelの選択
                    model_choice = gr.Dropdown(label="1. Model", info="Please select the LLM.", choices=[], value=None)

                    # 使用するアーキテクチャの設定
                    processor_choice = gr.Radio(label="2. Processor type", choices=PROCESSOR_LIST, value=None)

                    # モデルをロードする際のbitサイズの設定
                    load_bit_size_choice = gr.Radio(label="3. Load bit size", info="Select the bit size for model loading.", choices=LOAD_BIT_SIZE_LIST)

                # 初期状態のテキストボックスを配置
                model_info_text = gr.Textbox(label="Model information", lines=10, interactive=True, show_copy_button=True)

            # モデル情報を送信するボタンを配置
            submit_btn = gr.Button("4. Submit", variant="primary")
        
        #-----------------------------------------------------------
        # モデルのタイプに応じて、選択できるモデルの表示を変える
        #-----------------------------------------------------------
        @model_type_choice.change(inputs=model_type_choice, outputs=model_choice)
        def update_model_list(model_type_choice):
            
            model_list = list(MODEL_DICT[model_type_choice])
            
            return gr.Dropdown(choices=model_list, value=model_list[0], interactive=True)

        #-----------------------------------------------------------
        # 選択内容に応じてテキストボックスの表示を変える
        #-----------------------------------------------------------
        @processor_choice.change(inputs=processor_choice, outputs=load_bit_size_choice)
        def update_bit_size_radio(processor_choice):

            if processor_choice == "cuda":
                return gr.Radio(choices=LOAD_BIT_SIZE_LIST, interactive=True)
            
            elif processor_choice == "mps":
                return gr.Radio(choices=LOAD_BIT_SIZE_LIST_MPS, interactive=True)
            
            elif processor_choice == "cpu":
                return gr.Radio(choices=LOAD_BIT_SIZE_LIST_CPU, interactive=True)
            
            elif processor_choice == "auto":
                 pf = platform.system()
                 if pf == 'Darwin':
                     return gr.Radio(choices=LOAD_BIT_SIZE_LIST_MPS, interactive=True)
                 else:
                     return gr.Radio(choices=LOAD_BIT_SIZE_LIST, interactive=True)
            
            else:
                return gr.Radio(interactive=False)

        #　送信ボタンクリック時の動作を設定
        #　引数：model_name, processor_radio, load_bit_size_radioを関数set_model()に入力
        submit_btn.click(fn=self.set_model, inputs=[model_choice, processor_choice, load_bit_size_choice], outputs=model_info_text)

    #-----------------------------------------------------------
    # 使用するモデルの登録
    #-----------------------------------------------------------
    def set_model(self, model_choice, processor_choice, load_bit_size_choice):
        
        # データの初期化
        load_in_8bit = False
        load_in_4bit = False
        llm_int8_enable_fp32_cpu_offload=False

        # bitサイズの設定
        if load_bit_size_choice == "float32":
            load_bit_size = torch.float32
        
        elif load_bit_size_choice == "bfloat16":
            load_bit_size = torch.bfloat16
        
        elif load_bit_size_choice == "float16":
            load_bit_size = torch.float16
        
        elif load_bit_size_choice == "load_in_8bit":
            load_bit_size = torch.float16
            load_in_8bit = True
            llm_int8_enable_fp32_cpu_offload=True

        elif load_bit_size_choice == "load_in_4bit":
            load_bit_size = torch.float16
            load_in_4bit = True
        
        else:
            self.info += "The loading bit size is not set up. Please check again.\n"
            yield gr.Textbox(visible=True, value=self.info)
        
        # モデルが設定されているか確認
        # 設定されていない場合
        if model_choice == []:
            self.info += "The submit button was pressed, but the model is not set up. Please check again.\n"
            yield gr.Textbox(visible=True, value=self.info)
        
        # 設定されている場合
        else:
            # 表示する情報を作成
            self.info += "---------------------------------------------------------\n"
            self.info += "Model information\n"
            self.info += "---------------------------------------------------------\n"
            self.info += "Model: " + model_choice + "\n"
            self.info += "Processor: " + processor_choice + "\n"
            self.info += "Load bit size: " + load_bit_size_choice + "\n\n"
            self.info += "The model is now loading. Please wait a moment.....\n"
        
            # 設定されたモデルの情報を表示
            yield gr.Textbox(visible=True, value=self.info)
    
            # llmの具体的な設定
            model_factory = ModelFactory()
            self.llm = model_factory.create(
                name=model_choice, 
                processor=processor_choice, 
                load_bit_size=load_bit_size, 
                load_in_8bit=load_in_8bit,
                load_in_4bit=load_in_4bit,
                llm_int8_enable_fp32_cpu_offload=llm_int8_enable_fp32_cpu_offload
            )

            # 表示する情報を作成
            self.info += "Model loading is complete. Let's start a conversation.\n"
        
            # モデルの読み込み結果を表示
            yield gr.Textbox(visible=True, value=self.info)
    
    #-----------------------------------------------------------
    # 学習に関する情報登録のUI
    #-----------------------------------------------------------
    def set_train_ui(self):
        
        # UIの設定
        with gr.Blocks() as demo:
            with gr.Row():
                with gr.Column():
                    # 学習の種類を設定
                    train_type_choice = gr.Radio(label="1. Training type", choices=["Base", "Instruction Tuning"], value=None)
                    
                    # 学習の方法を設定
                    train_method_choice = gr.Radio(label="2. Training method", choices=["Full Fine Tuning", "LoRA"], value=None)
                    
                    # 学習用のデータセットの種類を選択
                    datastes_choice = gr.Dropdown(label="3. Datasets", info="Please select the datasets.", choices=["kunishou/databricks-dolly-15k-ja", "etc"], value=None)

                # 初期状態のテキストボックスを配置
                model_info_text = gr.Textbox(label="Train setting information", lines=10, interactive=True, show_copy_button=True)

            # 学習情報を送信するボタンを配置
            train_data_submit_btn = gr.Button("4. Train data submit")
            
            # 学習済みのモデル名を入力するテキストボックスを配置
            trained_model_name_text = gr.Textbox(label="5. Trained model name", interactive=True, show_copy_button=True)
            
            # 学習を開始するボタンを配置
            # 全ての設定が完了次第、反応するようにしたい
            train_start_btn = gr.Button("6. Training Start", variant="primary")
            
        #　送信ボタンクリック時の動作を設定
        train_data_submit_btn.click(fn=self.set_train_condition, inputs=[datastes_choice, train_type_choice, train_method_choice], outputs=model_info_text)
        
        #　送信ボタンクリック時の動作を設定
        train_start_btn.click(fn=self.training, inputs=[trained_model_name_text], outputs=model_info_text)
        
        
    #-----------------------------------------------------------
    # 使用するモデルの登録
    #-----------------------------------------------------------
    def set_train_condition(self, datasets_choice, train_type_choice, train_method_choice):
        
        # 学習方法を設定
        if train_method_choice == "Full Fine Tuning":
            
            if train_type_choice == "Base":
                self.train_method = FineTuning(
                    tokenizer=self.llm.tokenizer, 
                    model=self.llm.model
                )
            
            elif train_type_choice == "Instruction Tuning":
                self.train_method = InstructionFineTuning(
                    tokenizer=self.llm.tokenizer, 
                    model=self.llm.model
                )
                
                self.prompt_format = PromptInstructionTuningModel(
                    user_tag="ユーザー:", 
                    system_tag="システム:"
                )

            else:
                return self.train_info + "学習方式が設定されていません"
        
        elif train_method_choice == "LoRA":
            return self.train_info + "LoRAはまだ実装されていません"
        
        else:
            return self.train_info + "学習手法が設定されていません"
        
        self.train_info += "Training method: " + train_method_choice + "\n"
        self.train_info += "Training type: " + train_type_choice + "\n"
        self.train_info += "Datasets: " + datasets_choice + "\n"
        yield gr.Textbox(visible=True, value=self.train_info)

        self.train_info += "Creating a dataset for training. Please wait a moment.....\n"
        yield gr.Textbox(visible=True, value=self.train_info)
        
        # 学習用のデータセットを作成
        self.train_dataset = self.train_method.create_train_dataset(dataset_name=datasets_choice)
        
        self.train_info += "The dataset for training is complete! The training can now begin!\n"
        yield gr.Textbox(visible=True, value=self.train_info)
    
    
    #-----------------------------------------------------------
    # 使用するモデルの登録
    #-----------------------------------------------------------
    def training(self, trained_model_name_text):
        
        self.train_info += "Currently training. Please wait a moment.....\n"
        yield gr.Textbox(visible=True, value=self.train_info)

        # トレーニングを行う
        self.trainer = self.train_method.training(
            tokenizer=self.llm.tokenizer, 
            model=self.llm.model,
            prompt_format=self.prompt_format, 
            train_dataset=self.train_dataset
        )

        self.train_info += "Training is complete.\n"
        yield gr.Textbox(visible=True, value=self.train_info)
    
        # モデルの保存
        save_name = "../models/" + trained_model_name_text
        self.trainer.save_model(save_name)

        self.train_info += "The results of the training have been saved.\n"
        yield gr.Textbox(visible=True, value=self.train_info)
