
import sys
sys.path.append('../src/')

# モデルのキャッシュパスの変更
import os
os.environ['TRANSFORMERS_CACHE'] = '../models/cache/'

from UI.chatbot_gradio import ChatBotGradioUi
#======================================================================
# メインプログラム
#======================================================================
if __name__ == '__main__':

    # chatbotの生成
    gradio_ui = ChatBotGradioUi()

    # webブラウザを起動する
    gradio_ui.demo.queue()
    gradio_ui.demo.launch(share=False, inbrowser=True)