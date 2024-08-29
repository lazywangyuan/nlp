import gradio as gr
import time

def predict(user_input):
    res=''
    for i in '测试结果':
        time.sleep(1)
        res=res+i
        yield [['我试试',res]]


with gr.Blocks() as demo:
    gr.HTML("""<h1 align="center">Demo</h1>""")
    chatbot = gr.Chatbot()
    with gr.Row():
        with gr.Column(scale=4):
            with gr.Column(scale=12):
                user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=10, container=False)
            with gr.Column(min_width=32, scale=1):
                submitBtn = gr.Button("Submit")
        with gr.Column(scale=1):
            emptyBtn = gr.Button("Clear History")


    def user(query, history):
        return "", [history + [query, ""]]


    submitBtn.click(user, [user_input, chatbot], [user_input, chatbot], queue=False).then(
        predict, [user_input,chatbot], chatbot
    )
    emptyBtn.click(lambda: None, None, chatbot, queue=False)

demo.queue()
demo.launch()
