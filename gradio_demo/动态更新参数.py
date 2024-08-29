import gradio as gr

# 创建一个状态变量来存储Textbox的值
value_state = gr.State(value="")

def update_textbox(new_value):
    # 更新状态变量的值
    value_state.value = new_value
    return new_value

with gr.Blocks() as demo:
    with gr.Row():
        # 创建Textbox组件，并将其值绑定到value_state状态变量
        textbox = gr.Textbox(value=value_state, label="Dynamic Textbox")
        textbox_new = gr.Textbox(value='我测试下', label="Dynamic Textbox")

        # 创建一个按钮，当点击时，会调用update_textbox函数来更新Textbox的值
        update_button = gr.Button("Update Textbox")

        # 设置按钮的动作，当点击时，调用update_textbox函数
        update_button.click(fn=update_textbox, inputs=textbox_new, outputs=textbox)

demo.launch()
