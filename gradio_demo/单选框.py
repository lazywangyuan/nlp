import gradio as gr


def radio_demo(choice):
    return f"你选择了: {choice}"


# 创建单选框组件，选项为列表
radio = gr.Radio(['小红', '小白', '小胖', '小黑'], label='name', value='小红')

# 创建一个界面，并添加单选框组件和回调函数
gr.Interface(fn=radio_demo, inputs=radio, outputs="text").launch()