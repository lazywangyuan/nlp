import gradio as gr


def radio_demo(choice):
    # 这是一个列表
    return f"你选择了: {choice}"


# 创建单选框组件，选项为列表
radio = gr.CheckboxGroup(["山东", "湖南"], label="选择地区查看召回模型结果结果")

# 创建一个界面，并添加单选框组件和回调函数
gr.Interface(fn=radio_demo, inputs=radio, outputs="text").launch()