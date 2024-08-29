import gradio as gr
import re


def clear(*args):
    return ""

def greet(content, title):
    if content.strip() == "":
        return "文章内容不能为空，请重新输入。"
    if title.strip() == "":
        return "文章标题不能为空，请重新输入。"
    print('标题:' + title)
    print('内容:' + content)
    content = content.replace(' ', '')
    policy_type = match_policy_type(title)
    release_data, last_data = match_release_data(content)
    department_number, administrative_division, document_number = policy_level(content, last_data)
    period_validity = match_period_of_validity(content)
    return '政策有效期: ' + period_validity + '\n' + '政策类型: ' + policy_type + '\n' + '发布日期: ' + release_data + '\n' + \
           '政策级别: ' + department_number + '\n' + '行政区划: ' + administrative_division + '\n' + '发布部门: ' + document_number


with gr.Blocks() as demo:
    title = gr.Textbox(label="文章标题")
    content = gr.Textbox(label="文章内容")
    output = gr.Textbox(label="分析结果")
    greet_btn = gr.Button("提交")
    greet_btn.click(fn=greet, inputs=[content, title], outputs=output)
    greet_btn = gr.Button("清除")
    for clear_text in [title, output, content]:
        greet_btn.click(fn=clear, inputs=clear_text, outputs=clear_text)

demo.title = '政策字段提取'
# demo.launch()
demo.launch(server_name='10.10.22.219', server_port=7863, root_path="/policy", share=False, inbrowser=True)
# all_test()
