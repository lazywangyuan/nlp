import gradio as gr


def greet(name):
    return "Hello " + name + "!"


all_list = ['2021年浦东新区科技发展基金“海博计划”创新创业青年人才资助专项申报指南.docx',
            '2023年度浦东新区关于促进技能人才发展专项申报指南.docx',
            '2023年浦东新区“明珠计划”高峰人才项目申报指南.docx',
            '2023年浦东新区“明珠计划”工程师项目申报指南.docx',
            '2021年浦东新区科技发展基金“海博计划”创新创业青年人才资助专项申报指南.docx']
with gr.Blocks() as demo:
    with gr.Row():
        with gr.Column():
            name = gr.Textbox(label="数据库中的文件名称",
                              value='\n'.join(all_list))
        with gr.Column():
            select_doc = gr.Dropdown(
                all_list, label="数据库中的文件名称"
            )
            greet_btn = gr.Button("删除数据")
            greet_btn.click(fn=greet, inputs=name, outputs=[], api_name="greet")

if __name__ == "__main__":
    demo.launch()
