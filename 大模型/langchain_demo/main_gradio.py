import gradio as gr

with gr.Blocks() as demo:
    with gr.Tab("政务问答"):
        gr.HTML("""<h1 align="center">政务问答</h1>""")
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=4):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=5, container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("提交")
                    emptyBtn = gr.Button("清空历史")
        submitBtn.click(predict, user_input, chatbot)
        emptyBtn.click(lambda: None, None, chatbot, queue=False)

    with gr.Tab("上传数据"):
        with gr.Row():
            add_name = gr.Textbox(label="数据库中的文件名称",
                                  value=original_document_ids, lines=10)
            files = gr.File(label="添加文件", show_label=True)
            outputUploadKn = gr.Textbox(label="数据详情")
        with gr.Row():
            btnUploadKn = gr.Button(value="上传文件", visible=True)
            greet_btn = gr.Button("页面刷新")
            greet_btn.click(fn=updata_data, outputs=add_name)
        with gr.Row():
            resolving = gr.Textbox(label="解析过程")
        btnUploadKn.click(decide_data_type, [files], [outputUploadKn, resolving, add_name])

    with gr.Tab("删除数据"):
        with gr.Row():
            with gr.Column():
                name = gr.Textbox(label="数据库中的文件名称",
                                  value=original_document_ids, lines=10)
            with gr.Column():
                isin_name = gr.Textbox(label="输入数据库中存在的文件名称")
                greet_btn = gr.Button("删除数据")
                greet_btn.click(fn=del_data, inputs=isin_name, outputs=name)
                greet_btn = gr.Button("页面刷新")
                greet_btn.click(fn=updata_data, outputs=name)

    with gr.Tab("模型问答"):
        chatbot = gr.Chatbot()
        with gr.Row():
            with gr.Column(scale=4):
                with gr.Column(scale=4):
                    user_input = gr.Textbox(show_label=False, placeholder="Input...", lines=5, container=False)
                with gr.Column(min_width=32, scale=1):
                    submitBtn = gr.Button("提交")
                    emptyBtn = gr.Button("清空历史")
        submitBtn.click(predict_normal, user_input, chatbot)
        emptyBtn.click(lambda: None, None, chatbot, queue=False)

    with gr.Tab("简化标题"):
        gr.Markdown("简化标题")
        # chatbot_kn = gr.Chatbot(label="会话历史")
        sheet_data = gr.State()
        col = gr.State()
        # inputKn = gr.Textbox(label="用户问题")
        with gr.Row():
            one_text = gr.Textbox(label="输入需要简化语句")
        with gr.Row():
            one_output = gr.Textbox(label="分析结果")
        with gr.Row():
            submitBtnKn_one_text = gr.Button("提交", variant="primary")

        with gr.Row():
            files = gr.File(label="添加文件", show_label=True)
            outputUploadKn = gr.Textbox(label="文件中存在的列名称")
        with gr.Row():
            btnUploadKn = gr.Button(value="上传文件", visible=True)
            btnUploadKn.click(process_pdf, [files], [outputUploadKn])
        with gr.Row():
            col_text = gr.Textbox(label="输入需要简化的列名称")
        with gr.Row():
            submitBtnKn = gr.Button("提交", variant="primary")
        with gr.Row():
            # output_model = gr.Textbox(label="经过大模型输出结果")
            output = gr.components.File(label="Download Result")
            # emptyBtnKn = gr.Button("清空历史")
            # emptyBtnKn.click(reset_state, outputs=[history_kn], show_progress=True)
        submitBtnKn.click(gen_with_kn, [col_text], [output])
        submitBtnKn_one_text.click(gen_one, [one_text], [one_output])


    def user(query, history):
        print('--------开始-------')
        print("", [history + [[query, ""]]])
        return "", [history + [[query, ""]]]

# demo.queue().launch()
demo.queue().launch(server_name='0.0.0.0', server_port=7569, share=False, inbrowser=True)
