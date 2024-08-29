import gradio as gr
import joblib
from sentence_transformers import SentenceTransformer
import numpy as np
import pandas as pd
from zhipuai import ZhipuAI
from docx import Document
import pdfplumber
import chromadb
import re
import time

# 请填写您自己的APIKey
client = ZhipuAI(api_key="d9810efee86a4979e5e0c3696cf49f8f.pzC2q5q6PxLF3pLa")

words_list = []
articles = []
# 232
with open('train_pair_sim.pkl', 'rb') as file_cache:
    new = joblib.load(file_cache)
model = SentenceTransformer('/opt/modules/bge-m3')


# 每次去刷新参数
def refresh_chromadb():
    global client_chromadb, original_document, cut_document, original_document_ids
    client_chromadb = chromadb.PersistentClient(path="./")
    original_document = client_chromadb.get_collection(name="original_document")
    cut_document = client_chromadb.get_collection(name="cut_document")
    original_document_ids = '\n'.join(original_document.get()['ids'])


refresh_chromadb()


# 232
def recall_data(text, top):
    embeddings = model.encode(text, normalize_embeddings=True).tolist()
    start_time = time.time()
    original_text = []
    original_score = []
    original_results = original_document.query(
        query_embeddings=[embeddings],
        n_results=top
    )
    for ori_text, score in zip(original_results['documents'][0], original_results['distances'][0]):
        res_score = 1 - score
        if res_score > 0.7:
            original_text.append(ori_text)
            original_score.append(res_score)

    cut_document_results = cut_document.query(
        query_embeddings=[embeddings],
        n_results=top
    )

    # sim原文、置信度、原始文章id
    for sim_text, score, ori_ids in zip(cut_document_results['documents'][0], cut_document_results['distances'][0],
                                        cut_document_results['metadatas'][0]):
        res_score = 1 - score
        if res_score > 0.7:
            original = original_document.get(
                ids=ori_ids['source']
            )
            original_text.append(original['documents'][0])
            original_score.append(res_score)
    # 重排序
    rank_ori_text = []
    sorted_items = sorted(enumerate(original_score), key=lambda item: item[1], reverse=True)
    for ori_text_index, score in sorted_items:
        rank_ori_text.append(original_text[ori_text_index])
    return rank_ori_text


his_text = []
his_answer = []


def predict(question):
    his_text.append(question)
    res_question = question
    prompt_tool = """你是为市民和企业主提供办事服务政策和流程咨询的智能助手，请阅读可参考的文档资料，结合你的专业知识，细致、专业、重点突出地回答用户的问题。
注意：（1）用户问题可能需要判断、总结、结合表格数据推理等，请注意辨别然后做出对应的推理回答。
（2）回答要全面，不要遗漏要点，同时回答要确保与问题相关，不要多回答其他无关信息。
（3）回答要注意总结、分点、换行等格式，保证较好的可读性。
（4）如果根据参考资料无法回答,请给出解释并回复"对不起，没有足够的信息"，不要试图编造参考资料里面没有的内容。
（5）回答尽量严谨。
（6）回答的时候，如果引用了具体的文档内容，不要以”根据【内容片段1】“等方式进行回答，要说出具体的文档名称，如根据《XX》文档内容。
（7）如果回答找不到相关引用文档，不要用基模回答，可以直接回复"对不起，当前没有足够的政策法规知识可以回答您的问题"。
（8）若尝试回答问题的时候，你有了多个知识片段来支撑你的回答，不要融合你的知识来源，尽可能保持知识片段的独立性来回答问题。
（9）回答中不要出现“可能”字眼。"""
    # question = chatbot[:-1][:-1][0]
    recall_ori_list = recall_data(question, 1)
    if len(recall_ori_list) == 1:
        tool_text = recall_ori_list[0]
        res_question = "{0}\n可参考的文档资料：\n{1}用户问题：\n{2}".format(prompt_tool, tool_text, question)
    elif len(recall_ori_list) > 1:
        recall_ori = '------另一篇参考文档------'.join(recall_ori_list)
        res_question = "{0}\n可参考的文档资料：\n{1}用户问题：\n{2}".format(prompt_tool, recall_ori, question)
    elif len(recall_ori_list) == 0:
        res_question = "{0}\n可参考的文档资料:找不到相关引用文档：\n用户问题：\n{1}".format(prompt_tool, question)
    response = client.chat.completions.create(
        model="glm-4",  # 填写需要调用的模型名称
        messages=[{"role": "user", "content": res_question}],
        stream=True,
    )
    res = ''
    try:
        while True:
            value = next(response)
            res = res + value.choices[0].delta.content
            his_answer.append(res)
            yield [[question, res]]
    except StopIteration:
        print("模型回答结果")
        print(res)


def reset_state():
    return [], []


def add_data():
    # 使用 os.path.basename 获取文件名
    # file_name = os.path.basename(path)
    file_name = str(path).replace('/', '\\').split("\\")[-1]
    # 判断文件是否存在
    original = original_document.get(
        where={"source": {"$eq": file_name}},  # 表示 metadata 中 "author" 字段值等于 "jack" 的文档
    )
    # 先将名字转化为向量保存起来
    if original['ids']:
        print('该文件名称重复')
    else:
        # 将原始文件进行切片操作，然后转向量操作
        embeddings_1 = model.encode(file_name, normalize_embeddings=True).tolist()
        original_document.add(
            embeddings=[embeddings_1],
            documents=docx_span,
            metadatas={"source": file_name},
            ids=file_name
        )
        paragraphs = split_text(docx_span, 500)
        for big_i, big_p in enumerate(paragraphs):
            paragraphs_sim = split_text(big_p, 20)
            # 打印结果
            for sim_i, sim_p in enumerate(paragraphs_sim):
                embeddings = model.encode(sim_p, normalize_embeddings=True).tolist()
                cut_document.add(
                    embeddings=[embeddings],
                    documents=sim_p,
                    metadatas={"source": file_name},
                    ids=str(file_name) + '_' + str(big_i) + '_' + str(sim_i)
                )
    return '解析完成'


def read_docx(flag):
    docx_span = ''
    try:
        doc = Document(path)
        full_text = []
        for para in doc.paragraphs:
            full_text.append(para.text)
        docx_span = '\n'.join(full_text)
        flag = 1
    except:
        print('使用原始读取方式')
    if len(docx_span) == 0:
        try:
            with open(path, 'r', encoding='utf-8') as f:
                docx_span = f.read()
                flag = 1
        except:
            print('docx文档类型都无法读取')
    return docx_span, flag


def read_txt(flag):
    with open(path, 'r', encoding='utf-8') as f:
        docx_span = f.read()
        flag = 1
    return docx_span, flag


def read_pdf(flag):
    # 打开PDF文件
    with pdfplumber.open(path) as pdf:
        # 遍历每一页
        docx_span = '\n'.join([page.extract_text() for page in pdf.pages])
        flag = 1
    return docx_span, flag


def read_md(flag):
    with open(path, 'r', encoding='utf-8') as file:
        markdown_text = file.read()
    return markdown_text, flag


def decide_data_type(file_path):
    global path
    path = file_path.name
    global docx_span
    print(path)
    # 根据文件扩展名判断文件类型
    flag = 0
    if (path.find('.docx') != -1) | (path.find('.doc') != -1):
        docx_span, flag = read_docx(flag)
    elif path.find('pdf') != -1:
        docx_span, flag = read_pdf(flag)
    elif path.find('.txt') != -1:
        docx_span, flag = read_txt(flag)
    elif path.find('.md') != -1:
        docx_span, flag = read_md(flag)
    if flag == 0:
        print(path)
        print('没有处理数据')
    res_add = add_data()
    refresh_chromadb()
    return docx_span, res_add, original_document_ids


def split_text(text, max_length):
    result = []
    new_list = []
    # 去掉空格以及换行符号
    text = re.sub(r'\s+', '', text)
    for i in range(0, len(text), max_length):
        result.append(text[i:i + max_length])
    if len(result) >= 2:
        new_list.extend(result[:-2])
        new_list.append(result[-2] + result[-1])
    elif len(result) == 1:
        new_list.extend(result)
    return new_list


def updata_data():
    refresh_chromadb()
    return original_document_ids


def del_data(select_doc_name):
    # 判断文件是否存在
    original = original_document.get(
        ids=select_doc_name
    )
    if original:
        original_document.delete(ids=select_doc_name)
        refresh_chromadb()
        return original_document_ids
    else:
        return '文档不存在，请核对'


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


    def user(query, history):
        print('--------开始-------')
        print("", [history + [[query, ""]]])
        return "", [history + [[query, ""]]]

# demo.queue().launch()
demo.queue().launch(server_name='10.10.22.232', server_port=7568, share=False, inbrowser=True)
# demo.queue().launch(server_name='10.10.22.232', server_port=7542, root_path="/title", share=False, inbrowser=True,
#                    show_api=False)
