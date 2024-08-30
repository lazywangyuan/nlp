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
client = ZhipuAI(api_key="b654a95fa7bc2c7f3b4f4bd2d004aa5e.46PNkluMsjtNiRxt")

words_list = []
articles = []
# 本地
# model = SentenceTransformer(r'D:\models\bge-m3')


# 232
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
    print('------------问题-----------')
    print(question)
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
    print('--------寻找出的文章-------------')
    print(recall_ori_list)
    if len(recall_ori_list) == 1:
        tool_text = recall_ori_list[0]
        res_question = "{0}\n可参考的文档资料：\n{1}用户问题：\n{2}".format(prompt_tool, tool_text, question)
    elif len(recall_ori_list) > 1:
        recall_ori = '------另一篇参考文档------'.join(recall_ori_list)
        res_question = "{0}\n可参考的文档资料：\n{1}用户问题：\n{2}".format(prompt_tool, recall_ori, question)
    elif len(recall_ori_list) == 0:
        res_question = "{0}\n可参考的文档资料:找不到相关引用文档：\n用户问题：\n{1}".format(prompt_tool, question)
    print('--------拼接提示词后的输入-------------')
    print(res_question)
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


def predict_normal(question):
    his_text.append(question)
    print('------------问题-----------')
    print(question)
    res_question = question
    response = client.chat.completions.create(
        model="glm-4-flash",  # 填写需要调用的模型名称
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
    print(file_name)
    # 判断文件是否存在
    original = original_document.get(
        where={"source": {"$eq": file_name}},  # 表示 metadata 中 "author" 字段值等于 "jack" 的文档
    )
    print(original)
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
        print('-----500数据---------')
        print(paragraphs)
        for big_i, big_p in enumerate(paragraphs):
            paragraphs_sim = split_text(big_p, 20)
            print('-----20数据---------')
            print(paragraphs_sim)
            # 打印结果
            for sim_i, sim_p in enumerate(paragraphs_sim):
                embeddings = model.encode(sim_p, normalize_embeddings=True).tolist()
                cut_document.add(
                    embeddings=[embeddings],
                    documents=sim_p,
                    metadatas={"source": file_name},
                    ids=str(file_name) + '_' + str(big_i) + '_' + str(sim_i)
                )
                print('---------sim部分-------------')
                print(str(file_name) + '_' + str(big_i) + '_' + str(sim_i))

    print(original_document.get())
    print(cut_document.get())
    print('数据加入完成')
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
    print('完成')
    print(docx_span)
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


def process_pdf(file_path):
    global file_name
    file_name = file_path.name
    global data
    try:
        print(file_path.name)
        if file_path.name.endswith('.csv'):
            data = pd.read_csv(file_path.name)
        elif file_path.name.endswith('.xlsx') or file_path.name.endswith('.xls'):
            data = pd.read_excel(file_path.name)
        column_names = data.columns.values
        res = '||'.join(column_names)
    except:
        res = '请输入正确的文件，只能处理excel文件和csv文件'
    return res


def title_predict(question):
    print('------------问题-----------')
    print(question)
    prompt_tool = """Role: 标题提取专家
    ## Profile:
    - author: 智能小申
    - version: 1.2
    - language: 中文
    - description: 用户需要一个能够简化长标题但保留关键信息的提示词，以适应快速阅读和信息传递的需求。

    ## Constrains:
    - 输入信息如果大于20个字，你需要用小于20个字来描述该信息
    - 输入信息如果小于20个字，自己返回该信息
    - 如果标题中的时间、地点，关键词汇，对结果有唯一导向性则进行保留
    - 尽量不要自己创造新字，简化后的字尽量是问句中的
    - 任何条件下不要违反角色
    - 你不擅长客套, 不会进行没有意义的夸奖和客气对话
    - 解释完概念即结束对话, 不会询问是否有其它问题

    ## Skills:
    - 能分析出用户输入的真实意图
    - 擅长阅长标题摘要压缩、信息提取、用户问句压缩
    - 惜字如金, 不说废话
    - 编辑能力、信息筛选、语言精炼。

    ## Workflows:
    1. 角色初始化：作为一个标题提取专家，我已经掌握了如何使用简短的文字（小于20个字符）对用户输入进行总结。
    2. 接收用户输入：用户输入一段文本
    3. 理解分析：充分理解用户输入文本的内容，确保简化后的标题在20个字以内，同时保留关键信息。。
    4. 输出结果：将总结反馈给用户。
    5. 如果标题中的时间、地点，关键词汇，对结果有唯一导向性则进行保留

    ### Initialization:
    作为标题提取专家，擅长长句压缩成短句、长标题摘要压缩、信息提取、用户问句压缩等。
    用20字文本对用户输入进行总结。
    ### example:
原标题：上海市教育委员会 中共上海市委组织部 上海市经济和信息化委员会 上海市民政局 上海市财政局 上海市人力资源和社会保障局 上海市住房和城乡建设管理委员会 上海市市场监督管理局 上海市国有资产监督管理委员会 国家税务总局上海市税务局 上海市房屋管理局关于做好2022年上海市高校毕业生就业创业工作的通知
简化标题：2022年高校毕业生就业创业通知
 原标题：中共上海市教育卫生工作委员会 上海市教育委员会 中共上海市委宣传部 上海市文化和旅游局 上海市财政局 上海市人力资源和社会保障局 上海市文教结合工作协调小组办公室关于印发《上海市文教结合工作三年行动计划（2022-2024年）》和《上海市文教结合2022年工作要点》的通知
简化标题：文教结合工作三年行动计划及工作要点
 原标题：上海市教育委员会 中共上海市委组织部 上海市经济和信息化委员会 上海市民政局 上海市财政局 上海市人力资源和社会保障局 上海市市场监督管理局 上海市国有资产监督管理委员会 国家税务总局上海市税务局 上海市房屋管理局关于做好2023年上海市高校毕业生就业创业工作的通知
简化标题：2023年高校毕业生就业创业通知
 原标题：上海市商务委员会 上海市财政局 中国人民银行上海分行 国家税务总局上海市税务局 中国银行保险监督管理委员会上海监管局 国家外汇管理局上海市分局 上海市地方金融监督管理局 上海市文化和旅游局关于支持线下零售、住宿餐饮、外资外贸等市场主体纾困发展有关工作的通知
简化标题：市场主体纾困发展通知
 原标题：上海市民政局 上海市人力资源和社会保障局 上海市教育委员会 上海市房屋管理局 上海市司法局 上海市医疗保障局 上海市总工会 上海市妇女联合会 共青团上海市委员会 上海市大数据中心关于印发《社会救助“一件事”业务流程优化再造改革工作方案》的通知
简化标题：网信安全员培训项目批复
 原标题：上海市教育委员会 中共上海市委组织部 上海市经济和信息化委员会 上海市民政局 上海市财政局 上海市人力资源和社会保障局 上海市市场监督管理局 上海市国有资产监督管理委员会 上海市税务局关于做好2021年上海高校毕业生就业创业工作的通知
简化标题：2021年高校毕业生就业创业通知
 原标题：中共上海市教育卫生工作委员会 上海市教育委员会 上海市精神文明建设委员会办公室 上海市未成年人保护委员会办公室 共青团上海市委员会 上海市妇女联合会 上海市青少年学生校外活动联席会议办公室关于做好2021年上海市未成年人暑期工作的通知
简化标题：2021年上海市未成年人暑期工作通知
 原标题：中共上海市教育卫生工作委员会 上海市教育委员会 上海市精神文明建设委员会办公室 上海市未成年人保护委员会办公室 共青团上海市委员会 上海市妇女联合会 上海市青少年学生校外活动联席会议办公室关于做好2022年上海市未成年人暑期工作的通知
简化标题：2022年上海市未成年人暑期工作通知
 原标题：关于延续实施外籍个人有关津补贴 个人所得税政策的公告 Announcement on Extending IIT-related Policies for Expat Individuals  财政部 税务总局公告2023年第29号
简化标题：外籍个人津补贴政策延长
 原标题：中共上海市教育卫生工作委员会 上海市教育委员会上海市精神文明建设委员会办公室 上海市未成年人保护委员会办公室 共青团上海市委员会 上海市妇女联合会 上海市青少年学生校外活动联席会议办公室关于做好2023年上海市未成年人暑期工作的通知
简化标题：2023年上海市未成年人暑期工作通知
 原标题：【第14号公告】《<首次公开发行股票注册管理办法>第十二条、第十三条、第三十一条、第四十四条、第四十五条和<公开发行证券的公司信息披露内容与格式准则第57号——招股说明书>第七条有关规定的适用意见——证券期货法律适用意见第17号》
简化标题：证券期货法律适用意见第17号
 原标题：上海市民政局关于同意青浦区练塘镇等2个镇、赵巷镇赵巷居委会等7个居委会、华新镇凤溪居委会等4个居委会、香花桥街道金巷居委会等4个居委会分别为上海市社区建设示范镇、和谐社区建设示范居委会、社区建设模范居委会、社区建设示范居委会的批复
简化标题：关于社区建设示范镇、示范居委会、模范居委会批复
 原标题：上海市质量技术监督局关于征集对本市地方标准《铝合金挤压型材单位产品能源消耗限额》、《啤酒单位产品能源消耗限额》、《在用工业换热器能效测试及评价方法》、《涤纶（短纤）单位产品能源消耗限额》和《印染布单位产品综合能源消耗限额》意见的函
简化标题：征集对本市地方标准能源消耗限额的意见函
 原标题：上海市绿化和市容管理局关于转发《上海市住房和城乡建设管理委员会关于批准发布〈上海市建筑和装饰工程预算定额（SH01-31-2016）〉等7本工程预算定额及〈上海市建设工程施工费用计算规则（SHT0-33-2016）〉的通知》的通知
简化标题：绿化市容局转发工程预算定额及施工费用计算规则的通知
 原标题：上海市质量技术监督局关于征集对本市地方标准《铸钢件单位产品能源消耗限额》、《钢质热模锻件单位产品能源消耗限额》、《电动轮胎式集装箱门式起重机-高架滑触线式能源消耗指标标准限额和计算方法》和《数据中心机房单位能源消耗限额》意见的函
简化标题：征集对本市地方标准能源消耗限额的意见函
 原标题：上海市普陀区市场监管领域联合“双随机、一公开” 监管联席会议办公室上海市普陀区市场监督管理局关于印发《普陀区市场监管领域部门联合抽查事项清单 （第三版）》和《2023年度普陀区“双随机、一公开” 部门联合抽查工作计划》的 通知
简化标题：普陀区市场监管领域联合抽查事项清单及工作计划的通知
 原标题：上海市商务委员会 上海市规划和自然资源局 上海市住房和城乡建设管理委员会 上海市财政局 上海市文化和旅游局 上海市市场监督管理局 上海市绿化和市容管理局关于印发《上海市商圈能级提升三年行动方案（2024-2026年）》的通知
简化标题：商圈能级提升三年行动方案
 原标题：静安区市场监管领域联合“双随机、一公开”监管联席会议办公室 上海市静安区市场监督管理局关于印发《静安区市场监管领域部门联合抽查事项清单（第三版）》和《2024年度静安区“双随机、一公开”部门联合抽查工作计划》的通知
简化标题：静安区市场监管领域联合抽查事项清单及工作计划的通知
 原标题：转发《国家减灾委员会关于认真贯彻落实李克强总理等国务院领导重要批示精神扎实做好强降雨天气过程减灾救灾工作的紧急通知》《国家防汛抗旱总指挥部办公室关于贯彻落实李克强总理重要批示精神进一步做好防汛抗洪工作的通知》的通知
简化标题：做好减灾救灾工作 防汛抗洪工作的通知
 原标题：（2024年2月18日起施行，有效期至2029年2月17日）上海市市场监督管理局 国家税务总局上海市税务局 上海市人力资源和社会保障局 上海市医疗保障局 上海市公积金管理中心关于全面深化经营主体退出便利化改革的意见
简化标题：深化经营主体退出便利化改革的意见
 原标题：中共上海市教育卫生工作委员会 上海市教育委员会 中共上海市委宣传部 上海市文化和旅游局 上海市财政局 上海市人力资源和社会保障局 上海市文教结合工作协调小组办公室关于印发《2021年上海市文教结合工作要点》的通知
简化标题：2021年上海市文教结合工作要点的通知
 原标题：上海市奉贤区人民政府关于印发《奉贤区贯彻落实〈上海市被征收农民集体所有土地农业人员就业和社会保障办法〉的实施意见》的通知
简化标题：奉贤区落实征地农民就业保障实施意见的通知
 原标题：上海市宝山区教育局 上海市宝山区人力资源和社会保障局关于印发《进一步优化本区中小学专业技术岗位设置管理的实施方案》的通知
简化标题：宝山区优化小学专业技术岗位设置管理方案的通知
 原标题：上海市崇明区人民政府办公室关于印发本区“十四五”期间全面推进“15分钟社区生活圈”的行动方案（2023-2025）的通知
简化标题：崇明区印发推进社区生活圈方案的通知
 原标题：奉贤区发展改革委关于2020年奉贤区南桥镇八一路（沈陆中心路～环城东路）等四条“四好农村路”建设工程可行性研究报告的批复
简化标题：奉贤发改委关于“四好农村路”建设可行性研究报告的批复
 原标题：中共黄浦区委黄浦区人民政府关于坚持党建引领城区精细化治理科学规划建设“10分钟社区生活圈”“一街一路”示范区域的实施意见
简化标题：黄浦区建设社区生活圈 一街一路实施意见
 原标题：金山区人民政府办公室关于转发区科委制订的《推动工业互联网创新升级实施“工赋金山”工作方案（2020-2022年）》的通知
简化标题：金山区科委制定推动工业互联网创新升级实施方案
 原标题：关于印发《中国（上海）自由贸易试验区临港新片区工业互联网赋能重点产业集群发展专项行动方案（2022-2025年）》的通知
简化标题：印发临港工业互联网赋能重点产业集群行动方案
 原标题：临港新片区管委会、上海市商务委员会关于发布《关于支持临港新片区深化高水平制度型开放推动服务贸易创新发展的实施方案》的通知
简化标题：临港深化高水平制度型开放推动服务贸易创新发展的实施方案
 原标题：闵行区人民政府关于公布本区第九批非物质文化遗产代表性项目名录和第七批非物质文化遗产项目代表性传承人、主要传承人名单的通知
简化标题：闵行区公布非遗项目名录和非遗传承人名单"""
    res_question = "{0}\n 原标题：\n{1}".format(prompt_tool, question)
    print(res_question)
    response = client.chat.completions.create(
        model="glm-4-flash",  # 填写需要调用的模型名称
        messages=[{"role": "user", "content": res_question}]
    )
    res = response.choices[0].message.content
    return str(res).replace('简化标题：', '')


def gen_with_kn(target_column, progress=gr.Progress()):
    if target_column in data.columns.values:
        progress(0, desc="开始...")
        simple_list = []
        for i in progress.tqdm(data[target_column].values.tolist()):
            llm_text = title_predict(str(i))
            simple_list.append(llm_text)
        data['简化后的标题'] = simple_list
        if file_name.endswith('.csv'):
            data.to_csv(file_name, index=False)
        elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
            data.to_excel(file_name, index=False)
        return file_name
    else:
        raise gr.Error("请输入数据中有的列")


def gen_one(text):
    if len(text) > 0:
        print(text)
        llm_text = title_predict(str(text))
        print(llm_text)
        return llm_text
    else:
        raise gr.Error("请输入数据中有的列")


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
# demo.queue().launch(server_name='0.0.0.0', server_port=7569, share=False, inbrowser=True)
demo.queue().launch(server_name='10.10.22.232', server_port=7888, root_path="/title", share=False, inbrowser=True,
                    show_api=False)
