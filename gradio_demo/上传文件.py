import gradio as gr
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertModel, AdamW
from torch import nn
import os
import pandas as pd

# 本地
# model_path = 'D:\models\chinese-roberta-wwm-ext'
# 232
model_path = '/opt/modules/chinese-roberta-wwm-ext'
# 检查是否有可用的GPU
device = torch.device("cpu")


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            return_token_type_ids=False,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.float)
        }


class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(model_path)
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return self.sigmoid(logits)


class KnTopic():
    def __init__(self):
        self.all_type = ['交通出行', '交通运输', '人力资源', '优待抚恤', '住房保障', '入伍服役', '公共安全', '公共服务', '公安消防', '公用事业', '农林牧渔', '出境入境',
                         '医疗卫生', '司法公证', '商务贸易', '国土和规划建设', '婚姻登记', '安全生产', '就业创业', '年检年审', '应对气候变化', '慈善公益', '户籍办理',
                         '投资审批', '抵押质押',
                         '招标拍卖', '教育科研', '文体教育', '文化体育', '旅游观光', '档案文物', '检验检疫', '死亡殡葬', '民族宗教', '水务气象', '海关口岸', '消费维权',
                         '环保绿化',
                         '生育收养', '知识产权', '社会保障（社会保险、社会救助）', '离职退休', '科技创新', '网络通信', '职业资格', '融资信贷', '行政缴费', '证件办理',
                         '质量技术',
                         '资质认证', '法人注销', '税收财政', '设立变更', '资质认定', '社会保障', '准营准办', '其他']
        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        loaded_paras = torch.load('/opt/nlp/cls_multi_label_question/output/model50loss0.015Accuracy0.89.pt',
                                  map_location=torch.device('cpu'))
        self.model = BertClassifier(num_labels=57)
        self.model.load_state_dict(loaded_paras)  # 用本地已有模型来重新初始化网络权重参数
        self.model.eval()  # 将模型设置为评估模式
        print('模型加载完成')

    def predict(self, text):
        texts_train = [text]
        labels_num_train = [[0 for i in range(len(self.all_type))]]
        data_size = 1
        print(texts_train)
        dataset = TextDataset(texts_train, labels_num_train, self.tokenizer)
        test_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)
        true_labels = []
        predicted_labels = []
        self.model.eval()  # 将模型设置为评估模式
        with torch.no_grad():
            for batch in test_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = self.model(batch['input_ids'], batch['attention_mask'])
                # predicted = (outputs > 0.8).float()  # 将预测转换为0或1
                true_labels.extend(batch['labels'].cpu().tolist())
                predicted_labels.extend(outputs.cpu().tolist())
        all_pre_lab_list = []
        all_pre_sco_list = []
        for more_pre in predicted_labels:
            for num, pre in enumerate(more_pre):
                if pre > 0.8:
                    all_pre_lab_list.append(self.all_type[num])
                    all_pre_sco_list.append(str(round(pre, 3)))
            print(all_pre_lab_list)
            print(all_pre_sco_list)
            if len(all_pre_lab_list) > 1:
                return '分类1:' + all_pre_lab_list[0] + '\n分类1置信度:' + all_pre_sco_list[0] + '\n' + \
                       '分类2:' + all_pre_lab_list[1] + '\n分类2置信度:' + all_pre_sco_list[1]
            elif len(all_pre_lab_list) == 1:
                return '分类:' + all_pre_lab_list[0] + '\n置信度:' + all_pre_sco_list[0]
            else:
                return '分类:其他\n置信度:1'


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


pre_model = KnTopic()


# def train_type():
#     df_train = pd.read_excel('常见问题分类每类100个总表训练集.xlsx')
#     texts_test = df_train['问题'].values.tolist()
#     labels_num_test = []
#     df_train['业务标签'] = df_train['分类1']
#     df_train.loc[~(df_train['分类2'].isnull()), '业务标签'] = df_train['分类1'] + ',' + df_train['分类2']
#     for more_label in df_train['业务标签'].values.tolist():
#         ori_label = [0 for i in range(len(pre_model.all_type))]
#         labels_num_test.append(ori_label)
#     labels_num_test = labels_num_test
#     # 创建数据集和数据加载器
#     dataset = TextDataset(texts_test, labels_num_test, pre_model.tokenizer)
#     test_loader = DataLoader(dataset, batch_size=data_size, shuffle=False)
#     return test_loader, texts_test, df_train

def gen_with_kn(progress=gr.Progress()):
    pre_text_1 = []
    pre_text_2 = []
    pre_score_1 = []
    pre_score_2 = []
    progress(0, desc="开始...")
    simple_list = []
    labels_num_test = []
    texts_test = data['问题'].values.tolist()
    for more_label in texts_test:
        ori_label = [0 for i in range(len(pre_model.all_type))]
        labels_num_test.append(ori_label)
    # 创建数据集和数据加载器
    dataset = TextDataset(texts_test, labels_num_test, pre_model.tokenizer)
    test_loader = DataLoader(dataset, batch_size=1, shuffle=False)
    true_labels = []
    predicted_labels = []
    pre_model.model.eval()
    with torch.no_grad():
        for batch in progress.tqdm(test_loader):
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = pre_model.model(batch['input_ids'], batch['attention_mask'])
            true_labels.extend(batch['labels'].cpu().tolist())
            predicted_labels.extend(outputs.cpu().tolist())
    for more_pre in predicted_labels:
        all_pre_lab_list = []
        all_pre_sco_list = []
        for num, pre in enumerate(more_pre):
            if pre > 0.8:
                all_pre_lab_list.append(pre_model.all_type[num])
                all_pre_sco_list.append(str(round(pre, 3)))
        if len(all_pre_lab_list) > 1:
            pre_text_1.append(all_pre_lab_list[0])
            pre_text_2.append(all_pre_lab_list[1])
            pre_score_1.append(all_pre_sco_list[0])
            pre_score_2.append(all_pre_sco_list[1])

        elif len(all_pre_lab_list) == 1:
            pre_text_1.append(all_pre_lab_list[0])
            pre_score_1.append(all_pre_sco_list[0])
            pre_text_2.append('')
            pre_score_2.append('')
        else:
            pre_text_1.append('')
            pre_score_1.append('')
            pre_text_2.append('')
            pre_score_2.append('')

    df = pd.DataFrame(
        {'text': texts_test, '分类一': pre_text_1, '分类二': pre_text_2, '分类一置信度': pre_score_1, '分类二置信度': pre_score_2})
    if file_name.endswith('.csv'):
        df.to_csv(file_name, index=False)
    elif file_name.endswith('.xlsx') or file_name.endswith('.xls'):
        df.to_excel(file_name, index=False)
    return file_name


def greet(text):
    print(text)
    if text.strip() == "":
        return "问题不能为空，请重新输入。"
    res = pre_model.predict(text)
    print(res)
    return str(res)


def clear(*args):
    return ""


with gr.Blocks() as demo:
    title = gr.Textbox(label="知识")
    output = gr.Textbox(label="分析结果")
    greet_btn = gr.Button("提交")
    greet_btn.click(fn=greet, inputs=[title], outputs=output)
    greet_btn = gr.Button("清除")
    for clear_text in [title, output]:
        greet_btn.click(fn=clear, inputs=clear_text, outputs=clear_text)
    with gr.Row():
        files = gr.File(label="添加文件", show_label=True)
        outputUploadKn = gr.Textbox(label="文件中存在的列名称")
    with gr.Row():
        btnUploadKn = gr.Button(value="上传文件", visible=True)
        btnUploadKn.click(process_pdf, [files], [outputUploadKn])
    with gr.Row():
        submitBtnKn = gr.Button("提交", variant="primary")
    with gr.Row():
        # output_model = gr.Textbox(label="经过大模型输出结果")
        output_dir = gr.components.File(label="Download Result")
    submitBtnKn.click(gen_with_kn, [], [output_dir])
demo.title = '知识分类'
# demo.launch()
# demo.launch(server_name='0.0.0.0',server_port=7863, app_name="/policy", share=False, inbrowser=True, show_api=False)
# demo.launch(server_name='10.10.22.232', server_port=7592, share=False, inbrowser=True)
demo.launch(server_name='10.10.22.232', server_port=7891, root_path="/question", share=False, inbrowser=True)
