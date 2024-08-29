# 导入必要的库
from transformers import BertTokenizer, BertForSequenceClassification
import torch

# 加载预训练的 BERT 分词器和分类模型
tokenizer = BertTokenizer.from_pretrained(r'path_to_save_model')
model = BertForSequenceClassification.from_pretrained('path_to_save_model', num_labels=3)

# 待分类的文本
texts = '关于贯彻落实2016森林城市建设座谈会精神情况的报告'

# 类别标签映射
label2id = {0: '教育科研',
            1: '交通出行',
            2: '环保绿化'}

# 使用分词器对文本进行编码，并转换为 PyTorch 张量
input_ids = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)['input_ids']

# 使用模型进行推理
with torch.no_grad():
    outputs = model(input_ids)

    # 打印模型输出
    print(outputs)

    # 获取预测结果
    _, predicted = torch.max(outputs.logits, 1)

    # 映射预测结果到类别标签
    res = label2id[predicted.tolist()[0]]

    # 打印预测结果
    print(res)
