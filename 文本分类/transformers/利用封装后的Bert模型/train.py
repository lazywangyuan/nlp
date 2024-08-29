# 导入必要的库
import torch
from transformers import BertTokenizer, BertForSequenceClassification
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import os

# 准备数据
data = [("上海市人民政府办公厅关于印发《上海市“无废城市”建设工作方案》的通知", "环保绿化"),
        ("关于贯彻落实2016森林城市建设座谈会精神情况的报告", "环保绿化"),
        ("“蝶舞申城” 上海动物园第十一届蝴蝶展盛大开幕", "环保绿化"),
        ("“进博会停车预约系统”简介", "交通出行"),
        ("《配建机动车停车场（库）竣工验收》", "交通出行"),
        ("自5月22日起，上海地铁3、6、10、16号线恢复运营", "交通出行"),
        ("有哪几条交通线路可以到达四川北路街道社区事务受理中心？", "交通出行"),
        ("“储备教师”的薪资待遇如何？", "教育科研"),
        ("“储备教师”招录教师的条件是什么？", "教育科研"),
        ("“上海空中课堂”的客服渠道有哪些？", "教育科研"), ]

# 分割数据集
train_size = int(0.8 * len(data))
train_data, test_data = data[:train_size], data[train_size:]

# 加载预训练的BERT模型和tokenizer
tokenizer = BertTokenizer.from_pretrained(r'D:\models\bert-base-chinese')
model = BertForSequenceClassification.from_pretrained(r'D:\models\bert-base-chinese', num_labels=3)  # 假设有3个类别
label2id = {'教育科研': 0,
            '交通出行': 1,
            '环保绿化': 2}


# 数据预处理
def preprocess_data(data):
    # 将文本和标签分别提取出来
    texts, labels = zip(*data)

    # 使用BERT的tokenizer对文本进行处理，返回PyTorch Tensor
    input_ids = tokenizer(texts, return_tensors='pt', padding=True, truncation=True)['input_ids']

    # 将文本标签转换为对应的数字标签
    labels = torch.tensor([label2id[label] for label in labels])

    return TensorDataset(input_ids, labels)


# 转换为PyTorch的Dataset
train_dataset = preprocess_data(train_data)
test_dataset = preprocess_data(test_data)

# 使用DataLoader加载数据
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

# 定义优化器和损失函数
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
loss_fn = torch.nn.CrossEntropyLoss()

train_losses = []
test_losses = []

# 训练模型
epochs = 5
for epoch in range(epochs):
    model.train()
    for batch in train_loader:
        inputs, labels = batch
        optimizer.zero_grad()

        # 前向传播
        outputs = model(inputs, labels=labels)
        loss = outputs.loss

        # 反向传播和优化
        loss.backward()
        optimizer.step()

        train_losses.append(loss.item())

    # 评估模型
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in test_loader:
            inputs, labels = batch

            # 前向传播
            outputs = model(inputs)
            _, predicted = torch.max(outputs.logits, 1)

            # 统计正确预测数量
            total += labels.size(0)
            loss = loss_fn(outputs.logits, labels)
            correct += (predicted == labels).sum().item()

    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)  # 保存测试集损失

    accuracy = correct / total
    print(f"Test Accuracy: {accuracy}")

# 绘制训练集和测试集的损失曲线
plt.plot(train_losses, label='Training Loss')
plt.plot(test_losses, label='Test Loss')
plt.xlabel('Iterations')
plt.ylabel('Loss')
plt.legend()
plt.show()

# 保存模型
model.save_pretrained("path_to_save_model")
tokenizer.save_pretrained("path_to_save_model")
