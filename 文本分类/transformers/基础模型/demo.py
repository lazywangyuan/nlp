from transformers import BertTokenizer, BertModel, AdamW
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch
import pandas as pd
# 1. 数据预处理
# 假设texts和labels是已经准备好的文本数据和标签

texts = ["我爱机器学习", "深度学习太棒了", "机器学习很有趣", "我讨厌你"]
labels = ['positive', 'positive', 'positive', "negative"]  # 假设1代表正面情感

# 将标签转换为数值型
label_encoder = {"positive": 1, "negative": 0}
labels = [label_encoder[label] for label in labels]

# 2. 加载预训练的BERT分词器和模型
tokenizer = BertTokenizer.from_pretrained('D:\models\chinese-roberta-wwm-ext')


# 2. 定义BERT模型
class BertClassifier(nn.Module):
    def __init__(self, num_labels):
        super(BertClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('D:\models\chinese-roberta-wwm-ext')
        self.dropout = nn.Dropout(0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)
        return logits


# 3. 实例化模型
model = BertClassifier(num_labels=2)


# 3. 定义数据集类
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
            'labels': torch.tensor(label, dtype=torch.long)
        }


# 创建数据集和数据加载器
dataset = TextDataset(texts, labels, tokenizer)
train_loader = DataLoader(dataset, batch_size=2, shuffle=True)

# 4. 定义训练参数和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
optimizer = AdamW(model.parameters(), lr=2e-5)

# 5. 训练模型
model.train()
criterion = nn.CrossEntropyLoss()
for epoch in range(100):  # 训练3个epoch
    for batch in train_loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        outputs = model(batch['input_ids'], batch['attention_mask'])
        loss = criterion(outputs, batch['labels'])
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        print(f'Epoch: {epoch}, Loss: {loss.item()}')

# 6. 评估模型
# 这里省略了评估代码，评估时需要将模型设置为评估模式model.eval()，并关闭梯度计算torch.no_grad()

# 注意：实际应用中需要更大的数据集和更复杂的数据预处理步骤
