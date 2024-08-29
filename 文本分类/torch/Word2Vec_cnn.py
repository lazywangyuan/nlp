import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from gensim.models import Word2Vec
import numpy as np
import random

# 创建数据集
class TextDataset(Dataset):
    def __init__(self, texts, labels, word2vec):
        self.texts = texts
        self.labels = labels
        self.word2vec = word2vec

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        text_vector = np.array([self.word2vec.wv[word] if word in self.word2vec.wv else np.zeros(self.word2vec.vector_size) for word in text])
        return torch.tensor(text_vector), torch.tensor(label)

# 构建Word2Vec模型
def build_word2vec(sentences, vector_size=100, window=5, min_count=1, workers=4, sg=0):
    model = Word2Vec(sentences=sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers, sg=sg)
    model.train(sentences, total_examples=model.corpus_count, epochs=model.epochs)
    return model

# 定义CNN模型
class TextCNN(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_classes):
        super(TextCNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.conv1 = nn.Conv1d(in_channels=embedding_dim, out_channels=100, kernel_size=3)
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.fc = nn.Linear(100, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(0, 2, 1)  # 调整维度
        x = self.conv1(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

# 训练模型
def train(model, dataset, word2vec, optimizer, criterion, num_epochs=10):
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True)
    for epoch in range(num_epochs):
        for texts, labels in train_loader:
            optimizer.zero_grad()
            outputs = model(texts)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}')

# 句子和标签
sentences = [
    ["我爱北京天安门"],  # 类别1
    ["我喜欢编程"],       # 类别2
    ["我讨厌下雨"],       # 类别1
    ["我喜欢机器学习"],   # 类别2
    ["我讨厌数学"]        # 类别3
]
labels = [1, 2, 1, 2, 3]  # 假设1, 2, 3是类别标签

# 构建Word2Vec模型
word2vec_model = build_word2vec(sentences)

# 将句子转换为索引
vocab_size = len(word2vec_model.wv)
word2index = {word: i for i, word in enumerate(word2vec_model.wv.index_to_key)}
index2word = {i: word for word, i in word2index.items()}

# 将句子转换为索引
indexed_sentences = [[word2index.get(word, 0) for word in sentence] for sentence in sentences]

# 创建数据集
dataset = TextDataset(indexed_sentences, labels, word2vec_model)

# 定义模型参数
embedding_dim = 100
num_classes = 4  # 根据类别数量调整

# 创建模型
model = TextCNN(vocab_size, embedding_dim, num_classes)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练模型
train(model, dataset, word2vec_model, optimizer, criterion, num_epochs=10)