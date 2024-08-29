import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
# 假设我们有以下特征
features = torch.randn(1000, 10)  # 示例特征数据
labels = torch.randint(0, 2, (1000,))  # 示例标签数据

# 宽模型（Wide Model）
class WideModel(nn.Module):
    def __init__(self, num_features):
        super(WideModel, self).__init__()
        self.wide_layers = nn.Sequential(
            nn.Linear(num_features, 100),
            nn.ReLU(),
            nn.Linear(100, 50),
            nn.ReLU(),
            nn.Linear(50, 1)
        )

    def forward(self, x):
        return self.wide_layers(x)

# 深模型（Deep Model）
class DeepModel(nn.Module):
    def __init__(self, num_features):
        super(DeepModel, self).__init__()
        self.deep_layers = nn.Sequential(
            nn.Linear(num_features, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )

    def forward(self, x):
        return self.deep_layers(x)

# 训练数据
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Wide & Deep模型
class WideAndDeep(nn.Module):
    def __init__(self, wide_model, deep_model):
        super(WideAndDeep, self).__init__()
        self.wide = wide_model
        self.deep = deep_model

    def forward(self, x):
        wide_output = self.wide(x)
        deep_output = self.deep(x)
        return wide_output + deep_output

# 实例化模型
wide_model = WideModel(num_features=X_train.shape[1])
deep_model = DeepModel(num_features=X_train.shape[1])
wide_and_deep_model = WideAndDeep(wide_model, deep_model)

# 定义损失函数和优化器
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(wide_and_deep_model.parameters(), lr=0.001)

# 训练模型
for epoch in range(100):  # 示例：训练10个epochs
    optimizer.zero_grad()
    outputs = wide_and_deep_model(X_train)
    loss = criterion(outputs.squeeze(), y_train.float())
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 1 == 0:
        print(f'Epoch [{epoch+1}/10], Loss: {loss.item()}')

# 测试模型
with torch.no_grad():
    outputs = wide_and_deep_model(X_test)
    predictions = (outputs.squeeze() > 0).float()
    accuracy = (predictions == y_test).sum().item() / y_test.size(0)
print(f'Accuracy: {accuracy * 100}%')
