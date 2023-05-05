import torch
import torch.nn as nn
import torch.optim as optim

# 模型定义
class SimpleBinaryClassifier(nn.Module):
    def __init__(self, input_size):
        super(SimpleBinaryClassifier, self).__init__()
        self.linear = nn.Linear(input_size, 1)

    def forward(self, x):
        return self.linear(x)

# 数据准备（仅作示例）
input_size = 10
batch_size = 32
num_samples = 100
X = torch.randn(num_samples, input_size)
y = torch.randint(0, 2, (num_samples,)).float().view(-1, 1)

# 初始化模型、损失函数和优化器
model = SimpleBinaryClassifier(input_size)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=0.00001)

# 训练过程
num_epochs = 20000
for epoch in range(num_epochs):
    for i in range(0, num_samples, batch_size):
        inputs = X[i:i+batch_size]
        targets = y[i:i+batch_size]

        # 模型前向传播
        outputs = model(inputs)

        # 计算损失
        loss = criterion(outputs, targets)

        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')
