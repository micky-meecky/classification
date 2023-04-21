#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: test.py
@datatime: 4/21/2023 10:24 AM
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler
from torchvision.datasets import MNIST
from torchvision import datasets, transforms
from models import Net
# 定义数据转换
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

# 定义测试集
# 定义数据集和转换
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

# 划分数据集

test_indices = torch.arange(len(test_dataset))


# 定义数据加载器
train_sampler = SubsetRandomSampler(train_indices)
val_sampler = SubsetRandomSampler(val_indices)
test_sampler = SubsetRandomSampler(test_indices)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=train_sampler)
val_loader = DataLoader(val_dataset, batch_size=64, sampler=val_sampler)
test_loader = DataLoader(test_dataset, batch_size=64, sampler=test_sampler)

# 查看测试集
dataiter = iter(test_loader)

# 加载模型
model = Net()
model.load_state_dict(torch.load('model.pth'))

# 测试模型
correct = 0
total = 0
with torch.no_grad():
    for data in test_loader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        print('测试集上的准确率: %.3f %%' % (100 * correct / total))

# 计算准确率
accuracy = correct / total
print('Accuracy of the network on the test images: %d %%' % (100 * accuracy))


