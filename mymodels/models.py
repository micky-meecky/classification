#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: models.py
@datatime: 4/21/2023 10:17 AM
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import models
import matplotlib.pyplot as plt



class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.model = models.resnet18(pretrained=True)
        self.model.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.model.head = nn.Identity()
        self.fc1 = nn.Linear(1000, 120)
        self.dropout1 = nn.Dropout(0.5)
        self.fc2 = nn.Linear(120, 84)
        self.dropout2 = nn.Dropout(0.5)
        self.fc3 = nn.Linear(84, 3)

    def forward(self, x):
        x = self.model(x)
        x = x.view(x.size(0), -1)
        input_size = x.size(1)
        # print('input_size is ', input_size)
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x

