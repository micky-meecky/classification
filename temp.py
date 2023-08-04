#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: temp.py
@datatime: 8/3/2023 7:55 PM
"""

import torch
from torchscale.architecture.config import RetNetConfig
from torchscale.architecture.retnet import RetNetDecoder

config = RetNetConfig(vocab_size=64000)
retnet = RetNetDecoder(config)

print(retnet)

# 测试retnet，构建一个输入
input = torch.randn(1, 3, 224, 224)
output = retnet(input)

print(output.shape)







