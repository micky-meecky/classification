#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: temptest.py
@datatime: 6/28/2023 7:55 PM
"""
# 随机生成testp的值，他是一个列表，列表中的元素是0-1之间的随机浮点数, 注意是浮点数
import random

testp = [random.random() for i in range(10)]

for i in range(len(testp) - 3):
    # 随机生成testp的值，他是一个列表，列表中的元素是0-1之间的随机浮点数

    print('第' + str(i+1) + '个实验结果：', end=', ')
    print(testp[i], end=', ')
    print(testp[i+1], end=', ')
    print(testp[i+2], end=', ')
    print(testp[i+3])
    testp = [random.random() for i in range(10)]

