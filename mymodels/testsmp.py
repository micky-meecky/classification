#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: testsmp.py
@datatime: 7/23/2023 9:22 AM
"""


def build_model(self):
    # 在这里自己搭建自己的网络(网络结构)
    import segmentation_models_pytorch as smp

    aux_params = dict(
        pooling='avg',  # one of 'avg', 'max'
        dropout=0.5,  # dropout ratio, default is None
        activation='softmax',  # activation function, default is None
        classes=3,  # define number of output labels
    )

    self.unet = smp.UnetPlusPlus(encoder_name="timm-regnety_002",
                                 encoder_weights='imagenet',
                                 in_channels=self.img_ch,
                                 classes=1)
    # decoder_attention_type='scse')
    #  aux_params=aux_params)
    if self.sizeFlag:
        self.sizeFlag = False
        self.sizetotal = sum([param.nelement() for param in self.unet.parameters()])
        print("Number of parameter: %.2fM" % (self.sizetotal / 1e6))
    ######
    # 优化器修改
    self.optimizer = optim.Adam(list(self.unet.parameters()),
                                self.lr, [self.beta1, self.beta2])