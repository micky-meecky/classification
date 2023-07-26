#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
@author: 杨亚峰<yang_armageddon@163.com>
@version: 1.0.0
@license:  Apache Licence
@editor: Pycharm yyf
@file: test.py
@datatime: 7/26/2023 11:03 PM
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, in_channels, kernel_size=2, stride=2)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diff_y = x2.size()[2] - x1.size()[2]
        diff_x = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, [diff_x // 2, diff_x - diff_x // 2,
                        diff_y // 2, diff_y - diff_y // 2])
        x = torch.cat([x2, x1], dim=1)
        self.conv = ConvBlock(self.in_channels + self.out_channels, self.out_channels)
        return self.conv(x)


class ResNetUNet(nn.Module):
    def __init__(self, n_classes):
        super().__init__()

        base_model = models.resnet101(pretrained=True)
        base_layers = list(base_model.children())

        self.enc1 = nn.Sequential(*base_layers[:3])  # 64 out channels
        self.enc2 = nn.Sequential(*base_layers[3:5])  # 256 out channels
        self.enc3 = base_layers[5]  # 512 out channels
        self.enc4 = base_layers[6]  # 1024 out channels
        self.enc5 = base_layers[7]  # 2048 out channels

        self.up1 = UpBlock(2048, 1024)
        self.up2 = UpBlock(1024, 512)
        self.up3 = UpBlock(512, 256)
        self.up4 = UpBlock(256, 64)

        self.out = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        enc3 = self.enc3(enc2)
        enc4 = self.enc4(enc3)
        enc5 = self.enc5(enc4)

        up1 = self.up1(enc5, enc4)
        up2 = self.up2(up1, enc3)
        up3 = self.up3(up2, enc2)
        up4 = self.up4(up3, enc1)

        out = self.out(up4)
        return out

model = ResNetUNet(n_classes=2)
print(model)

x = torch.randn(1, 3, 256, 256)
y = model(x)
print(y.shape)

